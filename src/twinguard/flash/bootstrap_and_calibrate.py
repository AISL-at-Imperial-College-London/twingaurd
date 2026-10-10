"""Bootstrap uncertainty and conformal calibration for the frozen flash model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from twinguard.uncertainty.flash_bootstrap import (
    ENSEMBLE_CANDIDATES,
    PREDICTIVE_SCALE_FLOORS,
    PREDICTIVE_TRAJECTORIES,
    coherent_predictive_summary,
)
from twinguard.conformal.flash_conformal import (
    ALPHA_LEVELS,
    DIRECTION_NAMES,
    PRIMARY_ALPHA,
    corrected_quantile,
    normalized_action_distance,
    safety_direction_score,
)
from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.discrepancy.flash_state_space import (
    ENGINEERING_SCALES,
    FEATURE_NAMES,
    FlashStateSpaceParameters,
    fit_flash_state_space_em,
    parameters_from_payload,
    parameters_to_payload,
)
from twinguard.scenarios.flash_population_design import (
    ACTION_NAMES,
    BOUNDARY_DELTA_NORMALIZED,
    BOUNDARY_TOP_RANKS,
    MEASURED_OUTPUTS,
    OBSERVATION_COUNT,
    PRIMARY_FAULT_FAMILIES,
    PROCESS_BUNDLE_SHA256,
    SafetySpecification,
    boundary_candidate_pool,
    faulted_plant_parameters,
    latin_hypercube_actions,
    purpose_rng,
    purpose_seed,
    sample_primary_fault,
    validate_candidate_action,
)
from twinguard.plants.flash_drum import FlashAction, FlashParameters, FlashPlant, PhysicalStateError
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT, assert_isolated_output_path


POPULATION_DESIGN_SHA256 = "e976e1afc20fbe8f2da120717983dc58ed8a4ce6997d6a33bb15fb07edb2383a"
BOOTSTRAP_TARGET = 100
BOOTSTRAP_MAX_ATTEMPTS = 150
BOOTSTRAP_SEED_START = 31_000_000
ORDINARY_CALIBRATION_COUNT = 600
BOUNDARY_CALIBRATION_COUNT = 600
ORDINARY_CALIBRATION_SEED_START = 24_000_000
BOUNDARY_CALIBRATION_SEED_START = 25_000_000
CALIBRATION_ROLES = ("calibration_ordinary", "calibration_boundary")


_BOOTSTRAP_RESIDUAL: np.ndarray | None = None
_BOOTSTRAP_FEATURES: np.ndarray | None = None
_PREDICTIVE_MODELS: list[FlashStateSpaceParameters] | None = None


def _safe_path(relative: str) -> Path:
    path = assert_isolated_output_path(EXPERIMENT_ROOT / relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def _verify_frozen_inputs() -> tuple[dict[str, object], str]:
    preregistration = EXPERIMENT_ROOT / "population_preregistration.md"
    if hashlib.sha256(preregistration.read_bytes()).hexdigest() != POPULATION_DESIGN_SHA256:
        raise RuntimeError("population design preregistration hash mismatch")
    selected_path = EXPERIMENT_ROOT / "selected_flash_model.json"
    selected_hash = hashlib.sha256(selected_path.read_bytes()).hexdigest()
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    if selected["model"]["latent_dimension"] != 3:
        raise RuntimeError("calibration requires the frozen model selection k=3 model class")
    if selected["selection_status"] != "NO_MODEL_PASSED_PREDECLARED_FALLBACK":
        raise RuntimeError("unexpected model selection status")
    return selected, selected_hash


def _init_bootstrap(training_path: str) -> None:
    global _BOOTSTRAP_RESIDUAL, _BOOTSTRAP_FEATURES
    with np.load(training_path) as saved:
        _BOOTSTRAP_RESIDUAL = np.asarray(saved["residual"], dtype=float)
        _BOOTSTRAP_FEATURES = np.asarray(saved["features"], dtype=float)


def _bootstrap_task(attempt: int) -> tuple[int, int, FlashStateSpaceParameters | None, str]:
    if _BOOTSTRAP_RESIDUAL is None or _BOOTSTRAP_FEATURES is None:
        raise RuntimeError("bootstrap worker is not initialized")
    seed = BOOTSTRAP_SEED_START + attempt
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(_BOOTSTRAP_RESIDUAL), len(_BOOTSTRAP_RESIDUAL))
    try:
        model = fit_flash_state_space_em(
            _BOOTSTRAP_RESIDUAL[indices],
            _BOOTSTRAP_FEATURES[indices],
            3,
            33_000_003,
            max_iterations=100,
            relative_tolerance=1e-5,
        )
        finite = all(
            np.all(np.isfinite(getattr(model, name)))
            for name in ("A", "B", "C", "D", "Q", "R", "initial_covariance")
        )
        covariance_ok = bool(
            np.all(np.linalg.eigvalsh(model.Q) > 0.0)
            and np.all(np.linalg.eigvalsh(model.R) > 0.0)
            and np.all(np.linalg.eigvalsh(model.initial_covariance) > 0.0)
        )
        reasons = []
        if not model.converged:
            reasons.append("not_converged")
        if not finite:
            reasons.append("nonfinite_parameters")
        if not covariance_ok:
            reasons.append("nonpositive_covariance")
        if not model.spectral_radius < 0.995:
            reasons.append("spectral_radius_not_below_0.995")
        return attempt, seed, model if not reasons else None, "|".join(reasons)
    except Exception as exc:
        return attempt, seed, None, f"{type(exc).__name__}:{exc}"


def fit_bootstrap_models(workers: int) -> tuple[list[FlashStateSpaceParameters], list[dict[str, object]]]:
    training_path = _safe_path("model_selection_data/training_method_inputs.npz")
    accepted: list[FlashStateSpaceParameters] = []
    manifest: list[dict[str, object]] = []
    next_attempt = 0
    while len(accepted) < BOOTSTRAP_TARGET and next_attempt < BOOTSTRAP_MAX_ATTEMPTS:
        batch_size = min(25, BOOTSTRAP_MAX_ATTEMPTS - next_attempt)
        attempts = list(range(next_attempt, next_attempt + batch_size))
        with ProcessPoolExecutor(
            max_workers=min(workers, batch_size),
            initializer=_init_bootstrap,
            initargs=(str(training_path),),
        ) as pool:
            results = list(pool.map(_bootstrap_task, attempts, chunksize=1))
        for attempt, seed, model, reason in sorted(results):
            use = model is not None and len(accepted) < BOOTSTRAP_TARGET
            if use:
                accepted.append(model)
            manifest.append(
                {
                    "bootstrap_attempt": attempt,
                    "bootstrap_seed": seed,
                    "resampled_complete_episode_count": 667,
                    "fit_converged": bool(model is not None),
                    "numerically_accepted": use,
                    "accepted_model_index": len(accepted) - 1 if use else "",
                    "rejection_reason": reason if model is None else ("target_already_filled" if not use else ""),
                    "latent_dimension": 3,
                    "spectral_radius": model.spectral_radius if model is not None else "",
                    "iterations": model.iterations if model is not None else "",
                    "fit_seconds": model.fit_seconds if model is not None else "",
                    "final_parameter_change": model.final_parameter_change if model is not None else "",
                    "performance_based_rejection": False,
                }
            )
        next_attempt += batch_size
        _write_csv(_safe_path("flash_bootstrap_manifest.csv"), manifest)
        _safe_path("flash_bootstrap_models.json").write_text(
            json.dumps([parameters_to_payload(model) for model in accepted], indent=2)
            + "\n",
            encoding="utf-8",
        )
        print(
            f"calibration bootstrap attempts={next_attempt}, accepted={len(accepted)}",
            flush=True,
        )
    if len(accepted) < BOOTSTRAP_TARGET:
        raise RuntimeError(
            f"only {len(accepted)} of {BOOTSTRAP_TARGET} bootstrap fits accepted "
            f"within {BOOTSTRAP_MAX_ATTEMPTS} predeclared attempts"
        )
    return accepted, manifest


def load_bootstrap_models() -> tuple[list[FlashStateSpaceParameters], list[dict[str, str]]]:
    payloads = json.loads(_safe_path("flash_bootstrap_models.json").read_text(encoding="utf-8"))
    models = [parameters_from_payload(payload) for payload in payloads]
    with _safe_path("flash_bootstrap_manifest.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        manifest = list(csv.DictReader(handle))
    if len(models) != BOOTSTRAP_TARGET:
        raise RuntimeError("cached bootstrap ensemble is incomplete")
    return models, manifest


def _init_predictive(models_payload: Sequence[Mapping[str, object]]) -> None:
    global _PREDICTIVE_MODELS
    _PREDICTIVE_MODELS = [parameters_from_payload(payload) for payload in models_payload]


def _selection_ensemble_task(
    task: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
) -> tuple[int, dict[int, tuple[np.ndarray, np.ndarray]]]:
    if _PREDICTIVE_MODELS is None:
        raise RuntimeError("predictive worker is not initialized")
    index, observed_residual, observed_features, future_features, nominal_future = task
    summaries: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for candidate_index, ensemble_size in enumerate(ENSEMBLE_CANDIDATES):
        summary = coherent_predictive_summary(
            _PREDICTIVE_MODELS,
            observed_residual,
            observed_features,
            future_features,
            nominal_future,
            ensemble_size=ensemble_size,
            seed=purpose_seed(23_000_000 + index, 601 + candidate_index),
        )
        summaries[ensemble_size] = (summary.center, summary.scale)
    return index, summaries


def select_ensemble_size(
    models: Sequence[FlashStateSpaceParameters], workers: int
) -> tuple[int, list[dict[str, object]]]:
    with np.load(_safe_path("model_selection_data/selection_operational_inputs.npz")) as saved:
        selection = {name: np.asarray(saved[name]) for name in saved.files}
    tasks = [
        (
            index,
            selection["observed_residual"][index],
            selection["observed_features"][index],
            selection["future_features"][index],
            selection["nominal_future"][index],
        )
        for index in range(len(selection["observed_residual"]))
    ]
    payloads = [parameters_to_payload(model) for model in models]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_predictive, initargs=(payloads,)
    ) as pool:
        results = list(pool.map(_selection_ensemble_task, tasks, chunksize=1))
    results.sort(key=lambda item: item[0])
    diagnostics: list[dict[str, object]] = []
    for ensemble_size in ENSEMBLE_CANDIDATES:
        centers = np.stack([item[1][ensemble_size][0] for item in results])
        scales = np.stack([item[1][ensemble_size][1] for item in results])
        reference_centers = np.stack([item[1][100][0] for item in results])
        reference_scales = np.stack([item[1][100][1] for item in results])
        center_difference = np.median(
            np.abs(centers - reference_centers) / ENGINEERING_SCALES,
            axis=(0, 1),
        )
        scale_ratio = np.median(
            scales / np.maximum(reference_scales, PREDICTIVE_SCALE_FLOORS),
            axis=(0, 1),
        )
        passed = bool(
            np.all(center_difference <= 0.05)
            and np.all((scale_ratio >= 0.95) & (scale_ratio <= 1.05))
        )
        diagnostics.append(
            {
                "ensemble_size": ensemble_size,
                "selection_episode_count": len(results),
                "median_pressure_center_difference_normalized": center_difference[0],
                "median_level_center_difference_normalized": center_difference[1],
                "median_temperature_center_difference_normalized": center_difference[2],
                "median_pressure_scale_ratio_to_j100": scale_ratio[0],
                "median_level_scale_ratio_to_j100": scale_ratio[1],
                "median_temperature_scale_ratio_to_j100": scale_ratio[2],
                "predeclared_convergence_gate_passed": passed,
            }
        )
    passing = [
        int(row["ensemble_size"])
        for row in diagnostics
        if bool(row["predeclared_convergence_gate_passed"])
    ]
    selected = min(passing) if passing else 100
    for row in diagnostics:
        row["selected"] = int(row["ensemble_size"]) == selected
    _write_csv(_safe_path("calibration_data/ensemble_selection_diagnostics.csv"), diagnostics)
    return selected, diagnostics


def _state_valid(states: np.ndarray, parameters: FlashParameters) -> np.ndarray:
    total = states[:, 0]
    benzene = states[:, 1]
    capacity = (
        benzene * parameters.benzene_cp_liquid_j_mol_k
        + (total - benzene) * parameters.toluene_cp_liquid_j_mol_k
    )
    temperature = parameters.reference_temperature_k + states[:, 2] / capacity
    return (
        np.all(np.isfinite(states), axis=1)
        & (total >= parameters.minimum_holdup_mol)
        & (total <= parameters.maximum_holdup_mol)
        & (benzene > 0.0)
        & (benzene < total)
        & (temperature >= parameters.minimum_temperature_k)
        & (temperature <= parameters.maximum_temperature_k)
    )


def _batch_rates(
    states: np.ndarray, actions: np.ndarray, parameters: FlashParameters
) -> np.ndarray:
    total, benzene, energy = states.T
    x = benzene / total
    capacity = (
        benzene * parameters.benzene_cp_liquid_j_mol_k
        + (total - benzene) * parameters.toluene_cp_liquid_j_mol_k
    )
    temperature = parameters.reference_temperature_k + energy / capacity
    p_benzene = 10.0 ** (
        parameters.benzene_antoine.a
        - parameters.benzene_antoine.b_kelvin
        / (temperature + parameters.benzene_antoine.c_kelvin)
    )
    p_toluene = 10.0 ** (
        parameters.toluene_antoine.a
        - parameters.toluene_antoine.b_kelvin
        / (temperature + parameters.toluene_antoine.c_kelvin)
    )
    pressure = x * p_benzene + (1.0 - x) * p_toluene
    y = x * p_benzene / pressure
    level = total / parameters.liquid_capacity_mol
    liquid_flow = (
        parameters.liquid_valve_coefficient_mol_s
        * actions[:, 2]
        * np.sqrt(np.maximum(level, 0.0))
    )
    vapor_flow = (
        parameters.vapor_valve_coefficient_mol_s_bar_sqrt
        * actions[:, 1]
        * np.sqrt(np.maximum(pressure - parameters.downstream_pressure_bar, 0.0))
    )
    feed = actions[:, 3]
    total_rate = feed - liquid_flow - vapor_flow
    benzene_rate = (
        feed * parameters.feed_benzene_fraction - liquid_flow * x - vapor_flow * y
    )
    cp_b, cp_t = (
        parameters.benzene_cp_liquid_j_mol_k,
        parameters.toluene_cp_liquid_j_mol_k,
    )
    h_feed = (
        parameters.feed_benzene_fraction * cp_b
        + (1.0 - parameters.feed_benzene_fraction) * cp_t
    ) * (parameters.feed_temperature_k - parameters.reference_temperature_k)
    h_liquid = (x * cp_b + (1.0 - x) * cp_t) * (
        temperature - parameters.reference_temperature_k
    )
    tr_b = temperature / parameters.benzene_vaporization.critical_temperature_k
    tr_t = temperature / parameters.toluene_vaporization.critical_temperature_k
    latent_b = (
        1000.0
        * parameters.benzene_vaporization.a_kj_mol
        * np.exp(-parameters.benzene_vaporization.alpha * tr_b)
        * (1.0 - tr_b) ** parameters.benzene_vaporization.beta
    )
    latent_t = (
        1000.0
        * parameters.toluene_vaporization.a_kj_mol
        * np.exp(-parameters.toluene_vaporization.beta * tr_t)
        * (1.0 - tr_t) ** parameters.toluene_vaporization.beta
    )
    h_vapor = (y * cp_b + (1.0 - y) * cp_t) * (
        temperature - parameters.reference_temperature_k
    ) + y * latent_b + (1.0 - y) * latent_t
    energy_rate = (
        feed * h_feed
        + actions[:, 0]
        - liquid_flow * h_liquid
        - vapor_flow * h_vapor
    )
    return np.column_stack([total_rate, benzene_rate, energy_rate])


def batch_nominal_dt_margins(
    decision_state: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    """Compute nominal safety margins using batched one-second RK4 integration."""

    parameters = FlashParameters()
    states = np.broadcast_to(np.asarray(decision_state, dtype=float), (len(actions), 3)).copy()
    active = np.ones(len(actions), dtype=bool)
    max_violation = np.full(len(actions), -np.inf)
    nominal_state = FlashDigitalTwin().nominal_state().as_array()
    for second in range(1, 1201):
        safe = states.copy()
        safe[~active] = nominal_state
        k1 = _batch_rates(safe, actions, parameters)
        stage2 = states + 0.5 * k1
        active &= _state_valid(stage2, parameters)
        safe2 = stage2.copy()
        safe2[~active] = nominal_state
        k2 = _batch_rates(safe2, actions, parameters)
        stage3 = states + 0.5 * k2
        active &= _state_valid(stage3, parameters)
        safe3 = stage3.copy()
        safe3[~active] = nominal_state
        k3 = _batch_rates(safe3, actions, parameters)
        stage4 = states + k3
        active &= _state_valid(stage4, parameters)
        safe4 = stage4.copy()
        safe4[~active] = nominal_state
        k4 = _batch_rates(safe4, actions, parameters)
        states[active] += (k1[active] + 2.0 * k2[active] + 2.0 * k3[active] + k4[active]) / 6.0
        active &= _state_valid(states, parameters)
        if second % 10 == 0:
            total, benzene, energy = states.T
            capacity = (
                benzene * parameters.benzene_cp_liquid_j_mol_k
                + (total - benzene) * parameters.toluene_cp_liquid_j_mol_k
            )
            temperature = parameters.reference_temperature_k + energy / capacity
            x = benzene / total
            p_b = 10.0 ** (
                parameters.benzene_antoine.a
                - parameters.benzene_antoine.b_kelvin
                / (temperature + parameters.benzene_antoine.c_kelvin)
            )
            p_t = 10.0 ** (
                parameters.toluene_antoine.a
                - parameters.toluene_antoine.b_kelvin
                / (temperature + parameters.toluene_antoine.c_kelvin)
            )
            pressure = x * p_b + (1.0 - x) * p_t
            level = total / parameters.liquid_capacity_mol
            violation = np.max(
                np.column_stack(
                    [
                        (pressure - 0.90) / 0.10,
                        (level - 0.85) / 0.10,
                        (0.20 - level) / 0.10,
                        (temperature - 365.0) / 5.0,
                    ]
                ),
                axis=1,
            )
            max_violation[active] = np.maximum(
                max_violation[active], violation[active]
            )
    margins = -max_violation
    margins[~active] = -np.inf
    return margins


def _boundary_action(decision_state: np.ndarray, seed: int) -> tuple[FlashAction, float, bool, int]:
    pool = boundary_candidate_pool(purpose_seed(seed, 503))
    margins = batch_nominal_dt_margins(decision_state, pool)
    valid_indices = np.flatnonzero(np.isfinite(margins))
    if not len(valid_indices):
        raise RuntimeError("boundary pool has no valid nominal-DT action")
    ordering = valid_indices[
        np.lexsort((valid_indices, np.abs(margins[valid_indices])))
    ]
    eligible = ordering[np.abs(margins[ordering]) <= BOUNDARY_DELTA_NORMALIZED]
    candidates = eligible if len(eligible) else ordering[:1]
    top = candidates[:BOUNDARY_TOP_RANKS]
    selected = int(top[seed % len(top)])
    return (
        validate_candidate_action(pool[selected]),
        float(margins[selected]),
        bool(len(eligible)),
        int(len(eligible)),
    )


@dataclass(frozen=True)
class CalibrationEpisode:
    episode_id: str
    episode_seed: int
    observed_residual: np.ndarray
    observed_features: np.ndarray
    future_features: np.ndarray
    nominal_future: np.ndarray
    actual_future: np.ndarray
    candidate_action: np.ndarray
    completed: bool
    evaluator_metadata: dict[str, object]


def _feature(action: FlashAction, nominal_action: FlashAction, output: Mapping[str, float]) -> np.ndarray:
    current = action.as_array()
    return np.concatenate(
        [current, current - nominal_action.as_array(), [output[name] for name in MEASURED_OUTPUTS]]
    )


def _calibration_task(task: tuple[str, int, int, np.ndarray | None]) -> CalibrationEpisode:
    role, index, seed, ordinary_action = task
    if role not in CALIBRATION_ROLES:
        raise ValueError(f"calibration prohibited population role: {role}")
    family = PRIMARY_FAULT_FAMILIES[index % 3]
    truth = sample_primary_fault(family, seed)
    nominal_parameters = FlashParameters()
    plant_parameters = faulted_plant_parameters(nominal_parameters, truth)
    initial_rng = purpose_rng(seed, 101)
    noise_rng = purpose_rng(seed, 307)
    template = FlashDigitalTwin(nominal_parameters)
    initial_state = template.state_from_nxt(
        float(initial_rng.uniform(550.0, 650.0)),
        float(initial_rng.uniform(0.42, 0.48)),
        float(initial_rng.uniform(348.0, 352.0)),
    )
    plant = FlashPlant(plant_parameters, initial_state)
    twin = FlashDigitalTwin(nominal_parameters, initial_state)
    nominal_action = twin.nominal_action()
    observed_residual = np.full((50, 3), np.nan)
    observed_features = np.full((50, len(FEATURE_NAMES)), np.nan)
    future_features = np.full((120, len(FEATURE_NAMES)), np.nan)
    nominal_future = np.full((120, 3), np.nan)
    actual_future = np.full((120, 3), np.nan)
    sensor_sd = np.asarray([0.002, 0.002, 0.10])
    completed_observations = 0
    completed_future = 0
    domain_exit = ""
    boundary_margin = float("nan")
    boundary_supported = False
    eligible_count = 0
    action = nominal_action
    try:
        for second in range(1, 501):
            plant.step(nominal_action, 1.0)
            twin.step(nominal_action, 1.0)
            if second % 10:
                continue
            p_out = plant.outputs(plant.state)
            d_out = twin.outputs(twin.state)
            physical = np.asarray([p_out[name] for name in MEASURED_OUTPUTS])
            nominal = np.asarray([d_out[name] for name in MEASURED_OUTPUTS])
            observed_residual[completed_observations] = (
                physical + noise_rng.normal(0.0, sensor_sd) - nominal
            )
            observed_features[completed_observations] = _feature(
                nominal_action, nominal_action, d_out
            )
            completed_observations += 1
        if role == "calibration_boundary":
            action, boundary_margin, boundary_supported, eligible_count = _boundary_action(
                twin.state.as_array(), seed
            )
        else:
            if ordinary_action is None:
                raise RuntimeError("ordinary calibration action missing")
            action = validate_candidate_action(ordinary_action)
        for second in range(1, 1201):
            plant.step(action, 1.0)
            twin.step(action, 1.0)
            if second % 10:
                continue
            p_out = plant.outputs(plant.state)
            d_out = twin.outputs(twin.state)
            actual_future[completed_future] = [p_out[name] for name in MEASURED_OUTPUTS]
            nominal_future[completed_future] = [d_out[name] for name in MEASURED_OUTPUTS]
            future_features[completed_future] = _feature(action, nominal_action, d_out)
            completed_future += 1
    except PhysicalStateError as exc:
        domain_exit = str(exc)
    completed = completed_observations == 50 and completed_future == 120
    if ordinary_action is None and completed_observations < 50:
        action = nominal_action
    return CalibrationEpisode(
        episode_id=f"flash_{role}_{index:04d}",
        episode_seed=seed,
        observed_residual=observed_residual,
        observed_features=observed_features,
        future_features=future_features,
        nominal_future=nominal_future,
        actual_future=actual_future,
        candidate_action=action.as_array(),
        completed=completed,
        evaluator_metadata={
            "episode_id": f"flash_{role}_{index:04d}",
            "population_role": role,
            "episode_seed_evaluator_only": seed,
            "fault_family_evaluator_only": family.value,
            "fault_severity_evaluator_only": truth.normalized_severity,
            "fault_metadata_available_to_method": False,
            "completed_observations": completed_observations,
            "completed_future": completed_future,
            "complete_trajectory": completed,
            "numerical_domain_exit_evaluator_only": domain_exit,
            "nominal_dt_boundary_margin": boundary_margin,
            "nominal_dt_boundary_band_found": boundary_supported,
            "nominal_dt_boundary_eligible_count": eligible_count,
        },
    )


def generate_calibration_populations(workers: int) -> tuple[dict[str, list[CalibrationEpisode]], list[dict[str, object]]]:
    ordinary_actions = latin_hypercube_actions(
        ORDINARY_CALIBRATION_COUNT, ORDINARY_CALIBRATION_SEED_START + 401
    )
    role_tasks = {
        "calibration_ordinary": [
            (
                "calibration_ordinary",
                index,
                ORDINARY_CALIBRATION_SEED_START + index,
                ordinary_actions[index],
            )
            for index in range(ORDINARY_CALIBRATION_COUNT)
        ],
        "calibration_boundary": [
            (
                "calibration_boundary",
                index,
                BOUNDARY_CALIBRATION_SEED_START + index,
                None,
            )
            for index in range(BOUNDARY_CALIBRATION_COUNT)
        ],
    }
    populations: dict[str, list[CalibrationEpisode]] = {}
    evaluator_rows: list[dict[str, object]] = []
    for role in CALIBRATION_ROLES:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            episodes = list(pool.map(_calibration_task, role_tasks[role], chunksize=1))
        populations[role] = [episode for episode in episodes if episode.completed]
        evaluator_rows.extend(episode.evaluator_metadata for episode in episodes)
        print(
            f"calibration {role}: complete={len(populations[role])}, "
            f"domain_exits={len(episodes)-len(populations[role])}",
            flush=True,
        )
        complete = populations[role]
        np.savez_compressed(
            _safe_path(f"calibration_data/{role}_operational_inputs.npz"),
            episode_id=np.asarray([episode.episode_id for episode in complete]),
            observed_residual=np.stack([episode.observed_residual for episode in complete]),
            observed_features=np.stack([episode.observed_features for episode in complete]),
            future_features=np.stack([episode.future_features for episode in complete]),
            nominal_future=np.stack([episode.nominal_future for episode in complete]),
            candidate_action=np.stack([episode.candidate_action for episode in complete]),
        )
    _write_csv(_safe_path("calibration_data/CALIBRATION_EVALUATOR_MANIFEST.csv"), evaluator_rows)
    return populations, evaluator_rows


def _calibration_prediction_task(
    task: tuple[str, int, CalibrationEpisode, int]
) -> dict[str, object]:
    if _PREDICTIVE_MODELS is None:
        raise RuntimeError("predictive worker is not initialized")
    role, index, episode, ensemble_size = task
    summary = coherent_predictive_summary(
        _PREDICTIVE_MODELS,
        episode.observed_residual,
        episode.observed_features,
        episode.future_features,
        episode.nominal_future,
        ensemble_size=ensemble_size,
        seed=purpose_seed(episode.episode_seed, 601),
    )
    score, direction, horizon = safety_direction_score(
        episode.actual_future, summary.center, summary.scale
    )
    floor_hits = summary.scale <= PREDICTIVE_SCALE_FLOORS
    return {
        "episode_id": episode.episode_id,
        "calibration_population": role,
        "whole_trajectory_safety_score": score,
        "maximizing_direction": direction,
        "maximizing_horizon_index": horizon,
        "normalized_action_distance": normalized_action_distance(
            episode.candidate_action
        ),
        "mondrian_group": "",
        "mondrian_supported": False,
        "mondrian_reason": "BINS_NOT_PREREGISTERED",
        "minimum_pressure_scale_bar": float(np.min(summary.scale[:, 0])),
        "median_pressure_scale_bar": float(np.median(summary.scale[:, 0])),
        "minimum_level_scale_fraction": float(np.min(summary.scale[:, 1])),
        "median_level_scale_fraction": float(np.median(summary.scale[:, 1])),
        "minimum_temperature_scale_k": float(np.min(summary.scale[:, 2])),
        "median_temperature_scale_k": float(np.median(summary.scale[:, 2])),
        "pressure_scale_floor_hit_fraction": float(np.mean(floor_hits[:, 0])),
        "level_scale_floor_hit_fraction": float(np.mean(floor_hits[:, 1])),
        "temperature_scale_floor_hit_fraction": float(np.mean(floor_hits[:, 2])),
        "ensemble_size": ensemble_size,
        "predictive_trajectory_count": PREDICTIVE_TRAJECTORIES,
    }


def calibrate(
    models: Sequence[FlashStateSpaceParameters],
    populations: Mapping[str, Sequence[CalibrationEpisode]],
    ensemble_size: int,
    workers: int,
) -> tuple[list[dict[str, object]], dict[str, dict[str, dict[str, object]]]]:
    tasks = [
        (role, index, episode, ensemble_size)
        for role in CALIBRATION_ROLES
        for index, episode in enumerate(populations[role])
    ]
    payloads = [parameters_to_payload(model) for model in models]
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_predictive, initargs=(payloads,)
    ) as pool:
        rows = list(pool.map(_calibration_prediction_task, tasks, chunksize=1))
    quantiles: dict[str, dict[str, dict[str, object]]] = {}
    for role in CALIBRATION_ROLES:
        scores = [
            float(row["whole_trajectory_safety_score"])
            for row in rows
            if row["calibration_population"] == role
        ]
        quantiles[role] = {}
        for alpha in ALPHA_LEVELS:
            quantile, rank = corrected_quantile(scores, alpha)
            quantiles[role][str(alpha)] = {
                "alpha": alpha,
                "sample_count": len(scores),
                "corrected_rank": rank,
                "quantile": quantile,
                "scope": "pooled_across_F1_F2_F3_and_actions_within_population",
            }
    _write_csv(_safe_path("flash_calibration_scores.csv"), rows)
    return rows, quantiles


def _build_report(
    bootstrap_manifest: Sequence[Mapping[str, object]],
    ensemble_diagnostics: Sequence[Mapping[str, object]],
    ensemble_size: int,
    calibration_rows: Sequence[Mapping[str, object]],
    quantiles: Mapping[str, Mapping[str, Mapping[str, object]]],
    evaluator_rows: Sequence[Mapping[str, object]],
) -> str:
    bootstrap_rejections: dict[str, int] = {}
    for row in bootstrap_manifest:
        reason = str(row["rejection_reason"])
        if reason:
            bootstrap_rejections[reason] = bootstrap_rejections.get(reason, 0) + 1
    ensemble_lines = [
        f"| {row['ensemble_size']} | {row['median_pressure_center_difference_normalized']:.4f} / "
        f"{row['median_level_center_difference_normalized']:.4f} / "
        f"{row['median_temperature_center_difference_normalized']:.4f} | "
        f"{row['median_pressure_scale_ratio_to_j100']:.4f} / "
        f"{row['median_level_scale_ratio_to_j100']:.4f} / "
        f"{row['median_temperature_scale_ratio_to_j100']:.4f} | "
        f"{'PASS' if row['predeclared_convergence_gate_passed'] else 'FAIL'} |"
        for row in ensemble_diagnostics
    ]
    quantile_lines = []
    for role in CALIBRATION_ROLES:
        for alpha in ALPHA_LEVELS:
            item = quantiles[role][str(alpha)]
            quantile_lines.append(
                f"| {role} | {alpha:.2f} | {item['sample_count']} | "
                f"{item['corrected_rank']} | {item['quantile']:.6f} |"
            )
    scale_lines = []
    for role in CALIBRATION_ROLES:
        selected = [row for row in calibration_rows if row["calibration_population"] == role]
        scale_lines.append(
            f"| {role} | "
            f"{np.median([float(row['median_pressure_scale_bar']) for row in selected]):.6f} | "
            f"{np.median([float(row['median_level_scale_fraction']) for row in selected]):.6f} | "
            f"{np.median([float(row['median_temperature_scale_k']) for row in selected]):.6f} | "
            f"{np.mean([float(row['pressure_scale_floor_hit_fraction']) for row in selected]):.4f} / "
            f"{np.mean([float(row['level_scale_floor_hit_fraction']) for row in selected]):.4f} / "
            f"{np.mean([float(row['temperature_scale_floor_hit_fraction']) for row in selected]):.4f} |"
        )
    complete_counts = {
        role: sum(
            row["population_role"] == role and bool(row["complete_trajectory"])
            for row in evaluator_rows
        )
        for role in CALIBRATION_ROLES
    }
    return f"""# Flash calibration Bootstrap and Conformal Calibration

## Frozen Inputs and Scope

calibration uses the 667 complete model selection training episodes for whole-episode bootstrap,
the 143 complete model selection operational inputs for J selection, and only the
population design ordinary/boundary calibration seed blocks for conformal scores. No selection
truth, final, stress, or rare-event truth is read. The frozen mean is the model selection
`k=3` `NO_MODEL_PASSED_PREDECLARED_FALLBACK`; calibration does not repair its
documented mean-bias limitation.

## Bootstrap Parameter Uncertainty

Accepted {BOOTSTRAP_TARGET} complete-episode bootstrap fits from
{len(bootstrap_manifest)} numerical attempts. Rejections were based only on
convergence, covariance positivity, finiteness, and spectral radius below 0.995.
No predictive-performance or test-data rejection occurred. Rejection counts:
`{json.dumps(bootstrap_rejections, sort_keys=True)}`.

## Ensemble Selection

Normalized order is pressure / level / temperature. The smallest candidate
passing center-difference <=0.05 and scale-ratio 0.95-1.05 is selected.

| J | Median normalized center difference vs J100 | Median scale ratio vs J100 | Result |
|---:|---:|---:|---|
{chr(10).join(ensemble_lines)}

Selected `J={ensemble_size}`. Every prediction uses `M=2000` coherent paths;
each path retains one bootstrap model for all 120 horizons and propagates
correlated latent/output noise.

## Pooled Whole-Trajectory Calibration

One score maximizes high pressure, high temperature, high level, and low level
over the full horizon. Quantiles use `ceil((n+1)(1-alpha))`.

| Calibration population | alpha | Complete N | Rank | q |
|---|---:|---:|---:|---:|
{chr(10).join(quantile_lines)}

Reserved-to-complete counts are ordinary 600->{complete_counts['calibration_ordinary']}
and boundary 600->{complete_counts['calibration_boundary']}. Domain exits are
retained in the evaluator manifest and never replaced. Ordinary and nominal-DT
boundary quantiles remain separate as population design required.

## Predictive Scale Diagnostics

| Population | Median sigma_P [bar] | Median sigma_L | Median sigma_T [K] | Mean floor-hit fraction P/L/T |
|---|---:|---:|---:|---:|
{chr(10).join(scale_lines)}

Scale floors are 0.005 bar, 0.005 level fraction, and 0.25 K.

## Action-Conditional Extension

`d(a)=||(a-a_nominal)/(a_max-a_min)||_2` is implemented using only the candidate
action. However, population design did not declare Mondrian bin edges or a minimum group sample
count. Therefore no group was formed and no group-specific q was estimated.
The secondary method is frozen as `UNSUPPORTED_NOT_CERTIFIED` with reason
`BINS_NOT_PREREGISTERED`; inventing bins after calibration would violate the
preregistration.

## Decision Semantics

For a supported pooled quantile, certify `SAFE` only when `P_upper<=0.90 bar`,
`T_upper<=365 K`, `L_upper<=0.85`, and `L_lower>=0.20` at every horizon. All
other cases, including unsupported Mondrian groups, return `NOT CERTIFIED`.
No UNSAFE certificate exists. calibration contains no test result or safety-performance
claim.
"""


def _hash_artifacts() -> None:
    paths = [
        EXPERIMENT_ROOT / "flash_bootstrap_manifest.csv",
        EXPERIMENT_ROOT / "flash_bootstrap_models.json",
        EXPERIMENT_ROOT / "flash_calibration_scores.csv",
        EXPERIMENT_ROOT / "flash_conformal_config.json",
        EXPERIMENT_ROOT / "calibration_report.md",
    ]
    paths.extend(
        sorted(path for path in (EXPERIMENT_ROOT / "calibration_data").rglob("*") if path.is_file())
    )
    lines = ["# Frozen calibration artifact SHA-256 manifest", ""]
    for path in paths:
        relative = path.relative_to(EXPERIMENT_ROOT).as_posix()
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {relative}")
    _safe_path("calibration_artifact_hashes.txt").write_text(
        "\n".join(lines) + "\n", encoding="ascii"
    )


def main(workers: int, reuse_bootstrap: bool) -> None:
    selected_mean, selected_mean_hash = _verify_frozen_inputs()
    if reuse_bootstrap:
        models, bootstrap_manifest = load_bootstrap_models()
    else:
        models, bootstrap_manifest = fit_bootstrap_models(workers)
    ensemble_size, ensemble_diagnostics = select_ensemble_size(models, workers)
    populations, evaluator_rows = generate_calibration_populations(workers)
    calibration_rows, quantiles = calibrate(
        models, populations, ensemble_size, workers
    )
    config = {
        "schema_version": "FLASH_CONFORMAL_CONFIG_1.0",
        "process_bundle_sha256": PROCESS_BUNDLE_SHA256,
        "population_design_preregistration_sha256": POPULATION_DESIGN_SHA256,
        "selected_model_selection_model_sha256": selected_mean_hash,
        "selected_mean_model": {
            "latent_dimension": 3,
            "selection_status": selected_mean["selection_status"],
            "no_model_passed": selected_mean["no_model_passed"],
        },
        "roles_accessed": [
            "training",
            "model_selection_operational_inputs_only",
            "calibration_ordinary",
            "calibration_boundary",
        ],
        "prohibited_truth_accessed": [],
        "bootstrap": {
            "unit": "complete_training_episode",
            "training_episode_count": 667,
            "target_accepted": BOOTSTRAP_TARGET,
            "maximum_attempts": BOOTSTRAP_MAX_ATTEMPTS,
            "accepted": len(models),
            "rejection_rules": [
                "not_converged",
                "nonfinite_parameters",
                "nonpositive_covariance",
                "spectral_radius_not_below_0.995",
            ],
        },
        "ensemble": {
            "candidates": list(ENSEMBLE_CANDIDATES),
            "selected": ensemble_size,
            "predictive_trajectories": PREDICTIVE_TRAJECTORIES,
            "coherent_over_horizon": True,
        },
        "score": {
            "directions": list(DIRECTION_NAMES),
            "formula": "max_h,d direction-specific signed error / max(sigma,epsilon)",
            "scale_floors": PREDICTIVE_SCALE_FLOORS.tolist(),
            "alpha_levels": list(ALPHA_LEVELS),
            "primary_alpha": PRIMARY_ALPHA,
            "finite_sample_rank": "ceil((n_cal+1)*(1-alpha)), capped at n_cal",
        },
        "pooled_quantiles": quantiles,
        "mondrian": {
            "distance_formula": "L2 norm of (a-a_nominal)/(a_max-a_min)",
            "supported": False,
            "bin_edges": None,
            "minimum_group_count": None,
            "groups": [],
            "reason": "BINS_NOT_PREREGISTERED",
            "unsupported_decision": "NOT CERTIFIED",
        },
        "decision_labels": ["SAFE", "NOT CERTIFIED"],
        "unsafe_certificate_defined": False,
    }
    _safe_path("flash_conformal_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = _build_report(
        bootstrap_manifest,
        ensemble_diagnostics,
        ensemble_size,
        calibration_rows,
        quantiles,
        evaluator_rows,
    )
    _safe_path("calibration_report.md").write_text(report, encoding="utf-8")
    _hash_artifacts()
    print(
        f"calibration frozen: accepted={len(models)}, J={ensemble_size}, "
        f"ordinary_q95={quantiles['calibration_ordinary']['0.05']['quantile']:.6f}, "
        f"boundary_q95={quantiles['calibration_boundary']['0.05']['quantile']:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument("--reuse-bootstrap", action="store_true")
    arguments = parser.parse_args()
    main(arguments.workers, arguments.reuse_bootstrap)
