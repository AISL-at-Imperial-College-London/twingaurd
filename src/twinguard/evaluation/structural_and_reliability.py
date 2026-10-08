"""Prospective E7 structural-mismatch and large reliability experiments."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
from math import sqrt
import os
from pathlib import Path
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import beta

from twinguard.cstr import episode_design as b5
from twinguard.cstr import freeze_discrepancy_model as b10
from twinguard.cstr import safety_certification as c1

from twinguard.flash import evaluate_primary as e5
from twinguard.evaluation import boundary_challenge as e6
from twinguard.conformal.flash_conformal import normalized_action_distance
from twinguard.digital_twins.flash_drum import FlashDigitalTwin
from twinguard.discrepancy.flash_state_space import FEATURE_NAMES
from twinguard.scenarios.flash_population_design import (
    MEASURED_OUTPUTS,
    SafetySpecification,
    latin_hypercube_actions,
    purpose_rng,
    purpose_seed,
    validate_candidate_action,
)
from twinguard.plants.flash_drum import FlashAction, FlashParameters, PhysicalStateError
from twinguard.plants.flash_drum_valve_lag import VaporValveLagPlant, sample_vapor_valve_lag
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT, REPOSITORY_ROOT, assert_isolated_output_path
from twinguard.utils.provenance import frozen_sha256, frozen_sha256_if_released
from twinguard.flash.bootstrap_and_calibrate import _boundary_action


STRUCTURAL_COUNT = 600
STRUCTURAL_SEED_START = 28_000_000
FLASH_RELIABILITY_COUNT = 3_000
FLASH_RELIABILITY_SEED_START = 29_000_000
CSTR_RELIABILITY_COUNT = 3_000
CSTR_RELIABILITY_SEED_START = 50_000_000
CSTR_Q_PLUS = 5.464557586938352
CSTR_T_MAX_K = 313.0
DESIGN_PATH = EXPERIMENT_ROOT / "RESS_E7_DESIGN.json"
FROZEN_MANIFEST_PATH = EXPERIMENT_ROOT / "RESS_E7_FROZEN_INPUT_HASHES.txt"


_FLASH_OUTPUT: Path | None = None
_FLASH_ORDINARY_Q: float | None = None
_FLASH_BOUNDARY_Q: float | None = None
_CSTR_OUTPUT: Path | None = None
_CSTR_MODELS = None
_CSTR_TEMPLATE = None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe_path(relative: str) -> Path:
    path = assert_isolated_output_path(EXPERIMENT_ROOT / relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_quantiles() -> tuple[float, float]:
    config = json.loads((EXPERIMENT_ROOT / "flash_conformal_config.json").read_text())
    ordinary = float(config["pooled_quantiles"]["calibration_ordinary"]["0.05"]["quantile"])
    boundary = float(config["pooled_quantiles"]["calibration_boundary"]["0.05"]["quantile"])
    return ordinary, boundary


def _action_strategy(index: int) -> str:
    return "ordinary" if index % 2 == 0 else "boundary"


def _wait_after(path: Path) -> None:
    threshold = path.stat().st_mtime_ns
    while time.time_ns() <= threshold:
        time.sleep(0.001)


def _artifact_paths(root: Path, episode_id: str, prefix: str = "") -> tuple[Path, Path, Path, Path]:
    stem = f"{prefix}{episode_id}"
    return (
        root / "candidates" / f"{stem}.json",
        root / "predictions" / f"{stem}.npz",
        root / "decisions" / f"{stem}.json",
        root / "truths" / f"{stem}.npz",
    )


def _flash_init(output: str) -> None:
    global _FLASH_OUTPUT, _FLASH_ORDINARY_Q, _FLASH_BOUNDARY_Q
    ordinary_q, boundary_q = _load_quantiles()
    frozen = {
        "pooled_q": ordinary_q,
        "selected_mean_model_sha256": _sha256(EXPERIMENT_ROOT / "selected_flash_model.json"),
        "bootstrap_models_sha256": _sha256(EXPERIMENT_ROOT / "flash_bootstrap_models.json"),
        "conformal_config_sha256": _sha256(EXPERIMENT_ROOT / "flash_conformal_config.json"),
    }
    e5._init_worker(output, frozen)
    _FLASH_OUTPUT = Path(output)
    _FLASH_ORDINARY_Q, _FLASH_BOUNDARY_Q = ordinary_q, boundary_q


def _structural_observation(seed: int) -> dict[str, object]:
    lag = sample_vapor_valve_lag(seed)
    nominal_parameters = FlashParameters()
    initial_rng = purpose_rng(seed, 101)
    template = FlashDigitalTwin(nominal_parameters)
    initial_state = template.state_from_nxt(
        float(initial_rng.uniform(550.0, 650.0)),
        float(initial_rng.uniform(0.42, 0.48)),
        float(initial_rng.uniform(348.0, 352.0)),
    )
    twin = FlashDigitalTwin(nominal_parameters, initial_state)
    nominal_action = twin.nominal_action()
    plant = VaporValveLagPlant(
        nominal_parameters,
        initial_state,
        lag.time_constant_s,
        nominal_action.vapor_valve_opening,
    )
    residual = np.full((50, 3), np.nan)
    features = np.full((50, len(FEATURE_NAMES)), np.nan)
    observed_outputs = np.full((50, 3), np.nan)
    nominal_outputs = np.full((50, 3), np.nan)
    noise_rng = purpose_rng(seed, 307)
    sensor_sd = np.asarray([0.002, 0.002, 0.10])
    completed = 0
    domain_exit = ""
    try:
        for second in range(1, 501):
            plant.step(nominal_action, 1.0)
            twin.step(nominal_action, 1.0)
            if second % 10:
                continue
            physical = e5._outputs(plant)
            nominal = e5._outputs(twin)
            observed_outputs[completed] = physical + noise_rng.normal(0.0, sensor_sd)
            nominal_outputs[completed] = nominal
            residual[completed] = observed_outputs[completed] - nominal
            features[completed] = e5._feature(
                nominal_action, nominal_action, twin.outputs(twin.state)
            )
            completed += 1
    except PhysicalStateError as exc:
        domain_exit = str(exc)
    return {
        "family": "F4_unmodelled_vapor_valve_lag",
        "lag": lag,
        "initial_state": initial_state,
        "initial_outputs": template.outputs(initial_state),
        "plant": plant,
        "twin": twin,
        "nominal_action": nominal_action,
        "residual": residual[:completed],
        "features": features[:completed],
        "observed_outputs": observed_outputs[:completed],
        "nominal_outputs": nominal_outputs[:completed],
        "completed": completed,
        "domain_exit": domain_exit,
    }


def _generic_truth_rollout(observation: Mapping[str, object], candidate: FlashAction) -> tuple[np.ndarray, str]:
    if int(observation["completed"]) != 50:
        return np.empty((0, 3)), str(observation["domain_exit"])
    plant = observation["plant"]
    values: list[np.ndarray] = []
    domain_exit = ""
    try:
        for second in range(1, 1201):
            plant.step(candidate, 1.0)
            if second % 10 == 0:
                values.append(e5._outputs(plant))
    except PhysicalStateError as exc:
        domain_exit = str(exc)
    return np.asarray(values), domain_exit


def _reconstruct_flash_decision(prediction: Mapping[str, np.ndarray]) -> tuple[dict[str, str], dict[str, float]]:
    complete = str(prediction["prediction_status"]) == "COMPLETE"
    nominal = np.asarray(prediction["nominal_horizon_outputs"])
    mean = np.asarray(prediction["mean_corrected_horizon_outputs"])
    center = np.asarray(prediction["ensemble_center_horizon_outputs"])
    scale = np.asarray(prediction["ensemble_scale_horizon_outputs"])
    q = float(prediction["pooled_quantile"])
    if complete:
        decisions = {
            "M0_nominal_DT": "CERTIFIED SAFE" if e5._point_margin(nominal) >= 0 else "NOT CERTIFIED",
            "M1_discrepancy_mean": "CERTIFIED SAFE" if e5._point_margin(mean) >= 0 else "NOT CERTIFIED",
            "M2_raw_bootstrap": e5.safety_decision(center, scale, e5.RAW_ONE_SIDED_95).replace("SAFE", "CERTIFIED SAFE"),
            "M3_pooled_conformal": e5.safety_decision(center, scale, q).replace("SAFE", "CERTIFIED SAFE"),
            "M4_action_conditioned": "NOT CERTIFIED",
        }
        margins = {
            "M0_nominal_DT": e5._point_margin(nominal),
            "M1_discrepancy_mean": e5._point_margin(mean),
            "M2_raw_bootstrap": e5._envelope_margin(center, scale, e5.RAW_ONE_SIDED_95),
            "M3_pooled_conformal": e5._envelope_margin(center, scale, q),
            "M4_action_conditioned": float("-inf"),
        }
    else:
        decisions = {method: "NOT CERTIFIED" for method in e5.METHODS}
        margins = {method: float("-inf") for method in e5.METHODS}
    return decisions, margins


def _flash_task(task: tuple[str, int, int, list[float] | None]) -> dict[str, object]:
    if _FLASH_OUTPUT is None or _FLASH_ORDINARY_Q is None or _FLASH_BOUNDARY_Q is None:
        raise RuntimeError("E7 flash worker is not initialized")
    population, index, seed, ordinary_vector = task
    structural = population == "structural"
    episode_id = (
        f"flash_structural_{index:05d}"
        if structural
        else f"flash_reliability_{index:05d}"
    )
    root = _FLASH_OUTPUT / population
    paths = _artifact_paths(root, episode_id)
    if all(path.exists() for path in paths):
        return e6._artifact_audit(episode_id, *paths)

    observation = _structural_observation(seed) if structural else e5._simulate_observation(index, seed)
    strategy = _action_strategy(index)
    if int(observation["completed"]) == 50 and strategy == "boundary":
        candidate, nominal_margin, in_band, eligible_count = _boundary_action(
            observation["twin"].state.as_array(), seed
        )
    elif int(observation["completed"]) == 50:
        if ordinary_vector is None:
            raise RuntimeError("ordinary E7 episode missing its frozen LHS action")
        candidate = validate_candidate_action(ordinary_vector)
        nominal, _, _ = e5._nominal_horizon(
            observation["twin"], candidate, observation["nominal_action"]
        )
        nominal_margin = e5._point_margin(nominal)
        in_band = False
        eligible_count = 0
    else:
        candidate = observation["nominal_action"]
        nominal_margin = float("-inf")
        in_band = False
        eligible_count = 0
    candidate = candidate if isinstance(candidate, FlashAction) else FlashAction.from_vector(candidate)
    q = _FLASH_BOUNDARY_Q if strategy == "boundary" else _FLASH_ORDINARY_Q
    candidate_path, prediction_path, decision_path, truth_path = paths
    if not candidate_path.exists():
        payload = {
            "schema_version": "RESS_E7_FLASH_CANDIDATE_1.0",
            "episode_id": episode_id,
            "population": population,
            "action_selection_strategy": strategy,
            "candidate_action": candidate.as_array().tolist(),
            "nominal_dt_margin": nominal_margin,
            "within_boundary_band": in_band,
            "boundary_eligible_count": eligible_count,
            "selection_source": "nominal_DT_only" if strategy == "boundary" else "frozen_centered_LHS",
            "matching_calibration_population": f"calibration_{strategy}",
            "frozen_pooled_quantile": q,
        }
        e5._exclusive_write(candidate_path, e5._json_bytes(payload))

    created_prediction = False
    metrics: dict[str, float] = {}
    if not prediction_path.exists():
        e5._POOLED_Q = q
        prediction, decisions, metrics = e5._build_prediction(
            observation, candidate, episode_id, seed
        )
        prediction["schema_version"] = np.asarray("RESS_E7_FLASH_PREDICTION_1.0")
        _wait_after(candidate_path)
        e5._exclusive_write(prediction_path, e5._npz_bytes(prediction))
        created_prediction = True
    prediction = e5._load_npz(prediction_path)
    decisions, reconstructed_margins = _reconstruct_flash_decision(prediction)
    if not decision_path.exists():
        latencies = {
            method: float(metrics.get(f"latency_{method}", 0.0))
            for method in e5.METHODS
        }
        payload = {
            "schema_version": "RESS_E7_FLASH_DECISION_1.0",
            "episode_id": episode_id,
            "prediction_sha256": _sha256(prediction_path),
            "decisions": decisions,
            "normalized_margins": reconstructed_margins,
            "raw_multiplier": e5.RAW_ONE_SIDED_95,
            "pooled_quantile": q,
            "mondrian_status": "UNSUPPORTED_NOT_CERTIFIED:E2_BINS_NOT_PREDECLARED",
            "latency_seconds": latencies,
        }
        e5._validate_pretruth_payload(payload, e5.DECISION_ALLOWED_KEYS)
        _wait_after(prediction_path)
        e5._exclusive_write(decision_path, e5._json_bytes(payload))
    decision = json.loads(decision_path.read_text())
    if decision["prediction_sha256"] != _sha256(prediction_path):
        raise RuntimeError("E7 flash decision does not bind prediction")

    if not truth_path.exists():
        actual, truth_exit = _generic_truth_rollout(observation, candidate)
        if structural:
            lag = observation["lag"]
            evaluator = {
                "fault_family_evaluator_only": np.asarray("F4_unmodelled_vapor_valve_lag"),
                "fault_severity_evaluator_only": np.asarray(lag.normalized_severity),
                "fault_severity_label_evaluator_only": np.asarray(
                    "low" if lag.normalized_severity < 1 / 3 else "medium" if lag.normalized_severity < 2 / 3 else "high"
                ),
                "fault_signed_magnitude_evaluator_only": np.asarray(lag.time_constant_s),
                "changed_parameter_evaluator_only": np.asarray("NOT_APPLICABLE_STRUCTURAL_MECHANISM"),
                "nominal_parameter_evaluator_only": np.asarray(np.nan),
                "faulty_parameter_evaluator_only": np.asarray(np.nan),
                "structural_mechanism_evaluator_only": np.asarray("first_order_vapor_valve_actuator_lag"),
                "actuator_time_constant_s_evaluator_only": np.asarray(lag.time_constant_s),
            }
        else:
            hidden = observation["hidden"]
            evaluator = {
                "fault_family_evaluator_only": np.asarray(hidden.family.value),
                "fault_severity_evaluator_only": np.asarray(hidden.normalized_severity),
                "fault_severity_label_evaluator_only": np.asarray(hidden.severity_label),
                "fault_signed_magnitude_evaluator_only": np.asarray(hidden.signed_magnitude),
                "changed_parameter_evaluator_only": np.asarray(hidden.changed_parameter),
                "nominal_parameter_evaluator_only": np.asarray(hidden.nominal_value),
                "faulty_parameter_evaluator_only": np.asarray(hidden.faulty_value),
            }
        payload = {
            "schema_version": np.asarray("RESS_E7_FLASH_TRUTH_1.0"),
            "episode_id": np.asarray(episode_id),
            "episode_seed_evaluator_only": np.asarray(seed),
            **evaluator,
            "initial_state_evaluator_only": observation["initial_state"].as_array(),
            "initial_outputs_evaluator_only": np.asarray(
                [observation["initial_outputs"][name] for name in MEASURED_OUTPUTS]
            ),
            "actual_horizon_outputs": actual,
            "completed_horizon_samples": np.asarray(len(actual)),
            "truth_domain_exit_evaluator_only": np.asarray(truth_exit),
            "prediction_sha256": np.asarray(_sha256(prediction_path)),
            "decision_sha256": np.asarray(_sha256(decision_path)),
        }
        _wait_after(decision_path)
        e5._exclusive_write(truth_path, e5._npz_bytes(payload))
    return e6._artifact_audit(episode_id, *paths)


def _flash_tasks(population: str, count: int, seed_start: int) -> list[tuple[str, int, int, list[float] | None]]:
    ordinary_count = count // 2
    lhs = latin_hypercube_actions(ordinary_count, purpose_seed(seed_start, 401))
    tasks = []
    ordinary_index = 0
    for index in range(count):
        vector = lhs[ordinary_index].tolist() if _action_strategy(index) == "ordinary" else None
        if vector is not None:
            ordinary_index += 1
        tasks.append((population, index, seed_start + index, vector))
    return tasks


def run_flash_population(population: str, count: int, seed_start: int, workers: int) -> list[dict[str, object]]:
    root = _safe_path(f"e7_artifacts/{population}")
    root.mkdir(parents=True, exist_ok=True)
    audit: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_flash_init, initargs=(str(root),)
    ) as pool:
        for completed, row in enumerate(
            pool.map(_flash_task, _flash_tasks(population, count, seed_start), chunksize=1), 1
        ):
            audit.append(row)
            if completed % 100 == 0:
                print(f"E7 {population}={completed}/{count}", flush=True)
    audit.sort(key=lambda row: str(row["episode_id"]))
    _write_csv(_safe_path(f"e7_artifacts/{population.upper()}_ARTIFACT_AUDIT.csv"), audit)
    return audit


def evaluate_flash_population(
    population: str, audit: Sequence[Mapping[str, object]]
) -> list[dict[str, object]]:
    rows = e5.evaluate_artifacts(audit)
    candidate_paths = {
        str(item["episode_id"]): Path(str(item["candidate_path"])) for item in audit
    }
    for row in rows:
        candidate = json.loads(candidate_paths[str(row["episode_id"])].read_text())
        row["population"] = population
        row["action_selection_strategy"] = candidate["action_selection_strategy"]
        row["matching_calibration_population"] = candidate["matching_calibration_population"]
        row["frozen_pooled_quantile"] = candidate["frozen_pooled_quantile"]
    return rows


def _cstr_init(output: str) -> None:
    global _CSTR_OUTPUT, _CSTR_MODELS, _CSTR_TEMPLATE
    _CSTR_OUTPUT = Path(output)
    _CSTR_MODELS = c1._load_models(REPOSITORY_ROOT / "artifacts/cstr/bootstrap_ensemble", 10)
    _CSTR_TEMPLATE = b5.build_template(b5.MilestoneB5Config())


def _cstr_task(task: tuple[int, int]) -> dict[str, object]:
    if _CSTR_OUTPUT is None or _CSTR_MODELS is None or _CSTR_TEMPLATE is None:
        raise RuntimeError("E7 CSTR worker is not initialized")
    index, seed = task
    category = b10.CATEGORIES[index // (CSTR_RELIABILITY_COUNT // 3)]
    episode_id = f"cstr_reliability_{index:05d}_{category}"
    paths = _artifact_paths(_CSTR_OUTPUT, episode_id, "CSTR_RESS_RELIABILITY_")
    if all(path.exists() for path in paths):
        return e6._artifact_audit(episode_id, *paths)
    candidate_path, prediction_path, decision_path, truth_path = paths
    nominal, plant, residual, online, previous, magnitude = c1._reconstruct_decision_state(
        seed, b5.MilestoneB5Config(), _CSTR_TEMPLATE
    )
    action = float(
        b5._select_action_after_fit(category, np.random.default_rng(seed + 2_000_000))
    )
    if not candidate_path.exists():
        payload = {
            "schema_version": "CSTR_RESS_RELIABILITY_CANDIDATE_1.0",
            "episode_id": episode_id,
            "action_lps": action,
            "action_category": category,
            "selection_source": "frozen_category_sampler",
        }
        e5._exclusive_write(candidate_path, e5._json_bytes(payload))
    branch = copy.deepcopy(nominal)
    nominal_future = np.empty(c1.HORIZON)
    future = np.empty((c1.HORIZON, 4))
    prior = previous
    for horizon in range(c1.HORIZON):
        nominal_future[horizon] = float(
            branch.simulate_step(Fin_sp_normal=action, Fin_sp_startup=action)["T_true"]
        )
        future[horizon] = [action, action - prior, nominal_future[horizon], 51 + horizon]
        prior = action
    item = b10.EpisodeInputs(
        episode_id,
        category,
        seed,
        residual,
        online,
        future,
        nominal_future,
        action,
        truth_path,
        plant,
        seed + 4_000_000,
    )
    if not prediction_path.exists():
        _wait_after(candidate_path)
        c1.predict_frozen_u3(item, _CSTR_MODELS, 2000, prediction_path)
    with np.load(prediction_path, allow_pickle=False) as saved:
        nominal_saved = np.asarray(saved["nominal_temperature_k"])
        center = np.asarray(saved["predictive_center_k"])
        scale = np.maximum(np.asarray(saved["predictive_scale_k"]), c1.EPSILON_SCALE_K)
    if not decision_path.exists():
        upper = center + CSTR_Q_PLUS * scale
        payload = {
            "schema_version": "CSTR_RESS_RELIABILITY_DECISION_1.0",
            "episode_id": episode_id,
            "prediction_sha256": _sha256(prediction_path),
            "nominal_dt_decision": "CERTIFIED SAFE" if float(np.max(nominal_saved)) <= CSTR_T_MAX_K else "NOT CERTIFIED",
            "frozen_cstr_decision": "CERTIFIED SAFE" if float(np.max(upper)) <= CSTR_T_MAX_K else "NOT CERTIFIED",
            "q_plus": CSTR_Q_PLUS,
            "temperature_limit_k": CSTR_T_MAX_K,
            "nominal_margin_k": CSTR_T_MAX_K - float(np.max(nominal_saved)),
            "conformal_margin_k": CSTR_T_MAX_K - float(np.max(upper)),
        }
        _wait_after(prediction_path)
        e5._exclusive_write(decision_path, e5._json_bytes(payload))
    if not truth_path.exists():
        _wait_after(decision_path)
        actual = c1.reveal_physical_temperature(item, prediction_path)
        with np.load(truth_path, allow_pickle=False) as saved:
            payload = {key: np.asarray(saved[key]) for key in saved.files}
        payload.update(
            {
                "schema_version": np.asarray("CSTR_RESS_RELIABILITY_TRUTH_1.0"),
                "episode_seed_evaluator_only": np.asarray(seed),
                "hidden_ua_fouling_magnitude_evaluator_only": np.asarray(magnitude),
                "actual_physical_temperature_k": actual,
                "prediction_sha256": np.asarray(_sha256(prediction_path)),
                "decision_sha256": np.asarray(_sha256(decision_path)),
            }
        )
        np.savez_compressed(truth_path, **payload)
    return e6._artifact_audit(episode_id, *paths)


def run_cstr_reliability(workers: int) -> list[dict[str, object]]:
    root = _safe_path("e7_artifacts/cstr_reliability")
    root.mkdir(parents=True, exist_ok=True)
    tasks = [
        (index, CSTR_RELIABILITY_SEED_START + index)
        for index in range(CSTR_RELIABILITY_COUNT)
    ]
    audit: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_cstr_init, initargs=(str(root),)
    ) as pool:
        for completed, row in enumerate(pool.map(_cstr_task, tasks, chunksize=1), 1):
            audit.append(row)
            if completed % 100 == 0:
                print(f"E7 cstr_reliability={completed}/{CSTR_RELIABILITY_COUNT}", flush=True)
    audit.sort(key=lambda row: str(row["episode_id"]))
    _write_csv(_safe_path("e7_artifacts/CSTR_RELIABILITY_ARTIFACT_AUDIT.csv"), audit)
    return audit


def evaluate_cstr(audit: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in audit:
        candidate = json.loads(Path(str(item["candidate_path"])).read_text())
        decision = json.loads(Path(str(item["decision_path"])).read_text())
        with np.load(Path(str(item["prediction_path"])), allow_pickle=False) as saved:
            nominal = np.asarray(saved["nominal_temperature_k"])
            center = np.asarray(saved["predictive_center_k"])
            scale = np.maximum(np.asarray(saved["predictive_scale_k"]), c1.EPSILON_SCALE_K)
        with np.load(Path(str(item["truth_path"])), allow_pickle=False) as saved:
            actual = np.asarray(saved["actual_physical_temperature_k"])
            seed = int(saved["episode_seed_evaluator_only"])
            magnitude = float(saved["hidden_ua_fouling_magnitude_evaluator_only"])
        actual_safe = bool(float(np.max(actual)) <= CSTR_T_MAX_K)
        severity = np.maximum(actual - CSTR_T_MAX_K, 0.0)
        for method, predicted, certified, margin, covered in (
            (
                "nominal_DT",
                nominal,
                decision["nominal_dt_decision"] == "CERTIFIED SAFE",
                decision["nominal_margin_k"],
                bool(np.all(actual <= nominal)),
            ),
            (
                "frozen_CSTR_conformal",
                center,
                decision["frozen_cstr_decision"] == "CERTIFIED SAFE",
                decision["conformal_margin_k"],
                bool(np.all(actual <= center + CSTR_Q_PLUS * scale)),
            ),
        ):
            rows.append(
                {
                    "episode_id": item["episode_id"],
                    "population": "cstr_reliability",
                    "method": method,
                    "episode_seed_evaluator_only": seed,
                    "fault_family_evaluator_only": "UA_fouling",
                    "fault_magnitude_evaluator_only": magnitude,
                    "action_category": candidate["action_category"],
                    "action_lps": candidate["action_lps"],
                    "actual_safe": actual_safe,
                    "actual_unsafe": not actual_safe,
                    "certified_safe": certified,
                    "false_safe": certified and not actual_safe,
                    "true_safety_margin_k": CSTR_T_MAX_K - float(np.max(actual)),
                    "predicted_safety_margin_k": margin,
                    "severity_max_k": float(np.max(severity)),
                    "severity_integrated_k_s": float(np.sum(severity)),
                    "prediction_rmse_k": float(np.sqrt(np.mean((predicted - actual) ** 2))),
                    "upper_trajectory_covered": covered,
                    "prediction_sha256": item["prediction_sha256"],
                    "decision_sha256": item["decision_sha256"],
                    "truth_sha256": item["truth_sha256"],
                    "chronology_verified": item["chronology_verified"],
                }
            )
    return rows


def _intervals(successes: int, count: int) -> dict[str, object]:
    if count <= 0:
        return {
            "estimate": "",
            "exact_95_lower": "",
            "exact_95_upper": "",
            "wilson_95_lower": "",
            "wilson_95_upper": "",
        }
    alpha = 0.05
    exact_lower = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, count - successes + 1))
    exact_upper = 1.0 if successes == count else float(beta.ppf(1 - alpha / 2, successes + 1, count - successes))
    z = 1.959963984540054
    estimate = successes / count
    denominator = 1 + z * z / count
    center = (estimate + z * z / (2 * count)) / denominator
    radius = z / denominator * sqrt(estimate * (1 - estimate) / count + z * z / (4 * count * count))
    return {
        "estimate": estimate,
        "exact_95_lower": exact_lower,
        "exact_95_upper": exact_upper,
        "wilson_95_lower": max(0.0, center - radius),
        "wilson_95_upper": min(1.0, center + radius),
    }


def summarize(rows: Sequence[Mapping[str, object]], population: str) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        selected = [row for row in rows if row["method"] == method]
        count = len(selected)
        certified = sum(bool(row["certified_safe"]) for row in selected)
        false_safe = sum(bool(row["false_safe"]) for row in selected)
        safe = sum(bool(row["actual_safe"]) for row in selected)
        safe_certified = sum(
            bool(row["actual_safe"]) and bool(row["certified_safe"]) for row in selected
        )
        joint = _intervals(false_safe, count)
        conditional = _intervals(false_safe, certified)
        coverage_key = (
            "whole_trajectory_safety_direction_covered"
            if "whole_trajectory_safety_direction_covered" in selected[0]
            else "upper_trajectory_covered"
        )
        coverage = [
            bool(row[coverage_key])
            for row in selected
            if row[coverage_key] != ""
        ]
        false_severity_key = "severity_max" if "severity_max" in selected[0] else "severity_max_k"
        false_integrated_key = "severity_integrated_s" if "severity_integrated_s" in selected[0] else "severity_integrated_k_s"
        false_rows = [row for row in selected if bool(row["false_safe"])]
        summaries.append(
            {
                "population": population,
                "method": method,
                "episode_count": count,
                "actual_safe_count": safe,
                "actual_unsafe_count": count - safe,
                "certified_safe_count": certified,
                "certification_rate": certified / count,
                "not_certified_rate": 1 - certified / count,
                "truly_safe_action_certification_rate": safe_certified / safe if safe else "",
                "false_safe_count": false_safe,
                "joint_false_safe_probability": joint["estimate"],
                "joint_false_safe_exact_95_lower": joint["exact_95_lower"],
                "joint_false_safe_exact_95_upper": joint["exact_95_upper"],
                "joint_false_safe_wilson_95_lower": joint["wilson_95_lower"],
                "joint_false_safe_wilson_95_upper": joint["wilson_95_upper"],
                "conditional_false_safe_probability": conditional["estimate"],
                "conditional_false_safe_exact_95_lower": conditional["exact_95_lower"],
                "conditional_false_safe_exact_95_upper": conditional["exact_95_upper"],
                "conditional_false_safe_wilson_95_lower": conditional["wilson_95_lower"],
                "conditional_false_safe_wilson_95_upper": conditional["wilson_95_upper"],
                "trajectory_coverage": float(np.mean(coverage)) if coverage else "",
                "maximum_false_safe_severity": max(
                    (float(row[false_severity_key]) for row in false_rows), default=0.0
                ),
                "total_false_safe_severity": sum(
                    float(row[false_severity_key]) for row in false_rows
                ),
                "maximum_false_safe_integrated_severity": max(
                    (float(row[false_integrated_key]) for row in false_rows), default=0.0
                ),
                "total_false_safe_integrated_severity": sum(
                    float(row[false_integrated_key]) for row in false_rows
                ),
            }
        )
    return summaries


def _verify_design() -> dict[str, object]:
    if not DESIGN_PATH.exists() or not FROZEN_MANIFEST_PATH.exists():
        raise RuntimeError("freeze the E7 design before generating reserved truth")
    design = json.loads(DESIGN_PATH.read_text())
    if not design["frozen_before_truth"]:
        raise RuntimeError("E7 design is not marked frozen")
    if _sha256(FROZEN_MANIFEST_PATH) != design["frozen_input_manifest_sha256"]:
        raise RuntimeError("E7 frozen-input manifest changed")
    for relative, expected in design["e7_source_hashes"].items():
        if frozen_sha256(relative) != expected:
            raise RuntimeError(f"E7 source changed after design freeze: {relative}")
    for line in FROZEN_MANIFEST_PATH.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", 1)
        observed = frozen_sha256_if_released(relative)
        if observed is not None and observed != expected:
            raise RuntimeError(f"previous frozen result changed: {relative}")
    return design


def run_all(workers: int) -> None:
    _verify_design()
    structural_audit = run_flash_population(
        "structural", STRUCTURAL_COUNT, STRUCTURAL_SEED_START, workers
    )
    structural_rows = evaluate_flash_population("structural", structural_audit)
    _write_csv(_safe_path("FLASH_STRUCTURAL_RESULTS.csv"), structural_rows)
    _write_csv(
        _safe_path("FLASH_STRUCTURAL_SUMMARY.csv"),
        summarize(structural_rows, "flash_structural"),
    )

    flash_audit = run_flash_population(
        "flash_reliability",
        FLASH_RELIABILITY_COUNT,
        FLASH_RELIABILITY_SEED_START,
        workers,
    )
    flash_rows = evaluate_flash_population("flash_reliability", flash_audit)
    _write_csv(_safe_path("FLASH_RELIABILITY_RESULTS.csv"), flash_rows)
    _write_csv(
        _safe_path("FLASH_RELIABILITY_SUMMARY.csv"),
        summarize(flash_rows, "flash_reliability"),
    )

    cstr_audit = run_cstr_reliability(workers)
    cstr_rows = evaluate_cstr(cstr_audit)
    _write_csv(_safe_path("CSTR_RELIABILITY_RESULTS.csv"), cstr_rows)
    _write_csv(
        _safe_path("CSTR_RELIABILITY_SUMMARY.csv"),
        summarize(cstr_rows, "cstr_reliability"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    args = parser.parse_args()
    run_all(args.workers)
