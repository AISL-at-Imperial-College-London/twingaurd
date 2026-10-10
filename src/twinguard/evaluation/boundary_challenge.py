"""Boundary-reliability challenge for frozen flash and CSTR methods."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import csv
import hashlib
import json
from math import ceil, sqrt
import os
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from twinguard.cstr import episode_design as episode_design
from twinguard.cstr import freeze_discrepancy_model as discrepancy_freeze
from twinguard.cstr import safety_certification as certification

from twinguard.flash import evaluate_primary as primary
from twinguard.uncertainty.flash_bootstrap import PREDICTIVE_SCALE_FLOORS, coherent_predictive_summary
from twinguard.conformal.flash_conformal import conformal_envelope, normalized_action_distance, safety_decision
from twinguard.discrepancy.flash_state_space import parameters_from_payload, predict_discrepancy
from twinguard.scenarios.flash_population_design import (
    MEASURED_OUTPUTS,
    PRIMARY_FAULT_FAMILIES,
    SafetySpecification,
    purpose_seed,
)
from twinguard.plants.flash_drum import FlashAction
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT, REPOSITORY_ROOT, assert_isolated_output_path
from twinguard.flash.bootstrap_and_calibrate import _boundary_action


DESIGN_PATH = EXPERIMENT_ROOT / "boundary_challenge_design.json"
FLASH_COUNT = 1200
FLASH_SEED_START = 27_000_000
CSTR_COUNT = 300
CSTR_SEED_START = 9_300_000
CONFIDENCE_LEVELS = (0.90, 0.925, 0.95, 0.975, 0.99)
RAW_MULTIPLIER = 1.6448536269514722
CSTR_ACTIONS = np.linspace(0.012, 0.045, 129)
CSTR_BOUNDARY_DELTA_K = 0.25
CSTR_Q_PLUS = 5.464557586938352
CSTR_T_MAX_K = 313.0


_FLASH_MODELS = None
_FLASH_MEAN_MODEL = None
_FLASH_Q: dict[float, float] | None = None
_FLASH_OUTPUT: Path | None = None
_CSTR_MODELS = None
_CSTR_TEMPLATE = None
_CSTR_OUTPUT: Path | None = None


def _safe_path(relative: str) -> Path:
    path = assert_isolated_output_path(EXPERIMENT_ROOT / relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_design() -> dict[str, object]:
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    if not design["frozen_before_boundary_truth"]:
        raise RuntimeError("boundary challenge design was not frozen before truth")
    hashes = design["frozen_source_hashes"]
    sources = {
        "flash_primary_hash_manifest_sha256": EXPERIMENT_ROOT / "primary_evaluation_artifact_hashes.txt",
        "cstr_one_sided_quantiles_sha256": REPOSITORY_ROOT / "artifacts/cstr/primary_evaluation/one_sided_quantiles.csv",
        "cstr_safety_calibration_scores_sha256": REPOSITORY_ROOT / "artifacts/cstr/primary_evaluation/safety_calibration_scores.csv",
        "cstr_bootstrap_models_sha256": REPOSITORY_ROOT / "artifacts/cstr/bootstrap_ensemble/bootstrap_models.json",
    }
    for key, path in sources.items():
        if _sha256(path) != hashes[key]:
            raise RuntimeError(f"frozen boundary challenge source hash mismatch: {path}")
    primary.verify_frozen_inputs()
    for line in (EXPERIMENT_ROOT / "primary_evaluation_artifact_hashes.txt").read_text().splitlines():
        if line and not line.startswith("#"):
            expected, relative = line.split("  ", maxsplit=1)
            if _sha256(EXPERIMENT_ROOT / relative) != expected:
                raise RuntimeError(f"primary evaluation artifact changed before boundary challenge: {relative}")
    return design


def boundary_quantiles_from_calibration() -> dict[float, float]:
    with (EXPERIMENT_ROOT / "flash_calibration_scores.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        scores = sorted(
            float(row["whole_trajectory_safety_score"])
            for row in csv.DictReader(handle)
            if row["calibration_population"] == "calibration_boundary"
        )
    if len(scores) != 463:
        raise RuntimeError("frozen boundary calibration count mismatch")
    result = {}
    for confidence in CONFIDENCE_LEVELS:
        rank = min(ceil((len(scores) + 1) * confidence), len(scores))
        result[confidence] = scores[rank - 1]
    return result


def _wilson(successes: int, count: int) -> tuple[float, float]:
    if not count:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / count
    denominator = 1 + z * z / count
    center = (p + z * z / (2 * count)) / denominator
    radius = z / denominator * sqrt(p * (1 - p) / count + z * z / (4 * count * count))
    return max(0.0, center - radius), min(1.0, center + radius)


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty boundary challenge CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _wait_after(path: Path) -> None:
    threshold = path.stat().st_mtime_ns
    while time.time_ns() <= threshold:
        time.sleep(0.001)


def _artifact_audit(
    episode_id: str, candidate: Path, prediction: Path, decision: Path, truth: Path
) -> dict[str, object]:
    paths = (candidate, prediction, decision, truth)
    timestamps = [path.stat().st_mtime_ns for path in paths]
    if not timestamps[0] < timestamps[1] < timestamps[2] < timestamps[3]:
        raise RuntimeError(f"boundary challenge chronology failure: {episode_id}")
    decision_payload = json.loads(decision.read_text(encoding="utf-8"))
    with np.load(truth, allow_pickle=False) as revealed:
        if decision_payload["prediction_sha256"] != _sha256(prediction):
            raise RuntimeError("decision prediction hash mismatch")
        if str(revealed["prediction_sha256"]) != _sha256(prediction):
            raise RuntimeError("truth prediction hash mismatch")
        if str(revealed["decision_sha256"]) != _sha256(decision):
            raise RuntimeError("truth decision hash mismatch")
    return {
        "episode_id": episode_id,
        "candidate_path": candidate.as_posix(),
        "prediction_path": prediction.as_posix(),
        "decision_path": decision.as_posix(),
        "truth_path": truth.as_posix(),
        "candidate_sha256": _sha256(candidate),
        "prediction_sha256": _sha256(prediction),
        "decision_sha256": _sha256(decision),
        "truth_sha256": _sha256(truth),
        "candidate_saved_ns": timestamps[0],
        "prediction_saved_ns": timestamps[1],
        "decision_saved_ns": timestamps[2],
        "truth_revealed_ns": timestamps[3],
        "chronology_verified": True,
    }


def _flash_init(output: str, quantiles: Mapping[float, float]) -> None:
    global _FLASH_MODELS, _FLASH_MEAN_MODEL, _FLASH_Q, _FLASH_OUTPUT
    selected = json.loads((EXPERIMENT_ROOT / "selected_flash_model.json").read_text())
    bootstraps = json.loads((EXPERIMENT_ROOT / "flash_bootstrap_models.json").read_text())
    _FLASH_MEAN_MODEL = parameters_from_payload(selected["model"])
    _FLASH_MODELS = [parameters_from_payload(item) for item in bootstraps]
    _FLASH_Q = {float(key): float(value) for key, value in quantiles.items()}
    _FLASH_OUTPUT = Path(output)


def _flash_paths(episode_id: str) -> tuple[Path, Path, Path, Path]:
    if _FLASH_OUTPUT is None:
        raise RuntimeError("flash boundary challenge worker not initialized")
    return tuple(
        _FLASH_OUTPUT / folder / f"{episode_id}.{extension}"
        for folder, extension in (
            ("candidates", "json"),
            ("predictions", "npz"),
            ("decisions", "json"),
            ("truths", "npz"),
        )
    )


def _flash_point_margin(values: np.ndarray) -> float:
    if len(values) != 120:
        return float("-inf")
    return SafetySpecification().trajectory_margin(
        {name: values[:, index] for index, name in enumerate(MEASURED_OUTPUTS)}
    )


def _flash_envelope_margin(center: np.ndarray, scale: np.ndarray, q: float) -> float:
    if len(center) != 120:
        return float("-inf")
    envelope = conformal_envelope(center, scale, q)
    safety = SafetySpecification()
    violation = np.column_stack(
        [
            (envelope.pressure_upper - safety.pressure_max_bar) / safety.pressure_scale_bar,
            (envelope.level_upper - safety.level_max_fraction) / safety.level_scale_fraction,
            (safety.level_min_fraction - envelope.level_lower) / safety.level_scale_fraction,
            (envelope.temperature_upper - safety.temperature_max_k) / safety.temperature_scale_k,
        ]
    )
    return -float(np.max(violation))


def _flash_task(task: tuple[int, int]) -> dict[str, object]:
    if _FLASH_MODELS is None or _FLASH_MEAN_MODEL is None or _FLASH_Q is None:
        raise RuntimeError("flash boundary challenge predictor missing")
    index, seed = task
    episode_id = f"flash_boundary_{index:05d}"
    candidate_path, prediction_path, decision_path, truth_path = _flash_paths(episode_id)
    if all(path.exists() for path in (candidate_path, prediction_path, decision_path, truth_path)):
        return _artifact_audit(episode_id, candidate_path, prediction_path, decision_path, truth_path)

    observation = primary._simulate_observation(index, seed)
    if int(observation["completed"]) == 50:
        selected, nominal_selection_margin, boundary_found, eligible_count = _boundary_action(
            observation["twin"].state.as_array(), seed
        )
    else:
        selected = observation["nominal_action"]
        nominal_selection_margin = float("-inf")
        boundary_found = False
        eligible_count = 0
    candidate = selected if isinstance(selected, FlashAction) else FlashAction.from_vector(selected)
    if not candidate_path.exists():
        candidate_payload = {
            "schema_version": "FLASH_BOUNDARY_CANDIDATE_1.0",
            "episode_id": episode_id,
            "candidate_action": candidate.as_array().tolist(),
            "nominal_dt_margin": nominal_selection_margin,
            "within_predeclared_boundary_band": boundary_found,
            "nominal_dt_boundary_eligible_count": eligible_count,
            "candidate_pool_size": 153,
            "selection_source": "nominal_DT_only",
        }
        primary._exclusive_write(candidate_path, primary._json_bytes(candidate_payload))

    if not prediction_path.exists():
        nominal, features, prediction_exit = primary._nominal_horizon(
            observation["twin"], candidate, observation["nominal_action"]
        ) if int(observation["completed"]) == 50 else (np.empty((0, 3)), np.empty((0, 11)), str(observation["domain_exit"]))
        if len(nominal) == 120:
            discrepancy, _ = predict_discrepancy(
                _FLASH_MEAN_MODEL,
                np.asarray(observation["residual"]),
                np.asarray(observation["features"]),
                features,
            )
            mean_center = nominal + discrepancy
            probabilistic = coherent_predictive_summary(
                _FLASH_MODELS,
                np.asarray(observation["residual"]),
                np.asarray(observation["features"]),
                features,
                nominal,
                ensemble_size=10,
                trajectory_count=2000,
                seed=purpose_seed(seed, 601),
            )
            center, scale, status = probabilistic.center, probabilistic.scale, "COMPLETE"
        else:
            mean_center = center = scale = np.empty((0, 3))
            status = "PREDICTION_DOMAIN_EXIT"
        prediction_payload = {
            "schema_version": np.asarray("FLASH_BOUNDARY_PREDICTION_1.0"),
            "episode_id": np.asarray(episode_id),
            "candidate_action": candidate.as_array(),
            "observed_residual": np.asarray(observation["residual"]),
            "observed_features": np.asarray(observation["features"]),
            "prediction_status": np.asarray(status),
            "prediction_domain_exit": np.asarray(prediction_exit),
            "nominal_horizon_outputs": nominal,
            "mean_horizon_outputs": mean_center,
            "ensemble_center_horizon_outputs": center,
            "ensemble_scale_horizon_outputs": scale,
            "confidence_levels": np.asarray(CONFIDENCE_LEVELS),
            "boundary_quantiles": np.asarray([_FLASH_Q[level] for level in CONFIDENCE_LEVELS]),
            "raw_multiplier": np.asarray(RAW_MULTIPLIER),
            "ensemble_size": np.asarray(10),
            "predictive_trajectory_count": np.asarray(2000),
        }
        _wait_after(candidate_path)
        primary._exclusive_write(prediction_path, primary._npz_bytes(prediction_payload))
    prediction = primary._load_npz(prediction_path)

    if not decision_path.exists():
        complete = str(prediction["prediction_status"]) == "COMPLETE"
        nominal = np.asarray(prediction["nominal_horizon_outputs"])
        mean_center = np.asarray(prediction["mean_horizon_outputs"])
        center = np.asarray(prediction["ensemble_center_horizon_outputs"])
        scale = np.asarray(prediction["ensemble_scale_horizon_outputs"])
        decisions = {
            "nominal_DT": "CERTIFIED SAFE" if complete and _flash_point_margin(nominal) >= 0 else "NOT CERTIFIED",
            "discrepancy_mean": "CERTIFIED SAFE" if complete and _flash_point_margin(mean_center) >= 0 else "NOT CERTIFIED",
            "raw_bootstrap_95": (
                safety_decision(center, scale, RAW_MULTIPLIER).replace("SAFE", "CERTIFIED SAFE")
                if complete else "NOT CERTIFIED"
            ),
            "action_conditioned": "NOT CERTIFIED",
        }
        margins = {
            "nominal_DT": _flash_point_margin(nominal),
            "discrepancy_mean": _flash_point_margin(mean_center),
            "raw_bootstrap_95": _flash_envelope_margin(center, scale, RAW_MULTIPLIER),
            "action_conditioned": float("-inf"),
        }
        for level in CONFIDENCE_LEVELS:
            label = f"pooled_{level:.3f}"
            decisions[label] = (
                safety_decision(center, scale, _FLASH_Q[level]).replace("SAFE", "CERTIFIED SAFE")
                if complete else "NOT CERTIFIED"
            )
            margins[label] = _flash_envelope_margin(center, scale, _FLASH_Q[level])
        decision_payload = {
            "schema_version": "FLASH_BOUNDARY_DECISION_1.0",
            "episode_id": episode_id,
            "prediction_sha256": _sha256(prediction_path),
            "decisions": decisions,
            "normalized_margins": margins,
            "action_conditioned_status": "UNSUPPORTED_NOT_CERTIFIED:BINS_NOT_PREREGISTERED",
        }
        _wait_after(prediction_path)
        primary._exclusive_write(decision_path, primary._json_bytes(decision_payload))

    if not truth_path.exists():
        actual, truth_exit = primary._truth_rollout(observation, candidate)
        hidden = observation["hidden"]
        truth_payload = {
            "schema_version": np.asarray("FLASH_BOUNDARY_TRUTH_1.0"),
            "episode_id": np.asarray(episode_id),
            "episode_seed_evaluator_only": np.asarray(seed),
            "fault_family_evaluator_only": np.asarray(hidden.family.value),
            "fault_severity_evaluator_only": np.asarray(hidden.normalized_severity),
            "fault_severity_label_evaluator_only": np.asarray(hidden.severity_label),
            "actual_horizon_outputs": actual,
            "completed_horizon_samples": np.asarray(len(actual)),
            "truth_domain_exit_evaluator_only": np.asarray(truth_exit),
            "prediction_sha256": np.asarray(_sha256(prediction_path)),
            "decision_sha256": np.asarray(_sha256(decision_path)),
        }
        _wait_after(decision_path)
        primary._exclusive_write(truth_path, primary._npz_bytes(truth_payload))
    return _artifact_audit(episode_id, candidate_path, prediction_path, decision_path, truth_path)


def run_flash_boundary(workers: int, quantiles: Mapping[float, float]) -> list[dict[str, object]]:
    output = _safe_path("boundary_challenge_episodes/flash")
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(index, FLASH_SEED_START + index) for index in range(FLASH_COUNT)]
    audit = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_flash_init,
        initargs=(str(output), quantiles),
    ) as pool:
        for completed, row in enumerate(pool.map(_flash_task, tasks, chunksize=1), 1):
            audit.append(row)
            if completed % 100 == 0:
                print(f"boundary challenge flash boundary={completed}/{FLASH_COUNT}", flush=True)
    audit.sort(key=lambda row: str(row["episode_id"]))
    _write_csv(_safe_path("boundary_challenge_episodes/flash_boundary_ledger.csv"), audit)
    return audit


def _flash_truth_metrics(actual: np.ndarray, complete: bool) -> tuple[bool, float, float, float]:
    if not len(actual):
        return False, float("-inf"), 0.0, 0.0
    safety = SafetySpecification()
    margins = safety.violation_margins(
        {name: actual[:, index] for index, name in enumerate(MEASURED_OUTPUTS)}
    )
    maximum = float(np.max(margins))
    return (
        bool(complete and maximum <= 0.0),
        -maximum if complete else float("-inf"),
        float(np.max(np.maximum(margins, 0.0))),
        float(np.sum(np.maximum(margins, 0.0)) * 10.0),
    )


def evaluate_flash_boundary(audit: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for item in audit:
        candidate = json.loads(Path(str(item["candidate_path"])).read_text())
        prediction = primary._load_npz(Path(str(item["prediction_path"])))
        decision = json.loads(Path(str(item["decision_path"])).read_text())
        truth = primary._load_npz(Path(str(item["truth_path"])))
        actual = np.asarray(truth["actual_horizon_outputs"])
        actual_safe, true_margin, severity_max, severity_integrated = _flash_truth_metrics(
            actual, len(actual) == 120
        )
        method_specs: list[tuple[str, str, float | str]] = [
            ("nominal_DT", "nominal_DT", ""),
            ("discrepancy_mean", "discrepancy_mean", ""),
            ("raw_bootstrap_95", "raw_bootstrap", 0.95),
        ]
        method_specs.extend(
            (f"pooled_{level:.3f}", "pooled_conformal", level)
            for level in CONFIDENCE_LEVELS
        )
        method_specs.append(("action_conditioned", "action_conditioned", ""))
        for decision_key, method, confidence in method_specs:
            certified = decision["decisions"][decision_key] == "CERTIFIED SAFE"
            rows.append(
                {
                    "episode_id": item["episode_id"],
                    "population": "flash_boundary_challenge",
                    "method": method,
                    "confidence_level": confidence,
                    "episode_seed_evaluator_only": int(truth["episode_seed_evaluator_only"]),
                    "fault_family_evaluator_only": str(truth["fault_family_evaluator_only"]),
                    "fault_severity_evaluator_only": float(truth["fault_severity_evaluator_only"]),
                    "candidate_heat_duty_w": prediction["candidate_action"][0],
                    "candidate_vapor_valve_opening": prediction["candidate_action"][1],
                    "candidate_liquid_valve_opening": prediction["candidate_action"][2],
                    "candidate_feed_flow_mol_s": prediction["candidate_action"][3],
                    "normalized_action_distance": normalized_action_distance(prediction["candidate_action"]),
                    "nominal_dt_boundary_margin": candidate["nominal_dt_margin"],
                    "within_predeclared_boundary_band": candidate["within_predeclared_boundary_band"],
                    "nominal_dt_boundary_eligible_count": candidate["nominal_dt_boundary_eligible_count"],
                    "prediction_complete": str(prediction["prediction_status"]) == "COMPLETE",
                    "truth_complete": len(actual) == 120,
                    "actual_safe": actual_safe,
                    "actual_unsafe": not actual_safe,
                    "certified_safe": certified,
                    "false_safe": certified and not actual_safe,
                    "true_normalized_safety_margin": true_margin,
                    "predicted_normalized_safety_margin": decision["normalized_margins"][decision_key],
                    "severity_max": severity_max,
                    "severity_integrated_s": severity_integrated,
                    "severity_is_lower_bound": len(actual) != 120,
                    "prediction_sha256": item["prediction_sha256"],
                    "decision_sha256": item["decision_sha256"],
                    "truth_sha256": item["truth_sha256"],
                    "chronology_verified": item["chronology_verified"],
                }
            )
    return rows


def _cstr_init(output: str) -> None:
    global _CSTR_MODELS, _CSTR_TEMPLATE, _CSTR_OUTPUT
    _CSTR_MODELS = certification._load_models(REPOSITORY_ROOT / "artifacts/cstr/bootstrap_ensemble", 10)
    _CSTR_TEMPLATE = episode_design.build_template(episode_design.EpisodeDesignConfig())
    _CSTR_OUTPUT = Path(output)


def _cstr_paths(episode_id: str) -> tuple[Path, Path, Path, Path]:
    if _CSTR_OUTPUT is None:
        raise RuntimeError("CSTR boundary challenge worker not initialized")
    return tuple(
        _CSTR_OUTPUT / folder / f"CSTR_BOUNDARY_{episode_id}.{extension}"
        for folder, extension in (
            ("candidates", "json"),
            ("predictions", "npz"),
            ("decisions", "json"),
            ("truths", "npz"),
        )
    )


def _select_cstr_boundary_action(nominal, seed: int) -> tuple[float, float, int, bool, np.ndarray]:
    """Screen the frozen nominal CSTR action grid in one equation-equivalent batch.

    The scalar simulator invokes RK45 separately for every action and second. All
    branches are deterministic and remain in NORMAL phase over this horizon, so
    concatenating their states removes integrator setup overhead without changing
    the frozen dynamics, controller laws, tolerances, or candidate population.
    """
    if (
        nominal.current_phase.name != "NORMAL"
        or not nominal._startup_done
        or nominal.t + certification.HORIZON * nominal.dt >= nominal.T_startup + nominal.T_normal
        or nominal.fault_cfg.enable
        or nominal.conc_noise_pct != 0.0
        or nominal.flow_meas_noise_pct != 0.0
        or nominal.sensor_T_noise_std != 0.0
    ):
        raise RuntimeError("CSTR boundary challenge batched screen requires the frozen deterministic NORMAL nominal twin")

    count = len(CSTR_ACTIONS)
    state = np.repeat(np.asarray(nominal.x, dtype=float)[None, :], count, axis=0)
    trajectories = np.empty((count, certification.HORIZON), dtype=float)

    inlet_opening = np.full(count, nominal.valve_in.opening, dtype=float)
    coolant_opening = np.full(count, nominal.cool_valve_in.opening, dtype=float)
    pump_speed = np.full(count, nominal.pump_out.speed, dtype=float)
    flow_integral = np.full(count, nominal.pid_flow.integ, dtype=float)
    flow_previous = np.full(count, nominal.pid_flow.prev_err, dtype=float)
    level_integral = np.full(count, nominal.pid_level.integ, dtype=float)
    level_previous = np.full(count, nominal.pid_level.prev_err, dtype=float)
    temperature_integral = np.full(count, nominal.pid_temp.integ, dtype=float)
    temperature_previous = np.full(count, nominal.pid_temp.prev_err, dtype=float)

    def pid_step(pid, setpoint, process_value, integral, previous, reverse=False):
        error = process_value - setpoint if reverse else setpoint - process_value
        derivative = (error - previous) / nominal.dt
        unsaturated = pid.Kp * error + integral + pid.Kd * derivative
        command = np.clip(unsaturated, pid.umin, pid.umax)
        saturated = command != unsaturated
        integrate = (~saturated) | ((command == pid.umax) & (error < 0)) | (
            (command == pid.umin) & (error > 0)
        )
        integral = integral + integrate * pid.Ki * error * nominal.dt
        return command, integral, error

    current_time = float(nominal.t)
    for horizon in range(certification.HORIZON):
        volume = state[:, 0]
        temperature = state[:, 5]
        inlet_flow_before_update = nominal.valve_in.Cv_max * inlet_opening

        inlet_command, flow_integral, flow_previous = pid_step(
            nominal.pid_flow,
            CSTR_ACTIONS,
            inlet_flow_before_update,
            flow_integral,
            flow_previous,
        )

        level = volume / nominal.A_cs
        level_error = level - nominal.L_sp
        level_derivative = (level_error - level_previous) / nominal.dt
        level_unsaturated = (
            nominal.pid_level.Kp * level_error
            + level_integral
            + nominal.pid_level.Kd * level_derivative
        )
        level_pi = np.clip(level_unsaturated, nominal.pid_level.umin, nominal.pid_level.umax)
        level_saturated = level_pi != level_unsaturated
        level_integrate = (~level_saturated) | (
            (level_pi == nominal.pid_level.umax) & (level_error < 0)
        ) | ((level_pi == nominal.pid_level.umin) & (level_error > 0))
        level_integral = (
            level_integral
            + level_integrate * nominal.pid_level.Ki * level_error * nominal.dt
        )
        level_previous = level_error
        pump_command = np.clip(
            np.clip(inlet_flow_before_update / nominal.pump_out.Kq, 0.0, 1.0) + level_pi,
            0.0,
            1.0,
        )

        coolant_command, temperature_integral, temperature_previous = pid_step(
            nominal.pid_temp,
            nominal.T_sp,
            temperature,
            temperature_integral,
            temperature_previous,
            reverse=True,
        )

        inlet_alpha = 1.0 - np.exp(-nominal.dt / max(1e-6, nominal.valve_in.tau))
        coolant_alpha = 1.0 - np.exp(-nominal.dt / max(1e-6, nominal.cool_valve_in.tau))
        pump_alpha = 1.0 - np.exp(-nominal.dt / max(1e-6, nominal.pump_out.tau))
        inlet_opening = np.clip(
            inlet_opening + inlet_alpha * (inlet_command - inlet_opening), 0.0, 1.0
        )
        coolant_opening = np.clip(
            coolant_opening + coolant_alpha * (coolant_command - coolant_opening), 0.0, 1.0
        )
        pump_speed = np.clip(
            pump_speed + pump_alpha * (pump_command - pump_speed), 0.0, 1.0
        )
        inlet_flow = nominal.valve_in.Cv_max * inlet_opening
        outlet_flow = nominal.pump_out.Kq * pump_speed
        coolant_flow = nominal.cool_valve_in.Cv_max * coolant_opening

        def rhs(_time, flattened):
            values = flattened.reshape(count, 6)
            volume_rhs = np.maximum(values[:, 0], 0.1)
            energy_volume = np.maximum(values[:, 0], 1.0)
            ca, cb, cc, cd, temp = (
                values[:, 1], values[:, 2], values[:, 3], values[:, 4], values[:, 5]
            )
            k_a = 0.1 * np.exp(-5000.0 / (8.314 * temp))
            k_ad = 0.05 * np.exp(-7000.0 / (8.314 * temp))
            reaction_1 = k_ad * ca * cd
            reaction_2 = k_a * ca
            ua_scaled = nominal.UA * coolant_flow / max(1e-6, nominal.Fc_in_base)
            derivative = np.column_stack(
                (
                    inlet_flow - outlet_flow,
                    (inlet_flow / volume_rhs) * nominal.CA_in_base
                    - (outlet_flow / volume_rhs) * ca
                    - reaction_1
                    - reaction_2,
                    -(outlet_flow / volume_rhs) * cb + reaction_1,
                    -(outlet_flow / volume_rhs) * cc + reaction_2,
                    (inlet_flow / volume_rhs) * nominal.CD_in_base
                    - (outlet_flow / volume_rhs) * cd
                    - reaction_1,
                    (inlet_flow / volume_rhs) * (nominal.Tin_base - temp)
                    + ((-nominal.DH1) * reaction_1 + (-nominal.DH2) * reaction_2)
                    / nominal.Cp
                    - (ua_scaled / (values[:, 0] * nominal.Cp))
                    * (temp - nominal.Tc_base)
                    - (50.0 / (energy_volume * nominal.Cp)) * (temp - 300.0),
                )
            )
            return derivative.reshape(-1)

        solution = solve_ivp(
            rhs,
            (current_time, current_time + nominal.dt),
            state.reshape(-1),
            method="RK45",
            rtol=1e-7,
            atol=1e-9,
            max_step=0.05,
        )
        if not solution.success:
            raise RuntimeError(f"batched nominal CSTR integration failed: {solution.message}")
        state = solution.y[:, -1].reshape(count, 6)
        trajectories[:, horizon] = state[:, 5]
        current_time += nominal.dt

    margins = CSTR_T_MAX_K - np.max(trajectories, axis=1)
    ordering = np.lexsort((np.arange(len(margins)), np.abs(margins)))
    eligible = ordering[np.abs(margins[ordering]) <= CSTR_BOUNDARY_DELTA_K]
    candidates = eligible if len(eligible) else ordering[:1]
    top = candidates[:8]
    selected = int(top[seed % len(top)])
    return (
        float(CSTR_ACTIONS[selected]),
        float(margins[selected]),
        int(len(eligible)),
        bool(len(eligible)),
        trajectories[selected],
    )


def _cstr_task(task: tuple[int, int]) -> dict[str, object]:
    if _CSTR_MODELS is None or _CSTR_TEMPLATE is None:
        raise RuntimeError("CSTR boundary challenge predictor missing")
    index, seed = task
    episode_id = f"cstr_boundary_{index:04d}"
    candidate_path, prediction_path, decision_path, truth_path = _cstr_paths(episode_id)
    if all(path.exists() for path in (candidate_path, prediction_path, decision_path, truth_path)):
        return _artifact_audit(episode_id, candidate_path, prediction_path, decision_path, truth_path)
    nominal, plant, residual, online, previous, hidden_magnitude = certification._reconstruct_decision_state(
        seed, episode_design.EpisodeDesignConfig(), _CSTR_TEMPLATE
    )
    action, nominal_margin, eligible_count, band_found, nominal_future = _select_cstr_boundary_action(
        nominal, seed
    )
    if not candidate_path.exists():
        payload = {
            "schema_version": "CSTR_BOUNDARY_CANDIDATE_1.0",
            "episode_id": episode_id,
            "action_lps": action,
            "nominal_dt_margin_k": nominal_margin,
            "within_boundary_band": band_found,
            "eligible_count": eligible_count,
            "pool_size": len(CSTR_ACTIONS),
            "selection_source": "nominal_DT_only",
        }
        primary._exclusive_write(candidate_path, primary._json_bytes(payload))
    future = np.empty((certification.HORIZON, 4))
    prior = previous
    for horizon in range(certification.HORIZON):
        future[horizon] = [action, action - prior, nominal_future[horizon], 51 + horizon]
        prior = action
    item = discrepancy_freeze.EpisodeInputs(
        episode_id,
        certification._action_category(action),
        seed + 100_000,
        residual,
        online,
        future,
        nominal_future,
        action,
        truth_path,
        copy.deepcopy(plant),
        seed + 4_000_000,
    )
    if not prediction_path.exists():
        _wait_after(candidate_path)
        certification.predict_frozen_u3(item, _CSTR_MODELS, 2000, prediction_path)
    with np.load(prediction_path, allow_pickle=False) as saved:
        center = np.asarray(saved["predictive_center_k"])
        scale = np.maximum(np.asarray(saved["predictive_scale_k"]), certification.EPSILON_SCALE_K)
    if not decision_path.exists():
        upper = center + CSTR_Q_PLUS * scale
        payload = {
            "schema_version": "CSTR_BOUNDARY_DECISION_1.0",
            "episode_id": episode_id,
            "prediction_sha256": _sha256(prediction_path),
            "nominal_dt_decision": "CERTIFIED SAFE" if nominal_margin >= 0 else "NOT CERTIFIED",
            "frozen_cstr_decision": "CERTIFIED SAFE" if float(np.max(upper)) <= CSTR_T_MAX_K else "NOT CERTIFIED",
            "q_plus": CSTR_Q_PLUS,
            "temperature_limit_k": CSTR_T_MAX_K,
            "conformal_margin_k": CSTR_T_MAX_K - float(np.max(upper)),
        }
        _wait_after(prediction_path)
        primary._exclusive_write(decision_path, primary._json_bytes(payload))
    if not truth_path.exists():
        _wait_after(decision_path)
        actual = certification.reveal_physical_temperature(item, prediction_path)
        with np.load(truth_path, allow_pickle=False) as saved:
            truth_payload = {key: np.asarray(saved[key]) for key in saved.files}
        truth_payload.update(
            {
                "episode_seed_evaluator_only": np.asarray(seed),
                "hidden_ua_fouling_magnitude_evaluator_only": np.asarray(hidden_magnitude),
                "prediction_sha256": np.asarray(_sha256(prediction_path)),
                "decision_sha256": np.asarray(_sha256(decision_path)),
                "actual_physical_temperature_k": actual,
            }
        )
        # reveal_physical_temperature creates truth only after the decision. Replacing
        # that just-created truth extends its evaluator schema before it is audited.
        np.savez_compressed(truth_path, **truth_payload)
    return _artifact_audit(episode_id, candidate_path, prediction_path, decision_path, truth_path)


def run_cstr_boundary(workers: int) -> list[dict[str, object]]:
    output = _safe_path("boundary_challenge_episodes/cstr")
    output.mkdir(parents=True, exist_ok=True)
    tasks = [(index, CSTR_SEED_START + index) for index in range(CSTR_COUNT)]
    audit = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_cstr_init,
        initargs=(str(output),),
    ) as pool:
        for completed, row in enumerate(pool.map(_cstr_task, tasks, chunksize=1), 1):
            audit.append(row)
            if completed % 50 == 0:
                print(f"boundary challenge CSTR boundary={completed}/{CSTR_COUNT}", flush=True)
    audit.sort(key=lambda row: str(row["episode_id"]))
    _write_csv(_safe_path("boundary_challenge_episodes/cstr_boundary_ledger.csv"), audit)
    return audit


def evaluate_cstr_boundary(audit: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = []
    for item in audit:
        candidate = json.loads(Path(str(item["candidate_path"])).read_text())
        decision = json.loads(Path(str(item["decision_path"])).read_text())
        with np.load(Path(str(item["truth_path"])), allow_pickle=False) as truth:
            actual = np.asarray(truth["actual_physical_temperature_k"])
            seed = int(truth["episode_seed_evaluator_only"])
        actual_max = float(np.max(actual))
        actual_safe = actual_max <= CSTR_T_MAX_K
        severity_max = max(0.0, actual_max - CSTR_T_MAX_K)
        severity_integrated = float(np.sum(np.maximum(actual - CSTR_T_MAX_K, 0.0)))
        for method, key in (
            ("nominal_DT", "nominal_dt_decision"),
            ("frozen_CSTR_conformal", "frozen_cstr_decision"),
        ):
            certified = decision[key] == "CERTIFIED SAFE"
            rows.append(
                {
                    "episode_id": item["episode_id"],
                    "population": "CSTR_BOUNDARY_CHALLENGE",
                    "method": method,
                    "confidence_level": "" if method == "nominal_DT" else 0.95,
                    "episode_seed_evaluator_only": seed,
                    "action_lps": candidate["action_lps"],
                    "nominal_dt_boundary_margin_k": candidate["nominal_dt_margin_k"],
                    "within_boundary_band": candidate["within_boundary_band"],
                    "eligible_count": candidate["eligible_count"],
                    "actual_safe": actual_safe,
                    "actual_unsafe": not actual_safe,
                    "certified_safe": certified,
                    "false_safe": certified and not actual_safe,
                    "true_safety_margin_k": CSTR_T_MAX_K - actual_max,
                    "predicted_safety_margin_k": (
                        candidate["nominal_dt_margin_k"]
                        if method == "nominal_DT"
                        else decision["conformal_margin_k"]
                    ),
                    "severity_max_k": severity_max,
                    "severity_integrated_k_s": severity_integrated,
                    "prediction_sha256": item["prediction_sha256"],
                    "decision_sha256": item["decision_sha256"],
                    "truth_sha256": item["truth_sha256"],
                    "chronology_verified": item["chronology_verified"],
                }
            )
    return rows


def aggregate_risk(
    rows: Sequence[Mapping[str, object]], process: str, method: str, confidence: object
) -> dict[str, object]:
    selected = [
        row for row in rows
        if row["method"] == method
        and (confidence == "" or str(row["confidence_level"]) == str(confidence))
    ]
    count = len(selected)
    unsafe = sum(bool(row["actual_unsafe"]) for row in selected)
    actually_safe = count - unsafe
    certified = sum(bool(row["certified_safe"]) for row in selected)
    false_safe = sum(bool(row["false_safe"]) for row in selected)
    true_safe_certified = sum(bool(row["certified_safe"]) and bool(row["actual_safe"]) for row in selected)
    joint = _wilson(false_safe, count)
    conditional = _wilson(false_safe, certified)
    severity_name = "severity_max" if process == "flash" else "severity_max_k"
    integrated_name = "severity_integrated_s" if process == "flash" else "severity_integrated_k_s"
    false_severity = [float(row[severity_name]) for row in selected if bool(row["false_safe"])]
    false_integrated = [float(row[integrated_name]) for row in selected if bool(row["false_safe"])]
    return {
        "process": process,
        "population": selected[0]["population"],
        "method": method,
        "confidence_level": confidence,
        "episode_count": count,
        "actual_unsafe_count": unsafe,
        "actual_unsafe_rate": unsafe / count,
        "certified_safe_count": certified,
        "certification_availability": certified / count,
        "truly_safe_action_certification_rate": true_safe_certified / actually_safe if actually_safe else 0.0,
        "false_safe_count": false_safe,
        "joint_false_safe_risk": false_safe / count,
        "joint_false_safe_wilson_95_lower": joint[0],
        "joint_false_safe_wilson_95_upper": joint[1],
        "conditional_false_safe_risk": false_safe / certified if certified else 0.0,
        "conditional_false_safe_wilson_95_lower": conditional[0],
        "conditional_false_safe_wilson_95_upper": conditional[1],
        "not_certified_rate": 1.0 - certified / count,
        "maximum_false_safe_severity": max(false_severity, default=0.0),
        "total_false_safe_severity": sum(false_severity),
        "maximum_integrated_false_safe_severity": max(false_integrated, default=0.0),
        "total_integrated_false_safe_severity": sum(false_integrated),
        "expected_severity_weighted_unsafe_validation_risk": sum(false_severity) / count,
        "expected_integrated_unsafe_validation_risk": sum(false_integrated) / count,
    }


def risk_rows(flash: Sequence[Mapping[str, object]], cstr: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows = [aggregate_risk(flash, "flash", "nominal_DT", "")]
    rows.extend(
        aggregate_risk(flash, "flash", "pooled_conformal", level)
        for level in CONFIDENCE_LEVELS
    )
    rows.extend(
        [
            aggregate_risk(cstr, "CSTR", "nominal_DT", ""),
            aggregate_risk(cstr, "CSTR", "frozen_CSTR_conformal", 0.95),
        ]
    )
    return rows


def _save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(_safe_path(f"boundary_challenge_figures/{name}.png"), dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(
    flash: Sequence[Mapping[str, object]], cstr: Sequence[Mapping[str, object]], risk: Sequence[Mapping[str, object]]
) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    frontier = [row for row in risk if row["process"] == "flash" and row["method"] == "pooled_conformal"]
    nominal = next(row for row in risk if row["process"] == "flash" and row["method"] == "nominal_DT")
    representative = frontier[0]
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(
        [float(nominal["certification_availability"]), float(representative["certification_availability"])],
        [float(nominal["joint_false_safe_risk"]), float(representative["joint_false_safe_risk"])],
        s=48,
        label="Observed joint risk",
    )
    ax.scatter(
        [float(nominal["certification_availability"]), float(representative["certification_availability"])],
        [float(nominal["joint_false_safe_wilson_95_upper"]), float(representative["joint_false_safe_wilson_95_upper"])],
        marker="s",
        s=42,
        label="Wilson 95% upper",
    )
    ax.annotate("Nominal DT", (float(nominal["certification_availability"]), float(nominal["joint_false_safe_wilson_95_upper"])), xytext=(-62, 8), textcoords="offset points")
    ax.annotate(
        "Pooled conformal\nall levels 90-99%",
        (float(representative["certification_availability"]), float(representative["joint_false_safe_wilson_95_upper"])),
        xytext=(10, 10),
        textcoords="offset points",
    )
    ax.set(xlabel="Certification availability", ylabel="False-safe risk", xlim=(-0.025, 0.65), ylim=(-0.015, 0.43))
    ax.legend(frameon=False)
    _save(fig, "01_risk_availability_frontier")

    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    ax.scatter(
        [float(nominal["certification_availability"]), float(representative["certification_availability"])],
        [float(nominal["expected_severity_weighted_unsafe_validation_risk"]), float(representative["expected_severity_weighted_unsafe_validation_risk"])],
        s=52,
        color="#ca3c25",
    )
    ax.annotate("Nominal DT", (float(nominal["certification_availability"]), float(nominal["expected_severity_weighted_unsafe_validation_risk"])), xytext=(-62, 8), textcoords="offset points")
    ax.annotate(
        "Pooled conformal\nall levels 90-99%",
        (float(representative["certification_availability"]), float(representative["expected_severity_weighted_unsafe_validation_risk"])),
        xytext=(10, 10),
        textcoords="offset points",
    )
    ax.set(
        xlabel="Certification availability",
        ylabel="Expected severity-weighted unsafe-validation risk",
        xlim=(-0.025, 0.65),
        ylim=(-0.02, 0.55),
    )
    _save(fig, "02_severity_risk_frontier")

    flash95 = [row for row in flash if row["method"] == "pooled_conformal" and float(row["confidence_level"]) == 0.95 and bool(row["truth_complete"])]
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    ax.scatter([float(row["true_normalized_safety_margin"]) for row in flash95], [float(row["nominal_dt_boundary_margin"]) for row in flash95], s=11, alpha=0.35, label="Nominal DT")
    ax.scatter([float(row["true_normalized_safety_margin"]) for row in flash95], [float(row["predicted_normalized_safety_margin"]) for row in flash95], s=11, alpha=0.35, label="95% conformal")
    ax.axhline(0, color="#555555", lw=0.8); ax.axvline(0, color="#555555", lw=0.8)
    ax.set(xlabel="True normalized safety margin", ylabel="Predicted normalized safety margin")
    ax.legend(frameon=False)
    _save(fig, "03_near_boundary_true_vs_predicted_margins")

    groups = [
        next(row for row in risk if row["process"] == "flash" and row["method"] == "nominal_DT"),
        next(row for row in risk if row["process"] == "flash" and row["method"] == "pooled_conformal" and float(row["confidence_level"]) == 0.95),
        next(row for row in risk if row["process"] == "CSTR" and row["method"] == "nominal_DT"),
        next(row for row in risk if row["process"] == "CSTR" and row["method"] == "frozen_CSTR_conformal"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    positions = np.arange(4)
    ax.bar(positions - 0.18, [float(row["joint_false_safe_risk"]) for row in groups], 0.36, label="Joint false-safe", color="#ca3c25")
    ax.bar(positions + 0.18, [float(row["certification_availability"]) for row in groups], 0.36, label="Certification", color="#1d6996")
    ax.set_xticks(positions, ["Flash\nnominal", "Flash\nconformal", "CSTR\nnominal", "CSTR\nconformal"])
    ax.set_ylabel("Probability"); ax.legend(frameon=False)
    _save(fig, "04_cstr_vs_flash_boundary_behavior")


def build_report(
    flash: Sequence[Mapping[str, object]], cstr: Sequence[Mapping[str, object]], risk: Sequence[Mapping[str, object]], quantiles: Mapping[float, float]
) -> str:
    flash_nominal = next(row for row in risk if row["process"] == "flash" and row["method"] == "nominal_DT")
    flash95 = next(row for row in risk if row["process"] == "flash" and row["method"] == "pooled_conformal" and float(row["confidence_level"]) == 0.95)
    cstr_nominal = next(row for row in risk if row["process"] == "CSTR" and row["method"] == "nominal_DT")
    cstr95 = next(row for row in risk if row["process"] == "CSTR" and row["method"] == "frozen_CSTR_conformal")
    frontier_lines = []
    for row in risk:
        if row["process"] == "flash" and row["method"] == "pooled_conformal":
            frontier_lines.append(
                f"| {100*float(row['confidence_level']):g}% | {row['certification_availability']:.3f} | "
                f"{row['truly_safe_action_certification_rate']:.3f} | {row['joint_false_safe_risk']:.3f} | "
                f"{row['joint_false_safe_wilson_95_upper']:.3f} | {row['conditional_false_safe_risk']:.3f} | "
                f"{row['maximum_false_safe_severity']:.3f} | {row['total_integrated_false_safe_severity']:.3f} |"
            )
    boundary_found = sum(
        bool(row["within_predeclared_boundary_band"])
        for row in flash if row["method"] == "nominal_DT"
    )
    flash_unsafe = int(flash95["actual_unsafe_count"])
    useful = (
        float(flash95["certification_availability"]) > 0.0
        and float(flash95["joint_false_safe_risk"]) < float(flash_nominal["joint_false_safe_risk"])
    )
    return f"""# boundary challenge Boundary Reliability Results

## Frozen Design And Chronology

Flash uses all 1,200 population design-reserved boundary seeds. Candidate pools and selection
use only nominal-DT trajectories and the frozen `|m_DT|<=0.15` criterion.
{boundary_found}/1200 states had at least one in-band candidate; fallbacks were
not regenerated. Prediction and decision artifacts were serialized before each
plant future reveal. The true challenge mix is {1200-flash_unsafe} safe and
{flash_unsafe} unsafe actions; it was not rebalanced after truth.

The five flash quantiles are finite-sample order statistics of only the 463
frozen boundary-calibration scores: `{json.dumps({str(k): v for k, v in quantiles.items()})}`.

## Flash Risk-Availability Frontier

| Confidence | Availability | Truly-safe certification | Joint false-safe | Wilson upper | Conditional false-safe | Max severity | Integrated false-safe severity |
|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(frontier_lines)}

At 95%, nominal-DT reliance is {flash_nominal['certification_availability']:.3f}
with joint false-safe risk {flash_nominal['joint_false_safe_risk']:.3f}. The
pooled method certifies {flash95['certification_availability']:.3f}, with joint
risk {flash95['joint_false_safe_risk']:.3f} and Wilson upper
{flash95['joint_false_safe_wilson_95_upper']:.3f}. Thus the method
{'reduces unsafe reliance without complete abstention' if useful else 'removes observed unsafe validation only through complete abstention and does not demonstrate nontrivial boundary-risk reduction'}.
Zero false-safe events are not required for this conclusion.

## Separate CSTR Stress Extension

The new 300-action CSTR denominator is separate from primary certification/mixed-degradation evaluation. Its nominal-DT
availability/risk are {cstr_nominal['certification_availability']:.3f}/
{cstr_nominal['joint_false_safe_risk']:.3f}; the frozen 95% CSTR method gives
{cstr95['certification_availability']:.3f}/{cstr95['joint_false_safe_risk']:.3f}.
The CSTR reduction likewise occurs through complete abstention on this challenge.
No primary certification or mixed-degradation evaluation artifact or denominator was altered.

## Conclusion

### {'BOUNDARY RELIABILITY IMPROVED WITHOUT COMPLETE ABSTENTION' if useful else 'BOUNDARY RELIABILITY NOT DEMONSTRATED'}
"""


def _hash_outputs() -> None:
    paths = [
        DESIGN_PATH,
        EXPERIMENT_ROOT / "flash_boundary_episode_results.csv",
        EXPERIMENT_ROOT / "cstr_boundary_episode_results.csv",
        EXPERIMENT_ROOT / "boundary_risk_availability.csv",
        EXPERIMENT_ROOT / "boundary_challenge_report.md",
    ]
    paths.extend(sorted((EXPERIMENT_ROOT / "boundary_challenge_figures").glob("*.png")))
    paths.extend(sorted((EXPERIMENT_ROOT / "boundary_challenge_episodes").rglob("*.*")))
    lines = ["# Frozen boundary challenge artifact SHA-256 manifest", ""]
    for path in paths:
        lines.append(f"{_sha256(path)}  {path.relative_to(EXPERIMENT_ROOT).as_posix()}")
    _safe_path("boundary_challenge_artifact_hashes.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def main(workers: int, skip_cstr: bool = False) -> None:
    design = _load_design()
    quantiles = boundary_quantiles_from_calibration()
    expected = {
        float(level): float(value)
        for level, value in zip(design["flash"]["confidence_levels"], design["flash"]["quantiles"])
    }
    if quantiles != expected:
        raise RuntimeError("calibration-only boundary quantiles differ from frozen boundary challenge design")
    flash_audit = run_flash_boundary(workers, quantiles)
    flash_rows = evaluate_flash_boundary(flash_audit)
    _write_csv(_safe_path("flash_boundary_episode_results.csv"), flash_rows)
    if skip_cstr:
        return
    cstr_audit = run_cstr_boundary(workers)
    cstr_rows = evaluate_cstr_boundary(cstr_audit)
    _write_csv(_safe_path("cstr_boundary_episode_results.csv"), cstr_rows)
    risk = risk_rows(flash_rows, cstr_rows)
    _write_csv(_safe_path("boundary_risk_availability.csv"), risk)
    _safe_path("boundary_challenge_report.md").write_text(
        build_report(flash_rows, cstr_rows, risk, quantiles), encoding="utf-8"
    )
    create_figures(flash_rows, cstr_rows, risk)
    _hash_outputs()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    parser.add_argument("--skip-cstr", action="store_true")
    args = parser.parse_args()
    main(args.workers, args.skip_cstr)
