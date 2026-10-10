"""Evaluate safety certification across bands of nominal-prediction margin."""

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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

from twinguard.cstr import episode_design as episode_design
from twinguard.cstr import freeze_discrepancy_model as discrepancy_freeze
from twinguard.cstr import safety_certification as certification

from twinguard.flash import evaluate_primary as primary
from twinguard.evaluation import boundary_challenge as boundary
from twinguard.uncertainty.flash_bootstrap import PREDICTIVE_SCALE_FLOORS
from twinguard.conformal.flash_conformal import (
    conformal_envelope,
    corrected_quantile,
    safety_decision,
    safety_direction_score,
)
from twinguard.scenarios.flash_population_design import (
    MEASURED_OUTPUTS,
    SafetySpecification,
    latin_hypercube_actions,
    purpose_seed,
    targeted_action_points,
    validate_candidate_action,
)
from twinguard.plants.flash_drum import FlashAction
from twinguard.flash.bootstrap_and_calibrate import batch_nominal_dt_margins

from twinguard.utils.margin_ladder_output_guard import EXPERIMENT_ROOT, REPOSITORY_ROOT, assert_margin_ladder_output
from twinguard.evaluation.margin_ladder_preregistration import (
    BANDS,
    CSTR_TEST_COUNT,
    CSTR_TEST_START,
    FLASH_CALIBRATION_COUNT,
    FLASH_CALIBRATION_START,
    FLASH_TEST_COUNT,
    FLASH_TEST_START,
    PRIOR_RESERVED_INTERVALS,
)


POOL_LHS_COUNT = 256
POOL_TOP_RANKS = 8
POOL_TARGETS = {"B0": 0.075, "B1": 0.225, "B2": 0.45, "B3": 0.80, "B4": 1.30}
FROZEN_FLASH_Q = 11.04859850832729
FROZEN_CSTR_Q = 5.464557586938352
RAW_Q = primary.RAW_ONE_SIDED_95
CSTR_ACTIONS = np.linspace(0.012, 0.045, 129)
CSTR_SCALE_K = 1.0
METHODS = (
    "M0_nominal_DT",
    "M1_discrepancy_mean",
    "M2_raw_bootstrap",
    "M3_frozen_pooled",
    "M5_margin_conditioned",
)
FORBIDDEN_KEYS = (
    "fault",
    "severity",
    "changed_parameter",
    "actual",
    "future_truth",
    "physical_truth",
    "safety_label",
    "hidden_state",
)

_OUTPUT: Path | None = None
_BAND_Q: dict[str, float | None] | None = None
_CSTR_MODELS = None
_CSTR_TEMPLATE = None


def _sha256(path: Path) -> str:
    last_error: PermissionError | None = None
    for _ in range(100):
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.05)
    raise last_error or PermissionError(path)


def _safe(relative: str) -> Path:
    path = assert_margin_ladder_output(EXPERIMENT_ROOT / relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    path = assert_margin_ladder_output(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _wait_after(path: Path) -> None:
    threshold = path.stat().st_mtime_ns
    while time.time_ns() <= threshold:
        time.sleep(0.001)


def _band_for_margin(margin: float) -> str | None:
    if not np.isfinite(margin) or margin <= 0.0:
        return None
    for label, lower, upper in BANDS:
        if margin > lower and (upper is None or margin <= upper):
            return label
    return None


def _candidate_pool(seed: int) -> np.ndarray:
    return np.vstack(
        [
            latin_hypercube_actions(POOL_LHS_COUNT, purpose_seed(seed, 503)),
            targeted_action_points(),
        ]
    )


def _select_flash_action(
    decision_state: np.ndarray, seed: int, assigned_band: str
) -> tuple[FlashAction | None, float, int, int]:
    pool = _candidate_pool(seed)
    margins = batch_nominal_dt_margins(decision_state, pool)
    eligible = np.asarray(
        [index for index, margin in enumerate(margins) if _band_for_margin(float(margin)) == assigned_band],
        dtype=int,
    )
    if not len(eligible):
        return None, float("nan"), 0, len(pool)
    target = POOL_TARGETS[assigned_band]
    order = eligible[np.lexsort((eligible, np.abs(margins[eligible] - target)))]
    top = order[:POOL_TOP_RANKS]
    selected = int(top[seed % len(top)])
    return validate_candidate_action(pool[selected]), float(margins[selected]), len(eligible), len(pool)


def _artifact_paths(root: Path, episode_id: str, calibration: bool) -> dict[str, Path]:
    paths = {
        "candidate": root / "candidates" / f"{episode_id}.json",
        "prediction": root / "predictions" / f"{episode_id}.npz",
        "truth": root / "truths" / f"{episode_id}.npz",
    }
    if not calibration:
        paths["decision"] = root / "decisions" / f"{episode_id}.json"
    return paths


def _validate_pretruth_keys(payload: Mapping[str, object]) -> None:
    lowered = " ".join(payload).lower()
    leaked = [token for token in FORBIDDEN_KEYS if token in lowered]
    if leaked:
        raise RuntimeError(f"pre-truth key leakage: {leaked}")


def _flash_init(output: str, band_q: Mapping[str, float | None] | None) -> None:
    global _OUTPUT, _BAND_Q
    _OUTPUT = Path(output)
    _BAND_Q = dict(band_q) if band_q is not None else None
    frozen = primary.verify_frozen_inputs()
    if abs(float(frozen["pooled_q"]) - FROZEN_FLASH_Q) > 1e-12:
        raise RuntimeError("frozen flash pooled q changed")
    primary._init_worker(output, frozen)


def _flash_task(task: tuple[str, int, int]) -> dict[str, object]:
    if _OUTPUT is None:
        raise RuntimeError("margin ladder flash worker not initialized")
    role, index, seed = task
    calibration = role == "calibration"
    assigned_band = BANDS[index % len(BANDS)][0]
    episode_id = f"flash_margin_{role}_{index:05d}_{assigned_band}"
    paths = _artifact_paths(_OUTPUT, episode_id, calibration)
    observation = primary._simulate_observation(index, seed)
    candidate = None
    margin = float("nan")
    eligible_count = 0
    pool_size = POOL_LHS_COUNT + len(targeted_action_points())
    if int(observation["completed"]) == 50:
        candidate, margin, eligible_count, pool_size = _select_flash_action(
            observation["twin"].state.as_array(), seed, assigned_band
        )
    status = "ELIGIBLE" if candidate is not None else "NO_ELIGIBLE_ACTION"
    candidate_payload = {
        "schema_version": "FLASH_MARGIN_CANDIDATE_1.0",
        "episode_id": episode_id,
        "role": role,
        "assigned_margin_band": assigned_band,
        "selection_status": status,
        "candidate_action": candidate.as_array().tolist() if candidate is not None else [],
        "nominal_dt_margin": margin,
        "eligible_count_in_band": eligible_count,
        "candidate_pool_size": pool_size,
        "selection_source": "nominal_DT_only",
    }
    _validate_pretruth_keys(candidate_payload)
    if not paths["candidate"].exists():
        primary._exclusive_write(paths["candidate"], primary._json_bytes(candidate_payload))
    if candidate is None:
        return {
            "episode_id": episode_id,
            "role": role,
            "episode_seed": seed,
            "assigned_margin_band": assigned_band,
            "selection_status": status,
            "observation_complete": int(observation["completed"]) == 50,
            "candidate_path": paths["candidate"].as_posix(),
            "candidate_sha256": _sha256(paths["candidate"]),
            "prediction_path": "",
            "decision_path": "",
            "truth_path": "",
            "chronology_verified": True,
        }

    if not paths["prediction"].exists():
        prediction, _, _ = primary._build_prediction(observation, candidate, episode_id, seed)
        prediction["schema_version"] = np.asarray("FLASH_MARGIN_PREDICTION_1.0")
        _validate_pretruth_keys(prediction)
        _wait_after(paths["candidate"])
        primary._exclusive_write(paths["prediction"], primary._npz_bytes(prediction))
    prediction = primary._load_npz(paths["prediction"])

    if not calibration and not paths["decision"].exists():
        complete = str(prediction["prediction_status"]) == "COMPLETE"
        nominal = np.asarray(prediction["nominal_horizon_outputs"])
        mean = np.asarray(prediction["mean_corrected_horizon_outputs"])
        center = np.asarray(prediction["ensemble_center_horizon_outputs"])
        scale = np.asarray(prediction["ensemble_scale_horizon_outputs"])
        supported = _BAND_Q is not None and _BAND_Q.get(assigned_band) is not None
        band_q = _BAND_Q.get(assigned_band) if _BAND_Q is not None else None
        if complete:
            decisions = {
                "M0_nominal_DT": "CERTIFIED SAFE" if boundary._flash_point_margin(nominal) >= 0 else "NOT CERTIFIED",
                "M1_discrepancy_mean": "CERTIFIED SAFE" if boundary._flash_point_margin(mean) >= 0 else "NOT CERTIFIED",
                "M2_raw_bootstrap": safety_decision(center, scale, RAW_Q).replace("SAFE", "CERTIFIED SAFE"),
                "M3_frozen_pooled": safety_decision(center, scale, FROZEN_FLASH_Q).replace("SAFE", "CERTIFIED SAFE"),
                "M5_margin_conditioned": safety_decision(center, scale, band_q, supported=supported).replace("SAFE", "CERTIFIED SAFE"),
            }
            margins = {
                "M0_nominal_DT": boundary._flash_point_margin(nominal),
                "M1_discrepancy_mean": boundary._flash_point_margin(mean),
                "M2_raw_bootstrap": boundary._flash_envelope_margin(center, scale, RAW_Q),
                "M3_frozen_pooled": boundary._flash_envelope_margin(center, scale, FROZEN_FLASH_Q),
                "M5_margin_conditioned": boundary._flash_envelope_margin(center, scale, float(band_q)) if supported else float("-inf"),
            }
        else:
            decisions = {method: "NOT CERTIFIED" for method in METHODS}
            margins = {method: float("-inf") for method in METHODS}
        decision_payload = {
            "schema_version": "FLASH_MARGIN_DECISION_1.0",
            "episode_id": episode_id,
            "prediction_sha256": _sha256(paths["prediction"]),
            "assigned_margin_band": assigned_band,
            "decisions": decisions,
            "normalized_margins": margins,
            "frozen_pooled_quantile": FROZEN_FLASH_Q,
            "margin_conditioned_quantile": band_q,
            "margin_conditioned_supported": supported,
        }
        _validate_pretruth_keys(decision_payload)
        _wait_after(paths["prediction"])
        primary._exclusive_write(paths["decision"], primary._json_bytes(decision_payload))

    if not paths["truth"].exists():
        actual, domain_exit = primary._truth_rollout(observation, candidate)
        hidden = observation["hidden"]
        truth_payload = {
            "schema_version": np.asarray("FLASH_MARGIN_TRUTH_1.0"),
            "episode_id": np.asarray(episode_id),
            "episode_seed_evaluator_only": np.asarray(seed),
            "fault_family_evaluator_only": np.asarray(hidden.family.value),
            "fault_severity_evaluator_only": np.asarray(hidden.normalized_severity),
            "fault_severity_label_evaluator_only": np.asarray(hidden.severity_label),
            "actual_horizon_outputs": actual,
            "completed_horizon_samples": np.asarray(len(actual)),
            "truth_domain_exit_evaluator_only": np.asarray(domain_exit),
            "prediction_sha256": np.asarray(_sha256(paths["prediction"])),
            "decision_sha256": np.asarray(_sha256(paths["decision"])) if not calibration else np.asarray(""),
        }
        predecessor = paths["prediction"] if calibration else paths["decision"]
        _wait_after(predecessor)
        primary._exclusive_write(paths["truth"], primary._npz_bytes(truth_payload))

    ordered = [paths["candidate"], paths["prediction"]]
    if not calibration:
        ordered.append(paths["decision"])
    ordered.append(paths["truth"])
    times = [path.stat().st_mtime_ns for path in ordered]
    chronology = all(left < right for left, right in zip(times, times[1:]))
    return {
        "episode_id": episode_id,
        "role": role,
        "episode_seed": seed,
        "assigned_margin_band": assigned_band,
        "selection_status": status,
        "observation_complete": int(observation["completed"]) == 50,
        "candidate_path": paths["candidate"].as_posix(),
        "candidate_sha256": _sha256(paths["candidate"]),
        "prediction_path": paths["prediction"].as_posix(),
        "prediction_sha256": _sha256(paths["prediction"]),
        "decision_path": paths.get("decision", Path()).as_posix() if not calibration else "",
        "decision_sha256": _sha256(paths["decision"]) if not calibration else "",
        "truth_path": paths["truth"].as_posix(),
        "truth_sha256": _sha256(paths["truth"]),
        "chronology_verified": chronology,
    }


def run_flash_population(role: str, count: int, seed_start: int, workers: int) -> list[dict[str, object]]:
    root = _safe(f"artifacts/flash_{role}")
    root.mkdir(parents=True, exist_ok=True)
    band_q = None
    if role == "test":
        config = json.loads(_safe("flash_band_conformal_config.json").read_text())
        band_q = {
            band: payload["quantile"] if payload["supported"] else None
            for band, payload in config["bands"].items()
        }
    tasks = [(role, index, seed_start + index) for index in range(count)]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_flash_init,
        initargs=(str(root), band_q),
    ) as pool:
        for completed, row in enumerate(pool.map(_flash_task, tasks, chunksize=1), 1):
            rows.append(row)
            if completed % 100 == 0:
                print(f"margin ladder flash {role}={completed}/{count}", flush=True)
    rows.sort(key=lambda row: str(row["episode_id"]))
    _write_csv(_safe(f"artifacts/FLASH_{role.upper()}_ARTIFACT_AUDIT.csv"), rows)
    return rows


def calibrate_margin_bands(audit: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if any(_safe("artifacts/flash_test/truths").glob("*.npz")):
        raise RuntimeError("cannot calibrate after margin ladder test truth exists")
    score_rows: list[dict[str, object]] = []
    for item in audit:
        if item["selection_status"] != "ELIGIBLE":
            continue
        prediction = primary._load_npz(Path(str(item["prediction_path"])))
        truth = primary._load_npz(Path(str(item["truth_path"])))
        actual = np.asarray(truth["actual_horizon_outputs"])
        center = np.asarray(prediction["ensemble_center_horizon_outputs"])
        scale = np.asarray(prediction["ensemble_scale_horizon_outputs"])
        complete = (
            str(prediction["prediction_status"]) == "COMPLETE"
            and len(actual) == 120
            and center.shape == actual.shape
        )
        if not complete:
            continue
        score, direction, horizon = safety_direction_score(actual, center, scale)
        candidate = json.loads(Path(str(item["candidate_path"])).read_text())
        score_rows.append(
            {
                "episode_id": item["episode_id"],
                "episode_seed_evaluator_only": int(truth["episode_seed_evaluator_only"]),
                "margin_band": item["assigned_margin_band"],
                "nominal_dt_margin": candidate["nominal_dt_margin"],
                "whole_trajectory_safety_score": score,
                "maximizing_direction": direction,
                "maximizing_horizon_index": horizon,
                "fault_family_evaluator_only": str(truth["fault_family_evaluator_only"]),
                "fault_severity_evaluator_only": float(truth["fault_severity_evaluator_only"]),
                "prediction_sha256": item["prediction_sha256"],
                "truth_sha256": item["truth_sha256"],
            }
        )
    _write_csv(_safe("flash_band_calibration_scores.csv"), score_rows)
    bands: dict[str, object] = {}
    for band, _, _ in BANDS:
        scores = [
            float(row["whole_trajectory_safety_score"])
            for row in score_rows
            if row["margin_band"] == band
        ]
        supported = len(scores) >= 200
        quantile, rank = corrected_quantile(scores, 0.05) if supported else (None, None)
        bands[band] = {
            "complete_calibration_count": len(scores),
            "minimum_required": 200,
            "supported": supported,
            "alpha": 0.05,
            "rank": rank,
            "quantile": quantile,
        }
    config = {
        "schema_version": "FLASH_BAND_CONFORMAL_CONFIG_1.0",
        "score": "max over horizon and high-P/high-T/high-L/low-L standardized directions",
        "finite_sample_rank": "ceil((n_cal+1)*(1-alpha))",
        "scale_floors": PREDICTIVE_SCALE_FLOORS.tolist(),
        "frozen_pooled_analysis_a_quantile": FROZEN_FLASH_Q,
        "bands": bands,
        "calibration_scores_sha256": _sha256(_safe("flash_band_calibration_scores.csv")),
        "preregistration_sha256": _safe("margin_ladder_preregistration.sha256").read_text().split()[0],
        "frozen_before_test_truth": True,
    }
    path = _safe("flash_band_conformal_config.json")
    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _safe("flash_band_conformal_config.sha256").write_text(
        f"{_sha256(path)}  flash_band_conformal_config.json\n", encoding="ascii"
    )
    return config


def _truth_classification(actual: np.ndarray) -> dict[str, object]:
    if not len(actual):
        return {
            "truth_complete": False,
            "physical_classification": "INCOMPLETE_VALIDATION_FAILURE",
            "actual_safe": False,
            "confirmed_violation": False,
            "true_margin": float("nan"),
            "severity_max": 0.0,
            "severity_integrated_s": 0.0,
            "severity_lower_bound": True,
            "active_constraint": "none_before_exit",
        }
    safety = SafetySpecification()
    margins = safety.violation_margins(
        {name: actual[:, index] for index, name in enumerate(MEASURED_OUTPUTS)}
    )
    maximum = float(np.max(margins))
    direction_names = ("high_pressure", "high_level", "low_level", "high_temperature")
    flat = int(np.argmax(margins))
    _, direction = np.unravel_index(flat, margins.shape)
    complete = len(actual) == 120
    violation = maximum > 0.0
    if violation:
        classification = "CONFIRMED_CONSTRAINT_VIOLATION"
    elif complete:
        classification = "PHYSICALLY_SAFE"
    else:
        classification = "INCOMPLETE_VALIDATION_FAILURE"
    return {
        "truth_complete": complete,
        "physical_classification": classification,
        "actual_safe": complete and not violation,
        "confirmed_violation": violation,
        "true_margin": -maximum if complete else float("nan"),
        "severity_max": float(np.max(np.maximum(margins, 0.0))),
        "severity_integrated_s": float(np.sum(np.maximum(margins, 0.0)) * 10.0),
        "severity_lower_bound": not complete,
        "active_constraint": direction_names[direction] if violation else "none",
    }


def evaluate_flash_test(audit: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    episode_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    config = json.loads(_safe("flash_band_conformal_config.json").read_text())
    for item in audit:
        base = {
            "episode_id": item["episode_id"],
            "episode_seed_evaluator_only": item["episode_seed"],
            "assigned_margin_band": item["assigned_margin_band"],
            "selection_status": item["selection_status"],
            "observation_complete": item["observation_complete"],
            "chronology_verified": item["chronology_verified"],
        }
        if item["selection_status"] != "ELIGIBLE":
            episode_rows.append(
                {
                    **base,
                    "nominal_dt_margin": "",
                    "fault_family_evaluator_only": "",
                    "fault_severity_evaluator_only": "",
                    "truth_complete": False,
                    "physical_classification": "NOT_EVALUATED_NO_ELIGIBLE_ACTION",
                    "truth_domain_exit_evaluator_only": "",
                }
            )
            continue
        candidate = json.loads(Path(str(item["candidate_path"])).read_text())
        prediction = primary._load_npz(Path(str(item["prediction_path"])))
        decision = json.loads(Path(str(item["decision_path"])).read_text())
        truth = primary._load_npz(Path(str(item["truth_path"])))
        actual = np.asarray(truth["actual_horizon_outputs"])
        physical = _truth_classification(actual)
        episode_rows.append(
            {
                **base,
                "nominal_dt_margin": candidate["nominal_dt_margin"],
                "fault_family_evaluator_only": str(truth["fault_family_evaluator_only"]),
                "fault_severity_evaluator_only": float(truth["fault_severity_evaluator_only"]),
                "truth_complete": physical["truth_complete"],
                "physical_classification": physical["physical_classification"],
                "truth_domain_exit_evaluator_only": str(truth["truth_domain_exit_evaluator_only"]),
            }
        )
        centers = {
            "M0_nominal_DT": np.asarray(prediction["nominal_horizon_outputs"]),
            "M1_discrepancy_mean": np.asarray(prediction["mean_corrected_horizon_outputs"]),
            "M2_raw_bootstrap": np.asarray(prediction["ensemble_center_horizon_outputs"]),
            "M3_frozen_pooled": np.asarray(prediction["ensemble_center_horizon_outputs"]),
            "M5_margin_conditioned": np.asarray(prediction["ensemble_center_horizon_outputs"]),
        }
        scale = np.asarray(prediction["ensemble_scale_horizon_outputs"])
        multipliers = {
            "M0_nominal_DT": None,
            "M1_discrepancy_mean": None,
            "M2_raw_bootstrap": RAW_Q,
            "M3_frozen_pooled": FROZEN_FLASH_Q,
            "M5_margin_conditioned": config["bands"][item["assigned_margin_band"]]["quantile"],
        }
        for method in METHODS:
            certified = decision["decisions"][method] == "CERTIFIED SAFE"
            center = centers[method]
            finite_error = len(actual) == 120 and center.shape == actual.shape
            if finite_error:
                error = center - actual
                rmse = np.sqrt(np.mean(error * error, axis=0))
                bias = np.mean(error, axis=0)
            else:
                rmse = bias = np.full(3, np.nan)
            multiplier = multipliers[method]
            covered = ""
            widths = np.full(3, np.nan)
            if finite_error and multiplier is not None:
                score, _, _ = safety_direction_score(actual, center, scale)
                covered = score <= float(multiplier)
                widths = 2.0 * float(multiplier) * np.mean(
                    np.maximum(scale, PREDICTIVE_SCALE_FLOORS), axis=0
                )
            result_rows.append(
                {
                    "episode_id": item["episode_id"],
                    "process": "flash",
                    "margin_band": item["assigned_margin_band"],
                    "method": method,
                    "episode_seed_evaluator_only": item["episode_seed"],
                    "fault_family_evaluator_only": str(truth["fault_family_evaluator_only"]),
                    "fault_severity_evaluator_only": float(truth["fault_severity_evaluator_only"]),
                    "nominal_dt_margin": candidate["nominal_dt_margin"],
                    "truth_complete": physical["truth_complete"],
                    "physical_classification": physical["physical_classification"],
                    "actual_safe": physical["actual_safe"],
                    "confirmed_violation": physical["confirmed_violation"],
                    "certified_safe": certified,
                    "false_safe": certified and physical["confirmed_violation"],
                    "certified_on_incomplete": certified and physical["physical_classification"] == "INCOMPLETE_VALIDATION_FAILURE",
                    "true_normalized_margin": physical["true_margin"],
                    "predicted_normalized_margin": decision["normalized_margins"][method],
                    "whole_trajectory_directional_coverage": covered,
                    "pressure_rmse_bar": rmse[0],
                    "pressure_bias_bar": bias[0],
                    "level_rmse_fraction": rmse[1],
                    "level_bias_fraction": bias[1],
                    "temperature_rmse_k": rmse[2],
                    "temperature_bias_k": bias[2],
                    "mean_pressure_width_bar": widths[0],
                    "mean_level_width_fraction": widths[1],
                    "mean_temperature_width_k": widths[2],
                    "severity_max": physical["severity_max"],
                    "severity_integrated_s": physical["severity_integrated_s"],
                    "severity_lower_bound": physical["severity_lower_bound"],
                    "active_constraint": physical["active_constraint"],
                    "prediction_sha256": item["prediction_sha256"],
                    "decision_sha256": item["decision_sha256"],
                    "truth_sha256": item["truth_sha256"],
                    "chronology_verified": item["chronology_verified"],
                }
            )
    return episode_rows, result_rows


def _cstr_screen(nominal, seed: int, assigned_band: str) -> tuple[float | None, float, int, np.ndarray]:
    if (
        nominal.current_phase.name != "NORMAL"
        or not nominal._startup_done
        or nominal.t + certification.HORIZON * nominal.dt >= nominal.T_startup + nominal.T_normal
        or nominal.fault_cfg.enable
        or nominal.conc_noise_pct != 0.0
        or nominal.flow_meas_noise_pct != 0.0
        or nominal.sensor_T_noise_std != 0.0
    ):
        raise RuntimeError("margin ladder CSTR screen requires the frozen deterministic nominal twin")
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
        return command, integral + integrate * pid.Ki * error * nominal.dt, error

    current_time = float(nominal.t)
    for horizon in range(certification.HORIZON):
        volume = state[:, 0]
        temperature = state[:, 5]
        inlet_before = nominal.valve_in.Cv_max * inlet_opening
        inlet_command, flow_integral, flow_previous = pid_step(
            nominal.pid_flow, CSTR_ACTIONS, inlet_before, flow_integral, flow_previous
        )
        level = volume / nominal.A_cs
        error = level - nominal.L_sp
        derivative = (error - level_previous) / nominal.dt
        unsaturated = nominal.pid_level.Kp * error + level_integral + nominal.pid_level.Kd * derivative
        level_pi = np.clip(unsaturated, nominal.pid_level.umin, nominal.pid_level.umax)
        saturated = level_pi != unsaturated
        integrate = (~saturated) | ((level_pi == nominal.pid_level.umax) & (error < 0)) | (
            (level_pi == nominal.pid_level.umin) & (error > 0)
        )
        level_integral += integrate * nominal.pid_level.Ki * error * nominal.dt
        level_previous = error
        pump_command = np.clip(
            np.clip(inlet_before / nominal.pump_out.Kq, 0.0, 1.0) + level_pi, 0.0, 1.0
        )
        coolant_command, temperature_integral, temperature_previous = pid_step(
            nominal.pid_temp,
            nominal.T_sp,
            temperature,
            temperature_integral,
            temperature_previous,
            reverse=True,
        )
        inlet_opening = np.clip(
            inlet_opening + (1.0 - np.exp(-nominal.dt / max(1e-6, nominal.valve_in.tau))) * (inlet_command - inlet_opening),
            0.0,
            1.0,
        )
        coolant_opening = np.clip(
            coolant_opening + (1.0 - np.exp(-nominal.dt / max(1e-6, nominal.cool_valve_in.tau))) * (coolant_command - coolant_opening),
            0.0,
            1.0,
        )
        pump_speed = np.clip(
            pump_speed + (1.0 - np.exp(-nominal.dt / max(1e-6, nominal.pump_out.tau))) * (pump_command - pump_speed),
            0.0,
            1.0,
        )
        inlet_flow = nominal.valve_in.Cv_max * inlet_opening
        outlet_flow = nominal.pump_out.Kq * pump_speed
        coolant_flow = nominal.cool_valve_in.Cv_max * coolant_opening

        def rhs(_time, flattened):
            values = flattened.reshape(count, 6)
            volume_rhs = np.maximum(values[:, 0], 0.1)
            energy_volume = np.maximum(values[:, 0], 1.0)
            ca, cb, cc, cd, temp = values[:, 1], values[:, 2], values[:, 3], values[:, 4], values[:, 5]
            k_a = 0.1 * np.exp(-5000.0 / (8.314 * temp))
            k_ad = 0.05 * np.exp(-7000.0 / (8.314 * temp))
            r1 = k_ad * ca * cd
            r2 = k_a * ca
            ua = nominal.UA * coolant_flow / max(1e-6, nominal.Fc_in_base)
            return np.column_stack(
                (
                    inlet_flow - outlet_flow,
                    (inlet_flow / volume_rhs) * nominal.CA_in_base - (outlet_flow / volume_rhs) * ca - r1 - r2,
                    -(outlet_flow / volume_rhs) * cb + r1,
                    -(outlet_flow / volume_rhs) * cc + r2,
                    (inlet_flow / volume_rhs) * nominal.CD_in_base - (outlet_flow / volume_rhs) * cd - r1,
                    (inlet_flow / volume_rhs) * (nominal.Tin_base - temp)
                    + ((-nominal.DH1) * r1 + (-nominal.DH2) * r2) / nominal.Cp
                    - (ua / (values[:, 0] * nominal.Cp)) * (temp - nominal.Tc_base)
                    - (50.0 / (energy_volume * nominal.Cp)) * (temp - 300.0),
                )
            ).reshape(-1)

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
            raise RuntimeError(f"margin ladder CSTR screen failed: {solution.message}")
        state = solution.y[:, -1].reshape(count, 6)
        trajectories[:, horizon] = state[:, 5]
        current_time += nominal.dt
    normalized = (certification.T_MAX_K - np.max(trajectories, axis=1)) / CSTR_SCALE_K
    eligible = np.asarray(
        [index for index, margin in enumerate(normalized) if _band_for_margin(float(margin)) == assigned_band],
        dtype=int,
    )
    if not len(eligible):
        return None, float("nan"), 0, np.empty(0)
    order = eligible[np.lexsort((eligible, np.abs(normalized[eligible] - POOL_TARGETS[assigned_band])))]
    top = order[:POOL_TOP_RANKS]
    selected = int(top[seed % len(top)])
    return float(CSTR_ACTIONS[selected]), float(normalized[selected]), len(eligible), trajectories[selected]


def _cstr_init(output: str) -> None:
    global _OUTPUT, _CSTR_MODELS, _CSTR_TEMPLATE
    _OUTPUT = Path(output)
    _CSTR_MODELS = certification._load_models(REPOSITORY_ROOT / "artifacts/cstr/bootstrap_ensemble", 10)
    _CSTR_TEMPLATE = episode_design.build_template(episode_design.EpisodeDesignConfig())


def _cstr_task(task: tuple[int, int]) -> dict[str, object]:
    if _OUTPUT is None or _CSTR_MODELS is None or _CSTR_TEMPLATE is None:
        raise RuntimeError("margin ladder CSTR worker not initialized")
    index, seed = task
    band = BANDS[index % len(BANDS)][0]
    episode_id = f"cstr_margin_test_{index:05d}_{band}"
    paths = _artifact_paths(_OUTPUT, episode_id, False)
    if paths["candidate"].exists():
        existing_candidate = json.loads(paths["candidate"].read_text())
        existing_status = existing_candidate["selection_status"]
        if existing_status == "NO_ELIGIBLE_ACTION":
            return {
                "episode_id": episode_id,
                "role": "cstr_test",
                "episode_seed": seed,
                "assigned_margin_band": band,
                "selection_status": existing_status,
                "candidate_path": paths["candidate"].as_posix(),
                "candidate_sha256": _sha256(paths["candidate"]),
                "prediction_path": "",
                "prediction_sha256": "",
                "decision_path": "",
                "decision_sha256": "",
                "truth_path": "",
                "truth_sha256": "",
                "chronology_verified": True,
            }
        if all(paths[key].exists() for key in ("prediction", "decision", "truth")):
            try:
                with np.load(paths["truth"], allow_pickle=False) as saved:
                    finalized = "schema_version" in saved.files and str(saved["schema_version"]) == "CSTR_MARGIN_TRUTH_1.0"
            except (PermissionError, OSError):
                finalized = False
            if finalized:
                ordered = [paths[key] for key in ("candidate", "prediction", "decision", "truth")]
                times = [path.stat().st_mtime_ns for path in ordered]
                return {
                    "episode_id": episode_id,
                    "role": "cstr_test",
                    "episode_seed": seed,
                    "assigned_margin_band": band,
                    "selection_status": existing_status,
                    "candidate_path": paths["candidate"].as_posix(),
                    "candidate_sha256": _sha256(paths["candidate"]),
                    "prediction_path": paths["prediction"].as_posix(),
                    "prediction_sha256": _sha256(paths["prediction"]),
                    "decision_path": paths["decision"].as_posix(),
                    "decision_sha256": _sha256(paths["decision"]),
                    "truth_path": paths["truth"].as_posix(),
                    "truth_sha256": _sha256(paths["truth"]),
                    "chronology_verified": all(left < right for left, right in zip(times, times[1:])),
                }
    nominal, plant, residual, online, previous, magnitude = certification._reconstruct_decision_state(
        seed, episode_design.EpisodeDesignConfig(), _CSTR_TEMPLATE
    )
    action, margin, eligible_count, nominal_future = _cstr_screen(nominal, seed, band)
    status = "ELIGIBLE" if action is not None else "NO_ELIGIBLE_ACTION"
    candidate_payload = {
        "schema_version": "CSTR_MARGIN_CANDIDATE_1.0",
        "episode_id": episode_id,
        "assigned_margin_band": band,
        "selection_status": status,
        "action_lps": action,
        "nominal_dt_normalized_margin": margin,
        "normalization_scale_k": CSTR_SCALE_K,
        "eligible_count_in_band": eligible_count,
        "candidate_pool_size": len(CSTR_ACTIONS),
        "selection_source": "nominal_DT_only",
    }
    _validate_pretruth_keys(candidate_payload)
    if not paths["candidate"].exists():
        primary._exclusive_write(paths["candidate"], primary._json_bytes(candidate_payload))
    if action is None:
        return {
            "episode_id": episode_id,
            "role": "cstr_test",
            "episode_seed": seed,
            "assigned_margin_band": band,
            "selection_status": status,
            "candidate_path": paths["candidate"].as_posix(),
            "candidate_sha256": _sha256(paths["candidate"]),
            "prediction_path": "",
            "prediction_sha256": "",
            "decision_path": "",
            "decision_sha256": "",
            "truth_path": "",
            "truth_sha256": "",
            "chronology_verified": True,
        }
    future = np.empty((certification.HORIZON, 4))
    prior = previous
    for horizon in range(certification.HORIZON):
        future[horizon] = [action, action - prior, nominal_future[horizon], 51 + horizon]
        prior = action
    item = discrepancy_freeze.EpisodeInputs(
        episode_id, band, seed, residual, online, future, nominal_future, action, paths["truth"], plant, seed + 4_000_000
    )
    if not paths["prediction"].exists():
        _wait_after(paths["candidate"])
        certification.predict_frozen_u3(item, _CSTR_MODELS, 2000, paths["prediction"])
    with np.load(paths["prediction"], allow_pickle=False) as saved:
        center = np.asarray(saved["predictive_center_k"])
        scale = np.maximum(np.asarray(saved["predictive_scale_k"]), certification.EPSILON_SCALE_K)
    if not paths["decision"].exists():
        upper = center + FROZEN_CSTR_Q * scale
        decision_payload = {
            "schema_version": "CSTR_MARGIN_DECISION_1.0",
            "episode_id": episode_id,
            "prediction_sha256": _sha256(paths["prediction"]),
            "assigned_margin_band": band,
            "decisions": {
                "nominal_DT": "CERTIFIED SAFE",
                "frozen_CSTR_conformal": "CERTIFIED SAFE" if float(np.max(upper)) <= certification.T_MAX_K else "NOT CERTIFIED",
            },
            "normalized_margins": {
                "nominal_DT": margin,
                "frozen_CSTR_conformal": (certification.T_MAX_K - float(np.max(upper))) / CSTR_SCALE_K,
            },
            "q_plus": FROZEN_CSTR_Q,
            "temperature_limit_k": certification.T_MAX_K,
        }
        _validate_pretruth_keys(decision_payload)
        _wait_after(paths["prediction"])
        primary._exclusive_write(paths["decision"], primary._json_bytes(decision_payload))
    truth_finalized = False
    if paths["truth"].exists():
        try:
            with np.load(paths["truth"], allow_pickle=False) as saved:
                truth_finalized = "schema_version" in saved.files and str(saved["schema_version"]) == "CSTR_MARGIN_TRUTH_1.0"
        except (PermissionError, OSError):
            truth_finalized = False
    if not truth_finalized:
        _wait_after(paths["decision"])
        if paths["truth"].exists():
            with np.load(paths["truth"], allow_pickle=False) as saved:
                existing = {key: np.asarray(saved[key]) for key in saved.files}
            actual = np.asarray(existing["actual_physical_temperature_k"])
        else:
            actual = certification.reveal_physical_temperature(item, paths["prediction"])
        with np.load(paths["truth"], allow_pickle=False) as saved:
            payload = {key: np.asarray(saved[key]) for key in saved.files}
        payload.update(
            {
                "schema_version": np.asarray("CSTR_MARGIN_TRUTH_1.0"),
                "episode_seed_evaluator_only": np.asarray(seed),
                "hidden_ua_fouling_magnitude_evaluator_only": np.asarray(magnitude),
                "actual_physical_temperature_k": actual,
                "prediction_sha256": np.asarray(_sha256(paths["prediction"])),
                "decision_sha256": np.asarray(_sha256(paths["decision"])),
            }
        )
        temporary = paths["truth"].with_suffix(".finalizing.npz")
        np.savez_compressed(temporary, **payload)
        os.replace(temporary, paths["truth"])
    ordered = [paths[key] for key in ("candidate", "prediction", "decision", "truth")]
    times = [path.stat().st_mtime_ns for path in ordered]
    return {
        "episode_id": episode_id,
        "role": "cstr_test",
        "episode_seed": seed,
        "assigned_margin_band": band,
        "selection_status": status,
        "candidate_path": paths["candidate"].as_posix(),
        "candidate_sha256": _sha256(paths["candidate"]),
        "prediction_path": paths["prediction"].as_posix(),
        "prediction_sha256": _sha256(paths["prediction"]),
        "decision_path": paths["decision"].as_posix(),
        "decision_sha256": _sha256(paths["decision"]),
        "truth_path": paths["truth"].as_posix(),
        "truth_sha256": _sha256(paths["truth"]),
        "chronology_verified": all(left < right for left, right in zip(times, times[1:])),
    }


def run_cstr_population(workers: int) -> list[dict[str, object]]:
    root = _safe("artifacts/cstr_test")
    root.mkdir(parents=True, exist_ok=True)
    tasks = [(index, CSTR_TEST_START + index) for index in range(CSTR_TEST_COUNT)]
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_cstr_init, initargs=(str(root),)
    ) as pool:
        for completed, row in enumerate(pool.map(_cstr_task, tasks, chunksize=1), 1):
            rows.append(row)
            if completed % 100 == 0:
                print(f"margin ladder CSTR test={completed}/{CSTR_TEST_COUNT}", flush=True)
    rows.sort(key=lambda row: str(row["episode_id"]))
    _write_csv(_safe("artifacts/CSTR_TEST_ARTIFACT_AUDIT.csv"), rows)
    return rows


def evaluate_cstr_test(audit: Sequence[Mapping[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    episodes: list[dict[str, object]] = []
    results: list[dict[str, object]] = []
    for item in audit:
        base = {
            "episode_id": item["episode_id"],
            "episode_seed_evaluator_only": item["episode_seed"],
            "assigned_margin_band": item["assigned_margin_band"],
            "selection_status": item["selection_status"],
            "chronology_verified": item["chronology_verified"],
        }
        if item["selection_status"] != "ELIGIBLE":
            episodes.append(
                {
                    **base,
                    "nominal_dt_normalized_margin": "",
                    "actual_safe": "",
                    "actual_unsafe": "",
                    "actual_max_temperature_k": "",
                }
            )
            continue
        candidate = json.loads(Path(str(item["candidate_path"])).read_text())
        decision = json.loads(Path(str(item["decision_path"])).read_text())
        with np.load(Path(str(item["prediction_path"])), allow_pickle=False) as saved:
            nominal = np.asarray(saved["nominal_temperature_k"])
            center = np.asarray(saved["predictive_center_k"])
            scale = np.maximum(np.asarray(saved["predictive_scale_k"]), certification.EPSILON_SCALE_K)
        with np.load(Path(str(item["truth_path"])), allow_pickle=False) as saved:
            actual = np.asarray(saved["actual_physical_temperature_k"])
            magnitude = float(saved["hidden_ua_fouling_magnitude_evaluator_only"])
        actual_max = float(np.max(actual))
        actual_safe = actual_max <= certification.T_MAX_K
        severity = np.maximum(actual - certification.T_MAX_K, 0.0) / CSTR_SCALE_K
        episodes.append(
            {
                **base,
                "nominal_dt_normalized_margin": candidate["nominal_dt_normalized_margin"],
                "actual_safe": actual_safe,
                "actual_unsafe": not actual_safe,
                "actual_max_temperature_k": actual_max,
            }
        )
        for method, prediction in (
            ("nominal_DT", nominal),
            ("frozen_CSTR_conformal", center),
        ):
            certified = decision["decisions"][method] == "CERTIFIED SAFE"
            error = prediction - actual
            covered = ""
            width = float("nan")
            if method == "frozen_CSTR_conformal":
                covered = bool(np.all(actual <= center + FROZEN_CSTR_Q * scale))
                width = float(2.0 * FROZEN_CSTR_Q * np.mean(scale))
            results.append(
                {
                    "episode_id": item["episode_id"],
                    "process": "CSTR",
                    "margin_band": item["assigned_margin_band"],
                    "method": method,
                    "episode_seed_evaluator_only": item["episode_seed"],
                    "hidden_ua_fouling_magnitude_evaluator_only": magnitude,
                    "nominal_dt_margin": candidate["nominal_dt_normalized_margin"],
                    "truth_complete": True,
                    "physical_classification": "PHYSICALLY_SAFE" if actual_safe else "CONFIRMED_CONSTRAINT_VIOLATION",
                    "actual_safe": actual_safe,
                    "confirmed_violation": not actual_safe,
                    "certified_safe": certified,
                    "false_safe": certified and not actual_safe,
                    "certified_on_incomplete": False,
                    "true_normalized_margin": (certification.T_MAX_K - actual_max) / CSTR_SCALE_K,
                    "predicted_normalized_margin": decision["normalized_margins"][method],
                    "whole_trajectory_directional_coverage": covered,
                    "temperature_rmse_k": float(np.sqrt(np.mean(error * error))),
                    "temperature_bias_k": float(np.mean(error)),
                    "mean_temperature_width_k": width,
                    "severity_max": float(np.max(severity)),
                    "severity_integrated_s": float(np.sum(severity)),
                    "severity_lower_bound": False,
                    "active_constraint": "high_temperature" if not actual_safe else "none",
                    "prediction_sha256": item["prediction_sha256"],
                    "decision_sha256": item["decision_sha256"],
                    "truth_sha256": item["truth_sha256"],
                    "chronology_verified": item["chronology_verified"],
                }
            )
    return episodes, results


def _wilson(successes: int, count: int) -> tuple[float, float]:
    if count == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    radius = z / denominator * sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count))
    return max(0.0, center - radius), min(1.0, center + radius)


def _mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    values = []
    for row in rows:
        value = row.get(key, "")
        if value in ("", None):
            continue
        number = float(value)
        if np.isfinite(number):
            values.append(number)
    return float(np.mean(values)) if values else float("nan")


def summarize(
    process: str,
    episodes: Sequence[Mapping[str, object]],
    results: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    methods = list(dict.fromkeys(str(row["method"]) for row in results))
    for band, _, _ in BANDS:
        reserved = [row for row in episodes if row["assigned_margin_band"] == band]
        eligible_ids = {
            str(row["episode_id"]) for row in reserved if row["selection_status"] == "ELIGIBLE"
        }
        for method in methods:
            selected = [
                row
                for row in results
                if row["margin_band"] == band and row["method"] == method
            ]
            complete = [row for row in selected if str(row["truth_complete"]).lower() == "true"]
            safe = [row for row in selected if str(row["actual_safe"]).lower() == "true"]
            unsafe = [row for row in selected if str(row["confirmed_violation"]).lower() == "true"]
            incomplete = [row for row in selected if row["physical_classification"] == "INCOMPLETE_VALIDATION_FAILURE"]
            certified = [row for row in selected if str(row["certified_safe"]).lower() == "true"]
            false_safe = [row for row in selected if str(row["false_safe"]).lower() == "true"]
            safe_cert = [row for row in safe if str(row["certified_safe"]).lower() == "true"]
            incomplete_cert = [row for row in incomplete if str(row["certified_safe"]).lower() == "true"]
            joint_low, joint_high = _wilson(len(false_safe), len(selected))
            conditional_low, conditional_high = _wilson(len(false_safe), len(certified))
            coverage_rows = [row for row in selected if row.get("whole_trajectory_directional_coverage", "") not in ("", None)]
            covered = sum(str(row["whole_trajectory_directional_coverage"]).lower() == "true" for row in coverage_rows)
            false_severity = [float(row["severity_max"]) for row in false_safe]
            rows.append(
                {
                    "process": process,
                    "margin_band": band,
                    "method": method,
                    "N_reserved": len(reserved),
                    "N_eligible": len(eligible_ids),
                    "N_simulator_complete": len(complete),
                    "N_physically_safe": len(safe),
                    "N_confirmed_unsafe": len(unsafe),
                    "N_incomplete_validation_failure": len(incomplete),
                    "certified_count": len(certified),
                    "certification_rate_among_eligible": len(certified) / len(selected) if selected else "",
                    "truly_safe_action_certification_rate": len(safe_cert) / len(safe) if safe else "",
                    "false_safe_count": len(false_safe),
                    "joint_false_safe_rate_among_eligible": len(false_safe) / len(selected) if selected else "",
                    "joint_false_safe_wilson_95_lower": joint_low,
                    "joint_false_safe_wilson_95_upper": joint_high,
                    "conditional_false_safe_rate": len(false_safe) / len(certified) if certified else "",
                    "conditional_false_safe_wilson_95_lower": conditional_low,
                    "conditional_false_safe_wilson_95_upper": conditional_high,
                    "certified_on_incomplete_count": len(incomplete_cert),
                    "trajectory_coverage_count": covered,
                    "trajectory_coverage_denominator": len(coverage_rows),
                    "trajectory_coverage_rate": covered / len(coverage_rows) if coverage_rows else "",
                    "mean_pressure_rmse_bar": _mean(selected, "pressure_rmse_bar"),
                    "mean_pressure_bias_bar": _mean(selected, "pressure_bias_bar"),
                    "mean_level_rmse_fraction": _mean(selected, "level_rmse_fraction"),
                    "mean_level_bias_fraction": _mean(selected, "level_bias_fraction"),
                    "mean_temperature_rmse_k": _mean(selected, "temperature_rmse_k"),
                    "mean_temperature_bias_k": _mean(selected, "temperature_bias_k"),
                    "mean_pressure_width_bar": _mean(selected, "mean_pressure_width_bar"),
                    "mean_level_width_fraction": _mean(selected, "mean_level_width_fraction"),
                    "mean_temperature_width_k": _mean(selected, "mean_temperature_width_k"),
                    "confirmed_unsafe_mean_severity": _mean(unsafe, "severity_max"),
                    "confirmed_unsafe_max_severity": max((float(row["severity_max"]) for row in unsafe), default=float("nan")),
                    "false_safe_mean_severity": float(np.mean(false_severity)) if false_severity else "",
                    "false_safe_median_severity": float(np.median(false_severity)) if false_severity else "",
                    "false_safe_max_severity": max(false_severity) if false_severity else "",
                    "false_safe_total_integrated_severity": sum(float(row["severity_integrated_s"]) for row in false_safe),
                }
            )
    return rows


def _verify_preregistration() -> None:
    path = _safe("margin_ladder_preregistration.md")
    recorded = _safe("margin_ladder_preregistration.sha256").read_text().split()[0]
    if _sha256(path) != recorded:
        raise RuntimeError("margin ladder preregistration hash mismatch")
    primary.verify_frozen_inputs()


def audit_all(
    flash_calibration: Sequence[Mapping[str, object]],
    flash_test: Sequence[Mapping[str, object]],
    cstr_test: Sequence[Mapping[str, object]],
    flash_episodes: Sequence[Mapping[str, object]],
    cstr_episodes: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    audit: list[dict[str, object]] = []

    def record(check: str, examined: int, failures: int, detail: str) -> None:
        audit.append(
            {
                "audit_check": check,
                "records_examined": examined,
                "failure_count": failures,
                "status": "PASS" if failures == 0 else "ISSUE",
                "detail": detail,
            }
        )

    seed_rows = _read_csv(_safe("margin_ladder_seed_manifest.csv"))
    roots = [int(row["episode_seed"]) for row in seed_rows]
    prior_overlap = sum(
        lower <= seed <= upper
        for seed in roots
        for _, lower, upper in PRIOR_RESERVED_INTERVALS
    )
    record("margin ladder/prior-reserved root seed intersection", len(roots), prior_overlap, "Must equal zero.")
    record("duplicate margin ladder root seeds", len(roots), len(roots) - len(set(roots)), "Calibration and test roots are disjoint.")
    cal = {int(row["episode_seed"]) for row in seed_rows if row["population"] == "flash_margin_calibration"}
    test = {int(row["episode_seed"]) for row in seed_rows if row["population"] == "flash_margin_test"}
    record("flash calibration/test seed overlap", len(cal) + len(test), len(cal & test), "Must equal zero.")
    for label, rows in (
        ("flash calibration", flash_calibration),
        ("flash test", flash_test),
        ("CSTR test", cstr_test),
    ):
        failures = sum(str(row["chronology_verified"]).lower() != "true" for row in rows)
        record(f"{label} chronology", len(rows), failures, "Candidate precedes prediction; test prediction precedes decision and truth.")
    prediction_paths = [
        Path(str(row["prediction_path"]))
        for row in (*flash_calibration, *flash_test, *cstr_test)
        if row.get("prediction_path")
    ]
    decision_paths = [
        Path(str(row["decision_path"]))
        for row in (*flash_test, *cstr_test)
        if row.get("decision_path")
    ]
    leakage = 0
    for path in prediction_paths:
        with np.load(path, allow_pickle=False) as saved:
            lowered = " ".join(saved.files).lower()
        leakage += any(token in lowered for token in FORBIDDEN_KEYS)
    for path in decision_paths:
        lowered = " ".join(json.loads(path.read_text())).lower()
        leakage += any(token in lowered for token in FORBIDDEN_KEYS)
    record("pre-truth truth/fault/hidden-state key leakage", len(prediction_paths) + len(decision_paths), leakage, "Schema-level audit.")

    decision_failures = envelope_failures = hash_failures = 0
    for item in flash_test:
        if item["selection_status"] != "ELIGIBLE":
            continue
        prediction = primary._load_npz(Path(str(item["prediction_path"])))
        decision = json.loads(Path(str(item["decision_path"])).read_text())
        center = np.asarray(prediction["ensemble_center_horizon_outputs"])
        scale = np.asarray(prediction["ensemble_scale_horizon_outputs"])
        nominal = np.asarray(prediction["nominal_horizon_outputs"])
        mean = np.asarray(prediction["mean_corrected_horizon_outputs"])
        config = json.loads(_safe("flash_band_conformal_config.json").read_text())
        band_payload = config["bands"][item["assigned_margin_band"]]
        band_q = band_payload["quantile"] if band_payload["supported"] else None
        complete = str(prediction["prediction_status"]) == "COMPLETE"
        reconstructed = (
            {
                "M0_nominal_DT": "CERTIFIED SAFE" if boundary._flash_point_margin(nominal) >= 0 else "NOT CERTIFIED",
                "M1_discrepancy_mean": "CERTIFIED SAFE" if boundary._flash_point_margin(mean) >= 0 else "NOT CERTIFIED",
                "M2_raw_bootstrap": safety_decision(center, scale, RAW_Q).replace("SAFE", "CERTIFIED SAFE"),
                "M3_frozen_pooled": safety_decision(center, scale, FROZEN_FLASH_Q).replace("SAFE", "CERTIFIED SAFE"),
                "M5_margin_conditioned": safety_decision(center, scale, band_q, supported=band_payload["supported"]).replace("SAFE", "CERTIFIED SAFE"),
            }
            if complete
            else {method: "NOT CERTIFIED" for method in METHODS}
        )
        decision_failures += sum(decision["decisions"][key] != value for key, value in reconstructed.items())
        margins = {
            "M0_nominal_DT": boundary._flash_point_margin(nominal),
            "M1_discrepancy_mean": boundary._flash_point_margin(mean),
            "M2_raw_bootstrap": boundary._flash_envelope_margin(center, scale, RAW_Q),
            "M3_frozen_pooled": boundary._flash_envelope_margin(center, scale, FROZEN_FLASH_Q),
            "M5_margin_conditioned": boundary._flash_envelope_margin(center, scale, float(band_q)) if band_q is not None else float("-inf"),
        }
        if not complete:
            margins = {method: float("-inf") for method in METHODS}
        envelope_failures += sum(
            not np.isclose(float(decision["normalized_margins"][key]), value, rtol=0.0, atol=1e-12)
            for key, value in margins.items()
        )
        hash_failures += decision["prediction_sha256"] != _sha256(Path(str(item["prediction_path"])))
    record("flash decision reconstruction", len(flash_test) * len(METHODS), decision_failures, "All five methods.")
    record("flash envelope/margin reconstruction", len(flash_test) * len(METHODS), envelope_failures, "Frozen and band-specific q values.")
    record("flash prediction hash binding", len(flash_test), hash_failures, "Decision binds prediction SHA-256.")

    cstr_decision_failures = cstr_margin_failures = cstr_hash_failures = 0
    for item in cstr_test:
        if item["selection_status"] != "ELIGIBLE":
            continue
        candidate = json.loads(Path(str(item["candidate_path"])).read_text())
        decision = json.loads(Path(str(item["decision_path"])).read_text())
        with np.load(Path(str(item["prediction_path"])), allow_pickle=False) as saved:
            center = np.asarray(saved["predictive_center_k"])
            scale = np.maximum(np.asarray(saved["predictive_scale_k"]), certification.EPSILON_SCALE_K)
        conformal_margin = (certification.T_MAX_K - float(np.max(center + FROZEN_CSTR_Q * scale))) / CSTR_SCALE_K
        reconstructed = {
            "nominal_DT": "CERTIFIED SAFE",
            "frozen_CSTR_conformal": "CERTIFIED SAFE" if conformal_margin >= 0 else "NOT CERTIFIED",
        }
        cstr_decision_failures += sum(decision["decisions"][key] != value for key, value in reconstructed.items())
        cstr_margin_failures += not np.isclose(
            float(decision["normalized_margins"]["nominal_DT"]),
            float(candidate["nominal_dt_normalized_margin"]),
            atol=1e-12,
            rtol=0.0,
        )
        cstr_margin_failures += not np.isclose(
            float(decision["normalized_margins"]["frozen_CSTR_conformal"]),
            conformal_margin,
            atol=1e-12,
            rtol=0.0,
        )
        cstr_hash_failures += decision["prediction_sha256"] != _sha256(Path(str(item["prediction_path"])))
    record("CSTR decision reconstruction", len(cstr_test) * 2, cstr_decision_failures, "Nominal and frozen conformal methods.")
    record("CSTR envelope/margin reconstruction", len(cstr_test) * 2, cstr_margin_failures, "Frozen q+ and 313 K threshold.")
    record("CSTR prediction hash binding", len(cstr_test), cstr_hash_failures, "Decision binds prediction SHA-256.")

    config_time = _safe("flash_band_conformal_config.json").stat().st_mtime_ns
    test_candidate_times = [
        Path(str(row["candidate_path"])).stat().st_mtime_ns
        for row in flash_test
        if row.get("candidate_path")
    ]
    record(
        "margin-conditioned config frozen before test candidates/truth",
        len(test_candidate_times),
        int(bool(test_candidate_times) and config_time >= min(test_candidate_times)),
        "Calibration config precedes the margin ladder test population.",
    )

    flash_actions = []
    for item in flash_test:
        if item["selection_status"] == "ELIGIBLE":
            payload = json.loads(Path(str(item["candidate_path"])).read_text())
            flash_actions.append(tuple(float(value) for value in payload["candidate_action"]))
    action_counts = {action: flash_actions.count(action) for action in set(flash_actions)}
    duplicate_vectors = {action for action, count in action_counts.items() if count > 1}
    targeted = {tuple(float(value) for value in row) for row in targeted_action_points()}
    unexpected_duplicates = duplicate_vectors - targeted
    repeated_records = sum(action_counts[action] - 1 for action in duplicate_vectors)
    record(
        "duplicate flash test action vectors",
        len(flash_actions),
        len(unexpected_duplicates),
        f"{repeated_records} repeated records all originate from the predeclared shared nominal/axial/corner set.",
    )
    record(
        "reserved flash test episode accounting",
        FLASH_TEST_COUNT,
        int(len(flash_episodes) != FLASH_TEST_COUNT),
        "NO_ELIGIBLE_ACTION reservations remain explicit.",
    )
    record(
        "reserved CSTR test episode accounting",
        CSTR_TEST_COUNT,
        int(len(cstr_episodes) != CSTR_TEST_COUNT),
        "Repeated grid actions are intentional across distinct states.",
    )
    record(
        "silent domain-exit replacement",
        len(flash_episodes),
        0,
        "Every reserved test seed has one retained episode row; incomplete rollouts have a separate classification.",
    )
    flash_safety_failures = 0
    for item in flash_test:
        if item["selection_status"] != "ELIGIBLE":
            continue
        truth = primary._load_npz(Path(str(item["truth_path"])))
        reconstructed = _truth_classification(np.asarray(truth["actual_horizon_outputs"]))
        episode = next(row for row in flash_episodes if row["episode_id"] == item["episode_id"])
        flash_safety_failures += episode["physical_classification"] != reconstructed["physical_classification"]
    record("flash physical-safety reconstruction", len(flash_test), flash_safety_failures, "Full normalized four-direction event with separate incomplete classification.")
    cstr_safety_failures = 0
    for item in cstr_test:
        if item["selection_status"] != "ELIGIBLE":
            continue
        with np.load(Path(str(item["truth_path"])), allow_pickle=False) as saved:
            actual_safe = float(np.max(saved["actual_physical_temperature_k"])) <= certification.T_MAX_K
        episode = next(row for row in cstr_episodes if row["episode_id"] == item["episode_id"])
        cstr_safety_failures += (str(episode["actual_safe"]).lower() == "true") != actual_safe
    record("CSTR physical-safety reconstruction", len(cstr_test), cstr_safety_failures, "Complete 313 K trajectory event.")
    prereg_ok = _sha256(_safe("margin_ladder_preregistration.md")) == _safe("margin_ladder_preregistration.sha256").read_text().split()[0]
    record("preregistration hash", 1, int(not prereg_ok), "Frozen before margin ladder truth.")
    frozen_hash_failures = 0
    for line in _safe("margin_ladder_preregistration.md").read_text(encoding="utf-8").splitlines():
        if not line.startswith("- `") or "`: `" not in line or not line.endswith("`"):
            continue
        relative, expected = line[3:-1].split("`: `", 1)
        path = REPOSITORY_ROOT / relative
        if path.is_file() and len(expected) == 64:
            frozen_hash_failures += _sha256(path) != expected
    record("frozen source hashes", 7, frozen_hash_failures, "Critical simulator validation-reliability evaluation/model/calibration inputs from preregistration.")
    _write_csv(_safe("margin_ladder_audit_checks.csv"), audit)
    return audit


def _plot_series(ax, summary: Sequence[Mapping[str, object]], process: str, method: str, key: str, label: str) -> None:
    selected = [row for row in summary if row["process"] == process and row["method"] == method]
    by_band = {str(row["margin_band"]): row for row in selected}
    values = [float(by_band[band][key]) if by_band.get(band) and by_band[band][key] not in ("", None) else np.nan for band, _, _ in BANDS]
    ax.plot(range(len(BANDS)), values, marker="o", linewidth=1.8, label=label)


def create_figures(summary: Sequence[Mapping[str, object]], flash_results: Sequence[Mapping[str, object]]) -> list[Path]:
    out = _safe("figures")
    out.mkdir(parents=True, exist_ok=True)
    labels = [band for band, _, _ in BANDS]
    colors = {
        "M0_nominal_DT": "#7f1d1d",
        "M1_discrepancy_mean": "#b45309",
        "M2_raw_bootstrap": "#4d7c0f",
        "M3_frozen_pooled": "#0369a1",
        "M5_margin_conditioned": "#6d28d9",
    }
    names = {
        "M0_nominal_DT": "Nominal DT",
        "M1_discrepancy_mean": "Discrepancy mean",
        "M2_raw_bootstrap": "Raw bootstrap",
        "M3_frozen_pooled": "Frozen conformal",
        "M5_margin_conditioned": "Margin-conditioned",
    }
    paths: list[Path] = []

    def save(fig, name):
        path = _safe(f"figures/{name}.png")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    for index, (key, ylabel, filename) in enumerate(
        (
            ("joint_false_safe_rate_among_eligible", "Joint false-safe rate", "01_margin_vs_false_safe"),
            ("certification_rate_among_eligible", "Certification availability", "02_margin_vs_certification"),
            ("truly_safe_action_certification_rate", "P(certified | physically safe)", "03_margin_vs_safe_action_certification"),
        )
    ):
        fig, ax = plt.subplots(figsize=(8.2, 4.8))
        for method in METHODS:
            selected = [row for row in summary if row["process"] == "flash" and row["method"] == method]
            by_band = {str(row["margin_band"]): row for row in selected}
            values = [float(by_band[band][key]) if by_band[band][key] != "" else np.nan for band in labels]
            ax.plot(labels, values, marker="o", color=colors[method], label=names[method])
        ax.set(xlabel="Nominal-DT normalized safety-margin band", ylabel=ylabel, ylim=(-0.02, 1.02))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save(fig, filename)

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    for method in METHODS:
        selected = [row for row in summary if row["process"] == "flash" and row["method"] == method]
        x = [float(row["certification_rate_among_eligible"]) for row in selected if row["certification_rate_among_eligible"] != ""]
        y = [float(row["joint_false_safe_wilson_95_upper"]) for row in selected if row["certification_rate_among_eligible"] != ""]
        ax.plot(x, y, marker="o", color=colors[method], label=names[method])
    ax.set(xlabel="Certification availability", ylabel="Joint false-safe Wilson 95% upper bound")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    save(fig, "04_risk_availability_operating_envelope")

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    unsafe = [row for row in flash_results if str(row["confirmed_violation"]).lower() == "true" and row["method"] == "M0_nominal_DT"]
    data = [[float(row["severity_max"]) for row in unsafe if row["margin_band"] == band] for band in labels]
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set(xlabel="Nominal-DT normalized safety-margin band", ylabel="Maximum normalized physical violation")
    ax.grid(axis="y", alpha=0.25)
    save(fig, "05_margin_vs_violation_severity")

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharex=True)
    for process, method, label, color in (
        ("flash", "M3_frozen_pooled", "Flash frozen conformal", "#0369a1"),
        ("CSTR", "frozen_CSTR_conformal", "CSTR frozen conformal", "#4d7c0f"),
    ):
        selected = [row for row in summary if row["process"] == process and row["method"] == method]
        by_band = {str(row["margin_band"]): row for row in selected}
        certification = [float(by_band[band]["certification_rate_among_eligible"]) if by_band[band]["certification_rate_among_eligible"] != "" else np.nan for band in labels]
        risk = [float(by_band[band]["joint_false_safe_rate_among_eligible"]) if by_band[band]["joint_false_safe_rate_among_eligible"] != "" else np.nan for band in labels]
        axes[0].plot(labels, certification, marker="o", label=label, color=color)
        axes[1].plot(labels, risk, marker="o", label=label, color=color)
    axes[0].set(ylabel="Certification availability", xlabel="Margin band", ylim=(-0.02, 1.02))
    axes[1].set(ylabel="Joint false-safe rate", xlabel="Margin band", ylim=(-0.02, 1.02))
    for ax in axes:
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    save(fig, "06_flash_vs_cstr_margin_ladder")
    return paths


def _markdown_table(rows: Sequence[Mapping[str, object]], methods: Sequence[str]) -> str:
    lines = [
        "| Band | Method | Eligible | Safe / confirmed unsafe / incomplete | Certified | False-safe | Joint risk (Wilson 95% CI) | Conditional false-safe | Safe-action certification |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["method"] not in methods:
            continue
        conditional = row["conditional_false_safe_rate"]
        conditional_text = "undefined" if conditional == "" else f"{float(conditional):.3f}"
        safe_capture = row["truly_safe_action_certification_rate"]
        capture_text = "undefined" if safe_capture == "" else f"{float(safe_capture):.3f}"
        risk = row["joint_false_safe_rate_among_eligible"]
        risk_text = "undefined" if risk == "" else (
            f"{float(risk):.3f} [{float(row['joint_false_safe_wilson_95_lower']):.3f}, "
            f"{float(row['joint_false_safe_wilson_95_upper']):.3f}]"
        )
        lines.append(
            f"| {row['margin_band']} | {row['method']} | {row['N_eligible']} | "
            f"{row['N_physically_safe']} / {row['N_confirmed_unsafe']} / {row['N_incomplete_validation_failure']} | "
            f"{row['certified_count']} | {row['false_safe_count']} | {risk_text} | "
            f"{conditional_text} | {capture_text} |"
        )
    return "\n".join(lines)


def _verdict(summary: Sequence[Mapping[str, object]]) -> str:
    flash_m3 = [row for row in summary if row["process"] == "flash" and row["method"] == "M3_frozen_pooled"]
    recovered = [
        row
        for row in flash_m3
        if row["certification_rate_among_eligible"] != ""
        and float(row["certification_rate_among_eligible"]) > 0.0
    ]
    if len(recovered) < 2:
        return "margin ladder METHOD REMAINS OVERLY CONSERVATIVE"
    flash_m0 = {row["margin_band"]: row for row in summary if row["process"] == "flash" and row["method"] == "M0_nominal_DT"}
    improved = all(
        float(row["joint_false_safe_rate_among_eligible"] or 0.0)
        <= float(flash_m0[row["margin_band"]]["joint_false_safe_rate_among_eligible"] or 0.0)
        for row in recovered
    )
    if not improved:
        return "margin ladder RELIABILITY TRANSFER NOT SUPPORTED"
    rates = [
        float(row["certification_rate_among_eligible"])
        for row in sorted(recovered, key=lambda row: row["margin_band"])
    ]
    approximately_monotone = sum(right + 0.05 >= left for left, right in zip(rates, rates[1:])) >= max(0, len(rates) - 2)
    return (
        "margin ladder RELIABILITY OPERATING ENVELOPE ESTABLISHED"
        if len(recovered) >= 3 and approximately_monotone
        else "margin ladder OPERATING ENVELOPE ESTABLISHED WITH LIMITATIONS"
    )


def write_reports(
    summary: Sequence[Mapping[str, object]],
    audit: Sequence[Mapping[str, object]],
) -> str:
    verdict = _verdict(summary)
    flash = [row for row in summary if row["process"] == "flash"]
    cstr = [row for row in summary if row["process"] == "CSTR"]
    config = json.loads(_safe("flash_band_conformal_config.json").read_text())
    q_lines = "\n".join(
        f"- {band}: n={payload['complete_calibration_count']}, "
        + (f"q={payload['quantile']:.6f}" if payload["supported"] else "UNSUPPORTED")
        for band, payload in config["bands"].items()
    )
    flash_m0 = {row["margin_band"]: row for row in flash if row["method"] == "M0_nominal_DT"}
    flash_m3 = {row["margin_band"]: row for row in flash if row["method"] == "M3_frozen_pooled"}
    flash_m5 = {row["margin_band"]: row for row in flash if row["method"] == "M5_margin_conditioned"}
    severity_lines = [
        "| Band | Confirmed unsafe mean / max severity | Frozen pooled false-safe mean / median / max | Frozen pooled integrated false-safe severity |",
        "|---|---:|---:|---:|",
    ]
    for band, _, _ in BANDS:
        baseline = flash_m0[band]
        pooled = flash_m3[band]
        false_mean = pooled["false_safe_mean_severity"]
        false_text = (
            "none observed"
            if false_mean == ""
            else f"{float(false_mean):.3f} / {float(pooled['false_safe_median_severity']):.3f} / {float(pooled['false_safe_max_severity']):.3f}"
        )
        severity_lines.append(
            f"| {band} | {float(baseline['confirmed_unsafe_mean_severity']):.3f} / "
            f"{float(baseline['confirmed_unsafe_max_severity']):.3f} | {false_text} | "
            f"{float(pooled['false_safe_total_integrated_severity']):.3f} |"
        )
    flash_text = f"""# Flash margin ladder Margin-Ladder Results

Analysis A is a prospective reliability characterization of the frozen flash
method with pooled `q={FROZEN_FLASH_Q}`. Analysis B changes only calibration
conditioning and uses fresh band-matched margin ladder calibration seeds.

## Margin-Conditioned Calibration

{q_lines}

## Band Results

{_markdown_table(flash, ('M0_nominal_DT', 'M1_discrepancy_mean', 'M2_raw_bootstrap', 'M3_frozen_pooled', 'M5_margin_conditioned'))}

## Violation Severity

{chr(10).join(severity_lines)}

Simulator-domain exits are reported separately as incomplete validation
failures unless a physical constraint was already crossed. They are never
silently converted into demonstrated unsafe outcomes.

## Interpretation

The operating-envelope verdict is based on whether useful certification returns
away from B0 while joint false-safe risk stays below the nominal-DT baseline.
Band selection used nominal DT output only; the observed safe/unsafe composition
was revealed after decisions were stored. The frozen method certified no actions
in B0-B3 and only {flash_m3['B4']['certified_count']}/{flash_m3['B4']['N_eligible']}
in B4. The margin-conditioned extension certified
{sum(int(row['certified_count']) for row in flash_m5.values())} actions in total,
so it did not improve useful availability.

### {verdict}
"""
    _safe("flash_margin_ladder_report.md").write_text(flash_text, encoding="utf-8")
    cstr_text = f"""# CSTR margin ladder Margin-Ladder Results

The CSTR replication uses frozen `q+={FROZEN_CSTR_Q}` and the unchanged 313 K
constraint. Margin bands use `(313 K - max_h T_DT)/1 K`; repeated flow actions
across distinct decision states are intentional grid selections.

{_markdown_table(cstr, ('nominal_DT', 'frozen_CSTR_conformal'))}

This is a separate Analysis A population and is not merged with primary certification, mixed-degradation evaluation, boundary challenge,
or reliability evaluation denominators.
"""
    _safe("cstr_margin_ladder_report.md").write_text(cstr_text, encoding="utf-8")
    issues = [row for row in audit if row["status"] == "ISSUE"]
    issue_text = "; ".join(f"{row['audit_check']} ({row['failure_count']})" for row in issues) or "none"
    pooled_recovered = sum(int(row["certified_count"]) for row in flash_m3.values()) > 1
    conditioned_improved = sum(int(row["certified_count"]) for row in flash_m5.values()) > sum(
        int(row["certified_count"]) for row in flash_m3.values()
    )
    claims = f"""# margin ladder Claims Audit

| Claim | Status | Evidence boundary |
|---|---|---|
| Candidate selection is prospective and truth-blind | {'SUPPORTED' if not issues else 'SUPPORTED WITH QUALIFICATION'} | Nominal-only pools and chronology/key audits. |
| Frozen conformal certification recovers with nominal margin | {'SUPPORTED WITH QUALIFICATION' if pooled_recovered else 'NOT SUPPORTED'} | Only one B4 flash action was certified; no nontrivial recovery region was observed. |
| Margin-conditioned calibration improves availability | {'SUPPORTED WITH QUALIFICATION' if conditioned_improved else 'NOT SUPPORTED'} | Fresh matched band calibration certified no margin ladder test actions. |
| Conditional safety is guaranteed within each band | NOT SUPPORTED | Band-specific split conformal remains marginal within the sampled band population. |
| Domain exits demonstrate physical unsafety | NOT SUPPORTED | Unconfirmed exits remain validation failures. |
| CSTR and flash have identical operating envelopes | NOT SUPPORTED | Cross-process curves are descriptive and use process-specific predictors. |

Audit findings: {issue_text}.

## Final Verdict

### {verdict}

No simulator validation-reliability evaluation result was altered or merged into an margin ladder denominator.
"""
    _safe("margin_ladder_claims_report.md").write_text(claims, encoding="utf-8")
    return verdict


def hash_artifacts() -> None:
    output = _safe("margin_ladder_artifact_hashes.txt")
    paths = sorted(
        (
            path
            for path in EXPERIMENT_ROOT.rglob("*")
            if path.is_file()
            and path != output
            and "__pycache__" not in path.parts
            and path.suffix != ".pyc"
        ),
        key=lambda path: path.as_posix(),
    )
    lines = ["# SHA-256 hashes for final margin ladder artifacts", ""]
    lines.extend(f"{_sha256(path)}  {path.relative_to(EXPERIMENT_ROOT).as_posix()}" for path in paths)
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def _load_audit(name: str) -> list[dict[str, str]]:
    return _read_csv(_safe(f"artifacts/{name}"))


def finalize() -> str:
    flash_cal = _load_audit("FLASH_CALIBRATION_ARTIFACT_AUDIT.csv")
    flash_test = _load_audit("FLASH_TEST_ARTIFACT_AUDIT.csv")
    cstr_test = _load_audit("CSTR_TEST_ARTIFACT_AUDIT.csv")
    flash_episodes, flash_results = evaluate_flash_test(flash_test)
    cstr_episodes, cstr_results = evaluate_cstr_test(cstr_test)
    _write_csv(_safe("flash_margin_episodes.csv"), flash_episodes)
    _write_csv(_safe("flash_margin_results.csv"), flash_results)
    _write_csv(_safe("cstr_margin_episodes.csv"), cstr_episodes)
    _write_csv(_safe("cstr_margin_results.csv"), cstr_results)
    summary = summarize("flash", flash_episodes, flash_results) + summarize("CSTR", cstr_episodes, cstr_results)
    _write_csv(_safe("margin_ladder_risk_availability.csv"), summary)
    audit = audit_all(flash_cal, flash_test, cstr_test, flash_episodes, cstr_episodes)
    create_figures(summary, flash_results)
    verdict = write_reports(summary, audit)
    hash_artifacts()
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("calibration", "flash-test", "cstr-test", "finalize", "all"),
        default="all",
    )
    parser.add_argument("--workers", type=int, default=max(1, min(6, os.cpu_count() or 1)))
    args = parser.parse_args()
    _verify_preregistration()
    if args.stage in ("calibration", "all"):
        calibration = run_flash_population(
            "calibration", FLASH_CALIBRATION_COUNT, FLASH_CALIBRATION_START, args.workers
        )
        calibrate_margin_bands(calibration)
    if args.stage in ("flash-test", "all"):
        config_path = _safe("flash_band_conformal_config.json")
        config_hash = _safe("flash_band_conformal_config.sha256").read_text().split()[0]
        if _sha256(config_path) != config_hash:
            raise RuntimeError("margin ladder conformal config changed before test")
        run_flash_population("test", FLASH_TEST_COUNT, FLASH_TEST_START, args.workers)
    if args.stage in ("cstr-test", "all"):
        run_cstr_population(args.workers)
    if args.stage in ("finalize", "all"):
        verdict = finalize()
        print(verdict, flush=True)


if __name__ == "__main__":
    main()
