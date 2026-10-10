"""Prospective untouched ordinary-final evaluation for the flash transfer case."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
from io import BytesIO
import json
from math import sqrt
import os
from pathlib import Path
import time
from typing import Iterable, Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twinguard.uncertainty.flash_bootstrap import (
    PREDICTIVE_SCALE_FLOORS,
    PREDICTIVE_TRAJECTORIES,
    coherent_predictive_summary,
)
from twinguard.conformal.flash_conformal import conformal_envelope, normalized_action_distance, safety_decision
from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.discrepancy.flash_state_space import (
    FEATURE_NAMES,
    FlashStateSpaceParameters,
    parameters_from_payload,
    predict_discrepancy,
)
from twinguard.scenarios.flash_population_design import (
    ACTION_NAMES,
    MEASURED_OUTPUTS,
    PRIMARY_FAULT_FAMILIES,
    PROCESS_BUNDLE_SHA256,
    SafetySpecification,
    faulted_plant_parameters,
    latin_hypercube_actions,
    purpose_rng,
    purpose_seed,
    sample_primary_fault,
    validate_candidate_action,
)
from twinguard.plants.flash_drum import FlashAction, FlashParameters, FlashPlant, PhysicalStateError
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT, REPOSITORY_ROOT, assert_isolated_output_path


POPULATION_DESIGN_SHA256 = "e976e1afc20fbe8f2da120717983dc58ed8a4ce6997d6a33bb15fb07edb2383a"
FINAL_COUNT = 1200
FINAL_SEED_START = 26_000_000
ENSEMBLE_SIZE = 10
PRIMARY_ALPHA = 0.05
RAW_ONE_SIDED_95 = 1.6448536269514722
METHODS = (
    "M0_nominal_DT",
    "M1_discrepancy_mean",
    "M2_raw_bootstrap",
    "M3_pooled_conformal",
    "M4_action_conditioned",
)
PREDICTION_ALLOWED_KEYS = {
    "schema_version",
    "episode_id",
    "candidate_action",
    "observation_time_s",
    "observed_residual",
    "observed_features",
    "prediction_status",
    "prediction_domain_exit",
    "nominal_horizon_outputs",
    "mean_corrected_horizon_outputs",
    "ensemble_center_horizon_outputs",
    "ensemble_scale_horizon_outputs",
    "scale_floors",
    "raw_multiplier",
    "pooled_quantile",
    "ensemble_size",
    "predictive_trajectory_count",
    "selected_mean_model_sha256",
    "bootstrap_models_sha256",
    "conformal_config_sha256",
}
DECISION_ALLOWED_KEYS = {
    "schema_version",
    "episode_id",
    "prediction_sha256",
    "decisions",
    "normalized_margins",
    "raw_multiplier",
    "pooled_quantile",
    "mondrian_status",
    "latency_seconds",
}
FORBIDDEN_PRETRUTH_TOKENS = (
    "fault",
    "severity",
    "changed_parameter",
    "actual",
    "future_truth",
    "physical_truth",
    "safety_label",
)


_MODELS: list[FlashStateSpaceParameters] | None = None
_MEAN_MODEL: FlashStateSpaceParameters | None = None
_POOLED_Q: float | None = None
_MODEL_HASHES: dict[str, str] | None = None
_OUTPUT_ROOT: Path | None = None


def _safe_path(relative: str) -> Path:
    path = assert_isolated_output_path(EXPERIMENT_ROOT / relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _tree_hash(paths: Sequence[tuple[str, Path]]) -> str:
    digest = hashlib.sha256()
    for relative, path in paths:
        digest.update(relative.encode() + b"\0" + _sha256(path).encode() + b"\n")
    return digest.hexdigest()


def verify_frozen_inputs() -> dict[str, object]:
    if _sha256(EXPERIMENT_ROOT / "population_preregistration.md") != POPULATION_DESIGN_SHA256:
        raise RuntimeError("population design preregistration hash mismatch")

    hash_manifest = EXPERIMENT_ROOT / "calibration_artifact_hashes.txt"
    for line in hash_manifest.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        expected, relative = line.split("  ", maxsplit=1)
        if _sha256(EXPERIMENT_ROOT / relative) != expected:
            raise RuntimeError(f"frozen calibration hash mismatch: {relative}")

    config_path = EXPERIMENT_ROOT / "flash_conformal_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    selected_path = EXPERIMENT_ROOT / "selected_flash_model.json"
    bootstrap_path = EXPERIMENT_ROOT / "flash_bootstrap_models.json"
    if config["ensemble"]["selected"] != ENSEMBLE_SIZE:
        raise RuntimeError("frozen ensemble size is not J=10")
    if config["ensemble"]["predictive_trajectories"] != PREDICTIVE_TRAJECTORIES:
        raise RuntimeError("frozen predictive trajectory count mismatch")
    if config["mondrian"]["supported"] is not False:
        raise RuntimeError("unexpected supported Mondrian configuration")
    q = config["pooled_quantiles"]["calibration_ordinary"]["0.05"]["quantile"]
    return {
        "config": config,
        "pooled_q": float(q),
        "selected_mean_model_sha256": _sha256(selected_path),
        "bootstrap_models_sha256": _sha256(bootstrap_path),
        "conformal_config_sha256": _sha256(config_path),
        "calibration_stage_hash_manifest_sha256": _sha256(hash_manifest),
    }


def _exclusive_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def _json_bytes(payload: Mapping[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _npz_bytes(payload: Mapping[str, object]) -> bytes:
    stream = BytesIO()
    np.savez_compressed(stream, **payload)
    return stream.getvalue()


def _validate_pretruth_payload(payload: Mapping[str, object], allowed: set[str]) -> None:
    if set(payload) != allowed:
        raise ValueError(f"pretruth schema mismatch: {set(payload) ^ allowed}")
    lowered = " ".join(payload).lower()
    forbidden = [token for token in FORBIDDEN_PRETRUTH_TOKENS if token in lowered]
    if forbidden:
        raise ValueError(f"pretruth payload contains forbidden keys: {forbidden}")


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in saved.files}


def _feature(action: FlashAction, nominal: FlashAction, output: Mapping[str, float]) -> np.ndarray:
    vector = action.as_array()
    return np.concatenate(
        [vector, vector - nominal.as_array(), [output[name] for name in MEASURED_OUTPUTS]]
    )


def _outputs(model: FlashPlant | FlashDigitalTwin) -> np.ndarray:
    values = model.outputs(model.state)
    return np.asarray([values[name] for name in MEASURED_OUTPUTS], dtype=float)


def _simulate_observation(index: int, seed: int) -> dict[str, object]:
    family = PRIMARY_FAULT_FAMILIES[index % 3]
    hidden = sample_primary_fault(family, seed)
    nominal_parameters = FlashParameters()
    initial_rng = purpose_rng(seed, 101)
    template = FlashDigitalTwin(nominal_parameters)
    initial_state = template.state_from_nxt(
        float(initial_rng.uniform(550.0, 650.0)),
        float(initial_rng.uniform(0.42, 0.48)),
        float(initial_rng.uniform(348.0, 352.0)),
    )
    plant = FlashPlant(faulted_plant_parameters(nominal_parameters, hidden), initial_state)
    twin = FlashDigitalTwin(nominal_parameters, initial_state)
    nominal_action = twin.nominal_action()
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
            physical = _outputs(plant)
            nominal = _outputs(twin)
            observed_outputs[completed] = physical + noise_rng.normal(0.0, sensor_sd)
            nominal_outputs[completed] = nominal
            residual[completed] = observed_outputs[completed] - nominal
            features[completed] = _feature(
                nominal_action, nominal_action, twin.outputs(twin.state)
            )
            completed += 1
    except PhysicalStateError as exc:
        domain_exit = str(exc)
    return {
        "family": hidden.family.value,
        "hidden": hidden,
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


def _nominal_horizon(
    twin: FlashDigitalTwin, candidate: FlashAction, nominal_action: FlashAction
) -> tuple[np.ndarray, np.ndarray, str]:
    model = FlashDigitalTwin(twin.parameters, twin.state)
    outputs: list[np.ndarray] = []
    features: list[np.ndarray] = []
    domain_exit = ""
    try:
        for second in range(1, 1201):
            model.step(candidate, 1.0)
            if second % 10:
                continue
            values = model.outputs(model.state)
            outputs.append(np.asarray([values[name] for name in MEASURED_OUTPUTS]))
            features.append(_feature(candidate, nominal_action, values))
    except PhysicalStateError as exc:
        domain_exit = str(exc)
    return np.asarray(outputs), np.asarray(features), domain_exit


def _envelope_margin(center: np.ndarray, scale: np.ndarray, multiplier: float) -> float:
    if len(center) != 120:
        return float("-inf")
    envelope = conformal_envelope(center, scale, multiplier)
    safety = SafetySpecification()
    violations = np.column_stack(
        [
            (envelope.pressure_upper - safety.pressure_max_bar) / safety.pressure_scale_bar,
            (envelope.level_upper - safety.level_max_fraction) / safety.level_scale_fraction,
            (safety.level_min_fraction - envelope.level_lower) / safety.level_scale_fraction,
            (envelope.temperature_upper - safety.temperature_max_k) / safety.temperature_scale_k,
        ]
    )
    return -float(np.max(violations))


def _point_margin(center: np.ndarray) -> float:
    if len(center) != 120:
        return float("-inf")
    return SafetySpecification().trajectory_margin(
        {name: center[:, index] for index, name in enumerate(MEASURED_OUTPUTS)}
    )


def _build_prediction(
    observation: Mapping[str, object], candidate: FlashAction, episode_id: str, seed: int
) -> tuple[dict[str, object], dict[str, str], dict[str, float]]:
    if _MODELS is None or _MEAN_MODEL is None or _POOLED_Q is None or _MODEL_HASHES is None:
        raise RuntimeError("primary evaluation prediction worker is not initialized")
    started = time.perf_counter()
    nominal = np.empty((0, 3))
    mean_corrected = np.empty((0, 3))
    ensemble_center = np.empty((0, 3))
    ensemble_scale = np.empty((0, 3))
    prediction_status = "OBSERVATION_DOMAIN_EXIT"
    prediction_exit = str(observation["domain_exit"])
    nominal_elapsed = mean_elapsed = bootstrap_elapsed = 0.0
    if int(observation["completed"]) == 50:
        nominal_started = time.perf_counter()
        nominal, future_features, prediction_exit = _nominal_horizon(
            observation["twin"], candidate, observation["nominal_action"]
        )
        nominal_elapsed = time.perf_counter() - nominal_started
        if len(nominal) == 120:
            mean_started = time.perf_counter()
            discrepancy, _ = predict_discrepancy(
                _MEAN_MODEL,
                np.asarray(observation["residual"]),
                np.asarray(observation["features"]),
                future_features,
            )
            mean_corrected = nominal + discrepancy
            mean_elapsed = time.perf_counter() - mean_started
            bootstrap_started = time.perf_counter()
            summary = coherent_predictive_summary(
                _MODELS,
                np.asarray(observation["residual"]),
                np.asarray(observation["features"]),
                future_features,
                nominal,
                ensemble_size=ENSEMBLE_SIZE,
                trajectory_count=PREDICTIVE_TRAJECTORIES,
                seed=purpose_seed(seed, 601),
            )
            ensemble_center = summary.center
            ensemble_scale = summary.scale
            bootstrap_elapsed = time.perf_counter() - bootstrap_started
            prediction_status = "COMPLETE"
        else:
            prediction_status = "NOMINAL_DT_DOMAIN_EXIT"

    payload: dict[str, object] = {
        "schema_version": np.asarray("FLASH_PRIMARY_PREDICTION_1.0"),
        "episode_id": np.asarray(episode_id),
        "candidate_action": candidate.as_array(),
        "observation_time_s": np.arange(1, int(observation["completed"]) + 1) * 10.0,
        "observed_residual": np.asarray(observation["residual"]),
        "observed_features": np.asarray(observation["features"]),
        "prediction_status": np.asarray(prediction_status),
        "prediction_domain_exit": np.asarray(prediction_exit),
        "nominal_horizon_outputs": nominal,
        "mean_corrected_horizon_outputs": mean_corrected,
        "ensemble_center_horizon_outputs": ensemble_center,
        "ensemble_scale_horizon_outputs": ensemble_scale,
        "scale_floors": PREDICTIVE_SCALE_FLOORS,
        "raw_multiplier": np.asarray(RAW_ONE_SIDED_95),
        "pooled_quantile": np.asarray(_POOLED_Q),
        "ensemble_size": np.asarray(ENSEMBLE_SIZE),
        "predictive_trajectory_count": np.asarray(PREDICTIVE_TRAJECTORIES),
        "selected_mean_model_sha256": np.asarray(_MODEL_HASHES["selected"]),
        "bootstrap_models_sha256": np.asarray(_MODEL_HASHES["bootstrap"]),
        "conformal_config_sha256": np.asarray(_MODEL_HASHES["config"]),
    }
    _validate_pretruth_payload(payload, PREDICTION_ALLOWED_KEYS)
    if prediction_status == "COMPLETE":
        decisions = {
            "M0_nominal_DT": "CERTIFIED SAFE" if _point_margin(nominal) >= 0 else "NOT CERTIFIED",
            "M1_discrepancy_mean": "CERTIFIED SAFE" if _point_margin(mean_corrected) >= 0 else "NOT CERTIFIED",
            "M2_raw_bootstrap": safety_decision(
                ensemble_center, ensemble_scale, RAW_ONE_SIDED_95
            ).replace("SAFE", "CERTIFIED SAFE"),
            "M3_pooled_conformal": safety_decision(
                ensemble_center, ensemble_scale, _POOLED_Q
            ).replace("SAFE", "CERTIFIED SAFE"),
            "M4_action_conditioned": "NOT CERTIFIED",
        }
        margins = {
            "M0_nominal_DT": _point_margin(nominal),
            "M1_discrepancy_mean": _point_margin(mean_corrected),
            "M2_raw_bootstrap": _envelope_margin(
                ensemble_center, ensemble_scale, RAW_ONE_SIDED_95
            ),
            "M3_pooled_conformal": _envelope_margin(
                ensemble_center, ensemble_scale, _POOLED_Q
            ),
            "M4_action_conditioned": float("-inf"),
        }
    else:
        decisions = {method: "NOT CERTIFIED" for method in METHODS}
        margins = {method: float("-inf") for method in METHODS}
    latency = {
        "M0_nominal_DT": nominal_elapsed,
        "M1_discrepancy_mean": nominal_elapsed + mean_elapsed,
        "M2_raw_bootstrap": nominal_elapsed + bootstrap_elapsed,
        "M3_pooled_conformal": time.perf_counter() - started,
        "M4_action_conditioned": time.perf_counter() - started,
    }
    return payload, decisions, {**margins, **{f"latency_{key}": value for key, value in latency.items()}}


def _truth_rollout(observation: Mapping[str, object], candidate: FlashAction) -> tuple[np.ndarray, str]:
    if int(observation["completed"]) != 50:
        return np.empty((0, 3)), str(observation["domain_exit"])
    plant: FlashPlant = observation["plant"]
    values: list[np.ndarray] = []
    domain_exit = ""
    try:
        for second in range(1, 1201):
            plant.step(candidate, 1.0)
            if second % 10 == 0:
                values.append(_outputs(plant))
    except PhysicalStateError as exc:
        domain_exit = str(exc)
    return np.asarray(values), domain_exit


def _init_worker(output_root: str, frozen: Mapping[str, object]) -> None:
    global _MODELS, _MEAN_MODEL, _POOLED_Q, _MODEL_HASHES, _OUTPUT_ROOT
    _OUTPUT_ROOT = Path(output_root)
    selected = json.loads((EXPERIMENT_ROOT / "selected_flash_model.json").read_text())
    bootstraps = json.loads((EXPERIMENT_ROOT / "flash_bootstrap_models.json").read_text())
    _MEAN_MODEL = parameters_from_payload(selected["model"])
    _MODELS = [parameters_from_payload(payload) for payload in bootstraps]
    _POOLED_Q = float(frozen["pooled_q"])
    _MODEL_HASHES = {
        "selected": str(frozen["selected_mean_model_sha256"]),
        "bootstrap": str(frozen["bootstrap_models_sha256"]),
        "config": str(frozen["conformal_config_sha256"]),
    }


def _episode_paths(episode_id: str) -> tuple[Path, Path, Path]:
    if _OUTPUT_ROOT is None:
        raise RuntimeError("primary evaluation worker output root missing")
    return (
        _OUTPUT_ROOT / "predictions" / f"{episode_id}.npz",
        _OUTPUT_ROOT / "decisions" / f"{episode_id}.json",
        _OUTPUT_ROOT / "truths" / f"{episode_id}.npz",
    )


def _audit_existing_artifacts(
    episode_id: str, prediction_path: Path, decision_path: Path, truth_path: Path
) -> dict[str, object]:
    prediction_hash = _sha256(prediction_path)
    decision_hash = _sha256(decision_path)
    truth_hash = _sha256(truth_path)
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    truth = _load_npz(truth_path)
    if decision["episode_id"] != episode_id:
        raise RuntimeError("decision episode mismatch")
    if decision["prediction_sha256"] != prediction_hash:
        raise RuntimeError("decision does not bind prediction")
    if str(truth["prediction_sha256"]) != prediction_hash:
        raise RuntimeError("truth does not bind prediction")
    if str(truth["decision_sha256"]) != decision_hash:
        raise RuntimeError("truth does not bind decision")
    prediction_ns = prediction_path.stat().st_mtime_ns
    decision_ns = decision_path.stat().st_mtime_ns
    truth_ns = truth_path.stat().st_mtime_ns
    if not prediction_ns < decision_ns < truth_ns:
        raise RuntimeError("strict prediction-decision-truth chronology failure")
    return {
        "episode_id": episode_id,
        "prediction_path": prediction_path.as_posix(),
        "decision_path": decision_path.as_posix(),
        "truth_path": truth_path.as_posix(),
        "prediction_sha256": prediction_hash,
        "decision_sha256": decision_hash,
        "truth_sha256": truth_hash,
        "prediction_saved_ns": prediction_ns,
        "decision_saved_ns": decision_ns,
        "truth_revealed_ns": truth_ns,
        "chronology_verified": True,
    }


def _episode_task(task: tuple[int, int, np.ndarray]) -> dict[str, object]:
    index, seed, action_vector = task
    episode_id = f"flash_final_ordinary_{index:05d}"
    prediction_path, decision_path, truth_path = _episode_paths(episode_id)
    if prediction_path.exists() and decision_path.exists() and truth_path.exists():
        return _audit_existing_artifacts(
            episode_id, prediction_path, decision_path, truth_path
        )
    observation = _simulate_observation(index, seed)
    candidate = validate_candidate_action(action_vector)

    if not prediction_path.exists():
        prediction, decisions, metrics = _build_prediction(
            observation, candidate, episode_id, seed
        )
        _exclusive_write(prediction_path, _npz_bytes(prediction))
    else:
        prediction = _load_npz(prediction_path)
        if str(prediction["episode_id"]) != episode_id:
            raise RuntimeError("prediction episode mismatch")
        if decision_path.exists():
            existing = json.loads(decision_path.read_text(encoding="utf-8"))
            decisions = existing["decisions"]
            metrics = {
                **existing["normalized_margins"],
                **{
                    f"latency_{method}": existing["latency_seconds"][method]
                    for method in METHODS
                },
            }
        else:
            _, decisions, metrics = _build_prediction(
                observation, candidate, episode_id, seed
            )
    prediction_hash = _sha256(prediction_path)
    prediction_saved_ns = prediction_path.stat().st_mtime_ns

    if not decision_path.exists():
        decision_payload = {
            "schema_version": "FLASH_PRIMARY_DECISION_1.0",
            "episode_id": episode_id,
            "prediction_sha256": prediction_hash,
            "decisions": decisions,
            "normalized_margins": {method: metrics[method] for method in METHODS},
            "raw_multiplier": RAW_ONE_SIDED_95,
            "pooled_quantile": _POOLED_Q,
            "mondrian_status": "UNSUPPORTED_NOT_CERTIFIED:BINS_NOT_PREREGISTERED",
            "latency_seconds": {
                method: metrics[f"latency_{method}"] for method in METHODS
            },
        }
        _validate_pretruth_payload(decision_payload, DECISION_ALLOWED_KEYS)
        while time.time_ns() <= prediction_saved_ns:
            time.sleep(0.001)
        _exclusive_write(decision_path, _json_bytes(decision_payload))
    decision_payload = json.loads(decision_path.read_text(encoding="utf-8"))
    decision_hash = _sha256(decision_path)
    decision_saved_ns = decision_path.stat().st_mtime_ns
    if decision_payload["prediction_sha256"] != prediction_hash:
        raise RuntimeError("decision does not bind the prediction artifact")

    if not truth_path.exists():
        actual, truth_exit = _truth_rollout(observation, candidate)
        hidden = observation["hidden"]
        truth_payload = {
            "schema_version": np.asarray("FLASH_PRIMARY_TRUTH_1.0"),
            "episode_id": np.asarray(episode_id),
            "episode_seed_evaluator_only": np.asarray(seed),
            "fault_family_evaluator_only": np.asarray(hidden.family.value),
            "fault_severity_evaluator_only": np.asarray(hidden.normalized_severity),
            "fault_severity_label_evaluator_only": np.asarray(hidden.severity_label),
            "fault_signed_magnitude_evaluator_only": np.asarray(hidden.signed_magnitude),
            "changed_parameter_evaluator_only": np.asarray(hidden.changed_parameter),
            "nominal_parameter_evaluator_only": np.asarray(hidden.nominal_value),
            "faulty_parameter_evaluator_only": np.asarray(hidden.faulty_value),
            "initial_state_evaluator_only": observation["initial_state"].as_array(),
            "initial_outputs_evaluator_only": np.asarray(
                [observation["initial_outputs"][name] for name in MEASURED_OUTPUTS]
            ),
            "actual_horizon_outputs": actual,
            "completed_horizon_samples": np.asarray(len(actual)),
            "truth_domain_exit_evaluator_only": np.asarray(truth_exit),
            "prediction_sha256": np.asarray(prediction_hash),
            "decision_sha256": np.asarray(decision_hash),
        }
        while time.time_ns() <= decision_saved_ns:
            time.sleep(0.001)
        _exclusive_write(truth_path, _npz_bytes(truth_payload))
    truth = _load_npz(truth_path)
    if str(truth["prediction_sha256"]) != prediction_hash:
        raise RuntimeError("truth does not bind prediction")
    if str(truth["decision_sha256"]) != decision_hash:
        raise RuntimeError("truth does not bind decision")
    truth_revealed_ns = truth_path.stat().st_mtime_ns
    if not prediction_saved_ns < decision_saved_ns < truth_revealed_ns:
        raise RuntimeError("strict prediction-decision-truth chronology failure")
    return _audit_existing_artifacts(
        episode_id, prediction_path, decision_path, truth_path
    )


def _trajectory_mapping(values: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[:, index] for index, name in enumerate(MEASURED_OUTPUTS)}


def _coverage(actual: np.ndarray, center: np.ndarray, scale: np.ndarray, multiplier: float) -> bool:
    if len(actual) != 120 or len(center) != 120:
        return False
    denominator = np.maximum(scale, PREDICTIVE_SCALE_FLOORS)
    upper = center + multiplier * denominator
    lower_level = center[:, 1] - multiplier * denominator[:, 1]
    return bool(
        np.all(actual[:, 0] <= upper[:, 0])
        and np.all(actual[:, 2] <= upper[:, 2])
        and np.all(actual[:, 1] <= upper[:, 1])
        and np.all(actual[:, 1] >= lower_level)
    )


def _wilson(successes: int, count: int) -> tuple[float, float]:
    if count == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    radius = z / denominator * sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count))
    return max(0.0, center - radius), min(1.0, center + radius)


def _active_constraint(margins: np.ndarray) -> str:
    if not len(margins):
        return "simulator_domain_exit"
    names = ("high_pressure", "high_level", "low_level", "high_temperature")
    return names[int(np.unravel_index(np.argmax(margins), margins.shape)[1])]


def _initial_region(initial_state: np.ndarray) -> str:
    liquid_moles = float(initial_state[0])
    if liquid_moles < 550.0 + 100.0 / 3.0:
        return "low_initial_holdup"
    if liquid_moles < 550.0 + 200.0 / 3.0:
        return "mid_initial_holdup"
    return "high_initial_holdup"


def _action_distance_group(distance: float) -> str:
    if distance < 0.40:
        return "d_lt_0.40"
    if distance < 0.60:
        return "d_0.40_to_0.60"
    return "d_ge_0.60"


def evaluate_artifacts(audit_rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    safety = SafetySpecification()
    for audit in sorted(audit_rows, key=lambda item: str(item["episode_id"])):
        prediction = _load_npz(Path(str(audit["prediction_path"])))
        decision = json.loads(Path(str(audit["decision_path"])).read_text())
        truth = _load_npz(Path(str(audit["truth_path"])))
        actual = np.asarray(truth["actual_horizon_outputs"], dtype=float)
        truth_complete = len(actual) == 120
        prediction_complete = str(prediction["prediction_status"]) == "COMPLETE"
        margins = safety.violation_margins(_trajectory_mapping(actual)) if len(actual) else np.empty((0, 4))
        severity_max = float(np.max(np.maximum(margins, 0.0))) if len(margins) else 0.0
        severity_integrated = float(np.sum(np.maximum(margins, 0.0)) * 10.0)
        actual_safe = bool(
            truth_complete and len(margins) and float(np.max(margins)) <= 0.0
        )
        true_margin = -float(np.max(margins)) if truth_complete else float("-inf")
        candidate = np.asarray(prediction["candidate_action"], dtype=float)
        action_distance = normalized_action_distance(candidate)
        nominal = np.asarray(prediction["nominal_horizon_outputs"], dtype=float)
        mean_corrected = np.asarray(prediction["mean_corrected_horizon_outputs"], dtype=float)
        center = np.asarray(prediction["ensemble_center_horizon_outputs"], dtype=float)
        scale = np.asarray(prediction["ensemble_scale_horizon_outputs"], dtype=float)
        centers = {
            "M0_nominal_DT": nominal,
            "M1_discrepancy_mean": mean_corrected,
            "M2_raw_bootstrap": center,
            "M3_pooled_conformal": center,
            "M4_action_conditioned": center,
        }
        multipliers: dict[str, float | None] = {
            "M0_nominal_DT": 0.0,
            "M1_discrepancy_mean": 0.0,
            "M2_raw_bootstrap": RAW_ONE_SIDED_95,
            "M3_pooled_conformal": float(prediction["pooled_quantile"]),
            "M4_action_conditioned": None,
        }
        for method in METHODS:
            predicted = centers[method]
            complete_pair = prediction_complete and truth_complete and len(predicted) == 120
            errors = actual - predicted if complete_pair else np.empty((0, 3))
            multiplier = multipliers[method]
            method_scale = (
                np.zeros_like(predicted)
                if method in ("M0_nominal_DT", "M1_discrepancy_mean")
                else scale
            )
            covered: bool | str = ""
            if complete_pair and multiplier is not None:
                if multiplier == 0.0:
                    covered = bool(
                        np.all(actual[:, 0] <= predicted[:, 0])
                        and np.all(actual[:, 2] <= predicted[:, 2])
                        and np.all(actual[:, 1] <= predicted[:, 1])
                        and np.all(actual[:, 1] >= predicted[:, 1])
                    )
                else:
                    covered = _coverage(actual, predicted, method_scale, multiplier)
            certified = decision["decisions"][method] == "CERTIFIED SAFE"
            false_safe = certified and not actual_safe
            width_multiplier = multiplier if multiplier is not None else float("nan")
            widths = (
                2.0 * width_multiplier * np.maximum(method_scale, PREDICTIVE_SCALE_FLOORS)
                if complete_pair and multiplier is not None and multiplier > 0.0
                else np.zeros((120, 3)) if complete_pair and multiplier == 0.0 else np.empty((0, 3))
            )
            rows.append(
                {
                    "episode_id": str(audit["episode_id"]),
                    "population": "final_ordinary",
                    "method": method,
                    "episode_seed_evaluator_only": int(truth["episode_seed_evaluator_only"]),
                    "fault_family_evaluator_only": str(truth["fault_family_evaluator_only"]),
                    "fault_severity_evaluator_only": float(truth["fault_severity_evaluator_only"]),
                    "fault_severity_label_evaluator_only": str(truth["fault_severity_label_evaluator_only"]),
                    "action_heat_duty_w": candidate[0],
                    "action_vapor_valve_opening": candidate[1],
                    "action_liquid_valve_opening": candidate[2],
                    "action_feed_flow_mol_s": candidate[3],
                    "normalized_action_distance": action_distance,
                    "action_distance_group": _action_distance_group(action_distance),
                    "initial_operating_region": _initial_region(
                        np.asarray(truth["initial_state_evaluator_only"])
                    ),
                    "prediction_complete": prediction_complete,
                    "truth_complete": truth_complete,
                    "truth_domain_exit_evaluator_only": str(truth["truth_domain_exit_evaluator_only"]),
                    "actual_safe": actual_safe,
                    "actual_unsafe": not actual_safe,
                    "certified_safe": certified,
                    "decision": decision["decisions"][method],
                    "false_safe": false_safe,
                    "true_normalized_safety_margin": true_margin,
                    "predicted_normalized_safety_margin": decision["normalized_margins"][method],
                    "active_constraint": _active_constraint(margins),
                    "severity_max": severity_max,
                    "severity_integrated_s": severity_integrated,
                    "severity_is_lower_bound": not truth_complete,
                    "pressure_rmse_bar": float(np.sqrt(np.mean(errors[:, 0] ** 2))) if len(errors) else "",
                    "pressure_bias_bar": float(np.mean(predicted[:, 0] - actual[:, 0])) if len(errors) else "",
                    "level_rmse_fraction": float(np.sqrt(np.mean(errors[:, 1] ** 2))) if len(errors) else "",
                    "level_bias_fraction": float(np.mean(predicted[:, 1] - actual[:, 1])) if len(errors) else "",
                    "temperature_rmse_k": float(np.sqrt(np.mean(errors[:, 2] ** 2))) if len(errors) else "",
                    "temperature_bias_k": float(np.mean(predicted[:, 2] - actual[:, 2])) if len(errors) else "",
                    "whole_trajectory_safety_direction_covered": covered,
                    "mean_pressure_width_bar": float(np.mean(widths[:, 0])) if len(widths) else "",
                    "median_pressure_width_bar": float(np.median(widths[:, 0])) if len(widths) else "",
                    "mean_level_width_fraction": float(np.mean(widths[:, 1])) if len(widths) else "",
                    "median_level_width_fraction": float(np.median(widths[:, 1])) if len(widths) else "",
                    "mean_temperature_width_k": float(np.mean(widths[:, 2])) if len(widths) else "",
                    "median_temperature_width_k": float(np.median(widths[:, 2])) if len(widths) else "",
                    "online_latency_seconds": decision["latency_seconds"][method],
                    "prediction_sha256": audit["prediction_sha256"],
                    "decision_sha256": audit["decision_sha256"],
                    "truth_sha256": audit["truth_sha256"],
                    "prediction_saved_ns": audit["prediction_saved_ns"],
                    "decision_saved_ns": audit["decision_saved_ns"],
                    "truth_revealed_ns": audit["truth_revealed_ns"],
                    "chronology_verified": audit["chronology_verified"],
                }
            )
    return rows


def _numeric(values: Iterable[object]) -> np.ndarray:
    return np.asarray([float(value) for value in values if value != ""], dtype=float)


def aggregate(rows: Sequence[Mapping[str, object]], group_type: str, group_value: str) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        count = len(selected)
        unsafe = sum(bool(row["actual_unsafe"]) for row in selected)
        safe = count - unsafe
        certified = sum(bool(row["certified_safe"]) for row in selected)
        false_safe = sum(bool(row["false_safe"]) for row in selected)
        truly_safe_certified = sum(
            bool(row["certified_safe"]) and bool(row["actual_safe"]) for row in selected
        )
        joint_ci = _wilson(false_safe, count)
        conditional_ci = _wilson(false_safe, certified)
        coverage_values = [
            bool(row["whole_trajectory_safety_direction_covered"])
            for row in selected
            if row["whole_trajectory_safety_direction_covered"] != ""
        ]
        false_severity = [
            float(row["severity_max"]) for row in selected if bool(row["false_safe"])
        ]
        false_integrated = [
            float(row["severity_integrated_s"])
            for row in selected
            if bool(row["false_safe"])
        ]
        result: dict[str, object] = {
            "group_type": group_type,
            "group_value": group_value,
            "method": method,
            "episode_count": count,
            "prediction_complete_count": sum(bool(row["prediction_complete"]) for row in selected),
            "truth_complete_count": sum(bool(row["truth_complete"]) for row in selected),
            "actual_unsafe_count": unsafe,
            "actual_unsafe_rate": unsafe / count,
            "certified_safe_count": certified,
            "certification_rate": certified / count,
            "false_safe_count": false_safe,
            "joint_false_safe_probability": false_safe / count,
            "joint_false_safe_wilson_95_lower": joint_ci[0],
            "joint_false_safe_wilson_95_upper": joint_ci[1],
            "conditional_false_safe_probability": false_safe / certified if certified else 0.0,
            "conditional_false_safe_wilson_95_lower": conditional_ci[0],
            "conditional_false_safe_wilson_95_upper": conditional_ci[1],
            "truly_safe_action_certification_rate": truly_safe_certified / safe if safe else 0.0,
            "not_certified_rate": 1.0 - certified / count,
            "whole_trajectory_safety_direction_coverage": float(np.mean(coverage_values)) if coverage_values else "",
            "total_false_safe_severity": float(np.sum(false_severity)),
            "mean_false_safe_severity": float(np.mean(false_severity)) if false_severity else 0.0,
            "maximum_false_safe_severity": float(np.max(false_severity)) if false_severity else 0.0,
            "total_false_safe_integrated_severity_s": float(np.sum(false_integrated)),
            "mean_false_safe_integrated_severity_s": float(np.mean(false_integrated)) if false_integrated else 0.0,
            "maximum_false_safe_integrated_severity_s": float(np.max(false_integrated)) if false_integrated else 0.0,
        }
        for name in (
            "pressure_rmse_bar",
            "pressure_bias_bar",
            "level_rmse_fraction",
            "level_bias_fraction",
            "temperature_rmse_k",
            "temperature_bias_k",
            "mean_pressure_width_bar",
            "median_pressure_width_bar",
            "mean_level_width_fraction",
            "median_level_width_fraction",
            "mean_temperature_width_k",
            "median_temperature_width_k",
            "online_latency_seconds",
        ):
            values = _numeric(row[name] for row in selected)
            result[name] = float(np.mean(values)) if len(values) and "median_" not in name and name != "online_latency_seconds" else (float(np.median(values)) if len(values) else "")
        output.append(result)
    return output


def subgroup_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    definitions: list[tuple[str, str, list[Mapping[str, object]]]] = []
    for field, group_type in (
        ("fault_family_evaluator_only", "fault_family"),
        ("fault_severity_label_evaluator_only", "fault_severity"),
        ("action_distance_group", "normalized_action_distance"),
        ("initial_operating_region", "initial_operating_region"),
        ("active_constraint", "active_constraint"),
    ):
        for value in sorted({str(row[field]) for row in rows}):
            definitions.append((group_type, value, [row for row in rows if str(row[field]) == value]))
    output: list[dict[str, object]] = []
    for group_type, value, selected in definitions:
        output.extend(aggregate(selected, group_type, value))
    return output


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, name: str) -> None:
    path = _safe_path(f"primary_evaluation_figures/{name}.png")
    fig.tight_layout()
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def create_figures(rows: Sequence[Mapping[str, object]], ablation: Sequence[Mapping[str, object]]) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    m3_rows = [row for row in rows if row["method"] == "M3_pooled_conformal"]
    complete = [row for row in m3_rows if bool(row["truth_complete"]) and bool(row["prediction_complete"])]
    representative = min(complete, key=lambda row: abs(float(row["true_normalized_safety_margin"])))
    prediction = _load_npz(Path(str(REPOSITORY_ROOT / Path(str(representative["prediction_path"])))) if "prediction_path" in representative else EXPERIMENT_ROOT / "primary_evaluation_episodes" / "predictions" / f"{representative['episode_id']}.npz")
    truth = _load_npz(EXPERIMENT_ROOT / "primary_evaluation_episodes" / "truths" / f"{representative['episode_id']}.npz")
    t = np.arange(1, 121) * 10.0 / 60.0
    actual = np.asarray(truth["actual_horizon_outputs"])
    nominal = np.asarray(prediction["nominal_horizon_outputs"])
    center = np.asarray(prediction["ensemble_center_horizon_outputs"])
    scale = np.asarray(prediction["ensemble_scale_horizon_outputs"])
    envelope = conformal_envelope(center, scale, float(prediction["pooled_quantile"]))
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.0), sharex=True)
    labels = ("Pressure [bar]", "Level fraction", "Temperature [K]")
    uppers = (envelope.pressure_upper, envelope.level_upper, envelope.temperature_upper)
    lowers = (center[:, 0] - float(prediction["pooled_quantile"]) * np.maximum(scale[:, 0], PREDICTIVE_SCALE_FLOORS[0]), envelope.level_lower, center[:, 2] - float(prediction["pooled_quantile"]) * np.maximum(scale[:, 2], PREDICTIVE_SCALE_FLOORS[2]))
    for index, ax in enumerate(axes):
        ax.fill_between(t, lowers[index], uppers[index], color="#8ecae6", alpha=0.35, label="95% pooled envelope")
        ax.plot(t, actual[:, index], color="#111111", lw=1.5, label="Plant truth")
        ax.plot(t, nominal[:, index], color="#e76f51", lw=1.2, ls="--", label="Nominal DT")
        ax.plot(t, center[:, index], color="#1d6996", lw=1.2, label="Corrected center")
        ax.set_ylabel(labels[index])
    axes[-1].set_xlabel("Future time [min]")
    axes[0].legend(ncol=2, frameon=False)
    _save_figure(fig, "01_representative_corrected_rollout")

    true_margin = np.asarray([float(row["true_normalized_safety_margin"]) for row in complete])
    nominal_margin = np.asarray([float(next(item for item in rows if item["episode_id"] == row["episode_id"] and item["method"] == "M0_nominal_DT")["predicted_normalized_safety_margin"]) for row in complete])
    conformal_margin = np.asarray([float(row["predicted_normalized_safety_margin"]) for row in complete])
    for number, predicted_margin, label, name in (
        (2, nominal_margin, "Nominal-DT margin", "02_true_vs_nominal_margin"),
        (3, conformal_margin, "Conformal margin", "03_true_vs_conformal_margin"),
    ):
        fig, ax = plt.subplots(figsize=(5.4, 4.6))
        ax.scatter(true_margin, predicted_margin, s=12, alpha=0.45, color="#1d6996")
        bounds = [min(true_margin.min(), predicted_margin.min()), max(true_margin.max(), predicted_margin.max())]
        ax.plot(bounds, bounds, color="#555555", ls="--", lw=1)
        ax.axhline(0, color="#ca3c25", lw=0.8)
        ax.axvline(0, color="#ca3c25", lw=0.8)
        ax.set_xlabel("True normalized safety margin")
        ax.set_ylabel(label)
        _save_figure(fig, name)

    m3 = next(row for row in ablation if row["group_type"] == "pooled" and row["method"] == "M3_pooled_conformal")
    certified_safe = sum(bool(row["certified_safe"]) and bool(row["actual_safe"]) for row in m3_rows)
    false_safe = int(m3["false_safe_count"])
    abstain_safe = sum(not bool(row["certified_safe"]) and bool(row["actual_safe"]) for row in m3_rows)
    abstain_unsafe = sum(not bool(row["certified_safe"]) and bool(row["actual_unsafe"]) for row in m3_rows)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    matrix = np.asarray([[certified_safe, false_safe], [abstain_safe, abstain_unsafe]])
    image = ax.imshow(matrix, cmap="Blues")
    for (i, j), value in np.ndenumerate(matrix):
        ax.text(j, i, str(value), ha="center", va="center")
    ax.set_xticks([0, 1], ["Actually safe", "Actually unsafe"])
    ax.set_yticks([0, 1], ["Certified SAFE", "NOT CERTIFIED"])
    fig.colorbar(image, ax=ax, label="Episodes")
    _save_figure(fig, "04_certification_confusion")

    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    pooled_coverage = float(m3["whole_trajectory_safety_direction_coverage"])
    ax.bar(["Pooled M3", "Action-conditioned M4"], [pooled_coverage, 0.0], color=["#1d6996", "#999999"])
    ax.text(1, 0.03, "Unsupported\nNOT CERTIFIED", ha="center", va="bottom")
    ax.axhline(0.95, color="#ca3c25", ls="--", lw=1, label="95% target")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Whole-trajectory coverage")
    ax.legend(frameon=False)
    _save_figure(fig, "05_pooled_vs_action_conditioned_coverage")

    pooled = [row for row in ablation if row["group_type"] == "pooled"]
    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    x = np.arange(len(pooled))
    ax.bar(x - 0.18, [float(row["joint_false_safe_probability"]) for row in pooled], 0.36, label="Joint false-safe", color="#ca3c25")
    ax.bar(x + 0.18, [float(row["certification_rate"]) for row in pooled], 0.36, label="Certification", color="#1d6996")
    ax.set_xticks(x, [str(row["method"]).split("_")[0] for row in pooled])
    ax.set_ylabel("Probability")
    ax.legend(frameon=False)
    _save_figure(fig, "06_false_safe_risk_and_certification")

    unsafe = [row for row in m3_rows if bool(row["actual_unsafe"])]
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    groups = sorted({str(row["active_constraint"]) for row in unsafe})
    data = [[float(row["severity_max"]) for row in unsafe if row["active_constraint"] == group] for group in groups]
    ax.boxplot(data, tick_labels=[group.replace("_", "\n") for group in groups], showfliers=False)
    ax.set_ylabel("Maximum normalized excursion")
    _save_figure(fig, "07_safety_violation_severity")

    with (REPOSITORY_ROOT / "artifacts/cstr/primary_evaluation/safety_summary.csv").open(newline="", encoding="utf-8") as handle:
        cstr_row = next(row for row in csv.DictReader(handle)
                        if row["method"] == "S3_one_sided" and row["action_category"] == "pooled" and row["nominal_level"] == "0.95")
    cstr_false = float(cstr_row["joint_false_safe_probability"])
    cstr_cert = float(cstr_row["SAFE_certification_rate"])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    x = np.arange(2)
    ax.bar(x - 0.18, [cstr_false, float(m3["joint_false_safe_probability"])], 0.36, label="Joint false-safe", color="#ca3c25")
    ax.bar(x + 0.18, [cstr_cert, float(m3["certification_rate"])], 0.36, label="Certification", color="#1d6996")
    ax.set_xticks(x, ["CSTR", "Flash"])
    ax.set_ylabel("Probability")
    ax.legend(frameon=False)
    _save_figure(fig, "08_cstr_vs_flash_comparison")


def verdict(pooled_m3: Mapping[str, object], config: Mapping[str, object]) -> str:
    useful = (
        float(pooled_m3["joint_false_safe_wilson_95_upper"]) <= 0.01
        and float(pooled_m3["certification_rate"]) >= 0.10
    )
    if not useful:
        return "FLASH TRANSFER NOT VALIDATED"
    if config["selected_mean_model"]["no_model_passed"]:
        return "FLASH TRANSFER VALIDATED WITH LIMITATIONS"
    return "FLASH TRANSFER VALIDATED"


def build_report(
    frozen: Mapping[str, object], ablation: Sequence[Mapping[str, object]], result: str
) -> str:
    pooled = {row["method"]: row for row in ablation if row["group_type"] == "pooled"}
    lines = []
    for method in METHODS:
        row = pooled[method]
        coverage = row["whole_trajectory_safety_direction_coverage"]
        lines.append(
            f"| {method} | {row['certified_safe_count']} ({row['certification_rate']:.3f}) | "
            f"{row['false_safe_count']} ({row['joint_false_safe_probability']:.3f}) | "
            f"[{row['joint_false_safe_wilson_95_lower']:.3f}, {row['joint_false_safe_wilson_95_upper']:.3f}] | "
            f"{coverage if coverage == '' else f'{coverage:.3f}'} |"
        )
    m3 = pooled["M3_pooled_conformal"]
    domain_count = int(m3["episode_count"]) - int(m3["truth_complete_count"])
    pass_false = float(m3["joint_false_safe_wilson_95_upper"]) <= 0.01
    pass_cert = float(m3["certification_rate"]) >= 0.10
    return f"""# Flash primary evaluation Untouched Final Results

## Frozen Prospective Design

All 1,200 reserved ordinary-final seeds were evaluated with the frozen model selection/calibration
artifacts. Prediction NPZ and decision JSON artifacts were exclusively written
and hashed before the corresponding plant future was simulated. Artifact keys
exclude fault metadata, future plant truth, and safety labels. Chronology passed
for every episode. No fit, bootstrap model, calibration score, quantile, safety
limit, or decision threshold changed in primary evaluation.

M2 is the raw one-sided 95% Gaussian bootstrap envelope (`z={RAW_ONE_SIDED_95}`)
without conformal inflation. M3 uses the ordinary pooled `q={frozen['pooled_q']}`.
M4 is unsupported and always NOT CERTIFIED because population design froze no Mondrian bins.

## Reliability

Actual unsafe episodes under the complete-horizon definition: {m3['actual_unsafe_count']}/{m3['episode_count']}.
Simulator-domain exits count as not physically safe because a complete in-domain
horizon is required. Their normalized severity uses the available prefix and is
therefore a lower bound. There were {domain_count} incomplete truth horizons.

| Method | Certified SAFE | False-safe | Joint false-safe | Wilson 95% CI | Directional coverage |
|---|---:|---:|---:|---:|---:|
{chr(10).join(lines)}

For M3, conditional false-safe is {m3['conditional_false_safe_probability']:.4f},
truly-safe-action certification is {m3['truly_safe_action_certification_rate']:.4f},
and NOT CERTIFIED is {m3['not_certified_rate']:.4f}. False-safe maximum normalized
severity is {m3['maximum_false_safe_severity']:.4f}; total severity is
{m3['total_false_safe_severity']:.4f}. Total integrated false-safe excursion is
{m3['total_false_safe_integrated_severity_s']:.4f} normalized-seconds, with mean
{m3['mean_false_safe_integrated_severity_s']:.4f} and maximum
{m3['maximum_false_safe_integrated_severity_s']:.4f}.

## Prediction And Runtime

M3 complete-pair RMSE is {m3['pressure_rmse_bar']:.5f} bar pressure,
{m3['level_rmse_fraction']:.5f} level fraction, and
{m3['temperature_rmse_k']:.4f} K temperature. Median online latency is
{m3['online_latency_seconds']:.4f} s. The subgroup rows in
`flash_primary_ablation.csv` are descriptive only; they do not establish
conditional conformal guarantees.

## Frozen Gates

- Joint false-safe Wilson upper <=0.01: **{'PASS' if pass_false else 'FAIL'}**.
- Certification rate >=0.10: **{'PASS' if pass_cert else 'FAIL'}**.
- model selection mean-model selection gate: **FAIL; preregistered k=3 fallback retained**.
- Mondrian extension: **UNSUPPORTED / NOT CERTIFIED**.

## Final Verdict

### {result}
"""


def _hash_outputs() -> None:
    paths = [
        EXPERIMENT_ROOT / "flash_primary_episode_results.csv",
        EXPERIMENT_ROOT / "flash_primary_ablation.csv",
        EXPERIMENT_ROOT / "primary_evaluation_report.md",
        EXPERIMENT_ROOT / "primary_evaluation_ledger.csv",
    ]
    paths.extend(sorted((EXPERIMENT_ROOT / "primary_evaluation_figures").glob("*.png")))
    paths.extend(sorted((EXPERIMENT_ROOT / "primary_evaluation_episodes").rglob("*.*")))
    lines = ["# Frozen primary evaluation artifact SHA-256 manifest", ""]
    for path in paths:
        lines.append(f"{_sha256(path)}  {path.relative_to(EXPERIMENT_ROOT).as_posix()}")
    _safe_path("primary_evaluation_artifact_hashes.txt").write_text("\n".join(lines) + "\n", encoding="ascii")


def main(workers: int) -> str:
    frozen = verify_frozen_inputs()
    output = _safe_path("primary_evaluation_episodes")
    output.mkdir(parents=True, exist_ok=True)
    actions = latin_hypercube_actions(FINAL_COUNT, FINAL_SEED_START + 401)
    tasks = [(index, FINAL_SEED_START + index, actions[index]) for index in range(FINAL_COUNT)]
    audit: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(str(output), frozen),
    ) as pool:
        for completed, row in enumerate(pool.map(_episode_task, tasks, chunksize=1), 1):
            audit.append(row)
            if completed % 100 == 0 or completed == FINAL_COUNT:
                print(f"primary evaluation prospective episodes complete={completed}/{FINAL_COUNT}", flush=True)
    audit.sort(key=lambda row: str(row["episode_id"]))
    _write_csv(_safe_path("primary_evaluation_ledger.csv"), audit)
    rows = evaluate_artifacts(audit)
    _write_csv(_safe_path("flash_primary_episode_results.csv"), rows)
    ablation = aggregate(rows, "pooled", "all_final_ordinary") + subgroup_rows(rows)
    result = verdict(
        next(row for row in ablation if row["group_type"] == "pooled" and row["method"] == "M3_pooled_conformal"),
        frozen["config"],
    )
    _write_csv(_safe_path("flash_primary_ablation.csv"), ablation)
    _safe_path("primary_evaluation_report.md").write_text(build_report(frozen, ablation, result), encoding="utf-8")
    create_figures(rows, ablation)
    _hash_outputs()
    print(result, flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) - 1)))
    arguments = parser.parse_args()
    main(arguments.workers)
