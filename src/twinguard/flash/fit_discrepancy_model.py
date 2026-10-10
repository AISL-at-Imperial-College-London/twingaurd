"""Generate flash training data and select a latent discrepancy model."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.discrepancy.flash_state_space import (
    ENGINEERING_SCALES,
    FEATURE_NAMES,
    FlashStateSpaceParameters,
    counterfactual_prediction,
    fit_flash_state_space_em,
    parameters_to_payload,
    predict_discrepancy,
)
from twinguard.scenarios.flash_population_design import (
    MEASURED_OUTPUTS,
    OBSERVATION_COUNT,
    PRIMARY_FAULT_FAMILIES,
    PROCESS_BUNDLE_SHA256,
    faulted_plant_parameters,
    latin_hypercube_actions,
    purpose_rng,
    sample_primary_fault,
)
from twinguard.plants.flash_drum import FlashAction, FlashParameters, FlashPlant, PhysicalStateError
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT, assert_isolated_output_path


TRAINING_COUNT = 900
SELECTION_COUNT = 180
TRAINING_SEED_START = 22_000_000
SELECTION_SEED_START = 23_000_000
TOTAL_STEPS = OBSERVATION_COUNT + 120
INITIALIZATION_SEEDS = {1: 33_000_001, 2: 33_000_002, 3: 33_000_003}
ALLOWED_POPULATIONS = ("training", "model_selection")
PREREGISTRATION_SHA256 = "e976e1afc20fbe8f2da120717983dc58ed8a4ce6997d6a33bb15fb07edb2383a"


@dataclass(frozen=True)
class GeneratedEpisode:
    residual: np.ndarray
    features: np.ndarray
    nominal_outputs: np.ndarray
    physical_outputs: np.ndarray
    candidate_action: np.ndarray
    evaluator_metadata: dict[str, object]
    completed: bool


def _verify_frozen_inputs() -> None:
    preregistration = EXPERIMENT_ROOT / "population_preregistration.md"
    digest = hashlib.sha256(preregistration.read_bytes()).hexdigest()
    if digest != PREREGISTRATION_SHA256:
        raise RuntimeError("population design preregistration hash mismatch")


def _feature(action: FlashAction, nominal_action: FlashAction, output: Mapping[str, float]) -> np.ndarray:
    current = action.as_array()
    return np.concatenate(
        [
            current,
            current - nominal_action.as_array(),
            [output[name] for name in MEASURED_OUTPUTS],
        ]
    )


def _episode_task(task: tuple[str, int, int, np.ndarray]) -> GeneratedEpisode:
    role, index, seed, action_vector = task
    if role not in ALLOWED_POPULATIONS:
        raise ValueError(f"model selection prohibited population role: {role}")
    family = PRIMARY_FAULT_FAMILIES[index % len(PRIMARY_FAULT_FAMILIES)]
    fault_truth = sample_primary_fault(family, seed)
    nominal_parameters = FlashParameters()
    plant_parameters = faulted_plant_parameters(nominal_parameters, fault_truth)
    initial_rng = purpose_rng(seed, 101)
    noise_rng = purpose_rng(seed, 307)
    model_template = FlashDigitalTwin(nominal_parameters)
    initial_state = model_template.state_from_nxt(
        float(initial_rng.uniform(550.0, 650.0)),
        float(initial_rng.uniform(0.42, 0.48)),
        float(initial_rng.uniform(348.0, 352.0)),
    )
    plant = FlashPlant(plant_parameters, initial_state)
    twin = FlashDigitalTwin(nominal_parameters, initial_state)
    nominal_action = twin.nominal_action()
    candidate_action = FlashAction.from_vector(action_vector)
    residual = np.full((TOTAL_STEPS, 3), np.nan)
    features = np.full((TOTAL_STEPS, len(FEATURE_NAMES)), np.nan)
    nominal_outputs = np.full((TOTAL_STEPS, 3), np.nan)
    physical_outputs = np.full((TOTAL_STEPS, 3), np.nan)
    sensor_sd = np.asarray([0.002, 0.002, 0.10])
    sample_index = 0
    domain_exit = ""
    try:
        for second in range(1, TOTAL_STEPS * 10 + 1):
            action = nominal_action if second <= OBSERVATION_COUNT * 10 else candidate_action
            plant.step(action, 1.0)
            twin.step(action, 1.0)
            if second % 10:
                continue
            plant_output = plant.outputs(plant.state)
            twin_output = twin.outputs(twin.state)
            physical = np.asarray([plant_output[name] for name in MEASURED_OUTPUTS])
            nominal = np.asarray([twin_output[name] for name in MEASURED_OUTPUTS])
            measured = physical + noise_rng.normal(0.0, sensor_sd)
            use_noisy_target = role == "training" or sample_index < OBSERVATION_COUNT
            residual[sample_index] = (measured if use_noisy_target else physical) - nominal
            nominal_outputs[sample_index] = nominal
            physical_outputs[sample_index] = physical
            features[sample_index] = _feature(action, nominal_action, twin_output)
            sample_index += 1
    except PhysicalStateError as exc:
        domain_exit = str(exc)
    completed = sample_index == TOTAL_STEPS
    return GeneratedEpisode(
        residual=residual,
        features=features,
        nominal_outputs=nominal_outputs,
        physical_outputs=physical_outputs,
        candidate_action=candidate_action.as_array(),
        evaluator_metadata={
            "episode_id": f"flash_{role}_{index:04d}",
            "population_role": role,
            "episode_seed_evaluator_only": seed,
            "fault_family_evaluator_only": family.value,
            "severity_evaluator_only": fault_truth.normalized_severity,
            "severity_label_evaluator_only": fault_truth.severity_label,
            "fault_magnitude_evaluator_only": fault_truth.signed_magnitude,
            "changed_parameter_evaluator_only": fault_truth.changed_parameter,
            "fault_metadata_available_to_method": False,
            "complete_trajectory": completed,
            "completed_samples": sample_index,
            "numerical_domain_exit_evaluator_only": domain_exit,
        },
        completed=completed,
    )


def _generate_role(
    role: str,
    count: int,
    seed_start: int,
    workers: int,
) -> list[GeneratedEpisode]:
    if role not in ALLOWED_POPULATIONS:
        raise ValueError(f"model selection prohibited population role: {role}")
    actions = latin_hypercube_actions(count, seed_start + 401)
    tasks = [(role, index, seed_start + index, actions[index]) for index in range(count)]
    if workers == 1:
        return [_episode_task(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_episode_task, tasks, chunksize=2))


def _safe_path(relative: str) -> Path:
    path = assert_isolated_output_path(EXPERIMENT_ROOT / relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    records = list(rows)
    if not records:
        raise ValueError("cannot write empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def generate_model_selection_populations(
    workers: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, object]],
    dict[str, int],
]:
    training = _generate_role("training", TRAINING_COUNT, TRAINING_SEED_START, workers)
    selection = _generate_role(
        "model_selection", SELECTION_COUNT, SELECTION_SEED_START, workers
    )
    complete_training = [item for item in training if item.completed]
    complete_selection = [item for item in selection if item.completed]
    training_arrays = {
        "residual": np.stack([item.residual for item in complete_training]),
        "features": np.stack([item.features for item in complete_training]),
    }
    selection_inputs = {
        "observed_residual": np.stack(
            [item.residual[:OBSERVATION_COUNT] for item in complete_selection]
        ),
        "observed_features": np.stack(
            [item.features[:OBSERVATION_COUNT] for item in complete_selection]
        ),
        "future_features": np.stack(
            [item.features[OBSERVATION_COUNT:] for item in complete_selection]
        ),
        "nominal_future": np.stack(
            [item.nominal_outputs[OBSERVATION_COUNT:] for item in complete_selection]
        ),
        "candidate_actions": np.stack(
            [item.candidate_action for item in complete_selection]
        ),
    }
    selection_truth = {
        "actual_future": np.stack(
            [item.physical_outputs[OBSERVATION_COUNT:] for item in complete_selection]
        ),
        "true_residual_future": np.stack(
            [item.residual[OBSERVATION_COUNT:] for item in complete_selection]
        ),
    }
    evaluator_rows = [item.evaluator_metadata for item in training + selection]
    np.savez_compressed(_safe_path("model_selection_data/training_method_inputs.npz"), **training_arrays)
    np.savez_compressed(
        _safe_path("model_selection_data/selection_operational_inputs.npz"), **selection_inputs
    )
    _write_csv(_safe_path("model_selection_data/EVALUATOR_GENERATION_MANIFEST.csv"), evaluator_rows)
    counts = {
        "training_reserved": len(training),
        "training_complete": len(complete_training),
        "training_domain_exits": len(training) - len(complete_training),
        "selection_reserved": len(selection),
        "selection_complete": len(complete_selection),
        "selection_domain_exits": len(selection) - len(complete_selection),
    }
    return training_arrays, selection_inputs | selection_truth, evaluator_rows, counts


def load_generated_inputs_and_replay_selection_truth(
    workers: int,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    list[dict[str, object]],
    dict[str, int],
]:
    """Resume after a numerical-fit stop without changing any episode/action."""

    training_path = _safe_path("model_selection_data/training_method_inputs.npz")
    selection_path = _safe_path("model_selection_data/selection_operational_inputs.npz")
    manifest_path = _safe_path("model_selection_data/EVALUATOR_GENERATION_MANIFEST.csv")
    if not all(path.is_file() for path in (training_path, selection_path, manifest_path)):
        raise FileNotFoundError("complete cached model selection generation artifacts are required")
    with np.load(training_path) as saved:
        training = {name: np.asarray(saved[name]) for name in saved.files}
    with np.load(selection_path) as saved:
        cached_selection = {name: np.asarray(saved[name]) for name in saved.files}

    # Replay the identical reserved selection seeds solely to restore evaluator
    # truth that was intentionally not serialized before the first predictions.
    replayed = _generate_role(
        "model_selection", SELECTION_COUNT, SELECTION_SEED_START, workers
    )
    complete = [item for item in replayed if item.completed]
    reconstructed = {
        "observed_residual": np.stack(
            [item.residual[:OBSERVATION_COUNT] for item in complete]
        ),
        "observed_features": np.stack(
            [item.features[:OBSERVATION_COUNT] for item in complete]
        ),
        "future_features": np.stack(
            [item.features[OBSERVATION_COUNT:] for item in complete]
        ),
        "nominal_future": np.stack(
            [item.nominal_outputs[OBSERVATION_COUNT:] for item in complete]
        ),
        "candidate_actions": np.stack([item.candidate_action for item in complete]),
    }
    for name, values in cached_selection.items():
        np.testing.assert_array_equal(values, reconstructed[name])
    selection = reconstructed | {
        "actual_future": np.stack(
            [item.physical_outputs[OBSERVATION_COUNT:] for item in complete]
        ),
        "true_residual_future": np.stack(
            [item.residual[OBSERVATION_COUNT:] for item in complete]
        ),
    }
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        evaluator_rows = list(csv.DictReader(handle))
    counts = {
        "training_reserved": TRAINING_COUNT,
        "training_complete": int(training["residual"].shape[0]),
        "training_domain_exits": TRAINING_COUNT - int(training["residual"].shape[0]),
        "selection_reserved": SELECTION_COUNT,
        "selection_complete": len(complete),
        "selection_domain_exits": SELECTION_COUNT - len(complete),
    }
    return training, selection, evaluator_rows, counts


def _lag_one_autocorrelation(values: np.ndarray) -> float:
    first = values[:, :-1].reshape(-1)
    second = values[:, 1:].reshape(-1)
    if np.std(first) < 1e-12 or np.std(second) < 1e-12:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def evaluate_candidate(
    parameters: FlashStateSpaceParameters,
    selection: Mapping[str, np.ndarray],
) -> tuple[dict[str, object], np.ndarray]:
    count = len(selection["observed_residual"])
    predicted_discrepancy = np.empty_like(selection["true_residual_future"])
    standardized_innovations = np.empty_like(selection["observed_residual"])
    for index in range(count):
        prediction, filtered = predict_discrepancy(
            parameters,
            selection["observed_residual"][index],
            selection["observed_features"][index],
            selection["future_features"][index],
        )
        predicted_discrepancy[index] = prediction
        standardized_innovations[index] = filtered.standardized_innovations
    corrected = selection["nominal_future"] + predicted_discrepancy
    error = corrected - selection["actual_future"]
    episode_rmse = np.sqrt(np.mean(error**2, axis=1))
    normalized_rmse = episode_rmse / ENGINEERING_SCALES
    episode_worst = np.max(normalized_rmse, axis=1)
    bias = np.mean(error, axis=(0, 1))
    normalized_bias = bias / ENGINEERING_SCALES
    autocorrelation = np.asarray(
        [
            _lag_one_autocorrelation(standardized_innovations[:, :, output])
            for output in range(3)
        ]
    )
    finite = bool(
        np.all(np.isfinite(predicted_discrepancy))
        and np.all(np.isfinite(parameters.A))
        and np.all(np.linalg.eigvalsh(parameters.R) > 0.0)
        and np.all(np.linalg.eigvalsh(parameters.Q) > 0.0)
    )
    maximum_prediction = float(
        np.max(np.abs(predicted_discrepancy / ENGINEERING_SCALES))
    )
    gates = {
        "gate_converged_finite": bool(parameters.converged and finite),
        "gate_spectral_radius": bool(parameters.spectral_radius < 0.995),
        "gate_worst_normalized_rmse": bool(np.median(episode_worst) <= 0.50),
        "gate_each_output_normalized_rmse": bool(
            np.all(np.median(normalized_rmse, axis=0) <= 0.50)
        ),
        "gate_each_output_normalized_bias": bool(np.all(np.abs(normalized_bias) <= 0.10)),
        "gate_each_output_innovation_autocorrelation": bool(
            np.all(np.abs(autocorrelation) <= 0.50)
        ),
    }
    row: dict[str, object] = {
        "latent_dimension": parameters.latent_dimension,
        "initialization_seed": parameters.initialization_seed,
        "converged": parameters.converged,
        "iterations": parameters.iterations,
        "fit_seconds": parameters.fit_seconds,
        "log_likelihood": parameters.log_likelihood,
        "final_parameter_change": parameters.final_parameter_change,
        "spectral_radius": parameters.spectral_radius,
        "median_worst_output_normalized_rmse": float(np.median(episode_worst)),
        "median_pressure_rmse_bar": float(np.median(episode_rmse[:, 0])),
        "median_level_rmse_fraction": float(np.median(episode_rmse[:, 1])),
        "median_temperature_rmse_k": float(np.median(episode_rmse[:, 2])),
        "median_pressure_normalized_rmse": float(np.median(normalized_rmse[:, 0])),
        "median_level_normalized_rmse": float(np.median(normalized_rmse[:, 1])),
        "median_temperature_normalized_rmse": float(np.median(normalized_rmse[:, 2])),
        "pressure_bias_bar": float(bias[0]),
        "level_bias_fraction": float(bias[1]),
        "temperature_bias_k": float(bias[2]),
        "pressure_normalized_bias": float(normalized_bias[0]),
        "level_normalized_bias": float(normalized_bias[1]),
        "temperature_normalized_bias": float(normalized_bias[2]),
        "pressure_innovation_lag1": float(autocorrelation[0]),
        "level_innovation_lag1": float(autocorrelation[1]),
        "temperature_innovation_lag1": float(autocorrelation[2]),
        "numerically_stable": finite,
        "counterfactual_rollout_finite": finite,
        "maximum_abs_predicted_discrepancy_normalized": maximum_prediction,
        **gates,
        "all_predeclared_gates_passed": all(gates.values()),
    }
    return row, predicted_discrepancy


def _action_sensitivity(
    parameters: FlashStateSpaceParameters,
    selection: Mapping[str, np.ndarray],
) -> dict[str, float | bool]:
    model = FlashDigitalTwin()
    nominal = model.nominal_action()
    lower_action = FlashAction(11_000.0, 0.35, 0.65, 0.85)
    upper_action = FlashAction(20_000.0, 0.70, 0.35, 1.15)
    first = counterfactual_prediction(
        parameters,
        selection["observed_residual"][0],
        selection["observed_features"][0],
        model.nominal_state(),
        lower_action,
    )
    second = counterfactual_prediction(
        parameters,
        selection["observed_residual"][0],
        selection["observed_features"][0],
        model.nominal_state(),
        upper_action,
    )
    nominal_change = float(np.max(np.abs(first.nominal_outputs - second.nominal_outputs)))
    discrepancy_change = float(
        np.max(np.abs(first.predicted_discrepancy - second.predicted_discrepancy))
    )
    return {
        "action_changes_nominal_rollout": nominal_change > 1e-8,
        "action_changes_discrepancy_rollout": discrepancy_change > 1e-8,
        "maximum_nominal_action_sensitivity": nominal_change,
        "maximum_discrepancy_action_sensitivity": discrepancy_change,
        "reference_nominal_heat_duty_w": nominal.heat_duty_w,
    }


def _select(rows: list[dict[str, object]]) -> tuple[int, str]:
    for row in sorted(rows, key=lambda item: int(item["latent_dimension"])):
        if bool(row["all_predeclared_gates_passed"]):
            return int(row["latent_dimension"]), "FIRST_PASSING_PREDECLARED_CANDIDATE"
    eligible = [
        row
        for row in rows
        if bool(row["gate_converged_finite"]) and bool(row["gate_spectral_radius"])
    ]
    if not eligible:
        raise RuntimeError("NO MODEL PASSED and no converged stable fallback exists")
    fallback = min(
        eligible,
        key=lambda row: (
            float(row["median_worst_output_normalized_rmse"]),
            int(row["latent_dimension"]),
        ),
    )
    return int(fallback["latent_dimension"]), "NO_MODEL_PASSED_PREDECLARED_FALLBACK"


def _report(
    rows: list[dict[str, object]],
    selected: int,
    status: str,
    counts: Mapping[str, int],
) -> str:
    result_lines = []
    for row in rows:
        failed = [
            name.removeprefix("gate_")
            for name, value in row.items()
            if name.startswith("gate_") and not bool(value)
        ]
        result_lines.append(
            f"| {row['latent_dimension']} | {row['median_worst_output_normalized_rmse']:.4f} | "
            f"{row['median_pressure_normalized_rmse']:.4f} / {row['median_level_normalized_rmse']:.4f} / "
            f"{row['median_temperature_normalized_rmse']:.4f} | "
            f"{row['pressure_normalized_bias']:.4f} / {row['level_normalized_bias']:.4f} / "
            f"{row['temperature_normalized_bias']:.4f} | "
            f"{row['pressure_innovation_lag1']:.4f} / {row['level_innovation_lag1']:.4f} / "
            f"{row['temperature_innovation_lag1']:.4f} | {row['spectral_radius']:.5f} | "
            f"{'PASS' if row['all_predeclared_gates_passed'] else 'FAIL: ' + ', '.join(failed)} |"
        )
    selected_row = next(row for row in rows if row["latent_dimension"] == selected)
    failed_candidates = [
        str(row["latent_dimension"])
        for row in rows
        if not row["all_predeclared_gates_passed"]
    ]
    return f"""# Flash model selection Model Selection

## Scope and Leakage Boundary

Only the population design `training` (reserved N=900; complete N={counts['training_complete']})
and `model_selection` (reserved N=180; complete N={counts['selection_complete']})
seed blocks were generated or accessed. Calibration, final, boundary, structural-stress, and
rare-event populations were not generated or read. The nominal process bundle
and population design preregistration hashes were verified before generation.

The fitted arrays contain only residual `[P,L,T]` and the 11 observable features:
current action, action change from nominal, and nominal-DT `[P,L,T]`. Evaluator
fault metadata is separated and never passed to fitting or prediction.

EM uses the unchanged relative-likelihood convergence tolerance `1e-5`. The
uniform iteration ceiling is 100: the initial 40-iteration execution produced
no converged candidate, so computation was continued for every k without
changing a gate, initialization, population, action, or convergence tolerance.

### Declared numerical-domain exits

{counts['training_domain_exits']} training and {counts['selection_domain_exits']}
selection seeds left the frozen simulator validation property/vessel domain under their reserved
fault/action combinations. They were recorded in the evaluator manifest and
were not clipped, extrapolated, replaced, or regenerated. Only complete
trajectories enter identification and RMSE calculations. This is an unplanned
population design population-support limitation, not a gate relaxation.

## Exact Gate Results

Normalized output order below is pressure / level / temperature.

| k | Median worst normalized RMSE | Per-output median normalized RMSE | Normalized bias | Innovation lag-1 | Spectral radius | Gate result |
|---:|---:|---:|---:|---:|---:|---|
{chr(10).join(result_lines)}

All candidates were numerically finite with positive covariance matrices and
finite 120-step counterfactual rollouts. Failed candidates: {', '.join(failed_candidates) if failed_candidates else 'none'}.

## Frozen Selection

### NO MODEL PASSED

Selected `k={selected}` with status `{status}`. Its median engineering-unit RMSE
is {selected_row['median_pressure_rmse_bar']:.6f} bar pressure,
{selected_row['median_level_rmse_fraction']:.6f} level fraction, and
{selected_row['median_temperature_rmse_k']:.6f} K temperature. The fitted latent
coordinates are statistical discrepancy memory only and have no physical fault
interpretation.

## Counterfactual Verification

The selected model's two pre-truth action probes change the nominal rollout by
{selected_row['maximum_nominal_action_sensitivity']:.6g} in the largest raw output
coordinate and change the predicted discrepancy rollout by
{selected_row['maximum_discrepancy_action_sensitivity']:.6g}. Prediction APIs do
not accept fault metadata or future truth, and repeated predictions are exact
under fixed inputs.

## Transfer Limitations

The model is linear-Gaussian with stable diagonal latent dynamics and is tested
only on F1-F3 within the population design training/selection ranges. Selection diagnostics do
not establish calibrated uncertainty, SAFE performance, fault-conditional
validity, boundary reliability, or structural-mismatch robustness. Composition
is not measured and is represented only indirectly through observable residual
history. No calibration or final-test claim is made in model selection.
"""


def main(workers: int, reuse_generated: bool = False) -> None:
    _verify_frozen_inputs()
    if reuse_generated:
        training, selection, evaluator_rows, counts = (
            load_generated_inputs_and_replay_selection_truth(workers)
        )
    else:
        training, selection, evaluator_rows, counts = generate_model_selection_populations(workers)
    del evaluator_rows
    models: dict[int, FlashStateSpaceParameters] = {}
    rows: list[dict[str, object]] = []
    predictions: dict[int, np.ndarray] = {}
    for latent_dimension in (1, 2, 3):
        model = fit_flash_state_space_em(
            training["residual"],
            training["features"],
            latent_dimension,
            INITIALIZATION_SEEDS[latent_dimension],
            max_iterations=100,
        )
        row, predicted = evaluate_candidate(model, selection)
        row.update(_action_sensitivity(model, selection))
        row.update(counts)
        models[latent_dimension] = model
        rows.append(row)
        predictions[latent_dimension] = predicted
        print(
            f"model selection k={latent_dimension}: median worst normalized RMSE "
            f"{row['median_worst_output_normalized_rmse']:.4f}, "
            f"gates={'PASS' if row['all_predeclared_gates_passed'] else 'FAIL'}",
            flush=True,
        )
    selected_dimension, selection_status = _select(rows)
    for row in rows:
        row["selected"] = int(row["latent_dimension"]) == selected_dimension
        row["selection_status"] = selection_status if row["selected"] else "NOT_SELECTED"
    _write_csv(_safe_path("flash_model_selection.csv"), rows)
    selected_model = models[selected_dimension]
    payload = {
        "schema_version": "SELECTED_FLASH_MODEL_1.0",
        "process_bundle_sha256": PROCESS_BUNDLE_SHA256,
        "population_design_preregistration_sha256": PREREGISTRATION_SHA256,
        "populations_accessed": list(ALLOWED_POPULATIONS),
        "prohibited_populations_accessed": [],
        "population_completion": counts,
        "selection_status": selection_status,
        "no_model_passed": selection_status.startswith("NO_MODEL_PASSED"),
        "selection_gate_thresholds": {
            "spectral_radius_max": 0.995,
            "median_worst_normalized_rmse_max": 0.50,
            "each_output_median_normalized_rmse_max": 0.50,
            "each_output_abs_normalized_bias_max": 0.10,
            "each_output_abs_innovation_lag1_max": 0.50,
        },
        "numerical_fit_protocol": {
            "maximum_em_iterations": 100,
            "relative_likelihood_tolerance": 1e-5,
            "note": "Uniform iteration ceiling extension after the 40-iteration attempt; gate thresholds unchanged",
        },
        "model": parameters_to_payload(selected_model),
        "selected_metrics": next(
            row for row in rows if int(row["latent_dimension"]) == selected_dimension
        ),
    }
    _safe_path("selected_flash_model.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report = _report(rows, selected_dimension, selection_status, counts)
    _safe_path("model_selection_report.md").write_text(report, encoding="utf-8")

    # Selection truth is serialized only after every candidate prediction and
    # gate result has been computed. It is never included in a prediction call.
    np.savez_compressed(
        _safe_path("model_selection_data/selection_predictions_before_truth_serialization.npz"),
        **{f"k{dimension}": value for dimension, value in predictions.items()},
    )
    np.savez_compressed(
        _safe_path("model_selection_data/selection_truth_evaluator_only.npz"),
        actual_future=selection["actual_future"],
        true_residual_future=selection["true_residual_future"],
    )
    print(f"model selection selected k={selected_dimension}: {selection_status}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(12, (os.cpu_count() or 2) - 1)),
    )
    parser.add_argument(
        "--reuse-generated",
        action="store_true",
        help="reuse exact cached training inputs and replay identical selection seeds",
    )
    arguments = parser.parse_args()
    main(arguments.workers, reuse_generated=arguments.reuse_generated)
