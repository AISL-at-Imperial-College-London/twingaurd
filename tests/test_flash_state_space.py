from __future__ import annotations

import inspect
import csv
import json

import numpy as np

from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.discrepancy.flash_state_space import (
    ENGINEERING_SCALES,
    FEATURE_NAMES,
    FlashStateSpaceParameters,
    counterfactual_prediction,
    fit_flash_state_space_em,
    kalman_filter,
)
from twinguard.scenarios.flash_population_design import FORBIDDEN_OPERATIONAL_TOKENS
from twinguard.plants.flash_drum import FlashAction
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT


def _parameters() -> FlashStateSpaceParameters:
    twin = FlashDigitalTwin()
    action = twin.nominal_action().as_array()
    output = twin.outputs(twin.nominal_state())
    feature_mean = np.concatenate(
        [action, np.zeros(4), [output["pressure_bar"], output["level_fraction"], output["temperature_k"]]]
    )
    feature_scale = np.asarray([5000.0, 0.2, 0.2, 0.2, 5000.0, 0.2, 0.2, 0.2, 0.1, 0.1, 5.0])
    B = np.zeros((1, len(FEATURE_NAMES)))
    B[0, 4:8] = [0.03, -0.02, 0.02, 0.01]
    D = np.zeros((3, len(FEATURE_NAMES)))
    D[:, 4:8] = [[0.04, -0.03, 0.01, 0.01], [0.01, -0.02, -0.04, 0.03], [0.05, -0.04, 0.01, -0.02]]
    return FlashStateSpaceParameters(
        A=np.asarray([[0.8]]),
        B=B,
        C=np.asarray([[0.5], [0.25], [0.8]]),
        D=D,
        Q=np.asarray([[0.02]]),
        R=np.diag([0.03, 0.03, 0.03]),
        initial_mean=np.zeros(1),
        initial_covariance=np.eye(1),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        residual_scale=ENGINEERING_SCALES.copy(),
        log_likelihood=0.0,
        iterations=1,
        converged=True,
        initialization_seed=1,
        fit_seconds=0.0,
        final_parameter_change=0.0,
    )


def _history(parameters: FlashStateSpaceParameters) -> tuple[np.ndarray, np.ndarray]:
    features = np.broadcast_to(parameters.feature_mean, (50, len(FEATURE_NAMES))).copy()
    residual = np.zeros((50, 3))
    residual[:, 0] = np.linspace(0.0, 0.015, 50)
    residual[:, 1] = np.linspace(0.0, -0.01, 50)
    residual[:, 2] = np.linspace(0.0, 0.8, 50)
    return residual, features


def test_multivariable_filter_shapes_and_covariances() -> None:
    parameters = _parameters()
    residual, features = _history(parameters)
    result = kalman_filter(parameters, residual, features)
    assert result.filtered_mean.shape == (50, 1)
    assert result.innovations.shape == (50, 3)
    assert result.innovation_covariance.shape == (50, 3, 3)
    assert np.all(np.linalg.eigvalsh(result.innovation_covariance) > 0.0)


def test_prediction_api_cannot_receive_fault_or_future_truth() -> None:
    names = set(inspect.signature(counterfactual_prediction).parameters)
    assert not any(token in name for name in names for token in FORBIDDEN_OPERATIONAL_TOKENS)


def test_future_truth_cannot_change_prediction() -> None:
    parameters = _parameters()
    residual, features = _history(parameters)
    action = FlashDigitalTwin().nominal_action()
    state = FlashDigitalTwin().nominal_state()
    first = counterfactual_prediction(parameters, residual, features, state, action)
    fake_future_truth = np.full((120, 3), 1e9)
    fake_future_truth *= -3.0
    second = counterfactual_prediction(parameters, residual, features, state, action)
    np.testing.assert_array_equal(first.corrected_outputs, second.corrected_outputs)


def test_hidden_fault_parameters_cannot_change_prediction_except_via_history() -> None:
    parameters = _parameters()
    residual, features = _history(parameters)
    twin = FlashDigitalTwin()
    action = twin.nominal_action()
    baseline = counterfactual_prediction(
        parameters, residual, features, twin.nominal_state(), action
    )
    # There is no plant/fault argument; evaluator metadata has no route in.
    repeated = counterfactual_prediction(
        parameters, residual.copy(), features.copy(), twin.nominal_state(), action
    )
    np.testing.assert_array_equal(
        baseline.predicted_discrepancy, repeated.predicted_discrepancy
    )
    changed_history = residual.copy()
    changed_history[-10:, 0] += 0.03
    changed = counterfactual_prediction(
        parameters, changed_history, features, twin.nominal_state(), action
    )
    assert not np.array_equal(
        baseline.predicted_discrepancy, changed.predicted_discrepancy
    )


def test_candidate_action_changes_nominal_and_discrepancy_rollouts_before_truth() -> None:
    parameters = _parameters()
    residual, features = _history(parameters)
    state = FlashDigitalTwin().nominal_state()
    first = counterfactual_prediction(
        parameters, residual, features, state, FlashAction(12_000.0, 0.40, 0.65, 0.85)
    )
    second = counterfactual_prediction(
        parameters, residual, features, state, FlashAction(20_000.0, 0.70, 0.35, 1.15)
    )
    assert not np.array_equal(first.nominal_outputs, second.nominal_outputs)
    assert not np.array_equal(first.predicted_discrepancy, second.predicted_discrepancy)


def test_counterfactual_predictions_are_exactly_reproducible() -> None:
    parameters = _parameters()
    residual, features = _history(parameters)
    twin = FlashDigitalTwin()
    first = counterfactual_prediction(
        parameters, residual, features, twin.nominal_state(), twin.nominal_action()
    )
    second = counterfactual_prediction(
        parameters, residual, features, twin.nominal_state(), twin.nominal_action()
    )
    np.testing.assert_array_equal(first.nominal_outputs, second.nominal_outputs)
    np.testing.assert_array_equal(first.predicted_discrepancy, second.predicted_discrepancy)
    np.testing.assert_array_equal(first.corrected_outputs, second.corrected_outputs)


def test_small_multivariable_em_fit_is_stable_and_dimensionally_correct() -> None:
    rng = np.random.default_rng(33000999)
    features = rng.normal(size=(12, 20, len(FEATURE_NAMES)))
    residual = np.empty((12, 20, 3))
    state = np.zeros((12, 1))
    for step in range(20):
        state = 0.75 * state + 0.05 * features[:, step, :1]
        residual[:, step, 0] = 0.02 * state[:, 0]
        residual[:, step, 1] = -0.015 * state[:, 0]
        residual[:, step, 2] = 0.8 * state[:, 0]
    residual += rng.normal(0.0, [0.002, 0.002, 0.1], residual.shape)
    model = fit_flash_state_space_em(
        residual, features, 1, 33000999, max_iterations=3
    )
    assert model.A.shape == (1, 1)
    assert model.C.shape == (3, 1)
    assert model.D.shape == (3, len(FEATURE_NAMES))
    assert model.R.shape == (3, 3)
    assert model.spectral_radius < 0.995
    assert np.all(np.linalg.eigvalsh(model.R) > 0.0)


def test_model_selection_method_inputs_exclude_evaluator_metadata() -> None:
    with np.load(EXPERIMENT_ROOT / "model_selection_data" / "training_method_inputs.npz") as saved:
        assert set(saved.files) == {"residual", "features"}
        assert saved["residual"].shape == (667, 170, 3)
        assert saved["features"].shape == (667, 170, len(FEATURE_NAMES))
    with np.load(
        EXPERIMENT_ROOT / "model_selection_data" / "selection_operational_inputs.npz"
    ) as saved:
        assert set(saved.files) == {
            "observed_residual",
            "observed_features",
            "future_features",
            "nominal_future",
            "candidate_actions",
        }
        assert saved["observed_residual"].shape == (143, 50, 3)
        assert saved["nominal_future"].shape == (143, 120, 3)
        assert not any(
            token in name for name in saved.files for token in FORBIDDEN_OPERATIONAL_TOKENS
        )


def test_model_selection_used_only_training_and_model_selection_roles() -> None:
    with (
        EXPERIMENT_ROOT / "model_selection_data" / "EVALUATOR_GENERATION_MANIFEST.csv"
    ).open(newline="", encoding="utf-8") as handle:
        roles = {row["population_role"] for row in csv.DictReader(handle)}
    assert roles == {"training", "model_selection"}
    payload = json.loads(
        (EXPERIMENT_ROOT / "selected_flash_model.json").read_text(encoding="utf-8")
    )
    assert payload["populations_accessed"] == ["training", "model_selection"]
    assert payload["prohibited_populations_accessed"] == []


def test_no_candidate_passed_and_k3_is_only_the_predeclared_fallback() -> None:
    with (EXPERIMENT_ROOT / "flash_model_selection.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["latent_dimension"]) for row in rows] == [1, 2, 3]
    assert {row["all_predeclared_gates_passed"] for row in rows} == {"False"}
    selected = [row for row in rows if row["selected"] == "True"]
    assert len(selected) == 1
    assert selected[0]["latent_dimension"] == "3"
    assert selected[0]["selection_status"] == "NO_MODEL_PASSED_PREDECLARED_FALLBACK"


def test_selection_predictions_are_serialized_before_evaluator_truth() -> None:
    prediction = (
        EXPERIMENT_ROOT
        / "model_selection_data"
        / "selection_predictions_before_truth_serialization.npz"
    )
    truth = EXPERIMENT_ROOT / "model_selection_data" / "selection_truth_evaluator_only.npz"
    assert prediction.stat().st_mtime_ns <= truth.stat().st_mtime_ns
