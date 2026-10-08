from __future__ import annotations

import numpy as np
import pytest

from twinguard.discrepancy.baselines.bayesian_linear_residual import BayesianResidualDynamicsModel
from twinguard.cstr.episode_design import FROZEN_FEATURES
from twinguard.cstr.recursive_rollout_diagnostics import (
    spectral_radii,
    standardized_to_raw_coefficients,
    transition_matrices,
    true_future_feature_matrix,
)


def test_transition_matrix_matches_recursive_update() -> None:
    coefficients = np.asarray([0.4, -0.2, 0.3, 1.1, -0.7, 0.05])
    matrix = transition_matrices(coefficients, FROZEN_FEATURES)
    state = np.asarray([2.0, -0.5])
    autonomous_delta = coefficients[1] * state[0] + coefficients[2] * state[1]
    expected = np.asarray([state[0] + autonomous_delta, autonomous_delta])
    assert matrix @ state == pytest.approx(expected)
    assert np.allclose(matrix, [[0.8, 0.3], [-0.2, 0.3]])


def test_known_stable_and_unstable_spectral_radii() -> None:
    stable = np.asarray([[0.5, 0.0], [0.0, 0.2]])
    unstable = np.asarray([[1.1, 0.0], [0.0, 0.2]])
    assert spectral_radii(stable) == pytest.approx(0.5)
    assert spectral_radii(unstable) == pytest.approx(1.1)
    assert spectral_radii(np.stack((stable, unstable))) == pytest.approx([0.5, 1.1])


def test_standardized_sample_conversion_preserves_predictions() -> None:
    rng = np.random.default_rng(4)
    features = rng.normal(size=(30, len(FROZEN_FEATURES)))
    features[:, 2] *= 0.02
    targets = rng.normal(size=30)
    model = BayesianResidualDynamicsModel(FROZEN_FEATURES)
    posterior = model.fit(features, targets)
    standardized, _ = model.sample_posterior(posterior, 8, rng)
    raw = standardized_to_raw_coefficients(posterior, standardized)
    test_features = rng.normal(size=(8, len(FROZEN_FEATURES)))
    standardized_design = posterior.transform_features(test_features)
    raw_design = np.column_stack((np.ones(8), test_features))
    standardized_prediction = np.einsum("ij,ij->i", standardized_design, standardized)
    raw_prediction = np.einsum("ij,ij->i", raw_design, raw)
    assert raw_prediction == pytest.approx(standardized_prediction)


def test_teacher_forced_features_use_true_previous_state() -> None:
    actual = np.asarray([1.2, 1.7, 1.1])
    nominal = np.asarray([300.0, 301.0, 302.0])
    actions = np.asarray([0.03, 0.03, 0.03])
    features, targets = true_future_feature_matrix(
        actual,
        nominal,
        actions,
        initial_residual=1.0,
        initial_delta=0.2,
        previous_action=0.02,
    )
    assert features[:, 0] == pytest.approx([1.0, 1.2, 1.7])
    assert features[:, 1] == pytest.approx([0.2, 0.2, 0.5])
    assert features[:, 3] == pytest.approx([0.01, 0.0, 0.0])
    assert targets == pytest.approx([0.2, 0.5, -0.6])
