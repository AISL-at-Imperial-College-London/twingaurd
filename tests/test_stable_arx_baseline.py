from __future__ import annotations

import numpy as np
import pytest

from twinguard.discrepancy.baselines.stable_arx import (
    BayesianDirectARXModel,
    arx_feature_names,
    companion_matrices,
    draw_stability_conditioned_samples,
    free_running_residual_rollout,
    spectral_radii,
)


def test_arx_features_are_exactly_observable_allowlist() -> None:
    assert arx_feature_names(2) == (
        "r_lag_1", "r_lag_2", "u_t", "delta_u_t", "y_dt_next"
    )
    with pytest.raises(ValueError):
        arx_feature_names(0)


def test_companion_matrix_and_radius() -> None:
    # intercept, a1, a2, three exogenous coefficients
    raw = np.asarray([0.0, 0.5, 0.2, 1.0, 2.0, 3.0])
    matrix = companion_matrices(raw, 2)
    assert np.allclose(matrix, [[0.5, 0.2], [1.0, 0.0]])
    assert spectral_radii(matrix) == pytest.approx(
        max(abs(np.linalg.eigvals(matrix)))
    )


def test_stability_sampler_rejects_without_clipping() -> None:
    rng = np.random.default_rng(7)
    residual = np.zeros(80)
    for index in range(2, residual.size):
        residual[index] = 0.7 * residual[index - 1] - 0.1 * residual[index - 2] + rng.normal(0, 0.1)
    x = np.column_stack((residual[1:-1], residual[:-2], np.zeros((78, 3))))
    y = residual[2:]
    model = BayesianDirectARXModel(2)
    posterior = model.fit(x, y)
    coefficients, variances, radii, stats = draw_stability_conditioned_samples(
        model, posterior, 100, np.random.default_rng(8)
    )
    assert coefficients.shape == (100, 6)
    assert variances.shape == radii.shape == (100,)
    assert np.all(radii < 1.0)
    assert stats["stable_draws_accepted"] == 100


def test_direct_rollout_uses_lag_history() -> None:
    model = BayesianDirectARXModel(1)
    x = np.column_stack((np.linspace(0, 1, 30), np.zeros((30, 3))))
    y = 0.5 * x[:, 0]
    posterior = model.fit(x, y)
    coefficient = posterior.standardized_coefficient_mean[None, :]
    result = free_running_residual_rollout(
        posterior,
        coefficient,
        np.zeros((1, 3)),
        np.zeros(3),
        np.zeros(3),
        np.asarray([1.0]),
        0.0,
        1,
    )
    assert result.shape == (1, 3)
    assert result[0, 1] < result[0, 0]
