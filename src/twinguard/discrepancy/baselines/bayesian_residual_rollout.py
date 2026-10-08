"""Coherent recursive posterior rollouts for residual-dynamics models."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from twinguard.discrepancy.baselines.bayesian_linear_residual import (
    BayesianLinearPosterior,
    BayesianResidualDynamicsModel,
)
from twinguard.discrepancy.baselines.gaussian_residual import (
    BayesianGaussianResidualModel,
    GaussianPosterior,
)


@dataclass(frozen=True)
class CounterfactualRolloutResult:
    residual_trajectories: np.ndarray
    corrected_output_trajectories: np.ndarray
    posterior_sampling_time_s: float
    recursive_rollout_time_s: float

    def pointwise_interval(
        self, coverage: float
    ) -> tuple[np.ndarray, np.ndarray]:
        if not 0.0 < coverage < 1.0:
            raise ValueError("coverage must lie in (0, 1)")
        tail = 0.5 * (1.0 - coverage)
        return (
            np.quantile(self.corrected_output_trajectories, tail, axis=0),
            np.quantile(
                self.corrected_output_trajectories, 1.0 - tail, axis=0
            ),
        )


def _validate_future_sequences(
    nominal_twin_trajectory: np.ndarray, action_sequence: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    nominal = np.asarray(nominal_twin_trajectory, dtype=float)
    actions = np.asarray(action_sequence, dtype=float)
    if nominal.ndim != 1 or actions.ndim != 1 or nominal.size != actions.size:
        raise ValueError("nominal trajectory and action sequence must be aligned vectors")
    if nominal.size == 0 or not np.all(np.isfinite(nominal)) or not np.all(
        np.isfinite(actions)
    ):
        raise ValueError("future sequences must be non-empty and finite")
    return nominal, actions


def rollout_nominal_only(
    nominal_twin_trajectory: np.ndarray, n_trajectories: int
) -> CounterfactualRolloutResult:
    nominal = np.asarray(nominal_twin_trajectory, dtype=float)
    if n_trajectories <= 0 or nominal.ndim != 1:
        raise ValueError("invalid nominal rollout request")
    residuals = np.zeros((n_trajectories, nominal.size), dtype=float)
    return CounterfactualRolloutResult(
        residual_trajectories=residuals,
        corrected_output_trajectories=np.repeat(
            nominal[None, :], n_trajectories, axis=0
        ),
        posterior_sampling_time_s=0.0,
        recursive_rollout_time_s=0.0,
    )


def rollout_static_gaussian(
    posterior: GaussianPosterior,
    nominal_twin_trajectory: np.ndarray,
    n_trajectories: int,
    rng: np.random.Generator,
) -> CounterfactualRolloutResult:
    nominal = np.asarray(nominal_twin_trajectory, dtype=float)
    sampling_start = time.perf_counter()
    biases = BayesianGaussianResidualModel.sample_constant_biases(
        posterior, n_trajectories, rng
    )
    sampling_time = time.perf_counter() - sampling_start
    rollout_start = time.perf_counter()
    residuals = np.repeat(biases[:, None], nominal.size, axis=1)
    corrected = nominal[None, :] + residuals
    rollout_time = time.perf_counter() - rollout_start
    return CounterfactualRolloutResult(
        residual_trajectories=residuals,
        corrected_output_trajectories=corrected,
        posterior_sampling_time_s=sampling_time,
        recursive_rollout_time_s=rollout_time,
    )


def recursive_residual_dynamics_rollout(
    posterior: BayesianLinearPosterior,
    nominal_twin_trajectory: np.ndarray,
    action_sequence: np.ndarray,
    initial_residual: float,
    initial_delta_residual: float,
    previous_action: float,
    n_trajectories: int,
    rng: np.random.Generator,
) -> CounterfactualRolloutResult:
    """Recursively propagate one posterior model draw per trajectory.

    A coefficient vector and noise variance are sampled once for each trajectory
    and retained for the entire horizon.  Fresh process innovations are sampled
    at each step; future residual states are never reset or independently drawn.
    """

    nominal, actions = _validate_future_sequences(
        nominal_twin_trajectory, action_sequence
    )
    if not np.isfinite(initial_residual) or not np.isfinite(initial_delta_residual):
        raise ValueError("initial residual state must be finite")
    if n_trajectories <= 0:
        raise ValueError("n_trajectories must be positive")

    sampling_start = time.perf_counter()
    coefficients, noise_variances = BayesianResidualDynamicsModel.sample_posterior(
        posterior, n_trajectories, rng
    )
    sampling_time = time.perf_counter() - sampling_start

    rollout_start = time.perf_counter()
    residual_paths = np.empty((n_trajectories, nominal.size), dtype=float)
    current_residual = np.full(n_trajectories, initial_residual, dtype=float)
    current_delta = np.full(n_trajectories, initial_delta_residual, dtype=float)
    prior_action = float(previous_action)

    for step, (nominal_next, action_now) in enumerate(zip(nominal, actions)):
        feature_values = {
            "r_t": current_residual,
            "delta_r_t": current_delta,
            "u_t": np.full(n_trajectories, action_now),
            "delta_u_t": np.full(n_trajectories, action_now - prior_action),
            "y_dt_next": np.full(n_trajectories, nominal_next),
        }
        raw_features = np.column_stack(
            [feature_values[name] for name in posterior.feature_names]
        )
        design = posterior.transform_features(raw_features)
        conditional_delta_mean = np.einsum("ij,ij->i", design, coefficients)
        innovations = rng.normal(size=n_trajectories) * np.sqrt(noise_variances)
        next_delta = conditional_delta_mean + innovations
        next_residual = current_residual + next_delta
        residual_paths[:, step] = next_residual
        current_residual = next_residual
        current_delta = next_delta
        prior_action = float(action_now)

    corrected_paths = nominal[None, :] + residual_paths
    rollout_time = time.perf_counter() - rollout_start
    return CounterfactualRolloutResult(
        residual_trajectories=residual_paths,
        corrected_output_trajectories=corrected_paths,
        posterior_sampling_time_s=sampling_time,
        recursive_rollout_time_s=rollout_time,
    )
