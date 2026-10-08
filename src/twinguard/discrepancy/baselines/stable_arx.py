"""Bayesian direct-residual ARX models with stability-conditioned sampling.

This module deliberately models only online-observable quantities.  It fits the
ordinary conjugate Bayesian linear model first; stability is imposed afterwards
by rejecting posterior coefficient draws whose AR companion matrix has spectral
radius greater than or equal to one.  Coefficients are never clipped.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from twinguard.discrepancy.baselines.bayesian_linear_residual import (
    BayesianLinearPosterior,
    BayesianResidualDynamicsModel,
)


EXOGENOUS_FEATURES = ("u_t", "delta_u_t", "y_dt_next")
FORBIDDEN_FEATURE_TOKENS = (
    "fault", "ua", "fouling", "hidden", "parameter", "magnitude", "plant_state"
)


def arx_feature_names(order: int) -> tuple[str, ...]:
    """Return the frozen direct-ARX feature order for a positive lag order."""
    if not isinstance(order, int) or isinstance(order, bool) or order < 1:
        raise ValueError("ARX order must be a positive integer")
    return tuple(f"r_lag_{lag}" for lag in range(1, order + 1)) + EXOGENOUS_FEATURES


def assert_arx_feature_names(order: int, names: tuple[str, ...]) -> None:
    expected = arx_feature_names(order)
    if tuple(names) != expected:
        raise ValueError(f"ARX({order}) features must be exactly {expected}")
    suspicious = [
        name for name in names
        if any(token in name.lower() for token in FORBIDDEN_FEATURE_TOKENS)
    ]
    if suspicious:
        raise ValueError(f"hidden feature(s) rejected: {sorted(suspicious)}")


@dataclass(frozen=True)
class BayesianDirectARXModel:
    """Normal--inverse-gamma regression for direct residual ARX(p) dynamics."""

    order: int
    prior_precision: float = 0.01
    prior_alpha: float = 2.0
    prior_beta: float = 0.25

    @property
    def feature_names(self) -> tuple[str, ...]:
        return arx_feature_names(self.order)

    def __post_init__(self) -> None:
        assert_arx_feature_names(self.order, self.feature_names)
        if self.prior_precision <= 0.0:
            raise ValueError("prior_precision must be positive")
        if self.prior_alpha <= 1.0 or self.prior_beta <= 0.0:
            raise ValueError("inverse-gamma prior must have a finite variance mean")

    def fit(self, features: np.ndarray, targets: np.ndarray) -> BayesianLinearPosterior:
        x = np.asarray(features, dtype=float)
        y = np.asarray(targets, dtype=float)
        if x.ndim != 2 or x.shape[1] != len(self.feature_names):
            raise ValueError(f"features must have shape (n, {len(self.feature_names)})")
        if y.ndim != 1 or y.size != x.shape[0] or x.shape[0] < 2:
            raise ValueError("targets must align with at least two feature rows")
        if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
            raise ValueError("training data must be finite")
        feature_mean = np.mean(x, axis=0)
        feature_scale = np.std(x, axis=0, ddof=0)
        feature_scale = np.where(feature_scale > 1.0e-12, feature_scale, 1.0)
        design = np.column_stack(
            (np.ones(x.shape[0]), (x - feature_mean[None, :]) / feature_scale[None, :])
        )
        dimension = design.shape[1]
        precision = self.prior_precision * np.eye(dimension) + design.T @ design
        factor = cho_factor(precision, lower=True, check_finite=True)
        precision_inverse = cho_solve(factor, np.eye(dimension), check_finite=True)
        coefficient_mean = cho_solve(factor, design.T @ y, check_finite=True)
        alpha = self.prior_alpha + 0.5 * x.shape[0]
        beta = self.prior_beta + 0.5 * (
            float(y @ y) - float(coefficient_mean @ precision @ coefficient_mean)
        )
        return BayesianLinearPosterior(
            feature_names=self.feature_names,
            n_observations=int(x.shape[0]),
            standardization_mean=np.asarray(feature_mean),
            standardization_scale=np.asarray(feature_scale),
            standardized_coefficient_mean=np.asarray(coefficient_mean),
            standardized_precision_inverse=np.asarray(precision_inverse),
            alpha=float(alpha),
            beta=max(float(beta), np.finfo(float).tiny),
        )

    @staticmethod
    def sample_posterior(
        posterior: BayesianLinearPosterior,
        n_samples: int,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray]:
        return BayesianResidualDynamicsModel.sample_posterior(
            posterior, n_samples, rng
        )


def standardized_to_raw_coefficients(
    posterior: BayesianLinearPosterior, standardized_coefficients: np.ndarray
) -> np.ndarray:
    """Convert standardized coefficient draws to original feature units."""
    values = np.asarray(standardized_coefficients, dtype=float)
    one = values.ndim == 1
    if one:
        values = values[None, :]
    expected = len(posterior.feature_names) + 1
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"expected coefficient samples with {expected} columns")
    transform = np.zeros((expected, expected), dtype=float)
    transform[0, 0] = 1.0
    transform[1:, 0] = -posterior.standardization_mean / posterior.standardization_scale
    transform[1:, 1:] = np.diag(1.0 / posterior.standardization_scale)
    raw = values @ transform
    return raw[0] if one else raw


def companion_matrices(raw_coefficients: np.ndarray, order: int) -> np.ndarray:
    """Construct one AR companion matrix per raw coefficient vector."""
    coefficients = np.asarray(raw_coefficients, dtype=float)
    one = coefficients.ndim == 1
    if one:
        coefficients = coefficients[None, :]
    expected = 1 + order + len(EXOGENOUS_FEATURES)
    if coefficients.ndim != 2 or coefficients.shape[1] != expected:
        raise ValueError(f"raw coefficients must have {expected} columns")
    matrices = np.zeros((coefficients.shape[0], order, order), dtype=float)
    matrices[:, 0, :] = coefficients[:, 1 : 1 + order]
    if order > 1:
        matrices[:, 1:, :-1] = np.eye(order - 1)[None, :, :]
    return matrices[0] if one else matrices


def spectral_radii(matrices: np.ndarray) -> np.ndarray | float:
    values = np.asarray(matrices, dtype=float)
    one = values.ndim == 2
    if one:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[1] != values.shape[2]:
        raise ValueError("spectral radius requires square companion matrices")
    radii = np.max(np.abs(np.linalg.eigvals(values)), axis=1)
    return float(radii[0]) if one else np.asarray(radii, dtype=float)


def coefficient_spectral_radii(
    posterior: BayesianLinearPosterior,
    standardized_coefficients: np.ndarray,
    order: int,
) -> np.ndarray:
    raw = standardized_to_raw_coefficients(posterior, standardized_coefficients)
    return np.atleast_1d(spectral_radii(companion_matrices(raw, order))).astype(float)


def draw_stability_conditioned_samples(
    model: BayesianDirectARXModel,
    posterior: BayesianLinearPosterior,
    requested_samples: int,
    coefficient_rng: np.random.Generator,
    *,
    max_attempts: int = 2_000_000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, float | int]]:
    """Sample the ordinary posterior and retain only dynamically stable draws."""
    if requested_samples <= 0:
        raise ValueError("requested_samples must be positive")
    accepted_coefficients: list[np.ndarray] = []
    accepted_variances: list[np.ndarray] = []
    accepted_radii: list[np.ndarray] = []
    attempted = 0
    stable_candidates = 0
    raw_unstable = 0
    first_batch = True
    while sum(item.shape[0] for item in accepted_coefficients) < requested_samples:
        need = requested_samples - sum(item.shape[0] for item in accepted_coefficients)
        batch = requested_samples if first_batch else max(
            256, min(8192, int(math.ceil(1.2 * need)))
        )
        if attempted + batch > max_attempts:
            batch = max_attempts - attempted
        if batch <= 0:
            raise RuntimeError(
                f"ARX({model.order}) stability acceptance was computationally "
                f"problematic: fewer than {requested_samples} stable draws in "
                f"{max_attempts} attempts"
            )
        coefficients, variances = model.sample_posterior(
            posterior, batch, coefficient_rng
        )
        radii = coefficient_spectral_radii(posterior, coefficients, model.order)
        stable_index = np.flatnonzero(radii < 1.0)
        stable_candidates += stable_index.size
        if first_batch:
            raw_size = min(requested_samples, batch)
            raw_unstable = int(np.count_nonzero(radii[:raw_size] >= 1.0))
            first_batch = False
        take = stable_index[:need]
        attempted += batch
        if take.size:
            accepted_coefficients.append(coefficients[take])
            accepted_variances.append(variances[take])
            accepted_radii.append(radii[take])
    coefficients = np.vstack(accepted_coefficients)[:requested_samples]
    variances = np.concatenate(accepted_variances)[:requested_samples]
    radii = np.concatenate(accepted_radii)[:requested_samples]
    if np.any(radii >= 1.0):
        raise RuntimeError("unstable coefficient draw survived rejection sampling")
    return coefficients, variances, radii, {
        "raw_sample_count": requested_samples,
        "raw_unstable_count": raw_unstable,
        "raw_unstable_fraction": raw_unstable / requested_samples,
        "posterior_draws_attempted": attempted,
        "stable_draws_accepted": requested_samples,
        "stable_draw_acceptance_rate": stable_candidates / attempted,
    }


def arx_training_data(
    residuals: np.ndarray,
    actions: np.ndarray,
    nominal_next: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build direct ARX rows; arrays describe a complete observed sequence."""
    r = np.asarray(residuals, dtype=float)
    u = np.asarray(actions, dtype=float)
    ydt = np.asarray(nominal_next, dtype=float)
    if r.ndim != 1 or u.shape != r.shape or ydt.shape != r.shape:
        raise ValueError("residual, action, and nominal arrays must be aligned vectors")
    if r.size <= order:
        raise ValueError("observation sequence is too short for requested AR order")
    rows: list[list[float]] = []
    targets: list[float] = []
    for target_index in range(order, r.size):
        action = float(u[target_index])
        previous_action = float(u[target_index - 1])
        lags = [float(r[target_index - lag]) for lag in range(1, order + 1)]
        rows.append(lags + [action, action - previous_action, float(ydt[target_index])])
        targets.append(float(r[target_index]))
    return np.asarray(rows, dtype=float), np.asarray(targets, dtype=float)


def teacher_forced_features(
    actual_residual: np.ndarray,
    nominal_next: np.ndarray,
    actions: np.ndarray,
    initial_history: np.ndarray,
    previous_action: float,
    order: int,
) -> np.ndarray:
    """Build held-out one-step rows from true prior residuals only."""
    actual = np.asarray(actual_residual, dtype=float)
    nominal = np.asarray(nominal_next, dtype=float)
    future_actions = np.asarray(actions, dtype=float)
    history = list(np.asarray(initial_history, dtype=float))
    if len(history) < order:
        raise ValueError("initial history does not contain all requested lags")
    rows = []
    prior_action = float(previous_action)
    for index, next_residual in enumerate(actual):
        action = float(future_actions[index])
        lags = [history[-lag] for lag in range(1, order + 1)]
        rows.append(lags + [action, action - prior_action, float(nominal[index])])
        history.append(float(next_residual))
        prior_action = action
    return np.asarray(rows, dtype=float)


def free_running_residual_rollout(
    posterior: BayesianLinearPosterior,
    standardized_coefficients: np.ndarray,
    innovations: np.ndarray,
    nominal_next: np.ndarray,
    actions: np.ndarray,
    initial_history: np.ndarray,
    previous_action: float,
    order: int,
) -> np.ndarray:
    """Recursively propagate direct residual paths with fixed draw per path."""
    coefficients = np.asarray(standardized_coefficients, dtype=float)
    noise = np.asarray(innovations, dtype=float)
    sample_count, horizon = noise.shape
    if coefficients.shape != (sample_count, len(posterior.feature_names) + 1):
        raise ValueError("coefficient and innovation sample counts must align")
    history_values = np.asarray(initial_history, dtype=float)
    if history_values.size < order:
        raise ValueError("initial history does not contain all requested lags")
    history = np.tile(history_values[-order:][None, :], (sample_count, 1))
    result = np.empty((sample_count, horizon), dtype=float)
    prior_action = float(previous_action)
    for step in range(horizon):
        action = float(actions[step])
        lag_columns = [history[:, -lag] for lag in range(1, order + 1)]
        raw_features = np.column_stack(
            lag_columns
            + [
                np.full(sample_count, action),
                np.full(sample_count, action - prior_action),
                np.full(sample_count, float(nominal_next[step])),
            ]
        )
        design = posterior.transform_features(raw_features)
        next_residual = np.einsum("ij,ij->i", design, coefficients) + noise[:, step]
        result[:, step] = next_residual
        if order > 1:
            history[:, :-1] = history[:, 1:]
        history[:, -1] = next_residual
        prior_action = action
    if not np.all(np.isfinite(result)):
        raise RuntimeError("non-finite stable ARX rollout")
    return result
