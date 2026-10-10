"""Multivariable latent Gaussian discrepancy model for the flash drum.

The latent coordinates are statistical memory states only. They are never
interpreted as fault identities or physical parameters. Residual output order is
[pressure bar, level fraction, temperature K].
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import minimize

from twinguard.flash.digital_twin import FlashDigitalTwin
from twinguard.scenarios.flash_population_design import (
    ACTION_NAMES,
    MEASURED_OUTPUTS,
    PREDICTION_HORIZON_S,
    SAMPLING_INTERVAL_S,
    validate_candidate_action,
)
from twinguard.plants.flash_drum import FlashAction, FlashState


STABILITY_LIMIT = 0.995
ENGINEERING_SCALES = np.asarray([0.10, 0.10, 5.0])
FEATURE_NAMES = tuple(ACTION_NAMES) + tuple(f"delta_{name}" for name in ACTION_NAMES) + (
    "nominal_pressure_bar",
    "nominal_level_fraction",
    "nominal_temperature_k",
)


def _positive_definite(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    return (eigenvectors * np.maximum(eigenvalues, floor)) @ eigenvectors.T


@dataclass(frozen=True)
class FlashStateSpaceParameters:
    A: np.ndarray
    B: np.ndarray
    C: np.ndarray
    D: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    initial_mean: np.ndarray
    initial_covariance: np.ndarray
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    residual_scale: np.ndarray
    log_likelihood: float
    iterations: int
    converged: bool
    initialization_seed: int
    fit_seconds: float
    final_parameter_change: float

    @property
    def latent_dimension(self) -> int:
        return int(self.A.shape[0])

    @property
    def spectral_radius(self) -> float:
        return float(np.max(np.abs(np.linalg.eigvals(self.A))))

    def standardize_features(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != len(FEATURE_NAMES):
            raise ValueError("feature dimension mismatch")
        return (array - self.feature_mean) / self.feature_scale

    def standardize_residuals(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=float)
        if array.shape[-1] != len(MEASURED_OUTPUTS):
            raise ValueError("residual output dimension mismatch")
        return array / self.residual_scale


@dataclass(frozen=True)
class MultivariableFilterResult:
    filtered_mean: np.ndarray
    filtered_covariance: np.ndarray
    predicted_mean: np.ndarray
    predicted_covariance: np.ndarray
    innovations: np.ndarray
    innovation_covariance: np.ndarray
    standardized_innovations: np.ndarray
    log_likelihood: float


@dataclass(frozen=True)
class CorrectedPrediction:
    nominal_outputs: np.ndarray
    predicted_discrepancy: np.ndarray
    corrected_outputs: np.ndarray
    future_features: np.ndarray


def kalman_filter(
    parameters: FlashStateSpaceParameters,
    residual: np.ndarray,
    features: np.ndarray,
) -> MultivariableFilterResult:
    y = parameters.standardize_residuals(np.asarray(residual, dtype=float))
    x = parameters.standardize_features(np.asarray(features, dtype=float))
    if y.ndim != 2 or x.ndim != 2 or y.shape[0] != x.shape[0]:
        raise ValueError("residual and feature sequences must align")
    steps, output_count = y.shape
    latent_count = parameters.latent_dimension
    filtered_mean = np.empty((steps, latent_count))
    filtered_covariance = np.empty((steps, latent_count, latent_count))
    predicted_mean = np.empty_like(filtered_mean)
    predicted_covariance = np.empty_like(filtered_covariance)
    innovations = np.empty((steps, output_count))
    innovation_covariance = np.empty((steps, output_count, output_count))
    standardized = np.empty_like(innovations)
    log_likelihood = 0.0
    for step in range(steps):
        if step == 0:
            prior_mean = parameters.initial_mean
            prior_covariance = parameters.initial_covariance
        else:
            prior_mean = parameters.A @ filtered_mean[step - 1] + parameters.B @ x[step]
            prior_covariance = _positive_definite(
                parameters.A @ filtered_covariance[step - 1] @ parameters.A.T + parameters.Q
            )
        expected = parameters.C @ prior_mean + parameters.D @ x[step]
        covariance = _positive_definite(
            parameters.C @ prior_covariance @ parameters.C.T + parameters.R
        )
        innovation = y[step] - expected
        inverse = np.linalg.inv(covariance)
        gain = prior_covariance @ parameters.C.T @ inverse
        posterior_mean = prior_mean + gain @ innovation
        posterior_covariance = _positive_definite(
            prior_covariance - gain @ parameters.C @ prior_covariance
        )
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise FloatingPointError("innovation covariance is not positive definite")
        log_likelihood += -0.5 * (
            output_count * np.log(2.0 * np.pi) + logdet + innovation @ inverse @ innovation
        )
        predicted_mean[step] = prior_mean
        predicted_covariance[step] = prior_covariance
        filtered_mean[step] = posterior_mean
        filtered_covariance[step] = posterior_covariance
        innovations[step] = innovation
        innovation_covariance[step] = covariance
        standard_deviation = np.sqrt(np.maximum(np.diag(covariance), 1e-12))
        standardized[step] = innovation / standard_deviation
    return MultivariableFilterResult(
        filtered_mean,
        filtered_covariance,
        predicted_mean,
        predicted_covariance,
        innovations,
        innovation_covariance,
        standardized,
        float(log_likelihood),
    )


def _stable_dynamic_row(
    xx: np.ndarray,
    yx: np.ndarray,
    yy: float,
    initial_a: float,
    initial_b: np.ndarray,
) -> tuple[float, np.ndarray, float]:
    theta = np.arctanh(np.clip(initial_a / STABILITY_LIMIT, -0.999, 0.999))
    initial = np.concatenate(([theta], initial_b))

    def objective(values: np.ndarray) -> float:
        coefficients = np.concatenate(
            ([STABILITY_LIMIT * np.tanh(values[0])], values[1:])
        )
        return float(yy - 2.0 * coefficients @ yx + coefficients @ xx @ coefficients)

    result = minimize(
        objective,
        initial,
        method="L-BFGS-B",
        options={"maxiter": 100, "ftol": 1e-12},
    )
    a = float(STABILITY_LIMIT * np.tanh(result.x[0]))
    return a, np.asarray(result.x[1:]), max(objective(result.x), 1e-8)


def _initial_parameters(
    residual: np.ndarray,
    features: np.ndarray,
    latent_dimension: int,
    initialization_seed: int,
) -> FlashStateSpaceParameters:
    rng = np.random.default_rng(initialization_seed)
    feature_mean = np.mean(features, axis=(0, 1))
    feature_scale = np.std(features, axis=(0, 1))
    feature_scale = np.where(feature_scale > 1e-12, feature_scale, 1.0)
    standardized_features = (features - feature_mean) / feature_scale
    standardized_residual = residual / ENGINEERING_SCALES
    flat_x = standardized_features.reshape(-1, standardized_features.shape[-1])
    flat_y = standardized_residual.reshape(-1, standardized_residual.shape[-1])
    D = np.linalg.lstsq(flat_x, flat_y, rcond=None)[0].T
    unexplained = flat_y - flat_x @ D.T
    covariance = _positive_definite(np.cov(unexplained, rowvar=False), 1e-5)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    loadings = eigenvectors[:, order[:latent_dimension]]
    C = loadings * np.sqrt(np.maximum(eigenvalues[order[:latent_dimension]], 1e-3))
    modes = np.linspace(0.45, 0.85, latent_dimension)
    modes += rng.normal(0.0, 0.015, latent_dimension)
    A = np.diag(np.clip(modes, -0.95, 0.95))
    return FlashStateSpaceParameters(
        A=A,
        B=rng.normal(0.0, 0.01, (latent_dimension, flat_x.shape[1])),
        C=C,
        D=D,
        Q=np.eye(latent_dimension) * 0.05,
        R=_positive_definite(covariance * 0.5, 1e-5),
        initial_mean=np.zeros(latent_dimension),
        initial_covariance=np.eye(latent_dimension),
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        residual_scale=ENGINEERING_SCALES.copy(),
        log_likelihood=-np.inf,
        iterations=0,
        converged=False,
        initialization_seed=initialization_seed,
        fit_seconds=0.0,
        final_parameter_change=float("nan"),
    )


def _batched_em_update(
    current: FlashStateSpaceParameters,
    y: np.ndarray,
    x: np.ndarray,
) -> tuple[FlashStateSpaceParameters, float, float]:
    episodes, steps, outputs = y.shape
    latent = current.latent_dimension
    features = x.shape[2]
    A, B, C, D, Q, R = current.A, current.B, current.C, current.D, current.Q, current.R
    filtered_mean = np.empty((episodes, steps, latent))
    predicted_mean = np.empty_like(filtered_mean)
    filtered_covariance = np.empty((steps, latent, latent))
    predicted_covariance = np.empty_like(filtered_covariance)
    total_likelihood = 0.0
    for step in range(steps):
        if step == 0:
            prior_mean = np.broadcast_to(current.initial_mean, (episodes, latent))
            prior_covariance = current.initial_covariance
        else:
            prior_mean = filtered_mean[:, step - 1] @ A.T + x[:, step] @ B.T
            prior_covariance = _positive_definite(
                A @ filtered_covariance[step - 1] @ A.T + Q
            )
        expected = prior_mean @ C.T + x[:, step] @ D.T
        covariance = _positive_definite(C @ prior_covariance @ C.T + R)
        inverse = np.linalg.inv(covariance)
        innovation = y[:, step] - expected
        gain = prior_covariance @ C.T @ inverse
        filtered_mean[:, step] = prior_mean + innovation @ gain.T
        filtered_covariance[step] = _positive_definite(
            prior_covariance - gain @ C @ prior_covariance
        )
        predicted_mean[:, step] = prior_mean
        predicted_covariance[step] = prior_covariance
        sign, logdet = np.linalg.slogdet(covariance)
        if sign <= 0:
            raise FloatingPointError("non-positive innovation covariance")
        quadratic = np.einsum("ei,ij,ej->e", innovation, inverse, innovation)
        total_likelihood += float(
            np.sum(-0.5 * (outputs * np.log(2.0 * np.pi) + logdet + quadratic))
        )

    smoothed_mean = filtered_mean.copy()
    smoothed_covariance = filtered_covariance.copy()
    lag_covariance = np.zeros_like(filtered_covariance)
    for step in range(steps - 2, -1, -1):
        gain = filtered_covariance[step] @ A.T @ np.linalg.inv(
            predicted_covariance[step + 1]
        )
        smoothed_mean[:, step] = filtered_mean[:, step] + (
            smoothed_mean[:, step + 1] - predicted_mean[:, step + 1]
        ) @ gain.T
        smoothed_covariance[step] = _positive_definite(
            filtered_covariance[step]
            + gain
            @ (smoothed_covariance[step + 1] - predicted_covariance[step + 1])
            @ gain.T
        )
        lag_covariance[step + 1] = smoothed_covariance[step + 1] @ gain.T

    dimension = latent + features
    observation_xx = np.zeros((dimension, dimension))
    observation_xx[:latent, :latent] = episodes * np.sum(
        smoothed_covariance, axis=0
    ) + np.einsum("etk,etl->kl", smoothed_mean, smoothed_mean)
    state_feature = np.einsum("etk,etp->kp", smoothed_mean, x)
    observation_xx[:latent, latent:] = state_feature
    observation_xx[latent:, :latent] = state_feature.T
    observation_xx[latent:, latent:] = np.einsum("etp,etq->pq", x, x)
    observation_yx = np.empty((outputs, dimension))
    observation_yx[:, :latent] = np.einsum("ety,etk->yk", y, smoothed_mean)
    observation_yx[:, latent:] = np.einsum("ety,etp->yp", y, x)
    H = observation_yx @ np.linalg.pinv(observation_xx)
    new_C, new_D = H[:, :latent], H[:, latent:]
    yy = np.einsum("ety,etz->yz", y, y)
    new_R = _positive_definite(
        (yy - H @ observation_yx.T) / (episodes * steps), 1e-6
    )

    new_A = np.zeros_like(A)
    new_B = np.zeros_like(B)
    new_q = np.empty(latent)
    future_x = x[:, 1:]
    for mode in range(latent):
        xx = np.zeros((features + 1, features + 1))
        previous = smoothed_mean[:, :-1, mode]
        current_mean = smoothed_mean[:, 1:, mode]
        xx[0, 0] = episodes * np.sum(smoothed_covariance[:-1, mode, mode]) + np.sum(
            previous**2
        )
        state_feature = np.einsum("et,etp->p", previous, future_x)
        xx[0, 1:] = state_feature
        xx[1:, 0] = state_feature
        xx[1:, 1:] = np.einsum("etp,etq->pq", future_x, future_x)
        yx = np.empty(features + 1)
        yx[0] = episodes * np.sum(lag_covariance[1:, mode, mode]) + np.sum(
            current_mean * previous
        )
        yx[1:] = np.einsum("et,etp->p", current_mean, future_x)
        y_squared = episodes * np.sum(smoothed_covariance[1:, mode, mode]) + np.sum(
            current_mean**2
        )
        a, b, sse = _stable_dynamic_row(
            xx, yx, float(y_squared), A[mode, mode], B[mode]
        )
        new_A[mode, mode] = a
        new_B[mode] = b
        new_q[mode] = max(sse / (episodes * (steps - 1)), 1e-6)

    new_initial_mean = np.mean(smoothed_mean[:, 0], axis=0)
    new_initial_covariance = _positive_definite(
        smoothed_covariance[0]
        + np.cov(smoothed_mean[:, 0], rowvar=False, ddof=0),
        1e-6,
    )
    old_vector = np.concatenate(
        [A.ravel(), B.ravel(), C.ravel(), D.ravel(), Q.ravel(), R.ravel()]
    )
    new_vector = np.concatenate(
        [
            new_A.ravel(),
            new_B.ravel(),
            new_C.ravel(),
            new_D.ravel(),
            np.diag(new_q).ravel(),
            new_R.ravel(),
        ]
    )
    change = float(np.linalg.norm(new_vector - old_vector) / max(1.0, np.linalg.norm(old_vector)))
    updated = FlashStateSpaceParameters(
        A=new_A,
        B=new_B,
        C=new_C,
        D=new_D,
        Q=np.diag(new_q),
        R=new_R,
        initial_mean=new_initial_mean,
        initial_covariance=new_initial_covariance,
        feature_mean=current.feature_mean,
        feature_scale=current.feature_scale,
        residual_scale=current.residual_scale,
        log_likelihood=total_likelihood,
        iterations=current.iterations + 1,
        converged=False,
        initialization_seed=current.initialization_seed,
        fit_seconds=0.0,
        final_parameter_change=change,
    )
    return updated, total_likelihood, change


def fit_flash_state_space_em(
    residual: np.ndarray,
    features: np.ndarray,
    latent_dimension: int,
    initialization_seed: int,
    *,
    max_iterations: int = 40,
    relative_tolerance: float = 1e-5,
) -> FlashStateSpaceParameters:
    if latent_dimension not in (1, 2, 3):
        raise ValueError("population design permits latent dimensions 1, 2, and 3 only")
    raw_y = np.asarray(residual, dtype=float)
    raw_x = np.asarray(features, dtype=float)
    if raw_y.ndim != 3 or raw_y.shape[2] != 3:
        raise ValueError("residual must have shape (episodes, time, 3)")
    if raw_x.ndim != 3 or raw_x.shape[:2] != raw_y.shape[:2]:
        raise ValueError("features must align with residual episodes and time")
    parameters = _initial_parameters(raw_y, raw_x, latent_dimension, initialization_seed)
    y = parameters.standardize_residuals(raw_y)
    x = parameters.standardize_features(raw_x)
    previous_likelihood = -np.inf
    converged = False
    started = time.perf_counter()
    change = float("nan")
    for _ in range(max_iterations):
        parameters, likelihood, change = _batched_em_update(parameters, y, x)
        if np.isfinite(previous_likelihood) and abs(likelihood - previous_likelihood) <= (
            relative_tolerance * (1.0 + abs(previous_likelihood))
        ):
            converged = True
            break
        previous_likelihood = likelihood
    elapsed = time.perf_counter() - started
    return FlashStateSpaceParameters(
        **{
            **parameters.__dict__,
            "converged": converged,
            "fit_seconds": elapsed,
            "final_parameter_change": change,
        }
    )


def predict_discrepancy(
    parameters: FlashStateSpaceParameters,
    observed_residual: np.ndarray,
    observed_features: np.ndarray,
    future_features: np.ndarray,
) -> tuple[np.ndarray, MultivariableFilterResult]:
    filtered = kalman_filter(parameters, observed_residual, observed_features)
    standardized_future = parameters.standardize_features(future_features)
    state = filtered.filtered_mean[-1].copy()
    predicted = np.empty((len(future_features), 3))
    for step, feature in enumerate(standardized_future):
        state = parameters.A @ state + parameters.B @ feature
        predicted[step] = parameters.C @ state + parameters.D @ feature
    return predicted * parameters.residual_scale, filtered


def nominal_future_and_features(
    decision_state: FlashState,
    candidate_action: FlashAction | Sequence[float],
    nominal_action: FlashAction | Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    candidate = validate_candidate_action(candidate_action)
    twin = FlashDigitalTwin()
    reference = (
        twin.nominal_action()
        if nominal_action is None
        else validate_candidate_action(nominal_action)
    )
    trajectory = twin.simulate(
        candidate,
        PREDICTION_HORIZON_S,
        initial_state=decision_state,
        integration_step_s=1.0,
        sample_interval_s=SAMPLING_INTERVAL_S,
    )
    nominal_outputs = np.column_stack(
        [trajectory[name][1:] for name in MEASURED_OUTPUTS]
    )
    action = candidate.as_array()
    delta = action - reference.as_array()
    feature_prefix = np.concatenate([action, delta])
    future_features = np.column_stack(
        [
            np.broadcast_to(feature_prefix, (len(nominal_outputs), len(feature_prefix))),
            nominal_outputs,
        ]
    )
    return nominal_outputs, future_features


def counterfactual_prediction(
    parameters: FlashStateSpaceParameters,
    observed_residual: np.ndarray,
    observed_features: np.ndarray,
    decision_dt_state: FlashState,
    candidate_action: FlashAction | Sequence[float],
) -> CorrectedPrediction:
    nominal, future_features = nominal_future_and_features(
        decision_dt_state, candidate_action
    )
    discrepancy, _ = predict_discrepancy(
        parameters, observed_residual, observed_features, future_features
    )
    return CorrectedPrediction(
        nominal_outputs=nominal,
        predicted_discrepancy=discrepancy,
        corrected_outputs=nominal + discrepancy,
        future_features=future_features,
    )


def parameters_to_payload(parameters: FlashStateSpaceParameters) -> dict[str, object]:
    arrays = (
        "A",
        "B",
        "C",
        "D",
        "Q",
        "R",
        "initial_mean",
        "initial_covariance",
        "feature_mean",
        "feature_scale",
        "residual_scale",
    )
    payload = {name: getattr(parameters, name).tolist() for name in arrays}
    payload.update(
        {
            "latent_dimension": parameters.latent_dimension,
            "spectral_radius": parameters.spectral_radius,
            "log_likelihood": parameters.log_likelihood,
            "iterations": parameters.iterations,
            "converged": parameters.converged,
            "initialization_seed": parameters.initialization_seed,
            "fit_seconds": parameters.fit_seconds,
            "final_parameter_change": parameters.final_parameter_change,
            "feature_names": list(FEATURE_NAMES),
            "output_names": list(MEASURED_OUTPUTS),
            "latent_state_interpretation": "statistical_discrepancy_state_only",
        }
    )
    return payload


def parameters_from_payload(payload: Mapping[str, object]) -> FlashStateSpaceParameters:
    array_names = (
        "A",
        "B",
        "C",
        "D",
        "Q",
        "R",
        "initial_mean",
        "initial_covariance",
        "feature_mean",
        "feature_scale",
        "residual_scale",
    )
    arrays = {name: np.asarray(payload[name], dtype=float) for name in array_names}
    return FlashStateSpaceParameters(
        **arrays,
        log_likelihood=float(payload["log_likelihood"]),
        iterations=int(payload["iterations"]),
        converged=bool(payload["converged"]),
        initialization_seed=int(payload["initialization_seed"]),
        fit_seconds=float(payload["fit_seconds"]),
        final_parameter_change=float(payload["final_parameter_change"]),
    )
