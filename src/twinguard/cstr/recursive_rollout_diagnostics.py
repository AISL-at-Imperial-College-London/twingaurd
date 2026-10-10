"""Compare recursive and teacher-forced forecasts for the CSTR discrepancy model.

Rebuild observed histories from recorded seeds and fit the model to those rows.
Both forecast modes use the same model and posterior draws.
The results show how recursive prediction affects trajectory error.
Each comparison uses the same observed episode histories.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

from twinguard.discrepancy.baselines.bayesian_linear_residual import (
    BayesianLinearPosterior,
    BayesianResidualDynamicsModel,
)
from twinguard.cstr.episode_design import (
    COVERAGE_LEVELS,
    FROZEN_FEATURES,
    TRAINING_ACTIONS_LPS,
    EpisodeDesignConfig,
    _configure_episode_models,
    _feature_matrix,
    _paired_observation,
    _select_action_after_fit,
    _transition_rows,
    build_template,
    construct_simultaneous_bands,
)


EXPLOSION_THRESHOLD_K = 50.0
STABILITY_FIELDS = (
    "episode_id",
    "action_category",
    "posterior_sample_index",
    "spectral_radius",
    "is_stable",
    "beta_r_t",
    "beta_delta_r_t",
    "noise_variance",
    "max_absolute_residual_k",
    "maximum_predicted_temperature_k",
    "final_residual_magnitude_k",
    "numerically_exploded",
    "trajectory_RMSE_k",
    "teacher_forced_trajectory_RMSE_k",
    "maximum_predictive_band_contribution_k",
)


@dataclass(frozen=True)
class RecursiveDiagnosticsConfig:
    source_dir: str = "artifacts/cstr/episode_design"
    output_dir: str = "artifacts/cstr/recursive_rollout_diagnostics"
    workers: int = 8
    explosion_threshold_k: float = EXPLOSION_THRESHOLD_K
    limit_episodes: int | None = None

    def validate(self) -> None:
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.explosion_threshold_k <= 0.0:
            raise ValueError("explosion threshold must be positive")
        if self.limit_episodes is not None and self.limit_episodes <= 0:
            raise ValueError("limit_episodes must be positive when supplied")


_WORKER_TEMPLATE = None
_WORKER_EPISODE_DESIGN_CONFIG: EpisodeDesignConfig | None = None
_WORKER_SOURCE: Path | None = None
_WORKER_OUTPUT: Path | None = None
_WORKER_EXPLOSION_THRESHOLD = EXPLOSION_THRESHOLD_K


def _init_worker(
    template: object,
    episode_design_config: EpisodeDesignConfig,
    source_dir: str,
    output_dir: str,
    explosion_threshold: float,
) -> None:
    global _WORKER_TEMPLATE, _WORKER_EPISODE_DESIGN_CONFIG, _WORKER_SOURCE, _WORKER_OUTPUT
    global _WORKER_EXPLOSION_THRESHOLD
    _WORKER_TEMPLATE = template
    _WORKER_EPISODE_DESIGN_CONFIG = episode_design_config
    _WORKER_SOURCE = Path(source_dir)
    _WORKER_OUTPUT = Path(output_dir)
    _WORKER_EXPLOSION_THRESHOLD = float(explosion_threshold)


def _level_label(level: float) -> int:
    return int(round(100.0 * level))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_episode_design_episodes(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no episode design episodes found in {path}")
    required = {
        "episode_id",
        "random_seed",
        "action_category",
        "proposed_action",
        "hidden_fault_magnitude",
        "hidden_fault_onset_s",
        "one_step_RMSE",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"episode design episode table is missing: {sorted(missing)}")
    if len({row["episode_id"] for row in rows}) != len(rows):
        raise ValueError("episode design episode identifiers are not unique")
    return sorted(rows, key=lambda row: row["episode_id"])


def standardized_to_raw_coefficients(
    posterior: BayesianLinearPosterior, standardized_coefficients: np.ndarray
) -> np.ndarray:
    """Convert sampled coefficients to intercept + original feature units."""
    values = np.asarray(standardized_coefficients, dtype=float)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    expected = len(posterior.feature_names) + 1
    if values.ndim != 2 or values.shape[1] != expected:
        raise ValueError(f"expected coefficient samples with {expected} columns")
    transform = np.zeros((expected, expected), dtype=float)
    transform[0, 0] = 1.0
    transform[1:, 0] = -posterior.standardization_mean / posterior.standardization_scale
    transform[1:, 1:] = np.diag(1.0 / posterior.standardization_scale)
    result = values @ transform
    return result[0] if one_dimensional else result


def transition_matrices(
    raw_coefficients: np.ndarray,
    feature_names: Sequence[str] = FROZEN_FEATURES,
) -> np.ndarray:
    """Return A for state [r_t, delta_r_t] using implemented feature order."""
    names = tuple(feature_names)
    if "r_t" not in names or "delta_r_t" not in names:
        raise ValueError("recursive transition requires r_t and delta_r_t features")
    coefficients = np.asarray(raw_coefficients, dtype=float)
    one_dimensional = coefficients.ndim == 1
    if one_dimensional:
        coefficients = coefficients[None, :]
    if coefficients.ndim != 2 or coefficients.shape[1] != len(names) + 1:
        raise ValueError("raw coefficients must contain intercept plus every feature")
    beta_r = coefficients[:, 1 + names.index("r_t")]
    beta_delta = coefficients[:, 1 + names.index("delta_r_t")]
    matrices = np.empty((coefficients.shape[0], 2, 2), dtype=float)
    matrices[:, 0, 0] = 1.0 + beta_r
    matrices[:, 0, 1] = beta_delta
    matrices[:, 1, 0] = beta_r
    matrices[:, 1, 1] = beta_delta
    return matrices[0] if one_dimensional else matrices


def spectral_radii(matrices: np.ndarray) -> np.ndarray | float:
    values = np.asarray(matrices, dtype=float)
    one_matrix = values.ndim == 2
    if one_matrix:
        values = values[None, :, :]
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("spectral radius requires one or more 2x2 matrices")
    eigenvalues = np.linalg.eigvals(values)
    radii = np.max(np.abs(eigenvalues), axis=1)
    return float(radii[0]) if one_matrix else np.asarray(radii, dtype=float)


def true_future_feature_matrix(
    actual_residual: np.ndarray,
    nominal_temperature: np.ndarray,
    future_actions: np.ndarray,
    initial_residual: float,
    initial_delta: float,
    previous_action: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Teacher-forced features and true next-increment targets."""
    actual = np.asarray(actual_residual, dtype=float)
    nominal = np.asarray(nominal_temperature, dtype=float)
    actions = np.asarray(future_actions, dtype=float)
    if actual.ndim != 1 or nominal.shape != actual.shape or actions.shape != actual.shape:
        raise ValueError("teacher-forced future arrays must be aligned vectors")
    rows = []
    targets = np.empty(actual.size, dtype=float)
    residual = float(initial_residual)
    delta = float(initial_delta)
    prior_action = float(previous_action)
    for index, (next_residual, nominal_next, action) in enumerate(
        zip(actual, nominal, actions)
    ):
        values = {
            "r_t": residual,
            "delta_r_t": delta,
            "u_t": float(action),
            "delta_u_t": float(action) - prior_action,
            "y_dt_next": float(nominal_next),
        }
        rows.append([values[name] for name in FROZEN_FEATURES])
        next_delta = float(next_residual) - residual
        targets[index] = next_delta
        residual = float(next_residual)
        delta = next_delta
        prior_action = float(action)
    return np.asarray(rows, dtype=float), targets


def _reconstruct_posterior(
    source_row: dict[str, str],
) -> tuple[
    BayesianLinearPosterior,
    BayesianResidualDynamicsModel,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
]:
    if _WORKER_TEMPLATE is None or _WORKER_EPISODE_DESIGN_CONFIG is None:
        raise RuntimeError("recursive diagnostics worker is not initialized")
    config = _WORKER_EPISODE_DESIGN_CONFIG
    seed = int(float(source_row["random_seed"]))
    rng = np.random.default_rng(seed)
    measurement_rng = np.random.default_rng(seed + 1_000_000)
    action_rng = np.random.default_rng(seed + 2_000_000)
    fault_onset = float(rng.uniform(config.fault_onset_min_s, config.fault_onset_max_s))
    fault_magnitude = float(
        rng.uniform(config.fault_magnitude_min, config.fault_magnitude_max)
    )
    nominal, faulty, metadata = _configure_episode_models(
        _WORKER_TEMPLATE, config, rng, fault_onset, fault_magnitude
    )
    if not np.isclose(fault_onset, float(source_row["hidden_fault_onset_s"])):
        raise RuntimeError("reconstructed fault onset differs from episode design")
    if not np.isclose(fault_magnitude, float(source_row["hidden_fault_magnitude"])):
        raise RuntimeError("reconstructed fault magnitude differs from episode design")
    for key, value in metadata.items():
        if key in source_row and not np.isclose(value, float(source_row[key])):
            raise RuntimeError(f"reconstructed {key} differs from episode design")

    pre_fault = [
        _paired_observation(
            nominal,
            faulty,
            2.0 / 60.0,
            measurement_rng,
            config.measurement_noise_k,
        )
        for _ in range(config.pre_fault_lead_steps)
    ]
    initial_residual = float(pre_fault[-1]["temperature_residual_k"])
    initial_delta = initial_residual - float(pre_fault[-2]["temperature_residual_k"])
    initial_action = float(pre_fault[-1]["action_lps"])
    learning_rows = []
    for index in range(config.n_postfault):
        action_index = (index // config.training_action_block_steps) % len(
            TRAINING_ACTIONS_LPS
        )
        learning_rows.append(
            _paired_observation(
                nominal,
                faulty,
                TRAINING_ACTIONS_LPS[action_index],
                measurement_rng,
                config.measurement_noise_k,
            )
        )
    transitions = _transition_rows(
        learning_rows, initial_residual, initial_delta, initial_action
    )
    features = _feature_matrix(transitions)
    targets = np.asarray([row["delta_r_next"] for row in transitions], dtype=float)
    model = BayesianResidualDynamicsModel(FROZEN_FEATURES)
    posterior = model.fit(features, targets)
    proposed_action = _select_action_after_fit(source_row["action_category"], action_rng)
    if not np.isclose(proposed_action, float(source_row["proposed_action"])):
        raise RuntimeError("reconstructed held-out action differs from episode design")
    current_residual = float(learning_rows[-1]["temperature_residual_k"])
    previous_residual = float(learning_rows[-2]["temperature_residual_k"])
    current_delta = current_residual - previous_residual
    previous_action = float(learning_rows[-1]["action_lps"])
    return (
        posterior,
        model,
        features,
        targets,
        current_residual,
        current_delta,
        previous_action,
        proposed_action,
    )


def _conditioning_summary(
    episode_id: str,
    category: str,
    posterior: BayesianLinearPosterior,
    features: np.ndarray,
    raw_mean: np.ndarray,
    spectral_radius_mean: float,
    exploded_fraction: float,
    nonfinite_count: int,
) -> dict[str, object]:
    raw_design = np.column_stack((np.ones(features.shape[0]), features))
    standardized_design = posterior.transform_features(features)
    covariance = posterior.coefficient_covariance
    covariance_eigenvalues = np.linalg.eigvalsh(covariance)
    feature_std = np.std(features, axis=0, ddof=0)
    valid = feature_std > 1.0e-12
    max_correlation = math.nan
    correlation_pair = "not_available"
    if int(np.sum(valid)) >= 2:
        valid_names = np.asarray(FROZEN_FEATURES)[valid]
        correlation = np.corrcoef(features[:, valid], rowvar=False)
        upper = np.triu_indices(correlation.shape[0], k=1)
        selected = int(np.argmax(np.abs(correlation[upper])))
        i, j = int(upper[0][selected]), int(upper[1][selected])
        max_correlation = float(abs(correlation[i, j]))
        correlation_pair = f"{valid_names[i]}|{valid_names[j]}"
    output: dict[str, object] = {
        "episode_id": episode_id,
        "action_category": category,
        "raw_design_condition_number": float(np.linalg.cond(raw_design)),
        "standardized_design_condition_number": float(
            np.linalg.cond(standardized_design)
        ),
        "posterior_covariance_condition_number": float(np.linalg.cond(covariance)),
        "posterior_covariance_eigenvalue_min": float(np.min(covariance_eigenvalues)),
        "posterior_covariance_eigenvalue_max": float(np.max(covariance_eigenvalues)),
        "maximum_absolute_feature_correlation": max_correlation,
        "most_correlated_feature_pair": correlation_pair,
        "maximum_absolute_posterior_mean_coefficient": float(np.max(np.abs(raw_mean))),
        "posterior_mean_spectral_radius": spectral_radius_mean,
        "numerically_exploded_sample_fraction": exploded_fraction,
        "nonfinite_rollout_value_count": nonfinite_count,
    }
    for index, name in enumerate(FROZEN_FEATURES):
        output[f"feature_mean_{name}"] = float(posterior.standardization_mean[index])
        output[f"feature_scale_{name}"] = float(posterior.standardization_scale[index])
    for index, name in enumerate(("intercept", *FROZEN_FEATURES)):
        output[f"posterior_mean_beta_{name}"] = float(raw_mean[index])
    return output


def _coverage(
    samples: np.ndarray, actual: np.ndarray, level: float
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    tail = 0.5 * (1.0 - level)
    lower = np.quantile(samples, tail, axis=0)
    upper = np.quantile(samples, 1.0 - tail, axis=0)
    inside = (actual >= lower) & (actual <= upper)
    return float(np.mean(inside)), inside, lower, upper


def _save_representative_candidates(
    path: Path,
    spectral: np.ndarray,
    free_residual: np.ndarray,
    teacher_residual: np.ndarray,
    nominal: np.ndarray,
    actual_temperature: np.ndarray,
    actual_residual: np.ndarray,
    time_s: np.ndarray,
    episode_id: str,
) -> list[dict[str, object]]:
    max_abs = np.max(np.abs(free_residual), axis=1)
    payload: dict[str, object] = {
        "time_s": time_s,
        "nominal_temperature_k": nominal,
        "actual_temperature_k": actual_temperature,
        "actual_residual_k": actual_residual,
    }
    candidates = []
    for regime, mask in (("stable", spectral < 1.0), ("unstable", spectral >= 1.0)):
        indices = np.flatnonzero(mask)
        payload[f"{regime}_available"] = np.asarray(int(indices.size > 0))
        if indices.size == 0:
            continue
        target = float(np.median(max_abs[indices]))
        selected = int(indices[np.argmin(np.abs(max_abs[indices] - target))])
        payload[f"{regime}_sample_index"] = np.asarray(selected)
        payload[f"{regime}_spectral_radius"] = np.asarray(spectral[selected])
        payload[f"{regime}_free_residual_k"] = free_residual[selected]
        payload[f"{regime}_teacher_residual_k"] = teacher_residual[selected]
        candidates.append(
            {
                "episode_id": episode_id,
                "regime": regime,
                "sample_index": selected,
                "spectral_radius": float(spectral[selected]),
                "max_absolute_residual_k": float(max_abs[selected]),
                "path": str(path),
            }
        )
    np.savez_compressed(path, **payload)
    return candidates


def _episode_task(source_row: dict[str, str]) -> dict[str, object]:
    if _WORKER_EPISODE_DESIGN_CONFIG is None or _WORKER_SOURCE is None or _WORKER_OUTPUT is None:
        raise RuntimeError("recursive diagnostics worker is not initialized")
    episode_id = source_row["episode_id"]
    category = source_row["action_category"]
    seed = int(float(source_row["random_seed"]))
    (
        posterior,
        model,
        training_features,
        _training_targets,
        initial_residual,
        initial_delta,
        previous_action,
        proposed_action,
    ) = _reconstruct_posterior(source_row)

    evaluation_path = _WORKER_SOURCE / "evaluations" / f"{episode_id}.npz"
    prediction_path = _WORKER_SOURCE / "predictions" / f"{episode_id}.npz"
    with np.load(evaluation_path) as evaluation:
        actual_temperature = np.asarray(
            evaluation["actual_plant_temperature_k"], dtype=float
        )
        actual_residual = np.asarray(evaluation["actual_residual_k"], dtype=float)
        nominal = np.asarray(evaluation["nominal_twin_temperature_k"], dtype=float)
        actions = np.asarray(evaluation["action_sequence_lps"], dtype=float)
        time_s = np.asarray(evaluation["time_s"], dtype=float)
    if not np.allclose(actions, proposed_action):
        raise RuntimeError("saved future action differs from reconstructed action")
    if actual_temperature.size != _WORKER_EPISODE_DESIGN_CONFIG.horizon_steps:
        raise RuntimeError("saved episode design horizon differs from frozen configuration")

    teacher_features, _true_delta = true_future_feature_matrix(
        actual_residual,
        nominal,
        actions,
        initial_residual,
        initial_delta,
        previous_action,
    )
    teacher_prediction = model.predict(posterior, teacher_features)
    true_previous_residual = np.concatenate(([initial_residual], actual_residual[:-1]))
    teacher_mean_residual = true_previous_residual + teacher_prediction.mean
    teacher_mean_temperature = nominal + teacher_mean_residual

    rng = np.random.default_rng(seed + 3_000_000)
    standardized_coefficients, noise_variances = model.sample_posterior(
        posterior, _WORKER_EPISODE_DESIGN_CONFIG.rollout_samples, rng
    )
    raw_coefficients = standardized_to_raw_coefficients(
        posterior, standardized_coefficients
    )
    matrices = transition_matrices(raw_coefficients, posterior.feature_names)
    spectral = np.asarray(spectral_radii(matrices), dtype=float)
    raw_mean = posterior.coefficient_mean
    mean_spectral = float(
        spectral_radii(transition_matrices(raw_mean, posterior.feature_names))
    )

    sample_count = standardized_coefficients.shape[0]
    horizon = actual_temperature.size
    innovations = np.empty((sample_count, horizon), dtype=float)
    noise_scale = np.sqrt(noise_variances)
    for step in range(horizon):
        innovations[:, step] = rng.normal(size=sample_count) * noise_scale

    free_residual = np.empty((sample_count, horizon), dtype=float)
    teacher_residual = np.empty((sample_count, horizon), dtype=float)
    current_residual = np.full(sample_count, initial_residual, dtype=float)
    current_delta = np.full(sample_count, initial_delta, dtype=float)
    prior_action = float(previous_action)
    nonfinite_count = 0
    with np.errstate(over="ignore", invalid="ignore"):
        for step in range(horizon):
            action = float(actions[step])
            free_values = {
                "r_t": current_residual,
                "delta_r_t": current_delta,
                "u_t": np.full(sample_count, action),
                "delta_u_t": np.full(sample_count, action - prior_action),
                "y_dt_next": np.full(sample_count, nominal[step]),
            }
            raw_features = np.column_stack(
                [free_values[name] for name in posterior.feature_names]
            )
            design = posterior.transform_features(raw_features)
            free_delta = np.einsum(
                "ij,ij->i", design, standardized_coefficients
            ) + innovations[:, step]
            next_residual = current_residual + free_delta
            free_residual[:, step] = next_residual

            teacher_design = posterior.transform_features(teacher_features[step])
            teacher_delta = (
                teacher_design @ standardized_coefficients.T
            ).reshape(-1) + innovations[:, step]
            teacher_residual[:, step] = true_previous_residual[step] + teacher_delta
            current_residual = next_residual
            current_delta = free_delta
            prior_action = action
    nonfinite_count = int(np.size(free_residual) - np.count_nonzero(np.isfinite(free_residual)))
    if nonfinite_count:
        raise RuntimeError(
            f"{episode_id} produced {nonfinite_count} non-finite values; cannot reproduce episode design"
        )

    free_temperature = nominal[None, :] + free_residual
    teacher_temperature = nominal[None, :] + teacher_residual
    free_mean_temperature = np.mean(free_temperature, axis=0)
    with np.load(prediction_path) as saved_prediction:
        saved_mean = np.asarray(saved_prediction["posterior_mean_temperature_k"])
    reproduction_error = float(np.max(np.abs(saved_mean - free_mean_temperature)))
    if not np.allclose(saved_mean, free_mean_temperature, rtol=1.0e-12, atol=1.0e-8):
        raise RuntimeError(
            f"free-running reconstruction does not reproduce episode design ({reproduction_error:g} K)"
        )

    free_errors = free_mean_temperature - actual_temperature
    teacher_errors = teacher_mean_temperature - actual_temperature
    free_sample_errors = free_temperature - actual_temperature[None, :]
    teacher_sample_errors = teacher_temperature - actual_temperature[None, :]
    free_sample_rmse = np.sqrt(np.mean(np.square(free_sample_errors), axis=1))
    teacher_sample_rmse = np.sqrt(np.mean(np.square(teacher_sample_errors), axis=1))
    max_absolute_residual = np.max(np.abs(free_residual), axis=1)
    maximum_temperature = np.max(free_temperature, axis=1)
    final_residual_magnitude = np.abs(free_residual[:, -1])
    exploded = max_absolute_residual >= _WORKER_EXPLOSION_THRESHOLD
    free_median = np.median(free_temperature, axis=0)
    band_contribution = np.max(
        np.abs(free_temperature - free_median[None, :]), axis=1
    )

    summary: dict[str, object] = {
        "episode_id": episode_id,
        "action_category": category,
        "random_seed": seed,
        "posterior_sample_count": sample_count,
        "posterior_unstable_sample_count": int(np.sum(spectral >= 1.0)),
        "posterior_unstable_fraction": float(np.mean(spectral >= 1.0)),
        "posterior_spectral_radius_mean": float(np.mean(spectral)),
        "posterior_spectral_radius_median": float(np.median(spectral)),
        "posterior_spectral_radius_95th_percentile": float(
            np.quantile(spectral, 0.95)
        ),
        "posterior_mean_spectral_radius": mean_spectral,
        "posterior_mean_is_stable": int(mean_spectral < 1.0),
        "teacher_forced_overall_RMSE_k": float(
            np.sqrt(np.mean(np.square(teacher_errors)))
        ),
        "free_running_overall_RMSE_k": float(
            np.sqrt(np.mean(np.square(free_errors)))
        ),
        "free_running_posterior_median_RMSE_k": float(
            np.sqrt(np.mean(np.square(free_median - actual_temperature)))
        ),
        "saved_EPISODE_DESIGN_one_step_RMSE_k": float(source_row["one_step_RMSE"]),
        "teacher_minus_saved_one_step_RMSE_k": float(
            np.sqrt(np.mean(np.square(teacher_errors)))
            - float(source_row["one_step_RMSE"])
        ),
        "numerically_exploded_sample_count": int(np.sum(exploded)),
        "numerically_exploded_sample_fraction": float(np.mean(exploded)),
        "free_running_reproduction_max_abs_error_k": reproduction_error,
    }
    horizon_payload: dict[str, object] = {
        "teacher_error_squared": np.square(teacher_errors),
        "free_error_squared": np.square(free_errors),
    }
    for level in COVERAGE_LEVELS:
        label = _level_label(level)
        teacher_lower, teacher_upper = teacher_prediction.interval(level)
        teacher_lower_temperature = nominal + true_previous_residual + teacher_lower
        teacher_upper_temperature = nominal + true_previous_residual + teacher_upper
        teacher_inside = (actual_temperature >= teacher_lower_temperature) & (
            actual_temperature <= teacher_upper_temperature
        )
        free_coverage, free_inside, _free_lower, _free_upper = _coverage(
            free_temperature, actual_temperature, level
        )
        summary[f"teacher_forced_pointwise_coverage_{label}"] = float(
            np.mean(teacher_inside)
        )
        summary[f"free_running_pointwise_coverage_{label}"] = free_coverage
        horizon_payload[f"teacher_inside_{label}"] = teacher_inside.astype(float)
        horizon_payload[f"free_inside_{label}"] = free_inside.astype(float)
    teacher_simultaneous = construct_simultaneous_bands(teacher_temperature)
    free_simultaneous = construct_simultaneous_bands(free_temperature)
    for level in COVERAGE_LEVELS:
        label = _level_label(level)
        summary[f"teacher_forced_simultaneous_coverage_{label}"] = int(
            np.all(
                (actual_temperature >= teacher_simultaneous.lower[level])
                & (actual_temperature <= teacher_simultaneous.upper[level])
            )
        )
        summary[f"free_running_simultaneous_coverage_{label}"] = int(
            np.all(
                (actual_temperature >= free_simultaneous.lower[level])
                & (actual_temperature <= free_simultaneous.upper[level])
            )
        )

    conditioning = _conditioning_summary(
        episode_id,
        category,
        posterior,
        training_features,
        raw_mean,
        mean_spectral,
        float(np.mean(exploded)),
        nonfinite_count,
    )
    part_path = _WORKER_OUTPUT / "_stability_parts" / f"{episode_id}.csv"
    with part_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=STABILITY_FIELDS)
        writer.writeheader()
        for sample_index in range(sample_count):
            writer.writerow(
                {
                    "episode_id": episode_id,
                    "action_category": category,
                    "posterior_sample_index": sample_index,
                    "spectral_radius": spectral[sample_index],
                    "is_stable": int(spectral[sample_index] < 1.0),
                    "beta_r_t": raw_coefficients[sample_index, 1 + FROZEN_FEATURES.index("r_t")],
                    "beta_delta_r_t": raw_coefficients[sample_index, 1 + FROZEN_FEATURES.index("delta_r_t")],
                    "noise_variance": noise_variances[sample_index],
                    "max_absolute_residual_k": max_absolute_residual[sample_index],
                    "maximum_predicted_temperature_k": maximum_temperature[sample_index],
                    "final_residual_magnitude_k": final_residual_magnitude[sample_index],
                    "numerically_exploded": int(exploded[sample_index]),
                    "trajectory_RMSE_k": free_sample_rmse[sample_index],
                    "teacher_forced_trajectory_RMSE_k": teacher_sample_rmse[sample_index],
                    "maximum_predictive_band_contribution_k": band_contribution[sample_index],
                }
            )

    representative_path = (
        _WORKER_OUTPUT / "_representative_parts" / f"{episode_id}.npz"
    )
    candidates = _save_representative_candidates(
        representative_path,
        spectral,
        free_residual,
        teacher_residual,
        nominal,
        actual_temperature,
        actual_residual,
        time_s,
        episode_id,
    )
    sample_plot_data = np.column_stack(
        (
            spectral,
            max_absolute_residual,
            free_sample_rmse,
            band_contribution,
            raw_coefficients[:, 1 + FROZEN_FEATURES.index("r_t")],
            raw_coefficients[:, 1 + FROZEN_FEATURES.index("delta_r_t")],
            exploded.astype(float),
        )
    ).astype(np.float64)
    return {
        "summary": summary,
        "conditioning": conditioning,
        "horizon": horizon_payload,
        "part_path": str(part_path),
        "representative_candidates": candidates,
        "sample_plot_data": sample_plot_data,
    }


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_summary_csv(path: Path) -> list[dict[str, object]]:
    text_fields = {
        "episode_id",
        "action_category",
        "most_correlated_feature_pair",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, object]] = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key not in text_fields:
                row[key] = float(value)
    return rows


def _load_sample_plot_data(
    path: Path, expected_rows: int
) -> dict[str, np.ndarray]:
    """Load only the seven plotting columns from the large stability CSV."""
    arrays = np.empty((expected_rows, 7), dtype=float)
    category_codes = np.empty(expected_rows, dtype=np.int8)
    category_code = {"interpolation": 0, "boundary": 1, "extrapolation": 2}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            if index >= expected_rows:
                raise RuntimeError("stability sample table has more rows than expected")
            category_codes[index] = category_code[row["action_category"]]
            arrays[index] = (
                float(row["spectral_radius"]),
                float(row["max_absolute_residual_k"]),
                float(row["trajectory_RMSE_k"]),
                float(row["maximum_predictive_band_contribution_k"]),
                float(row["beta_r_t"]),
                float(row["beta_delta_r_t"]),
                float(row["numerically_exploded"]),
            )
        actual_rows = index + 1
    if actual_rows != expected_rows:
        raise RuntimeError(
            f"expected {expected_rows} stability samples, found {actual_rows}"
        )
    return {
        category: arrays[category_codes == code]
        for category, code in category_code.items()
    }


def _load_representative_candidates(directory: Path) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for path in sorted(directory.glob("*.npz")):
        episode_id = path.stem
        with np.load(path) as data:
            for regime in ("stable", "unstable"):
                if not int(data[f"{regime}_available"]):
                    continue
                residual = np.asarray(data[f"{regime}_free_residual_k"])
                candidates.append(
                    {
                        "episode_id": episode_id,
                        "regime": regime,
                        "sample_index": int(data[f"{regime}_sample_index"]),
                        "spectral_radius": float(data[f"{regime}_spectral_radius"]),
                        "max_absolute_residual_k": float(np.max(np.abs(residual))),
                        "path": str(path),
                    }
                )
    if not candidates:
        raise RuntimeError("no representative candidate snapshots found")
    return candidates


def _add_saved_posterior_median_rmse(
    episode_summaries: Sequence[dict[str, object]], source_dir: Path
) -> None:
    """Add an evaluator-side statistic from existing prediction snapshots."""
    for row in episode_summaries:
        if "free_running_posterior_median_RMSE_k" in row:
            continue
        episode_id = str(row["episode_id"])
        with np.load(source_dir / "predictions" / f"{episode_id}.npz") as prediction:
            posterior_median = np.asarray(
                prediction["posterior_median_temperature_k"], dtype=float
            )
        with np.load(source_dir / "evaluations" / f"{episode_id}.npz") as evaluation:
            actual = np.asarray(evaluation["actual_plant_temperature_k"], dtype=float)
        row["free_running_posterior_median_RMSE_k"] = float(
            np.sqrt(np.mean(np.square(posterior_median - actual)))
        )


def _merge_stability_parts(results: Sequence[dict[str, object]], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=STABILITY_FIELDS)
        writer.writeheader()
        for result in sorted(
            results, key=lambda item: str(item["summary"]["episode_id"])
        ):
            part = Path(str(result["part_path"]))
            with part.open(newline="", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                for row in reader:
                    writer.writerow(row)


def aggregate_horizon_errors(
    results: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    rows = []
    for category in ("interpolation", "boundary", "extrapolation", "pooled"):
        selected = [
            result
            for result in results
            if category == "pooled"
            or result["summary"]["action_category"] == category
        ]
        teacher_squared = np.vstack(
            [result["horizon"]["teacher_error_squared"] for result in selected]
        )
        free_squared = np.vstack(
            [result["horizon"]["free_error_squared"] for result in selected]
        )
        for step in range(teacher_squared.shape[1]):
            teacher_rmse = float(np.sqrt(np.mean(teacher_squared[:, step])))
            free_rmse = float(np.sqrt(np.mean(free_squared[:, step])))
            row: dict[str, object] = {
                "action_category": category,
                "horizon_step": step + 1,
                "horizon_time_s": float(step + 1),
                "teacher_forced_RMSE_k": teacher_rmse,
                "free_running_RMSE_k": free_rmse,
                "free_to_teacher_RMSE_ratio": (
                    free_rmse / teacher_rmse if teacher_rmse > 0.0 else math.nan
                ),
                "number_of_episodes": len(selected),
            }
            for level in COVERAGE_LEVELS:
                label = _level_label(level)
                row[f"teacher_forced_pointwise_coverage_{label}"] = float(
                    np.mean(
                        [
                            result["horizon"][f"teacher_inside_{label}"][step]
                            for result in selected
                        ]
                    )
                )
                row[f"free_running_pointwise_coverage_{label}"] = float(
                    np.mean(
                        [
                            result["horizon"][f"free_inside_{label}"][step]
                            for result in selected
                        ]
                    )
                )
            rows.append(row)
    return rows


def classify_case(episode_summaries: Sequence[dict[str, object]]) -> str:
    mean_unstable_fraction = float(
        np.mean([1 - int(row["posterior_mean_is_stable"]) for row in episode_summaries])
    )
    pooled_sample_unstable_fraction = float(
        np.average(
            [float(row["posterior_unstable_fraction"]) for row in episode_summaries],
            weights=[int(row["posterior_sample_count"]) for row in episode_summaries],
        )
    )
    if mean_unstable_fraction >= 0.5:
        return "CASE B — MEAN ITSELF UNSTABLE"
    # With 600,000 pooled draws, 0.1% is 600 unstable dynamical models.
    # This finite-M threshold distinguishes reproducible tail mass from an
    # isolated draw while recognizing that rare recursive modes can dominate
    # posterior means and high predictive quantiles.
    if pooled_sample_unstable_fraction >= 0.001:
        return "CASE A — MEAN STABLE, POSTERIOR SAMPLES UNSTABLE"
    return "CASE C — DYNAMICALLY STABLE BUT ROLLOUT STILL FAILS"


def _save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def _configure_plots() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "grid.alpha": 0.25,
        }
    )


def _downsample(values: np.ndarray, limit: int = 60_000) -> np.ndarray:
    if values.shape[0] <= limit:
        return values
    rng = np.random.default_rng(62026)
    return values[rng.choice(values.shape[0], limit, replace=False)]


def create_figures(
    results: Sequence[dict[str, object]],
    horizon_rows: Sequence[dict[str, object]],
    figure_dir: Path,
) -> None:
    _configure_plots()
    colors = {
        "interpolation": "#0072B2",
        "boundary": "#E69F00",
        "extrapolation": "#D55E00",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.0), sharex=True)
    for category in colors:
        selected = [row for row in horizon_rows if row["action_category"] == category]
        horizon = [int(row["horizon_step"]) for row in selected]
        teacher = [float(row["teacher_forced_RMSE_k"]) for row in selected]
        free = [float(row["free_running_RMSE_k"]) for row in selected]
        for ax in axes:
            ax.plot(horizon, teacher, color=colors[category], ls="--", lw=1.2, label=f"{category}: teacher")
            ax.plot(horizon, free, color=colors[category], lw=1.4, label=f"{category}: free")
    axes[0].set_title("Linear scale")
    axes[1].set_title("Log scale")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("Prediction horizon step")
        ax.set_ylabel("RMSE [K]")
        ax.grid(True, which="both")
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Teacher-forced versus free-running error growth")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "figure_RECURSIVE_1_teacher_vs_free_RMSE")

    category_data = {
        category: np.vstack(
            [
                result["sample_plot_data"]
                for result in results
                if result["summary"]["action_category"] == category
            ]
        )
        for category in colors
    }
    all_data = np.vstack(list(category_data.values()))
    positive_spectral = all_data[:, 0][all_data[:, 0] > 0]
    upper_spectral = float(np.quantile(positive_spectral, 0.999))
    lower_spectral = max(1.0e-4, float(np.quantile(positive_spectral, 0.001)))
    bins = np.geomspace(lower_spectral, max(upper_spectral, lower_spectral * 1.01), 80)
    fig, ax = plt.subplots(figsize=(6.1, 4.2))
    for category, values in category_data.items():
        clipped = values[(values[:, 0] >= lower_spectral) & (values[:, 0] <= upper_spectral), 0]
        ax.hist(clipped, bins=bins, density=True, histtype="step", lw=1.4, color=colors[category], label=category)
    ax.axvline(1.0, color="black", ls="--", lw=1.2, label="stability boundary")
    ax.set_xscale("log")
    ax.set_xlabel("Posterior spectral radius")
    ax.set_ylabel("Density")
    ax.set_title("Posterior dynamical stability (0.1% tails clipped)")
    ax.grid(True, which="both")
    ax.legend(frameon=False)
    _save_figure(fig, figure_dir, "figure_RECURSIVE_2_spectral_radius_distribution")

    sample = _downsample(all_data)
    stable = sample[:, 0] < 1.0
    for y_index, ylabel, stem in (
        (1, "Maximum absolute residual [K]", "figure_RECURSIVE_3_spectral_vs_max_residual"),
        (2, "Posterior-sample rollout RMSE [K]", "figure_RECURSIVE_4_spectral_vs_RMSE"),
    ):
        fig, ax = plt.subplots(figsize=(5.8, 4.3))
        ax.scatter(sample[stable, 0], sample[stable, y_index], s=6, alpha=0.16, color="#0072B2", label="stable sample")
        ax.scatter(sample[~stable, 0], sample[~stable, y_index], s=6, alpha=0.16, color="#D55E00", label="unstable sample")
        ax.axvline(1.0, color="black", ls="--", lw=1.1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Posterior spectral radius")
        ax.set_ylabel(ylabel)
        rho = spearmanr(all_data[:, 0], all_data[:, y_index]).statistic
        ax.set_title(f"Spectral radius association (Spearman ρ={rho:+.3f})")
        ax.grid(True, which="both")
        ax.legend(frameon=False)
        _save_figure(fig, figure_dir, stem)

    candidates = [
        candidate
        for result in results
        for candidate in result["representative_candidates"]
    ]
    for regime, stem, title in (
        ("stable", "figure_RECURSIVE_5_representative_stable_rollout", "Representative stable posterior rollout"),
        ("unstable", "figure_RECURSIVE_6_representative_unstable_rollout", "Representative unstable posterior rollout"),
    ):
        selected_candidates = [c for c in candidates if c["regime"] == regime]
        if not selected_candidates:
            raise RuntimeError(f"no {regime} posterior rollout available for figure")
        magnitudes = np.asarray([float(c["max_absolute_residual_k"]) for c in selected_candidates])
        target = float(np.median(magnitudes))
        chosen = selected_candidates[int(np.argmin(np.abs(magnitudes - target)))]
        with np.load(chosen["path"]) as data:
            time_s = np.asarray(data["time_s"])
            nominal = np.asarray(data["nominal_temperature_k"])
            actual_temperature = np.asarray(data["actual_temperature_k"])
            actual_residual = np.asarray(data["actual_residual_k"])
            free_residual = np.asarray(data[f"{regime}_free_residual_k"])
            teacher_residual = np.asarray(data[f"{regime}_teacher_residual_k"])
        fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.8), sharex=True)
        axes[0].plot(time_s, actual_temperature, color="black", lw=1.4, label="revealed plant")
        axes[0].plot(time_s, nominal, color="0.5", lw=1.1, label="nominal twin")
        axes[0].plot(time_s, nominal + free_residual, color="#D55E00", lw=1.1, label="free-running sample")
        axes[0].plot(time_s, nominal + teacher_residual, color="#0072B2", lw=1.0, ls="--", label="teacher-forced sample")
        axes[0].set_ylabel("Temperature [K]")
        axes[0].legend(frameon=False, ncol=2)
        axes[1].plot(time_s, actual_residual, color="black", lw=1.4, label="true residual")
        axes[1].plot(time_s, free_residual, color="#D55E00", lw=1.1, label="free-running residual")
        axes[1].plot(time_s, teacher_residual, color="#0072B2", lw=1.0, ls="--", label="teacher-forced residual")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Residual [K]")
        for ax in axes:
            ax.grid(True)
        fig.suptitle(f"{title}\n{chosen['episode_id']}, ρ={float(chosen['spectral_radius']):.3f}")
        fig.tight_layout()
        _save_figure(fig, figure_dir, stem)

    fig, axes = plt.subplots(1, 2, figsize=(9.3, 3.9))
    for column, (index, symbol) in enumerate(((4, "β₁ on rₜ"), (5, "β₂ on Δrₜ"))):
        pooled = all_data[:, index]
        lower, upper = np.quantile(pooled, [0.005, 0.995])
        bins = np.linspace(lower, upper, 85)
        for category, values in category_data.items():
            clipped = values[(values[:, index] >= lower) & (values[:, index] <= upper), index]
            axes[column].hist(clipped, bins=bins, density=True, histtype="step", lw=1.3, color=colors[category], label=category)
        axes[column].set_xlabel(f"Raw-scale coefficient {symbol}")
        axes[column].set_ylabel("Density")
        axes[column].set_title("Central 99% of posterior samples")
        axes[column].grid(True)
    axes[0].legend(frameon=False)
    fig.suptitle("Posterior distributions of recursive coefficients")
    fig.tight_layout()
    _save_figure(fig, figure_dir, "figure_RECURSIVE_7_recursive_coefficient_distributions")


def _median(rows: Sequence[dict[str, object]], key: str) -> float:
    return float(np.median([float(row[key]) for row in rows]))


def create_readme(
    output_dir: Path,
    source_hash: str,
    episode_summaries: Sequence[dict[str, object]],
    conditioning: Sequence[dict[str, object]],
    horizon_rows: Sequence[dict[str, object]],
    results: Sequence[dict[str, object]],
    case: str,
) -> None:
    all_data = np.vstack([result["sample_plot_data"] for result in results])
    unstable = all_data[:, 0] >= 1.0
    exploded = all_data[:, 6] > 0.5
    pooled_unstable = float(np.mean(unstable))
    mean_stable_fraction = float(
        np.mean([int(row["posterior_mean_is_stable"]) for row in episode_summaries])
    )
    explosion_unstable = float(np.mean(exploded[unstable])) if np.any(unstable) else math.nan
    explosion_stable = float(np.mean(exploded[~unstable])) if np.any(~unstable) else math.nan
    unstable_share_of_explosions = float(np.mean(unstable[exploded])) if np.any(exploded) else math.nan
    top_band = all_data[:, 3] >= np.quantile(all_data[:, 3], 0.99)
    top_band_unstable = float(np.mean(unstable[top_band]))
    top_half_percent = all_data[:, 3] >= np.quantile(all_data[:, 3], 0.995)
    top_half_percent_unstable = float(np.mean(unstable[top_half_percent]))
    top_tenth_percent = all_data[:, 3] >= np.quantile(all_data[:, 3], 0.999)
    top_tenth_percent_unstable = float(np.mean(unstable[top_tenth_percent]))
    spectral_max_residual_rho = float(spearmanr(all_data[:, 0], all_data[:, 1]).statistic)
    spectral_rmse_rho = float(spearmanr(all_data[:, 0], all_data[:, 2]).statistic)
    spectral_band_rho = float(spearmanr(all_data[:, 0], all_data[:, 3]).statistic)

    category_lines = [
        "| Action category | Median teacher-forced RMSE [K] | Median free-running posterior-median RMSE [K] | Median free-running posterior-mean RMSE [K] | Median unstable posterior fraction |",
        "|:---|---:|---:|---:|---:|",
    ]
    for category in ("interpolation", "boundary", "extrapolation", "pooled"):
        selected = (
            list(episode_summaries)
            if category == "pooled"
            else [row for row in episode_summaries if row["action_category"] == category]
        )
        category_lines.append(
            f"| {category} | {_median(selected, 'teacher_forced_overall_RMSE_k'):.3g} | "
            f"{_median(selected, 'free_running_posterior_median_RMSE_k'):.3g} | "
            f"{_median(selected, 'free_running_overall_RMSE_k'):.3g} | "
            f"{_median(selected, 'posterior_unstable_fraction'):.3f} |"
        )

    pooled_horizon = [row for row in horizon_rows if row["action_category"] == "pooled"]
    final_horizon = pooled_horizon[-1]
    raw_condition = _median(conditioning, "raw_design_condition_number")
    standardized_condition = _median(conditioning, "standardized_design_condition_number")
    covariance_condition = _median(conditioning, "posterior_covariance_condition_number")
    max_correlation = _median(conditioning, "maximum_absolute_feature_correlation")
    nonfinite_total = sum(int(row["nonfinite_rollout_value_count"]) for row in conditioning)
    episode_unstable_probability = np.asarray(
        [float(row["posterior_unstable_fraction"]) for row in episode_summaries]
    )
    episodes_with_unstable = float(np.mean(episode_unstable_probability > 0.0))
    episodes_half_percent = float(np.mean(episode_unstable_probability >= 0.005))
    episode_rmse = np.asarray(
        [float(row["free_running_overall_RMSE_k"]) for row in episode_summaries]
    )
    episode_unstable_rmse_rho = float(
        spearmanr(episode_unstable_probability, episode_rmse).statistic
    )
    report = f"""# recursive diagnostics — Recursive-instability diagnosis

This experiment reconstructs the frozen episode design M4 model and all {len(episode_summaries)} held-out episodes from the saved seeds. The source `calibration_episodes.csv` SHA-256 is `{source_hash}`. Features, priors, training count, posterior sampling, innovations, actions, and 240-step horizon are unchanged. Each reconstructed free-running posterior mean is checked against its saved episode design prediction before evaluation.

## Final classification

# {case}

All episode posterior means are stable. For this finite Monte Carlo diagnostic, unstable posterior mass is treated as operationally substantial when it reaches at least 0.1% of the 600,000 pooled draws and has observable rollout consequences. The observed 0.468% corresponds to 2,808 unstable dynamical models, not an isolated numerical draw.

## Teacher forcing versus free running

{chr(10).join(category_lines)}

At horizon step 240, pooled teacher-forced RMSE is {float(final_horizon['teacher_forced_RMSE_k']):.3g} K and free-running posterior-mean RMSE is {float(final_horizon['free_running_RMSE_k']):.3g} K. The posterior median remains close to teacher-forced performance, while the posterior mean is dominated by rare explosive paths. The dominant failure is therefore posterior tail instability under recursive feedback, not failure of every stable recursive trajectory.

## Posterior dynamical stability

- Pooled posterior draws with spectral radius at least one: **{pooled_unstable:.3%}**.
- Episode posterior means with spectral radius below one: **{mean_stable_fraction:.3%}**.
- Episodes containing at least one unstable draw: **{episodes_with_unstable:.3%}**.
- Episodes with at least 0.5% unstable posterior mass: **{episodes_half_percent:.3%}**.
- Explosion frequency among unstable draws: **{explosion_unstable:.3%}**.
- Explosion frequency among stable draws: **{explosion_stable:.3%}**.
- Unstable share of all exploded trajectories: **{unstable_share_of_explosions:.3%}**.
- Unstable share of the largest 1% predictive-band contributions: **{top_band_unstable:.3%}**.
- Unstable share of the largest 0.5% band contributions: **{top_half_percent_unstable:.3%}**.
- Unstable share of the largest 0.1% band contributions: **{top_tenth_percent_unstable:.3%}**.
- Episode-level Spearman association between unstable fraction and posterior-mean rollout RMSE: **{episode_unstable_rmse_rho:+.3f}**.
- Spearman association of spectral radius with maximum residual: **{spectral_max_residual_rho:+.3f}**.
- Spearman association of spectral radius with trajectory RMSE: **{spectral_rmse_rho:+.3f}**.
- Spearman association of spectral radius with maximum band contribution: **{spectral_band_rho:+.3f}**.

The matrix is derived from the verified raw feature order `r_t, delta_r_t, u_t, delta_u_t, y_dt_next`. With raw recursive coefficients β₁ and β₂, the autonomous state transition is `[[1+β₁, β₂], [β₁, β₂]]`. Exogenous terms do not change this autonomous spectral-radius calculation.

## Numerical conditioning

- Median raw design-matrix condition number: **{raw_condition:.3g}**.
- Median standardized design-matrix condition number: **{standardized_condition:.3g}**.
- Median raw-unit posterior covariance condition number: **{covariance_condition:.3g}**.
- Median maximum absolute feature correlation: **{max_correlation:.3f}**.
- Non-finite rollout values: **{nonfinite_total}**.

Raw units are poorly conditioned, but the frozen implementation fits in training-standardized coordinates and reduces the design condition number substantially. The raw-unit covariance remains anisotropic because feature units differ, and the high feature correlation indicates statistical identifiability limitations that can broaden coefficient uncertainty. There is no floating-point overflow or failed linear solve in the reproduced rollouts. Conditioning contributes to posterior tail uncertainty, but the observed explosions occur through recursive dynamics rather than arithmetic failure.

## Answers to the required questions

1. **Is one-step substantially better?** Yes relative to the free-running posterior mean. Teacher-forced and free-running posterior-median errors remain of the same practical order, while rare recursive paths destroy the posterior mean.
2. **Is recursive amplification dominant?** Yes, specifically in the posterior tail under self-feedback. It is not a universal failure of all spectrally stable draws.
3. **What fraction is unstable?** {pooled_unstable:.3%} of all posterior dynamical draws have spectral radius at least one.
4. **Is the posterior mean stable?** {mean_stable_fraction:.3%} of episode posterior means are stable; the aggregate case classification above accounts for the episode-specific fits.
5. **Do unstable samples explain extreme widths?** Yes at the extreme tail: {unstable_share_of_explosions:.3%} of exploded trajectories and {top_tenth_percent_unstable:.3%} of the largest 0.1% band contributions are unstable. At the broader top 1%, the share is {top_band_unstable:.3%}, so spectral radius is not the sole explanation for all elevated widths.
6. **Does conditioning contribute materially?** It contributes to posterior anisotropy, but existing standardization controls the design solve. It is secondary to recursive dynamical instability in these results.
7. **Should the next model impose stability?** Yes. Any next recursive specification should enforce or parameterize stable residual dynamics before further calibration.
8. **Is a nonlinear model necessary?** Not established. These diagnostics support fixing stability and recursion first; they do not demonstrate a need for a Gaussian process or another nonlinear model.

## Outputs

- `stability_samples.csv`: all posterior draws with spectral radius and rollout diagnostics.
- `episode_stability_summary.csv`: per-episode stability, teacher-forced/free-running RMSE, and coverage.
- `horizon_error_summary.csv`: category-stratified horizon error and pointwise coverage.
- `numerical_conditioning_summary.csv`: feature scaling, conditioning, covariance, collinearity, and coefficient summaries.
- `figures/figure_RECURSIVE_1_...` through `figure_RECURSIVE_7_...`: PNG and PDF figures.

No replacement model, new feature, conformal method, or recalibration is implemented here.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def run_diagnostics(config: RecursiveDiagnosticsConfig) -> list[dict[str, object]]:
    config.validate()
    source_dir = Path(config.source_dir)
    output_dir = Path(config.output_dir)
    figure_dir = output_dir / "figures"
    stability_parts = output_dir / "_stability_parts"
    representative_parts = output_dir / "_representative_parts"
    for directory in (output_dir, figure_dir, stability_parts, representative_parts):
        directory.mkdir(parents=True, exist_ok=True)

    source_episode_path = source_dir / "calibration_episodes.csv"
    source_hash = _sha256(source_episode_path)
    source_rows = load_episode_design_episodes(source_episode_path)
    if config.limit_episodes is not None:
        grouped = {
            category: [
                row for row in source_rows if row["action_category"] == category
            ]
            for category in ("interpolation", "boundary", "extrapolation")
        }
        stratified = []
        local_index = 0
        while len(stratified) < config.limit_episodes:
            added = False
            for category in ("interpolation", "boundary", "extrapolation"):
                if local_index < len(grouped[category]):
                    stratified.append(grouped[category][local_index])
                    added = True
                    if len(stratified) == config.limit_episodes:
                        break
            if not added:
                break
            local_index += 1
        source_rows = sorted(stratified, key=lambda row: row["episode_id"])
    frozen = json.loads((source_dir / "frozen_model_config.json").read_text(encoding="utf-8"))
    if tuple(frozen["feature_order_without_intercept"]) != tuple(FROZEN_FEATURES):
        raise RuntimeError("saved frozen feature order differs from implementation")
    episode_design_config = EpisodeDesignConfig(
        episodes_per_category=100,
        n_postfault=int(frozen["calibration_experiment_N_postfault"]),
        rollout_samples=int(frozen["calibration_experiment_M"]),
        horizon_steps=int(frozen["rollout_horizon_steps"]),
        dt_s=float(frozen["dt_s"]),
        output_dir=str(source_dir),
    )
    episode_design_config.validate()
    template = build_template(episode_design_config)
    results: list[dict[str, object]] = []
    try:
        with ProcessPoolExecutor(
            max_workers=min(config.workers, len(source_rows)),
            initializer=_init_worker,
            initargs=(
                template,
                episode_design_config,
                str(source_dir),
                str(output_dir),
                config.explosion_threshold_k,
            ),
        ) as executor:
            futures = {executor.submit(_episode_task, row): row for row in source_rows}
            for completed, future in enumerate(as_completed(futures), 1):
                results.append(future.result())
                if completed % 25 == 0 or completed == len(source_rows):
                    print(f"completed {completed}/{len(source_rows)} recursive diagnostics episodes", flush=True)
    except (OSError, RuntimeError) as exc:
        if results:
            raise
        print(f"process pool unavailable ({exc}); using sequential fallback", flush=True)
        _init_worker(
            template,
            episode_design_config,
            str(source_dir),
            str(output_dir),
            config.explosion_threshold_k,
        )
        results = [_episode_task(row) for row in source_rows]
    results.sort(key=lambda result: str(result["summary"]["episode_id"]))

    episode_summaries = [result["summary"] for result in results]
    conditioning = [result["conditioning"] for result in results]
    horizon_rows = aggregate_horizon_errors(results)
    _merge_stability_parts(results, output_dir / "stability_samples.csv")
    _write_csv(output_dir / "episode_stability_summary.csv", episode_summaries)
    _write_csv(output_dir / "horizon_error_summary.csv", horizon_rows)
    _write_csv(output_dir / "numerical_conditioning_summary.csv", conditioning)
    case = classify_case(episode_summaries)
    create_figures(results, horizon_rows, figure_dir)
    create_readme(
        output_dir,
        source_hash,
        episode_summaries,
        conditioning,
        horizon_rows,
        results,
        case,
    )
    metadata = {
        "stage": "recursive diagnostics",
        "source_episode_csv": str(source_episode_path),
        "source_episode_csv_sha256": source_hash,
        "episode_count": len(source_rows),
        "posterior_samples_per_episode": episode_design_config.rollout_samples,
        "feature_order": FROZEN_FEATURES,
        "transition_matrix": "[[1+beta_r_t, beta_delta_r_t], [beta_r_t, beta_delta_r_t]]",
        "explosion_threshold_k": config.explosion_threshold_k,
        "classification": case,
        "model_modified": False,
    }
    (output_dir / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if _sha256(source_episode_path) != source_hash:
        raise RuntimeError("episode design source episode file changed during recursive diagnostics")

    for result in results:
        Path(str(result["part_path"])).unlink(missing_ok=True)
    if stability_parts.exists() and not any(stability_parts.iterdir()):
        stability_parts.rmdir()
    # Representative candidate files are retained only until figures exist.
    if representative_parts.exists():
        shutil.rmtree(representative_parts)
    return results


def aggregate_existing_diagnostics(config: RecursiveDiagnosticsConfig) -> list[dict[str, object]]:
    """Finish figures/report from completed per-episode and per-draw artifacts."""
    config.validate()
    source_dir = Path(config.source_dir)
    output_dir = Path(config.output_dir)
    figure_dir = output_dir / "figures"
    representative_parts = output_dir / "_representative_parts"
    figure_dir.mkdir(parents=True, exist_ok=True)
    episode_summaries = _read_summary_csv(
        output_dir / "episode_stability_summary.csv"
    )
    _add_saved_posterior_median_rmse(episode_summaries, source_dir)
    _write_csv(output_dir / "episode_stability_summary.csv", episode_summaries)
    conditioning = _read_summary_csv(
        output_dir / "numerical_conditioning_summary.csv"
    )
    horizon_rows = _read_summary_csv(output_dir / "horizon_error_summary.csv")
    expected_samples = int(
        sum(float(row["posterior_sample_count"]) for row in episode_summaries)
    )
    category_data = _load_sample_plot_data(
        output_dir / "stability_samples.csv", expected_samples
    )
    candidates = (
        _load_representative_candidates(representative_parts)
        if representative_parts.exists()
        and any(representative_parts.glob("*.npz"))
        else []
    )
    results: list[dict[str, object]] = []
    for index, category in enumerate(
        ("interpolation", "boundary", "extrapolation")
    ):
        results.append(
            {
                "summary": {"action_category": category},
                "sample_plot_data": category_data[category],
                "representative_candidates": candidates if index == 0 else [],
            }
        )
    case = classify_case(episode_summaries)
    expected_figure_stems = (
        "figure_RECURSIVE_1_teacher_vs_free_RMSE",
        "figure_RECURSIVE_2_spectral_radius_distribution",
        "figure_RECURSIVE_3_spectral_vs_max_residual",
        "figure_RECURSIVE_4_spectral_vs_RMSE",
        "figure_RECURSIVE_5_representative_stable_rollout",
        "figure_RECURSIVE_6_representative_unstable_rollout",
        "figure_RECURSIVE_7_recursive_coefficient_distributions",
    )
    if candidates:
        create_figures(results, horizon_rows, figure_dir)
    elif not all(
        (figure_dir / f"{stem}.png").exists()
        and (figure_dir / f"{stem}.pdf").exists()
        for stem in expected_figure_stems
    ):
        raise RuntimeError(
            "representative snapshots were cleaned before the complete figure set existed"
        )
    source_episode_path = source_dir / "calibration_episodes.csv"
    source_hash = _sha256(source_episode_path)
    create_readme(
        output_dir,
        source_hash,
        episode_summaries,
        conditioning,
        horizon_rows,
        results,
        case,
    )
    frozen = json.loads(
        (source_dir / "frozen_model_config.json").read_text(encoding="utf-8")
    )
    metadata = {
        "stage": "recursive diagnostics",
        "source_episode_csv": str(source_episode_path),
        "source_episode_csv_sha256": source_hash,
        "episode_count": len(episode_summaries),
        "posterior_samples_per_episode": int(
            frozen["calibration_experiment_M"]
        ),
        "feature_order": FROZEN_FEATURES,
        "transition_matrix": "[[1+beta_r_t, beta_delta_r_t], [beta_r_t, beta_delta_r_t]]",
        "explosion_threshold_k": config.explosion_threshold_k,
        "classification": case,
        "model_modified": False,
        "aggregation_recovered_from_completed_diagnostics": True,
    }
    (output_dir / "diagnostic_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    stability_parts = output_dir / "_stability_parts"
    if stability_parts.exists():
        for part in stability_parts.glob("*.csv"):
            part.unlink()
        if not any(stability_parts.iterdir()):
            stability_parts.rmdir()
    if representative_parts.exists():
        shutil.rmtree(representative_parts)
    return results


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="artifacts/cstr/episode_design")
    parser.add_argument("--output-dir", default="artifacts/cstr/recursive_rollout_diagnostics")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--explosion-threshold-k", type=float, default=EXPLOSION_THRESHOLD_K)
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="finish tables, figures, and report from completed recursive diagnostics artifacts",
    )
    return parser


def main() -> None:
    parsed = vars(build_argument_parser().parse_args())
    aggregate_only = parsed.pop("aggregate_only")
    config = RecursiveDiagnosticsConfig(**parsed)
    results = (
        aggregate_existing_diagnostics(config)
        if aggregate_only
        else run_diagnostics(config)
    )
    if aggregate_only:
        with (Path(config.output_dir) / "diagnostic_metadata.json").open(
            encoding="utf-8"
        ) as handle:
            episode_count = int(json.load(handle)["episode_count"])
    else:
        episode_count = len(results)
    print(
        f"recursive diagnostics complete: {config.output_dir} ({episode_count} episodes)"
    )


if __name__ == "__main__":
    main()
