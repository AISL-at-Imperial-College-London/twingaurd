"""Compare stability-conditioned Bayesian ARX models for CSTR residuals."""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twinguard.cstr import episode_design as episode_design
from twinguard.cstr import recursive_rollout_diagnostics as recursive_diagnostics
from twinguard.discrepancy.baselines.stable_arx import (
    BayesianDirectARXModel,
    arx_training_data,
    coefficient_spectral_radii,
    draw_stability_conditioned_samples,
    free_running_residual_rollout,
    teacher_forced_features,
)


ORDERS = (1, 2, 3, 4, 5)
SELECTION_EPISODES_PER_CATEGORY = 20
MEANINGFUL_IMPROVEMENT_FRACTION = 0.05
MAX_LAG = 10


@dataclass(frozen=True)
class ArxOrderStudyConfig:
    source_dir: str = "artifacts/cstr/episode_design"
    stability_dir: str = "artifacts/cstr/stability_diagnostics"
    recursive_dir: str = "artifacts/cstr/recursive_rollout_diagnostics"
    output_dir: str = "artifacts/cstr/stable_arx_order_study"
    workers: int = 8
    limit_episodes: int | None = None
    posterior_samples: int = 2000

    def validate(self) -> None:
        if self.workers <= 0 or self.posterior_samples <= 0:
            raise ValueError("workers and posterior_samples must be positive")
        if self.limit_episodes is not None and self.limit_episodes <= 0:
            raise ValueError("limit_episodes must be positive")
        if self.limit_episodes is None and self.posterior_samples != 2000:
            raise ValueError("the full ARX order study experiment freezes M at 2000")


_TEMPLATE: object | None = None
_EPISODE_DESIGN_CONFIG: episode_design.EpisodeDesignConfig | None = None
_SOURCE_DIR: Path | None = None
_SAMPLES = 2000


def _init_worker(template: object, config: episode_design.EpisodeDesignConfig, source: str, samples: int) -> None:
    global _TEMPLATE, _EPISODE_DESIGN_CONFIG, _SOURCE_DIR, _SAMPLES
    _TEMPLATE = template
    _EPISODE_DESIGN_CONFIG = config
    _SOURCE_DIR = Path(source)
    _SAMPLES = int(samples)


def _reconstruct_observations(source_row: dict[str, str]) -> tuple[list[dict], list[dict], float]:
    """Rebuild observed CSTR histories from stored seeds and verify the saved data."""
    if _TEMPLATE is None or _EPISODE_DESIGN_CONFIG is None:
        raise RuntimeError("ARX order study worker is not initialized")
    config = _EPISODE_DESIGN_CONFIG
    seed = int(float(source_row["random_seed"]))
    rng = np.random.default_rng(seed)
    measurement_rng = np.random.default_rng(seed + 1_000_000)
    action_rng = np.random.default_rng(seed + 2_000_000)
    fault_onset = float(rng.uniform(config.fault_onset_min_s, config.fault_onset_max_s))
    fault_magnitude = float(rng.uniform(config.fault_magnitude_min, config.fault_magnitude_max))
    nominal, faulty, metadata = episode_design._configure_episode_models(
        _TEMPLATE, config, rng, fault_onset, fault_magnitude
    )
    if not np.isclose(fault_onset, float(source_row["hidden_fault_onset_s"])):
        raise RuntimeError("reconstructed fault onset differs from episode design")
    if not np.isclose(fault_magnitude, float(source_row["hidden_fault_magnitude"])):
        raise RuntimeError("reconstructed fault magnitude differs from episode design")
    for key, value in metadata.items():
        if key in source_row and not np.isclose(value, float(source_row[key])):
            raise RuntimeError(f"reconstructed {key} differs from episode design")
    pre = [
        episode_design._paired_observation(nominal, faulty, 2.0 / 60.0, measurement_rng, config.measurement_noise_k)
        for _ in range(config.pre_fault_lead_steps)
    ]
    learning = []
    for index in range(config.n_postfault):
        action_index = (index // config.training_action_block_steps) % len(episode_design.TRAINING_ACTIONS_LPS)
        learning.append(
            episode_design._paired_observation(
                nominal, faulty, episode_design.TRAINING_ACTIONS_LPS[action_index],
                measurement_rng, config.measurement_noise_k
            )
        )
    proposed = episode_design._select_action_after_fit(source_row["action_category"], action_rng)
    if not np.isclose(proposed, float(source_row["proposed_action"])):
        raise RuntimeError("reconstructed held-out action differs from episode design")
    return pre, learning, proposed


def _rmse(prediction: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(prediction) - np.asarray(truth)))))


def _episode_task(source_row: dict[str, str]) -> list[dict[str, object]]:
    if _SOURCE_DIR is None or _EPISODE_DESIGN_CONFIG is None:
        raise RuntimeError("ARX order study worker is not initialized")
    episode_id = source_row["episode_id"]
    seed = int(float(source_row["random_seed"]))
    category = source_row["action_category"]
    split_role = source_row["split_role"]
    pre, learning, proposed_action = _reconstruct_observations(source_row)
    with np.load(_SOURCE_DIR / "evaluations" / f"{episode_id}.npz") as saved:
        actual_temperature = np.asarray(saved["actual_plant_temperature_k"], dtype=float)
        actual_residual = np.asarray(saved["actual_residual_k"], dtype=float)
        nominal = np.asarray(saved["nominal_twin_temperature_k"], dtype=float)
        actions = np.asarray(saved["action_sequence_lps"], dtype=float)
    if actual_residual.size != 240 or not np.allclose(actions, proposed_action):
        raise RuntimeError("saved episode design evaluation population is not the frozen H=240 action")

    all_observed = pre + learning
    residuals = np.asarray([row["temperature_residual_k"] for row in all_observed])
    observed_actions = np.asarray([row["action_lps"] for row in all_observed])
    observed_nominal = np.asarray([row["nominal_twin_temperature_k"] for row in all_observed])
    output: list[dict[str, object]] = []
    for order in ORDERS:
        # Exactly 50 post-fault targets, with p pre-existing residuals supplying lags.
        training_residual = residuals[-(50 + order):]
        training_actions = observed_actions[-(50 + order):]
        training_nominal = observed_nominal[-(50 + order):]
        x_train, y_train = arx_training_data(
            training_residual, training_actions, training_nominal, order
        )
        if y_train.size != 50:
            raise RuntimeError("ARX training set is not exactly N_postfault=50")
        model = BayesianDirectARXModel(order)
        fit_start = time.perf_counter()
        posterior = model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - fit_start

        teacher_x = teacher_forced_features(
            actual_residual, nominal, actions, residuals[-order:],
            observed_actions[-1], order
        )
        teacher_mean = posterior.transform_features(teacher_x) @ posterior.standardized_coefficient_mean
        innovations = actual_residual - teacher_mean

        sample_start = time.perf_counter()
        coefficients, variances, radii, stability = draw_stability_conditioned_samples(
            model, posterior, _SAMPLES,
            np.random.default_rng(seed + 10_000_000 + order * 100_000)
        )
        sample_seconds = time.perf_counter() - sample_start
        rollout_rng = np.random.default_rng(seed + 20_000_000 + order * 100_000)
        sampled_innovations = rollout_rng.normal(size=(_SAMPLES, 240)) * np.sqrt(variances)[:, None]
        rollout_start = time.perf_counter()
        residual_paths = free_running_residual_rollout(
            posterior, coefficients, sampled_innovations, nominal, actions,
            residuals[-order:], observed_actions[-1], order
        )
        trajectories = nominal[None, :] + residual_paths
        posterior_mean = np.mean(trajectories, axis=0)
        posterior_median = np.median(trajectories, axis=0)
        stable_mean_coefficient = np.mean(coefficients, axis=0, keepdims=True)
        deterministic_residual = free_running_residual_rollout(
            posterior, stable_mean_coefficient, np.zeros((1, 240)), nominal, actions,
            residuals[-order:], observed_actions[-1], order
        )[0]
        deterministic_temperature = nominal + deterministic_residual
        rollout_seconds = time.perf_counter() - rollout_start
        raw_mean_radius = float(coefficient_spectral_radii(
            posterior, posterior.standardized_coefficient_mean, order
        )[0])
        stable_mean_radius = float(coefficient_spectral_radii(
            posterior, stable_mean_coefficient[0], order
        )[0])
        lower95, upper95 = np.quantile(trajectories, [0.025, 0.975], axis=0)
        output.append({
            "episode_id": episode_id,
            "action_category": category,
            "split_role": split_role,
            "random_seed": seed,
            "order": order,
            "proposed_action_lps": proposed_action,
            "training_sample_count": 50,
            "horizon_steps": 240,
            "teacher_forced_RMSE_k": _rmse(teacher_mean, actual_residual),
            "teacher_forced_MAE_k": float(np.mean(np.abs(innovations))),
            "teacher_forced_innovation_mean_k": float(np.mean(innovations)),
            "teacher_forced_innovation_median_k": float(np.median(innovations)),
            "posterior_mean_trajectory_RMSE_k": _rmse(posterior_mean, actual_temperature),
            "posterior_median_trajectory_RMSE_k": _rmse(posterior_median, actual_temperature),
            "deterministic_stable_coefficient_mean_RMSE_k": _rmse(deterministic_temperature, actual_temperature),
            "pointwise_95_coverage_diagnostic": float(np.mean((actual_temperature >= lower95) & (actual_temperature <= upper95))),
            "pointwise_95_mean_width_k_diagnostic": float(np.mean(upper95 - lower95)),
            "raw_posterior_mean_spectral_radius": raw_mean_radius,
            "raw_posterior_mean_is_stable": int(raw_mean_radius < 1.0),
            "stable_conditioned_mean_spectral_radius": stable_mean_radius,
            "fit_seconds": fit_seconds,
            "stability_sampling_seconds": sample_seconds,
            "rollout_seconds": rollout_seconds,
            "total_seconds": fit_seconds + sample_seconds + rollout_seconds,
            **stability,
            "spectral_radius_mean": float(np.mean(radii)),
            "spectral_radius_median": float(np.median(radii)),
            "spectral_radius_q05": float(np.quantile(radii, 0.05)),
            "spectral_radius_q95": float(np.quantile(radii, 0.95)),
            "spectral_radius_maximum": float(np.max(radii)),
            "innovation": innovations,
            "posterior_mean_error": posterior_mean - actual_temperature,
            "posterior_median_error": posterior_median - actual_temperature,
            "deterministic_error": deterministic_temperature - actual_temperature,
        })
    return output


def _acf_within_episodes(arrays: Sequence[np.ndarray], max_lag: int = MAX_LAG) -> np.ndarray:
    all_values = np.concatenate(arrays)
    center = float(np.mean(all_values))
    denominator = float(np.sum(np.square(all_values - center)))
    if denominator <= np.finfo(float).tiny:
        return np.concatenate(([1.0], np.zeros(max_lag)))
    acf = np.ones(max_lag + 1)
    for lag in range(1, max_lag + 1):
        numerator = sum(
            float(np.sum((values[:-lag] - center) * (values[lag:] - center)))
            for values in arrays
        )
        acf[lag] = numerator / denominator
    return acf


def _pacf_from_acf(acf: np.ndarray) -> np.ndarray:
    pacf = np.ones_like(acf)
    for lag in range(1, acf.size):
        matrix = np.asarray([[acf[abs(i-j)] for j in range(lag)] for i in range(lag)])
        try:
            pacf[lag] = np.linalg.solve(matrix, acf[1:lag+1])[-1]
        except np.linalg.LinAlgError:
            pacf[lag] = np.nan
    return pacf


def _groups(results: Sequence[dict[str, object]]) -> Iterable[tuple[int, str, str, list[dict[str, object]]]]:
    for order in ORDERS:
        for split in ("model_selection", "final_assessment", "all"):
            split_rows = [row for row in results if int(row["order"]) == order and (split == "all" or row["split_role"] == split)]
            if not split_rows:
                continue
            for category in ("interpolation", "boundary", "extrapolation", "pooled"):
                rows = split_rows if category == "pooled" else [row for row in split_rows if row["action_category"] == category]
                if rows:
                    yield order, split, category, rows


def build_innovation_summary(results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for order, split, category, rows in _groups(results):
        values = np.concatenate([np.asarray(row["innovation"]) for row in rows])
        acf = _acf_within_episodes([np.asarray(row["innovation"]) for row in rows])
        pacf = _pacf_from_acf(acf)
        record: dict[str, object] = {
            "order": order, "split_role": split, "action_category": category,
            "number_of_episodes": len(rows), "number_of_innovations": values.size,
            "innovation_mean_k": float(np.mean(values)),
            "innovation_median_k": float(np.median(values)),
            "innovation_RMSE_k": _rmse(values, np.zeros_like(values)),
            "innovation_MAE_k": float(np.mean(np.abs(values))),
            "maximum_absolute_ACF_lags_1_10": float(np.nanmax(np.abs(acf[1:]))),
            "maximum_absolute_PACF_lags_1_10": float(np.nanmax(np.abs(pacf[1:]))),
        }
        for lag in range(1, MAX_LAG + 1):
            record[f"ACF_lag_{lag}"] = float(acf[lag])
            record[f"PACF_lag_{lag}"] = float(pacf[lag])
        output.append(record)
    return output


def build_horizon_summary(results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for order, split, category, rows in _groups(results):
        for method, key in (
            ("teacher_forced_prediction", "innovation"),
            ("posterior_mean", "posterior_mean_error"),
            ("posterior_median", "posterior_median_error"),
            ("deterministic_stable_coefficient_mean", "deterministic_error"),
        ):
            errors = np.vstack([np.asarray(row[key]) for row in rows])
            if method == "teacher_forced_prediction":
                # Stored innovations are truth - prediction; this table uses the
                # common prediction - truth sign convention for bias.
                errors = -errors
            for horizon in range(errors.shape[1]):
                output.append({
                    "order": order, "split_role": split, "action_category": category,
                    "prediction_method": method, "horizon_step": horizon + 1,
                    "number_of_episodes": len(rows),
                    "RMSE_k": _rmse(errors[:, horizon], np.zeros(len(rows))),
                    "bias_prediction_minus_truth_k": float(np.mean(errors[:, horizon])),
                })
    return output


def build_order_comparison(
    results: Sequence[dict[str, object]], innovation_summary: Sequence[dict[str, object]]
) -> list[dict[str, object]]:
    output = []
    for order, split, category, rows in _groups(results):
        innovation = next(row for row in innovation_summary if int(row["order"]) == order and row["split_role"] == split and row["action_category"] == category)
        record: dict[str, object] = {
            "model": f"stable_direct_ARX_{order}", "order": order,
            "split_role": split, "action_category": category,
            "number_of_episodes": len(rows),
        }
        for name in (
            "teacher_forced_RMSE_k", "teacher_forced_MAE_k",
            "teacher_forced_innovation_mean_k", "posterior_mean_trajectory_RMSE_k",
            "posterior_median_trajectory_RMSE_k", "deterministic_stable_coefficient_mean_RMSE_k",
            "raw_unstable_fraction", "stable_draw_acceptance_rate", "fit_seconds",
            "stability_sampling_seconds", "rollout_seconds", "total_seconds",
            "pointwise_95_coverage_diagnostic", "pointwise_95_mean_width_k_diagnostic",
        ):
            values = np.asarray([float(row[name]) for row in rows])
            record[f"median_{name}"] = float(np.median(values))
            record[f"mean_{name}"] = float(np.mean(values))
        record["pooled_innovation_mean_k"] = innovation["innovation_mean_k"]
        record["maximum_absolute_innovation_ACF_lags_1_10"] = innovation["maximum_absolute_ACF_lags_1_10"]
        output.append(record)
    return output


def select_order(comparison: Sequence[dict[str, object]]) -> tuple[int, dict[int, float]]:
    """Freeze p using only the declared model-selection episodes."""
    candidates = [row for row in comparison if row["split_role"] == "model_selection" and row["action_category"] == "pooled"]
    if len(candidates) != len(ORDERS):
        raise RuntimeError("model-selection split does not contain every AR order")
    scores = {int(row["order"]): float(row["median_posterior_mean_trajectory_RMSE_k"]) for row in candidates}
    acceptance = {int(row["order"]): float(row["median_stable_draw_acceptance_rate"]) for row in candidates}
    feasible = [order for order in ORDERS if acceptance[order] >= 0.01]
    if not feasible:
        raise RuntimeError("stable posterior acceptance is computationally problematic for every order")
    best = min(scores[order] for order in feasible)
    selected = min(order for order in feasible if scores[order] <= (1.0 + MEANINGFUL_IMPROVEMENT_FRACTION) * best)
    return selected, scores


def _plain_rows(results: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    return [{key: value for key, value in row.items() if not isinstance(value, np.ndarray)} for row in results]


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _selection_roles(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    category_count: dict[str, int] = {}
    output = []
    for row in rows:
        item = dict(row)
        category = row["action_category"]
        index = category_count.get(category, 0)
        category_count[category] = index + 1
        item["split_role"] = "model_selection" if index < SELECTION_EPISODES_PER_CATEGORY else "final_assessment"
        output.append(item)
    return output


def _previous_comparison(selected_results: Sequence[dict[str, object]], stability_dir: Path, recursive_dir: Path, source_dir: Path) -> list[dict[str, object]]:
    stability_diagnostics_episode = {row["episode_id"]: row for row in _read_csv(stability_dir / "stable_posterior_episode_results.csv")}
    stability_diagnostics_innovations = _read_csv(stability_dir / "innovation_samples.csv")
    recursive_diagnostics_stability = {row["episode_id"]: row for row in _read_csv(recursive_dir / "episode_stability_summary.csv")}
    episode_design_episode = {row["episode_id"]: row for row in _read_csv(source_dir / "calibration_episodes.csv")}
    output = []
    final_rows = [row for row in selected_results if row["split_role"] == "final_assessment"]
    ids = {str(row["episode_id"]) for row in final_rows}
    for category in ("interpolation", "boundary", "extrapolation", "pooled"):
        current = final_rows if category == "pooled" else [row for row in final_rows if row["action_category"] == category]
        category_ids = ids if category == "pooled" else {str(row["episode_id"]) for row in final_rows if row["action_category"] == category}
        current_innovations = [np.asarray(row["innovation"]) for row in current]
        current_acf = _acf_within_episodes(current_innovations)
        output.append({
            "model": f"selected_stable_direct_ARX_{int(current[0]['order'])}",
            "order": int(current[0]["order"]), "split_role": "final_assessment",
            "action_category": category, "number_of_episodes": len(current),
            "teacher_forced_RMSE_k": float(np.median([float(row["teacher_forced_RMSE_k"]) for row in current])),
            "posterior_mean_trajectory_RMSE_k": float(np.median([float(row["posterior_mean_trajectory_RMSE_k"]) for row in current])),
            "posterior_median_trajectory_RMSE_k": float(np.median([float(row["posterior_median_trajectory_RMSE_k"]) for row in current])),
            "innovation_mean_k": float(np.mean(np.concatenate(current_innovations))),
            "maximum_absolute_innovation_ACF_lags_1_10": float(np.max(np.abs(current_acf[1:]))),
            "raw_unstable_fraction": float(np.mean([float(row["raw_unstable_fraction"]) for row in current])),
            "median_runtime_seconds_lower_bound": float(np.median([float(row["total_seconds"]) for row in current])),
            "runtime_note": "complete ARX fit+stability-rejection+rollout runtime",
        })
        innovations_by_episode: dict[str, list[float]] = {episode: [] for episode in category_ids}
        for row in stability_diagnostics_innovations:
            if row["episode_id"] in innovations_by_episode:
                innovations_by_episode[row["episode_id"]].append(float(row["innovation_k"]))
        arrays = [np.asarray(values) for values in innovations_by_episode.values() if values]
        acf = _acf_within_episodes(arrays)
        mean_rmse = np.asarray([float(stability_diagnostics_episode[e]["after_posterior_mean_RMSE_k"]) for e in category_ids])
        median_rmse = np.asarray([float(stability_diagnostics_episode[e]["after_posterior_median_RMSE_k"]) for e in category_ids])
        teacher_rmse = np.asarray([_rmse(values, np.zeros(len(values))) for values in arrays])
        runtimes = np.asarray([sum(float(episode_design_episode[e][key]) for key in ("posterior_fit_time", "posterior_sampling_time", "rollout_time")) for e in category_ids])
        unstable = np.asarray([float(recursive_diagnostics_stability[e]["posterior_unstable_fraction"]) for e in category_ids])
        all_innovations = np.concatenate(arrays)
        output.append({
            "model": "previous_stability_conditioned_M4", "order": "",
            "split_role": "final_assessment", "action_category": category,
            "number_of_episodes": len(category_ids),
            "teacher_forced_RMSE_k": float(np.median(teacher_rmse)),
            "posterior_mean_trajectory_RMSE_k": float(np.median(mean_rmse)),
            "posterior_median_trajectory_RMSE_k": float(np.median(median_rmse)),
            "innovation_mean_k": float(np.mean(all_innovations)),
            "maximum_absolute_innovation_ACF_lags_1_10": float(np.max(np.abs(acf[1:]))),
            "raw_unstable_fraction": float(np.mean(unstable)),
            "median_runtime_seconds_lower_bound": float(np.median(runtimes)),
            "runtime_note": "episode design fit+sampling+rollout only; stability diagnostics rejection overhead was not logged",
        })
    return output


def create_figures(comparison: Sequence[dict[str, object]], stability: Sequence[dict[str, object]], horizon: Sequence[dict[str, object]], innovations: Sequence[dict[str, object]], selected: int, previous: Sequence[dict[str, object]], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 300, "font.size": 9})
    def save(fig: plt.Figure, name: str) -> None:
        fig.tight_layout(); fig.savefig(output / f"{name}.png"); fig.savefig(output / f"{name}.pdf"); plt.close(fig)
    selection = [row for row in comparison if row["split_role"] == "model_selection" and row["action_category"] == "pooled"]
    fig, ax = plt.subplots(figsize=(6.4, 4)); x = np.asarray(ORDERS)
    ax.plot(x, [row["median_teacher_forced_RMSE_k"] for row in selection], "o-", label="teacher-forced")
    ax.plot(x, [row["median_posterior_mean_trajectory_RMSE_k"] for row in selection], "o-", label="free posterior mean")
    ax.plot(x, [row["median_posterior_median_trajectory_RMSE_k"] for row in selection], "o-", label="free posterior median")
    ax.axvline(selected, color="black", ls="--", label=f"selected p={selected}"); ax.set(xlabel="AR order p", ylabel="Median held-out RMSE (K)"); ax.legend(); ax.grid(alpha=.25); save(fig, "figure_B7B_1_order_multistep_performance")
    fig, ax = plt.subplots(figsize=(6.4, 4));
    for order in ORDERS:
        values = [float(row["spectral_radius_median"]) for row in stability if int(row["order"]) == order]
        ax.boxplot(values, positions=[order], widths=.55, showfliers=False)
    ax.axhline(1, color="red", ls="--", label="stability boundary"); ax.set(xlabel="AR order p", ylabel="Median stable-draw spectral radius"); ax.legend(); ax.grid(alpha=.25); save(fig, "figure_B7B_2_spectral_radius")
    fig, ax = plt.subplots(figsize=(6.4, 4)); ax.plot(x, [row["mean_raw_unstable_fraction"] for row in selection], "o-", label="raw unstable fraction"); ax.plot(x, [row["mean_stable_draw_acceptance_rate"] for row in selection], "s-", label="rejection acceptance"); ax.set(xlabel="AR order p", ylabel="Fraction", ylim=(0, 1.02)); ax.legend(); ax.grid(alpha=.25); save(fig, "figure_B7B_3_stability_acceptance")
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), sharey=True)
    for order in ORDERS:
        row = next(item for item in innovations if int(item["order"]) == order and item["split_role"] == "model_selection" and item["action_category"] == "pooled")
        axes[0].plot(range(1,11), [row[f"ACF_lag_{lag}"] for lag in range(1,11)], marker="o", label=f"p={order}")
        axes[1].plot(range(1,11), [row[f"PACF_lag_{lag}"] for lag in range(1,11)], marker="o", label=f"p={order}")
    axes[0].set_title("Innovation ACF"); axes[1].set_title("Innovation PACF")
    for ax in axes: ax.axhline(0,color="black",lw=.7); ax.set_xlabel("Lag"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Correlation"); axes[1].legend(ncol=2); save(fig, "figure_B7B_4_innovation_acf_pacf")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, category in zip(axes, ("interpolation", "boundary", "extrapolation")):
        for order in ORDERS:
            rows = [row for row in horizon if int(row["order"]) == order and row["split_role"] == "final_assessment" and row["action_category"] == category and row["prediction_method"] == "posterior_mean"]
            ax.plot([r["horizon_step"] for r in rows], [r["RMSE_k"] for r in rows], lw=1, label=f"p={order}")
        ax.set_title(category); ax.set_xlabel("Horizon step"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Recursive RMSE (K)"); axes[-1].legend(ncol=2); save(fig, "figure_B7B_5_horizon_rmse")
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, category in zip(axes, ("interpolation", "boundary", "extrapolation")):
        rows = [row for row in horizon if int(row["order"]) == selected and row["split_role"] == "final_assessment" and row["action_category"] == category and row["prediction_method"] == "posterior_mean"]
        ax.plot([r["horizon_step"] for r in rows], [r["bias_prediction_minus_truth_k"] for r in rows]); ax.axhline(0,color="black",lw=.7); ax.set_title(category); ax.set_xlabel("Horizon step"); ax.grid(alpha=.25)
    axes[0].set_ylabel("Prediction bias (K)"); save(fig, "figure_B7B_6_selected_horizon_bias")
    selected_final = [row for row in comparison if int(row["order"]) == selected and row["split_role"] == "final_assessment" and row["action_category"] in ("interpolation", "boundary", "extrapolation", "pooled")]
    categories = ["interpolation", "boundary", "extrapolation", "pooled"]
    fig, ax = plt.subplots(figsize=(7, 4)); positions = np.arange(4); width=.36
    ax.bar(positions-width/2, [next(r for r in previous if r["action_category"]==c and r["model"]=="previous_stability_conditioned_M4")["posterior_mean_trajectory_RMSE_k"] for c in categories], width, label="stability diagnostics")
    ax.bar(positions+width/2, [next(r for r in selected_final if r["action_category"]==c)["median_posterior_mean_trajectory_RMSE_k"] for c in categories], width, label=f"ARX({selected})")
    ax.set_xticks(positions, categories, rotation=20); ax.set_ylabel("Median free-running mean RMSE (K)"); ax.legend(); ax.grid(axis="y", alpha=.25); save(fig, "figure_B7B_7_selected_vs_B7A")
    fig, ax = plt.subplots(figsize=(6.4, 4)); ax.plot(x, [row["median_fit_seconds"] for row in selection], "o-", label="fit"); ax.plot(x, [row["median_stability_sampling_seconds"] for row in selection], "o-", label="stable sampling"); ax.plot(x, [row["median_rollout_seconds"] for row in selection], "o-", label="rollout"); ax.set(xlabel="AR order p", ylabel="Median seconds per episode"); ax.legend(); ax.grid(alpha=.25); save(fig, "figure_B7B_8_runtime")


def verdict(selected: int, comparison: Sequence[dict[str, object]], innovations: Sequence[dict[str, object]]) -> str:
    row = next(r for r in comparison if int(r["order"]) == selected and r["split_role"] == "final_assessment" and r["action_category"] == "pooled")
    interp = next(r for r in comparison if int(r["order"]) == selected and r["split_role"] == "final_assessment" and r["action_category"] == "interpolation")
    innovation = next(r for r in innovations if int(r["order"]) == selected and r["split_role"] == "final_assessment" and r["action_category"] == "pooled")
    accurate = float(row["median_posterior_mean_trajectory_RMSE_k"]) <= 5.0 and float(interp["median_posterior_mean_trajectory_RMSE_k"]) <= 2.0
    low_bias = abs(float(row["pooled_innovation_mean_k"])) <= 0.5
    low_dependence = float(innovation["maximum_absolute_ACF_lags_1_10"]) <= 0.3
    if accurate and low_bias and low_dependence:
        return "MEAN MODEL ACCEPTED"
    if selected == max(ORDERS):
        p4 = next(r for r in comparison if int(r["order"]) == 4 and r["split_role"] == "model_selection" and r["action_category"] == "pooled")
        p5 = next(r for r in comparison if int(r["order"]) == 5 and r["split_role"] == "model_selection" and r["action_category"] == "pooled")
        improvement = 1.0 - float(p5["median_posterior_mean_trajectory_RMSE_k"]) / float(p4["median_posterior_mean_trajectory_RMSE_k"])
        if improvement >= MEANINGFUL_IMPROVEMENT_FRACTION and not low_dependence:
            return "INCREASE TEMPORAL ORDER"
    return "MEAN MODEL STRUCTURE INADEQUATE"


def create_report(output: Path, selected: int, scores: dict[int, float], comparison: Sequence[dict[str, object]], innovations: Sequence[dict[str, object]], previous: Sequence[dict[str, object]], final_verdict: str) -> None:
    selected_row = next(r for r in comparison if int(r["order"]) == selected and r["split_role"] == "final_assessment" and r["action_category"] == "pooled")
    selected_innov = next(r for r in innovations if int(r["order"]) == selected and r["split_role"] == "final_assessment" and r["action_category"] == "pooled")
    highest_row = next(r for r in comparison if int(r["order"]) == max(ORDERS) and r["split_role"] == "final_assessment" and r["action_category"] == "pooled")
    highest_innov = next(r for r in innovations if int(r["order"]) == max(ORDERS) and r["split_role"] == "final_assessment" and r["action_category"] == "pooled")
    old = next(r for r in previous if r["action_category"] == "pooled" and r["model"] == "previous_stability_conditioned_M4")
    score_text = ", ".join(f"p={p}: {score:.3f} K" for p, score in scores.items())
    report = f"""# ARX order study — Selected mean model

## Central result

The predeclared model-selection split selected **ARX({selected})**. The selection-only median free-running posterior-mean RMSE values were {score_text}. The rule selected the smallest feasible order within {MEANINGFUL_IMPROVEMENT_FRACTION:.0%} of the best selection score. None of the 240 final-assessment episodes entered this decision.

On the final-assessment split (80 episodes per action category), ARX({selected}) obtained median posterior-mean trajectory RMSE **{float(selected_row['median_posterior_mean_trajectory_RMSE_k']):.3f} K**, median posterior-median trajectory RMSE **{float(selected_row['median_posterior_median_trajectory_RMSE_k']):.3f} K**, and median deterministic stable-coefficient-mean RMSE **{float(selected_row['median_deterministic_stable_coefficient_mean_RMSE_k']):.3f} K**.

## Comparison with stability diagnostics

For the identical final-assessment episode IDs, stability diagnostics's corresponding median posterior-mean and posterior-median RMSE values were **{float(old['posterior_mean_trajectory_RMSE_k']):.3f} K** and **{float(old['posterior_median_trajectory_RMSE_k']):.3f} K**. Teacher-forced innovation RMSE changed from **{float(old['teacher_forced_RMSE_k']):.3f} K** to **{float(selected_row['median_teacher_forced_RMSE_k']):.3f} K**. The selected model's pooled innovation mean is **{float(selected_innov['innovation_mean_k']):+.3f} K** and maximum absolute within-episode ACF over lags 1–10 is **{float(selected_innov['maximum_absolute_ACF_lags_1_10']):.3f}**.

## Stability and computation

The ordinary posterior was fitted before conditioning. Coefficient draws with companion-matrix spectral radius at least one were rejected, never clipped. The final-assessment median raw unstable fraction is **{float(selected_row['median_raw_unstable_fraction']):.3f}** and median stable-draw acceptance rate is **{float(selected_row['median_stable_draw_acceptance_rate']):.3f}**. The median fit, stability-sampling, and rollout times are **{float(selected_row['median_fit_seconds']):.4f} s**, **{float(selected_row['median_stability_sampling_seconds']):.4f} s**, and **{float(selected_row['median_rollout_seconds']):.4f} s** per episode.

## Interpretation

Coverage and 95% width are retained only as diagnostics because stability diagnostics already rejected the constant-variance innovation assumption. They were not used for model selection or the verdict. The comparison is about conditional-mean accuracy, horizon bias, dependence, stability, and runtime. A poor result therefore diagnoses the direct linear ARX mean structure before any nonlinear model is considered.

The predeclared acceptance gates were: final-assessment pooled median posterior-mean RMSE at most 5 K, interpolation median RMSE at most 2 K, absolute pooled innovation bias at most 0.5 K, and maximum absolute within-episode innovation ACF over lags 1–10 at most 0.3. Failure at p=5 is classified as `INCREASE TEMPORAL ORDER` only when p=5 improves selection RMSE by at least 5% over p=4 and dependence remains excessive; otherwise the tested linear mean structure is inadequate.

The diagnosis is structural within the tested feature set, not a request for an automatically more complex nonlinear model. Increasing order from p={selected} to p={max(ORDERS)} lowers final-assessment median teacher-forced RMSE from **{float(selected_row['median_teacher_forced_RMSE_k']):.3f} K** to **{float(highest_row['median_teacher_forced_RMSE_k']):.3f} K**, but the free-running posterior-mean RMSE changes only from **{float(selected_row['median_posterior_mean_trajectory_RMSE_k']):.3f} K** to **{float(highest_row['median_posterior_mean_trajectory_RMSE_k']):.3f} K** and maximum innovation ACF remains **{float(highest_innov['maximum_absolute_ACF_lags_1_10']):.3f}**. The horizon plots show systematic, action-regime-dependent degradation shared by all five orders. Thus extra residual lags fit local transitions better but do not supply the evolving state/action information needed for accurate long-horizon counterfactual means. That missing conditioning should be diagnosed before nonlinear modelling is proposed.

## Answer

**{final_verdict}**

This is the answer to whether a stability-constrained higher-order ARX model provides a sufficiently accurate and stable mean model for counterfactual residual rollout under the predeclared criteria in the runner.
"""
    (output / "SELECTED_MODEL.md").write_text(report, encoding="utf-8")


def run_experiment(config: ArxOrderStudyConfig) -> int:
    config.validate()
    source = Path(config.source_dir); output = Path(config.output_dir); output.mkdir(parents=True, exist_ok=True)
    rows = _selection_roles(recursive_diagnostics.load_episode_design_episodes(source / "calibration_episodes.csv"))
    if config.limit_episodes is not None:
        rows = rows[:config.limit_episodes]
    episode_design_config = episode_design.EpisodeDesignConfig(output_dir=config.source_dir)
    template = episode_design.build_template(episode_design_config)
    with ProcessPoolExecutor(
        max_workers=min(config.workers, len(rows)), initializer=_init_worker,
        initargs=(template, episode_design_config, str(source), config.posterior_samples)
    ) as pool:
        nested = list(pool.map(_episode_task, rows, chunksize=1))
    results = [item for group in nested for item in group]
    innovations = build_innovation_summary(results)
    horizon = build_horizon_summary(results)
    comparison = build_order_comparison(results, innovations)
    selected, scores = select_order(comparison)
    selected_results = [row for row in results if int(row["order"]) == selected]
    stability_fields = (
        "episode_id", "action_category", "split_role", "random_seed", "order",
        "raw_sample_count", "raw_unstable_count", "raw_unstable_fraction",
        "posterior_draws_attempted", "stable_draws_accepted", "stable_draw_acceptance_rate",
        "raw_posterior_mean_spectral_radius", "raw_posterior_mean_is_stable",
        "stable_conditioned_mean_spectral_radius", "spectral_radius_mean",
        "spectral_radius_median", "spectral_radius_q05", "spectral_radius_q95",
        "spectral_radius_maximum",
    )
    stability = [{key: row[key] for key in stability_fields} for row in results]
    previous = _previous_comparison(selected_results, Path(config.stability_dir), Path(config.recursive_dir), source)
    final_verdict = verdict(selected, comparison, innovations)
    for row in comparison:
        row["selected_order"] = int(int(row["order"]) == selected)
        row["selection_threshold_fraction"] = MEANINGFUL_IMPROVEMENT_FRACTION
    _write_csv(output / "arx_order_comparison.csv", comparison)
    _write_csv(output / "selected_model_results.csv", _plain_rows(selected_results))
    _write_csv(output / "posterior_stability_summary.csv", stability)
    _write_csv(output / "horizon_error_summary.csv", horizon)
    _write_csv(output / "innovation_summary.csv", innovations)
    _write_csv(output / "selected_vs_stability_diagnostics_comparison.csv", previous)
    create_figures(comparison, stability, horizon, innovations, selected, previous, output / "figures")
    create_report(output, selected, scores, comparison, innovations, previous, final_verdict)
    metadata = {
        "stage": "ARX order study", "episode_count": len(rows), "orders": list(ORDERS),
        "N_postfault": 50, "horizon_steps": 240, "posterior_samples": config.posterior_samples,
        "selection_episodes_per_category": SELECTION_EPISODES_PER_CATEGORY,
        "final_assessment_episodes_per_category": 80 if config.limit_episodes is None else None,
        "selection_rule": "smallest feasible p within 5% of best selection-split median free-running posterior-mean RMSE",
        "mean_model_acceptance_gates": {
            "pooled_final_median_posterior_mean_RMSE_k_max": 5.0,
            "interpolation_final_median_posterior_mean_RMSE_k_max": 2.0,
            "absolute_pooled_innovation_bias_k_max": 0.5,
            "maximum_absolute_innovation_ACF_lags_1_10_max": 0.3
        },
        "selected_order": selected, "verdict": final_verdict,
        "stability_conditioning": "ordinary posterior followed by rejection at companion spectral radius >= 1; no clipping",
        "variance_calibrated": False,
    }
    (output / "experiment_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return len(rows)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", default="artifacts/cstr/episode_design")
    parser.add_argument("--stability-dir", default="artifacts/cstr/stability_diagnostics")
    parser.add_argument("--recursive-dir", default="artifacts/cstr/recursive_rollout_diagnostics")
    parser.add_argument("--output-dir", default="artifacts/cstr/stable_arx_order_study")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--limit-episodes", type=int)
    parser.add_argument("--posterior-samples", type=int, default=2000)
    return parser


def main() -> None:
    config = ArxOrderStudyConfig(**vars(build_argument_parser().parse_args()))
    count = run_experiment(config)
    print(f"ARX order study complete: {config.output_dir} ({count} episodes)")


if __name__ == "__main__":
    main()
