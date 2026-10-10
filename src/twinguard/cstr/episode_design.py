"""Build distinct CSTR histories for model fitting, calibration, and evaluation.

Operational inputs exclude hidden faults and future physical outcomes.
Seed blocks remain disjoint across population roles.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from twinguard.discrepancy.baselines.bayesian_linear_residual import (
    MODEL_FEATURES,
    BayesianResidualDynamicsModel,
    assert_online_feature_names,
)
from twinguard.discrepancy.baselines.bayesian_residual_rollout import recursive_residual_dynamics_rollout
from twinguard.plants.cstr import CSTRSimulation, FaultConfig, FaultManager


COVERAGE_LEVELS = (0.50, 0.80, 0.90, 0.95, 0.99)
FROZEN_MODEL_NAME = "M4_full_bayesian"
FROZEN_FEATURES = MODEL_FEATURES[FROZEN_MODEL_NAME]
TRAINING_ACTIONS_LPS = (0.023, 0.026, 0.032)
PREDICTION_KEYS_WITHOUT_TRUTH = frozenset(
    {
        "time_s",
        "action_sequence_lps",
        "nominal_twin_temperature_k",
        "posterior_mean_temperature_k",
        "posterior_median_temperature_k",
        "pointwise_50_lower_k",
        "pointwise_50_upper_k",
        "pointwise_80_lower_k",
        "pointwise_80_upper_k",
        "pointwise_90_lower_k",
        "pointwise_90_upper_k",
        "pointwise_95_lower_k",
        "pointwise_95_upper_k",
        "pointwise_99_lower_k",
        "pointwise_99_upper_k",
        "simultaneous_50_lower_k",
        "simultaneous_50_upper_k",
        "simultaneous_80_lower_k",
        "simultaneous_80_upper_k",
        "simultaneous_90_lower_k",
        "simultaneous_90_upper_k",
        "simultaneous_95_lower_k",
        "simultaneous_95_upper_k",
        "simultaneous_99_lower_k",
        "simultaneous_99_upper_k",
    }
)


@dataclass(frozen=True)
class EpisodeDesignConfig:
    seed: int = 42_000
    episodes_per_category: int = 100
    n_postfault: int = 50
    rollout_samples: int = 2000
    horizon_steps: int = 240
    dt_s: float = 1.0
    pre_fault_lead_steps: int = 20
    template_time_s: float = 1200.0
    fault_onset_min_s: float = 1260.0
    fault_onset_max_s: float = 1360.0
    fault_magnitude_min: float = 0.25
    fault_magnitude_max: float = 0.55
    fault_time_constant_s: float = 1.0
    measurement_noise_k: float = 0.10
    temperature_perturbation_sd_k: float = 0.20
    volume_perturbation_sd_l: float = 0.03
    concentration_perturbation_sd_fraction: float = 0.01
    training_action_block_steps: int = 5
    workers: int = 8
    output_dir: str = "artifacts/cstr/episode_design"

    @property
    def categories(self) -> tuple[str, ...]:
        return ("interpolation", "boundary", "extrapolation")

    @property
    def episode_count(self) -> int:
        return self.episodes_per_category * len(self.categories)

    def validate(self) -> None:
        if self.n_postfault != 50:
            raise ValueError("episode design freezes N_postfault at 50")
        if self.rollout_samples != 2000:
            raise ValueError("episode design freezes posterior trajectories at M=2000")
        if self.horizon_steps != 240 or self.dt_s != 1.0:
            raise ValueError(" B horizon and dt must remain 240 and 1 s")
        if self.episodes_per_category <= 0 or self.workers <= 0:
            raise ValueError("episode and worker counts must be positive")
        if not 0.0 < self.fault_magnitude_min < self.fault_magnitude_max < 0.95:
            raise ValueError("invalid hidden fault-magnitude range")
        if self.training_action_block_steps <= 0:
            raise ValueError("training_action_block_steps must be positive")
        assert_online_feature_names(FROZEN_FEATURES)


@dataclass(frozen=True)
class SimultaneousBands:
    median: np.ndarray
    robust_scale: np.ndarray
    lower: dict[float, np.ndarray]
    upper: dict[float, np.ndarray]
    standardized_maximum_quantiles: dict[float, float]


_WORKER_TEMPLATE: CSTRSimulation | None = None
_WORKER_CONFIG: EpisodeDesignConfig | None = None
_WORKER_OUTPUT: str | None = None


def frozen_model_configuration(config: EpisodeDesignConfig) -> dict:
    def source_hash(path: str) -> str:
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    return {
        "stage": "episode design posterior calibration",
        "model_frozen": True,
        "model_name": FROZEN_MODEL_NAME,
        "model_class": "BayesianResidualDynamicsModel",
        "feature_order_without_intercept": list(FROZEN_FEATURES),
        "implicit_intercept": True,
        "prediction_target": "delta_r_next",
        "residual_definition": "plant measured temperature - nominal twin temperature",
        "prior": {
            "sigma_squared": "InverseGamma(alpha=2.0, beta=0.25)",
            "beta_given_sigma_squared": "Normal(0, sigma_squared / 0.01 * I)",
            "prior_precision": 0.01,
            "prior_alpha": 2.0,
            "prior_beta": 0.25,
        },
        "feature_standardization": "training-only mean and population standard deviation; constant columns use scale 1",
        "noise_model": "Gaussian process innovation; one sampled sigma_squared retained per trajectory",
        "posterior_sampling": "one beta and one sigma_squared sample retained for complete trajectory; fresh conditional Gaussian innovation each future step",
        "nominal_twin_parameters_changed": False,
        "hidden_fault_features_allowed": False,
        "dt_s": config.dt_s,
        "rollout_horizon_steps": config.horizon_steps,
        "reference_M": 1000,
        "calibration_experiment_M": config.rollout_samples,
        "calibration_experiment_N_postfault": config.n_postfault,
        "pointwise_levels": list(COVERAGE_LEVELS),
        "simultaneous_band_method": "max standardized deviation from posterior median using pointwise MAD scale",
        "source_sha256": {
            "bayesian_residual_dynamics.py": source_hash(
                "bayesian_residual_dynamics.py"
            ),
            "counterfactual_rollout.py": source_hash("counterfactual_rollout.py"),
            "residual_dynamics_study.py": source_hash(
                "residual_dynamics_study.py"
            ),
        },
    }


def build_template(config: EpisodeDesignConfig) -> CSTRSimulation:
    simulator = CSTRSimulation(
        dt=config.dt_s,
        Fin_sp_startup=2.0 / 60.0,
        Fin_sp_normal=2.0 / 60.0,
        conc_noise_pct=0.0,
        flow_meas_noise_pct=0.0,
        sensor_T_noise_std=0.0,
        seed=config.seed,
        fault_cfg=FaultConfig(
            enable=False,
            fouling_on=False,
            pump_degrade_on=False,
            overflow_protection=True,
        ),
    )
    while simulator.t < config.template_time_s - 0.5 * config.dt_s:
        simulator.simulate_step(
            Fin_sp_normal=2.0 / 60.0,
            Fin_sp_startup=2.0 / 60.0,
        )
    return simulator


def _init_worker(
    template: CSTRSimulation, config: EpisodeDesignConfig, output_dir: str
) -> None:
    global _WORKER_TEMPLATE, _WORKER_CONFIG, _WORKER_OUTPUT
    _WORKER_TEMPLATE = template
    _WORKER_CONFIG = config
    _WORKER_OUTPUT = output_dir


def _configure_episode_models(
    template: CSTRSimulation,
    config: EpisodeDesignConfig,
    rng: np.random.Generator,
    fault_onset_s: float,
    fault_magnitude: float,
) -> tuple[CSTRSimulation, CSTRSimulation, dict]:
    nominal = copy.deepcopy(template)
    faulty = copy.deepcopy(template)
    start_time = fault_onset_s - config.pre_fault_lead_steps * config.dt_s
    nominal.t = start_time
    faulty.t = start_time

    volume_shift = float(rng.normal(0.0, config.volume_perturbation_sd_l))
    concentration_scale = float(
        1.0 + rng.normal(0.0, config.concentration_perturbation_sd_fraction)
    )
    temperature_shift = float(
        rng.normal(0.0, config.temperature_perturbation_sd_k)
    )
    perturbed_state = template.x.copy()
    perturbed_state[0] = max(0.1, perturbed_state[0] + volume_shift)
    perturbed_state[1:5] = np.maximum(
        0.0, perturbed_state[1:5] * concentration_scale
    )
    perturbed_state[5] += temperature_shift
    nominal.x = perturbed_state.copy()
    faulty.x = perturbed_state.copy()

    nominal_fault = FaultConfig(
        enable=False,
        fouling_on=False,
        pump_degrade_on=False,
        overflow_protection=True,
    )
    hidden_fault = FaultConfig(
        enable=True,
        fouling_on=True,
        fouling_t_start=fault_onset_s,
        fouling_tau=config.fault_time_constant_s,
        fouling_max=fault_magnitude,
        cool_stuck_open=False,
        cool_stuck_closed=False,
        pump_degrade_on=False,
        outlet_block_on=False,
        inlet_stuck_open=False,
        temp_bias_on=False,
        level_bias_on=False,
        leak_on=False,
        overflow_protection=True,
    )
    nominal.fault_cfg = nominal_fault
    nominal.faults = FaultManager(nominal_fault, UA_nominal=nominal.UA)
    faulty.fault_cfg = hidden_fault
    faulty.faults = FaultManager(hidden_fault, UA_nominal=faulty.UA)
    metadata = {
        "initial_volume_perturbation_l": volume_shift,
        "initial_concentration_scale": concentration_scale,
        "initial_temperature_perturbation_k": temperature_shift,
    }
    return nominal, faulty, metadata


def _paired_observation(
    nominal: CSTRSimulation,
    faulty: CSTRSimulation,
    action_lps: float,
    measurement_rng: np.random.Generator,
    measurement_noise_k: float,
) -> dict:
    nominal_out = nominal.simulate_step(
        Fin_sp_normal=action_lps, Fin_sp_startup=action_lps
    )
    faulty_out = faulty.simulate_step(
        Fin_sp_normal=action_lps, Fin_sp_startup=action_lps
    )
    if not np.isclose(float(nominal_out["t"]), float(faulty_out["t"])):
        raise RuntimeError("episode plant/twin time misalignment")
    plant_temperature = float(faulty_out["T_true"]) + float(
        measurement_rng.normal(0.0, measurement_noise_k)
    )
    twin_temperature = float(nominal_out["T_true"])
    return {
        "time_s": float(nominal_out["t"]),
        "action_lps": float(action_lps),
        "plant_temperature_k": plant_temperature,
        "nominal_twin_temperature_k": twin_temperature,
        "temperature_residual_k": plant_temperature - twin_temperature,
    }


def _transition_rows(
    observations: list[dict],
    initial_residual: float,
    initial_delta: float,
    initial_action: float,
) -> list[dict]:
    current_residual = float(initial_residual)
    current_delta = float(initial_delta)
    previous_action = float(initial_action)
    rows = []
    for observation in observations:
        next_residual = float(observation["temperature_residual_k"])
        next_delta = next_residual - current_residual
        action = float(observation["action_lps"])
        rows.append(
            {
                "r_t": current_residual,
                "delta_r_t": current_delta,
                "u_t": action,
                "delta_u_t": action - previous_action,
                "y_dt_next": float(observation["nominal_twin_temperature_k"]),
                "delta_r_next": next_delta,
                "r_next": next_residual,
            }
        )
        current_residual = next_residual
        current_delta = next_delta
        previous_action = action
    return rows


def _feature_matrix(rows: list[dict]) -> np.ndarray:
    assert_online_feature_names(FROZEN_FEATURES)
    return np.asarray(
        [[float(row[name]) for name in FROZEN_FEATURES] for row in rows],
        dtype=float,
    )


def _select_action_after_fit(
    category: str, rng: np.random.Generator
) -> float:
    if category == "interpolation":
        return float(rng.uniform(0.0270, 0.0310))
    if category == "boundary":
        if rng.random() < 0.5:
            return float(rng.uniform(0.0195, 0.0220))
        return float(rng.uniform(0.0325, 0.0340))
    if category == "extrapolation":
        if rng.random() < 0.5:
            return float(rng.uniform(0.0120, 0.0180))
        return float(rng.uniform(0.0380, 0.0450))
    raise ValueError(f"unknown action category: {category}")


def action_support_distance(action: float) -> float:
    return float(min(abs(action - training) for training in TRAINING_ACTIONS_LPS))


def construct_simultaneous_bands(
    trajectories: np.ndarray,
    levels: tuple[float, ...] = COVERAGE_LEVELS,
) -> SimultaneousBands:
    samples = np.asarray(trajectories, dtype=float)
    if samples.ndim != 2 or samples.shape[0] < 2:
        raise ValueError("simultaneous bands require a trajectory sample matrix")
    if not np.all(np.isfinite(samples)):
        raise ValueError("trajectory samples must be finite")
    median = np.median(samples, axis=0)
    mad = np.median(np.abs(samples - median[None, :]), axis=0)
    robust_scale = 1.4826 * mad
    positive = robust_scale[robust_scale > 0.0]
    reference = float(np.median(positive)) if positive.size else 1.0
    epsilon = max(1.0e-12, 1.0e-9 * reference)
    standardized_maximum = np.max(
        np.abs(samples - median[None, :]) / (robust_scale[None, :] + epsilon),
        axis=1,
    )
    lower = {}
    upper = {}
    quantiles = {}
    for level in levels:
        q = float(np.quantile(standardized_maximum, level, method="higher"))
        quantiles[level] = q
        lower[level] = median - q * robust_scale
        upper[level] = median + q * robust_scale
    return SimultaneousBands(
        median=median,
        robust_scale=robust_scale,
        lower=lower,
        upper=upper,
        standardized_maximum_quantiles=quantiles,
    )


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials:
        return float("nan"), float("nan")
    # 1.95996398454 is the two-sided 95% standard-normal quantile.
    if not np.isclose(confidence, 0.95):
        raise ValueError("this frozen experiment uses 95% Wilson intervals")
    z = 1.959963984540054
    rate = successes / trials
    denominator = 1.0 + z * z / trials
    center = (rate + z * z / (2.0 * trials)) / denominator
    half = (
        z
        * np.sqrt(rate * (1.0 - rate) / trials + z * z / (4.0 * trials**2))
        / denominator
    )
    lower = 0.0 if successes == 0 else max(0.0, center - half)
    upper = 1.0 if successes == trials else min(1.0, center + half)
    return float(lower), float(upper)


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _episode_task(task: tuple[int, str, int]) -> dict:
    episode_index, category, random_seed = task
    if _WORKER_TEMPLATE is None or _WORKER_CONFIG is None or _WORKER_OUTPUT is None:
        raise RuntimeError("worker was not initialized")
    config = _WORKER_CONFIG
    rng = np.random.default_rng(random_seed)
    measurement_rng = np.random.default_rng(random_seed + 1_000_000)
    action_rng = np.random.default_rng(random_seed + 2_000_000)
    fault_onset = float(
        rng.uniform(config.fault_onset_min_s, config.fault_onset_max_s)
    )
    fault_magnitude = float(
        rng.uniform(config.fault_magnitude_min, config.fault_magnitude_max)
    )
    nominal, faulty, initial_metadata = _configure_episode_models(
        _WORKER_TEMPLATE,
        config,
        rng,
        fault_onset,
        fault_magnitude,
    )

    pre_fault = []
    for _ in range(config.pre_fault_lead_steps):
        pre_fault.append(
            _paired_observation(
                nominal,
                faulty,
                2.0 / 60.0,
                measurement_rng,
                config.measurement_noise_k,
            )
        )
    if len(pre_fault) < 2 or not np.isclose(nominal.t, fault_onset):
        raise AssertionError("fault-boundary construction failed")
    initial_residual = float(pre_fault[-1]["temperature_residual_k"])
    initial_delta = initial_residual - float(
        pre_fault[-2]["temperature_residual_k"]
    )
    initial_action = float(pre_fault[-1]["action_lps"])

    learning_rows = []
    for index in range(config.n_postfault):
        action_index = (
            index // config.training_action_block_steps
        ) % len(TRAINING_ACTIONS_LPS)
        learning_rows.append(
            _paired_observation(
                nominal,
                faulty,
                TRAINING_ACTIONS_LPS[action_index],
                measurement_rng,
                config.measurement_noise_k,
            )
        )
    prediction_time = float(nominal.t)
    if len(learning_rows) != config.n_postfault:
        raise AssertionError("incorrect post-fault training count")
    if max(float(row["time_s"]) for row in learning_rows) > prediction_time:
        raise AssertionError("future row entered training data")
    training_transitions = _transition_rows(
        learning_rows, initial_residual, initial_delta, initial_action
    )
    features = _feature_matrix(training_transitions)
    targets = np.asarray(
        [row["delta_r_next"] for row in training_transitions], dtype=float
    )
    model = BayesianResidualDynamicsModel(FROZEN_FEATURES)
    fit_start = time.perf_counter()
    posterior = model.fit(features, targets)
    fit_time = time.perf_counter() - fit_start

    # The held-out action is selected only after the posterior has been frozen.
    proposed_action = _select_action_after_fit(category, action_rng)
    support_distance = action_support_distance(proposed_action)
    current_residual = float(learning_rows[-1]["temperature_residual_k"])
    previous_residual = float(learning_rows[-2]["temperature_residual_k"])
    current_delta = current_residual - previous_residual
    previous_action = float(learning_rows[-1]["action_lps"])

    nominal_future_model = copy.deepcopy(nominal)
    nominal_future = []
    for _ in range(config.horizon_steps):
        out = nominal_future_model.simulate_step(
            Fin_sp_normal=proposed_action,
            Fin_sp_startup=proposed_action,
        )
        nominal_future.append(
            {
                "time_s": float(out["t"]),
                "action_lps": proposed_action,
                "nominal_twin_temperature_k": float(out["T_true"]),
            }
        )
    if min(row["time_s"] for row in nominal_future) <= prediction_time:
        raise AssertionError("future prediction times overlap training")
    nominal_temperature = np.asarray(
        [row["nominal_twin_temperature_k"] for row in nominal_future]
    )
    future_actions = np.full(config.horizon_steps, proposed_action)
    rollout_result = recursive_residual_dynamics_rollout(
        posterior,
        nominal_temperature,
        future_actions,
        current_residual,
        current_delta,
        previous_action,
        config.rollout_samples,
        np.random.default_rng(random_seed + 3_000_000),
    )
    trajectories = rollout_result.corrected_output_trajectories
    if not np.all(np.isfinite(trajectories)):
        raise RuntimeError("non-finite posterior trajectory in calibration episode")
    posterior_mean = np.mean(trajectories, axis=0)
    posterior_median = np.median(trajectories, axis=0)
    pointwise = {}
    for level in COVERAGE_LEVELS:
        tail = 0.5 * (1.0 - level)
        pointwise[level] = (
            np.quantile(trajectories, tail, axis=0),
            np.quantile(trajectories, 1.0 - tail, axis=0),
        )
    simultaneous = construct_simultaneous_bands(trajectories)

    episode_id = f"episode_{episode_index:04d}_{category}"
    prediction_path = (
        Path(_WORKER_OUTPUT) / "predictions" / f"{episode_id}.npz"
    )
    prediction_payload = {
        "time_s": np.asarray([row["time_s"] for row in nominal_future]),
        "action_sequence_lps": future_actions,
        "nominal_twin_temperature_k": nominal_temperature,
        "posterior_mean_temperature_k": posterior_mean,
        "posterior_median_temperature_k": posterior_median,
    }
    for level in COVERAGE_LEVELS:
        label = int(100 * level)
        prediction_payload[f"pointwise_{label}_lower_k"] = pointwise[level][0]
        prediction_payload[f"pointwise_{label}_upper_k"] = pointwise[level][1]
        prediction_payload[f"simultaneous_{label}_lower_k"] = simultaneous.lower[
            level
        ]
        prediction_payload[f"simultaneous_{label}_upper_k"] = simultaneous.upper[
            level
        ]
    if set(prediction_payload) != PREDICTION_KEYS_WITHOUT_TRUTH:
        raise AssertionError("prediction snapshot schema changed or contains truth")
    np.savez_compressed(prediction_path, **prediction_payload)
    prediction_hash_before_reveal = _file_sha256(prediction_path)

    # Hidden future is revealed only after the complete prediction is on disk.
    plant_future_model = copy.deepcopy(faulty)
    future_measurement_rng = np.random.default_rng(random_seed + 4_000_000)
    actual_future = []
    for nominal_row in nominal_future:
        out = plant_future_model.simulate_step(
            Fin_sp_normal=proposed_action,
            Fin_sp_startup=proposed_action,
        )
        actual = float(out["T_true"]) + float(
            future_measurement_rng.normal(0.0, config.measurement_noise_k)
        )
        if not np.isclose(float(out["t"]), nominal_row["time_s"]):
            raise RuntimeError("revealed future lost time alignment")
        actual_future.append(actual)
    actual = np.asarray(actual_future)
    if _file_sha256(prediction_path) != prediction_hash_before_reveal:
        raise AssertionError("saved prediction changed after truth reveal")

    future_observations = [
        {
            **nominal_row,
            "plant_temperature_k": actual[index],
            "temperature_residual_k": actual[index]
            - nominal_row["nominal_twin_temperature_k"],
        }
        for index, nominal_row in enumerate(nominal_future)
    ]
    one_step_transitions = _transition_rows(
        future_observations,
        current_residual,
        current_delta,
        previous_action,
    )
    one_step_prediction = model.predict(
        posterior, _feature_matrix(one_step_transitions)
    )
    one_step_residual_mean = np.asarray(
        [row["r_t"] for row in one_step_transitions]
    ) + one_step_prediction.mean
    actual_residual = actual - nominal_temperature
    one_step_rmse = float(
        np.sqrt(np.mean((one_step_residual_mean - actual_residual) ** 2))
    )
    rollout_rmse = float(np.sqrt(np.mean((posterior_mean - actual) ** 2)))

    evaluation_path = (
        Path(_WORKER_OUTPUT) / "evaluations" / f"{episode_id}.npz"
    )
    np.savez_compressed(
        evaluation_path,
        actual_plant_temperature_k=actual,
        actual_residual_k=actual_residual,
        **prediction_payload,
    )

    output = {
        "episode_id": episode_id,
        "random_seed": random_seed,
        "action_category": category,
        "proposed_action": proposed_action,
        "action_support_distance": support_distance,
        "hidden_fault_id": "UA_fouling",
        "hidden_fault_magnitude": fault_magnitude,
        "hidden_fault_onset_s": fault_onset,
        "N_postfault": config.n_postfault,
        "training_sample_count": posterior.n_observations,
        "rollout_RMSE": rollout_rmse,
        "one_step_RMSE": one_step_rmse,
        "posterior_fit_time": fit_time,
        "posterior_sampling_time": rollout_result.posterior_sampling_time_s,
        "rollout_time": rollout_result.recursive_rollout_time_s,
        "total_prediction_time": fit_time
        + rollout_result.posterior_sampling_time_s
        + rollout_result.recursive_rollout_time_s,
        "prediction_snapshot_path": str(prediction_path),
        "prediction_snapshot_sha256": prediction_hash_before_reveal,
        "evaluation_snapshot_path": str(evaluation_path),
        "prediction_saved_before_reveal": 1,
        "numerically_stable": int(
            bool(np.max(np.abs(rollout_result.residual_trajectories)) < 50.0)
        ),
        **initial_metadata,
    }
    for level in COVERAGE_LEVELS:
        label = int(100 * level)
        point_lower, point_upper = pointwise[level]
        point_inside = (actual >= point_lower) & (actual <= point_upper)
        point_width = point_upper - point_lower
        simultaneous_lower = simultaneous.lower[level]
        simultaneous_upper = simultaneous.upper[level]
        simultaneous_width = simultaneous_upper - simultaneous_lower
        output[f"pointwise_covered_points_{label}"] = int(np.sum(point_inside))
        output[f"pointwise_evaluated_points_{label}"] = int(actual.size)
        output[f"pointwise_cov_{label}"] = float(np.mean(point_inside))
        output[f"width_{label}"] = float(np.mean(point_width))
        output[f"maximum_width_{label}"] = float(np.max(point_width))
        output[f"median_width_{label}"] = float(np.median(point_width))
        output[f"simultaneous_cover_{label}"] = int(
            bool(
                np.all(
                    (actual >= simultaneous_lower)
                    & (actual <= simultaneous_upper)
                )
            )
        )
        output[f"simultaneous_width_{label}"] = float(
            np.mean(simultaneous_width)
        )
        output[f"simultaneous_maximum_width_{label}"] = float(
            np.max(simultaneous_width)
        )
        output[f"simultaneous_median_width_{label}"] = float(
            np.median(simultaneous_width)
        )
        output[f"simultaneous_q_{label}"] = (
            simultaneous.standardized_maximum_quantiles[level]
        )
    return output


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def calibration_summary(episodes: list[dict]) -> list[dict]:
    rows = []
    groups = ["interpolation", "boundary", "extrapolation", "pooled"]
    for group in groups:
        selected = (
            episodes if group == "pooled" else [row for row in episodes if row["action_category"] == group]
        )
        for level in COVERAGE_LEVELS:
            label = int(100 * level)
            point_success = int(
                sum(row[f"pointwise_covered_points_{label}"] for row in selected)
            )
            point_trials = int(
                sum(row[f"pointwise_evaluated_points_{label}"] for row in selected)
            )
            point_ci = wilson_interval(point_success, point_trials)
            rows.append(
                {
                    "action_category": group,
                    "band_type": "pointwise",
                    "nominal_coverage": level,
                    "successes": point_success,
                    "trials": point_trials,
                    "empirical_coverage": point_success / point_trials,
                    "wilson_95_lower": point_ci[0],
                    "wilson_95_upper": point_ci[1],
                    "average_width_k": float(
                        np.mean([row[f"width_{label}"] for row in selected])
                    ),
                    "median_episode_width_k": float(
                        np.median([row[f"width_{label}"] for row in selected])
                    ),
                    "episode_count": len(selected),
                    "confidence_interval_note": "Wilson interval treats future points as binomial trials; serial dependence makes it approximate",
                }
            )
            simultaneous_success = int(
                sum(row[f"simultaneous_cover_{label}"] for row in selected)
            )
            simultaneous_trials = len(selected)
            simultaneous_ci = wilson_interval(
                simultaneous_success, simultaneous_trials
            )
            rows.append(
                {
                    "action_category": group,
                    "band_type": "simultaneous",
                    "nominal_coverage": level,
                    "successes": simultaneous_success,
                    "trials": simultaneous_trials,
                    "empirical_coverage": simultaneous_success
                    / simultaneous_trials,
                    "wilson_95_lower": simultaneous_ci[0],
                    "wilson_95_upper": simultaneous_ci[1],
                    "average_width_k": float(
                        np.mean(
                            [
                                row[f"simultaneous_width_{label}"]
                                for row in selected
                            ]
                        )
                    ),
                    "median_episode_width_k": float(
                        np.median(
                            [
                                row[f"simultaneous_width_{label}"]
                                for row in selected
                            ]
                        )
                    ),
                    "episode_count": len(selected),
                    "confidence_interval_note": "Wilson interval over independent episode-level whole-trajectory outcomes",
                }
            )
    return rows


def runtime_summary(episodes: list[dict]) -> list[dict]:
    rows = []
    for group in ("interpolation", "boundary", "extrapolation", "pooled"):
        selected = episodes if group == "pooled" else [row for row in episodes if row["action_category"] == group]
        output = {"action_category": group, "episode_count": len(selected)}
        for field in (
            "posterior_fit_time",
            "posterior_sampling_time",
            "rollout_time",
            "total_prediction_time",
        ):
            values = np.asarray([row[field] for row in selected], dtype=float)
            output[f"{field}_mean_s"] = float(np.mean(values))
            output[f"{field}_std_s"] = float(
                np.std(values, ddof=1 if values.size > 1 else 0)
            )
            output[f"{field}_median_s"] = float(np.median(values))
        rows.append(output)
    return rows


def support_analysis(episodes: list[dict]) -> list[dict]:
    distance = np.asarray([row["action_support_distance"] for row in episodes])
    outcomes = {
        "pointwise_width_95": np.asarray([row["width_95"] for row in episodes]),
        "rollout_RMSE": np.asarray([row["rollout_RMSE"] for row in episodes]),
        "simultaneous_width_95": np.asarray(
            [row["simultaneous_width_95"] for row in episodes]
        ),
        "pointwise_coverage_95": np.asarray(
            [row["pointwise_cov_95"] for row in episodes]
        ),
        "simultaneous_coverage_95": np.asarray(
            [row["simultaneous_cover_95"] for row in episodes]
        ),
    }
    rows = []
    for name, values in outcomes.items():
        if np.std(distance) == 0.0 or np.std(values) == 0.0:
            pearson_statistic = pearson_p = float("nan")
            spearman_statistic = spearman_p = float("nan")
        else:
            pearson = pearsonr(distance, values)
            spearman = spearmanr(distance, values)
            pearson_statistic, pearson_p = float(pearson.statistic), float(
                pearson.pvalue
            )
            spearman_statistic, spearman_p = float(spearman.statistic), float(
                spearman.pvalue
            )
        rows.append(
            {
                "outcome": name,
                "episode_count": len(episodes),
                "pearson_correlation": pearson_statistic,
                "pearson_p_value": pearson_p,
                "spearman_correlation": spearman_statistic,
                "spearman_p_value": spearman_p,
                "interpretation_restriction": "association only; no causal interpretation",
            }
        )
    return rows


def _save_figure(fig: plt.Figure, figure_dir: Path, stem: str) -> None:
    fig.savefig(figure_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(
    episodes: list[dict], summary: list[dict], figure_dir: Path
) -> None:
    categories = ("interpolation", "boundary", "extrapolation", "pooled")
    colors = dict(
        zip(categories, ("tab:blue", "tab:orange", "tab:red", "black"))
    )
    for band_type, stem, title in (
        ("pointwise", "figure_EPISODE_DESIGN_1_pointwise_calibration", "Pointwise calibration"),
        ("simultaneous", "figure_EPISODE_DESIGN_2_simultaneous_calibration", "Whole-trajectory simultaneous calibration"),
    ):
        fig, ax = plt.subplots(figsize=(7.5, 5.2))
        for category in categories:
            rows = [
                row
                for row in summary
                if row["band_type"] == band_type
                and row["action_category"] == category
            ]
            x = np.asarray([row["nominal_coverage"] for row in rows])
            y = np.asarray([row["empirical_coverage"] for row in rows])
            lower = np.asarray([row["wilson_95_lower"] for row in rows])
            upper = np.asarray([row["wilson_95_upper"] for row in rows])
            ax.errorbar(
                x,
                y,
                yerr=np.maximum(0.0, np.vstack((y - lower, upper - y))),
                marker="o",
                capsize=3,
                color=colors[category],
                label=category,
            )
        ax.plot([0.45, 1.0], [0.45, 1.0], color="grey", ls=":")
        ax.set(
            xlabel="Nominal posterior probability",
            ylabel="Empirical coverage",
            title=title,
            xlim=(0.47, 1.01),
            ylim=(0.0, 1.03),
        )
        ax.grid(alpha=0.2)
        ax.legend()
        _save_figure(fig, figure_dir, stem)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for category in categories:
        point = [row for row in summary if row["band_type"] == "pointwise" and row["action_category"] == category]
        simultaneous = [row for row in summary if row["band_type"] == "simultaneous" and row["action_category"] == category]
        axes[0].plot([row["nominal_coverage"] for row in point], [row["average_width_k"] for row in point], marker="o", color=colors[category], label=category)
        axes[1].plot([row["nominal_coverage"] for row in simultaneous], [row["average_width_k"] for row in simultaneous], marker="o", color=colors[category])
    axes[0].set_title("Pointwise sharpness"); axes[1].set_title("Simultaneous-band sharpness")
    for ax in axes:
        ax.set(xlabel="Nominal probability", ylabel="Average width [K]")
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout(); _save_figure(fig, figure_dir, "figure_EPISODE_DESIGN_3_sharpness")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    plot_fields = (
        ("width_95", "Pointwise 95% width [K]"),
        ("simultaneous_width_95", "Simultaneous 95% width [K]"),
        ("rollout_RMSE", "Rollout RMSE [K]"),
    )
    for ax, (field, label) in zip(axes, plot_fields):
        for category in categories[:-1]:
            rows = [row for row in episodes if row["action_category"] == category]
            ax.scatter([row["action_support_distance"] for row in rows], [row[field] for row in rows], s=14, alpha=0.45, color=colors[category], label=category)
        ax.set(xlabel="Action-support distance [L/s]", ylabel=label)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Action-support distance associations (not causal)")
    fig.tight_layout(); _save_figure(fig, figure_dir, "figure_EPISODE_DESIGN_4_support_distance")

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=False)
    for ax, category in zip(axes, categories[:-1]):
        rows = sorted([row for row in episodes if row["action_category"] == category], key=lambda row: row["rollout_RMSE"])
        representative = rows[len(rows) // 2]
        data = np.load(representative["evaluation_snapshot_path"])
        t = data["time_s"]
        ax.plot(t, data["nominal_twin_temperature_k"], color="grey", ls="--", label="Nominal twin")
        ax.plot(t, data["posterior_median_temperature_k"], color="tab:blue", label="Posterior median")
        ax.fill_between(t, data["simultaneous_95_lower_k"], data["simultaneous_95_upper_k"], color="tab:blue", alpha=0.12, label="95% simultaneous band")
        ax.fill_between(t, data["pointwise_95_lower_k"], data["pointwise_95_upper_k"], color="tab:blue", alpha=0.28, label="95% pointwise interval")
        ax.plot(t, data["actual_plant_temperature_k"], color="crimson", label="Hidden plant revealed")
        ax.set_title(f"{category}: {representative['episode_id']}")
        ax.set_ylabel("Temperature [K]"); ax.grid(alpha=0.2)
    axes[0].legend(fontsize=8, ncol=3)
    axes[-1].set_xlabel("Time [s]")
    fig.tight_layout(); _save_figure(fig, figure_dir, "figure_EPISODE_DESIGN_5_representative_episodes")


def create_readme(
    config: EpisodeDesignConfig,
    episodes: list[dict],
    summary: list[dict],
    runtimes: list[dict],
    support: list[dict],
    output_dir: Path,
) -> None:
    pooled_95 = {
        row["band_type"]: row
        for row in summary
        if row["action_category"] == "pooled"
        and np.isclose(float(row["nominal_coverage"]), 0.95)
    }
    runtime_pooled = next(row for row in runtimes if row["action_category"] == "pooled")
    category_lines = []
    for category in config.categories:
        point = next(row for row in summary if row["action_category"] == category and row["band_type"] == "pointwise" and np.isclose(float(row["nominal_coverage"]), 0.95))
        simultaneous = next(row for row in summary if row["action_category"] == category and row["band_type"] == "simultaneous" and np.isclose(float(row["nominal_coverage"]), 0.95))
        category_lines.append(
            f"| {category} | {point['empirical_coverage']:.3f} [{point['wilson_95_lower']:.3f}, {point['wilson_95_upper']:.3f}] | {point['average_width_k']:.3f} / {point['median_episode_width_k']:.3f} | {simultaneous['empirical_coverage']:.3f} [{simultaneous['wilson_95_lower']:.3f}, {simultaneous['wilson_95_upper']:.3f}] | {simultaneous['average_width_k']:.3f} / {simultaneous['median_episode_width_k']:.3f} |"
        )
    support_lines = [
        f"- {row['outcome']}: Pearson {row['pearson_correlation']:+.3f}; Spearman {row['spearman_correlation']:+.3f}."
        for row in support
    ]
    unstable = sum(not bool(row["numerically_stable"]) for row in episodes)
    report = f"""# episode design — Frozen-posterior calibration dataset

This folder evaluates the unchanged  B **M4 full Bayesian residual-dynamics model**. No feature, prior, noise model, numerical setting, posterior sampling rule, or rollout horizon was tuned from these episodes. No conformal calibration is performed.

There are **{len(episodes)} independent evaluation episodes**: {config.episodes_per_category} interpolation, {config.episodes_per_category} boundary, and {config.episodes_per_category} extrapolation episodes. Each uses N={config.n_postfault} observed post-fault transitions and M={config.rollout_samples} posterior trajectories over {config.horizon_steps} future points. Hidden UA-fouling magnitude, onset, initial state, measurement sequence, and held-out action vary by episode. Hidden metadata is written only to `calibration_episodes.csv` for evaluator-side stratification and never enters the learner.

Every prediction-only NPZ in `predictions/` was written before the corresponding faulty future was simulated. Its schema contains no true future values, and its SHA-256 digest was checked again after truth reveal. Evaluator snapshots containing truth are stored separately in `evaluations/`.

## Descriptive 95% results

| Category | Pointwise coverage [Wilson 95% CI] | Pointwise width avg./median [K] | Simultaneous coverage [Wilson 95% CI] | Simultaneous width avg./median [K] |
|:---|---:|---:|---:|---:|
{chr(10).join(category_lines)}
| pooled | {pooled_95['pointwise']['empirical_coverage']:.3f} [{pooled_95['pointwise']['wilson_95_lower']:.3f}, {pooled_95['pointwise']['wilson_95_upper']:.3f}] | {pooled_95['pointwise']['average_width_k']:.3f} / {pooled_95['pointwise']['median_episode_width_k']:.3f} | {pooled_95['simultaneous']['empirical_coverage']:.3f} [{pooled_95['simultaneous']['wilson_95_lower']:.3f}, {pooled_95['simultaneous']['wilson_95_upper']:.3f}] | {pooled_95['simultaneous']['average_width_k']:.3f} / {pooled_95['simultaneous']['median_episode_width_k']:.3f} |

Pointwise Wilson intervals treat future points as binomial trials and are approximate because points within a trajectory are serially dependent. Simultaneous-band Wilson intervals use independent episode-level successes and are the appropriate whole-trajectory finite-population uncertainty statement. Coverage and width must be read together; coverage above nominal is not automatically better.

The simultaneous band uses the empirical quantile of each posterior path's maximum absolute deviation from the posterior median after division by the pointwise MAD scale. It is an approximate posterior simultaneous credible band, distinct from a pointwise interval.

## Action-support associations

{chr(10).join(support_lines)}

These are descriptive associations only and are not interpreted causally.

## Runtime and integrity

- Mean fit time: {runtime_pooled['posterior_fit_time_mean_s'] * 1000:.3f} ms.
- Mean posterior sampling time: {runtime_pooled['posterior_sampling_time_mean_s'] * 1000:.3f} ms.
- Mean recursive rollout time: {runtime_pooled['rollout_time_mean_s'] * 1000:.3f} ms.
- Episodes exceeding the 50 K sampled-residual stability threshold: {unstable}/{len(episodes)}.

This stage produces a calibration dataset only. It does not recalibrate the model or decide a replacement model.
"""
    (output_dir / "README.md").write_text(report, encoding="utf-8")


def run_calibration(config: EpisodeDesignConfig) -> list[dict]:
    config.validate()
    output_dir = Path(config.output_dir)
    prediction_dir = output_dir / "predictions"
    evaluation_dir = output_dir / "evaluations"
    figure_dir = output_dir / "figures"
    for directory in (output_dir, prediction_dir, evaluation_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    with (output_dir / "frozen_model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(frozen_model_configuration(config), handle, indent=2)

    template = build_template(config)
    tasks = []
    episode_index = 0
    for category in config.categories:
        for local_index in range(config.episodes_per_category):
            tasks.append(
                (
                    episode_index,
                    category,
                    config.seed + 100_000 * config.categories.index(category) + local_index,
                )
            )
            episode_index += 1

    episodes = []
    try:
        with ProcessPoolExecutor(
            max_workers=min(config.workers, config.episode_count),
            initializer=_init_worker,
            initargs=(template, config, str(output_dir)),
        ) as executor:
            futures = {executor.submit(_episode_task, task): task for task in tasks}
            for completed, future in enumerate(as_completed(futures), start=1):
                episodes.append(future.result())
                if completed % 25 == 0 or completed == len(tasks):
                    print(
                        f"completed {completed}/{len(tasks)} calibration episodes",
                        flush=True,
                    )
    except (OSError, RuntimeError) as exc:
        if episodes:
            raise
        print(f"process pool unavailable ({exc}); using sequential fallback", flush=True)
        _init_worker(template, config, str(output_dir))
        episodes = [_episode_task(task) for task in tasks]
    episodes.sort(key=lambda row: row["episode_id"])

    summary = calibration_summary(episodes)
    runtimes = runtime_summary(episodes)
    support = support_analysis(episodes)
    _write_csv(output_dir / "calibration_episodes.csv", episodes)
    _write_csv(output_dir / "calibration_summary.csv", summary)
    _write_csv(output_dir / "runtime_summary.csv", runtimes)
    _write_csv(output_dir / "action_support_analysis.csv", support)
    create_figures(episodes, summary, figure_dir)
    create_readme(config, episodes, summary, runtimes, support, output_dir)
    return episodes


def _load_episode_rows(path: Path) -> list[dict]:
    text_fields = {
        "episode_id",
        "action_category",
        "hidden_fault_id",
        "prediction_snapshot_path",
        "prediction_snapshot_sha256",
        "evaluation_snapshot_path",
    }
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key not in text_fields:
                row[key] = float(value)
    return rows


def regenerate_aggregates(config: EpisodeDesignConfig) -> list[dict]:
    """Rebuild summaries and figures without rerunning or modifying episodes."""

    config.validate()
    output_dir = Path(config.output_dir)
    episode_path = output_dir / "calibration_episodes.csv"
    episodes = _load_episode_rows(episode_path)
    if len(episodes) != config.episode_count:
        raise RuntimeError(
            f"expected {config.episode_count} existing episodes, found {len(episodes)}"
        )
    summary = calibration_summary(episodes)
    runtimes = runtime_summary(episodes)
    support = support_analysis(episodes)
    _write_csv(output_dir / "calibration_summary.csv", summary)
    _write_csv(output_dir / "runtime_summary.csv", runtimes)
    _write_csv(output_dir / "action_support_analysis.csv", support)
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    create_figures(episodes, summary, figure_dir)
    create_readme(config, episodes, summary, runtimes, support, output_dir)
    return episodes


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="artifacts/cstr/episode_design")
    parser.add_argument("--episodes-per-category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42_000)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="regenerate summaries and figures from existing episode CSV",
    )
    return parser


def main() -> None:
    parsed = vars(build_argument_parser().parse_args())
    aggregate_only = parsed.pop("aggregate_only")
    config = EpisodeDesignConfig(**parsed)
    episodes = (
        regenerate_aggregates(config) if aggregate_only else run_calibration(config)
    )
    print(f"episode design complete: {config.output_dir} ({len(episodes)} episodes)")


if __name__ == "__main__":
    main()
