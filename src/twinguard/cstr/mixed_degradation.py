"""Evaluate fault-blind CSTR revalidation under multiple degradation mechanisms."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twinguard.cstr import episode_design as episode_design
from twinguard.cstr import freeze_discrepancy_model as discrepancy_freeze
from twinguard.cstr import bootstrap_ensemble as bootstrap_ensemble
from twinguard.cstr import conformal_calibration as trajectory_conformal
from twinguard.cstr import safety_certification as certification
from twinguard.cstr import feed_temperature_fault as cstr_fault_check
from twinguard.plants.cstr import FaultManager
from twinguard.discrepancy.cstr_state_space import fit_scalar_state_space_em_checkpoints, kalman_filter


T_MAX_K = 313.0
HORIZON = 240
K = 1
FEATURE_NAMES = ("u_t", "delta_u_t", "y_dt_t", "elapsed_since_detection_s")
FAULTS = ("ua_fouling", "feed_temperature_step")
CATEGORIES = ("interpolation", "boundary", "extrapolation")
SEVERITY_RANGES = {
    "ua_fouling": {"low": (0.25, 0.35), "medium": (0.35, 0.45), "high": (0.45, 0.55)},
    "feed_temperature_step": {"low": (5.0, 15.0), "medium": (15.0, 27.5), "high": (27.5, 40.0)},
}
ROLE_SEEDS = {"training": 2_100_000, "selection": 2_200_000, "calibration": 2_300_000, "final_test": 2_400_000}


@dataclass(frozen=True)
class Config:
    output_dir: str = "artifacts/cstr/mixed_degradation"
    training_count: int = 600
    selection_count: int = 60
    calibration_count: int = 400
    final_count: int = 600
    training_steps: int = 120
    trajectories: int = 2000
    bootstrap_target: int = 50
    ensemble_size: int = 10
    workers: int = min(4, os.cpu_count() or 1)

    def validate(self) -> None:
        if self.training_count != 600 or self.final_count < 600:
            raise ValueError("mixed-degradation evaluation requires 600 training and at least 600 final episodes")
        if self.trajectories != 2000 or self.bootstrap_target != 50 or self.ensemble_size != 10:
            raise ValueError("the transferred method freezes M=2000, bootstrap J=50, online J=10")
        if self.training_steps != 120 or self.selection_count < 30 or self.calibration_count < 200:
            raise ValueError("insufficient training, selection, or calibration population")
        if len(set(ROLE_SEEDS.values())) != len(ROLE_SEEDS):
            raise ValueError("population seed blocks overlap")


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"empty output table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fault_assignment(index: int) -> tuple[str, str]:
    fault = FAULTS[index % len(FAULTS)]
    severity = ("low", "medium", "high")[(index // len(FAULTS)) % 3]
    return fault, severity


def _configure_pair(template, config: episode_design.EpisodeDesignConfig, seed: int, fault: str, severity_label: str):
    rng = np.random.default_rng(seed)
    onset = float(rng.uniform(config.fault_onset_min_s, config.fault_onset_max_s))
    lo, hi = SEVERITY_RANGES[fault][severity_label]
    severity = float(rng.uniform(lo, hi))
    nominal = copy.deepcopy(template); plant = copy.deepcopy(template)
    nominal.t = onset - config.pre_fault_lead_steps * config.dt_s; plant.t = nominal.t
    state = template.x.copy()
    volume_shift = float(rng.normal(0.0, config.volume_perturbation_sd_l))
    concentration_scale = float(1 + rng.normal(0.0, config.concentration_perturbation_sd_fraction))
    temperature_shift = float(rng.normal(0.0, config.temperature_perturbation_sd_k))
    state[0] = max(.1, state[0] + volume_shift); state[1:5] = np.maximum(0, state[1:5] * concentration_scale); state[5] += temperature_shift
    nominal.x = state.copy(); plant.x = state.copy()
    nominal_cfg = cstr_fault_check._clean_fault(enable=False)
    if fault == "ua_fouling":
        plant_cfg = cstr_fault_check._clean_fault(fouling_on=True, fouling_t_start=onset, fouling_tau=1.0, fouling_max=severity)
        units = "fractional UA loss"
    else:
        plant_cfg = cstr_fault_check._clean_fault(feed_temp_step_on=True, feed_temp_step_t=onset, feed_temp_step_K=severity)
        units = "K feed-temperature increase"
    nominal.fault_cfg = nominal_cfg; nominal.faults = FaultManager(nominal_cfg, nominal.UA)
    plant.fault_cfg = plant_cfg; plant.faults = FaultManager(plant_cfg, plant.UA)
    metadata = {"fault_mechanism_evaluator_only": fault, "severity_label_evaluator_only": severity_label, "severity_value_evaluator_only": severity, "severity_units_evaluator_only": units, "fault_onset_s_evaluator_only": onset, "initial_volume_perturbation_l_evaluator_only": volume_shift, "initial_concentration_scale_evaluator_only": concentration_scale, "initial_temperature_perturbation_k_evaluator_only": temperature_shift}
    return nominal, plant, metadata


_TEMPLATE = None
_EPISODE_DESIGN_CONFIG = None


def _init_population_worker(template, config):
    global _TEMPLATE, _EPISODE_DESIGN_CONFIG
    _TEMPLATE = template; _EPISODE_DESIGN_CONFIG = config


def _training_task(task):
    index, seed, steps = task; fault, severity_label = _fault_assignment(index)
    nominal, plant, metadata = _configure_pair(_TEMPLATE, _EPISODE_DESIGN_CONFIG, seed, fault, severity_label)
    rng = np.random.default_rng(seed); measurement_rng = np.random.default_rng(seed + 1_000_000); previous = 2 / 60
    for _ in range(_EPISODE_DESIGN_CONFIG.pre_fault_lead_steps): episode_design._paired_observation(nominal, plant, previous, measurement_rng, _EPISODE_DESIGN_CONFIG.measurement_noise_k)
    residual = np.empty(steps); z = np.empty((steps, 4)); actions = np.empty(steps); remaining = 0; action = previous; changes = 0
    for t in range(steps):
        if remaining <= 0:
            remaining = int(rng.integers(5, 16)); action = float(rng.uniform(.012, .045) if rng.random() < .25 else rng.uniform(.019, .036)); changes += int(t > 0)
        row = episode_design._paired_observation(nominal, plant, action, measurement_rng, _EPISODE_DESIGN_CONFIG.measurement_noise_k)
        residual[t] = row["temperature_residual_k"]; actions[t] = action; z[t] = [action, action - previous, row["nominal_twin_temperature_k"], t + 1]; previous = action; remaining -= 1
    manifest = {"episode_id": f"cstr_mixed_training_{index:04d}", "population_role": "training", "random_seed": seed, **metadata, "action_category": "variable_training_sequence", "operational_features": "|".join(FEATURE_NAMES), "fault_metadata_available_to_method": 0, "steps": steps, "action_min_lps": float(np.min(actions)), "action_max_lps": float(np.max(actions)), "action_changes": changes}
    return residual, z, manifest


def generate_training(config: Config, template, b5config, output: Path):
    tasks = [(i, ROLE_SEEDS["training"] + i, config.training_steps) for i in range(config.training_count)]
    with ProcessPoolExecutor(max_workers=config.workers, initializer=_init_population_worker, initargs=(template, b5config)) as pool:
        results = list(pool.map(_training_task, tasks, chunksize=2))
    residual = np.stack([x[0] for x in results]); z = np.stack([x[1] for x in results])
    path = output / "training_population.npz"; np.savez_compressed(path, residual=residual, z=z)
    return residual, z, [x[2] for x in results], path


def _episode_task(task):
    role, index, seed = task; fault, severity_label = _fault_assignment(index)
    category = CATEGORIES[(index // 6) % 3]
    nominal, plant, metadata = _configure_pair(_TEMPLATE, _EPISODE_DESIGN_CONFIG, seed, fault, severity_label)
    measurement_rng = np.random.default_rng(seed + 1_000_000); action_rng = np.random.default_rng(seed + 2_000_000); prior = 2 / 60
    for _ in range(_EPISODE_DESIGN_CONFIG.pre_fault_lead_steps): episode_design._paired_observation(nominal, plant, prior, measurement_rng, _EPISODE_DESIGN_CONFIG.measurement_noise_k)
    learning = []
    for t in range(_EPISODE_DESIGN_CONFIG.n_postfault):
        action = episode_design.TRAINING_ACTIONS_LPS[(t // _EPISODE_DESIGN_CONFIG.training_action_block_steps) % len(episode_design.TRAINING_ACTIONS_LPS)]
        learning.append(episode_design._paired_observation(nominal, plant, action, measurement_rng, _EPISODE_DESIGN_CONFIG.measurement_noise_k))
    proposed = episode_design._select_action_after_fit(category, action_rng); previous = prior; online = []
    for t, row in enumerate(learning):
        action = float(row["action_lps"]); online.append([action, action - previous, row["nominal_twin_temperature_k"], t + 1]); previous = action
    nominal_branch = copy.deepcopy(nominal); nominal_future = []; future = []
    for h in range(HORIZON):
        out = nominal_branch.simulate_step(Fin_sp_normal=proposed, Fin_sp_startup=proposed); temp = float(out["T_true"]); nominal_future.append(temp); future.append([proposed, proposed - previous, temp, 51 + h]); previous = proposed
    episode_id = f"cstr_mixed_{role}_{index:04d}_{category}"; truth_path = Path("__OUTPUT__") / f"revealed_{role}" / f"{episode_id}.npz"
    item = discrepancy_freeze.EpisodeInputs(episode_id, category, seed, np.asarray([x["temperature_residual_k"] for x in learning]), np.asarray(online), np.asarray(future), np.asarray(nominal_future), proposed, truth_path, plant, None)
    manifest = {"episode_id": episode_id, "population_role": role, "random_seed": seed, **metadata, "action_category": category, "proposed_action_lps": proposed, "postfault_observation_count": 50, "fault_metadata_available_to_method": 0, "truth_generated_after_prediction": 1}
    return item, manifest


def generate_episodes(config: Config, template, b5config, output: Path, role: str, count: int):
    tasks = [(role, i, ROLE_SEEDS[role] + i) for i in range(count)]
    with ProcessPoolExecutor(max_workers=config.workers, initializer=_init_population_worker, initargs=(template, b5config)) as pool:
        results = list(pool.map(_episode_task, tasks, chunksize=2))
    episodes = []
    for item, manifest in results:
        truth_path = output / f"revealed_{role}" / f"{item.episode_id}.npz"
        episodes.append(discrepancy_freeze.EpisodeInputs(item.episode_id, item.category, item.random_seed, item.residual_online, item.z_online, item.z_future, item.nominal_future, item.proposed_action, truth_path, item.hidden_plant, None))
    return episodes, [x[1] for x in results]


def _reveal_physical(item: discrepancy_freeze.EpisodeInputs, prediction_path: Path) -> np.ndarray:
    if not prediction_path.exists() or item.hidden_plant is None:
        raise RuntimeError("truth reveal requires a saved prediction and hidden plant")
    actual = []
    for _ in range(HORIZON):
        actual.append(float(item.hidden_plant.simulate_step(Fin_sp_normal=item.proposed_action, Fin_sp_startup=item.proposed_action)["T_true"]))
    values = np.asarray(actual); item.truth_path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(item.truth_path, actual_physical_temperature_k=values)
    return values


def selection_diagnostics(model, episodes):
    rows = []
    for item in episodes:
        paths, mean, _, _, _ = bootstrap_ensemble._paths_for_model(model, item, 200, item.random_seed + 121_000_000)
        path = Path(item.truth_path).parent.parent / "selection_predictions" / f"{item.episode_id}.npz"; path.parent.mkdir(parents=True, exist_ok=True); np.savez_compressed(path, predictive_mean_temperature_k=mean)
        actual = _reveal_physical(item, path); filtered = kalman_filter(model, item.residual_online, item.z_online); error = mean - actual
        rows.append({"episode_id": item.episode_id, "recursive_RMSE_k": float(np.sqrt(np.mean(error ** 2))), "prediction_bias_k": float(np.mean(error)), "innovation_bias_k": float(np.mean(filtered.innovations)), "spectral_radius": model.spectral_radius})
    pooled = {"number_of_selection_episodes": len(rows), "median_recursive_RMSE_k": float(np.median([r["recursive_RMSE_k"] for r in rows])), "absolute_mean_prediction_bias_k": abs(float(np.mean([r["prediction_bias_k"] for r in rows]))), "absolute_mean_innovation_bias_k": abs(float(np.mean([r["innovation_bias_k"] for r in rows]))), "spectral_radius": model.spectral_radius}
    pooled["k1_passed_predefined_mean_dynamics_gate"] = int(pooled["median_recursive_RMSE_k"] <= 3.0 and pooled["absolute_mean_prediction_bias_k"] <= 1.0 and pooled["spectral_radius"] < 1)
    return rows, pooled


_PRED_MODELS = None
_PRED_OUTPUT = None
_Q_PLUS = None
_Q_MINUS = None
_META = None


def _init_predict_worker(models, output, q_plus=None, q_minus=None, metadata=None):
    global _PRED_MODELS, _PRED_OUTPUT, _Q_PLUS, _Q_MINUS, _META
    _PRED_MODELS = models; _PRED_OUTPUT = Path(output); _Q_PLUS = q_plus; _Q_MINUS = q_minus; _META = metadata or {}


def _calibration_task(item):
    path = _PRED_OUTPUT / "calibration_predictions" / f"{item.episode_id}.npz"; pred = certification.predict_frozen_u3(item, _PRED_MODELS, 2000, path); before = _sha256(path); actual = _reveal_physical(item, path)
    if before != _sha256(path): raise AssertionError("prediction changed after calibration reveal")
    plus, minus = certification.one_sided_scores(actual, pred["center"], pred["scale"])
    return {"episode_id": item.episode_id, "upper_score": plus, "lower_score": minus, "prediction_saved_before_truth_reveal": 1}


def calibrate(config: Config, episodes, models, output: Path):
    with ProcessPoolExecutor(max_workers=config.workers, initializer=_init_predict_worker, initargs=(models, str(output))) as pool:
        rows = list(pool.map(_calibration_task, episodes, chunksize=1))
    q_plus, rank_plus = trajectory_conformal.corrected_quantile(np.asarray([r["upper_score"] for r in rows]), .95)
    q_minus, rank_minus = trajectory_conformal.corrected_quantile(np.asarray([r["lower_score"] for r in rows]), .95)
    return rows, float(q_plus), float(q_minus), int(rank_plus), int(rank_minus)


def _final_task(item):
    path = _PRED_OUTPUT / "final_predictions" / f"{item.episode_id}.npz"; pred = certification.predict_frozen_u3(item, _PRED_MODELS, 2000, path); prediction_ns = time.time_ns(); before = _sha256(path)
    start = time.perf_counter(); scale = np.maximum(pred["scale"], certification.EPSILON_SCALE_K); upper = pred["center"] + _Q_PLUS * scale; lower = pred["center"] - _Q_MINUS * scale; certified = float(np.max(upper)) <= T_MAX_K; conformal_seconds = time.perf_counter() - start
    decision_path = _PRED_OUTPUT / "final_decisions" / f"{item.episode_id}.json"; decision_path.parent.mkdir(parents=True, exist_ok=True); decision_path.write_text(json.dumps({"episode_id": item.episode_id, "decision": "CERTIFIED SAFE" if certified else "NOT CERTIFIED", "q_plus": _Q_PLUS, "T_max_k": T_MAX_K, "prediction_sha256": before}, indent=2), encoding="utf-8"); decision_ns = time.time_ns()
    actual = _reveal_physical(item, path); reveal_ns = time.time_ns()
    if before != _sha256(path) or not prediction_ns <= decision_ns <= reveal_ns: raise AssertionError("predict-decide-reveal chronology failure")
    meta = _META[item.episode_id]; actual_safe = float(np.max(actual)) <= T_MAX_K; nominal_safe = float(np.max(item.nominal_future)) <= T_MAX_K
    return {"episode_id": item.episode_id, "random_seed": item.random_seed, "action_category": item.category, **meta, "proposed_action_lps": item.proposed_action, "actual_max_temperature_k": float(np.max(actual)), "actual_safe": int(actual_safe), "actual_unsafe": int(not actual_safe), "nominal_dt_safe": int(nominal_safe), "nominal_dt_false_safe": int(nominal_safe and not actual_safe), "nominal_upper_trajectory_covered": int(np.all(actual <= item.nominal_future)), "certified_safe": int(certified), "decision": "CERTIFIED SAFE" if certified else "NOT CERTIFIED", "false_safe": int(certified and not actual_safe), "upper_trajectory_covered": int(np.all(actual <= upper)), "mean_conformal_width_k": float(np.mean(upper - lower)), "median_conformal_width_k": float(np.median(upper - lower)), "prediction_RMSE_k": float(np.sqrt(np.mean((pred["center"] - actual) ** 2))), "true_safety_margin_k": T_MAX_K - float(np.max(actual)), "conformal_safety_margin_k": T_MAX_K - float(np.max(upper)), "online_latency_seconds": pred["prediction_seconds"] + conformal_seconds, "prediction_sha256": before, "decision_sha256": _sha256(decision_path), "prediction_saved_ns": prediction_ns, "decision_saved_ns": decision_ns, "truth_revealed_ns": reveal_ns, "chronology_verified": 1}


def evaluate_final(config: Config, episodes, models, q_plus, q_minus, manifest, output):
    metadata = {r["episode_id"]: {k: v for k, v in r.items() if k.endswith("_evaluator_only")} for r in manifest}
    with ProcessPoolExecutor(max_workers=config.workers, initializer=_init_predict_worker, initargs=(models, str(output), q_plus, q_minus, metadata)) as pool:
        return list(pool.map(_final_task, episodes, chunksize=1))


def _wilson(successes, trials):
    return certification.wilson(int(successes), int(trials)) if trials else (float("nan"), float("nan"))


def metric_row(rows, method, population="CSTR mixed degradation"):
    n = len(rows)
    if method == "nominal_DT":
        safe = sum(r["nominal_dt_safe"] for r in rows); false = sum(r["nominal_dt_false_safe"] for r in rows); coverage = np.mean([r["nominal_upper_trajectory_covered"] for r in rows]); width = 0.0; latency = 0.0
    else:
        safe = sum(r["certified_safe"] for r in rows); false = sum(r["false_safe"] for r in rows); coverage = np.mean([r["upper_trajectory_covered"] for r in rows]); width = np.mean([r["mean_conformal_width_k"] for r in rows]); latency = np.median([r["online_latency_seconds"] for r in rows])
    joint_lo, joint_hi = _wilson(false, n); cond_lo, cond_hi = _wilson(false, safe)
    return {"population": population, "method": method, "number_of_episodes": n, "upper_trajectory_coverage": coverage, "SAFE_certification_rate": safe / n, "SAFE_count": safe, "false_safe_count": false, "joint_false_safe_probability": false / n, "joint_false_safe_CI_lower": joint_lo, "joint_false_safe_CI_upper": joint_hi, "conditional_false_safe_rate": false / safe if safe else np.nan, "conditional_false_safe_CI_lower": cond_lo, "conditional_false_safe_CI_upper": cond_hi, "NOT_CERTIFIED_rate": 1 - safe / n, "mean_conformal_width_k": width, "median_online_latency_seconds": latency}


def subgroup_rows(rows, q_plus):
    fault_rows = []; severity_rows = []
    for fault in FAULTS:
        selected = [r for r in rows if r["fault_mechanism_evaluator_only"] == fault]
        fault_rows.append({"fault_mechanism": fault, "N": len(selected), "actual_unsafe_fraction": np.mean([r["actual_unsafe"] for r in selected]), "nominal_DT_joint_false_safe_probability": np.mean([r["nominal_dt_false_safe"] for r in selected]), "proposed_method_SAFE_rate": np.mean([r["certified_safe"] for r in selected]), "proposed_method_false_safe_count": sum(r["false_safe"] for r in selected), "upper_envelope_coverage": np.mean([r["upper_trajectory_covered"] for r in selected]), "median_conformal_width_k": np.median([r["median_conformal_width_k"] for r in selected]), "median_prediction_RMSE_k": np.median([r["prediction_RMSE_k"] for r in selected]), "pooled_q_plus": q_plus, "formal_subgroup_guarantee_claimed": 0})
        for severity in ("low", "medium", "high"):
            group = [r for r in selected if r["severity_label_evaluator_only"] == severity]
            severity_rows.append({"fault_mechanism": fault, "severity_label": severity, "N": len(group), "actual_unsafe_fraction": np.mean([r["actual_unsafe"] for r in group]), "nominal_DT_joint_false_safe_probability": np.mean([r["nominal_dt_false_safe"] for r in group]), "proposed_method_SAFE_rate": np.mean([r["certified_safe"] for r in group]), "proposed_method_false_safe_count": sum(r["false_safe"] for r in group), "upper_envelope_coverage": np.mean([r["upper_trajectory_covered"] for r in group]), "median_conformal_width_k": np.median([r["median_conformal_width_k"] for r in group]), "median_prediction_RMSE_k": np.median([r["prediction_RMSE_k"] for r in group]), "pooled_q_plus": q_plus, "formal_subgroup_guarantee_claimed": 0})
    return fault_rows, severity_rows


def original_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle: rows = list(csv.DictReader(handle))
    output = []
    for source_method, method in (("S0_nominal_DT", "nominal_DT"), ("S3_one_sided", "proposed_method")):
        r = next(x for x in rows if x["method"] == source_method and x["action_category"] == "pooled" and np.isclose(float(x["nominal_level"]), .95))
        output.append({"population": "CSTR primary evaluation", "method": method, "number_of_episodes": int(r["number_of_episodes"]), "upper_trajectory_coverage": float(r["upper_trajectory_empirical_coverage"]), "SAFE_certification_rate": float(r["SAFE_certification_rate"]), "SAFE_count": int(r["SAFE_count"]), "false_safe_count": int(r["false_safe_count"]), "joint_false_safe_probability": float(r["joint_false_safe_probability"]), "joint_false_safe_CI_lower": float(r["joint_false_safe_CI_lower"]), "joint_false_safe_CI_upper": float(r["joint_false_safe_CI_upper"]), "conditional_false_safe_rate": float(r["conditional_false_safe_rate"]), "conditional_false_safe_CI_lower": float(r["false_safe_CI_lower"]), "conditional_false_safe_CI_upper": float(r["false_safe_CI_upper"]), "NOT_CERTIFIED_rate": 1.0 - float(r["SAFE_certification_rate"]), "mean_conformal_width_k": float(r["mean_envelope_width_k"]), "median_online_latency_seconds": float(r["median_latency"])})
    return output


def create_figures(output, summary, faults, severities, final_rows, calibration_rows):
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    def save(fig, name): fig.tight_layout(); fig.savefig(figures / f"{name}.png", dpi=180); plt.close(fig)
    multi = [r for r in summary if r["population"] == "CSTR mixed degradation"]
    proposed = next(r for r in multi if r["method"] == "proposed_method"); nominal = next(r for r in multi if r["method"] == "nominal_DT")
    fig, ax = plt.subplots(figsize=(7, 4)); ax.bar(["Nominal false-safe", "Proposed false-safe", "Proposed SAFE", "Upper coverage"], [nominal["joint_false_safe_probability"], proposed["joint_false_safe_probability"], proposed["SAFE_certification_rate"], proposed["upper_trajectory_coverage"]], color=["#a33a3a", "#a33a3a", "#286f45", "#2d5f8b"]); ax.set_ylim(0,1); ax.set_ylabel("Episode fraction"); save(fig,"01_pooled_safety_metrics")
    fig, axes = plt.subplots(1,2,figsize=(10,4)); labels=[r["fault_mechanism"].replace("_"," ") for r in faults]; axes[0].bar(labels,[r["proposed_method_SAFE_rate"] for r in faults],color="#286f45"); axes[1].bar(labels,[r["upper_envelope_coverage"] for r in faults],color="#2d5f8b"); axes[0].set_ylabel("SAFE certification rate"); axes[1].set_ylabel("Upper trajectory coverage"); [ax.tick_params(axis="x",rotation=12) for ax in axes]; save(fig,"02_fault_specific_results")
    fig, axes=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,fault in zip(axes,FAULTS):
        group=[r for r in severities if r["fault_mechanism"]==fault]; ax.plot([r["severity_label"] for r in group],[r["upper_envelope_coverage"] for r in group],marker="o",label="Coverage"); ax.plot([r["severity_label"] for r in group],[r["proposed_method_SAFE_rate"] for r in group],marker="s",label="SAFE rate"); ax.axhline(.95,color="black",ls="--",lw=.8); ax.set_title(fault.replace("_"," ")); ax.legend()
    axes[0].set_ylabel("Episode fraction"); save(fig,"03_severity_diagnostics")
    fig,ax=plt.subplots(figsize=(6.5,4)); ax.scatter([r["true_safety_margin_k"] for r in final_rows],[r["conformal_safety_margin_k"] for r in final_rows],c=["#286f45" if r["certified_safe"] else "#777777" for r in final_rows],s=13,alpha=.55); ax.axhline(0,color="black",lw=.8); ax.axvline(0,color="black",lw=.8); ax.set(xlabel="True safety margin (K)",ylabel="Conformal safety margin (K)"); save(fig,"04_true_vs_conformal_margin")
    fig,ax=plt.subplots(figsize=(6.5,4)); ax.hist([r["upper_score"] for r in calibration_rows],bins=30,color="#2d5f8b",alpha=.8); ax.set(xlabel="Calibration upper trajectory score",ylabel="Episodes"); save(fig,"05_calibration_scores")
    fig,ax=plt.subplots(figsize=(7,4)); original=next(r for r in summary if r["population"]=="CSTR primary evaluation" and r["method"]=="proposed_method"); x=np.arange(4); ax.bar(x-.18,[original["joint_false_safe_probability"],original["SAFE_certification_rate"],original["upper_trajectory_coverage"],original["mean_conformal_width_k"]],.36,label="Original primary certification"); ax.bar(x+.18,[proposed["joint_false_safe_probability"],proposed["SAFE_certification_rate"],proposed["upper_trajectory_coverage"],proposed["mean_conformal_width_k"]],.36,label="CSTR mixed degradation"); ax.set_xticks(x,["False-safe","SAFE rate","Coverage","Width (K)"]); ax.legend(); save(fig,"06_original_vs_multifault")


def verdict(summary, fault_rows):
    proposed = next(r for r in summary if r["population"] == "CSTR mixed degradation" and r["method"] == "proposed_method")
    strong = proposed["joint_false_safe_probability"] <= .05 and proposed["upper_trajectory_coverage"] >= .90
    useful = proposed["SAFE_certification_rate"] >= .15 and all(r["proposed_method_SAFE_rate"] >= .05 for r in fault_rows)
    if strong and useful: return "MULTI-FAULT REVALIDATION SUCCESSFUL"
    if strong: return "MULTI-FAULT REVALIDATION CONSERVATIVE"
    return "MULTI-FAULT MODEL INADEQUATE"


def create_reports(output, config, summary, faults, severities, q_plus, q_minus, rank_plus, selection, decision):
    proposed = next(r for r in summary if r["population"] == "CSTR mixed degradation" and r["method"] == "proposed_method"); nominal = next(r for r in summary if r["population"] == "CSTR mixed degradation" and r["method"] == "nominal_DT"); original = next(r for r in summary if r["population"] == "CSTR primary evaluation" and r["method"] == "proposed_method"); original_nominal = next(r for r in summary if r["population"] == "CSTR primary evaluation" and r["method"] == "nominal_DT")
    fault_text="\n".join(f"| {r['fault_mechanism']} | {r['N']} | {r['actual_unsafe_fraction']:.3f} | {r['proposed_method_SAFE_rate']:.3f} | {r['proposed_method_false_safe_count']} | {r['upper_envelope_coverage']:.3f} |" for r in faults)
    paper=f"""# mixed-degradation evaluation - Multi-fault CSTR revalidation

The transferred fault-blind method retained k=1, the four observable features, whole-episode bootstrap, J=10/M=2000 prediction, standardized episode-level conformal scores, and prospective predict-decide-reveal ordering. All state-space parameters, 50 bootstrap fits, and q+={q_plus:.4f} (finite-sample rank {rank_plus}/{config.calibration_count}) were learned anew from balanced multi-fault populations. The final test used the practical minimum of {config.final_count} episodes because end-to-end CSTR integration and ensemble rollout made 1,000 materially more expensive. No final episode entered fitting, model diagnostics, bootstrap, or calibration.

The predefined k=1 training/selection gate passed: selection median RMSE was {selection['median_recursive_RMSE_k']:.3f} K, absolute mean prediction bias was {selection['absolute_mean_prediction_bias_k']:.3f} K, and spectral radius was {selection['spectral_radius']:.5f}. Dimensions 2 and 3 were therefore not evaluated.

## Pooled final results

The proposed method certified {proposed['SAFE_certification_rate']:.3f} of episodes, upper-envelope coverage was {proposed['upper_trajectory_coverage']:.3f}, and {proposed['false_safe_count']} false-safe decisions gave joint probability {proposed['joint_false_safe_probability']:.3f} (Wilson 95% CI [{proposed['joint_false_safe_CI_lower']:.3f}, {proposed['joint_false_safe_CI_upper']:.3f}]). Conditional false-safe rate was {proposed['conditional_false_safe_rate']:.3f} (Wilson [{proposed['conditional_false_safe_CI_lower']:.3f}, {proposed['conditional_false_safe_CI_upper']:.3f}]); median latency was {proposed['median_online_latency_seconds']:.3f} s. Nominal-DT joint false-safe was {nominal['joint_false_safe_probability']:.3f}.

| Fault subgroup | N | Actual unsafe | SAFE rate | False-safe count | Upper coverage |
|---|---:|---:|---:|---:|---:|
{fault_text}

These fault rows are post-decision subgroup diagnostics under pooled calibration, not formal per-fault guarantees. High-severity UA undercoverage **persists** at {next(r['upper_envelope_coverage'] for r in severities if r['fault_mechanism']=='ua_fouling' and r['severity_label']=='high'):.3f}; high-severity feed-temperature coverage is similarly {next(r['upper_envelope_coverage'] for r in severities if r['fault_mechanism']=='feed_temperature_step' and r['severity_label']=='high'):.3f}. This conditional undercoverage does not contradict the pooled conformal guarantee, but it is the strongest deployment limitation.

## Original primary certification comparison

| Metric | CSTR primary evaluation | CSTR mixed degradation |
|---|---:|---:|
| Nominal-DT joint false-safe | {original_nominal['joint_false_safe_probability']:.3f} | {nominal['joint_false_safe_probability']:.3f} |
| Proposed joint false-safe | {original['joint_false_safe_probability']:.3f} | {proposed['joint_false_safe_probability']:.3f} |
| Proposed SAFE certification | {original['SAFE_certification_rate']:.3f} | {proposed['SAFE_certification_rate']:.3f} |
| Proposed upper coverage | {original['upper_trajectory_coverage']:.3f} | {proposed['upper_trajectory_coverage']:.3f} |
| Mean conformal width (K) | {original['mean_conformal_width_k']:.3f} | {proposed['mean_conformal_width_k']:.3f} |
| Median online latency (s) | {original['median_online_latency_seconds']:.3f} | {proposed['median_online_latency_seconds']:.3f} |

### {decision}
"""
    (output/"mixed_degradation_report.md").write_text(paper,encoding="utf-8"); (output/"decision_summary.md").write_text(f"### {decision}\n",encoding="utf-8")


def run(config: Config) -> str:
    config.validate(); output=Path(config.output_dir); output.mkdir(parents=True,exist_ok=True); b5config=episode_design.EpisodeDesignConfig(); template=episode_design.build_template(b5config)
    residual,z,manifest,training_path=generate_training(config,template,b5config,output)
    fixed=fit_scalar_state_space_em_checkpoints(residual,z,91_005,(100,))[100]
    selection_episodes,selection_manifest=generate_episodes(config,template,b5config,output,"selection",config.selection_count); selection_rows,selection=selection_diagnostics(fixed,selection_episodes)
    if not selection["k1_passed_predefined_mean_dynamics_gate"]: raise RuntimeError("k=1 failed predefined mean-dynamics gate; k=2/3 diagnostic extension required before final testing")
    models,bootstrap_manifest=bootstrap_ensemble.fit_bootstrap_ensemble(training_path,config.bootstrap_target,75,config.workers)
    (output/"frozen_model.json").write_text(json.dumps({"latent_dimension":1,"parameters":discrepancy_freeze._parameter_payload(fixed),"selection_gate":selection},indent=2),encoding="utf-8"); (output/"bootstrap_models.json").write_text(json.dumps([discrepancy_freeze._parameter_payload(p) for p in models],indent=2),encoding="utf-8"); _write_csv(output/"bootstrap_manifest.csv",bootstrap_manifest); _write_csv(output/"selection_diagnostics.csv",selection_rows)
    calibration_episodes,calibration_manifest=generate_episodes(config,template,b5config,output,"calibration",config.calibration_count); calibration_rows,q_plus,q_minus,rank_plus,rank_minus=calibrate(config,calibration_episodes,models[:config.ensemble_size],output); _write_csv(output/"calibration_scores.csv",calibration_rows)
    frozen_at_ns=time.time_ns(); final_episodes,final_manifest=generate_episodes(config,template,b5config,output,"final_test",config.final_count)
    final_rows=evaluate_final(config,final_episodes,models[:config.ensemble_size],q_plus,q_minus,final_manifest,output)
    all_manifest=[*manifest,*selection_manifest,*calibration_manifest,*final_manifest]; _write_csv(output/"mixed_degradation_population_manifest.csv",all_manifest); _write_csv(output/"mixed_degradation_test_episodes.csv",final_rows)
    summary=[*original_rows(Path("artifacts/cstr/primary_evaluation/safety_summary.csv")),metric_row(final_rows,"nominal_DT"),metric_row(final_rows,"proposed_method")]; fault_rows,severity_rows=subgroup_rows(final_rows,q_plus); _write_csv(output/"mixed_degradation_summary.csv",summary); _write_csv(output/"mixed_degradation_by_fault.csv",fault_rows); _write_csv(output/"mixed_degradation_by_severity.csv",severity_rows)
    decision=verdict(summary,fault_rows); create_figures(output,summary,fault_rows,severity_rows,final_rows,calibration_rows); create_reports(output,config,summary,fault_rows,severity_rows,q_plus,q_minus,rank_plus,selection,decision)
    (output/"configuration.json").write_text(json.dumps({"configuration":asdict(config),"method":{"latent_dimension":K,"feature_names":FEATURE_NAMES,"bootstrap_unit":"whole episode","bootstrap_fits":50,"online_ensemble_size":10,"trajectories":2000,"upper_score":"max_h (T_true-m)/max(s,epsilon)","q_plus":q_plus,"q_minus_for_width_reporting":q_minus,"q_plus_rank":rank_plus,"q_minus_rank":rank_minus,"T_max_k":T_MAX_K,"safe_rule":"max_h U_conf(h) <= 313 K"},"population_seed_blocks":ROLE_SEEDS,"model_selection":selection,"method_frozen_before_final_generation_ns":frozen_at_ns,"verdict_rule":{"successful":"joint false-safe <=.05, upper coverage >=.90, pooled SAFE >=.15, each fault SAFE >=.05","conservative":"safety gates pass but usefulness gate fails","inadequate":"safety gates fail"},"verdict":decision},indent=2),encoding="utf-8")
    return decision


def _read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def refresh_existing(config: Config) -> str:
    """Rebuild evaluator summaries from frozen final artifacts without new decisions."""
    output=Path(config.output_dir); raw=_read_csv(output/"mixed_degradation_test_episodes.csv"); final_rows=[]
    integer_fields={"random_seed","actual_safe","actual_unsafe","nominal_dt_safe","nominal_dt_false_safe","certified_safe","false_safe","upper_trajectory_covered","chronology_verified"}
    float_fields={"proposed_action_lps","actual_max_temperature_k","severity_value_evaluator_only","fault_onset_s_evaluator_only","initial_volume_perturbation_l_evaluator_only","initial_concentration_scale_evaluator_only","initial_temperature_perturbation_k_evaluator_only","mean_conformal_width_k","median_conformal_width_k","prediction_RMSE_k","true_safety_margin_k","conformal_safety_margin_k","online_latency_seconds"}
    for source in raw:
        row=dict(source)
        for field in integer_fields: row[field]=int(float(row[field]))
        for field in float_fields: row[field]=float(row[field])
        with np.load(output/"final_predictions"/f"{row['episode_id']}.npz") as pred: nominal=np.asarray(pred["nominal_temperature_k"],float)
        with np.load(output/"revealed_final_test"/f"{row['episode_id']}.npz") as truth: actual=np.asarray(truth["actual_physical_temperature_k"],float)
        row["nominal_upper_trajectory_covered"]=int(np.all(actual<=nominal)); final_rows.append(row)
    _write_csv(output/"mixed_degradation_test_episodes.csv",final_rows)
    configuration=json.loads((output/"configuration.json").read_text(encoding="utf-8")); q_plus=float(configuration["method"]["q_plus"]); q_minus=float(configuration["method"]["q_minus_for_width_reporting"]); selection=configuration["model_selection"]
    summary=[*original_rows(Path("artifacts/cstr/primary_evaluation/safety_summary.csv")),metric_row(final_rows,"nominal_DT"),metric_row(final_rows,"proposed_method")]; fault_rows,severity_rows=subgroup_rows(final_rows,q_plus); _write_csv(output/"mixed_degradation_summary.csv",summary); _write_csv(output/"mixed_degradation_by_fault.csv",fault_rows); _write_csv(output/"mixed_degradation_by_severity.csv",severity_rows)
    calibration_rows=_read_csv(output/"calibration_scores.csv"); calibration_rows=[{**r,"upper_score":float(r["upper_score"]),"lower_score":float(r["lower_score"])} for r in calibration_rows]
    decision=verdict(summary,fault_rows); create_figures(output,summary,fault_rows,severity_rows,final_rows,calibration_rows); create_reports(output,config,summary,fault_rows,severity_rows,q_plus,q_minus,int(configuration["method"]["q_plus_rank"]),selection,decision); configuration["verdict"]=decision; (output/"configuration.json").write_text(json.dumps(configuration,indent=2),encoding="utf-8"); return decision


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir",default=Config.output_dir); p.add_argument("--workers",type=int,default=Config.workers); p.add_argument("--refresh-existing",action="store_true"); args=p.parse_args(); refresh=args.refresh_existing; delattr(args,"refresh_existing"); config=Config(**vars(args))
    print(refresh_existing(config) if refresh else run(config))
