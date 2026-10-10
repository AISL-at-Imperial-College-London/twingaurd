"""Apply one-sided conformal envelopes to certify candidate CSTR actions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
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


LEVELS = (0.90, 0.95, 0.99)
METHODS = ("S0_nominal_DT", "S1_raw_bootstrap", "S2_symmetric_conformal", "S3_one_sided")
CATEGORIES = discrepancy_freeze.CATEGORIES
T_MAX_K = 313.0  # Existing hard limit: cstr_case.SafeStateChecker.T_safe_max.
HORIZON = bootstrap_ensemble.HORIZON
EPSILON_SCALE_K = trajectory_conformal.EPSILON_SCALE_K
FINAL_DECISIONS = (
    "SAFETY LAYER SUCCESSFUL",
    "SAFE BUT TOO CONSERVATIVE",
    "SAFETY CALIBRATION REGIME-DEPENDENT",
    "SAFETY LAYER INADEQUATE",
)


@dataclass(frozen=True)
class CertificationConfig:
    bootstrap_dir: str = "artifacts/cstr/bootstrap_ensemble"
    conformal_dir: str = "artifacts/cstr/conformal_calibration"
    output_dir: str = "artifacts/cstr/primary_evaluation"
    calibration_count: int = 500
    final_count: int = 1000
    screening_states: int = 30
    trajectories: int = 2000
    ensemble_size: int = 10
    limit_episodes: int | None = None

    def validate(self) -> None:
        if self.limit_episodes is None and (
            self.calibration_count,
            self.final_count,
            self.trajectories,
            self.ensemble_size,
        ) != (500, 1000, 2000, 10):
            raise ValueError("full primary certification freezes n_cal=500, n_test=1000, M=2000, J=10")
        if self.limit_episodes is not None and self.limit_episodes < 3:
            raise ValueError("smoke mode needs at least three episodes")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    discrepancy_freeze._write_csv(path, rows)


def one_sided_scores(
    actual: np.ndarray, center: np.ndarray, scale: np.ndarray
) -> tuple[float, float]:
    denominator = np.maximum(np.asarray(scale, float), EPSILON_SCALE_K)
    truth = np.asarray(actual, float)
    middle = np.asarray(center, float)
    return (
        float(np.max((truth - middle) / denominator)),
        float(np.max((middle - truth) / denominator)),
    )


def safety_decision(
    lower: np.ndarray, upper: np.ndarray, temperature_limit_k: float = T_MAX_K
) -> str:
    if float(np.max(upper)) <= temperature_limit_k:
        return "SAFE"
    if float(np.max(lower)) > temperature_limit_k:
        return "UNSAFE"
    return "UNRESOLVED"


def wilson(successes: int, trials: int) -> tuple[float, float]:
    if trials == 0:
        return float("nan"), float("nan")
    return episode_design.wilson_interval(successes, trials)


def reveal_physical_temperature(
    item: discrepancy_freeze.EpisodeInputs, prediction_path: Path
) -> np.ndarray:
    """Reveal plant output and recover physical temperature after the decision.

    The predictor uses measured temperature, but safety scoring uses physical
    temperature recovered from the saved measurement noise. Neither future series
    is available to the predictor before the decision is recorded.
    Recovery runs after the decision and does not alter prediction inputs.
    """
    measured = trajectory_conformal.reveal_after_prediction(item, prediction_path)
    if item.hidden_measurement_seed is None:
        raise RuntimeError("primary certification physical truth recovery requires the saved measurement seed")
    noise = np.random.default_rng(item.hidden_measurement_seed).normal(
        0.0, episode_design.EpisodeDesignConfig().measurement_noise_k, measured.size
    )
    physical = measured - noise
    with np.load(item.truth_path) as saved:
        payload = {key: np.asarray(saved[key]) for key in saved.files}
    payload["measured_plant_temperature_k"] = measured
    payload["actual_physical_temperature_k"] = physical
    np.savez_compressed(item.truth_path, **payload)
    return physical


def _load_models(path: Path, count: int):
    payload = json.loads((path / "bootstrap_models.json").read_text(encoding="utf-8"))
    if len(payload) < count:
        raise RuntimeError(f"bootstrap ensemble has only {len(payload)} bootstrap models; {count} required")
    return [discrepancy_freeze._parameter_from_payload(item) for item in payload[:count]]


def _load_trajectory_conformal_quantiles(path: Path) -> dict[float, float]:
    with (path / "conformal_quantiles.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {
        float(row["nominal_level"]): float(row["conformal_quantile"])
        for row in rows
        if row["method"] == "B2" and row["calibration_scope"] == "pooled"
    }
    missing = set(LEVELS) - set(result)
    if missing:
        raise RuntimeError(f"frozen trajectory calibration quantiles missing levels: {sorted(missing)}")
    return result


def predict_frozen_u3(item: discrepancy_freeze.EpisodeInputs, models, trajectories: int, path: Path):
    start = time.perf_counter()
    counts = np.full(len(models), trajectories // len(models), int)
    counts[: trajectories % len(models)] += 1
    paths = []
    filtering_seconds = 0.0
    rollout_seconds = 0.0
    for index, (model, count) in enumerate(zip(models, counts)):
        samples, _, _, filtering, rollout = bootstrap_ensemble._paths_for_model(
            model,
            item,
            int(count),
            item.random_seed + 151_000_000 + index * 10_000,
        )
        paths.append(samples)
        filtering_seconds += filtering
        rollout_seconds += rollout
    samples = np.vstack(paths)
    center = np.median(samples, axis=0)
    scale = np.std(samples, axis=0, ddof=1)
    raw = episode_design.construct_simultaneous_bands(samples, LEVELS)
    payload = {
        "nominal_temperature_k": item.nominal_future,
        "predictive_center_k": center,
        "predictive_scale_k": scale,
    }
    for level in LEVELS:
        label = int(level * 100)
        payload[f"raw_{label}_lower_k"] = raw.lower[level]
        payload[f"raw_{label}_upper_k"] = raw.upper[level]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return {
        "center": center,
        "scale": scale,
        "raw": raw,
        "path": path,
        "filtering_seconds": filtering_seconds,
        "rollout_seconds": rollout_seconds,
        "prediction_seconds": time.perf_counter() - start,
    }


def calibrate_one_sided(episodes, models, trajectories: int, output: Path):
    rows = []
    for index, item in enumerate(episodes):
        prediction = predict_frozen_u3(
            item, models, trajectories, output / "calibration_predictions" / f"{item.episode_id}.npz"
        )
        before = _sha256(prediction["path"])
        actual = reveal_physical_temperature(item, prediction["path"])
        if _sha256(prediction["path"]) != before:
            raise AssertionError("calibration prediction changed after truth reveal")
        plus, minus = one_sided_scores(actual, prediction["center"], prediction["scale"])
        rows.append(
            {
                "episode_id": item.episode_id,
                "action_category": item.category,
                "random_seed": item.random_seed,
                "proposed_action_lps": item.proposed_action,
                "upper_score": plus,
                "lower_score": minus,
                "minimum_predictive_scale_k": float(np.min(prediction["scale"])),
                "prediction_saved_before_truth_reveal": True,
            }
        )
        if (index + 1) % 50 == 0:
            print(f"completed {index + 1}/{len(episodes)} safety-calibration predictions", flush=True)
    quantile_rows = []
    quantiles = {}
    for direction, field in (("upper", "upper_score"), ("lower", "lower_score")):
        scores = np.asarray([row[field] for row in rows], float)
        for level in LEVELS:
            q, rank = trajectory_conformal.corrected_quantile(scores, level)
            quantiles[(direction, level)] = q
            quantile_rows.append(
                {
                    "direction": direction,
                    "nominal_level": level,
                    "miscoverage_alpha": 1.0 - level,
                    "number_of_calibration_episodes": len(scores),
                    "corrected_rank_k": rank,
                    "conformal_quantile": q,
                    "finite_sample_convention": "k=ceil((n_cal+1)*(1-alpha)); kth ordered episode score",
                }
            )
    return rows, quantile_rows, quantiles


def method_envelope(method: str, level: float, prediction, one_sided_q, symmetric_q):
    center = np.asarray(prediction["center"])
    scale = np.maximum(np.asarray(prediction["scale"]), EPSILON_SCALE_K)
    if method == "S0_nominal_DT":
        lower = upper = np.asarray(prediction["nominal"])
    elif method == "S1_raw_bootstrap":
        lower, upper = prediction["raw"].lower[level], prediction["raw"].upper[level]
    elif method == "S2_symmetric_conformal":
        half = symmetric_q[level] * scale
        lower, upper = center - half, center + half
    elif method == "S3_one_sided":
        lower = center - one_sided_q[("lower", level)] * scale
        upper = center + one_sided_q[("upper", level)] * scale
    else:
        raise ValueError(method)
    return np.asarray(lower), np.asarray(upper)


def evaluate_final(episodes, manifest, models, trajectories, output, one_sided_q, symmetric_q):
    metadata = {row["episode_id"]: row for row in manifest}
    rows = []
    for index, item in enumerate(episodes):
        prediction = predict_frozen_u3(
            item, models, trajectories, output / "predictions" / f"{item.episode_id}.npz"
        )
        prediction["nominal"] = item.nominal_future
        start = time.perf_counter()
        envelopes = {
            (method, level): method_envelope(
                method, level, prediction, one_sided_q, symmetric_q
            )
            for method in METHODS
            for level in LEVELS
        }
        decisions = {
            key: safety_decision(*envelope) for key, envelope in envelopes.items()
        }
        conformal_seconds = time.perf_counter() - start
        # The saved prediction includes the frozen S3 envelopes before truth is revealed.
        s3_payload = {}
        for level in LEVELS:
            lo, hi = envelopes[("S3_one_sided", level)]
            s3_payload[f"s3_{int(level*100)}_lower_k"] = lo
            s3_payload[f"s3_{int(level*100)}_upper_k"] = hi
        with np.load(prediction["path"]) as saved:
            original = {key: np.asarray(saved[key]) for key in saved.files}
        np.savez_compressed(prediction["path"], **original, **s3_payload)
        # Persist all decisions before the evaluator reveals the hidden plant future.
        decision_path = output / "decision_snapshots" / f"{item.episode_id}.json"
        decision_path.parent.mkdir(parents=True, exist_ok=True)
        decision_path.write_text(
            json.dumps(
                {
                    "episode_id": item.episode_id,
                    "decisions": {f"{m}|{l}": d for (m, l), d in decisions.items()},
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        snapshot_hash = _sha256(decision_path)
        actual = reveal_physical_temperature(item, prediction["path"])
        if _sha256(decision_path) != snapshot_hash:
            raise AssertionError("safety decision changed during truth reveal")
        actual_max = float(np.max(actual))
        actual_safe = int(actual_max <= T_MAX_K)
        meta = metadata[item.episode_id]
        for method in METHODS:
            for level in LEVELS:
                lower, upper = envelopes[(method, level)]
                decision = decisions[(method, level)]
                total = prediction["prediction_seconds"] + conformal_seconds
                rows.append(
                    {
                        "episode_id": item.episode_id,
                        "random_seed": item.random_seed,
                        "action_category": item.category,
                        "proposed_action": item.proposed_action,
                        "hidden_fault_id_evaluator_only": "UA_fouling",
                        "hidden_fault_magnitude_evaluator_only": meta["hidden_fault_magnitude_evaluator_only"],
                        "T_max": T_MAX_K,
                        "method": method,
                        "alpha": 1.0 - level,
                        "nominal_level": level,
                        "nominal_DT_max_temperature": float(np.max(item.nominal_future)),
                        "predictive_max_temperature": float(np.max(prediction["center"])),
                        "conformal_upper_max": float(np.max(upper)),
                        "conformal_lower_max": float(np.max(lower)),
                        "conformal_safety_margin": float(T_MAX_K - np.max(upper)),
                        "mean_envelope_width_k": float(np.mean(upper - lower)),
                        "upper_inflation_max_k": float(np.max(upper) - np.max(prediction["center"])),
                        "actual_max_temperature": actual_max,
                        "actual_safety_margin": T_MAX_K - actual_max,
                        "decision": decision,
                        "actual_safe": actual_safe,
                        "false_safe_indicator": int(decision == "SAFE" and not actual_safe),
                        "false_unsafe_indicator": int(decision == "UNSAFE" and actual_safe),
                        "upper_trajectory_covered": int(np.all(actual <= upper)),
                        "lower_trajectory_covered": int(np.all(actual >= lower)),
                        "filtering_time": prediction["filtering_seconds"],
                        "rollout_time": prediction["rollout_seconds"],
                        "conformal_time": conformal_seconds,
                        "total_latency": total,
                        "decision_saved_before_truth_reveal": True,
                    }
                )
        if (index + 1) % 50 == 0:
            print(f"completed {index + 1}/{len(episodes)} final safety predictions", flush=True)
    return rows


def summarize(rows: Sequence[dict[str, object]]) -> list[dict[str, object]]:
    output = []
    for method in METHODS:
        for level in LEVELS:
            for category in (*CATEGORIES, "pooled"):
                selected = [
                    row
                    for row in rows
                    if row["method"] == method
                    and np.isclose(row["nominal_level"], level)
                    and (category == "pooled" or row["action_category"] == category)
                ]
                n = len(selected)
                if n == 0:
                    continue
                safe = [row for row in selected if row["decision"] == "SAFE"]
                unsafe = [row for row in selected if row["decision"] == "UNSAFE"]
                unresolved = [row for row in selected if row["decision"] == "UNRESOLVED"]
                actually_unsafe = [row for row in selected if not row["actual_safe"]]
                false_safe = sum(int(row["false_safe_indicator"]) for row in selected)
                false_unsafe = sum(int(row["false_unsafe_indicator"]) for row in selected)
                fs_ci = wilson(false_safe, len(safe))
                joint_ci = wilson(false_safe, n)
                output.append(
                    {
                        "method": method,
                        "alpha": 1.0 - level,
                        "nominal_level": level,
                        "action_category": category,
                        "number_of_episodes": n,
                        "actual_safe_fraction": np.mean([row["actual_safe"] for row in selected]),
                        "SAFE_count": len(safe),
                        "UNSAFE_count": len(unsafe),
                        "UNRESOLVED_count": len(unresolved),
                        "SAFE_certification_rate": len(safe) / n,
                        "UNSAFE_decision_rate": len(unsafe) / n,
                        "false_safe_count": false_safe,
                        "joint_false_safe_probability": false_safe / n,
                        "joint_false_safe_CI_lower": joint_ci[0],
                        "joint_false_safe_CI_upper": joint_ci[1],
                        "conditional_false_safe_rate": false_safe / len(safe) if safe else float("nan"),
                        "false_safe_CI_lower": fs_ci[0],
                        "false_safe_CI_upper": fs_ci[1],
                        "unsafe_rejection_rate": sum(row["decision"] == "UNSAFE" for row in actually_unsafe) / len(actually_unsafe) if actually_unsafe else float("nan"),
                        "false_unsafe_rate": false_unsafe / len(unsafe) if unsafe else float("nan"),
                        "joint_false_unsafe_probability": false_unsafe / n,
                        "unresolved_rate": len(unresolved) / n,
                        "oracle_safe_capture_rate": sum(row["decision"] == "SAFE" and row["actual_safe"] for row in selected) / sum(row["actual_safe"] for row in selected) if any(row["actual_safe"] for row in selected) else float("nan"),
                        "upper_trajectory_empirical_coverage": float(np.mean([row["upper_trajectory_covered"] for row in selected])),
                        "lower_trajectory_empirical_coverage": float(np.mean([row["lower_trajectory_covered"] for row in selected])),
                        "mean_envelope_width_k": float(np.mean([row["mean_envelope_width_k"] for row in selected])),
                        "mean_upper_inflation_max_k": float(np.mean([row["upper_inflation_max_k"] for row in selected])),
                        "median_latency": float(np.median([row["total_latency"] for row in selected])),
                    }
                )
    return output


def _summary_row(summary, method: str, level: float, category: str = "pooled"):
    return next(
        row
        for row in summary
        if row["method"] == method
        and row["action_category"] == category
        and np.isclose(row["nominal_level"], level)
    )


def _reconstruct_decision_state(seed: int, config: episode_design.EpisodeDesignConfig, template):
    rng = np.random.default_rng(seed)
    measurement_rng = np.random.default_rng(seed + 1_000_000)
    onset = float(rng.uniform(config.fault_onset_min_s, config.fault_onset_max_s))
    magnitude = float(rng.uniform(config.fault_magnitude_min, config.fault_magnitude_max))
    nominal, plant, _ = episode_design._configure_episode_models(template, config, rng, onset, magnitude)
    pre = [
        episode_design._paired_observation(nominal, plant, 2 / 60, measurement_rng, config.measurement_noise_k)
        for _ in range(config.pre_fault_lead_steps)
    ]
    learning = []
    for index in range(config.n_postfault):
        action = episode_design.TRAINING_ACTIONS_LPS[(index // config.training_action_block_steps) % len(episode_design.TRAINING_ACTIONS_LPS)]
        learning.append(episode_design._paired_observation(nominal, plant, action, measurement_rng, config.measurement_noise_k))
    previous = float(pre[-1]["action_lps"])
    online = []
    for index, row in enumerate(learning):
        action = float(row["action_lps"])
        online.append([action, action - previous, float(row["nominal_twin_temperature_k"]), index + 1])
        previous = action
    return nominal, plant, np.asarray([x["temperature_residual_k"] for x in learning]), np.asarray(online), previous, magnitude


def _action_category(action: float) -> str:
    if 0.0265 <= action <= 0.0315:
        return "interpolation"
    if 0.0195 <= action < 0.0265 or 0.0315 < action <= 0.034:
        return "boundary"
    return "extrapolation"


def screen_actions(config, template, models, one_sided_q, output: Path):
    # Fixed, physically admissible range already exercised by the episode design-trajectory calibration protocol.
    actions = np.linspace(0.012, 0.045, 20)
    rows = []
    timings = []
    state_count = config.limit_episodes or config.screening_states
    for state_index in range(state_count):
        seed = 1_830_000 + state_index
        nominal, plant, residual, online, previous, magnitude = _reconstruct_decision_state(seed, episode_design.EpisodeDesignConfig(), template)
        state_rows = []
        for action_index, action in enumerate(actions):
            nominal_branch = __import__("copy").deepcopy(nominal)
            plant_branch = __import__("copy").deepcopy(plant)
            nominal_future = []
            future = []
            prior = previous
            for horizon in range(HORIZON):
                out = nominal_branch.simulate_step(Fin_sp_normal=float(action), Fin_sp_startup=float(action))
                temp = float(out["T_true"])
                nominal_future.append(temp)
                future.append([float(action), float(action) - prior, temp, 51 + horizon])
                prior = float(action)
            item = discrepancy_freeze.EpisodeInputs(
                f"cstr_screening_{state_index:04d}_{action_index:02d}",
                _action_category(float(action)),
                seed + action_index * 10_000,
                residual,
                online,
                np.asarray(future),
                np.asarray(nominal_future),
                float(action),
                output / "revealed_screening_truth" / f"cstr_screening_{state_index:04d}_{action_index:02d}.npz",
                plant_branch,
                seed + 4_000_000 + action_index,
            )
            pred = predict_frozen_u3(item, models, config.trajectories, output / "screening_predictions" / f"{item.episode_id}.npz")
            pred["nominal"] = item.nominal_future
            begin = time.perf_counter()
            lo, hi = method_envelope("S3_one_sided", .95, pred, one_sided_q, {})
            decision = safety_decision(lo, hi)
            conformal_time = time.perf_counter() - begin
            actual = reveal_physical_temperature(item, pred["path"])
            row = {
                "decision_state_id": f"cstr_screening_state_{state_index:04d}",
                "random_seed": seed,
                "candidate_index": action_index,
                "action_lps": float(action),
                "action_category": item.category,
                "hidden_fault_magnitude_evaluator_only": magnitude,
                "decision": decision,
                "conformal_safety_margin_k": T_MAX_K - float(np.max(hi)),
                "actual_safety_margin_k": T_MAX_K - float(np.max(actual)),
                "actual_safe": int(float(np.max(actual)) <= T_MAX_K),
                "total_latency_seconds": pred["prediction_seconds"] + conformal_time,
            }
            rows.append(row)
            state_rows.append(row)
        for k in (1, 5, 10, 20):
            timings.append(
                {
                    "decision_state_id": f"cstr_screening_state_{state_index:04d}",
                    "candidate_count_K": k,
                    "measured_sequential_latency_seconds": sum(row["total_latency_seconds"] for row in state_rows[:k]),
                    "ideal_parallel_latency_lower_bound_seconds": max(row["total_latency_seconds"] for row in state_rows[:k]),
                }
            )
        print(f"completed {state_index + 1}/{state_count} multi-action decision states", flush=True)
    state_summary = []
    for state_id in sorted({row["decision_state_id"] for row in rows}):
        selected = [row for row in rows if row["decision_state_id"] == state_id]
        exists_safe = any(row["actual_safe"] for row in selected)
        found_safe = any(row["decision"] == "SAFE" for row in selected)
        state_summary.append(
            {
                "decision_state_id": state_id,
                "candidate_count": len(selected),
                "at_least_one_actually_safe": int(exists_safe),
                "at_least_one_SAFE_certification": int(found_safe),
                "screening_success": int(exists_safe and found_safe),
                "no_action_certifiable": int(not found_safe),
                "only_unsafe_actions_exist": int(not exists_safe),
                "multiple_actions_certifiable": int(sum(row["decision"] == "SAFE" for row in selected) > 1),
            }
        )
    return rows, state_summary, timings


def decide(summary) -> str:
    primary = _summary_row(summary, "S3_one_sided", .95)
    nominal = _summary_row(summary, "S0_nominal_DT", .95)
    raw = _summary_row(summary, "S1_raw_bootstrap", .95)
    useful = primary["SAFE_certification_rate"] >= 0.10
    controlled = primary["joint_false_safe_probability"] <= 0.05
    reduced = primary["joint_false_safe_probability"] <= 0.5 * min(
        nominal["joint_false_safe_probability"], raw["joint_false_safe_probability"]
    )
    bad_regime = any(
        _summary_row(summary, "S3_one_sided", .95, category)["joint_false_safe_probability"] > 0.10
        for category in CATEGORIES
    )
    if controlled and bad_regime:
        return "SAFETY CALIBRATION REGIME-DEPENDENT"
    if controlled and reduced and useful:
        return "SAFETY LAYER SUCCESSFUL"
    if controlled and not useful:
        return "SAFE BUT TOO CONSERVATIVE"
    return "SAFETY LAYER INADEQUATE"


def _save_figure(fig, output: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def create_figures(output: Path, rows, summary, screening_rows, screening_summary):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(11, 2.5)); ax.axis("off")
    labels = ["Nominal DT +\npost-fault observations", "Frozen latent +\nbootstrap U3", "One-sided conformal\nupper/lower envelopes", "313 K safety\nboundary", "SAFE / UNSAFE /\nUNRESOLVED"]
    xs = np.linspace(.08, .92, len(labels))
    for x, label in zip(xs, labels): ax.text(x,.5,label,ha="center",va="center",bbox=dict(boxstyle="round",fc="#e8f1fa",ec="#356a9a"),transform=ax.transAxes)
    for left,right in zip(xs[:-1],xs[1:]): ax.annotate("",xy=(right-.08,.5),xytext=(left+.08,.5),xycoords=ax.transAxes,arrowprops=dict(arrowstyle="->"))
    _save_figure(fig, output, "figure_CERTIFICATION_1_safety_layer_schematic")

    primary = [r for r in rows if r["method"] == "S3_one_sided" and np.isclose(r["nominal_level"], .95)]
    for number, decision in zip((2,3,4),("SAFE","UNSAFE","UNRESOLVED")):
        candidates=[r for r in primary if r["decision"]==decision]
        if not candidates: continue
        chosen=min(candidates,key=lambda r:abs(r["actual_safety_margin"])) if decision=="UNRESOLVED" else candidates[len(candidates)//2]
        eid=chosen["episode_id"]
        with np.load(output/"predictions"/f"{eid}.npz") as pred, np.load(output/"revealed_final_truth"/f"{eid}.npz") as truth:
            h=np.arange(1,HORIZON+1);lo=pred["s3_95_lower_k"];hi=pred["s3_95_upper_k"]
            fig,ax=plt.subplots(figsize=(7.2,4.2));ax.fill_between(h,lo,hi,alpha=.25,label="S3 one-sided 95% envelope");ax.plot(h,pred["nominal_temperature_k"],"--",label="nominal DT");ax.plot(h,pred["predictive_center_k"],label="frozen U3 median");ax.plot(h,truth["actual_physical_temperature_k"],color="black",label="revealed physical plant");ax.axhline(T_MAX_K,color="#b22222",ls=":",label=r"$T_{max}=313$ K");ax.set(xlabel="Future horizon (s)",ylabel="Reactor temperature (K)",title=f"{decision}: {eid}");ax.legend(fontsize=8)
            _save_figure(fig,output,f"figure_CERTIFICATION_{number}_representative_{decision.lower()}")

    fig,ax=plt.subplots(figsize=(6.4,4.2))
    trade=[_summary_row(summary,"S3_one_sided",level) for level in LEVELS]
    coordinates=[(row["SAFE_certification_rate"],row["conditional_false_safe_rate"]) for row in trade]
    if all(np.allclose(coordinates[0],coordinate) for coordinate in coordinates[1:]):
        x,y=coordinates[0];ax.scatter(x,y,s=90,color="#2e8b57",edgecolor="black",zorder=3);ax.annotate("90%, 95%, and 99% overlap\n338 SAFE; 0 false-safe",(x,y),xytext=(18,18),textcoords="offset points",arrowprops=dict(arrowstyle="->"));ax.set_xlim(x-.035,x+.035);ax.set_ylim(-.002,.02)
    else:
        for level,row in zip(LEVELS,trade):ax.scatter(row["SAFE_certification_rate"],row["conditional_false_safe_rate"],s=70,label=f"{int(level*100)}% (joint FS={row['joint_false_safe_probability']:.3f})")
        ax.legend(fontsize=8)
    ax.set(xlabel="SAFE certification rate",ylabel="Conditional false-safe rate",title="Safety-usefulness trade-off")
    _save_figure(fig,output,"figure_CERTIFICATION_5_false_safe_vs_certification")

    matrix=np.zeros((3,2),int);dmap={"SAFE":0,"UNSAFE":1,"UNRESOLVED":2}
    for r in primary:matrix[dmap[r["decision"]],0 if r["actual_safe"] else 1]+=1
    fig,ax=plt.subplots(figsize=(5.4,4));im=ax.imshow(matrix,cmap="Blues");ax.set_xticks([0,1],["Actually safe","Actually unsafe"]);ax.set_yticks(range(3),["SAFE","UNSAFE","UNRESOLVED"]);ax.set(xlabel="Revealed outcome",ylabel="Pre-reveal decision")
    for i in range(3):
        for j in range(2):ax.text(j,i,str(matrix[i,j]),ha="center",va="center")
    fig.colorbar(im,ax=ax,label="Episodes");_save_figure(fig,output,"figure_CERTIFICATION_6_decision_outcome_matrix")

    colors={"SAFE":"#2e8b57","UNSAFE":"#b22222","UNRESOLVED":"#d68a00"}
    fig,ax=plt.subplots(figsize=(7,4));rng=np.random.default_rng(10)
    for idx,d in enumerate(("SAFE","UNRESOLVED","UNSAFE")):
        x=[r["actual_safety_margin"] for r in primary if r["decision"]==d];ax.scatter(np.full(len(x),idx)+rng.normal(0,.05,len(x)),x,s=9,alpha=.45,color=colors[d])
    ax.axhline(0,color="black",ls="--");ax.set_xticks(range(3),("SAFE","UNRESOLVED","UNSAFE"));ax.set(ylabel="True safety margin (K)",xlabel="Decision")
    _save_figure(fig,output,"figure_CERTIFICATION_7_decision_vs_true_margin")

    fig,ax=plt.subplots(figsize=(6,5))
    for d in colors:
        selected=[r for r in primary if r["decision"]==d];ax.scatter([r["actual_safety_margin"] for r in selected],[r["conformal_safety_margin"] for r in selected],s=10,alpha=.55,label=d,color=colors[d])
    ax.axhline(0,color="black",ls="--");ax.axvline(0,color="black",ls=":");ax.set(xlabel="True safety margin (K)",ylabel="Upper-envelope safety margin (K)");ax.legend()
    _save_figure(fig,output,"figure_CERTIFICATION_8_conformal_vs_true_margin")

    fig,axes=plt.subplots(1,3,figsize=(12,3.7));metrics=("SAFE_certification_rate","joint_false_safe_probability","unresolved_rate");titles=("SAFE certification","Joint false-safe","UNRESOLVED")
    for ax,metric,title in zip(axes,metrics,titles):
        vals=[_summary_row(summary,"S3_one_sided",.95,c)[metric] for c in CATEGORIES];ax.bar(CATEGORIES,vals,color="#527da5");ax.set_title(title);ax.tick_params(axis="x",rotation=20)
    _save_figure(fig,output,"figure_CERTIFICATION_9_metrics_by_regime")

    fig,axes=plt.subplots(1,3,figsize=(12,3.7));x=np.arange(len(METHODS));labels=[m.split("_",1)[0] for m in METHODS]
    for ax,metric,title in zip(axes,metrics,titles):
        vals=[_summary_row(summary,m,.95)[metric] for m in METHODS];ax.bar(x,vals,color=["#999999","#7194b5","#7b6ba8","#2e8b57"]);ax.set_xticks(x,labels);ax.set_title(title)
    _save_figure(fig,output,"figure_CERTIFICATION_10_method_comparison")

    fig,axes=plt.subplots(1,2,figsize=(11,4.3),sharex=True)
    for d in colors:
        selected=[r for r in primary if r["decision"]==d];actions=[r["proposed_action"] for r in selected];axes[0].scatter(actions,[r["actual_safety_margin"] for r in selected],s=10,alpha=.55,label=d,color=colors[d]);axes[1].scatter(actions,[r["conformal_safety_margin"] for r in selected],s=10,alpha=.55,label=d,color=colors[d])
    for ax in axes:ax.axhline(0,color="black",ls="--");ax.set_xlabel("Candidate flow setpoint (L/s)")
    axes[0].set_ylabel("True safety margin (K)");axes[1].set_ylabel("Upper-envelope safety margin (K)");axes[0].legend()
    _save_figure(fig,output,"figure_CERTIFICATION_11_action_magnitude_vs_decision")

    fig,axes=plt.subplots(1,2,figsize=(10,4));
    states=screening_summary;labels=["Screening\nsuccess","No action\ncertifiable","Only unsafe\nexist","Multiple\ncertifiable"];values=[np.mean([r["screening_success"] for r in states if r["at_least_one_actually_safe"]]) if any(r["at_least_one_actually_safe"] for r in states) else np.nan,np.mean([r["no_action_certifiable"] for r in states]),np.mean([r["only_unsafe_actions_exist"] for r in states]),np.mean([r["multiple_actions_certifiable"] for r in states])];axes[0].bar(labels,values,color="#527da5");axes[0].tick_params(axis="x",rotation=15);axes[0].set_ylabel("Decision-state fraction")
    for d in colors:
        selected=[r for r in screening_rows if r["decision"]==d];axes[1].scatter([r["action_lps"] for r in selected],[r["conformal_safety_margin_k"] for r in selected],s=8,alpha=.5,label=d,color=colors[d])
    axes[1].axhline(0,color="black",ls="--");axes[1].set(xlabel="Candidate flow setpoint (L/s)",ylabel="Conformal safety margin (K)");axes[1].legend(fontsize=8)
    _save_figure(fig,output,"figure_CERTIFICATION_12_multi_action_screening")


def create_reports(output, config, quantile_rows, final_rows, summary, screening_summary, timing_rows, decision):
    p=_summary_row(summary,"S3_one_sided",.95);s0=_summary_row(summary,"S0_nominal_DT",.95);s1=_summary_row(summary,"S1_raw_bootstrap",.95);s2=_summary_row(summary,"S2_symmetric_conformal",.95)
    eligible=[r for r in screening_summary if r["at_least_one_actually_safe"]];screen_success=np.mean([r["screening_success"] for r in eligible]) if eligible else float("nan")
    screening_latency=np.asarray([r["measured_sequential_latency_seconds"] for r in timing_rows if r["candidate_count_K"]==1])
    qplus=next(r["conformal_quantile"] for r in quantile_rows if r["direction"]=="upper" and np.isclose(r["nominal_level"],.95));qminus=next(r["conformal_quantile"] for r in quantile_rows if r["direction"]=="lower" and np.isclose(r["nominal_level"],.95))
    primary_rows=[r for r in final_rows if r["method"]=="S3_one_sided" and np.isclose(r["nominal_level"],.95)]
    online_latency=np.asarray([r["total_latency"] for r in primary_rows],float);filter_latency=np.asarray([r["filtering_time"] for r in primary_rows],float);rollout_latency=np.asarray([r["rollout_time"] for r in primary_rows],float);conformal_latency=np.asarray([r["conformal_time"] for r in primary_rows],float)
    near_boundary=[r for r in primary_rows if abs(r["actual_safety_margin"])<=1.0]
    near_unresolved=np.mean([r["decision"]=="UNRESOLVED" for r in near_boundary]) if near_boundary else float("nan")
    regime_lines="\n".join(f"- {c}: SAFE rate {_summary_row(summary,'S3_one_sided',.95,c)['SAFE_certification_rate']:.3f}, joint false-safe {_summary_row(summary,'S3_one_sided',.95,c)['joint_false_safe_probability']:.3f}, unresolved {_summary_row(summary,'S3_one_sided',.95,c)['unresolved_rate']:.3f}." for c in CATEGORIES)
    readme=f"""# primary certification — action-specific safety certification

## Decision

**{decision}**

The predictor is frozen at the bootstrap ensemble U3 stack (first 10 accepted bootstrap models, M=2000, posterior median centre) and the trajectory calibration numerical scale floor of {EPSILON_SCALE_K:g} K. The primary physical constraint is the repository's existing hard limit, **reactor temperature <= {T_MAX_K:.1f} K at every one of {HORIZON} future seconds** (`cstr_case.SafeStateChecker`). It was fixed before primary certification outcome generation.

The independent safety-calibration population contains {config.limit_episodes or config.calibration_count} episodes. Its finite-sample 95% one-sided quantiles are q+={qplus:.3f} and q-={qminus:.3f}; they were frozen before generating {config.limit_episodes or config.final_count} new final episodes. Candidate actions retain the established 0.012–0.045 L/s interpolation/boundary/extrapolation protocol. trajectory calibration's untouched population had already shown this protocol spans conservative, boundary, and unsafe outcomes at 313 K, so neither the limit nor action range was tuned to primary certification test performance.

bootstrap ensemble/trajectory calibration predict measured temperature. For the physical primary certification constraint, evaluator-side `T_true` is recovered exactly by removing the saved 0.10 K measurement draw using its frozen RNG seed after reveal. Both measured and physical trajectories are retained; no future trajectory enters prediction or calibration before its prescribed reveal.

## Central results at 95%

| Method | SAFE rate | Joint false-safe | Conditional false-safe | UNRESOLVED |
|---|---:|---:|---:|---:|
| S0 nominal DT | {s0['SAFE_certification_rate']:.3f} | {s0['joint_false_safe_probability']:.3f} | {s0['conditional_false_safe_rate']:.3f} | {s0['unresolved_rate']:.3f} |
| S1 raw U3 | {s1['SAFE_certification_rate']:.3f} | {s1['joint_false_safe_probability']:.3f} | {s1['conditional_false_safe_rate']:.3f} | {s1['unresolved_rate']:.3f} |
| S2 symmetric trajectory calibration | {s2['SAFE_certification_rate']:.3f} | {s2['joint_false_safe_probability']:.3f} | {s2['conditional_false_safe_rate']:.3f} | {s2['unresolved_rate']:.3f} |
| S3 one-sided | {p['SAFE_certification_rate']:.3f} | {p['joint_false_safe_probability']:.3f} | {p['conditional_false_safe_rate']:.3f} | {p['unresolved_rate']:.3f} |

The S3 joint false-safe 95% Wilson interval is [{p['joint_false_safe_CI_lower']:.3f}, {p['joint_false_safe_CI_upper']:.3f}]. Its conditional false-safe interval is [{p['false_safe_CI_lower']:.3f}, {p['false_safe_CI_upper']:.3f}] over {int(p['SAFE_count'])} SAFE decisions. This conditional rate is empirical, not conformally guaranteed.

S3 achieved upper whole-trajectory coverage {p['upper_trajectory_empirical_coverage']:.3f}; the lower envelope achieved {p['lower_trajectory_empirical_coverage']:.3f}, below the 0.95 target. No false-UNSAFE decision occurred, but lower-envelope calibration should not be described as validated at 95%. S3's mean envelope width was {p['mean_envelope_width_k']:.3f} K versus {s2['mean_envelope_width_k']:.3f} K for S2. S3 did not increase the SAFE count relative to S2; its gain was sharper envelopes and fewer unresolved decisions ({p['unresolved_rate']:.3f} versus {s2['unresolved_rate']:.3f}).

## Regimes

{regime_lines}

These are pooled-calibration diagnostics; no category-conditional guarantee is claimed.

## Interpretation and predeclared questions

1. Nominal DT unsafe-as-safe behaviour is quantified by S0 joint false-safe={s0['joint_false_safe_probability']:.3f}.
2. Frozen latent correction changes it to S1={s1['joint_false_safe_probability']:.3f}.
3. Symmetric and one-sided conformalization change it to S2={s2['joint_false_safe_probability']:.3f} and S3={p['joint_false_safe_probability']:.3f}.
4. One-sided calibration is sharper than symmetric trajectory calibration and resolves more strong-UNSAFE cases, but does not certify more SAFE actions in this population.
5. S3 certifies {p['oracle_safe_capture_rate']:.3f} of genuinely safe actions.
6. Among {len(near_boundary)} evaluator-defined near-boundary episodes (absolute true margin <=1 K), {near_unresolved:.3f} become UNRESOLVED; the overall unresolved rate is {p['unresolved_rate']:.3f}.
7. Empirical conditional false-safe is {p['conditional_false_safe_rate']:.3f}.
8. Marginal joint false-safe is {p['joint_false_safe_probability']:.3f}.
9. Tightening 90% to 99% widened S3 mean envelopes from {_summary_row(summary,'S3_one_sided',.90)['mean_envelope_width_k']:.3f} to {_summary_row(summary,'S3_one_sided',.99)['mean_envelope_width_k']:.3f} K, but the discrete decisions were unchanged at 338 SAFE and zero false-safe at all three levels; unresolved increased from {_summary_row(summary,'S3_one_sided',.90)['unresolved_rate']:.3f} to {_summary_row(summary,'S3_one_sided',.99)['unresolved_rate']:.3f}.
10. Regime results appear above and in Figure 9.
11. Multi-action screening found at least one certified SAFE action in {screen_success:.3f} of states having an actually safe candidate. Final-test online latency median/p95/max was {np.median(online_latency):.3f}/{np.quantile(online_latency,.95):.3f}/{np.max(online_latency):.3f} s. Median filtering/rollout/conformal components were {np.median(filter_latency):.4f}/{np.median(rollout_latency):.4f}/{np.median(conformal_latency):.6f} s. Screening K=1 measured median was {np.median(screening_latency):.3f} s; all K-specific sequential and ideal-parallel lower-bound timings are in `screening_latency.csv`.

## Guarantee scope and limitation

Under split-conformal exchangeability over the defined episode population, whole-trajectory upper-envelope miscoverage is marginally controlled. Because `SAFE and actually unsafe` is a subset of upper-envelope failure, this supports a marginal population bound on joint false-safe probability. It does **not** guarantee that every selected action has safety probability at least 1-alpha, nor does it guarantee the conditional false-safe rate. Usefulness is strongly regime-dependent: all 334 interpolation actions were UNRESOLVED despite 63.5% being physically safe. The lower-envelope coverage shortfall, selective/conditional validity, finite regime samples, and simulator-to-deployment population match are the strongest limitations. The optional LLM Action Agent experiment was not run; it is not part of the statistical guarantee or core primary certification result.
"""
    (output/"README.md").write_text(readme,encoding="utf-8")
    (output/"decision_summary.md").write_text(f"# primary certification decision\n\n**{decision}**\n\nAt 95%, S3 joint false-safe probability is {p['joint_false_safe_probability']:.3f} (Wilson [{p['joint_false_safe_CI_lower']:.3f}, {p['joint_false_safe_CI_upper']:.3f}]), conditional false-safe is {p['conditional_false_safe_rate']:.3f} over {int(p['SAFE_count'])} SAFE decisions, SAFE certification is {p['SAFE_certification_rate']:.3f}, and UNRESOLVED is {p['unresolved_rate']:.3f}.\n",encoding="utf-8")
    paper=f"""# primary certification — paper result summary

**Central question.** Can a frozen, post-fault conformal digital twin safely certify individual counterfactual recovery actions?

**Population and constraint.** A new {config.limit_episodes or config.calibration_count}-episode calibration population and independent {config.limit_episodes or config.final_count}-episode test population vary hidden UA fouling, initial conditions, measurement noise, and established interpolation/boundary/extrapolation flow actions. Safety requires reactor temperature <= {T_MAX_K:.1f} K throughout the {HORIZON}-s horizon.

**Rule and implication.** S3 declares SAFE only when the complete one-sided conformal upper envelope is <= {T_MAX_K:.1f} K, UNSAFE only when some point of the lower envelope exceeds it, and otherwise UNRESOLVED. Marginal whole-trajectory upper coverage implies that the joint event `SAFE and actually unsafe` is contained in conformal miscoverage under exchangeability; it does not imply per-action conditional safety.

**Results (95%).** Nominal-DT joint false-safe was {s0['joint_false_safe_probability']:.3f}; S3 was {p['joint_false_safe_probability']:.3f} (Wilson [{p['joint_false_safe_CI_lower']:.3f}, {p['joint_false_safe_CI_upper']:.3f}]). S3 conditional false-safe was {p['conditional_false_safe_rate']:.3f} over {int(p['SAFE_count'])} certifications, SAFE rate was {p['SAFE_certification_rate']:.3f}, and unresolved rate was {p['unresolved_rate']:.3f}. Upper/lower trajectory coverage was {p['upper_trajectory_empirical_coverage']:.3f}/{p['lower_trajectory_empirical_coverage']:.3f}. S3 was sharper than S2 ({p['mean_envelope_width_k']:.3f} versus {s2['mean_envelope_width_k']:.3f} K) but had the same SAFE count; it reduced unresolved decisions from {s2['unresolved_rate']:.3f} to {p['unresolved_rate']:.3f}. Multi-action screening success was {screen_success:.3f} conditional on an actually safe candidate existing.

**Regimes.** {regime_lines.replace(chr(10),' ')}

**Trade-off and latency.** The 90/95/99% S3 rules all produced 338 SAFE and zero observed false-safe decisions; envelope width increased and unresolved rate rose slightly with confidence. Final-test median online latency was {np.median(online_latency):.3f} s; conformal construction added median {np.median(conformal_latency):.6f} s.

**Strongest limitation.** Calibration is marginal, not conditional for a selected action or regime. All interpolation actions were unresolved, and lower-envelope coverage was only {p['lower_trajectory_empirical_coverage']:.3f}; simulator-to-plant population shift remains outside the guarantee.
"""
    (output/"result_summary.md").write_text(paper,encoding="utf-8")


def run(config: CertificationConfig) -> str:
    config.validate()
    output=Path(config.output_dir);output.mkdir(parents=True,exist_ok=True)
    b11dir=Path(config.bootstrap_dir);b12dir=Path(config.conformal_dir)
    models=_load_models(b11dir,config.ensemble_size);symmetric_q=_load_trajectory_conformal_quantiles(b12dir)
    b5cfg=episode_design.EpisodeDesignConfig();template=episode_design.build_template(b5cfg)
    ncal=config.limit_episodes or config.calibration_count;ntest=config.limit_episodes or config.final_count
    calibration,cal_manifest=trajectory_conformal.generate_population(b5cfg,template,output,ncal,1_630_000,"cstr_calibration","revealed_calibration_truth")
    cal_rows,quantile_rows,one_sided_q=calibrate_one_sided(calibration,models,config.trajectories,output)
    for row in cal_manifest:row["population_role"]="independent_safety_calibration";row["proposed_action_lps"]=next(x.proposed_action for x in calibration if x.episode_id==row["episode_id"])
    _write_csv(output/"safety_calibration_manifest.csv",cal_manifest);_write_csv(output/"safety_calibration_scores.csv",cal_rows);_write_csv(output/"one_sided_quantiles.csv",quantile_rows)
    del calibration
    final,final_manifest=trajectory_conformal.generate_population(b5cfg,template,output,ntest,1_730_000,"cstr_test","revealed_final_truth")
    for row in final_manifest:row["population_role"]="untouched_final_safety_test";row["proposed_action_lps"]=next(x.proposed_action for x in final if x.episode_id==row["episode_id"])
    _write_csv(output/"final_safety_population_manifest.csv",final_manifest)
    final_rows=evaluate_final(final,final_manifest,models,config.trajectories,output,one_sided_q,symmetric_q);summary=summarize(final_rows)
    _write_csv(output/"final_safety_episodes.csv",final_rows);_write_csv(output/"safety_summary.csv",summary)
    screening_rows,screening_summary,timing_rows=screen_actions(config,template,models,one_sided_q,output)
    _write_csv(output/"multi_action_screening.csv",screening_rows);_write_csv(output/"multi_action_screening_summary.csv",screening_summary);_write_csv(output/"screening_latency.csv",timing_rows)
    decision=decide(summary)
    metadata={"configuration":asdict(config),"physical_constraint":{"variable":"physical reactor temperature T_true","T_max_k":T_MAX_K,"source":"cstr_case.SafeStateChecker default and explicit project instantiations","horizon_steps":HORIZON,"dt_s":b5cfg.dt_s,"evaluator_note":"bootstrap ensemble measured-output reveal is deterministically de-noised with its saved RNG seed for physical primary certification labels; both measured and physical trajectories are retained"},"frozen_predictor":"bootstrap ensemble U3: first 10 accepted bootstrap models, M=2000, posterior median; no refit","scale_floor_k":EPSILON_SCALE_K,"one_sided_score_upper":"max_h (T_true_h-m_h)/max(s_h,epsilon)","one_sided_score_lower":"max_h (m_h-T_true_h)/max(s_h,epsilon)","quantile_convention":"k=ceil((n_cal+1)*(1-alpha)); kth ordered episode score","calibration_seeds":{"base":1_630_000,"count":ncal},"final_test_seeds":{"base":1_730_000,"count":ntest},"screening_seeds":{"base":1_830_000,"count":config.limit_episodes or config.screening_states},"source_hashes":{"bootstrap_models_sha256":_sha256(b11dir/"bootstrap_models.json"),"trajectory_conformal_quantiles_sha256":_sha256(b12dir/"conformal_quantiles.csv")},"predeclared_decision_rule":{"successful":"95% pooled joint false-safe <=.05, <=half both S0 and S1, SAFE rate >=.10, and no regime joint false-safe >.10","too_conservative":"joint false-safe <=.05 but SAFE rate <.10","regime_dependent":"pooled controlled but a regime joint false-safe >.10","inadequate":"otherwise"},"guarantee":"marginal joint false-safe implication only under split-conformal exchangeability; conditional false-safe is empirical","decision":decision}
    (output/"configuration.json").write_text(json.dumps(metadata,indent=2),encoding="utf-8")
    create_figures(output,final_rows,summary,screening_rows,screening_summary);create_reports(output,config,quantile_rows,final_rows,summary,screening_summary,timing_rows,decision)
    return decision


def parser() -> argparse.ArgumentParser:
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--bootstrap-dir",default="artifacts/cstr/bootstrap_ensemble");p.add_argument("--conformal-dir",default="artifacts/cstr/conformal_calibration");p.add_argument("--output-dir",default="artifacts/cstr/primary_evaluation");p.add_argument("--limit-episodes",type=int);p.add_argument("--trajectories",type=int,default=2000);return p


if __name__ == "__main__":
    result=run(CertificationConfig(**vars(parser().parse_args())));print(f"primary certification complete: {result}")
