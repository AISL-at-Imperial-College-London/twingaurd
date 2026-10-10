#!/usr/bin/env python3
"""
Retrospective conformal controls for the flash calibration and test populations.

This script uses the frozen calibration design and saved test records.
The test population contains 1200 episodes.

It does NOT:
- refit the flash discrepancy model,
- refit bootstrap models,
- regenerate test predictions or truth,
- modify frozen calibration or test artifacts,
- tune anything using test outcomes.

Why calibration truth is regenerated:
The calibration bundle includes operational inputs and adaptive conformal scores,
but not complete calibration truth and mean-corrected trajectories.
To evaluate fixed-width controls, the script deterministically rebuilds
ordinary calibration plant trajectories from the declared seeds and
frozen candidate actions. It verifies that every
regenerated observation/future nominal input matches the saved calibration
input arrays before using the regenerated physical future.

Fixed-scale controls:
    M0  nominal DT point prediction
    C0  nominal DT + fixed-engineering-scale conformal back-off
    M1  discrepancy mean point prediction
    C1  discrepancy mean + fixed-engineering-scale conformal back-off
    M2  existing raw bootstrap envelope
    M3  existing adaptive-scale pooled conformal envelope

C0/C1 use the same four safety directions as the paper:
    high pressure, high temperature, high level, low level

but replace the state/action-dependent bootstrap scale with fixed declared
engineering scales [0.10 bar, 0.10 level fraction, 5 K]. Thus the width is
constant across episodes/actions/horizon in normalized engineering units.

Outputs go to a separate directory. Frozen calibration and test files are not overwritten.

Run from repository root:
    python experiments/evaluate_flash_conformal_controls.py

The flash-study artifacts, including the per-episode files of the primary evaluation, must exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

try:
    from scipy.stats import beta
except Exception:
    beta = None

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from twinguard.flash import bootstrap_and_calibrate as calibration_stage
from twinguard.uncertainty.flash_bootstrap import PREDICTIVE_SCALE_FLOORS
from twinguard.conformal.flash_conformal import (
    safety_decision,
    safety_direction_score,
)
from twinguard.discrepancy.flash_state_space import (
    ENGINEERING_SCALES,
    parameters_from_payload,
    predict_discrepancy,
)
from twinguard.scenarios.flash_population_design import (
    MEASURED_OUTPUTS,
    SafetySpecification,
)


DEFAULT_ALPHAS = (0.10, 0.05, 0.025, 0.01, 0.005)
ORDINARY_SEED_START = 24_000_000
EXPECTED_FINAL = 1200

EXPECTED = {
    "M0_nominal_DT": {"certified": 779, "false_safe": 207},
    "M1_discrepancy_mean": {"certified": 672, "false_safe": 103},
    "M2_raw_bootstrap": {"certified": 623, "false_safe": 72},
    "M3_pooled_conformal": {"certified": 177, "false_safe": 5},
}
EXPECTED_M3_COVERAGE = 0.9399332591768632

METHOD_ORDER = (
    "M0_nominal_DT",
    "C0_nominal_fixedscale_conformal",
    "M1_discrepancy_mean",
    "C1_corrected_fixedscale_conformal",
    "M2_raw_bootstrap",
    "M3_pooled_conformal",
)


def parse_alphas(value: str) -> tuple[float, ...]:
    result = tuple(float(x.strip()) for x in value.split(",") if x.strip())
    if not result or any(not 0.0 < x < 1.0 for x in result):
        raise argparse.ArgumentTypeError("alphas must be comma-separated values in (0,1)")
    return result


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as saved:
        return {name: np.asarray(saved[name]) for name in saved.files}


def scalar_str(value: np.ndarray | object) -> str:
    array = np.asarray(value)
    return str(array.item() if array.ndim == 0 else value)


def corrected_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    values = np.sort(np.asarray(scores, dtype=float))
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("conformal scores must be a non-empty finite vector")
    rank = int(math.ceil((values.size + 1) * (1.0 - alpha)))
    # Match flash_conformal.corrected_quantile convention.
    rank = min(rank, values.size)
    return float(values[rank - 1]), rank


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n <= 0:
        return float("nan"), float("nan")
    p = k / n
    den = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / den
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def clopper_pearson(k: int, n: int, confidence: float = 0.95) -> tuple[float, float]:
    if n <= 0 or beta is None:
        return float("nan"), float("nan")
    tail = (1.0 - confidence) / 2.0
    lower = 0.0 if k == 0 else float(beta.ppf(tail, k, n - k + 1))
    upper = 1.0 if k == n else float(beta.ppf(1.0 - tail, k + 1, n - k))
    return lower, upper


def trajectory_mapping(values: np.ndarray) -> dict[str, np.ndarray]:
    return {name: values[:, index] for index, name in enumerate(MEASURED_OUTPUTS)}


def actual_safe(actual: np.ndarray, safety: SafetySpecification) -> bool:
    if len(actual) != 120:
        return False
    margins = safety.violation_margins(trajectory_mapping(actual))
    return bool(len(margins) and float(np.max(margins)) <= 0.0)


def point_certified(center: np.ndarray, safety: SafetySpecification) -> bool:
    if len(center) != 120:
        return False
    margins = safety.violation_margins(trajectory_mapping(center))
    return bool(len(margins) and float(np.max(margins)) <= 0.0)


def directional_coverage(
    actual: np.ndarray,
    center: np.ndarray,
    scale: np.ndarray,
    multiplier: float,
) -> bool | None:
    if len(actual) != 120 or len(center) != 120 or len(scale) != 120:
        return None
    denominator = np.maximum(np.asarray(scale, float), PREDICTIVE_SCALE_FLOORS)
    upper = np.asarray(center, float) + multiplier * denominator
    lower_level = np.asarray(center, float)[:, 1] - multiplier * denominator[:, 1]
    return bool(
        np.all(actual[:, 0] <= upper[:, 0])
        and np.all(actual[:, 2] <= upper[:, 2])
        and np.all(actual[:, 1] <= upper[:, 1])
        and np.all(actual[:, 1] >= lower_level)
    )


def fixed_scale(length: int) -> np.ndarray:
    return np.repeat(np.asarray(ENGINEERING_SCALES, float)[None, :], length, axis=0)


def load_selected_mean_model(experiment_root: Path):
    payload = json.loads((experiment_root / "selected_flash_model.json").read_text(encoding="utf-8"))
    return parameters_from_payload(payload["model"])


def load_frozen_ordinary_inputs(experiment_root: Path) -> dict[str, np.ndarray]:
    path = experiment_root / "calibration_data" / "calibration_ordinary_operational_inputs.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    return load_npz(path)


def regenerate_calibration_truth(
    experiment_root: Path,
    workers: int,
) -> tuple[list[dict[str, object]], dict[str, np.ndarray]]:
    """
    Regenerate physical calibration futures from the frozen action design.

    Before using any score, compare regenerated online and future nominal inputs
    with the saved calibration-input bundle.
    """
    frozen = load_frozen_ordinary_inputs(experiment_root)
    episode_ids = [str(x) for x in np.asarray(frozen["episode_id"]).tolist()]
    candidate_actions = np.asarray(frozen["candidate_action"], float)

    tasks = []
    for row_index, episode_id in enumerate(episode_ids):
        try:
            original_index = int(episode_id.rsplit("_", 1)[1])
        except Exception as exc:
            raise RuntimeError(f"cannot parse calibration episode index from {episode_id}") from exc
        seed = ORDINARY_SEED_START + original_index
        tasks.append(
            ("calibration_ordinary", original_index, seed, candidate_actions[row_index])
        )

    if workers <= 1:
        episodes = [calibration_stage._calibration_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            episodes = list(pool.map(calibration_stage._calibration_task, tasks, chunksize=1))

    if len(episodes) != len(episode_ids):
        raise RuntimeError("calibration regeneration count mismatch")

    checks = []
    truths: dict[str, np.ndarray] = {}
    for row_index, (episode_id, episode) in enumerate(zip(episode_ids, episodes)):
        if episode.episode_id != episode_id:
            raise RuntimeError(
                f"regenerated episode id mismatch: expected {episode_id}, got {episode.episode_id}"
            )
        if not episode.completed:
            raise RuntimeError(
                f"{episode_id} was frozen as complete but regenerated as incomplete"
            )

        comparisons = {
            "observed_residual": (
                np.asarray(episode.observed_residual),
                np.asarray(frozen["observed_residual"][row_index]),
            ),
            "observed_features": (
                np.asarray(episode.observed_features),
                np.asarray(frozen["observed_features"][row_index]),
            ),
            "future_features": (
                np.asarray(episode.future_features),
                np.asarray(frozen["future_features"][row_index]),
            ),
            "nominal_future": (
                np.asarray(episode.nominal_future),
                np.asarray(frozen["nominal_future"][row_index]),
            ),
            "candidate_action": (
                np.asarray(episode.candidate_action),
                np.asarray(frozen["candidate_action"][row_index]),
            ),
        }

        max_diffs = {}
        for name, (regen, saved) in comparisons.items():
            if regen.shape != saved.shape:
                raise RuntimeError(f"{episode_id}: {name} shape mismatch")
            diff = float(np.max(np.abs(regen - saved))) if regen.size else 0.0
            max_diffs[name] = diff
            if not np.allclose(regen, saved, rtol=1e-10, atol=1e-10):
                raise RuntimeError(
                    f"{episode_id}: regenerated {name} does not match frozen calibration input; "
                    f"max abs diff={diff:.3e}"
                )

        truths[episode_id] = np.asarray(episode.actual_future, float)
        checks.append(
            {
                "episode_id": episode_id,
                "episode_seed": episode.episode_seed,
                "max_abs_diff_observed_residual": max_diffs["observed_residual"],
                "max_abs_diff_observed_features": max_diffs["observed_features"],
                "max_abs_diff_future_features": max_diffs["future_features"],
                "max_abs_diff_nominal_future": max_diffs["nominal_future"],
                "max_abs_diff_candidate_action": max_diffs["candidate_action"],
                "frozen_input_match": 1,
            }
        )
    return checks, truths


def build_constant_width_calibration_scores(
    experiment_root: Path,
    truths: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    frozen = load_frozen_ordinary_inputs(experiment_root)
    episode_ids = [str(x) for x in np.asarray(frozen["episode_id"]).tolist()]
    selected_model = load_selected_mean_model(experiment_root)

    c0_scores = []
    c1_scores = []
    rows = []

    for row_index, episode_id in enumerate(episode_ids):
        observed_residual = np.asarray(frozen["observed_residual"][row_index], float)
        observed_features = np.asarray(frozen["observed_features"][row_index], float)
        future_features = np.asarray(frozen["future_features"][row_index], float)
        nominal = np.asarray(frozen["nominal_future"][row_index], float)
        actual = np.asarray(truths[episode_id], float)

        discrepancy, _ = predict_discrepancy(
            selected_model,
            observed_residual,
            observed_features,
            future_features,
        )
        corrected = nominal + discrepancy
        scale = fixed_scale(len(nominal))

        c0, c0_direction, c0_horizon = safety_direction_score(actual, nominal, scale)
        c1, c1_direction, c1_horizon = safety_direction_score(actual, corrected, scale)

        c0_scores.append(c0)
        c1_scores.append(c1)
        rows.append(
            {
                "episode_id": episode_id,
                "C0_fixedscale_score": c0,
                "C0_maximizing_direction": c0_direction,
                "C0_maximizing_horizon_index": c0_horizon,
                "C1_fixedscale_score": c1,
                "C1_maximizing_direction": c1_direction,
                "C1_maximizing_horizon_index": c1_horizon,
            }
        )

    return {
        "C0_nominal_fixedscale_conformal": np.asarray(c0_scores, float),
        "C1_corrected_fixedscale_conformal": np.asarray(c1_scores, float),
    }, rows


def load_adaptive_calibration_scores(experiment_root: Path) -> np.ndarray:
    rows = read_csv(experiment_root / "flash_calibration_scores.csv")
    values = [
        float(row["whole_trajectory_safety_score"])
        for row in rows
        if row["calibration_population"] == "calibration_ordinary"
    ]
    if not values:
        raise RuntimeError("no ordinary adaptive calibration scores found")
    return np.asarray(values, float)


def frozen_q95(experiment_root: Path) -> float:
    config = json.loads((experiment_root / "flash_conformal_config.json").read_text(encoding="utf-8"))
    return float(config["pooled_quantiles"]["calibration_ordinary"]["0.05"]["quantile"])


def final_episode_paths(experiment_root: Path) -> list[tuple[str, Path, Path]]:
    pred_dir = experiment_root / "primary_evaluation_episodes" / "predictions"
    truth_dir = experiment_root / "primary_evaluation_episodes" / "truths"
    predictions = {p.stem: p for p in pred_dir.glob("*.npz")}
    truths = {p.stem: p for p in truth_dir.glob("*.npz")}
    if set(predictions) != set(truths):
        raise RuntimeError(
            f"primary evaluation prediction/truth mismatch: pred-only={len(set(predictions)-set(truths))}, "
            f"truth-only={len(set(truths)-set(predictions))}"
        )
    if len(predictions) != EXPECTED_FINAL:
        raise RuntimeError(
            f"expected {EXPECTED_FINAL} frozen primary evaluation final episodes, got {len(predictions)}"
        )
    return [(eid, predictions[eid], truths[eid]) for eid in sorted(predictions)]


def method_components(
    prediction: Mapping[str, np.ndarray],
    method: str,
    q: float | None,
) -> tuple[np.ndarray, np.ndarray | None, float | None, bool]:
    status = scalar_str(prediction["prediction_status"])
    if status != "COMPLETE":
        return np.empty((0, 3)), None, None, False

    nominal = np.asarray(prediction["nominal_horizon_outputs"], float)
    mean_corrected = np.asarray(prediction["mean_corrected_horizon_outputs"], float)
    ensemble_center = np.asarray(prediction["ensemble_center_horizon_outputs"], float)
    ensemble_scale = np.asarray(prediction["ensemble_scale_horizon_outputs"], float)

    if method == "M0_nominal_DT":
        return nominal, None, 0.0, True
    if method == "C0_nominal_fixedscale_conformal":
        if q is None:
            raise ValueError("C0 requires conformal quantile")
        return nominal, fixed_scale(len(nominal)), q, True
    if method == "M1_discrepancy_mean":
        return mean_corrected, None, 0.0, True
    if method == "C1_corrected_fixedscale_conformal":
        if q is None:
            raise ValueError("C1 requires conformal quantile")
        return mean_corrected, fixed_scale(len(mean_corrected)), q, True
    if method == "M2_raw_bootstrap":
        raw = float(np.asarray(prediction["raw_multiplier"]).item())
        return ensemble_center, ensemble_scale, raw, True
    if method == "M3_pooled_conformal":
        if q is None:
            raise ValueError("M3 requires conformal quantile")
        return ensemble_center, ensemble_scale, q, True
    raise KeyError(method)


def summarize_final(
    experiment_root: Path,
    method: str,
    q: float | None,
) -> dict[str, object]:
    safety = SafetySpecification()
    certified = 0
    false_safe = 0
    safe_count = 0
    safe_certified = 0
    prediction_complete_count = 0
    truth_complete_count = 0
    coverage_values: list[bool] = []

    for episode_id, pred_path, truth_path in final_episode_paths(experiment_root):
        prediction = load_npz(pred_path)
        truth = load_npz(truth_path)
        actual = np.asarray(truth["actual_horizon_outputs"], float)

        center, scale, multiplier, prediction_complete = method_components(
            prediction, method, q
        )
        truth_complete = len(actual) == 120
        prediction_complete_count += int(prediction_complete)
        truth_complete_count += int(truth_complete)

        is_safe = actual_safe(actual, safety)
        safe_count += int(is_safe)

        if not prediction_complete:
            is_certified = False
        elif scale is None:
            is_certified = point_certified(center, safety)
        else:
            is_certified = (
                safety_decision(center, scale, float(multiplier), safety=safety) == "SAFE"
            )

        certified += int(is_certified)
        safe_certified += int(is_certified and is_safe)
        false_safe += int(is_certified and not is_safe)

        if prediction_complete and truth_complete:
            if scale is None:
                # Match primary evaluation's zero-width directional coverage convention.
                zero_scale = np.zeros_like(center)
                coverage = directional_coverage(actual, center, zero_scale, 0.0)
            else:
                coverage = directional_coverage(
                    actual, center, scale, float(multiplier)
                )
            if coverage is not None:
                coverage_values.append(bool(coverage))

    n = EXPECTED_FINAL
    wlo, whi = wilson_interval(false_safe, n)
    clo, chi = clopper_pearson(false_safe, n)
    return {
        "method": method,
        "episode_count": n,
        "prediction_complete_count": prediction_complete_count,
        "truth_complete_count": truth_complete_count,
        "conformal_quantile": "" if q is None else q,
        "certified_count": certified,
        "certification_availability": certified / n,
        "physically_safe_count": safe_count,
        "safe_certified_count": safe_certified,
        "safe_action_certification_rate": safe_certified / safe_count if safe_count else float("nan"),
        "false_safe_count": false_safe,
        "joint_false_safe_frequency": false_safe / n,
        "conditional_false_safe_frequency": false_safe / certified if certified else float("nan"),
        "whole_trajectory_safety_direction_coverage": (
            float(np.mean(coverage_values)) if coverage_values else float("nan")
        ),
        "coverage_denominator_complete_pairs": len(coverage_values),
        "joint_false_safe_wilson_95_lower": wlo,
        "joint_false_safe_wilson_95_upper": whi,
        "joint_false_safe_clopper_pearson_95_lower": clo,
        "joint_false_safe_clopper_pearson_95_upper": chi,
    }


def verify_existing_primary(
    experiment_root: Path,
    m3_q95: float,
) -> dict[str, dict[str, object]]:
    reconstructed = {
        "M0_nominal_DT": summarize_final(experiment_root, "M0_nominal_DT", None),
        "M1_discrepancy_mean": summarize_final(experiment_root, "M1_discrepancy_mean", None),
        "M2_raw_bootstrap": summarize_final(experiment_root, "M2_raw_bootstrap", None),
        "M3_pooled_conformal": summarize_final(
            experiment_root, "M3_pooled_conformal", m3_q95
        ),
    }
    failures = []
    for method, expected in EXPECTED.items():
        row = reconstructed[method]
        if row["certified_count"] != expected["certified"]:
            failures.append(
                f"{method}: certified expected {expected['certified']}, "
                f"got {row['certified_count']}"
            )
        if row["false_safe_count"] != expected["false_safe"]:
            failures.append(
                f"{method}: false-safe expected {expected['false_safe']}, "
                f"got {row['false_safe_count']}"
            )
    if not np.isclose(
        float(reconstructed["M3_pooled_conformal"]["whole_trajectory_safety_direction_coverage"]),
        EXPECTED_M3_COVERAGE,
        atol=5e-7,
    ):
        failures.append(
            "M3 coverage mismatch: expected "
            f"{EXPECTED_M3_COVERAGE:.12f}, got "
            f"{reconstructed['M3_pooled_conformal']['whole_trajectory_safety_direction_coverage']}"
        )
    if failures:
        raise RuntimeError("FINAL primary evaluation RECONSTRUCTION FAILED:\n- " + "\n- ".join(failures))
    return reconstructed


def make_plot(rows: list[dict[str, object]], output: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for row in rows:
        ax.scatter(
            float(row["certification_availability"]),
            float(row["joint_false_safe_frequency"]),
            s=55,
        )
        ax.annotate(
            str(row["method"]).split("_")[0],
            (
                float(row["certification_availability"]),
                float(row["joint_false_safe_frequency"]),
            ),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Certification availability")
    ax.set_ylabel("Observed joint false-safe frequency")
    ax.set_xlim(left=-0.02)
    ax.set_ylim(bottom=-0.005)
    fig.tight_layout()
    fig.savefig(output / "flash_control_comparison.png", dpi=220)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    experiment_root = Path(args.experiment_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    required = [
        experiment_root / "calibration_data" / "calibration_ordinary_operational_inputs.npz",
        experiment_root / "flash_calibration_scores.csv",
        experiment_root / "flash_conformal_config.json",
        experiment_root / "selected_flash_model.json",
        experiment_root / "primary_evaluation_episodes" / "predictions",
        experiment_root / "primary_evaluation_episodes" / "truths",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "missing required frozen flash artifacts:\n- " + "\n- ".join(missing)
        )

    print("Regenerating ONLY the frozen calibration ordinary calibration physical futures...")
    print("This does not touch primary evaluation and does not refit any model.")
    verification_rows, calibration_truths = regenerate_calibration_truth(
        experiment_root, args.workers
    )
    write_csv(output / "flash_calibration_regeneration_checks.csv", verification_rows)

    fixed_scores, fixed_score_rows = build_constant_width_calibration_scores(
        experiment_root, calibration_truths
    )
    write_csv(output / "flash_fixedscale_calibration_scores.csv", fixed_score_rows)

    adaptive_scores = load_adaptive_calibration_scores(experiment_root)
    score_sets = {
        **fixed_scores,
        "M3_pooled_conformal": adaptive_scores,
    }

    quantile_rows: list[dict[str, object]] = []
    q_lookup: dict[tuple[str, float], float] = {}
    for method, values in score_sets.items():
        for alpha in args.alphas:
            q, rank = corrected_quantile(values, alpha)
            q_lookup[(method, alpha)] = q
            quantile_rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "target_coverage": 1.0 - alpha,
                    "n_calibration_complete": len(values),
                    "corrected_rank_k": rank,
                    "quantile": q,
                    "fixed_engineering_scales": (
                        "0.10 bar; 0.10 level fraction; 5 K"
                        if method.startswith(("C0_", "C1_"))
                        else "adaptive bootstrap scale"
                    ),
                }
            )

    if 0.05 not in args.alphas:
        raise ValueError("alpha list must include 0.05 for the paper comparison")

    frozen_q = frozen_q95(experiment_root)
    recomputed_m3_q = q_lookup[("M3_pooled_conformal", 0.05)]
    if not np.isclose(frozen_q, recomputed_m3_q, rtol=0.0, atol=1e-12):
        raise RuntimeError(
            f"adaptive q95 mismatch: frozen={frozen_q:.12g}, "
            f"recomputed={recomputed_m3_q:.12g}"
        )

    print("Reconstructing frozen primary evaluation headline results...")
    original = verify_existing_primary(experiment_root, frozen_q)

    c0 = summarize_final(
        experiment_root,
        "C0_nominal_fixedscale_conformal",
        q_lookup[("C0_nominal_fixedscale_conformal", 0.05)],
    )
    c1 = summarize_final(
        experiment_root,
        "C1_corrected_fixedscale_conformal",
        q_lookup[("C1_corrected_fixedscale_conformal", 0.05)],
    )

    comparison = [
        original["M0_nominal_DT"],
        c0,
        original["M1_discrepancy_mean"],
        c1,
        original["M2_raw_bootstrap"],
        original["M3_pooled_conformal"],
    ]

    alpha_rows: list[dict[str, object]] = []
    for alpha in args.alphas:
        for method in (
            "C0_nominal_fixedscale_conformal",
            "C1_corrected_fixedscale_conformal",
            "M3_pooled_conformal",
        ):
            row = summarize_final(experiment_root, method, q_lookup[(method, alpha)])
            row.update({"alpha": alpha, "target_coverage": 1.0 - alpha})
            alpha_rows.append(row)

    write_csv(output / "flash_control_conformal_quantiles.csv", quantile_rows)
    write_csv(output / "flash_control_comparison_alpha_0p05.csv", comparison)
    write_csv(output / "flash_alpha_sensitivity.csv", alpha_rows)
    make_plot(comparison, output)

    report = [
        "# Final flash comparison-control analysis",
        "",
        "The final-test calculations use the exact frozen primary evaluation 1200-episode artifacts.",
        "The new C0/C1 calibration scores use deterministic reconstruction of the",
        "frozen calibration ordinary calibration futures; every regenerated operational input",
        "is checked against the frozen calibration operational-input bundle before use.",
        "No model is refit and no primary evaluation trajectory is regenerated.",
        "",
        "## Frozen-result reconstruction",
        f"- Frozen/recomputed adaptive q95: {frozen_q:.12g}",
    ]
    for method in (
        "M0_nominal_DT",
        "M1_discrepancy_mean",
        "M2_raw_bootstrap",
        "M3_pooled_conformal",
    ):
        row = original[method]
        report.append(
            f"- {method}: certified={row['certified_count']}/1200, "
            f"false-safe={row['false_safe_count']}/1200, "
            f"coverage={row['whole_trajectory_safety_direction_coverage']:.6f}"
        )

    report += ["", "## Comparison 95% controls"]
    for row in comparison:
        report.append(
            f"- {row['method']}: availability={row['certification_availability']:.4f}, "
            f"false-safe={row['joint_false_safe_frequency']:.4f}, "
            f"coverage={row['whole_trajectory_safety_direction_coverage']:.4f}"
        )

    report += [
        "",
        "Interpretation guardrails:",
        "- C0 isolates conformal back-off around the nominal DT.",
        "- C1 isolates discrepancy-mean correction under a fixed normalized width.",
        "- M3 adds state/action-dependent bootstrap scale before conformal calibration.",
        "- Fixed C0/C1 scales are the declared engineering scales, not test-tuned widths.",
        "- The guarantee remains population-marginal; alpha is not a SIL PFD.",
    ]

    (output / "flash_conformal_controls_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("FINAL primary evaluation RECONSTRUCTION: PASS")
    for row in comparison:
        print(
            f"{row['method']}: "
            f"availability={row['certification_availability']:.4f}, "
            f"false-safe={row['joint_false_safe_frequency']:.4f}, "
            f"coverage={row['whole_trajectory_safety_direction_coverage']:.4f}"
        )
    print(f"Outputs: {output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--experiment-dir",
        default="artifacts/flash_study",
        help="Path to the frozen artifacts/flash_study experiment directory.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/flash_conformal_controls",
        help="New output directory; frozen calibration/primary evaluation files are never modified.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Workers used only to regenerate calibration ordinary calibration plant futures.",
    )
    parser.add_argument(
        "--alphas",
        type=parse_alphas,
        default=DEFAULT_ALPHAS,
        help="comma-separated alpha values",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
