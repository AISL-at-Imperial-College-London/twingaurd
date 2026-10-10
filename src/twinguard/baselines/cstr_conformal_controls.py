#!/usr/bin/env python3
"""
Retrospective conformal controls for the primary CSTR safety evaluation.

This script reads the frozen primary CSTR evaluation artifacts:
    artifacts/cstr/primary_evaluation/

It does NOT:
- regenerate any plant trajectory,
- refit the discrepancy model,
- regenerate bootstrap models,
- overwrite primary evaluation artifacts,
- tune anything on final-test outcomes.

It reconstructs the paper's final 500-calibration / 1000-test CSTR experiment,
compares the main M3/S3 method with fixed-width conformal controls:

    M0  nominal DT (point prediction)
    C0  nominal DT + constant-width one-sided conformal back-off
    M1  discrepancy-corrected predictive center (point prediction)
    C1  corrected center + constant-width one-sided conformal back-off
    M2  existing raw 95% bootstrap upper envelope
    M3  corrected center + adaptive-scale one-sided conformal envelope
        (the paper's S3 one-sided method)

The constant-width baselines use scores in kelvin:
    C0: max_h(T_true - T_nominal)
    C1: max_h(T_true - T_corrected_center)

The adaptive method uses:
    M3: max_h((T_true - T_corrected_center) / max(sigma, eps))

Safety:
    CERTIFIED SAFE iff max_h upper_bound(h) <= 313 K.

Outputs go to a separate directory; primary evaluation files are never modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

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


T_MAX_K = 313.0
EPSILON_SCALE_K = 1.0e-6
DEFAULT_ALPHAS = (0.10, 0.05, 0.025, 0.01, 0.005)

EXPECTED_CALIBRATION = 500
EXPECTED_TEST = 1000
EXPECTED_M3_CERTIFIED = 338
EXPECTED_M3_FALSE_SAFE = 0
EXPECTED_M3_COVERAGE = 0.957
EXPECTED_M0_FALSE_SAFE = 394

METHOD_ORDER = (
    "M0_nominal_DT",
    "C0_nominal_constant_conformal",
    "M1_corrected_center",
    "C1_corrected_constant_conformal",
    "M2_raw_bootstrap_95",
    "M3_adaptive_conformal",
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


def corrected_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    values = np.sort(np.asarray(scores, dtype=float))
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError("conformal scores must be a non-empty finite vector")
    rank = int(math.ceil((values.size + 1) * (1.0 - alpha)))
    if rank > values.size:
        return float("inf"), rank
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


def paired_files(pred_dir: Path, truth_dir: Path) -> list[tuple[str, Path, Path]]:
    predictions = {p.stem: p for p in pred_dir.glob("*.npz")}
    truths = {p.stem: p for p in truth_dir.glob("*.npz")}
    if set(predictions) != set(truths):
        raise RuntimeError(
            f"prediction/truth episode mismatch: "
            f"pred-only={len(set(predictions)-set(truths))}, "
            f"truth-only={len(set(truths)-set(predictions))}"
        )
    if not predictions:
        raise RuntimeError(f"no paired artifacts found in {pred_dir} and {truth_dir}")
    return [(eid, predictions[eid], truths[eid]) for eid in sorted(predictions)]


def calibration_scores(primary_dir: Path) -> dict[str, np.ndarray]:
    pairs = paired_files(
        primary_dir / "calibration_predictions",
        primary_dir / "revealed_calibration_truth",
    )
    if len(pairs) != EXPECTED_CALIBRATION:
        raise RuntimeError(
            f"expected {EXPECTED_CALIBRATION} final C1 calibration episodes, got {len(pairs)}"
        )

    c0, c1, m3 = [], [], []
    for episode_id, pred_path, truth_path in pairs:
        pred = load_npz(pred_path)
        truth = load_npz(truth_path)
        for key in ("nominal_temperature_k", "predictive_center_k", "predictive_scale_k"):
            if key not in pred:
                raise KeyError(f"{pred_path} missing {key}")
        if "actual_physical_temperature_k" not in truth:
            raise KeyError(f"{truth_path} missing actual_physical_temperature_k")

        nominal = np.asarray(pred["nominal_temperature_k"], float)
        center = np.asarray(pred["predictive_center_k"], float)
        scale = np.asarray(pred["predictive_scale_k"], float)
        actual = np.asarray(truth["actual_physical_temperature_k"], float)
        if not (nominal.shape == center.shape == scale.shape == actual.shape):
            raise RuntimeError(f"shape mismatch in {episode_id}")

        c0.append(float(np.max(actual - nominal)))
        c1.append(float(np.max(actual - center)))
        m3.append(float(np.max((actual - center) / np.maximum(scale, EPSILON_SCALE_K))))

    return {
        "C0_nominal_constant_conformal": np.asarray(c0),
        "C1_corrected_constant_conformal": np.asarray(c1),
        "M3_adaptive_conformal": np.asarray(m3),
    }


def frozen_upper_q95(primary_dir: Path) -> float:
    rows = read_csv(primary_dir / "one_sided_quantiles.csv")
    matches = [
        row for row in rows
        if row["direction"] == "upper" and np.isclose(float(row["nominal_level"]), 0.95)
    ]
    if len(matches) != 1:
        raise RuntimeError("could not uniquely identify frozen 95% C1 upper quantile")
    return float(matches[0]["conformal_quantile"])


def upper_bound(pred: dict[str, np.ndarray], method: str, q: float | None) -> np.ndarray:
    nominal = np.asarray(pred["nominal_temperature_k"], float)
    center = np.asarray(pred["predictive_center_k"], float)
    scale = np.asarray(pred["predictive_scale_k"], float)

    if method == "M0_nominal_DT":
        return nominal
    if method == "C0_nominal_constant_conformal":
        if q is None:
            raise ValueError("C0 requires q")
        return nominal + q
    if method == "M1_corrected_center":
        return center
    if method == "C1_corrected_constant_conformal":
        if q is None:
            raise ValueError("C1 requires q")
        return center + q
    if method == "M2_raw_bootstrap_95":
        if "raw_95_upper_k" not in pred:
            raise KeyError("C1 prediction artifact missing raw_95_upper_k")
        return np.asarray(pred["raw_95_upper_k"], float)
    if method == "M3_adaptive_conformal":
        if q is None:
            raise ValueError("M3 requires q")
        return center + q * np.maximum(scale, EPSILON_SCALE_K)
    raise KeyError(method)


def summarize_test(
    primary_dir: Path,
    method: str,
    q: float | None,
) -> dict[str, object]:
    pairs = paired_files(primary_dir / "predictions", primary_dir / "revealed_final_truth")
    if len(pairs) != EXPECTED_TEST:
        raise RuntimeError(f"expected {EXPECTED_TEST} final C1 episodes, got {len(pairs)}")

    certified = 0
    false_safe = 0
    safe_count = 0
    safe_certified = 0
    covered = 0

    for episode_id, pred_path, truth_path in pairs:
        pred = load_npz(pred_path)
        truth = load_npz(truth_path)
        actual = np.asarray(truth["actual_physical_temperature_k"], float)
        upper = upper_bound(pred, method, q)
        if upper.shape != actual.shape:
            raise RuntimeError(f"test shape mismatch in {episode_id}")

        is_safe = bool(np.max(actual) <= T_MAX_K)
        is_certified = bool(np.max(upper) <= T_MAX_K)
        is_covered = bool(np.all(actual <= upper))

        safe_count += int(is_safe)
        certified += int(is_certified)
        safe_certified += int(is_safe and is_certified)
        false_safe += int(is_certified and not is_safe)
        covered += int(is_covered)

    n = len(pairs)
    wlo, whi = wilson_interval(false_safe, n)
    clo, chi = clopper_pearson(false_safe, n)
    return {
        "method": method,
        "test_episodes": n,
        "conformal_quantile": "" if q is None else q,
        "certified_count": certified,
        "certification_availability": certified / n,
        "physically_safe_count": safe_count,
        "safe_certified_count": safe_certified,
        "safe_action_certification_rate": safe_certified / safe_count if safe_count else float("nan"),
        "false_safe_count": false_safe,
        "joint_false_safe_frequency": false_safe / n,
        "conditional_false_safe_frequency": false_safe / certified if certified else float("nan"),
        "upper_trajectory_coverage": covered / n,
        "joint_false_safe_wilson_95_lower": wlo,
        "joint_false_safe_wilson_95_upper": whi,
        "joint_false_safe_clopper_pearson_95_lower": clo,
        "joint_false_safe_clopper_pearson_95_upper": chi,
    }


def verify_frozen_m3(primary_dir: Path, recomputed_q95: float) -> dict[str, object]:
    frozen_q = frozen_upper_q95(primary_dir)
    if not np.isclose(recomputed_q95, frozen_q, rtol=0.0, atol=1e-10):
        raise RuntimeError(
            f"recomputed M3 q95={recomputed_q95:.12g} does not match "
            f"frozen C1 q95={frozen_q:.12g}"
        )

    m0 = summarize_test(primary_dir, "M0_nominal_DT", None)
    m3 = summarize_test(primary_dir, "M3_adaptive_conformal", frozen_q)

    failures = []
    if m0["false_safe_count"] != EXPECTED_M0_FALSE_SAFE:
        failures.append(
            f"M0 false-safe expected {EXPECTED_M0_FALSE_SAFE}, got {m0['false_safe_count']}"
        )
    if m3["certified_count"] != EXPECTED_M3_CERTIFIED:
        failures.append(
            f"M3 certifications expected {EXPECTED_M3_CERTIFIED}, got {m3['certified_count']}"
        )
    if m3["false_safe_count"] != EXPECTED_M3_FALSE_SAFE:
        failures.append(
            f"M3 false-safe expected {EXPECTED_M3_FALSE_SAFE}, got {m3['false_safe_count']}"
        )
    if not np.isclose(
        float(m3["upper_trajectory_coverage"]),
        EXPECTED_M3_COVERAGE,
        atol=5e-4,
    ):
        failures.append(
            f"M3 coverage expected about {EXPECTED_M3_COVERAGE}, "
            f"got {m3['upper_trajectory_coverage']}"
        )
    if failures:
        raise RuntimeError("FINAL C1 RECONSTRUCTION FAILED:\n- " + "\n- ".join(failures))
    return {"frozen_q95": frozen_q, "m0": m0, "m3": m3}


def make_plot(rows: list[dict[str, object]], output: Path) -> None:
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.5))
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
    fig.savefig(output / "cstr_control_comparison.png", dpi=220)
    plt.close(fig)


def main(args: argparse.Namespace) -> None:
    primary_dir = Path(args.primary_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    required = [
        primary_dir / "calibration_predictions",
        primary_dir / "revealed_calibration_truth",
        primary_dir / "predictions",
        primary_dir / "revealed_final_truth",
        primary_dir / "one_sided_quantiles.csv",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("missing required frozen C1 artifacts:\n- " + "\n- ".join(missing))

    scores = calibration_scores(primary_dir)

    quantile_rows: list[dict[str, object]] = []
    q_lookup: dict[tuple[str, float], float] = {}
    for method, values in scores.items():
        for alpha in args.alphas:
            q, rank = corrected_quantile(values, alpha)
            q_lookup[(method, alpha)] = q
            quantile_rows.append(
                {
                    "method": method,
                    "alpha": alpha,
                    "target_coverage": 1.0 - alpha,
                    "n_calibration": len(values),
                    "corrected_rank_k": rank,
                    "quantile": q,
                    "quantile_is_infinite": int(not np.isfinite(q)),
                }
            )

    if 0.05 not in args.alphas:
        raise ValueError("alpha list must include 0.05 for the paper comparison")

    verification = verify_frozen_m3(
        primary_dir, q_lookup[("M3_adaptive_conformal", 0.05)]
    )

    comparison = [
        summarize_test(primary_dir, "M0_nominal_DT", None),
        summarize_test(
            primary_dir,
            "C0_nominal_constant_conformal",
            q_lookup[("C0_nominal_constant_conformal", 0.05)],
        ),
        summarize_test(primary_dir, "M1_corrected_center", None),
        summarize_test(
            primary_dir,
            "C1_corrected_constant_conformal",
            q_lookup[("C1_corrected_constant_conformal", 0.05)],
        ),
        summarize_test(primary_dir, "M2_raw_bootstrap_95", None),
        summarize_test(
            primary_dir,
            "M3_adaptive_conformal",
            q_lookup[("M3_adaptive_conformal", 0.05)],
        ),
    ]

    alpha_rows: list[dict[str, object]] = []
    for alpha in args.alphas:
        for method in (
            "C0_nominal_constant_conformal",
            "C1_corrected_constant_conformal",
            "M3_adaptive_conformal",
        ):
            row = summarize_test(primary_dir, method, q_lookup[(method, alpha)])
            row.update({"alpha": alpha, "target_coverage": 1.0 - alpha})
            alpha_rows.append(row)

    write_csv(output / "cstr_control_conformal_quantiles.csv", quantile_rows)
    write_csv(output / "cstr_control_comparison_alpha_0p05.csv", comparison)
    write_csv(output / "cstr_alpha_sensitivity.csv", alpha_rows)
    make_plot(comparison, output)

    report = [
        "# Final CSTR comparison-control analysis",
        "",
        "This analysis uses the exact frozen C1 500-episode calibration population",
        "and exact frozen C1 1000-episode untouched final-test population.",
        "No trajectories were regenerated and no model was refit.",
        "",
        "## Frozen-result reconstruction",
        f"- Frozen upper q95: {verification['frozen_q95']:.12g}",
        f"- Nominal-DT false-safe: {verification['m0']['false_safe_count']}/1000",
        f"- M3 certified: {verification['m3']['certified_count']}/1000",
        f"- M3 false-safe: {verification['m3']['false_safe_count']}/1000",
        f"- M3 upper-trajectory coverage: {verification['m3']['upper_trajectory_coverage']:.4f}",
        "",
        "## Comparison 95% controls",
    ]
    for row in comparison:
        report.append(
            f"- {row['method']}: availability={row['certification_availability']:.4f}, "
            f"false-safe={row['joint_false_safe_frequency']:.4f}, "
            f"coverage={row['upper_trajectory_coverage']:.4f}"
        )
    report += [
        "",
        "Interpretation:",
        "- C0 asks whether conformalising the nominal DT alone yields useful authority.",
        "- C1 asks whether discrepancy correction improves authority under a constant-width back-off.",
        "- M3 asks whether state/action-dependent scale improves authority beyond constant width.",
        "- The formal guarantee remains marginal over the represented episode population.",
    ]
    (output / "cstr_conformal_controls_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("FINAL C1 RECONSTRUCTION: PASS")
    for row in comparison:
        print(
            f"{row['method']}: "
            f"availability={row['certification_availability']:.4f}, "
            f"false-safe={row['joint_false_safe_frequency']:.4f}, "
            f"coverage={row['upper_trajectory_coverage']:.4f}"
        )
    print(f"Outputs: {output.resolve()}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", default="artifacts/cstr/primary_evaluation")
    parser.add_argument(
        "--output-dir",
        default="runs/cstr_conformal_controls",
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
