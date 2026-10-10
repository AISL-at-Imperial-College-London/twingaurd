"""Derive the paper-facing tables in results/tables/ from the frozen experiment outputs.

Every table is a selection or relabelling of rows from frozen summary CSVs under artifacts/;
no number is computed here except the truth-completeness counts, which are counted from the
frozen per-episode tables. The unsupported "M4_action_conditioned" placeholder rows are not
copied into the paper tables (they remain in the frozen per-episode files).

    python scripts/generate_paper_tables.py           # (re)write results/tables
    python scripts/generate_paper_tables.py --check   # verify results/tables is up to date
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLASH = ROOT / "artifacts" / "flash_study"
LADDER = ROOT / "artifacts" / "flash_margin_ladder"
CSTR = ROOT / "artifacts" / "cstr"
TABLES = ROOT / "results" / "tables"

UNSUPPORTED = {"M4_action_conditioned", "action_conditioned"}
LABELS = {
    "M0_nominal_DT": "nominal digital twin (M0)",
    "M1_discrepancy_mean": "discrepancy-corrected mean (M1)",
    "M2_raw_bootstrap": "raw bootstrap band (M2)",
    "M3_pooled_conformal": "TwinGuard: bootstrap + whole-trajectory conformal (M3)",
    "nominal_DT": "nominal digital twin",
    "pooled_conformal": "TwinGuard: bootstrap + whole-trajectory conformal",
    "frozen_CSTR_conformal": "TwinGuard: bootstrap + one-sided whole-trajectory conformal",
    "proposed_method": "TwinGuard: bootstrap + one-sided whole-trajectory conformal",
    "S0_nominal_DT": "nominal digital twin",
    "S3_one_sided": "TwinGuard: bootstrap + one-sided whole-trajectory conformal",
}


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render(rows: list[dict[str, object]]) -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def completeness(table: Path) -> dict[str, tuple[int, int]]:
    counts: dict[str, list[int]] = {}
    for row in read(table):
        entry = counts.setdefault(row["method"], [0, 0])
        entry[0 if row["truth_complete"] == "True" else 1] += 1
    return {method: (c[0], c[1]) for method, c in counts.items()}


def headline() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(experiment, process, row, method, n, certified, availability, false_safe, joint, lower, upper, coverage,
            source, selector, complete=("", "")):
        rows.append({
            "experiment": experiment, "process": process, "method": LABELS.get(method, method), "method_id": method,
            "episodes": row[n], "truth_complete_episodes": complete[0], "truth_incomplete_episodes": complete[1],
            "certified_safe": row[certified], "certification_availability": row[availability],
            "false_safe": row[false_safe], "joint_false_safe": row[joint],
            "joint_false_safe_95_lower": row.get(lower, ""), "joint_false_safe_95_upper": row.get(upper, ""),
            "whole_trajectory_coverage": row.get(coverage, "") if coverage else "",
            "source_file": source, "source_row": selector,
        })

    for row in read(CSTR / "primary_evaluation/safety_summary.csv"):
        if row["action_category"] == "pooled" and row["nominal_level"] == "0.95" and row["method"] in ("S0_nominal_DT", "S3_one_sided"):
            add("CSTR primary prospective evaluation", "CSTR", row, row["method"], "number_of_episodes", "SAFE_count",
                "SAFE_certification_rate", "false_safe_count", "joint_false_safe_probability", "joint_false_safe_CI_lower",
                "joint_false_safe_CI_upper", "upper_trajectory_empirical_coverage", "artifacts/cstr/primary_evaluation/safety_summary.csv",
                f"method={row['method']}, nominal_level=0.95, action_category=pooled")
    for row in read(CSTR / "mixed_degradation/mixed_degradation_summary.csv"):
        if row["population"] == "CSTR mixed degradation":
            add("CSTR mixed-degradation fault-blind transfer", "CSTR", row, row["method"], "number_of_episodes", "SAFE_count",
                "SAFE_certification_rate", "false_safe_count", "joint_false_safe_probability", "joint_false_safe_CI_lower",
                "joint_false_safe_CI_upper", "upper_trajectory_coverage" if row["method"] == "proposed_method" else None,
                "artifacts/cstr/mixed_degradation/mixed_degradation_summary.csv", f"population={row['population']}, method={row['method']}")
    primary = completeness(FLASH / "flash_primary_episode_results.csv")
    for row in read(FLASH / "flash_primary_ablation.csv"):
        if row["group_type"] == "pooled" and row["method"] not in UNSUPPORTED:
            add("Flash primary prospective evaluation / component ablation", "flash", row, row["method"], "episode_count",
                "certified_safe_count", "certification_rate", "false_safe_count", "joint_false_safe_probability",
                "joint_false_safe_wilson_95_lower", "joint_false_safe_wilson_95_upper", "whole_trajectory_safety_direction_coverage",
                "artifacts/flash_study/flash_primary_ablation.csv", f"group_type=pooled, method={row['method']}", primary[row["method"]])
    for row in read(FLASH / "boundary_risk_availability.csv"):
        if row["process"] == "flash" and not (row["method"] == "nominal_DT" or row["confidence_level"] == "0.95"):
            continue
        add(f"{'Flash' if row['process'] == 'flash' else 'CSTR'} boundary-region challenge", row["process"].lower(), row, row["method"],
            "episode_count", "certified_safe_count", "certification_availability", "false_safe_count", "joint_false_safe_risk",
            "joint_false_safe_wilson_95_lower", "joint_false_safe_wilson_95_upper", None,
            "artifacts/flash_study/boundary_risk_availability.csv", f"process={row['process']}, method={row['method']}, confidence_level={row['confidence_level'] or 'n/a'}")
    for table, summary, experiment in (("flash_structural_episode_results.csv", "flash_structural_summary.csv", "Flash structural mismatch (actuator lag)"),
                                       ("flash_reliability_episode_results.csv", "flash_reliability_summary.csv", "Flash secondary mixed-regime reliability audit")):
        complete = completeness(FLASH / table)
        for row in read(FLASH / summary):
            if row["method"] not in UNSUPPORTED:
                add(experiment, "flash", row, row["method"], "episode_count", "certified_safe_count", "certification_rate",
                    "false_safe_count", "joint_false_safe_probability", "joint_false_safe_exact_95_lower", "joint_false_safe_exact_95_upper",
                    "trajectory_coverage", f"artifacts/flash_study/{summary}", f"method={row['method']}", complete[row["method"]])
    for row in read(FLASH / "cstr_reliability_summary.csv"):
        add("CSTR secondary mixed-regime reliability audit", "CSTR", row, row["method"], "episode_count", "certified_safe_count",
            "certification_rate", "false_safe_count", "joint_false_safe_probability", "joint_false_safe_exact_95_lower",
            "joint_false_safe_exact_95_upper", "trajectory_coverage", "artifacts/flash_study/cstr_reliability_summary.csv", f"method={row['method']}")
    return rows


def retrospective_baselines() -> list[dict[str, object]]:
    """Retrospective controls at alpha = 0.05 next to the reconstructed frozen methods (post hoc, not prospective)."""
    base = ROOT / "artifacts" / "retrospective_baselines"
    rows: list[dict[str, object]] = []

    def add(process, method, n, q, certified, availability, false_safe, joint, coverage, coverage_n, source):
        rows.append({"process": process, "method": method, "test_episodes": n, "conformal_quantile": q,
                     "certified_safe": certified, "certification_availability": availability, "false_safe": false_safe,
                     "joint_false_safe": joint, "whole_trajectory_coverage": coverage, "coverage_denominator": coverage_n,
                     "source_file": source})

    source = "artifacts/retrospective_baselines/cstr_conformal_controls/cstr_control_comparison_alpha_0p05.csv"
    for r in read(ROOT / source):
        add("CSTR", r["method"], r["test_episodes"], r["conformal_quantile"], r["certified_count"], r["certification_availability"],
            r["false_safe_count"], r["joint_false_safe_frequency"], r["upper_trajectory_coverage"], r["test_episodes"], source)
    source = "artifacts/retrospective_baselines/cstr_persistent_offset/offset_summary.csv"
    for r in read(ROOT / source):
        add("CSTR", r["method"], r["n_test"], r["q_K"], r["certified_count"], r["availability"], r["false_safe_count"],
            r["joint_false_safe"], r["upper_trajectory_coverage"], r["n_test"], source)
    source = "artifacts/retrospective_baselines/flash_conformal_controls/flash_control_comparison_alpha_0p05.csv"
    for r in read(ROOT / source):
        add("flash", r["method"], r["episode_count"], r["conformal_quantile"], r["certified_count"], r["certification_availability"],
            r["false_safe_count"], r["joint_false_safe_frequency"], r["whole_trajectory_safety_direction_coverage"],
            r["coverage_denominator_complete_pairs"], source)
    source = "artifacts/retrospective_baselines/flash_persistent_offset/offset_summary.csv"
    for r in read(ROOT / source):
        add("flash", r["method"], r["n_test_reserved"], r["quantile"], r["certified_count"], r["availability"], r["false_safe_count"],
            r["joint_false_safe"], r["coverage_complete_pairs"], r["n_coverage_pairs"], source)
    return rows


def without_unsupported(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("method") not in UNSUPPORTED]


def build() -> dict[str, str]:
    flash_quantiles = []
    import json
    config = json.loads((FLASH / "flash_conformal_config.json").read_text(encoding="utf-8"))
    for population, levels in config["pooled_quantiles"].items():
        for alpha, entry in levels.items():
            flash_quantiles.append({"calibration_population": population, "alpha": alpha, "calibration_episodes": entry["sample_count"],
                                    "corrected_rank": entry["corrected_rank"], "conformal_quantile": entry["quantile"],
                                    "source": "artifacts/flash_study/flash_conformal_config.json"})
    reliability = [{"process": "flash", **row} for row in without_unsupported(read(FLASH / "flash_reliability_summary.csv"))]
    reliability += [{"process": "CSTR", **row} for row in read(FLASH / "cstr_reliability_summary.csv")]
    keys = list(dict.fromkeys(k for row in reliability for k in row))
    reliability = [{k: row.get(k, "") for k in keys} for row in reliability]
    return {
        "paper_headline_results.csv": render(headline()),
        "cstr_primary_results.csv": render(read(CSTR / "primary_evaluation/safety_summary.csv")),
        "cstr_one_sided_conformal_quantiles.csv": render(read(CSTR / "primary_evaluation/one_sided_quantiles.csv")),
        "cstr_mixed_degradation_results.csv": render(read(CSTR / "mixed_degradation/mixed_degradation_summary.csv")),
        "cstr_mixed_degradation_by_fault.csv": render(read(CSTR / "mixed_degradation/mixed_degradation_by_fault.csv")),
        "cstr_mixed_degradation_by_severity.csv": render(read(CSTR / "mixed_degradation/mixed_degradation_by_severity.csv")),
        "flash_component_ablation.csv": render(without_unsupported(read(FLASH / "flash_primary_ablation.csv"))),
        "flash_model_selection.csv": render(read(FLASH / "flash_model_selection.csv")),
        "flash_conformal_quantiles.csv": render(flash_quantiles),
        "boundary_risk_availability.csv": render(without_unsupported(read(FLASH / "boundary_risk_availability.csv"))),
        "flash_structural_mismatch_results.csv": render(without_unsupported(read(FLASH / "flash_structural_summary.csv"))),
        "mixed_regime_audit_results.csv": render(reliability),
        "margin_ladder_risk_availability.csv": render(read(LADDER / "margin_ladder_risk_availability.csv")),
        "flash_healthy_parity_summary.csv": render(read(FLASH / "simulator_validation/HEALTHY_SCENARIO_SUMMARY.csv")),
        "flash_action_step_responses.csv": render(read(FLASH / "simulator_validation/ACTION_STEP_RESPONSES.csv")),
        "retrospective_baseline_comparison.csv": render(retrospective_baselines()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    tables = build()
    stale = []
    for name, text in tables.items():
        path = TABLES / name
        if args.check:
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8", newline="")
    if args.check:
        print("results/tables up to date" if not stale else f"stale tables: {stale}")
        return 1 if stale else 0
    print(f"wrote {len(tables)} tables to {TABLES.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
