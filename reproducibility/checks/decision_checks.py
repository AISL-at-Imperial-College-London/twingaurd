"""Certification-decision consistency and reconstruction of every summary table from per-episode rows.

These checks need no hard-coded result values. They recompute counts from the frozen
per-episode tables and compare them with the frozen summary tables that the paper reports.
"""

from __future__ import annotations

from collections import defaultdict

from .common import CSTR, FLASH, LADDER, CheckResult, as_float, read_csv, truthy


def _row_consistency(name, rows, rule, detail):
    bad = [row.get("episode_id", "?") + ":" + row.get("method", "") for row in rows if not rule(row)]
    return CheckResult(f"decision_consistency:{name}", "decision_reconstruction", name, len(rows), len(bad), detail=detail, examples=bad[:5])


def check_row_consistency() -> list[CheckResult]:
    results = []
    for name, table in (("flash primary", "flash_primary_episode_results.csv"), ("flash structural mismatch", "flash_structural_episode_results.csv"),
                        ("flash mixed-regime audit", "flash_reliability_episode_results.csv")):
        rows = read_csv(FLASH / table)
        results.append(_row_consistency(name, rows, lambda r: (
            truthy(r["certified_safe"]) == (r["decision"] == "CERTIFIED SAFE")
            and truthy(r["false_safe"]) == (truthy(r["certified_safe"]) and truthy(r["actual_unsafe"]))
            and truthy(r["actual_safe"]) != truthy(r["actual_unsafe"])),
            "certified <=> decision 'CERTIFIED SAFE'; false-safe <=> certified and physically unsafe; safe xor unsafe"))
    for name, path in (("flash boundary", FLASH / "flash_boundary_episode_results.csv"), ("cstr boundary", FLASH / "cstr_boundary_episode_results.csv"),
                       ("cstr mixed-regime audit", FLASH / "cstr_reliability_episode_results.csv")):
        results.append(_row_consistency(name, read_csv(path), lambda r: (
            truthy(r["false_safe"]) == (truthy(r["certified_safe"]) and truthy(r["actual_unsafe"]))
            and truthy(r["actual_safe"]) != truthy(r["actual_unsafe"])),
            "false-safe <=> certified and physically unsafe; safe xor unsafe"))
    results.append(_row_consistency("cstr primary", read_csv(CSTR / "primary_evaluation/final_safety_episodes.csv"), lambda r: (
        truthy(r["false_safe_indicator"]) == (r["decision"] == "SAFE" and not truthy(r["actual_safe"]))),
        "false-safe <=> decision SAFE and physical temperature limit exceeded"))
    results.append(_row_consistency("cstr mixed degradation", read_csv(CSTR / "mixed_degradation/mixed_degradation_test_episodes.csv"), lambda r: (
        truthy(r["certified_safe"]) == (r["decision"] == "CERTIFIED SAFE")
        and truthy(r["false_safe"]) == (truthy(r["certified_safe"]) and truthy(r["actual_unsafe"]))
        and truthy(r["nominal_dt_false_safe"]) == (truthy(r["nominal_dt_safe"]) and truthy(r["actual_unsafe"]))),
        "proposed and nominal-DT false-safe indicators follow from decision and physical outcome"))
    for name, path in (("flash margin ladder", LADDER / "flash_margin_results.csv"), ("cstr margin ladder", LADDER / "cstr_margin_results.csv")):
        results.append(_row_consistency(name, read_csv(path), lambda r: (
            not truthy(r["false_safe"]) or (truthy(r["certified_safe"]) and not truthy(r["actual_safe"]))),
            "false-safe implies certified and not physically safe"))
    return results


def _count(rows, column):
    return sum(truthy(row[column]) for row in rows)


def _coverage(rows, column):
    values = [truthy(row[column]) for row in rows if row.get(column, "") != ""]
    return (sum(values) / len(values)) if values else float("nan"), len(values)


def _compare(check, subject, pairs, coverage_pairs=()):
    """pairs: (label, recomputed, reported) integer comparisons; coverage pairs are reported, not failed."""
    failures = [f"{label}: recomputed {a} vs reported {b}" for label, a, b in pairs if int(a) != int(float(b))]
    notes = []
    for label, value, denominator, reported in coverage_pairs:
        reported_value = as_float(reported)
        status = "matches" if abs(value - reported_value) < 1e-9 else "differs"
        notes.append(f"{label} {value:.6f} over {denominator} rows ({status} reported {reported})")
    return CheckResult(check, "decision_reconstruction", subject, len(pairs), len(failures),
                       detail="; ".join(notes) if notes else "counts recomputed from per-episode rows", examples=failures[:5])


def _group(rows, keys):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in keys)].append(row)
    return groups


def check_summary_reconstruction() -> list[CheckResult]:
    results = []
    # flash primary + component ablation (pooled and every reported subgroup)
    episodes = read_csv(FLASH / "flash_primary_episode_results.csv")
    by_method = _group(episodes, ("method",))
    for row in read_csv(FLASH / "flash_primary_ablation.csv"):
        subset = by_method[(row["method"],)]
        if row["group_type"] != "pooled":
            column = next((c for c in subset[0] if c.endswith(("_evaluator_only", "_group", "_region", "_constraint"))
                           and row["group_value"] in {r[c] for r in subset}), None)
            if column is None:
                results.append(CheckResult(f"summary:flash ablation {row['group_type']}={row['group_value']} {row['method']}",
                                           "decision_reconstruction", "flash_primary_ablation.csv", 0, 0, status="INFO",
                                           detail="no per-episode column carries this subgroup label"))
                continue
            subset = [r for r in subset if r[column] == row["group_value"]]
        coverage, denominator = _coverage(subset, "whole_trajectory_safety_direction_covered")
        results.append(_compare(f"summary:flash ablation {row['group_type']}={row['group_value']} {row['method']}", "flash_primary_ablation.csv",
                                [("episodes", len(subset), row["episode_count"]), ("certified", _count(subset, "certified_safe"), row["certified_safe_count"]),
                                 ("false_safe", _count(subset, "false_safe"), row["false_safe_count"]),
                                 ("actual_unsafe", _count(subset, "actual_unsafe"), row["actual_unsafe_count"])],
                                [("coverage", coverage, denominator, row["whole_trajectory_safety_direction_coverage"])] if row["method"] != "M4_action_conditioned" else []))
    # boundary challenge
    flash_boundary = _group(read_csv(FLASH / "flash_boundary_episode_results.csv"), ("method", "confidence_level"))
    cstr_boundary = _group(read_csv(FLASH / "cstr_boundary_episode_results.csv"), ("method", "confidence_level"))
    for row in read_csv(FLASH / "boundary_risk_availability.csv"):
        groups = flash_boundary if row["process"] == "flash" else cstr_boundary
        key = (row["method"], row["confidence_level"])
        subset = groups.get(key) or [r for (m, _), rs in groups.items() if m == row["method"] for r in rs]
        results.append(_compare(f"summary:boundary {row['process']} {row['method']} {row['confidence_level']}", "boundary_risk_availability.csv",
                                [("episodes", len(subset), row["episode_count"]), ("certified", _count(subset, "certified_safe"), row["certified_safe_count"]),
                                 ("false_safe", _count(subset, "false_safe"), row["false_safe_count"]),
                                 ("actual_unsafe", _count(subset, "actual_unsafe"), row["actual_unsafe_count"])]))
    # structural mismatch and mixed-regime audits
    for table, summary, coverage_column in (("flash_structural_episode_results.csv", "flash_structural_summary.csv", "whole_trajectory_safety_direction_covered"),
                                            ("flash_reliability_episode_results.csv", "flash_reliability_summary.csv", "whole_trajectory_safety_direction_covered"),
                                            ("cstr_reliability_episode_results.csv", "cstr_reliability_summary.csv", "upper_trajectory_covered")):
        groups = _group(read_csv(FLASH / table), ("method",))
        for row in read_csv(FLASH / summary):
            subset = groups[(row["method"],)]
            coverage, denominator = _coverage(subset, coverage_column)
            results.append(_compare(f"summary:{summary} {row['method']}", summary,
                                    [("episodes", len(subset), row["episode_count"]), ("certified", _count(subset, "certified_safe"), row["certified_safe_count"]),
                                     ("false_safe", _count(subset, "false_safe"), row["false_safe_count"]),
                                     ("actual_unsafe", _count(subset, "actual_unsafe"), row["actual_unsafe_count"])],
                                    [("coverage", coverage, denominator, row["trajectory_coverage"])] if row["method"] != "M4_action_conditioned" else []))
    # CSTR primary (every method, level and action category)
    primary = read_csv(CSTR / "primary_evaluation/final_safety_episodes.csv")
    groups = _group(primary, ("method", "nominal_level"))
    for row in read_csv(CSTR / "primary_evaluation/safety_summary.csv"):
        subset = [r for r in groups[(row["method"], row["nominal_level"])]
                  if row["action_category"] == "pooled" or r["action_category"] == row["action_category"]]
        coverage, denominator = _coverage(subset, "upper_trajectory_covered")
        results.append(_compare(f"summary:cstr primary {row['method']} {row['nominal_level']} {row['action_category']}", "safety_summary.csv",
                                [("episodes", len(subset), row["number_of_episodes"]),
                                 ("SAFE", sum(r["decision"] == "SAFE" for r in subset), row["SAFE_count"]),
                                 ("UNSAFE", sum(r["decision"] == "UNSAFE" for r in subset), row["UNSAFE_count"]),
                                 ("false_safe", _count(subset, "false_safe_indicator"), row["false_safe_count"])],
                                [("upper coverage", coverage, denominator, row["upper_trajectory_empirical_coverage"])]))
    # CSTR mixed degradation (+ its comparison rows, which restate the CSTR primary result)
    mixed = read_csv(CSTR / "mixed_degradation/mixed_degradation_test_episodes.csv")
    primary_summary = {(r["method"], r["nominal_level"], r["action_category"]): r for r in read_csv(CSTR / "primary_evaluation/safety_summary.csv")}
    for row in read_csv(CSTR / "mixed_degradation/mixed_degradation_summary.csv"):
        if row["population"] == "CSTR mixed degradation":
            safe_col, false_col, cov_col = (("certified_safe", "false_safe", "upper_trajectory_covered") if row["method"] == "proposed_method"
                                            else ("nominal_dt_safe", "nominal_dt_false_safe", "nominal_upper_trajectory_covered"))
            coverage, denominator = _coverage(mixed, cov_col)
            results.append(_compare(f"summary:cstr mixed degradation {row['method']}", "mixed_degradation_summary.csv",
                                    [("episodes", len(mixed), row["number_of_episodes"]), ("certified", _count(mixed, safe_col), row["SAFE_count"]),
                                     ("false_safe", _count(mixed, false_col), row["false_safe_count"])],
                                    [("upper coverage", coverage, denominator, row["upper_trajectory_coverage"])]))
        else:
            source = primary_summary[("S3_one_sided" if row["method"] == "proposed_method" else "S0_nominal_DT", "0.95", "pooled")]
            results.append(_compare(f"summary:cstr primary restated in mixed-degradation table ({row['method']})", "mixed_degradation_summary.csv",
                                    [("episodes", source["number_of_episodes"], row["number_of_episodes"]),
                                     ("certified", source["SAFE_count"], row["SAFE_count"]),
                                     ("false_safe", source["false_safe_count"], row["false_safe_count"])]))
    # margin ladder
    ladder_rows = read_csv(LADDER / "flash_margin_results.csv") + read_csv(LADDER / "cstr_margin_results.csv")
    for row in read_csv(LADDER / "margin_ladder_risk_availability.csv"):
        subset = [r for r in ladder_rows if r["process"].lower() == row["process"].lower() and r["method"] == row["method"]
                  and (row["margin_band"] in ("all", "pooled", "ALL") or r["margin_band"] == row["margin_band"])]
        results.append(_compare(f"summary:margin ladder {row['process']} {row['margin_band']} {row['method']}", "margin_ladder_risk_availability.csv",
                                [("eligible", len(subset), row["N_eligible"]), ("certified", _count(subset, "certified_safe"), row["certified_count"]),
                                 ("false_safe", _count(subset, "false_safe"), row["false_safe_count"])]))
    return results
