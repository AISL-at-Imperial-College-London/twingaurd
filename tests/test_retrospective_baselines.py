"""Retrospective baselines: per-episode decisions reproduce their summaries; reconstructed M0-M3 match the frozen results."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "artifacts" / "retrospective_baselines"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("process,certified,false_safe,covered,n_key", [
    ("cstr", "certified", "false_safe", "covered", "n_test"),
    ("flash", "certified", "false_safe", "covered", "n_test_reserved"),
])
def test_offset_summary_matches_per_episode_decisions(process, certified, false_safe, covered, n_key) -> None:
    folder = BASE / f"{process}_persistent_offset"
    decisions = _rows(folder / "offset_test_decisions.csv")
    summary = _rows(folder / "offset_summary.csv")[0]
    assert len(decisions) == int(summary[n_key]) == len({r["episode_id"] for r in decisions})
    assert sum(int(r[certified]) for r in decisions) == int(summary["certified_count"])
    assert sum(int(r[false_safe]) for r in decisions) == int(summary["false_safe_count"])
    pairs = [int(r[covered]) for r in decisions if r[covered] != ""]
    coverage_key = "upper_trajectory_coverage" if process == "cstr" else "coverage_complete_pairs"
    assert math.isclose(sum(pairs) / len(pairs), float(summary[coverage_key]), rel_tol=0, abs_tol=1e-12)


def test_offset_calibration_population_sizes() -> None:
    assert len(_rows(BASE / "cstr_persistent_offset" / "offset_calibration_scores.csv")) == 500
    assert len(_rows(BASE / "flash_persistent_offset" / "offset_calibration_scores.csv")) == 447


@pytest.mark.parametrize("folder,frozen,mapping", [
    ("cstr_conformal_controls/cstr_control_comparison_alpha_0p05.csv",
     ("artifacts/cstr/primary_evaluation/safety_summary.csv"),
     {"M0_nominal_DT": "S0_nominal_DT", "M3_adaptive_conformal": "S3_one_sided"}),
    ("flash_conformal_controls/flash_control_comparison_alpha_0p05.csv",
     ("artifacts/flash_study/flash_primary_ablation.csv"),
     {m: m for m in ("M0_nominal_DT", "M1_discrepancy_mean", "M2_raw_bootstrap", "M3_pooled_conformal")}),
])
def test_reconstructed_frozen_methods_match_frozen_results(folder, frozen, mapping) -> None:
    controls = {r["method"]: r for r in _rows(BASE / folder)}
    reference = _rows(ROOT / frozen)
    for method, frozen_method in mapping.items():
        if "safety_summary" in frozen:
            row = next(r for r in reference if r["method"] == frozen_method and r["nominal_level"] == "0.95" and r["action_category"] == "pooled")
            expected = (int(row["SAFE_count"]), int(row["false_safe_count"]))
        else:
            row = next(r for r in reference if r["group_type"] == "pooled" and r["method"] == frozen_method)
            expected = (int(row["certified_safe_count"]), int(row["false_safe_count"]))
        assert (int(controls[method]["certified_count"]), int(controls[method]["false_safe_count"])) == expected, method
