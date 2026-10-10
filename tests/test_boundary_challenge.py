import csv
import json
from pathlib import Path

import numpy as np
import pytest

from twinguard.evaluation import boundary_challenge as boundary


ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "flash_study"


def _rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_boundary_uses_frozen_design_and_calibration_only_quantiles():
    design = json.loads((ROOT / "boundary_challenge_design.json").read_text(encoding="utf-8"))
    assert design["frozen_before_boundary_truth"] is True
    assert design["flash"]["seed_start"] == 27_000_000
    assert design["flash"]["episode_count"] == 1_200
    assert design["cstr"]["seed_start"] == 9_300_000
    assert design["cstr"]["episode_count"] == 300
    expected = dict(
        zip(design["flash"]["confidence_levels"], design["flash"]["quantiles"])
    )
    assert boundary.boundary_quantiles_from_calibration() == expected


def test_boundary_results_have_complete_independent_denominators_and_chronology():
    flash = _rows("flash_boundary_episode_results.csv")
    cstr = _rows("cstr_boundary_episode_results.csv")
    assert len(flash) == 1_200 * 9
    assert len(cstr) == 300 * 2
    assert len({row["episode_id"] for row in flash}) == 1_200
    assert len({row["episode_id"] for row in cstr}) == 300
    assert all(row["population"] == "flash_boundary_challenge" for row in flash)
    assert all(row["population"] == "CSTR_BOUNDARY_CHALLENGE" for row in cstr)
    assert all(row["chronology_verified"] == "True" for row in flash + cstr)
    assert all(row["truth_sha256"] for row in flash + cstr)
    incomplete = {row["episode_id"] for row in flash if row["truth_complete"] == "False"}
    assert incomplete
    assert all(
        row["severity_is_lower_bound"] == "True"
        for row in flash
        if row["episode_id"] in incomplete
    )


def test_boundary_frontier_metrics_are_internally_consistent():
    risk = _rows("boundary_risk_availability.csv")
    assert len(risk) == 8
    flash = [
        row
        for row in risk
        if row["process"] == "flash" and row["method"] == "pooled_conformal"
    ]
    assert [float(row["confidence_level"]) for row in flash] == list(boundary.CONFIDENCE_LEVELS)
    for row in risk:
        count = int(row["episode_count"])
        certified = int(row["certified_safe_count"])
        false_safe = int(row["false_safe_count"])
        assert np.isclose(float(row["certification_availability"]), certified / count)
        assert np.isclose(float(row["joint_false_safe_risk"]), false_safe / count)
        assert np.isclose(float(row["not_certified_rate"]), 1.0 - certified / count)
    assert all(int(row["certified_safe_count"]) == 0 for row in flash)
