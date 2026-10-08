import csv
import json
from pathlib import Path

import numpy as np
import pytest

from twinguard.evaluation import boundary_challenge as e6


ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "flash_study"
ARCHIVE_PRESENT = (ROOT / "e6_artifacts" / "flash" / "predictions").is_dir()


def _rows(name: str) -> list[dict[str, str]]:
    with (ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_e6_uses_frozen_design_and_calibration_only_quantiles():
    design = json.loads((ROOT / "RESS_E6_DESIGN.json").read_text(encoding="utf-8"))
    assert design["frozen_before_boundary_truth"] is True
    assert design["flash"]["seed_start"] == 27_000_000
    assert design["flash"]["episode_count"] == 1_200
    assert design["cstr"]["seed_start"] == 9_300_000
    assert design["cstr"]["episode_count"] == 300
    expected = dict(
        zip(design["flash"]["confidence_levels"], design["flash"]["quantiles"])
    )
    assert e6.boundary_quantiles_from_calibration() == expected


def test_e6_results_have_complete_independent_denominators_and_chronology():
    flash = _rows("FLASH_BOUNDARY_RESULTS.csv")
    cstr = _rows("CSTR_BOUNDARY_RESULTS.csv")
    assert len(flash) == 1_200 * 9
    assert len(cstr) == 300 * 2
    assert len({row["episode_id"] for row in flash}) == 1_200
    assert len({row["episode_id"] for row in cstr}) == 300
    assert all(row["population"] == "flash_boundary_challenge" for row in flash)
    assert all(row["population"] == "CSTR_RESS_STRESS_BOUNDARY" for row in cstr)
    assert all(row["chronology_verified"] == "True" for row in flash + cstr)
    assert all(row["truth_sha256"] for row in flash + cstr)
    incomplete = {row["episode_id"] for row in flash if row["truth_complete"] == "False"}
    assert incomplete
    assert all(
        row["severity_is_lower_bound"] == "True"
        for row in flash
        if row["episode_id"] in incomplete
    )


@pytest.mark.skipif(not ARCHIVE_PRESENT, reason="needs the frozen per-episode archive extracted at the repository root (see data/README.md)")
def test_e6_pretruth_artifacts_are_fault_blind_and_bound_by_hash():
    forbidden = ("fault", "severity", "truth", "actual_safe", "changed_parameter")
    roots = (
        ROOT / "e6_artifacts" / "flash",
        ROOT / "e6_artifacts" / "cstr",
    )
    for process_root in roots:
        candidates = sorted((process_root / "candidates").glob("*.json"))
        predictions = sorted((process_root / "predictions").glob("*.npz"))
        decisions = sorted((process_root / "decisions").glob("*.json"))
        assert candidates and len(candidates) == len(predictions) == len(decisions)
        for path in candidates + decisions:
            keys = json.loads(path.read_text(encoding="utf-8")).keys()
            assert not any(token in key.lower() for key in keys for token in forbidden)
        for path in predictions:
            with np.load(path, allow_pickle=False) as payload:
                assert not any(
                    token in key.lower() for key in payload.files for token in forbidden
                )


def test_e6_frontier_metrics_are_internally_consistent():
    risk = _rows("RISK_AVAILABILITY_RESULTS.csv")
    assert len(risk) == 8
    flash = [
        row
        for row in risk
        if row["process"] == "flash" and row["method"] == "pooled_conformal"
    ]
    assert [float(row["confidence_level"]) for row in flash] == list(e6.CONFIDENCE_LEVELS)
    for row in risk:
        count = int(row["episode_count"])
        certified = int(row["certified_safe_count"])
        false_safe = int(row["false_safe_count"])
        assert np.isclose(float(row["certification_availability"]), certified / count)
        assert np.isclose(float(row["joint_false_safe_risk"]), false_safe / count)
        assert np.isclose(float(row["not_certified_rate"]), 1.0 - certified / count)
    assert all(int(row["certified_safe_count"]) == 0 for row in flash)
    assert "BOUNDARY RELIABILITY NOT DEMONSTRATED" in (
        ROOT / "RESS_E6_RESULTS.md"
    ).read_text(encoding="utf-8")
