"""Regression checks for the frozen E7 reliability evidence package."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
E7 = ROOT / "artifacts" / "flash_study"
ARCHIVE_PRESENT = (E7 / "e7_artifacts" / "structural").is_dir()


def _rows(name: str) -> list[dict[str, str]]:
    with (E7 / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_e7_design_hash_is_frozen() -> None:
    design = E7 / "RESS_E7_DESIGN.json"
    recorded = (E7 / "RESS_E7_DESIGN.sha256").read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(design.read_bytes()).hexdigest() == recorded
    payload = json.loads(design.read_text(encoding="utf-8"))
    assert payload["flash_reliability"]["episode_count"] == 3000
    assert payload["cstr_reliability"]["episode_count"] == 3000
    assert payload["structural_amendment"]["episode_count"] == 600


def test_large_reliability_denominators_and_zero_event_intervals() -> None:
    flash = {row["method"]: row for row in _rows("FLASH_RELIABILITY_SUMMARY.csv")}
    cstr = {row["method"]: row for row in _rows("CSTR_RELIABILITY_SUMMARY.csv")}

    assert {int(row["episode_count"]) for row in flash.values()} == {3000}
    assert {int(row["episode_count"]) for row in cstr.values()} == {3000}

    cstr_frozen = cstr["frozen_CSTR_conformal"]
    assert int(cstr_frozen["false_safe_count"]) == 0
    assert 0.0 < float(cstr_frozen["joint_false_safe_exact_95_upper"]) < 0.002

    unsupported = flash["M4_action_conditioned"]
    assert int(unsupported["certified_safe_count"]) == 0
    assert unsupported["conditional_false_safe_probability"] == ""
    assert unsupported["conditional_false_safe_exact_95_upper"] == ""


@pytest.mark.skipif(not ARCHIVE_PRESENT, reason="needs the frozen per-episode archive extracted at the repository root (see data/README.md)")
def test_e7_artifacts_are_complete_and_prospective() -> None:
    expected = {
        "STRUCTURAL_ARTIFACT_AUDIT.csv": 600,
        "FLASH_RELIABILITY_ARTIFACT_AUDIT.csv": 3000,
        "CSTR_RELIABILITY_ARTIFACT_AUDIT.csv": 3000,
    }
    for name, count in expected.items():
        rows = _rows(f"e7_artifacts/{name}")
        assert len(rows) == count
        assert all(row["chronology_verified"].lower() == "true" for row in rows)
        assert all((ROOT / row["candidate_path"]).exists() for row in rows)
        assert all((ROOT / row["prediction_path"]).exists() for row in rows)
        assert all((ROOT / row["decision_path"]).exists() for row in rows)
        assert all((ROOT / row["truth_path"]).exists() for row in rows)


def test_final_audit_has_no_e7_integrity_findings() -> None:
    rows = _rows("RESS_FINAL_AUDIT_DETAILS.csv")
    e7_issues = [
        row
        for row in rows
        if row["status"] == "ISSUE"
        and (
            row["population"].startswith("E7_")
            or row["population"].endswith("E7_RELIABILITY")
        )
    ]
    assert e7_issues == []
    frozen = [row for row in rows if row["audit_check"] == "frozen SHA-256 baseline"]
    assert len(frozen) == 1
    assert frozen[0]["status"] == "PASS"
    assert int(frozen[0]["records_examined"]) == 34222


def test_final_verdict_is_limitations_qualified() -> None:
    claims = (E7 / "RESS_FINAL_CLAIMS_AUDIT.md").read_text(encoding="utf-8")
    results = (E7 / "RESS_E7_RESULTS.md").read_text(encoding="utf-8")
    verdict = "RESS-POSSIBLE WITH SPECIFIC LIMITATIONS"
    assert verdict in claims
    assert verdict in results
    assert "0 observed events / 3000" in results
