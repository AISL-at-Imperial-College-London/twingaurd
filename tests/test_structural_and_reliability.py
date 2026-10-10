"""Checks structural-mismatch and mixed-regime reliability artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
RELIABILITY_ROOT = ROOT / "artifacts" / "flash_study"


def _rows(name: str) -> list[dict[str, str]]:
    with (RELIABILITY_ROOT / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_reliability_design_hash_is_frozen() -> None:
    design = RELIABILITY_ROOT / "reliability_design.json"
    recorded = (RELIABILITY_ROOT / "reliability_design.sha256").read_text(encoding="ascii").split()[0]
    assert hashlib.sha256(design.read_bytes()).hexdigest() == recorded
    payload = json.loads(design.read_text(encoding="utf-8"))
    assert payload["flash_reliability"]["episode_count"] == 3000
    assert payload["cstr_reliability"]["episode_count"] == 3000
    assert payload["structural_amendment"]["episode_count"] == 600


def test_large_reliability_denominators_and_zero_event_intervals() -> None:
    flash = {row["method"]: row for row in _rows("flash_reliability_summary.csv")}
    cstr = {row["method"]: row for row in _rows("cstr_reliability_summary.csv")}

    assert {int(row["episode_count"]) for row in flash.values()} == {3000}
    assert {int(row["episode_count"]) for row in cstr.values()} == {3000}

    cstr_frozen = cstr["frozen_CSTR_conformal"]
    assert int(cstr_frozen["false_safe_count"]) == 0
    assert 0.0 < float(cstr_frozen["joint_false_safe_exact_95_upper"]) < 0.002

    unsupported = flash["M4_action_conditioned"]
    assert int(unsupported["certified_safe_count"]) == 0
    assert unsupported["conditional_false_safe_probability"] == ""
    assert unsupported["conditional_false_safe_exact_95_upper"] == ""


def test_final_audit_has_no_reliability_integrity_findings() -> None:
    rows = _rows("final_audit_checks.csv")
    reliability_issues = [
        row
        for row in rows
        if row["status"] == "ISSUE"
        and (
            row["population"].startswith("RELIABILITY_")
            or row["population"].endswith("RELIABILITY")
        )
    ]
    assert reliability_issues == []
    frozen = [row for row in rows if row["audit_check"] == "frozen SHA-256 baseline"]
    assert len(frozen) == 1
    assert frozen[0]["status"] == "PASS"
    assert int(frozen[0]["records_examined"]) == 34222


