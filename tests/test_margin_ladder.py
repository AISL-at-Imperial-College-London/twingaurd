"""Isolation and prospective-design tests for E6B."""

from __future__ import annotations

import csv
import hashlib
import inspect
import json
from pathlib import Path

import pytest

from twinguard.utils.margin_ladder_output_guard import EXPERIMENT_ROOT, REPOSITORY_ROOT, assert_e6b_output
from twinguard.evaluation.margin_ladder import _band_for_margin, _select_flash_action
from twinguard.evaluation.margin_ladder_preregistration import (
    CSTR_TEST_COUNT,
    CSTR_TEST_START,
    FLASH_CALIBRATION_COUNT,
    FLASH_CALIBRATION_START,
    FLASH_TEST_COUNT,
    FLASH_TEST_START,
    PRIOR_RESERVED_INTERVALS,
)


def test_e6b_output_isolation_rejects_frozen_roots() -> None:
    assert assert_e6b_output(EXPERIMENT_ROOT / "new.csv").parent == EXPERIMENT_ROOT.resolve()
    for frozen in (
        REPOSITORY_ROOT / "artifacts" / "flash_study" / "FLASH_FINAL_RESULTS.csv",
        REPOSITORY_ROOT / "artifacts" / "cstr" / "primary_evaluation" / "final_safety_episodes.csv",
    ):
        with pytest.raises(PermissionError):
            assert_e6b_output(frozen)


def test_margin_band_boundaries_are_frozen() -> None:
    assert _band_for_margin(0.0) is None
    assert _band_for_margin(0.15) == "B0"
    assert _band_for_margin(0.15000001) == "B1"
    assert _band_for_margin(0.30) == "B1"
    assert _band_for_margin(0.60) == "B2"
    assert _band_for_margin(1.00) == "B3"
    assert _band_for_margin(1.000001) == "B4"


def test_seed_roots_are_disjoint_from_prior_reservations() -> None:
    new_intervals = (
        (FLASH_CALIBRATION_START, FLASH_CALIBRATION_START + FLASH_CALIBRATION_COUNT - 1),
        (FLASH_TEST_START, FLASH_TEST_START + FLASH_TEST_COUNT - 1),
        (CSTR_TEST_START, CSTR_TEST_START + CSTR_TEST_COUNT - 1),
    )
    for lower, upper in new_intervals:
        for _, old_lower, old_upper in PRIOR_RESERVED_INTERVALS:
            assert max(lower, old_lower) > min(upper, old_upper)
    for index, (lower, upper) in enumerate(new_intervals):
        for other_lower, other_upper in new_intervals[index + 1 :]:
            assert max(lower, other_lower) > min(upper, other_upper)


def test_flash_selector_has_no_plant_or_truth_argument() -> None:
    assert tuple(inspect.signature(_select_flash_action).parameters) == (
        "decision_state",
        "seed",
        "assigned_band",
    )


def test_preregistration_hash_when_present() -> None:
    prereg = EXPERIMENT_ROOT / "E6B_PREDECLARATION.md"
    recorded = EXPERIMENT_ROOT / "E6B_PREDECLARATION.sha256"
    if prereg.exists() or recorded.exists():
        assert prereg.exists() and recorded.exists()
        expected = recorded.read_text(encoding="ascii").split()[0]
        assert hashlib.sha256(prereg.read_bytes()).hexdigest() == expected


def test_completed_population_and_calibration_accounting() -> None:
    flash_path = EXPERIMENT_ROOT / "FLASH_E6B_EPISODES.csv"
    cstr_path = EXPERIMENT_ROOT / "CSTR_E6B_EPISODES.csv"
    config_path = EXPERIMENT_ROOT / "FLASH_E6B_CONFORMAL_CONFIG.json"
    if not flash_path.exists():
        pytest.skip("E6B execution artifacts are not present")

    with flash_path.open(newline="", encoding="utf-8") as handle:
        flash_rows = list(csv.DictReader(handle))
    with cstr_path.open(newline="", encoding="utf-8") as handle:
        cstr_rows = list(csv.DictReader(handle))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    assert len(flash_rows) == FLASH_TEST_COUNT
    assert len(cstr_rows) == CSTR_TEST_COUNT
    assert {row["assigned_margin_band"] for row in flash_rows} == {"B0", "B1", "B2", "B3", "B4"}
    assert all(entry["supported"] for entry in config["bands"].values())
    assert all(entry["complete_calibration_count"] >= 200 for entry in config["bands"].values())


def test_completed_audit_and_verdict() -> None:
    audit_path = EXPERIMENT_ROOT / "E6B_ARTIFACT_AUDIT.csv"
    claims_path = EXPERIMENT_ROOT / "E6B_CLAIMS_AUDIT.md"
    if not audit_path.exists():
        pytest.skip("E6B execution artifacts are not present")

    with audit_path.open(newline="", encoding="utf-8") as handle:
        audit_rows = list(csv.DictReader(handle))
    assert audit_rows
    assert all(row["status"] == "PASS" and int(row["failure_count"]) == 0 for row in audit_rows)
    assert "### E6B METHOD REMAINS OVERLY CONSERVATIVE" in claims_path.read_text(encoding="utf-8")
