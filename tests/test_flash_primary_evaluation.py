"""Audits the prospective flash evaluation artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


import numpy as np
import pytest

from twinguard.conformal.flash_conformal import safety_decision
from twinguard.flash.evaluate_primary import (
    DECISION_ALLOWED_KEYS,
    FINAL_COUNT,
    FORBIDDEN_PRETRUTH_TOKENS,
    METHODS,
    PREDICTION_ALLOWED_KEYS,
    verify_frozen_inputs,
)


ROOT = Path(__file__).resolve().parents[1]
FLASH = ROOT / "artifacts" / "flash_study"
ARTIFACTS = FLASH / "primary_evaluation_episodes"


def _rows(name: str) -> list[dict[str, str]]:
    with (FLASH / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_calibration_stage_inputs_still_validate() -> None:
    frozen = verify_frozen_inputs()
    assert frozen["pooled_q"] == 11.04859850832729
    assert frozen["config"]["ensemble"]["selected"] == 10
    assert frozen["config"]["mondrian"]["supported"] is False


def test_primary_metrics_and_frozen_verdict_are_reproducible() -> None:
    ablation = _rows("flash_primary_ablation.csv")
    m3 = next(
        row
        for row in ablation
        if row["group_type"] == "pooled" and row["method"] == "M3_pooled_conformal"
    )
    assert int(m3["episode_count"]) == 1200
    assert int(m3["certified_safe_count"]) == 177
    assert int(m3["false_safe_count"]) == 5
    assert float(m3["joint_false_safe_wilson_95_upper"]) <= 0.01
    assert float(m3["certification_rate"]) >= 0.10


