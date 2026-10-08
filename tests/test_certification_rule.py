"""Certification rule and finite-sample conformal quantiles, checked against the frozen calibration files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from twinguard.conformal.flash_conformal import corrected_quantile, safety_decision
from twinguard.cstr import safety_certification as cstr

ROOT = Path(__file__).resolve().parents[1]
FLASH = ROOT / "artifacts" / "flash_study"
CSTR = ROOT / "artifacts" / "cstr"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_corrected_quantile_uses_finite_sample_rank() -> None:
    scores = np.arange(1.0, 20.0)  # n = 19
    value, rank = corrected_quantile(scores, 0.05)
    assert rank == int(np.ceil(20 * 0.95)) == 19
    assert value == 19.0
    value, rank = corrected_quantile(scores, 0.10)
    assert rank == 18 and value == 18.0


def test_flash_decision_requires_whole_envelope_inside_limits() -> None:
    center = np.tile([0.5, 0.5, 340.0], (120, 1))
    tight = np.tile([1e-4, 1e-4, 1e-2], (120, 1))
    wide = np.tile([1.0, 1.0, 100.0], (120, 1))
    assert safety_decision(center, tight, 1.0) == "SAFE"
    assert safety_decision(center, wide, 1.0) == "NOT CERTIFIED"
    assert safety_decision(center, tight, None) == "NOT CERTIFIED"
    assert safety_decision(center, tight, 1.0, supported=False) == "NOT CERTIFIED"


def test_cstr_decision_is_safe_only_when_upper_envelope_stays_below_limit() -> None:
    limit = cstr.T_MAX_K
    assert cstr.safety_decision(np.full(5, limit - 2), np.full(5, limit - 1)) == "SAFE"
    assert cstr.safety_decision(np.full(5, limit + 1), np.full(5, limit + 2)) == "UNSAFE"
    assert cstr.safety_decision(np.full(5, limit - 1), np.full(5, limit + 1)) == "UNRESOLVED"


def test_frozen_flash_quantiles_are_order_statistics_of_their_own_calibration_population() -> None:
    config = json.loads((FLASH / "flash_conformal_config.json").read_text(encoding="utf-8"))
    scores = _rows(FLASH / "flash_calibration_scores.csv")
    for population in ("calibration_ordinary", "calibration_boundary"):
        own = [float(r["whole_trajectory_safety_score"]) for r in scores if r["calibration_population"] == population]
        for alpha_text, entry in config["pooled_quantiles"][population].items():
            value, rank = corrected_quantile(own, float(alpha_text))
            assert value == entry["quantile"]
            assert rank == entry["corrected_rank"]


def test_frozen_cstr_one_sided_quantile_matches_calibration_scores() -> None:
    quantiles = _rows(CSTR / "primary_evaluation" / "one_sided_quantiles.csv")
    scores = _rows(CSTR / "primary_evaluation" / "safety_calibration_scores.csv")
    upper = [row for row in quantiles if row["direction"] == "upper"]
    assert upper, "one-sided quantile table must contain the upper side"
    values = np.sort([float(r["upper_score"]) for r in scores])
    for row in upper:
        rank = int(row["corrected_rank_k"])
        assert rank == min(int(np.ceil((len(values) + 1) * (1.0 - float(row["miscoverage_alpha"])))), len(values))
        assert float(row["conformal_quantile"]) == values[rank - 1]
