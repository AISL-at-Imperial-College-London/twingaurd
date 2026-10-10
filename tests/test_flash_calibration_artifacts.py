"""Checks flash calibration artifacts and evaluator-only data leakage."""

from __future__ import annotations

import csv
import hashlib
import json
from math import ceil
from pathlib import Path



ROOT = Path(__file__).resolve().parents[1]
FLASH = ROOT / "artifacts" / "flash_study"


def _csv_rows(name: str) -> list[dict[str, str]]:
    with (FLASH / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_bootstrap_manifest_uses_only_numerical_rejection_rules() -> None:
    rows = _csv_rows("flash_bootstrap_manifest.csv")
    accepted = [row for row in rows if row["numerically_accepted"] == "True"]

    assert len(rows) == 150
    assert len(accepted) == 100
    assert {row["bootstrap_seed"] for row in rows} == {
        str(seed) for seed in range(31_000_000, 31_000_150)
    }
    assert all(row["performance_based_rejection"] == "False" for row in rows)
    assert {
        row["rejection_reason"] for row in rows if row["rejection_reason"]
    } <= {"not_converged", "target_already_filled"}


def test_conformal_config_records_no_test_truth_access() -> None:
    config = json.loads((FLASH / "flash_conformal_config.json").read_text())

    assert config["prohibited_truth_accessed"] == []
    assert all("final" not in role for role in config["roles_accessed"])
    assert config["ensemble"]["selected"] == 10
    assert config["ensemble"]["predictive_trajectories"] == 2000
    assert config["score"]["directions"] == [
        "high_pressure",
        "high_temperature",
        "high_level",
        "low_level",
    ]
    assert config["unsafe_certificate_defined"] is False
    assert config["mondrian"]["supported"] is False
    assert config["mondrian"]["unsupported_decision"] == "NOT CERTIFIED"


def test_scores_are_complete_calibration_episodes_only() -> None:
    scores = _csv_rows("flash_calibration_scores.csv")
    populations = {row["calibration_population"] for row in scores}
    counts = {
        population: sum(
            row["calibration_population"] == population for row in scores
        )
        for population in populations
    }

    assert populations == {"calibration_ordinary", "calibration_boundary"}
    assert counts == {"calibration_ordinary": 447, "calibration_boundary": 463}
    assert {row["maximizing_direction"] for row in scores} == {
        "high_pressure",
        "high_temperature",
        "high_level",
        "low_level",
    }
    assert all(row["ensemble_size"] == "10" for row in scores)
    assert all(row["predictive_trajectory_count"] == "2000" for row in scores)


def test_stored_quantiles_match_corrected_score_order_statistics() -> None:
    scores = _csv_rows("flash_calibration_scores.csv")
    config = json.loads((FLASH / "flash_conformal_config.json").read_text())

    for population, alpha_results in config["pooled_quantiles"].items():
        values = sorted(
            float(row["whole_trajectory_safety_score"])
            for row in scores
            if row["calibration_population"] == population
        )
        for alpha_text, result in alpha_results.items():
            alpha = float(alpha_text)
            rank = min(ceil((len(values) + 1) * (1.0 - alpha)), len(values))
            assert result["sample_count"] == len(values)
            assert result["corrected_rank"] == rank
            assert result["quantile"] == values[rank - 1]


def test_frozen_artifact_hashes_match() -> None:
    manifest = FLASH / "calibration_artifact_hashes.txt"
    entries = [
        line.split("  ", maxsplit=1)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]

    assert entries
    for expected, relative in entries:
        actual = hashlib.sha256((FLASH / relative).read_bytes()).hexdigest()
        assert actual == expected, relative
