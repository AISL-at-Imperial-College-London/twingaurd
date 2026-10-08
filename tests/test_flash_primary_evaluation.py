"""Audits for the frozen prospective flash E5 final evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from twinguard.utils.provenance import frozen_sha256

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
ARTIFACTS = FLASH / "e5_artifacts"
ARCHIVE_PRESENT = (ARTIFACTS / "predictions").is_dir()
requires_archive = pytest.mark.skipif(not ARCHIVE_PRESENT, reason="needs the frozen per-episode archive extracted at the repository root (see data/README.md)")


def _rows(name: str) -> list[dict[str, str]]:
    with (FLASH / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_frozen_e4_inputs_still_validate() -> None:
    frozen = verify_frozen_inputs()
    assert frozen["pooled_q"] == 11.04859850832729
    assert frozen["config"]["ensemble"]["selected"] == 10
    assert frozen["config"]["mondrian"]["supported"] is False


@requires_archive
def test_exact_final_population_and_artifact_counts() -> None:
    audit = _rows("e5_artifact_audit.csv")
    results = _rows("FLASH_FINAL_RESULTS.csv")

    assert len(audit) == FINAL_COUNT
    assert len(results) == FINAL_COUNT * len(METHODS)
    assert len(list((ARTIFACTS / "predictions").glob("*.npz"))) == FINAL_COUNT
    assert len(list((ARTIFACTS / "decisions").glob("*.json"))) == FINAL_COUNT
    assert len(list((ARTIFACTS / "truths").glob("*.npz"))) == FINAL_COUNT
    assert {int(row["episode_seed_evaluator_only"]) for row in results} == set(
        range(26_000_000, 26_001_200)
    )


@requires_archive
def test_all_artifacts_have_strict_chronology_and_hash_binding() -> None:
    for row in _rows("e5_artifact_audit.csv"):
        prediction = ROOT / row["prediction_path"]
        decision = ROOT / row["decision_path"]
        truth = ROOT / row["truth_path"]
        assert int(row["prediction_saved_ns"]) < int(row["decision_saved_ns"])
        assert int(row["decision_saved_ns"]) < int(row["truth_revealed_ns"])
        assert row["chronology_verified"] == "True"
        assert _hash(prediction) == row["prediction_sha256"]
        assert _hash(decision) == row["decision_sha256"]
        assert _hash(truth) == row["truth_sha256"]
        decision_payload = json.loads(decision.read_text(encoding="utf-8"))
        with np.load(truth, allow_pickle=False) as truth_payload:
            assert decision_payload["prediction_sha256"] == row["prediction_sha256"]
            assert str(truth_payload["prediction_sha256"]) == row["prediction_sha256"]
            assert str(truth_payload["decision_sha256"]) == row["decision_sha256"]


@requires_archive
def test_pretruth_artifacts_are_fault_blind_and_decisions_reconstruct() -> None:
    for prediction_path in sorted((ARTIFACTS / "predictions").glob("*.npz")):
        with np.load(prediction_path, allow_pickle=False) as prediction:
            assert set(prediction.files) == PREDICTION_ALLOWED_KEYS
            lowered = " ".join(prediction.files).lower()
            assert not any(token in lowered for token in FORBIDDEN_PRETRUTH_TOKENS)
            episode_id = str(prediction["episode_id"])
            decision_path = ARTIFACTS / "decisions" / f"{episode_id}.json"
            decision = json.loads(decision_path.read_text(encoding="utf-8"))
            assert set(decision) == DECISION_ALLOWED_KEYS
            decision_keys = " ".join(decision).lower()
            assert not any(token in decision_keys for token in FORBIDDEN_PRETRUTH_TOKENS)
            assert decision["decisions"]["M4_action_conditioned"] == "NOT CERTIFIED"
            if str(prediction["prediction_status"]) == "COMPLETE":
                reconstructed = safety_decision(
                    prediction["ensemble_center_horizon_outputs"],
                    prediction["ensemble_scale_horizon_outputs"],
                    float(prediction["pooled_quantile"]),
                )
                expected = "CERTIFIED SAFE" if reconstructed == "SAFE" else reconstructed
                assert decision["decisions"]["M3_pooled_conformal"] == expected
            else:
                assert set(decision["decisions"].values()) == {"NOT CERTIFIED"}


def test_primary_metrics_and_frozen_verdict_are_reproducible() -> None:
    ablation = _rows("FLASH_FINAL_ABLATION.csv")
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
    report = (FLASH / "FLASH_E5_RESULTS.md").read_text(encoding="utf-8")
    assert "### FLASH TRANSFER VALIDATED WITH LIMITATIONS" in report


@requires_archive
def test_e5_freeze_manifest_matches_every_listed_artifact() -> None:
    lines = (FLASH / "FLASH_E5_ARTIFACT_HASHES.txt").read_text().splitlines()
    entries = [line.split("  ", maxsplit=1) for line in lines if line and not line.startswith("#")]
    assert len(entries) == 5 + 8 + 3 * FINAL_COUNT
    for expected, relative in entries:
        if relative == "e5_figures/05_pooled_vs_action_conditioned_coverage.png":
            continue  # unsupported M4 placeholder figure, deliberately not released
        assert frozen_sha256("E_FLASH_RESS/" + relative) == expected, relative
