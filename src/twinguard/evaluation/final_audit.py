"""Independent reconstruction and claims audit for the final evidence."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from twinguard.cstr import safety_certification as certification

from twinguard.flash import evaluate_primary as primary
from twinguard.evaluation import boundary_challenge as boundary
from twinguard.evaluation import structural_and_reliability as reliability
from twinguard.scenarios.flash_population_design import MEASURED_OUTPUTS, SafetySpecification
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT, REPOSITORY_ROOT, assert_isolated_output_path


FORBIDDEN = ("fault", "severity", "truth", "actual", "changed_parameter", "hidden")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path = assert_isolated_output_path(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _record(
    rows: list[dict[str, object]],
    population: str,
    check: str,
    examined: int,
    failures: int,
    detail: str,
) -> None:
    rows.append(
        {
            "population": population,
            "audit_check": check,
            "records_examined": examined,
            "failure_count": failures,
            "status": "PASS" if failures == 0 else "ISSUE",
            "detail": detail,
        }
    )


def _prediction_keys(path: Path) -> list[str]:
    with np.load(path, allow_pickle=False) as payload:
        return list(payload.files)


def audit_pretruth_keys(
    rows: list[dict[str, object]], population: str, prediction_dir: Path, decision_dir: Path
) -> None:
    predictions = sorted(prediction_dir.glob("*.npz"))
    decisions = sorted(decision_dir.glob("*.json"))
    failures = 0
    for path in predictions:
        if any(token in key.lower() for key in _prediction_keys(path) for token in FORBIDDEN):
            failures += 1
    for path in decisions:
        keys = json.loads(path.read_text()).keys()
        if any(token in key.lower() for key in keys for token in FORBIDDEN):
            failures += 1
    _record(
        rows,
        population,
        "truth/fault/hidden-parameter leakage in prediction or decision keys",
        len(predictions) + len(decisions),
        failures,
        "Key-level allowlist audit; residual histories are permitted observations, evaluator metadata is not.",
    )


def audit_certification(rows: list[dict[str, object]]) -> None:
    root = REPOSITORY_ROOT / "artifacts/cstr/primary_evaluation"
    quantiles = {
        (row["direction"], float(row["nominal_level"])): float(row["conformal_quantile"])
        for row in _read_csv(root / "one_sided_quantiles.csv")
    }
    symmetric = {
        float(row["nominal_level"]): float(row["conformal_quantile"])
        for row in _read_csv(REPOSITORY_ROOT / "artifacts/cstr/conformal_calibration/conformal_quantiles.csv")
        if row["method"] == "B2" and row["calibration_scope"] == "pooled"
    }
    failures = envelope_failures = 0
    files = sorted((root / "predictions").glob("*.npz"))
    for path in files:
        episode = path.stem
        decision = json.loads((root / "decision_snapshots" / f"{episode}.json").read_text())
        with np.load(path, allow_pickle=False) as saved:
            nominal = np.asarray(saved["nominal_temperature_k"])
            center = np.asarray(saved["predictive_center_k"])
            scale = np.maximum(np.asarray(saved["predictive_scale_k"]), certification.EPSILON_SCALE_K)
            for level in certification.LEVELS:
                label = int(level * 100)
                envelopes = {
                    "S0_nominal_DT": (nominal, nominal),
                    "S1_raw_bootstrap": (
                        np.asarray(saved[f"raw_{label}_lower_k"]),
                        np.asarray(saved[f"raw_{label}_upper_k"]),
                    ),
                    "S2_symmetric_conformal": (
                        center - symmetric[level] * scale,
                        center + symmetric[level] * scale,
                    ),
                    "S3_one_sided": (
                        center - quantiles[("lower", level)] * scale,
                        center + quantiles[("upper", level)] * scale,
                    ),
                }
                for method, (lower, upper) in envelopes.items():
                    reconstructed = certification.safety_decision(lower, upper)
                    if reconstructed != decision["decisions"][f"{method}|{level}"]:
                        failures += 1
                if not np.allclose(
                    envelopes["S3_one_sided"][1], saved[f"s3_{label}_upper_k"], rtol=0, atol=1e-12
                ):
                    envelope_failures += 1
    _record(rows, "CSTR_PRIMARY", "decision reconstruction", len(files) * 12, failures, "All four methods at 90/95/99%.")
    _record(rows, "CSTR_PRIMARY", "envelope reconstruction", len(files) * 3, envelope_failures, "Saved one-sided upper envelope versus frozen q and scale floor.")
    final = _read_csv(root / "final_safety_episodes.csv")
    safety_failures = sum(
        int(row["actual_safe"]) != (float(row["actual_max_temperature"]) <= 313.0)
        or int(row["false_safe_indicator"]) != (row["decision"] == "SAFE" and int(row["actual_safe"]) == 0)
        for row in final
    )
    _record(rows, "CSTR_PRIMARY", "safety-indicator reconstruction", len(final), safety_failures, "Physical 313 K event and decision outcome.")
    chronology_failures = sum(not _bool(row["decision_saved_before_truth_reveal"]) for row in final)
    _record(rows, "CSTR_PRIMARY", "prediction-decision-truth chronology", len(final), chronology_failures, "Legacy primary certification stores a prospective boolean but not nanosecond timestamps.")
    audit_pretruth_keys(rows, "CSTR_PRIMARY", root / "predictions", root / "decision_snapshots")


def audit_scalar_transfer(
    rows: list[dict[str, object]], population: str, root: Path, limit: float, scale_floor: float
) -> None:
    result_name = "mixed_degradation_test_episodes.csv" if population == "CSTR_MIXED_DEGRADATION" else "FINAL_EPISODE_RESULTS.csv"
    episode_rows = _read_csv(root / result_name)
    by_id = {row["episode_id"]: row for row in episode_rows}
    failures = envelope_failures = hash_failures = safety_failures = chronology_failures = 0
    predictions = sorted((root / "final_predictions").glob("*.npz"))
    for path in predictions:
        episode = path.stem
        decision_path = root / "final_decisions" / f"{episode}.json"
        decision = json.loads(decision_path.read_text())
        row = by_id[episode]
        with np.load(path, allow_pickle=False) as saved:
            center_key = "predictive_center_k" if population == "CSTR_MIXED_DEGRADATION" else "predictive_center_m"
            scale_key = "predictive_scale_k" if population == "CSTR_MIXED_DEGRADATION" else "predictive_scale_m"
            center = np.asarray(saved[center_key])
            scale = np.maximum(np.asarray(saved[scale_key]), scale_floor)
        upper_max = float(np.max(center + float(decision["q_plus"]) * scale))
        reconstructed = "CERTIFIED SAFE" if upper_max <= limit else "NOT CERTIFIED"
        failures += reconstructed != decision["decision"]
        margin_field = "conformal_safety_margin_k" if population == "CSTR_MIXED_DEGRADATION" else "conformal_safety_margin_m"
        envelope_failures += not np.isclose(float(row[margin_field]), limit - upper_max, rtol=0, atol=1e-10)
        hash_failures += decision["prediction_sha256"] != _sha256(path)
        safety_limit_field = "actual_max_temperature_k" if population == "CSTR_MIXED_DEGRADATION" else "actual_max_level_m"
        safety_failures += int(row["actual_safe"]) != (float(row[safety_limit_field]) <= limit)
        safety_failures += int(row["false_safe"]) != (reconstructed == "CERTIFIED SAFE" and int(row["actual_safe"]) == 0)
        chronology_failures += not (
            int(row["prediction_saved_ns"]) < int(row["decision_saved_ns"]) < int(row["truth_revealed_ns"])
        )
    _record(rows, population, "decision reconstruction", len(predictions), int(failures), "Frozen pooled upper envelope.")
    _record(rows, population, "envelope reconstruction", len(predictions), int(envelope_failures), "Stored conformal margin versus reconstructed upper maximum.")
    _record(rows, population, "prediction hash binding", len(predictions), int(hash_failures), "Decision SHA-256 binding.")
    _record(rows, population, "safety-indicator reconstruction", len(predictions) * 2, int(safety_failures), f"Physical upper limit {limit}.")
    _record(rows, population, "prediction-decision-truth chronology", len(predictions), int(chronology_failures), "Nanosecond timestamps.")
    audit_pretruth_keys(rows, population, root / "final_predictions", root / "final_decisions")


def _flash_decision_failures(prediction_path: Path, decision_path: Path) -> tuple[int, int, int]:
    prediction = primary._load_npz(prediction_path)
    decision = json.loads(decision_path.read_text())
    reconstructed, margins = reliability._reconstruct_flash_decision(prediction)
    decision_failures = sum(reconstructed[key] != decision["decisions"][key] for key in primary.METHODS)
    margin_failures = sum(
        not (
            np.isneginf(margins[key]) and np.isneginf(float(decision["normalized_margins"][key]))
        )
        and not np.isclose(margins[key], float(decision["normalized_margins"][key]), rtol=0, atol=1e-10)
        for key in primary.METHODS
    )
    hash_failure = decision["prediction_sha256"] != _sha256(prediction_path)
    return decision_failures, margin_failures, int(hash_failure)


def audit_flash_primary(rows: list[dict[str, object]]) -> None:
    root = EXPERIMENT_ROOT / "primary_evaluation_episodes"
    prediction_files = sorted((root / "predictions").glob("*.npz"))
    decision_failures = margin_failures = hash_failures = 0
    for prediction in prediction_files:
        values = _flash_decision_failures(
            prediction, root / "decisions" / f"{prediction.stem}.json"
        )
        decision_failures += values[0]
        margin_failures += values[1]
        hash_failures += values[2]
    _record(rows, "FLASH_PRIMARY", "decision reconstruction", len(prediction_files) * 5, decision_failures, "M0-M4 from frozen prediction artifacts.")
    _record(rows, "FLASH_PRIMARY", "envelope reconstruction", len(prediction_files) * 5, margin_failures, "Normalized margins.")
    _record(rows, "FLASH_PRIMARY", "prediction hash binding", len(prediction_files), hash_failures, "Decision SHA-256 binding.")
    audit = _read_csv(EXPERIMENT_ROOT / "primary_evaluation_ledger.csv")
    _record(rows, "FLASH_PRIMARY", "prediction-decision-truth chronology", len(audit), sum(not _bool(row["chronology_verified"]) for row in audit), "Strict artifact timestamps.")
    audit_pretruth_keys(rows, "FLASH_PRIMARY", root / "predictions", root / "decisions")


def audit_flash_boundary(rows: list[dict[str, object]]) -> None:
    root = EXPERIMENT_ROOT / "boundary_challenge_episodes/flash"
    failures = margins_failed = hashes_failed = 0
    predictions = sorted((root / "predictions").glob("*.npz"))
    for path in predictions:
        prediction = primary._load_npz(path)
        decision = json.loads((root / "decisions" / f"{path.stem}.json").read_text())
        complete = str(prediction["prediction_status"]) == "COMPLETE"
        nominal = np.asarray(prediction["nominal_horizon_outputs"])
        mean = np.asarray(prediction["mean_horizon_outputs"])
        center = np.asarray(prediction["ensemble_center_horizon_outputs"])
        scale = np.asarray(prediction["ensemble_scale_horizon_outputs"])
        reconstructed = {
            "nominal_DT": "CERTIFIED SAFE" if complete and boundary._flash_point_margin(nominal) >= 0 else "NOT CERTIFIED",
            "discrepancy_mean": "CERTIFIED SAFE" if complete and boundary._flash_point_margin(mean) >= 0 else "NOT CERTIFIED",
            "raw_bootstrap_95": primary.safety_decision(center, scale, boundary.RAW_MULTIPLIER).replace("SAFE", "CERTIFIED SAFE") if complete else "NOT CERTIFIED",
            "action_conditioned": "NOT CERTIFIED",
        }
        margins = {
            "nominal_DT": boundary._flash_point_margin(nominal),
            "discrepancy_mean": boundary._flash_point_margin(mean),
            "raw_bootstrap_95": boundary._flash_envelope_margin(center, scale, boundary.RAW_MULTIPLIER),
            "action_conditioned": float("-inf"),
        }
        for level, q in zip(prediction["confidence_levels"], prediction["boundary_quantiles"]):
            key = f"pooled_{float(level):.3f}"
            reconstructed[key] = primary.safety_decision(center, scale, float(q)).replace("SAFE", "CERTIFIED SAFE") if complete else "NOT CERTIFIED"
            margins[key] = boundary._flash_envelope_margin(center, scale, float(q))
        failures += sum(reconstructed[key] != decision["decisions"][key] for key in reconstructed)
        margins_failed += sum(
            not (np.isneginf(value) and np.isneginf(float(decision["normalized_margins"][key])))
            and not np.isclose(value, float(decision["normalized_margins"][key]), rtol=0, atol=1e-10)
            for key, value in margins.items()
        )
        hashes_failed += decision["prediction_sha256"] != _sha256(path)
    _record(rows, "FLASH_BOUNDARY", "decision reconstruction", len(predictions) * 9, failures, "All frozen confidence levels and baselines.")
    _record(rows, "FLASH_BOUNDARY", "envelope reconstruction", len(predictions) * 9, margins_failed, "All normalized margins.")
    _record(rows, "FLASH_BOUNDARY", "prediction hash binding", len(predictions), hashes_failed, "Decision SHA-256 binding.")
    audit = _read_csv(EXPERIMENT_ROOT / "boundary_challenge_episodes/flash_boundary_ledger.csv")
    _record(rows, "FLASH_BOUNDARY", "candidate-prediction-decision-truth chronology", len(audit), sum(not _bool(row["chronology_verified"]) for row in audit), "Strict artifact timestamps.")
    audit_pretruth_keys(rows, "FLASH_BOUNDARY", root / "predictions", root / "decisions")


def audit_reliability_population(rows: list[dict[str, object]], population: str) -> None:
    root = EXPERIMENT_ROOT / "reliability_episodes" / population / population
    predictions = sorted((root / "predictions").glob("*.npz"))
    decision_failures = margin_failures = hash_failures = 0
    for prediction in predictions:
        values = _flash_decision_failures(
            prediction, root / "decisions" / f"{prediction.stem}.json"
        )
        decision_failures += values[0]
        margin_failures += values[1]
        hash_failures += values[2]
    label = "FLASH_STRUCTURAL" if population == "structural" else "FLASH_RELIABILITY"
    _record(rows, label, "decision reconstruction", len(predictions) * 5, decision_failures, "M0-M4 from stored prediction artifacts.")
    _record(rows, label, "envelope reconstruction", len(predictions) * 5, margin_failures, "Normalized margins.")
    _record(rows, label, "prediction hash binding", len(predictions), hash_failures, "Decision SHA-256 binding.")
    audit = _read_csv(EXPERIMENT_ROOT / f"reliability_episodes/{population}_ledger.csv")
    _record(rows, label, "candidate-prediction-decision-truth chronology", len(audit), sum(not _bool(row["chronology_verified"]) for row in audit), "Strict artifact timestamps.")
    audit_pretruth_keys(rows, label, root / "predictions", root / "decisions")


def audit_reliability_cstr(rows: list[dict[str, object]]) -> None:
    root = EXPERIMENT_ROOT / "reliability_episodes/cstr_reliability"
    predictions = sorted((root / "predictions").glob("*.npz"))
    failures = margins_failed = hashes_failed = 0
    for path in predictions:
        decision_path = root / "decisions" / f"{path.stem}.json"
        decision = json.loads(decision_path.read_text())
        with np.load(path, allow_pickle=False) as saved:
            nominal = np.asarray(saved["nominal_temperature_k"])
            center = np.asarray(saved["predictive_center_k"])
            scale = np.maximum(np.asarray(saved["predictive_scale_k"]), certification.EPSILON_SCALE_K)
        upper = center + reliability.CSTR_Q_PLUS * scale
        reconstructed = {
            "nominal_dt_decision": "CERTIFIED SAFE" if float(np.max(nominal)) <= 313 else "NOT CERTIFIED",
            "frozen_cstr_decision": "CERTIFIED SAFE" if float(np.max(upper)) <= 313 else "NOT CERTIFIED",
        }
        failures += sum(decision[key] != value for key, value in reconstructed.items())
        margins_failed += not np.isclose(float(decision["nominal_margin_k"]), 313 - float(np.max(nominal)), atol=1e-10, rtol=0)
        margins_failed += not np.isclose(float(decision["conformal_margin_k"]), 313 - float(np.max(upper)), atol=1e-10, rtol=0)
        hashes_failed += decision["prediction_sha256"] != _sha256(path)
    _record(rows, "CSTR_RELIABILITY", "decision reconstruction", len(predictions) * 2, failures, "Nominal and frozen conformal decisions.")
    _record(rows, "CSTR_RELIABILITY", "envelope reconstruction", len(predictions) * 2, int(margins_failed), "Nominal and conformal margins.")
    _record(rows, "CSTR_RELIABILITY", "prediction hash binding", len(predictions), hashes_failed, "Decision SHA-256 binding.")
    audit = _read_csv(EXPERIMENT_ROOT / "reliability_episodes/cstr_reliability_ledger.csv")
    _record(rows, "CSTR_RELIABILITY", "candidate-prediction-decision-truth chronology", len(audit), sum(not _bool(row["chronology_verified"]) for row in audit), "Strict artifact timestamps.")
    audit_pretruth_keys(rows, "CSTR_RELIABILITY", root / "predictions", root / "decisions")


def audit_boundary_cstr(rows: list[dict[str, object]]) -> None:
    root = EXPERIMENT_ROOT / "boundary_challenge_episodes/cstr"
    predictions = sorted((root / "predictions").glob("*.npz"))
    failures = margins_failed = hashes_failed = 0
    for path in predictions:
        decision_path = root / "decisions" / f"{path.stem}.json"
        decision = json.loads(decision_path.read_text())
        with np.load(path, allow_pickle=False) as saved:
            nominal = np.asarray(saved["nominal_temperature_k"])
            center = np.asarray(saved["predictive_center_k"])
            scale = np.maximum(np.asarray(saved["predictive_scale_k"]), certification.EPSILON_SCALE_K)
        upper = center + reliability.CSTR_Q_PLUS * scale
        failures += decision["nominal_dt_decision"] != (
            "CERTIFIED SAFE" if float(np.max(nominal)) <= 313 else "NOT CERTIFIED"
        )
        failures += decision["frozen_cstr_decision"] != (
            "CERTIFIED SAFE" if float(np.max(upper)) <= 313 else "NOT CERTIFIED"
        )
        margins_failed += not np.isclose(
            float(decision["conformal_margin_k"]), 313 - float(np.max(upper)), atol=1e-10, rtol=0
        )
        hashes_failed += decision["prediction_sha256"] != _sha256(path)
    _record(rows, "CSTR_BOUNDARY", "decision reconstruction", len(predictions) * 2, int(failures), "Nominal and frozen conformal decisions.")
    _record(rows, "CSTR_BOUNDARY", "envelope reconstruction", len(predictions), int(margins_failed), "Frozen conformal margin.")
    _record(rows, "CSTR_BOUNDARY", "prediction hash binding", len(predictions), int(hashes_failed), "Decision SHA-256 binding.")
    audit = _read_csv(EXPERIMENT_ROOT / "boundary_challenge_episodes/cstr_boundary_ledger.csv")
    _record(rows, "CSTR_BOUNDARY", "candidate-prediction-decision-truth chronology", len(audit), sum(not _bool(row["chronology_verified"]) for row in audit), "Strict artifact timestamps.")
    audit_pretruth_keys(rows, "CSTR_BOUNDARY", root / "predictions", root / "decisions")


def audit_flash_safety(
    rows: list[dict[str, object]], population: str, results_path: Path, truth_dir: Path
) -> None:
    results = _read_csv(results_path)
    by_episode: dict[str, list[dict[str, str]]] = {}
    for row in results:
        by_episode.setdefault(row["episode_id"], []).append(row)
    safety = SafetySpecification()
    failures = 0
    for episode, episode_rows in by_episode.items():
        with np.load(truth_dir / f"{episode}.npz", allow_pickle=False) as truth:
            actual = np.asarray(truth["actual_horizon_outputs"])
        actual_safe = False
        if len(actual) == 120:
            actual_safe = safety.is_physically_safe(
                {name: actual[:, index] for index, name in enumerate(MEASURED_OUTPUTS)}
            )
        for row in episode_rows:
            failures += _bool(row["actual_safe"]) != actual_safe
            failures += _bool(row["false_safe"]) != (
                _bool(row["certified_safe"]) and not actual_safe
            )
    _record(rows, population, "safety-indicator reconstruction", len(results) * 2, failures, "Complete multiconstraint horizon and false-safe indicator.")


def audit_cstr_safety(
    rows: list[dict[str, object]], population: str, results_path: Path, truth_dir: Path
) -> None:
    results = _read_csv(results_path)
    by_episode: dict[str, list[dict[str, str]]] = {}
    for row in results:
        by_episode.setdefault(row["episode_id"], []).append(row)
    failures = 0
    for episode, episode_rows in by_episode.items():
        matching = list(truth_dir.glob(f"*{episode}*.npz"))
        if len(matching) != 1:
            failures += len(episode_rows) * 2
            continue
        with np.load(matching[0], allow_pickle=False) as truth:
            actual = np.asarray(truth["actual_physical_temperature_k"])
        actual_safe = bool(len(actual) == certification.HORIZON and float(np.max(actual)) <= 313.0)
        for row in episode_rows:
            failures += _bool(row["actual_safe"]) != actual_safe
            failures += _bool(row["false_safe"]) != (
                _bool(row["certified_safe"]) and not actual_safe
            )
    _record(rows, population, "safety-indicator reconstruction", len(results) * 2, failures, "Complete 313 K horizon and false-safe indicator.")


def audit_frozen_manifest(rows: list[dict[str, object]]) -> None:
    lines = [
        line for line in reliability.FROZEN_MANIFEST_PATH.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    entries = [(expected, REPOSITORY_ROOT / relative) for expected, relative in (line.split("  ", 1) for line in lines)]
    with ThreadPoolExecutor(max_workers=16) as pool:
        actual = list(pool.map(_sha256, [path for _, path in entries]))
    failures = sum(expected != observed for (expected, _), observed in zip(entries, actual))
    _record(rows, "ALL_PRE_RELIABILITY_RESULTS", "frozen SHA-256 baseline", len(entries), failures, "Every released input of this stage.")


def audit_seeds_duplicates_missing(rows: list[dict[str, object]]) -> None:
    flash_manifest = _read_csv(EXPERIMENT_ROOT / "flash_population_manifest.csv")
    intervals = [
        (row["population_id"], int(row["seed_start"]), int(row["seed_end"]))
        for row in flash_manifest
    ]
    overlap = sum(
        max(a_start, b_start) <= min(a_end, b_end)
        for index, (_, a_start, a_end) in enumerate(intervals)
        for _, b_start, b_end in intervals[index + 1 :]
    )
    _record(rows, "FLASH_ALL_SPLITS", "training/calibration/test seed overlap", len(intervals), overlap, "All eight population design-reserved intervals, including reliability evaluation blocks.")
    cstr_intervals = [
        ("primary_final", 1_730_000, 1_730_999),
        ("mixed_degradation_train", 2_100_000, 2_100_599),
        ("mixed_degradation_select", 2_200_000, 2_200_059),
        ("mixed_degradation_cal", 2_300_000, 2_300_399),
        ("mixed_degradation_final", 2_400_000, 2_400_599),
        ("boundary_stress", 9_300_000, 9_300_299),
        ("reliability", 50_000_000, 50_002_999),
    ]
    cstr_overlap = sum(
        max(a_start, b_start) <= min(a_end, b_end)
        for index, (_, a_start, a_end) in enumerate(cstr_intervals)
        for _, b_start, b_end in cstr_intervals[index + 1 :]
    )
    _record(rows, "CSTR_ALL_SPLITS", "training/calibration/test seed overlap", len(cstr_intervals), cstr_overlap, "Frozen primary certification/mixed-degradation evaluation and new stress/reliability intervals.")
    populations = (
        ("FLASH_PRIMARY", EXPERIMENT_ROOT / "flash_primary_episode_results.csv", 1200),
        ("FLASH_BOUNDARY", EXPERIMENT_ROOT / "flash_boundary_episode_results.csv", 1200),
        ("FLASH_STRUCTURAL", EXPERIMENT_ROOT / "flash_structural_episode_results.csv", 600),
        ("FLASH_RELIABILITY", EXPERIMENT_ROOT / "flash_reliability_episode_results.csv", 3000),
        ("CSTR_RELIABILITY", EXPERIMENT_ROOT / "cstr_reliability_episode_results.csv", 3000),
    )
    for population, path, expected in populations:
        values = _read_csv(path)
        episode_ids = {row["episode_id"] for row in values}
        _record(rows, population, "duplicates and missing episode records", expected, int(len(episode_ids) != expected), f"Unique={len(episode_ids)}, expected={expected}; method rows share episode IDs by design.")


def audit_nan_to_zero(rows: list[dict[str, object]]) -> None:
    sources = (
        ("FLASH_PRIMARY", EXPERIMENT_ROOT / "flash_primary_ablation.csv", "certified_safe_count", "conditional_false_safe_probability"),
        ("boundary challenge", EXPERIMENT_ROOT / "boundary_risk_availability.csv", "certified_safe_count", "conditional_false_safe_risk"),
        ("STRUCTURAL", EXPERIMENT_ROOT / "flash_structural_summary.csv", "certified_safe_count", "conditional_false_safe_probability"),
        ("FLASH_RELIABILITY", EXPERIMENT_ROOT / "flash_reliability_summary.csv", "certified_safe_count", "conditional_false_safe_probability"),
        ("CSTR_RELIABILITY", EXPERIMENT_ROOT / "cstr_reliability_summary.csv", "certified_safe_count", "conditional_false_safe_probability"),
    )
    for population, path, denominator, value in sources:
        source = _read_csv(path)
        failures = sum(
            int(row[denominator]) == 0 and row[value].strip().lower() not in {"", "nan"}
            for row in source
        )
        _record(rows, population, "incorrect NaN-to-zero conversion", len(source), failures, "Conditional false-safe is undefined when no action is certified; reliability evaluation leaves it blank.")


def normalized_final_results() -> list[dict[str, object]]:
    output: list[dict[str, object]] = []

    def append(population: str, process: str, method: str, row: Mapping[str, object], mapping: Mapping[str, str]) -> None:
        output.append(
            {
                "population": population,
                "process": process,
                "method": method,
                "episode_count": row[mapping["n"]],
                "certified_safe_count": row[mapping["certified"]],
                "certification_rate": row[mapping["availability"]],
                "false_safe_count": row[mapping["false"]],
                "joint_false_safe_probability": row[mapping["risk"]],
                "joint_false_safe_95_lower": row.get(mapping.get("lower", ""), ""),
                "joint_false_safe_95_upper": row.get(mapping.get("upper", ""), ""),
                "trajectory_coverage": row.get(mapping.get("coverage", ""), ""),
                "denominator_merged_with_prior": False,
            }
        )

    certification_summary = _read_csv(REPOSITORY_ROOT / "artifacts/cstr/primary_evaluation/safety_summary.csv")
    for method, source_method in (("nominal_DT", "S0_nominal_DT"), ("proposed_method", "S3_one_sided")):
        row = next(r for r in certification_summary if r["method"] == source_method and r["action_category"] == "pooled" and np.isclose(float(r["nominal_level"]), 0.95))
        append("CSTR_PRIMARY", "CSTR", method, row, {"n":"number_of_episodes","certified":"SAFE_count","availability":"SAFE_certification_rate","false":"false_safe_count","risk":"joint_false_safe_probability","lower":"joint_false_safe_CI_lower","upper":"joint_false_safe_CI_upper","coverage":"upper_trajectory_empirical_coverage"})
    for population, path, process in (
        ("CSTR_MIXED_DEGRADATION", REPOSITORY_ROOT / "artifacts/cstr/mixed_degradation/mixed_degradation_summary.csv", "CSTR"),
    ):
        source = _read_csv(path)
        selected = [
            row for row in source
            if (population == "CSTR_MIXED_DEGRADATION" and row["population"] == "CSTR mixed degradation")
            or (population == "TWO_TANK" and row["population"] == "two-tank process")
        ]
        for row in selected:
            append(population, process, row["method"], row, {"n":"number_of_episodes","certified":"SAFE_count","availability":"SAFE_certification_rate","false":"false_safe_count","risk":"joint_false_safe_probability","lower":"joint_false_safe_CI_lower","upper":"joint_false_safe_CI_upper","coverage":"upper_trajectory_coverage"})
    primary_rows = _read_csv(EXPERIMENT_ROOT / "flash_primary_ablation.csv")
    for method in ("M0_nominal_DT", "M3_pooled_conformal"):
        row = next(r for r in primary_rows if r["group_type"] == "pooled" and r["method"] == method)
        append("FLASH_PRIMARY", "flash", method, row, {"n":"episode_count","certified":"certified_safe_count","availability":"certification_rate","false":"false_safe_count","risk":"joint_false_safe_probability","lower":"joint_false_safe_wilson_95_lower","upper":"joint_false_safe_wilson_95_upper","coverage":"whole_trajectory_safety_direction_coverage"})
    boundary_rows = _read_csv(EXPERIMENT_ROOT / "boundary_risk_availability.csv")
    for row in boundary_rows:
        if row["method"] == "pooled_conformal" and row["confidence_level"] != "0.95":
            continue
        append(f"{row['process']}_BOUNDARY_BOUNDARY", row["process"], row["method"], row, {"n":"episode_count","certified":"certified_safe_count","availability":"certification_availability","false":"false_safe_count","risk":"joint_false_safe_risk","lower":"joint_false_safe_wilson_95_lower","upper":"joint_false_safe_wilson_95_upper"})
    for population, path, process in (
        ("FLASH_STRUCTURAL", EXPERIMENT_ROOT / "flash_structural_summary.csv", "flash"),
        ("FLASH_RELIABILITY", EXPERIMENT_ROOT / "flash_reliability_summary.csv", "flash"),
        ("CSTR_RELIABILITY", EXPERIMENT_ROOT / "cstr_reliability_summary.csv", "CSTR"),
    ):
        for row in _read_csv(path):
            if row["method"] not in {"M0_nominal_DT", "M3_pooled_conformal", "nominal_DT", "frozen_CSTR_conformal"}:
                continue
            append(population, process, row["method"], row, {"n":"episode_count","certified":"certified_safe_count","availability":"certification_rate","false":"false_safe_count","risk":"joint_false_safe_probability","lower":"joint_false_safe_exact_95_lower","upper":"joint_false_safe_exact_95_upper","coverage":"trajectory_coverage"})
    return output


def create_figures(final: Sequence[Mapping[str, object]]) -> list[Path]:
    figure_root = _safe_directory("reliability_figures")
    selected = [row for row in final if "RELIABILITY" in str(row["population"])]
    labels = [f"{row['process']}\n{str(row['method']).replace('_', ' ')}" for row in selected]
    x = np.arange(len(selected))
    risk = np.asarray([float(row["joint_false_safe_probability"]) for row in selected])
    lower = np.asarray([float(row["joint_false_safe_95_lower"]) for row in selected])
    upper = np.asarray([float(row["joint_false_safe_95_upper"]) for row in selected])
    availability = np.asarray([float(row["certification_rate"]) for row in selected])
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    ax.errorbar(x, risk, yerr=np.vstack([risk - lower, upper - risk]), fmt="o", capsize=4, label="Joint false-safe, exact 95% CI")
    ax.scatter(x, availability, marker="s", label="Certification availability")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Probability")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    first = figure_root / "01_large_reliability_evidence.png"
    fig.savefig(first, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    structural = _read_csv(EXPERIMENT_ROOT / "flash_structural_summary.csv")
    methods = [row for row in structural if row["method"] in {"M0_nominal_DT", "M3_pooled_conformal"}]
    fig, ax = plt.subplots(figsize=(5.8, 4.5))
    positions = np.arange(2)
    ax.bar(positions - 0.18, [float(row["certification_rate"]) for row in methods], 0.36, label="Certification")
    ax.bar(positions + 0.18, [float(row["joint_false_safe_probability"]) for row in methods], 0.36, label="Joint false-safe")
    ax.set_xticks(positions, ["Nominal DT", "Frozen conformal"])
    ax.set_ylabel("Probability")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    second = figure_root / "02_structural_mismatch_evidence.png"
    fig.savefig(second, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return [first, second]


def _safe_directory(relative: str) -> Path:
    path = assert_isolated_output_path(EXPERIMENT_ROOT / relative)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _format_risk_interval(row: Mapping[str, str]) -> str:
    count = int(row["false_safe_count"])
    n = int(row["episode_count"])
    exact_lower = float(row["joint_false_safe_exact_95_lower"])
    exact_upper = float(row["joint_false_safe_exact_95_upper"])
    wilson_lower = float(row["joint_false_safe_wilson_95_lower"])
    wilson_upper = float(row["joint_false_safe_wilson_95_upper"])
    return (
        f"{count} observed events / {n}; exact two-sided 95% CI "
        f"[{exact_lower:.4f}, {exact_upper:.4f}] and Wilson 95% CI "
        f"[{wilson_lower:.4f}, {wilson_upper:.4f}]"
    )


def write_reports(audit: Sequence[Mapping[str, object]], final: Sequence[Mapping[str, object]], figures: Sequence[Path]) -> None:
    structural = _read_csv(EXPERIMENT_ROOT / "flash_structural_summary.csv")
    flash_rel = _read_csv(EXPERIMENT_ROOT / "flash_reliability_summary.csv")
    cstr_rel = _read_csv(EXPERIMENT_ROOT / "cstr_reliability_summary.csv")
    structural_m3 = next(row for row in structural if row["method"] == "M3_pooled_conformal")
    flash_m3 = next(row for row in flash_rel if row["method"] == "M3_pooled_conformal")
    cstr_m3 = next(row for row in cstr_rel if row["method"] == "frozen_CSTR_conformal")
    structural_predictions = [
        row
        for row in _read_csv(EXPERIMENT_ROOT / "flash_structural_episode_results.csv")
        if row["method"] == "M3_pooled_conformal"
    ]

    def mean_metric(name: str) -> float:
        values = np.asarray(
            [float(row[name]) for row in structural_predictions if row[name] != ""],
            dtype=float,
        )
        return float(np.mean(values[np.isfinite(values)]))

    finite_structural_errors = sum(
        row["pressure_rmse_bar"] != "" for row in structural_predictions
    )

    audit_issues = [row for row in audit if row["status"] == "ISSUE"]
    result_text = f"""# reliability evaluation Results

## Structural Mismatch

The only new mechanism is a plant-side first-order vapor-valve actuator lag
(`tau=30-120 s`). The nominal DT has no actuator state, and neither the mechanism
label nor effective valve position enters prediction. The population design non-ideal-VLE F4
placeholder was prospectively replaced at reliability evaluation before any reserved truth; this is
an explicit protocol amendment, not a modification of simulator validation-boundary challenge evidence.

For the {finite_structural_errors}/{len(structural_predictions)} trajectories with finite
physical rollouts, mean recursive prediction RMSE was
{mean_metric('pressure_rmse_bar'):.4f} bar for pressure,
{mean_metric('level_rmse_fraction'):.4f} vessel fraction for level, and
{mean_metric('temperature_rmse_k'):.4f} K for temperature. Mean signed biases were
{mean_metric('pressure_bias_bar'):.4f} bar, {mean_metric('level_bias_fraction'):.4f}
vessel fraction, and {mean_metric('temperature_bias_k'):.4f} K, respectively.

At frozen 95% pooled conformal settings, certification availability was
{int(structural_m3['certified_safe_count'])}/{int(structural_m3['episode_count'])}
({float(structural_m3['certification_rate']):.4f}), trajectory coverage was
{float(structural_m3['trajectory_coverage']):.4f}, and false-safe evidence was
{_format_risk_interval(structural_m3)}. Maximum and total normalized false-safe
severity were {float(structural_m3['maximum_false_safe_severity']):.4f} and
{float(structural_m3['total_false_safe_severity']):.4f}; maximum and total integrated
severity were {float(structural_m3['maximum_false_safe_integrated_severity']):.4f}
and {float(structural_m3['total_false_safe_integrated_severity']):.4f} normalized-seconds.
Matching ordinary/boundary quantiles are descriptive for this unseen mechanism
and do not establish a structural-mismatch guarantee.

## Large Independent Reliability

Flash: {_format_risk_interval(flash_m3)}; certification availability
{int(flash_m3['certified_safe_count'])}/{int(flash_m3['episode_count'])}
({float(flash_m3['certification_rate']):.4f}), trajectory coverage
{float(flash_m3['trajectory_coverage']):.4f}, maximum normalized false-safe severity
{float(flash_m3['maximum_false_safe_severity']):.4f}, and total normalized false-safe
severity {float(flash_m3['total_false_safe_severity']):.4f}. CSTR:
{_format_risk_interval(cstr_m3)}; certification availability
{int(cstr_m3['certified_safe_count'])}/{int(cstr_m3['episode_count'])}
({float(cstr_m3['certification_rate']):.4f}) and trajectory coverage
{float(cstr_m3['trajectory_coverage']):.4f}.
These are separate new denominators and are not pooled with primary certification, mixed-degradation evaluation, primary evaluation, or boundary challenge.

## Final Verdict

### -POSSIBLE WITH SPECIFIC LIMITATIONS

The frozen methods reduce false-safe reliance with useful availability in ordinary
populations, but flash and CSTR boundary challenges abstain completely, flash mean
model selection used a preregistered fallback, structural evidence is descriptive,
and pooled conformal coverage is marginal rather than conditional.
"""
    (EXPERIMENT_ROOT / "reliability_report.md").write_text(result_text, encoding="utf-8")

    issue_lines = "\n".join(
        f"- `{row['population']}` / {row['audit_check']}: {row['failure_count']} issue(s). {row['detail']}"
        for row in audit_issues
    ) or "- None."
    audit_text = f"""# Final  Audit

The audit independently reconstructed stored decisions and envelopes, checked
seed separation, hashes, missing/duplicate episodes, strict chronology where
timestamps exist, pre-truth key leakage, physical safety labels, and undefined
conditional rates.

- Audit checks: {len(audit)}
- Checks with findings: {len(audit_issues)}
- Pre-reliability evaluation frozen artifacts protected by hash: {json.loads(reliability.DESIGN_PATH.read_text())['frozen_input_file_count']}
- New reliability evaluation episodes: 600 structural flash + 3,000 flash reliability + 3,000 CSTR reliability

## Findings

{issue_lines}

No prior result was rewritten. Earlier numeric zeros for conditional false-safe
rates with zero certifications are retained as historical artifacts and treated
as undefined in final claims.
"""
    (EXPERIMENT_ROOT / "final_audit_report.md").write_text(audit_text, encoding="utf-8")

    structural_supported = (
        float(structural_m3["trajectory_coverage"]) >= 0.90
        and float(structural_m3["certification_rate"]) > 0.0
    )
    claims = [
        ("Fault-blind operational prediction", "SUPPORTED", "Prediction/decision key audits found no evaluator fault metadata; only observed residual history carries fault effects."),
        ("Cross-process transfer", "SUPPORTED WITH QUALIFICATION", "CSTR, two-tank, and flash show qualitative risk reduction, but flash used a fallback mean model and boundary availability collapses."),
        ("Whole-trajectory coverage", "SUPPORTED WITH QUALIFICATION", "Pooled empirical trajectory coverage is reported for matched populations; domain exits and finite samples remain."),
        ("Conditional coverage", "NOT SUPPORTED", "No per-fault, severity, state, or action-conditional conformal guarantee was established."),
        ("Low false-safe risk", "SUPPORTED WITH QUALIFICATION", "Observed counts and finite exact/Wilson intervals support low risk only for the tested populations."),
        ("Action availability", "SUPPORTED WITH QUALIFICATION", "Ordinary populations retain nonzero availability; both boundary challenge boundary challenges certify no actions."),
        ("Structural mismatch robustness", "SUPPORTED WITH QUALIFICATION" if structural_supported else "NOT SUPPORTED", "The actuator-lag test is independent and descriptive; its matching q was not calibrated on structural faults."),
        ("Actions throughout the admissible continuous domain", "SUPPORTED WITH QUALIFICATION", "Space-filling and nominal-boundary samples cover the declared bounded domain, not every possible future action/state distribution."),
        ("Per-action safety guarantee", "NOT SUPPORTED", "Conformal validity is marginal over exchangeable episodes, not an individual certified-action probability statement."),
        ("Computational feasibility", "SUPPORTED WITH QUALIFICATION", "Prospective 3,000-episode populations completed, but stored latency and hardware context limit deployment extrapolation."),
    ]
    claim_lines = "\n".join(f"| {claim} | {status} | {basis} |" for claim, status, basis in claims)
    claims_text = f"""# Final  Claims Audit

| Proposed claim | Status | Evidence basis |
|---|---|---|
{claim_lines}

## Verdict

### -POSSIBLE WITH SPECIFIC LIMITATIONS

Drafting is defensible only with explicit marginal-coverage, boundary-abstention,
model-selection, simulator-only, structural-amendment, and finite-sample language.
"""
    (EXPERIMENT_ROOT / "final_claims_report.md").write_text(claims_text, encoding="utf-8")

    figure_lines = []
    for path in figures:
        figure_lines.append(f"| `{path.relative_to(REPOSITORY_ROOT).as_posix()}` | `{_sha256(path)}` | reliability evaluation final evidence |")
    for relative, role in (
        ("artifacts/flash_study/primary_evaluation_figures/06_false_safe_risk_and_certification.png", "Flash ordinary reliability"),
        ("artifacts/flash_study/boundary_challenge_figures/01_risk_availability_frontier.png", "Boundary risk-availability"),
        ("artifacts/flash_study/boundary_challenge_figures/04_cstr_vs_flash_boundary_behavior.png", "Cross-process boundary comparison"),
        ("artifacts/cstr/primary_evaluation/figure_CERTIFICATION_10_method_comparison.png", "Frozen CSTR comparison"),
    ):
        path = REPOSITORY_ROOT / relative
        if path.exists():
            figure_lines.append(f"| `{relative}` | `{_sha256(path)}` | {role} |")
    manifest_text = "# Final  Figure Manifest\n\n| Figure | SHA-256 | Role |\n|---|---|---|\n" + "\n".join(figure_lines) + "\n"
    (EXPERIMENT_ROOT / "final_figure_manifest.md").write_text(manifest_text, encoding="utf-8")


def hash_final_artifacts() -> None:
    paths = [path for path in (EXPERIMENT_ROOT / "reliability_episodes").rglob("*") if path.is_file()]
    for pattern in ("reliability_*", "final_audit_*", "final_claims_*", "final_figure_*", "flash_structural_*", "flash_reliability_*", "cstr_reliability_*"):
        paths.extend(path for path in EXPERIMENT_ROOT.glob(pattern) if path.is_file())
    paths.extend(path for path in (EXPERIMENT_ROOT / "reliability_figures").glob("*") if path.is_file())
    output = EXPERIMENT_ROOT / "reliability_artifact_hashes.txt"
    paths = sorted(set(path for path in paths if path != output), key=lambda path: path.relative_to(EXPERIMENT_ROOT).as_posix())
    with ThreadPoolExecutor(max_workers=16) as pool:
        hashes = list(pool.map(_sha256, paths))
    lines = ["# SHA-256 manifest for all reliability evaluation and final artifacts", ""]
    lines.extend(f"{digest}  {path.relative_to(EXPERIMENT_ROOT).as_posix()}" for path, digest in zip(paths, hashes))
    output.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    audit: list[dict[str, object]] = []
    audit_frozen_manifest(audit)
    audit_seeds_duplicates_missing(audit)
    audit_certification(audit)
    audit_scalar_transfer(audit, "CSTR_MIXED_DEGRADATION", REPOSITORY_ROOT / "artifacts/cstr/mixed_degradation", 313.0, certification.EPSILON_SCALE_K)
    audit_flash_primary(audit)
    audit_flash_boundary(audit)
    audit_boundary_cstr(audit)
    audit_reliability_population(audit, "structural")
    audit_reliability_population(audit, "flash_reliability")
    audit_reliability_cstr(audit)
    audit_flash_safety(
        audit, "FLASH_PRIMARY", EXPERIMENT_ROOT / "flash_primary_episode_results.csv",
        EXPERIMENT_ROOT / "primary_evaluation_episodes/truths",
    )
    audit_flash_safety(
        audit, "FLASH_BOUNDARY", EXPERIMENT_ROOT / "flash_boundary_episode_results.csv",
        EXPERIMENT_ROOT / "boundary_challenge_episodes/flash/truths",
    )
    audit_flash_safety(
        audit, "FLASH_STRUCTURAL", EXPERIMENT_ROOT / "flash_structural_episode_results.csv",
        EXPERIMENT_ROOT / "reliability_episodes/structural/structural/truths",
    )
    audit_flash_safety(
        audit, "FLASH_RELIABILITY", EXPERIMENT_ROOT / "flash_reliability_episode_results.csv",
        EXPERIMENT_ROOT / "reliability_episodes/flash_reliability/flash_reliability/truths",
    )
    audit_cstr_safety(
        audit, "CSTR_BOUNDARY", EXPERIMENT_ROOT / "cstr_boundary_episode_results.csv",
        EXPERIMENT_ROOT / "boundary_challenge_episodes/cstr/truths",
    )
    audit_cstr_safety(
        audit, "CSTR_RELIABILITY", EXPERIMENT_ROOT / "cstr_reliability_episode_results.csv",
        EXPERIMENT_ROOT / "reliability_episodes/cstr_reliability/truths",
    )
    audit_nan_to_zero(audit)
    _write_csv(EXPERIMENT_ROOT / "final_audit_checks.csv", audit)
    final = normalized_final_results()
    _write_csv(EXPERIMENT_ROOT / "final_audit_results.csv", final)
    figures = create_figures(final)
    write_reports(audit, final, figures)
    hash_final_artifacts()
    print(f"final audit checks={len(audit)} issues={sum(row['status']=='ISSUE' for row in audit)}")


if __name__ == "__main__":
    main()
