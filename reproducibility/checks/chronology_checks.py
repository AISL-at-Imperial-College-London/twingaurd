"""Prediction-before-truth chronology, from the recorded per-episode ledgers.

Chronology is checked on the nanosecond timestamps that each stage recorded when it wrote the
candidate, prediction, decision and truth files, and on the chronology flag stored with each
episode. The ledgers also record the SHA-256 of each per-episode file.
"""

from __future__ import annotations

from collections import defaultdict

from .common import CSTR, FLASH, LADDER, CheckResult, read_csv, truthy

LEDGERS = (
    ("flash primary", FLASH / "primary_evaluation_ledger.csv"),
    ("flash boundary", FLASH / "boundary_challenge_episodes/flash_boundary_ledger.csv"),
    ("cstr boundary", FLASH / "boundary_challenge_episodes/cstr_boundary_ledger.csv"),
    ("flash structural mismatch", FLASH / "reliability_episodes/structural_ledger.csv"),
    ("flash mixed-regime audit", FLASH / "reliability_episodes/flash_reliability_ledger.csv"),
    ("cstr mixed-regime audit", FLASH / "reliability_episodes/cstr_reliability_ledger.csv"),
    ("flash margin-ladder calibration", LADDER / "artifacts/FLASH_CALIBRATION_ARTIFACT_AUDIT.csv"),
    ("flash margin-ladder test", LADDER / "artifacts/FLASH_TEST_ARTIFACT_AUDIT.csv"),
    ("cstr margin-ladder test", LADDER / "artifacts/CSTR_TEST_ARTIFACT_AUDIT.csv"),
)
STAGES = ("candidate", "prediction", "decision", "truth")


def check_ledger_chronology() -> list[CheckResult]:
    results = []
    for name, ledger in LEDGERS:
        rows = read_csv(ledger)
        order_failures = flag_failures = 0
        examples = []
        for row in rows:
            if row.get("chronology_verified", "") != "" and not truthy(row["chronology_verified"]):
                flag_failures += 1
                examples.append(f"{row['episode_id']}: chronology_verified false")
            times = [int(row[f"{stage}_saved_ns" if stage != "truth" else "truth_revealed_ns"])
                     for stage in STAGES
                     if row.get(f"{stage}_saved_ns" if stage != "truth" else "truth_revealed_ns", "") not in ("", None)]
            if times and any(later <= earlier for earlier, later in zip(times, times[1:])):
                order_failures += 1
                examples.append(f"{row['episode_id']}: timestamps not strictly increasing")
        has_times = any(k.endswith("_ns") for k in (rows[0] if rows else {}))
        results.append(CheckResult(f"chronology:{name}", "prediction_before_truth", ledger.name, len(rows),
                                   order_failures + flag_failures,
                                   detail=("strict candidate<prediction<decision<truth nanosecond order and recorded chronology flag"
                                           if has_times else "recorded chronology flag (this ledger stores no timestamps)"),
                                   examples=examples[:5]))
    return results


def check_table_chronology() -> list[CheckResult]:
    """Per-episode result tables: recorded timestamps and hashes agree with the ledgers."""
    results = []
    for name, table, ledger in (
        ("flash primary", FLASH / "flash_primary_episode_results.csv", FLASH / "primary_evaluation_ledger.csv"),
        ("flash structural mismatch", FLASH / "flash_structural_episode_results.csv", FLASH / "reliability_episodes/structural_ledger.csv"),
        ("flash mixed-regime audit", FLASH / "flash_reliability_episode_results.csv", FLASH / "reliability_episodes/flash_reliability_ledger.csv"),
        ("flash boundary", FLASH / "flash_boundary_episode_results.csv", FLASH / "boundary_challenge_episodes/flash_boundary_ledger.csv"),
        ("cstr boundary", FLASH / "cstr_boundary_episode_results.csv", FLASH / "boundary_challenge_episodes/cstr_boundary_ledger.csv"),
        ("cstr mixed-regime audit", FLASH / "cstr_reliability_episode_results.csv", FLASH / "reliability_episodes/cstr_reliability_ledger.csv"),
    ):
        ledger_rows = {row["episode_id"]: row for row in read_csv(ledger)}
        failures, examined, examples = 0, 0, []
        for row in read_csv(table):
            source = ledger_rows.get(row["episode_id"])
            examined += 1
            if source is None:
                failures += 1
                examples.append(f"{row['episode_id']}: missing from ledger")
                continue
            for key in ("prediction_sha256", "decision_sha256", "truth_sha256", "prediction_saved_ns", "decision_saved_ns", "truth_revealed_ns"):
                if key in row and key in source and row[key] != source[key]:
                    failures += 1
                    examples.append(f"{row['episode_id']}: {key} differs from ledger")
                    break
            if "chronology_verified" in row and not truthy(row["chronology_verified"]):
                failures += 1
        results.append(CheckResult(f"table_ledger_agreement:{name}", "prediction_before_truth", table.name, examined, failures,
                                   detail="per-episode result rows carry the ledger's artifact hashes and timestamps", examples=examples[:5]))

    mixed = read_csv(CSTR / "mixed_degradation/mixed_degradation_test_episodes.csv")
    bad = [row["episode_id"] for row in mixed
           if not (int(row["prediction_saved_ns"]) < int(row["decision_saved_ns"]) < int(row["truth_revealed_ns"]))
           or not truthy(row["chronology_verified"])]
    results.append(CheckResult("chronology:cstr mixed degradation", "prediction_before_truth", "mixed_degradation_test_episodes.csv",
                               len(mixed), len(bad), detail="strict prediction<decision<truth nanosecond order", examples=bad[:5]))

    primary = read_csv(CSTR / "primary_evaluation/final_safety_episodes.csv")
    flags = defaultdict(set)
    for row in primary:
        flags[row["episode_id"]].add(truthy(row["decision_saved_before_truth_reveal"]))
    bad = [episode for episode, values in flags.items() if values != {True}]
    results.append(CheckResult("chronology:cstr primary", "prediction_before_truth", "final_safety_episodes.csv", len(flags), len(bad),
                               detail="decision snapshot saved before truth reveal (recorded flag)", examples=bad[:5]))
    audit = read_csv(CSTR / "primary_decision_audit/artifact_chronology_audit.csv")
    bad = [row["episode_id"] for row in audit if not truthy(row["chronology_valid"]) or not truthy(row["complete_record"])]
    results.append(CheckResult("chronology:cstr primary (decision reconstruction audit)", "prediction_before_truth",
                               "artifact_chronology_audit.csv", len(audit), len(bad),
                               detail="file-time order recorded by the read-only reconstruction audit of the per-episode files", examples=bad[:5]))
    return results
