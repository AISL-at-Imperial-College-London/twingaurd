"""Prediction-before-truth chronology and hash binding, from the recorded per-episode ledgers.

Filesystem modification times of the development copy are no longer trustworthy (files were
copied and line endings converted), so chronology is checked on the nanosecond timestamps the
stages recorded at write time, and on the hash chain that binds each truth file to the earlier
prediction and decision files.
"""

from __future__ import annotations

import re
from collections import defaultdict

from .common import CSTR, FLASH, LADDER, CheckResult, read_csv, truthy

LEDGERS = (
    ("flash primary", FLASH / "e5_artifact_audit.csv", FLASH / "FLASH_E5_ARTIFACT_HASHES.txt", "artifacts/flash_study/"),
    ("flash boundary", FLASH / "e6_artifacts/FLASH_RESS_STRESS_ARTIFACT_AUDIT.csv", FLASH / "RESS_E6_ARTIFACT_HASHES.txt", "artifacts/flash_study/"),
    ("cstr boundary", FLASH / "e6_artifacts/CSTR_RESS_STRESS_ARTIFACT_AUDIT.csv", FLASH / "RESS_E6_ARTIFACT_HASHES.txt", "artifacts/flash_study/"),
    ("flash structural mismatch", FLASH / "e7_artifacts/STRUCTURAL_ARTIFACT_AUDIT.csv", FLASH / "RESS_FINAL_ARTIFACT_HASHES.txt", "artifacts/flash_study/"),
    ("flash mixed-regime audit", FLASH / "e7_artifacts/FLASH_RELIABILITY_ARTIFACT_AUDIT.csv", FLASH / "RESS_FINAL_ARTIFACT_HASHES.txt", "artifacts/flash_study/"),
    ("cstr mixed-regime audit", FLASH / "e7_artifacts/CSTR_RELIABILITY_ARTIFACT_AUDIT.csv", FLASH / "RESS_FINAL_ARTIFACT_HASHES.txt", "artifacts/flash_study/"),
    ("flash margin-ladder calibration", LADDER / "artifacts/FLASH_CALIBRATION_ARTIFACT_AUDIT.csv", LADDER / "E6B_FINAL_ARTIFACT_HASHES.txt", "artifacts/flash_margin_ladder/"),
    ("flash margin-ladder test", LADDER / "artifacts/FLASH_TEST_ARTIFACT_AUDIT.csv", LADDER / "E6B_FINAL_ARTIFACT_HASHES.txt", "artifacts/flash_margin_ladder/"),
    ("cstr margin-ladder test", LADDER / "artifacts/CSTR_TEST_ARTIFACT_AUDIT.csv", LADDER / "E6B_FINAL_ARTIFACT_HASHES.txt", "artifacts/flash_margin_ladder/"),
)
STAGES = ("candidate", "prediction", "decision", "truth")


def _manifest(path):
    entries = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-f]{64})  (.+)$", line)
        if match:
            entries[match.group(2)] = match.group(1)
    return entries


def check_ledger_chronology() -> list[CheckResult]:
    results = []
    for name, ledger, manifest_path, prefix in LEDGERS:
        rows = read_csv(ledger)
        manifest = _manifest(manifest_path)
        order_failures = flag_failures = binding_failures = present = 0
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
            for stage in STAGES:
                path, digest = row.get(f"{stage}_path", ""), row.get(f"{stage}_sha256", "")
                if not path or not digest:
                    continue
                present += 1
                relative = path[len(prefix):] if path.startswith(prefix) else path
                if manifest.get(relative) != digest:
                    binding_failures += 1
                    if len(examples) < 5:
                        examples.append(f"{row['episode_id']}: {stage} hash not in frozen manifest")
        has_times = any(k.endswith("_ns") for k in (rows[0] if rows else {}))
        results.append(CheckResult(f"chronology:{name}", "prediction_before_truth", ledger.name, len(rows),
                                   order_failures + flag_failures,
                                   detail=("strict candidate<prediction<decision<truth nanosecond order and recorded chronology flag"
                                           if has_times else "recorded chronology flag (this ledger stores no timestamps)"),
                                   examples=examples[:5]))
        results.append(CheckResult(f"hash_binding:{name}", "hash_verification", ledger.name, present, binding_failures,
                                   detail=f"every artifact hash in the ledger equals the frozen manifest entry ({manifest_path.name})"))
    return results


def check_table_chronology() -> list[CheckResult]:
    """Per-episode result tables: recorded timestamps and hashes agree with the ledgers."""
    results = []
    for name, table, ledger in (
        ("flash primary", FLASH / "FLASH_FINAL_RESULTS.csv", FLASH / "e5_artifact_audit.csv"),
        ("flash structural mismatch", FLASH / "FLASH_STRUCTURAL_RESULTS.csv", FLASH / "e7_artifacts/STRUCTURAL_ARTIFACT_AUDIT.csv"),
        ("flash mixed-regime audit", FLASH / "FLASH_RELIABILITY_RESULTS.csv", FLASH / "e7_artifacts/FLASH_RELIABILITY_ARTIFACT_AUDIT.csv"),
        ("flash boundary", FLASH / "FLASH_BOUNDARY_RESULTS.csv", FLASH / "e6_artifacts/FLASH_RESS_STRESS_ARTIFACT_AUDIT.csv"),
        ("cstr boundary", FLASH / "CSTR_BOUNDARY_RESULTS.csv", FLASH / "e6_artifacts/CSTR_RESS_STRESS_ARTIFACT_AUDIT.csv"),
        ("cstr mixed-regime audit", FLASH / "CSTR_RELIABILITY_RESULTS.csv", FLASH / "e7_artifacts/CSTR_RELIABILITY_ARTIFACT_AUDIT.csv"),
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

    mixed = read_csv(CSTR / "mixed_degradation/MULTIFAULT_FINAL_EPISODES.csv")
    bad = [row["episode_id"] for row in mixed
           if not (int(row["prediction_saved_ns"]) < int(row["decision_saved_ns"]) < int(row["truth_revealed_ns"]))
           or not truthy(row["chronology_verified"])]
    results.append(CheckResult("chronology:cstr mixed degradation", "prediction_before_truth", "MULTIFAULT_FINAL_EPISODES.csv",
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
    results.append(CheckResult("chronology:cstr primary (frozen reconstruction audit)", "prediction_before_truth",
                               "artifact_chronology_audit.csv", len(audit), len(bad),
                               detail="file-time order recorded by the read-only reconstruction audit on the original artifacts", examples=bad[:5]))
    return results
