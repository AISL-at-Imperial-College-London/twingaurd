"""Checks that need the frozen per-episode archive extracted at the repository root."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from twinguard.scenarios.flash_population_design import FORBIDDEN_OPERATIONAL_TOKENS

from .common import FLASH, PROVENANCE, REPOSITORY_ROOT, CheckResult, read_csv, sha256_file


def archive_present() -> bool:
    return (FLASH / "e5_artifacts" / "predictions").is_dir()


def check_archive_integrity() -> list[CheckResult]:
    with gzip.open(PROVENANCE / "file_map.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["location"] == "archive"]
    missing, mismatched = [], []
    for row in rows:
        path = REPOSITORY_ROOT / row["release_path"]
        if not path.is_file():
            missing.append(row["release_path"])
        elif sha256_file(path) != row["sha256"]:
            mismatched.append(row["release_path"])
    return [CheckResult("archive_integrity", "hash_verification", "every archived file", len(rows), len(missing) + len(mismatched),
                        detail=f"missing={len(missing)}, hash mismatches={len(mismatched)}", examples=(missing + mismatched)[:10])]


def check_truth_binding() -> list[CheckResult]:
    """Truth files record the SHA-256 of the prediction and decision written before them."""
    results = []
    for name, root, decision_suffix in (("flash primary", FLASH / "e5_artifacts", ".json"),
                                        ("flash boundary", FLASH / "e6_artifacts/flash", ".json"),
                                        ("cstr boundary", FLASH / "e6_artifacts/cstr", ".json"),
                                        ("flash structural mismatch", FLASH / "e7_artifacts/structural/structural", ".json"),
                                        ("flash mixed-regime audit", FLASH / "e7_artifacts/flash_reliability/flash_reliability", ".json"),
                                        ("cstr mixed-regime audit", FLASH / "e7_artifacts/cstr_reliability", ".json")):
        truths = sorted((root / "truths").glob("*.npz"))
        failures, examples = 0, []
        for truth in truths:
            prediction = root / "predictions" / f"{truth.stem}.npz"
            decision = root / "decisions" / f"{truth.stem}{decision_suffix}"
            with np.load(truth, allow_pickle=False) as saved:
                bound_prediction = str(saved["prediction_sha256"]) if "prediction_sha256" in saved.files else None
                bound_decision = str(saved["decision_sha256"]) if "decision_sha256" in saved.files else None
            ok = (bound_prediction == sha256_file(prediction)) and (bound_decision == sha256_file(decision))
            if not ok:
                failures += 1
                examples.append(truth.stem)
        results.append(CheckResult(f"truth_binding:{name}", "prediction_before_truth", str(root.relative_to(REPOSITORY_ROOT)),
                                   len(truths), failures, detail="truth file binds the SHA-256 of the earlier prediction and decision files",
                                   examples=examples[:5]))
    return results


def check_pretruth_artifacts() -> list[CheckResult]:
    """Candidate, prediction and decision files (everything written before truth) hold no evaluator-only keys."""
    results = []
    roots = {"flash primary": FLASH / "e5_artifacts", "flash boundary": FLASH / "e6_artifacts/flash", "cstr boundary": FLASH / "e6_artifacts/cstr",
             "flash structural mismatch": FLASH / "e7_artifacts/structural/structural",
             "flash mixed-regime audit": FLASH / "e7_artifacts/flash_reliability/flash_reliability",
             "cstr mixed-regime audit": FLASH / "e7_artifacts/cstr_reliability"}

    def keys_of(path: Path) -> list[str]:
        if path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as saved:
                return list(saved.files)
        payload = json.loads(path.read_text(encoding="utf-8"))
        out = []
        stack = [payload]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                out.extend(item.keys())
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
        return out

    for name, root in roots.items():
        examined, bad = 0, []
        for stage in ("candidates", "predictions", "decisions"):
            for path in sorted((root / stage).glob("*")):
                examined += 1
                hits = [k for k in keys_of(path) if any(t in k.lower() for t in FORBIDDEN_OPERATIONAL_TOKENS)]
                if hits:
                    bad.append(f"{stage}/{path.name}: {hits[:3]}")
        results.append(CheckResult(f"pretruth_keys:{name}", "pretruth_leakage", str(root.relative_to(REPOSITORY_ROOT)), examined, len(bad),
                                   detail="pre-truth artifacts contain no fault/severity/truth keys", examples=bad[:5]))
    return results


def check_original_final_audit() -> list[CheckResult]:
    """Re-run the original final-audit reconstruction functions (read-only) and compare with the frozen audit."""
    from twinguard.cstr import safety_certification as c1
    from twinguard.evaluation import final_audit as audit

    rows: list[dict[str, object]] = []
    audit.audit_c1(rows)
    audit.audit_scalar_transfer(rows, "CSTR_C1_6B", REPOSITORY_ROOT / "artifacts/cstr/mixed_degradation", 313.0, c1.EPSILON_SCALE_K)
    audit.audit_flash_e5(rows)
    audit.audit_flash_e6(rows)
    audit.audit_e6_cstr(rows)
    audit.audit_e7_population(rows, "structural")
    audit.audit_e7_population(rows, "flash_reliability")
    audit.audit_e7_cstr(rows)
    for population, results_name, truths in (
        ("FLASH_E5", "FLASH_FINAL_RESULTS.csv", "e5_artifacts/truths"),
        ("FLASH_E6_BOUNDARY", "FLASH_BOUNDARY_RESULTS.csv", "e6_artifacts/flash/truths"),
        ("FLASH_E7_STRUCTURAL", "FLASH_STRUCTURAL_RESULTS.csv", "e7_artifacts/structural/structural/truths"),
        ("FLASH_E7_RELIABILITY", "FLASH_RELIABILITY_RESULTS.csv", "e7_artifacts/flash_reliability/flash_reliability/truths"),
    ):
        audit.audit_flash_safety(rows, population, FLASH / results_name, FLASH / truths)
    audit.audit_cstr_safety(rows, "CSTR_E6_BOUNDARY", FLASH / "CSTR_BOUNDARY_RESULTS.csv", FLASH / "e6_artifacts/cstr/truths")
    audit.audit_cstr_safety(rows, "CSTR_E7_RELIABILITY", FLASH / "CSTR_RELIABILITY_RESULTS.csv", FLASH / "e7_artifacts/cstr_reliability/truths")
    audit.audit_nan_to_zero(rows)
    frozen = {(r["population"], r["audit_check"]): r for r in read_csv(FLASH / "RESS_FINAL_AUDIT_DETAILS.csv")}
    results = []
    for row in rows:
        key = (str(row["population"]), str(row["audit_check"]))
        reference = frozen.get(key)
        same = (reference is not None and str(reference["records_examined"]) == str(row["records_examined"])
                and str(reference["failure_count"]) == str(row["failure_count"]))
        status = str(row["status"])
        if same and status == "ISSUE":
            outcome = dict(failures=int(row["failure_count"]), status="INFO")  # documented, frozen NaN->0 issues
        elif same:
            outcome = dict(failures=0, status="")
        else:
            outcome = dict(failures=max(1, int(row["failure_count"])), status="")
        results.append(CheckResult(f"original_final_audit:{key[0]}:{key[1]}", "decision_reconstruction", key[0], int(row["records_examined"]),
                                   detail=f"original audit status {status}; frozen record {'reproduced' if same else 'NOT reproduced'}: {row.get('detail', '')}",
                                   **outcome))
    return results
