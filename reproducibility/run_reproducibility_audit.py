"""Read-only reproducibility audit of the released TwinGuard results.

Usage (from the repository root):

    python reproducibility/run_reproducibility_audit.py

Nothing under ``artifacts/`` or ``results/`` is written. The report is written to
``reproducibility/reports/<UTC timestamp>/`` (git-ignored). The exit status is 1 if any check fails.

Checks:
  * SHA-256 integrity of every released result file, of the stage manifests, of the
    preregistration / design sidecars and of the hash references between design files
  * seed separation between all training / selection / calibration / test / stress populations
  * population completeness: declared seed blocks fully present, no missing or duplicate episodes
  * prediction-before-truth chronology from the recorded nanosecond timestamps
  * certification-decision consistency and reconstruction of every summary table from per-episode rows
  * fault-blindness of method inputs, fitted models and feature sets
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
os.environ.setdefault("TWINGUARD_ROOT", str(ROOT))
os.environ.setdefault("MPLBACKEND", "Agg")

from reproducibility.checks import (  # noqa: E402
    chronology_checks,
    decision_checks,
    integrity_checks,
    leakage_checks,
    seed_checks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=None, help="report directory (default: reproducibility/reports/<timestamp>)")
    args = parser.parse_args()

    steps = [
        ("released file integrity", integrity_checks.check_artifact_manifest),
        ("stage hash manifests", integrity_checks.check_stage_manifests),
        ("design and preregistration hashes", integrity_checks.check_design_references),
        ("seed separation", seed_checks.check_seed_separation),
        ("population completeness", seed_checks.check_population_completeness),
        ("method rows", seed_checks.check_method_rows),
        ("ledger chronology", chronology_checks.check_ledger_chronology),
        ("table chronology", chronology_checks.check_table_chronology),
        ("decision consistency", decision_checks.check_row_consistency),
        ("summary reconstruction", decision_checks.check_summary_reconstruction),
        ("fault-blind method inputs", leakage_checks.check_method_inputs),
        ("fault-blind fitted models", leakage_checks.check_frozen_models),
        ("evaluator-only labelling", leakage_checks.check_evaluator_column_labelling),
    ]

    results = []
    for label, step in steps:
        print(f"[audit] {label} ...", flush=True)
        results.extend(step())

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or ROOT / "reproducibility" / "reports" / stamp
    output.mkdir(parents=True, exist_ok=True)
    with (output / "audit_checks.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["check", "category", "subject", "examined", "failures", "status", "detail", "examples"])
        for r in results:
            writer.writerow([r.check, r.category, r.subject, r.examined, r.failures, r.status, r.detail, " | ".join(r.examples)])
    counts = Counter(r.status for r in results)
    by_category = Counter((r.category, r.status) for r in results)
    lines = ["# Reproducibility audit", "",
             f"Generated {stamp}. Checks: {len(results)}; " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())), "",
             "| Category | PASS | FAIL | INFO |", "|---|---:|---:|---:|"]
    for category in sorted({r.category for r in results}):
        lines.append(f"| {category} | {by_category[(category, 'PASS')]} | {by_category[(category, 'FAIL')]} | {by_category[(category, 'INFO')]} |")
    failed = [r for r in results if r.status == "FAIL"]
    if failed:
        lines += ["", "## Failed checks", ""] + [f"- `{r.check}`: {r.detail} {r.examples[:3]}" for r in failed]
    info = [r for r in results if r.status == "INFO"]
    if info:
        lines += ["", "## Informational", ""] + [f"- `{r.check}`: {r.detail}" for r in info]
    (output / "audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + len({r.category for r in results}) + 2]))
    if failed:
        print("\nFailed checks:\n" + "\n".join(f"  {r.check}: {r.examples[:3]}" for r in failed))
    print(f"\nreport: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
