"""Read-only reproducibility audit of the frozen TwinGuard results.

Usage (from the repository root):

    python reproducibility/run_reproducibility_audit.py            # git checkout only
    python reproducibility/run_reproducibility_audit.py --archive  # also the extracted per-episode archive

Nothing under ``artifacts/`` or ``results/`` is written. The report is written to
``reproducibility/reports/<UTC timestamp>/`` (git-ignored). The exit status is 1 if any check fails.

Checks run on the git checkout:
  * source provenance: each release module reverses to its frozen source bytes
  * frozen SHA-256 manifests, preregistration sidecars and design-file hashes
  * seed separation between all training / selection / calibration / test / stress populations
  * population completeness: declared seed blocks fully present, no missing or duplicate episodes
  * prediction-before-truth chronology from recorded nanosecond timestamps and ledger hash chains
  * certification-decision consistency and reconstruction of every summary table from per-episode rows
  * fault-blindness of method inputs, frozen models and feature sets

Additional checks with --archive:
  * integrity of every archived per-episode file
  * truth files bind the hashes of the earlier prediction and decision files
  * pre-truth artifacts (candidate/prediction/decision) contain no evaluator-only keys
  * the original final-audit reconstruction functions, compared with the frozen audit record
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
    archive_checks,
    chronology_checks,
    decision_checks,
    leakage_checks,
    provenance_checks,
    seed_checks,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--archive", action="store_true", help="require and audit the extracted frozen per-episode archive")
    parser.add_argument("--output", type=Path, default=None, help="report directory (default: reproducibility/reports/<timestamp>)")
    args = parser.parse_args()

    if args.archive and not archive_checks.archive_present():
        print("--archive given, but the frozen archive is not extracted at the repository root (see data/README.md)")
        return 2

    steps = [
        ("source provenance", provenance_checks.check_source_provenance),
        ("frozen hash manifests", lambda: provenance_checks.check_hash_manifests(require_archive=args.archive)),
        ("seed separation", seed_checks.check_seed_separation),
        ("population completeness", seed_checks.check_population_completeness),
        ("method rows", seed_checks.check_method_rows),
        ("ledger chronology", chronology_checks.check_ledger_chronology),
        ("table chronology", chronology_checks.check_table_chronology),
        ("decision consistency", decision_checks.check_row_consistency),
        ("summary reconstruction", decision_checks.check_summary_reconstruction),
        ("fault-blind method inputs", leakage_checks.check_method_inputs),
        ("fault-blind frozen models", leakage_checks.check_frozen_models),
        ("evaluator-only labelling", leakage_checks.check_evaluator_column_labelling),
    ]
    if args.archive:
        steps += [
            ("archive integrity", archive_checks.check_archive_integrity),
            ("truth hash binding", archive_checks.check_truth_binding),
            ("pre-truth artifact keys", archive_checks.check_pretruth_artifacts),
            ("original final audit", archive_checks.check_original_final_audit),
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
    lines = [f"# Reproducibility audit ({'git checkout + frozen archive' if args.archive else 'git checkout'})", "",
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
    (output / "AUDIT_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:4 + len({r.category for r in results}) + 2]))
    print(f"\nreport: {output}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
