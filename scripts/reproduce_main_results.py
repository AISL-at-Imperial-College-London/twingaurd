"""Reproduce the principal results.

Verification of the released results (minutes; nothing is recomputed from scratch):

    python scripts/reproduce_main_results.py

  runs the reproducibility audit, checks that results/tables and results/figures match the released
  outputs, and prints the headline table.

Prospective reruns (each stage reruns in a fresh workspace under runs/, never in artifacts/):

    python scripts/reproduce_main_results.py --rerun flash-simulator cstr-primary-smoke
    python scripts/reproduce_main_results.py --rerun flash-primary --workers 6

  After each stage, its regenerated summary files are compared cell by cell with the released ones.
  Available stages and approximate cost are listed by --list. Full reruns of the evaluation
  populations take hours. The structural / mixed-regime stage ends with a final audit that reads the
  per-episode files of the flash and CSTR primary evaluations, so rerun those stages first.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FLASH = ROOT / "artifacts" / "flash_study"
LADDER = ROOT / "artifacts" / "flash_margin_ladder"
CSTR = ROOT / "artifacts" / "cstr"

# stage -> (experiment script, extra args, kind, [(released file, file relative to the rerun output)], cost)
STAGES = {
    "flash-simulator": ("validate_flash_simulator.py", [], "flash",
                        [(FLASH / "simulator_validation/HEALTHY_SCENARIO_SUMMARY.csv", "simulator_validation/HEALTHY_SCENARIO_SUMMARY.csv"),
                         (FLASH / "simulator_validation/ACTION_STEP_RESPONSES.csv", "simulator_validation/ACTION_STEP_RESPONSES.csv"),
                         (FLASH / "simulator_validation/simulator_validation_summary.json", "simulator_validation/simulator_validation_summary.json")], "seconds"),
    "flash-model-selection": ("fit_and_select_flash_discrepancy_model.py", [], "flash",
                              [(FLASH / "flash_model_selection.csv", "flash_model_selection.csv")], "~1 h"),
    "flash-calibration": ("calibrate_flash_conformal.py", [], "flash",
                          [(FLASH / "flash_calibration_scores.csv", "flash_calibration_scores.csv")], "~1-2 h"),
    "flash-bootstrap-and-calibration": ("build_flash_bootstrap_and_calibrate.py", [], "flash",
                                        [(FLASH / "flash_bootstrap_manifest.csv", "flash_bootstrap_manifest.csv"),
                                         (FLASH / "flash_calibration_scores.csv", "flash_calibration_scores.csv")], "several hours"),
    "flash-primary": ("evaluate_flash_primary.py", [], "flash",
                      [(FLASH / "flash_primary_ablation.csv", "flash_primary_ablation.csv")], "~1 h (1200 episodes x 2000 paths)"),
    "boundary": ("evaluate_boundary_challenge.py", [], "flash",
                 [(FLASH / "boundary_risk_availability.csv", "boundary_risk_availability.csv")], "hours"),
    "structural-and-mixed-regime": ("evaluate_structural_mismatch_and_mixed_regime_audits.py", [], "flash",
                                    [(FLASH / "flash_structural_summary.csv", "flash_structural_summary.csv"),
                                     (FLASH / "flash_reliability_summary.csv", "flash_reliability_summary.csv"),
                                     (FLASH / "cstr_reliability_summary.csv", "cstr_reliability_summary.csv")], "many hours; needs primary per-episode files"),
    "margin-ladder": ("evaluate_flash_margin_ladder.py", ["--stage", "all"], "flash",
                      [(LADDER / "margin_ladder_risk_availability.csv", "margin_ladder_risk_availability.csv")], "many hours"),
    "cstr-bootstrap": ("build_cstr_bootstrap_ensemble.py", [], "cstr",
                       [(CSTR / "bootstrap_ensemble/bootstrap_model_manifest.csv", "bootstrap_model_manifest.csv")], "~1 h"),
    "cstr-conformal": ("calibrate_cstr_conformal.py", [], "cstr",
                       [(CSTR / "conformal_calibration/conformal_quantiles.csv", "conformal_quantiles.csv")], "~1 h"),
    "cstr-primary": ("evaluate_cstr_primary.py", [], "cstr",
                     [(CSTR / "primary_evaluation/one_sided_quantiles.csv", "one_sided_quantiles.csv"),
                      (CSTR / "primary_evaluation/safety_summary.csv", "safety_summary.csv")], "~1-2 h"),
    "cstr-primary-smoke": ("evaluate_cstr_primary.py", ["--limit-episodes", "3", "--trajectories", "200"], "cstr", [], "~1 min"),
    "cstr-mixed-degradation": ("evaluate_cstr_mixed_degradation.py", [], "cstr",
                               [(CSTR / "mixed_degradation/mixed_degradation_summary.csv", "mixed_degradation_summary.csv")], "hours"),
}


def _cells(path: Path) -> list[list[str]]:
    if path.suffix == ".json":
        return [[json.dumps(json.loads(path.read_text(encoding="utf-8")), sort_keys=True)]]
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.reader(handle))


def _same(a: str, b: str) -> bool:
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except ValueError:
        return False
    return (math.isnan(x) and math.isnan(y)) or math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)


def compare(released: Path, rerun: Path) -> str:
    if not rerun.is_file():
        return f"MISSING   {rerun}"
    a, b = _cells(released), _cells(rerun)
    if len(a) != len(b):
        return f"DIFFERENT {released.name}: {len(a)} vs {len(b)} rows"
    differing = sum(any(not _same(x, y) for x, y in zip(ra, rb)) or len(ra) != len(rb) for ra, rb in zip(a, b))
    return f"{'IDENTICAL' if differing == 0 else 'DIFFERENT'} {released.name}: {differing} of {len(a)} rows differ (numeric tolerance 1e-9)"


def rerun(stage: str, workers: int | None) -> int:
    script, extra, kind, comparisons, cost = STAGES[stage]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = ROOT / "runs" / f"{stage}_{stamp}"
    arguments = [sys.executable, str(ROOT / "experiments" / script)]
    arguments += ["--workspace", str(target)] if kind == "flash" else ["--output-dir", str(target)]
    arguments += extra + (["--workers", str(workers)] if workers and stage != "flash-simulator" and stage != "cstr-primary-smoke" else [])
    print(f"\n=== {stage} ({cost}) -> {target}", flush=True)
    status = subprocess.call(arguments, cwd=ROOT)
    for released, relative in comparisons:
        print("   ", compare(released, target / relative))
    return status


def verify() -> int:
    status = subprocess.call([sys.executable, str(ROOT / "reproducibility/run_reproducibility_audit.py")], cwd=ROOT)
    status |= subprocess.call([sys.executable, str(ROOT / "scripts/generate_paper_tables.py"), "--check"], cwd=ROOT)
    status |= subprocess.call([sys.executable, str(ROOT / "scripts/generate_paper_figures.py")], cwd=ROOT)
    print("\nHeadline results (results/tables/paper_headline_results.csv):")
    with (ROOT / "results/tables/paper_headline_results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            print(f"  {row['experiment'][:55]:55s} {row['method_id']:22s} N={row['episodes']:>5}  certified={row['certified_safe']:>5}"
                  f"  false-safe={row['false_safe']:>5}  coverage={row['whole_trajectory_coverage'][:6] or '-'}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rerun", nargs="+", choices=sorted(STAGES), metavar="STAGE", help="stages to rerun prospectively")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--list", action="store_true", help="list rerunnable stages")
    args = parser.parse_args()
    if args.list:
        for name, (script, _, _, _, cost) in STAGES.items():
            print(f"{name:32s} experiments/{script:58s} {cost}")
        return 0
    if args.rerun:
        return max(rerun(stage, args.workers) for stage in args.rerun)
    return verify()


if __name__ == "__main__":
    sys.exit(main())
