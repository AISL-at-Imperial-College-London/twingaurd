"""Verify or regenerate the paper figures in results/figures/.

    python scripts/generate_paper_figures.py
        Verify that every figure in results/figures/ matches the SHA-256 recorded in
        reproducibility/artifact_manifest.csv.

    python scripts/generate_paper_figures.py --regenerate --workspace runs/figures
        Re-draw the flash simulator figures (healthy-plant / twin parity and manipulated-action step
        responses) by rerunning the simulator validation in a fresh workspace. Byte identity is not
        expected across matplotlib versions, so the regenerated files are listed next to the released
        ones for visual comparison.

The other figures are drawn by the stage that produces their data:

    flash primary evaluation figures      experiments/evaluate_flash_primary.py
    CSTR vs flash boundary figure          experiments/evaluate_boundary_challenge.py
    structural-mismatch diagnostics        experiments/evaluate_structural_mismatch_and_mixed_regime_audits.py
    margin-ladder figures                  experiments/evaluate_flash_margin_ladder.py --stage finalize
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments._runner import FLASH_ROOT, prepare_workspace  # noqa: E402

SIMULATOR_FIGURES = {  # figure written by the simulator validation -> released paper figure
    "figures/healthy_plant_dt_parity.png": "results/figures/appendix/flash_healthy_plant_twin_parity.png",
    "figures/action_step_responses.png": "results/figures/appendix/flash_manipulated_action_step_responses.png",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify() -> int:
    with (ROOT / "reproducibility/artifact_manifest.csv").open(newline="", encoding="utf-8") as handle:
        recorded = {row["path"]: row["sha256"] for row in csv.DictReader(handle) if row["path"].startswith("results/figures/")}
    failures = 0
    for path in sorted(recorded):
        ok = (ROOT / path).is_file() and _sha256(ROOT / path) == recorded[path]
        failures += not ok
        print(f"{'OK      ' if ok else 'MISMATCH'} {path}")
    on_disk = {p.relative_to(ROOT).as_posix() for p in (ROOT / "results/figures").rglob("*") if p.is_file()}
    for path in sorted(on_disk - set(recorded)):
        failures += 1
        print(f"UNLISTED {path}")
    return 1 if failures else 0


def regenerate(workspace: Path) -> int:
    workspace = prepare_workspace(FLASH_ROOT, workspace, ["figures", "simulator_validation", "simulator_validation_report.md"])
    env = dict(os.environ, MPLBACKEND="Agg", TWINGUARD_ROOT=str(ROOT), TWINGUARD_FLASH_STUDY_ROOT=str(workspace),
               PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT / "src"), os.environ.get("PYTHONPATH", "")])))
    status = subprocess.call([sys.executable, "-m", "twinguard.flash.validate_simulator"], cwd=ROOT, env=env)
    for produced, released in SIMULATOR_FIGURES.items():
        regenerated = workspace / produced
        if not regenerated.is_file():
            print(f"missing    {regenerated}")
            status |= 1
            continue
        same = _sha256(regenerated) == _sha256(ROOT / released)
        print(f"{'identical' if same else 'redrawn  '}  {regenerated}  vs  {released}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--workspace", type=Path, default=ROOT / "runs" / "figures")
    args = parser.parse_args()
    return regenerate(args.workspace) if args.regenerate else verify()


if __name__ == "__main__":
    sys.exit(main())
