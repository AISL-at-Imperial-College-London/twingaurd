"""Verify or regenerate the paper figures in results/figures/.

    python scripts/generate_paper_figures.py
        Verify that every figure in results/figures/ is byte-identical to the figure produced by
        the frozen experiment run (SHA-256 recorded in reproducibility/provenance/file_map.csv).

    python scripts/generate_paper_figures.py --regenerate --workspace runs/figures
        Re-draw the figures with the original plotting functions in a fresh workspace:
          * flash simulator figures (parity, step responses): rerun of the simulator validation
          * flash primary figures (representative rollout, nominal/conformal margin geometry):
            rebuilt from the archived per-episode artifacts
          * CSTR vs flash boundary figure: rebuilt from the archived boundary artifacts
        Regeneration of the archived-artifact figures needs the frozen archive. Byte identity is not
        expected across matplotlib versions, so the regenerated files are listed next to their frozen
        counterparts for visual comparison.

Margin-ladder and structural-mismatch figures are drawn by their stage modules
(experiments/evaluate_flash_margin_ladder.py --stage finalize, and the final audit).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from _runner import FROZEN_FLASH_ROOT, prepare_workspace  # noqa: E402

FIGURE_SOURCES = {}
with gzip.open(ROOT / "reproducibility/provenance/file_map.csv.gz", "rt", newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row["release_path"].startswith("results/figures/"):
            FIGURE_SOURCES[row["release_path"]] = row

REGENERATE = r"""
import csv
from twinguard.utils.paths import FLASH_STUDY_ROOT as root

def ledger(path):
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))

if "{stage}" == "primary":
    from twinguard.flash import evaluate_primary as e5
    rows = e5.evaluate_artifacts(ledger(root / "e5_artifact_audit.csv"))
    e5.create_figures(rows, e5.aggregate(rows, "pooled", "all_final_ordinary") + e5.subgroup_rows(rows))
else:
    from twinguard.evaluation import boundary_challenge as e6
    flash = e6.evaluate_flash_boundary(ledger(root / "e6_artifacts/FLASH_RESS_STRESS_ARTIFACT_AUDIT.csv"))
    cstr = e6.evaluate_cstr_boundary(ledger(root / "e6_artifacts/CSTR_RESS_STRESS_ARTIFACT_AUDIT.csv"))
    e6.create_figures(flash, cstr, e6.risk_rows(flash, cstr))
"""


def verify() -> int:
    failures = 0
    for release_path, row in sorted(FIGURE_SOURCES.items()):
        digest = hashlib.sha256((ROOT / release_path).read_bytes()).hexdigest()
        status = "OK " if digest == row["sha256"] else "MISMATCH"
        failures += status != "OK "
        print(f"{status} {release_path}  <-  frozen {row['original_path']}")
    return 1 if failures else 0


def regenerate(workspace: Path) -> int:
    env = dict(os.environ, MPLBACKEND="Agg", TWINGUARD_ROOT=str(ROOT), TWINGUARD_FLASH_STUDY_ROOT=str(workspace.resolve()),
               PYTHONPATH=os.pathsep.join([str(ROOT / "src"), os.environ.get("PYTHONPATH", "")]))
    archive = (ROOT / "artifacts/flash_study/e5_artifacts/predictions").is_dir()
    prepare_workspace(FROZEN_FLASH_ROOT, workspace, ["figures", "e1_validation", "FLASH_E1_VALIDATION.md", "e5_figures", "e6_figures"],
                      include_archive=False)
    status = subprocess.call([sys.executable, "-m", "twinguard.flash.validate_simulator"], cwd=ROOT, env=env)
    if archive:
        for stage in ("primary", "boundary"):
            status |= subprocess.call([sys.executable, "-c", REGENERATE.format(stage=stage)], cwd=ROOT, env=env)
    else:
        print("frozen archive not extracted: skipping figures that are drawn from per-episode artifacts")
    for release_path, row in sorted(FIGURE_SOURCES.items()):
        relative = row["original_path"].split("/", 1)[1]
        regenerated = workspace / relative
        if row["original_path"].startswith("E_FLASH_RESS/") and regenerated.is_file():
            same = hashlib.sha256(regenerated.read_bytes()).hexdigest() == row["sha256"]
            print(f"{'identical' if same else 'redrawn  '} {regenerated}  vs  {release_path}")
    return status


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--workspace", type=Path, default=ROOT / "runs" / "figures")
    args = parser.parse_args()
    return regenerate(args.workspace) if args.regenerate else verify()


if __name__ == "__main__":
    sys.exit(main())
