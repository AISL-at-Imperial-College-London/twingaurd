"""Shared launcher for the experiment entry points.

Stage modules read and write relative to an experiment root. To keep the frozen evidence under
``artifacts/`` untouched, reruns always happen in a workspace:

* flash and margin-ladder stages: the workspace is a copy of the frozen experiment root with the
  outputs of the stage being rerun removed. The stage then sees exactly the frozen upstream
  inputs and regenerates its own outputs.
* CSTR stages: ``--output-dir`` defaults to ``runs/<stage>``; upstream inputs are read from the
  frozen ``artifacts/cstr/...`` directories.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FROZEN_FLASH_ROOT = REPOSITORY_ROOT / "artifacts" / "flash_study"
FROZEN_MARGIN_LADDER_ROOT = REPOSITORY_ROOT / "artifacts" / "flash_margin_ladder"
ARCHIVE_ROOT = REPOSITORY_ROOT / "archive"


def _environment(extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPOSITORY_ROOT / "src"), env.get("PYTHONPATH", "")]))
    env.update(extra)
    return env


def _copy_tree(source: Path, destination: Path) -> None:
    for path in source.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)


def prepare_workspace(frozen_root: Path, workspace: Path, stage_outputs: list[str], include_archive: bool) -> Path:
    """Copy the frozen root into ``workspace`` and delete the outputs the stage will regenerate."""
    workspace = workspace.resolve()
    artifacts = (REPOSITORY_ROOT / "artifacts").resolve()
    if workspace == artifacts or artifacts in workspace.parents:
        raise SystemExit("the workspace must not be inside artifacts/ (frozen evidence is read-only)")
    if workspace.exists() and any(workspace.iterdir()):
        raise SystemExit(f"workspace {workspace} is not empty; choose a new directory")
    workspace.mkdir(parents=True, exist_ok=True)
    _copy_tree(frozen_root, workspace)
    archived = ARCHIVE_ROOT / frozen_root.relative_to(REPOSITORY_ROOT)
    if include_archive and archived.is_dir():
        _copy_tree(archived, workspace)
    for relative in stage_outputs:
        target = workspace / relative
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()
    return workspace


def run_module(module: str, arguments: list[str], env: dict[str, str] | None = None) -> int:
    command = [sys.executable, "-m", module, *arguments]
    print("+", " ".join(command), flush=True)
    return subprocess.call(command, cwd=REPOSITORY_ROOT, env=_environment(env or {}))


def flash_stage(module: str, description: str, stage_outputs: list[str], needs_archive: bool = False,
                extra_arguments: list[str] | None = None, root_variable: str = "TWINGUARD_FLASH_STUDY_ROOT",
                frozen_root: Path = FROZEN_FLASH_ROOT) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="new empty directory for this rerun (e.g. runs/flash_primary)")
    args, passthrough = parser.parse_known_args()
    if needs_archive and not (ARCHIVE_ROOT / frozen_root.relative_to(REPOSITORY_ROOT)).is_dir():
        raise SystemExit("this stage verifies archived per-episode artifacts; extract the frozen archive into archive/ first")
    workspace = prepare_workspace(frozen_root, args.workspace, stage_outputs, include_archive=needs_archive)
    env = {root_variable: str(workspace), "TWINGUARD_ROOT": str(REPOSITORY_ROOT)}
    raise SystemExit(run_module(module, [*(extra_arguments or []), *passthrough], env))


def cstr_stage(module: str, description: str, default_output: str) -> None:
    parser = argparse.ArgumentParser(description=description, add_help=False)
    parser.add_argument("--output-dir")
    args, passthrough = parser.parse_known_args()
    output = args.output_dir or str(REPOSITORY_ROOT / "runs" / default_output)
    if (REPOSITORY_ROOT / "artifacts").resolve() in Path(output).resolve().parents:
        raise SystemExit("--output-dir must not be inside artifacts/ (frozen evidence is read-only)")
    raise SystemExit(run_module(module, ["--output-dir", output, *passthrough], {"TWINGUARD_ROOT": str(REPOSITORY_ROOT)}))
