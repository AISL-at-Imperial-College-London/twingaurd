"""Shared launcher for the experiment entry points.

Stage modules read and write relative to an experiment root. To keep the recorded results under
``artifacts/`` untouched, reruns always happen in a workspace:

* flash and margin-ladder stages: the workspace is a copy of the recorded experiment root with the
  outputs of the stage being rerun removed. The stage then sees exactly the recorded upstream
  inputs and regenerates its own outputs.
* CSTR stages: ``--output-dir`` defaults to ``runs/<stage>``; upstream inputs are read from the
  recorded ``artifacts/cstr/...`` directories.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FLASH_ROOT = REPOSITORY_ROOT / "artifacts" / "flash_study"
MARGIN_LADDER_ROOT = REPOSITORY_ROOT / "artifacts" / "flash_margin_ladder"


def _environment(extra: dict[str, str]) -> dict[str, str]:
    env = dict(os.environ)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(REPOSITORY_ROOT / "src"), env.get("PYTHONPATH", "")]))
    env.update(extra)
    return env


def prepare_workspace(recorded_root: Path, workspace: Path, stage_outputs: list[str]) -> Path:
    """Copy the recorded experiment root into ``workspace`` and delete the outputs the stage regenerates."""
    workspace = workspace.resolve()
    artifacts = (REPOSITORY_ROOT / "artifacts").resolve()
    if workspace == artifacts or artifacts in workspace.parents:
        raise SystemExit("the workspace must not be inside artifacts/ (recorded results are read-only)")
    if workspace.exists() and any(workspace.iterdir()):
        raise SystemExit(f"workspace {workspace} is not empty; choose a new directory")
    workspace.mkdir(parents=True, exist_ok=True)
    for path in recorded_root.rglob("*"):
        if path.is_file():
            target = workspace / path.relative_to(recorded_root)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
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


def flash_stage(module: str, description: str, stage_outputs: list[str], extra_arguments: list[str] | None = None,
                root_variable: str = "TWINGUARD_FLASH_STUDY_ROOT", recorded_root: Path = FLASH_ROOT) -> None:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--workspace", type=Path, required=True,
                        help="new empty directory for this rerun (e.g. runs/flash_primary)")
    args, passthrough = parser.parse_known_args()
    workspace = prepare_workspace(recorded_root, args.workspace, stage_outputs)
    env = {root_variable: str(workspace), "TWINGUARD_ROOT": str(REPOSITORY_ROOT)}
    raise SystemExit(run_module(module, [*(extra_arguments or []), *passthrough], env))


def cstr_stage(module: str, description: str, default_output: str) -> None:
    parser = argparse.ArgumentParser(description=description, add_help=False)
    parser.add_argument("--output-dir")
    args, passthrough = parser.parse_known_args()
    output = args.output_dir or str(REPOSITORY_ROOT / "runs" / default_output)
    if (REPOSITORY_ROOT / "artifacts").resolve() in Path(output).resolve().parents:
        raise SystemExit("--output-dir must not be inside artifacts/ (recorded results are read-only)")
    raise SystemExit(run_module(module, ["--output-dir", output, *passthrough], {"TWINGUARD_ROOT": str(REPOSITORY_ROOT)}))
