"""Repository-relative locations of the frozen artifacts and of rerun workspaces.

The frozen experiment directories under ``artifacts/`` are read-only evidence. Stage modules
read their inputs from, and write their outputs to, an experiment root. A rerun must therefore
point that root at a workspace copy, which ``scripts/reproduce_main_results.py`` prepares:

``TWINGUARD_FLASH_STUDY_ROOT``
    flash-study root (frozen default: ``artifacts/flash_study``)
``TWINGUARD_MARGIN_LADDER_ROOT``
    margin-ladder root (frozen default: ``artifacts/flash_margin_ladder``)
``TWINGUARD_ROOT``
    repository root, when it cannot be found from the package location
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_repository_root() -> Path:
    configured = os.environ.get("TWINGUARD_ROOT")
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "artifacts").is_dir():
            return parent
    return Path.cwd().resolve()


def _configured(variable: str, default: Path) -> Path:
    value = os.environ.get(variable)
    return Path(value).resolve() if value else default.resolve()


REPOSITORY_ROOT = _find_repository_root()
ARTIFACTS_ROOT = REPOSITORY_ROOT / "artifacts"
CSTR_ARTIFACTS_ROOT = ARTIFACTS_ROOT / "cstr"
FLASH_STUDY_ROOT = _configured("TWINGUARD_FLASH_STUDY_ROOT", ARTIFACTS_ROOT / "flash_study")
MARGIN_LADDER_ROOT = _configured("TWINGUARD_MARGIN_LADDER_ROOT", ARTIFACTS_ROOT / "flash_margin_ladder")
