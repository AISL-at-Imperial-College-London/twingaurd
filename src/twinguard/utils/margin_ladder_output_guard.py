"""Prevent margin-ladder runs from writing into frozen artifact directories."""

from __future__ import annotations

from pathlib import Path


from twinguard.utils.paths import MARGIN_LADDER_ROOT, REPOSITORY_ROOT

EXPERIMENT_ROOT = MARGIN_LADDER_ROOT


def assert_margin_ladder_output(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    root = EXPERIMENT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError(f"margin ladder may only write below {root}: {resolved}")
    return resolved

