"""Filesystem isolation for the post-E6 E6B extension."""

from __future__ import annotations

from pathlib import Path


from twinguard.utils.paths import MARGIN_LADDER_ROOT, REPOSITORY_ROOT

EXPERIMENT_ROOT = MARGIN_LADDER_ROOT


def assert_e6b_output(path: str | Path) -> Path:
    resolved = Path(path).resolve()
    root = EXPERIMENT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise PermissionError(f"E6B may only write below {root}: {resolved}")
    return resolved

