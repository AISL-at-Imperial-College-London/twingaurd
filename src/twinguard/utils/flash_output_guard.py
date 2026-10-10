"""Keep flash experiment outputs out of frozen artifact directories.

Each entry point must validate its output path before writing files with
``assert_isolated_output_path``.
Frozen CSTR artifacts are comparison inputs, not output targets.
The guard centralizes that check for the flash study.
"""

from __future__ import annotations

from pathlib import Path


from twinguard.utils.paths import FLASH_STUDY_ROOT, REPOSITORY_ROOT

EXPERIMENT_ROOT = FLASH_STUDY_ROOT

# Release layout: frozen CSTR artifacts are comparison inputs only, never flash outputs.
FROZEN_RESULT_RELATIVE_PATHS = (
    "artifacts/cstr",
)

FROZEN_RESULT_ROOTS = tuple(
    (REPOSITORY_ROOT / relative).resolve()
    for relative in FROZEN_RESULT_RELATIVE_PATHS
)

FROZEN_RESULT_ROOT_FILE_NAMES: tuple[str, ...] = ()


def _contains(root: Path, candidate: Path) -> bool:
    return candidate == root or root in candidate.parents


def assert_isolated_output_path(path: str | Path) -> Path:
    """Return a resolved flash output path or reject an unsafe target.

    Relative paths are interpreted from the repository root. Resolving before
    checking also rejects ``..`` traversal and existing links that escape the
    isolated experiment namespace.
    """

    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = REPOSITORY_ROOT / candidate
    resolved = candidate.resolve(strict=False)
    if any(_contains(root, resolved) for root in FROZEN_RESULT_ROOTS):
        raise ValueError(f"flash output overlaps frozen results: {resolved}")
    if not _contains(EXPERIMENT_ROOT, resolved):
        raise ValueError(
            f"flash output must remain under {EXPERIMENT_ROOT}: {resolved}"
        )
    return resolved
