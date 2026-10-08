"""Filesystem boundary for all E_FLASH_RESS output.

Every future flash entry point must validate its output directory with
``assert_isolated_output_path`` before creating a directory or file. Existing
CSTR and two-tank results are inputs for comparison only and are never output
targets for the flash study.
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

FROZEN_RESULT_ROOT_FILE_NAMES = (
    "milestone_b5_stderr.log",
    "milestone_b5_stdout.log",
    "milestone_b6_stderr.log",
    "milestone_b6_stdout.log",
    "milestone_b7a_stderr.log",
    "milestone_b7a_stdout.log",
    "milestone_b7b_stderr.log",
    "milestone_b7b_stdout.log",
    "milestone_b9_stderr.log",
    "milestone_b9_stdout.log",
    "milestone_c15c_shard0_stderr.log",
    "milestone_c15c_shard0_stdout.log",
    "milestone_c15c_shard1_stderr.log",
    "milestone_c15c_shard1_stdout.log",
    "milestone_c15c_shard2_stderr.log",
    "milestone_c15c_shard2_stdout.log",
    "milestone_c15c_shard3_stderr.log",
    "milestone_c15c_shard3_stdout.log",
    "milestone_c15c_stderr.log",
    "milestone_c15c_stdout.log",
    "milestone_c16b_stderr.log",
    "milestone_c16b_stdout.log",
    "second_case_final.zip",
)


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
