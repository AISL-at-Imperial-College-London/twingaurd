"""Resolve frozen-artifact hashes recorded under development-repository paths.

The frozen hash manifests, preregistrations and design files record SHA-256 values against the
paths that the files had in the development repository (for example
``E_FLASH_RESS/flash_process.py`` or ``results/milestone_c1/one_sided_quantiles.csv``). This
module maps such an original path to its release location and returns the SHA-256 of the
*frozen* bytes:

* data files are stored byte-exact. Where a recorded hash proved the frozen file used LF line
  endings, the LF form is stored. The digest is that of the stored file.
* source modules were moved into ``src/twinguard`` with line-level edits (imports, paths,
  frozen-hash resolution) listed in ``reproducibility/provenance/source_ledger.json``. Reversing
  those edits reproduces the frozen bytes, and their digest is returned. Any other change to a
  module therefore changes the digest, exactly as in the original frozen checks.
* per-episode ledgers whose path columns were sanitized return their recorded original digest,
  but only if the stored file still has its recorded sanitized digest.

Per-episode artifacts live in the frozen archive. ``FrozenArtifactUnavailable`` is raised when
the archive has not been extracted, and ``FrozenArtifactNotReleased`` for development-only files
that are not part of the release.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from twinguard.utils.paths import ARCHIVE_ROOT, PROVENANCE_ROOT, REPOSITORY_ROOT


class FrozenArtifactNotReleased(LookupError):
    """The original file is a development-only file that is not part of the release."""


class FrozenArtifactUnavailable(FileNotFoundError):
    """The original file is released in the frozen archive, but the archive is not present."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=1)
def _file_map() -> dict[str, dict[str, str]]:
    with gzip.open(PROVENANCE_ROOT / "file_map.csv.gz", "rt", newline="", encoding="utf-8") as handle:
        return {row["original_path"]: row for row in csv.DictReader(handle)}


@lru_cache(maxsize=1)
def _source_ledger() -> dict[str, dict]:
    return json.loads((PROVENANCE_ROOT / "source_ledger.json").read_text(encoding="utf-8"))["modules"]


@lru_cache(maxsize=1)
def _sanitized() -> dict[str, dict]:
    return json.loads((PROVENANCE_ROOT / "sanitized_ledgers.json").read_text(encoding="utf-8"))


def reconstruct_frozen_source(release_path: str) -> str:
    """Return the frozen source text of a release module by reversing its ledgered edits."""
    entry = _source_ledger()[release_path]
    lines = (REPOSITORY_ROOT / release_path).read_bytes().decode("utf-8").split("\n")
    for hunk in sorted(entry["hunks"], key=lambda item: item["release_line"], reverse=True):
        start = hunk["release_line"] - 1
        stop = start + len(hunk["release_lines"])
        if lines[start:stop] != hunk["release_lines"]:
            raise ValueError(f"{release_path} differs from its provenance ledger near line {hunk['release_line']}")
        lines[start:stop] = hunk["frozen_lines"]
    return "\n".join(lines)


def release_location(original_path: str) -> Path:
    """Return the existing release (or extracted-archive) path of an original frozen file."""
    entry = _file_map().get(original_path)
    if entry is None:
        raise FrozenArtifactNotReleased(original_path)
    candidates = [REPOSITORY_ROOT / entry["release_path"]]
    if entry["location"] == "archive":
        candidates.append(ARCHIVE_ROOT / entry["release_path"])
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FrozenArtifactUnavailable(
        f"{original_path} is stored in the frozen archive ({entry['release_path']}); "
        "extract the archive at the repository root or set TWINGUARD_ARCHIVE_ROOT"
    )


def frozen_sha256(original_path: str) -> str:
    """SHA-256 of the frozen bytes of ``original_path`` (a development-repository path)."""
    entry = _file_map().get(original_path)
    if entry is None:
        raise FrozenArtifactNotReleased(original_path)
    release_path = entry["release_path"]
    if entry["byte_form"] == "SOURCE_LEDGERED":
        return _sha256_bytes(reconstruct_frozen_source(release_path).encode("utf-8"))
    observed = _sha256_bytes(release_location(original_path).read_bytes())
    sanitized = _sanitized().get(release_path)
    if sanitized is not None and observed == sanitized["sanitized_sha256"]:
        return sanitized["original_sha256"]
    return observed


def frozen_sha256_if_released(original_path: str) -> str | None:
    """Like :func:`frozen_sha256`, but ``None`` for development-only files outside the release."""
    try:
        return frozen_sha256(original_path)
    except FrozenArtifactNotReleased:
        return None


def frozen_tree_hash(original_paths: Iterable[str]) -> str:
    """Tree digest ``sha256(concat(path + NUL + sha256(file) + LF))``, as used by the E1 process bundle."""
    digest = hashlib.sha256()
    for relative in original_paths:
        digest.update(relative.encode() + b"\0" + frozen_sha256(relative).encode() + b"\n")
    return digest.hexdigest()
