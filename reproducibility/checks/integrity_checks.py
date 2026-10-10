"""Integrity of the released result files.

* reproducibility/artifact_manifest.csv: SHA-256 and size of every file under artifacts/ and results/
* the SHA-256 manifests written by the flash and margin-ladder stages
* preregistration and design sidecars (``*.sha256``)
* hash references between design/configuration files (keys ending in ``_sha256``, and the
  input list of the margin-ladder preregistration)
"""

from __future__ import annotations

import re
from pathlib import Path

from .common import ARTIFACTS, FLASH, LADDER, REPOSITORY_ROOT, CheckResult, read_csv, read_json, sha256_file

ARTIFACT_MANIFEST = REPOSITORY_ROOT / "reproducibility" / "artifact_manifest.csv"
STAGE_MANIFESTS = (  # (manifest, directory its paths are relative to)
    (FLASH / "calibration_artifact_hashes.txt", FLASH),
    (FLASH / "primary_evaluation_artifact_hashes.txt", FLASH),
    (FLASH / "boundary_challenge_artifact_hashes.txt", FLASH),
    (FLASH / "reliability_artifact_hashes.txt", FLASH),
    (FLASH / "reliability_input_hashes.txt", REPOSITORY_ROOT),
    (LADDER / "margin_ladder_artifact_hashes.txt", LADDER),
)
# hashes of source files recorded at preregistration time (not result files)
SOURCE_HASH_KEYS = {"process_bundle_sha256"}


def _verify(check: str, subject: str, entries: list[tuple[str, Path]], detail: str) -> CheckResult:
    failures, examples = 0, []
    for expected, path in entries:
        if not path.is_file():
            failures += 1
            examples.append(f"missing {path.relative_to(REPOSITORY_ROOT).as_posix()}")
        elif sha256_file(path) != expected:
            failures += 1
            examples.append(f"hash differs {path.relative_to(REPOSITORY_ROOT).as_posix()}")
    return CheckResult(check, "hash_verification", subject, len(entries), failures, detail=detail, examples=examples[:5])


def check_artifact_manifest() -> list[CheckResult]:
    rows = read_csv(ARTIFACT_MANIFEST)
    listed = {row["path"] for row in rows}
    on_disk = {path.relative_to(REPOSITORY_ROOT).as_posix()
               for root in (ARTIFACTS, REPOSITORY_ROOT / "results") for path in root.rglob("*") if path.is_file()}
    entries = [(row["sha256"], REPOSITORY_ROOT / row["path"]) for row in rows]
    size_failures = [row["path"] for row in rows
                     if (REPOSITORY_ROOT / row["path"]).is_file() and (REPOSITORY_ROOT / row["path"]).stat().st_size != int(row["bytes"])]
    results = [_verify("artifact_manifest:hashes", ARTIFACT_MANIFEST.name, entries,
                       "every released result file matches its recorded SHA-256")]
    results.append(CheckResult("artifact_manifest:sizes", "hash_verification", ARTIFACT_MANIFEST.name, len(rows), len(size_failures),
                               detail="every released result file matches its recorded size", examples=size_failures[:5]))
    unlisted = sorted(on_disk - listed)
    results.append(CheckResult("artifact_manifest:coverage", "hash_verification", "artifacts/, results/", len(on_disk), len(unlisted),
                               detail="every file under artifacts/ and results/ is listed in the manifest", examples=unlisted[:5]))
    return results


def _manifest_entries(path: Path, base: Path) -> list[tuple[str, Path]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-f]{64})  (.+)$", line)
        if match:
            entries.append((match.group(1), base / match.group(2)))
    return entries


def check_stage_manifests() -> list[CheckResult]:
    return [_verify(f"stage_manifest:{manifest.name}", manifest.relative_to(REPOSITORY_ROOT).as_posix(),
                    _manifest_entries(manifest, base), "every entry of the stage's SHA-256 manifest matches the released file")
            for manifest, base in STAGE_MANIFESTS]


def _hash_keys(value, key=""):
    if isinstance(value, dict):
        for child_key, child in value.items():
            yield from _hash_keys(child, child_key)
    elif isinstance(value, list):
        for child in value:
            yield from _hash_keys(child, key)
    elif isinstance(value, str) and key.endswith("sha256") and re.fullmatch(r"[0-9a-f]{64}", value):
        yield key, value


def check_design_references() -> list[CheckResult]:
    sidecars = []
    for sidecar in sorted(ARTIFACTS.rglob("*.sha256")):
        digest, name = sidecar.read_text(encoding="utf-8").split()[:2]
        sidecars.append((digest, sidecar.parent / name))
    results = [_verify("design_sidecars", "*.sha256", sidecars, "each preregistration / design sidecar matches its document")]

    released = {}
    for path in ARTIFACTS.rglob("*"):
        if path.is_file():
            released.setdefault(sha256_file(path), path)
    references, unresolved, source_hashes = 0, [], 0
    for document in sorted(ARTIFACTS.rglob("*.json")):
        for key, digest in _hash_keys(read_json(document)):
            if key in SOURCE_HASH_KEYS:
                source_hashes += 1
                continue
            references += 1
            if digest not in released:
                unresolved.append(f"{document.relative_to(REPOSITORY_ROOT).as_posix()}:{key}")
    results.append(CheckResult("design_hash_references", "hash_verification", "artifacts/**/*.json", references, len(unresolved),
                               detail="every *_sha256 value in a design or configuration file equals the hash of a released file "
                                      f"({source_hashes} source-code bundle hashes recorded at preregistration are not result files)",
                               examples=unresolved[:5]))

    preregistration = LADDER / "margin_ladder_preregistration.md"
    entries = [(digest, REPOSITORY_ROOT / relative)
               for relative, digest in re.findall(r"^- `([^`]+)`: `([0-9a-f]{64})`", preregistration.read_text(encoding="utf-8"), re.M)]
    results.append(_verify("margin_ladder_preregistration_inputs", preregistration.name, entries,
                           "every input hash listed in the margin-ladder preregistration matches the released file"))
    return results
