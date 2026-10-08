"""Source provenance and frozen hash-manifest verification."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from twinguard.utils import provenance

from .common import FLASH, LADDER, PROVENANCE, REPOSITORY_ROOT, CheckResult, read_csv, read_json

KNOWN = {(row["original_path"], row["recorded_sha256"], row["observed_sha256"]): row["explanation"]
         for row in read_csv(REPOSITORY_ROOT / "reproducibility" / "known_discrepancies.csv")}


def check_source_provenance() -> list[CheckResult]:
    """Reversing each module's ledgered edits must reproduce its recorded frozen bytes."""
    ledger = read_json(PROVENANCE / "source_ledger.json")["modules"]
    failures, examples, recorded_ok, recorded_total = 0, [], 0, 0
    for release_path, entry in ledger.items():
        try:
            text = provenance.reconstruct_frozen_source(release_path)
            ok = hashlib.sha256(text.encode("utf-8")).hexdigest() == entry["frozen_sha256"]
        except (OSError, ValueError) as exc:
            ok = False
            examples.append(f"{release_path}: {exc}")
        if not ok:
            failures += 1
            examples.append(release_path)
        if entry["frozen_sha256_recorded_in_manifests"] is not None:
            recorded_total += 1
            recorded_ok += bool(entry["frozen_sha256_recorded_in_manifests"])
    results = [CheckResult("source_provenance_reversal", "provenance", "all release modules and tests", len(ledger), failures,
                           detail="release module = frozen source + ledgered import/path/hash-resolution edits only", examples=examples[:10])]
    results.append(CheckResult("source_matches_frozen_manifests", "provenance", "modules with a recorded frozen hash",
                               recorded_total, recorded_total - recorded_ok,
                               detail="reconstructed frozen source equals the SHA-256 recorded in E4/E5/E6/E7/E6B manifests"))
    return results


def _manifest_entries(path: Path, base: str) -> list[tuple[str, str]]:
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^([0-9a-f]{64})  (.+)$", line)
        if match:
            entries.append((match.group(1), base + match.group(2)))
    return entries


def _verify(name: str, entries: list[tuple[str, str]], require_archive: bool) -> CheckResult:
    verified = mismatched = archived_absent = not_released = known = 0
    examples, known_notes = [], []
    for expected, original in entries:
        try:
            observed = provenance.frozen_sha256(original)
        except provenance.FrozenArtifactNotReleased:
            not_released += 1
            continue
        except provenance.FrozenArtifactUnavailable:
            archived_absent += 1
            continue
        if observed == expected:
            verified += 1
        elif (original, expected, observed) in KNOWN:
            known += 1
            known_notes.append(original)
        else:
            mismatched += 1
            examples.append(original)
    failures = mismatched + (archived_absent if require_archive else 0)
    detail = (f"verified={verified}, mismatched={mismatched}, archived (not extracted)={archived_absent}, "
              f"development-only (not released)={not_released}"
              + (f", documented known discrepancies={known} ({', '.join(known_notes)}; see reproducibility/known_discrepancies.csv)" if known else ""))
    return CheckResult(f"frozen_manifest:{name}", "hash_verification", name, verified + mismatched + (archived_absent if require_archive else 0),
                       failures, detail=detail, examples=examples[:10])


def check_hash_manifests(require_archive: bool) -> list[CheckResult]:
    results = []
    manifests = (
        ("FLASH_E4_ARTIFACT_HASHES.txt", FLASH / "FLASH_E4_ARTIFACT_HASHES.txt", "E_FLASH_RESS/"),
        ("FLASH_E5_ARTIFACT_HASHES.txt", FLASH / "FLASH_E5_ARTIFACT_HASHES.txt", "E_FLASH_RESS/"),
        ("RESS_E6_ARTIFACT_HASHES.txt", FLASH / "RESS_E6_ARTIFACT_HASHES.txt", "E_FLASH_RESS/"),
        ("RESS_FINAL_ARTIFACT_HASHES.txt", FLASH / "RESS_FINAL_ARTIFACT_HASHES.txt", "E_FLASH_RESS/"),
        ("RESS_E7_FROZEN_INPUT_HASHES.txt", FLASH / "RESS_E7_FROZEN_INPUT_HASHES.txt", ""),
        ("E6B_FINAL_ARTIFACT_HASHES.txt", LADDER / "E6B_FINAL_ARTIFACT_HASHES.txt", "E6B_MARGIN_LADDER/"),
    )
    for name, path, base in manifests:
        results.append(_verify(name, _manifest_entries(path, base), require_archive))

    # sidecars and design-file hashes
    design_entries = []
    for sidecar, base in ((FLASH / "FLASH_E2_PREDECLARATION.sha256", "E_FLASH_RESS/"), (FLASH / "RESS_E7_DESIGN.sha256", "E_FLASH_RESS/"),
                          (LADDER / "E6B_PREDECLARATION.sha256", "E6B_MARGIN_LADDER/"),
                          (LADDER / "FLASH_E6B_CONFORMAL_CONFIG.sha256", "E6B_MARGIN_LADDER/")):
        digest, name = sidecar.read_text(encoding="utf-8").split()[:2]
        design_entries.append((digest, base + name))
    e6 = read_json(FLASH / "RESS_E6_DESIGN.json")["frozen_source_hashes"]
    design_entries += [
        (e6["flash_e5_hash_manifest_sha256"], "E_FLASH_RESS/FLASH_E5_ARTIFACT_HASHES.txt"),
        (e6["cstr_one_sided_quantiles_sha256"], "results/milestone_c1/one_sided_quantiles.csv"),
        (e6["cstr_safety_calibration_scores_sha256"], "results/milestone_c1/safety_calibration_scores.csv"),
        (e6["cstr_bootstrap_models_sha256"], "results/milestone_b11/bootstrap_models.json"),
    ]
    e7 = read_json(FLASH / "RESS_E7_DESIGN.json")
    design_entries.append((e7["frozen_input_manifest_sha256"], "E_FLASH_RESS/RESS_E7_FROZEN_INPUT_HASHES.txt"))
    design_entries += [(digest, original) for original, digest in e7["e7_source_hashes"].items()]
    for line in (LADDER / "E6B_PREDECLARATION.md").read_text(encoding="utf-8").splitlines():
        match = re.match(r"^- `([^`]+)`: `([0-9a-f]{64})`$", line)
        if match:
            design_entries.append((match.group(2), match.group(1)))
    c1 = read_json(Path(FLASH.parent / "cstr/primary_evaluation/configuration.json"))["source_hashes"]
    design_entries += [(c1["bootstrap_models_sha256"], "results/milestone_b11/bootstrap_models.json"),
                       (c1["b12_quantiles_sha256"], "results/milestone_b12/conformal_quantiles.csv")]
    results.append(_verify("preregistrations, sidecars and design files", design_entries, require_archive))

    # E1 process bundle (tree hash over flash plant, twin and configuration)
    from twinguard.scenarios.flash_population_design import PROCESS_BUNDLE_SHA256
    bundle = provenance.frozen_tree_hash(("E_FLASH_RESS/flash_process.py", "E_FLASH_RESS/flash_digital_twin.py",
                                          "E_FLASH_RESS/flash_config.yaml"))
    results.append(CheckResult("frozen_process_bundle", "hash_verification", "flash plant + nominal twin + configuration",
                               1, int(bundle != PROCESS_BUNDLE_SHA256), detail="E1 process-bundle tree hash"))
    return results
