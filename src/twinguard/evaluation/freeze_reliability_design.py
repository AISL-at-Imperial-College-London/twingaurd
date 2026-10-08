"""Freeze the E7 design and every pre-E7 result hash before reserved truth."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

from twinguard.evaluation import structural_and_reliability as e7
from twinguard.utils.flash_output_guard import EXPERIMENT_ROOT, REPOSITORY_ROOT


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exclusive_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def frozen_paths() -> list[Path]:
    paths = [path for path in (REPOSITORY_ROOT / "results").rglob("*") if path.is_file()]
    for path in EXPERIMENT_ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(EXPERIMENT_ROOT).as_posix()
        if (
            "__pycache__" in relative
            or relative.startswith("e7_")
            or relative.startswith("RESS_E7")
            or relative.startswith("RESS_FINAL")
            or relative in {
                "run_ress_e7.py",
                "flash_structural_mismatch.py",
                "freeze_ress_e7.py",
                "audit_ress_final.py",
            }
        ):
            continue
        paths.append(path)
    return sorted(set(paths), key=lambda path: path.relative_to(REPOSITORY_ROOT).as_posix())


def main() -> None:
    if e7.DESIGN_PATH.exists() or e7.FROZEN_MANIFEST_PATH.exists():
        raise FileExistsError("E7 design is already frozen and must not be overwritten")
    if (EXPERIMENT_ROOT / "e7_artifacts").exists():
        raise RuntimeError("E7 reserved artifacts exist before design freeze")
    paths = frozen_paths()
    with ThreadPoolExecutor(max_workers=16) as pool:
        hashes = list(pool.map(sha256, paths))
    lines = ["# SHA-256 inventory of every pre-E7 result artifact", ""]
    for path, digest in zip(paths, hashes):
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        lines.append(f"{digest}  {relative}")
    manifest = ("\n".join(lines) + "\n").encode("ascii")
    exclusive_write(e7.FROZEN_MANIFEST_PATH, manifest)
    ordinary_q, boundary_q = e7._load_quantiles()
    design = {
        "schema_version": "RESS_E7_DESIGN_1.0",
        "freeze_date": "2026-08-13",
        "frozen_before_truth": True,
        "frozen_input_file_count": len(lines) - 2,
        "frozen_input_manifest_sha256": sha256(e7.FROZEN_MANIFEST_PATH),
        "e7_source_hashes": {
            "E_FLASH_RESS/run_ress_e7.py": sha256(EXPERIMENT_ROOT / "run_ress_e7.py"),
            "E_FLASH_RESS/flash_structural_mismatch.py": sha256(
                EXPERIMENT_ROOT / "flash_structural_mismatch.py"
            ),
        },
        "structural_amendment": {
            "status": "PROSPECTIVE_BEFORE_E7_TRUTH",
            "deviation_from_e2_placeholder": (
                "E2 reserved nonideal VLE as F4; the explicit E7 instruction instead "
                "preferred unmodelled actuator dynamics. No E1-E6 artifact is changed."
            ),
            "single_mechanism": "plant-only first-order vapor-valve actuator lag",
            "nominal_dt_contains_mechanism": False,
            "effective_actuator_state_visible_to_method": False,
            "time_constant_s": [30.0, 120.0],
            "severity_sampling": "tau=30+90*s, s~Uniform[0,1] from purpose stream 811",
            "observation_protocol": "unchanged 50 samples at 10 s under nominal command",
            "episode_count": e7.STRUCTURAL_COUNT,
            "seed_start": e7.STRUCTURAL_SEED_START,
            "seed_end": e7.STRUCTURAL_SEED_START + e7.STRUCTURAL_COUNT - 1,
            "actions": "exactly half frozen centered-LHS ordinary and half nominal-DT boundary",
            "calibration_interpretation": "matching frozen q values applied descriptively; no structural-fault calibration claim",
        },
        "flash_reliability": {
            "episode_count": e7.FLASH_RELIABILITY_COUNT,
            "seed_start": e7.FLASH_RELIABILITY_SEED_START,
            "seed_end": e7.FLASH_RELIABILITY_SEED_START + e7.FLASH_RELIABILITY_COUNT - 1,
            "faults": "balanced cyclic F1/F2/F3, 1000 each",
            "actions": "1500 frozen centered-LHS ordinary and 1500 nominal-DT boundary",
            "ordinary_pooled_q_95": ordinary_q,
            "boundary_pooled_q_95": boundary_q,
            "denominator_policy": "new population, never merged with E5/E6",
        },
        "cstr_reliability": {
            "episode_count": e7.CSTR_RELIABILITY_COUNT,
            "seed_start": e7.CSTR_RELIABILITY_SEED_START,
            "seed_end": e7.CSTR_RELIABILITY_SEED_START + e7.CSTR_RELIABILITY_COUNT - 1,
            "fault": "UA fouling under frozen C1 generator",
            "actions": "1000 each interpolation/boundary/extrapolation from frozen sampler",
            "q_plus_95": e7.CSTR_Q_PLUS,
            "temperature_limit_k": e7.CSTR_T_MAX_K,
            "denominator_policy": "new population, never merged with C1/C1.6",
        },
        "prospective_protocol": [
            "candidate artifact",
            "prediction artifact",
            "decision artifact",
            "physical truth reveal",
        ],
        "intervals": {
            "confidence": 0.95,
            "exact": "two-sided Clopper-Pearson",
            "score": "Wilson",
            "zero_event_word_rule": "report 0 observed/N with finite upper confidence bound; never zero risk",
        },
    }
    encoded = (json.dumps(design, indent=2, sort_keys=True) + "\n").encode("utf-8")
    exclusive_write(e7.DESIGN_PATH, encoded)
    exclusive_write(
        EXPERIMENT_ROOT / "RESS_E7_DESIGN.sha256",
        f"{sha256(e7.DESIGN_PATH)}  RESS_E7_DESIGN.json\n".encode("ascii"),
    )
    print(
        f"froze E7 design with {design['frozen_input_file_count']} prior artifacts; "
        f"manifest={design['frozen_input_manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
