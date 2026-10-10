"""Define the margin-ladder populations, seed blocks, and evaluation settings."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

from twinguard.scenarios.flash_population_design import purpose_seed

from twinguard.utils.margin_ladder_output_guard import EXPERIMENT_ROOT, REPOSITORY_ROOT, assert_margin_ladder_output


FLASH_CALIBRATION_START = 70_000_000
FLASH_CALIBRATION_COUNT = 1_500
FLASH_TEST_START = 71_000_000
FLASH_TEST_COUNT = 2_000
CSTR_TEST_START = 72_000_000
CSTR_TEST_COUNT = 1_500
BANDS = (
    ("B0", 0.00, 0.15),
    ("B1", 0.15, 0.30),
    ("B2", 0.30, 0.60),
    ("B3", 0.60, 1.00),
    ("B4", 1.00, None),
)

PRIOR_RESERVED_INTERVALS = (
    ("CSTR_PRIMARY_calibration", 1_630_000, 1_630_999),
    ("CSTR_PRIMARY_final", 1_730_000, 1_730_999),
    ("CSTR_PRIMARY_screening", 1_830_000, 1_830_999),
    ("CSTR_MIXED_DEGRADATION_training", 2_100_000, 2_100_599),
    ("CSTR_MIXED_DEGRADATION_selection", 2_200_000, 2_200_059),
    ("CSTR_MIXED_DEGRADATION_calibration", 2_300_000, 2_300_399),
    ("CSTR_MIXED_DEGRADATION_final", 2_400_000, 2_400_599),
    ("TWO_TANK_training", 3_100_000, 3_100_479),
    ("TWO_TANK_selection", 3_200_000, 3_200_053),
    ("TWO_TANK_calibration", 3_300_000, 3_300_299),
    ("TWO_TANK_final", 3_400_000, 3_400_599),
    ("TWO_TANK_bootstrap", 3_500_000, 3_500_044),
    ("CSTR_boundary", 9_300_000, 9_300_299),
    ("FLASH_training", 22_000_000, 22_000_899),
    ("FLASH_selection", 23_000_000, 23_000_179),
    ("FLASH_calibration_ordinary", 24_000_000, 24_000_599),
    ("FLASH_calibration_boundary", 25_000_000, 25_000_599),
    ("FLASH_primary", 26_000_000, 26_001_199),
    ("FLASH_boundary", 27_000_000, 27_001_199),
    ("FLASH_structural", 28_000_000, 28_000_599),
    ("FLASH_rare_event_reserved", 29_000_000, 29_011_999),
    ("FLASH_bootstrap", 31_000_000, 31_000_149),
    ("CSTR_reliability", 50_000_000, 50_002_999),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_seed_manifest() -> None:
    rows: list[dict[str, object]] = []
    populations = (
        ("flash_margin_calibration", FLASH_CALIBRATION_START, FLASH_CALIBRATION_COUNT, "flash"),
        ("flash_margin_test", FLASH_TEST_START, FLASH_TEST_COUNT, "flash"),
        ("cstr_margin_test", CSTR_TEST_START, CSTR_TEST_COUNT, "cstr"),
    )
    for population, start, count, process in populations:
        for index in range(count):
            seed = start + index
            rows.append(
                {
                    "population": population,
                    "process": process,
                    "episode_index": index,
                    "episode_seed": seed,
                    "assigned_margin_band": BANDS[index % len(BANDS)][0],
                    "fault_family_assignment": (
                        f"F{index % 3 + 1}" if process == "flash" else "UA_fouling"
                    ),
                    "purpose_codes": (
                        "101;211;307;503;601" if process == "flash" else "frozen_CSTR_streams"
                    ),
                    "action_pool_seed": purpose_seed(seed, 503) if process == "flash" else "",
                    "predictive_seed": purpose_seed(seed, 601) if process == "flash" else seed + 4_000_000,
                }
            )
    roots = {int(row["episode_seed"]) for row in rows}
    if len(roots) != len(rows):
        raise RuntimeError("duplicate margin ladder episode seed")
    prior_overlap = {
        seed
        for seed in roots
        for _, lower, upper in PRIOR_RESERVED_INTERVALS
        if lower <= seed <= upper
    }
    if prior_overlap:
        raise RuntimeError(f"margin ladder/prior seed overlap: {sorted(prior_overlap)[:5]}")
    path = assert_margin_ladder_output(EXPERIMENT_ROOT / "margin_ladder_seed_manifest.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _source_hashes() -> dict[str, str]:
    files = (
        "artifacts/flash_study/flash_drum.py",
        "artifacts/flash_study/digital_twin.py",
        "artifacts/flash_study/selected_flash_model.json",
        "artifacts/flash_study/flash_bootstrap_models.json",
        "artifacts/flash_study/flash_conformal_config.json",
        "artifacts/flash_study/calibration_artifact_hashes.txt",
        "artifacts/flash_study/reliability_artifact_hashes.txt",
        "artifacts/cstr/bootstrap_ensemble/bootstrap_models.json",
        "artifacts/cstr/primary_evaluation/one_sided_quantiles.csv",
    )
    return {relative: _sha256(REPOSITORY_ROOT / relative) for relative in files}


def write_preregistration() -> None:
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    forbidden_truth = list(EXPERIMENT_ROOT.rglob("*truth*"))
    if forbidden_truth:
        raise RuntimeError("cannot preregister after margin ladder truth artifacts exist")
    _write_seed_manifest()
    hashes = _source_hashes()
    hash_lines = "\n".join(f"- `{path}`: `{digest}`" for path, digest in hashes.items())
    prior_lines = "\n".join(
        f"- `{name}`: `{lower}-{upper}`" for name, lower, upper in PRIOR_RESERVED_INTERVALS
    )
    text = f"""# margin ladder Prospective Preregistration

Status: frozen before any margin ladder physical future was generated. This is a
post-boundary challenge prospective extension; it was not preregistered in population design.

## Immutable Inputs

The CSTR and flash nominal twins, mean models, bootstrap ensembles, physical
limits, scale floors, ensemble sizes, predictive trajectory counts, and prior
conformal quantiles remain frozen. Analysis A uses the frozen flash ordinary
pooled `q=11.04859850832729` for every band because margin ladder spans the continuous
ordinary action domain. The margin ladder selection mechanism is new, so Analysis A is a
prospective reliability characterization rather than an automatic formal
split-conformal guarantee. CSTR uses frozen `q+=5.464557586938352`.

{hash_lines}

Repository baseline before margin ladder: git commit `756c6a2d4cb17fc31c1d7c081feaab3e69b9ca6e`.

## Seed Audit

Prior and reserved root intervals inspected:

{prior_lines}

New mutually disjoint roots:

- Flash margin calibration: `{FLASH_CALIBRATION_START}-{FLASH_CALIBRATION_START + FLASH_CALIBRATION_COUNT - 1}` (300 reserved per band).
- Flash margin test: `{FLASH_TEST_START}-{FLASH_TEST_START + FLASH_TEST_COUNT - 1}` (400 reserved per band).
- CSTR margin test: `{CSTR_TEST_START}-{CSTR_TEST_START + CSTR_TEST_COUNT - 1}` (300 reserved per band).

The complete episode and purpose-stream allocation is in
`margin_ladder_seed_manifest.csv`. Programmatic interval intersection must equal zero.

## Frozen Margin Ladder

The nominal-DT margin is `m_DT=-max_{{h,j}} g_j_DT(h,a)` and is computed without
plant future, fault metadata, a conformal decision, or a physical label.

| Band | Fixed interval | Selection target |
|---|---:|---:|
| B0 | `0.00 < m_DT <= 0.15` | 0.075 |
| B1 | `0.15 < m_DT <= 0.30` | 0.225 |
| B2 | `0.30 < m_DT <= 0.60` | 0.450 |
| B3 | `0.60 < m_DT <= 1.00` | 0.800 |
| B4 | `m_DT > 1.00` | 1.300 |

Each flash episode screens exactly 256 centered Latin-hypercube actions plus
the frozen targeted nominal/axial/corner set using the nominal DT only. From an
assigned band, eligible actions are ordered by distance to the fixed target,
then by pool index; one of the first eight is selected deterministically from
the episode seed. Absence of a candidate yields `NO_ELIGIBLE_ACTION`, without
replacement. Faults cycle F1/F2/F3 independently of truth and band.

## Analysis A

Use M0 nominal DT, M1 frozen discrepancy mean, M2 frozen raw bootstrap, and M3
frozen pooled conformal (`alpha=0.05`) without refitting or recalibration.

## Analysis B

Use the same frozen whole-trajectory safety-direction score. Compute one
finite-sample corrected 95% quantile independently for each B0-B4 band from the
fresh margin ladder calibration population. A band requires at least 200 complete scores;
otherwise it is `UNSUPPORTED` and every test action in that band is NOT
CERTIFIED. Bins may not be merged after truth.

## Chronology And Domain Exits

For test episodes: observation, nominal-only pool, band assignment, candidate
artifact, prediction artifact/hash, decision artifact/hash, physical future,
truth artifact, evaluation. Prediction and decision schemas prohibit fault,
severity, changed-parameter, evaluator-state, actual-safety, and future-truth
fields. A domain exit without a previously observed constraint violation is
`INCOMPLETE_VALIDATION_FAILURE`, not a demonstrated unsafe trajectory.

## CSTR Replication

CSTR Analysis A uses its frozen 313 K event and action grid. The normalized
margin is the already reported CSTR temperature margin in kelvin divided by a
fixed 1 K engineering scale; this preserves the old threshold and the existing
1 K near-boundary diagnostic. No CSTR Analysis B calibration is introduced.

## Verdict Rule

- `margin ladder RELIABILITY OPERATING ENVELOPE ESTABLISHED` if M3 has nonzero useful
  certification in at least three populated bands, false-safe risk remains
  materially below M0, and availability recovers approximately monotonically.
- `margin ladder OPERATING ENVELOPE ESTABLISHED WITH LIMITATIONS` if a nontrivial recovery
  region exists but monotonicity, coverage, domain exits, or residual risk needs
  qualification.
- `margin ladder METHOD REMAINS OVERLY CONSERVATIVE` if useful M3 certification does not
  recover away from B0.
- `margin ladder RELIABILITY TRANSFER NOT SUPPORTED` if false-safe behavior is not
  materially improved over M0 or cross-process behavior is inconsistent.
"""
    prereg = assert_margin_ladder_output(EXPERIMENT_ROOT / "margin_ladder_preregistration.md")
    prereg.write_text(text, encoding="utf-8")
    digest = _sha256(prereg)
    assert_margin_ladder_output(EXPERIMENT_ROOT / "margin_ladder_preregistration.sha256").write_text(
        f"{digest}  margin_ladder_preregistration.md\n", encoding="ascii"
    )


if __name__ == "__main__":
    write_preregistration()
