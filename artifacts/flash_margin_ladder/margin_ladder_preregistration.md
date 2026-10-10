# margin ladder Prospective Preregistration

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

- `artifacts/flash_study/selected_flash_model.json`: `56575a51e870ec57279a5d31476880a9e499617a1b7870ce0be59c1b21ab14d2`
- `artifacts/flash_study/flash_bootstrap_models.json`: `00bd43e83cc268fd7a5720c742610db6d8d8fb42e6e9b0658c5cd1fb91dd5f68`
- `artifacts/flash_study/flash_conformal_config.json`: `91364ac71702c9849c79bc2b7ac0772a1a540ff336521c41ea276656594adfad`
- `artifacts/flash_study/calibration_artifact_hashes.txt`: `99af69b723e83eeba8deefb94104cca997f3eb32ef3f3c2885e146163c48fff7`
- `artifacts/flash_study/reliability_artifact_hashes.txt`: `c17eddcf429b17fd40b4c82746b41e15ce7197bd342b2690286d917cef51c6ae`
- `artifacts/cstr/bootstrap_ensemble/bootstrap_models.json`: `247376d2cfdcb33696f922ef2895a008a9dbd00f03673c7f8728f0b5c70095ad`
- `artifacts/cstr/primary_evaluation/one_sided_quantiles.csv`: `35ebe6c02d0651819d6378aa6bc34104e596becd8a495ce3d7c18fcf1832d9b5`

Repository baseline before margin ladder: git commit `756c6a2d4cb17fc31c1d7c081feaab3e69b9ca6e`.

## Seed Audit

Prior and reserved root intervals inspected:

- `CSTR_PRIMARY_calibration`: `1630000-1630999`
- `CSTR_PRIMARY_final`: `1730000-1730999`
- `CSTR_PRIMARY_screening`: `1830000-1830999`
- `CSTR_MIXED_DEGRADATION_training`: `2100000-2100599`
- `CSTR_MIXED_DEGRADATION_selection`: `2200000-2200059`
- `CSTR_MIXED_DEGRADATION_calibration`: `2300000-2300399`
- `CSTR_MIXED_DEGRADATION_final`: `2400000-2400599`
- `TWO_TANK_training`: `3100000-3100479`
- `TWO_TANK_selection`: `3200000-3200053`
- `TWO_TANK_calibration`: `3300000-3300299`
- `TWO_TANK_final`: `3400000-3400599`
- `TWO_TANK_bootstrap`: `3500000-3500044`
- `CSTR_boundary`: `9300000-9300299`
- `FLASH_training`: `22000000-22000899`
- `FLASH_selection`: `23000000-23000179`
- `FLASH_calibration_ordinary`: `24000000-24000599`
- `FLASH_calibration_boundary`: `25000000-25000599`
- `FLASH_primary`: `26000000-26001199`
- `FLASH_boundary`: `27000000-27001199`
- `FLASH_structural`: `28000000-28000599`
- `FLASH_rare_event_reserved`: `29000000-29011999`
- `FLASH_bootstrap`: `31000000-31000149`
- `CSTR_reliability`: `50000000-50002999`

New mutually disjoint roots:

- Flash margin calibration: `70000000-70001499` (300 reserved per band).
- Flash margin test: `71000000-71001999` (400 reserved per band).
- CSTR margin test: `72000000-72001499` (300 reserved per band).

The complete episode and purpose-stream allocation is in
`margin_ladder_seed_manifest.csv`. Programmatic interval intersection must equal zero.

## Frozen Margin Ladder

The nominal-DT margin is `m_DT=-max_{h,j} g_j_DT(h,a)` and is computed without
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
