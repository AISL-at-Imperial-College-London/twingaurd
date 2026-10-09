# Frozen experiment artifacts

This directory contains the frozen inputs and outputs of the prospective experiments. They are
read-only evidence: the experiment entry points rerun stages in workspaces under `runs/` and never
write here.

File names inside each directory are exactly those written by the frozen runs, because the frozen
SHA-256 manifests refer to them. Directory names were changed to describe their content. The
original development paths are listed in `reproducibility/provenance/path_map.csv` and
`file_map.csv.gz`.

## `cstr/`

| Directory | Content | Code |
|---|---|---|
| `discrepancy_fit/` | CSTR training population (`training_population.npz`), latent-dimension selection, fitted parameters | `twinguard.cstr.fit_discrepancy_model` |
| `discrepancy_freeze/` | frozen latent discrepancy model (`frozen_latent_model.json`), EM robustness and variance diagnostics | `twinguard.cstr.freeze_discrepancy_model` |
| `bootstrap_ensemble/` | whole-episode bootstrap ensemble (`bootstrap_models.json`) and diagnostics | `twinguard.cstr.bootstrap_ensemble` |
| `conformal_calibration/` | trajectory-conformal calibration scores and quantiles | `twinguard.cstr.conformal_calibration` |
| `primary_evaluation/` | one-sided safety calibration (n = 500; `one_sided_quantiles.csv`) and the primary prospective test (N = 1000; `final_safety_episodes.csv`, `safety_summary.csv`) | `twinguard.cstr.safety_certification` |
| `primary_decision_audit/` | read-only reconstruction of every primary decision | `twinguard.cstr.primary_decision_audit` |
| `feed_temperature_fault/` | feed-temperature-step fault definition and smoke test | `twinguard.cstr.feed_temperature_fault` |
| `mixed_degradation/` | mixed-degradation transfer: retrained model, bootstrap, calibration (n = 400), test (N = 600) | `twinguard.cstr.mixed_degradation` |
| `final_summary/` | closing summary of the CSTR study written during development (no generating script; its numbers cite the files above). Read by the flash primary-evaluation comparison figure. | — |

Only in the archive:

- `episode_design/`
- `stable_arx_order_study/`
- `observable_conditioning_study/`

These are inputs to the CSTR discrepancy refit.

## `flash_study/`

This is the root directory of the flash study. It also holds the cross-case (CSTR + flash)
boundary, structural and mixed-regime evaluations, which were run in the same frozen namespace.

| Files | Content | Code |
|---|---|---|
| `flash_config.yaml`, `FLASH_PROCESS_SPECIFICATION.md`, `e1_validation/`, `FLASH_E1_VALIDATION.md` | simulator specification and healthy plant / nominal-twin validation | `twinguard.flash.validate_simulator` |
| `FLASH_E2_PREDECLARATION.md` (+ `.sha256`), `flash_population_manifest.csv`, `flash_seed_manifest.csv` | preregistered design and seed reservation | `twinguard.flash.reserve_populations` |
| `e3_data/`, `flash_model_selection.csv`, `selected_flash_model.json`, `FLASH_E3_MODEL_SELECTION.md` | discrepancy training / selection data and model selection | `twinguard.flash.fit_discrepancy_model` |
| `flash_bootstrap_*`, `e4_data/`, `flash_calibration_scores.csv`, `flash_conformal_config.json`, `FLASH_E4_*` | bootstrap ensemble and ordinary + boundary conformal calibration | `twinguard.flash.bootstrap_and_calibrate` |
| `FLASH_FINAL_RESULTS.csv`, `FLASH_FINAL_ABLATION.csv`, `e5_artifact_audit.csv`, `FLASH_E5_*` | primary prospective evaluation and component ablation | `twinguard.flash.evaluate_primary` |
| `RESS_E6_DESIGN.json`, `FLASH_BOUNDARY_RESULTS.csv`, `CSTR_BOUNDARY_RESULTS.csv`, `RISK_AVAILABILITY_RESULTS.csv`, `e6_artifacts/*.csv`, `RESS_E6_*` | boundary-region challenge | `twinguard.evaluation.boundary_challenge` |
| `RESS_E7_DESIGN.json` (+ `.sha256`), `RESS_E7_AMENDMENT_01.md`, `RESS_E7_FROZEN_INPUT_HASHES.txt` | frozen design of the structural / mixed-regime stage | `twinguard.evaluation.freeze_reliability_design` |
| `FLASH_STRUCTURAL_*`, `FLASH_RELIABILITY_*`, `CSTR_RELIABILITY_*`, `e7_artifacts/*.csv` | structural mismatch and secondary mixed-regime audits | `twinguard.evaluation.structural_and_reliability` |
| `RESS_FINAL_*`, `RESS_E7_RESULTS.md` | the original final audit record, claims audit and artifact hashes | `twinguard.evaluation.final_audit` |
| `FROZEN_INPUT_HASHES.txt` | package-level hashes of the development results at the start of the flash study (provenance only) | — |

**Per-episode artifacts are in the archive:**

- `e5_artifacts/`
- `e6_artifacts/{flash,cstr}/`
- `e7_artifacts/{structural,flash_reliability,cstr_reliability}/`
- the stage figure directories

## `flash_margin_ladder/`

The margin-ladder stress analysis:

- preregistration: `E6B_PREDECLARATION.md` (+ `.sha256`) and `E6B_SEED_MANIFEST.csv`;
- per-band calibration: `FLASH_E6B_CALIBRATION_SCORES.csv` and `FLASH_E6B_CONFORMAL_CONFIG.json`;
- per-episode results, risk-availability tables, audit and hashes.

The per-episode artifacts are in the archive (`artifacts/{cstr_test,flash_calibration,flash_test}/`).

The code is `twinguard.evaluation.margin_ladder`.

## Notes on method labels in frozen files

The frozen files keep their original method identifiers:

| Identifier | Meaning |
|---|---|
| `M0_nominal_DT`, `M1_discrepancy_mean`, `M2_raw_bootstrap`, `M3_pooled_conformal` | flash ablation; M3 is the TwinGuard method |
| `M4_action_conditioned` / `action_conditioned` | an unsupported placeholder that is always NOT CERTIFIED. It is not part of the TwinGuard method. |
| `M5_margin_conditioned` | a margin-ladder stress comparator |
| `S0`…`S3` | CSTR primary comparators; S3 is the TwinGuard one-sided conformal rule |
| `proposed_method` / `frozen_CSTR_conformal` | the TwinGuard CSTR rule |

## `retrospective_baselines/`

These are post-hoc comparison controls on the frozen populations. They are not prospective and
not part of the method. See `paper/result_mapping.md`.

| Directory | Content |
|---|---|
| `cstr_conformal_controls/` | CSTR C0/C1 constant-width conformal controls with the reconstructed M0–M3 results, plus α sensitivity |
| `flash_conformal_controls/` | flash C0/C1 fixed-scale controls with the reconstructed M0–M3 results, α sensitivity and E4 regeneration checks |
| `cstr_persistent_offset/` | CSTR Offset-C: calibration scores, per-episode test decisions and summary. Cache: `offset_residual_histories.npz` (pre-decision residuals rebuilt from the frozen seeds). |
| `flash_persistent_offset/` | flash Offset-C: calibration scores, per-episode decisions and summary. Cache: `e4_calibration_truths_verified.npz` (verified E4 calibration futures). |
