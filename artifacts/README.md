# Experiment artifacts

This directory contains the inputs and outputs of the prospective experiments: fitted models,
calibration sets, design files, SHA-256 manifests, per-episode result tables and per-episode
ledgers. The files are read-only. The experiment entry points in `experiments/` rerun stages in
workspaces under `runs/` and never write here.

The individual per-episode files (candidate actions, predictive envelopes, decisions and physical
truths) are not included; see `data/README.md`. The ledgers record each file's SHA-256 and the
nanosecond time at which it was written.

## `cstr/`

| Directory | Content | Code |
|---|---|---|
| `discrepancy_fit/` | CSTR training population (`training_population.npz`), latent-dimension selection, fitted parameters | `twinguard.cstr.fit_discrepancy_model` |
| `discrepancy_freeze/` | selected latent discrepancy model (`frozen_latent_model.json`), EM robustness and variance diagnostics | `twinguard.cstr.freeze_discrepancy_model` |
| `bootstrap_ensemble/` | whole-episode bootstrap ensemble (`bootstrap_models.json`) and diagnostics | `twinguard.cstr.bootstrap_ensemble` |
| `conformal_calibration/` | trajectory-conformal calibration scores and quantiles | `twinguard.cstr.conformal_calibration` |
| `primary_evaluation/` | one-sided safety calibration (n = 500; `one_sided_quantiles.csv`) and the primary prospective test (N = 1000; `final_safety_episodes.csv`, `safety_summary.csv`) | `twinguard.cstr.safety_certification` |
| `primary_decision_audit/` | read-only reconstruction of every primary certification decision from its stored prediction | `twinguard.cstr.primary_decision_audit` |
| `feed_temperature_fault/` | feed-temperature-step degradation: definition and smoke test | `twinguard.cstr.feed_temperature_fault` |
| `mixed_degradation/` | mixed-degradation transfer: retrained model, bootstrap ensemble, calibration (n = 400), test (N = 600) | `twinguard.cstr.mixed_degradation` |

## `flash_study/`

The flash-drum study. It also holds the cross-process (CSTR + flash) boundary, structural and
mixed-regime evaluations, which share this experiment root.

| Files | Content | Code |
|---|---|---|
| `flash_config.yaml`, `process_specification.md`, `simulator_validation/` | simulator specification and healthy plant / nominal-twin validation | `twinguard.flash.validate_simulator` |
| `population_preregistration.md` (+ `.sha256`), `flash_population_manifest.csv`, `flash_seed_manifest.csv` | preregistered population design and seed reservation | `twinguard.flash.reserve_populations` |
| `model_selection_data/`, `flash_model_selection.csv`, `selected_flash_model.json` | discrepancy training / selection data and model selection | `twinguard.flash.fit_discrepancy_model` |
| `flash_bootstrap_manifest.csv`, `flash_bootstrap_models.json`, `calibration_data/`, `flash_calibration_scores.csv`, `flash_conformal_config.json`, `calibration_artifact_hashes.txt` | bootstrap ensemble and ordinary + boundary conformal calibration | `twinguard.flash.bootstrap_and_calibrate` |
| `flash_primary_episode_results.csv`, `flash_primary_ablation.csv`, `primary_evaluation_ledger.csv`, `primary_evaluation_artifact_hashes.txt` | primary prospective evaluation and component ablation | `twinguard.flash.evaluate_primary` |
| `boundary_challenge_design.json`, `flash_boundary_episode_results.csv`, `cstr_boundary_episode_results.csv`, `boundary_risk_availability.csv`, `boundary_challenge_episodes/*_ledger.csv`, `boundary_challenge_artifact_hashes.txt` | boundary-region challenge | `twinguard.evaluation.boundary_challenge` |
| `reliability_design.json` (+ `.sha256`), `reliability_design_amendment.md`, `reliability_input_hashes.txt` | design of the structural / mixed-regime stage, fixed before its populations were generated | `twinguard.evaluation.freeze_reliability_design` |
| `flash_structural_*`, `flash_reliability_*`, `cstr_reliability_*`, `reliability_episodes/*_ledger.csv` | structural mismatch and secondary mixed-regime audits | `twinguard.evaluation.structural_and_reliability` |
| `final_audit_checks.csv`, `final_audit_results.csv`, `reliability_artifact_hashes.txt` | independent reconstruction audit across all populations | `twinguard.evaluation.final_audit` |

## `flash_margin_ladder/`

The margin-ladder stress analysis (`twinguard.evaluation.margin_ladder`):

- preregistration: `margin_ladder_preregistration.md` (+ `.sha256`) and `margin_ladder_seed_manifest.csv`;
- per-band calibration: `flash_band_calibration_scores.csv` and `flash_band_conformal_config.json` (+ `.sha256`);
- per-episode results (`*_margin_episodes.csv`, `*_margin_results.csv`), the risk-availability
  table, audit checks, per-episode ledgers (`artifacts/*.csv`) and hashes.

## Method labels

| Identifier | Meaning |
|---|---|
| `M0_nominal_DT`, `M1_discrepancy_mean`, `M2_raw_bootstrap`, `M3_pooled_conformal` | flash component ablation; M3 is the TwinGuard method |
| `M4_action_conditioned` / `action_conditioned` | an unsupported placeholder that is always NOT CERTIFIED. It is not part of the TwinGuard method. |
| `M5_margin_conditioned` | a margin-ladder stress comparator |
| `S0`…`S3` | CSTR primary comparators; S3 is the TwinGuard one-sided conformal rule |
| `proposed_method` / `frozen_CSTR_conformal` | the TwinGuard CSTR rule |

## `retrospective_baselines/`

Post-hoc comparison controls on the same calibration and test populations. They are not
prospective and not part of the method. See `paper/result_mapping.md`.

| Directory | Content |
|---|---|
| `cstr_conformal_controls/` | CSTR C0/C1 constant-width conformal controls with the reconstructed M0–M3 results, plus α sensitivity |
| `flash_conformal_controls/` | flash C0/C1 fixed-scale controls with the reconstructed M0–M3 results, α sensitivity and calibration regeneration checks |
| `cstr_persistent_offset/` | CSTR Offset-C: calibration scores, per-episode test decisions and summary. Cache: `offset_residual_histories.npz` (pre-decision residuals rebuilt from the episode seeds). |
| `flash_persistent_offset/` | flash Offset-C: calibration scores, per-episode decisions and summary. Cache: `calibration_truths_verified.npz` (verified calibration futures). |
