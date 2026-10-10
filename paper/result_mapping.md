# Manuscript result → file → script mapping

Table numbers follow the manuscript (Table 5 = flash ablation, Table 6 = primary results,
Table 7 = mixed-regime audit).

All paths are relative to the repository root. Throughout this document:

- **"Source"** is the file written by the prospective run. It is hash-verified by
  `reproducibility/run_reproducibility_audit.py`.
- **"Paper table"** is the file in `results/tables/`, derived from the source by
  `scripts/generate_paper_tables.py` (row selection and relabelling only).
- **`paper_headline_results.csv`** collects every number quoted in the text, one row per
  experiment and method. Its last two columns name the exact source file and row.

## Tables

### Table 5: Flash internal component ablation (M0 → M1 → M2 → M3)

| | |
|---|---|
| Paper table | `results/tables/flash_component_ablation.csv` (pooled rows: `group_type=pooled`; subgroups by fault family, severity, action distance, initial region and active constraint) |
| Source | `artifacts/flash_study/flash_primary_ablation.csv`, lines 2–5 (`group_type=pooled`, methods `M0_nominal_DT`, `M1_discrepancy_mean`, `M2_raw_bootstrap`, `M3_pooled_conformal`) |
| Per-episode evidence | `artifacts/flash_study/flash_primary_episode_results.csv` (1200 episodes × methods) and ledger `artifacts/flash_study/primary_evaluation_ledger.csv` |
| Script | `experiments/evaluate_flash_primary.py` → `twinguard.flash.evaluate_primary` (`evaluate_artifacts`, `aggregate`, `subgroup_rows`) |
| Values | M0: 779 certified / 207 false-safe (17.25%). M1: 672 / 103 (8.58%). M2: 623 / 72 (6.00%). M3: 177 / 5 (0.42%), availability 14.75%, coverage 0.9399 |

The source also has a fifth row, `M4_action_conditioned`. It is a hard-coded
"NOT CERTIFIED" placeholder for an unsupported action-conditioned calibration, and it is omitted
from the paper table.

### Table 6: Primary prospective results (CSTR and flash)

**CSTR (N = 1000).**

| | |
|---|---|
| Paper table | `results/tables/cstr_primary_results.csv`; headline rows in `paper_headline_results.csv` |
| Source | `artifacts/cstr/primary_evaluation/safety_summary.csv`:<br>• line 45 (`S3_one_sided, 0.95, pooled`): 338 certified (33.8%), 0 false-safe, upper whole-trajectory coverage 0.957<br>• line 9 (`S0_nominal_DT, 0.95, pooled`): nominal twin, 394 false-safe |
| Per-episode evidence | `artifacts/cstr/primary_evaluation/final_safety_episodes.csv`; reconstruction audit `artifacts/cstr/primary_decision_audit/` |
| Calibration | `artifacts/cstr/primary_evaluation/one_sided_quantiles.csv` (q⁺ = 5.464557586938352 at 95%, n = 500) |
| Script | `experiments/evaluate_cstr_primary.py` → `twinguard.cstr.safety_certification` |

**Flash (N = 1200).**

| | |
|---|---|
| Source | `artifacts/flash_study/flash_primary_ablation.csv`, line 5 (M3): 177 certified (14.75%), 5 false-safe (joint 0.42%), coverage 0.9399 |
| Script | `experiments/evaluate_flash_primary.py` |

The coverage of 0.9399 is computed over 899 episodes with complete prediction and truth. 292
episodes have truncated truth: the physical simulation left its domain, and these are counted as
unsafe.

### Table 7: Secondary mixed-regime reliability audit

**CSTR (N = 3000).** The population has 1000 episodes each in the interpolation, boundary and
extrapolation regimes.

| | |
|---|---|
| Paper table | `results/tables/mixed_regime_audit_results.csv` |
| Source | `artifacts/flash_study/cstr_reliability_summary.csv`, row `frozen_CSTR_conformal`: 1010 certified, 0 false-safe |
| Per-episode evidence | `cstr_reliability_episode_results.csv`, ledger `reliability_episodes/cstr_reliability_ledger.csv` |

**Flash (N = 3000).** The population has F1/F2/F3 at 1000 each, with 1500 ordinary-action and
1500 boundary-action episodes.

| | |
|---|---|
| Source | `artifacts/flash_study/flash_reliability_summary.csv`, row `M3_pooled_conformal`: 237 certified, 6 false-safe |
| Per-episode evidence | `flash_reliability_episode_results.csv`, ledger `reliability_episodes/flash_reliability_ledger.csv` |

**Both processes.**

| | |
|---|---|
| Script | `experiments/evaluate_structural_mismatch_and_mixed_regime_audits.py` → `twinguard.evaluation.structural_and_reliability` (one run; separate denominators) |
| Design | `artifacts/flash_study/reliability_design.json` (+ `.sha256`), fixed before any truth |

## Other reported results

### CSTR mixed-degradation / fault-blind transfer (heat-transfer fouling and feed-temperature shift; N = 600)

| | |
|---|---|
| Paper tables | `results/tables/cstr_mixed_degradation_results.csv`, `..._by_fault.csv`, `..._by_severity.csv` |
| Source | `artifacts/cstr/mixed_degradation/mixed_degradation_summary.csv`:<br>• line 4 (`nominal_DT`): 265 false-safe<br>• line 5 (`proposed_method`): 141 certified (23.5%), 0 false-safe, coverage 0.9483 |
| Per-episode evidence | `artifacts/cstr/mixed_degradation/mixed_degradation_test_episodes.csv` |
| Script | `experiments/evaluate_cstr_mixed_degradation.py` → `twinguard.cstr.mixed_degradation`; fault definition `experiments/check_cstr_feed_temperature_fault.py` |
| Note | Each episode carries one hidden mechanism drawn from the mixture. Faults are not concurrent. |

### Boundary-region challenge

| | |
|---|---|
| Paper table | `results/tables/boundary_risk_availability.csv` |
| Source | `artifacts/flash_study/boundary_risk_availability.csv` |
| Flash (N = 1200) | • line 2 (`nominal_DT`): 853 unsafe, 711 certified, 447 false-safe (37.25%)<br>• lines 3–7 (`pooled_conformal`, levels 0.90–0.99): 0 certified |
| CSTR (N = 300) | • line 8 (`nominal_DT`): 300 unsafe, all 300 certified<br>• line 9 (`frozen_CSTR_conformal`): 0 certified |
| Per-episode evidence | `flash_boundary_episode_results.csv`, `cstr_boundary_episode_results.csv`, ledgers in `boundary_challenge_episodes/` |
| Script | `experiments/evaluate_boundary_challenge.py` → `twinguard.evaluation.boundary_challenge` |
| Calibration | flash boundary quantiles from the separate boundary-calibration population (`artifacts/flash_study/flash_conformal_config.json`, `calibration_boundary`; `boundary_challenge_design.json`) |

The 853 unsafe flash episodes comprise 587 complete-and-unsafe trajectories and 266 domain exits.

### Flash structural mismatch (unmodelled vapour-valve actuator lag; reserved N = 600)

| | |
|---|---|
| Paper table | `results/tables/flash_structural_mismatch_results.csv` |
| Source | `artifacts/flash_study/flash_structural_summary.csv`, row `M3_pooled_conformal`: 43 certified, 0 false-safe |
| Completeness | 564 complete / 36 incomplete, counted from `flash_structural_episode_results.csv` (`truth_complete`); see `paper_headline_results.csv` |
| Script | `experiments/evaluate_structural_mismatch_and_mixed_regime_audits.py`; plant `twinguard.plants.flash_drum_valve_lag` |
| Note | Descriptive stress test. The ordinary/boundary quantiles are reused unchanged. There is no structural-fault calibration claim. |

### Margin-ladder stress analysis

| | |
|---|---|
| Paper table | `results/tables/margin_ladder_risk_availability.csv` |
| Sources | `artifacts/flash_margin_ladder/margin_ladder_risk_availability.csv`, `flash_margin_results.csv`, `cstr_margin_results.csv`, preregistration `margin_ladder_preregistration.md` (+ `.sha256`) |
| Script | `experiments/evaluate_flash_margin_ladder.py` → `twinguard.evaluation.margin_ladder` |
| Note | `M5_margin_conditioned` is a preregistered stress-analysis comparator (per-band calibration), not part of the TwinGuard method. |

### Discrepancy-model fitting, selection and calibration

**Flash.**

| | |
|---|---|
| Model selection | `results/tables/flash_model_selection.csv` ← `artifacts/flash_study/flash_model_selection.csv`, `selected_flash_model.json` |
| Script | `experiments/fit_and_select_flash_discrepancy_model.py` |
| Bootstrap ensemble | `artifacts/flash_study/flash_bootstrap_models.json`, `flash_bootstrap_manifest.csv` (`experiments/build_flash_bootstrap_and_calibrate.py`) |
| Conformal quantiles | `results/tables/flash_conformal_quantiles.csv` ← `artifacts/flash_study/flash_conformal_config.json`; scores `flash_calibration_scores.csv`. Ordinary (n = 447) and boundary (n = 463) calibration populations are kept separate. |

**CSTR.** Each stage's outputs, with the script that produces them:

| Stage | Outputs | Script |
|---|---|---|
| Discrepancy fit | `artifacts/cstr/discrepancy_fit/` | `experiments/fit_cstr_discrepancy_model.py` |
| Discrepancy freeze | `artifacts/cstr/discrepancy_freeze/` | `experiments/freeze_cstr_discrepancy_model.py` |
| Bootstrap ensemble | `artifacts/cstr/bootstrap_ensemble/` | `experiments/build_cstr_bootstrap_ensemble.py` |
| Trajectory conformal | `artifacts/cstr/conformal_calibration/` | `experiments/calibrate_cstr_conformal.py` |

### Simulator verification (flash)

| | |
|---|---|
| Healthy plant / nominal-twin parity | `results/tables/flash_healthy_parity_summary.csv`, figure `results/figures/appendix/flash_healthy_plant_twin_parity.png` |
| Manipulated-action directional responses | `results/tables/flash_action_step_responses.csv`, figure `results/figures/appendix/flash_manipulated_action_step_responses.png` |
| Script | `experiments/validate_flash_simulator.py` → `twinguard.flash.validate_simulator` |
| Rerun | regenerates these outputs bit-identically: `python scripts/reproduce_main_results.py --rerun flash-simulator` |

## Figures

| Figure | File | Drawn by | Stage output |
|---|---|---|---|
| Representative corrected flash counterfactual rollout | `results/figures/main/flash_representative_corrected_rollout.png` | `twinguard.flash.evaluate_primary.create_figures` | `primary_evaluation_figures/01_representative_corrected_rollout.png` |
| Nominal-twin vs true safety-margin geometry | `results/figures/main/flash_nominal_vs_true_safety_margin.png` | same | `primary_evaluation_figures/02_true_vs_nominal_margin.png` |
| Conformal vs true safety-margin geometry | `results/figures/main/flash_conformal_vs_true_safety_margin.png` | same | `primary_evaluation_figures/03_true_vs_conformal_margin.png` |
| CSTR vs flash boundary behaviour | `results/figures/main/cstr_vs_flash_boundary_behaviour.png` | `twinguard.evaluation.boundary_challenge.create_figures` | `boundary_challenge_figures/04_cstr_vs_flash_boundary_behavior.png` |
| Margin ladder: false-safe frequency | `results/figures/appendix/margin_ladder_false_safe_frequency.png` | `twinguard.evaluation.margin_ladder.create_figures` | `flash_margin_ladder/figures/01_margin_vs_false_safe.png` |
| Margin ladder: certification availability | `results/figures/appendix/margin_ladder_certification_availability.png` | same | `figures/02_margin_vs_certification.png` |
| Margin ladder: safe-action certification | `results/figures/appendix/margin_ladder_safe_action_certification.png` | same | `figures/03_margin_vs_safe_action_certification.png` |
| Healthy plant / nominal-twin parity | `results/figures/appendix/flash_healthy_plant_twin_parity.png` | `twinguard.flash.validate_simulator` | `figures/healthy_plant_dt_parity.png` |
| Manipulated-action step responses | `results/figures/appendix/flash_manipulated_action_step_responses.png` | same | `figures/action_step_responses.png` |
| Structural-mismatch diagnostics | `results/figures/appendix/flash_structural_mismatch_diagnostics.png` | `twinguard.evaluation.final_audit.create_figures` | `reliability_figures/02_structural_mismatch_evidence.png` |

The figures in `results/figures/` are the stage outputs listed above (paths relative to the stage's
experiment root); a rerun of the stage redraws them. `python scripts/generate_paper_figures.py`
verifies the released figures against their recorded hashes, and
`python scripts/generate_paper_figures.py --regenerate --workspace runs/figures` redraws the
simulator figures.

## Retrospective comparison baselines (post hoc; not prospective)

These controls were computed after the prospective study, on the same calibration and test
populations. They are not part of the TwinGuard method. The numbers below are at α = 0.05.

| | |
|---|---|
| Combined table | `results/tables/retrospective_baseline_comparison.csv`, built by `scripts/generate_paper_tables.py` |
| Test | `tests/test_retrospective_baselines.py` recomputes each Offset-C summary from its per-episode decisions. It also checks that the reconstructed M0–M3 rows equal the recorded primary results. |

| Process | Control | Output | Code / entry point | Result |
|---|---|---|---|---|
| CSTR | C0, C1 (constant-width one-sided conformal) and reconstructed M0–M3 | `artifacts/retrospective_baselines/cstr_conformal_controls/` | `twinguard.baselines.cstr_conformal_controls` / `experiments/evaluate_cstr_conformal_controls.py` | C0: 0 certified. C1: 194 certified, 0 false-safe. |
| CSTR | Offset-C (persistent offset) | `artifacts/retrospective_baselines/cstr_persistent_offset/` | `twinguard.baselines.cstr_persistent_offset` / `experiments/evaluate_cstr_persistent_offset.py` | 16 / 1000 certified, 0 false-safe, coverage 0.951 |
| Flash | C0, C1 (fixed engineering scales) and reconstructed M0–M3 | `artifacts/retrospective_baselines/flash_conformal_controls/` | `twinguard.baselines.flash_conformal_controls` / `experiments/evaluate_flash_conformal_controls.py` | C0: 0 certified. C1: 49 certified, 1 false-safe. |
| Flash | Offset-C (persistent offset) | `artifacts/retrospective_baselines/flash_persistent_offset/` | `twinguard.baselines.flash_persistent_offset` / `experiments/evaluate_flash_persistent_offset.py` | 207 / 1200 certified, 2 false-safe (both on simulator-domain exits), coverage 0.943 |

For comparison, TwinGuard (M3) certified 338 / 1000 with 0 false-safe on CSTR, and 177 / 1200 with
5 false-safe on flash.

**How the inputs were obtained.** Offset-C uses only information available before the decision:
the observed residual window and the nominal-twin rollout under the candidate action. Two inputs
were not stored by the prospective runs, so they were rebuilt from the episode seeds and cached next
to the outputs:

- **CSTR:** the pre-decision residual histories.
- **Flash:** the calibration futures. Every regenerated input matched the recorded calibration inputs
  exactly.
