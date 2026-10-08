# Manuscript result → file → script mapping

Table numbers follow the numbering used when this repository was prepared (Table 5 = flash
ablation, Table 6 = primary results, Table 7 = mixed-regime audit). **Confirm them against the
final manuscript before release.**

All paths are relative to the repository root. Throughout this document:

- **"Frozen source"** is the file written by the frozen prospective run. It is byte-exact and
  hash-verified by `reproducibility/run_reproducibility_audit.py`.
- **"Paper table"** is the file in `results/tables/`, derived from the frozen source by
  `scripts/generate_paper_tables.py` (row selection and relabelling only).
- **`paper_headline_results.csv`** collects every number quoted in the text, one row per
  experiment and method. Its last two columns name the exact frozen source file and row.

## Tables

### Table 5: Flash internal component ablation (M0 → M1 → M2 → M3)

| | |
|---|---|
| Paper table | `results/tables/flash_component_ablation.csv` (pooled rows: `group_type=pooled`; subgroups by fault family, severity, action distance, initial region and active constraint) |
| Frozen source | `artifacts/flash_study/FLASH_FINAL_ABLATION.csv`, lines 2–5 (`group_type=pooled`, methods `M0_nominal_DT`, `M1_discrepancy_mean`, `M2_raw_bootstrap`, `M3_pooled_conformal`) |
| Per-episode evidence | `artifacts/flash_study/FLASH_FINAL_RESULTS.csv` (1200 episodes × methods) and ledger `artifacts/flash_study/e5_artifact_audit.csv` |
| Script | `experiments/evaluate_flash_primary.py` → `twinguard.flash.evaluate_primary` (`evaluate_artifacts`, `aggregate`, `subgroup_rows`) |
| Values | M0: 779 certified / 207 false-safe (17.25%). M1: 672 / 103 (8.58%). M2: 623 / 72 (6.00%). M3: 177 / 5 (0.42%), availability 14.75%, coverage 0.9399 |

The frozen source also has a fifth row, `M4_action_conditioned`. It is a hard-coded
"NOT CERTIFIED" placeholder for an unsupported action-conditioned calibration, and it is omitted
from the paper table.

### Table 6: Primary prospective results (CSTR and flash)

**CSTR (N = 1000).**

| | |
|---|---|
| Paper table | `results/tables/cstr_primary_results.csv`; headline rows in `paper_headline_results.csv` |
| Frozen source | `artifacts/cstr/primary_evaluation/safety_summary.csv`:<br>• line 45 (`S3_one_sided, 0.95, pooled`): 338 certified (33.8%), 0 false-safe, upper whole-trajectory coverage 0.957<br>• line 9 (`S0_nominal_DT, 0.95, pooled`): nominal twin, 394 false-safe |
| Per-episode evidence | `artifacts/cstr/primary_evaluation/final_safety_episodes.csv`; reconstruction audit `artifacts/cstr/primary_decision_audit/` |
| Calibration | `artifacts/cstr/primary_evaluation/one_sided_quantiles.csv` (q⁺ = 5.464557586938352 at 95%, n = 500) |
| Script | `experiments/evaluate_cstr_primary.py` → `twinguard.cstr.safety_certification` |

**Flash (N = 1200).**

| | |
|---|---|
| Frozen source | `artifacts/flash_study/FLASH_FINAL_ABLATION.csv`, line 5 (M3): 177 certified (14.75%), 5 false-safe (joint 0.42%), coverage 0.9399 |
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
| Frozen source | `artifacts/flash_study/CSTR_RELIABILITY_SUMMARY.csv`, row `frozen_CSTR_conformal`: 1010 certified, 0 false-safe |
| Per-episode evidence | `CSTR_RELIABILITY_RESULTS.csv`, ledger `e7_artifacts/CSTR_RELIABILITY_ARTIFACT_AUDIT.csv` |

**Flash (N = 3000).** The population has F1/F2/F3 at 1000 each, with 1500 ordinary-action and
1500 boundary-action episodes.

| | |
|---|---|
| Frozen source | `artifacts/flash_study/FLASH_RELIABILITY_SUMMARY.csv`, row `M3_pooled_conformal`: 237 certified, 6 false-safe |
| Per-episode evidence | `FLASH_RELIABILITY_RESULTS.csv`, ledger `e7_artifacts/FLASH_RELIABILITY_ARTIFACT_AUDIT.csv` |

**Both processes.**

| | |
|---|---|
| Script | `experiments/evaluate_structural_mismatch_and_mixed_regime_audits.py` → `twinguard.evaluation.structural_and_reliability` (one frozen run; separate denominators) |
| Design | `artifacts/flash_study/RESS_E7_DESIGN.json` (+ `.sha256`), frozen before any truth |

## Other reported results

### CSTR mixed-degradation / fault-blind transfer (heat-transfer fouling and feed-temperature shift; N = 600)

| | |
|---|---|
| Paper tables | `results/tables/cstr_mixed_degradation_results.csv`, `..._by_fault.csv`, `..._by_severity.csv` |
| Frozen source | `artifacts/cstr/mixed_degradation/MULTIFAULT_SUMMARY.csv`:<br>• line 4 (`nominal_DT`): 265 false-safe<br>• line 5 (`proposed_method`): 141 certified (23.5%), 0 false-safe, coverage 0.9483 |
| Per-episode evidence | `artifacts/cstr/mixed_degradation/MULTIFAULT_FINAL_EPISODES.csv` |
| Script | `experiments/evaluate_cstr_mixed_degradation.py` → `twinguard.cstr.mixed_degradation`; fault definition `experiments/check_cstr_feed_temperature_fault.py` |
| Note | Each episode carries one hidden mechanism drawn from the mixture. Faults are not concurrent. |

### Boundary-region challenge

| | |
|---|---|
| Paper table | `results/tables/boundary_risk_availability.csv` |
| Frozen source | `artifacts/flash_study/RISK_AVAILABILITY_RESULTS.csv` |
| Flash (N = 1200) | • line 2 (`nominal_DT`): 853 unsafe, 711 certified, 447 false-safe (37.25%)<br>• lines 3–7 (`pooled_conformal`, levels 0.90–0.99): 0 certified |
| CSTR (N = 300) | • line 8 (`nominal_DT`): 300 unsafe, all 300 certified<br>• line 9 (`frozen_CSTR_conformal`): 0 certified |
| Per-episode evidence | `FLASH_BOUNDARY_RESULTS.csv`, `CSTR_BOUNDARY_RESULTS.csv`, ledgers in `e6_artifacts/` |
| Script | `experiments/evaluate_boundary_challenge.py` → `twinguard.evaluation.boundary_challenge` |
| Calibration | flash boundary quantiles from the separate boundary-calibration population (`artifacts/flash_study/flash_conformal_config.json`, `calibration_boundary`; `RESS_E6_DESIGN.json`) |

The 853 unsafe flash episodes comprise 587 complete-and-unsafe trajectories and 266 domain exits.

### Flash structural mismatch (unmodelled vapour-valve actuator lag; reserved N = 600)

| | |
|---|---|
| Paper table | `results/tables/flash_structural_mismatch_results.csv` |
| Frozen source | `artifacts/flash_study/FLASH_STRUCTURAL_SUMMARY.csv`, row `M3_pooled_conformal`: 43 certified, 0 false-safe |
| Completeness | 564 complete / 36 incomplete, counted from `FLASH_STRUCTURAL_RESULTS.csv` (`truth_complete`); see `paper_headline_results.csv` |
| Script | `experiments/evaluate_structural_mismatch_and_mixed_regime_audits.py`; plant `twinguard.plants.flash_drum_valve_lag` |
| Note | Descriptive stress test. The frozen ordinary/boundary quantiles are reused. There is no structural-fault calibration claim. |

### Margin-ladder stress analysis

| | |
|---|---|
| Paper table | `results/tables/margin_ladder_risk_availability.csv` |
| Frozen sources | `artifacts/flash_margin_ladder/E6B_RISK_AVAILABILITY.csv`, `FLASH_E6B_MARGIN_RESULTS.csv`, `CSTR_E6B_MARGIN_RESULTS.csv`, preregistration `E6B_PREDECLARATION.md` (+ `.sha256`) |
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

**CSTR.** Each stage's frozen outputs, with the script that produces them:

| Stage | Frozen outputs | Script |
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

| Figure | File | Drawn by | Frozen source |
|---|---|---|---|
| Representative corrected flash counterfactual rollout | `results/figures/main/flash_representative_corrected_rollout.png` | `twinguard.flash.evaluate_primary.create_figures` | `artifacts/flash_study/e5_figures/01_representative_corrected_rollout.png` (archive) |
| Nominal-twin vs true safety-margin geometry | `results/figures/main/flash_nominal_vs_true_safety_margin.png` | same | `e5_figures/02_true_vs_nominal_margin.png` |
| Conformal vs true safety-margin geometry | `results/figures/main/flash_conformal_vs_true_safety_margin.png` | same | `e5_figures/03_true_vs_conformal_margin.png` |
| CSTR vs flash boundary behaviour | `results/figures/main/cstr_vs_flash_boundary_behaviour.png` | `twinguard.evaluation.boundary_challenge.create_figures` | `e6_figures/04_cstr_vs_flash_boundary_behavior.png` |
| Margin ladder: false-safe frequency | `results/figures/appendix/margin_ladder_false_safe_frequency.png` | `twinguard.evaluation.margin_ladder.create_figures` | `flash_margin_ladder/figures/01_margin_vs_false_safe.png` |
| Margin ladder: certification availability | `results/figures/appendix/margin_ladder_certification_availability.png` | same | `figures/02_margin_vs_certification.png` |
| Margin ladder: safe-action certification | `results/figures/appendix/margin_ladder_safe_action_certification.png` | same | `figures/03_margin_vs_safe_action_certification.png` |
| Healthy plant / nominal-twin parity | `results/figures/appendix/flash_healthy_plant_twin_parity.png` | `twinguard.flash.validate_simulator` | `figures/healthy_plant_dt_parity.png` |
| Manipulated-action step responses | `results/figures/appendix/flash_manipulated_action_step_responses.png` | same | `figures/action_step_responses.png` |
| Structural-mismatch diagnostics | `results/figures/appendix/flash_structural_mismatch_diagnostics.png` | `twinguard.evaluation.final_audit.create_figures` | `e7_figures/02_structural_mismatch_evidence.png` |

The figures in `results/figures/` are byte-identical to the frozen figures
(`python scripts/generate_paper_figures.py`). To redraw them with the original plotting code, run
`python scripts/generate_paper_figures.py --regenerate --workspace runs/figures`.
