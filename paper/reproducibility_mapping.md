# Manuscript claim → reproducibility check mapping

**How to run the checks.**

```bash
python reproducibility/run_reproducibility_audit.py
pytest tests/test_seed_separation.py tests/test_prediction_before_truth.py tests/test_certification_rule.py
```

The audit writes `reproducibility/reports/<timestamp>/audit_checks.csv`, with one row per check
(ID, category, records examined, failures, status). The check IDs below are the prefixes used in
that file.

## Seed separation

**Claim:** training, model-selection, calibration, test and stress populations are disjoint, and
were reserved before any truth was generated.

| Evidence | Check |
|---|---|
| Pairwise intersection of episode root seeds is empty. This covers 10 flash populations (45 pairs) and 16 CSTR populations (120 pairs), including the margin-ladder blocks. | `seed_separation:flash`, `seed_separation:cstr` |
| Each population's observed seeds equal its declared contiguous block, read from the population manifests and designs:<br>• `flash_population_manifest.csv`<br>• `boundary_challenge_design.json`<br>• `reliability_design.json`<br>• CSTR primary / mixed-degradation `configuration.json`<br>• `margin_ladder_seed_manifest.csv` | `population_completeness:*` |
| Seed reservation preceded truth generation | The preregistrations `population_preregistration.md`, `reliability_design.json` and `margin_ladder_preregistration.md` are hash-verified (`design_sidecars`, `margin_ladder_preregistration_inputs`). |

## Prediction-before-truth chronology

**Claim:** every certification decision was fixed before the corresponding physical truth was
generated or revealed.

| Evidence | Check |
|---|---|
| Strict order candidate < prediction < decision < truth, from nanosecond timestamps recorded at write time (flash primary, boundary, structural, mixed-regime; CSTR boundary and mixed-regime) | `chronology:*` |
| The per-episode result tables carry exactly the ledger's hashes and timestamps | `table_ledger_agreement:*` |
| CSTR mixed degradation: prediction < decision < truth timestamps | `chronology:cstr mixed degradation` |
| CSTR primary: decision snapshot saved before truth reveal (recorded flag), plus the read-only decision reconstruction audit | `chronology:cstr primary`, `chronology:cstr primary (decision reconstruction audit)` |

## No pre-truth physical safety labels

**Claim:** no evaluator-only information reached the prediction or the decision.

| Evidence | Check |
|---|---|
| Discrepancy-model training/selection inputs and calibration operational inputs contain no fault, severity or truth arrays | `fault_blind_inputs:*` |

## No fault-identity leakage

**Claim:** the deployed method is fault blind.

| Evidence | Check |
|---|---|
| Fitted discrepancy models, bootstrap ensembles and conformal configurations hold no fault fields | `fault_blind_model:*` |
| Discrepancy-model features are action, action change and nominal-twin outputs only (flash), plus the time since detection (CSTR) | `fault_blind_features:flash`, `fault_blind_features:cstr` |
| All fault and severity columns in the evaluation tables are explicitly labelled `_evaluator_only` | `evaluator_labelling:*` |
| In code, faults enter only through plant parameter copies, and the nominal twins never see them | `tests/test_simulators.py` |

## Hash verification

**Claim:** the released files are the recorded prospective outputs.

| Evidence | Check |
|---|---|
| Every file under `artifacts/` and `results/` matches its SHA-256 and size in `reproducibility/artifact_manifest.csv`, and no file is unlisted | `artifact_manifest:*` |
| Every entry of the stage SHA-256 manifests (calibration, primary evaluation, boundary challenge, reliability evaluation, reliability-stage inputs, margin ladder) matches the released file | `stage_manifest:*` |
| Preregistration and design sidecars match their documents | `design_sidecars` |
| Every `*_sha256` reference in a design or configuration file equals the hash of a released file | `design_hash_references` |
| Every input hash listed in the margin-ladder preregistration matches the released file | `margin_ladder_preregistration_inputs` |

## Duplicate / missing episode checks

**Claim:** each reported denominator is complete, and no episode is counted twice.

| Evidence | Check |
|---|---|
| Declared seed block fully present, with no duplicates or extras | `population_completeness:*` |
| Every method (and confidence level) is scored on exactly the same episodes, once each | `method_rows:*` |

## Decision reconstruction

**Claim:** every reported count follows from the per-episode decisions, and every decision
follows from the stored prediction.

| Evidence | Check |
|---|---|
| Per-row consistency: certified ⇔ decision string; false-safe ⇔ certified ∧ physically unsafe; nominal-twin false-safe likewise | `decision_consistency:*` |
| Every row of every summary table is recomputed from per-episode rows, matching N, certified, false-safe and actual-unsafe counts. Coverage is also recomputed. This covers the flash ablation (pooled and all subgroups), boundary, structural, mixed-regime, CSTR primary (all methods, levels and regimes), mixed degradation and margin ladder. | `summary:*` |
| Frozen conformal quantiles equal the finite-sample order statistic of their own calibration scores (flash ordinary and boundary; CSTR one-sided) | `tests/test_certification_rule.py` |
| The final audit of the reliability stage reconstructed envelopes and decisions from the per-episode files, with prediction-hash binding, safety-label recomputation and NaN handling; its record is `final_audit_checks.csv` | `twinguard.evaluation.final_audit` |
| CSTR primary decision reconstruction from stored predictions | `experiments/audit_cstr_primary_decisions.py`; outputs in `artifacts/cstr/primary_decision_audit/` |

## Prospective re-execution

| Evidence | Command |
|---|---|
| Flash simulator validation regenerates its outputs identically | `python scripts/reproduce_main_results.py --rerun flash-simulator` |
| CSTR certification pipeline runs end to end from the released inputs | `python scripts/reproduce_main_results.py --rerun cstr-primary-smoke` |
| Full prospective reruns of any population, compared cell by cell with the released summaries | `python scripts/reproduce_main_results.py --rerun <stage>` |

## Retrospective baselines

| Evidence | Check |
|---|---|
| Each Offset-C summary equals the counts recomputed from its per-episode decisions. The calibration populations have the expected sizes (CSTR 500, flash 447 complete). | `tests/test_retrospective_baselines.py` |
| The baseline scripts reconstruct the M0–M3 results exactly before adding the controls | `tests/test_retrospective_baselines.py`; `flash_calibration_regeneration_checks.csv` (447/447 inputs match the recorded calibration inputs) |
