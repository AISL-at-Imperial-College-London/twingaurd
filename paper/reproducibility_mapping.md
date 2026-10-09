# Manuscript claim → reproducibility check mapping

**How to run the checks.**

```bash
python reproducibility/run_reproducibility_audit.py              # git checkout
python reproducibility/run_reproducibility_audit.py --archive    # with the frozen per-episode archive extracted
pytest tests/test_seed_separation.py tests/test_prediction_before_truth.py tests/test_certification_rule.py
```

The audit writes `reproducibility/reports/<timestamp>/audit_checks.csv`, with one row per check
(ID, category, records examined, failures, status). The check IDs below are the prefixes used in
that file. Checks marked **[archive]** need the extracted frozen archive. All other checks run on
the git checkout.

## Seed separation

**Claim:** training, model-selection, calibration, test and stress populations are disjoint, and
were reserved before any truth was generated.

| Evidence | Check |
|---|---|
| Pairwise intersection of episode root seeds is empty. This covers 10 flash populations (45 pairs) and 16 CSTR populations (120 pairs), including the margin-ladder blocks. | `seed_separation:flash`, `seed_separation:cstr` |
| Each population's observed seeds equal its declared contiguous block, read from the frozen manifests and designs:<br>• `flash_population_manifest.csv`<br>• `RESS_E6_DESIGN.json`<br>• `RESS_E7_DESIGN.json`<br>• C1 / mixed-degradation `configuration.json`<br>• `E6B_SEED_MANIFEST.csv` | `population_completeness:*` |
| Seed reservation preceded truth generation | The preregistrations `FLASH_E2_PREDECLARATION.md`, `RESS_E7_DESIGN.json` and `E6B_PREDECLARATION.md` are hash-verified (`frozen_manifest:preregistrations, sidecars and design files`). |

## Prediction-before-truth chronology

**Claim:** every certification decision was fixed before the corresponding physical truth was
generated or revealed.

| Evidence | Check |
|---|---|
| Strict order candidate < prediction < decision < truth, from nanosecond timestamps recorded at write time (flash primary, boundary, structural, mixed-regime; CSTR boundary and mixed-regime) | `chronology:*` |
| The per-episode result tables carry exactly the ledger's hashes and timestamps | `table_ledger_agreement:*` |
| CSTR mixed degradation: prediction < decision < truth timestamps | `chronology:cstr mixed degradation` |
| CSTR primary: decision snapshot saved before truth reveal (recorded flag), plus the original read-only reconstruction audit | `chronology:cstr primary`, `chronology:cstr primary (frozen reconstruction audit)` |
| **[archive]** Each truth file stores the SHA-256 of its earlier prediction and decision files, and these match the files | `truth_binding:*` |

## No pre-truth physical safety labels

**Claim:** no evaluator-only information reached the prediction or the decision.

| Evidence | Check |
|---|---|
| Discrepancy-model training/selection inputs and calibration operational inputs contain no fault, severity or truth arrays | `fault_blind_inputs:*` |
| **[archive]** Candidate, prediction and decision files contain no fault, severity or truth keys. The token list is imported from the method code (`FORBIDDEN_OPERATIONAL_TOKENS`). | `pretruth_keys:*` |

## No fault-identity leakage

**Claim:** the deployed method is fault blind.

| Evidence | Check |
|---|---|
| Frozen discrepancy models, bootstrap ensembles and conformal configurations hold no fault fields | `fault_blind_model:*` |
| Discrepancy-model features are action, action change and nominal-twin outputs only (flash), plus the time since detection (CSTR) | `fault_blind_features:flash`, `fault_blind_features:cstr` |
| All fault and severity columns in the evaluation tables are explicitly labelled `_evaluator_only` | `evaluator_labelling:*` |
| In code, faults enter only through plant parameter copies, and the nominal twins never see them | `tests/test_simulators.py` |

## Hash verification

**Claim:** the released files are the frozen prospective outputs, and the released code is the
code that produced them.

| Evidence | Check |
|---|---|
| Every frozen SHA-256 manifest is verified against the released bytes. This covers E4, E5, E6, the final E7 artifacts, the 34,222-entry E7 frozen-input manifest and the margin-ladder manifest. Archived entries are verified with `--archive`. | `frozen_manifest:*` |
| Preregistration sidecars, design-file hashes and C1 source hashes | `frozen_manifest:preregistrations, sidecars and design files` |
| Flash plant + nominal twin + configuration process-bundle tree hash | `frozen_process_bundle` |
| Each released module reverses, through its ledgered edits, to the frozen source bytes. Where a frozen hash was recorded, the reversed source matches it. | `source_provenance_reversal`, `source_matches_frozen_manifests` |
| Every ledger artifact hash appears in the corresponding frozen manifest | `hash_binding:*` |
| **[archive]** Every archived per-episode file matches its recorded SHA-256 | `archive_integrity` |

One documented exception, reported as known rather than as a pass:
`artifacts/flash_study/FLASH_E3_MODEL_SELECTION.md` (a narrative report) differs from its E7
freeze record. See `reproducibility/known_discrepancies.csv`.

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
| **[archive]** The original final-audit reconstruction re-runs on the archived artifacts and reproduces the frozen audit record (`RESS_FINAL_AUDIT_DETAILS.csv`). It covers envelope and decision reconstruction, prediction-hash binding, safety-label recomputation and NaN handling. | `original_final_audit:*` |
| CSTR primary decision reconstruction from stored predictions | `experiments/audit_cstr_primary_decisions.py` [archive]; frozen outputs in `artifacts/cstr/primary_decision_audit/` |

## Prospective re-execution

| Evidence | Command |
|---|---|
| Flash simulator validation regenerates its outputs identically | `python scripts/reproduce_main_results.py --rerun flash-simulator` |
| CSTR certification pipeline runs end to end from the frozen inputs | `python scripts/reproduce_main_results.py --rerun cstr-primary-smoke` |
| Full prospective reruns of any population, compared cell by cell with the frozen summaries | `python scripts/reproduce_main_results.py --rerun <stage>` |

## Retrospective baselines

| Evidence | Check |
|---|---|
| Each Offset-C summary equals the counts recomputed from its per-episode decisions. The calibration populations have the expected sizes (CSTR 500, flash 447 complete). | `tests/test_retrospective_baselines.py` |
| The baseline scripts reconstruct the frozen M0–M3 results exactly before adding the controls | `tests/test_retrospective_baselines.py`; `flash_e4_calibration_regeneration_checks.csv` (447/447 inputs match the frozen E4 inputs) |
| The baseline code is the analysis scripts plus ledgered import and path edits | `source_provenance_reversal` |
