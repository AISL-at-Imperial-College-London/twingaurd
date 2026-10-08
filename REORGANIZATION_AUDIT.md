# Reorganization audit (development repository → TwinGuard release)

This audit was written **before** any file was copied, renamed or edited. It records what the
development repository contains, what each component actually does (established by reading the
code and the data, not by filename), how the final RESS experiments depend on each other, and the
integrity problems found. The development repository itself was not modified.

Audit date: 2026-10-07. Development repository state: branch `main`, last commit `a8d166f`; the
directories `twingaurd/`, `E_FLASH_RESS/`, `E6B_MARGIN_LADDER/` and `prompts/` are untracked.

---

## 1. Repository facts

| Quantity | Value |
|---|---|
| Working-tree files (excluding `.git`) | 69,058 |
| Working-tree size (excluding `.git`) | 908.7 MB |
| `.git` object store | 354 MB |
| Total on disk | ≈ 1.26 GB |
| Python source files | 86 (≈ 31,100 lines) |
| `.npz` per-episode artifacts | 40,448 files, 571 MB |
| `.json` files (mostly per-episode decision/candidate records) | 27,801 files, 15 MB |
| `.csv` files | 186 files, 272 MB |

Files above the size thresholds:

| Threshold | Files |
|---|---|
| > 100 MB | `results/milestone_b6/stability_samples.csv` (143.2 MB; already git-ignored in the development repo) |
| > 50 MB | the same file only |
| > 10 MB | the file above, plus `results/milestone_b7a/innovation_samples.csv` (19.5 MB), `E_FLASH_RESS/FLASH_RELIABILITY_RESULTS.csv` (12.8 MB) and `results/milestone_b/one_step_row_results.csv` (10.4 MB) |

There are no single files that GitHub would reject, but the repository is dominated by
≈ 70,000 small per-episode artifact files in these trees:

| Tree | Files | Size |
|---|---|---|
| `E_FLASH_RESS/e7_artifacts` | 26,403 | 175.8 MB |
| `E6B_MARGIN_LADDER/artifacts` | 17,499 | 118.8 MB |
| `results/milestone_c1` | 5,237 | 67.5 MB |
| `results/milestone_b12` | 2,338 | 58.4 MB |
| `E_FLASH_RESS/e6_artifacts` | 6,002 | 36.9 MB |
| `results/milestone_b11` | 813 | 31.2 MB |
| `E_FLASH_RESS/e5_artifacts` | 3,600 | 29.2 MB |
| `results/milestone_b5` | 635 | 29.1 MB |

The per-file decisions are in `LARGE_FILE_DECISIONS.md`.

## 2. Method used

1. A full recursive inventory was taken: path, size and extension for all 69,058 files.
2. Four independent read-only code audits were run:
   - the flash core library
   - the flash E1–E5 runners and their outputs
   - the cross-case E6/E7, E6B and final audit
   - the CSTR package
3. Every local import was traced, and every path literal and every hash-verification site was
   located.
4. Each reference result in the release brief was located in the frozen CSVs and re-checked
   against the row that contains it.
5. Every frozen SHA-256 recorded in the hash manifests and sidecars was recomputed for the source
   files and design documents.
6. The development test suite was run read-only: `PYTHONDONTWRITEBYTECODE=1`, no pytest cache, and
   outputs only to `tmp_path`.

## 3. Development naming patterns found

| Pattern | Where | What it actually denotes (from the code) |
|---|---|---|
| `milestone_a`, `a5`, `b`, `b5`…`b12` | `twingaurd/*.py`, `results/milestone_*` | sequential CSTR development stages. b9–b12 are the final discrepancy → bootstrap → conformal chain; a, a5, b, b5-analysis, b7a and b8 are abandoned or superseded model studies |
| `milestone_c1`, `c1_5a`, `c1_5c`, `c1_6a`, `c1_6b` | same | C1 = primary CSTR certification; C1.5A = read-only reconstruction audit of C1; C1.5C = multi-action screening; C1.6A = feed-temperature fault definition; C1.6B = mixed-degradation transfer |
| `milestone_d1`, `d2`, `second_case_*`, `two_tank_process` | same | two-tank second process, superseded by the flash drum (`prompts/milestone_E0.txt`) |
| `E1`…`E7` (`run_flash_e*.py`, `run_ress_e*.py`, `FLASH_E*_*.md`, `RESS_E*_*`) | `E_FLASH_RESS/` | E1 simulator validation, E2 preregistration, E3 discrepancy model selection, E4 bootstrap + calibration, E5 primary evaluation, E6 boundary challenge, E7 structural + mixed-regime reliability |
| `E6B` | `E6B_MARGIN_LADDER/` | margin-ladder stress analysis (post-E6, separately preregistered) |
| `M0`…`M4` (flash), `M5` (E6B) | `run_flash_e5.py`, `run_ress_e7.py`, `run_e6b.py` | M0 nominal DT, M1 discrepancy mean, M2 raw bootstrap, M3 pooled conformal (the method), M4 action-conditioned (**never supported: hard-coded NOT CERTIFIED**), M5 margin-band-conditioned (E6B analysis B) |
| `M1`…`M4` (CSTR) | `milestone_b_residual_dynamics.py`, `milestone_b5_*` | **different meaning**: Milestone-B Bayesian residual-dynamics model variants. `FROZEN_MODEL_NAME = "M4_full_bayesian"` is a residual model, not an action-conditioned method |
| `S0`…`S3` | `milestone_c1_safety_certification.py` | S0 nominal DT, S1 raw U3 band, S2 symmetric B12 conformal, S3 one-sided conformal (the method) |
| `_smoke`, `_debug`, `shard0..3` | `results/` | smoke/debug runs and parallel shards. The c15c shard directories are listed in the frozen manifests but absent on disk |
| `*_stdout.log`, `*_stderr.log` | `results/` | raw console dumps |
| `FINAL_`, `RESS_FINAL_`, `cstr_final` | various | final summary/closure packages. `results/cstr_final/` has no generating script; it is an agent-authored closure package whose numbers cite source CSVs |
| `prompts/milestone_*.txt` | `prompts/` | coding-agent task instructions (development only) |

No `v2`/`v3`, `old`, `backup`, `temp`, `final_final` or `test2` files exist.

## 4. What each component does

### 4.1 CSTR (`twingaurd/`)

| Original module | What it does (verified in code) | Status |
|---|---|---|
| `cstr_digital_twin.py` | CSTR simulator. The same class is used as the **physical plant** (with a hidden `FaultConfig`) and as the **nominal digital twin** (fault disabled). `FaultManager` applies UA fouling and (added after the last commit, inert by default) a feed-temperature step to the plant only. | final |
| `latent_discrepancy_state_space.py` | scalar/diagonal linear-Gaussian latent discrepancy model: Kalman filter, EM (batched checkpoints), predictive rollout. The diagonal-EM function was added after the last commit and is purely additive. | final |
| `bayesian_residual_dynamics.py`, `counterfactual_rollout.py`, `probabilistic_residual_model.py`, `stable_arx_residual_dynamics.py` | earlier residual models | kept only because the final chain imports them at module level (via b5 and b7b) |
| `milestone_b5_posterior_calibration.py` | defines the CSTR episode configuration (`MilestoneB5Config`: noise, N=50, H=240, UA-fouling range), episode template, hidden-fault configuration, action-regime selection and Wilson interval. Its own `run()` calibrates the obsolete Milestone-B model. | library functions final; `run()` obsolete |
| `milestone_b6_recursive_instability.py` | instability study. Only `load_b5_episodes` is used downstream (by b9/b10). | helper only |
| `milestone_b7b_stable_higher_order_arx.py` | ARX order study. `_init_worker` and `_reconstruct_observations` are used by b9/b10. | helpers only |
| `milestone_b9_latent_discrepancy.py` | generates the CSTR training population, fits latent dimensions by EM and selects k | **final: CSTR discrepancy fitting** |
| `milestone_b10_latent_uncertainty.py` | freezes the latent model; `EpisodeInputs`; fresh test-input generation; variance diagnostics | **final: model freeze** |
| `milestone_b11_parameter_uncertainty.py` | whole-episode bootstrap of the latent model (J=50, seeds 1,210,000+) | **final: CSTR bootstrap** |
| `milestone_b12_conformal_trajectories.py` | episode-level whole-trajectory conformal calibration. B2 is pooled; B2M is a Mondrian diagnostic only. | **final: CSTR conformal** |
| `milestone_c1_safety_certification.py` | one-sided conformal certification: calibration n=500 (seeds 1,630,000+), prospective test N=1000 (seeds 1,730,000+); decision snapshot written before truth reveal. Provides `predict_frozen_u3`, `reveal_physical_temperature` and `_reconstruct_decision_state`, reused by E6/E6B/E7. | **final: CSTR primary** |
| `milestone_c1_5a_frozen_audit.py` | read-only reconstruction of all C1 decisions and the interpolation-regime diagnosis | final-supporting |
| `milestone_c1_6a_additional_faults.py` | feed-temperature-step fault definition and smoke test; `_clean_fault` is used by c16b | final-supporting |
| `milestone_c1_6b_multifault_revalidation.py` | mixed-degradation transfer. Each episode has **either** UA fouling **or** a feed-temperature step (not concurrent faults). Retrains k=1, bootstraps 50, calibrates (n=400) and tests prospectively (N=600; seeds 2.1M/2.2M/2.3M/2.4M). | **final: CSTR mixed degradation** |
| `milestone_c1_5c_multi_action_screening.py`, `aggregate_milestone_c1_5c.py` | K-candidate multi-action screening with a rounded q+ = 5.465 (frozen value 5.464557586938352). The aggregator runs at import time and needs shard directories that are absent. | not among the requested paper experiments; **manual review** |
| `milestone_d1_*`, `milestone_d2_*`, `two_tank_process.py` | two-tank second process, superseded by the flash drum. `audit_ress_final.py` imports only `SCALE_FLOOR_M` from d2 and audits `second_case_final`. | **manual review** (see §9) |
| `milestone_a*`, `milestone_b_residual_dynamics.py`, `milestone_b5_calibration_analysis.py`, `milestone_b7a_*`, `milestone_b8_*`, `observable_conditioned_arx.py` | abandoned or superseded residual-model studies | obsolete |
| `cstr_case.py` | LLM-agent control case (langgraph, langchain-openai, ollama, dotenv). Imports a missing `common.trace_logging` module and writes traces outside the repository. No final code imports it. | unrelated |
| `cstr_anomaly_threshold_test.py` | temperature-threshold detector used only by `cstr_case.py`. Not a test. | unrelated |

### 4.2 Flash drum (`E_FLASH_RESS/`)

| Original file | What it does | Status |
|---|---|---|
| `flash_process.py` | benzene/toluene dynamic flash model (3 states, Raoult VLE, NIST Antoine). `FlashPlant` is the physical plant; faults enter only through a modified `FlashParameters` copy. | final |
| `flash_digital_twin.py` | independent nominal twin with its own parameter copy | final |
| `flash_population.py` | `purpose_seed`; fault generator F1–F3; `EvaluatorFaultTruth` (evaluator-only); fault-blind `OperationalEpisodeInput` with a forbidden-token guard; `SafetySpecification`; LHS and boundary action designs | final |
| `flash_discrepancy_model.py` | latent linear-Gaussian discrepancy state space over [P, level, T]. Features are action, Δaction and nominal-twin outputs only. | final |
| `flash_bootstrap.py` | coherent bootstrap predictive summary (J models, 2000 paths) | final |
| `flash_conformal.py` | whole-trajectory directional score, finite-sample corrected quantile, envelope, SAFE / NOT CERTIFIED rule | final |
| `flash_structural_mismatch.py` | plant-only unmodelled first-order vapour-valve lag (structural mismatch) | final |
| `frozen_boundaries.py` | output-isolation guard. All flash writes must stay inside `E_FLASH_RESS/` and outside frozen CSTR results. | final infrastructure |
| `run_flash_e1.py` | healthy plant/twin parity (difference 0.0), 12 healthy cases, 7200 s convergence, four manipulated-action step responses (directional checks), VLE residuals. Seed 1601. | final |
| `write_flash_e2_preregistration.py` | reserves 8 disjoint seed blocks (`flash_population_manifest.csv`, `flash_seed_manifest.csv` with 17,280 rows) and writes the preregistration sidecar. `FLASH_E2_PREDECLARATION.md` itself is hand-written. | final |
| `run_flash_e3.py` | generates training (900) and model-selection (180) populations, fits EM k=1..3 and applies the preregistered gates. No candidate passed, so the preregistered fallback k=3 was used. | final |
| `run_flash_e4.py` | bootstrap (100 accepted of 150 attempts, seeds 31,000,000+), ensemble-size selection J=10 on operational selection inputs only, and ordinary (24M seeds) plus boundary (25M seeds) conformal calibration | final |
| `run_flash_e5.py` | prospective primary evaluation, N=1200 (26M seeds), M0–M3 ablation plus the M4 placeholder; exclusive-create chronology writes | final |
| `run_ress_e6.py` | boundary challenge: flash N=1200 (27M seeds) and CSTR N=300 (9.3M seeds). CSTR episodes are regenerated from seeds using the frozen C1 predictor. | final |
| `run_ress_e7.py` | structural mismatch, N=600 (28M seeds, vapour-valve lag); mixed-regime reliability audits for flash, N=3000 (29M seeds), and CSTR, N=3000 (50M seeds) | final |
| `freeze_ress_e7.py` | wrote the E7 design and the 34,222-entry frozen-input hash manifest | final (provenance) |
| `audit_ress_final.py` | final cross-experiment audit; writes `RESS_FINAL_*` into `E_FLASH_RESS/` | final (audit) |
| `FROZEN_INPUT_HASHES.txt`, `IMPLEMENTATION_PLAN.md`, `FLASH_PROCESS_SPECIFICATION.md` | E0/E1 planning and specification documents; no generating script | provenance / planning |

### 4.3 Margin ladder (`E6B_MARGIN_LADDER/`)

| File | Description |
|---|---|
| `write_preregistration.py` | Reserves seeds for three populations: flash calibration 70M (1500), flash test 71M (2000) and CSTR test 72M (1500). Also defines the nominal-margin bands B0–B4. |
| `run_e6b.py` | Runs the margin-ladder evaluation and has these stages: `calibration`, `flash-test`, `cstr-test`, `finalize`. Methods are M0–M3 plus **M5 (margin-band-conditioned)**; there is no M4. **Verdict: "E6B METHOD REMAINS OVERLY CONSERVATIVE".** |

## 5. Dependency closure of the final experiments

```
CSTR library: cstr_digital_twin ─┬─ latent_discrepancy_state_space
                                 └─ (b5 → bayesian_residual_dynamics, counterfactual_rollout → probabilistic_residual_model)
b9 (fit) ← b5, b6, b7b(→ stable_arx)          b10 (freeze) ← b5, b6, b7b, b9
b11 (bootstrap) ← b5, b10                     b12 (conformal) ← b5, b10, b11
c1 (primary) ← b5, b10, b11, b12              c1_5a (audit) ← none (reads files)
c16b (mixed degradation) ← b5, b10, b11, b12, c1, c16a, cstr_digital_twin, latent model

Flash: flash_process ← flash_digital_twin ← flash_population ← flash_discrepancy_model ← flash_bootstrap ← flash_conformal
E1 ← process, twin       E2 ← population       E3 ← twin, discrepancy, population, process
E4 ← bootstrap, conformal, twin, discrepancy, population, process
E5 ← E4 objects + all library modules
E6 ← E5, E4._boundary_action, library, CSTR(b5, b10, c1)
E7 ← E5, E6, E4._boundary_action, structural mismatch, library, CSTR(b5, b10, c1)
E6B ← E5, E6, E4.batch_nominal_dt_margins, library, CSTR(b5, b10, c1)
final audit ← E5, E6, E7, c1, d2 (SCALE_FLOOR_M only)
```

Fifteen CSTR modules form the closure required by C1. b6, b7b and b9 are pulled in only because
b10 imports them at module level. Nothing in the closure imports `cstr_case.py`, the A/B/B7a/B8
studies, `observable_conditioned_arx.py` or C1.5C.

## 6. Final paper experiments: code, frozen output, and verification of reference values

Every reference value in the release brief was found in a frozen file and matched exactly.

| Experiment | Code (original) | Frozen evidence (original path : row) | Reference check |
|---|---|---|---|
| CSTR primary | `milestone_c1_safety_certification.py` | `results/milestone_c1/safety_summary.csv`, line 45 (`S3_one_sided, 0.95, pooled`) | N=1000, SAFE 338 (0.338), false-safe 0, upper whole-trajectory coverage 0.957. **Exact.** Nominal S0 false-safe 394 (line 9). |
| CSTR mixed degradation | `milestone_c1_6b_multifault_revalidation.py` (+ `c1_6a`) | `results/milestone_c16b/MULTIFAULT_SUMMARY.csv`, lines 4–5 | N=600; nominal false-safe 265; proposed false-safe 0, SAFE 141 (0.235), coverage 0.948333. **Exact.** |
| CSTR discrepancy fitting/calibration | b9 → b10 → b11 → b12 → C1 calibration | `milestone_b9/model_parameters.json`, `b10/frozen_latent_model.json`, `b11/bootstrap_models.json`, `b12/conformal_quantiles.csv`, `c1/one_sided_quantiles.csv` (q+ = 5.464557586938352) | consumed unchanged by C1/E6/E6B/E7 |
| Flash simulator + nominal DT parity, step responses | `run_flash_e1.py` | `e1_validation/*`, `FLASH_E1_VALIDATION.md` | go=true, parity difference 0.0, 4/4 directional responses |
| Flash training + model selection | `run_flash_e3.py` | `flash_model_selection.csv`, `selected_flash_model.json` | k=3 by preregistered fallback (no candidate passed all gates) |
| Flash bootstrap, ordinary and boundary calibration | `run_flash_e4.py` | `flash_bootstrap_models.json`, `flash_conformal_config.json`, `flash_calibration_scores.csv` | J=10; q₀.₀₅: ordinary 11.048599 (n=447), boundary 11.349921 (n=463) |
| Flash primary (M3) | `run_flash_e5.py` | `FLASH_FINAL_ABLATION.csv`, line 5 | N=1200, certified 177 (0.1475), false-safe 5, joint 0.0041667, coverage 0.939933. **Exact.** |
| Flash ablation M0→M3 | `run_flash_e5.py` | `FLASH_FINAL_ABLATION.csv`, lines 2–5 | 779/207, 672/103, 623/72, 177/5. **Exact.** Independently re-derived from the 6,000 per-episode rows. |
| Flash boundary | `run_ress_e6.py` | `RISK_AVAILABILITY_RESULTS.csv`, rows 2–7 | N=1200, unsafe 853, nominal certified 711, nominal false-safe 447 (0.3725), conformal certified 0 at every level. **Exact.** |
| CSTR boundary | `run_ress_e6.py` | `RISK_AVAILABILITY_RESULTS.csv`, rows 8–9 | N=300, 300 unsafe, nominal certified 300, conformal certified 0. **Exact.** |
| Margin ladder | `E6B_MARGIN_LADDER/run_e6b.py` | `E6B_RISK_AVAILABILITY.csv`, `FLASH_E6B_*`, `CSTR_E6B_*` | M3 certified 0 in B0–B3 and 1/400 in B4; M5 certified 0 |
| Flash structural mismatch | `run_ress_e7.py` | `FLASH_STRUCTURAL_SUMMARY.csv`, row 5 (M3) | reserved 600, 564 complete, 36 incomplete, certified 43, 0 false-safe. **Exact.** |
| Mixed-regime audit, CSTR | `run_ress_e7.py` | `CSTR_RELIABILITY_SUMMARY.csv`, row 2 | N=3000, certified 1010, false-safe 0. **Exact.** |
| Mixed-regime audit, flash | `run_ress_e7.py` | `FLASH_RELIABILITY_SUMMARY.csv`, row 5 (M3) | N=3000, certified 237, false-safe 6. **Exact.** |
| Final audit | `audit_ress_final.py` | `RESS_FINAL_AUDIT_DETAILS.csv` | 66 checks: 64 PASS, 2 ISSUE (documented NaN→0 conversions in E5 and E6) |

Points for the manuscript text, found during verification. These are not release changes.

- **Flash primary coverage denominator.** The 94.0% coverage uses the 899 episodes where both the
  prediction and the truth horizon are complete, not all 1200.
- **Flash primary unsafe count.** 617 of 1200 episodes are unsafe, and 292 of those count as unsafe
  only because the simulation left its domain before the horizon.
- **Flash boundary unsafe count.** The 853 unsafe episodes are 587 complete-and-unsafe plus 266
  domain exits. Of the 447 nominal false-safes, 160 are on incomplete truths, and 158 of those
  violated a limit before the exit.
- **CSTR boundary band.** Only 132 of the 300 CSTR boundary episodes fall within the 0.25 K
  nominal-margin band.
- **"Mixed degradation" wording.** In CSTR mixed degradation each episode carries a *single* fault
  mechanism drawn from a mixture. Describe it as a mixture of fault mechanisms, not as combined
  faults.
- **"Mixed-regime" naming.** No code uses this phrase. It corresponds to the E7 large-reliability
  populations:
  - CSTR: 1000 episodes each in the interpolation, boundary and extrapolation regimes.
  - Flash: 1500 ordinary-action and 1500 boundary-action episodes, with F1–F3 at 1000 each.
- **Flash reliability seed reservation.** E2 reserved 12,000 seeds for the flash reliability
  population. E7 used the first 3000.
- **F4 naming.** The E2 manifest names F4 `nonideal_vle_structural_stress`. E7 actually used
  `F4_unmodelled_vapor_valve_lag`. The E7 design documents this choice.

## 7. Fault-blindness and separation (verified in code)

- **Fault-blind method code.** The deployable code never receives fault family, severity, changed
  parameters or future truth as arguments: flash discrepancy fit/predict, bootstrap summary,
  conformal envelope and decision rule, and the CSTR `predict_frozen_u3` path.
- **Plant-side faults.** Faults enter only through plant parameter copies:
  - `faulted_plant_parameters`
  - CSTR `FaultConfig` on the plant instance
  - `VaporValveLagPlant` for structural mismatch
- **Fault-blind inputs.** `OperationalEpisodeInput` rejects forbidden field tokens (fault,
  severity, truth, …). E5, E6 and E7 scan pre-truth artifacts for forbidden keys, and the final
  audit repeats that scan.
- **Truth stored last.** Evaluator truth is stored only in truth files, written after the
  prediction and decision. The truth file binds the hashes of the prediction and the decision.
- **Two caveats** (not leaks):
  - `VaporValveLagPlant.step()` returns the effective valve opening in its output dict.
  - The F3 sign is derived from seed parity.

  Neither value reaches method code paths.

The release must keep this separation. Plants, twins, scenario/fault generation, discrepancy
models, bootstrap, conformal/certification and evaluation therefore stay in separate modules.

## 8. Integrity problems in the development repository

Baseline test run on the development repository: **175 passed, 9 failed**. Environment: Python
3.13.1, numpy 2.1.3, scipy 1.17.1, matplotlib 3.10.9, pytest 9.0.3. All 9 failures trace to the
first four causes below. None indicates a scientific change.

1. **CRLF conversion after freezing.** `core.autocrlf=true` and the E-series directories are
   untracked, so text files now carry CRLF line endings. Every frozen SHA-256 failure was
   re-checked, and all reproduce exactly after CR stripping:
   - the E2 preregistration, the E7 design and the E6 design
   - the E7 frozen-input manifest
   - the process-bundle sources and the E4 `.py` entries
   - `run_flash_e5.py` and **all 1,200 E5 decision JSONs**

   E6B files were written with CRLF originally and match as-is.

   **Release action:** restore LF where the recorded hash proves LF was the frozen form, keep raw
   bytes everywhere else, and mark data files `-text` in `.gitattributes`.
2. **Absolute machine paths.** About 53,500 absolute paths from a collaborator's machine
   (Windows user-profile paths inside a cloud-synchronised folder) are embedded in nine per-episode
   ledger CSVs:
   - `e5_artifact_audit.csv`
   - the E6 and E7 `*_ARTIFACT_AUDIT.csv`
   - the E6B `*_ARTIFACT_AUDIT.csv`

   Matches of the username string inside hash manifests are coincidental hex substrings, not
   paths.

   **Release action:** rewrite the path columns to repository-relative paths, and record the
   original and sanitized SHA-256 of each file.
3. **Import-only edits after freezing.** On 2026-10-07, `run_ress_e6.py`, `run_ress_e7.py`,
   `run_e6b.py` and `audit_ress_final.py` were edited to `from twingaurd import …`. Reverting the
   import lines (and CR) reproduces the recorded hashes of `run_ress_e6.py`, `run_ress_e7.py` and
   `run_e6b.py`; no hash was recorded for `audit_ress_final.py`. The CSTR modules were moved into
   `twingaurd/` with relative imports. The only substantive post-commit code changes are the two
   additive changes noted in §4.1.
4. **Missing frozen inputs.** About 9,300 of the 34,222 entries in the E7 frozen-input manifest
   are missing on disk:
   - the `milestone_c15c` raw artifacts and shards
   - the c15c logs
   - `second_case_final.zip`
   - a matplotlib font cache

   As a result `run_ress_e7._verify_design` and the final audit's manifest check cannot pass even
   in the development repository.
5. **File-time chronology lost.** Filesystem mtimes no longer preserve the
   prediction < decision < truth order (all 1,200 E5 episodes violate it on disk). The original
   nanosecond timestamps survive in the per-episode result and ledger CSV columns, and those are
   used for chronology checks.
6. **Hard-coded narrative numbers.** Report-text and figure annotations hard-code the frozen
   numbers:
   - C1 at lines 585, 688 and 694
   - C1.5A at lines 182–243
   - several E6, E6B and final-audit verdict strings

   These are report-generation text only, not analysis code, and they are left unchanged.
7. **`b5` hashes bare module filenames relative to the working directory** (lines 165–170). The
   final experiments never call `b5.run`.

## 9. Items requiring a manual decision

| Item | Why uncertain | Release handling |
|---|---|---|
| Two-tank second process (`milestone_d1/d2`, `two_tank_process.py`, `results/second_case_*`) | Not in the requested experiment list; superseded by the flash drum. The final audit includes a two-tank section. | Excluded. The two-tank section of the final audit is removed (recorded as a source edit). Left in the development repository. |
| C1.5C multi-action screening | Cited in `results/cstr_final` rows R17–R19, but not in the requested list. Raw shards are absent and it uses a rounded q+. | Excluded; left in the development repository. |
| `results/cstr_final/` | No generating script (agent-authored closure). Read by the E5 comparison figure. | Kept as a frozen input and flagged as authored rather than generated. |
| `IMPLEMENTATION_PLAN.md`, `prompts/` | Development planning | Excluded |
| M4 action-conditioned placeholder | Required by result schemas and hash manifests | Kept inside frozen per-episode files and code (cannot be removed without changing frozen outputs). Removed from public tables and figures, and documented as unsupported. |
| `LICENSE`, `CITATION.cff` authors | Legal and authorship choices belong to the authors | Placeholders added and flagged |

## 10. M4 / action-conditioned content

- **The code never estimates or certifies with M4.**
  - In E4, Mondrian support is `False` (`E2_BINS_NOT_PREDECLARED`).
  - E5 and E7 hard-code M4 to NOT CERTIFIED with margin −∞.
  - E6 does the same for `action_conditioned`.
- **No final result depends on M4.**
- **M4 appears only in output artifacts, as a non-result:** M4/`action_conditioned` rows in the
  per-episode CSVs and summaries, and `e5_figures/05_pooled_vs_action_conditioned_coverage.png`
  (a zero-height bar).
- **Release handling:**
  - Do not present M4 in the README or in the paper tables (derived tables filter to M0–M3).
  - Drop figure 05.
  - Keep the frozen per-episode CSVs unchanged, because the hash manifests and the audit expect
    five methods.
- **E6B's M5 is not part of the TwinGuard method.** It is a preregistered stress-analysis
  comparator that certified nothing.
- **The CSTR "M4" and B12 "B2M" are unrelated to action-conditioned certification of the final
  method.**

## 11. Duplicates and near-duplicates

| Item | Finding | Decision |
|---|---|---|
| `results/milestone_b9_smoke`, `b10_smoke`, `b10_debug`, `b11_debug`, `b12_debug` | smoke/debug re-runs of the same stages | exclude |
| `results/milestone_c15c_shard0..3` vs `milestone_c15c` | shards and their aggregate (shards absent) | exclude (C1.5C out of scope) |
| root-level CSTR modules in git `HEAD` vs `twingaurd/` | same code moved; relative imports only, apart from the two additive changes | use the `twingaurd/` version |
| Boundary-action selection | three implementations: `flash_population.select_boundary_action` (tests only), `run_flash_e4._boundary_action` (153-point pool; used by E4/E6/E7) and E6B's own 256-LHS screen | **not merged.** Different pools and seeds define different populations. |
| CSTR nominal boundary screens | batched RK45 nominal CSTR in `run_ress_e6._select_cstr_boundary_action` and in E6B `_cstr_screen` | not merged (different band definitions) |
| `corrected_quantile`, `wilson` | implemented separately in b12/C1 and `flash_conformal`/E5/E7 | not merged (different frozen code paths) |
| `results/cstr_final/FINAL_CSTR_RESULTS.csv` | restates values from C1/C1.6B | keep as a frozen input; paper tables cite the primary sources |

## 12. Proposed release structure

These decisions follow from the findings above.

- **Code** moves into an installable package, `src/twinguard/`, organised by component:
  - `plants`, `digital_twins`, `scenarios`, `discrepancy`, `uncertainty` and `conformal`
  - stage modules for `cstr`, `flash` and the cross-case `evaluation`

  Each original module maps to **one** new module (no function-level splitting, which could
  silently alter behaviour). The only edits are import statements, file-path constants and
  frozen-hash resolution. They are recorded in a machine-checkable ledger
  (`reproducibility/provenance/source_ledger.json`): reversing the ledger reproduces the exact
  frozen SHA-256 of every module.
- **Frozen experiment directories** move to `artifacts/` under descriptive names. File names
  *inside* them are unchanged, because the frozen hash manifests and the code bind to them; a path
  map records every original → release path.
- **Per-episode artifact trees** (≈ 600 MB, ≈ 66,000 files) are excluded from git and staged as
  a frozen archive for deposition (e.g. Zenodo). Their SHA-256 values remain in git, in the
  shipped hash manifests and in the path-sanitized ledgers.
- **Paper-facing tables and figures** get semantic names under `results/`, generated from the
  frozen outputs by `scripts/`.
- **Reproducibility audit:** a new read-only `reproducibility/run_reproducibility_audit.py` runs
  on the git checkout alone and, with `--archive`, over the full frozen archive. It never
  overwrites frozen outputs.

---

## Appendix A: Grouped inventory of the development repository

One row per file, or per directory of homogeneous per-episode artifacts (marked `/*` or `/**`).
"Referenced by" lists repository modules that import the file (tests excluded). Classification
values are explained in §3–§4.

| Path or group | Files | Size | Types | Purpose | Referenced by | Classification |
|---|---:|---:|---|---|---|---|
| `.gitignore` | 1 | 0.5 kB | (none):1 | development packaging/config | n/a | replaced in release |
| `.pytest_cache/*` | 3 | 0.5 kB | (none):1 .tag:1 .md:1 | Python bytecode / pytest cache | no | cached |
| `.pytest_cache/v/cache/*` | 2 | 20.2 kB | (none):2 | Python bytecode / pytest cache | no | cached |
| `E6B_MARGIN_LADDER/CSTR_E6B_EPISODES.csv` | 1 | 0.14 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E6B_MARGIN_LADDER/CSTR_E6B_MARGIN_RESULTS.csv` | 1 | 1.19 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E6B_MARGIN_LADDER/CSTR_E6B_RESULTS.md` | 1 | 1.6 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E6B_MARGIN_LADDER/E6B_ARTIFACT_AUDIT.csv` | 1 | 2.1 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E6B_MARGIN_LADDER/E6B_CLAIMS_AUDIT.md` | 1 | 1.1 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E6B_MARGIN_LADDER/E6B_FINAL_ARTIFACT_HASHES.txt` | 1 | 2.27 MB | .txt:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E6B_MARGIN_LADDER/E6B_PREDECLARATION.md` | 1 | 6.3 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E6B_MARGIN_LADDER/E6B_PREDECLARATION.sha256` | 1 | 0.1 kB | .sha256:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E6B_MARGIN_LADDER/E6B_RISK_AVAILABILITY.csv` | 1 | 13.7 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E6B_MARGIN_LADDER/E6B_SEED_MANIFEST.csv` | 1 | 0.50 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E6B_MARGIN_LADDER/FLASH_E6B_CALIBRATION_SCORES.csv` | 1 | 0.35 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E6B_MARGIN_LADDER/FLASH_E6B_CONFORMAL_CONFIG.json` | 1 | 1.6 kB | .json:1 | frozen experiment input/output table or config | yes | final |
| `E6B_MARGIN_LADDER/FLASH_E6B_CONFORMAL_CONFIG.sha256` | 1 | 0.1 kB | .sha256:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E6B_MARGIN_LADDER/FLASH_E6B_EPISODES.csv` | 1 | 0.32 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E6B_MARGIN_LADDER/FLASH_E6B_MARGIN_RESULTS.csv` | 1 | 6.06 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E6B_MARGIN_LADDER/FLASH_E6B_RESULTS.md` | 1 | 4.3 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E6B_MARGIN_LADDER/__init__.py` | 1 | 0.1 kB | .py:1 | package marker | no (entry point or unused) | final (packaging) |
| `E6B_MARGIN_LADDER/__pycache__/*` | 4 | 0.12 MB | .pyc:4 | Python bytecode / pytest cache | no | cached |
| `E6B_MARGIN_LADDER/artifacts/CSTR_TEST_ARTIFACT_AUDIT.csv` | 1 | 1.19 MB | .csv:1 | per-episode chronology/hash ledger (contains absolute paths from author machine) | final audit, tests | final (ledger) |
| `E6B_MARGIN_LADDER/artifacts/FLASH_CALIBRATION_ARTIFACT_AUDIT.csv` | 1 | 1.17 MB | .csv:1 | per-episode chronology/hash ledger (contains absolute paths from author machine) | final audit, tests | final (ledger) |
| `E6B_MARGIN_LADDER/artifacts/FLASH_TEST_ARTIFACT_AUDIT.csv` | 1 | 1.89 MB | .csv:1 | per-episode chronology/hash ledger (contains absolute paths from author machine) | final audit, tests | final (ledger) |
| `E6B_MARGIN_LADDER/artifacts/cstr_test/candidates/*` | 1500 | 0.50 MB | .json:1500 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/artifacts/cstr_test/decisions/*` | 1166 | 0.51 MB | .json:1166 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/artifacts/cstr_test/predictions/*` | 1166 | 19.30 MB | .npz:1166 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/artifacts/cstr_test/truths/*` | 1166 | 10.27 MB | .npz:1166 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/artifacts/flash_calibration/candidates/*` | 1500 | 0.59 MB | .json:1500 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/artifacts/flash_calibration/predictions/*` | 1499 | 26.92 MB | .npz:1499 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/artifacts/flash_calibration/truths/*` | 1499 | 7.80 MB | .npz:1499 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/artifacts/flash_test/candidates/*` | 2000 | 0.75 MB | .json:2000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/artifacts/flash_test/decisions/*` | 2000 | 1.49 MB | .json:2000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/artifacts/flash_test/predictions/*` | 2000 | 35.89 MB | .npz:2000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/artifacts/flash_test/truths/*` | 2000 | 10.57 MB | .npz:2000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E6B_MARGIN_LADDER/figures/01_margin_vs_false_safe.png` | 1 | 84.8 kB | .png:1 | generated figure | no | final figure / candidate |
| `E6B_MARGIN_LADDER/figures/02_margin_vs_certification.png` | 1 | 88.4 kB | .png:1 | generated figure | no | final figure / candidate |
| `E6B_MARGIN_LADDER/figures/03_margin_vs_safe_action_certification.png` | 1 | 81.9 kB | .png:1 | generated figure | no | final figure / candidate |
| `E6B_MARGIN_LADDER/figures/04_risk_availability_operating_envelope.png` | 1 | 79.0 kB | .png:1 | generated figure | no | final figure / candidate |
| `E6B_MARGIN_LADDER/figures/05_margin_vs_violation_severity.png` | 1 | 47.4 kB | .png:1 | generated figure | no | final figure / candidate |
| `E6B_MARGIN_LADDER/figures/06_flash_vs_cstr_margin_ladder.png` | 1 | 74.1 kB | .png:1 | generated figure | no | final figure / candidate |
| `E6B_MARGIN_LADDER/frozen_boundaries.py` | 1 | 0.5 kB | .py:1 | output guard for margin-ladder outputs | yes: E6B_MARGIN_LADDER/run_e6b.py, E6B_MARGIN_LADDER/write_preregistration.py, E_FLASH_RESS/__init__.py, E_FLASH_RESS/audit_ress_final.py, E_FLASH_RESS/freeze_ress_e7.py, E_FLASH_RESS/run_flash_e1.py ... | final (infrastructure) |
| `E6B_MARGIN_LADDER/run_e6b.py` | 1 | 82.4 kB | .py:1 | margin-ladder stress analysis (bands B0-B4; M0-M3 + M5 margin-conditioned; CSTR) | no (entry point or unused) | final |
| `E6B_MARGIN_LADDER/write_preregistration.py` | 1 | 10.0 kB | .py:1 | margin-ladder seed reservation + preregistration | yes: E6B_MARGIN_LADDER/run_e6b.py | final |
| `E_FLASH_RESS/CSTR_BOUNDARY_RESULTS.csv` | 1 | 0.24 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/CSTR_RELIABILITY_RESULTS.csv` | 1 | 2.68 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/CSTR_RELIABILITY_SUMMARY.csv` | 1 | 1.2 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/FLASH_BOUNDARY_RESULTS.csv` | 1 | 5.84 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/FLASH_E1_VALIDATION.md` | 1 | 2.6 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/FLASH_E2_PREDECLARATION.md` | 1 | 14.4 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/FLASH_E2_PREDECLARATION.sha256` | 1 | 0.1 kB | .sha256:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E_FLASH_RESS/FLASH_E3_MODEL_SELECTION.md` | 1 | 3.6 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/FLASH_E4_ARTIFACT_HASHES.txt` | 1 | 1.3 kB | .txt:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E_FLASH_RESS/FLASH_E4_CALIBRATION.md` | 1 | 3.7 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/FLASH_E5_ARTIFACT_HASHES.txt` | 1 | 0.44 MB | .txt:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E_FLASH_RESS/FLASH_E5_RESULTS.md` | 1 | 2.5 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/FLASH_FINAL_ABLATION.csv` | 1 | 45.2 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/FLASH_FINAL_RESULTS.csv` | 1 | 4.81 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/FLASH_PROCESS_SPECIFICATION.md` | 1 | 5.5 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/FLASH_RELIABILITY_RESULTS.csv` | 1 | 12.79 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/FLASH_RELIABILITY_SUMMARY.csv` | 1 | 2.3 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/FLASH_STRUCTURAL_RESULTS.csv` | 1 | 2.59 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/FLASH_STRUCTURAL_SUMMARY.csv` | 1 | 2.2 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/FROZEN_INPUT_HASHES.txt` | 1 | 8.9 kB | .txt:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E_FLASH_RESS/IMPLEMENTATION_PLAN.md` | 1 | 16.7 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/RESS_E6_ARTIFACT_HASHES.txt` | 1 | 0.75 MB | .txt:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E_FLASH_RESS/RESS_E6_DESIGN.json` | 1 | 2.1 kB | .json:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/RESS_E6_RESULTS.md` | 1 | 2.1 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/RESS_E7_AMENDMENT_01.md` | 1 | 0.9 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/RESS_E7_DESIGN.json` | 1 | 3.0 kB | .json:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/RESS_E7_DESIGN.sha256` | 1 | 0.1 kB | .sha256:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E_FLASH_RESS/RESS_E7_FROZEN_INPUT_HASHES.txt` | 1 | 4.79 MB | .txt:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E_FLASH_RESS/RESS_E7_RESULTS.md` | 1 | 2.3 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/RESS_FINAL_ARTIFACT_HASHES.txt` | 1 | 4.22 MB | .txt:1 | frozen hash manifest / sidecar | verification code | final (provenance) |
| `E_FLASH_RESS/RESS_FINAL_AUDIT.md` | 1 | 1.0 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/RESS_FINAL_AUDIT_DETAILS.csv` | 1 | 7.7 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/RESS_FINAL_CLAIMS_AUDIT.md` | 1 | 2.1 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/RESS_FINAL_FIGURE_MANIFEST.md` | 1 | 1.1 kB | .md:1 | generated or hand-written report/design document | some hashed | final (provenance/report) |
| `E_FLASH_RESS/RESS_FINAL_RESULTS.csv` | 1 | 2.3 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/RISK_AVAILABILITY_RESULTS.csv` | 1 | 2.2 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/__init__.py` | 1 | 0.7 kB | .py:1 | flash package exports | no (entry point or unused) | final (packaging) |
| `E_FLASH_RESS/__pycache__/*` | 12 | 0.29 MB | .pyc:12 | Python bytecode / pytest cache | no | cached |
| `E_FLASH_RESS/audit_ress_final.py` | 1 | 46.7 kB | .py:1 | final cross-experiment audit (reconstruction, leakage, seeds, safety labels); writes RESS_FINAL_* | no (entry point or unused) | final (audit) |
| `E_FLASH_RESS/e1_validation/ACTION_STEP_RESPONSES.csv` | 1 | 0.16 MB | .csv:1 | simulator validation outputs | E7 manifest | final |
| `E_FLASH_RESS/e1_validation/E1_VALIDATION_SUMMARY.json` | 1 | 2.1 kB | .json:1 | simulator validation outputs | E7 manifest | final |
| `E_FLASH_RESS/e1_validation/HEALTHY_SCENARIO_SUMMARY.csv` | 1 | 1.7 kB | .csv:1 | simulator validation outputs | E7 manifest | final |
| `E_FLASH_RESS/e1_validation/HEALTHY_TRAJECTORIES.csv` | 1 | 0.18 MB | .csv:1 | simulator validation outputs | E7 manifest | final |
| `E_FLASH_RESS/e3_data/EVALUATOR_GENERATION_MANIFEST.csv` | 1 | 0.19 MB | .csv:1 | frozen training/selection/calibration inputs | E4/E5 reruns + hash manifests | final (frozen inputs) |
| `E_FLASH_RESS/e3_data/selection_operational_inputs.npz` | 1 | 1.08 MB | .npz:1 | frozen training/selection/calibration inputs | E4/E5 reruns + hash manifests | final (frozen inputs) |
| `E_FLASH_RESS/e3_data/selection_predictions_before_truth_serialization.npz` | 1 | 1.19 MB | .npz:1 | frozen training/selection/calibration inputs | E4/E5 reruns + hash manifests | final (frozen inputs) |
| `E_FLASH_RESS/e3_data/selection_truth_evaluator_only.npz` | 1 | 0.74 MB | .npz:1 | frozen training/selection/calibration inputs | E4/E5 reruns + hash manifests | final (frozen inputs) |
| `E_FLASH_RESS/e3_data/training_method_inputs.npz` | 1 | 5.06 MB | .npz:1 | frozen training/selection/calibration inputs | E4/E5 reruns + hash manifests | final (frozen inputs) |
| `E_FLASH_RESS/e4_data/CALIBRATION_EVALUATOR_MANIFEST.csv` | 1 | 0.20 MB | .csv:1 | frozen training/selection/calibration inputs | E4/E5 reruns + hash manifests | final (frozen inputs) |
| `E_FLASH_RESS/e4_data/calibration_boundary_operational_inputs.npz` | 1 | 3.51 MB | .npz:1 | frozen training/selection/calibration inputs | E4/E5 reruns + hash manifests | final (frozen inputs) |
| `E_FLASH_RESS/e4_data/calibration_ordinary_operational_inputs.npz` | 1 | 3.38 MB | .npz:1 | frozen training/selection/calibration inputs | E4/E5 reruns + hash manifests | final (frozen inputs) |
| `E_FLASH_RESS/e4_data/ensemble_selection_diagnostics.csv` | 1 | 0.8 kB | .csv:1 | frozen training/selection/calibration inputs | E4/E5 reruns + hash manifests | final (frozen inputs) |
| `E_FLASH_RESS/e5_artifact_audit.csv` | 1 | 0.87 MB | .csv:1 | per-episode chronology/hash ledger (contains absolute paths from author machine) | final audit, tests | final (ledger) |
| `E_FLASH_RESS/e5_artifacts/decisions/*` | 1200 | 1.13 MB | .json:1200 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e5_artifacts/predictions/*` | 1200 | 20.08 MB | .npz:1200 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e5_artifacts/truths/*` | 1200 | 8.01 MB | .npz:1200 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e5_figures/01_representative_corrected_rollout.png` | 1 | 0.19 MB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e5_figures/02_true_vs_nominal_margin.png` | 1 | 0.16 MB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e5_figures/03_true_vs_conformal_margin.png` | 1 | 0.18 MB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e5_figures/04_certification_confusion.png` | 1 | 36.9 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e5_figures/05_pooled_vs_action_conditioned_coverage.png` | 1 | 40.8 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e5_figures/06_false_safe_risk_and_certification.png` | 1 | 30.7 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e5_figures/07_safety_violation_severity.png` | 1 | 31.9 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e5_figures/08_cstr_vs_flash_comparison.png` | 1 | 31.0 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e6_artifacts/CSTR_RESS_STRESS_ARTIFACT_AUDIT.csv` | 1 | 0.30 MB | .csv:1 | per-episode chronology/hash ledger (contains absolute paths from author machine) | final audit, tests | final (ledger) |
| `E_FLASH_RESS/e6_artifacts/FLASH_RESS_STRESS_ARTIFACT_AUDIT.csv` | 1 | 1.14 MB | .csv:1 | per-episode chronology/hash ledger (contains absolute paths from author machine) | final audit, tests | final (ledger) |
| `E_FLASH_RESS/e6_artifacts/cstr/candidates/*` | 300 | 74.5 kB | .json:300 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e6_artifacts/cstr/decisions/*` | 300 | 0.10 MB | .json:300 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e6_artifacts/cstr/predictions/*` | 300 | 4.99 MB | .npz:300 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e6_artifacts/cstr/truths/*` | 300 | 2.58 MB | .npz:300 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e6_artifacts/flash/candidates/*` | 1200 | 0.42 MB | .json:1200 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e6_artifacts/flash/decisions/*` | 1200 | 1.07 MB | .json:1200 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e6_artifacts/flash/predictions/*` | 1200 | 20.06 MB | .npz:1200 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e6_artifacts/flash/truths/*` | 1200 | 6.21 MB | .npz:1200 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e6_figures/01_risk_availability_frontier.png` | 1 | 55.6 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e6_figures/02_severity_risk_frontier.png` | 1 | 52.7 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e6_figures/03_near_boundary_true_vs_predicted_margins.png` | 1 | 0.18 MB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e6_figures/04_cstr_vs_flash_boundary_behavior.png` | 1 | 37.6 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e7_artifacts/CSTR_RELIABILITY_ARTIFACT_AUDIT.csv` | 1 | 3.46 MB | .csv:1 | per-episode chronology/hash ledger (contains absolute paths from author machine) | final audit, tests | final (ledger) |
| `E_FLASH_RESS/e7_artifacts/FLASH_RELIABILITY_ARTIFACT_AUDIT.csv` | 1 | 3.25 MB | .csv:1 | per-episode chronology/hash ledger (contains absolute paths from author machine) | final audit, tests | final (ledger) |
| `E_FLASH_RESS/e7_artifacts/STRUCTURAL_ARTIFACT_AUDIT.csv` | 1 | 0.61 MB | .csv:1 | per-episode chronology/hash ledger (contains absolute paths from author machine) | final audit, tests | final (ledger) |
| `E_FLASH_RESS/e7_artifacts/cstr_reliability/candidates/*` | 3000 | 0.66 MB | .json:3000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/cstr_reliability/decisions/*` | 3000 | 1.22 MB | .json:3000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/cstr_reliability/predictions/*` | 3000 | 48.36 MB | .npz:3000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/cstr_reliability/truths/*` | 3000 | 26.09 MB | .npz:3000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/flash_reliability/flash_reliability/candidates/*` | 3000 | 1.46 MB | .json:3000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/flash_reliability/flash_reliability/decisions/*` | 3000 | 2.84 MB | .json:3000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/flash_reliability/flash_reliability/predictions/*` | 3000 | 52.04 MB | .npz:3000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/flash_reliability/flash_reliability/truths/*` | 3000 | 20.10 MB | .npz:3000 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/structural/structural/candidates/*` | 600 | 0.29 MB | .json:600 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/structural/structural/decisions/*` | 600 | 0.57 MB | .json:600 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/structural/structural/predictions/*` | 600 | 10.42 MB | .npz:600 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_artifacts/structural/structural/truths/*` | 600 | 4.48 MB | .npz:600 | per-episode candidate/prediction/decision/truth artifacts | final audit / reruns | raw data (frozen evidence) |
| `E_FLASH_RESS/e7_figures/01_large_reliability_evidence.png` | 1 | 56.0 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/e7_figures/02_structural_mismatch_evidence.png` | 1 | 33.4 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/figures/action_step_responses.png` | 1 | 0.13 MB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/figures/composition_vle_diagnostics.png` | 1 | 98.5 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/figures/healthy_plant_dt_parity.png` | 1 | 0.11 MB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/figures/level_trajectory.png` | 1 | 51.5 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/figures/pressure_trajectory.png` | 1 | 49.7 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/figures/temperature_trajectory.png` | 1 | 50.5 kB | .png:1 | generated figure | no | final figure / candidate |
| `E_FLASH_RESS/flash_bootstrap.py` | 1 | 3.0 kB | .py:1 | coherent bootstrap predictive summary (J models, 2000 paths) | yes: E6B_MARGIN_LADDER/run_e6b.py, E_FLASH_RESS/flash_conformal.py, E_FLASH_RESS/run_flash_e4.py, E_FLASH_RESS/run_flash_e5.py, E_FLASH_RESS/run_ress_e6.py | final |
| `E_FLASH_RESS/flash_bootstrap_manifest.csv` | 1 | 13.3 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/flash_bootstrap_models.json` | 1 | 0.55 MB | .json:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/flash_calibration_scores.csv` | 1 | 0.27 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/flash_config.yaml` | 1 | 1.0 kB | .yaml:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/flash_conformal.py` | 1 | 5.1 kB | .py:1 | whole-trajectory directional score, corrected quantile, envelope, SAFE/NOT CERTIFIED rule | yes: E6B_MARGIN_LADDER/run_e6b.py, E_FLASH_RESS/run_flash_e4.py, E_FLASH_RESS/run_flash_e5.py, E_FLASH_RESS/run_ress_e6.py, E_FLASH_RESS/run_ress_e7.py | final |
| `E_FLASH_RESS/flash_conformal_config.json` | 1 | 3.5 kB | .json:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/flash_digital_twin.py` | 1 | 0.6 kB | .py:1 | independent nominal flash digital twin (own parameter copy) | yes: E_FLASH_RESS/__init__.py, E_FLASH_RESS/flash_discrepancy_model.py, E_FLASH_RESS/flash_population.py, E_FLASH_RESS/run_flash_e1.py, E_FLASH_RESS/run_flash_e3.py, E_FLASH_RESS/run_flash_e4.py ... | final |
| `E_FLASH_RESS/flash_discrepancy_model.py` | 1 | 21.4 kB | .py:1 | flash latent linear-Gaussian discrepancy state-space model (EM, Kalman, counterfactual prediction) | yes: E_FLASH_RESS/flash_bootstrap.py, E_FLASH_RESS/run_flash_e3.py, E_FLASH_RESS/run_flash_e4.py, E_FLASH_RESS/run_flash_e5.py, E_FLASH_RESS/run_ress_e6.py, E_FLASH_RESS/run_ress_e7.py | final |
| `E_FLASH_RESS/flash_model_selection.csv` | 1 | 3.0 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/flash_population.py` | 1 | 13.4 kB | .py:1 | seeding (purpose_seed), fault generator F1-F3, evaluator-only fault truth, fault-blind operational record, safety specification, action designs | yes: E6B_MARGIN_LADDER/run_e6b.py, E6B_MARGIN_LADDER/write_preregistration.py, E_FLASH_RESS/audit_ress_final.py, E_FLASH_RESS/flash_conformal.py, E_FLASH_RESS/flash_discrepancy_model.py, E_FLASH_RESS/run_flash_e3.py ... | final |
| `E_FLASH_RESS/flash_population_manifest.csv` | 1 | 1.3 kB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/flash_process.py` | 1 | 22.4 kB | .py:1 | benzene/toluene dynamic flash-drum physics shared by plant and twin; FlashPlant | yes: E6B_MARGIN_LADDER/run_e6b.py, E_FLASH_RESS/__init__.py, E_FLASH_RESS/flash_digital_twin.py, E_FLASH_RESS/flash_discrepancy_model.py, E_FLASH_RESS/flash_population.py, E_FLASH_RESS/flash_structural_mismatch.py ... | final |
| `E_FLASH_RESS/flash_seed_manifest.csv` | 1 | 2.00 MB | .csv:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/flash_structural_mismatch.py` | 1 | 3.3 kB | .py:1 | plant-only unmodelled vapour-valve lag (structural mismatch) + lag sampler | yes: E_FLASH_RESS/run_ress_e7.py | final |
| `E_FLASH_RESS/freeze_ress_e7.py` | 1 | 6.2 kB | .py:1 | freezes E7 design + 34,222-entry frozen input hash manifest | no (entry point or unused) | final (provenance) |
| `E_FLASH_RESS/frozen_boundaries.py` | 1 | 3.5 kB | .py:1 | output-isolation guard for flash outputs | yes: E6B_MARGIN_LADDER/run_e6b.py, E6B_MARGIN_LADDER/write_preregistration.py, E_FLASH_RESS/__init__.py, E_FLASH_RESS/audit_ress_final.py, E_FLASH_RESS/freeze_ress_e7.py, E_FLASH_RESS/run_flash_e1.py ... | final (infrastructure) |
| `E_FLASH_RESS/run_flash_e1.py` | 1 | 19.5 kB | .py:1 | healthy plant/nominal-DT parity, step responses, convergence, VLE checks | no (entry point or unused) | final |
| `E_FLASH_RESS/run_flash_e3.py` | 1 | 28.8 kB | .py:1 | flash training/model-selection data generation, EM fit k=1..3, gated selection | no (entry point or unused) | final |
| `E_FLASH_RESS/run_flash_e4.py` | 1 | 42.6 kB | .py:1 | bootstrap ensemble (100 accepted), ensemble size J, ordinary+boundary conformal calibration | yes: E6B_MARGIN_LADDER/run_e6b.py, E_FLASH_RESS/run_ress_e6.py, E_FLASH_RESS/run_ress_e7.py | final |
| `E_FLASH_RESS/run_flash_e5.py` | 1 | 54.2 kB | .py:1 | primary prospective flash evaluation N=1200, M0-M3 ablation (+M4 unsupported placeholder) | yes: E6B_MARGIN_LADDER/run_e6b.py, E_FLASH_RESS/audit_ress_final.py, E_FLASH_RESS/run_ress_e6.py, E_FLASH_RESS/run_ress_e7.py | final |
| `E_FLASH_RESS/run_ress_e6.py` | 1 | 51.3 kB | .py:1 | boundary-region challenge: flash N=1200 + CSTR N=300 | yes: E6B_MARGIN_LADDER/run_e6b.py, E_FLASH_RESS/audit_ress_final.py, E_FLASH_RESS/run_ress_e7.py | final |
| `E_FLASH_RESS/run_ress_e7.py` | 1 | 33.6 kB | .py:1 | structural mismatch N=600 + mixed-regime reliability audits (flash N=3000, CSTR N=3000) | yes: E_FLASH_RESS/audit_ress_final.py, E_FLASH_RESS/freeze_ress_e7.py | final |
| `E_FLASH_RESS/selected_flash_model.json` | 1 | 8.9 kB | .json:1 | frozen experiment input/output table or config | yes | final |
| `E_FLASH_RESS/write_flash_e2_preregistration.py` | 1 | 5.0 kB | .py:1 | seed/population reservation + E2 preregistration sidecar | no (entry point or unused) | final |
| `__pycache__/*` | 28 | 1.23 MB | .pyc:28 | Python bytecode / pytest cache | no | cached |
| `prompts/milestone_C15C.txt` | 1 | 5.8 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_C16A.txt` | 1 | 3.9 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_C16B.txt` | 1 | 4.7 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_C16C.txt` | 1 | 4.1 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_D1.txt` | 1 | 2.6 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_D2.txt` | 1 | 3.1 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_E0.txt` | 1 | 3.3 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_E1.txt` | 1 | 5.0 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_E2.txt` | 1 | 5.2 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_E3.txt` | 1 | 3.5 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_E4.txt` | 1 | 3.6 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_E5.txt` | 1 | 3.3 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_E6.txt` | 1 | 2.9 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_E6_b.txt` | 1 | 19.8 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `prompts/milestone_E7.txt` | 1 | 3.1 kB | .txt:1 | coding-agent task prompt (development instructions) | no | debugging/development |
| `pyproject.toml` | 1 | 0.5 kB | .toml:1 | development packaging/config | n/a | replaced in release |
| `requirements.txt` | 1 | 0.1 kB | .txt:1 | development packaging/config | n/a | replaced in release |
| `results/cstr_final/<top-level files>` | 8 | 26.3 kB | .md:6 .csv:2 | frozen stage outputs: summaries, manifests, frozen models, calibration | yes (downstream stages/audit) | final |
| `results/milestone_a/<top-level files>` | 13 | 1.39 MB | .pdf:4 .png:4 .csv:3 .md:1 .json:1 | development milestone outputs | no | obsolete experiment |
| `results/milestone_a5/<top-level files>` | 17 | 2.28 MB | .csv:5 .pdf:5 .png:5 .md:1 .json:1 | development milestone outputs | no | obsolete experiment |
| `results/milestone_b/<top-level files>` | 32 | 21.63 MB | .csv:12 .pdf:9 .png:9 .md:1 .json:1 | development milestone outputs | no | obsolete experiment |
| `results/milestone_b10/<top-level files>` | 13 | 0.44 MB | .csv:9 .md:2 .json:2 | frozen stage outputs: summaries, manifests, frozen models, calibration | yes (downstream stages/audit) | final |
| `results/milestone_b10/figures/**` | 20 | 2.16 MB | .pdf:10 .png:10 | development figures (figures) | no | intermediate (figures) |
| `results/milestone_b10/predictions/**` | 240 | 17.86 MB | .npz:240 | per-episode frozen artifacts (predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b10/revealed_test_truth/**` | 240 | 0.90 MB | .npz:240 | per-episode frozen artifacts (revealed_test_truth) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b10_debug/predictions/**` | 3 | 0.23 MB | .npz:3 | debug/smoke run outputs | no | debugging/development |
| `results/milestone_b10_debug/revealed_test_truth/**` | 3 | 11.1 kB | .npz:3 | debug/smoke run outputs | no | debugging/development |
| `results/milestone_b10_smoke/<top-level files>` | 5 | 22.2 kB | .csv:4 .json:1 | debug/smoke run outputs | no | debugging/development |
| `results/milestone_b11/<top-level files>` | 15 | 0.87 MB | .csv:11 .md:2 .json:2 | frozen stage outputs: summaries, manifests, frozen models, calibration | yes (downstream stages/audit) | final |
| `results/milestone_b11/calibration_predictions/**` | 234 | 9.12 MB | .npz:234 | per-episode frozen artifacts (calibration_predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b11/figures/**` | 24 | 1.86 MB | .pdf:12 .png:12 | development figures (figures) | no | intermediate (figures) |
| `results/milestone_b11/predictions/**` | 240 | 18.28 MB | .npz:240 | per-episode frozen artifacts (predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b11/revealed_calibration_truth/**` | 60 | 0.22 MB | .npz:60 | per-episode frozen artifacts (revealed_calibration_truth) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b11/revealed_final_truth/**` | 240 | 0.90 MB | .npz:240 | per-episode frozen artifacts (revealed_final_truth) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b11_debug/predictions/**` | 3 | 0.23 MB | .npz:3 | debug/smoke run outputs | no | debugging/development |
| `results/milestone_b11_debug/truth/**` | 3 | 11.1 kB | .npz:3 | debug/smoke run outputs | no | debugging/development |
| `results/milestone_b12/<top-level files>` | 14 | 3.23 MB | .csv:10 .md:2 .json:1 .npy:1 | frozen stage outputs: summaries, manifests, frozen models, calibration | yes (downstream stages/audit) | final |
| `results/milestone_b12/calibration_predictions/**` | 500 | 21.43 MB | .npz:500 | per-episode frozen artifacts (calibration_predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b12/diagnostic_predictions/**` | 150 | 6.43 MB | .npz:150 | per-episode frozen artifacts (diagnostic_predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b12/figures/**` | 24 | 1.64 MB | .pdf:12 .png:12 | development figures (figures) | no | intermediate (figures) |
| `results/milestone_b12/predictions/**` | 500 | 21.42 MB | .npz:500 | per-episode frozen artifacts (predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b12/revealed_calibration_truth/**` | 500 | 1.86 MB | .npz:500 | per-episode frozen artifacts (revealed_calibration_truth) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b12/revealed_diagnostic_truth/**` | 150 | 0.56 MB | .npz:150 | per-episode frozen artifacts (revealed_diagnostic_truth) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b12/revealed_final_truth/**` | 500 | 1.87 MB | .npz:500 | per-episode frozen artifacts (revealed_final_truth) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_b12_debug/debug_predictions/**` | 3 | 0.13 MB | .npz:3 | debug/smoke run outputs | no | debugging/development |
| `results/milestone_b12_debug/truth/**` | 3 | 11.2 kB | .npz:3 | debug/smoke run outputs | no | debugging/development |
| `results/milestone_b5/<top-level files>` | 11 | 0.43 MB | .csv:7 .md:2 .json:2 | frozen upstream outputs (b9/b10 baselines/episodes) | yes (b9/b10 reruns) | intermediate |
| `results/milestone_b5/evaluations/**` | 300 | 13.63 MB | .npz:300 | per-episode frozen artifacts (evaluations) | inputs to b9/b10 reruns | intermediate (frozen upstream) |
| `results/milestone_b5/figures/**` | 24 | 2.53 MB | .pdf:12 .png:12 | development figures (figures) | inputs to b9/b10 reruns | intermediate (frozen upstream) |
| `results/milestone_b5/predictions/**` | 300 | 12.52 MB | .npz:300 | per-episode frozen artifacts (predictions) | inputs to b9/b10 reruns | intermediate (frozen upstream) |
| `results/milestone_b5_stderr.log` | 1 | 0.0 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_b5_stdout.log` | 1 | 0.5 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_b6/<top-level files>` | 6 | 143.61 MB | .csv:4 .md:1 .json:1 | development milestone outputs | no | obsolete experiment |
| `results/milestone_b6/figures/**` | 14 | 3.34 MB | .pdf:7 .png:7 | development figures (figures) | no | obsolete experiment |
| `results/milestone_b6_stderr.log` | 1 | 0.0 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_b6_stdout.log` | 1 | 0.4 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_b7a/<top-level files>` | 7 | 19.90 MB | .csv:5 .md:1 .json:1 | development milestone outputs | no | obsolete experiment |
| `results/milestone_b7a/figures/**` | 16 | 4.05 MB | .pdf:8 .png:8 | development figures (figures) | no | obsolete experiment |
| `results/milestone_b7a_stderr.log` | 1 | 0.0 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_b7a_stdout.log` | 1 | 0.5 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_b7b/<top-level files>` | 8 | 6.11 MB | .csv:6 .md:1 .json:1 | frozen upstream outputs (b9/b10 baselines/episodes) | yes (b9/b10 reruns) | intermediate |
| `results/milestone_b7b/figures/**` | 16 | 1.05 MB | .pdf:8 .png:8 | development figures (figures) | inputs to b9/b10 reruns | intermediate (frozen upstream) |
| `results/milestone_b7b_stderr.log` | 1 | 0.0 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_b7b_stdout.log` | 1 | 0.1 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_b8/<top-level files>` | 11 | 3.94 MB | .csv:9 .md:1 .json:1 | frozen upstream outputs (b9/b10 baselines/episodes) | yes (b9/b10 reruns) | intermediate |
| `results/milestone_b8/figures/**` | 18 | 1.34 MB | .pdf:9 .png:9 | development figures (figures) | inputs to b9/b10 reruns | intermediate (frozen upstream) |
| `results/milestone_b9/<top-level files>` | 15 | 1.95 MB | .csv:10 .md:2 .json:2 .npz:1 | frozen stage outputs: summaries, manifests, frozen models, calibration | yes (downstream stages/audit) | final |
| `results/milestone_b9/figures/**` | 20 | 1.56 MB | .pdf:10 .png:10 | development figures (figures) | no | intermediate (figures) |
| `results/milestone_b9_smoke/<top-level files>` | 15 | 0.28 MB | .csv:10 .md:2 .json:2 .npz:1 | debug/smoke run outputs | no | debugging/development |
| `results/milestone_b9_smoke/figures/**` | 20 | 1.65 MB | .pdf:10 .png:10 | debug/smoke run outputs | no | debugging/development |
| `results/milestone_b9_stderr.log` | 1 | 0.0 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_b9_stdout.log` | 1 | 0.1 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_c1/<top-level files>` | 37 | 7.26 MB | .pdf:12 .png:12 .csv:9 .md:3 .json:1 | frozen stage outputs: summaries, manifests, frozen models, calibration | yes (downstream stages/audit) | final |
| `results/milestone_c1/calibration_predictions/**` | 500 | 8.06 MB | .npz:500 | per-episode frozen artifacts (calibration_predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c1/decision_snapshots/**` | 1000 | 0.44 MB | .json:1000 | per-episode frozen artifacts (decision_snapshots) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c1/predictions/**` | 1000 | 26.59 MB | .npz:1000 | per-episode frozen artifacts (predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c1/revealed_calibration_truth/**` | 500 | 3.68 MB | .npz:500 | per-episode frozen artifacts (revealed_calibration_truth) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c1/revealed_final_truth/**` | 1000 | 7.36 MB | .npz:1000 | per-episode frozen artifacts (revealed_final_truth) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c1/revealed_screening_truth/**` | 600 | 4.42 MB | .npz:600 | per-episode frozen artifacts (revealed_screening_truth) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c1/screening_predictions/**` | 600 | 9.71 MB | .npz:600 | per-episode frozen artifacts (screening_predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c15c/<top-level files>` | 7 | 1.01 MB | .csv:4 .md:2 .json:1 | multi-action screening summaries (raw shards absent) | no | superseded/unused (manual review) |
| `results/milestone_c15c/figures/**` | 6 | 0.26 MB | .png:6 | development figures (figures) | no | obsolete experiment |
| `results/milestone_c16a/<top-level files>` | 4 | 1.29 MB | .md:2 .csv:2 | frozen stage outputs: summaries, manifests, frozen models, calibration | yes (downstream stages/audit) | final |
| `results/milestone_c16a/figures/**` | 4 | 0.31 MB | .png:4 | development figures (figures) | no | intermediate (figures) |
| `results/milestone_c16b/<top-level files>` | 14 | 2.22 MB | .csv:8 .json:3 .md:2 .npz:1 | frozen stage outputs: summaries, manifests, frozen models, calibration | yes (downstream stages/audit) | final |
| `results/milestone_c16b/calibration_predictions/**` | 400 | 6.52 MB | .npz:400 | per-episode frozen artifacts (calibration_predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c16b/figures/**` | 6 | 0.31 MB | .png:6 | development figures (figures) | no | intermediate (figures) |
| `results/milestone_c16b/final_decisions/**` | 600 | 0.14 MB | .json:600 | per-episode frozen artifacts (final_decisions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c16b/final_predictions/**` | 600 | 9.78 MB | .npz:600 | per-episode frozen artifacts (final_predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c16b/revealed_calibration/**` | 400 | 0.72 MB | .npz:400 | per-episode frozen artifacts (revealed_calibration) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c16b/revealed_final_test/**` | 600 | 1.08 MB | .npz:600 | per-episode frozen artifacts (revealed_final_test) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c16b/revealed_selection/**` | 60 | 0.11 MB | .npz:60 | per-episode frozen artifacts (revealed_selection) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c16b/selection_predictions/**` | 60 | 0.11 MB | .npz:60 | per-episode frozen artifacts (selection_predictions) | read by audit/rerun code | raw data (frozen evidence) |
| `results/milestone_c16b_stderr.log` | 1 | 0.0 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_c16b_stdout.log` | 1 | 0.1 kB | .log:1 | raw console dump | no | debugging/development |
| `results/milestone_c1_5a/<top-level files>` | 25 | 2.74 MB | .pdf:8 .png:8 .csv:6 .md:2 .json:1 | frozen stage outputs: summaries, manifests, frozen models, calibration | yes (downstream stages/audit) | final |
| `results/second_case_final/<top-level files>` | 14 | 4.30 MB | .csv:8 .json:3 .md:2 .npz:1 | two-tank second-process outputs | final audit only | superseded by flash drum (manual review) |
| `results/second_case_final/calibration_predictions/**` | 300 | 2.23 MB | .npz:300 | per-episode frozen artifacts (calibration_predictions) | no | obsolete experiment |
| `results/second_case_final/figures/**` | 7 | 0.35 MB | .png:7 | development figures (figures) | no | obsolete experiment |
| `results/second_case_final/final_decisions/**` | 600 | 0.14 MB | .json:600 | per-episode frozen artifacts (final_decisions) | no | obsolete experiment |
| `results/second_case_final/final_predictions/**` | 600 | 4.46 MB | .npz:600 | per-episode frozen artifacts (final_predictions) | no | obsolete experiment |
| `results/second_case_final/k1/**` | 54 | 0.40 MB | .npz:54 | per-episode frozen artifacts (k1) | no | obsolete experiment |
| `results/second_case_final/k2/**` | 54 | 0.40 MB | .npz:54 | per-episode frozen artifacts (k2) | no | obsolete experiment |
| `results/second_case_final/k3/**` | 54 | 0.40 MB | .npz:54 | per-episode frozen artifacts (k3) | no | obsolete experiment |
| `results/second_case_final/revealed_calibration/**` | 300 | 0.74 MB | .npz:300 | per-episode frozen artifacts (revealed_calibration) | no | obsolete experiment |
| `results/second_case_final/revealed_final_test/**` | 600 | 1.47 MB | .npz:600 | per-episode frozen artifacts (revealed_final_test) | no | obsolete experiment |
| `results/second_case_final/revealed_selection/**` | 54 | 0.13 MB | .npz:54 | per-episode frozen artifacts (revealed_selection) | no | obsolete experiment |
| `results/second_case_setup/<top-level files>` | 6 | 6.70 MB | .md:5 .csv:1 | two-tank second-process outputs | final audit only | superseded by flash drum (manual review) |
| `results/second_case_setup/figures/**` | 4 | 0.41 MB | .png:4 | development figures (figures) | no | obsolete experiment |
| `tests/__pycache__/*` | 32 | 0.53 MB | .pyc:32 | Python bytecode / pytest cache | no | cached |
| `tests/test_e6b_margin_ladder.py` | 1 | 4.5 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_e_flash_ress_isolation.py` | 1 | 2.8 kB | .py:1 | pytest module | n/a | development-layout specific (frozen dev results inventory) |
| `tests/test_flash_conformal.py` | 1 | 5.3 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_flash_discrepancy.py` | 1 | 9.3 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_flash_e2_design.py` | 1 | 7.0 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_flash_e4_calibration.py` | 1 | 3.9 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_flash_e5_final.py` | 1 | 5.4 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_flash_e6_boundary.py` | 1 | 3.9 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_flash_e7_structural.py` | 1 | 2.1 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_flash_physics.py` | 1 | 7.9 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_latent_discrepancy_state_space.py` | 1 | 2.8 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_milestone_a.py` | 1 | 4.6 kB | .py:1 | pytest module | n/a | obsolete experiment |
| `tests/test_milestone_b.py` | 1 | 3.5 kB | .py:1 | pytest module | n/a | obsolete experiment |
| `tests/test_milestone_b10_latent_uncertainty.py` | 1 | 2.3 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_milestone_b11_parameter_uncertainty.py` | 1 | 1.3 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_milestone_b12_conformal_trajectories.py` | 1 | 1.1 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_milestone_b5.py` | 1 | 2.7 kB | .py:1 | pytest module | n/a | mixed: asserts dev-layout file hashes (bare filenames) |
| `tests/test_milestone_b5_calibration_analysis.py` | 1 | 2.9 kB | .py:1 | pytest module | n/a | obsolete experiment |
| `tests/test_milestone_b6_recursive_instability.py` | 1 | 2.8 kB | .py:1 | pytest module | n/a | obsolete experiment |
| `tests/test_milestone_b7a_stability_conditioned.py` | 1 | 2.2 kB | .py:1 | pytest module | n/a | obsolete experiment |
| `tests/test_milestone_b7b_stable_arx.py` | 1 | 1.5 kB | .py:1 | pytest module | n/a | obsolete experiment |
| `tests/test_milestone_b8_observable_conditioning.py` | 1 | 1.6 kB | .py:1 | pytest module | n/a | obsolete experiment |
| `tests/test_milestone_b9_latent_discrepancy.py` | 1 | 1.2 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_milestone_c1_5a_frozen_audit.py` | 1 | 0.8 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_milestone_c1_6a_additional_faults.py` | 1 | 1.2 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_milestone_c1_6b_multifault_revalidation.py` | 1 | 1.0 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_milestone_c1_safety_certification.py` | 1 | 3.2 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_milestone_d1_second_process_setup.py` | 1 | 2.8 kB | .py:1 | pytest module | n/a | superseded by flash drum |
| `tests/test_milestone_d2_second_process_revalidation.py` | 1 | 2.2 kB | .py:1 | pytest module | n/a | superseded by flash drum |
| `tests/test_observable_conditioned_arx.py` | 1 | 1.4 kB | .py:1 | pytest module | n/a | obsolete experiment |
| `tests/test_ress_e7_final.py` | 1 | 3.5 kB | .py:1 | pytest module | n/a | final (test) |
| `tests/test_stable_arx_residual_dynamics.py` | 1 | 2.3 kB | .py:1 | pytest module | n/a | final (test) |
| `twingaurd/__init__.py` | 1 | 0.1 kB | .py:1 | package marker for the CSTR development package | no (entry point or unused) | final (packaging) |
| `twingaurd/__pycache__/*` | 28 | 1.35 MB | .pyc:28 | Python bytecode / pytest cache | no | cached |
| `twingaurd/aggregate_milestone_c1_5c.py` | 1 | 2.2 kB | .py:1 | one-off aggregation of C1.5C shards (module-level code; shards absent) | no (entry point or unused) | debugging/development |
| `twingaurd/bayesian_residual_dynamics.py` | 1 | 10.3 kB | .py:1 | conjugate Bayesian linear residual-dynamics model (development baseline); imported by the final chain through b5/counterfactual_rollout | yes: twingaurd/counterfactual_rollout.py, twingaurd/milestone_b5_posterior_calibration.py, twingaurd/milestone_b6_recursive_instability.py, twingaurd/milestone_b7a_stability_conditioned_diagnostics.py, twingaurd/milestone_b_residual_dynamics.py, twingaurd/observable_conditioned_arx.py ... | final-dependency (legacy baseline) |
| `twingaurd/counterfactual_rollout.py` | 1 | 6.3 kB | .py:1 | recursive posterior rollout of Bayesian residual model; imported by b5 | yes: twingaurd/milestone_b5_posterior_calibration.py, twingaurd/milestone_b_residual_dynamics.py | final-dependency (legacy baseline) |
| `twingaurd/cstr_anomaly_threshold_test.py` | 1 | 11.9 kB | .py:1 | online temperature-threshold detector used only by cstr_case.py (not a test) | yes: twingaurd/cstr_case.py | unrelated to paper |
| `twingaurd/cstr_case.py` | 1 | 93.2 kB | .py:1 | LLM-agent CSTR control case (langgraph/langchain/OpenAI); imports missing modules | no (entry point or unused) | unrelated to paper |
| `twingaurd/cstr_digital_twin.py` | 1 | 18.1 kB | .py:1 | CSTR simulator used as physical plant AND nominal digital twin; FaultConfig/FaultManager inject UA fouling and feed-temperature step (plant-side only) | yes: twingaurd/cstr_case.py, twingaurd/milestone_a5_diagnostic.py, twingaurd/milestone_a_minimal_probabilistic_revalidation.py, twingaurd/milestone_b5_posterior_calibration.py, twingaurd/milestone_c1_6a_additional_faults.py, twingaurd/milestone_c1_6b_multifault_revalidation.py | final |
| `twingaurd/latent_discrepancy_state_space.py` | 1 | 28.5 kB | .py:1 | scalar/diagonal linear-Gaussian latent discrepancy state-space model: Kalman filter, EM fitting, predictive rollout | yes: twingaurd/milestone_b10_latent_uncertainty.py, twingaurd/milestone_b11_parameter_uncertainty.py, twingaurd/milestone_b9_latent_discrepancy.py, twingaurd/milestone_c1_6b_multifault_revalidation.py, twingaurd/milestone_d2_second_process_revalidation.py | final |
| `twingaurd/milestone_a5_diagnostic.py` | 1 | 36.3 kB | .py:1 | Milestone A.5 residual diagnostics | no (entry point or unused) | obsolete experiment |
| `twingaurd/milestone_a_minimal_probabilistic_revalidation.py` | 1 | 29.1 kB | .py:1 | Milestone A constant-residual revalidation experiment | yes: twingaurd/milestone_a5_diagnostic.py, twingaurd/milestone_b_residual_dynamics.py | obsolete experiment |
| `twingaurd/milestone_b10_latent_uncertainty.py` | 1 | 44.0 kB | .py:1 | freezes latent model; EpisodeInputs; new test-input generation; variance diagnostics | yes: E6B_MARGIN_LADDER/run_e6b.py, E_FLASH_RESS/run_ress_e6.py, E_FLASH_RESS/run_ress_e7.py, twingaurd/milestone_b11_parameter_uncertainty.py, twingaurd/milestone_b12_conformal_trajectories.py, twingaurd/milestone_c1_5c_multi_action_screening.py ... | final (CSTR discrepancy freeze) |
| `twingaurd/milestone_b11_parameter_uncertainty.py` | 1 | 31.8 kB | .py:1 | whole-episode bootstrap ensemble of latent models (J=50) and predictive paths | yes: twingaurd/milestone_b12_conformal_trajectories.py, twingaurd/milestone_c1_6b_multifault_revalidation.py, twingaurd/milestone_c1_safety_certification.py | final (CSTR bootstrap) |
| `twingaurd/milestone_b12_conformal_trajectories.py` | 1 | 31.7 kB | .py:1 | episode-level whole-trajectory conformal calibration (B2 pooled; B2M Mondrian diagnostic) | yes: twingaurd/milestone_c1_6b_multifault_revalidation.py, twingaurd/milestone_c1_safety_certification.py, twingaurd/milestone_d2_second_process_revalidation.py | final (CSTR conformal) |
| `twingaurd/milestone_b5_calibration_analysis.py` | 1 | 38.6 kB | .py:1 | analysis of B5 calibration outputs | no (entry point or unused) | obsolete experiment |
| `twingaurd/milestone_b5_posterior_calibration.py` | 1 | 48.4 kB | .py:1 | CSTR episode configuration/template, hidden UA-fouling configuration, action-regime selection, Wilson interval; its own run() calibrates the Milestone-B Bayesian model | yes: E6B_MARGIN_LADDER/run_e6b.py, E_FLASH_RESS/run_ress_e6.py, E_FLASH_RESS/run_ress_e7.py, twingaurd/milestone_b10_latent_uncertainty.py, twingaurd/milestone_b11_parameter_uncertainty.py, twingaurd/milestone_b12_conformal_trajectories.py ... | final-dependency (library functions) + obsolete run() |
| `twingaurd/milestone_b6_recursive_instability.py` | 1 | 66.4 kB | .py:1 | recursive instability diagnostics; load_b5_episodes reused by b9/b10 | yes: twingaurd/milestone_b10_latent_uncertainty.py, twingaurd/milestone_b7a_stability_conditioned_diagnostics.py, twingaurd/milestone_b7b_stable_higher_order_arx.py, twingaurd/milestone_b8_observable_conditioning.py, twingaurd/milestone_b9_latent_discrepancy.py | final-dependency (one helper) + obsolete run() |
| `twingaurd/milestone_b7a_stability_conditioned_diagnostics.py` | 1 | 54.7 kB | .py:1 | stability-conditioned diagnostics | no (entry point or unused) | obsolete experiment |
| `twingaurd/milestone_b7b_stable_higher_order_arx.py` | 1 | 41.2 kB | .py:1 | ARX order study; _init_worker/_reconstruct_observations reused by b9/b10 | yes: twingaurd/milestone_b10_latent_uncertainty.py, twingaurd/milestone_b9_latent_discrepancy.py | final-dependency (helpers) + obsolete run() |
| `twingaurd/milestone_b8_observable_conditioning.py` | 1 | 52.3 kB | .py:1 | observable-conditioning study (outputs read as baselines by b9 run) | no (entry point or unused) | obsolete experiment |
| `twingaurd/milestone_b9_latent_discrepancy.py` | 1 | 48.5 kB | .py:1 | generates CSTR training population and fits/selects latent discrepancy model dimension | yes: twingaurd/milestone_b10_latent_uncertainty.py | final (CSTR discrepancy fitting) |
| `twingaurd/milestone_b_residual_dynamics.py` | 1 | 62.2 kB | .py:1 | Milestone B Bayesian residual-dynamics model comparison (M1-M4 = residual models, NOT action-conditioned) | no (entry point or unused) | obsolete experiment |
| `twingaurd/milestone_c1_5a_frozen_audit.py` | 1 | 30.6 kB | .py:1 | read-only reconstruction of C1 decisions + regime/interpolation diagnosis | no (entry point or unused) | final-supporting (audit) |
| `twingaurd/milestone_c1_5c_multi_action_screening.py` | 1 | 19.5 kB | .py:1 | multi-action screening (K candidate actions, pooled q+ rounded to 5.465); NOT action-conditioned conformal | yes: twingaurd/aggregate_milestone_c1_5c.py | superseded/unused by requested paper experiments (manual review) |
| `twingaurd/milestone_c1_6a_additional_faults.py` | 1 | 14.1 kB | .py:1 | feed-temperature-step fault definition + smoke test; _clean_fault used by c16b | yes: twingaurd/milestone_c1_6b_multifault_revalidation.py | final-supporting |
| `twingaurd/milestone_c1_6b_multifault_revalidation.py` | 1 | 37.1 kB | .py:1 | CSTR mixed-degradation (UA fouling OR feed-temperature step per episode) fault-blind transfer: retrain, bootstrap, calibrate, N=600 test | no (entry point or unused) | final (CSTR mixed-degradation) |
| `twingaurd/milestone_c1_safety_certification.py` | 1 | 49.7 kB | .py:1 | CSTR one-sided conformal safety certification: calibration n=500, prospective test N=1000 (primary CSTR result) | yes: E6B_MARGIN_LADDER/run_e6b.py, E_FLASH_RESS/audit_ress_final.py, E_FLASH_RESS/run_ress_e6.py, E_FLASH_RESS/run_ress_e7.py, twingaurd/milestone_c1_5c_multi_action_screening.py, twingaurd/milestone_c1_6b_multifault_revalidation.py ... | final (CSTR primary) |
| `twingaurd/milestone_d1_second_process_setup.py` | 1 | 23.7 kB | .py:1 | two-tank second-process setup | no (entry point or unused) | superseded by flash drum |
| `twingaurd/milestone_d2_second_process_revalidation.py` | 1 | 56.9 kB | .py:1 | two-tank second-process revalidation (only SCALE_FLOOR_M imported by final audit) | yes: E_FLASH_RESS/audit_ress_final.py | superseded by flash drum (manual review) |
| `twingaurd/observable_conditioned_arx.py` | 1 | 6.4 kB | .py:1 | observable-conditioned ARX (B.8 only) | yes: twingaurd/milestone_b8_observable_conditioning.py | obsolete experiment |
| `twingaurd/probabilistic_residual_model.py` | 1 | 4.7 kB | .py:1 | Milestone-A scalar Gaussian residual model; imported via counterfactual_rollout | yes: twingaurd/counterfactual_rollout.py, twingaurd/milestone_a5_diagnostic.py, twingaurd/milestone_a_minimal_probabilistic_revalidation.py, twingaurd/milestone_b_residual_dynamics.py | final-dependency (legacy baseline) |
| `twingaurd/stable_arx_residual_dynamics.py` | 1 | 14.1 kB | .py:1 | stability-conditioned ARX residual model; imported via b7b | yes: twingaurd/milestone_b7b_stable_higher_order_arx.py, twingaurd/milestone_b8_observable_conditioning.py, twingaurd/observable_conditioned_arx.py | final-dependency (legacy baseline) |
| `twingaurd/two_tank_process.py` | 1 | 4.9 kB | .py:1 | two-tank simulator | yes: twingaurd/milestone_d1_second_process_setup.py, twingaurd/milestone_d2_second_process_revalidation.py | superseded by flash drum (manual review) |
