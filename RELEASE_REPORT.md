# TwinGuard release report

Prepared 2026-10-08 from the development repository `conformal_set_restoration` (branch `main`,
last commit `a8d166f`, plus untracked working-tree directories). **The development repository was
not modified.** A full path/size inventory was taken before and after the build: 69,058 files,
none added, removed or changed.

## 1. Size and file counts

| | Development repository | TwinGuard release (git content) | Frozen archive (separate deposit) |
|---|---:|---:|---:|
| Files | 69,058 (+ `.git`) | 347 | 65,773 |
| Size | 908.7 MB working tree + 354 MB `.git` ≈ **1.26 GB** | **102.1 MB** | 578.2 MB |
| Largest file | 143.2 MB | 12.8 MB | < 1 MB |

The git content breaks down as follows:

| Directory | Size | Contents |
|---|---:|---|
| `artifacts/` | 96.0 MB | frozen models, calibration, manifests, per-episode result tables and ledgers |
| `reproducibility/` | 3.5 MB | audit, provenance ledger, compressed file map |
| `results/` | 1.3 MB | paper tables and figures |
| code, tests, docs | ≈ 1.3 MB | |

No Git LFS is needed. The release does not carry the development git history.

## 2. Large files

### Removed from git

| File / tree | Size | Action |
|---|---:|---|
| `results/milestone_b6/stability_samples.csv` | 143.2 MB | excluded (obsolete B6 study) |
| `results/milestone_b7a/innovation_samples.csv` | 19.5 MB | excluded (obsolete B7a study) |
| `results/milestone_b/one_step_row_results.csv` | 10.4 MB | excluded (obsolete Milestone B) |
| Per-episode artifact trees: E7 (175.8 MB), E6B (118.8 MB), C1 (≈ 60 MB), B12 (≈ 54 MB), E6 (36.9 MB), E5 (29.2 MB), B11, B10, C1.6B, B5 | ≈ 578 MB | moved to the frozen archive. SHA-256 values stay in git. |

### Retained in git, and why

| File | Size | Reason |
|---|---:|---|
| `artifacts/flash_study/FLASH_RELIABILITY_RESULTS.csv` | 12.8 MB | per-episode evidence for the flash mixed-regime audit (3000 × 5 methods). The audit uses it for denominators, chronology and decision reconstruction. |
| `artifacts/flash_margin_ladder/FLASH_E6B_MARGIN_RESULTS.csv` | 6.1 MB | per-episode margin-ladder results |
| `artifacts/flash_study/FLASH_BOUNDARY_RESULTS.csv` | 5.8 MB | per-episode flash boundary results |
| `artifacts/flash_study/e3_data/training_method_inputs.npz` (+ `e3_data`, `e4_data`, 15 MB total) | 5.1 MB | frozen training and calibration inputs. E5 verifies their hashes before running, and E4 trains on them. |
| Frozen SHA-256 manifests (`RESS_E7_FROZEN_INPUT_HASHES.txt` 4.8 MB, `RESS_FINAL_ARTIFACT_HASHES.txt` 4.2 MB, …) | ≈ 12.5 MB | compact proof of the archived per-episode files |

Full table: `LARGE_FILE_DECISIONS.md`.

## 3. Scientific experiments preserved

Each reference value given for the release was located in its frozen file and matched exactly.
The values are not hard-coded anywhere: `results/tables/paper_headline_results.csv` is generated
by row selection from the frozen summaries, and the audit recomputes every summary row from
per-episode records.

| Experiment | Code | Frozen result | Reference check |
|---|---|---|---|
| CSTR primary prospective | `twinguard.cstr.safety_certification` | `artifacts/cstr/primary_evaluation/safety_summary.csv` | N 1000, certified 338 (33.8%), false-safe 0, coverage 0.957 ✔ |
| CSTR mixed degradation (fouling + feed temperature) | `twinguard.cstr.mixed_degradation` | `artifacts/cstr/mixed_degradation/MULTIFAULT_SUMMARY.csv` | N 600; nominal false-safe 265; TwinGuard false-safe 0, certified 141 (23.5%), coverage 0.948 ✔ |
| CSTR discrepancy fit → freeze → bootstrap → conformal | `twinguard.cstr.{fit_discrepancy_model, freeze_discrepancy_model, bootstrap_ensemble, conformal_calibration}` | `artifacts/cstr/{discrepancy_fit, discrepancy_freeze, bootstrap_ensemble, conformal_calibration}` | frozen models consumed unchanged ✔ |
| Flash simulator, nominal twin, parity, step responses | `twinguard.flash.validate_simulator` | `artifacts/flash_study/e1_validation/` | rerun reproduces the outputs identically ✔ |
| Flash discrepancy training and model selection | `twinguard.flash.fit_discrepancy_model` | `flash_model_selection.csv`, `selected_flash_model.json` | k = 3 (preregistered fallback) ✔ |
| Flash bootstrap, ordinary and boundary calibration | `twinguard.flash.bootstrap_and_calibrate` | `flash_bootstrap_models.json`, `flash_conformal_config.json` | J = 10; q₀.₉₅ ordinary 11.0486 (n 447), boundary 11.3499 (n 463) ✔ |
| Flash primary prospective + M0→M3 ablation | `twinguard.flash.evaluate_primary` | `FLASH_FINAL_ABLATION.csv` | N 1200; M3: 177 (14.75%), false-safe 5 (0.42%), coverage 0.940. M0 779/207, M1 672/103, M2 623/72 ✔ |
| Boundary challenge (both processes) | `twinguard.evaluation.boundary_challenge` | `RISK_AVAILABILITY_RESULTS.csv` | flash N 1200: unsafe 853, nominal 711/447 (37.25%), conformal 0. CSTR N 300: all unsafe, nominal 300 certified, conformal 0 ✔ |
| Margin ladder | `twinguard.evaluation.margin_ladder` | `artifacts/flash_margin_ladder/` | preregistered; verdict "method remains overly conservative" ✔ |
| Flash structural mismatch (actuator lag) | `twinguard.evaluation.structural_and_reliability` | `FLASH_STRUCTURAL_SUMMARY.csv` | reserved 600, complete 564, incomplete 36, certified 43, false-safe 0 ✔ |
| Secondary mixed-regime audits | same | `CSTR_RELIABILITY_SUMMARY.csv`, `FLASH_RELIABILITY_SUMMARY.csv` | CSTR N 3000: certified 1010, false-safe 0. Flash N 3000: certified 237, false-safe 6 ✔ |
| Final cross-experiment audit | `twinguard.evaluation.final_audit` (+ new `reproducibility/`) | `RESS_FINAL_AUDIT_DETAILS.csv` | reproduced in archive mode (§4) |

**Separation is preserved.** These remain distinct modules:

- plant simulators;
- the nominal twins;
- the degradation generators;
- the discrepancy models;
- bootstrap;
- conformal calibration;
- the certification rule;
- evaluator-only truth.

No population was merged. The fault-blind method code paths were not touched beyond import and
path lines (see the source ledger).

**What changed in the code.** 37 source modules and 23 test modules were moved one-to-one, with no
function-level splitting. Every edit is listed in `reproducibility/provenance/source_ledger.json`.
Reversing the edits reproduces the frozen source hashes, which is checked on every audit run. The
edits fall into four kinds:

- imports;
- frozen-directory locations;
- frozen-hash resolution for moved files;
- removal of the two-tank section from the final audit.

## 4. Tests and checks run

The verification environment was Windows 11, Python 3.13.1, numpy 2.1.3, scipy 1.17.1,
matplotlib 3.10.9 and pytest 9.0.3.

| Check | Result |
|---|---|
| Development-repo baseline test suite (read-only, before any change) | 175 passed, 9 failed. All 9 failures were caused by CRLF conversion, absolute collaborator paths, the moved modules, or absent C1.5C shards (see `REORGANIZATION_AUDIT.md` §8). |
| Import/compile check of all release modules | all 60 source/test modules import; 115 development tests + 4 new test modules collect |
| Release test suite (git checkout) | **201 passed, 7 skipped.** Six tests need the archive; one documents the superseded Milestone-B model. |
| Release test suite with the archive extracted (clean clone + archive overlay) | **207 passed, 1 skipped** (the superseded Milestone-B model test) |
| Reproducibility audit, git checkout | **304 / 304 PASS** (source provenance, 17 hash-manifest checks, seed separation over 165 population pairs, 36 completeness checks, 18 chronology checks, 202 decision/summary reconstructions, 27 leakage checks) |
| Reproducibility audit with the archive (`--archive`) | **369 checks: 367 PASS, 0 FAIL, 2 INFO**. Archive integrity of all 65,773 files; truth→decision hash binding; pre-truth key scans; and the original final-audit reconstruction re-run on the archived artifacts, reproducing the frozen record. The 2 INFO rows are the two NaN→0 issues already documented in the frozen final audit. |
| `results/tables` regenerated from artifacts | up to date |
| `results/figures` vs frozen figures | 10 / 10 byte-identical |
| Prospective rerun: flash simulator validation (seed 1601) | **identical** to frozen: 13 parity rows, 756 step-response rows, summary JSON |
| Prospective rerun: CSTR primary certification smoke run (3 episodes, 200 paths), release code vs original development code | 9 output tables **cell-identical** (timing columns excluded) |
| Privacy scan (user-profile paths, user names, cloud-folder names) over the git content and the archive | none, apart from coincidental hex substrings inside SHA-256 values |
| Secrets scan (API keys, tokens, `.env`) | none |

Full reruns of the evaluation populations (hours each) were **not** executed, as instructed. They
are available via `scripts/reproduce_main_results.py --rerun <stage>`, which compares the result
with the frozen summaries cell by cell.

## 5. Unresolved issues

1. **`artifacts/flash_study/FLASH_E3_MODEL_SELECTION.md` differs from its E7 freeze hash** in both
   LF and CRLF form. Its file time is the E7 freeze date, so it was edited after hashing. It is
   shipped unchanged and listed in `reproducibility/known_discrepancies.csv`. The audit reports
   it explicitly. The machine-readable E3 outputs it describes match their hashes.
2. **The E7 frozen-input manifest covers 34,222 development files, of which 12,182 are not
   released** (obsolete studies, debug runs, C1.5C, two-tank). The 9,300 entries that were already
   missing from the development repository are among them. As a result:
   - `run_ress_e7._verify_design` and the final audit's manifest check could not pass even in the
     development repository.
   - In the release they verify every released or archived entry, and report the development-only
     entries separately (a ledgered edit).
3. **Rerunning the boundary and structural/mixed-regime stages requires the archive**, because
   their frozen design verifies the earlier per-episode artifacts first. Re-fitting the CSTR
   discrepancy model from scratch also requires archived B5/B7B/B8 inputs.
4. **Exact package versions of the frozen runs were not recorded.** The requirements give lower
   bounds only; the verification environment is listed above.
5. **Report-text and figure annotations hard-code frozen numbers.** This applies to C1, C1.5A and
   the E6/E6B/final-audit verdict strings. Rerunning those stages with different outcomes would
   leave stale prose in the generated Markdown, although the CSV results are always recomputed.
   The code was left unchanged, as required.
6. **Minor code quirks preserved as frozen:**
   - E7 writes artifacts into doubled directories (`structural/structural/`).
   - C1.6B's nominal-twin latency is written as 0.0.
   - B5's `frozen_model_configuration` hashes bare file names (affects only the superseded B5 run).
7. **Windows path length.** The longest archived path is 126 characters relative to the repository
   root. On Windows, a clone whose root path is longer than about 130 characters hits the
   260-character limit when the archive is extracted. During validation this caused spurious
   file-not-found errors until the clone was moved to a shorter path. The limit is documented in
   the README.
8. **Legal/authorship placeholders.** `LICENSE` (MIT) and the `CITATION.cff` authors, affiliations
   and DOI are placeholders for the authors to confirm.

## 6. Files requiring manual review

| Item | Location | Decision needed |
|---|---|---|
| Two-tank second process (D1/D2) | development repository only | Include it if the manuscript reports two-tank transfer evidence. If restored, the final audit's two-tank section can be reinstated from the ledger. |
| C1.5C multi-action screening | development repository only | Include it if the manuscript reports multi-action screening. Its raw shards are absent, and it used a rounded q⁺. |
| `FLASH_E3_MODEL_SELECTION.md` | shipped | Confirm the post-freeze edit was editorial only. |
| `artifacts/cstr/final_summary/` | shipped | An authored closure summary with no generating script, kept because the flash comparison figure reads it. |
| Table numbering in `paper/result_mapping.md` | shipped | Check against the final manuscript. |
| Archive deposition | `archive/` (git-ignored, 578 MB) | Deposit it (e.g. Zenodo), add the DOI to `README.md` and `data/README.md`, then delete or move `archive/` before publishing. |
| `LICENSE`, `CITATION.cff` | shipped | Confirm the license and authors. |

## 7. Commands to reproduce the principal RESS results

```bash
pip install -e ".[test]"
pytest                                                     # unit, regression and integrity tests
python scripts/reproduce_main_results.py                   # audit + tables + figures + headline table (git only)
# after extracting the frozen archive at the repository root:
python scripts/reproduce_main_results.py --archive

# prospective reruns (fresh workspace under runs/, compared with the frozen summaries)
python scripts/reproduce_main_results.py --list
python scripts/reproduce_main_results.py --rerun flash-simulator cstr-primary-smoke
python scripts/reproduce_main_results.py --rerun cstr-primary cstr-mixed-degradation flash-primary --workers 6
python scripts/reproduce_main_results.py --rerun boundary structural-and-mixed-regime margin-ladder --workers 6   # needs archive
```

## 8. Deliverables

| File | Purpose |
|---|---|
| `REORGANIZATION_AUDIT.md` | pre-change audit with the full grouped inventory |
| `LARGE_FILE_DECISIONS.md` | decision for every large file and artifact tree |
| `RENAME_MANIFEST.csv` | old path → new path → action for all 69,058 development files (558 rows, bulk directories aggregated) |
| `EXCLUDED_FILES.md` | exclusions grouped by category |
| `README.md` | public README |
| `paper/result_mapping.md` | manuscript tables and figures → files, rows and scripts |
| `paper/reproducibility_mapping.md` | manuscript claims → audit check IDs |
| `reproducibility/` | audit, seed and artifact manifests, provenance ledger, known discrepancies |
| `RELEASE_REPORT.md` | this report |

**Before pushing to GitHub:**

1. Remove or move `archive/` (it is git-ignored, but large) and `runs/` if present.
2. Run `git init`, then `git add .` and check `git status`. The `.gitignore` already excludes the
   archived paths, caches and run workspaces.
