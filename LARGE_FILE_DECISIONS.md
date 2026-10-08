# Large-file decisions

**Policy.** Ship generator code, fixed seeds, configuration and compact manifests in git. Do not
ship bulk generated data that can be regenerated. Frozen artifacts that prove prospective
chronology or reproduce published decisions are preserved in two ways:

- **In git**, as compact equivalents: per-episode hash/timestamp ledgers, hash manifests, and
  summary and per-episode result tables.
- **In a frozen archive**, as full per-episode files for deposition (e.g. Zenodo), staged by the
  release build.

**Git LFS is not recommended for any file.** No retained file exceeds 13 MB, and the bulk
artifact trees consist of tens of thousands of small files. LFS handles that pattern badly, and an
archived dataset with a DOI serves it better.

## Individual files > 10 MB

| Current file | Size | Purpose | Needed for paper? | Regenerable? | Recommended action | Reason |
|---|---:|---|---|---|---|---|
| `results/milestone_b6/stability_samples.csv` | 143.2 MB | posterior stability samples from the abandoned B6 instability study | no | yes (`milestone_b6` run) | EXCLUDE | obsolete experiment; already git-ignored in the development repository |
| `results/milestone_b7a/innovation_samples.csv` | 19.5 MB | innovation samples from the abandoned B7a study | no | yes | EXCLUDE | obsolete experiment |
| `E_FLASH_RESS/FLASH_RELIABILITY_RESULTS.csv` | 12.8 MB | per-episode results for the flash mixed-regime audit (3000 episodes × 5 methods) | **yes** (Table: mixed-regime audit) | only by a full E7 rerun (3000 episodes, ≈ hours) | KEEP in git (plain CSV) | primary per-episode evidence for a reported result. Read by the final audit and the release audit (duplicate/missing, chronology timestamps, decision consistency). Below GitHub's 50 MB warning threshold. |
| `results/milestone_b/one_step_row_results.csv` | 10.4 MB | Milestone-B one-step residual-model results | no | yes | EXCLUDE | obsolete experiment |

## Bulk per-episode artifact trees

These are prediction, decision and truth files, plus candidate files where the stage writes them.

| Current tree | Files | Size | Purpose | Needed for paper? | Regenerable? | Recommended action | Reason |
|---|---:|---:|---|---|---|---|---|
| `E_FLASH_RESS/e7_artifacts/{structural,flash_reliability,cstr_reliability}/` | 26,400 | 175.8 MB | structural-mismatch and mixed-regime episode artifacts | evidence for chronology and leakage checks | yes (E7 rerun, same seeds) | LARGE_ARTIFACT_EXCLUDED → frozen archive | per-episode hashes and nanosecond timestamps stay in git in the sanitized ledgers `*_ARTIFACT_AUDIT.csv` and in `RESS_FINAL_ARTIFACT_HASHES.txt` |
| `E6B_MARGIN_LADDER/artifacts/{cstr_test,flash_calibration,flash_test}/` | 17,496 | 118.8 MB | margin-ladder episode artifacts | evidence | yes (E6B rerun) | LARGE_ARTIFACT_EXCLUDED → archive | hashes are in `E6B_FINAL_ARTIFACT_HASHES.txt` and the sanitized ledgers |
| `results/milestone_c1/{predictions,decision_snapshots,revealed_final_truth,calibration_predictions,revealed_calibration_truth,screening_predictions,revealed_screening_truth}/` | 5,200 | ≈ 60 MB | CSTR primary episode artifacts | evidence (C1.5A reconstruction) | yes (C1 rerun) | LARGE_ARTIFACT_EXCLUDED → archive | per-episode results stay in `final_safety_episodes.csv`; the reconstruction audit stays in `primary_decision_audit/` |
| `E_FLASH_RESS/e6_artifacts/{flash,cstr}/` | 6,000 | 36.9 MB | boundary-challenge episode artifacts | evidence | yes (E6 rerun) | LARGE_ARTIFACT_EXCLUDED → archive | hashes are in `RESS_E6_ARTIFACT_HASHES.txt` and the sanitized ledgers |
| `E_FLASH_RESS/e5_artifacts/{predictions,decisions,truths}/` | 3,600 | 29.2 MB | flash primary episode artifacts | evidence | yes (E5 rerun) | LARGE_ARTIFACT_EXCLUDED → archive | hashes are in `FLASH_E5_ARTIFACT_HASHES.txt`. The decision JSONs are archived in their LF (frozen-hash) form. |
| `results/milestone_b12/*predictions*/`, `revealed_*` | 2,300 | ≈ 54 MB | CSTR conformal calibration episodes | no (upstream of C1) | yes | LARGE_ARTIFACT_EXCLUDED → archive | calibration scores and quantiles stay in git |
| `results/milestone_b11/*predictions*/`, `revealed_*` | 774 | ≈ 28 MB | bootstrap-ensemble validation episodes | no | yes | LARGE_ARTIFACT_EXCLUDED → archive | `bootstrap_models.json` and manifests stay in git |
| `results/milestone_b10/predictions/`, `revealed_test_truth/` | 480 | ≈ 19 MB | latent-model validation episodes | no | yes | LARGE_ARTIFACT_EXCLUDED → archive | the frozen model stays in git |
| `results/milestone_c16b/{calibration_predictions,final_predictions,final_decisions,revealed_*,selection_predictions}/` | 2,720 | ≈ 19 MB | CSTR mixed-degradation episode artifacts | evidence | yes (C1.6B rerun) | LARGE_ARTIFACT_EXCLUDED → archive | per-episode results stay in `MULTIFAULT_FINAL_EPISODES.csv` |
| `results/milestone_b5/{evaluations,predictions}/` (+ top-level CSVs) | 600 | ≈ 26 MB | B5 episodes, inputs to b9/b10 reruns | no | yes (B5 run) | LARGE_ARTIFACT_EXCLUDED → archive | only needed to re-fit the CSTR discrepancy model from scratch |
| `results/milestone_b7b/*.csv`, `results/milestone_b8/*.csv` | 20 | ≈ 10 MB | baseline tables read by the b9 run for comparison | no | yes | archive | only needed for a b9 rerun |
| `E_FLASH_RESS/e3_data/` | 5 | 8.3 MB | frozen flash training and model-selection inputs | **yes** (input to E4 bootstrap; hash-bound) | yes (E3 rerun ≈ 1080 episodes) | KEEP in git | needed so that E4/E5 can be rerun from git alone, and hash-verified |
| `E_FLASH_RESS/e4_data/` | 4 | 7.1 MB | frozen calibration operational inputs | **yes** (E5 verifies these hashes before running) | yes | KEEP in git | required for an E5 rerun from git |
| `results/milestone_c1/final_safety_episodes.csv` | 1 | 5.0 MB | per-episode CSTR primary results | **yes** | via a C1 rerun | KEEP in git | primary per-episode evidence |
| `E_FLASH_RESS/RESS_E7_FROZEN_INPUT_HASHES.txt`, `RESS_FINAL_ARTIFACT_HASHES.txt`, `E6B_FINAL_ARTIFACT_HASHES.txt`, `RESS_E6_ARTIFACT_HASHES.txt`, `FLASH_E5_ARTIFACT_HASHES.txt` | 5 | 12.5 MB | frozen SHA-256 manifests | **yes** (integrity) | no (they are the record) | KEEP in git, byte-exact (LF form where the recorded hash requires it) | compact proof of the archived artifacts' content |
| Development figures (`results/milestone_*/figures`, C1/C1.5A PNG/PDF, non-paper E-series PNGs) | ≈ 300 | ≈ 25 MB | exploratory and diagnostic plots | no | yes (stage scripts) | archive or EXCLUDE | paper figures are copied to `results/figures/` under semantic names |
| `.git` object store | — | 354 MB | development history | no | — | not carried over | the brief requires a fresh repository without development history |

## Archive vs. exclusion

- **Frozen archive** (staged under `archive/`, git-ignored). Per-episode artifacts of the
  experiments reported in the paper, plus the B5/B7B/B8 inputs needed to re-fit the CSTR
  discrepancy model. Every archived file keeps its release-relative path, so extracting the
  archive over the repository root restores the full layout. The release audit's `--archive`
  mode verifies it.
- **Excluded entirely** (remains only in the development repository):
  - obsolete milestone studies: A, A5, B, B5-analysis, B6, B7a, B8 outputs
  - smoke/debug runs and logs
  - the two-tank study and C1.5C (manual review)
  - the LLM-agent code and the prompts
