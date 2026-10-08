# Files not carried into the TwinGuard release

Nothing was deleted. Every file listed here remains in the development repository; the release
was built as a separate copy. `RENAME_MANIFEST.csv` gives the action for every one of the 69,058
development files (bulk directories are aggregated into `dir/*` rows with file counts).

| Category | Files | Action in `RENAME_MANIFEST.csv` |
|---|---:|---|
| Unnecessary raw artifact (moved to the frozen archive, not deleted) | 65,773 | `LARGE_ARTIFACT_EXCLUDED` |
| Uncertain / manual review | 2,666 (+1 shipped file flagged) | `REQUIRES_REVIEW` |
| Obsolete experiment | 157 | `EXCLUDED` |
| Generated / cache | 109 | `EXCLUDED` |
| Debugging / development | 91 | `EXCLUDED` |
| Superseded duplicate | 3 | `EXCLUDED` |
| Unrelated to final paper | 2 | `EXCLUDED` |
| Security / privacy | 0 excluded (see below) | — |

## Obsolete experiment

Earlier residual-model studies that the final method superseded. They were established by reading
the code, and none of them is imported by any final experiment.

| Original path | Files | What it is |
|---|---:|---|
| `twingaurd/milestone_a_minimal_probabilistic_revalidation.py`, `milestone_a5_diagnostic.py` + `results/milestone_a`, `milestone_a5` | 2 + 30 | constant-residual Gaussian revalidation and its diagnostics |
| `twingaurd/milestone_b_residual_dynamics.py` + `results/milestone_b` | 1 + 32 | Bayesian residual-dynamics model comparison. Its "M1–M4" are residual models, unrelated to the flash M0–M4 labels. |
| `twingaurd/milestone_b5_calibration_analysis.py` | 1 | analysis of the B5 posterior calibration |
| `twingaurd/milestone_b7a_stability_conditioned_diagnostics.py` + `results/milestone_b7a` | 1 + 23 | stability-conditioned diagnostics (includes the 19.5 MB `innovation_samples.csv`) |
| `twingaurd/milestone_b8_observable_conditioning.py`, `observable_conditioned_arx.py` + figures/metadata in `results/milestone_b8` | 2 + 20 | observable-conditioning study. Its 9 baseline CSVs, read by the CSTR discrepancy-fit rerun, are in the archive. |
| `results/milestone_b6` | 20 | recursive-instability study outputs (includes the 143 MB `stability_samples.csv`). The B6 module itself is kept because the final chain imports one helper from it. |
| `results/milestone_b7b` figures/metadata | 18 | ARX order study plots. Its 6 baseline CSVs are in the archive, and the module is kept as a dependency. |
| `E_FLASH_RESS/e5_figures/05_pooled_vs_action_conditioned_coverage.png` | 1 | plots the unsupported M4 action-conditioned placeholder (zero-height bar). Not part of the TwinGuard method. |
| `tests/test_milestone_a.py`, `test_milestone_b.py`, `test_milestone_b5_calibration_analysis.py`, `test_milestone_b7a_stability_conditioned.py`, `test_milestone_b8_observable_conditioning.py`, `test_observable_conditioned_arx.py` | 6 | tests of the excluded modules |

## Superseded duplicate

| Original path | Reason |
|---|---|
| `twingaurd/__init__.py`, `E_FLASH_RESS/__init__.py`, `E6B_MARGIN_LADDER/__init__.py` | package markers and re-exports, replaced by the `twinguard` package `__init__` files |

The root-level CSTR modules in git `HEAD` and the `twingaurd/` copies are one code base, moved
during development. The `twingaurd/` version is used: it differs from `HEAD` only by relative
imports and two additive changes, documented in `REORGANIZATION_AUDIT.md`.

## Debugging / development

| Original path | Files | Reason |
|---|---:|---|
| `results/milestone_b9_smoke`, `milestone_b10_smoke`, `milestone_b10_debug`, `milestone_b11_debug`, `milestone_b12_debug` | 58 | smoke and debug reruns of final stages |
| `results/*_stdout.log`, `results/*_stderr.log` | 12 | raw console dumps |
| `prompts/milestone_*.txt` | 15 | coding-agent task instructions used during development (contain no secrets) |
| `twingaurd/aggregate_milestone_c1_5c.py` | 1 | one-off aggregation that runs at import and needs shard directories that are absent |
| `E_FLASH_RESS/IMPLEMENTATION_PLAN.md` | 1 | planning document from the start of the flash study |
| `tests/test_e_flash_ress_isolation.py` | 1 | asserts the exact development `results/` inventory. Its output-guard checks are covered by the release guard and audit. |
| `.gitignore`, `pyproject.toml`, `requirements.txt` | 3 | development packaging, replaced by release files |

## Generated / cache

| Original path | Files |
|---|---:|
| `__pycache__/`, `twingaurd/__pycache__/`, `tests/__pycache__/`, `E_FLASH_RESS/__pycache__/`, `E6B_MARGIN_LADDER/__pycache__/` | 105 |
| `.pytest_cache/` | 4 |

A matplotlib font cache (`E_FLASH_RESS/.matplotlib/fontlist-v3.11.0.json`) is listed in the E7
frozen-input manifest, but it no longer exists in the development repository.

## Unnecessary raw artifact (moved to the frozen archive)

These files are **not discarded**. They are staged under `archive/` (git-ignored) for deposition,
and their SHA-256 values stay in git. They are the per-episode files of the prospective
evaluations, plus non-paper figures that their stage scripts regenerate.

| Original path | Files | Release path (inside the archive) |
|---|---:|---|
| `E_FLASH_RESS/e5_artifacts/{predictions,decisions,truths}` | 3,600 | `artifacts/flash_study/e5_artifacts/` |
| `E_FLASH_RESS/e6_artifacts/{flash,cstr}` | 6,000 | `artifacts/flash_study/e6_artifacts/` |
| `E_FLASH_RESS/e7_artifacts/{structural,flash_reliability,cstr_reliability}` | 26,400 | `artifacts/flash_study/e7_artifacts/` |
| `E6B_MARGIN_LADDER/artifacts/{cstr_test,flash_calibration,flash_test}` | 17,496 | `artifacts/flash_margin_ladder/artifacts/` |
| `results/milestone_c1/` per-episode directories and development figures | 5,224 | `artifacts/cstr/primary_evaluation/` |
| `results/milestone_c16b/` per-episode directories and figures | 2,726 | `artifacts/cstr/mixed_degradation/` |
| `results/milestone_b12`, `b11`, `b10`, `b9`, `c1_5a`, `c16a`: per-episode directories and figures | 3,662 | `artifacts/cstr/<stage>/` |
| `results/milestone_b5` (all), `b7b`/`b8` baseline CSVs | 650 | `artifacts/cstr/{episode_design,stable_arx_order_study,observable_conditioning_study}/` (inputs to the CSTR discrepancy refit) |
| Non-paper stage figures: `E_FLASH_RESS/figures`, `e5_figures`, `e6_figures`, `e7_figures`; `E6B_MARGIN_LADDER/figures` | 15 | same directories. The paper figures are in `results/figures/`. |

## Uncertain / manual review (left in the development repository)

| Original path | Files | Why it needs a decision |
|---|---:|---|
| `twingaurd/two_tank_process.py`, `milestone_d1_second_process_setup.py`, `milestone_d2_second_process_revalidation.py`, `results/second_case_setup`, `results/second_case_final`, `tests/test_milestone_d1/d2_*` | 2,652 | Two-tank second process. It is superseded by the flash drum as the second case study, and it is not in the requested experiment list. The original final audit had a two-tank section, which is removed in the release (ledgered source edit). **Restore it if the manuscript reports the two-tank study.** |
| `twingaurd/milestone_c1_5c_multi_action_screening.py`, `results/milestone_c15c` | 14 | CSTR multi-action screening (K candidate actions per state). It is cited by `artifacts/cstr/final_summary/FINAL_CSTR_RESULTS.csv` rows R17–R19, but not in the requested experiment list. Its raw shards are absent from the development repository, and it uses a rounded q⁺ = 5.465. **Restore it if the manuscript reports multi-action screening.** |
| `artifacts/flash_study/FLASH_E3_MODEL_SELECTION.md` (shipped) | 1 | Narrative E3 report. Its bytes differ from the E7 freeze record, so it was edited after hashing. The machine-readable E3 outputs match their hashes. It is reported by the audit as a documented known discrepancy. |

## Unrelated to final paper

| Original path | What it is |
|---|---|
| `twingaurd/cstr_case.py` | LLM-agent CSTR control case (langgraph / langchain-openai / ollama, `load_dotenv`). It imports a missing `common.trace_logging` module, and no final code imports it. The only link to the paper is the 313 K temperature limit, which is restated in the CSTR certification code. |
| `twingaurd/cstr_anomaly_threshold_test.py` | temperature-threshold detector used only by `cstr_case.py` (not a test) |

## Security / privacy

- **No secrets found.** No API keys, tokens, passwords or `.env` files appear in the development
  working tree or in its git history. The search covered `sk-…`, `api_key`, `OPENAI_API_KEY`,
  `HF_TOKEN` and password patterns. The only credential-related code is the `load_dotenv()` call in
  the excluded `cstr_case.py`.
- **Machine paths removed.** Nine per-episode ledger CSVs contained about 53,500 absolute paths
  from a collaborator's machine, including the user name and cloud-storage folder. The files are
  kept, because they are the chronology and hash evidence, but the paths were rewritten to
  repository-relative form. The original and sanitized SHA-256 are recorded in
  `reproducibility/provenance/sanitized_ledgers.json`.
- **No other machine paths** occur in any released file. A scan of the release for user-profile
  paths is part of the release checks in `RELEASE_REPORT.md`.
