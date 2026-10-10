# Data availability

All data in this study are **simulated**. No plant measurements or external datasets are used.

Every episode is generated from three ingredients:

- the plant simulators in `src/twinguard/plants/`;
- the hidden-degradation generators (`twinguard.scenarios`, `twinguard.cstr.episode_design`);
- fixed integer seeds.

`reproducibility/seed_manifest.csv` lists each population with its role and its declared and
observed seed block. The seed blocks were reserved in preregistration and design files before any
truth was generated:

- `artifacts/flash_study/flash_population_manifest.csv`
- `artifacts/flash_study/flash_seed_manifest.csv`
- `artifacts/flash_study/population_preregistration.md`
- `artifacts/flash_study/boundary_challenge_design.json`
- `artifacts/flash_study/reliability_design.json`
- `artifacts/flash_margin_ladder/margin_ladder_seed_manifest.csv`
- the CSTR stage `configuration.json` files

## What is in this repository

Everything needed to verify the reported numbers without regenerating data:

- fitted discrepancy models, bootstrap ensembles and conformal calibration sets;
- the training and calibration inputs of the flash study (`artifacts/flash_study/model_selection_data`,
  `calibration_data`);
- per-episode result tables for every evaluated population;
- per-episode ledgers with the SHA-256 and write time of every candidate, prediction, decision and
  truth file;
- the SHA-256 manifests written by each stage.

## Per-episode files

The individual per-episode files of the prospective evaluations (candidate actions, predictive
envelopes, certification decisions and physical truths; ≈ 66,000 files, ≈ 580 MB) are not
included. They are not needed to reproduce the reported numbers. Rerunning a stage with its entry
point in `experiments/` regenerates them in a workspace under `runs/`.

## Regenerating populations

Any population can be regenerated from its seeds with the scripts in `experiments/` (see the main
README). Stages that build on earlier stages read the earlier stage's outputs:

- **CSTR discrepancy fit** (`experiments/fit_cstr_discrepancy_model.py`) reads the CSTR episode
  population written by `experiments/run_cstr_episode_design.py`. It also writes comparison tables
  against earlier residual-model studies (stable ARX order, observable conditioning); pass their
  result directories with `--arx-dir` and `--observable-dir`.
- **Retrospective baselines** and the **final audit** of the structural / mixed-regime stage read
  the per-episode files of the flash and CSTR primary evaluations, so rerun those stages first.
