# Data availability

All data in this study are **simulated**. No plant measurements or external datasets are used.

Every episode is generated from three ingredients:

- the plant simulators in `src/twinguard/plants/`;
- the hidden-degradation generators (`twinguard.scenarios`, `twinguard.cstr.episode_design`);
- fixed integer seeds.

`reproducibility/seed_manifest.csv` lists each population with its role and its declared and
observed seed block. The seed blocks were reserved in frozen preregistration and design files
before any truth was generated:

- `artifacts/flash_study/flash_population_manifest.csv`
- `artifacts/flash_study/flash_seed_manifest.csv`
- `artifacts/flash_study/FLASH_E2_PREDECLARATION.md`
- `artifacts/flash_study/RESS_E6_DESIGN.json`
- `artifacts/flash_study/RESS_E7_DESIGN.json`
- `artifacts/flash_margin_ladder/E6B_SEED_MANIFEST.csv`
- the CSTR stage `configuration.json` files

## What is in git

The repository holds everything needed to verify the reported numbers without regenerating data:

- frozen discrepancy models, bootstrap ensembles and conformal calibration sets;
- the frozen training and calibration inputs of the flash study (`artifacts/flash_study/e3_data`,
  `e4_data`);
- per-episode result tables for every evaluated population;
- per-episode chronology and hash ledgers (`*_ARTIFACT_AUDIT.csv`, `e5_artifact_audit.csv`);
- the frozen SHA-256 manifests of all per-episode artifacts.

## Frozen per-episode archive

The individual per-episode files of the prospective evaluations are distributed as a separate
archive: candidate actions, predictive envelopes, certification decisions and physical truths,
≈ 66,000 files, ≈ 580 MB. They are not needed to reproduce the reported numbers, but they let
anyone:

- verify prediction-before-truth chronology file by file;
- rebuild every certification decision from its stored prediction;
- scan the pre-truth files for evaluator-only information.

> **TODO (authors):** deposit the archive (for example on Zenodo) and add its DOI and download
> link here and in `README.md`.

The archive uses repository-relative paths, so it is extracted at the repository root:

```bash
cd TwinGuard
unzip TwinGuard_frozen_archive.zip      # files land in artifacts/... (git-ignored)
python reproducibility/run_reproducibility_audit.py --archive
pytest                                  # archive-dependent tests now run instead of skipping
```

The SHA-256 of every archived file is recorded in git (`reproducibility/provenance/file_map.csv.gz`
and the frozen manifests), so a corrupted or altered archive is detected.

## Regenerating instead of downloading

Any population can be regenerated from its seeds with the scripts in `experiments/`, in a
workspace under `runs/` (see the main README). Two inputs come from the archive:

- **CSTR discrepancy-model refit** (`experiments/fit_cstr_discrepancy_model.py`,
  `freeze_cstr_discrepancy_model.py`) reads earlier B5 episodes and baseline tables, which are in
  the archive under `artifacts/cstr/episode_design/`, `stable_arx_order_study/` and
  `observable_conditioning_study/`.
- **Boundary and structural/mixed-regime stages** first verify the frozen per-episode artifacts
  of the earlier stages. This is part of their frozen design, so they also need the archive.
