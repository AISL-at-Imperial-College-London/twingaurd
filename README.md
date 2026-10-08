# TwinGuard

**Reliability-aware revalidation and selective certification for degraded digital twins.**

TwinGuard implements action-specific counterfactual revalidation for degraded digital twins. It
uses three components:

- fault-blind discrepancy modelling;
- bootstrap predictive uncertainty;
- whole-trajectory conformal calibration.

The repository accompanies the RESS manuscript *"Reliability-Aware Revalidation of Degraded
Digital Twins Using Fault-Blind Discrepancy Modelling and Conformal Prediction."* It contains:

- the code;
- the frozen calibration artifacts;
- per-episode evaluation records and hash manifests;
- the audit used to check the results reported in the paper.

## 1. Overview

A digital twin calibrated on a healthy plant becomes unreliable once the plant degrades. TwinGuard
asks a narrow question before a proposed control action is executed: *can the degraded plant's
response to this action be certified safe over the whole prediction horizon, or should the
decision be withheld?*

The deployed method is **fault blind**. It never sees fault identity, fault severity, changed
physical parameters, or the future physical trajectory. It uses only:

- observed plant outputs;
- the fixed nominal twin's predictions;
- the candidate action.

Every evaluation in this repository is **prospective**. Each episode is processed in this order,
with each step stored and hashed before the next:

1. candidate action
2. prediction
3. certification decision
4. evaluator-only physical truth

Two processes are studied:

| Process | Plant / nominal twin | Degradation (plant only) |
|---|---|---|
| CSTR | non-isothermal CSTR; the twin is the same model with nominal parameters | heat-transfer (UA) fouling; feed-temperature step |
| Flash drum | benzene/toluene dynamic flash (Raoult VLE, NIST correlations); independent nominal twin | vapour-outlet and liquid-outlet valve degradation, feed-composition shift; structural stress: unmodelled vapour-valve actuator lag |

## 2. Method at a glance

1. **Discrepancy model.** A latent linear-Gaussian state-space model is fitted by EM to the
   residual *plant − nominal twin*. Its features are the action, the action change and the nominal
   twin outputs; no fault information enters. The model selection gates were preregistered.
   - Code: `twinguard.discrepancy`.
2. **Bootstrap predictive uncertainty.** A whole-episode bootstrap ensemble of discrepancy models
   is filtered on the observed history. It is propagated as coherent trajectory samples under the
   candidate action, giving a predictive centre and scale for every horizon step.
   - Code: `twinguard.uncertainty`, `twinguard.cstr.bootstrap_ensemble`.
3. **Whole-trajectory conformal calibration.** On a separate calibration population, each episode
   gets one score: the maximum over the horizon (and over safety directions) of the standardised
   error in the safety-relevant direction. The finite-sample quantile at rank
   ⌈(n+1)(1−α)⌉ inflates the predictive band into a conformal envelope.
   - Code: `twinguard.conformal`, `twinguard.cstr.safety_certification`.
4. **Certification rule.** An action is certified safe only if the entire conformal envelope stays
   inside the safety limits for the whole horizon. Otherwise it is not certified, and the decision
   is withheld rather than declared unsafe.
5. **Prospective evaluation.** The physical truth is generated only after the decision is stored.
   The evaluator then labels the outcome:
   - false-safe: certified, but physically unsafe;
   - availability: the fraction of actions certified;
   - coverage: whether the true trajectory stayed inside the envelope.

   Each population keeps its own denominator.

**What the guarantee is, and is not.** The conformal coverage statement is *population-marginal*.
It holds over the episode population the calibration set was drawn from, under exchangeability of
calibration and test episodes, as stated in the manuscript. It is **not**:

- a per-action safety probability;
- a guarantee under arbitrary distribution shift;
- a general claim of robustness to structural model error.

The boundary, mixed-degradation, mixed-regime and structural-mismatch populations are stress
tests that probe where the marginal assumptions hold or break.

## 3. Repository structure

```
README.md  LICENSE  CITATION.cff  pyproject.toml  requirements.txt
src/twinguard/
  plants/            cstr.py (CSTR plant; also the nominal CSTR twin with faults off), flash_drum.py,
                     flash_drum_valve_lag.py (plant-only structural mismatch)
  digital_twins/     flash_drum.py (independent nominal flash twin)
  scenarios/         flash_population_design.py (seed streams, evaluator-only faults, safety spec, action designs)
  discrepancy/       cstr_state_space.py, flash_state_space.py, baselines/ (earlier residual models)
  uncertainty/       flash_bootstrap.py
  conformal/         flash_conformal.py (scores, finite-sample quantile, envelope, certification rule)
  cstr/              CSTR pipeline stages (episode design, discrepancy fit/freeze, bootstrap,
                     conformal calibration, primary certification, decision audit, mixed degradation)
  flash/             flash pipeline stages (simulator validation, population reservation,
                     discrepancy fit/selection, bootstrap + calibration, primary evaluation)
  evaluation/        boundary challenge, structural mismatch + mixed-regime audits, margin ladder, final audit
  utils/             repository paths, output guards, frozen-artifact provenance resolver
experiments/         one entry point per experiment (reruns go to runs/, never to artifacts/)
scripts/             reproduce_main_results.py, generate_paper_tables.py, generate_paper_figures.py
artifacts/           frozen experiment directories: models, calibration, manifests, per-episode result tables
  cstr/  flash_study/  flash_margin_ladder/
results/
  tables/            paper-facing tables with semantic names (derived from artifacts/)
  figures/main/      figures used in the main text
  figures/appendix/  appendix / repository figures
reproducibility/     read-only audit, provenance ledger, seed and artifact manifests, checks/
data/                data availability, population manifest, frozen archive instructions
paper/               result_mapping.md, reproducibility_mapping.md
tests/               unit, regression and integrity tests
```

Inside `artifacts/`, file names are kept exactly as written by the frozen runs, because the frozen
SHA-256 manifests refer to them. `reproducibility/provenance/file_map.csv` maps every original
development path to its release path.

## 4. Installation

Requires Python ≥ 3.10.

```bash
git clone <repository-url> TwinGuard && cd TwinGuard
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[test]"                               # or: pip install -r requirements.txt
pytest                                                  # ~2 min; archive-only tests are skipped
```

Run all commands from the repository root. Line endings matter for the hash checks:
`.gitattributes` keeps frozen artifacts byte-exact, so do not re-save them with an editor that
rewrites line endings.

## 5. Reproducing the main results

**Verify the frozen results** (minutes):

```bash
python scripts/reproduce_main_results.py              # git checkout only
python scripts/reproduce_main_results.py --archive    # after extracting the frozen archive (see data/README.md)
```

This runs the reproducibility audit, checks that `results/tables` and `results/figures` match the
frozen outputs, and prints the headline table. The audit recomputes every reported count from the
per-episode records, checks seed separation and chronology, and verifies frozen hashes.

**Rerun experiments prospectively.** Each stage runs in a fresh workspace under `runs/` and is
compared cell by cell with the frozen outputs:

```bash
python scripts/reproduce_main_results.py --list
python scripts/reproduce_main_results.py --rerun flash-simulator cstr-primary-smoke   # minutes
python scripts/reproduce_main_results.py --rerun flash-primary cstr-primary --workers 6  # hours
```

Individual experiments can also be run directly, for example
`python experiments/evaluate_flash_primary.py --workspace runs/flash_primary`.

`paper/result_mapping.md` lists the exact file, row and script behind every table and figure.

## 6. Experimental populations

All populations are disjoint seed blocks, declared in frozen manifests before their truth was
generated. They are never merged. The full list, with declared and observed seed ranges, is in
`reproducibility/seed_manifest.csv`.

| Process | Population | Role | Episodes |
|---|---|---|---:|
| CSTR | discrepancy training / freeze diagnostics / bootstrap diagnostics | model fitting | 600 / 240 / 300 |
| CSTR | trajectory-conformal calibration (+ size diagnostic) | calibration | 500 (+150) |
| CSTR | one-sided safety calibration | calibration | 500 |
| CSTR | primary prospective test | test | 1000 |
| CSTR | mixed degradation: training / selection / calibration / test | transfer | 600 / 60 / 400 / 600 |
| CSTR | boundary-region challenge | stress test | 300 |
| CSTR | secondary mixed-regime audit (interpolation / boundary / extrapolation) | reliability | 3000 |
| CSTR | margin ladder | stress test | 1500 |
| Flash | training / model selection | model fitting | 900 / 180 |
| Flash | ordinary calibration / boundary calibration | calibration | 600 / 600 |
| Flash | primary prospective test (ordinary actions) | test | 1200 |
| Flash | boundary-region challenge | stress test | 1200 |
| Flash | structural mismatch (actuator lag) | stress test, descriptive | 600 |
| Flash | secondary mixed-regime audit (ordinary + boundary actions, F1–F3) | reliability | 3000 |
| Flash | margin ladder: calibration / test | stress test | 1500 / 2000 |

Flash episodes whose physical simulation leaves the simulator's validity domain are kept as
incomplete, never replaced, and counted as unsafe outcomes. The headline table reports the number
of complete trajectories for each flash population.

## 7. Reproducibility safeguards

- **Seed separation.** Every population has a declared seed block. The audit checks that observed
  seeds equal the declared block, with nothing missing, duplicated or extra, and that no two
  populations share a seed.
- **Prediction before truth.** Each episode's candidate, prediction, decision and truth files
  were written in that order. Each later file stores the SHA-256 of the earlier ones. The
  nanosecond timestamps recorded at write time are kept in the per-episode ledgers.
- **Fault blindness.** Method inputs, frozen models and feature sets contain no fault, severity or
  truth fields. Pre-truth artifacts are scanned for evaluator-only keys (with `--archive`).
- **Hash verification.** The frozen SHA-256 manifests, preregistration sidecars and design files
  are verified against the released bytes.
- **Source provenance.** Each module under `src/twinguard` is the frozen experiment source plus a
  machine-checked list of import, path and hash-resolution edits.
  `reproducibility/provenance/source_ledger.json` reverses these edits to reproduce the frozen
  source hashes.
- **Decision reconstruction.** Every summary table is recomputed from per-episode rows. With the
  archive, every decision is rebuilt from the stored prediction envelope.

See `reproducibility/README.md` and `paper/reproducibility_mapping.md`.

## 8. Generating paper figures

```bash
python scripts/generate_paper_figures.py                                        # verify against frozen figures
python scripts/generate_paper_figures.py --regenerate --workspace runs/figures  # redraw with the original plotting code
```

`results/figures/main/` contains four figures:

- the representative corrected flash counterfactual rollout;
- nominal-twin vs. true safety-margin geometry;
- conformal vs. true safety-margin geometry;
- CSTR vs. flash boundary behaviour.

`results/figures/appendix/` contains:

- margin-ladder false-safe frequency, certification availability and safe-action certification;
- healthy plant / nominal-twin parity;
- manipulated-action step responses;
- structural-mismatch diagnostics.

## 9. Data availability

All data are simulated. Every population can be regenerated from the code and its seeds.

- **This repository (git):** frozen models, calibration sets, configurations, hash manifests,
  per-episode result tables and ledgers (≈ 100 MB).
- **Frozen archive:** the ≈ 66,000 per-episode candidate, prediction, decision and truth files
  (≈ 580 MB). These prove prospective chronology for every reported decision, and are deposited
  separately (see `data/README.md`). Extract the archive at the repository root to enable
  `--archive` checks and the archive-dependent tests.

  On Windows, the longest archived path is 126 characters relative to the repository root. Keep
  the repository root path under about 130 characters, or enable Windows long-path support,
  before extracting.

## 10. Citation

If you use this code or its frozen results, please cite the manuscript (see `CITATION.cff`):

```
<Authors>. Reliability-Aware Revalidation of Degraded Digital Twins Using Fault-Blind Discrepancy
Modelling and Conformal Prediction. Reliability Engineering & System Safety (submitted).
```

Licensed under the MIT License (see `LICENSE`).
