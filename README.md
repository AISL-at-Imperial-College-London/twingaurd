# TwinGuard

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![MIT License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Test suite: pytest](https://img.shields.io/badge/test%20suite-pytest-0A9EDC?logo=pytest&logoColor=white)](https://docs.pytest.org/)

**Reliability-aware revalidation and selective certification for degraded digital twins.**

TwinGuard implements action-specific counterfactual revalidation for degraded digital twins. It
uses three components:

- fault-blind discrepancy modelling;
- bootstrap predictive uncertainty;
- whole-trajectory conformal calibration.

This repository contains the TwinGuard implementation and its research materials:

- the code and tests;
- the calibration and evaluation artifacts;
- paper-facing tables, figures and reproducibility checks.

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

Every evaluation of TwinGuard in this repository is **prospective**. (The comparison baselines in
§5 are the exception: they were computed retrospectively.) Each episode is processed in this
order, with each step stored and hashed before the next:

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
README.md  LICENSE  pyproject.toml requirements.txt
src/twinguard/
  plants/            cstr.py (CSTR plant; also the nominal CSTR twin with faults off), flash_drum.py,
                     flash_drum_valve_lag.py (plant-only structural mismatch)
  scenarios/         flash_population_design.py (seed streams, evaluator-only faults, safety spec, action designs)
  discrepancy/       cstr_state_space.py, flash_state_space.py, baselines/ (earlier residual models)
  uncertainty/       flash_bootstrap.py
  conformal/         flash_conformal.py (scores, finite-sample quantile, envelope, certification rule)
  cstr/              CSTR pipeline stages (episode design, discrepancy fit/freeze, bootstrap,
                     conformal calibration, primary certification, decision audit, mixed degradation)
  flash/             nominal twin and pipeline stages (simulator validation, population reservation,
                     discrepancy fit/selection, bootstrap + calibration, primary evaluation)
  evaluation/        boundary challenge, structural mismatch + mixed-regime audits, margin ladder, final audit
  baselines/         retrospective comparison baselines (conformal controls C0/C1, persistent offset)
  utils/             repository paths, output guards
experiments/         one entry point per experiment (reruns go to runs/, never to artifacts/)
scripts/             reproduce_main_results.py, generate_paper_tables.py, generate_paper_figures.py
artifacts/           experiment outputs: models, calibration, manifests, per-episode result tables
  cstr/  flash_study/  flash_margin_ladder/  retrospective_baselines/
results/
  tables/            paper-facing tables with semantic names (derived from artifacts/)
  figures/main/      figures used in the main text
  figures/appendix/  appendix / repository figures
reproducibility/     read-only audit, seed and artifact manifests, checks/
data/                data availability
paper/               result_mapping.md, reproducibility_mapping.md
tests/               unit, regression and integrity tests
```

The SHA-256 of every released result file is recorded in `reproducibility/artifact_manifest.csv`.

## 4. Installation

Requires Python ≥ 3.10.

```bash
git clone https://github.com/AISL-at-Imperial-College-London/twinguard.git
cd twinguard
python -m venv .venv
```

Activate the environment using the command for your shell, then install the package and test
dependencies.

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

macOS / Linux:

```bash
source .venv/bin/activate
```

Install and run the tests:

```bash
pip install -e ".[test]"
pytest
```

Run all commands from the repository root. Line endings matter for the hash checks:
`.gitattributes` keeps the released artifacts byte-exact, so do not re-save them with an editor that
rewrites line endings.

## 5. Reproducing the main results

**Verify the released results** (minutes):

```bash
python scripts/reproduce_main_results.py
```

This runs the reproducibility audit, checks that `results/tables` and `results/figures` match the
released outputs, and prints the headline table. The audit recomputes every reported count from the
per-episode records, checks seed separation and chronology, and verifies the recorded hashes.

**Rerun experiments prospectively.** Each stage runs in a fresh workspace under `runs/` and is
compared cell by cell with the released outputs:

```bash
python scripts/reproduce_main_results.py --list
python scripts/reproduce_main_results.py --rerun flash-simulator cstr-primary-smoke   # minutes
python scripts/reproduce_main_results.py --rerun flash-primary cstr-primary --workers 6  # hours
```

Individual experiments can also be run directly, for example
`python experiments/evaluate_flash_primary.py --workspace runs/flash_primary`.

`paper/result_mapping.md` lists the exact file, row and script behind every table and figure.

### Retrospective comparison baselines

Three post-hoc controls were evaluated on the same calibration and test populations, after
the prospective study. They are **not** part of the TwinGuard method and **not** prospective
results.

| Control | Prediction | Calibration |
|---|---|---|
| C0 | nominal twin | constant-width conformal |
| C1 | discrepancy mean | constant-width conformal |
| Offset-C (persistent offset) | nominal twin under the candidate action + the mean of the last 10 observed pre-decision residuals | conformal |

Code: `src/twinguard/baselines/`. Entry points: `experiments/evaluate_*_conformal_controls.py` and
`experiments/evaluate_*_persistent_offset.py`. Outputs: `artifacts/retrospective_baselines/`.
Comparison table: `results/tables/retrospective_baseline_comparison.csv`. These runs read the
per-episode prediction and truth files of the primary evaluations, so rerun those stages first.

## 6. Experimental populations

All populations are disjoint seed blocks, declared in population manifests before their truth was
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
- **Fault blindness.** Method inputs, fitted models and feature sets contain no fault, severity or
  truth fields.
- **Hash verification.** Every released result file, the stage SHA-256 manifests, the
  preregistration sidecars and the hash references between design files are verified against the
  released bytes.
- **Decision reconstruction.** Every summary table is recomputed from per-episode rows.

See `reproducibility/README.md` and `paper/reproducibility_mapping.md`.

## 8. Generating paper figures

```bash
python scripts/generate_paper_figures.py                                        # verify against the recorded hashes
python scripts/generate_paper_figures.py --regenerate --workspace runs/figures  # redraw the simulator figures
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

- **This repository:** fitted models, calibration sets, configurations, hash manifests,
  per-episode result tables and ledgers (≈ 100 MB).
- **Per-episode files:** the ≈ 66,000 per-episode candidate, prediction, decision and truth files
  (≈ 580 MB) are not included. The experiment entry points regenerate them from their seeds; the
  ledgers record the SHA-256 and write time of each file. See `data/README.md`.

Licensed under the MIT License; see [LICENSE](LICENSE).
