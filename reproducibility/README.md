# Reproducibility

This directory holds the read-only audit of the released results.

| Path | Content |
|---|---|
| `run_reproducibility_audit.py` | Read-only audit. It never writes to `artifacts/` or `results/`. The report goes to `reports/<timestamp>/` (git-ignored). Exit status 1 if any check fails. |
| `checks/` | Check modules: file integrity and hash references, seeds and populations, chronology, decision reconstruction, fault-blindness |
| `seed_manifest.csv` | Every experimental population: process, role, declared seed block, observed seeds, source file |
| `artifact_manifest.csv` | Every file under `artifacts/` and `results/`: path, SHA-256 and size |

## Running the audit

```bash
python reproducibility/run_reproducibility_audit.py      # a few minutes
```

`paper/reproducibility_mapping.md` maps each manuscript claim to the check IDs.

## What is checked

- **File integrity.** Every released result file matches `artifact_manifest.csv`; every entry of
  the stage SHA-256 manifests matches its file; each preregistration / design sidecar
  (`*.sha256`) matches its document; every `*_sha256` value in a design or configuration file
  equals the hash of a released file.
- **Seed separation and population completeness.** Each population's observed seeds equal its
  declared seed block, with nothing missing, duplicated or extra, and no two populations of the same
  process share a seed.
- **Prediction before truth.** The candidate, prediction, decision and truth write times recorded
  in each ledger are strictly increasing, and every episode carries a positive chronology flag.
- **Decision consistency and summary reconstruction.** Every certification decision agrees with
  its recorded envelope and limits, and every summary table is recomputed from its per-episode
  rows.
- **Fault blindness.** Method inputs, fitted models and feature sets contain no fault, severity or
  truth fields, and evaluator-only columns are labelled as such.

## Rerun checks

Two prospective reruns were compared with the released outputs:

- The flash simulator validation (`python scripts/reproduce_main_results.py --rerun flash-simulator`)
  regenerates the released outputs identically.
- A CSTR primary-certification smoke run (`--rerun cstr-primary-smoke`) completes and writes the
  expected outputs.

## Software environment

Exact package versions used for the original runs were not recorded. The repository was verified
with:

- Python 3.13.1
- numpy 2.1.3
- scipy 1.17.1
- matplotlib 3.10.9
- pytest 9.0.3

Reruns on other versions may differ in the last floating-point digits. Monte Carlo seeds are fixed,
so differences, if any, are numerical rather than statistical.
