# Reproducibility

This directory holds the read-only audit of the frozen results and the provenance records that
link the released files to the frozen prospective runs.

| Path | Content |
|---|---|
| `run_reproducibility_audit.py` | Read-only audit. It never writes to `artifacts/` or `results/`. The report goes to `reports/<timestamp>/` (git-ignored). Exit status 1 if any check fails. |
| `checks/` | Check modules: provenance and hash manifests, seeds and populations, chronology, decision reconstruction, leakage, archive |
| `seed_manifest.csv` | Every experimental population: process, role, declared seed block, observed seeds, source file |
| `artifact_manifest.csv` | Every frozen file shipped in git: release path, original development path, SHA-256, size, byte form |
| `known_discrepancies.csv` | Recorded hash mismatches that are explained and reported, never silently accepted |
| `provenance/source_ledger.json` | For each module in `src/twinguard` and `tests/`: the original file, the frozen SHA-256 and the exact line-level edits made for the release |
| `provenance/file_map.csv.gz` | All 66,000+ released files (git and archive), with original path → release path, SHA-256 and byte form |
| `provenance/path_map.csv` | Directory renames (development → release) |
| `provenance/sanitized_ledgers.json` | Ledgers whose absolute author-machine paths were rewritten, with their original and sanitized SHA-256 |

## Running the audit

```bash
python reproducibility/run_reproducibility_audit.py              # ~3 min on a git checkout
python reproducibility/run_reproducibility_audit.py --archive    # ~15 min; requires the extracted archive
```

`paper/reproducibility_mapping.md` maps each manuscript claim to the check IDs.

## How the release relates to the frozen runs

The experiments were run, frozen and hash-recorded in a development repository whose directories
used internal milestone names (`E_FLASH_RESS/`, `results/milestone_c1/`, …). The release changes
names, not results.

**Data files** are copied byte-exact, with three byte-level adjustments, each proven by a hash:

- **LF restored.** The development copy had been converted to CRLF line endings by a Git setting
  after the files were frozen. Where a recorded frozen hash proves the frozen file used LF, the LF
  form is restored. This covers 25,583 files, including all flash decision records.
- **CRLF restored.** Where a recorded hash proves the frozen file used CRLF, that form is
  restored.
- **Paths sanitized.** Nine per-episode ledgers stored absolute paths from an author's machine.
  These were rewritten to repository-relative paths. Both the original and the sanitized SHA-256
  are recorded. The original digest can be checked by anyone who holds the private path prefix,
  whose SHA-256 is recorded.

**Source modules** were moved into the `twinguard` package. The edits are limited to:

- import statements;
- the location of frozen directories;
- resolving frozen hashes through `twinguard.utils.provenance` (so code that checks frozen hashes
  of moved files still works);
- removing the two-tank second-process section from the final audit (that study is not part of
  this release).

Every edit is listed in `provenance/source_ledger.json`. Reversing the edits reproduces the frozen
source bytes, which `source_provenance_reversal` checks on every audit run. No numerical constant
or algorithmic line was changed.

**Rerun checks performed for the release.** Two prospective reruns were compared with the frozen
or development outputs:

- The flash simulator validation regenerates the frozen outputs identically.
- A CSTR primary-certification smoke run gives cell-identical results (apart from timing columns)
  for the release code and the original development code.

## Software environment

Exact package versions used for the frozen runs were not recorded. The release was verified with:

- Python 3.13.1
- numpy 2.1.3
- scipy 1.17.1
- matplotlib 3.10.9
- pytest 9.0.3

Reruns on other versions may differ in the last floating-point digits. Monte Carlo seeds are fixed,
so differences, if any, are numerical rather than statistical.
