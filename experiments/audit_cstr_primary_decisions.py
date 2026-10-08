"""Read-only reconstruction of every CSTR primary certification decision from the frozen
prediction files (frozen output: artifacts/cstr/primary_decision_audit).

Requires the frozen archive extracted at the repository root, so that
artifacts/cstr/primary_evaluation contains predictions/, decision_snapshots/ and
revealed_final_truth/.
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.primary_decision_audit", __doc__, "cstr_primary_decision_audit")
