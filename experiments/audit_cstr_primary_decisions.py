"""Read-only reconstruction of every CSTR primary certification decision from its stored prediction
(recorded output: artifacts/cstr/primary_decision_audit). Pass --primary-dir pointing at a primary-evaluation
output directory that contains the per-episode prediction, decision and truth files (for example
the output of evaluate_cstr_primary.py).
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.primary_decision_audit", __doc__, "cstr_primary_decision_audit")
