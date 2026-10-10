"""CSTR primary prospective evaluation: one-sided conformal calibration (n=500) and safety
certification of N=1000 prospective episodes (recorded output: artifacts/cstr/primary_evaluation).

Use --limit-episodes and --trajectories for a quick smoke run.
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.safety_certification", __doc__, "cstr_primary_evaluation")
