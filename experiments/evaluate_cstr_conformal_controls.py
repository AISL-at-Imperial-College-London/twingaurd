"""Retrospective CSTR conformal controls on the frozen C1 calibration (n=500) and test (N=1000) populations:
nominal twin + constant-width one-sided conformal (C0) and discrepancy mean + constant-width conformal (C1),
next to the reconstructed M0-M3 results (frozen output: artifacts/retrospective_baselines/cstr_conformal_controls).
Requires the frozen archive extracted at the repository root.
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.baselines.cstr_conformal_controls", __doc__, "cstr_conformal_controls")
