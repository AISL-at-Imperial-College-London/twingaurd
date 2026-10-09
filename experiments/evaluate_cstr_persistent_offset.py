"""Retrospective persistent-offset baseline (Offset-C) for CSTR: nominal twin under the candidate action plus the
mean of the last 10 observed pre-decision residuals, with one-sided split-conformal calibration on the frozen
C1 calibration population (frozen output: artifacts/retrospective_baselines/cstr_persistent_offset).
Pre-decision residual histories are rebuilt from the frozen seeds and cached. Requires the frozen archive.
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.baselines.cstr_persistent_offset", __doc__, "cstr_persistent_offset")
