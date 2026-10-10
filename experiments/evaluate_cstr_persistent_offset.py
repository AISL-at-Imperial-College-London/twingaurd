"""Retrospective persistent-offset baseline (Offset-C) for CSTR: nominal twin under the candidate action
plus the mean of the last 10 observed pre-decision residuals, with one-sided split-conformal
calibration on the primary calibration population
(recorded output: artifacts/retrospective_baselines/cstr_persistent_offset). Pass --primary-dir pointing at
a primary-evaluation output directory that contains the per-episode prediction and truth files.
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.baselines.cstr_persistent_offset", __doc__, "cstr_persistent_offset")
