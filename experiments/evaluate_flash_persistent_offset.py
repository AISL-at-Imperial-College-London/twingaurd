"""Retrospective persistent-offset baseline (Offset-C) for flash: nominal twin under the candidate action plus the
mean of the last 10 observed pre-decision residuals, fixed engineering scales, conformal calibration on the
frozen E4 ordinary calibration population (frozen output: artifacts/retrospective_baselines/flash_persistent_offset).
Requires the frozen archive.
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.baselines.flash_persistent_offset", __doc__, "flash_persistent_offset")
