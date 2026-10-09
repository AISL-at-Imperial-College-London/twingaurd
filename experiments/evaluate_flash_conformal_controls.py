"""Retrospective flash conformal controls on the frozen E4 ordinary calibration (n=447 complete) and E5 test (N=1200)
populations: C0 and C1 with fixed engineering scales, next to the reconstructed M0-M3 results
(frozen output: artifacts/retrospective_baselines/flash_conformal_controls). E4 calibration futures are
regenerated from their seeds and verified against the frozen E4 inputs. Requires the frozen archive.
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.baselines.flash_conformal_controls", __doc__, "flash_conformal_controls")
