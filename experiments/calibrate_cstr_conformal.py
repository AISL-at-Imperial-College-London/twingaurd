"""Calibrate the CSTR episode-level whole-trajectory conformal quantiles
(recorded output: artifacts/cstr/conformal_calibration).
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.conformal_calibration", __doc__, "cstr_conformal_calibration")
