"""Recompute ensemble-size selection and the ordinary/boundary conformal calibration from the
frozen flash bootstrap ensemble (runs the calibration stage with --reuse-bootstrap).

Usage: python experiments/calibrate_flash_conformal.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.flash.bootstrap_and_calibrate",
        __doc__,
        [
        'e4_data',
        'flash_calibration_scores.csv',
        'flash_conformal_config.json',
        'FLASH_E4_CALIBRATION.md',
        'FLASH_E4_ARTIFACT_HASHES.txt',
        ],
        needs_archive=False,
        extra_arguments=['--reuse-bootstrap'],
    )
