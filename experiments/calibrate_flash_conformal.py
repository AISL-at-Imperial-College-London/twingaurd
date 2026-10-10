"""Recompute ensemble-size selection and the ordinary/boundary conformal calibration from the recorded
flash bootstrap ensemble (runs the calibration stage with --reuse-bootstrap).

Usage: python experiments/calibrate_flash_conformal.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.flash.bootstrap_and_calibrate",
        __doc__,
        [
            'calibration_data',
            'flash_calibration_scores.csv',
            'flash_conformal_config.json',
            'calibration_report.md',
            'calibration_artifact_hashes.txt',
        ],
        extra_arguments=['--reuse-bootstrap'],
    )
