"""Build the flash bootstrap ensemble (100 accepted fits), select the ensemble size J on operational
model-selection inputs, and calibrate the ordinary and boundary conformal quantiles on their two
separate calibration populations.

Usage: python experiments/build_flash_bootstrap_and_calibrate.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.flash.bootstrap_and_calibrate",
        __doc__,
        [
            'flash_bootstrap_manifest.csv',
            'flash_bootstrap_models.json',
            'calibration_data',
            'flash_calibration_scores.csv',
            'flash_conformal_config.json',
            'calibration_report.md',
            'calibration_artifact_hashes.txt',
        ],
        extra_arguments=[],
    )
