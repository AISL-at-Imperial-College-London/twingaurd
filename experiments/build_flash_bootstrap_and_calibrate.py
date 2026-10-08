"""Build the flash bootstrap ensemble (100 accepted fits), select the ensemble size J on
operational model-selection inputs, and calibrate the ordinary and boundary conformal
quantiles on their two separate calibration populations.

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
        'e4_data',
        'flash_calibration_scores.csv',
        'flash_conformal_config.json',
        'FLASH_E4_CALIBRATION.md',
        'FLASH_E4_ARTIFACT_HASHES.txt',
        ],
        needs_archive=False,
        extra_arguments=[],
    )
