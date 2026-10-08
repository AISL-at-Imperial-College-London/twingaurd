"""Generate the flash training (900) and model-selection (180) populations, fit the latent
discrepancy model for k=1..3 by EM and apply the preregistered selection gates.

Usage: python experiments/fit_and_select_flash_discrepancy_model.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.flash.fit_discrepancy_model",
        __doc__,
        [
        'e3_data',
        'flash_model_selection.csv',
        'selected_flash_model.json',
        'FLASH_E3_MODEL_SELECTION.md',
        ],
        needs_archive=False,
        extra_arguments=[],
    )
