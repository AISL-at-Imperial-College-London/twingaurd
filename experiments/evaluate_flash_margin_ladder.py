"""Margin-ladder stress analysis over nominal-DT margin bands B0-B4: flash (calibration 1500,
test 2000) and CSTR (test 1500). Use --stage {calibration,flash-test,cstr-test,finalize,all}.

Usage: python experiments/evaluate_flash_margin_ladder.py --workspace runs/<new-directory> [stage options]
"""

from _runner import FROZEN_MARGIN_LADDER_ROOT, flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.evaluation.margin_ladder",
        __doc__,
        [
        'artifacts',
        'FLASH_E6B_EPISODES.csv',
        'FLASH_E6B_MARGIN_RESULTS.csv',
        'CSTR_E6B_EPISODES.csv',
        'CSTR_E6B_MARGIN_RESULTS.csv',
        'E6B_RISK_AVAILABILITY.csv',
        'E6B_ARTIFACT_AUDIT.csv',
        'FLASH_E6B_RESULTS.md',
        'CSTR_E6B_RESULTS.md',
        'E6B_CLAIMS_AUDIT.md',
        'E6B_FINAL_ARTIFACT_HASHES.txt',
        'figures',
        ],
        needs_archive=False,
        extra_arguments=[],
        root_variable="TWINGUARD_MARGIN_LADDER_ROOT",
        frozen_root=FROZEN_MARGIN_LADDER_ROOT,
    )
