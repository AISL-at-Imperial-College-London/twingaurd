"""Margin-ladder stress analysis over nominal-DT margin bands B0-B4: flash (calibration 1500, test 2000)
and CSTR (test 1500). Use --stage {calibration,flash-test,cstr-test,finalize,all}.

Usage: python experiments/evaluate_flash_margin_ladder.py --workspace runs/<new-directory> [stage options]
"""

from _runner import MARGIN_LADDER_ROOT, flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.evaluation.margin_ladder",
        __doc__,
        [
            'artifacts',
            'flash_margin_episodes.csv',
            'flash_margin_results.csv',
            'cstr_margin_episodes.csv',
            'cstr_margin_results.csv',
            'margin_ladder_risk_availability.csv',
            'margin_ladder_audit_checks.csv',
            'flash_margin_ladder_report.md',
            'cstr_margin_ladder_report.md',
            'margin_ladder_claims_report.md',
            'margin_ladder_artifact_hashes.txt',
            'figures',
        ],
        extra_arguments=[],
        root_variable="TWINGUARD_MARGIN_LADDER_ROOT",
        recorded_root=MARGIN_LADDER_ROOT,
    )
