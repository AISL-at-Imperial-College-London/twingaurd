"""Flash primary prospective evaluation (N=1200 ordinary actions) together with the internal
component ablation M0 nominal DT -> M1 discrepancy mean -> M2 raw bootstrap -> M3 conformal.

The frozen per-episode schema also has an 'M4_action_conditioned' column. It is a hard-coded
NOT CERTIFIED placeholder (unsupported) and is not part of the method.

Usage: python experiments/evaluate_flash_primary.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.flash.evaluate_primary",
        __doc__,
        [
        'e5_artifacts',
        'e5_artifact_audit.csv',
        'FLASH_FINAL_RESULTS.csv',
        'FLASH_FINAL_ABLATION.csv',
        'FLASH_E5_RESULTS.md',
        'e5_figures',
        'FLASH_E5_ARTIFACT_HASHES.txt',
        ],
        needs_archive=False,
        extra_arguments=[],
    )
