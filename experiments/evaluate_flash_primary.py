"""Flash primary prospective evaluation (N=1200 ordinary actions) together with the internal component
ablation M0 nominal DT -> M1 discrepancy mean -> M2 raw bootstrap -> M3 conformal.

The per-episode schema also has an 'M4_action_conditioned' column: a fixed NOT CERTIFIED placeholder
for an unsupported action-conditioned calibration. It is not part of the method.

Usage: python experiments/evaluate_flash_primary.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.flash.evaluate_primary",
        __doc__,
        [
            'primary_evaluation_episodes',
            'primary_evaluation_ledger.csv',
            'flash_primary_episode_results.csv',
            'flash_primary_ablation.csv',
            'primary_evaluation_report.md',
            'primary_evaluation_figures',
            'primary_evaluation_artifact_hashes.txt',
        ],
        extra_arguments=[],
    )
