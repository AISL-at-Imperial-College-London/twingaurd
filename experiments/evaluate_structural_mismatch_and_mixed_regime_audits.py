"""Flash structural mismatch (N=600, plant-only vapour-valve actuator lag), followed by the secondary
mixed-regime reliability audits (flash N=3000, CSTR N=3000). The three populations are generated in
one run and kept as separate denominators.

Usage: python experiments/evaluate_structural_mismatch_and_mixed_regime_audits.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.evaluation.structural_and_reliability",
        __doc__,
        [
            'reliability_episodes',
            'flash_structural_episode_results.csv',
            'flash_structural_summary.csv',
            'flash_reliability_episode_results.csv',
            'flash_reliability_summary.csv',
            'cstr_reliability_episode_results.csv',
            'cstr_reliability_summary.csv',
        ],
        extra_arguments=[],
    )
