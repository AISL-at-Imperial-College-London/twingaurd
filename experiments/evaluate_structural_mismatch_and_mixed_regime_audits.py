"""Flash structural mismatch (N=600, plant-only vapour-valve actuator lag), followed by the
secondary mixed-regime reliability audits (flash N=3000, CSTR N=3000). The three
populations are generated in one frozen run and kept as separate denominators. Requires the
frozen archive (the frozen-input manifest covers earlier per-episode artifacts).

Usage: python experiments/evaluate_structural_mismatch_and_mixed_regime_audits.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.evaluation.structural_and_reliability",
        __doc__,
        [
        'e7_artifacts',
        'FLASH_STRUCTURAL_RESULTS.csv',
        'FLASH_STRUCTURAL_SUMMARY.csv',
        'FLASH_RELIABILITY_RESULTS.csv',
        'FLASH_RELIABILITY_SUMMARY.csv',
        'CSTR_RELIABILITY_RESULTS.csv',
        'CSTR_RELIABILITY_SUMMARY.csv',
        ],
        needs_archive=True,
        extra_arguments=[],
    )
