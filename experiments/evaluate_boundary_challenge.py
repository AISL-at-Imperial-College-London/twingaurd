"""Boundary-region challenge for both processes: flash N=1200 nominal-DT boundary actions
(boundary calibration population) and CSTR N=300 nominal-DT boundary actions.
Pass --skip-cstr to run the flash population only. Requires the frozen archive, because the
stage verifies the flash primary per-episode artifacts before running.

Usage: python experiments/evaluate_boundary_challenge.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.evaluation.boundary_challenge",
        __doc__,
        [
        'e6_artifacts',
        'FLASH_BOUNDARY_RESULTS.csv',
        'CSTR_BOUNDARY_RESULTS.csv',
        'RISK_AVAILABILITY_RESULTS.csv',
        'RESS_E6_RESULTS.md',
        'e6_figures',
        'RESS_E6_ARTIFACT_HASHES.txt',
        ],
        needs_archive=True,
        extra_arguments=[],
    )
