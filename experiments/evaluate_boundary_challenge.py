"""Boundary-region challenge for both processes: flash N=1200 nominal-DT boundary actions (boundary
calibration population) and CSTR N=300 nominal-DT boundary actions. Pass --skip-cstr to run the flash
population only.

Usage: python experiments/evaluate_boundary_challenge.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.evaluation.boundary_challenge",
        __doc__,
        [
            'boundary_challenge_episodes',
            'flash_boundary_episode_results.csv',
            'cstr_boundary_episode_results.csv',
            'boundary_risk_availability.csv',
            'boundary_challenge_report.md',
            'boundary_challenge_figures',
            'boundary_challenge_artifact_hashes.txt',
        ],
        extra_arguments=[],
    )
