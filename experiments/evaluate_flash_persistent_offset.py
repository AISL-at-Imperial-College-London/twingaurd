"""Retrospective persistent-offset baseline (Offset-C) for flash: nominal twin under the candidate action
plus the mean of the last 10 observed pre-decision residuals, fixed engineering scales, conformal
calibration on the ordinary calibration population
(recorded output: artifacts/retrospective_baselines/flash_persistent_offset). Needs the per-episode
files of the primary evaluation in the workspace (rerun evaluate_flash_primary.py first).

Usage: python experiments/evaluate_flash_persistent_offset.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.baselines.flash_persistent_offset",
        __doc__,
        [
        ],
        extra_arguments=[],
    )
