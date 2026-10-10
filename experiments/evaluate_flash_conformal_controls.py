"""Retrospective flash comparison controls on the ordinary calibration (n=447 complete) and primary test
(N=1200) populations: C0 and C1 with fixed engineering scales, next to the reconstructed M0-M3 results
(recorded output: artifacts/retrospective_baselines/flash_conformal_controls). Calibration futures are
regenerated from their seeds and verified against the recorded calibration inputs. Needs the
per-episode files of the primary evaluation in the workspace (rerun evaluate_flash_primary.py first).

Usage: python experiments/evaluate_flash_conformal_controls.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.baselines.flash_conformal_controls",
        __doc__,
        [
        ],
        extra_arguments=[],
    )
