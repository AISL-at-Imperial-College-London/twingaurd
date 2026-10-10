"""Flash-drum simulator validation: healthy plant / independent nominal-twin parity, manipulated-action
directional step responses, long-horizon convergence and VLE checks.

Usage: python experiments/validate_flash_simulator.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.flash.validate_simulator",
        __doc__,
        [
            'simulator_validation',
            'figures',
            'simulator_validation_report.md',
        ],
        extra_arguments=[],
    )
