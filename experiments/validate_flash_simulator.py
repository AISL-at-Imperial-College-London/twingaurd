"""Flash-drum simulator validation: healthy plant / independent nominal-twin parity,
manipulated-action directional step responses, long-horizon convergence and VLE checks.

Usage: python experiments/validate_flash_simulator.py --workspace runs/<new-directory> [stage options]
"""

from _runner import flash_stage

if __name__ == "__main__":
    flash_stage(
        "twinguard.flash.validate_simulator",
        __doc__,
        [
        'e1_validation',
        'figures',
        'FLASH_E1_VALIDATION.md',
        ],
        needs_archive=False,
        extra_arguments=[],
    )
