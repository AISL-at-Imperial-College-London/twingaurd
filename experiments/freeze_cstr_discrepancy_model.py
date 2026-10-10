"""Freeze the selected CSTR latent discrepancy model and run its variance diagnostics
(recorded output: artifacts/cstr/discrepancy_freeze). Reads the CSTR episode population
(see run_cstr_episode_design.py).
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.freeze_discrepancy_model", __doc__, "cstr_discrepancy_freeze")
