"""Freeze the selected CSTR latent discrepancy model and run its variance diagnostics
(frozen output: artifacts/cstr/discrepancy_freeze). Requires the frozen archive (B5 episodes;
pass --source-dir archive/artifacts/cstr/episode_design).
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.freeze_discrepancy_model", __doc__, "cstr_discrepancy_freeze")
