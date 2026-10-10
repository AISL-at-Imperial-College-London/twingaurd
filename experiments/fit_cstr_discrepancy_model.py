"""Fit the CSTR latent discrepancy model: generate the training population, run EM for each latent
dimension and select k (recorded output: artifacts/cstr/discrepancy_fit).

Reads the CSTR episode population and the ARX / observable-conditioning baseline tables; regenerate
them first with run_cstr_episode_design.py and pass --source-dir, --arx-dir and --observable-dir accordingly.
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.fit_discrepancy_model", __doc__, "cstr_discrepancy_fit")
