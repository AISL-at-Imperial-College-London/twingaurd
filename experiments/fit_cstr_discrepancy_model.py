"""Fit the CSTR latent discrepancy model: generate the training population, run EM for each
latent dimension and select k (frozen output: artifacts/cstr/discrepancy_fit).

Requires the frozen archive: the stage reads the B5 episodes and the B7B/B8 baseline tables
(pass --source-dir/--b7b-dir/--b8-dir pointing at archive/artifacts/cstr/...).
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.fit_discrepancy_model", __doc__, "cstr_discrepancy_fit")
