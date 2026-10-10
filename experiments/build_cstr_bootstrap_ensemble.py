"""Build the CSTR whole-episode bootstrap ensemble of latent discrepancy models
(recorded output: artifacts/cstr/bootstrap_ensemble).
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.bootstrap_ensemble", __doc__, "cstr_bootstrap_ensemble")
