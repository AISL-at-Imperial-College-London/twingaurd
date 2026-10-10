"""CSTR mixed-degradation fault-blind transfer. Each episode carries one hidden mechanism drawn from
{heat-transfer (UA) fouling, feed-temperature step}. Retrains the discrepancy model, rebuilds the
bootstrap ensemble, recalibrates (n=400) and evaluates N=600 prospective episodes
(recorded output: artifacts/cstr/mixed_degradation).
"""

from _runner import cstr_stage

if __name__ == "__main__":
    cstr_stage("twinguard.cstr.mixed_degradation", __doc__, "cstr_mixed_degradation")
