# Flash E5 Untouched Final Results

## Frozen Prospective Design

All 1,200 reserved ordinary-final seeds were evaluated with the frozen E3/E4
artifacts. Prediction NPZ and decision JSON artifacts were exclusively written
and hashed before the corresponding plant future was simulated. Artifact keys
exclude fault metadata, future plant truth, and safety labels. Chronology passed
for every episode. No fit, bootstrap model, calibration score, quantile, safety
limit, or decision threshold changed in E5.

M2 is the raw one-sided 95% Gaussian bootstrap envelope (`z=1.6448536269514722`)
without conformal inflation. M3 uses the ordinary pooled `q=11.04859850832729`.
M4 is unsupported and always NOT CERTIFIED because E2 froze no Mondrian bins.

## Reliability

Actual unsafe episodes under the complete-horizon definition: 617/1200.
Simulator-domain exits count as not physically safe because a complete in-domain
horizon is required. Their normalized severity uses the available prefix and is
therefore a lower bound. There were 292 incomplete truth horizons.

| Method | Certified SAFE | False-safe | Joint false-safe | Wilson 95% CI | Directional coverage |
|---|---:|---:|---:|---:|---:|
| M0_nominal_DT | 779 (0.649) | 207 (0.172) | [0.152, 0.195] | 0.000 |
| M1_discrepancy_mean | 672 (0.560) | 103 (0.086) | [0.071, 0.103] | 0.000 |
| M2_raw_bootstrap | 623 (0.519) | 72 (0.060) | [0.048, 0.075] | 0.241 |
| M3_pooled_conformal | 177 (0.147) | 5 (0.004) | [0.002, 0.010] | 0.940 |
| M4_action_conditioned | 0 (0.000) | 0 (0.000) | [0.000, 0.003] |  |

For M3, conditional false-safe is 0.0282,
truly-safe-action certification is 0.2950,
and NOT CERTIFIED is 0.8525. False-safe maximum normalized
severity is 1.0232; total severity is
1.6640. Total integrated false-safe excursion is
346.0094 normalized-seconds, with mean
69.2019 and maximum
258.8630.

## Prediction And Runtime

M3 complete-pair RMSE is 0.03355 bar pressure,
0.03685 level fraction, and
1.4181 K temperature. Median online latency is
1.8061 s. The subgroup rows in
`FLASH_FINAL_ABLATION.csv` are descriptive only; they do not establish
conditional conformal guarantees.

## Frozen Gates

- Joint false-safe Wilson upper <=0.01: **PASS**.
- Certification rate >=0.10: **PASS**.
- E3 mean-model selection gate: **FAIL; preregistered k=3 fallback retained**.
- Mondrian extension: **UNSUPPORTED / NOT CERTIFIED**.

## Final Verdict

### FLASH TRANSFER VALIDATED WITH LIMITATIONS
