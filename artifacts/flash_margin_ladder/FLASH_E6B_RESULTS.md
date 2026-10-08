# Flash E6B Margin-Ladder Results

Analysis A is a prospective reliability characterization of the frozen flash
method with pooled `q=11.04859850832729`. Analysis B changes only calibration
conditioning and uses fresh band-matched E6B calibration seeds.

## Margin-Conditioned Calibration

- B0: n=236, q=10.068458
- B1: n=242, q=9.612979
- B2: n=252, q=11.827788
- B3: n=262, q=11.203470
- B4: n=275, q=12.046427

## Band Results

| Band | Method | Eligible | Safe / confirmed unsafe / incomplete | Certified | False-safe | Joint risk (Wilson 95% CI) | Conditional false-safe | Safe-action certification |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | M0_nominal_DT | 400 | 139 / 258 / 3 | 400 | 258 | 0.645 [0.597, 0.690] | 0.645 | 1.000 |
| B0 | M1_discrepancy_mean | 400 | 139 / 258 / 3 | 224 | 86 | 0.215 [0.178, 0.258] | 0.384 | 0.971 |
| B0 | M2_raw_bootstrap | 400 | 139 / 258 / 3 | 37 | 0 | 0.000 [0.000, 0.010] | 0.000 | 0.266 |
| B0 | M3_frozen_pooled | 400 | 139 / 258 / 3 | 0 | 0 | 0.000 [0.000, 0.010] | undefined | 0.000 |
| B0 | M5_margin_conditioned | 400 | 139 / 258 / 3 | 0 | 0 | 0.000 [0.000, 0.010] | undefined | 0.000 |
| B1 | M0_nominal_DT | 400 | 182 / 218 / 0 | 400 | 218 | 0.545 [0.496, 0.593] | 0.545 | 1.000 |
| B1 | M1_discrepancy_mean | 400 | 182 / 218 / 0 | 279 | 99 | 0.247 [0.208, 0.292] | 0.355 | 0.989 |
| B1 | M2_raw_bootstrap | 400 | 182 / 218 / 0 | 87 | 4 | 0.010 [0.004, 0.025] | 0.046 | 0.456 |
| B1 | M3_frozen_pooled | 400 | 182 / 218 / 0 | 0 | 0 | 0.000 [0.000, 0.010] | undefined | 0.000 |
| B1 | M5_margin_conditioned | 400 | 182 / 218 / 0 | 0 | 0 | 0.000 [0.000, 0.010] | undefined | 0.000 |
| B2 | M0_nominal_DT | 400 | 206 / 194 / 0 | 400 | 194 | 0.485 [0.436, 0.534] | 0.485 | 1.000 |
| B2 | M1_discrepancy_mean | 400 | 206 / 194 / 0 | 298 | 94 | 0.235 [0.196, 0.279] | 0.315 | 0.990 |
| B2 | M2_raw_bootstrap | 400 | 206 / 194 / 0 | 216 | 30 | 0.075 [0.053, 0.105] | 0.139 | 0.903 |
| B2 | M3_frozen_pooled | 400 | 206 / 194 / 0 | 0 | 0 | 0.000 [0.000, 0.010] | undefined | 0.000 |
| B2 | M5_margin_conditioned | 400 | 206 / 194 / 0 | 0 | 0 | 0.000 [0.000, 0.010] | undefined | 0.000 |
| B3 | M0_nominal_DT | 400 | 247 / 152 / 1 | 400 | 152 | 0.380 [0.334, 0.428] | 0.380 | 1.000 |
| B3 | M1_discrepancy_mean | 400 | 247 / 152 / 1 | 335 | 87 | 0.217 [0.180, 0.261] | 0.260 | 1.000 |
| B3 | M2_raw_bootstrap | 400 | 247 / 152 / 1 | 300 | 54 | 0.135 [0.105, 0.172] | 0.180 | 0.992 |
| B3 | M3_frozen_pooled | 400 | 247 / 152 / 1 | 0 | 0 | 0.000 [0.000, 0.010] | undefined | 0.000 |
| B3 | M5_margin_conditioned | 400 | 247 / 152 / 1 | 0 | 0 | 0.000 [0.000, 0.010] | undefined | 0.000 |
| B4 | M0_nominal_DT | 400 | 281 / 119 / 0 | 400 | 119 | 0.297 [0.255, 0.344] | 0.297 | 1.000 |
| B4 | M1_discrepancy_mean | 400 | 281 / 119 / 0 | 355 | 75 | 0.188 [0.152, 0.229] | 0.211 | 0.996 |
| B4 | M2_raw_bootstrap | 400 | 281 / 119 / 0 | 340 | 65 | 0.163 [0.130, 0.202] | 0.191 | 0.979 |
| B4 | M3_frozen_pooled | 400 | 281 / 119 / 0 | 1 | 0 | 0.000 [0.000, 0.010] | 0.000 | 0.004 |
| B4 | M5_margin_conditioned | 400 | 281 / 119 / 0 | 0 | 0 | 0.000 [0.000, 0.010] | undefined | 0.000 |

## Violation Severity

| Band | Confirmed unsafe mean / max severity | Frozen pooled false-safe mean / median / max | Frozen pooled integrated false-safe severity |
|---|---:|---:|---:|
| B0 | 1.196 / 3.459 | none observed | 0.000 |
| B1 | 1.279 / 3.590 | none observed | 0.000 |
| B2 | 1.476 / 3.578 | none observed | 0.000 |
| B3 | 1.546 / 3.549 | none observed | 0.000 |
| B4 | 1.268 / 3.479 | none observed | 0.000 |

Simulator-domain exits are reported separately as incomplete validation
failures unless a physical constraint was already crossed. They are never
silently converted into demonstrated unsafe outcomes.

## Interpretation

The operating-envelope verdict is based on whether useful certification returns
away from B0 while joint false-safe risk stays below the nominal-DT baseline.
Band selection used nominal DT output only; the observed safe/unsafe composition
was revealed after decisions were stored. The frozen method certified no actions
in B0-B3 and only 1/400
in B4. The margin-conditioned extension certified
0 actions in total,
so it did not improve useful availability.

### E6B METHOD REMAINS OVERLY CONSERVATIVE
