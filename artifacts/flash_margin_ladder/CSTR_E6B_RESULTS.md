# CSTR E6B Margin-Ladder Results

The CSTR replication uses frozen `q+=5.464557586938352` and the unchanged 313 K
constraint. Margin bands use `(313 K - max_h T_DT)/1 K`; repeated flow actions
across distinct decision states are intentional grid selections.

| Band | Method | Eligible | Safe / confirmed unsafe / incomplete | Certified | False-safe | Joint risk (Wilson 95% CI) | Conditional false-safe | Safe-action certification |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| B0 | nominal_DT | 0 | 0 / 0 / 0 | 0 | 0 | undefined | undefined | undefined |
| B0 | frozen_CSTR_conformal | 0 | 0 / 0 / 0 | 0 | 0 | undefined | undefined | undefined |
| B1 | nominal_DT | 266 | 0 / 266 / 0 | 266 | 266 | 1.000 [0.986, 1.000] | 1.000 | undefined |
| B1 | frozen_CSTR_conformal | 266 | 0 / 266 / 0 | 0 | 0 | 0.000 [0.000, 0.014] | undefined | undefined |
| B2 | nominal_DT | 300 | 0 / 300 / 0 | 300 | 300 | 1.000 [0.987, 1.000] | 1.000 | undefined |
| B2 | frozen_CSTR_conformal | 300 | 0 / 300 / 0 | 0 | 0 | 0.000 [0.000, 0.013] | undefined | undefined |
| B3 | nominal_DT | 300 | 0 / 300 / 0 | 300 | 300 | 1.000 [0.987, 1.000] | 1.000 | undefined |
| B3 | frozen_CSTR_conformal | 300 | 0 / 300 / 0 | 0 | 0 | 0.000 [0.000, 0.013] | undefined | undefined |
| B4 | nominal_DT | 300 | 0 / 300 / 0 | 300 | 300 | 1.000 [0.987, 1.000] | 1.000 | undefined |
| B4 | frozen_CSTR_conformal | 300 | 0 / 300 / 0 | 0 | 0 | 0.000 [0.000, 0.013] | undefined | undefined |

This is a separate Analysis A population and is not merged with C1, C1.6, E6,
or E7 denominators.
