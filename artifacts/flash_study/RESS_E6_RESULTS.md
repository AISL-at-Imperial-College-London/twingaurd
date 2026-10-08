# RESS E6 Boundary Reliability Results

## Frozen Design And Chronology

Flash uses all 1,200 E2-reserved boundary seeds. Candidate pools and selection
use only nominal-DT trajectories and the frozen `|m_DT|<=0.15` criterion.
1200/1200 states had at least one in-band candidate; fallbacks were
not regenerated. Prediction and decision artifacts were serialized before each
plant future reveal. The true challenge mix is 347 safe and
853 unsafe actions; it was not rebalanced after truth.

The five flash quantiles are finite-sample order statistics of only the 463
frozen boundary-calibration scores: `{"0.9": 9.0228762395739, "0.925": 10.068982705941224, "0.95": 11.349920831470156, "0.975": 12.496388687845336, "0.99": 14.164732647450517}`.

## Flash Risk-Availability Frontier

| Confidence | Availability | Truly-safe certification | Joint false-safe | Wilson upper | Conditional false-safe | Max severity | Integrated false-safe severity |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 90% | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 |
| 92.5% | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 |
| 95% | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 |
| 97.5% | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 |
| 99% | 0.000 | 0.000 | 0.000 | 0.003 | 0.000 | 0.000 | 0.000 |

At 95%, nominal-DT reliance is 0.593
with joint false-safe risk 0.372. The
pooled method certifies 0.000, with joint
risk 0.000 and Wilson upper
0.003. Thus the method
removes observed unsafe validation only through complete abstention and does not demonstrate nontrivial boundary-risk reduction.
Zero false-safe events are not required for this conclusion.

## Separate CSTR Stress Extension

The new 300-action CSTR denominator is separate from C1/C1.6. Its nominal-DT
availability/risk are 1.000/
1.000; the frozen 95% CSTR method gives
0.000/0.000.
The CSTR reduction likewise occurs through complete abstention on this challenge.
No C1 or C1.6 artifact or denominator was altered.

## Conclusion

### BOUNDARY RELIABILITY NOT DEMONSTRATED
