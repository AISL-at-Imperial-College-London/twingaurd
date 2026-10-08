# RESS E7 Results

## Structural Mismatch

The only new mechanism is a plant-side first-order vapor-valve actuator lag
(`tau=30-120 s`). The nominal DT has no actuator state, and neither the mechanism
label nor effective valve position enters prediction. The E2 non-ideal-VLE F4
placeholder was prospectively replaced at E7 before any reserved truth; this is
an explicit protocol amendment, not a modification of E1-E6 evidence.

For the 564/600 trajectories with finite
physical rollouts, mean recursive prediction RMSE was
0.0218 bar for pressure,
0.0113 vessel fraction for level, and
0.8447 K for temperature. Mean signed biases were
-0.0012 bar, 0.0087
vessel fraction, and -0.0915 K, respectively.

At frozen 95% pooled conformal settings, certification availability was
43/600
(0.0717), trajectory coverage was
1.0000, and false-safe evidence was
0 observed events / 600; exact two-sided 95% CI [0.0000, 0.0061] and Wilson 95% CI [0.0000, 0.0064]. Maximum and total normalized false-safe
severity were 0.0000 and
0.0000; maximum and total integrated
severity were 0.0000
and 0.0000 normalized-seconds.
Matching ordinary/boundary quantiles are descriptive for this unseen mechanism
and do not establish a structural-mismatch guarantee.

## Large Independent Reliability

Flash: 6 observed events / 3000; exact two-sided 95% CI [0.0007, 0.0043] and Wilson 95% CI [0.0009, 0.0044]; certification availability
237/3000
(0.0790), trajectory coverage
0.9456, maximum normalized false-safe severity
0.9709, and total normalized false-safe
severity 1.9618. CSTR:
0 observed events / 3000; exact two-sided 95% CI [0.0000, 0.0012] and Wilson 95% CI [0.0000, 0.0013]; certification availability
1010/3000
(0.3367) and trajectory coverage
0.9560.
These are separate new denominators and are not pooled with C1, C1.6, E5, or E6.

## Final Verdict

### RESS-POSSIBLE WITH SPECIFIC LIMITATIONS

The frozen methods reduce false-safe reliance with useful availability in ordinary
populations, but flash and CSTR boundary challenges abstain completely, flash mean
model selection used a preregistered fallback, structural evidence is descriptive,
and pooled conformal coverage is marginal rather than conditional.
