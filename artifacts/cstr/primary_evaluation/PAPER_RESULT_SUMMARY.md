# Milestone C1 — paper result summary

**Central question.** Can a frozen, post-fault conformal digital twin safely certify individual counterfactual recovery actions?

**Population and constraint.** A new 500-episode calibration population and independent 1000-episode test population vary hidden UA fouling, initial conditions, measurement noise, and established interpolation/boundary/extrapolation flow actions. Safety requires reactor temperature <= 313.0 K throughout the 240-s horizon.

**Rule and implication.** S3 declares SAFE only when the complete one-sided conformal upper envelope is <= 313.0 K, UNSAFE only when some point of the lower envelope exceeds it, and otherwise UNRESOLVED. Marginal whole-trajectory upper coverage implies that the joint event `SAFE and actually unsafe` is contained in conformal miscoverage under exchangeability; it does not imply per-action conditional safety.

**Results (95%).** Nominal-DT joint false-safe was 0.394; S3 was 0.000 (Wilson [0.000, 0.004]). S3 conditional false-safe was 0.000 over 338 certifications, SAFE rate was 0.338, and unresolved rate was 0.493. Upper/lower trajectory coverage was 0.957/0.928. S3 was sharper than S2 (2.471 versus 3.539 K) but had the same SAFE count; it reduced unresolved decisions from 0.547 to 0.493. Multi-action screening success was 1.000 conditional on an actually safe candidate existing.

**Regimes.** - interpolation: SAFE rate 0.000, joint false-safe 0.000, unresolved 1.000. - boundary: SAFE rate 0.521, joint false-safe 0.000, unresolved 0.476. - extrapolation: SAFE rate 0.494, joint false-safe 0.000, unresolved 0.000.

**Trade-off and latency.** The 90/95/99% S3 rules all produced 338 SAFE and zero observed false-safe decisions; envelope width increased and unresolved rate rose slightly with confidence. Final-test median online latency was 0.167 s; conformal construction added median 0.000154 s.

**Strongest limitation.** Calibration is marginal, not conditional for a selected action or regime. All interpolation actions were unresolved, and lower-envelope coverage was only 0.928; simulator-to-plant population shift remains outside the guarantee.
