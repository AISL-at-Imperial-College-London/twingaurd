# Milestone C1.5A — frozen C1 audit

## Audit verdict

**FROZEN C1 RESULT REPRODUCED**

This milestone is read-only. No predictor, conformal quantile, safety limit, split, decision, or C1 artifact was modified. The audit reconstructed decisions from the frozen arrays and wrote only to `results/milestone_c1_5a/`.

## Prospective chronology and accounting

- Predictions: **1000**
- Decision snapshots: **1000**
- Revealed truths: **1000**
- Missing records: **0**
- Chronology violations: **0**
- Duplicate manifest IDs/seeds: **0/0**
- Duplicate episode-method-level rows: **0**
- Reconstructed decision/envelope/truth/indicator mismatches: **0/0/0/0**
- Prediction files with truth/fault/physical/plant-named keys: **0**
- C1 calibration-final and C1-final/B.12-final seed overlaps: **0/0**
- Frozen B.11 ensemble and B.12 quantile hash matches: **1/1**

Every decision timestamp strictly precedes its truth timestamp. Sixty-three prediction and decision files share the same filesystem timestamp because their writes completed within the timestamp resolution; none is inverted. Logical prediction-before-decision ordering is independently supported by exact reconstruction from prediction-only keys and by the C1 implementation, which writes and hashes the decision snapshot before calling the reveal function. Filesystem timestamps are audit evidence, not a cryptographically signed event log.

The code-path audit found that frozen U3 rollout consumes only observed residual history, observed/future covariates, the nominal future, and frozen model parameters. The hidden plant is accessed only by the subsequent reveal function. Prediction archives contain no actual, truth, physical-plant, or hidden-fault fields; decision JSON contains only the episode ID and the 12 reconstructed method/level decisions. SHA-256 fingerprints for every prediction, decision, and truth artifact are stored in `artifact_chronology_audit.csv` for future immutability checks.

## Interpolation diagnosis

Interpolation has 334 episodes; 0.635 are physically safe, but S3 certifies 0.000. The result reflects **both genuine boundary proximity and conservative full-trajectory uncertainty propagation**:

- 0.545 are within ±1 K of 313 K; 0.281 are within ±0.5 K.
- Median true safety margin is 0.482 K.
- The frozen U3 centre is physically below the limit in 0.997 of interpolation episodes, with median centre margin 0.699 K.
- Median upper conformal inflation over the centre maximum is 1.803 K, exceeding the available centre margin and making the median conformal margin -1.084 K.
- The 95th percentile critical q is 4.035; even the largest interpolation critical q (4.741) is below frozen q+=5.465, so no episode can be certified under the frozen rule.
- Interpolation upper-envelope coverage is only 0.916, not excessive coverage. Its calibration upper-score descriptive 95th percentile is 5.643, above pooled q+=5.465, and 0.078 of interpolation calibration scores exceed the pooled q.

Therefore universal UNRESOLVED decisions are **not evidence that the pooled q is unnecessarily high for interpolation**. Interpolation has the heaviest upper-score tail and worse-than-nominal final upper coverage; a category-specific tail diagnostic would demand at least as much inflation, not less. Lowering q to recover certification would trade away the already weak interpolation coverage. This is a diagnosis only—no Mondrian quantile or alternative decision was fitted or evaluated.

For context, the frozen raw U3 95% envelope certifies 52/334 interpolation actions but achieves only 0.716 upper trajectory coverage. This is not a valid safety alternative; it shows that the transition from some raw certification to zero conformal certification is the cost of repairing severe raw undercoverage.

The conservatism is nevertheless operationally real: nearly every centre trajectory is safe and 63.5% of actual trajectories are safe, yet none is certifiable. It arises because relatively small centre margins are dominated by a max-over-240 standardized uncertainty envelope required for marginal population-level validity. This distinction—decision conservatism without evidence of an overlarge pooled q—is the central C1.5A finding.

## Frozen C1 interpretation

The primary C1 upper-safety conclusion is reproduced with no accounting discrepancy or detectable evaluation leakage. This audit does not repair the lower envelope: its 92.8% coverage remains below nominal 95%, so strong UNSAFE certification is still not validated at 95%. C1's conformal statement remains marginal over the defined episode population, not conditional for a selected action or regime.
