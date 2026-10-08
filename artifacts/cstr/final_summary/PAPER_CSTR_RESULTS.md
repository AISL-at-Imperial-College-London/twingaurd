# Frozen CSTR results

## RQ1 — Why is nominal post-fault DT validation unsafe?

The nominal DT remains internally consistent but is unaware of the hidden plant change. In Milestone A, the post-fault residual has lag-one autocorrelation 0.959 and a positive time slope, while a constant-residual 95% band covers only 1.7% of the held-out trajectory at N=50. The discrepancy is therefore dynamic, not stationary correction noise.

This matters at the decision layer. In the frozen 1,000-episode C1 test, nominal-DT certification declares every action SAFE but produces 394 false-safe episodes: joint false-safe probability 0.394 (Wilson 95% CI 0.364–0.425). In the balanced 600-episode multi-fault test, the corresponding value is 0.442 (0.402–0.482). A nominal rollout alone is not a post-fault safety certificate.

## RQ2 — Can post-fault discrepancy be predicted without fault identification?

Yes, within the tested simulator populations. The learner uses residual observations and operational covariates, never fault identity, magnitude, changed physical parameters, hidden plant states, or future truth. Developmental dynamic residual models outperform static correction on their original held-out action split. More importantly, on the B.9 final assessment, the fault-blind k=1 latent model achieves median recursive RMSE 0.680 K, compared with 2.582 K for observable-conditioned F1.

Higher temporal order alone was not the solution: ARX p=5 obtains 2.936 K median recursive RMSE and retains maximum innovation autocorrelation near 0.996. Stability conditioning prevents explosive samples but does not supply missing evolving-state information. The latent state improves the conditional mean without being interpreted as a physical fault state.

## RQ3 — Is probabilistic uncertainty calibrated?

Not before conformal calibration. Raw B.10 U0 95% simultaneous coverage is 0.525. Adding whole-episode bootstrap parameter uncertainty raises B.11 U3 coverage to 0.617, so parameter uncertainty helps but does not fully explain predictive error.

With the predictor frozen, B.12 uses one max-standardized score per complete trajectory. The pooled B2 result reaches 0.934 simultaneous coverage (Wilson 0.909–0.953) on 500 new final episodes, with mean width 3.538 K. This supports useful marginal trajectory calibration under exchangeability; it does not establish conditional calibration by action or severity.

## RQ4 — Can conformal trajectory revalidation support SAFE certification?

Yes, selectively. C1 certifies SAFE only when the complete upper conformal envelope remains below 313 K. On 1,000 frozen UA-fouling episodes, upper coverage is 0.957, SAFE certification is 0.338, and no false-safe event is observed (Wilson joint upper bound 0.0038). The nominal-DT false-safe rate is 0.394.

The balanced multi-fault revalidation relearns all state-space parameters, bootstrap models, feature scaling, and `q+` from UA-fouling and feed-temperature populations. On 600 untouched final episodes, pooled upper coverage is 0.948, SAFE certification is 0.235, and no false-safe event is observed (Wilson upper bound 0.0064), while nominal-DT false-safe is 0.442. Fault-specific coverage is 0.950 for UA fouling and 0.947 for feed-temperature steps.

The qualification is material: high-severity coverage is 0.850 for UA and 0.840 for feed-temperature episodes. C1 lower-envelope coverage is 0.928. The defensible method is therefore upper-envelope SAFE certification with NOT CERTIFIED otherwise, under a marginal population guarantee.

## RQ5 — Is the selective framework operationally useful?

Single-action C1 certifies 33.8% of actions and abstains on every interpolation action. C1.5A reproduces the full chronology and shows that interpolation cases are genuinely close to the safety boundary; it does not justify lowering `q+`.

Action screening changes availability without changing the frozen predictor or decision rule. Across 100 new C1.5C states, conditional success rises from 40/56 eligible states at K=1 (0.714, Wilson 0.585–0.816) to 100/100 at K=20 (1.000, 0.963–1.000). The probability of at least one certified action rises from 0.400 to 1.000, with zero false-safe certifications among 2,000 screened actions. This is evidence that selective certification can be operationally useful when several physically admissible actions are screened prospectively.

## Frozen conclusion

The CSTR evidence supports a fault-blind, discrepancy-aware, parameter-ensemble, conformal safety layer that abstains when it cannot certify an action. It does not support claims of DT repair, universal conditional coverage, per-action 95% safety probability, strong UNSAFE certification, or hardware validation. The completed case is ready to transfer as a methodology specification, with every numerical quantity relearned for the next process.
