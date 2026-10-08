# Final RESS Claims Audit

| Proposed claim | Status | Evidence basis |
|---|---|---|
| Fault-blind operational prediction | SUPPORTED | Prediction/decision key audits found no evaluator fault metadata; only observed residual history carries fault effects. |
| Cross-process transfer | SUPPORTED WITH QUALIFICATION | CSTR, two-tank, and flash show qualitative risk reduction, but flash used a fallback mean model and boundary availability collapses. |
| Whole-trajectory coverage | SUPPORTED WITH QUALIFICATION | Pooled empirical trajectory coverage is reported for matched populations; domain exits and finite samples remain. |
| Conditional coverage | NOT SUPPORTED | No per-fault, severity, state, or action-conditional conformal guarantee was established. |
| Low false-safe risk | SUPPORTED WITH QUALIFICATION | Observed counts and finite exact/Wilson intervals support low risk only for the tested populations. |
| Action availability | SUPPORTED WITH QUALIFICATION | Ordinary populations retain nonzero availability; both E6 boundary challenges certify no actions. |
| Structural mismatch robustness | SUPPORTED WITH QUALIFICATION | The actuator-lag test is independent and descriptive; its matching q was not calibrated on structural faults. |
| Actions throughout the admissible continuous domain | SUPPORTED WITH QUALIFICATION | Space-filling and nominal-boundary samples cover the declared bounded domain, not every possible future action/state distribution. |
| Per-action safety guarantee | NOT SUPPORTED | Conformal validity is marginal over exchangeable episodes, not an individual certified-action probability statement. |
| Computational feasibility | SUPPORTED WITH QUALIFICATION | Prospective 3,000-episode populations completed, but stored latency and hardware context limit deployment extrapolation. |

## Verdict

### RESS-POSSIBLE WITH SPECIFIC LIMITATIONS

Drafting is defensible only with explicit marginal-coverage, boundary-abstention,
model-selection, simulator-only, structural-amendment, and finite-sample language.
