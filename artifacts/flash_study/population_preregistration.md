# artifacts/flash_study  population design Preregistration

Status: **FROZEN BEFORE POPULATION GENERATION**  
Protocol version: `population design-1.0.0`  
Freeze date: `2026-08-13`

This document defines the flash experiment before training, model selection,
calibration, or final evaluation. Later stages may append separate amendment
records but must not silently edit this file. The exact-file SHA-256 is stored in
`population_preregistration.sha256`.

## 1. Preconditions and Frozen Process

simulator validation is GO: the recorded simulator validation summary has `go=true`, zero plant/DT difference,
physical healthy trajectories, closed balances/VLE, and four meaningful action
responses. The frozen process bundle is the filename-delimited tree hash of:

| File | SHA-256 |
|---|---|
| `flash_drum.py` | `493255bdffb7d3218758c889bfb4ab33b814574c8027cda9b72f2de8e6f7b1ae` |
| `digital_twin.py` | `53d43a3923bab5ce7256757a93a2fbee85f20c89da52173aaa9d3db2ae53eeef` |
| `flash_config.yaml` | `810076236285a0937dba0507fc956f45b02eef168429e805ebdb1a39a7104db4` |
| Combined bundle | `1bc228ee61c628bf44795a0f67c9a476973f630dcd95f9932d88c96682325624` |

The nominal DT remains the simulator validation benzene/toluene ideal-Raoult equilibrium model.
No CSTR/two-tank parameter, fit, quantile, scale, action, or safety limit enters
this design.

## 2. Fault-Blind Information Boundary

The deployed method receives only episode ID, 50 time stamps, measured
post-event `[P,L,T]`, matching nominal-DT `[P,L,T]`, the observed action history,
and one candidate action. The root episode seed remains evaluator-side because
it determines fault sampling. The method never receives fault family, severity, magnitude,
changed parameter, plant composition/holdup/enthalpy state, future plant output,
or future safety label. Composition is evaluator-side because no online
composition analyzer is assumed. Fault metadata is written only after decision
artifacts in evaluator/truth records and is used only for descriptive subgroup
analysis. Artifact allowlists and chronology are mandatory later.

Fault detection is external and outside scope. At `t=0`, plant and independent
DT begin from the same last-known-valid state; the hidden plant change occurs
instantaneously while the DT stays nominal. The nominal action is held during
the observation window. Candidate-action physical truth does not exist until
after prediction and decision serialization.

## 3. Hidden Fault Families

Primary populations balance F1/F2/F3 exactly. Within each family normalized
severity `s~Uniform[0,1]`; low/medium/high are fixed thirds of `s`.

| ID | Hidden plant change | Frozen magnitude | Nominal DT | Training status |
|---|---|---|---|---|
| F1 | Vapor outlet effective capacity loss | `C_V,plant=(1-d)C_V,nom`, `d=0.10+0.35s` | nominal `C_V` | included |
| F2 | Liquid outlet effective capacity loss | `C_L,plant=(1-d)C_L,nom`, `d=0.10+0.35s` | nominal `C_L` | included |
| F3 | Unmeasured benzene feed-fraction shift | `z_F,plant=0.55+sigma(0.04+0.08s)`, signs alternate exactly by reserved seed parity; range 0.43-0.67 | `z_F=0.55` | included |
| F4 | Non-ideal liquid VLE structural mismatch | symmetric Margules activity model, `A_M in [0.15,0.35]`; exact implementation is frozen in a separate pre-generation reliability evaluation amendment | ideal Raoult | reserved stress only; excluded population design-calibration |

F1/F2 are actuator-capacity degradations; F3 is an unmeasured material-property
input shift. F4 changes the thermodynamic structure rather than an simulator validation parameter.
F4 is not implemented, simulated, trained, selected, or calibrated in population design.

## 4. Initial Conditions, Sampling, and Noise

Each episode's last valid state is sampled independently and uniformly as
`N_L in [550,650] mol`, `x_B in [0.42,0.48]`, and `T in [348,352] K`. Plant and
DT receive identical initial values. The fault is applied plant-side at `t=0`.

Sampling is every 10 s. The method receives `N_post=50` post-fault samples at
times 10 through 500 s, giving 500 s (about one third of the simulator validation 1500 s settling
display) to expose pressure, inventory, and thermal mismatch without using a
future action trajectory. Candidate evaluation begins at 500 s and covers
`H=1200 s` (120 future samples). RK4 integration is 1 s.

Observation noise is independent zero-mean Gaussian with frozen sensor standard
deviations `0.002 bar` for P, `0.002` level fraction for L, and `0.10 K` for T.
Noise applies only to method-visible plant observations. DT paths and evaluator
safety truth are noise-free physical quantities. Missingness is not introduced
in the primary experiment; any later missing-data stress requires an amendment.

## 5. Continuous Action Domain

The action order is `[Q,u_V,u_L,F]`. The exact nominal action computed from the
simulator validation balances is `[15317.3303572124 W, 0.5015947237, 0.4995841834, 1.0 mol/s]`.

| Component | simulator validation physical bounds | Maximum absolute change | population design admissible bounds |
|---|---:|---:|---:|
| Q | 10000-22000 W | 6000 W | 10000-21317.3303572124 W |
| u_V | 0.25-0.85 | 0.25 | 0.2515947237-0.7515947237 |
| u_L | 0.25-0.85 | 0.25 | 0.25-0.7495841834 |
| F | 0.75-1.25 mol/s | 0.25 mol/s | 0.75-1.25 mol/s |

Any finite continuous vector inside the population design bounds is admissible. Ordinary
actions use a centered, independently permuted Latin hypercube over all four
dimensions, one population-level design per split. Seeds fix every permutation.
The design is not regenerated in response to plant outcomes.

## 6. Physical Safety Event

Engineering limits are inherited unchanged from simulator validation and predate all faults:
`P_max=0.90 bar absolute`, `L_min=0.20`, `L_max=0.85`, `T_max=365 K`.
Fixed engineering scales are `scale_P=0.10 bar`, `scale_L=0.10` level fraction,
and `scale_T=5 K`. The four dimensionless violation directions are

```text
g_P      = (P - 0.90) / 0.10
g_L_high = (L - 0.85) / 0.10
g_L_low  = (0.20 - L) / 0.10
g_T      = (T - 365) / 5
```

Physical truth is SAFE iff `max(h,j) g_j,true(h) <= 0` over all 120 future
samples. The signed trajectory safety margin is `m=-max(h,j)g_j`; positive is
safe, zero is boundary, and negative is violation. No individual constraint may
be averaged away.

## 7. Nominal-DT Boundary Challenge

For each boundary episode, generate 128 centered-LHS actions plus 25 targeted
actions (nominal, each axial lower/upper extreme, and all 16 domain corners).
For every candidate, simulate only the independent nominal DT from its decision
state and compute `m_DT(a)=-max(h,j)g_j,DT(h,a)`. A nominal trajectory leaving
the validated property/vessel domain receives `m_DT=-infinity` and is excluded.

The fixed boundary band is `|m_DT|<=delta_boundary=0.15`, equivalent to 0.015
bar pressure, 0.015 level fraction, or 0.75 K when a single direction is active.
Sort eligible actions by `|m_DT|` then original candidate index; use
`episode_seed mod min(8,n_eligible)` to select among the eight closest. If none
is eligible, select the single closest valid DT action, mark `boundary_fallback`,
and do not regenerate. This mechanism has no plant/fault/truth input. Counts of
fallbacks and subsequently revealed unsafe outcomes will be reported honestly.

## 8. Reserved Populations and Seeds

Only seed reservations and design rows are created in population design. No observation,
training, calibration, final, boundary, stress, or rare-event trajectory is
generated. Exact rows are in `flash_population_manifest.csv` and
`flash_seed_manifest.csv`.

| Population | N | Seed block | Faults | Action | Earliest generation |
|---|---:|---:|---|---|---|
| training | 900 | 22000000-22000899 | F1/F2/F3, 300 each | ordinary LHS | model selection |
| model selection | 180 | 23000000-23000179 | F1/F2/F3, 60 each | ordinary LHS | model selection |
| ordinary conformal calibration | 600 | 24000000-24000599 | F1/F2/F3, 200 each | ordinary LHS | calibration after model freeze |
| boundary conformal calibration | 600 | 25000000-25000599 | F1/F2/F3, 200 each | DT boundary rule | calibration after model freeze |
| ordinary final test | 1200 | 26000000-26001199 | F1/F2/F3, 400 each | ordinary LHS | primary evaluation after full freeze |
| boundary challenge test | 1200 | 27000000-27001199 | F1/F2/F3, 400 each | DT boundary rule | boundary challenge after full freeze |
| structural mismatch stress | 600 | 28000000-28000599 | F4 only | 50% ordinary/50% boundary | reliability evaluation after full freeze |
| rare-event reliability | 12000 | 29000000-29011999 | F1/F2/F3, 4000 each | 50% ordinary/50% boundary | reliability evaluation after full freeze |

All blocks are disjoint. Within primary populations, family assignment cycles
F1/F2/F3. Each episode seed is the root for purpose-coded NumPy `SeedSequence`
substreams: initial state 101, fault magnitude 211, observation noise 307,
ordinary action 401, boundary pool 503, predictive paths 601, and truth rollout
701. Bootstrap roots are 31000000-31000149 and are never episode seeds.

Ordinary and boundary calibration are separate because nominal-DT boundary
selection changes the action/state population. Each final population uses only
its matching calibration quantile. Cross-application of quantiles is reported
as a sensitivity analysis without a formal guarantee.

## 9. Mean Discrepancy Model Selection

The transferred model is a multivariable linear-Gaussian latent discrepancy
state model for residual `[P_plant-P_DT,L_plant-L_DT,T_plant-T_DT]`. Allowed
operational covariates are nominal-DT `[P,L,T]`, current/candidate `[Q,u_V,u_L,F]`,
and action change from the observation action, all scaled using training data
only. Fault metadata and composition are prohibited. Full `R(3,3)` and `Q(k,k)`
are allowed; stable diagonal `A` is required.

Try latent dimensions `k=1,2,3` sequentially and select the first that passes all
selection gates. Gates are evaluated on the 180 selection episodes only:

1. finite converged fit and `spectral_radius(A)<0.995`;
2. median episode worst-output recursive RMSE, after division by engineering
   scales `[0.10,0.10,5]`, at most 0.50;
3. each output's median normalized recursive RMSE at most 0.50;
4. absolute pooled normalized prediction bias at most 0.10 for every output;
5. absolute lag-1 innovation autocorrelation at most 0.50 for every output.

If none passes, freeze the converged stable candidate with lowest criterion 2,
breaking ties by smaller `k`, label the mean model inadequate, and prohibit a
claim of unqualified transfer success. Gates may not be relaxed after seeing
calibration or test data.

## 10. Bootstrap Predictive Uncertainty

After `k` is frozen, target 100 accepted fits from unstratified whole-episode
bootstrap resamples of training episodes; evaluator fault labels are not used in
resampling. Attempt at most 150 roots (31000000-31000149). Accept only finite,
converged fits with positive-definite covariance matrices and
`spectral_radius(A)<0.995`; log every rejection.

Online ensemble candidates are `J in {10,25,50,100}` using the first J accepted
fits in root order. On selection data, choose the smallest J whose median
normalized center difference from J=100 is <=0.05 for every output and whose
median predictive-scale ratio to J=100 lies in `[0.95,1.05]` for every output;
fallback is J=100. Use 2000 coherent multivariable predictive trajectories per
episode. Predictive scale floors are `0.005 bar`, `0.005` level fraction, and
`0.25 K`.

## 11. Conformal Score and Alpha Levels

For predictive median `m_hr`, standard deviation `s_hr`, and truth `y_hr`, the
single violation-oriented whole-trajectory score is

```text
S = max_h {
  (P_true-m_P)/max(s_P,0.005),
  (L_true-m_L)/max(s_L,0.005),
  (m_L-L_true)/max(s_L,0.005),
  (T_true-m_T)/max(s_T,0.25)
}
```

The primary alpha is 0.05; alpha 0.10 and 0.01 are secondary. For each matching
calibration population use finite-sample rank `ceil((n_cal+1)(1-alpha))`, capped
at n, on scalar episode scores. This gives one simultaneous marginal envelope
over all times and constraint directions. It is not a conditional guarantee for
each fault, severity, state, or selected action.

## 12. Prospective Decision and Evaluation Semantics

For an upper direction use `m+q*s`; for low level use `m_L-q*s_L`. At the chosen
alpha, return exactly `CERTIFIED SAFE` iff pressure, high/low level, and
temperature conformal bounds satisfy all simulator validation limits at every future sample.
Return `NOT CERTIFIED` otherwise. No strong UNSAFE decision is defined.

Chronology is: observe 50 samples -> form candidate using its declared action
generator -> predict -> serialize and hash prediction -> decide -> serialize and
hash decision -> reveal/generate physical future -> serialize truth. `CERTIFIED
SAFE` with physically unsafe truth is a false-safe. Primary metrics are joint
false-safe probability with Wilson 95% CI, conditional false-safe among
certifications, certification/abstention rate, simultaneous coverage, per-output
RMSE/width, violation direction/severity, and latency. Fault/severity results are
descriptive only.

Ordinary primary evaluation and boundary boundary challenge results remain separate. A useful safety result is
predeclared as joint false-safe Wilson 95% upper bound <=0.01 and certification
rate >=0.10 in the corresponding primary population. Failure of either is
reported without retuning. The rare-event set estimates risk more precisely but
cannot be used to revise the predictor, quantile, action rule, or primary claims.

## 13. System-Specific Quantities

The following must be relearned or redeclared for another process: flash
equations/thermodynamics/property domain; state and measured outputs; sensor
noise; initial-condition distribution; fault families/ranges/balance; action
components, bounds, nominal values, rate/change limits, and generators;
sampling interval, observation count, and horizon; physical limits and margin
scales; boundary pool, tolerance, and selection ranks; seed blocks/population
sizes; residual outputs/covariates/scaling; allowed latent dimensions and gates;
bootstrap acceptance/size candidates; trajectory count/scale floors; calibration
strata; alpha levels; usefulness thresholds; and all numerical fitted parameters
and conformal quantiles. Only the fault-blind observe-predict-decide-reveal
architecture, whole-episode bootstrap principle, scalar whole-trajectory
conformal rank convention, and selective SAFE/NOT CERTIFIED semantics are
process-independent.
