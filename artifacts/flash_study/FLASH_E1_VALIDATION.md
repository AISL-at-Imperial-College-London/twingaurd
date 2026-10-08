# Flash E1 Validation

## Scope

This is a healthy-process physics validation only. No faults, discrepancy model,
training, bootstrap uncertainty, conformal calibration, or safety decisions were
introduced. Seed: `1601`.

## Validation Results

| Check | Result | Criterion | Status |
|---|---:|---:|---|
| Independent plant/DT maximum output difference | 0.000e+00 | < 1e-10 | PASS |
| Maximum total-balance residual per RK4 step | 5.684e-14 mol | < 1e-7 mol | PASS |
| Maximum component-balance residual per RK4 step | 2.842e-14 mol | < 1e-7 mol | PASS |
| Maximum energy-balance residual per RK4 step | 4.657e-10 J | < 1e-7 J | PASS |
| Maximum Raoult partial-pressure residual | 8.327e-17 bar | < 1e-12 bar | PASS |
| Healthy sampled trajectories physical | True (12 cases) | all | PASS |
| Perturbed long-horizon convergence ratio | 1.700e-02 | < 0.05 over 7200 s | PASS |
| Perturbed long-horizon trajectory physical | True | all samples | PASS |
| Fixed-seed replay | True | exact | PASS |
| Meaningful manipulated variables | 4 of 4 | >= 2 | PASS |

## Action Responses

All steps begin at 300 s from the exact nominal steady state. Reported changes
are final values relative to the unstepped baseline.

| Action increase | Delta pressure [bar] | Delta level [-] | Delta temperature [K] | Interpretation |
|---|---:|---:|---:|---|
| Heat duty +2.5 kW | 0.1121 | -0.0448 | 5.576 | More boiling raises T/P and removes liquid |
| Vapor valve +0.12 | -0.1129 | -0.0279 | -6.008 | More vapor withdrawal depressurizes/cools |
| Liquid valve +0.12 | 0.0000 | -0.1164 | 0.000 | More liquid withdrawal lowers holdup |
| Feed flow +0.15 mol/s | -0.0057 | 0.1411 | -0.638 | More subcooled feed raises holdup and cools |

## Numerical Domain

The simulator rejects action vectors outside the declared bounds and raises a
`PhysicalStateError` if a trajectory leaves the vessel or property-correlation
domain. It does not silently clip states. This validation used a 1 s fixed-step
RK4 integrator, 12 seeded perturbed healthy cases of 1200 s, and a 7200 s run
from `N_L=650 mol`, `x_B=0.47`, and `T=352 K` back toward nominal operation.

## Decision

### E1 GO

Plant/DT mismatch is zero at numerical precision, all extensive balances and
VLE identities close, sampled trajectories remain physical, the perturbed state
recovers stably over two hours, and all four manipulated variables produce
nontrivial, physically directed responses. Fault work may proceed in a later
milestone; none was introduced here.
