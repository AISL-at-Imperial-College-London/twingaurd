# Milestone C1.6A fault smoke-test verdict

| Mechanism | Trajectories | Safe | Boundary crossing | Min max-residual (K) | Max max-residual (K) | Numerically stable |
|---|---:|---:|---:|---:|---:|---:|
| ua fouling reference | 27 | 16 | 11 | 1.697 | 5.265 | 27/27 |
| feed temperature step | 27 | 16 | 11 | 0.428 | 9.246 | 27/27 |

Each mechanism used 27 trajectories: nine identical recovery actions at each of three severities. Every paired plant/DT trajectory began from the same state and received the same actions. Fault metadata appears only in evaluator-side summary columns; trajectory inputs contain no fault label or severity. The full conformal safety workflow was not invoked.

The feed-temperature fault is nontrivial and action-dependent, includes safe and unsafe outcomes, and remains numerically stable throughout the declared range. Its discrepancy shape is compared with the unchanged UA-fouling reference in the four figures.

### ADDITIONAL FAULTS READY
