# Dynamic Binary Flash-Drum Process Specification

## Purpose and Freeze Boundary

 simulator validation defines the healthy plant and its independent nominal digital
twin. No fault mechanism or learned revalidation component is present. All simulator validation
code and outputs live under `artifacts/flash_study`; the setup output guard rejects any
attempt to write to the frozen CSTR and two-tank result packages.

## Process

The process is a continuously fed, well-mixed equilibrium flash drum separating
benzene (more volatile) from toluene. It is deliberately different from the
CSTR: its nonlinearities arise from material and energy inventories, phase
equilibrium, pressure-dependent vapor withdrawal, and level-dependent liquid
withdrawal. There is no reaction.

The integrated state is

`z = [N_L, N_B,L, E_L]`,

where `N_L` is total liquid holdup [mol], `N_B,L` is liquid benzene holdup
[mol], and `E_L` is liquid sensible enthalpy relative to 298.15 K [J]. Derived
measurements are liquid composition `x_B`, temperature `T`, bubble pressure
`P`, equilibrium vapor composition `y_B`, and level fraction `L=N_L/1000`.

## Dynamic Balances

With liquid feed `F`, liquid withdrawal `W_L`, vapor withdrawal `W_V`, feed
benzene fraction `z_F`, and equilibrium phase compositions `x_B,y_B`:

```text
dN_L/dt       = F - W_L - W_V
dN_B,L/dt     = F z_F - W_L x_B - W_V y_B
dE_L/dt       = F h_L(T_F,z_F) + Q
                 - W_L h_L(T,x_B) - W_V h_V(T,y_B)
```

`E_L = [N_B,L Cp_B + (N_L-N_B,L) Cp_T](T-298.15)` is inverted after each
integration stage. `h_V` uses the same liquid reference path plus the
temperature-dependent pure-component latent heats. This is an open equilibrium
stage, not a vapor-holdup or pressure-vessel gas-dynamics model.

## Thermodynamics

The liquid and vapor are assumed ideal; activity and fugacity coefficients are
one. Raoult's law gives

```text
P = x_B Psat_B(T) + (1-x_B) Psat_T(T)
y_B = x_B Psat_B(T) / P.
```

Vapor pressures use `log10(P/bar)=A-B/(T/K+C)`. Benzene uses the NIST Eon et
al. range 333.4-373.5 K (`A=4.72583`, `B=1660.652 K`, `C=-1.461 K`); toluene
uses the NIST Willingham et al. range 308.52-384.66 K (`A=4.07827`,
`B=1343.943 K`, `C=-53.773 K`). [NIST benzene Antoine data](https://webbook.nist.gov/cgi/cbook.cgi?ID=C71432&Type=ANTOINE),
[NIST toluene Antoine data](https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Type=ANTOINE).

Latent heats use NIST reduced-temperature correlations. Benzene uses
`A=47.41 kJ/mol`, `alpha=0.1231`, `beta=0.3602`, `Tc=562.1 K`; toluene uses
form 2 with `A=53.09 kJ/mol`, `beta=0.2774`, `Tc=591.7 K`.
[NIST benzene latent heat](https://webbook.nist.gov/cgi/cbook.cgi?ID=C71432&Type=HVAP-FORM1),
[NIST toluene latent heat](https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Type=HVAP-FORM2).

Constant liquid heat capacities, a documented reduced-model approximation, are
135.69 J mol-1 K-1 for benzene and 157.09 J mol-1 K-1 for toluene at 298.15 K.
[NIST benzene condensed-phase data](https://webbook.nist.gov/cgi/cbook.cgi?Name=benzene&cTC=on),
[NIST toluene condensed-phase data](https://webbook.nist.gov/cgi/cbook.cgi?ID=C108883&Mask=3FFF&Units=SI).

The validated shared temperature domain is 333.5-373.0 K. Extrapolation beyond
the source correlations is prohibited.

## Withdrawals and Actions

The continuous candidate action is

`a = [Q, u_V, u_L, F]`.

| Variable | Meaning | Bound |
|---|---|---:|
| `Q` | heat duty | 10-22 kW |
| `u_V` | vapor valve opening | 0.25-0.85 |
| `u_L` | liquid valve opening | 0.25-0.85 |
| `F` | liquid feed flow | 0.75-1.25 mol/s |

Any finite continuous vector inside these bounds is accepted. The outlet laws
are `W_L=C_L u_L sqrt(L)` and `W_V=C_V u_V sqrt(max(P-P_down,0))`, with
`P_down=0.20 bar absolute`. Valve coefficients and the 1000 mol working liquid
capacity are equipment design parameters, not thermodynamic constants.

The exact nominal steady point is `N_L=600 mol`, `x_B=0.45`, `T=350 K`, with
`F=1 mol/s`; the corresponding heat and valve settings are computed from the
balances rather than hard-coded as qualitative action labels.

## Outputs and Predeclared Safety Limits

| Output | Operating safety constraint |
|---|---:|
| Bubble pressure `P` | `P <= 0.90 bar absolute` |
| Level fraction `L` | `0.20 <= L <= 0.85` |
| Temperature `T` | `T <= 365 K` |

These are assumed equipment operating limits declared before any fault model or
fault trajectory exists. The pressure limit is a vacuum-system high-high limit,
the level band preserves vapor disengagement space and liquid outlet coverage,
and the temperature limit retains margin inside the validated property domain.
They are not chemical critical properties and were not tuned to fault outcomes.

## Numerical Method and Interfaces

`FlashPlant` and `FlashDigitalTwin` are separate objects. Each owns an
independent immutable parameter tree and independent simulation state; the DT
never reads plant parameters after construction. Fixed-step RK4 uses a 1 s
integration step and rejects nonphysical intermediate or final states instead
of clipping them. Standard output sampling is 10 s and the declared recovery
horizon is 1200 s.

Known scope limits are the ideal-liquid assumption, instantaneous phase
equilibrium, constant liquid heat capacities, no vapor holdup, and no explicit
wall/ambient heat loss. These assumptions are acceptable for a transparent
reduced equilibrium-stage transfer case but must remain visible in later claims.
