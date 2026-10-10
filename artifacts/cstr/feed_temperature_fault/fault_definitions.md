# Fault definitions

## Preserved reference: UA fouling

- **Physical interpretation:** progressive heat-transfer surface fouling.
- **Modified parameter:** plant `UA_eff`; nominal `UA = 500 J/s/K`.
- **Diagnostic range:** 25%, 40%, and 55% loss, matching the established primary certification fault family.
- **Scope:** hidden plant only; nominal DT unchanged.
- **Status:** the existing `UA_eff` implementation was not modified.

## Additional mechanism: feed-temperature step

- **Physical interpretation:** upstream feed-heater excursion or loss of feed-temperature regulation causing hotter material to enter the reactor.
- **Modified input:** plant inlet temperature `Tin` through a hidden additive step.
- **Nominal value:** 350 K.
- **Faulty range:** 355 K, 370 K, and 390 K (+5, +20, +40 K), declared as low, medium, and high severity.
- **Distinctness:** this directly increases inlet sensible-energy flow and its effect scales with recovery feed rate. UA fouling instead reduces heat removal and approaches a severity-dependent heat-transfer plateau.
- **Scope:** hidden plant only. The nominal DT retains 350 K and receives no fault type, changed input, or severity.

Only the feed-temperature mechanism was added. Outlet-pump degradation was considered but rejected after a range probe because its temperature mismatch was <=0.14 K over 120 s, failing the nontrivial mismatch criterion. No false-safe behavior was used to set the feed-temperature range.
