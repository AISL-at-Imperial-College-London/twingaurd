# Final RESS Audit

The audit independently reconstructed stored decisions and envelopes, checked
seed separation, hashes, missing/duplicate episodes, strict chronology where
timestamps exist, pre-truth key leakage, physical safety labels, and undefined
conditional rates.

- Audit checks: 66
- Checks with findings: 2
- Pre-E7 frozen artifacts protected by hash: 34222
- New E7 episodes: 600 structural flash + 3,000 flash reliability + 3,000 CSTR reliability

## Findings

- `FLASH_E5` / incorrect NaN-to-zero conversion: 18 issue(s). Conditional false-safe is undefined when no action is certified; E7 leaves it blank.
- `E6` / incorrect NaN-to-zero conversion: 6 issue(s). Conditional false-safe is undefined when no action is certified; E7 leaves it blank.

No prior result was rewritten. Earlier numeric zeros for conditional false-safe
rates with zero certifications are retained as historical artifacts and treated
as undefined in final claims.
