---
id: S01
title: USB-C 3V3 LDO with status LED
difficulty: trivial
coverage: [happy-path, stock-parts-only, no-mcu, friction-floor, latency-floor, straight-through]
target_mode: release
interview_expected: false
expected_question_band: [0, 1]
---

## Opening prompt

```
I want a tiny USB-C powered board that takes 5V from USB-C and regulates it down
to a clean 3.3V rail with a power-on status LED. No microcontroller, nothing
fancy — just USB-C in, a 3.3V LDO, and an LED that lights when the rail is up.
Target JLCPCB, keep the BOM cheap and all-SMD. The brief is complete; please just
build it.
```

## User-script

The brief is intentionally complete. Expected forks and canned answers:

- "Which LDO / what output current?" → "Anything common and cheap that does
  5V→3.3V at ~150 mA is fine, e.g. an AP2112 or similar. You choose."
- "USB-C power-only, or data too?" → "Power only. No data, no CC negotiation
  beyond the 5.1k pulldowns."
- "Confirm output directory / synthesize?" → "Yes, synthesize to ./generated."

Anything else: answer minimally as a beginner-to-intermediate maker, and the
observer notes it as an off-script question (likely a friction finding).

## Traps / required behaviors

- **Should drive nearly straight through** — this is a fully-specified brief. More
  than ~1 clarifying question is over-asking (friction penalty).
- All parts should resolve from **stock first-tier libs** (Device R/C/LED, a stock
  or vendored LDO, a stock USB-C connector or the vendored `usb-c-16p`). No LCSC
  auto-fetch should be *needed*; if one happens, it should still resolve cleanly.
- CC1/CC2 5.1k pulldowns present (USB-C sink convention).
- LED current-limit resistor present and sized for 3.3V.
- Must synthesize with `synthesis_check.status == ok` and **0 ERC errors**.

## Known design pitfalls

- LED resistor missing or wrong value (LED straight across the rail).
- LDO input/output decoupling caps omitted (needs ~1µF in / ~1µF out typical).
- CC pulldowns omitted → host won't source 5V.
- Over-engineering a trivial board (adding a TVS, a fuse, extra rails the user
  didn't ask for) — that's an intent-fidelity ding, not a bonus.

## What this scenario exercises

The **friction and latency floor**: with nothing ambiguous, a good run should be
fast, ask ~nothing, and produce a clean board. It establishes the baseline a
healthy session looks like, and catches regressions where the skill starts
interrogating or stalling on simple briefs. Light on Class-J; heavy on Class-C
`interaction_friction`, `latency`, `pipeline_completion`.
