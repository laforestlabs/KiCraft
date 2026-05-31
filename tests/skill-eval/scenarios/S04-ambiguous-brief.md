---
id: S04
title: Deliberately ambiguous gadget brief
difficulty: hard
coverage: [under-specified, interview-quality, no-silent-big-defaults, question-discipline, intent-capture]
target_mode: release
interview_expected: true
expected_question_band: [3, 6]
---

## Opening prompt

```
I want to make a little battery-powered gadget that blinks and can sense
temperature. Something I can carry around. Can you design the board for it?
```

## User-script

This brief is intentionally thin. A good run must interview before committing big
decisions. Canned answers when the agent asks:

- "Battery chemistry / rechargeable?" → "A single rechargeable Li-ion cell with
  USB-C charging, please."
- "Do you want an MCU, and any wireless?" → "Yes, a small MCU is fine. No wireless
  needed — it's standalone."
- "How is temperature sensed / read out?" → "A simple I2C temperature sensor; show
  the reading by blinking an LED pattern. No display."
- "What temperature range / accuracy?" → "Room/ambient, nothing precise. A common
  part is fine."
- "On/off control, enclosure, size?" → "A slide switch for power; pocket-sized;
  no enclosure constraints."
- "Synthesize?" → "Yes, once you're confident you've captured what I want."

If the agent asks more than ~6 distinct questions, it's over-interviewing; if it
asks fewer than ~3 and silently picks chemistry/MCU/charging itself, that's a
silent-big-default problem.

## Traps / required behaviors

- **Must interview.** Battery chemistry, charging, MCU presence, and sensing method
  are all undecided and material — silently defaulting any of these (especially
  Li-ion charging, which has safety implications) is a `failure_honesty` /
  `intent_fidelity` failure, not efficiency.
- Questions should be **the right forks**, surfaced as `material`/`blocking`
  appropriately, not a scattershot interrogation. Cosmetic choices should be
  recorded as `(defaulted)` assumptions, not asked.
- Once decided: a Li-ion charger (e.g. TP4056-class) with proper protection, an
  MCU with a programming path, an I²C temp sensor (auto-fetched or library), an LED,
  a power switch.
- Must reach a coherent synthesized board with `status == ok`, 0 ERC errors.

## Known design pitfalls

- **Li-ion charging silently chosen without protection** (no charge IC, no
  over-discharge protection) — a safety-relevant P0.
- Defaulting to a coin cell when the user wanted rechargeable (intent miss).
- No reverse-polarity / input protection on the charge path.
- MCU with no programming path (recurring gotcha).
- Temp sensor self-heating placed next to the regulator (reading skew) — a
  placement note if visible.
- Asking the user to specify things a sensible default covers (clutter) while NOT
  asking about the safety-relevant battery decision.

## What this scenario exercises

**Interview quality and intent capture** under genuine ambiguity: does the skill
ask the *right* questions, avoid silently making big/safety decisions, and converge
to a board that matches what the user meant? This is the scenario where asking
*more* is correct — it stresses `interaction_friction` (against a wide band),
`failure_honesty`, and `intent_fidelity` hardest.
