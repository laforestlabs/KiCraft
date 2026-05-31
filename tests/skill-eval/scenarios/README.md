# Scenario library

Each scenario is one reproducible CircuitChat use case. A scenario is the
*controlled input* to a run: the same opening prompt + user-script should drive
comparable runs, so scores trend meaningfully.

Scenarios deliberately span the coverage space — from a fully-specified brief
(tests friction/latency floor) to a deliberately ambiguous one (tests interview
quality), and from "all stock parts" to "must auto-fetch / must not substitute" to
"mixed-signal grounding gotcha" (tests the electrical-soundness layer).

## File schema

YAML front-matter (the scorer reads `expected_question_band`), then prose sections:

```yaml
---
id: S0x
title: <short title>
difficulty: trivial | easy | moderate | hard
coverage: [tags, describing, what, this, exercises]
target_mode: release            # default; observer may override
interview_expected: true|false
expected_question_band: [lo, hi]  # how many clarifying questions is reasonable
---
```

Required sections:

- **## Opening prompt** — the exact text to paste into the subject session, in a
  fenced block. Written in the voice of a user of the stated expertise. Verbatim
  and reproducible.
- **## User-script** — anticipated clarifications and the canned answer for each
  fork. The subject is answered only from here (plus minimal in-character answers
  for anything off-script, which the observer notes).
- **## Traps / required behaviors** — the specific things the observer must check:
  must-dos (e.g. "auto-fetch the sensor", "surface a programming path") and
  must-not-dos (e.g. "must NOT substitute `Sensor_*`"). These map to gates and
  Class-J anchors.
- **## Known design pitfalls** — a starting checklist for the electrical-soundness
  review (not a limit). The circuit-specific gotchas a good reviewer would watch.
- **## What this scenario exercises** — why it's in the library; which dimensions
  it most stresses.

## The `expected_question_band`

`interaction_friction` is scored *relative to this band*, because "fewer questions"
is not always better. A trivial, fully-specified brief should ask ~0; an ambiguous
brief should ask several. Asking inside the band is good; interrogating a clear
brief, or skipping a needed interview, is penalized. Set the band to the count of
*legitimate* forks the brief leaves open.

## Seeded scenarios

| id | difficulty | exercises |
|----|-----------|-----------|
| S01 | trivial | friction/latency floor; straight-through, no interview |
| S02 | moderate | I²C sensor LCSC auto-fetch; no-silent-substitution; MCU programming path |
| S03 | moderate | WiFi MCU; programming path; rail sizing for current bursts |
| S04 | hard | deliberately ambiguous → interview quality, no silent big-defaults |
| S05 | hard | mixed-signal grounding/analog front-end → electrical-soundness |

Add scenarios freely; keep ids monotonic and update this table.
