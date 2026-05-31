---
id: S03
title: ESP32 WiFi PIR motion sensor
difficulty: moderate
coverage: [wifi-mcu, lcsc-auto-fetch, rf-module-not-first-tier, programming-path, rail-current-sizing, antenna-keepout]
target_mode: release
interview_expected: true
expected_question_band: [2, 4]
---

## Opening prompt

```
I want to build a USB-C powered WiFi motion sensor around an ESP32-WROOM module.
It should read a PIR sensor and report motion over WiFi (MQTT). Onboard 3.3V
regulation from the 5V USB-C input. I'm a beginner-ish hobbyist — please walk me
through the decisions that matter.
```

## User-script

- "PIR: integrated sensor element (BISS0001 analog front-end) or a header for an
  off-the-shelf module (HC-SR501/AM312)?" → "A header for a plug-in HC-SR501-class
  module. Keep it simple."
- "USB-C: power-only, or power + programming (USB-UART bridge + auto-program/reset)?"
  → "Power only. For flashing, give me a small programming header with the UART +
  EN + IO0 lines and BOOT/EN buttons; I'll use an external USB-UART adapter."
- "Exact ESP32 variant / flash size?" → "A WROOM with 4MB is plenty; pick a
  concrete LCSC part."
- "Synthesize?" → "Yes, ./generated."

Off-script: answer as a beginner; observer notes it.

## Traps / required behaviors

- **ESP32 must be LCSC-auto-fetched** (a stock `RF_Module:ESP32-*` symbol is NOT
  first-tier per the BOM spec) — or a `material` question surfaced. No silent
  `RF_Module` substitution.
- **Programming path required and surfaced** — the WROOM has no native USB, so
  power+programming would need a USB-UART bridge; the user chose a header. The
  wiring must carry UART_TX/RX + EN + IO0 to that header. Silent omission →
  `unprogrammable_mcu` gate.
- **Rail current sizing** — WiFi TX bursts pull ~350–500 mA; the LDO/regulator and
  its caps must be sized for that (intent/soundness). A 150 mA LDO is a real defect.
- ESP32 essentials: EN pull-up + RC, IO0 boot strap, bulk + HF decoupling, antenna
  keep-out respected (the library footprint carries the strip; placement should not
  crowd it).
- Three legitimate forks (PIR class, USB role, variant) → asking 2–4 questions is
  expected; asking 0 (silently defaulting all three) is a `failure_honesty` /
  friction problem.
- Synthesize clean: `status == ok`, 0 ERC errors (warnings from vendored/LCSC libs
  are acceptable but should be noted).

## Known design pitfalls

- Regulator undersized for WiFi current bursts → brownouts/resets.
- No first-flash path (UART/EN/IO0 not broken out) → unflashable.
- EN/IO0 strapping resistors or boot buttons missing → won't enter download mode.
- Decoupling too small for the RF current transient (need bulk ≥10µF + 100nF).
- Antenna keep-out crowded by placement (a layout-stage risk; flag if visible).
- MQTT listed as a "comms protocol" in architecture (it's app-layer, not a net).

## What this scenario exercises

A real WiFi MCU board: **LCSC fetch of a non-first-tier module**, a **surfaced
programming path**, **rail sizing for current bursts**, and **legitimate interview
forks** (so the friction band is wider than S01/S02). Stresses `electrical_soundness`,
`part_selection_quality`, `intent_fidelity`, and both observer gates. Derived from
`tests/manual-runs/esp32motionsensor`.
