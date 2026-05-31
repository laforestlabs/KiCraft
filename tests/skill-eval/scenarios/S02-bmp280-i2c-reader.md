---
id: S02
title: USB-C BMP280 I²C reader (CH32V003)
difficulty: moderate
coverage: [i2c-sensor, lcsc-auto-fetch, no-silent-substitution, mcu-programming-path, library-hits]
target_mode: release
interview_expected: false
expected_question_band: [1, 3]
---

## Opening prompt

```
I'd like to design a small USB-C powered board that reads a Bosch BMP280
pressure/temperature sensor over I2C using a CH32V003 microcontroller. Add a tact
switch for user input and a status LED. Power is USB-C (5V) down to 3.3V through a
low-quiescent-current LDO. Target JLCPCB, all-SMD, cheap BOM. I'm comfortable with
electronics but let me know if there's anything you need me to decide.
```

## User-script

- "How do you want to program/flash the CH32V003 the first time?" → "Expose the
  single-wire SWIO programming pin on a small pad or header so I can flash it with a
  WCH-LinkE. Don't drop it silently." *(This is the key fork — the agent should
  raise it; the right answer is a programming path, not omission.)*
- "BMP280 address / CSB / SDO handling?" → "Default I2C address is fine; tie CSB
  high for I2C mode, SDO to GND. Use your judgment."
- "Which LDO?" → "A genuinely low-Iq part (single-digit µA quiescent), e.g.
  AP2112-3.3 or better. The low-Iq matters."
- "Synthesize?" → "Yes, to ./generated."

Off-script: answer as an intermediate maker; observer notes it.

## Traps / required behaviors

- **BMP280 must be LCSC-auto-fetched** (it's a real sensor with an LCSC part), **or**
  a `material` question must surface listing the MPN + fetch command. It must **NOT**
  be silently substituted with stock `Sensor_Pressure:BMP280` and recorded as a
  routine `(defaulted)` assumption. → `silent_substitution` gate watch.
- **MCU programming path required.** CH32V003 SWIO (PD1) is shared with a GPIO;
  the wiring must expose SWIO to a pad/header *or* surface a `material` question.
  Silently shipping no programming path → `unprogrammable_mcu` gate. (This is the
  exact defect the bmp280 prototype run shipped.)
- Library bundles should be preferred where they exist (`usb-c-16p`, an
  `ap2112k-3v3`-class LDO, `ch32v003j4m6`, a tact-switch bundle) over guessed stock
  footprints. A hallucinated footprint name (e.g. a truncated `SW_SPST_PTS645`) that
  fails to resolve is a Class-C finding.
- **low-Iq** LDO constraint honored (intent_fidelity).
- Synthesize clean: `status == ok`, 0 ERC errors.

## Known design pitfalls

- No SWIO/programming path → unflashable board (P0).
- BMP280 decoupling (100 nF close to VDD) omitted.
- I²C pull-ups (SDA/SCL ~4.7k to 3V3) missing — bus won't work.
- CSB/SDO left floating (mode/address indeterminate).
- LDO not actually low-Iq despite the stated constraint.
- Tact switch with no series/pull resistor or debounce consideration.

## What this scenario exercises

The headline failure modes from the original prototype run: **silent part
substitution** and a **dropped MCU programming path**, plus **footprint
hallucination** and **library-vs-stock** preference. Stresses `part_selection_quality`,
`electrical_soundness`, both observer gates, and `computing_error_cleanliness`.
Derived from `tests/manual-runs/bmp280-reader`.
