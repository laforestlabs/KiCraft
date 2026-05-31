---
id: S05
title: Load-cell amplifier (mixed-signal front-end)
difficulty: hard
coverage: [mixed-signal, grounding, analog-front-end, reference, decoupling, electrical-soundness]
target_mode: release
interview_expected: true
expected_question_band: [2, 4]
---

## Opening prompt

```
I'm building a digital scale. I have a standard 4-wire load cell (Wheatstone
bridge, ~2mV/V) and I want a board that amplifies and digitizes its output and
sends the reading to a small MCU. USB-C powered, 3.3V logic. Beginner-intermediate.
Help me get the analog part right — that's what I'm worried about.
```

## User-script

- "ADC approach: a dedicated 24-bit bridge ADC (HX711) or an op-amp + the MCU's
  ADC?" → "An HX711 is exactly what I had in mind — it's the standard part for this."
- "Which MCU, and how is it programmed?" → "Any small 3.3V MCU you like with an I2C
  or simple digital interface; give it a programming header/pads."
- "Load cell connection — header or terminal block?" → "A 4-pin header is fine; the
  cell plugs in."
- "Excitation voltage / reference?" → "Use the HX711's on-chip regulator/reference
  as designed; nothing exotic."
- "Synthesize?" → "Yes, ./generated."

Off-script: answer as a beginner-intermediate maker focused on the analog side.

## Traps / required behaviors

- **This scenario lives or dies on `electrical_soundness`.** The schematic can be
  ERC-clean and still be a bad scale. The observer must check the analog front-end,
  not just connectivity:
  - **Grounding** — bridge/analog return kept clean; no obvious ground loop between
    the load-cell return, HX711 AGND, and the digital/MCU ground. A single-point /
    star-ground intent should be visible or at least surfaced.
  - **HX711 reference/regulator** decoupling per datasheet (AVDD/DVDD caps, the
    regulator output cap, the ratiometric reference to the bridge excitation).
    Ratiometric measurement only works if the bridge excitation and the ADC
    reference are the same node — check that.
  - **Bridge connection** — E+/E−/A+/A− to the right HX711 pins; input filtering if
    present.
- **Programming path** for the MCU (recurring requirement).
- HX711 should be **auto-fetched or library** (not a guessed stock symbol).
- Must synthesize `status == ok`, 0 ERC errors.

## Known design pitfalls

- **Digital and analog grounds tied carelessly** → noise into a 2mV/V signal swamps
  the measurement. The single biggest thing to grade here.
- **Non-ratiometric reference** — ADC reference not derived from the same supply as
  the bridge excitation → readings drift with supply.
- HX711 AVDD/regulator caps missing → noisy/unstable conversions.
- Long unguarded analog traces near the switching/USB section (layout note).
- MCU programming path dropped.
- Input protection / RC filtering on the bridge lines omitted where it'd help.

## What this scenario exercises

The **electrical-soundness / "gotcha" layer** at its hardest: a mixed-signal design
where correctness is about grounding, references, and decoupling rather than
netlist completeness. ERC will likely pass regardless; the score must reflect
whether the *analog design* is sound. Heaviest stress on `electrical_soundness`
(weight 16) and `part_selection_quality`; moderate interview forks.
