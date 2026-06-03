---
id: S06
title: ESP32 WiFi LED-matrix mood light with accelerometer
difficulty: hard
coverage: [wifi-mcu, led-array, array-leaf-grid-placement, array-leaf-route-cache, i2c-accelerometer, lcsc-auto-fetch, programming-path, gpio-budget, rail-current-sizing, antenna-keepout]
target_mode: release
interview_expected: true
expected_question_band: [3, 5]
---

## Opening prompt

```
I want to build a small USB-C powered "mood light" gadget around an ESP32-WROOM.
It drives a little grid of plain single-color LEDs (think a 4x4 array, 16 LEDs) and
also reads a 3-axis accelerometer so it can react to being tilted or tapped, and it
reports state over WiFi. Power comes from the 5V USB-C input with onboard 3.3V
regulation. I'm a hobbyist who can solder but I'm shaky on the electrical details,
so please walk me through the decisions that matter.
```

## User-script

- "How should the LED grid be driven: straight from GPIO pins (one series resistor
  per LED, simplest, but it eats a lot of pins) or through a shift register / LED
  driver IC (more LEDs from fewer pins, more complexity)?" → "Keep it simple: drive
  them directly from GPIO, one series resistor on each LED. A 4x4 grid of 16 LEDs is
  exactly what I want."
- "Same LED part for all 16?" → "Yes, all identical, a plain green 0603 LED."
- "Which accelerometer and bus: an I²C part like the LIS3DH or ADXL345 (two wires,
  simplest), or SPI (faster, more pins)?" → "I²C is fine. Pick a common one with a
  concrete LCSC part: I have no preference, LIS3DH is good if you like it."
- "USB-C: power-only, or power + programming (USB-UART bridge with auto-program/
  reset)?" → "Power only. Give me a small programming header that breaks out UART
  TX/RX + EN + IO0, plus BOOT and EN buttons; I'll flash with an external USB-UART
  adapter."
- "Exact ESP32 variant / flash size?" → "A WROOM with 4MB is plenty, pick a concrete
  LCSC part."
- "Synthesize?" → "Yes, ./generated."

Off-script: answer as a shaky-on-electrical hobbyist; the observer notes any
off-script answer.

## Traps / required behaviors

- **The 16-LED block must be recognized and emitted as an array leaf** (an explicit
  `rows`/`cols`/serpentine array hint over the 16 LED refs, with their series
  resistors as the leaf's only other parts). This is the whole point of the run: it
  is the construct PR#8 grid-places deterministically and PR#9 route-caches. Failure
  tell: the LEDs are handed to the force/SA placement solver and end up scattered
  (the exact regression PR#8 fixed), or never grouped as a grid at all.
- **Pure array leaf preserved** — the LED leaf should contain *only* the LED grid +
  two-terminal passives (16 LEDs + 16 series resistors, optionally a local decoupling
  cap). If the subject injects a shift register / constant-current driver / matrix
  multiplexing transistors (parts with >2 pads) into that leaf, it (a) contradicts the
  user's explicit "direct from GPIO, keep it simple" intent and (b) defeats
  `leaf_is_fully_array`, so the route-cache path is never taken. Flag as intent +
  quality.
- **Grid placement actually applied** — in the synthesized PCB the 16 LEDs sit on a
  regular pitched grid (serpentine ordering), `locked`, flagged array members, not
  optimizer-scattered. Series resistors placed in a strip near the grid.
- **GPIO budget honored (the hard part).** Direct-driving 16 LEDs needs 16 output
  pins *on top of* I²C (SDA/SCL), the UART programming lines (TX/RX), EN and IO0. A
  WROOM does not have 16 free general-purpose outputs once you exclude: the input-only
  pins **GPIO34–39** (cannot drive an LED), the flash pins **GPIO6–11** (unusable),
  and ideally the boot-strapping pins **GPIO0/2/12/15** (driving these at boot can
  brick the boot sequence). A sound design notices the pressure and either assigns
  strapping pins deliberately with the boot levels respected, or pushes back on the
  count. **LEDs wired to input-only or flash pins is a P1 defect.**
- **Accelerometer LCSC auto-fetched**, not a stock `Sensor_Motion:*` / `Sensor:*`
  symbol (silent substitution → gate). I²C bus correctly formed: pull-ups present
  **once** on SDA/SCL (not duplicated, not absent), the accel's address/CS-strap pin
  tied to a defined level, local decoupling on its VDD.
- **Programming path required and surfaced** — WROOM has no native USB; the user took
  power-only, so the header must carry UART TX/RX + EN + IO0, with EN pull-up + RC and
  the IO0 boot strap and BOOT/EN buttons. Silent omission → `unprogrammable_mcu` gate.
- **Rail current sizing** — the 3.3V regulator must cover ESP32 WiFi TX bursts
  (~350–500 mA) **plus** the LED array (16 × ~5–10 mA ≈ up to ~160 mA if all lit) plus
  the accel. A 150 mA LDO is a real defect; bulk (≥10 µF) + HF (100 nF) decoupling
  sized for the RF transient.
- **ESP32 itself LCSC-fetched** (a stock `RF_Module:ESP32-*` symbol is not first-tier
  per the BOM spec). No silent `RF_Module` substitution.
- Antenna keep-out respected by placement (the WROOM library footprint carries the
  strip; the LED grid / parts must not crowd it).
- Synthesize clean: `status == ok`, 0 ERC errors (warnings from vendored/LCSC libs are
  acceptable but should be noted).

## Known design pitfalls

- **LEDs assigned to input-only GPIO34–39 or flash GPIO6–11** → won't light / board
  won't boot. The headline electrical gotcha here.
- LEDs on strapping pins (0/2/12/15) without respecting required boot levels → boot
  into the wrong mode / fail to flash.
- **LED array scattered by the force/SA solver** instead of grid-placed → the PR#8
  regression; flag if the layout shows it.
- Over-engineering the LED block with a driver IC the user declined → intent
  infidelity *and* breaks the route-cache eligibility.
- Regulator undersized for WiFi-burst + full-array combined load → brownouts/resets.
- Missing per-LED series resistors (LEDs straight onto GPIO) → overcurrent / dead pins.
- I²C pull-ups missing, duplicated, or wrong value; accel address pin left floating.
- No first-flash path (UART/EN/IO0 not broken out) → unflashable.
- Decoupling too small for the RF current transient (need bulk ≥10 µF + 100 nF).
- Antenna keep-out crowded by the grid placement.

## What this scenario exercises

The **array-leaf pipeline added in PR#8 (programmatic grid placement) and PR#9 (routed-
copper cache for deterministic array leaves)**, driven from a realistic mixed board: a
16-LED direct-drive matrix (the pure array leaf), an **I²C accelerometer LCSC fetch with
no silent substitution**, and a **WiFi MCU** carrying a surfaced **programming path** and
**rail sizing for current bursts**. The standout judgment layer is the **GPIO budget**:
fitting 16 direct LED drives plus I²C plus the programming lines onto a WROOM without
landing on input-only or flash pins. Stresses `electrical_soundness` (GPIO budget, array
placement, rail sizing, programming path), `part_selection_quality`, `intent_fidelity`,
and the array placement/route-cache code specifically. Runs in **release** target: the
pipx `kicraft` is an editable install, so the installed CLI already carries PR#8/#9.
