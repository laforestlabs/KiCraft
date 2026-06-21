Stage 3: Architecture. You are running inside the KiCraft stage sub-agent. Your job is to draft the `architecture` slot of the conversation state and commit it. Follow SKILL.md's "Workflow (follow exactly, in order)" section — this file specifies what the slot must look like.

Given the captured `intent` and `functional_spec` (both available in the `state` field of stage-prep's output), commit to concrete topologies, regulation strategy, MCU presence, comms protocols, rail voltages, and crucially the SHEET HIERARCHY plus inter-sheet connectivity.

**Library reuse.** The `extras.leaves_block` field from stage-prep is the curated catalog of pre-tested sheet implementations the user has built up (or `null` if the library is empty). Reusing one is faster and lower-risk than designing from scratch — but only when the leaf's interface actually fits. Do NOT run `kicraft list-leaves` yourself; the prep output already contains everything.

**Core component defaults.** The `extras.core_defaults_block` field from stage-prep (when present) lists the curated default part per common functional block (regulator tiers, sensors, drivers, interface chips); rows with a `bundle` are already vendored in the parts library, ready to use with zero fetching. Use it when committing to topologies, and NAME the default family in `assumptions` for each block it covers (e.g. `"LDO 3.3V <=500mA: ME6211C33 per core defaults (defaulted)"`): the BOM stage then adopts those exact parts without researching alternatives.

Slot shape (`Architecture`):

- `topologies`: dict mapping block name to the topology choice (e.g. `BOOST` → `"Inductive synchronous boost converter"`, `CHARGER` → `"Standalone linear Li-ion charger IC with USB power-path"`).
- `rail_voltages`: dict of net name → voltage (e.g. `{"VBUS": 5.0, "+3V3": 3.3, "VBAT": 4.2}`).
- `comms_protocols`: list (e.g. `["I2C", "USB 2.0 FS"]`).
- `mcu_present`: bool.
- `sheets`: one `Sheet` per functional block (typically). Each Sheet has:
  - `name` — uppercase with spaces, regex `^[A-Z][A-Z0-9 ]*[A-Z0-9]$` (e.g. `"USB INPUT"`, `"BOOST 5V"`).
  - `stem` — uppercase with underscores, regex `^[A-Z][A-Z0-9_]*$` (e.g. `"USB_INPUT"`, `"BOOST_5V"`). KiCraft uses this as the filename stem.
  - `function` — one-sentence description.
  - `from_library` — `"<name>@<version>"` when reusing a leaf, else null.
  - `library_instance` — 1 for the first instance of a reused leaf, 2 for the second, etc. Null when `from_library` is null. BOTH MUST BE SET OR BOTH NULL.
  - `replication_group` — a shared key (e.g. `"STEPPER_AXIS"`) when this sheet is one of several structurally-identical from-scratch copies; else null.
  - `replication_instance` — 1-based index within the group (1 for the representative, 2, 3 …). Null when `replication_group` is null. BOTH MUST BE SET OR BOTH NULL. Cannot be combined with `from_library`.
- `power_nets`: list of every recognized power/ground net (`VBUS`, `+3V3`, `GND`, etc.). Use canonical names — `VBUS` not `BATT_POSITIVE`, `GND` not `EARTH`.
- `inter_sheet_nets`: every signal that crosses sheet boundaries. Each `InterSheetNet` has `name` and `endpoints` (≥2). Each `SheetPin` endpoint has `sheet` (must match a `Sheet.name`) and `direction` (`input` / `output` / `bidirectional` / `passive`).
  - Power rails: usually `bidirectional` at both ends.
  - Plain signals: `output` at the source, `input` at the sink.
  - Use `passive` only when direction genuinely doesn't apply (rare).
- `assumptions`: defaults applied, each ending `(defaulted)`.

Constraints (enforced by Pydantic):

- Sheet names unique. Sheet stems unique.
- Every `inter_sheet_nets` endpoint must reference a known `Sheet.name`.
- Each `InterSheetNet` needs at least 2 endpoints.
- `Sheet.from_library` and `library_instance` must both be set or both null. `library_instance >= 1`. `from_library` must contain `@`.
- `Sheet.replication_group` and `replication_instance` must both be set or both null. `replication_instance >= 1`. A replication sheet cannot also be a `from_library` sheet.

Recognized power-net name patterns: `VCC`, `VDD`, `VBAT`, `VBUS`, `VSYS`, `+5V`, `+3V3`, `+3.3V`, `5V`, `3V3`, `3.3V`, `+12V`, `12V`, etc.; `GND`, `PGND`, `AGND`, `DGND`. Non-power signal nets are everything else.

Library reuse — additional rules enforced by `stage-commit`:

- For each library leaf you pick, the leaf's hierarchical-label interface MUST match this sheet's endpoints in `inter_sheet_nets` *exactly* (same names + directions, set equality). Use the leaf's label names verbatim.
- For multiple instances of the same leaf, `library_instance` values must be sequential `1..N` with no gaps. Pick distinct `Sheet.name` / `Sheet.stem` for each (e.g. `CHARGER` and `CHARGER_2`).
- If no leaf is a good match, design the sheet from scratch (both `from_library` and `library_instance` null).
- Reevaluate every turn — picking, dropping, or switching a leaf between turns is fine.

**Repeated identical blocks — declare a replication group (solve once, reuse the layout).** When the design needs N structurally-identical copies of the SAME from-scratch sub-circuit — e.g. "3 axes of stepper drivers", a 4-channel relay bank, dual identical motor stages — emit **N separate sheets that share one `replication_group`** with sequential `replication_instance` 1..N (instance 1 is the representative). This is the from-scratch analogue of `library_instance`. Give each sheet its own distinct `name`/`stem` (e.g. `STEPPER_AXIS_X` / `_Y` / `_Z`); the BOM assigns each its own refdes range and wiring its own nets, so ERC still sees N independent circuits with no shorts between them. The signal that drives each instance (STEP/DIR/EN per axis, the per-channel control line) is a normal `inter_sheet_net`. The layout engine then solves ONLY the representative's placement+routing and reuses it for every sibling — far less compute and an identical, predictable layout per copy. A functional-spec block with `count > 1` maps directly to one replication group of that many sheets. Use this only when the copies are truly identical sub-circuits; distinct-but-similar blocks stay separate sheets with no group.

Reminder: every block from the Functional Spec gets a Sheet — don't drop any (a `count`-N block expands to N grouped sheets). Every functional-spec `connection` either appears in `inter_sheet_nets` or is entirely local to one sheet if you reorganized.

**Physical integration — one sheet per IC domain (don't split one chip across sheets; don't merge distinct chips onto one).** The Functional Spec lists *abstract* blocks, but the sheet hierarchy must follow the **physical ICs**: a sheet is one IC plus the support parts that wire directly to it (its connector(s), inductor, CC/DP–DN passives, decoupling, feedback network). Decide this HERE, not at wiring, and get BOTH directions right (the same "decide here, not at wiring" rule as the programming interface below):

- **Merge — don't split one IC across sheets.** When a single highly-integrated IC implements several functional-spec blocks at once, co-locate those blocks on ONE sheet named for the physical domain (e.g. `POWER PATH`), not split into `INPUT` / `CHARGER` / `BOOST` / …. Single-SoC topologies to keep on one sheet: a USB-PD power-bank controller (IP2368 / IP53xx class — PD sink+source, charger, and boost in one chip), a single-chip charger+boost or power-path IC, a PMIC driving several rails. If the split doesn't match the chip, the BOM scatters that IC's pins across sheets and wiring hits a late blocking question about a cross-sheet net it cannot resolve (wiring does not own `inter_sheet_nets`, so an undeclared cross-sheet signal becomes a dangling label).

- **Split — don't merge distinct ICs onto one sheet.** Each distinct non-trivial IC gets its OWN sheet — the MCU/SoC, every separate sensor, a display/OLED, each standalone regulator — *even when they share a bus*. The bus is the boundary: I2C / SPI / UART / a power rail between two chips is exactly what `inter_sheet_nets` is for. Collapsing a multi-IC board onto one sheet yields a single oversized leaf the layout engine places and routes but then REJECTS at leaf-acceptance, so the build never produces a routed parent board. Floor: roughly one sheet per distinct IC; fold only trivial discretes (a lone connector, an RC, a status LED) into the nearest IC's sheet. E.g. an ESP32-S3 plant monitor whose spec lists `USB_C_INPUT, LDO_3V3, ESP32_S3, BME280, SOIL_MOISTURE, OLED` → about five sheets (`POWER` = USB-C + LDO, `MCU` = ESP32-S3, `BME280`, `SOIL`, `OLED`), NOT one `PLANT MONITOR` sheet — and the +3V3/GND rails, the I2C bus (SDA/SCL), the soil ADC line, and the USB D+/D− programming pair between them are all `inter_sheet_nets`.

- **Array sheets are pure.** A sheet that holds a component array (an addressable-LED matrix, a keypad, a resistor-network bank — anything you'd list in BOM `arrays`) must contain ONLY the array members and their per-member companions (e.g. one decoupling cap beside each LED). Put every OTHER part — the power/data header that feeds the array, a driver IC, a level shifter — on its own dedicated sheet, with the shared nets (the array's power rails + its data in/out) declared in `inter_sheet_nets`. The placer grids the array members on a locked grid and co-locates only their 2-pin power companions; a stray header or IC left on the array sheet gets stranded far from the grid and bloats the board. E.g. "4×8 LED array with a back-side power/data header" → a `LED ARRAY` sheet of just the LEDs + decaps and a separate `HEADER` sheet for the connector, joined by `5V`/`GND`/`DATA` inter-sheet nets (the header also takes `side: "back"` in the BOM, not an edge zone).

This co-locates or separates blocks, it never drops them (see the reminder above): a connection between two blocks that land on the same sheet (one IC owns both) becomes sheet-local, so omit it from `inter_sheet_nets`. Minimize inter-sheet nets *within one chip's domain* — a switching node (LX), USB data, and CC lines should almost never leave their IC's sheet — but do NOT minimize them by merging distinct ICs; cross-IC buses belong in `inter_sheet_nets`. (Exception: a `from_library` leaf has a fixed boundary and interface — keep it as its own sheet and match its labels verbatim.)

**Programming interface (when `mcu_present` and the MCU needs external first-time flashing).** An MCU like an ESP32 cannot self-program: its USB or UART0 (TXD0/RXD0) + EN + IO0 (or SWD/JTAG on other parts) need a path to the outside world. DECIDE this here, not at wiring, so it flows into the BOM and gets connected without a late blocking question (which forces a BOM re-run). If the intent/spec already names a part or method, honor it. Otherwise:

- **Prefer an MCU variant with native USB (default recommendation).** When the user asked for a family generically ("an ESP32") without pinning a specific part, pick a native-USB variant: an ESP32-S3 / C3 / S2 / C6 module. Default to the smaller vendored `esp32-s3-mini-1` (~15.4x20mm); step up to the larger vendored `esp32-s3-wroom-1` only when the design needs its extra broken-out GPIO. Flash over the module's built-in USB by routing the board's USB data lines straight to the MCU's native USB pins. No bridge chip, fewest parts, simplest board.
- **USB-UART bridge only for the classic ESP32.** If the design specifically needs the classic ESP32 (ESP32-WROOM-32, which has no native USB), default to an onboard USB-to-UART bridge taken from the core-defaults `usb-uart-bridge` row (the vendored `ch340c` bundle), with DTR/RTS auto-reset to EN/IO0 (so it still flashes over USB with no button dance), sharing the board's USB data lines.

Record the choice in `assumptions` ending `(defaulted)` (e.g. `"MCU: ESP32-S3-MINI-1, flashed over native USB, no bridge (defaulted)"` or `"Programming: onboard CH340C USB-UART per core defaults, auto-reset to EN/IO0 (defaulted)"`) and reflect it in `topologies` (plus a sheet if it is its own block). BOM then adds any bridge/auto-reset parts; wiring connects the path.

Open-question discipline matches earlier stages.
