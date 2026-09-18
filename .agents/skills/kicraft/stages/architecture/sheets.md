Architecture section: the board shape.

This call answers **one section** of the architecture intent — the physical board the design is
built on: its sheets, its rails, and the decisions that belong to no single part. Parts, signals
and obligation ownership are asked for separately, and they may only reference what you state here,
so state the whole board shape once and do not leave a sheet or a rail to be discovered later.

Fields (only these — do not emit `requirements`, `signals` or `obligations`):

- `topologies`: dict mapping a short block name to the chosen topology, e.g. `"POWER"` →
  `"Synchronous buck 5 V to 3.3 V at 1 A"`. Carry the current/voltage rating here when the brief
  states one.
- `comms_protocols`: list, e.g. `["I2C", "USB 2.0 FS"]`.
- `mcu_present`: bool — whether the board has a programmable MCU.
- `power.rails`: dict of net name → `{"voltage": <volts>, "from": "<requirement_id>.<port>"}`.
  `from` names the port that *generates* the rail (a regulator's `output`, a receptacle's `vbus`);
  use `null` for a rail (other than ground) that no part generates — the design-level rail-source
  check reports it rather than inventing a source. Ground is implicit and is never a rail entry.
- `sheets`: one entry per physical sheet. All four required fields plus the optional reuse fields:
  - `name` — uppercase letters/digits with spaces (regex `^[A-Z0-9](?:[A-Z0-9 ]*[A-Z0-9])?$`).
    No hyphens: `"AUDIO CHANNEL 1-2"` is invalid; use a space or spell it out.
  - `stem` — uppercase with underscores (regex `^[A-Z0-9][A-Z0-9_]*$`); KiCraft uses it as the
    filename stem, and the compiler uses it to name a sheet it adds for an `edge:` interface.
  - `role` — short lowercase token for what the sheet is (`mcu`, `power`, `power_input`,
    `regulator`, `sensor`, `driver`, `display`, `interface`, `connector`, …).
  - `function` — one sentence.
  - `from_library` / `library_instance` — a reused Leaf Library entry (`"<name>@<version>"` and
    1-based instance), both set or both null.
  - `replication_group` / `replication_instance` — N sheets that are the same from-scratch circuit
    (3 stepper axes, a 4-channel bank): shared group key, sequential 1..N. Both set or both null,
    never with `from_library`.
- `standard_form_factor`: the template id when the user approved a standard form factor, else null.
- `assumptions`: defaults you applied, each ending `(defaulted)`.

Rules that decide the shape:

- **One sheet per IC domain.** Each distinct non-trivial IC gets its own sheet — the MCU, every
  separate sensor, a display, each standalone regulator — even when they share a bus; the bus is
  exactly what a signal is for. Do not split one integrated IC across sheets (a PD sink+charger, a
  PMIC), and keep an array sheet to the array members plus their per-member companions.
- **Rail-source completeness.** Every declared non-input rail names the port that generates it. An
  ESP32-S3 design declares a 3.3 V rail and a 5 V-to-3.3 V regulator sized for at least 1 A: write
  the rail as `+3V3` (or `3.3V`) and state the converter's rating beside it in `topologies` and in
  the regulator sheet's `function` — e.g. `{"POWER": "Synchronous buck converter, VBUS to 3.3V at
  1 A"}`.
- **Two same-voltage rails are distinct nets** (e.g. `VBUS` and `+5V`): declare the fuse, switch,
  filter, net tie, or converter that connects them, or use one name.
- **Input contract.** Choose the smallest common source contract that covers the named output
  budget, the onboard load, conversion loss and transient margin; never the largest a PD source
  offers. A contract ≥ 2× the downstream converter with ≥ 30 W unused is gross overprovision.
- **External-load budget.** When the board powers a display, LED string, motor or heater, give the
  maximum output current and an input-source budget with headroom (and ask one `blocking: true`
  question first if the brief does not supply the current). The regulated converter's rating must
  exceed the load current so onboard circuits keep headroom.
- **Standard form factor.** A declared standard form factor requires its real owned stacking
  connectors, not generic headers added after wiring; their pin maps come from the template, so
  state only the template id here.
- Emit only the sheets the design actually has: a sheet nobody's part is on is refused, and an
  invented sheet is a place the next section will be told to put a part.
