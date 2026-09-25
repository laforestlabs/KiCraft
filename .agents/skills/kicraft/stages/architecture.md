Stage 3: Architecture. State the design; the compiler writes the wiring bookkeeping.

This stage used to ask you to hand-write derived data — canonical net names, a `ports` value per
requirement that had to equal a net name declared elsewhere, an endpoint list per net, rail nets,
and the connector a signal needs. Those are *consequences* of the design, and they are now derived
from what you state here by `derive_architecture`. You state intent — **`signals` and each
requirement's `supply` are the inputs that matter** — and the compiler writes:

- the canonical net names, one per signal you declare;
- every `requirements[].ports` binding (which port sits on which net) on **both** ends of a signal;
- each requirement's **supply and ground pins**: `supply` is bound to the family's own supply port
  (a regulator's `input`, a logic part's `vdd`, a driver's `vm`), and the family's reference port
  (its `gnd`/`vss`) to `GND` — and a lowerer's published `vdd`/`gnd` contacts the same way;
- the spare-port ties a reviewed recipe allows to float no longer: a required, groundable port your
  design does not use is tied to `GND` with a note;
- `inter_sheet_nets`, with the endpoint sheets and directions the parts imply;
- `power_nets` and `rail_voltages` from the rails you declare;
- the physical connector an off-board (`edge:`) signal needs, including the rails that peer asked
  for;
- the top-level `obligations` list, from the obligations the intent and functional spec committed;
- the pin/net map of a `standard_form_factor`'s stacking connectors, from the approved template.

When the prompt carries a `STANDARD FORM FACTOR` block, that template's fixed connectors are the
board's host interface and the block lists every role with its exact pin/net map. Emit exactly one
requirement per listed role (`role: "connector"`, `family: "pin-header"`, `parameters:
{"rows": 1, "gender": "female"}`, `standard_stacking_role: <role>`, owned by the functional block
that owns the host interface) — never one composite or renamed header in their place.

Do not write any of those. There is no `ports`, `inter_sheet_nets`, `power_nets` or
`rail_voltages` field in this slot, and a port you name must exist (see *Port keys*). The optional
refinements — `ties`, `supply_bindings`, `reference_bindings`, `declared_ports` — are honoured
where you state them and derived where you do not; never write them to satisfy a check.

Slot shape:

- `topologies`: dict mapping a short block name to the chosen topology (e.g. `"POWER"` →
  `"Synchronous buck 5 V to 3.3 V"`).
- `comms_protocols`: list (e.g. `["I2C", "USB 2.0 FS"]`).
- `mcu_present`: bool.
- `power.rails`: dict of net name → `{"voltage": <volts>, "from": "<requirement_id>.<port>"}`.
  `from` names the port that generates the rail (a regulator's `output`, a receptacle's `vbus`);
  use `null` for a rail (other than ground) that no part you named generates — the design-level
  rail-source check reports it rather than inventing a source. Ground is implicit: bind a part to
  it with `"gnd"` (recipes carry that port) or a `ties` entry.
- `sheets`: one entry per physical sheet, all four required fields plus the optional reuse fields:
  - `name` — uppercase letters/digits with spaces (regex `^[A-Z0-9](?:[A-Z0-9 ]*[A-Z0-9])?$`).
    No hyphens: `"AUDIO CHANNEL 1-2"` is invalid; use a space or spell it out.
  - `stem` — uppercase with underscores (regex `^[A-Z0-9][A-Z0-9_]*$`); KiCraft uses it as the
    filename stem, and the compiler uses it to name a sheet it adds for an `edge:` interface.
  - `role` — short lowercase token for what the sheet is (`mcu`, `power`, `power_input`,
    `regulator`, `sensor`, `driver`, `display`, `interface`, `connector`, …).
  - `function` — one sentence.
  - `from_library` / `library_instance` — a reused Leaf Library entry (`"<name>@<version>"` and
    1-based instance), both set or both null. Its hierarchical labels must match your signals
    exactly; the compiler binds the nets, not the labels.
  - `replication_group` / `replication_instance` — N sheets that are the same from-scratch circuit
    (3 stepper axes, a 4-channel bank): shared group key, sequential 1..N. Both set or both null,
    never with `from_library`.
- `requirements`: one per physical part/block implementation. Fields:
  - `id` (stable lowercase), `sheet` (an exact `sheets[].name`), `role` (one of the CircuitRole
    values: `mcu_core`, `power_input`, `regulator`, `programming`, `bus_interface`, `sensor`,
    `driver`, `analog_block`, `user_io`, `connector`).
  - `family` and `exact_part`: the recipe/lowerer family from `extras.circuit_recipes` /
    `extras.circuit_lowerers`, and the exact ordering code when the user named one. Preserve the
    user's part identity; do not substitute a different family.
  - `parameters`: the family's own bounded keys (`rows`, `gender`, `output_voltage`,
    `supply_voltage`, …). Use only keys the reference data advertises.
  - `supply`: the rail this part is powered from (a declared `power.rails` name). The compiler binds
    it to the family's supply port (a regulator's input, a logic part's vdd, a driver's vm); a
    family that publishes no supply port — a status LED draws its current from its `drive` signal —
    simply has no pin on that rail, and saying so is not an error. Omit `supply` only for a part
    that genuinely draws nothing from a declared rail.
  - `programming`: how a programmable part is flashed — `native_usb`, `usb_uart_bridge`, `swd`,
    `updi`, `bootsel`, `none`. See *Programming*.
  - `interfaces`: the recipe's interface names you use (`i2c_controller`, `spi_controller`,
    `uart`, `can_controller`, `pwm`, `adc`, `parallel_output`). The compiler derives this list from
    the ports you bind: a bus member (`sda`/`scl`, `sclk`/`mosi`/`miso`/`cs`, `tx`/`rx`,
    `can_tx`/`can_rx`, `pwm_*`, `adc_*`, `parallel_<i>`) requests its interface, an interface you
    name with no such port bound is not a request, and `parallel_output` takes its count from the
    `parallel_<i>` ports you bind. You do not have to get the list exactly right; the ports decide.
  - `functional_blocks`: the exact committed Functional Spec block names this part implements.
    Every block needs an owner; several requirements may implement one block.
  - `obligations`: the committed intent/functional-spec obligation rows this part implements,
    copied verbatim (`kind` and `original_obligation_id` identify them). Attach each committed
    obligation to the requirement that implements it — several requirements may carry the same row
    when your design implements one obligation in more than one place (three binding posts), and a
    `quantity` count over the whole design may stay at the top level. The top-level `obligations`
    list itself is written for you from the committed set: do not copy the rows into it.
  - `ties`: rare. The compiler already ties the family's ground, its unused groundable ports, its
    default-tied control ports, and a standard template's stacking pin map. State a tie only for a
    direct connection no signal names and no rule derives: a connector shell to `GND`, an enable
    pin to the rail it runs on. The net must be `GND`, a declared rail, or a net one of your
    signals names. A port a recipe advertises with a default tie (a regulator's `enable`, a
    follower's `feedback`) is tied to the port it names when your design wires neither — the
    always-on / unity-gain configuration — so bind it only when the brief asks for the
    controllable configuration, and that binding wins.
  - `supply_bindings` / `reference_bindings`: rare refinements for a part with several supply or
    reference domains (an isolated side named `GND_ISO`); a domain must be a declared zero-volt
    rail. The compiler binds the single supply/reference case itself — do not restate it.
  - `declared_ports`: **only** for a part with no curated recipe (see *Parts the code has never
    seen*); on a curated or lowerer family there is nothing to declare.
- `signals`: one entry per named application signal. `name` is the net name (canonical, unique,
  not a declared rail); `from` is `"<requirement_id>.<port>"`, `to` is one peer reference or a list
  of them, each `"<requirement_id>.<port>"` or `"edge:<NAME>"`. Optional:
  - `rails`: rails an off-board peer needs on its connector (a 5 V LED string behind 3.3 V logic).
    Name each one exactly as you declared it under `power.rails` — `["+5V"]` is refused when the
    rail you declared is `VBUS`. A signal whose source port already carries a declared rail is that
    rail: its peers join the rail rather than a second net on the same pin.
  - `start` / `end`: expand `{n}` in `name`, `from` and `to` over an inclusive numeric range — use
    this for a bus or a repeated channel ([`HUB75_R{n}` – `hub75.r{n}`, 0..1], never a hand-typed
    list with a hole in it). `{n}` must appear exactly once in each of the three strings.
- `assumptions`: defaults you applied, each ending `(defaulted)`.

**Port keys.** A signal reference names a port of the part it connects. Valid keys are:
- every `ports` name the selected recipe advertises in `extras.circuit_recipes` (its reviewed
  interface), including supply/ground and conditional ports;
- the advertized keys of the recipe's `interfaces` you declared, for bus members (`sda`/`scl`,
  `sclk`/`mosi`/`miso`/`cs`, `tx`/`rx`, `can_tx`/`can_rx`, `pwm_*`, `adc_*`, `parallel_<i>`);
- MCU application pins, by capability: `output_<id>`, `input_<id>`, `touch_<id>` (and `gpio<n>`
  when you mean a specific numbered pin). The pin allocator owns the assignment;
- the lowerer's keyed ports for a lowerer family (`pin1` … `pinN` for `pin-header`, in physical
  pin order — the pins are ordered by the order you bind them, so bind each connector pin once).

**How the compiler binds ports.** A port key may be bound exactly once, and the compiler is the one
that binds it:

- A signal binds its source port to its own net and each peer port to the same net. Two signals that
  name the same **peer** port are refused, naming the ports that were available: that really is two
  nets on one pin.
- Two signals that leave the **same** source port are one physical net under two names — the first
  name owns it, the second signal's peers join it, and the join is recorded as a derived assumption.
  That is legal, not a habit: prefer one signal per net.
- A port a rail or ground already owns keeps that net; a signal out of it joins the peers to the
  rail instead of naming a second net.

A family whose port list has a single signal port (`switch-input`'s `signal`; `connector-bank`'s
`signal0…signalN`) realises **one instance per requirement** — three microstep switches are three
requirements, or the family's numbered port pattern, never one `signal` port bound three times.

A key that is none of the published ones is refused, naming the ports that do exist. Do not invent a
port to carry a signal the part cannot. A required port of a selected recipe nobody binds is
refused by name: wire it with a signal or a `ties` entry. A port whose signal continues off the
board (a WS2812 driver's `data_out` feeding the string) is bound like any other, to the same `edge:`
peer as the rest of that interface.

**Off-board interfaces.** Send a signal to `edge:<NAME>` and the compiler adds the connector: a
USB data socket when the source is a native-USB MCU's `usb_dm`/`usb_dp` (both must be sent to the
same edge), otherwise a one-row pin header on its own sheet, carrying your signals in order, then
ground, then the rails you listed in `rails`. So `LED_DATA` → `edge:LED_STRING` with
`"rails": ["+5V"]` gives a three-pin output: data, ground, 5 V.

**Design obligations that remain yours** (each has a check behind it):
- **One sheet per IC domain.** Each distinct non-trivial IC gets its own sheet — the MCU, every
  separate sensor, a display, each standalone regulator — even when they share a bus; the bus is
  exactly what a signal is for. Do not split one integrated IC across sheets (a PD sink+charger,
  a PMIC), and keep an array sheet to the array members plus their per-member companions.
- **Every block owned.** Every Functional Spec block needs explicit `functional_blocks` ownership
  by a requirement on a sheet; a sheet name or an assumption is not implementation evidence.
- **Rail-source completeness.** Every declared non-input rail names the port that generates it. An
  ESP32-S3 design declares a 3.3 V rail and a 5 V-to-3.3 V regulator sized for at least 1 A: write
  the rail as `+3V3` (or `3.3V`) and state the converter's current rating beside it in `topologies`
  and in the regulator sheet's `function` — e.g. `{"POWER": "Synchronous buck converter, VBUS to
  3.3V at 1 A"}`. Two same-voltage rails (e.g. `VBUS` and `+5V`) are distinct nets: declare the
  fuse, switch, filter, net tie, or converter that connects them, or use one name.
- **Input contract.** Choose the smallest common source contract that covers the named output
  budget, the onboard load, conversion loss and transient margin; never the largest a PD source
  offers. A contract ≥ 2× the downstream converter with ≥ 30 W unused is gross overprovision.
- **External-load budget.** When the board powers a display, LED string, motor or heater, give the
  maximum output current and an input-source budget with headroom (and ask one `blocking: true`
  question first if the brief does not supply the current). The regulated converter's rating must
  exceed the load current so onboard circuits keep headroom.

**Programming.** Decide it here, from the part you chose:
- **Native USB (ESP32-S3 MINI/WROOM, ESP32-C3 MINI, RP2040):** set `"programming": "native_usb"`,
  bind the family's `usb_dm`/`usb_dp` and send **both** to `edge:<a USB name>`. That is the
  mandatory physical data connector — `native_usb` cannot program over a header, and adding a
  UART/SWD header instead is a repairable error, never an alternative. Do not add a bridge.
  RP2040 also needs its BOOTSEL path (its recipe owns it); an ESP32 needs a BOOT strap pull path
  during reset (the native-USB recipes own it).
  The socket also carries power: an edge that carries the USB data pair must have the rail that
  socket exposes declared under `power.rails`, at about 5 V and named `VBUS` or `+5V`. The socket
  itself is written by the compiler for the edge, so it is NOT a requirement you declare and you
  cannot source the rail from it: state `{"VBUS": {"voltage": 5.0, "from": null}}` (the rail is
  the host's, and the design assumptions record it). Naming an undeclared requirement here is
  refused (`unknown_signal_requirement`), and a USB data edge with no such rail is refused
  (`usb_connector_supply_unknown`). Never tie this rail to the board's battery or 3.3 V rail.
- **Classic ESP32 (ESP32-WROOM-32):** add the vendored `ch340c` bridge as its own requirement with
  `"programming": "usb_uart_bridge"`, `supply` on the same 3.3 V rail as the MCU, and typed UART
  ports crossed device-relative: bridge `tx` → MCU `uart_rx`, bridge `rx` → MCU `uart_tx`, with
  `dtr_n`/`rts_n` on the MCU's fixed ports. Set the bridge's `supply_voltage` to the rail it
  actually runs on, and never drive 3.3 V MCU pins from a 5 V bridge.
- **USB power for a battery design:** the USB port is data/programming only; VBUS is a separate
  host-supplied net, never tied to VBAT or the 3.3 V rail, and needs no power mux.

**Parts the code has never seen.** A curated recipe owns its connections. For anything else:

- A part with no recipe is acceptable when you state its interface: give the requirement
  `declared_ports`, one entry per pin function — `{"key": "...", "pin": "<the ordering code's own
  contact>", "direction": "input|output|bidirectional|passive|power", "function": "<what it
  does>"}`. `pin` is what makes the claim checkable: the BOM resolves the real symbol and refuses a
  function claimed on a contact that part does not have, and a claim that states no pin at all.
  The compiler then treats the part like a curated one: it binds the wires you named, exposes the
  off-board connector, and records the interface as a **claim** in the review and in the
  assumptions — the pin functions are yours, not verified against a curated table. Wires you name
  across a declared interface are the risk the review cannot check: get the pins and functions
  right.
- Without `declared_ports`, a requirement whose ports a signal needs is **refused once**, naming
  the part: nothing can be wired to a part whose connections are unknown. State the interface (or
  use a family with a curated recipe) and re-emit.
- Whatever the provenance, the BOM stage still refuses anything that does not resolve to a real
  symbol, footprint and orderable part; never invent a library prefix or package name.

**Reference data (from stage-prep `extras`).**
- `extras.circuit_recipes`: the curated exact recipes — `family`, `exact_part`, `ports`,
  `interfaces`, `owned_parts`, `parameter_defaults`, `allocatable_capabilities`. Naming a listed
  family and part is what selects it; do not duplicate it in a selection field. Each recipe's
  `owned_parts` are already implemented inside it — do not add a second crystal or pull network for
  an owned role, and do not invent external ports for its internal nets.
- `extras.circuit_lowerers`: generic connectors, passive networks and repeated simple channels.
  Prefer one when it fits; emit its `parameter_keys` and its advertised port keys, including every
  `required_port_keys`. For `pin-header`, `rows` is 1 or 2 parallel contact rows, not the total pin
  count, and the port keys are `pin1` … `pinN` in physical order.
- `extras.core_defaults_block`: the curated default part per common block. Use it and name the
  default family in `assumptions` so the BOM adopts that exact part.
- `extras.leaves_block`: pre-tested sheet implementations; reuse one when its interface fits.

**Clarifying questions.** Always produce the complete slot with safe defaults; a question never
replaces the slot. Ask only when no safe default exists and a wrong answer would change the
manufactured board — at most 3, one decision each, 2-4 suggested answers each, marked
`blocking: true` when the build cannot proceed without the answer. Never ask to confirm a default or
a fact already in the brief or the design state.
