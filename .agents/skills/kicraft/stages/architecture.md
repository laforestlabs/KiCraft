Stage 3: Architecture. Draft the complete `architecture` slot and commit it through the workflow in the parent `SKILL.md`; this file defines the slot contract.

Given the captured `intent` and `functional_spec` (both available in the `state` field of stage-prep's output), commit to concrete topologies, regulation strategy, MCU presence, comms protocols, rail voltages, and crucially the SHEET HIERARCHY plus inter-sheet connectivity.

**Library reuse.** The `extras.leaves_block` field from stage-prep is the curated catalog of pre-tested sheet implementations the user has built up (or `null` if the library is empty). Reusing one is faster and lower-risk than designing from scratch — but only when the leaf's interface actually fits. Do NOT run `kicraft list-leaves` yourself; the prep output already contains everything.

**Core component defaults.** The `extras.core_defaults_block` field from stage-prep (when present) lists the curated default part per common functional block (regulator tiers, sensors, drivers, interface chips); rows with a `bundle` are already vendored in the parts library, ready to use with zero fetching. Use it when committing to topologies, and NAME the default family in `assumptions` for each block it covers (e.g. `"LDO 3.3V <=500mA: ME6211C33 per core defaults (defaulted)"`): the BOM stage then adopts those exact parts without researching alternatives.

Slot shape (`Architecture`):

- `topologies`: dict mapping block name to the topology choice (e.g. `BOOST` → `"Inductive synchronous boost converter"`, `CHARGER` → `"Standalone linear Li-ion charger IC with USB power-path"`).
- `rail_voltages`: dict of net name → voltage (e.g. `{"VBUS": 5.0, "+3V3": 3.3, "VBAT": 4.2}`).
- `comms_protocols`: list (e.g. `["I2C", "USB 2.0 FS"]`).
- `mcu_present`: bool.
- `sheets`: one `Sheet` per functional block (typically). Each Sheet has:
  - `name` — uppercase letters/digits with spaces, regex `^[A-Z0-9](?:[A-Z0-9 ]*[A-Z0-9])?$` (e.g. `"USB INPUT"` or `"5V BUCK"`). NO hyphens or punctuation — `"AUDIO CHANNEL 1-2"` is invalid; use a space (`"AUDIO CHANNEL 1 2"`) or spell it out (`"AUDIO CHANNEL 1 AND 2"`).
  - `stem` — uppercase with underscores, regex `^[A-Z0-9][A-Z0-9_]*$` (e.g. `"USB_INPUT"` or `"5V_BUCK"`). KiCraft uses this as the filename stem.
  - `function` — one-sentence description.
  - `from_library` — `"<name>@<version>"` when reusing a leaf, else null.
  - `library_instance` — 1 for the first instance of a reused leaf, 2 for the second, etc. Null when `from_library` is null. BOTH MUST BE SET OR BOTH NULL.
  - `replication_group` — a shared key (e.g. `"STEPPER_AXIS"`) when this sheet is one of several structurally-identical from-scratch copies; else null.
  - `replication_instance` — 1-based index within the group (1 for the representative, 2, 3 …). Null when `replication_group` is null. BOTH MUST BE SET OR BOTH NULL. Cannot be combined with `from_library`.
- `requirements`: a nonempty list of typed physical implementation contracts.
  Each has a stable lowercase `id`, owning `sheet`, `role`, `family`, explicit
  `ports`, bounded `parameters`, and nonempty `functional_blocks` containing
  exact names from the committed Functional Spec. Preserve explicit part
  identities in `exact_part`. Every functional block needs an owner, including
  blocks merged onto another block's sheet. Several requirements may serve one
  block; a composite requirement may cover several blocks only when it really
  implements them all. A sheet name, assumption, or empty requirement is not
  implementation evidence.
  `ports` maps logical port keys to actual net names, for example
  `{"vbus": "VBUS", "gnd": "GND", "cc1": "CC1"}`. Values are not
  directions such as `"power"`, `"ground"`, or `"bidirectional"`; directions
  belong on sheet endpoints. Every declared inter-sheet endpoint needs its
  exact net name bound by a requirement on that endpoint's sheet.
- `power_nets`: list of every recognized power/ground net (`VBUS`, `+3V3`, `GND`, etc.). Use canonical names — `VBUS` not `BATT_POSITIVE`, `GND` not `EARTH`. These entries are nets, not sheets. Physical power converters, battery holders, and power connectors still need owning sheets and typed requirements.
- `inter_sheet_nets`: every signal that crosses sheet boundaries. Each `InterSheetNet` has `name` and `endpoints` (≥2). Each `SheetPin` endpoint has `sheet` (must match a `Sheet.name`) and `direction` (`input` / `output` / `bidirectional` / `passive`).
  - Net names are unique. A shared net is ONE record containing all participating
    sheet endpoints, not one record per functional connection or pair of sheets.
    Combine the participants; do not rename or omit a branch to evade uniqueness.
  - For sequential numeric nets, use `inter_sheet_net_ranges`, not a long repeated
    `inter_sheet_nets` array: `name_pattern` contains exactly `{n}`, inclusive
    `start`/`end`, and common endpoints. Split discontinuous GPIO sets into separate
    ranges; every GPIO must belong to the selected recipe's `allocatable_gpios`
    evidence in `extras.circuit_recipes`. Never fill holes in that set, use its
    highest number as a guessed contiguous range, or drop a required net to fit.
    Ranges expand before canonical validation; never overlap an explicit net or
    another range. If the chosen device cannot carry a required contract, correct
    the device/interface decision rather than silently deleting the contract.
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

Reminder: every physical block from the Functional Spec needs explicit requirement ownership. Preserve repeated circuit instances with replication groups or the primitive's bounded channel count, as appropriate. Bare supply/ground nets do not get sheets, but `category: "power"` does NOT mean a block is merely a net: regulators, power inputs, and load connectors remain hardware. Every functional-spec connection either crosses declared sheet boundaries or is local to one sheet through its explicit owners.

**Physical integration — one sheet per IC domain (don't split one chip across sheets; don't merge distinct chips onto one).** The Functional Spec lists *abstract* blocks, but the sheet hierarchy must follow the **physical ICs**: a sheet is one IC plus the support parts that wire directly to it (its connector(s), inductor, CC/DP–DN passives, decoupling, feedback network). Decide this HERE, not at wiring, and get BOTH directions right (the same "decide here, not at wiring" rule as the programming interface below):

- **Merge — don't split one IC across sheets.** When a single highly-integrated IC implements several functional-spec blocks at once, co-locate those blocks on ONE sheet named for the physical domain (e.g. `POWER PATH`), not split into `INPUT` / `CHARGER` / `BOOST` / …. Single-SoC topologies to keep on one sheet: a USB-PD power-bank controller (IP2368 / IP53xx class — PD sink+source, charger, and boost in one chip), a single-chip charger+boost or power-path IC, a PMIC driving several rails. If the split doesn't match the chip, the BOM scatters that IC's pins across sheets and wiring hits a late blocking question about a cross-sheet net it cannot resolve (wiring does not own `inter_sheet_nets`, so an undeclared cross-sheet signal becomes a dangling label).

- **Split — don't merge distinct ICs onto one sheet.** Each distinct non-trivial IC gets its OWN sheet — the MCU/SoC, every separate sensor, a display/OLED, each standalone regulator — *even when they share a bus*. The bus is the boundary: I2C / SPI / UART / a power rail between two chips is exactly what `inter_sheet_nets` is for. Collapsing a multi-IC board onto one sheet yields a single oversized leaf the layout engine places and routes but then REJECTS at leaf-acceptance, so the build never produces a routed parent board. Floor: roughly one sheet per distinct IC; fold only trivial discretes (a lone connector, an RC, a status LED) into the nearest IC's sheet. E.g. an ESP32-S3 plant monitor whose spec lists `USB_C_INPUT, LDO_3V3, ESP32_S3, BME280, SOIL_MOISTURE, OLED` → about five sheets (`POWER` = USB-C + LDO, `MCU` = ESP32-S3, `BME280`, `SOIL`, `OLED`), NOT one `PLANT MONITOR` sheet — and the +3V3/GND rails, the I2C bus (SDA/SCL), the soil ADC line, and the USB D+/D− programming pair between them are all `inter_sheet_nets`.

- **Array sheets are pure.** A sheet that holds a component array (an addressable-LED matrix, a keypad, a resistor-network bank — anything you'd list in BOM `arrays`) must contain ONLY the array members and their per-member companions (e.g. one decoupling cap beside each LED). Put every OTHER part — the power/data header that feeds the array, a driver IC, a level shifter — on its own dedicated sheet, with the shared nets (the array's power rails + its data in/out) declared in `inter_sheet_nets`. The placer grids the array members on a locked grid and co-locates only their 2-pin power companions; a stray header or IC left on the array sheet gets stranded far from the grid and bloats the board. E.g. "4×8 LED array with a back-side power/data header" → a `LED ARRAY` sheet of just the LEDs + decaps and a separate `HEADER` sheet for the connector, joined by `5V`/`GND`/`DATA` inter-sheet nets (the header also takes `side: "back"` in the BOM, not an edge zone).

This co-locates or separates blocks, it never drops them (see the reminder above): a connection between two blocks that land on the same sheet (one IC owns both) becomes sheet-local, so omit it from `inter_sheet_nets`. Minimize inter-sheet nets *within one chip's domain* — a switching node (LX), USB data, and CC lines should almost never leave their IC's sheet — but do NOT minimize them by merging distinct ICs; cross-IC buses belong in `inter_sheet_nets`. (Exception: a `from_library` leaf has a fixed boundary and interface — keep it as its own sheet and match its labels verbatim.)

**Programming interface (when `mcu_present` and the MCU needs external first-time flashing).** An MCU like an ESP32 cannot self-program: its USB or UART0 (TXD0/RXD0) + EN + IO0 (or SWD/JTAG on other parts) need a path to the outside world. DECIDE this here, not at wiring, so it flows into the BOM and gets connected without a late blocking question (which forces a BOM re-run). If the intent/spec already names a part or method, honor it. Otherwise:

- **Prefer an MCU variant with native USB (default recommendation).** When the user asked for a family generically ("an ESP32") without pinning a specific part, pick a native-USB variant: an ESP32-S3 / C3 / S2 / C6 module. Default to the smaller vendored `esp32-s3-mini-1` (~15.4x20mm); step up to the larger vendored `esp32-s3-wroom-1` only when the design needs its extra broken-out GPIO. Flash over the module's built-in USB by routing the board's USB data lines straight to the MCU's native USB pins. No bridge chip, fewest parts, simplest board.
- **Native-USB family ⇒ a physical USB data connector is MANDATORY.** For ESP32-S3 MINI/WROOM, ESP32-C3 MINI and RP2040 the selected recipe always programs over its own USB pair — `native_usb` is fixed on and there is no header escape. Whenever you select one of these, add a USB device requirement (role `connector`, family `usb-c-usb2-device`) on its own USB sheet, bind the MCU's `usb_dm` to `USB_D_N` and `usb_dp` to `USB_D_P`, and declare each of those as an inter-sheet net with TWO bidirectional endpoints (the MCU sheet and the USB sheet). Do NOT add a UART/SWD programming header for these families; missing USB is a repairable architecture error (the build rejects a header-only choice), never a reason to substitute a header.
- **USB-UART bridge only for the classic ESP32.** If the design specifically needs the classic ESP32 (ESP32-WROOM-32, which has no native USB), default to an onboard USB-to-UART bridge taken from the core-defaults `usb-uart-bridge` row (the vendored `ch340c` bundle), with DTR/RTS auto-reset to EN/IO0 (so it still flashes over USB with no button dance), sharing the board's USB data lines.
- **Typed UART ports are device-relative, not net-name-relative.** Cross the bridge's
  `tx` output to the MCU's fixed `uart_rx` input, and the bridge's `rx` input to
  the MCU's fixed `uart_tx` output. For bridge `tx: UART_TX, rx: UART_RX`, bind
  MCU `uart_rx: UART_TX, uart_tx: UART_RX`; do not match TX to TX by label.
  Bridge `dtr_n` / `rts_n` bind the same MCU fixed control ports and net values
  with `auto_reset: true`; the recipe owns the reset circuit, not GPIO allocation.
  Keep all application GPIO contracts separate and preserve their full capacity.
- **CH340C supply mode must match the actual rail.** Bind `vdd` to a proven rail
  and set numeric `supply_voltage: 3.3` or `5.0`; `rail_voltages` is authoritative,
  and an unknown `VBUS` label is not evidence of 5 V. At 3.3 V, V3 is tied to VCC;
  at 5 V, V3 has only its 100 nF bypass to ground. A direct CH340C-to-ESP32 UART
  uses the same 3.3 V supply net and ground for both devices, not a 5 V bridge
  driving 3.3 V MCU pins. WCH CH340DS1 v3D §6.2 guarantees non-USB VOH ≥4.4 V
  at VCC=5 V; ESP32-C3-MINI-1 datasheet v2.2 Table 6-3 limits input high to
  VDD+0.3 V (3.6 V at 3.3 V). Separate supplies require real typed isolation or
  level translation with distinct conductors, or an explicit common-rail repair;
  neither matching voltage labels nor fixing UART directions/GPIO allocation
  establishes a safe common power domain. Do not silently rebind power rails.

- **RP2040: native USB plus a BOOTSEL path.** The RP2040 family default programs over native USB — bind `usb_dm`/`usb_dp` to `USB_D_N`/`USB_D_P` and pair it with the USB data connector. First flash and recovery also need BOOTSEL held at reset, which the recipe's BOOTSEL button already provides by switching QSPI_CS to GND; do NOT add an SWD programming header. A USB connector alone is still not sufficient without that BOOTSEL path (§9.29 rejects it at commit).
- **ESP32 family (native-USB or bridged): the BOOT strap needs a pull path during reset.** Ship BOOT+EN/RESET buttons (the native-USB recipes already own them), a USB-UART bridge with DTR/RTS auto-reset, or dedicated strap test pads — a bare USB connector is NOT sufficient for download mode (§9.29 rejects it at commit).
- **Programming USB power for a battery design.** The USB port is data/programming only. VBUS is a separate host-supplied 5 V net wired to the receptacle and its ESD protection, sharing GND with the board; keep the battery/buck supplies separate. Never tie VBUS to VBAT or +3V3, and add no power mux or power-input header — note in `assumptions` that battery/external board power is needed while programming.

Record the choice in `assumptions` ending `(defaulted)` (e.g. `"MCU: ESP32-S3-MINI-1, flashed over native USB, no bridge (defaulted)"` or `"Programming: onboard CH340C USB-UART per core defaults, auto-reset to EN/IO0 (defaulted)"`) and reflect it in `topologies` (plus a sheet if it is its own block). BOM then adds any bridge/auto-reset parts; wiring connects the path.

**Verified circuit recipes.** `extras.circuit_recipes` lists exact versioned
recipes. Normally, naming a listed family and exact part in `requirements` lets
KiCraft resolve the recipe deterministically at commit; do not duplicate those
requirements in `recipe_selections`. Explicit selections remain supported when
needed: use the exact `recipe`, a stable `instance`, the required sheet-role
mapping, and bounded parameters. The current production ESP32 example is the
exact `ESP32-S3-MINI-1-N8`; do not assume another MCU family or package is registered. An
unsupported member of a protected family (for example an unregistered
ESP32-S3-WROOM variant) stops with a blocking diagnostic instead of falling
through to BOM/wiring. Prefer a matching recipe over inventing an MCU support
circuit. USB and GPIO are bidirectional at sheet boundaries. Programming labels
alone never prove a path.

Every MCU needs a typed `requirements` entry with `role: "mcu_core"` and its
implementing sheet, even when other connector or power requirements are present.
Keep the requirement model-owned when no registered recipe fits.

Every other physical function also needs an owning typed requirement: connectors,
switches, battery holders, LEDs, sensors, and support blocks are not exceptions.
A sheet containing several distinct functions has several requirements. Use the
listed primitive family's explicit parameters and ports when that circuit fits;
otherwise keep the requirement model-owned.

The catalogue lists alternative implementations, not the board's inventory.
Implement only the committed functional specification and explicit intent.
Never add another MCU, motor driver, or other unrelated circuit merely because
its recipe appears in the reference data.

Each recipe's `owned_parts` lists the roles, quantities, and values already
implemented inside that recipe. Do not create a second crystal, pull network,
or other support circuit for an owned role, or invent external ports for its
internal nets. For example, a recipe owning an 8MHz crystal keeps that crystal
inside its MCU sheet; it does not expose OSC_IN/OSC_OUT unless those ports are
explicitly listed. Unowned buttons, headers, and other requested functions
still need their own requirements, even on a recipe's sheet.

Recipe `external_ports` describe the fixed interface, and `parameter_defaults`
describe its default configuration. They are definitions, not requirement
bindings: supply actual net names in `requirements[].ports`. MCU application
bindings are additional logical ports, not limited to the fixed `external_ports`.
`allocatable_capabilities` maps each capability to its eligible-pin count under
normal allocation rules, even for MCUs without numeric `allocatable_gpios`.
Counts overlap: one pin may support several capabilities, so do not add them
as independent capacity. The allocator remains authoritative for joint assignment.
The recipe's `interfaces` metadata supplies the exact application interface names
and accepted logical `keys` for every member; declare the interface in
`requirements[].interfaces` and bind one listed key per member. Bare I2C/SPI/UART
keys and their listed protocol-prefixed aliases are equivalent only for
application pins; duplicate spellings must bind the identical net, never different
nets. Arbitrary suffixes or numbered protocol prefixes are not aliases. Fixed MCU
programming ports remain device-relative `uart_tx`/`uart_rx`; a separate application
UART uses `tx`/`rx` and must not reuse the programming conductors.

For each selected recipe, preserve every required external port listed in
`extras.circuit_recipes`, including supply, ground, and conditional native-USB
ports. Bind each used signal to its actual circuit peer; when crossing sheets,
declare an explicit inter-sheet net carrying the owning sheet endpoint. Signal
names never belong in `power_nets`, including differential bus nets. A connector's
prose description does not define its bus ports: its typed `requirements[].ports`
must declare the intended net values and pin order. KiCraft may complete a missing
net only from those typed peer ports, never by guessing the connector bus or pinout.

For optional connector ports, bind only signals the circuit actually uses;
the compiler marks omitted unused terminals NC. A PD-only USB-C receptacle needs
VBUS/GND/CC1/CC2 bindings, not invented SBU1/SBU2 or USB-data singleton nets.
A requested full breakout must instead bind those SBU and USB-data signals to
the real header peer's typed ports. Optional does not permit dropping a requested
function, and naming a net without its physical peer does not implement a breakout.

Bind each recipe's supply port explicitly to the actual architecture rail:
for example, an MCU powered directly from a coin cell uses `"vdd": "VBAT"`,
not an undeclared `VDD` or an invented 3.3 V regulator. A voltage entry alone
does not assign that rail to a device. Preserve differential polarity: `D+`
and `D-` (also `USB_D_P` and `USB_D_N`) are distinct signal nets, never aliases
of each other or power rails.

Every MCU application endpoint, whether sheet-local or inter-sheet, needs its
MCU-side logical port/capability binding. LED and touch net names alone do not
allocate MCU pins. For example, an ATtiny1614 coin-cell application can bind
`{"vdd": "VBAT", "gnd": "GND", "output_led0": "LED0", "touch_pad0": "TOUCH0"}`.
The fixed supply bindings remain required; `output_led0` and `touch_pad0` request
additional pins using the advertised `output` and reviewed `touch` capabilities.
Digital outputs use `output_<id>` ports; inputs use `input_<id>`; touch uses
`touch_<id>`. Use only the selected recipe's verified capabilities: generic GPIO
is not evidence of a touch/analog peripheral. Unsupported touch requests fail
rather than substituting GPIO. If the required capability is not represented,
make the missing device/interface contract explicit rather than emitting a
supply-only MCU requirement and leaving its signals unowned.

`mcu_present: false` cannot override a uniquely named MCU on its implementing
sheet. Bind that MCU's real supply explicitly (coin cell: `vdd: VBAT`), then
provide its complete application contract.

For an MCU-side CAN controller, use `interfaces: ["can_controller"]` and ports
`can_tx` / `can_rx`. KiCraft can derive these only from one verified CAN
transceiver peer's explicit typed `tx` / `rx` net bindings and their shared
inter-sheet endpoints. Empty ports on both chips are not a CAN contract.
The STM32F103C8T6 recipe supports CAN_RX on PB8 (physical pin 45) and CAN_TX on
PB9 (pin 46), with `can_remap: "pb8-pb9"`; firmware must enable that remap.
Do not substitute GPIO, UART, CANH/CANL, or the USB pins for these capabilities.
An SN65HVD230 has no `CAN_TERM` pin: a switchable terminator is a separate
physical resistor/switch circuit across CANH/CANL, not an invented transceiver
endpoint. Repair its explicit topology/ports without silently deleting nets.

A compact recovery is still a complete architecture, not an abbreviated design.
Preserve upstream MCU/interface decisions, topology and rail choices, typed
requirements, required nets, and sheet hierarchy. Repair only the named invalid
contract; do not omit unrelated fields or replace required signals with power.
After `unavailable_recipe_gpio`, use the diagnostic's finite
`allocatable_gpios` set, not the rejected numbers as a starting point for a
larger range. C3 recovery must not grow a GPIO list beyond the package or route
USB ID/CC pins into arbitrary MCU endpoints. The native USB pair, programming
pins, and allocatable GPIO set are different contracts. Keep power regulation
and typed connector pin order intact; use compact ranges only for genuinely
contiguous advertised GPIOs.

**Deterministic circuit lowerers.** `extras.circuit_lowerers` lists exact,
versioned transformations for generic connectors, passive networks, and repeated
simple channels. Prefer one when it matches the requested circuit. Emit a
`requirements` entry whose `family` is one listed family, whose `parameters`
use only the advertised `parameter_keys`, and whose `ports` bind the advertised
`port_keys` to final application net names. Include every `required_port_keys`
entry. For ranged ports such as `bit0..bitN` or `signal0..signalN`, emit every
indexed key implied by the count.
For connectors with one named port per pin/terminal, dictionary insertion order
is physical pin order. Families match exactly: never paraphrase, concatenate,
or infer one from topology prose. KiCraft derives both BOM and wiring from the
same lowerer artifact; an incomplete or out-of-range requirement deliberately
falls through to an ordinary BOM work unit.
When the user has not specified an exact part and the registered lowerer fits,
leave `exact_part` null rather than inventing an MPN. Unsupported parameters or
an incompatible exact part cannot be silently ignored by deterministic lowering;
the model-owned implementation must preserve those constraints.

For `pin-header`, `rows` is the number of parallel contact rows: **1 or 2**,
not the total pin count. Declare contiguous `pin1` through `pinN` port keys;
each key names that physical pin, regardless of JSON object order. Omit `rows`
for the default single-row header.

**External-load power budget.** When the accepted functional spec says the board
supplies power to a display, LED string, motor, heater, or other external load,
the architecture must include both the maximum output-current budget and an
input-source power budget with headroom for the board and conversion losses. If
the brief and prior answers do not supply the output current, return one
`blocking: true` question before choosing the USB-C/PD sink or power-path
topology. Never equate the external-load current with input current: a 5 V / 5 A
external-load budget needs a higher-voltage input contract and a dedicated buck
converter because USB-PD contracts cannot exceed 5 A. The regulated 5 V
converter's output-current rating must also exceed the external-load current so
the onboard circuits retain headroom. Include the negotiated input
voltage/current and the regulated output voltage/current explicitly. When
a phrase such as "configured for 5 V" could mean either the PD contract or the
regulated load output, ask which is required instead of guessing. Never size an
external load path from an unrelated regulator rating.

**Right-size the input contract.** Choose the smallest common source contract
that covers the named regulated-output budget, onboard load, realistic conversion
loss, and transient margin. Do not maximize source voltage/current merely because
USB-PD offers it: for a 30 W regulated load, a common 15 V / 3 A contract is
normally preferred over 20 V / 5 A unless another requirement consumes the extra
55 W. State the margin and the requirement that justifies it. Treat a source
contract at least 2× the downstream converter capacity with at least 30 W of
unused capacity as gross overprovision that must be corrected or explicitly
justified.

**Rail-source completeness.** Every declared non-input power rail must name the
regulator or converter topology that generates it. A distinct regulator IC is
its own physical sheet and connects to its source and output rails through
`inter_sheet_nets`. An ESP32-S3 design must declare a 3.3V rail and a
5V-to-3.3V regulator sized for at least 1A; never silently drop the rail or leave
its source for the BOM stage to invent.

**Power-net identity.** Use one canonical net name for a direct electrical
connection. Two same-voltage rails such as `VBUS` and `+5V` are distinct nets;
keep both only when the architecture names the fuse, switch, filter, net tie, or
converter that connects them. Do not emit both as unexplained outputs of one
sheet or distribute different loads between them without that relationship.

For a configurable USB-PD trigger, distinguish the USB-C negotiation bus from
the controller's voltage-selection interface. CC1/CC2 connect the receptacle
to the PD controller; they are not voltage-selector controls. Declare the
selected controller's actual configuration-pin/net contracts and selector
topology from its supported recipe or part evidence. A `VBUS`/`VOUT` pair
alone does not describe how a requested PD voltage is selected.
The listed `usb-pd-selectable-trigger` / `ch224k-pd-selectable@1` recipe owns
the CH224K and its physical SP3T selector together on one `power` sheet.
Its required ports are `cc1`, `cc2`, `vbus`, `gnd`; its supported parameters are
`voltage_options: "9/12/20V"`, `selection_mode: "resistor-sp3t"`, and
`selector_part: "SS13D07VG4"`. A separate receptacle exposes CC1/CC2 to this
sheet; do not add receptacle-side Rd or invent external VSEL pins/selector
sheets for this integrated recipe. If an earlier architecture already
committed external selector contracts, explicitly repair that topology rather
than folding or deleting its endpoints silently.


Open-question discipline matches earlier stages. First produce a complete
architecture using safe engineering defaults. If a question has a safe default,
it is not blocking: apply that default in the first draft and record the choice
in `assumptions` ending `(defaulted)`. Return a blocking question only when no
safe default exists and the answer materially changes the manufactured board.
Ask one decision per question and include 2–4 concise suggested answers in
`options`; never ask merely to confirm a default or a fact already stated in the
brief or design state.
