Architecture section: one part per block.

This call answers **one section** of the architecture intent — the parts that implement the design:
`requirements[]`, one entry per physical part or block implementation. The sheets and rails are
already fixed and are listed in the prompt; the signals that connect these parts, and which
requirement implements which committed obligation, are asked for separately.

Answer exactly one field — `requirements` — and do not emit `sheets`, `signals` or `obligations`.

Requirement fields:

- `id` (stable lowercase), `sheet` (an exact `sheets[].name` from the prompt), `role` (one of the
  CircuitRole values: `mcu_core`, `power_input`, `regulator`, `programming`, `bus_interface`,
  `sensor`, `driver`, `analog_block`, `user_io`, `connector`).
- `family` and `exact_part`: the recipe/lowerer family from `extras.circuit_recipes` /
  `extras.circuit_lowerers`, and the exact ordering code when the user named one. Preserve the
  user's part identity; do not substitute a different family.
- `parameters`: the family's own bounded keys (`rows`, `gender`, `output_voltage`,
  `supply_voltage`, …). Use only keys the reference data advertises.
- `supply`: the rail this part is powered from (a declared `power.rails` name). The compiler binds
  it to the family's supply port (a regulator's input, a logic part's vdd, a driver's vm); a family
  that publishes no supply port — a status LED draws its current from its `drive` signal — simply
  has no pin on that rail, and saying so is not an error. Omit `supply` only for a part that
  genuinely draws nothing from a declared rail.
- `programming`: how a programmable part is flashed — `native_usb`, `usb_uart_bridge`, `swd`,
  `updi`, `bootsel`, `none`. See *Programming* below.
- `interfaces`: the recipe's interface names you use (`i2c_controller`, `spi_controller`, `uart`,
  `can_controller`, `pwm`, `adc`, `parallel_output`). The compiler derives this list from the ports
  you bind in the signals section: a bus member (`sda`/`scl`, `sclk`/`mosi`/`miso`/`cs`, `tx`/`rx`,
  `can_tx`/`can_rx`, `pwm_*`, `adc_*`, `parallel_<i>`) requests its interface, and an interface you
  name with no such port bound is not a request. You do not have to get the list exactly right.
- `functional_blocks`: the exact committed Functional Spec block names this part implements. Every
  block needs an owner; several requirements may implement one block. The schema enumerates the
  accepted names.
- `obligations`: leave this empty. Ownership of the committed typed obligations is asked for in its
  own call, and the compiler writes each row verbatim from the committed intent/functional spec.
- `ties`: rare. The compiler already ties the family's ground, its unused groundable ports, and a
  standard template's stacking pin map. State a tie only for a direct connection no signal names
  and no rule derives: a connector shell to `GND`, an enable pin to the rail it runs on. The net
  must be `GND`, a declared rail, or a net one of your signals names.
- `supply_bindings` / `reference_bindings`: rare refinements for a part with several supply or
  reference domains (an isolated side named `GND_ISO`); a domain must be a declared zero-volt rail.
  The compiler binds the single supply/reference case itself — do not restate it.
- `declared_ports`: **only** for a part with no curated recipe (see below); on a curated or lowerer
  family there is nothing to declare.

Rules:

- **Every Functional Spec block needs an explicit owner** on a sheet; a sheet name or an assumption
  is not implementation evidence. Every requirement must be on a sheet that exists.
- **A curated recipe owns its connections.** A `family` listed in `extras.circuit_recipes` is
  implemented by that reviewed recipe: its `owned_parts` (crystal, pull network, decoupling) are
  already inside it — do not add a second crystal or pull network for an owned role, and do not
  invent external ports for its internal nets. A `family` listed in `extras.circuit_lowerers` is a
  generic connector, passive network or repeated simple channel; emit its `parameter_keys` and its
  advertised port keys, including every `required_port_keys`. For `pin-header`, `rows` is 1 or 2
  parallel contact rows, not the total pin count.
- **A family with a single signal port is one instance per requirement** (`switch-input`'s
  `signal`; `connector-bank`'s `signal0…signalN`): three microstep switches are three requirements,
  or the family's numbered port pattern, never one `signal` port bound three times.
- **Parts the code has never seen.** A part with no recipe is acceptable when you state its
  interface: give the requirement `declared_ports`, one entry per pin function — `{"key": "...",
  "pin": "<the ordering code's own contact>", "direction":
  "input|output|bidirectional|passive|power", "function": "<what it does>"}`. `pin` is what makes
  the claim checkable: the BOM resolves the real symbol and refuses a function claimed on a contact
  that part does not have, and a claim that states no pin at all. Without `declared_ports`, a
  requirement whose ports a signal needs is refused once, naming the part.
- **Use `extras.core_defaults_block`**: the curated default part per common block. Adopt it and
  name the default family in `assumptions` so the BOM adopts that exact part.
- **Programming.** `native_usb` (ESP32-S3 MINI/WROOM, ESP32-C3 MINI, RP2040) means the family's
  `usb_dm`/`usb_dp` are the mandatory data connector — the signals section sends both to one
  `edge:` name. A classic ESP32-WROOM-32 needs the vendored `ch340c` bridge as its own requirement
  with `programming: usb_uart_bridge`, supplied from the same 3.3 V rail as the MCU. Never drive a
  3.3 V MCU's pins from a 5 V bridge, and never tie VBUS to VBAT or a regulated rail.
- Never invent a library prefix, a family, a parameter key or a package name; the BOM stage refuses
  anything that does not resolve to a real symbol, footprint and orderable part.
