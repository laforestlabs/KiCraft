Architecture section: cross-part wiring.

This call answers **one section** of the architecture intent — the signals: `signals[]`, one entry
per named application signal. The sheets, rails and parts are already fixed and are listed in the
prompt. This is the only place nets are named.

Answer exactly one field — `signals` — and do not emit `sheets`, `requirements` or `obligations`.

Fields:

- `name` — the net name: canonical, unique, not a declared rail.
- `from` — `"<requirement_id>.<port>"`, the source port.
- `to` — one peer reference or a list of them, each `"<requirement_id>.<port>"` or `"edge:<NAME>"`.
- `rails` — rails an off-board peer needs on its connector (a 5 V LED string behind 3.3 V logic).
  Name each one exactly as declared under `power.rails` — `["+5V"]` is refused when the rail you
  declared is `VBUS`. A signal whose source port already carries a declared rail *is* that rail:
  its peers join the rail rather than a second net on the same pin.
- `start` / `end` — expand `{n}` in `name`, `from` and `to` over an inclusive numeric range; use
  this for a bus or a repeated channel (`[HUB75_R{n}` – `hub75.r{n}`, 0..1]), never a hand-typed
  list with a hole in it. `{n}` must appear exactly once in each of the three strings.

**Port keys.** A signal reference names a port of the part it connects. Valid keys are:

- every `ports` name the selected recipe advertises in `extras.circuit_recipes` (its reviewed
  interface), including supply/ground and conditional ports;
- the advertised keys of the recipe's `interfaces` you declared, for bus members (`sda`/`scl`,
  `sclk`/`mosi`/`miso`/`cs`, `tx`/`rx`, `can_tx`/`can_rx`, `pwm_*`, `adc_*`, `parallel_<i>`);
- MCU application pins, by capability: `output_<id>`, `input_<id>`, `touch_<id>` (and `gpio<n>` when
  you mean a specific numbered pin). The pin allocator owns the assignment;
- the lowerer's keyed ports for a lowerer family (`pin1` … `pinN` for `pin-header`, in physical pin
  order — the pins are ordered by the order you bind them, so bind each connector pin once);
- the part's own `declared_ports` keys, for a part with no curated recipe.

A key that is none of the published ones is refused, naming the ports that do exist. Do not invent a
port to carry a signal the part cannot. A required port of a selected recipe nobody binds is refused
by name: wire it with a signal or a `ties` entry.

**How the compiler binds ports.** A port key may be bound exactly once, and the compiler binds it:

- A signal binds its source port to its own net and each peer port to the same net. Two signals that
  name the same **peer** port are refused, naming the ports that were available: that really is two
  nets on one pin.
- Two signals that leave the **same** source port are one physical net under two names — the first
  name owns it, the second signal's peers join it, and the join is recorded as a derived assumption.
  That is legal, not a habit: prefer one signal per net.
- A port a rail or ground already owns keeps that net; a signal out of it joins its peers to the
  rail instead of naming a second net.
- A port whose signal continues off the board (a WS2812 driver's `data_out` feeding the string) is
  bound like any other, to the same `edge:` peer as the rest of that interface.

**Off-board interfaces.** Send a signal to `edge:<NAME>` and the compiler adds the connector: a USB
data socket when the source is a native-USB MCU's `usb_dm`/`usb_dp` (both must be sent to the same
edge), otherwise a one-row pin header on its own sheet, carrying your signals in order, then ground,
then the rails you listed in `rails`. So `LED_DATA` → `edge:LED_STRING` with `"rails": ["+5V"]`
gives a three-pin output: data, ground, 5 V.

Emit only the signals the design actually has. A board whose parts share nothing but rails
legitimately answers `{"signals": []}` — never invent a net to fill the list. Do not name a
declared rail as a signal, and do not use a signal to move power between two names for the same
net: declare one rail name instead.
