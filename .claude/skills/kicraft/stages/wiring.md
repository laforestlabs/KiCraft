Stage 5: Wiring. Produce the final assignment for every component pin.

Use `intent`, `functional_spec`, `architecture`, and the committed canonical BOM.
`extras.symbol_pinouts` is the authoritative pin inventory. Use exact pin
numbers; never invent pins, use pin names as numbers, or read symbol files.

Slot shape:

```json
{"pins": [
  {"ref": "U1", "pin": "1", "net": "+3V3"},
  {"ref": "U1", "pin": "2", "net": "GND"},
  {"ref": "U1", "pin": "3", "no_connect": true}
]}
```

Each `(ref, pin)` appears exactly once. Connected records have `net`; deliberately
unused records have `no_connect: true`. Never emit sheets, connection rows,
endpoint lists, or correction operations. KiCraft derives canonical
`NetConnection` rows by grouping connected pins by their BOM sheet and net.

Wire every repeated component instance and every required supply, programming,
feedback, sense, bypass, pull, and sheet-local signal. Use architecture power and
inter-sheet net names verbatim where applicable.

Every two-terminal series component must separate two distinct nets. A resistor,
capacitor, inductor, ferrite, diode, or fuse with both pins assigned to the same
net is electrically shorted out. For USB termination resistors, use upstream and
downstream names such as `USB_DP_MCU`/`USB_DP` and `USB_DN_MCU`/`USB_DN`.

**Programming-pin check (when `architecture.mcu_present` is true).** The wiring slot MUST give every MCU a first-time programming path. Provide exactly one of:

1. a net assigning the MCU programming pin(s) to a dedicated programming header or connector;
2. a net assigning the programming pin(s) to a labeled test point or pad;
3. a `material: true` open question when the programming pin conflicts with active required use.

When the architecture stage has already provided a programming interface (e.g. an onboard CH340C USB-UART bridge, or a programming header) in the BOM, you MUST connect it per (1) and MUST NOT ask: a USB-UART bridge wired to UART0 with DTR/RTS auto-reset to EN/IO0 satisfies (1). Reserve (3) for the genuine case where no interface exists and a programming pin is shared with an active GPIO.

Silently omitting the programming path is forbidden, even when the package shares its programming pin with an active GPIO. Single-wire SWIO parts (e.g. the CH32V003) **always** expose programming on a shared GPIO — that is the norm, not a reason to drop the net.

Net coverage is enforced at commit:

- Every `(ref, pin)` defined by the symbol appears exactly once in `pins`.
- Use `no_connect: true` only for deliberately floating pins.
- Omitted, duplicate, multiply-connected, and connected-plus-no-connect pins fail.

**BOM shortfall — repair it, never ask the user.** If required support parts are missing, do not mark the affected pin `no_connect: true` and do not invent a ref. Emit exactly one blocking question tagged for automatic BOM repair with a precise parts instruction:

```json
{"questions": [{"text": "The nRF52840 (U1) needs a decoupling cap on each of DEC1-DEC6 and DECUSB; the BOM has only 4x 100nF (C7-C10). Add three more 100nF 0402/0603 caps for the remaining DEC pins and a 4.7uF cap for DECUSB, on the PROCESSOR sheet, clustered with U1.", "blocking": true, "reconcile_target": "bom"}]}
```

The pipeline re-drives the BOM stage with that instruction, then re-runs wiring — the deficit is fixed and the user is never asked. Only use an **untagged** question (no `reconcile_target`) for a genuine design-intent choice the user alone can make (e.g. a shared programming pin, per §21 option 3). A tagged repair question is not a fallback for laziness: name the exact parts.

Constraints:

- Every `ref` exists in the committed BOM.
- Every `pin` exists in that component's KiCad symbol.
- `net` is non-empty and should reuse architecture-owned names where applicable.
- Correction returns a complete replacement using this same `pins` schema.

Re-running wiring replaces the canonical connections and no-connect list wholesale.
