Stage 4: BOM. Given the Architecture, pick real parts and assign each to a sheet.

Write the `bom` slot of `.kicraft/state.json`. Shape (`BOM`):

- `parts`: list of `BomPart`. Each MUST have:
  - `ref` — standard EDA designator matching `^[A-Z]+[0-9]+[A-Z0-9_-]*$` (e.g. `U1`, `C12`, `R3`, `RT1`, `BT1`, `J2`). NOT `1U`, `u1`, `MyPart`, or `U-1`.
  - `value` — human-friendly value or MPN (`"10k"`, `"1uF"`, `"BQ24072RGT"`).
  - `symbol` — KiCad symbol in `Library:Name` form (e.g. `"Device:R"`, `"Regulator_Linear:AP2112K-3.3"`). Default-install KiCad 9 ships these at `/usr/share/kicad/symbols/`. Stick to stock libraries when possible.
  - `footprint` — KiCad footprint in `Library:Name` form (e.g. `"Resistor_SMD:R_0402_1005Metric"`, `"Package_TO_SOT_SMD:SOT-23-5"`). NEVER empty. Stock libraries include `Capacitor_SMD`, `Resistor_SMD`, `Inductor_SMD`, `LED_SMD`, `Diode_SMD`, `Package_TO_SOT_SMD`, `Package_DFN_QFN`, `Connector_USB`, etc.
  - `sheet` — must match a `Sheet.name` from the Architecture exactly.
  - `mpn` / `datasheet` / `sourcing_note` — optional but useful.
  - `source_leaf` — set only if a leaf installer added this part; leave null otherwise.

Additional top-level fields:

- `ic_groups`: dict mapping an IC's `ref` to the list of supporting passives that should physically cluster with it (decoupling caps, feedback resistors, inductors). **This is the single most impactful input to placement quality. Spend time on it.**
- `group_labels`: dict mapping IC `ref` to a short silkscreen label (e.g. `"U2": "CHARGER"`).
- `thermal_refs`: list of refs that dissipate significant heat (regulators, power MOSFETs).
- `signal_flow_order`: IC refs in the order signals flow through them (input → ... → output).
- `component_zones`: per-ref placement hints, e.g. `{"J1": {"edge": "left"}, "BT1": {"zone": "bottom"}, "H4": {"corner": "top-left"}}`.
- `assumptions`: defaults applied, each ending `(defaulted)`.

Constraints (enforced by Pydantic):

- Refs unique across the whole BOM.
- Every `ic_groups` key and member must be in `parts`.
- Every entry in `thermal_refs`, `signal_flow_order`, and `component_zones` keys must be in `parts`.
- `ref` / `symbol` / `footprint` shapes match the regexes above.

Sourcing memory: prefer cheap LCSC-stocked parts over premium brands. The `sourcing_note` field is a good place to record LCSC part numbers.
