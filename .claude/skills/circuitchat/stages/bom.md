Stage 4: BOM. You are running inside the CircuitChat stage sub-agent. Your job is to draft the `bom` slot of the conversation state and commit it. Follow SKILL.md's "Workflow (follow exactly, in order)" section — this file specifies what the slot must look like.

Given the `architecture` (available in the `state` field of stage-prep's output), pick real parts and assign each to a sheet.

Slot shape (`BOM`):

- `parts`: list of `BomPart`. Each MUST have:
  - `ref` — standard EDA designator matching `^[A-Z]+[0-9]+[A-Z0-9_-]*$` (e.g. `U1`, `C12`, `R3`, `RT1`, `BT1`, `J2`). NOT `1U`, `u1`, `MyPart`, or `U-1`.
  - `value` — human-friendly value or MPN (`"10k"`, `"1uF"`, `"BQ24072RGT"`).
  - `symbol` — KiCad symbol in `Library:Name` form. See "Symbol & footprint sources" below for where these come from.
  - `footprint` — KiCad footprint in `Library:Name` form (e.g. `"Resistor_SMD:R_0402_1005Metric"`, `"Package_TO_SOT_SMD:SOT-23-5"`). NEVER empty.
  - `sheet` — must match a `Sheet.name` from the Architecture exactly.
  - `mpn` / `datasheet` / `sourcing_note` — optional but useful.
  - `source_leaf` — set only if a leaf installer added this part; leave null otherwise.

Additional top-level fields (still inside the BOM slot):

- `ic_groups`: dict mapping an IC's `ref` to the list of supporting passives that should physically cluster with it (decoupling caps, feedback resistors, inductors). **This is the single most impactful input to placement quality. Spend time on it.**
- `group_labels`: dict mapping IC `ref` to a short silkscreen label (e.g. `"U2": "CHARGER"`).
- `thermal_refs`: list of refs that dissipate significant heat (regulators, power MOSFETs).
- `signal_flow_order`: IC refs in the order signals flow through them (input → ... → output).
- `component_zones`: per-ref placement hints, e.g. `{"J1": {"edge": "left"}, "BT1": {"zone": "bottom"}, "H4": {"corner": "top-left"}}`.
- `assumptions`: defaults applied, each ending `(defaulted)`.

DO NOT include `connections` or `no_connect_pins` in the BOM slot. Those fields are owned by the wiring stage; `stage-commit bom` preserves any pre-existing values automatically.

Constraints (enforced by Pydantic):

- Refs unique across the whole BOM.
- Every `ic_groups` key and member must be in `parts`.
- Every entry in `thermal_refs`, `signal_flow_order`, and `component_zones` keys must be in `parts`.
- `ref` / `symbol` / `footprint` shapes match the regexes above.

## Symbol & footprint sources

For every part, you must resolve a real `symbol` and `footprint` reference. The resolver searches **four parts-library tiers** before falling back to stock KiCad — and `stage-prep` has already done the lookup for you.

The `extras.parts_block` field from `stage-prep bom` is the **curated parts table**: every symbol+footprint bundle available to this project. Each row gives the exact strings to put in `BomPart.symbol` and `BomPart.footprint`. Use them verbatim. Do not invent variant spellings. If `parts_block` is null, the parts library is empty — skip directly to the stock-libraries paragraph.

Beyond the parts block, the default-install KiCad 9 stock libraries at `/usr/share/kicad/{symbols,footprints}/` cover the obvious passives and packages: `Device` (R, C, L, LED, D_TVS, Thermistor_NTC, ...), `Resistor_SMD`, `Capacitor_SMD`, `Inductor_SMD`, `LED_SMD`, `Diode_SMD`, `Package_TO_SOT_SMD`, `Package_DFN_QFN`, `Connector_USB`, `Connector_PinHeader_2.54mm`, `Battery`, `Power_Protection`, `Switch`, etc. Use these freely.

### When a needed part is in neither the parts block nor stock KiCad

This is the case to handle deliberately — never silently substitute an inferior part. Route based on the captured `state.intent.inferred_expertise`:

- **`beginner` or `intermediate`** — auto-fetch from LCSC (since the default fab is JLCPCB). If you know the LCSC part number from the architecture or a prior session, or you can identify a reasonable LCSC equivalent from the MPN, call:

  ```
  kicraft-circuitchat add-part --from-lcsc C<NNNNN> --into project
  ```

  This writes a bundle to `<project>/.kicraft/parts/<name>/` and the resolver picks it up immediately. Re-run `stage-prep bom` after the fetch so the new part appears in `extras.parts_block`. Record what you did in `assumptions`, ending the line with `(defaulted)` — e.g. `"Auto-added IP2368 from LCSC C2837135 to the project parts library (defaulted)"`.

  If `add-part` fails (network error, LCSC ID unknown, parser failure on the EasyEDA data), fall through to the advanced path below: surface a `material: true` question listing the missing MPN, the attempted LCSC number, and the failure mode. Do not substitute.

- **`advanced`** — never auto-fetch silently. Surface a `material: true` open question of the form:

  > MPN `IP2368-BZ` is not in the parts library and not in stock KiCad. I can fetch it from LCSC (C2837135) into the project parts library, or you can add it manually under `.kicraft/parts/ip2368/`. Which would you like?

  Leave the BOM out the missing part (or mark its line with a placeholder MPN and `(awaiting parts-library entry)` in `sourcing_note`) so the user has something concrete to react to.

The principle: **the BOM must either reference a real, resolvable symbol+footprint pair, or surface the gap explicitly.** Off-board substitutions, downgraded connectors, and "we don't have this so we'll use a header instead" decisions belong as `material: true` questions, not as silent `(defaulted)` assumptions.

## Sourcing memory

Prefer cheap LCSC-stocked parts over premium brands. The `sourcing_note` field is a good place to record LCSC part numbers for stock-library parts; for parts-library entries, the `sourcing` dict in the manifest already carries the canonical vendor IDs.
