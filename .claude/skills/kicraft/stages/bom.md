Stage 4: BOM. You are running inside the KiCraft stage sub-agent. Your job is to draft the `bom` slot of the conversation state and commit it. Follow SKILL.md's "Workflow (follow exactly, in order)" section — this file specifies what the slot must look like.

Given the `architecture` (available in the `state` field of stage-prep's output), pick real parts and assign each to a sheet.

**Programming interface.** Reflect the architecture's programming decision:

- **Native-USB MCU** (an ESP32-S3 / C3 / S2 / C6, the default): no bridge part. The MCU's native USB pins connect to the board's USB connector (wiring handles it); nothing extra to add here beyond the MCU itself.
- **Classic ESP32 with a USB-UART bridge**: include the bridge chosen by the architecture stage (the core-defaults `usb-uart-bridge` row, CH340C / C84681, by default; or the vendored `ch340n` bundle when it is already in the parts block), two small-signal NPN transistors + base resistors for the DTR/RTS auto-reset to EN/IO0, and the bridge's decoupling. Cluster them with the bridge in `ic_groups`.

Either way the wiring stage connects the path, so no programming question reaches it.

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
- `component_zones`: per-ref placement hints, e.g. `{"J1": {"edge": "left"}, "BT1": {"zone": "bottom"}, "H4": {"corner": "top-left"}}`. **Rule, not just an example:** any connector that mates off-board (USB receptacle, barrel jack, an enclosure-wall header) MUST get an `edge` or `corner` zone so it lands at the board edge — pick the meaningful edge from the intent (e.g. the side the cable enters). If you omit it, synthesis defaults edge-mount connector families to an edge as a backstop, but choosing the right edge yourself produces a better board.
- `arrays`: list of regular matrix/array blocks of repeated, identical components (e.g. an addressable-LED matrix, a keypad, a resistor-network bank). Each entry is `{"refs": [...], "rows": R, "cols": C, "pitch_mm": <optional>, "serpentine": true}`. The downstream autoplacer lays these out **programmatically as a serpentine grid** instead of running the force/simulated-annealing solver over them (which does not converge at array scale). **List `refs` in data-chain / logical order** (e.g. `D1, D2, … D200` following the DIN→DOUT chain) so the serpentine fill keeps consecutive parts physically adjacent and routing short. `rows*cols` MUST equal `len(refs)`; take the dimensions from the intent (a "10x20" matrix → `rows: 10, cols: 20`, 200 refs). Omit `pitch_mm` to let the placer derive it from the footprint courtyard. Only include genuine repeated grids here — not ordinary clusters (use `ic_groups` for those).
- `placement_hints` (OPTIONAL): per-passive *schematic*-layout intent. The schematic placer already clusters each 2-pin passive next to the pin it serves and rotates it so its far pin points into open space — inferring the role from the netlist (a cap on rail+gnd is decoupling, a resistor on rail+signal is a pull-up, …). Add a hint only to **override or disambiguate** that inference, e.g. an RC where the "served" pin isn't obvious, a cap between two rails, or a passive that should hug a different IC than the netlist implies. Each entry is `{"ref": "C7", "role": "decoupling", "anchor_ref": "U2", "anchor_pin": "12", "rail_net": "+3V3"}`. `role` is one of `decoupling | bulk | pullup | pulldown | series | feedback | other` (required); `anchor_ref` / `anchor_pin` / `rail_net` are each optional (the placer fills any gap from `connections`). Hints are cheap to omit — leave them out unless a sheet renders badly without them.
- `assumptions`: defaults applied, each ending `(defaulted)`.

DO NOT include `connections` or `no_connect_pins` in the BOM slot. Those fields are owned by the wiring stage; `stage-commit bom` preserves any pre-existing values automatically.

Constraints (enforced by Pydantic):

- Refs unique across the whole BOM.
- Every `ic_groups` key and member must be in `parts`.
- Every entry in `thermal_refs`, `signal_flow_order`, and `component_zones` keys must be in `parts`.
- Every `arrays[*].refs` entry must be in `parts`, no ref may appear in two arrays, and `rows*cols == len(refs)`.
- Every `placement_hints[*].ref` and `anchor_ref` (when set) must be in `parts`.
- `ref` / `symbol` / `footprint` shapes match the regexes above.

## Part selection

**Core defaults: adopt before researching.** `extras.core_defaults_block` from `stage-prep bom` (when present) is the curated registry of default parts, one per common functional block, each with a verified LCSC C-number. Precedence: a matching curated bundle in `extras.parts_block` first, then the core default, then research. When a required function matches a core-defaults row and no stated constraint disqualifies it (voltage/current beyond the row's qualifier, package or assembly limits, an explicit user-named part), adopt the default directly: fetch its bundle with the given C-number in ONE call (the `add_part_from_lcsc` tool, or `kicraft add-part --from-lcsc <C#> --into project`), and do NOT call `lookup_lcsc_id` or `search_symbols` for that part. For the passive-series rows (no C-number), use the named package with the stock `Device:R` / `Device:C` symbols per the footprint defaults below. Record each adoption in `assumptions` (e.g. `"LDO 3.3V: ME6211C33M5G-N per core defaults (defaulted)"`). Check the block's "Package caveats" section before adopting a flagged row.

**Prefer the smaller stocked variant.** When a part comes in several package or module sizes that are all stocked on LCSC and electrically suitable, pick the physically smaller one (e.g. `esp32-s3-mini-1` over `esp32-s3-wroom-1`) to save board area, UNLESS a stated constraint overrides it: a cost driver, the smaller part dropping pins / IO / power rating the design needs, or (for passives) the assembly method favoring a larger hand-solderable size.

## Symbol & footprint sources

For every part, you must resolve a real `symbol` and `footprint` reference. The resolver searches **four parts-library tiers** before falling back to stock KiCad — and `stage-prep` has already done the lookup for you.

The `extras.parts_block` field from `stage-prep bom` is the **curated parts table**: every symbol+footprint bundle available to this project. Each row gives the exact strings to put in `BomPart.symbol` and `BomPart.footprint`. Use them verbatim. Do not invent variant spellings. If `parts_block` is null, the parts library is empty — skip directly to the stock-libraries paragraph.

Beyond the parts block, the following default-install KiCad 9 stock libraries (at `/usr/share/kicad/{symbols,footprints}/`) are **first-tier** sources for obvious passives and packages — use them freely:

- `Device` (R, C, L, LED, D, D_TVS, D_Schottky, D_Zener, Thermistor_NTC, Crystal, Ferrite_Bead)
- `Resistor_SMD`, `Capacitor_SMD`, `Inductor_SMD`, `LED_SMD`, `Diode_SMD`, `Ferrite_Bead_SMD`
- `Package_TO_SOT_SMD`, `Package_DFN_QFN`, `Package_SO`
- `Connector_USB`, `Connector_PinHeader_2.54mm`, `Connector_PinHeader_1.27mm`, `Connector_JST`, `Connector_Generic`
- `Battery`, `Power_Protection`, `Switch`

**Any other stock-KiCad library — `Sensor_*`, `MCU_*`, `RF_*`, `Regulator_*`, `Interface_*`, `Amplifier_*`, vendor-named libraries — is NOT first-tier.** Treat a symbol from one of those exactly like a parts-block/fetch miss: route it through the section below (auto-fetch for beginner/intermediate, or a `material: true` question). Stock symbols for sensors, MCUs, regulators, and interface ICs are frequently out of date or differ from the part the user actually wants, so picking one silently is precisely the substitution this stage exists to prevent.

### When a needed part is in neither the parts block nor stock KiCad

This is the case to handle deliberately — never silently substitute an inferior part. Route based on the captured `state.intent.inferred_expertise`:

- **`beginner` or `intermediate`** — auto-fetch from LCSC (since the default fab is JLCPCB). First resolve the MPN to an LCSC part number — don't guess the `C<NNNNN>`:

  ```
  kicraft lookup-lcsc-id "<MPN>"
  ```

  On a clean hit it prints `{"ok": true, "lcsc": "C<NNNNN>", ...}`. If it returns `"ok": false` with a `candidates` list, choose the right one (or surface a `material: true` question if you can't tell). Then fetch:

  ```
  kicraft add-part --from-lcsc C<NNNNN> --into project
  ```

  This writes a bundle to `<project>/.kicraft/parts/<name>/` and the resolver picks it up immediately. Re-run `stage-prep bom` after the fetch so the new part appears in `extras.parts_block`. Record what you did in `assumptions`, ending the line with `(defaulted)` — e.g. `"Auto-added IP2368 from LCSC C2837135 to the project parts library (defaulted)"`.

  If `add-part` fails (network error, LCSC ID unknown, parser failure on the EasyEDA data), fall through to the advanced path below: surface a `material: true` question listing the missing MPN, the attempted LCSC number, and the failure mode. Do not substitute.

- **`advanced`** — never auto-fetch silently. Surface a `material: true` open question of the form:

  > MPN `IP2368-BZ` is not in the parts library and not in stock KiCad. Options:
  > 1. fetch from LCSC: `kicraft add-part --from-lcsc C2837135 --into project`
  > 2. download .kicad_sym + .kicad_mod from SnapEDA / Ultra Librarian / a silicon-vendor library, then: `kicraft add-part --symbol <path> --footprint <path> --mpn IP2368-BZ --into project`
  > 3. roll your own under `.kicraft/parts/<slug>/` and run `kicraft validate-part .kicraft/parts/<slug> --update-hash`
  >
  > Which do you want to do?

  Leave the missing part out of the BOM (or mark its line with a placeholder MPN and `(awaiting parts-library entry)` in `sourcing_note`) so the user has something concrete to react to.

The principle: **the BOM must either reference a real, resolvable symbol+footprint pair, or surface the gap explicitly.** Off-board substitutions, downgraded connectors, and "we don't have this so we'll use a header instead" decisions belong as `material: true` questions, not as silent `(defaulted)` assumptions.

## Sourcing memory

Prefer cheap LCSC-stocked parts over premium brands. The `sourcing_note` field is a good place to record LCSC part numbers for stock-library parts; for parts-library entries, the `sourcing` dict in the manifest already carries the canonical vendor IDs.
