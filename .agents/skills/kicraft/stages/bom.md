Stage 4: BOM. Draft the complete `bom` slot and commit it through the workflow in the parent `SKILL.md`; this file defines the slot contract.

Given the `architecture` (available in the `state` field of stage-prep's output), pick real parts and assign each to a sheet.

**Programming interface.** Reflect the architecture's programming decision:

- **Native-USB MCU** (ESP32-S3/C3/S2/C6): native USB means no USB-UART bridge, but a USB connector alone is not sufficient for first download and recovery. Include BOOT and EN/RESET buttons or equivalent labeled strap test pads, required pulls, and complete decoupling as explicit groups on the MCU sheet.
- **Classic ESP32 with a USB-UART bridge**: include the CH340C bridge, two small-signal NPN transistors, base resistors for DTR/RTS auto-reset to EN/IO0, and bridge decoupling as distinct component groups on the MCU sheet.

Either way the wiring stage connects the path, so no programming question reaches it.

When architecture selected a circuit recipe, its owned parts are expanded and
locked before model groups. Do not duplicate or substitute recipe roles.
Board-fabricated features such as castellations, edge fingers, vias, and holes
are never purchasable groups: do not assign them headers, MPNs, sourcing notes,
or placeholder footprints.

Slot shape:

- `groups`: the only component representation. Each entry has:
  - `id`: stable lowercase identifier unique in this response.
  - `reference_prefix`: uppercase EDA prefix such as `U`, `R`, `C`, or `J`.
  - `quantity`: number of identical components; use 1 for a unique part.
  - `value`, `symbol`, `footprint`, `sheet`: shared component fields.
  - optional `mpn`, `datasheet`, `sourcing_note`, and `side`.

KiCraft assigns references sequentially per prefix in group order. Never emit
individual parts, explicit references, start/end ranges, connections, or
no-connect records.

**Decoupling completeness — provision every cap the datasheet shows, not a token one or two.** An IC's power integrity is your job here, not the wiring stage's (wiring can only connect parts, it cannot add them — an under-provisioned IC stalls the whole design). For every IC, give it the decoupling its datasheet's reference schematic specifies:

- One bypass cap (usually 100nF) **per dedicated supply/decoupling pin**. A chip that exposes several `DEC*` / `VDD*` / `VDDA` / `AVDD` / bypass pins needs one cap each — e.g. an nRF52840 has DEC1–DEC6 + DECUSB (so ~6× 100nF + a 4.7µF on DECUSB, per its datasheet), an STM32 has a 100nF on every VDD/VDDA pair. Do not ship two caps for a chip that has six supply pins.
- Plus the bulk/reservoir and special-purpose capacitors the datasheet calls out. Give each distinct value/package its own group.

When in doubt, err toward the datasheet's typical-application count. Sizing decoupling correctly here is cheaper than a re-drive later.

**Adjustable-regulator feedback dividers are checked numerically.** For an adjustable-output regulator (TPS54xx, LM2596/LM2576, MP15xx/MP23xx, XL4015, ...), size the feedback divider from the chip's datasheet Vref so `Vout = Vref x (1 + Rtop/Rbot)` hits the rail declared in `architecture.rail_voltages` within a few percent — a divider that misses the named rail by >10% is rejected at commit (§9.32).

Additional top-level fields:

- `arrays`: regular placement patterns. Each entry names one complete component
  group through `group_id`; KiCraft derives the member refs. Grid arrays require
  `rows` and `cols` whose product equals the group quantity. Ring arrays require
  at least three members and omit rows/cols. Optional placement fields are
  `pitch_mm`, `serpentine`, `radius_mm`, and `start_angle_deg`.
- `assumptions`: defaults applied, each ending `(defaulted)`.
- `substitutions`: one `wanted`/`got`/`reason` record for every deviation from a
  part named by the intent, specification, or architecture.

Constraints:

- Group ids are unique.
- Quantities expand to at most 500 total parts and 450 parts per sheet.
- Every group sheet exactly matches an architecture sheet.
- Each group may appear in at most one array.
- Symbols and footprints use real `Library:Name` identifiers and must resolve.
- References, array member lists, and canonical BOM placement fields are derived
  by KiCraft rather than supplied by the model.

## Part selection

**Stock is a hard gate.** Never specify a part that is out of stock. A pick must be in stock BOTH for JLCPCB assembly AND at the lcsc.com retail storefront — these are separate inventories, and common passives are routinely dry at one while plentiful at the other. `lookup_lcsc_id` reports both (`stock` = JLCPCB assembly, `retail_stock` = live lcsc.com retail); the commit gate (§9.26) bounces any pick that fails either, so picking an in-stock part the first time saves the retry. For generic passives there are always dozens of in-stock equivalents — never fight for a specific dry C#.

**Core defaults: adopt before researching.** `extras.core_defaults_block` from `stage-prep bom` (when present) is the curated registry of default parts, one per common functional block. Precedence: a matching curated bundle in `extras.parts_block` first, then the core default, then research. When a required function matches a core-defaults row and no stated constraint disqualifies it (voltage/current beyond the row's qualifier, package or assembly limits, an explicit user-named part), adopt the default directly, and do NOT call `lookup_lcsc_id` or `search_symbols` for that part. How to adopt depends on the row kind: rows with a `bundle` are ALREADY in the parts library, so take the exact `<bundle>:<symbol>` / `<bundle>:<footprint>` strings from `extras.parts_block` or `list_parts` with NO fetch at all; passive-series rows (no C-number) use the named package with the stock `Device:R` / `Device:C` symbols per the footprint defaults below; rows with only a C-number are fetched in ONE call (the `add_part_from_lcsc` tool, or `kicraft add-part --from-lcsc <C#> --into project`). Record each adoption in `assumptions` (e.g. `"LDO 3.3V: ME6211C33M5G-N per core defaults (defaulted)"`). Check the block's "Package caveats" section before adopting a flagged row.

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

**Multi-unit (dual/quad) devices — one usable section per expanded component.** If the design needs N independent sections, set the component group's quantity to N; do not expect KiCraft to instantiate the unused units of one package.

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

  If `add-part` fails (network error, LCSC ID unknown, parser failure on the EasyEDA data), fall through to the expert path below: surface a `material: true` question listing the missing MPN, the attempted LCSC number, and the failure mode. Do not substitute.

- **`expert`** — never auto-fetch silently. Surface a `material: true` open question that names the gap and the concrete choices, WITHOUT shell commands (the user answering may have no terminal):

  > MPN `IP2368-BZ` is not in the parts library and not in stock KiCad. I can fetch it from LCSC as `C2837135` (verified in stock), or substitute the closest stocked equivalent — or paste a different LCSC C-number and I'll use that. Which do you want?

  If the user replies with an LCSC C-number, fetch it with the `add_part_from_lcsc` tool on the next pass and use that bundle's own symbol/footprint ids.

  Leave the missing part out of the BOM (or mark its line with a placeholder MPN and `(awaiting parts-library entry)` in `sourcing_note`) so the user has something concrete to react to.

The principle: **the BOM must either reference a real, resolvable symbol+footprint pair, or surface the gap explicitly.** Off-board substitutions, downgraded connectors, and "we don't have this so we'll use a header instead" decisions belong as `material: true` questions, not as silent `(defaulted)` assumptions.

## Sourcing memory

Prefer cheap **in-stock** LCSC parts over premium brands — in stock meaning both JLCPCB assembly and lcsc.com retail (see "Stock is a hard gate" above). The `sourcing_note` field is a good place to record LCSC part numbers for stock-library parts; for parts-library entries, the `sourcing` dict in the manifest already carries the canonical vendor IDs.
