Stage 4: BOM. You are running inside the KiCraft stage sub-agent. Your job is to draft the `bom` slot of the conversation state and commit it. Follow SKILL.md's "Workflow (follow exactly, in order)" section — this file specifies what the slot must look like.

Given the `architecture` (available in the `state` field of stage-prep's output), pick real parts and assign each to a sheet.

**Programming interface.** Reflect the architecture's programming decision:

- **Native-USB MCU** (an ESP32-S3 / C3 / S2 / C6, the default): no bridge part. The MCU's native USB pins connect to the board's USB connector (wiring handles it); nothing extra to add here beyond the MCU itself.
- **Classic ESP32 with a USB-UART bridge**: include the bridge chosen by the architecture stage (the core-defaults `usb-uart-bridge` row — the vendored `ch340c` bundle, CH340C / C84681), two small-signal NPN transistors + base resistors for the DTR/RTS auto-reset to EN/IO0, and the bridge's decoupling. Cluster them with the bridge in `ic_groups`.

Either way the wiring stage connects the path, so no programming question reaches it.

Slot shape (`BOM`):

- `parts`: list of `BomPart`. Each MUST have:
  - `ref` — standard EDA designator matching `^[A-Z]+[0-9]+[A-Z0-9_-]*$` (e.g. `U1`, `C12`, `R3`, `RT1`, `BT1`, `J2`). NOT `1U`, `u1`, `MyPart`, or `U-1`.
  - `value` — human-friendly value or MPN (`"10k"`, `"1uF"`, `"BQ24072RGT"`).
  - `symbol` — KiCad symbol in `Library:Name` form. See "Symbol & footprint sources" below for where these come from.
  - `footprint` — KiCad footprint in `Library:Name` form (e.g. `"Resistor_SMD:R_0402_1005Metric"`, `"Package_TO_SOT_SMD:SOT-23-5"`). NEVER empty.
  - `sheet` — must match a `Sheet.name` from the Architecture exactly.
  - `mpn` / `datasheet` / `sourcing_note` — optional but useful.
  - `side` (OPTIONAL) — `"front"` (default) or `"back"`. Set `"back"` for any part the intent places on the back of the board (e.g. "a header on the **back side**", a back-mounted connector or battery clip). General to any part; the part is flipped onto B.Cu at placement time. A back-side *internal* connector (a power/data pin header that does NOT mate through an enclosure wall) gets `side: "back"` and **no** `component_zones` edge — see the `component_zones` rule below.
  - `source_leaf` — set only if a leaf installer added this part; leave null otherwise.

**Decoupling completeness — provision every cap the datasheet shows, not a token one or two.** An IC's power integrity is your job here, not the wiring stage's (wiring can only connect parts, it cannot add them — an under-provisioned IC stalls the whole design). For every IC, give it the decoupling its datasheet's reference schematic specifies:

- One bypass cap (usually 100nF) **per dedicated supply/decoupling pin**. A chip that exposes several `DEC*` / `VDD*` / `VDDA` / `AVDD` / bypass pins needs one cap each — e.g. an nRF52840 has DEC1–DEC6 + DECUSB (so ~6× 100nF + a 4.7µF on DECUSB, per its datasheet), an STM32 has a 100nF on every VDD/VDDA pair. Do not ship two caps for a chip that has six supply pins.
- Plus the bulk/reservoir cap(s) the datasheet calls out (often a 1–10µF near the main rail), and any special-purpose cap (VDDH/DC-DC, USB, PLL loop, crystal load caps).
- Cluster every one of these with the IC in `ic_groups[<ic_ref>]` so placement keeps them adjacent.

When in doubt, err toward the datasheet's typical-application count. Sizing decoupling correctly here is cheaper than a re-drive later.

Additional top-level fields (still inside the BOM slot):

- `ic_groups`: dict mapping an IC's `ref` to the list of supporting passives that should physically cluster with it (decoupling caps, feedback resistors, inductors). **This is the single most impactful input to placement quality. Spend time on it.**
- `group_labels`: dict mapping IC `ref` to a short silkscreen label (e.g. `"U2": "CHARGER"`).
- `thermal_refs`: list of refs that dissipate significant heat (regulators, power MOSFETs).
- `signal_flow_order`: IC refs in the order signals flow through them (input → ... → output).
- `component_zones`: per-ref placement hints, e.g. `{"J1": {"edge": "left"}, "BT1": {"zone": "bottom"}, "H4": {"corner": "top-left"}}`. **Rule, not just an example:** any connector that mates off-board (USB receptacle, barrel jack, an enclosure-wall header) MUST get an `edge` or `corner` zone so it lands at the board edge — pick the meaningful edge from the intent (e.g. the side the cable enters). If you omit it, synthesis defaults edge-mount connector families to an edge as a backstop, but choosing the right edge yourself produces a better board. **Inverse rule:** an *internal* connector that does NOT mate off-board — a power/data pin header feeding the board itself, including a "back-side" header — must NOT get an edge/corner zone (use `side: "back"` for the layer instead). An edge zone strands such a header at a far board edge and bloats the outline.
- `arrays`: list of regular repeated-component patterns (an addressable-LED matrix, a keypad, a resistor-network bank, an LED ring). The downstream autoplacer lays these out **programmatically** instead of running the force/simulated-annealing solver over them (which does not converge at array scale). **List `refs` in data-chain / logical order** (e.g. `D1, D2, … D200` following the DIN→DOUT chain) so consecutive parts stay physically adjacent and routing short. Two patterns:
  - **grid** (default): `{"refs": [...], "rows": R, "cols": C, "pitch_mm": <optional>, "serpentine": true}` — a serpentine matrix. `rows*cols` MUST equal `len(refs)`; take the dimensions from the intent (a "10x20" matrix → `rows: 10, cols: 20`, 200 refs).
  - **ring**: `{"refs": [...], "pattern": "ring", "radius_mm": <optional>, "start_angle_deg": <optional>}` — members evenly spaced on a circle, in chain order. **Use this whenever the intent says ring / circle / evenly spaced around ("12 WS2812B evenly spaced in a circle").** Do NOT send `rows`/`cols` with a ring. Set `radius_mm` when the intent fixes the board/ring size — put the LEDs near the board edge (a "60 mm round ring board" → `radius_mm: 24`); omit it to get the tightest legal ring.

  Omit `pitch_mm` to let the placer derive spacing from the footprint courtyard. Only include genuine repeated patterns here — not ordinary clusters (use `ic_groups` for those).
- `placement_hints` (OPTIONAL): per-passive *schematic*-layout intent. The schematic placer already clusters each 2-pin passive next to the pin it serves and rotates it so its far pin points into open space — inferring the role from the netlist (a cap on rail+gnd is decoupling, a resistor on rail+signal is a pull-up, …). Add a hint only to **override or disambiguate** that inference, e.g. an RC where the "served" pin isn't obvious, a cap between two rails, or a passive that should hug a different IC than the netlist implies. Each entry is `{"ref": "C7", "role": "decoupling", "anchor_ref": "U2", "anchor_pin": "12", "rail_net": "+3V3"}`. `role` is one of `decoupling | bulk | pullup | pulldown | series | feedback | other` (required); `anchor_ref` / `anchor_pin` / `rail_net` are each optional (the placer fills any gap from `connections`). Hints are cheap to omit — leave them out unless a sheet renders badly without them.
- `assumptions`: defaults applied, each ending `(defaulted)`.

DO NOT include `connections` or `no_connect_pins` in the BOM slot. Those fields are owned by the wiring stage; `stage-commit bom` preserves any pre-existing values automatically.

Constraints (enforced by Pydantic):

- Refs unique across the whole BOM.
- Every `ic_groups` key and member must be in `parts`.
- Every entry in `thermal_refs`, `signal_flow_order`, and `component_zones` keys must be in `parts`.
- Every `arrays[*].refs` entry must be in `parts`, no ref may appear in two arrays. Grid: `rows*cols == len(refs)`. Ring: >= 3 refs, no `rows`/`cols`.
- Every `placement_hints[*].ref` and `anchor_ref` (when set) must be in `parts`.
- `ref` / `symbol` / `footprint` shapes match the regexes above.

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

**Multi-unit (dual/quad) devices — one section per part.** KiCraft instantiates exactly ONE unit of a symbol per BOM part (the emitter places `(unit 1)`; the wiring stage only sees unit 1's pins). So a dual/quad op-amp, comparator, or logic-gate package yields a SINGLE usable section, not two/four. If the design needs N independent sections (e.g. a 4-channel buffer = four op-amps), add N SEPARATE parts (`U1`, `U2`, … each its own `ref`) — do NOT rely on the extra amplifiers inside one dual/quad chip; those units stay unwired and the channels are dead even though ERC passes. Using a quad package for a single section is fine; using one quad and expecting four channels is the trap.

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

- **`expert`** — never auto-fetch silently. Surface a `material: true` open question of the form:

  > MPN `IP2368-BZ` is not in the parts library and not in stock KiCad. Options:
  > 1. fetch from LCSC: `kicraft add-part --from-lcsc C2837135 --into project`
  > 2. download .kicad_sym + .kicad_mod from SnapEDA / Ultra Librarian / a silicon-vendor library, then: `kicraft add-part --symbol <path> --footprint <path> --mpn IP2368-BZ --into project`
  > 3. roll your own under `.kicraft/parts/<slug>/` and run `kicraft validate-part .kicraft/parts/<slug> --update-hash`
  >
  > Which do you want to do?

  Leave the missing part out of the BOM (or mark its line with a placeholder MPN and `(awaiting parts-library entry)` in `sourcing_note`) so the user has something concrete to react to.

The principle: **the BOM must either reference a real, resolvable symbol+footprint pair, or surface the gap explicitly.** Off-board substitutions, downgraded connectors, and "we don't have this so we'll use a header instead" decisions belong as `material: true` questions, not as silent `(defaulted)` assumptions.

## Sourcing memory

Prefer cheap **in-stock** LCSC parts over premium brands — in stock meaning both JLCPCB assembly and lcsc.com retail (see "Stock is a hard gate" above). The `sourcing_note` field is a good place to record LCSC part numbers for stock-library parts; for parts-library entries, the `sourcing` dict in the manifest already carries the canonical vendor IDs.
