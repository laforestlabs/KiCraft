# CircuitChat Wiring — Implementation Spec (v1)

A spec for making CircuitChat synthesis produce **fully wired, human-readable** KiCad 9 schematics from a `ConversationState`. The pipeline today places components and emits hierarchical labels but draws no wires, no junctions, and no power symbols; the resulting schematics are unroutable and unreadable. This spec defines the complete fix end to end — no shortcuts, no stub-with-label hacks.

This is the implementation brief for the agent picking this up. The design decisions below were settled in conversation 2026-05-19 (session `20260519T014114Z_UNNAMED`); treat them as fixed unless explicitly revisited with the requester.

---

## 1. Context

### What works today

`kicraft-circuitchat synthesize` emits a KiCad 9 hierarchical schematic with the right sheet structure: root sheet + N leaf sheets, correct sheet pins, correct refdes uniqueness, all symbols/footprints resolvable in stock KiCad 9 libraries. Validators §9.1–§9.8 all pass.

### What's broken (the one real blocker)

Every leaf sheet from the from-scratch path (`_emit_leaf` in `kicraft/circuitchat/synthesis/emitter.py:255-310`) contains:

- N `(symbol …)` instances laid out in a 5-column grid
- M `(hierarchical_label …)` blocks along the left edge

And nothing else. Specifically: **zero `(wire …)`, zero `(junction …)`, zero `(label …)`, zero `(no_connect …)`, zero `(power …)` symbols** across every leaf. Hierarchical labels are not connected to any pin by a wire — they float at the sheet edge.

Empirical measurement on `ESP32_LED_ACCEL` (session `20260519T014114Z_UNNAMED`): 9 leaves, 156 placed symbols, 37 hier labels, **0 wires, 0 junctions, 0 power symbols, 0 net labels**. `kicad-cli sch export netlist` produces a netlist with 156 `(comp …)` entries and an empty `(nets)` block. Downstream tools (`solve-subcircuits`, `compose-subcircuits`, `autoexperiment`) all require `(nets)` to be populated; they fail with `tier=partial_leaves leafs=0/0 parent_route=fail`.

### Why it's broken at the source

The schema in `kicraft/circuitchat/models.py` carries **sheet-level** connectivity (`Architecture.inter_sheet_nets`) but no **part-pin-level** connectivity. The LLM never produces pin-to-net mappings, and the emitter has no code path that would render such mappings as wires even if they existed. `architecture.topologies` is a free-text description never consumed by synthesis. The leaf-library mechanism (which would deliver pre-wired sheets) is intentionally a *downstream* artifact — populated by promoting working subsheets from real projects — so it can't be the fix; it depends on this fix.

---

## 2. Goals and non-goals

### Goals

- Emit a fully connected schematic for every from-scratch leaf: every electrical connection rendered as actual `(wire …)` segments plus, where appropriate, `(power …)` symbols and `(label …)`. Hierarchical labels physically connected by wire to the pins inside the sheet.
- Schematics conform to industry readability conventions (§3): orthogonal wiring on a 1.27 mm grid, signal flow left-to-right, power-up / GND-down via power symbols, decoupling caps visually adjacent to their IC pin, no 4-way junctions, all pins accounted for (connected or `(no_connect …)`).
- Populated `.kicad_pcb` with one `(footprint …)` block per BOM part and a `(net …)` block per electrical net, so `kicad-cli sch export netlist` and `kicad-cli pcb update` produce a routable design.
- Deterministic synthesis: same `state.json` → byte-identical output (modulo UUIDs, which must be stable across re-runs of the same state via seeded generation).
- Strict validation: refuse synthesis success if any leaf has parts but zero connectivity.

### Non-goals (v1)

- Schematic *quality* matching a senior EE's hand-drawn sheet. Algorithmic schematics are inherently less polished. Target: legible enough that a reviewer can follow signal flow and verify intent in <2 minutes per leaf, no manual cleanup required to run the downstream pipeline.
- Bus notation (`D[0..7]`) — every net rendered individually in v1.
- Multi-unit symbol support (e.g. dual op-amps). Use unit 1 only; if the LLM picks a multi-unit symbol, treat it as a single instance.
- Custom symbols / footprints — stock KiCad 9 libraries only (matches existing constraint).
- Sub-mm component placement optimization. Grid snap is enough.
- ERC clean-room. We aim for ERC errors == 0 on power, GND, and inter-sheet nets; warnings (e.g. unconnected outputs) are acceptable in v1 if explicitly `(no_connect …)`-ed.

---

## 3. Industry best practices being encoded

These rules drive the placement and routing algorithms in §7–§9. Sources: [Schematic Design Best Practices, Schemalyzer](https://www.schemalyzer.com/en/blog/schematic-review/best-practices/schematic-design-best-practices), [Rules to Make Schematics Clear, EMA Design Automation](https://www.ema-eda.com/ema-resources/blog/rules-to-make-schematics-clear-and-easy-to-understand/), [IPC-2612-1 Schematic Symbol Generation](https://pcbsync.com/ipc-2612-1/), [IEEE 315 Graphic Symbols](https://standards.ieee.org/ieee/315/719/), [Principles of Schematic and PCB Layout Design, Arshon Inc.](https://arshon.com/blog/principles-of-schematic-and-pcb-layout-design-a-practical-guide-for-reliable-electronics-and-faster-product-development/).

### Wire-level rules (R-rules)

- **R1 Grid.** All pin endpoints, wire endpoints, junctions, labels, and power symbols lie on a 1.27 mm (50 mil) grid. Symbol origins on a 2.54 mm (100 mil) grid.
- **R2 Orthogonal only.** Wire segments are axis-aligned. No 45° wires.
- **R3 Pin exit.** Every wire leaving a symbol pin advances at least one 2.54 mm grid step in the pin's exit direction before any 90° turn.
- **R4 No 4-way junctions.** A junction may merge at most three wires. Logical 4-way connections must be split into two offset 3-way T-junctions ≥2.54 mm apart.
- **R5 Junction marker.** Every T-intersection of distinct net segments carries a `(junction …)` token. Wires that touch but do *not* connect must offset by ≥0.254 mm to avoid ambiguity.
- **R6 No wire-over-symbol.** Wires never pass through a symbol's bounding box.

### Component placement rules (P-rules)

- **P1 Signal flow.** Inputs on the left edge of the sheet, outputs on the right edge, where "input"/"output" are determined by the sheet's `inter_sheet_nets` endpoint directions.
- **P2 Voltage convention.** Highest-voltage net (from `Architecture.rail_voltages`) at the top of the sheet, GND at the bottom.
- **P3 Decoupling proximity.** Every decoupling capacitor (a capacitor whose two endpoints are a power net and GND, with at least one endpoint shared with an IC power pin) is placed within 5.08 mm (2 grid units of 2.54 mm) of its associated IC pin.
- **P4 Pullup proximity.** Pull-up/pull-down resistors are placed within 5.08 mm of the pin they pull.
- **P5 Anchor centering.** The leaf's "anchor IC" (the highest-pin-count IC, ties broken by ref dictionary order) is placed at the sheet's spatial center (148.5, 105 mm on A4 portrait).
- **P6 Functional grouping.** Parts in the same `BOM.ic_groups[<anchor_ref>]` are placed in a contiguous spatial cluster around their anchor.

### Net rendering rules (N-rules)

- **N1 Power symbols, not wires.** Every endpoint of a net in `Architecture.power_nets` is rendered with a `(symbol "power:<NAME>" …)` block (KiCad convention: `power:+3V3`, `power:GND`, `power:VBUS`, etc.), pointing in the canonical direction (up for positive rails, down for ground). No long wire trunks for power.
- **N2 Hierarchical label wiring.** For every `inter_sheet_nets` endpoint of the current sheet, a wire physically connects the `(hierarchical_label …)` at the sheet edge to the pin(s) inside the sheet that share that net.
- **N3 Net labels at branches.** Every local net with ≥3 connected pins has a `(label "<net_name>" …)` placed at the routing-tree root. Two-pin local nets are unlabeled (the wire alone is sufficient).
- **N4 Net naming.** Net names follow `Architecture.inter_sheet_nets` and `Architecture.power_nets` verbatim. Sheet-local nets use descriptive names from the wiring stage (§6), never auto-generated `Net-1`/`Net-2` style.
- **N5 No-connect markers.** Every symbol pin not in any `BOM.connections` entry receives a `(no_connect …)` marker. This makes "intentionally unused" explicit and lets ERC pass.

### Component metadata rules (M-rules)

- **M1 Reference + value visible.** Every part shows its refdes and value as visible properties. Footprint, Datasheet, Description hidden (matches current behavior).
- **M2 Refdes above, value below.** Refdes property positioned above the symbol, value below. (Currently the emitter does the inverse for some symbols; align this.)
- **M3 Pin numbers visible** where the symbol defines them. (Default KiCad behavior; just don't suppress.)
- **M4 Sheet title block.** Every sheet sets `(title_block (title "<Sheet.name>") (date "<archive_date>") (rev "1") (company "<project_stem>"))`.

---

## 4. Architectural overview

Five concrete artifacts need to be added or changed. Each is independent enough that the implementing agent can build them in the order given.

```
┌─────────────────────────────────────────────────────────────────────────┐
│ models.py: + BOM.connections: list[NetConnection]                       │
│            + NetConnection, PinEndpoint                                 │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ symbol_pinout.py (NEW): parses .kicad_sym, returns per-symbol pin list  │
│   - (name, number, position, exit_direction, electrical_type)           │
│ cli_app.py: + `kicraft-circuitchat lookup-symbol <Lib:Name>`            │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ .claude/skills/circuitchat/stages/wiring.md (NEW):                      │
│   New 5th LLM stage. Reads architecture + bom, emits connections.       │
│ Calls `lookup-symbol` per part. Pydantic-validates pin existence.       │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ placement.py (NEW): roles → grid positions, deterministic               │
│ router.py (NEW): orthogonal autoroute, junction insertion, label/power  │
│                  symbol placement                                       │
│ emitter.py: _emit_leaf() now calls placement + router; emits wires +    │
│             junctions + labels + power symbols + no_connects            │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ kicad_pcb_stub.py: populate from BOM — (footprint …) + (net …) blocks  │
└─────────────────────────────────────────────────────────────────────────┘
              │
              ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ validation.py: + §9.9 connectivity, §9.10 pin-existence,                │
│                + §9.11 net coverage, + §9.12 ERC                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Schema additions (`kicraft/circuitchat/models.py`)

Add two new models and one new field on `BOM`. Place these between the existing `BomPart` definition and the `BOM` class.

```python
class PinEndpoint(BaseModel):
    """One pin's participation in a net.

    `ref` matches a BomPart.ref. `pin` is the pin number as defined in the
    KiCad symbol (matches the `(pin "<number>" …)` token in .kicad_sym).
    For multi-unit symbols, this addresses unit 1 only (v1 constraint).
    """
    model_config = ConfigDict(extra="forbid")

    ref: str
    pin: str

    @field_validator("ref")
    @classmethod
    def _ref_pattern(cls, v: str) -> str:
        if not REF_RE.match(v):
            raise ValueError(f"PinEndpoint.ref {v!r} must match {REF_RE.pattern}")
        return v


class NetConnection(BaseModel):
    """One electrical net inside the schematic.

    `net_name` is either:
    - A name appearing in Architecture.power_nets (rendered as power symbol),
    - A name appearing in Architecture.inter_sheet_nets (rendered with the
      hier label wired through to all endpoints on this sheet),
    - A sheet-local descriptive name (rendered as wire-only or wire+label
      depending on N3).
    """
    model_config = ConfigDict(extra="forbid")

    net_name: str
    endpoints: list[PinEndpoint]
    sheet: str  # Sheet.name this net lives on

    @model_validator(mode="after")
    def _has_endpoints(self):
        if len(self.endpoints) < 1:
            raise ValueError(f"NetConnection {self.net_name!r} has no endpoints")
        return self


# Extend BOM:
class BOM(BaseModel):
    # ... existing fields ...
    connections: list[NetConnection] = Field(default_factory=list)

    @model_validator(mode="after")
    def _connection_refs_known(self):
        ref_set = {p.ref for p in self.parts}
        for c in self.connections:
            for ep in c.endpoints:
                if ep.ref not in ref_set:
                    raise ValueError(
                        f"NetConnection {c.net_name!r} references unknown ref {ep.ref!r}"
                    )
        return self

    @model_validator(mode="after")
    def _connection_sheets_known(self):
        # Cross-validation against Architecture happens at the conversation
        # level; here we only check internal consistency.
        sheets_used = {c.sheet for c in self.connections}
        part_sheets = {p.sheet for p in self.parts}
        unknown = sheets_used - part_sheets
        if unknown:
            raise ValueError(
                f"NetConnection.sheet values not represented in BOM.parts: {sorted(unknown)}"
            )
        return self
```

Add at the top alongside the existing regex constants:

```python
PIN_NUMBER_RE = re.compile(r"^[A-Z0-9~_/]+$")  # KiCad allows alphanumeric + a few specials
```

Apply to `PinEndpoint.pin` via a field_validator.

**No changes to `IntentSlot`, `FunctionalSpec`, `Architecture`** — wiring is BOM-side.

---

## 6. New LLM stage: `wiring`

New file: `.claude/skills/circuitchat/stages/wiring.md`. Slots in between BOM and synthesis. Promote it to a first-class stage in `SKILL.md` so the per-turn workflow knows about it.

### Stage contract

**Reads:** the full `state.json` (intent + functional_spec + architecture + bom).

**Writes:** `BOM.connections`. Does NOT modify any other field.

**Prerequisites:** all four prior slots populated. If not, the skill refuses to run `wiring` and instead surfaces the missing slot.

### Process the LLM follows

1. For every part in `BOM.parts`, look up the symbol pin inventory:
   ```
   kicraft-circuitchat lookup-symbol <symbol>
   ```
   This returns JSON like:
   ```json
   {
     "symbol": "RF_Module:ESP32-S3-WROOM-1",
     "pins": [
       {"number": "1", "name": "GND", "electrical_type": "power_in"},
       {"number": "2", "name": "3V3", "electrical_type": "power_in"},
       {"number": "3", "name": "EN", "electrical_type": "input"},
       ...
     ]
   }
   ```
   The model caches results per-symbol within the turn.

2. For each sheet, produce a `NetConnection` for every electrical net **on that sheet**:
   - Every entry in `Architecture.power_nets` that has at least one endpoint on this sheet's parts (e.g. `GND`, `+3V3`).
   - Every entry in `Architecture.inter_sheet_nets` whose `endpoints` includes this sheet, expanded into the specific pins on this sheet's parts that participate.
   - Every sheet-local net (entirely within one sheet) — e.g. the feedback divider on an LDO, the I²C address pins on an IMU.

3. For every part pin not assigned to any net, the LLM has two choices:
   - Mark as no-connect by adding an entry to a per-sheet "no_connect_pins" list (new optional field — see below) OR
   - Reject the BOM as incomplete and surface as a blocking question.

   For v1, support no-connects via an extra `BOM` field:
   ```python
   no_connect_pins: list[PinEndpoint] = Field(default_factory=list)
   ```

4. Validation step: every part pin in this leaf must be accounted for (in a `NetConnection` OR in `no_connect_pins`). This is enforced by §9.11 (see §11). The stage must keep iterating until this holds.

### Output discipline

Same as the existing stages: the wiring stage owns the `connections` list; re-running it replaces it wholesale. If the user revises BOM, wiring re-runs.

### Stage instruction file template

```
Stage 5: Wiring. Given Intent, Functional Spec, Architecture, and BOM,
produce the explicit pin-to-net mapping that synthesis renders as actual
wires + power symbols + labels.

Read the current `.kicraft/state.json`. For every part, run:

    kicraft-circuitchat lookup-symbol <Library:Symbol>

…and treat the returned pin list as the canonical pin inventory. Never
invent pin numbers.

Then for each Sheet in the architecture, produce a NetConnection for:

- every power net (Architecture.power_nets) whose pins land on parts in
  this sheet,
- every inter-sheet net (Architecture.inter_sheet_nets) whose endpoints
  include this sheet, expanded into the specific pins on this sheet,
- every sheet-local net you derive from the topology
  (`Architecture.topologies[sheet]`) — feedback dividers, sense lines,
  bypass paths, etc.

Account for every pin: either it joins a NetConnection or it goes into
`no_connect_pins`. No pin may be silently omitted.

[… rest of the prompt …]
```

The implementing agent should write the full prompt text in the file. Keep it under 60 lines, like the existing stage files.

---

## 7. New CLI helper: `lookup-symbol`

Add to `kicraft/circuitchat/cli_app.py`:

```python
def _cmd_lookup_symbol(args: argparse.Namespace) -> int:
    from .synthesis.symbol_pinout import lookup_pins, SymbolNotFoundError
    try:
        info = lookup_pins(args.symbol)
    except SymbolNotFoundError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(json.dumps(info, indent=2))
    return 0
```

New module: `kicraft/circuitchat/synthesis/symbol_pinout.py`.

```python
"""Pin-level inventory for KiCad 9 stock symbols.

Reads the `(symbol …)` block from /usr/share/kicad/symbols/<Library>.kicad_sym
(reusing the parser in symbol_library.py) and extracts each (pin …) child.

Returns a dict:
  {
    "symbol": "Library:Name",
    "unit_count": int,                  # multi-unit symbols
    "pins": [
      {
        "number": str,                  # "1", "A", "EN", etc.
        "name": str,                    # pin name from symbol
        "electrical_type": str,         # input | output | bidirectional |
                                        # tri_state | passive | unspecified |
                                        # power_in | power_out | open_collector |
                                        # open_emitter | no_connect | free
        "position": {"x": float, "y": float},  # symbol-local coords (mm)
        "orientation": int,             # 0 | 90 | 180 | 270 — direction the
                                        # pin extends OUT of the symbol body
        "length": float,                # pin stub length (mm)
        "unit": int                     # 1-indexed; v1 callers should use 1
      },
      ...
    ]
  }
```

The function MUST handle `(extends "<base>" …)` symbols by resolving the inherited pin list from the base symbol (KiCad symbol inheritance). Existing `symbol_library.py` resolves `(extends …)` for `(lib_symbols)` block emission — extract or reuse that logic here.

Performance: cache parsed `.kicad_sym` files for the process lifetime (a simple `@functools.lru_cache` is sufficient). Stock libraries are read-only.

Tests at `tests/test_symbol_pinout.py`:
- Resistor `Device:R` returns 2 pins, both passive.
- ESP32 module `RF_Module:ESP32-S3-WROOM-1` returns 45 pins of mixed electrical types.
- An extends-based symbol returns the same pins as its base.
- Missing symbol raises `SymbolNotFoundError`.

---

## 8. Placement algorithm

New module: `kicraft/circuitchat/synthesis/placement.py`.

### Inputs

- `architecture: Architecture` (for rail voltages, inter_sheet_nets, topologies)
- `sheet: Sheet` (which one we're laying out)
- `parts: list[BomPart]` (this sheet's parts)
- `connections: list[NetConnection]` (this sheet's nets)
- `pin_lookup: callable(lib_id) -> PinInfo` (from §7)

### Output

```python
@dataclass(frozen=True)
class PlacedPart:
    ref: str
    x_mm: float
    y_mm: float
    rotation_deg: int           # 0 | 90 | 180 | 270
    mirror: str | None          # None | "x" | "y"
    role: str                   # anchor | decoupling | pullup | inline | peripheral
```

`place_sheet(...) -> list[PlacedPart]`.

### Algorithm

A4 portrait sheet: 297 mm × 210 mm. Usable region (margins for title block): x ∈ [25.4, 271.6], y ∈ [25.4, 184.6]. Center: (148.5, 105).

**Step 1 — classify each part by role.**

```python
def classify(part, connections, pin_lookup):
    pins = pin_lookup(part.symbol).pins
    # Anchor heuristic: ≥8 pins AND lib_id starts with one of
    # ['MCU_', 'RF_Module:', 'Regulator_', 'Battery_Management:',
    #  'Sensor_', 'Driver_', 'Interface_']
    if len(pins) >= 8 and any(part.symbol.startswith(p) for p in ANCHOR_PREFIXES):
        return "anchor_candidate"
    # Decoupling: 2-pin capacitor whose two endpoints are a power_net and GND
    if part.symbol.startswith(("Device:C", "Device:CP")):
        endpoint_nets = nets_for_ref(part.ref, connections)
        if len(endpoint_nets) == 2 and is_power_or_ground_name(endpoint_nets[0]) \
                                   and is_power_or_ground_name(endpoint_nets[1]):
            return "decoupling"
    # Pullup/pulldown: 2-pin resistor connecting a signal to a power rail
    if part.symbol.startswith("Device:R"):
        nets = nets_for_ref(part.ref, connections)
        if len(nets) == 2 and any(is_power_or_ground_name(n) for n in nets):
            return "pullup"
    # Inline: 2-pin passive whose neither endpoint is a power rail
    if part.symbol.startswith(("Device:R", "Device:L", "Device:C", "Device:D")):
        return "inline"
    # Everything else
    return "peripheral"
```

Per sheet: choose anchor by taking the single `anchor_candidate` with the most connections to other parts. Ties broken by ref dictionary order. If no anchor_candidate exists, the highest-pin-count peripheral becomes the anchor.

**Step 2 — orient the anchor.**

For each of the 8 orientations (4 rotations × 2 mirror states), score:
- +10 per power_in pin landing on the top edge
- +10 per ground pin landing on the bottom edge
- +5 per "input" pin landing on the left edge
- +5 per "output" pin landing on the right edge
- −2 per power_in pin not on the top edge
- −2 per ground pin not on the bottom edge

"Edge" means: the pin's exit direction after applying the rotation/mirror. Pick the orientation with the highest score; ties broken by lowest rotation_deg.

**Step 3 — place the anchor** at (148.5, 105). Snap to 2.54 mm grid.

**Step 4 — place pin neighbors (decoupling + pullup).**

For each pin_neighbor, find its associated anchor pin (the pin shared via a NetConnection). Compute the pin's absolute position in schematic coords after anchor placement. Then:

- If the pin exits to the right (orientation 0°): place the neighbor at (pin.x + 7.62, pin.y) with rotation 0°.
- If exits left: (pin.x − 7.62, pin.y) with rotation 180°.
- If exits up: (pin.x, pin.y − 7.62) with rotation 270°.
- If exits down: (pin.x, pin.y + 7.62) with rotation 90°.

7.62 mm = 3 × 2.54 mm — enough clearance for the part body and pin stubs.

If multiple neighbors share an anchor pin, stack them along the pin's exit axis at 5.08 mm intervals.

**Step 5 — place inline parts** between their two endpoints. If both endpoints are placed, midpoint between them (grid-snapped). If only one is placed, offset from the placed endpoint by 7.62 mm in the direction that doesn't collide with existing placement.

**Step 6 — place peripherals.**

Each peripheral connects to ≥1 already-placed part (via NetConnection) or to a hierarchical label. Compute the centroid of its connection points. Place the peripheral at the centroid, snapped to grid, with collision avoidance: if the bounding box overlaps any placed part, shift by 5.08 mm in the centroid-to-edge direction until no overlap.

Peripheral orientation: align the "primary input" pin toward the centroid.

**Step 7 — place hierarchical-label endpoints.**

`Architecture.inter_sheet_nets` endpoints already place hier labels at the sheet edge in the current emitter. Refine: for each inter-sheet net, find the pin (on this sheet) it connects to via NetConnection, and place the hierarchical label aligned with that pin's y-coordinate (for left-side nets) or x-coordinate (top/bottom). Choose which edge based on the net's signal flow direction (P1):
- nets with this sheet as `input` → left edge
- nets with this sheet as `output` → right edge
- power nets → no hier label, rendered as power symbols (N1)
- nets where this sheet is `bidirectional` → choose left if the net source is upstream, right otherwise

**Step 8 — collision resolution.**

If after steps 3–7 any two parts' bounding boxes overlap (use a per-symbol-class default size from `lookup-symbol`), nudge the later-placed part along the longer axis by 2.54 mm until clear. Hard error if no resolution found in 20 iterations — let synthesis raise `SynthesisValidationError("placement: cannot resolve collision for <ref>")`.

### Sheet-size escalation

For very dense sheets (e.g. `LED_MATRIX` with 112 parts) where the anchor + neighbors don't fit on A4, the algorithm detects bbox overflow and escalates:
- Try A3 portrait (297 × 420). Update title block paper size.
- Try A2.
- Raise an error if A2 still doesn't fit; this is a signal to the user to split the sheet (typically what the leaf-library spec is for, but in v1 just raise).

For repetitive arrays (detected heuristically: ≥10 parts sharing the same symbol, e.g. 96 WS2812B), use a different layout: grid of N columns × ceil(parts/N) rows on a 12.7 mm pitch. The first iteration ships the heuristic; a `--layout=grid` override hook may come later.

---

## 9. Routing algorithm

New module: `kicraft/circuitchat/synthesis/router.py`.

### Inputs

- `placed_parts: list[PlacedPart]` from §8
- `connections: list[NetConnection]`
- `pin_lookup` (resolves pin position given placed part)
- `sheet_bounds: (min_x, min_y, max_x, max_y)`

### Output

```python
@dataclass(frozen=True)
class WireSegment:
    x1_mm: float
    y1_mm: float
    x2_mm: float
    y2_mm: float

@dataclass(frozen=True)
class Junction:
    x_mm: float
    y_mm: float

@dataclass(frozen=True)
class NetLabel:
    text: str
    x_mm: float
    y_mm: float
    angle_deg: int
    kind: str   # "local" | "global" — v1 emits "local" only

@dataclass(frozen=True)
class PowerSymbol:
    lib_id: str   # "power:+3V3", "power:GND", etc.
    x_mm: float
    y_mm: float
    angle_deg: int

@dataclass(frozen=True)
class NoConnect:
    x_mm: float
    y_mm: float

@dataclass(frozen=True)
class RoutedSheet:
    wires: list[WireSegment]
    junctions: list[Junction]
    labels: list[NetLabel]
    power_symbols: list[PowerSymbol]
    no_connects: list[NoConnect]
```

`route_sheet(...) -> RoutedSheet`.

### Algorithm

**Phase A — power-net rendering (rule N1).**

For each net in `Architecture.power_nets` that appears in `BOM.connections` for this sheet:
- Skip generating wires for the net entirely.
- For each `(ref, pin)` endpoint: compute the pin's absolute position. Drop a `PowerSymbol` at the pin position, oriented:
  - `power:GND`-style (any net matching `GND_NET_PATTERNS` in models.py): symbol below the pin, angle 0° (KiCad GND symbol points down).
  - Positive rails (`+3V3`, `+5V`, `VBUS`, `VBAT`, `VSYS`, etc.): symbol above the pin, angle 180° (KiCad rail symbols point up).
- If the pin's exit direction is not vertical (rail pin on left/right side of symbol): emit a 2.54 mm stub wire from the pin to a turn point, then a 2.54 mm vertical segment to the power symbol. This satisfies R3 and is the *only* allowed exception to N1's "no wires" rule for power.

**Phase B — sheet-local + inter-sheet signal routing.**

For each remaining `NetConnection` (not a power net):

1. Resolve all endpoint positions. If the net is an inter-sheet net, also include the hierarchical-label position as an endpoint.

2. Compute a routing tree:
   - 2 endpoints → single Manhattan path (Phase C, below).
   - N>2 endpoints → minimum Manhattan spanning tree. For up to ~10 endpoints (always the case in v1), Prim's algorithm on the complete graph of Manhattan distances is fast enough. Where intermediate nodes are needed for routing (T-junctions), add a Steiner point at the median x and median y of the endpoint cluster.

3. For each tree edge, compute a Manhattan path (Phase C).

4. Internal nodes of the tree become `Junction` markers (rule R4: ensure no node has degree ≥4 by splitting into two offset 3-junctions ≥2.54 mm apart).

5. If the net has ≥3 endpoints, emit a `NetLabel` at the Steiner-point centroid (rule N3).

**Phase C — Manhattan path with obstacle avoidance.**

Pathfind from (x1, y1) to (x2, y2):

1. Generate the two L-shaped candidates: (x1→x2 then y1→y2) and (y1→y2 then x1→x2).
2. Check each against the obstacle set: bounding boxes of all `PlacedPart`s, dilated by 1.27 mm. A path "collides" if any of its segments crosses an obstacle interior.
3. If both candidates collide, fall back to A* on the grid (1.27 mm cells) with obstacles. Cost = segment length + 5 × turn count (penalizes squiggly routes).
4. Honor R3: the first 2.54 mm of every path lies along the pin's exit direction.
5. Honor R2 (orthogonal) and R1 (grid) by construction.

**Phase D — no-connect markers (rule N5).**

For every pin not appearing in any `NetConnection` for this sheet and not in `no_connect_pins`: this is an error (caught by §9.11). For pins explicitly in `no_connect_pins`: emit a `NoConnect` at the pin's absolute position.

### Determinism

Sort endpoint lists by (ref, pin_number) before building the routing tree so output is deterministic. Tie-break Manhattan paths by preferring horizontal-first when equal-cost.

---

## 10. Emitter changes (`kicraft/circuitchat/synthesis/emitter.py`)

`_emit_leaf` becomes:

```python
def _emit_leaf(project_dir, project_stem, sheet_inst):
    placed = place_sheet(...)            # § 8
    routed = route_sheet(...)            # § 9

    lib_block = build_lib_symbols_block(...)
    symbol_blocks = [
        _emit_symbol_instance(part, p.x_mm, p.y_mm, p.rotation_deg, p.mirror, ...)
        for part, p in zip(sheet_inst.parts, placed)
    ]

    wire_blocks = [_emit_wire(w) for w in routed.wires]
    junction_blocks = [_emit_junction(j) for j in routed.junctions]
    label_blocks = [_emit_net_label(l) for l in routed.labels]
    power_blocks = [_emit_power_symbol(p) for p in routed.power_symbols]
    noconn_blocks = [_emit_no_connect(n) for n in routed.no_connects]
    hier_label_blocks = [_emit_hierarchical_label(h) for h in routed.hier_labels]

    body = "\n".join(
        symbol_blocks + power_blocks + wire_blocks + junction_blocks
        + label_blocks + noconn_blocks + hier_label_blocks
    )
    ...
```

New helper functions to add:

- `_emit_wire(seg: WireSegment) -> str` — produces `(wire (pts (xy x1 y1) (xy x2 y2)) (stroke (width 0) (type default)) (uuid …))`
- `_emit_junction(j: Junction) -> str` — `(junction (at x y) (diameter 0) (color 0 0 0 0) (uuid …))`
- `_emit_net_label(l: NetLabel) -> str` — `(label "<text>" (at x y angle) (effects (font (size 1.27 1.27)) (justify left)) (uuid …))`
- `_emit_power_symbol(p: PowerSymbol) -> str` — emits a `(symbol (lib_id "power:<NAME>") …)` block; also append the corresponding `power:<NAME>` symbol to the `lib_symbols` set so the sheet header includes it.
- `_emit_no_connect(n: NoConnect) -> str` — `(no_connect (at x y) (uuid …))`
- `_emit_symbol_instance` gains `rotation_deg` and `mirror` parameters and renders them into the `(at x y rotation)` and `(mirror x|y)` tokens.

UUID determinism: introduce `_uuid_seeded(salt: str) -> str` that derives a UUIDv5 from a session-scoped namespace + salt (e.g. `f"{sheet.stem}/{ref}/{pin}"`). All UUIDs become deterministic across re-runs of the same `state.json`, which is required for diff-based auditing.

The `lib_symbols` block now also needs to include `power:+3V3`, `power:GND`, etc. for every power symbol used. `symbol_library.py` already pulls from `/usr/share/kicad/symbols/power.kicad_sym` if requested by name; extend the call site to enumerate the power symbols needed.

---

## 11. PCB population (`kicraft/circuitchat/synthesis/kicad_pcb_stub.py`)

Today `kicad_pcb_stub.write_empty_pcb` calls `pcbnew.NewBoard()` and saves. Replace with a function that:

1. Creates the board via `pcbnew.NewBoard()`.
2. For every `BomPart`:
   - Resolves the footprint via `pcbnew.PCB_IO.FootprintLoad(library_path, footprint_name)` (or the modern equivalent — KiCad 9 has shifted; the implementing agent should check `kicraft/cli/solve_subcircuits.py` for the working call).
   - Adds the footprint to the board at a scattered initial position (e.g. spread across a 200×150 mm grid, snapped to 1 mm). The `_autoplacer.json` carries the *real* placement plan; the PCB just needs the parts present.
3. For every `NetConnection`, adds a `pcbnew.NETINFO_ITEM` with the net name.
4. Connects each footprint pad to its net via `pad.SetNetCode(...)` per the connections list.
5. Saves the board.

The `solve-subcircuits` pipeline now finds matching components and a non-empty ratsnest. The `_autoplacer.json` retains its existing role of *driving* placement; the PCB stub just *contains* the components.

If `pcbnew` is unavailable (CI without KiCad bindings), `write_empty_pcb` raises a clear error rather than silently writing an empty board. Update test infrastructure to skip these tests when pcbnew is missing (pattern already exists in `tests/test_cli_help.py:_skip_if_pcbnew_import_error`).

---

## 12. New validators (`kicraft/circuitchat/synthesis/validation.py`)

Add four new `CheckResult` producers to the `run_validations()` pipeline.

- **§9.9 connectivity.** For every `.kicad_sch` file: parse with `kicad-skip` (or regex on the s-expr — the existing code already does s-expr handling). For every leaf with `len(symbols) ≥ 2`: require `(len(wires) + len(power_symbols)) > 0`. Fail otherwise with the offending sheet name.

- **§9.10 pin existence.** Every `(ref, pin)` in `BOM.connections` and `BOM.no_connect_pins` must exist in the corresponding symbol's pin list (looked up via §7). Catches LLM pin-number hallucination.

- **§9.11 net coverage.** For every part in the BOM: every pin defined in the symbol must appear in either a `NetConnection.endpoints` or `no_connect_pins`. No silent omissions.

- **§9.12 ERC.** Run `kicad-cli sch erc <root.kicad_sch> --output erc.rpt` and parse. Require zero `error:` lines (warnings are tolerated). Skip if `kicad-cli` is unavailable.

Update existing §9.7 to include the `power:*` symbols in its uniqueness check (they're all the same ref `#PWR0xxx` family — the test must allow that or distinguish power refs).

---

## 13. Phasing — order of work for the implementing agent

Strict order. Each phase is independently testable. Do not start phase N until phase N−1 ships green.

### Phase 1 — Schema + pin lookup (foundation, low risk)

- Add `PinEndpoint`, `NetConnection` and the `connections` + `no_connect_pins` fields to `models.py`.
- Build `symbol_pinout.py` with `lookup_pins(lib_id)`.
- Add `kicraft-circuitchat lookup-symbol` subcommand.
- Tests for `lookup_pins` covering R, C, ESP32 module, extends-resolved symbols, missing symbol.

Ship criterion: `kicraft-circuitchat lookup-symbol Device:R` returns the expected JSON.

### Phase 2 — Wiring stage (LLM-side, no synthesis changes yet)

- Write `.claude/skills/circuitchat/stages/wiring.md` (full prompt).
- Update `SKILL.md`: add `wiring` to the stage ordering and per-turn workflow.
- Update `cli_app.py`'s `validate` subcommand: if `bom.connections` is non-empty, cross-check pin existence (§9.10) and net coverage (§9.11).

Ship criterion: rerun the existing `ESP32_LED_ACCEL` session through the wiring stage and produce a non-empty `bom.connections` that passes `validate`.

### Phase 3 — Placement engine

- `placement.py` with `place_sheet()`.
- Unit tests on canonical leaves: 2-resistor divider, LDO (4 parts), TP4056 charger (6 parts), ESP32 module (9 parts).
- Acceptance criterion: anchor at sheet center, power pins facing up, decoupling caps within 5.08 mm of their IC pin.

Ship criterion: `place_sheet` returns deterministic, grid-aligned, collision-free placements.

### Phase 4 — Router

- `router.py` with `route_sheet()`.
- Power-symbol rendering (Phase A).
- 2-pin and N-pin Manhattan routing (Phase B + C).
- Junction insertion with no-4-way enforcement (R4).
- Unit tests: a 3-pin net produces 1 junction; a 4-pin net produces 2 offset junctions; obstacle avoidance reroutes around a placed part.

Ship criterion: `route_sheet` returns `RoutedSheet` whose wires satisfy R1–R6.

### Phase 5 — Emitter integration

- Update `_emit_leaf` to call `place_sheet` + `route_sheet` and emit all new s-expr tokens.
- Add new emit helpers (`_emit_wire`, `_emit_junction`, `_emit_net_label`, `_emit_power_symbol`, `_emit_no_connect`).
- Seeded UUIDs (`_uuid_seeded`).
- Update `lib_symbols` to include `power:*` symbols.

Ship criterion: re-running `kicraft-circuitchat synthesize` on `ESP32_LED_ACCEL` produces 9 leaf sheets with non-zero wire counts and KiCad opens them without errors.

### Phase 6 — PCB population

- Rewrite `kicad_pcb_stub.py` per §11.
- Tests gated on pcbnew availability (skip if missing).

Ship criterion: `autoexperiment` no longer reports `leafs=0/0` on a freshly synthesized project.

### Phase 7 — Validators

- Add §9.9 / §9.10 / §9.11 / §9.12 to `validation.py`.
- Wire into `run_validations`.

Ship criterion: synthesizing a deliberately broken state (e.g. a wiring stage that misses a pin) fails synth with a precise error pointing at the missing pin.

### Phase 8 — Smoke: full end-to-end

- Run a fresh `/circuitchat` session for a small project (3-sheet LDO + ESP32 + LED).
- Confirm: synthesis exits 0, `kicad-cli sch erc` reports 0 errors, `autoexperiment` runs to non-zero scores, KiCad opens the schematic and a human can read signal flow in <2 minutes per sheet.

---

## 14. Verification

Per-phase ship criteria are above. End-to-end acceptance test:

1. Fresh session: `/circuitchat` from a blank directory. Describe a USB-C powered ESP32-S3 board with a 3V3 LDO and a 5-LED indicator chain.
2. Walk through all five stages (intent → functional_spec → architecture → bom → wiring), let the skill archive at each boundary.
3. Synthesize.
4. Open `<project>.kicad_pro` in KiCad 9.
5. Inspect each leaf sheet visually:
   - Power rails at top, GND at bottom (P2).
   - Anchor IC centered (P5).
   - Decoupling caps adjacent to their IC pins (P3).
   - All wires orthogonal (R2), grid-aligned (R1), no 4-way junctions (R4).
   - Every IC pin either wired or marked `(no_connect)`.
   - Net labels present at every branch with ≥3 pins.
   - Power symbols used for every power-net endpoint — no long power trunks.
6. Run `kicad-cli sch erc`. Zero errors.
7. Run `kicad-cli sch export netlist`. `(nets)` block non-empty; net count matches `len(BOM.connections)`.
8. Open `.kicad_pcb`. Footprints present, ratsnest visible.
9. Run `autoexperiment`. Scores > 0; `leafs > 0`.

A human reviewer (the requester) reads one sheet cold and writes a one-line verdict in `~/.kicraft/sessions/<id>/feedback.md`. If the verdict is "I can follow this without referring to the BOM," v1 ships.

---

## 15. Out of scope for v1

- Multi-unit symbol support (op-amp dual, hex inverter, etc.). Constrain to unit 1.
- Bus notation (`D[0..7]`).
- Schematic differential-pair routing visualization.
- ERC zero-warnings (zero-errors is the bar; warnings tolerated).
- Custom symbol/footprint libraries.
- Sub-mm placement micro-optimization.
- Beautification beyond P/R/N/M rules — e.g. label-text alignment, font selection, decorative graphics.
- Auto-promotion of finished sheets to the leaf library (separate spec, downstream).
- Schematic title block customization beyond M4.

---

## 16. Open questions for the implementing agent

These are decisions to make during implementation. Surface each one to the requester before committing if the choice is non-obvious.

- **Q1 — Use kicad-skip or hand-rolled s-expr parsing?** `kicad-skip>=0.2` is a declared dependency in pyproject.toml but currently unused. It can read/write `.kicad_sch` and probably `.kicad_sym`. Investigate during Phase 1; pick the cleaner integration. If kicad-skip's API is too lossy for round-trip determinism, stick with the regex/manual-s-expr approach in `symbol_library.py`.

- **Q2 — Steiner-point heuristic for multi-pin nets.** The median-of-medians heuristic in §9 is simple but not optimal. Acceptable for v1. If LED_MATRIX or other dense sheets produce visibly ugly routing, revisit with a proper Steiner-tree algorithm (still a small instance, ~10 pins max per net).

- **Q3 — Sheet-size escalation policy.** §8 escalates A4 → A3 → A2 on overflow. For very-large repetitive arrays (LED_MATRIX again), a grid layout on A4 may be preferable to A2 with a tiny anchor. Implementing agent decides per-sheet via the heuristic in §8.

- **Q4 — Wiring stage placement: own stage vs extension of BOM.** v1 introduces it as a separate 5th stage for clarity (separate prompt file, separate slot semantics). If during Phase 2 it becomes evident that BOM and wiring co-evolve (the LLM keeps regenerating BOM to fix wiring), fold them. Default: keep separate.

- **Q5 — Deterministic UUID namespace.** UUIDs need to be stable across re-runs of identical state. Proposed: UUIDv5 with namespace = `uuid5(URL_NAMESPACE, project_stem)` and name = `<sheet_stem>/<role>/<index>`. Validate this strategy produces unique UUIDs across all KiCad's expected ranges (no collisions).

- **Q6 — Power symbol library inclusion.** When `lib_symbols` is built, every power symbol used must appear once. Power symbol names follow a strict KiCad convention (`+3V3`, `+5V`, `GND`, `VBUS`, `VBAT`, `VSYS`, …). If a custom rail name is used (e.g. `VPP`), the symbol may not exist in stock libraries. Decision for v1: only allow power_nets matching stock power-symbol names; reject in validation otherwise.

- **Q7 — Hierarchical label position.** P1 specifies left edge for input nets, right edge for output nets, derived from `inter_sheet_nets[*].endpoints[*].direction`. For `bidirectional` nets, the agent chooses based on "upstream" inference, which is fuzzy. Concrete rule: bidirectional nets default to right edge for the current sheet unless every other sheet using the net places it on the left. Tighten if visible problems emerge.

- **Q8 — Performance ceiling.** Synthesis runs in <10s today for ESP32_LED_ACCEL. The new placement + routing layers should keep total synthesis <60s for projects up to ~200 parts. If routing for LED_MATRIX (112 parts) blows past this, the grid-array shortcut from §8 must kick in.

---

## 17. Where this leaves the leaf library

Once Phases 1–8 ship, schematics are human-readable and electrically complete. The leaf library can then be populated *from successful synthesis output* — open a fully-wired sheet in KiCad, polish it if desired, and promote it via the leaf-library promote path (separate spec, `docs/leaf_library_spec.md` §3). The architecture stage's `from_library` mechanism then has real leaves to match against, and future projects benefit from human-curated layouts in addition to the algorithmic ones produced by §8–§9.

This is the inversion the requester called out: the library is downstream, not upstream. Synthesis must work without it first.
