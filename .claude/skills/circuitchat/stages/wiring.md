Stage 5: Wiring. Given Intent, Functional Spec, Architecture, and BOM, produce the explicit pin-to-net mapping that synthesis renders as PCB nets (Stage A) and schematic wires + power symbols (Stage B).

Write the `bom.connections` and `bom.no_connect_pins` fields of `.kicraft/state.json`. Do NOT modify any other field.

Shape (added to the existing `BOM`):

- `connections`: list of `NetConnection`. Each MUST have:
  - `net_name` — either an `Architecture.power_nets` entry, an `Architecture.inter_sheet_nets[*].name`, or a descriptive sheet-local name (no `Net-1`/auto-generated style).
  - `endpoints`: list of `PinEndpoint`, each `{"ref": "<BomPart.ref>", "pin": "<pin_number>"}`.
  - `sheet` — must match a `Sheet.name` from the Architecture exactly.

- `no_connect_pins`: list of `PinEndpoint` for pins explicitly left disconnected.

Process:

1. Read `.kicraft/state.json`. Refuse if any of `intent`, `functional_spec`, `architecture`, `bom` is missing.

2. For every part in `bom.parts`, run:

   `kicraft-circuitchat lookup-symbol <part.symbol>`

   The output JSON lists every pin `{number, name, electrical_type, ...}`. Treat this as the canonical pin inventory. NEVER invent pin numbers; NEVER use a pin name where a pin number is required. Cache the result per-symbol within the turn.

3. For each sheet in `architecture.sheets`, produce a `NetConnection` for:
   - every `architecture.power_nets` entry (e.g. `GND`, `+3V3`) with at least one endpoint on this sheet's parts,
   - every `architecture.inter_sheet_nets[*]` whose endpoints include this sheet, expanded into the specific pins on this sheet's parts that participate (use the hierarchical label name as `net_name`),
   - every sheet-local net you derive from `architecture.topologies[sheet]` (feedback dividers, sense lines, bypass paths, I2C pull-ups, boot caps, etc.). Give each a short uppercase name like `FB_SENSE` or `SDA_LOCAL`.

4. Account for every pin. Every (ref, pin) defined in the symbol must appear in either `connections.endpoints` OR `no_connect_pins`. Use `no_connect_pins` for pins the design intentionally leaves floating (e.g. NC pads, unused GPIOs). No pin may be silently omitted.

Constraints (enforced by Pydantic + `kicraft-circuitchat validate`):

- Every endpoint `ref` must exist in `bom.parts`.
- Every endpoint `pin` must exist in the part's KiCad symbol (per `lookup-symbol`).
- Every `connection.sheet` must equal one of `bom.parts[*].sheet`.
- `net_name` is free-form but should match `Architecture.power_nets` and `Architecture.inter_sheet_nets[*].name` verbatim where applicable.

Output discipline (same as other stages): the wiring stage owns `connections` and `no_connect_pins`. Re-running replaces both wholesale. If the user revises the BOM, re-run wiring.

After writing, run `kicraft-circuitchat validate .kicraft/state.json`. On non-zero, READ the error, FIX the slot, re-write, re-validate. Do not proceed until validation passes.
