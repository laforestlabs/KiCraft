Stage 5: Wiring. You are running inside the KiCraft stage sub-agent. Your job is to draft the wiring fields of the BOM (`bom.connections` and `bom.no_connect_pins`) and commit them. Follow SKILL.md's "Workflow (follow exactly, in order)" section — this file specifies what the slot must look like.

Given the `intent`, `functional_spec`, `architecture`, and `bom` (all available in the `state` field of stage-prep's output), produce the explicit pin-to-net mapping that synthesis renders as PCB nets (Stage A) and schematic wires + power symbols (Stage B).

**The pin inventory is in `extras.symbol_pinouts`.** Stage-prep batches every distinct BomPart.symbol's pin list into one dict, keyed by `Library:Name`. Each value is the same JSON schema you would get from a per-part `lookup-symbol` call (`number`, `name`, `electrical_type`, `position`, etc.). **NEVER invent pin numbers; NEVER use a pin name where a pin number is required; NEVER call `lookup-symbol` yourself; and NEVER read symbol files from `/usr/share/kicad/symbols/` (or anywhere) — the data is already in the prep output.** If `stage-prep wiring` cannot resolve a symbol it exits non-zero with an `offenders` list instead of returning a pinout: that means the BOM has a bad symbol — fix the BOM (re-fetch or correct it) and re-run `stage-prep wiring`, or surface a `material: true` question. Reading the symbol library yourself to work around a missing pinout is a hard-rule violation.

Slot-file shape — write a JSON object with just these two fields (the rest of the BOM is preserved automatically):

- `connections`: list of `NetConnection`. Each MUST have:
  - `net_name` — either an `architecture.power_nets` entry, an `architecture.inter_sheet_nets[*].name`, or a descriptive sheet-local name (no `Net-1`/auto-generated style).
  - `endpoints`: list of `PinEndpoint`, each `{"ref": "<BomPart.ref>", "pin": "<pin_number>"}`.
  - `sheet` — must match a `Sheet.name` from the Architecture exactly.
- `no_connect_pins`: list of `PinEndpoint` for pins explicitly left disconnected.

Drafting strategy (per sheet in `architecture.sheets`):

- Produce a `NetConnection` for every `architecture.power_nets` entry (e.g. `GND`, `+3V3`) with at least one endpoint on this sheet's parts.
- Produce a `NetConnection` for every `architecture.inter_sheet_nets[*]` whose endpoints include this sheet, expanded into the specific pins on this sheet's parts that participate (use the hierarchical label name as `net_name`).
- Produce a `NetConnection` for every sheet-local net implied by `architecture.topologies[sheet]` (feedback dividers, sense lines, bypass paths, I2C pull-ups, boot caps, etc.). Give each a short uppercase name like `FB_SENSE` or `SDA_LOCAL`.

**Programming-pin check (when `architecture.mcu_present` is true).** The wiring slot MUST give every MCU a first-time programming path. Provide exactly one of:

1. a net connecting the MCU's programming pin(s) — SWD/SWIO/SWCLK, UART boot, JTAG, USB-DFU, etc. — to a dedicated programming header/connector part;
2. a net connecting the programming pin(s) to a labeled test point / pad, recorded in `assumptions` with a user-facing note (e.g. `"flash via the Vcc/Gnd/SWIO pads next to U1 (defaulted)"`);
3. a `material: true` open question naming the shared pin — e.g. *"MCU programming pin PD1/SWIO is shared with active GPIO X; how do you want to program this board the first time?"* — leaving the programming net out of `connections`.

Silently omitting the programming path is forbidden, even when the package shares its programming pin with an active GPIO. Single-wire SWIO parts (e.g. the CH32V003) **always** expose programming on a shared GPIO — that is the norm, not a reason to drop the net.

Net coverage (enforced by stage-commit):

- Every (ref, pin) defined in the symbol must appear in EITHER `connections.endpoints` OR `no_connect_pins`.
- Use `no_connect_pins` for pins the design intentionally leaves floating (NC pads, unused GPIOs).
- No pin may be silently omitted. The `9.11 net coverage` check in stage-commit will reject the slot otherwise.

Constraints (enforced by Pydantic + stage-commit):

- Every endpoint `ref` must exist in `bom.parts`.
- Every endpoint `pin` must exist in the part's KiCad symbol (per `symbol_pinouts`).
- Every `connection.sheet` must equal one of `bom.parts[*].sheet`.
- `net_name` is free-form but should match `architecture.power_nets` and `architecture.inter_sheet_nets[*].name` verbatim where applicable.

Output discipline: the wiring stage owns `connections` and `no_connect_pins`. Re-running replaces both wholesale. If the user revises the BOM, re-run wiring.
