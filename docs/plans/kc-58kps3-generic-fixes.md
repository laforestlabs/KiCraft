# KC-58KPS3 → Generic KiCraft Pipeline Fixes

**Date:** 2026-06-27
**Origin:** Investigation of failed board KC-58KPS3 (project 1/528, "bench power supply")

## Failure summary

| # | Failure | Root cause | Type |
|---|---|---|---|
| 1 | J2/J3 stranded −128.89 mm from top edge | Sheet "USB PD INPUT" has J1 (bottom‑zoned USB‑C) + J2/J3 (top‑zoned banana jacks). A single leaf can only satisfy one edge — the composer pinned J1, stranding J2/J3. | Per‑design (LLM architecture error) |
| 2 | `VDD_3V3` unconnected (USB PD INPUT leaf) | CH224K VDD pin sits on net `VDD_3V3`, which is **not** classified as a power net by `is_power_or_ground_name`. The architecture never declared a 3.3V inter‑sheet net to USB PD INPUT, so the leaf solver treats it as an internal signal net and rejects the leaf. | Pipeline gap: power‑net classifier too narrow |
| 3 | `COMP_NET` unconnected (BUCK 12V leaf) | TPS54331 compensation network wired in schematic but PCB router couldn't connect it. Placement/routing failure within a single leaf. | Per‑design (placement/routing) |
| 4 | TPS54331 (U2) solder‑mask bridge ×6 | Footprint SOIC‑8 PowerPAD apertures bridge to adjacent pin pads. Hit 5/54 runs. | **Systematic** — footprint‑library bug |

---

## Changes

### 1. Auto-split sheets with opposite‑edge connectors (synthesis stage)

**What:** New function `isolate_opposite_edge_connectors` in `array_decaps.py`, called during synthesis right after `isolate_array_sheets`. Detects sheets whose `component_zones` have connectors on opposite edges (`top`/`bottom` or `left`/`right`). Moves the minority-edge connectors to a dedicated sheet → their own leaf, so every leaf has compatible edge zones. Connections are re-split per sheet and cross‑sheet signal nets are declared inter‑sheet.

**Why:** A single rigid leaf can only satisfy one edge per axis. The composer even documents this: "If no rotation satisfies every edge constraint — e.g. two parts pinned to opposite edges of one rigid leaf — the candidate set is left untouched and a warning is logged." Rather than rejecting the architecture at BOM‑commit time (which would force the LLM to re‑architecture), the pipeline auto‑fixes it at synthesis time, the same way `isolate_array_sheets` already moves stray parts off array sheets.

**Where:** `kicraft/design/synthesis/array_decaps.py` — new function + `kicraft/design/synthesize.py` — call site.

**Safety net:** `check_sheet_connector_edge_conflicts` (§9.24) in `validation.py` still runs at BOM commit. If the auto‑split handles every case (as it should), this check never fires. If a future architecture somehow bypasses the auto‑split, the check provides a clear error message.

**Edge zone preservation:** Unlike `isolate_array_sheets` (which drops edge/corner zones on displaced parts since they're internal strays), moved connectors KEEP their edge zone — the new leaf contains only same‑edge connectors and can satisfy the constraint.

**Acceptance:**
- KC‑58KPS3's BOM (J1 bottom + J2/J3 top on "USB PD INPUT") is auto‑split: J1 moves to a new "HEADER" sheet, original sheet keeps J2/J3.
- Post‑split validation passes (no opposite‑edge conflicts).
- J1 retains `{"edge": "bottom"}` in its component_zones.
- 243 validation/synthesis tests pass unchanged.

---

### 2. Broaden `is_power_or_ground_name` to catch local supply‑net names

**What:** Extend `is_power_or_ground_name` in `kicraft/design/models.py` so that net names like `VDD_3V3`, `VCC_5V`, `+3V3_A`, `VDDIO` are classified as power nets.

**Why:** Currently `is_power_or_ground_name` only matches the canonical regex patterns (`POWER_NET_PATTERNS`). These match `+3V3`, `/3V3`, `3V3` but NOT `VDD_3V3`. The function gates:
- `reconcile_inter_sheet_nets` — power nets are preserved verbatim (not reconciled)
- `check_no_dangling_signal_nets` — power nets are exempt (a lone power pin is fine)
- `check_inter_sheet_nets_realized` — power nets are exempt (they join through power symbols)

When `VDD_3V3` is misclassified as a signal net, none of these protections apply, and the leaf solver's `no_unconnected` gate treats it as a local signal net that must be internally connected.

**Where:** `kicraft/design/models.py:67-73` — `is_power_or_ground_name`.

**Approach:** Add a second matching tier: if the stripped name contains any token from `_PWR_NET_TOKENS` (already defined in `validation.py`, mirror it or move to `models.py`), classify as power. The existing regex patterns stay as the primary match (they're stricter and catch canonical forms).

Concretely, `is_power_or_ground_name("VDD_3V3")` should return `True` because it contains the token `"VDD"` (a known supply prefix) AND the token `"3V3"` (a known voltage).

**Tokens to add** (from `validation.py`'s `_PWR_NET_TOKENS`):
```
VDD, VCC, VBAT, VBUS, VSYS, VIN, VOUT, VREG, VPP,
3V3, 5V, 1V8, 2V5, 12V
```

**Acceptance:**
- `is_power_or_ground_name("VDD_3V3")` → `True`
- `is_power_or_ground_name("VCC_5V")` → `True`
- `is_power_or_ground_name("VDDIO")` → `True` (contains VDD)
- `is_power_or_ground_name("COMP_NET")` → `False` (compensation network, not power)
- `is_power_or_ground_name("DATA0")` → `False` (data line)
- Existing canonical forms still match: `is_power_or_ground_name("+3V3")` → `True`

---

### 3. TPS54331 footprint: fix solder‑mask bridge (SOIC‑8 PowerPAD)

**What:** Adjust the solder‑mask expansion on the TPS54331's PowerPAD and adjacent pin pads so the mask apertures don't bridge. This is a **footprint‑library** fix affecting every design that uses the TPS54331 (5/54 runs hit this).

**Where:** `kicraft/parts_library/tps54331/tps54331.pretty/SOP-8_L5.0-W4.0-P1.27-LS6.0-BL.kicad_mod`

**Root cause:** The SOIC‑8 exposed‑pad footprint's solder‑mask apertures for the thermal pad overlap/bridge with the apertures for pins 5‑8 (GND side). The thermal pad and the pin pads share a GND net, but the solder‑mask bridge is between `<no net>` (the PTH pad annulus) and `GND` (the pin pad) — these are different copper features.

**Acceptance:**
- `kicad-cli pcb drc` on a board with TPS54331 reports 0 `solder_mask_bridge` errors clustered on U2.
- Re‑run on an existing failed board (e.g. project 492, 521, 523, 527, 528) to confirm.

---

## NOT doing (explicit descoping)

- **COMP_NET routing failure:** This is a leaf‑level routing problem specific to the TPS54331 compensation network layout. A generic fix would require per‑family placement contracts for compensation networks — too complex for now. The leaf already rejects itself (no_unconnected gate works correctly), so the build correctly fails. The LLM just couldn't produce a routable compensation layout within the leaf's solve budget.
- **BOM generic‑passive sourcing:** Generic `Device:C`/`Device:R` parts with no LCSC part numbers is a known quality gap but doesn't cause build failures. Addressed separately.
- **Board area waste (81%):** The sprawl is a consequence of the connector stranding forcing a huge board. Fixing (1) should shrink the board naturally.
