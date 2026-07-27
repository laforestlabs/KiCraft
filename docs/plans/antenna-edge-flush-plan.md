# Chip-antenna edge-flush placement — plan

**Status:** open (surfaced 2026-07-23, KC-69TGAP). Owning module:
`autoplacer/brain/placement_solver.py` (leaf edge-flush) +
`cli/compose_subcircuits.py::_repair_parent_outline` /
`_compose_state.edge_zoned_outline_sides` (parent outline flush).

## Symptom

A chip/PCB antenna zoned to a board edge lands a few mm **inboard** of that
edge, and the connector-stranding fab gate — correctly — rejects the board.
An antenna is not like an SWD header (which we just made access-only): it is an
RF radiator that *must* sit at the board edge with its emitting end flush/over
the edge and a ground keepout behind it. So the gate flagging it is right; the
**placement** is what's wrong.

KC-69TGAP (`~/.kicraft/projects/1/660`, nRF52840 BLE beacon) after the
congestion-growth + SWD fixes (`6e18f79`, `c7dd0cc`) de-bloated from
178×161 mm to 68×93 mm and dropped its DRC errors to 0 — but the replay then
failed with:

```
reasons=['unconnected_nets', 'connector_stranded:ANT1@-1.51mm(top)']
```

`ANT1` (`2450AT43B100E`, footprint `ANT-SMD_4P-L7.0-W2.0-R`) is zoned
`{edge: top}` but its mouth bbox sits **1.51 mm inboard** of the top edge
(tolerance `connector_edge_inboard_tol_mm = 1.0`). On the original *bloated*
board ANT1 happened to land within tol (only J1 was flagged); once the board
packed tight, the antenna's inboard offset exceeded tol. So this is a latent
placement-quality gap that the de-bloat *exposed*, not one it caused.

## Root cause (hypothesis — confirm in step 1)

The deterministic **body-flush** edge logic in `_pin_edge_components`
(`placement_solver.py` ~1106–1204: `_connector_edge_x` / `_connector_edge_y` /
`_orient_for_edge`) is written for **connectors** — it flushes a connector body
against its zoned edge and orients its mouth outward. A part that is
edge-zoned but not classified as a connector (an `ANT*` chip antenna) gets only
the softer edge *bias* (`edge_jitter_mm`, `connector_edge_inset_mm`) and the
`edge_compliance` scoring nudge — not the hard flush snap — so it settles a
jitter-width inboard.

There is a second layer at parent compose: `edge_zoned_outline_sides`
(`_compose_state.py`) governs which outline sides `_repair_parent_outline` keeps
**flush** with the zoned part instead of adding breathing-room margin. If the
antenna's zoned edge is not in that set (or the antenna lives in a leaf whose
edge intent isn't propagated to the parent), outline repair adds margin and
buries the antenna inboard. This is the "compose-level mouth-line alignment"
item noted open as **N5b** in `kicraft-codebase-review-2026-07-19` /
`c1-v2-pathfinding-design.md`.

## Plan

1. **Confirm the layer.** Replay KC-69TGAP with `--no-route` (deterministic
   placement only) and read `ANT1`'s gap from `connector_edge_gaps` at (a) the
   leaf board and (b) the composed parent. If the leaf already places it inboard
   → it's the `_pin_edge_components` flush classification. If the leaf is flush
   but the parent isn't → it's `_repair_parent_outline` / `edge_zoned_outline_sides`.
   (Measure both inside ONE replay — never across runs.)

2. **Classify antennas as edge-flush parts (Lever A, leaf).** Extend the
   connector body-flush path in `_pin_edge_components` to also fire for chip
   antennas — matched by ref prefix `ANT` and/or footprint family
   (`*ANT-SMD*`, `*chip-ant*`, `*2450at*`). An antenna's RF end should flush to
   (or slightly overhang) its zoned edge exactly like a connector mouth. Keep
   the keepout: the ground-plane cutout behind a chip antenna must be honored
   (check `gnd_pour.py` / leaf keepout so the flush snap doesn't drop the
   antenna onto poured copper).

3. **Keep the antenna's edge flush at compose (Lever B, parent).** Ensure the
   antenna's zoned edge is added to `edge_zoned_outline_sides` so
   `_repair_parent_outline` holds the outline flush to the antenna rather than
   margining it inboard. This is the concrete N5b follow-up for the antenna case.

4. **Prefer overhang over flush for true edge-radiators (optional).** A chip
   antenna is happiest with its tip *at or just past* the board edge (like a
   USB-C body overhang). Consider treating `ANT*` like `edge_constrained_refs`
   so a small positive overhang is allowed and only *pads* outside the outline
   fail geometry validation — mirrors the USB-C shell handling.

## Verify

- Replay KC-69TGAP (`quality good`, seed 0): expect `connector_stranded:ANT1`
  **gone** (gap ≥ −1.0 mm), no new geometry violations, board still ~68×93 mm.
- Re-check the self-eval `run_13_nrf52-beacon` and any other antenna board
  (`run_14_lora-node`, `run_29_round-led-ring` has an RF part) for regressions.

## Guard

- A fast placement unit test: an `ANT`-ref part with `{edge: <side>}` lands with
  its mouth bbox within `connector_edge_inboard_tol_mm` of that edge (mirrors
  the connector flush tests in `tests/test_connector_edge_gap.py` /
  `test_usb_edge_connector_placement.py`).

## Prior art / links

- Gate that flags it (correct): `autoplacer/brain/connector_edge_gap.py`
  (`connector_edge_gaps` / `_connector_stranded_refs`); the SWD access-only
  exemption that de-scoped the *wrong* part shipped in `c7dd0cc`
  (`kicraft-kc69tgap-congestion-leafinternal-swd-fixes`).
- N5b compose-level mouth-line alignment — `docs/plans/c1-v2-pathfinding-design.md`,
  memory `kicraft-codebase-review-2026-07-19`.
- Connector-stranding family — `kicraft-connector-stranding-edge-flush`,
  `kicraft-screw-terminal-orientation-gate`,
  `docs/plans/usb-c-edge-connector-stranding-three-bugs.md`.
