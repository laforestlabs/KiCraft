# C1 v2 phase 3 — real pathfinding + rip-up (design handoff)

Status 2026-07-21: phases 1–2 SHIPPED (this wave); phase 3 designed here,
unbuilt. Owner of the remaining dense-board routing-completion class
(self-eval 2026-07-20: 7 boards, run_10 unc=24 stress fixture).

## What already shipped (and what it proved)

- **Track-endpoint anchors** (`unconnected_repair` + `BreakoutSpec.start_xy/net`):
  a track/via DRC endpoint now anchors a tie at its bare coordinate, with the
  net's nearest pad as the netclass stand-in. `no_pad_anchor` is vestigial.
- **Board-diagonal gap cap** (`_gap_cap_mm`): the fixed 60 mm cap skipped every
  castellated fan-out edge (62–86 mm) unattempted; the default now scales to
  the board diagonal. Explicit `signal_repair_max_mm` still wins.
- **Free-anchor strict margins**: a free-coordinate tie takes
  `strict_same_fp=True` foreign-pad margins — the relaxed same-footprint rule
  let a USB_DN tie graze the stand-in footprint's USB_DP pad at 0.05 mm.
- **Geometry-worse accept gates**: BOTH repair wrappers
  (`_attempt_signal_unconnected_repair`, `_attempt_illegal_geometry_repair`)
  now refuse a result whose `malformed_board_geometry` /
  `obviously_illegal_routed_geometry` flags got worse — run_10 proved a tie
  can close an open by stamping illegal copper and the old gate accepted it.

**Post-fix run_10 evidence (the phase-3 brief):** all 21 remaining edges are
now honestly `no_clear_path` with ~16k pruned candidates — the straight/L/
dogleg family is exhausted, not budget-starved. The QFN-56 QSPI escape and
the 14-edge GPIO fan-out need a router, not more candidate shapes.

## Phase 3 design

New module `kicraft/autoplacer/brain/repair_pathfinder.py`, called from
`repair_unconnected_signals` when the candidate family fails an edge
(i.e. replacing the `no_clear_path` skip), behind
`signal_repair_pathfinder_enabled` (default True once fixtures pass).

1. **Grid A\* per edge.** Neighborhood = bbox(src, tgt) + 10 mm margin.
   Two layers (F.Cu/B.Cu), pitch 0.2 mm, 8-dir moves + layer-change move
   (via cost ≈ 25 steps; via legality re-checked at stamp time anyway).
   Obstacles from the existing `_copper_obstacles` index (foreign nets
   only), inflated by `pre_margin_mm` + halfwidth. Pours are NOT obstacles
   (same rules as today: pour nets excluded from repair, pours refill
   around new copper).
2. **String-pulling.** Simplify the grid path to minimal waypoints
   (greedy farthest-visible using `_path_clear` on each shortcut), then
   stamp through `add_breakout_stubs` — the authoritative guards stay the
   only thing that writes copper. Layer changes split the path into
   per-layer specs joined by `via_at_end`.
3. **Bounded rip-up when A\* fails.** Re-run A\* with foreign TRACK cells
   passable at high cost (pads/vias of foreign nets stay hard obstacles).
   The foreign segments the winning path crosses form the rip set; accept
   it only if ≤ `signal_repair_ripup_max_segments` (default 3) and none is
   locked leaf copper (`IsLocked()`). Rip via `board.Remove`, stamp the
   tie, then re-enter the repair loop so the ripped nets' new opens get
   their own tie attempt (they are ordinary edges on the next DRC pass —
   iterate rip-up ≤ 2 rounds to avoid churn).
4. **Containment.** No new accept logic needed: the wrapper already
   re-validates and byte-reverts unless unconnected drops, shorts hold,
   and the geometry flags don't worsen. Rip-up risk is therefore bounded
   to wasted compute, not shipped damage.

## Fixtures / measurement

- run_10 RP2040 (unc=24): QSPI escape + GPIO fan-out; target unc < 10 from
  the repair pass alone (direct-wrapper harness used this wave:
  copy project dir incl. `.kicad_pro`, `validate_routed_board`, call the
  wrapper — NOT a bare-board copy; validate stamps default netclass rules
  into a board missing its project and manufactures fake violations).
- runs 06/13/14/24/27: measure JOINTLY with the N2 geometry-repair pass —
  the rip pass cleans the flags but its re-close currently loses by one
  open on 13/14; a pathfinder tie is exactly what's missing.
- Full-batch: next self-eval target ≥28/34 (plan §measurement discipline).

## Config knobs (proposed)

- `signal_repair_pathfinder_enabled` (True)
- `signal_repair_grid_mm` (0.2), `signal_repair_via_cost_steps` (25)
- `signal_repair_ripup_max_segments` (3), `signal_repair_ripup_rounds` (2)
