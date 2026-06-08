# Handoff: dense power-leaf shorts / doesn't route (USB-C + LDO)

**Status: RESOLVED** (2026-06-08). The shorting copper is gone at its source; the
fix is general (every dense leaf, not just POWER) and covered by regression tests.
This doc records the *correct* diagnosis and the fix. An earlier version of this
handoff misdiagnosed the cause — see "What the first diagnosis got wrong" below, so
the next engineer doesn't re-derive the dead end.

**Symptom (user-reported):** generated PCBs "look terrible" — the USB-C + LDO power
leaf is placed strung-out across a ~150 mm strip, and `kicraft build` exits **rc=6**.

**Canonical repro:** the ESP32-S3 plant monitor, `POWER` sheet = `J1` USB-C
`TYPE-C-31-M-12`, `U1` LDO `AP2112K-3.3`, `D1` ESD `USBLC6`, `R1`/`R2` CC pulldowns,
`C1`–`C4`. Saved boards under
`logs/self_eval/fixtest_20260608T170355Z/.../subcircuits/4ac73ca7…__65aa31e6f3/`
(`round_*_leaf_pre_freerouting.kicad_pcb` = post-placement, post-stamp;
`round_*_leaf_routed.kicad_pcb` = after FreeRouting). `logs/` is gitignored.

---

## Root cause

A **VBUS perimeter power-tie was stamped straight across `U1`'s own pads**, before
FreeRouting ran, with no collision check.

1. `U1` (AP2112K) has **two VBUS pads** — pin 1 (VIN) and pin 3 (EN tied to VIN for
   always-on). Two pads on one power net is the "spread power connector" signature
   `auto_power_tie_specs` keys on, so it fired a VBUS tie on the LDO.
2. `perimeter_tie_specs` connected those two pads with a path that walks around the
   footprint's bounding box. It used **`fp.GetBoundingBox()`**, which silkscreen,
   courtyard and the reference designator inflate well beyond the copper and make
   **asymmetric**. For `U1` the inflated box put the *nearest* border on the far
   (right) side, so the lead-in legs ran from the VBUS pads (left column) straight
   across pad 5 (+3V3) and pad 4 (NC) in the right column — a short.
3. `add_breakout_stubs` stamped the tie as **locked** copper and routing ran with
   `freerouting_preserve_existing_copper=True`, so the short was preserved into the
   routed board (DRC: `Track [VBUS] ↔ Pad 5 [+3V3]` and `↔ Pad 4 [<no net>]`).

Because the tie crosses `U1`'s *own* pads, the short is **placement-independent** —
it reproduced identically on the tight (~16 mm) *and* the sprawled (~104 mm) rounds.
Every round was rejected for `illegal_routed_geometry` (shorts > 0) and scored
`-inf`, so the auto-pin had only bad options and the board failed to compose.

## The fix (`kicraft/autoplacer/brain/breakout_stubs.py`)

1. **`perimeter_tie_specs` walks the *pad field*, not `fp.GetBoundingBox()`.** New
   `_pads_bbox_mm()` returns the union of pad boxes (+margin). It hugs the copper
   symmetrically, so the nearest border is genuinely nearest and the lead-in legs
   from the two farthest same-net pads (always convex-hull-extremal) stay clear of
   every pad. This makes the `U1` tie route correctly instead of across its pads.
2. **`add_breakout_stubs` enforces a hard invariant for *every* waypoint path:** a
   segment that comes within clearance of a pad on another net (or a no-net pad) is
   a short, so the whole spec is dropped (`…:waypoint_crosses_pad`). Previously only
   the *radial* branch was guarded; the waypoint branch trusted the caller ("the
   caller owns collision-avoidance"). New `_foreign_pads()` / `_segment_clears_pads()`
   are shared by this guard and the radial `_safe_radial_length()`.
3. **Dead code removed:** `radial_escape_point` and `radial_breakout_specs` were
   test-only (no production caller; the radial path is built inline). Gone, with
   their tests.

(1) makes the tie correct on the placements we want to win; (2) is the universal
backstop that catches anything (1) can't see — e.g. a *neighbouring* footprint's
pad in the tie path. Together they hold the invariant for all leaves and any future
waypoint producer (incl. the curated `cfg['breakout_specs']` path).

> Note on the originally-planned "Step 3" (choose the perimeter side that crosses
> the fewest pads): it is **unnecessary**. With a pad-field box + margin, the
> border walk is outside every pad regardless of side, so there is nothing to
> choose. Adding a side-selection pass would have been redundant.

## Verification

- `pytest tests/test_breakout_stubs.py` — 14 pass, incl. two new regression tests:
  `test_perimeter_tie_uses_pad_field_not_inflated_bbox` (reproduces the U1 inflated-
  bbox geometry; old path crosses 2 pads, new path 0) and
  `test_waypoint_spec_crossing_foreign_pad_is_dropped` (the invariant).
- Re-stamped the **real** POWER `round_*_leaf_pre_freerouting` boards through the
  actual `auto_power_tie_specs → auto_signal_escape_specs → add_breakout_stubs`
  sequence, then `kicad-cli pcb drc`: **0 `shorting_items`, 0 `solder_mask_bridge`**
  on all three rounds (was 2 + 2 each). Round 0000/0001 (tight) stamp the tie
  cleanly; round 0002 (sprawl) the guard drops it (a neighbour pad intrudes) — both
  short-free.

Fast inner loop (no FreeRouting): re-stamp a `round_*_leaf_pre_freerouting.kicad_pcb`
and `kicad-cli pcb drc`. Full routed build (`kicraft build … --no-archive`) needs
`java` on PATH (currently absent here); the short lived in the *stamped* copper, so
removing it pre-route removes it from the routed board.

## What the first diagnosis got wrong

The original handoff claimed FreeRouting routed VBUS over `U1`'s pads because the
pads "are not obstacles in the DSN", and prescribed "make every pad a routing
keepout in the DSN export". Both are false:

- `export_dsn` uses KiCad's native `pcbnew.ExportSpecctraDSN`, which emits **every**
  pad (incl. no-net) as a `padstack` obstacle — confirmed by reading the exported
  DSN (`U1`'s image lists all 5 pins). There is nothing to "make a keepout".
- The shorting track is **pre-stamped, fixed-length (4.1525 mm) copper**, identical
  on every round — FreeRouting never routed it. The "tight fails, sprawl routes
  clean → sprawl pinned" chain was wrong: the sprawl shorts too.

Do **not** add a footprint-internal escape to the *shorts* gate (a tempting earlier
idea) — that would teach acceptance to ignore a real short and ship a non-fab board.

## Secondary issues (not this bug; track separately)

- **Global fine-pitch clearance collapse.** `_resolve_fine_pitch_rule`
  (`freerouting_runner.py`) lowers the routing clearance **board-wide** to the
  0.1 mm floor (and track to 0.15 mm) whenever `min_intra_footprint_pad_gap_mm`
  finds a sub-0.2 mm gap — which J1's *intrinsic* USB-C pad pitch always triggers.
  That relaxes clearance far from the connector too. Not the cause of this short,
  but a fab-margin risk worth localizing (relax only around the dense part).
- **~5 remaining `unconnected_items`** on the routed POWER leaf (interface/CC nets).
  With `max_unconnected=0`, clearing the short is necessary but may not be
  sufficient for rc=0 — this is the next layer once a full routed build can run.

## Already-landed context

`feat/build-two-phase-leaf-parent` (merged): `kicraft build` runs leaf phase
(`--leaves-only`, auto-pin best per leaf) → parent phase (`--parents-only`).
Necessary plumbing; with the tie short gone, the tight rounds now score for real and
the auto-pin selects a compact POWER placement instead of the sprawl.
