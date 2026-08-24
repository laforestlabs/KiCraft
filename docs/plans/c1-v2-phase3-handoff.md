> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# C1 v2 phase 3 — session handoff (grid A* + rip-up for the last-net repair pass)

> **STATUS 2026-07-21: SHELVED, not next.** After critical discussion the
> user ruled that building our own router contradicts KiCraft's core design
> choice of outsourcing routing to freerouting. The active plan is
> `docs/plans/power-first-routing-handoff.md` (power-first levers, then the
> missing-repair-pass investigation). This document stays as the fallback
> design if those levers prove insufficient.

Written 2026-07-21 (session box-investigate). Fresh-session brief for
implementing the phase-3 pathfinder. The authoritative technical design is
`docs/plans/c1-v2-pathfinding-design.md` — read it in full before writing
code; this document adds the operating context, the code map, the newest
live evidence, and the verification recipes a new session needs.

## What "C1" is, in one paragraph

C1 is the failure-cluster label (from the 2026-07-19 codebase review) for
KiCraft's single biggest remaining rc7 mode: a board routes almost
completely — 0 shorts, hundreds of traces — but freerouting **walls off the
last net(s)**, ending with `unconnected=1..2` at the fab gate. Space is not
the problem (boards fail at <20% utilization); the remaining net has no
clear escape path through the copper that already exists. After freerouting,
a deterministic repair pass (`repair_unconnected_signals`) tries to close
the remaining opens by stamping short tie tracks, but its candidate family
is only straight/L/dogleg shapes: when those are all screened out it reports
`no_clear_path` and gives up. C1 **v2** is the second-generation fix
program: phases 1–2 (shipped `b725d8f`, 2026-07-21) fixed the repair pass's
anchoring/screening/acceptance so that its remaining failures are *honest*
`no_clear_path`. Phase 3 — this task — replaces that give-up with a real
router: **grid A\* pathfinding + string-pulling + bounded rip-up**.

## Why now (live evidence)

- **Post-phase-1–2 run_10 (RP2040 stress fixture, unc=24):** all 21
  remaining edges are honestly `no_clear_path` with ~16k pruned candidates.
  The QFN-56 QSPI escape and the 14-edge GPIO fan-out need a router, not
  more candidate shapes. (Design doc §"What already shipped".)
- **KC-ZRAUR7 (1/655, 2026-07-21, LIVE web build):** USB-C→dual-USB-A
  splitter, rc7 with exactly 1 unconnected. The stuck net *changed between
  compose rounds* (ILIM rounds 1–2, VBUS round 3), congestion-growth ran to
  its 3.5× area cap, and the final board is 55% empty — the walled-off
  signature, not a space problem. Same brief passed as KC-3WN46Z (1/652)
  hours earlier with a smaller synthesized design: this brief sits right on
  the routing-capability boundary, which is exactly where phase 3 pays.
- **Breadth:** GND-strand + unconnected=1 clusters together are the #1 rc7
  blocker by breadth across live boards (see memory
  `kicraft-unconnected-1-cluster-walled-off-signal-power`,
  `kicraft-gnd-plane-strand-walled-off-breadth`).

## Code map (read in this order)

| File | What it is |
| --- | --- |
| `docs/plans/c1-v2-pathfinding-design.md` | The design: A* grid spec, string-pulling, rip-up bounds, containment, config knobs. Follow it. |
| `kicraft/autoplacer/brain/unconnected_repair.py` (~440 lines) | The repair pass. `repair_unconnected_signals()` is the entry; `_copper_obstacles()` builds the obstacle index the A* must reuse; the `no_clear_path` verdict near line 423–428 is the exact spot the pathfinder replaces. |
| `kicraft/cli/_compose_route.py` | The two accept-gated wrappers: `_attempt_signal_unconnected_repair` (line ~449) and `_attempt_illegal_geometry_repair` (line ~570). Both byte-revert unless unconnected drops, shorts hold, and geometry flags don't worsen — this containment is why rip-up risk is bounded to wasted compute. Do not add new accept logic. |
| `kicraft/autoplacer/brain/breakout_stubs.py` | `add_breakout_stubs()` (line ~763) — the ONLY thing that writes copper. The pathfinder emits `BreakoutSpec`s (waypoints, `start_xy/net` anchors, `via_at_end`) and stamps through it so every existing DRC guard applies. |
| `kicraft/autoplacer/config.py` | Where the new `signal_repair_*` knobs land (see design doc §Config knobs). |
| `kicraft/autoplacer/brain/gnd_pour.py` | Has its own `no_clear_path` skip (line ~525) for GND stub routing — same disease, NOT in phase-3 scope, but keep it in mind: a later phase can reuse the pathfinder here (the GND-strand cluster's owning fix). |

New module: `kicraft/autoplacer/brain/repair_pathfinder.py`, called from
`repair_unconnected_signals` when the candidate family fails an edge,
behind `signal_repair_pathfinder_enabled`.

## Verification recipes (all $0, no LLM)

1. **Direct-wrapper harness (fastest loop, used to build phases 1–2):**
   copy a fixture project dir **including `.kicad_pro`/`.kicad_prl` and
   `*_autoplacer.json`** (NEVER a bare `.kicad_pcb` copy — validate stamps
   default 0.20 mm netclass rules into a bare copy and manufactures fake
   violations; memory `kicraft-validate-board-needs-project-context`), run
   `validate_routed_board`, call `_attempt_signal_unconnected_repair`
   directly, re-validate. Measure unconnected before/after in the SAME
   script run.
2. **Replay:** `python -m kicraft.design.cli_app replay --project <copy>
   --quality good --seed 0` re-runs the full place/route tail on the frozen
   seed. NEVER compare artifacts across two separate replay runs (memory
   `kicraft-replay-cross-run-contamination`); routing is seed/hash-noisy, so
   claim wins only on N-of-3.
3. **Fixtures:**
   - run_10 RP2040 stress board (self-eval batch 20260720T113207Z), unc=24;
     target: unc < 10 from the repair pass alone.
   - runs 06/13/14/24/27 of the same batch: measure JOINTLY with the N2
     geometry-repair pass — its re-close currently loses by one open on
     13/14; a pathfinder tie is exactly the missing piece.
   - KC-ZRAUR7 = `~/.kicraft/projects/1/655` (VBUS open, live board):
     replay the frozen workspace; expect unconnected 1→0.
4. **Full batch:** next self-eval target ≥28/34 fab-ready (was 24/34 on
   20260720T113207Z). Single-brief deltas are noise; only batch-level
   movement counts (memory `kicraft-self-eval-2026-06-24-findings`).

## Guardrails / working agreements

- **Fix at the source; no masking gates.** The pathfinder must close nets,
  never relax acceptance. The byte-revert containment in the wrappers stays
  exactly as is.
- `autoplacer/` is load-bearing: surgical changes, small diffs, tests for
  every behavior change.
- Locked leaf copper (`IsLocked()`) is never rippable.
- Pours are not obstacles and pour nets are excluded from repair (pours
  refill around new copper) — same rules as today.
- Deploy = restart BOTH services (`deploy/restart-web.sh`,
  `deploy/restart-build-worker.sh`); pipeline changes are invisible to the
  live site until the build worker restarts.

## Done criteria

1. `repair_pathfinder.py` implements A* (0.2 mm grid, 2 layers, via cost
   ~25 steps) + string-pulling + ≤3-segment/≤2-round rip-up per the design
   doc, config-gated, stamping only through `add_breakout_stubs`.
2. run_10 harness: unc 24 → <10; no geometry-flag regressions.
3. KC-ZRAUR7 replay closes VBUS (1→0) on at least 2 of 3 replays.
4. Unit tests: pathfinder grid/obstacle/string-pull cases + a rip-up
   bounded-set case; suite green (pre-existing reds: parts
   test_3d_models/test_maturity, test_kicraft_lookup_lcsc easyeda test).
5. Self-eval batch scheduled (real $ — ask the user first) with target
   ≥28/34; memory + this doc updated with results.
