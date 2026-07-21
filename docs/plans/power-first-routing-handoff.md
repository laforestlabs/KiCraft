# Power-first routing — research + fix handoff (then: the missing repair pass)

Written 2026-07-21 (session box-investigate), after a back-and-forth on
KC-ZRAUR7. **Direction decision (user):** KiCraft's core design choice is to
outsource routing to freerouting — do NOT build our own router. The C1 v2
phase-3 A* pathfinder (`docs/plans/c1-v2-pathfinding-design.md`,
`docs/plans/c1-v2-phase3-handoff.md`) is ON THE SHELF as a fallback, not the
next step. The next step is to **help freerouting do its job**: give power
nets the priority the pipeline currently gives nothing, and then find out why
the phase-1/2 repair pass left no trace on the failed board.

Work order (user-prescribed): **Workstream A first, then Workstream B.**
(B is cheap; if A's research shows the repair pass would have caught the gap
anyway, say so before sinking more into A's heavier levers.)

## The motivating failure (fixture)

KC-ZRAUR7 = `~/.kicraft/projects/1/655` (LIVE, 2026-07-21, rc7, unc=1).
USB-C→dual-USB-A splitter. VBUS (netclass **Power, 0.5mm track / 0.153mm
clearance** vs 0.2mm Default) ends split in two islands:

- Island A (USB-C input): `J1.A4B9`, `J1.B4A9`, `C1.1` (10uF), `C2.1`,
  `U1.5` — includes a 26.0mm B.Cu run dangling at ≈(133.7, 114.0).
- Island B (both TPS2553 limiter INs, tied together): `U2.1` (160.9,112.9),
  `U3.1` (160.9,83.5), `C3.1`, `C5.1`, `R4.1`, `R8.1` — nearest approach a
  0.95mm F.Cu stub at (151.78, 111.09).
- The one missing connection spans ~18mm of HALF-EMPTY board (55% empty,
  congestion-growth ran to its 3.5x cap). Stuck net rotated across rounds
  (ILIM r1–2, VBUS r3) → class failure, not one cursed geometry.

## Verified facts (do not re-derive; file:line as of `7efccc5`)

1. **No net priority exists anywhere in the stack.** The invocation is
   `java -jar freerouting.jar -de <dsn> -do <ses> -mp <passes>`
   (`freerouting_runner.py:1199-1220`) — no ordering/priority/weight flags,
   and v1.9.0 has none. (Pinned to 1.9.0: 2.1 ignores `max_passes`.)
2. **freerouting 1.9.0 routes in arbitrary order** (read from its
   `BatchAutorouter.java`, tag v1.9.0): each pass collects incomplete items
   into a LinkedList by iterating the board item list sequentially — NO
   sorting by airline length / net class / width. Rip-up cost escalates
   linearly per pass (`start_ripup_costs * pass_no`); per-pass time budget
   doubles. Effective order ≈ DSN component/pad enumeration order.
3. **Structural priority = what is stamped before freerouting.** Stage
   order in `cli/_compose_route.py` `_route_parent_board`:
   leaf copper stamped + LOCKED (`export_dsn(lock_existing_traces=True)`,
   `freerouting_runner.py:640`; config `freerouting_preserve_existing_copper`
   at `_compose_route.py:94`) → breakout stubs + shield ties stamped
   (`:143-190`) → freerouting (GND probe first, GND-strip fallback
   `:196-221`) → SES import → GND pour/thermal vias/edge spine +
   `pour_power_planes` / `repair_parent_gnd_islands` / `repair_stranded_power`
   (`:261-336`, `brain/gnd_pour.py`) → `validate_routed_board` (`:347`) →
   `_attempt_illegal_geometry_repair` (`:370`) →
   `_attempt_signal_unconnected_repair` (`:382`).
4. **Wide nets are structurally last-in-practice:** VBUS needs a ~0.8mm
   clear corridor vs ~0.5mm for signals; thin nets that grab a channel in
   an earlier encounter can only be displaced by rip-up whose price rises
   every pass; leaf islands' locked copper walls the escape routes.
5. **The repair pass left no trace on 1/655:** round_0003
   `parent_pipeline.json` `routed_validation` has NO repair keys and the
   build log has no repair/tie lines. Unknown whether the wrapper ran.

## Workstream A — power-first levers (research, then fix the winner)

Evaluate on the frozen KC-ZRAUR7 workspace ($0, no LLM). For each lever:
prototype minimally, replay N-of-3, record unconnected + shorts + geometry
flags. Pick by evidence, not elegance. Candidate levers, in order of
architectural fit:

- **A1. Pre-stamp a power trunk as locked copper before freerouting.**
  Before DSN export, deterministically route the parent-level power-class
  interconnect (VBUS/VOUT_*: the netclass_patterns "Power" nets minus GND)
  as locked tracks, stamped ONLY through `add_breakout_stubs`
  (`brain/breakout_stubs.py:763`) so every DRC guard applies. Precedent:
  shield ties (`_compose_route.py:143`), GND edge spine. Power becomes
  literally the first-routed copper; freerouting then routes signals around
  a fixed spine — exactly how a human routes a PSU board. Open questions:
  corridor choice (straight/L between island anchor points? reuse the
  phase-1/2 candidate generator in `brain/unconnected_repair.py` at
  compose-time, pre-congestion?); what if the trunk fails to place (fall
  through to today's behavior, never block); interaction with congestion
  growth (a trunk may REDUCE needed area).
- **A2. Two-phase freerouting.** Phase 1: DSN with only power-class nets
  (strip the rest via `_strip_nets_from_dsn`, `freerouting_runner.py:980`);
  import SES; lock. Phase 2: full DSN as today. Costs ~2x route wall-clock
  (parent budget already scales per component, `_scale_parent_route_budget`).
  Simpler than A1 (no new geometry code) but slower and still subject to
  freerouting's whims within phase 1.
- **A3. Why didn't `pour_power_planes` / `repair_stranded_power` cover
  VBUS on 1/655?** Read their gating in `brain/gnd_pour.py` + config; run
  them directly on the failed board copy. If a pour CAN bridge the 18mm gap
  legally, this may be the cheapest fix of all — understand why it didn't
  fire (config default? B.Cu-only? net-name filter?) before building A1/A2.
  Do A3 FIRST — it is an afternoon and may reframe the workstream.
- **A4. DSN item-order bias.** Since 1.9.0 routes in item-list encounter
  order (fact 2), reordering the DSN so power-net pads/components enumerate
  first biases the first-pass claim on channels. Fragile, undocumented,
  version-coupled — research-only unless A1–A3 all disappoint; if it wins
  experiments, pin freerouting version + add a canary test.
- **A5. Newer freerouting.** Survey freerouting releases > 1.9.0 for (a) a
  fixed `max_passes`, (b) any real net-ordering/priority option. Timebox
  this; a version bump is its own risk (the 1.9.0 gotcha list in memory:
  DSN keepout-ignore, Omega-hang, submicron clearance).

Recommended sequence inside A: **A3 (investigate) → A1 prototype vs A2
prototype on 1/655 → pick one, productionize behind a config flag**
(e.g. `parent_power_first` default on once fixtures pass, kill switch kept).

## Workstream B — the missing repair pass (after A)

The phase-1/2 repair (`_attempt_signal_unconnected_repair`,
`_compose_route.py:449`) should have attempted a straight/L/dogleg tie
across an 18mm gap on an empty board — candidates that trivially fit. Yet
round_0003 shows no evidence it ran.

1. Read the wrapper's invocation condition (`:370-395`) and any config
   gates; establish from code whether the 1/655 build path could reach it
   (e.g. does it run for every rejected round, only the kept round, or only
   a specific tier?). The `[KEPT]`/discard round bookkeeping in the log
   (`Round 1/3 ... [KEPT]`, later rounds discarded) is a suspect: the
   repair may run on a different board object than the one promoted.
2. Reproduce: copy 1/655's generated tree **with `.kicad_pro`/`.kicad_prl`
   and `*_autoplacer.json`** (NEVER a bare .kicad_pcb — validate stamps
   default netclass rules into bare copies and manufactures fake
   violations), run `validate_routed_board`, call the wrapper directly.
   - Closes VBUS → the bug is the missing invocation; fix the call site.
   - Doesn't close → find which screen kills the candidates
     (`no_clear_path` vs gap cap vs anchor choice) — that result feeds the
     shelved phase-3 evidence file either way.
3. Whatever the outcome: make the repair pass LOUD — it should always log
   attempted/closed/skipped per net into the build log and persist a
   `repair_summary` into `parent_pipeline.json` state, so its absence can
   never again be ambiguous. (This logging fix is in scope for B regardless.)

## Verification / measurement

- Primary fixture: replay 1/655 (`replay --quality good --seed 0` on a
  copied workspace) — target VBUS unconnected 1→0 on ≥2 of 3 replays.
  NEVER compare artifacts across separate replay runs; measure inside one.
- Secondary: run_10 RP2040 stress fixture (unc=24) — a power trunk won't
  fix a QSPI escape, so expect little movement; it guards against
  regressions from the pre-stamped copper (shorts/geometry flags).
- Batch: next self-eval, target >=28/34 fab-ready (baseline 24/34,
  20260720T113207Z). Single-brief deltas are noise (N-of-3 discipline).
- Suite: pre-existing reds are parts test_3d_models/test_maturity and
  test_kicraft_lookup_lcsc easyeda fall-through; everything else green.

## Guardrails

- Fix at the source; no masking gates, no acceptance loosening. The
  byte-revert containment in the repair wrappers stays as is.
- `autoplacer/` + `cli/_compose_route.py` are load-bearing: surgical diffs,
  a test per behavior change, kill-switch config for new route-path steps.
- Pre-stamped power copper must go through `add_breakout_stubs` guards —
  nothing else writes copper.
- Deploy = restart BOTH services (`deploy/restart-web.sh`,
  `deploy/restart-build-worker.sh`).

## Done criteria

1. A3 answered in writing (why power pours didn't cover 1/655).
2. One power-first lever shipped behind a config flag; 1/655 replays close
   VBUS (>=2 of 3); run_10 shows no regression.
3. B answered in writing: why the repair pass left no trace, with the fix
   (invocation or screening) landed + the always-log/persist change.
4. Self-eval batch scheduled (real $ — ask the user first), target >=28/34;
   memory + this doc updated with results.
