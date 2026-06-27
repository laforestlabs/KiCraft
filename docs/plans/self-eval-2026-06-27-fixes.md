# Self-eval 2026-06-27 — most-impactful fixes (implementation plan)

**Audience:** an implementing agent. **Source run:** `logs/self_eval/20260627T032049Z`
(28-brief `BENCHMARK_PROMPTS` corpus, design=`deepseek-v4-flash`, judge=`minimax-m3`,
spend ≈ $0.99 + ≈ $0.10 re-grade).

This plan is built on the **judge-independent build reality** (parent-board
`_verify_routed_board` verdicts + KiCad DRC on the promoted boards), *not* the LLM
letter grades — grades measure synthesis quality and are a separate axis from
fab-readiness (a board can grade **B** and still be not-fab-ready).

---

## 0. DONE — Judge truncation fix  *(committed `b6b7adf`)*

16/28 runs originally scored `final=None` because the
Class-J judge (`minimax-m3`, a reasoning model) was called with `max_tokens=1600`
and burned the whole budget on reasoning tokens before emitting the JSON answer
(`_extract_json` → `"no JSON object found in reply"`). Identical failure mode the
review gate already fixed (`review_max_tokens 3000→24000`); the judge never got
the same treatment.

Changes:
- `kicraft/server/config.py` — new `Settings.eval_judge_max_tokens: int = 24000`
  (+ `KICRAFT_EVAL_JUDGE_MAX_TOKENS` env, parsed in `from_env`).
- `kicraft/eval/judge.py` — `grade_class_j` default `max_tokens` 1600 → 24000.
- `kicraft/eval/run_web.py` — `evaluate_project(..., judge_max_tokens=)` threaded
  to `grade_class_j`; `main` reads the Settings knob.
- `kicraft/eval/self_eval.py` — `evaluate_one(..., judge_max_tokens=)` threaded;
  both call sites pass `s.eval_judge_max_tokens`.

Re-grading the 16 dropped runs with the fix recovered every grade (16/16),
giving a complete 28/28 scoreboard.

---

## 1. DONE — Cluster B: Parent GND-pour islands  *(committed `3d5877e`)*

**Files changed:** `gnd_pour.py`, `_compose_route.py`

**Symptom.** Parent verdict `unconnected=N` where the open "nets" are
GND-zone ↔ GND-zone "missing connection" pairs.

**Fix.**
- `gnd_pour.py`: new `repair_parent_gnd_islands()` — convergence-loop repair
  (max 5 iters) with via-stitching for cross-layer overlapping B.Cu↔F.Cu fill
  islands, falling back to track-stitching for same-layer islands. Collision
  guard against existing copper.
- `_compose_route.py`: replaced single-pass `repair_stranded_gnd` with the new
  convergence-loop function in the parent route pipeline.

**Verification:** `tests/test_power_pour.py` (23), `tests/test_parent_route_gnd_plane.py` (3) — all pass.
Full suite: 2042 pass, 0 regressions from this change.

**Impact.** Targets #26 (8 GND islands — largest open-count board) and the GND
component of #9, #22, #24.

---

## 2. DONE — Cluster E: Synthesis honesty + schematic  *(committed `3d5877e`)*

**Files changed:** `validation.py`, `cli_app.py`, `symbol_library.py`, `router.py`

Three independent fixes:

**E1 — silent_substitution** (#4 speaker-crossover). When the BOM commits a
part that is a class substitution of what the brief named (e.g. "binding-post
terminals" → screw-terminal-5mm-2p), surface it as an open_question rather
than committing silently. New `check_named_part_substitutions` detector in
`validation.py`; wired into BOM commit in `cli_app.py`.

**E2 — ERC error** (#27 stepper-a4988). New `_normalize_regulator_output_pins`
retypes easyeda2kicad's `power_in` to `power_out` for VREG/VOUT/VCP regulator
output pins. Added `_normalize_ic_pins` aggregator; wired into
`extract_symbol_block`.

**E3 — netlist faithfulness** (#11 fpc-breakout). New `_label_position_free`
check prevents "label slide" net merges by ensuring no existing label from a
different net occupies the same position before placing a label.

**Verification:** `tests/test_kicraft_symbol_library.py` (24),
`tests/test_kicraft_netlist_faithfulness.py` (5), `tests/test_kicraft_synthesis.py` (19),
`tests/test_schematic_layout_erc.py` (2), `tests/test_power_net_fallback.py` (5) — all pass.

---

## 3. INVESTIGATED — Cluster A: the rotation-stranding mechanism is REFUTED

**The plan's stated mechanism does not hold.** Reproduced deterministically
(`replay --no-route` on run_25/run_05/run_02/run_09/run_22) and traced the code:

- The failing designs carry **no subcircuit blocks** — they are single-leaf
  boards whose `component_zones` are plain connectors, so the
  `_connector_edge_x`/`_connector_edge_y` subcircuit/`anchor_offset_mm` branch
  never executes.
- An edge-pinned block is **never re-rotated** after `_pin_edge_components`:
  it is added to `self._pinned_targets` and `_optimize_rotations` skips any ref
  in `_pinned_targets` (`placement_solver.py:2157`); `_sa_refine` only moves the
  `unlocked` set. `_restore_pinned_positions` only translates, reusing the
  current rotation. So "Y→X after 270°" has no trigger.
- `connector_stranded` is a KiCraft synthetic acceptance-gate verdict, not a
  KiCad DRC type; it never appears in the `drc` counts. It is independent of the
  `copper_edge_clearance` failures the corpus actually shows.

**The real adjacent cluster is `copper_edge_clearance` (5 designs: run_02, 05,
09, 22, 25).** Root cause (reproduced): `_compute_final_outline`
(`compose_subcircuits.py:740`) snaps each EDGE-constrained side to the connector
mouth anchor (`return c_val`) and discards the placed-child geometry `g_val`, so
a corner-mount pad or a leaf-routed track sitting 0–0.2 mm proud of the mouth is
left over the edge.

**Deferred — needs the copper envelope, not the body bbox.** The naive fix
("keep clearance from `g_val`") regresses mouthed connectors with
`connector_edge_overhang_mm`: `placed_bboxes` is the *content* bbox (body +
copper), so it cannot tell a connector's overhanging **mouth/body** (which is
supposed to stick past the edge) from stray **copper** (pads/tracks) that must be
cleared (verified: it breaks `test_edge_connector_pinned_to_parent_bottom_outline`
and `test_compute_final_outline_edge_pinned_no_margin`). A correct fix must thread
the per-child **copper** extent (`child_layer_envelopes`, excluding the body) into
`_compute_final_outline`, or extend `_repair_parent_outline`
(`_compose_validate.py`) to enumerate transformed child footprint pads/traces (it
is currently blind to intra-leaf copper, which is also why
`geometry_validation.outside_pad_count` reads 0 while pads sit outside).

---

## 4. DONE — Cluster C: `courtyards_overlap`  *(uncommitted)*

**Files changed:** `placement_utils.py`, `placement_solver.py`, `compose_subcircuits.py`
(+ tests in `test_courtyard_overlap_resolution.py`).

**Real root cause (reproduced on run_06, instrumented).** Two compounding bugs
at the **parent-compose** level — block-level courtyard overlaps were 0 right
after `solve()` but 2 after the post-solve geometry steps:

1. **`_ensure_edge_blocks_extremal` reintroduces overlaps after the solver's
   last courtyard pass.** The solver runs Step 16 (`_resolve_courtyard_overlaps`)
   at the end of `solve()`, but compose then runs `_slide_constrained_to_cluster`
   + `_ensure_edge_blocks_extremal`, which align same-edge blocks to the same
   perpendicular extreme — collapsing the X-separation Step 16 relied on. This is
   the KC-59PTZA "final pass isn't last" pattern, one level up.
2. **The Step-16 exemption conflated copper-compat with courtyard-compat.**
   `_blocker_pair_compatible` (= `can_overlap_sparse`) returns True for two
   same-side THT pin-headers whose annular rings don't touch — but their
   courtyards share ONE layer (F.CrtYd) and DO produce a real `courtyards_overlap`
   DRC. The exemption is meant only for genuine opposite-side stacks (F.CrtYd vs
   B.CrtYd, different layers).

**Fix (three surgical changes):**
- `placement_utils._back_courtyard(comp)`: True iff a block sits on the back
  copper layer (`block_side == "back"` or `block_force_back_only`).
- `placement_solver._resolve_courtyard_overlaps`: exempt a pair only when
  `_blocker_pair_compatible(a,b) AND _back_courtyard(a) != _back_courtyard(b)`
  (genuine opposite-side stack). Same-side compatible pairs are now separated.
- `placement_solver._push_clear`: when the smaller-overlap-axis push is fully
  blocked by the board edge (an edge-pinned connector can't move further out on
  its pinned axis), fall back to the other axis instead of giving up.
- `compose_subcircuits._compose_artifacts`: re-run `_resolve_courtyard_overlaps`
  after `_ensure_edge_blocks_extremal`, so the courtyard pass is the GENUINE last
  geometry step (only unlocked blocks move; edge flush/extremity preserved).

**Verification (`replay --no-route`, parent courtyard measured with
`courtyard_overlap.measure_courtyard_overlaps`):**
- run_06 usb-c-full-breakout: 2 gross → **0**
- run_23 can-node: 2 → **0**; run_04 speaker-crossover: 1 → **0**;
  run_12 esp32-s3-sensor: 1 → **0**
- run_01 rc-lowpass-bnc: 1 → 1 (unchanged; pre-existing intra-leaf J1↔RV1, not
  regressed)
- **No regression:** run_19 relay-quad (full routed build) produces an IDENTICAL
  verdict under old and new code (`connector_stranded:J4@-4.85mm`,
  `unconnected_nets`) — run_19's full-build rejection is pre-existing
  replay-vs-original nondeterminism, not this change. run_06's ~4 stranded
  connectors are also pre-existing (the original strands J1/J4/J5/J3 — 5
  connectors crammed on one short edge is a synthesis over-constraint).
- Full test suite: 2046 passed (+4 new courtyard tests), same 14 pre-existing
  failures, 0 regressions.

---

## 5. NOT DONE — Cluster D: Route/infra fail (no routable parent)

**Designs:** #10 rp2040-min, #17 led-cc-driver (rc=6), #20 encoder-oled-panel
(rc=1, a hard error). Memory flags these as historically never routing. The
diagnosis agent was cut off before reporting; treat as individual model/route
issues, not one shared bug. Diagnose run_20's rc=1 exception first (most likely
to be a generic code defect worth a guard).

---

## 6. Verification results

Full test suite: **2042 passed, 14 failed** (all pre-existing: missing
fixtures, optional modules, flaky tests). **0 regressions** from committed
changes.

Key test files verified:
- `tests/test_power_pour.py` — 23 pass (GND strand repair)
- `tests/test_parent_route_gnd_plane.py` — 3 pass
- `tests/test_kicraft_symbol_library.py` — 24 pass
- `tests/test_kicraft_netlist_faithfulness.py` — 5 pass
- `tests/test_kicraft_synthesis.py` — 19 pass
- `tests/test_kicraft_synthesis_stageb.py` — 1 pass
- `tests/test_schematic_layout_erc.py` — 2 pass
- `tests/test_power_net_fallback.py` — 5 pass
