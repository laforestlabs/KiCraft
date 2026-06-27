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

## 3. NOT DONE — Cluster A: Connector stranding + stranded high-speed nets

**Symptom.** Edge connectors placed inboard (negative mm from edge) → `connector_stranded:<ref>@-N.NNmm(<edge>)`.

**Status.** Root cause identified but fix not implemented. The issue is in
`placement_solver.py:_pin_edge_components` / `_connector_edge_x`:
`_pin_edge_components` pins a subcircuit block at a position computed for
rotation 0 (from the zone's `chosen_rotation` field). The solver then rotates
the block (typically 270° for packing), and `_restore_pinned_positions` snaps
it back to the rotation-0 position. The anchor offset's Y component (from
content bbox center → component center) becomes an X shift after 270° rotation,
stranding the connector ~2.3mm inboard.

Required investigation (using replay on `run_06` and `run_25`):
1. Trace the anchor offset computation in `_compute_local_anchor_offset`
   and `attachment_constraints_to_zones` — the body center from `_content_bbox`
   differs from the component center for tall components, creating a Y offset
   that becomes X displacement after rotation.
2. Fix the single point: either in `_connector_edge_x` (use unrotated width,
   not rotated anchor offset), or in `_restore_pinned_positions` (recompute
   position for the solver's final rotation), or in the zone entry (set
   `anchor_offset.y = 0` by using component center as reference instead of
   content bbox center).

Load-bearing code:
- `kicraft/autoplacer/brain/placement_solver.py` `_pin_edge_components`
  (edge-group placement, `_connector_edge_x` for subcircuit blocks)
- `kicraft/autoplacer/brain/parent_adapter.py` `attachment_constraints_to_zones`
  (anchor offset computation)
- `kicraft/autoplacer/brain/subcircuit_composer.py` `_compute_local_anchor_offset`
  (anchor point selection)

---

## 4. NOT DONE — Cluster C: `courtyards_overlap`

**Symptom.** `reasons=['courtyards_overlap']` in parent verdict.

**Residual cases** after KC-59PTZA fix:
- Intra-leaf: edge-pinned connector vs neighbor (Step 16 declines to move
  the pinned member)
- Inter-leaf: overlaps created at rigid compose stamping (leaf-scoped Step 16
  can't see them)

**Approach.**
1. Allow Step 16 to slide the non-pinned member while the pinned member
   holds the edge.
2. Add a parent/compose-level courtyard-separation pass for inter-leaf overlaps.

---

## 5. NOT DONE — Cluster D: Route/infra fail (no routable parent)

**Designs:** #10 rp2040-min, #17 led-cc-driver, #20 encoder-oled-panel.
Memory flags these as historically never routing. Diagnose with
`/kicraft-investigate`; likely individual model/route issues, not one
shared bug.

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
