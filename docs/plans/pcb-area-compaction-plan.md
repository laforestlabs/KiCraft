# PCB area compaction — investigation + resolution plan

**Status:** Phases 0–4 IMPLEMENTED 2026-07-02 (same day as the investigation); Phase 5
(parent packing) deferred pending fleet re-baseline, CMA-ES retune (Phase 2 item 4) still
to be launched. Implementation summary at the bottom of this file.
**Owner:** next implementing agent. Read the whole causal chain before touching anything —
every mechanism below was verified against run `~/.kicraft/projects/1/554` (KC-4W7KNW) and a
30-board fleet scan; file:line references are to `main @ d0d34e4`.

## 0. The problem, quantified

KC-4W7KNW ("high-side load switch", 11 parts, 361 mm² of footprints) shipped fab-ready at
**189.1 × 25.9 mm = 4 903 mm² — 7.4 % utilization, 7.3:1 aspect ratio**, with parts strung
along a line with ~60 mm empty gaps (J1 at x≈55, its only neighbor R1 at x≈122).

This is the fleet norm, not an outlier. Newest 30 promoted boards
(`projects/1/525..554`):

- **median utilization 14.7 %** (sum of footprint bboxes / board area)
- 15/30 boards below 15 % utilization
- 17/30 boards with aspect ratio > 2:1
- six boards ≥ 197 mm wide (525, 526, 527, 535, 539, 545, 551) — the seed scatter
  grid's 200 mm width bleeding straight through to the shipped board

A sane hand-layout for KC-4W7KNW is roughly 40 × 30 mm (~30 % utilization) — the pipeline
shipped **4×** that area.

## 1. Root-cause chain (verified, in causal order)

The waste is created at leaf-canvas derivation, installed by placement initialization,
unrecoverable by the optimizer, invisible to the scorer, unchecked by the acceptance
gates, and only cosmetically trimmed afterwards. Five mechanisms compound:

### RC1 — the leaf canvas inherits the synthesis scatter grid, not the parts

- `design/synthesis/kicad_pcb_stub.py:101-122` — `write_empty_pcb` scatters footprints on
  a **20 mm-pitch, 10-column grid starting at (20, 20)** ("the autoplacer.json carries the
  real placement plan, this just gets parts on the board"). 11 parts → a ~180 × 20 mm row.
- `autoplacer/brain/subcircuit_extractor.py:471-528` — `_derive_local_envelope` sets the
  leaf solve canvas to **the seed component bbox + margin**. For KC-4W7KNW:
  seed bbox ~177 × 22 mm + `subcircuit_margin_mm` (default **10.82**, tuned range
  6.97–13.38; this run used 9.26) → **195.2 × 40.2 mm canvas**.
- Nothing anywhere derives the canvas from **component area**. The comment in the stub
  says the scatter is throwaway; the extractor then treats its geometry as load-bearing.

### RC2 — `signal_flow_order` spreads the flow chain across the full canvas width

- `autoplacer/brain/placement_solver.py:1356-1365` — `_place_clusters` computes
  `flow_x_targets` **evenly spaced across the usable canvas width**
  (`frac = (i+0.5)/len(signal_flow)`). This design's `signal_flow_order = [J1, Q1, Q2, J2]`
  on a 195 mm canvas → targets at ~12 %, 38 %, 62 %, 88 % of 177 mm usable width.
  The final board shows exactly that: J1@55, Q1@147, Q2@180, J2@236.
- Signal-flow ordering is a *good* structure — the bug is that its spacing scales with
  the (arbitrary, RC1-inflated) canvas instead of with the parts.

### RC3 — the optimizer physically cannot recover the sprawl

- Force step is capped: `placement_solver.py:1628` `max_step = 1.5 * damping`, with
  `damping *= cooling_factor` (default **0.97**, `config.py:42`) per iteration →
  total movement budget ≈ Σ 1.5·0.97^k ≈ **50 mm**, in practice far less because
  attraction/repulsion equilibrate. J1 needed ~90 mm.
- SA refine explores with `sa_refine_move_radius_mm = 2.0` over 300 iterations
  (`placement_solver.py:405-417`) — diffusion; cannot close 60 mm gaps either.
- The solver already *knows* this: the adaptive-early-exit comment
  (`placement_solver.py:370-383`) documents that "PlacementScore.total saturates near
  100 on sprawled layouts" and that `max_iterations` is the backstop "for cases where
  compaction is unreachable under the current force balance". For leaves there is no
  outline cap to fail such a round (see RC5) — so it ships.

### RC4 — the scorer normalizes sprawl away

- `placement_scorer.py:81-99` — `_score_net_distance` normalizes total ratsnest length by
  `board_diagonal × n_nets`. A bigger canvas ⇒ bigger denominator ⇒ **the same absolute
  sprawl scores better**. This run: 60 mm gaps still scored `net_distance = 81/100`.
- `_score_compactness` (`placement_scorer.py:112-123`) is deliberately gentle
  ("Not heavily penalized": 7.4 % fill → 31.8/100) and its weight is small;
  `psw_aspect_ratio` weight is **0.02** (`config.py:234`), so this board's
  `aspect_ratio = 0.0` cost ~2 points.
- Net effect: the accepted round scored **71.5 total** with `compactness 31.8`,
  `aspect_ratio 0.0`, `net_distance 81`, `crossover 100`. Sprawl costs ≈ 15 of 100
  points; clean routability buys them all back.

### RC5 — no gate anywhere checks area

- Leaf acceptance gates (`leaf_acceptance.py`, observed in debug.json):
  `anchor_completeness, board_exists, drc_clearance, no_gross_courtyard_overlap,
  no_illegal_geometry, no_python_exception, no_shorts, no_unconnected, routed_board` —
  **all binary routability/DRC. Utilization/aspect are not gated, not even warned.**
- `leaf_size_reduction` (`autoplacer/brain/leaf_size_reduction.py`,
  candidates in `leaf_geometry.py:203+`) only **crops the outline border** in
  2/1/0.5/0.25 mm steps (default `leaf_size_reduction_max_attempts=3`, `max_passes=1`),
  floored at the placed-geometry bbox + 0.5 mm. It cannot touch internal gaps —
  on this run it took 195.2 × 40.2 → 189.4 × 24.3 and stopped (geometry bounds:
  189.6 × 23.3).
- Parent side: `cli/inspect_parent.py:262-283` already computes `wasted_area_mm2` /
  `stacked_area_mm2` fractions — but only as a report, never a gate or score input.
- KC-4W7KNW is single-leaf, so parent compose faithfully wrapped the sprawled leaf.

## 2. Resolution plan

Ordered by leverage. Phases are independently shippable; each has its own verification.
P1+P2 are the core fix; P3-P5 are compounding improvements.

**Global invariants for every phase:**

- The autoplacer is load-bearing: surgical diffs, every change behind a config key with
  the old behavior as fallback, so the tuner and replay A/Bs can compare arms.
- Array leaves (`array_placement.py`, skip force/SA) and wrapped loose-component leaves
  (re-based to (0,0)) must be exempted or handled explicitly — their canvases are
  already content-derived.
- Fab-ready rate must not regress: the acceptance floor stays routability. Area is an
  optimization target and (eventually) a *soft* gate, never a new hard rc6/rc7 source.
  (Established feedback: fix at the source, no masking gates.)

### Phase 0 — measurement (do first, ~half day)

Make waste visible before changing behavior, so every later phase has a baseline.

1. Compute and persist per-leaf and per-parent `area_utilization`
   (Σ footprint courtyard area / board area), `aspect_ratio`, and
   `bbox_utilization` (Σ area / placed-bbox area) into `debug.json`
   `solve_summary` + `run_status.json` + parent `parent_pipeline.json`. Most inputs
   already exist (`leaf_size_reduction.py:100-104` computes density;
   `inspect_parent.py:262+` computes waste).
2. Add the three numbers to the `[build] 4/5 verify:` line and the web build panel.
3. Baseline script: run the fleet scan (this doc §0's numbers) as
   `scripts/board_utilization_report.py` so post-change comparisons are one command.

### Phase 1 — right-size the leaf canvas from content (the fix, ~1-2 days)

**Change:** derive the solve canvas from component area + interface structure instead of
the seed scatter bbox.

- Site: `cli/solve_subcircuits.py:680-690` (the `extract_leaf_board_state` call) or
  directly in `_derive_local_envelope`. Compute
  `target_area = Σ courtyard_area / fill_target` with
  `fill_target = leaf_canvas_fill_target` (new config key, start **0.28**, tuner range
  0.15–0.45), aspect from a simple heuristic: near-square by default; widen toward the
  edge-zone axis when `component_zones` pin connectors to opposite edges; respect
  ArraySpec grid dims for array leaves (canvas ≥ grid + ring).
- Keep a floor: canvas ≥ largest courtyard + 2×margin per axis, and canvas ≥ the
  routability minimum the size-reduction pass uses.
- **Grow-on-failure ladder:** when no round routes at fill 0.28, retry the leaf at 0.22,
  then 0.17 (config: `leaf_canvas_fill_ladder`). The per-round retry loop already exists
  in `_solve_leaf_subcircuit` (rounds) — thread the ladder through round index or the
  autoexperiment round. `enable_board_size_search` + `_apply_board_outline`
  (`hardware/adapter.py:573-576`) already provide the mechanism to stamp a changed
  outline.
- RC2 falls out for free: `_place_clusters` flow targets spread across a *right-sized*
  canvas is exactly the intended structure.
- Feature flag: `leaf_canvas_mode = "content" | "seed-bbox"` (default `content` after
  verification; `seed-bbox` preserves today's behavior byte-for-byte for replay A/B).

**Expected effect on KC-4W7KNW:** 361 mm² / 0.28 ≈ 1 290 mm² ≈ 41 × 31 mm canvas —
matches the hand-layout estimate.

### Phase 2 — make sprawl expensive to the scorer (~1 day + tuner run)

1. `_score_net_distance` (`placement_scorer.py:85-99`): normalize by a **content-derived
   scale** — `sqrt(Σ component area) × n_nets` (or the component-bbox diagonal) — instead
   of the canvas diagonal, so the score is canvas-invariant.
2. Re-curve `_score_compactness` so <10 % fill scores near 0 (currently 36) — or fold it
   into `bbox_packing` and retire it.
3. Raise `psw_aspect_ratio` default (0.02 → ~0.08) and re-balance `psw_bbox_packing`.
4. **Re-run the CMA-ES tuner** (`kicraft/tuning/`, area axes already in the reward)
   after 1-3: the current psw defaults were tuned against the old metric semantics.
   $0 via `replay`; check the i-series protocol in the tuning memory before launching.

### Phase 3 — deterministic compaction pass (post-SA, pre-route, ~1-2 days)

Even with a right-sized canvas, force equilibrium leaves slack. Add a **squeeze pass**
after SA refine / alignment repair, before routing:

- Per-axis sweep toward the placed-bbox centroid: sort components by distance, slide
  each as far as legality allows (courtyard + clearance + keepouts + locked/edge-pinned
  parts respected) — same move primitive family as the composer's `_push_clear` /
  extremal slide, reused not reinvented.
- Then re-run `leaf_size_reduction` (its geometry bounds are now genuinely smaller) and
  let its existing reroute-threshold logic decide whether a reroute is needed.
- Config: `leaf_compaction_pass = true`, off in `seed-bbox` mode.
- This is the surgical, verifiable alternative to raising force budgets (RC3): do NOT
  try to fix sprawl by raising `max_step`/iterations — that destabilizes converged
  behavior the whole tuned config depends on.

### Phase 4 — area visibility in acceptance + review (~half day)

- Add `area_utilization` / `aspect_ratio` to `leaf_acceptance` as **WARNING-level**
  observations (structured, surfaced in run_status + web), not hard gates.
  Thresholds: warn below 15 % util (part count ≥ 5) or aspect > 4.
- Parent: include utilization in `routed_validation` output and the `[build]` verify
  line (Phase 0 did the plumbing; this makes it part of the acceptance record).
- Revisit making it a soft re-solve trigger (spend one extra round at a tighter canvas
  when warned) only after Phase 1-3 data shows how often warnings still fire.

### Phase 5 — parent-level packing (only if multi-leaf waste remains)

Phases 1-3 shrink the leaves; the composer's candidate search + Step 16 + extremal slide
already pack leaf blocks. After the fleet re-baseline, if multi-leaf parents still show
>2× waste vs Σ leaf areas, extend the composer's sprawl penalty / outline derivation
(`subcircuit_composer.py:2300+`, `_derive_board_outline`) — separate plan; do not start
here.

## 3. Verification protocol (every phase)

1. **Unit:** new tests per changed module (canvas derivation, scorer normalization
   invariance — same layout, two canvas sizes, same score; squeeze-pass legality).
2. **Replay A/B ($0):** `kicraft replay` over the corpus with `leaf_canvas_mode` old vs
   new, PYTHONHASHSEED pinned; compare fab-ready rate, utilization, aspect, wall-clock,
   DRC counts via the Phase-0 report script. **Gate: fab-ready rate not worse; median
   utilization ≥ 25 %.** Remember run-to-run noise: N-of-3 + median for any LLM-path
   comparison; replay itself is deterministic.
3. **Targeted replays:** KC-4W7KNW (this board: expect ≤ ~45 × 35 mm), one dense-array
   design (KC-SMQ3HX class), one multi-leaf USB-PD design (530-class), one
   edge-connector-heavy design (BNC/USB-C class) — the classes with canvas-sensitive
   history.
4. **Self-eval** after P1+P2 land: full rubric run; compare area medians and fab-ready
   vs the 2026-06-24 baseline (11/28 fab-ready, mean 71.1).

## 4. Effort + sequencing summary

| Phase | What | Effort | Risk |
|---|---|---|---|
| 0 | utilization metrics + baseline script | 0.5 d | none |
| 1 | content-derived leaf canvas + grow ladder | 1-2 d | route failures on dense leaves → ladder + flag |
| 2 | scorer canvas-invariance + retune | 1 d + tuner | score-shape regressions → replay A/B |
| 3 | post-SA squeeze pass + re-crop | 1-2 d | legality bugs → reuse composer primitives, heavy tests |
| 4 | warning-level area acceptance | 0.5 d | none (warnings only) |
| 5 | parent packing (conditional) | separate plan | — |

Do 0 → 1 → verify → 2 → verify → 3 → 4. Phase 1 alone should collapse the worst of the
fleet (every ~200 mm-wide board is RC1 bleed-through).

## 5. Implementation record (2026-07-02)

Phases 0–4 implemented in one pass; every change is config-gated with the old behavior as
fallback. Key artifacts:

- **Phase 0:** `placement_utils.board_utilization_metrics` (solver-side) +
  `inspect_parent.board_utilization` (pcbnew-side, exact courtyard sums). Persisted:
  leaf `debug.json` (`solve_summary.scheduling_metadata.board_metrics` +
  `extra.board_metrics`), parent `parent_pipeline.json`
  (`state.packing_metadata.board_metrics`), `run_status.json`
  (`hierarchy.board_metrics` via `autoexperiment._extract_parent_board_metrics`),
  the `[build] 4/5 verify:` line (`util= aspect= bbox_util=` suffix), web build panel
  KV rows. Baseline: `scripts/board_utilization_report.py` (reproduced the §0 numbers:
  15/30 below 15 % util).
- **Phase 1:** `derive_content_canvas` + `set_extraction_canvas`
  (`subcircuit_extractor.py`); ladder in `_solve_leaf_subcircuit` (content fills →
  seed-bbox terminal fallback, per-attempt extraction tracked with the winning round).
  Config: `leaf_canvas_mode` ("content" default / "seed-bbox"),
  `leaf_canvas_fill_target` 0.28 (tuner range 0.15–0.45), `leaf_canvas_fill_ladder`
  [0.22, 0.17]. Array leaves exempt (grid-placed; already content-sized).
  **Byte-parity verified:** seed-bbox mode reproduces pre-change placements exactly
  (USB_PD_TRIGGER corpus fixture, PYTHONHASHSEED=0).
- **Phase 2:** `_score_net_distance` normalized by 2×√(2·ΣA/fill) in "content" mode
  (`placement_score_net_scale`, canvas-invariant; 2× headroom keeps gradient on
  seed-bbox/parent canvases); `_score_compactness` strict curve
  (`placement_compactness_curve`, ≤5 % fill → 0). **Both are implemented + tested but
  OPT-IN (default None = legacy)**, and `psw_aspect_ratio` stays 0.02: replay A/B showed
  a global flip regresses parent route (530/535) and even a leaf-scoped flip regresses
  the 535 J1 leaf (routes at fill 0.28 under legacy scoring, ladders to seed-bbox and
  strands under content scoring) — the psw weights were tuned against the legacy score
  shapes. **These knobs are the CMA-ES retune campaign's search surface (item 4, NOT yet
  run; i-series protocol in the tuning memory); flip defaults only with retuned weights.**
- **Phase 3:** `brain/leaf_compaction.compact_toward_centroid` — per-axis
  nearest-centroid-first slide, clearance/keep-out/keep-in/board-bound aware,
  deterministic; wired as Step 15.5 in `PlacementSolver.solve()` gated on
  `leaf_compaction_pass` (None = follow canvas mode; forced off for seed-bbox attempts
  including the ladder's terminal fallback).
- **Phase 4:** `leaf_acceptance` `area_utilization` observation gate (ALWAYS passes;
  structured `warning: True` + notes; thresholds `leaf_area_warn_utilization` 0.15 @ ≥5
  parts, `leaf_area_warn_aspect` 4.0) fed from `validation["board_metrics"]` at persist
  time; parent-side `gate["warnings"]` entries in `_promote_verify_fab` on shipped
  wasteful boards.

**Targeted A/B (replay --quality fast, seed 0; "old" = `leaf_canvas_mode="seed-bbox"`,
"new" = shipping defaults: content canvas + squeeze, legacy scoring):**

| board | old (seed-bbox) | new (shipping defaults) |
|---|---|---|
| 554 KC-4W7KNW | rc0, 186×27 mm, 7.1 % util, aspect 6.9 | rc0, **36.6×37.1 mm, 25.8 % util, aspect 1.03** |
| 533 RC/BNC | rc0, 33×23 (already tight) | rc0, 33×22 (unchanged) |
| 535 LED array | rc6 (connector stranded −109 mm) | **rc0, ships** (custom-outline resolver governs its area) |
| 530 USB-PD multi-leaf | rc6 (parent search shorts @187×167) | rc7 (routes clean, 2 courtyard overlaps; parent packing = Phase 5) |

Fab-ready 2/4 → 3/4; no board got worse; Phase-4 warnings fire on the wasteful boards.
Remaining waste lives in the array/custom-outline class and parent-level packing — both
explicitly Phase 5. Tests: `test_content_canvas.py`, `test_leaf_compaction.py`,
`test_scorer_canvas_invariance.py`, `test_leaf_acceptance_area.py`.

**Verification gotcha for future A/Bs:** a replay arm that fails BEFORE promote leaves
the copied production `<stem>.kicad_pcb` untouched — measuring it "as the arm's board"
silently compares against production. Check the arm's rc (and `[build] 3/5 promoted`)
before measuring.
