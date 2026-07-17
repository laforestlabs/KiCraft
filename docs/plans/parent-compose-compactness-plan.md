# Parent-compose compactness — KC-AXHQTP investigation + fix plan

**Status:** Fix 1 + Fix 2 IMPLEMENTED 2026-07-16 (`placement-streamline`, atop `d26ce89`).
Fix 3 aspect-sweep is now live for free; the optional overhead sweep is intentionally
NOT built (Fix 2's per-candidate re-fit subsumes it — see below). Investigated on run
`~/.kicraft/projects/1/626` = KC-AXHQTP.
This is the deferred **Phase 5** of `docs/plans/pcb-area-compaction-plan.md` ("parent-level
packing — separate plan"), whose trigger condition (">2× waste vs Σ leaf areas on multi-leaf
parents") is now met at **5.3×**. Phases 0–4/6 of that plan fixed the *leaf* side and worked:
this board's leaves are compact (17–36 mm each). The waste owner is now the **parent compose**.
File:line refs are to `placement-streamline @ d26ce89`.

### Implementation summary (what landed)

- **Fix 1** — new `_edge_bank_geometry` helper (`cli/compose_subcircuits.py`) does per-edge
  bank accounting (opposing banks add depths; same-edge members stack); `_seed_outline_dimensions`
  calls it and DROPS the old `sum*0.6` single-row floors. Guard: `tests/test_parent_seed_edge_banks.py`.
- **Fix 2** — new `_refit_seed_from_placement` measures the true content need from the pass-1
  placement (interior union bbox ∪ per-edge banks). `_compose_artifacts` gained a
  `seed_size_override` arg and now stamps `ParentCompositionState.refit_seed`; `_search_best_layout`
  runs a pass-2 candidate on that tighter seed (worklist entry `("r", refit_seed)`), so pass-1
  stays in the candidate pool as the route-congestion fallback. Kill switch:
  `candidate_search.parent_refit = false`.
- **Fix 3** — the per-candidate aspect sweep is live again (Fix 1 removed the floor that made all
  candidates identical); no code needed. The overhead sweep was NOT added: Fix 2 right-sizes each
  candidate directly, so sweeping `parent_seed_area_overhead` would be redundant double-tuning.

## 0. The problem, quantified

KC-AXHQTP ("four-channel relay board", 35 parts, 2,460 mm² of footprints, 4,053 mm² of leaf
content bboxes) shipped fab-ready at **213.3 × 101.0 mm = 21,539 mm² — 11.0 % utilization,
2.11:1 aspect**. The render shows input terminals+optos hugging the far-left edge, relays+output
terminals hugging the far-right edge, the ULN2003 driver alone in the dead center, and ~60 mm of
empty copper on either side of it.

Fleet (all web builds with a `util=` line, 2026-07-06..16, n=40): median ≈ 26 %, eight boards
< 20 %, worst 3.8 / 8.2 / 11.4 %. The leaf-side fixes moved the floor up; multi-leaf parents with
edge-pinned connectors are the remaining systematic sink.

## 1. Root-cause chain (verified, in causal order)

The sprawl is created by seed sizing, installed by edge pinning, unrecoverable by the optimizer,
invisible to the candidate search, and only warned about at the very end.

### RC-P1 — seed edge-span floor sums opposing edges' widths into one row (the arithmetic bug)

`cli/compose_subcircuits.py:373-395` (`_seed_outline_dimensions`): every left/right-edge-pinned
child appends its **width** to one `horizontal_widths` list, and
`seed_w = max(seed_w, sum(horizontal_widths) + spacing·(k+1))`.

This treats all left- AND right-pinned children as if they sat **side by side in a single
horizontal row**. But left/right-edge children stack **vertically** along their edge: they consume
*height*, not summed width. This design pins 9 of 10 leaves (J1,J6–J9 → left; J2–J5 → right, from
`component_zones`), so:

- area-driven base: √(4,053 × 2.5) ≈ **103 mm**
- edge-span floor: Σ(9 leaf widths) + spacing ≈ **218.4 mm**  ← wins

**Reproduced exactly** by calling the real `derive_attachment_constraints` +
`_seed_outline_dimensions` on the run's promoted artifacts: **seed = 218.4 × 115.2, invariant to
`area_overhead` 2.5 → 1.0** and to the candidate aspect sweep — every tuning knob is dead because
the floor overrides them. The shipped outline (213.3 × 101.0) is the seed minus edge margins.

The `sum_w*0.6` fallback floor (`compose_subcircuits.py:367`) has the same single-row axis
blindness (here it contributes 141 mm — also above the area basis).

### RC-P2 — blocks are locked to the *seed* edges for the whole solve; nothing ever re-fits

- `solve()` runs `_pin_edge_components` **first** (`placement_solver.py:198-204`): with
  `unlock_all_footprints` unset (this run), the 9 edge blocks are **locked** flush to the
  218-mm-wide seed canvas edges before any optimization.
- Force/SA/compaction only move the interior (the driver block wandered the huge middle; the
  phase-bbox trace of unlocked comps shows a 72×74 mm core mid-solve).
- `_restore_pinned_positions` (`placement_solver.py:658-667`, def at `:1481`) re-snaps pinned
  blocks to the recorded seed-edge targets at the end. **There is no pass that shrinks the canvas
  to content and re-flushes the pins to the smaller outline.** Board size ≡ seed size whenever any
  edge constraint exists.
- The final outline is fitted around placed geometry (`_derive_board_outline`), so it faithfully
  wraps the seed-sized placement: 213 × 101.

### RC-P3 — the parent has no compaction pass at all

The Phase-3 deterministic squeeze (`compact_toward_centroid`) is **leaf-only by explicit design**:
`placement_solver.py:690` gates it on `leaf_compaction_pass`, and "the parent/compose path never
sets the flag". (It also slides only *unlocked* parts, so it couldn't move the pinned blocks
anyway — Fix 2 is required regardless.)

### RC-P4 — the candidate search cannot recover

`compose_subcircuits.py:2611-2639` sweeps seed aspect 0.6 → 1.7 across the K=4 candidates, but the
RC-P1 floor overrides `seed_w` for every aspect → **all 4 candidates identical**
(outline 213.3×101, scores 11.40–11.47). The composite score *sees* the sprawl
(`area_utilization=11.6`, `packing_density=17.7/100`) but has no smaller candidate to prefer.
`parent_seed_area_overhead` (`:2628`) is read from config but nothing varies it — and per RC-P1 it
wouldn't matter here anyway.

### RC-P5 — the only guard is a warn-only line at the very end

`[build] 4/5 verify: … WARNING: board area utilization 11.0% is below 15%` — correct per the
fix-at-source policy (no masking gate on a fab-ready board), but nothing upstream acts on it.

**Note what is NOT wrong:** pinning screw terminals to board edges is correct intent
(accessibility/wiring). The fix is to make the board edges come to the connectors, never to unpin
them.

## 2. Resolution plan

Ordered by leverage; each independently shippable, all $0-verifiable via replay.

### Fix 1 — per-edge span accounting in `_seed_outline_dimensions` (~0.5 day)

Rewrite the constraint-floor block (`compose_subcircuits.py:373-395`) to accumulate **per edge**:

- left/right children: Σ **heights** per edge → `seed_h ≥ max(left_stack, right_stack) + spacing`;
  their widths contribute `max(w)` per edge → `seed_w ≥ maxw(left) + maxw(right) + spacing·3`
  (room for interior between opposing banks).
- top/bottom children: symmetric (Σ widths per edge floors `seed_w`; max heights floor `seed_h`).
- corner children: contribute max, not sum, to both axes.
- Replace the `sum_w*0.6` single-row fallback (`:362-368`) with the per-edge floors + the existing
  max-single-child floor; keep the area-driven base as the primary sizing.

Expected for KC-AXHQTP: seed ≈ 103 × 103 (area basis; edge floors ≈ 58 w / 92 h no longer bind)
→ board ≈ 100 × 100 → util ~24 % from this fix alone. **Guard:** pinned unit test — synthetic
5-left/4-right artifact set must yield `seed_w` from max-per-edge widths, `seed_h` covering the
tallest stack; plus top/bottom + corner cases. (No test exists today; only `test_shape_seed_cap.py`
touches this function's callers.)

### Fix 2 — post-solve outline re-fit + edge re-flush (the structural fix, ~1-2 days)

Even a correct seed is an *estimate*; the placement's natural size is only known after the solve.
Add a **second, right-sized solve pass** at the composer level (preferred over patching `solve()`
internals — the solver is load-bearing):

1. Pass 1: solve on the (Fix-1-sized) seed as today.
2. Measure the content need per axis from the pass-1 placement: interior bbox ∪ per-edge block
   stacks + spacing + edge margins.
3. If the measured need is < the seed by more than ~10 %, re-solve with seed = measured need
   (same rng seed, same cfg). `place_solve_ms ≈ 1 s`, so this is cheap even ×4 candidates.
4. Keep whichever pass stamps + routes better (existing candidate acceptance already does this
   ranking — the re-solve is just another candidate).

Expected for KC-AXHQTP: edge stacks ≈ 92 mm tall, opposing banks 18+35 mm wide + driver 15 mm
interior → seed₂ ≈ 78 × 95 → **board ≈ 7,400 mm², util ~33 %, aspect ~1.2** (vs 21,539 mm² today
— a ~2.9× area reduction).

### Fix 3 — make the search explore tightness again (~0.5 day, after 1+2)

- With the floors fixed, the existing per-candidate aspect sweep (RC-P4) becomes live again — no
  code needed, verify it in the replay.
- Optionally have the RoundScheduler (`cli/_round_scheduler.py`) sweep
  `parent_seed_area_overhead` down (2.5 → 1.5) across rounds with grow-on-route-fail as the escape
  hatch. Do NOT retune score weights yet (CMA-ES retune is a separate deferred item in the Phase-2
  plan; avoid double-tuning).

### Explicitly out of scope

- Any leaf-path change (`leaf_compaction_pass`, canvas derivation) — owned by Phases 0–4/6, done.
- A utilization hard gate — would mask, not fix; the warn line stays as the tripwire.
- Unpinning/re-routing connector edge intent.

## 3. Verification (all $0, deterministic replay)

1. Unit tests for `_seed_outline_dimensions` (Fix 1 guard, above).
2. **Replay KC-AXHQTP**: copy `projects/1/626/generated/FOUR_CHANNEL_RELAY` to a scratch dir,
   `cli_app replay --quality good --seed 0`; measure outline/util/DRC **inside that one replay**
   (never across runs). Expect: outline ≤ ~110×105 after Fix 1, ≤ ~85×100 after Fix 2; util ≥
   25–30 %; DRC 0 shorts / 0 unconnected preserved; rc0.
3. **Fleet spot-check**: same replay recipe on `1/618` (8.2 %), `1/614` (17.2 %), `1/622`
   (25.5 %) — median util must rise with **zero fab-ready regressions**.
4. Next self-eval batch: watch fab-ready rate (route congestion risk) + median util; apply the
   N-of-3 noise rule before believing per-board deltas.

## 4. Risks / watchpoints

- **Route congestion on tighter parents** → rc6/rc7 risk. Mitigations: candidate DRC acceptance
  already rejects unroutable placements; Fix 2's pass-1 result remains a fallback candidate;
  grow-on-fail in Fix 3. Keep `spacing_mm` untouched.
- **GND pour fragmentation** (the walled-off C1 class): smaller boards → less pour area. Watch
  `unconnected_nets` in the fleet replays; if GND strands rise, that is C1-v2's problem, not a
  reason to re-sprawl.
- **Breakout-stub clearances** on denser parents — the +10 µm guard (`9035faf`) is fresh; watch
  for near-miss clearance DRC in replays.
