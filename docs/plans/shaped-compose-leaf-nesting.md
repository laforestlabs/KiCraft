# Shaped-compose leaf nesting — implementation plan

**Status:** planned (design verified against code + run artifacts, 2026-07-13, HEAD `3b21eaa`).
**Goal:** a brief that requests a shaped outline with a hollow leaf (⌀60 LED ring, etc.) composes
with the companion leaf NESTED in the hollow, so the requested shape actually fits. Anchor
cases: projects `1/601` (KC-9G4YPT) and `1/600` (KC-CV4NE3) — 2 leaves, LED RING annulus
56.7×57.0 mm + MCU 21.7×20.2 mm, currently packed side-by-side → circumscribed 82.9 mm vs
requested 60 → rect fallback → outline-conformance gate fails the build (electrically the
boards are COMPLETE: 0 shorts / 0 unconnected, batch 20260713T051225Z).

## Root-cause chain — four independent blockers, all required

1. **Same-side overlap veto.** `can_overlap_sparse`
   (`kicraft/autoplacer/brain/subcircuit_composer.py:2045`) hard-forbids ANY bbox overlap
   between two leaves committing the same copper side. Ring (front SMT) + MCU (front SMT) are
   held apart by every consumer — solver repulsion (`placement_solver.py:2810`), overlap escape
   (`:3222,:3280`), courtyard pass (`:3444`), compose validation
   (`compose_subcircuits.py:1868-1916`) — even though the ring interior is empty FR4.
2. **No proposal mechanism.** Nothing moves a leaf into another's interior; the only
   co-location pass is opposite-layer stacking (`_stack_compatible_blocks`,
   `placement_solver.py:2592`).
3. **AABB circumscribe.** `_fit_requested_shape` (`kicraft/cli/_compose_validate.py:369`)
   circumscribes around the content AABB (`layout_editor/outline.py:309`); the annulus AABB's
   diagonal gives ⌀80.4 even when perfectly nested. The fit must test the true OCCUPIED
   geometry (ring occupied max radius ≈ 27.5 mm → ⌀55, fits ⌀60).
4. **Spurious edge-pin on the guest.** `1/601` carries `component_zones: {"J1": {"edge":
   "bottom"}}` **while the captured intent constraint says "no edge connectors"** (verified in
   state.json). `derive_attachment_constraints` (`subcircuit_composer.py:283-394`, strict=True)
   turns that into a hard edge zone, so the MCU leaf can never nest.

**Scope caveat (named):** self-eval run_29 is the per-LED-sheet variant (14 leaves, no annulus
leaf) — nesting cannot fix it; its at-source fix is synthesis-side (ring ArraySpec members must
share one sheet). File separately; do not count run_29 as a success criterion here.

## Key design decisions

- **Interior holes from geometry, not ArraySpec.** New frozen field
  `LeafBlockerSet.interior_free_rects` computed in `extract_leaf_blocker_set`
  (`subcircuit_composer.py:158,1981`) via a ~1 mm occupancy grid over the leaf outline:
  flood-fill from the boundary marks outside space; remaining empty regions are holes; keep the
  largest inscribed axis-aligned rect per hole above `nest_min_hole_side_mm`. The ring's
  interior companions (C3/C4 sit at r≈18.8 inside the annulus) are dodged automatically — an
  "inner radius" model from the ArraySpec would be wrong.
- **Allowance = containment, never partial overlap.** A containment branch in
  `can_overlap_sparse`: when the same-side veto would fire, allow iff the smaller leaf's
  occupied bbox (world frame) sits entirely inside ONE host `interior_free_rect` deflated by
  `nest_margin_mm`; host rotation must be cardinal (else False, conservative). Because
  `_blocker_pair_compatible` (`placement_utils.py:111`) is position-dependent, scorer,
  repulsion, escape and validation all accept a properly nested pair while any seam overlap
  stays rejected — the seam-short failure the veto exists for stays impossible.
- **Proposal = deterministic solver pass** `_nest_blocks_in_interior_holes` (Step 8.8, after
  `_stack_compatible_blocks` at `placement_solver.py:557-568`, before `_resolve_overlaps`).
  Hosts: subcircuit blocks with usable holes; guests: unlocked subcircuit blocks with no strict
  edge/corner zone that fit (current then allowed cardinal rotations). New
  `Component.block_nested_anchor` mirrors `block_stacked_anchor` (`types.py:187`) and is
  honored by `_resolve_overlaps` (`:3283-3298`) and `_resolve_courtyard_overlaps` (`:3444` —
  must exempt nested pairs, currently requires opposite backsides). Deterministic ordering
  (hole area desc, shared-net guests first, then area, then ref).
- **No new candidate type.** The K=4 search (`compose_subcircuits.py:2422`) runs the solver per
  seed; scoring picks nesting up free (`bbox_packing` 0.30 weight, sprawl penalty, and the
  `shape_fitted` −30 + lexicographic winner preference at `:2649-2664,:2798`).
- **Routing is favorable, keepouts are NOT protection.** Stamped leaf copper is locked
  pre-routes; the parent GND pour ties both leaves; DATA/+5V cross the ~7 mm inter-LED chords.
  FreeRouting 1.9.0 ignores DSN keepouts/boundary for wires (known gotcha) — the interior is
  protected only by real copper; an escaping wire is caught honestly by post-route validation.
- **Gates stay untouched.** The outline-conformance gate is the measurement, not the fix; the
  rect-fallback behavior stays.

## PR sequence (each lands green; $0 replay = rebuild 1/601 + 1/600 from state.json)

**PR-N1 — representation + allowance (no behavior change alone).**
`interior_free_rects` + grid computation; `nest_margin_mm`, `nest_min_hole_side_mm` in
DEFAULT_CONFIG; containment branch in `can_overlap_sparse`. Tests
(`tests/test_leaf_interior_nesting.py`, style of `test_blocker_aware_overlap.py`): annulus
fixture finds the hole and dodges interior decaps; nested pair → True; partial overlap /
side-by-side → False (seam regression pinned); non-cardinal host → False; determinism. Verify:
replays byte-behavior unchanged.

**PR-N2 — solver nest-proposal pass.** Pass + anchor field + skip exemptions; guests with
strict edge/corner zones excluded; gated on `parent_placement.leaf_nesting: "auto"` (fires only
when a non-rect outline shape is requested — rollout scoping, not masking). Tests in the
`test_stack_compatible_blocks.py` pattern incl. courtyard-pass keeps the nest intact. Verify:
1/601 unchanged (guest still edge-pinned — honest); nesting proven at fixture level.

**PR-N3 — occupied-geometry shape fit.** `circumscribe` (`layout_editor/outline.py:309`,
`shapes/__init__.py:170`) gains optional `content_rects`; `_fit_requested_shape` passes
occupied rects (component `physical_bbox()` + trace/via rects); `inscribed_rect_bound`
untouched (seed cap stays conservative). Tests: annulus → circle ≈ occupied diameter not AABB
diagonal; solid rect → identical to today. Verify: 1/601 logged circumscribed drops 82.9 →
~57-ish (gate outcome still rect-fallback until PR-N4 — honest intermediate).

**PR-N4 — edge-pin demotion wave (flips the genre).** In `_search_best_layout`: if a non-rect
shape+size was requested and ZERO candidates shape-fitted and a strict-edge-pinned guest would
fit a host hole, run ONE more K-wave with those `component_zones` entries dropped (same seeds,
deterministic), loudly logged + recorded in `winner_state.candidate_search.edge_pins_demoted`
and promote provenance. `_connector_stranded_refs` (`cli_app.py:3692`) learns to skip demoted
refs via the provenance record (else it would flag the nested J1 "stranded"). Companion
at-source fix: placement-stage commit normalization strips edge zones when intent constraints
say "no edge connectors" (LLM-run-only path; the compose-side wave is what replay verifies).
Verify ($0, headline): 1/601 + 1/600 → wave-2 nests MCU, shape fitted ~[60,60], routed parent,
`outline-shape CONFORMANT`, 0 shorts/0 unconnected, no connector_stranded; run each twice
(route noise); re-replay one green rectangular project for no-regression.

## Rollback / kill switches

`parent_placement.leaf_nesting: "off"`; `candidate_search.edge_demotion: false`;
`circumscribe(content_rects=None)` → AABB behavior. PRs are separable.

## Risks

- SA/late passes un-nesting the guest → anchor skips; if insufficient, lock guest post-nest
  (array-member precedent `array_placement.py:360`).
- FreeRouting wire escaping the circular boundary → post-route validation rejects the
  candidate (cost, not corruption); all copper sits ≥2.5 mm inside the circle.
- 475 mm star genre may or may not benefit (PR-N3 polygon branch + nesting); re-measure after
  landing, promise nothing.
