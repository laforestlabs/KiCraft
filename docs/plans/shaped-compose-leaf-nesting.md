> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# Shaped-compose leaf nesting — implementation plan

**Status:** PR-N1..N4 LANDED 2026-07-13 (`a546274`, `56f48b3`, `2567701`, `21389c4`).
**PR-N5 LANDED 2026-07-15** in three commits — r1+r2 `6069d22`, p1+p2 `7b7b52d`, plus a
fifth leg r3 `b78851e` discovered during verification (non-cardinal member courtyard AABBs
were eating the hole the same way trace AABBs did; exact rotated-body rasterization +
0.5 mm hole grid). Verification surfaced two more N5-class fixes, both landed: the nest LANDING centres the
guest's occupied bbox, not its content pos (`2c8084f` — fatal at tight slack), and the
shape-fit content measures exact rotated bodies + subdivided traces (`ed2cb3e` — the fourth
consumer of the rotated-AABB lie; the fit read ⌀63.1 for content that truly fits ⌀58).

**Ring interior discipline is measured DONE** on a rebuilt 1/601: innermost copper r=20.1
(was 14.9), hole 12.9×16.0 → **23.96×23.87** (the predicted ~24), band decaps +
closed-loop +5V bus stamp clean. **THE GENRE FLIP IS DEMONSTRATED END-TO-END
(2026-07-15):** a zone-stripped 1/601 rebuild (equivalent to a live post-N4 run — synthesis
no longer emits the contradicting `J1 {edge: bottom}` zone) produced **rc0: MCU leaf nested
inside the ring, routed 0 shorts / 0 unconnected, circular Edge.Cuts ⌀61.0 delivered vs
⌀60 requested → outline-shape CONFORMANT, independent kicad-cli DRC 0/0, fab package
exported** (scratchpad replay_601z5). Regressions: KC-HN59RJ ring replay DRC 0/0; 1/606
rectangular rc0 fab.

**Remaining flakiness, with its owner:** the guest MCU leaf's solved size varies
20.5–23.8 mm across otherwise-identical rebuilds (~30% utilization; the fit boundary is
~21.9), so the flip fires on the rounds whose leaf lands tight — 2 of 5 zone-stripped
rebuilds nested, 1 reached rc0. Owning follow-up: leaf grid-assignment tuning (the known
placement-streamline open item) and/or a guest-leaf re-solve after edge-pin demotion. NOT
further margin cuts — the remaining stack (0.5 extraction + 1.0 margin + 1.0 standoff) is
the honest floor. The compose log prints every near-miss explicitly
(`nest-demotion fit check failed ... short 0.00/0.52 mm`). Frozen pre-N4 replays (stock
1/601 state.json) still carry the contradicting zone in `bom.component_zones`, which also
pins J1 inside the guest leaf's own solve and inflates it. Separate NEW finding: 1/600
decomposes into THREE leaves and its 13×13 POWER leaf cannot fit inside ⌀60 once the hole
is taken (one guest per hole; 18.2+12.7 > 24.4 rules out co-nesting) — its nest+route
works (0 shorts / 0 unconnected with the MCU inside the ring), so the ⌀60 request is
honestly unsatisfiable at that decomposition; owner = synthesis sheet-merge for small
power sheets on shaped briefs.

Implementation corrections vs the draft below, for the record: (a) the r2 "morphological
close" is actually a reachability sweep — a true closing fails to seal a ring's DIAGONAL
gaps (the structuring element nestles into the gap mouth from outside and erosion destroys
the bridge); dilate, flood the centre positions, dilate the reachable set back. (b) p1
decaps are RADIAL (power pad inward on the chord sagitta), not tangential — a tangential
cap puts its GND pad in the bus chord's path and the foreign-pad guard drops the tie.

**Blocker 5 (OPEN, owns the genre now): ring-leaf interior discipline.** Scoped 2026-07-13
by probing the accepted 1/601 leaf artifact — the original "two ~35 mm chords" theory was
only part of the story; the measured decomposition is FOUR independent contributors (see
the PR-N5 section at the bottom for the full design). DSN keepouts still CANNOT protect the
interior (FreeRouting 1.9.0 ignores keepouts/boundary for wires — the known gotcha), so the
physical legs speak the only language FR respects: locked pre-routed copper.
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

## PR-N5 — ring interior discipline (blocker 5; flips the genre)

### Measured root cause (1/601 accepted ring leaf, probed 2026-07-13)

Today's hole is **12.9×16.0 mm**; the MCU guest needs **24.4×22.0** (21.4×19.0 occupied —
essentially its whole 21.5×19.3 outline, nothing to reclaim on the guest side — plus 2×1.5
`nest_margin_mm`). Four independent contributors, each measured by recomputing
`compute_interior_free_rects` on the real blocker set with that contributor removed:

1. **Trace-AABB bloat (representation).** `_trace_blocker_rects`
   (`subcircuit_composer.py:1635`) makes ONE axis-aligned bbox per segment; a 45° chord in
   the annulus band becomes a fat 8.9×8.9 square whose corner reaches r=12.8 while the real
   copper stays at r≥20. Twelve rotated-LED hops tile these squares over the interior.
2. **Real interior +5V routing.** FreeRouting feeds the interior decaps with arcs dipping to
   r=15.2–18.0. (Zero DATA nets cross — the chain chords are already band-disciplined; the
   original "35 mm data chords" theory was wrong.)
3. **Interior decaps by design.** `_place_companion_decaps` ring branch
   (`array_placement.py:529-562`) drops C3/C4 radially INWARD at r≈17.8 — its comment "the
   ring interior is otherwise empty" is exactly the premise nesting invalidates.
4. **Closing-band standoff (hole rule).** The hole excludes `nest_min_hole_side_mm/2` = 4 mm
   around ALL copper — a gap-sealing artifact doing double duty as clearance. Electrically
   ~0.5 mm pad-margin + ~1 mm standoff is already generous vs the 0.2 mm rule.

No subset suffices (measured): tight traces alone → 14.9×16.0; + interior routing cleared →
14.9×16.0; + decaps out → 17.9×14.0; + decoupled standoff 1.5 → 23.9×20.0 (still short);
**all four at standoff 1.0 / margin 1.0 → ~25.9×22.0 vs needed 23.4×21.0 — fits with real
slack.** Every leg is load-bearing.

The ring's pad geometry makes the physical legs deterministic and rotationally symmetric
(`_orient_ring` rotates every member with the circle): on 1/601, **+5V pads all at r=21.1
(inner corner), DATA at r=23.4–24.9 (mid-band), GND all at r=26.9 (outer corner)**. A +5V
bus is 12 identical ~10.9 mm pad-to-pad chords whose sagitta dips only to r=20.4. Leaf zones
are NOT stamped into the parent (`_stamp_subcircuit_subprocess.py` payload carries only
components/traces/vias/silk), so the stamped +5V *tracks* are the distribution that matters
at parent scope; GND is owned by the parent pour.

### Legs (one PR, two commits: representation first, physical second)

**N5-r1 — tight trace rects for the hole computation only.** In `extract_leaf_blocker_set`
(`subcircuit_composer.py:2185`), compute the holes from a copy of the blocker set whose
trace rects are subdivided into ≤`_NEST_GRID_MM` pieces (each piece's bbox still a copper
superset). The published blocker set keeps today's conservative one-rect-per-segment AABBs,
so `can_overlap_sparse`, repulsion, escape and validation are untouched — zero seam-regression
surface.

**N5-r2 — decouple sealing from standoff in `compute_interior_free_rects`
(`subcircuit_composer.py:2010`).** Keep the morphological close at `min_side_mm/2` for
TOPOLOGY (outside/interior labeling — inter-member gaps stay sealed, open bays stay open),
but exclude from the hole only a dilation by new cfg `nest_hole_standoff_mm` (default 1.0)
around occupied cells. Drop `nest_margin_mm` 1.5 → 1.0 (`config.py:379`). Effective real
copper-to-copper spacing stays ≥ ~3 mm (0.5 pad-margin each side + 1.0 + 1.0). Update the
N1 tests that pin the closing-band-excluded semantics — deliberate rule change.

**N5-p1 — ring decaps into the band, not the interior.** Rework the ring branch of
`_place_companion_decaps` (`array_placement.py:529`): place decap *k* at the gap-midpoint
angle between members k and k+1, tangentially oriented at the +5V-pad radius (≈21 on 1/601),
+5V pad toward the bus, GND pad tied by a `via_at_end` stub to the B.Cu pour. Existing
perimeter fallback stays as the too-tight escape hatch. This is canonical ring construction
(a real LED-ring board keeps the middle clear — often physically cut out), not a nesting
hack; it applies to every ring. Kill switch `array_ring_band_decaps` (default on).

**N5-p2 — deterministic +5V ring bus (`array_router.py`).** New `array_ring_power_specs`:
for every fully-present `pattern=="ring"` array, emit `BreakoutSpec` ties chaining the
members' +5V pads around the circle (member→member chord, or member→decap→member where a
band decap occupies the gap) — a CLOSED loop, so one or two guard-dropped ties cannot
disconnect the bus. Compose into `_breakout_specs` in `leaf_routing.py` (:638-652) as a
SEPARATE list from `_arr_specs` so the array stamp gate (:683-711) keeps measuring in-row
DATA hops only. The existing `add_breakout_stubs` foreign-pad/copper guards + the
no-silent-handoff log stay the honesty layer. Kill switch `array_ring_power_bus` (default
on). GND ties are NOT stamped initially — LED GND pads sit on the outer corner (r=26.9)
where FR's shortest paths never enter the interior, and the pours own GND; if replay still
shows interior GND dips, extend the same generator with outer-arc GND ties (one-line
decision point, recorded in the verify notes).

### Contingency (only if replay still shows interior crossings)

**N5-e — temporary REAL obstacle:** netless dual-layer pad footprint parked at the ring
centre during the leaf route, removed after `import_routed_copper`, before acceptance
(array-leaf-purity precedent). Not in the initial PR — pre-routing should leave FR nothing
to cross with; add only on measured evidence.

### Predicted outcome

+5V bus innermost copper ≈ r 20.4 → interior free disc r ≈ 18.9 → hole ≈ 24×26+ vs needed
23.4×21.0. Then the already-landed machinery takes over: N1 containment allows the pair, N2
Step 8.8 nests the MCU, N4 wave-2 demotes the contradicting `J1 {edge: bottom}` pin, N3
occupied-geometry fit measures ~⌀55 vs requested ⌀60 → CONFORMANT.

### Tests

- `tests/test_leaf_interior_nesting.py`: hole-with-standoff semantics (annulus fixture hole
  GROWS vs closing-band rule; corridor sealing still holds; seam regression stays pinned);
  subdivided-trace tightness (diagonal chord no longer eats the hole; straight segment
  identical).
- `array_router` ring-bus test (style of the daisy-chain tests): 12-member ring fixture →
  12 closed-loop +5V ties, chain order deterministic, decap-in-gap reroutes through it;
  power specs excluded from the stamp-gate stats.
- `array_placement` band-decap test: gap-midpoint placement, rotation-with-ring, perimeter
  fallback when the gap is too tight.

### Verify ($0, headline = the genre flip)

1. Rebuild 1/601 + 1/600 from state.json, twice each (route noise): ring leaf accepted with
   interior probe showing hole ≥ 23.4×21.0 and no track with true min-distance < ~19 from
   ring centre; N2 nests the MCU, N4 demotes J1, shape fitted ~[60,60], parent routed,
   `outline-shape CONFORMANT`, 0 shorts / 0 unconnected, no connector_stranded.
2. Regression: replay the KC-HN59RJ ring project (run_33 — band decaps + bus change hit it;
   must stay ERC 0 / DRC 0/0) and one green rectangular project (no array path touched).
3. Probe scripts from scoping live in the session scratchpad; re-derive from this section's
   numbers if needed (they are one-file pcbnew/composer probes).
