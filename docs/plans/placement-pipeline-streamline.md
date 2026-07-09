# Leaf placement pipeline — streamline plan

**Status:** Phase 0 done (instrument + baseline); Phases 1–4 proposed. **Date:** 2026-07-08.
**Decisions locked:** "orderly" = functional rows (passives grouped by anchor); packing
appetite = moderate tighten (small passive pitch, ICs/connectors keep routing clearance).
**Goal:** replace the accreted 26-step leaf finishing tail with a small, structure-first
pipeline that produces **tightly packed, orderly** leaf layouts — passives in neat
rows/grids at a uniform pitch and consistent orientation — instead of the organic-scatter
look force+SA currently yields. Deliver the visual/functional outcome the current patch
stack cannot, without regressing any hardened DRC guard and without touching parent
composition or routing.

**Relationship to `refactor-roadmap.md` principle #3** ("don't rewrite the place/route
core"): this is a *scoped* exception. We are not rewriting the force/SA optimizer, the
router, or parent compose. We are restructuring the **leaf-only** tidiness + legalization
subsystem, which the coupling audit (below) proved is cleanly separable. The optimizer core
is reused, not replaced.

---

## 1. Why the current pipeline is out of control

`PlacementSolver.solve()` (`autoplacer/brain/placement_solver.py:136-673`) is a 2-phase
optimizer (force-directed loop → SA refine) wrapped in a **~26-step deterministic tail**
(Steps 0.5 … 16 with sub-steps 8.5/8.7/9.1/9.2/9.3/9.5/13.5/15.5), followed by a **second**
tidiness+legality pass back in `solve_subcircuits._solve_one_round` (`:484-559`).

**Root structural flaw: there is no ordering invariant.** Each finishing pass moves parts,
which can violate what an earlier pass guaranteed, so a corrective pass was appended behind
it — which violates again, requiring re-snap / re-clamp / re-restore. The code says so
itself at Step 16 (`ee56574`): *"Steps 13-15 all move parts AFTER the last `_resolve_overlaps`,
so separated same-side pairs drift back into overlap and reach the router."*

Measured redundancy (line refs in `placement_solver.py` unless noted):

| Concern | # independent code paths | # invocations | Where |
| --- | --- | --- | --- |
| Overlap / courtyard resolution | 4 engines | 7+ | `_resolve_overlaps` ×6 (319, 514, 529, 538, 548, 596) + companion clamp (solve_subcircuits:551); `legalize_components` (554); `_resolve_courtyard_overlaps` (654); `leaf_compaction._pair_blocks` |
| Passive ordering | 2 systems (don't know about each other) | 2 | `_apply_orderedness` (Step 8.5, 493) + `apply_leaf_passive_ordering` (solve_subcircuits:504) — a leaf can be re-ordered twice |
| Alignment / tidiness | 4 systems | — | `_align_large_pairs` (Step 1.3) · `apply_alignment_repair` (424) · `_apply_orderedness` (493) · `_re_snap_aligned_pairs` ×5 (481, 487, 495, 517, 640) |
| Pad clamp / board clamp | — | ~10 | `_clamp_pads_to_board` ×7 (320, 561, 583, 600, 609, 641, 655) + `_clamp_to_board` ×2 (558, 582) + Step 12 re-clamp loop |
| Pinned-position restore | — | ~4 | `_restore_pinned_positions` ×2 (589, 597) + Step 15 `_pinned_targets` + companion clamp restore |
| Area shrink | 2 passes | 2 | `compact_toward_centroid` (Step 15.5) + `attempt_leaf_size_reduction` (post-solve) |

**Why the layouts look "random," specifically:**
1. **Orientation has zero signal.** `DEFAULT_PLACEMENT_WEIGHTS["rotation_score"] = 0.00`
   (`types.py:364`). Each passive's rotation is chosen independently to minimize *its own*
   routing cost (`_optimize_rotations`, `:1710`). Nothing rewards neighbors agreeing on an
   orientation → adjacent resistors settle at 0° vs 90° at random. That disagreement *is*
   the random look.
2. **The tidy pass is weak and clobbered.** `_apply_orderedness` blends only 30% toward the
   grid (`orderedness=0.20–0.35`), never touches rotation, and runs at Step 8.5 — *before*
   overlap resolution (9), compaction (15.5), and courtyard separation (16), all of which
   move parts again with no re-tidy. The neat rows are a suggestion later legalization erases.
3. **Packing is loose by tuning.** Leaf `placement_clearance_mm` is density-adaptive
   `max(0.5, 3·(1-density))` (`leaf_size_reduction.py:108-111`), CMA-ES-tuned for routability,
   not density. Compaction can only close slack down to that clearance.

---

## 2. Leaf/parent coupling — what is safe to restructure

Leaf and parent call the **same** `solve()`; divergence is 100% cfg-flag driven. Verified
separation:

- **Leaf-only (safe to restructure):** Step 8.5 orderedness (`orderedness` cfg), Step 15.5
  `compact_toward_centroid` (`leaf_compaction_pass`), the `prefer_legal` selection/repair
  branches (333-347, 550-554), everything in `local_solver_config`
  (`leaf_size_reduction.py:18-225`), and the leaf post-passes in `solve_subcircuits`
  (`:504`, `:510`, `:540`, `:1050`).
- **Parent-only (DO NOT TOUCH):** Step 8.7 block stacking (`_stack_compatible_blocks`),
  `_optimize_block_rotation`, `block_opposite_side` score, opposite-side attraction, all
  `kind=="subcircuit"` handling, and the compose-side geometry passes in
  `compose_subcircuits.py` (extremal slide 1490, edge-extremity 1503, courtyard re-resolve 1524).
- **Shared core (change only behind leaf gates):** clustering, force loop, SA refine, swap,
  grid snap, `_resolve_overlaps`, clamp/pad-containment, pinned restore, keep-out/keep-in,
  Step 16 courtyard. Note `smt_opposite_tht` is a *real-component* force, not parent-only.

**Design rule:** all new behavior is gated by a leaf cfg flag set only in
`local_solver_config`. The parent cfg path (`compose_subcircuits.py:1462-1469`) sets none of
them, so parent composition is byte-identical throughout.

---

## 3. Target architecture — structure-first, 5 stages, no re-do loops

Organizing principle: **the layout has a STRUCTURE — a placement tree of anchors and their
passive groups — and every stage operates on that structure as first-class, moving whole
groups rigidly so later stages preserve earlier invariants.** Tidiness stops being a weak
cosmetic afterthought and becomes the primary representation. There is exactly one pass per
concern, in dependency order, so nothing needs re-snapping.

```
  Stage 1  STRUCTURE   → build PlacementPlan (pins, anchors, passive groups, arrays)   [deterministic]
  Stage 2  COARSE      → force+SA arrange anchors + group super-nodes                  [optimizer, reduced graph]
  Stage 3  LOCAL       → lay each group's passives as rows/grids, uniform orientation  [deterministic] ← the tidy pass
  Stage 4  LEGALIZE    → one structure-preserving pass (overlaps/keepouts/pads/edge)   [deterministic] ← replaces the 26-step tail
  Stage 5  COMPACT     → slide whole groups toward centroid within legality            [deterministic]
                       → route (unchanged)
```

### Stage 1 — STRUCTURE (deterministic, once)
Build a `PlacementPlan`: edge/corner-pinned parts fixed to edges; anchors = ICs / regulators
/ connectors; **each passive assigned to exactly one anchor** by net-topology, formed into
ordered chains/rows; arrays detected. **Unifies** the three existing assignment algorithms
(`_apply_orderedness` IC-group binning, `apply_leaf_passive_ordering` topology chains, and the
scorer's `_score_topology_structure` re-derivation) into **one** `assign_passive_groups`.
Absorbs: `_align_large_pairs`, IC-group weighting, sibling grouping, `detect_alignment_groups`.

### Stage 2 — COARSE (optimizer, reduced graph)
Run the existing force+SA optimizer on a **reduced graph of anchors + group super-nodes**
(~5 nodes, not ~40 individual parts). Passives ride with their group centroid. Because SA
moves groups, not individual caps, it **cannot scramble intra-group order** — which is why
the current pipeline needs a post-SA re-tidy and this one does not. Keep objective terms:
`net_distance`, `crossover`, `bbox_packing` (and its early-exit gate `c4b7de2`).

### Stage 3 — LOCAL (deterministic, once) — the single tidy pass
For each group, place its passives as clean rows / grids:
- fixed **pitch** = part extent + a *moderate* clearance (this is the "moderate tighten":
  passives pack at this small pitch; anchors/connectors keep routing clearance);
- **uniform orientation** per line (horizontal row → 0°, vertical column → 90°) — this is
  the fix for random orientation; rotation becomes deterministic, not a scoring afterthought;
- placed on the anchor-relative side connectivity implies; snapped to grid.
Arrays already do exactly this (`array_placement`); extend the same discipline to all groups.
**Replaces:** `_apply_orderedness`, `apply_leaf_passive_ordering`, `apply_alignment_repair`,
all `_re_snap_aligned_pairs`.

### Stage 4 — LEGALIZE (one structure-preserving pass)
A single legalizer that resolves overlaps / keep-outs / courtyard / pad-inside-board, but
**moves whole groups/rows rigidly** and slides pinned parts *along* their edge — so it never
breaks intra-group alignment or pinned constraints, and converges in one pass because it
respects the structure instead of fighting it. The hardened DRC guards become **inputs/
constraints**, not sequential patches (see §4). **Absorbs:** Steps 9, 9.1, 9.2, 9.3, 9.5,
10, 11, 12, 13, 13.5, 14, 15, 16, and the post-solve `_repair_leaf_placement_legality` +
companion clamp.

### Stage 5 — COMPACT (structure-preserving)
Generalize `compact_toward_centroid` to slide **whole groups** toward the placed centroid as
far as legality allows — tightens without re-randomizing intra-group order. `leaf_size_reduction`
stays as an outer loop but should trigger far less often because Stages 3+5 pack tighter.

### Scoring simplification (leaf path)
Leaf scorer shrinks from ~13 terms to the ~5 Stage 2 needs: `net_distance`, `crossover`,
`bbox_packing`, `board_containment`, `courtyard_overlap`. Retire as SA drivers (keep as
diagnostics only): `rotation_score` (now deterministic in Stage 3), `topology_structure` +
`group_coherence` (now deterministic structure), `compactness` (already 0). `block_opposite_side`
stays parent-side. No parent weights change.

---

## 4. Landmine → constraint mapping (do-not-regress)

Every hardened guard survives as a **constraint** in the new structure, not a deleted pass:

| Guard (commit) | Defends against | Where it lives in the new pipeline |
| --- | --- | --- |
| Step 16 courtyard sep (`ee56574`/`062dfbb`) | `courtyards_overlap` (#1 blocker, 10 designs) | Stage 4 overlap constraint — last-by-construction; the "runs too early" bug is gone because there is one pass |
| Step 15 pinned keep-out slide (`87b7f87`) | pinned connector in neighbour's antenna keep-out | Stage 4: pinned parts slide *along-edge* within the one legalizer |
| Step 9.2 antenna keep-out (`b21e2b3`) | parts in ESP32 RF near-field | Stage 4: keep-out rects are inputs |
| companion clamp (`38be624`) | `copper_edge_clearance` (9/9 corpus) | Stage 1: passives on an edge-connector anchor are assigned inboard of its pad face *by construction* |
| Step 9.3 array-grid escape (`5bac2c7`) | stray part trapped in locked LED grid | Stage 1 locks arrays; Stage 4 moves the stray part, not the grid |
| bbox_packing early-exit (`c4b7de2`) | sprawl rejected by parent outline-cap | Stage 2 keeps the gate |
| Step 9.1 keep-in (`b3bb5d5`) | parent mounting-hole/keep-in | Stage 4 keep-in rects are inputs |
| adaptive legality retry (`45e54e0`) | `illegal_unrepaired_leaf_placement` on dense leaves | Stage 4 convergence guarantee should remove the need; keep a retry backstop |
| alignment repair (`20af4b3`) | SA scrambling batteries/LED rows/headers | Stage 3 places these deterministically; SA never sees individual members |
| block stacking (`fbce780`) | dual-layer parent wasting >50% area | untouched — parent-only |

---

## 5. Landing strategy — incremental, gated, replay-verified

Load-bearing code. Rules from memory: *surgical fixes only*, *run the suite before shipping*,
*measure leaf+parent in ONE replay*, *no fallback previews*, *fix at source, no masking gates*.
Each phase ships independently behind a leaf flag, verified by `kicraft replay` ($0, no LLM)
across the self-eval corpus + a self-eval run, with DRC parity (0 shorts / 0 unconnected /
no new courtyard overlaps) as the hard gate and the tidiness metric as the win condition.

- **Phase 0 — Instrument & baseline. ✅ DONE (2026-07-08).** Added
  `autoplacer/brain/leaf_tidiness.py` (pure metric: orientation-consensus %, row-alignment
  residual mm, packing fill %), `scripts/leaf_tidiness_report.py` (reads any tree of
  `solved_layout.json` — the leaf's own record artifact — and rolls up per-design + corpus),
  and `tests/test_leaf_tidiness.py` (12 tests, green). **No live-path change** — the report
  recomputes from frozen geometry, so it measures baseline now and Phase-1 A/B later by
  re-running on the new `solved_layout.json`. Live-embed of the metric into the artifact is
  deferred to Phase 1 (gated, alongside real changes). Baseline over yesterday's full 34-brief
  self-eval batch (`logs/self_eval/20260707T193651Z`, 111 leaves / 23 designs, 46 leaves with
  a passive group):

  | Metric | Corpus mean | Meaning | Target |
  | --- | --- | --- | --- |
  | orientation consensus (grouped) | **73.1%** | 27% of grouped passives point against their group's dominant axis — the "random orientation" look | → ~100% (Stage 3 sets it deterministically) |
  | orientation consensus (leaf) | **81.4%** | same, over all passives in a leaf | → ~95%+ |
  | alignment residual | **4.02 mm** | off-axis scatter within a "row" (0 = straight row) | → < ~0.5 mm |
  | packing fill | **47.9%** | ~half the placement bbox is empty copper | → higher (Stage 3 pitch + Stage 5 group compaction) |

  Worst offenders to watch as Phase-1 canaries: residual — `TWO_WAY_CROSSOVER` 9.26mm,
  `A_MINIMAL_STM32F103` 7.22mm; orientation — `1A_LED_DRIVER` 50%, `A_CHAMFERED_CORNER` 53.8%,
  `AN_R_2R` 58.3%; fill — `TPS5430_BUCK_CONVERTER` 26.2%, `A_MINIMAL_STM32F103` 26.8%,
  `AN_ESP32_S3` 28.4%. Re-run: `.venv/bin/python scripts/leaf_tidiness_report.py [CORPUS_DIR]`.
- **Phase 1 — Stage 3 (structured local layout). ✅ CODE DONE (2026-07-08), replay-verifying.**
  New `autoplacer/brain/leaf_structured_layout.py`: `apply_structured_local_layout` lays each
  functional group as a straight row/column at a courtyard-legal fixed pitch with uniform
  orientation (group dominant axis, minimal-churn rotate). **Atomic per group** with a bounded
  perpendicular-dominant shift search — commits a whole row only if every member is legal
  (on-board, clear of every non-member courtyard, out of keep-outs), else leaves the group as
  the solver had it. Grouping unified into `leaf_tidiness.assign_passive_groups` (Stage-1 down
  payment), shared by metric + packer. Wired as **Step 15.7** in `solve()` (after compaction,
  before the final courtyard pass); legacy Step 8.5 `orderedness` + post-solve
  `apply_leaf_passive_ordering` disabled when the flag is on. Leaf-only (flag set in
  `local_solver_config`, default on); parent path untouched. Tests: `test_leaf_structured_layout.py`
  (6) + `test_leaf_tidiness.py` (12) green; full leaf/placement suite 156 pass / 1 pre-existing
  env failure.

  **Frozen-corpus A/B** (`scripts/phase1_packer_ab.py`, packer applied to the 111 frozen leaves —
  faithful since the packer is a pure post-placement transform):

  | Metric | before | after | Δ |
  | --- | --- | --- | --- |
  | orientation consensus (grouped) | 73.1% | **89.6%** | +16.5 |
  | orientation consensus (leaf) | 81.4% | **90.6%** | +9.2 |
  | alignment residual | 4.02 mm | **1.32 mm** | −2.71 |
  | packing fill | 47.9% | **49.9%** | +2.0 |

  43/56 groups placed, **0 new courtyard overlaps, 0 new off-board** — under an artificial tight
  board bound, so real solve() (true outline) should match or beat this.

  **Real-pipeline confirmation (run_09 stm32-min via `solve_subcircuits.py`, seed 0):** the
  packer fires (`Structured layout: 1 row(s), 3 passive(s) aligned`) and its output **survives
  the post-solve legality tail** — the persisted POWER leaf came out 100% orientation consensus /
  0.0 mm residual (4 passives). This proves the late-placement slot works: tidiness is not
  clobbered by the downstream repair. **Safe by construction:** atomic + legality-checked, so it
  cannot add a courtyard overlap or off-board part regardless of design.

  **Routing-parity sweep (N-of-3 medians, `scripts/phase1_routing_parity.py`) caught a REAL
  regression — now guarded.** First sweep: `HIGH_SIDE_SWITCH` 0→0 unconnected (ok), but
  `MINIMAL_RP2040` (dense MCU) **19→24** median unconnected (on-seed-1 spiked to 41). Root cause:
  SA places passives for routability; a pure-geometry tidy row overrides that and, on a dense
  leaf, stretches/walls signal nets. **Fix — routability guard:** the packer now commits a tidy
  row only if it grows the group's signal-net **HPWL** (high-fanout power/GND nets excluded) by
  ≤ `leaf_structured_max_hpwl_increase` (default 15%); else it keeps the solver's placement. Costs
  almost nothing on the corpus (frozen A/B 88.6% orient / 1.49 mm resid vs 89.6% / 1.32 mm without
  the guard; 41 vs 43 groups placed), while backing off exactly where geometry fought the router.
  **HPWL guard did NOT fix it.** RP2040 re-verified 22→**26** (still every ON seed worse than every
  OFF seed — real). Root cause diagnosed: the RP2040 MCU leaf is dense (12 comps, **43 nets**, 3.6
  nets/comp) vs HIGH_SIDE (6 nets, 0.55) — and its passives connect almost entirely to power/GND
  (only 3 signal nets), which the HPWL guard *excludes*, so it never engages. **The harm is
  CONGESTION, not wirelength:** tidy passive rows block the MCU's 43-net routing fabric; a
  pre-route wirelength proxy can't see that.

  **Deeper cause (architectural):** the packer runs AFTER a full SA solve that already placed every
  passive individually for routability; overriding that with pure geometry discards SA's work,
  which only bites on congested leaves. The clean fix is Stage 2/Phase 3 — SA arranges GROUPS, the
  packer owns intra-group — so structure and routability never compete. Until then Phase 1's safe
  scope is **low-congestion leaves only** (a net-density gate: skip when nets/comp exceeds ~1.8).
  **Lesson:** legality-safe ≠ routability-safe, and routability harm on dense leaves is congestion,
  not a cheap pre-route proxy. Flag keeps it fully reversible.
- **Phase 2 — Stage 4 (unified structure-preserving legalizer).** Collapse the 4-engine /
  7-invocation overlap+clamp+restore tail into one pass. Highest cleanup ROI, highest risk.
  Keep the old tail behind a flag for A/B. *Gate:* per-design DRC parity across the entire
  corpus via replay; identical or better courtyard/keep-out/edge outcomes.
- **Phase 3 — Stage 2 (reduced-graph coarse) + Stage 5 (group compaction).** The optimizer
  change. *Gate:* board area, routability, and solve wall-clock all ≥ baseline.
- **Phase 4 — Prune.** Delete superseded passes, dead scoring terms, and orphan cfg knobs;
  CMA-ES re-tune the reduced leaf weight vector + pitch/clearance. *Gate:* full self-eval.

**Rollback:** every phase is a flag; flip off to restore prior behavior byte-for-byte.

---

## 5b. Reframe (2026-07-08) — group-as-unit, not tidy-as-post-pass

Phase 1 built tidiness as a **post-pass on top of individual-passive SA**, and the
routing-parity sweep exposed why that's the wrong shape: SA optimizes each passive's position
for routability, the packer overrides it with geometry, and on dense leaves that fights the
router (RP2040 19→24 unconnected). Guards/gates to referee the fight are *more of the same
patch disease* — the thing this plan exists to remove.

**The generalizable fix is a representation change, not another pass:** the free variable
becomes the **rigid, internally-tidy group**, not the individual component. A functional group
(anchor + its passives) is *always* a tidy row/grid by construction; SA moves/rotates/reflows
**groups**, so every state it visits is already tidy and routability is optimized *within* the
tidy space. Tidiness and routability stop competing — there is no untidy intermediate to clean
up, hence no packer/orderedness/passive-ordering/re-snap/gate. This deletes the competing
systems rather than arbitrating between them, and generalizes with zero per-leaf conditionals
(dense vs sparse handled identically — SA just finds fewer routable tidy arrangements on dense
leaves, never an untidy one). The Phase-1 packer collapses into one primitive: "render a
group's internal tidy layout," called when SA places a group. This supersedes the Stage-2/3
split above (coarse-then-local could still fight); the group is a first-class SA atom throughout.

## 5c. Visual diagnostic framework (built 2026-07-08)

Proof requires the eye, not just the metric (a metric can miss what looks wrong). Added
`autoplacer/brain/leaf_layout_svg.py` (`render_leaf_svg`) + `scripts/leaf_layout_viz.py`:
annotated per-leaf SVG + HTML gallery showing courtyards/pads tinted by functional group, each
group's ideal axis with per-member residual ticks, **misoriented passives flagged** (warning),
courtyard overlaps/off-board (critical), and a **per-net MST ratsnest** so routing congestion is
visible — the picture that makes the tidiness-vs-routability tension legible. Uses the same
`assign_passive_groups`/`leaf_tidiness` as the numbers (picture and metric can't disagree) and
the data-viz skill's validated palette. This is the before/after proof harness for the
group-as-unit work: render the corpus pre- and post-redesign alongside the metric deltas.

## 5d. Corrected approach (2026-07-08) — soft tidiness, routing wins

Two *hard*-tidiness approaches were built and both **regressed dense-leaf routing** (verified by
N-of-3 routing sweeps on the RP2040 43-net MCU leaf):
- **Packer post-pass** (Phase 1): classic 19 unconnected → 24. HPWL guard didn't help (harm is
  congestion, not wirelength).
- **Group-as-unit / rigid groups** (§5b): classic 23 → 27. A rigid 9-cap row around a 43-net MCU
  over-constrains — the group-SA had 1–2 movable anchors, no freedom to thread the fabric.

**Finding: crisp tidiness and best routability genuinely conflict on a congested leaf — the tidy
arrangement space does not contain the best-routable layout. It's irreducible, so *any* method
that imposes crisp tidiness as a constraint/post-pass/rigid-group pays for it in dense routing.**

**The generalizable fix (user-endorsed): tidiness as a SOFT term in the placement objective**,
co-optimized by the existing full-DOF SA — not a constraint. Implemented: `PlacementScore.tidiness`
(orientation consensus + row-alignment of each functional group, 0–100), scored by
`PlacementScorer._score_tidiness` (grouping memoized net-based; short-circuits + zero cost when
unweighted so default/parent scoring is byte-identical). Weight `psw_tidiness` = 0.15 for leaves
(subordinate to routing's net_distance 0.20 + crossover 0.17), set in `local_solver_config`.

Why this is right *and* simplifying:
- **Sparse leaves** → routing score saturates → the tidiness term dominates → crisp rows, free.
- **Dense leaves** → routing terms dominate → SA sacrifices tidiness exactly where congestion
  demands → **no hard routing regression**, graceful degradation (how good human layouts look).
- **One objective, one optimizer, zero conditionals, no gate, no rigid groups, no post-pass** —
  and it lets us **delete** orderedness / passive-ordering / packer / group-rigid (the LOC win).
- The packer + group-rigid code stays behind flags (both **default OFF**) for A/B until soft
  tidiness is validated (routing parity on RP2040 + tidiness gain on sparse leaves), then deleted.

## 6. Open decisions (resolve before Phase 1)

1. **"Orderly" definition** — strict global lattice (every passive on one board-wide grid) vs
   **functional rows** (passives grouped by anchor, aligned within a group, groups may differ).
   *Recommendation: functional rows* — matches how the board actually connects and how humans
   read it; a global lattice fights connectivity.
2. **Scope gate** — all leaves, or only passive-heavy (`≥4 passives`, the existing
   `apply_leaf_passive_ordering` gate)? *Recommendation: reuse the existing gate* so trivial
   leaves are untouched.
3. **Pitch target** — "moderate tighten" = passive pitch `extent + ~0.6mm`, anchors keep
   routing clearance. Confirm the 0.6mm (vs a density-adaptive pitch) after Phase 1 replay.
4. **Big-bang vs incremental** — *Recommendation: incremental per §5.* A single rewrite of
   `solve()` risks all 10 landmines at once with no A/B.
