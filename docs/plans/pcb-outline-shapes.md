# Plan: arbitrary PCB outline shapes (incl. compound shapes like a snowman)

**Status:** proposed (2026-06-30, rev 2)
**Goal:** when a brief asks for a non-rectangular board — from "round coaster" to
"snowman ornament" — produce a fabricable board whose `Edge.Cuts` matches the requested
shape, with the circuit built *inside* that shape. Plus a self-eval category that proves
it.

**Committed scope (2026-06-30):** v1 ships **Tiers 1 → 2 → 3** — parametric shapes, then
single arbitrary polygons (inscribed-rect placement), then **compound/multi-region shapes
like the snowman** (leaf→region mapping). Tier 4 (full polygon-containment placement) is
deferred. **Geometry engine: Shapely (GEOS).** Each tier ships independently and gets its
own self-eval coverage.

**Progress** (work branch `feat/outline-shapes-phase3b`; 1–3b also on
`feat/outline-shapes-phase1` + `main` via `c01d3b2`):
- ✅ **Phase 1 — capture** (`3fd1a33`, `b17c752`). Shape captured from the brief into
  `IntentSlot.form_factor`.
- ✅ **Phase 2 — flow** (`628c0e6`). `form_factor` → `autoplacer.json` `board_outline` →
  `ParentCompositionState.requested_shape`.
- ✅ **Phase 3a — parametric circumscribe** (`bbb468b`). Tier 1 end-to-end: a brief-
  requested circle / rounded_rect / chamfered_rect is grown around the placed content at
  compose (`outline.circumscribe` + `_compose_validate._fit_requested_shape`, wired into
  `_compose_stamp`) and flows through the existing shape-aware stamp + validate + zone-clip
  path. No placement changes, no new dep.
- ✅ **Phase 3b — Shapely named/compound shapes** (`9d8f10d` foundation, `675e85a`
  integration). `kicraft/shapes/` library (hexagon…snowman, snowman = boolean union of
  circles) + polygon `circumscribe()`. Wired into compose via a `fitted_polygon` state
  channel kept OFF the JS-mirrored `OutlineSpec`: `_fit_requested_shape` dispatches
  parametric→`manual_outline` vs named→`fitted_polygon`; `_validate_parent_geometry`
  duck-types the containment checker (`OutlineSpec` or shapely `PolygonOutline`);
  `_compose_stamp` stamps the ring straight to `Edge.Cuts`. **Verification gap:** a full
  place+route build emitting an actual snowman board hasn't been run yet — only geometry/
  wiring units. Close it in Phase 6 (self-eval).

**Next:** Phase 4 (compound/Tier-3 region mapping — leaf→blob anchors + necks) and/or
Phase 6 (self-eval category) to lock 3b in with a real shaped build. Edge-connector
handling (Phase 5) for shapes without a flat run.

---

## What already works (verified)

Non-rectangular outlines are already first-class in the **manual layout editor** path:

| Capability | Where | Evidence |
| --- | --- | --- |
| Shape model `rect/rounded_rect/circle/chamfered_rect` w/ `polyline()`, analytic containment, mounting holes | `kicraft/layout_editor/outline.py` (`OutlineSpec`) | full read |
| Edge.Cuts stamping from an arbitrary polyline (leaves + parent) | `_stamp_subcircuit_subprocess.py:109-131`, `_parent_stamp_subprocess.py:70-86`, `hardware/adapter.py:952-1025` | polyline branch |
| Parent Edge.Cuts from `OutlineSpec.polyline()` | `cli/_compose_stamp.py:148-161` | reads `state.manual_outline` |
| Shape-aware geometry validation (outside body/pad/trace/via) | `cli/_compose_validate.py:209-230` | `contains_rect/contains_point` |
| **Routing honors the polygon boundary** (DSN carries the polyline) | `freerouting_runner.py:506` | `tests/test_outline_shape_stamping.py::test_circle_boundary_reaches_dsn` |
| **Copper pour is fine over any outline** | KiCad `ZONE_FILLER` | clips fill to `Edge.Cuts` + applies copper-to-edge clearance automatically |
| Leaf→sub-region placement anchor (`zone`) | `subcircuit_composer.py:194,554`; `models.py:578` | existing `zone` constraint target |

**Correction from rev 1:** pouring a rectangular zone over a shaped board is *not* a
fab problem — KiCad clips the zone fill to the board outline and pulls back by the edge
clearance. The "shape-aware pour" phase is **dropped**.

So the drawing/routing/pour layers already work for any polygon. The unsolved problems are
all about **getting a shape from the brief** and **placing the circuit inside a
non-rectangular (and possibly non-convex) region.**

---

## The real gaps

1. **Capture** — synthesis has no shape concept. `manual_outline` only ever comes from the
   editor (`compose_subcircuits.py:1811`). No brief → shape path. `PlacementBoard`
   (`models.py:589`) has no shape field.
2. **Representation of arbitrary/compound shapes** — `OutlineSpec` only knows 4 parametric
   shapes. A snowman is a *union of primitives* → a non-convex polygon. Need (a) a way to
   compose primitives into one polygon and (b) to carry an arbitrary point loop.
3. **Containment correctness** — `OutlineSpec.contains_rect` is the convex 4-corner test
   (`outline.py:259`). It is **wrong** for non-convex shapes (a rect spanning a snowman's
   neck can have all corners inside but its middle outside). Validation must use true
   polygon-in-polygon for arbitrary shapes.
4. **Build the board *inside* the region (placement)** — the solver only knows a
   rectangular AABB. To place a circuit inside a heart/star/snowman, we need a
   region-aware placement strategy. **This is the crux of the feature.**
5. **Edge features on arbitrary outlines** — edge connectors need a locally-flat edge run;
   mounting holes want the extremities. Minor, mostly reuse.

---

## Architecture: shape as a region, board built inside it

Three layers. Layers 1 and 3 are mostly new-but-easy; Layer 2 is the design's center.

### Layer 1 — represent & compose the shape

- **Generalize `OutlineSpec`** with `shape="polygon"`, carrying a closed CW point loop
  (`points`) plus optional interior cutouts (`holes`). The parametric shapes stay as the
  fast analytic path; `polygon` is the general path.
- **Composition = a tiny shape program (CSG DSL)** the LLM emits: a list of **primitives**
  (`circle`, `ellipse`, `rect`, `regular_polygon`, `capsule`) each with center / size /
  rotation, combined with boolean **union / difference**. A snowman is
  `union(circle r20 @0,0; circle r14 @0,-30; circle r10 @0,-50)`. A keyhole is
  `circle ∪ rect`. Evaluate the program to one validated polygon (+ holes).
- **Named shape library on top:** `snowman`, `star`, `heart`, `hexagon`, `gear`, `cat`,
  … each expands to a parametric shape program. The LLM picks a name + scale for the
  common case, or writes a custom program for novel shapes. Robust 90% + open-ended tail.
- **Validity & repair:** every composed/LLM polygon is validated (closed, simple,
  min area, min neck width) and repaired (`make_valid` / self-intersection fix / convex-
  hull fallback) or rejected with a surfaced reason. Never stamp a degenerate outline.

### Layer 2 — build the board inside the region (placement)

Three escalating strategies; choose per shape. Earlier = safer + more reuse.

- **(A) Inscribed-rectangle placement** *(single-blob shapes: heart, star, gear, cat)* —
  compute the **largest axis-aligned rectangle inside the shape**, run the existing
  rectangular solver inside it, and let the shape's extremities (star points, etc.) be
  board material for decoration / LEDs / mounting holes. **Zero solver changes**;
  containment guaranteed (inscribed rect ⊂ shape); copper clips to the outline; routing
  uses the polygon. This is the safe foundation and already "builds a board inside" the
  shape.
- **(B) Region-decomposed leaf mapping** *(compound shapes: the snowman)* — for a CSG
  union, the primitives **are** the regions (head / torso / base). Inscribe a rect per
  region, then **assign the design's leaves/subcircuits to regions** and constrain each
  leaf to its region — riding the existing composer `zone`/anchor system
  (`subcircuit_composer.py:194`; extend the `zone` vocabulary or add a `region` target).
  The parent composer already drops rectangular leaves into the parent; we just pin each
  leaf to a blob. Result: a multi-blob board, each blob hosting a subcircuit, unioned
  under one shaped `Edge.Cuts`. **No solver-core surgery** — a region-assignment heuristic
  + an anchor type. This is the KiCraft-native way to use the *whole* snowman.
- **(C) True polygon-containment placement** *(full-area, tightest, future)* — replace the
  solver's AABB clamp with `polygon.contains(courtyard)` so parts flow into concave/pointy
  areas. Most powerful, most invasive (touches the load-bearing solver legality/scoring).
  Explicitly **tier 3 / deferred** given the "surgical fixes only / don't touch place-route
  core" rule.

### Layer 3 — edges, holes, connectors (mostly reuse)

- Routing boundary, copper pour, Edge.Cuts stamping: **already handled** (see table).
- **Containment validation:** use true polygon-in-polygon (`.contains`) for `polygon`
  shapes; keep the fast analytic path for the parametric shapes (`_compose_validate.py`).
- **Edge connectors:** find the longest near-straight run of the outline and place the
  connector there; if none qualifies (e.g. a pure circle), advisory fallback (downgrade
  to a shape with a flat, or keep the connector internal) + an `open_question`.
- **Mounting holes:** place in the extremities the shape gives us; generalize
  `OutlineSpec.mounting_hole_position` to "inset from nearest boundary."

---

## The snowman, end to end

1. **Capture:** intent recognizes "snowman ornament" → `placement.board.shape = "snowman"`
   (a named shape program), scale from any stated size.
2. **Compose:** evaluate `union(3 circles)` → one non-convex snowman polygon (+ neck
   width check).
3. **Decompose:** the 3 circles are 3 regions; inscribe a placement rect in each.
4. **Map leaves → regions:** assign the design's subcircuits to head / torso / base (pack
   several into a blob if there are fewer blobs than leaves; favor putting weakly-coupled
   subcircuits in separate blobs to keep neck-crossing nets few).
5. **Compose + stamp:** parent composer places each leaf in its region rect; stamp the
   snowman polyline as `Edge.Cuts`.
6. **Route:** FreeRouting routes within the polygon; inter-blob nets cross the **necks**.
7. **Pour + validate:** GND pour clips to the snowman outline; geometry validation uses
   polygon containment.

**The honest hard part — necks.** The neck between blobs is a narrow copper channel.
Interconnect-heavy designs won't fit a pinched shape. v1 mitigations: (a) a
`min_neck_width_mm` parameter that widens necks during composition, (b) region assignment
that minimizes neck-crossing nets, (c) a surfaced advisory when interconnect demand
exceeds neck capacity. This feature targets *decorative / low-interconnect* boards (an LED
snowman ornament), not a 200-pin board crammed into a star — set expectations accordingly.

---

## Geometry engine (a real decision)

Compound shapes need robust boolean **union** (for the Edge.Cuts outline), **offset/buffer**
(margins, necks), **point/rect-in-polygon** (containment), and **validity repair**.
Hand-rolling robust polygon booleans is a well-known trap, and the repo currently has
**no** geometry lib (numpy/scipy/shapely absent; only `matplotlib`).

- **Chosen: `shapely` (GEOS-backed).** One mature dep gives union/difference, `buffer`
  offset, `.contains`, `make_valid`, and largest-inscribed-circle out of the box.
  Largest-inscribed-*rectangle* is a small search on top. Add to `pyproject.toml` deps;
  it becomes the first heavyweight geometry dep (currently only `matplotlib`).
- The parametric shapes (circle/rounded/chamfer/regular-polygon) need **no** engine — they
  keep the existing hand-rolled polyline generators. The engine is only pulled in for
  `polygon`/compound shapes.

---

## Scope tiers (pick v1)

- **Tier 1 — parametric convex** (`circle`, `rounded_rect`, `chamfered_rect`,
  `regular_polygon`/hexagon): capture + circumscribe/inscribe + stamp. **No new dep.**
  Covers "round", "hex", "rounded corners". Smallest, ships fastest.
- **Tier 2 — single arbitrary polygon** (heart, star, gear, cat) via the shape DSL + named
  library + **inscribed-rectangle placement (Strategy A)**. Adds the geometry engine and
  polygon containment. Covers most "fun shape" briefs with full reuse of place+route.
- **Tier 3 — compound / multi-region** (snowman, keyhole) via **region-decomposed leaf
  mapping (Strategy B)** + neck handling. The "wow" capability; medium effort, no solver
  surgery.
- **Tier 4 — full polygon-containment placement (Strategy C):** deferred.

**Committed:** land **Tier 1 → Tier 2 → Tier 3** in sequence (each independently shippable;
the self-eval locks each in). Tier 4 held.

---

## Implementation phases

### Phase 1 — capture (all tiers) — ✅ DONE
Canonical capture is **`IntentSlot.form_factor`**, not `PlacementBoard` — the intent stage
*always* runs (placement is optional user-rules), and the requested shape is genuinely
user intent. Built:
- `design/models.py`: `FormFactor` (shape + `corner_radius_mm`/`chamfer_mm`/`size_mm` +
  provenance `note`) + `IntentSlot.form_factor`. Lenient validation — unknown/library
  shape names (`hexagon`, `snowman`, …) tolerated so a novel shape never bricks the commit.
- `design/synthesis/form_factor.py`: `extract_form_factor()` — deterministic keyword
  classifier with a precision bias (strong patterns fire alone; weak bare words need
  board/shape/diameter context; EE false-friends like *star ground* / *hex inverter* /
  *heart rate* / *Nth round* are excluded). Pulls an advisory `size_mm` from Ø / "N mm
  diameter".
- `cli_app._cmd_stage_commit`: on an intent commit, fills `intent.form_factor` from the
  committed intent text + `brief.txt` when the model didn't set a non-rect shape; echoes
  it in the commit summary. Mirrors the `reconcile_inter_sheet_nets` normalizer pattern.
- `intent.md` documents `form_factor` so the LLM can also set it directly.
- Tests: `tests/test_form_factor.py` (46 — model, extractor positives/negatives/size,
  stage-commit integration incl. brief fallback).

Deferred (not needed for brief→shape): a `placement.board.shape` *override* surface in the
rules panel. Add later if users want to set/override the shape post-intent.

### Phase 2 — flow (all tiers) — ✅ DONE
- `design/synthesis/autoplacer.py`: `write_autoplacer_json(form_factor=…)` emits a
  top-level `board_outline` block (`{shape, corner_radius_mm?, chamfer_mm?, size_mm?}`)
  for any non-rect shape; rect/absent emits nothing. `synthesize.py` passes
  `state.intent.form_factor`. (A future `placement.board` shape override would win here.)
- `cli/_compose_state.py`: `ParentCompositionState.requested_shape` (shape intent only —
  no min/max; distinct from the authoritative `manual_outline`).
- `cli/compose_subcircuits.py`: populated from `cfg["board_outline"]` when no manual
  layout is present. Verified `load_project_config` passes the block through to `cfg`.

### Phase 3 — shape evaluation + outline fit
- New `shapes/` module: shape-program evaluator (named library → primitives → boolean
  union/difference → validated polygon), built on the geometry engine. Generalize
  `OutlineSpec` to `polygon`.
- At parent compose: **Tier 1** circumscribe/inscribe the parametric shape around the
  content AABB; **Tier 2** compute the largest inscribed rectangle of the polygon and run
  the rectangular solver inside it; write the resulting `OutlineSpec` into the field the
  stamp/validate path reads (`state.manual_outline`-equivalent for the auto path).
- Make `_repair_parent_outline` (`_compose_validate.py:17`) aware of the auto-shape path
  (today it bails whenever an outline is set, `:55-60`).

### Phase 4 — region mapping (Tier 3)
- Region decomposition (CSG primitives → blobs) + inscribed rect per blob.
- Extend the composer `zone`/anchor vocabulary to named regions; a heuristic that assigns
  leaves → regions minimizing neck-crossing nets; `min_neck_width_mm` widening.

### Phase 5 — containment + edges
- Swap validation to true polygon containment for `polygon` shapes (`_compose_validate.py`).
- Edge-connector flat-run detection + advisory fallback; mounting-hole generalization.

### Phase 6 — self-eval category (per tier as it lands)
- `kicraft/tuning/benchmark.py`: new archetype `shaped_outline` — existing briefs cloned
  with a shape phrase + an explicit `outline_shape` (+ params) key. Cover: round (no
  connector), rounded-rect **with** USB-C (flat-run path), chamfered badge, a single
  polygon (star/heart), and a compound (snowman) once Tier 3 lands.
- Deterministic classifier `classify_edge_cuts_shape(pcb_path)` in
  `kicraft/render/edge_cuts.py` (segment count, arc presence, circularity, corner-angle
  histogram, blob count for compound) — round-trip self-tested against each `OutlineSpec`.
- New Class-C dimension `outline_shape_correctness` in `eval/rubric.yaml` + scorer in
  `eval/scoring.py` (register in `CLASS_C_SCORERS`) + metric wiring in `eval/metrics_web.py`
  (expected shape from the benchmark entry). Deterministic, $0, reproducible. **Weighted
  dimension, not a hard gate** — fab-readiness is already gated by existing
  pipeline/DRC dimensions, so a shaped board that won't route still gets caught. Harness +
  report need no change (new dimension auto-picked-up).

---

## Testing
- **Unit:** shape-program evaluator (snowman → expected blob count/area, neck width);
  inscribed-rectangle (always ⊂ shape); polygon containment vs the convex shortcut;
  `classify_edge_cuts_shape` round-trips every shape.
- **Integration** (mock-LLM web driver, build-in-place recipe): "round 50 mm sensor" and
  (Tier 3) "LED snowman ornament" end to end → Edge.Cuts classifies correctly, all parts
  inside the polygon, `build_rc == 0`.
- **Regression:** rectangular briefs unchanged through stamp/pour; existing
  `test_outline_shape_stamping.py` stays green.

## Out of scope (v1)
- Tier 4 full polygon-containment placement (solver surgery).
- Dense/high-interconnect designs in pinched shapes (neck capacity).
- Image/SVG tracing as a shape source (could feed the polygon path later).
- Do not touch `placement_solver.py` core — the whole design avoids it.

## Key file references
- Shape model / generators: `kicraft/layout_editor/outline.py` (`OutlineSpec`)
- New shape DSL + geometry engine: `kicraft/shapes/` (new), dep `shapely`
- Capture (done): `kicraft/design/models.py` (`FormFactor`, `IntentSlot.form_factor`), `kicraft/design/synthesis/form_factor.py` (`extract_form_factor`), `cli_app._cmd_stage_commit`
- Flow: `kicraft/design/synthesis/autoplacer.py:~199`, `kicraft/cli/_compose_state.py:123`, `kicraft/cli/compose_subcircuits.py:1811`
- Fit + validate: `kicraft/cli/_compose_validate.py:17,157,209`
- Stamp: `kicraft/cli/_compose_stamp.py:148`
- Region anchors: `kicraft/autoplacer/brain/subcircuit_composer.py:194,554`; `kicraft/design/models.py:578`
- Routing boundary (works): `kicraft/autoplacer/freerouting_runner.py:506`
- Self-eval: `kicraft/tuning/benchmark.py`, `kicraft/eval/{rubric.yaml,scoring.py,metrics_web.py}`, `kicraft/render/edge_cuts.py`
