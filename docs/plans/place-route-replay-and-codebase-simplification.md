# Plan: deterministic place+route replay + placement/compose simplification

**Status:** proposed (for another agent to implement)
**Author context:** written 2026-06-14 after a USB‑C edge‑connector fix chain (orientation, overhang, parent‑local) where we **could not deterministically validate a placement change** — `kicraft build` re-runs LLM synthesis, which re-partitions the sheet hierarchy, so a rebuilt board has a *different* structure than the one whose bug we were chasing. That pain motivates Part 1. The difficulty of even *locating* the bug (it lived in a duplicated "parent-local" path that mirrors the leaf path) motivates Part 2.

> **Line numbers in this doc are approximate** (the tree moves). Always re-grep the named function/symbol before editing.

---

## Part 1 — `replay` command: re-run ONLY place + route, deterministically

> **STATUS: IMPLEMENTED 2026-06-14.** `kicraft replay` ships (both input modes),
> seed + env pinned, placement determinism verified, fixture + tests + corpus
> harness committed. See **“Implemented”** at the end of Part 1 for what landed,
> the determinism scope that held, and the one follow-up it surfaced.

### Goal
A CLI command that takes an **already-synthesized** project workspace and re-runs placement + routing (+ promote/verify/fab) **without touching the LLM synthesis stages**, so a code change can be tested against a *fixed* input and produce a *reproducible* board. This is the deterministic test harness the connector work lacked.

### Why the current `build` can't do this
`kicraft build <state.json> <out_dir>` runs 5 steps (`_cmd_build`, `kicraft/design/cli_app.py:2488`):

| Step | Function | Deterministic? |
|---|---|---|
| 1/5 synthesize (schematic + seed PCB + ERC) | `run_synth` → `kicraft/design/synthesize.py:run()` | **NO** — LLM-driven; re-partitions sheets into leaves vs parent-local each run |
| 2/5 place + route | `_layout_route_fab` → `_run_layout(quality, root_sch, pcb)` (`cli_app.py:2178`) | YES (seeded SA + freerouting) |
| 3/5 promote routed parent | `_promote_verify_fab` (`cli_app.py:2322`) | YES |
| 4/5 verify (shorts/unconnected) | `_verify_routed_board` | YES |
| 5/5 export fab package | `export_fab` | YES |

The deterministic boundary is **end of step 1**: once the `.kicad_sch` files, `<stem>_autoplacer.json`, and seed `<stem>.kicad_pcb` exist on disk, steps 2–5 are pure-from-disk. `_cmd_manual_route` (`cli_app.py:2406`) already proves the pattern (it skips synthesis and goes straight to compose+route from a saved layout) — but it routes a *manual* layout, bypassing the auto-placer. We want the **auto** place+route re-run.

### Design
Add a `replay` subcommand that is **`_cmd_build` minus step 1**:

1. Resolve the synthesized workspace. Two input modes (support both):
   - `replay <state.json> <out_dir>` — read `state.artifacts` (an `ArtifactPaths`, `kicraft/design/models.py:~604`) for `project_dir`, `root_sch`, `project_stem`.
   - `replay --project <dir>` — discover artifacts on disk by stem: `<stem>.kicad_sch`, `<stem>.kicad_pcb`, `<stem>_autoplacer.json`, `<stem>.kicad_pro`. (Preferred for testing — no `state.json` needed.)
2. **Validate** all required artifacts exist; fail loudly (rc=3) listing any missing file. Do **not** call `run_synth`.
3. Call the existing seam directly: `_layout_route_fab(args, state, state_path, artifacts, results=[], stem, project_dir, root_sch, pcb)` (`cli_app.py:2573`), which already runs steps 2–5. Reuse `build_slot(...)` so it shares the host flock queue.
4. Flags: `--quality {fast,draft,good,best}` (default `fast` for quick iteration), `--seed <int>` (see Determinism), `--no-fab` (skip step 5/5 for speed), `--route/--no-route`.

Registration mirrors `p_build` (`cli_app.py:2912`); `set_defaults(func=_cmd_replay)`.

### Determinism requirements (the whole point)
- **Pin the RNG seed.** The SA placement uses a seed (`PlacementSolver(..., seed=...)`); `autoexperiment`/`solve_hierarchy` thread it. The replay command MUST pass a fixed, explicit `--seed` (default e.g. `0`) all the way down so two replays of the same workspace produce the same placement. Audit the seed plumbing from `_run_layout` → `autoexperiment_main` / `_solve_hierarchy_main` and ensure no `Date.now()/random` leaks in (the build worker / autoexperiment may derive a time-based run id — make the placement RNG independent of it).
- **Confirm reproducibility as an acceptance test** (below). If it's not bit-reproducible, identify the nondeterminism source (unordered dict iteration, time-seeded RNG, freerouting threads) and fix or document it. Freerouting itself may not be bit-deterministic; if so, scope the determinism guarantee to **placement** (the part we change most) and treat routing as best-effort-stable.

### Implementation steps
1. `kicraft/design/cli_app.py`: add `_cmd_replay(args)` (model it on `_cmd_build` but delete the step-1 block and the `run_synth` call; reuse `_resolve_artifacts`/`_load_state` helpers already used by build). Add `p_replay = sub.add_parser("replay", ...)` next to `p_build`.
2. Add a small `_resolve_synthesized_workspace(state_or_project) -> ArtifactPaths` helper that supports both input modes and validates file existence.
3. Thread an explicit `--seed` through `_run_layout` if it isn't already a parameter (it currently takes only `quality, root_sch, pcb` — add `seed`).
4. Update `--help`/README and the `self-eval` docs to mention `replay` for debugging a single saved run.

### Acceptance criteria
- `kicraft replay --project <dir> --quality fast --seed 0` on a synthesized workspace produces a routed board **without** invoking any LLM stage (assert no network/LLM calls; assert the `.kicad_sch` files are byte-identical before/after).
- Running it **twice** yields placements with identical component positions+rotations (assert via a geometry diff of the parent `.kicad_pcb`). Document routing determinism scope.
- A saved workspace that previously produced a stranded connector reproduces the **same** stranding on replay (this is the deterministic repro Part 3 needs).
- New test: `tests/test_replay_command.py` — point at a tiny committed fixture workspace; assert rc=0, board produced, and position-stability across two runs.

### Effort
Small (~1 day). Almost entirely reuse; the real work is auditing/pinning the seed and writing the reproducibility test.

### Implemented (2026-06-14)
**What landed:**
- `kicraft replay` (`cli_app.py`): both input modes (`replay STATE.json OUT_DIR`
  and `replay --project DIR`), `--quality {fast,draft,good,best}` (default
  `fast`), `--seed` (default 0), `--route/--no-route`, `--no-fab`, `--archive`
  (replay skips the session archive by default). Helpers: `_cmd_replay`,
  `_resolve_synthesized_workspace` (validates the artifacts the placer consumes,
  rc 3 on missing; `_autoplacer.json` is a warning — it's UI-only, the placer
  never reads it), `_discover_stem`, `_find_state_json`, `_find_placed_parent`.
  Reuses the `_layout_route_fab` → `_run_layout`/`_promote_verify_fab` seam under
  `build_slot`. A post-run guard re-reads the root `.kicad_sch` and fails rc 8 if
  anything mutated it — the no-synthesis invariant, asserted, not assumed.
- **Seed plumbed** through `_run_layout(seed, route)` → solve-hierarchy (new
  `--seed`, threaded to `solve_subcircuits` via `_solve_leaves`) and
  autoexperiment (`--seed` already existed; was unset = *random*). `seed=None` is
  preserved as the `build` default (no `--seed` forwarded → build's search stays
  random), so **build behavior is unchanged**; only `replay` pins a seed.
- **Env pinned** by `_pin_deterministic_placement_env()` (replay only):
  `PYTHONHASHSEED=0` + single-thread numpy (`OMP/OPENBLAS/MKL/NUMEXPR_NUM_THREADS=1`),
  `setdefault` so a caller can override.
- Fixture `tests/fixtures/replay_workspace/USB_PD_TRIGGER/` (a real USB-C, 4-leaf
  workspace; ~256 KB), `tests/test_replay_command.py` (15 unit + 1 opt-in e2e
  gated by `KICRAFT_REPLAY_E2E=1`), and `scripts/replay_corpus.py` (golden-diff
  harness — the “make replay-corpus” the sequencing calls for; `--update` writes
  goldens, no-arg checks; `*.golden.json` committed).

**Determinism — what actually held (the audit the plan asked for):**
- **`PYTHONHASHSEED` is the dominant nondeterminism source.** The placement
  solver iterates `set`/`dict` of string refs and dedups force-states via
  `hash(...)` (`placement_solver.py:~2223`), all salted per-process by the hash
  seed. Unpinned, two seeded runs of the same workspace diverge at **mm scale**
  (verified). Pinned, leaf placement is **byte-identical**.
- **Placement IS reproducible; the composed parent is NOT.** Each leaf's
  `leaf_pre_freerouting.kicad_pcb` (the placement output) is 0-diff across two
  independent replays with the env pinned. The promoted *parent*
  (`<stem>.kicad_pcb`) still differs run-to-run because compose consumes the
  **routed** leaf boards, and FreeRouting is not bit-deterministic. This is
  exactly the plan's scoped guarantee: **placement deterministic, routing
  best-effort.** The determinism test + corpus therefore assert on the per-leaf
  placement boards, not the parent.

**Follow-up this surfaced (worth a fix, deferred — belongs with Part 3 / compose
cleanup):** `replay --no-route` skips *parent* routing but the leaf solve still
runs FreeRouting on each leaf (confirmed: `solve_subcircuits` routes leaves even
with `args.route=False` — the route flag plumbs correctly but routing still
happens; root cause not yet pinned). Two payoffs once fixed: (a) `--no-route`
becomes genuinely fast (leaf routing is most of the wall-clock), and (b) if
compose then reads the deterministic `leaf_pre_freerouting` boards, the **parent
becomes deterministic too**, upgrading the determinism guarantee from leaf-level
to board-level — directly enabling Part 3's "same stranding on replay" repro.

---

## Part 2 — Simplify the placement / compose codebase

### Why
The connector bug was hard to *find* because the same logical operation ("pin an edge connector facing outward") exists in **two parallel code paths** (leaf vs parent-local) plus scattered rotation/anchor/frame math. The core modules are huge and patch-laden:

| File | LOC | Notes |
|---|---|---|
| `kicraft/cli/compose_subcircuits.py` | ~4300 | god-file: CLI + compose + outline + snap + validate + stamp |
| `kicraft/autoplacer/brain/placement_solver.py` | ~3900 | **57** fallback/legacy/clamp markers |
| `kicraft/autoplacer/brain/subcircuit_composer.py` | ~2450 | constraints + blockers + anchors + rebase |
| `kicraft/cli/solve_subcircuits.py` | ~1700 | |
| `kicraft/autoplacer/hardware/adapter.py` | ~1135 | |

### Guiding principles
1. **One path per concept.** If two code paths do "the same thing for a slightly different container," collapse them. Branches and fallbacks are where bugs hide.
2. **Centralize core mechanisms.** Coordinate-frame transforms, rotation conventions, and edge/anchor math must each live in exactly one place with one tested convention.
3. **Fail loud, not soft.** Replace silent fallbacks/clamps with explicit errors + diagnostics. A fallback that "rescues" a bad input also hides the bug that produced it.
4. **Shrink the god-files** into focused modules so a reader can find a behavior by file name.
5. Every collapse must be **guarded by the Part 1 replay harness** — re-run a fixed corpus before/after and diff the boards.

### Lever 2.1 — "Top-level parent sheets contain only leaves" (eliminate the parent-local component path)
This is the highest-value simplification and the user's lead suggestion.

**Finding (feasible):** A sheet is a leaf iff it has zero child sheets (`hierarchy_parser.py:~510`). "Parent-local" components are whatever ends up constrained but not inside any child artifact (`extract_parent_local_components`, `subcircuit_extractor.py:301`). For connectors this creates a **second** placement path:
- **Leaf path:** `subcircuit_composer._filter_rotations_for_connector_opening` + `_compute_local_anchor_offset` + attachment constraints → solver rotates/pins the leaf.
- **Parent-local path:** `compose_subcircuits._snap_parent_local` + `_rotate_component_in_place` (added in the recent fix) → post-solve rotate+snap.

These are duplicate logic for the same goal. The recent parent-local connector fix (`ae29d82`) is itself evidence of the duplication — we had to re-implement orientation in the second path.

**Proposal:** enforce the invariant **"a top-level/parent sheet contains only child leaves (plus board-level structural refs — mounting holes / fiducials)."** Any loose connector on a parent sheet is **auto-wrapped into a single-component leaf** before composition, so it flows through the *one* leaf path.

**What must stay parent-local:** mounting holes `H*` (board structure, often synthesized at compose time, `compose_subcircuits.py:~1445`) and possibly fiducials/test points. Keep an explicit, named `board_level_components` set for these; everything else must be in a leaf.

**Migration:**
1. Add `wrap_single_component_as_leaf(comp, nets) -> LoadedSubcircuitArtifact` (the synthetic-block machinery already exists: `parent_adapter.artifact_to_component`, `synthetic_block_ref`). A trivial one-component `SubCircuitLayout` with no internal traces.
2. In `_compose_artifacts` (or earlier, at extraction), detect non-board-level parent-local refs and wrap each as a leaf; append to `loaded_artifacts`. Stop extracting them as parent-local.
3. **Delete** `_snap_parent_local`'s connector branch + `_rotate_component_in_place` (`compose_subcircuits.py:1098–1217`); keep only mounting-hole/board-level snapping (or fold holes into the same mechanism).
4. Simplify `AttachmentConstraint.source` (`subcircuit_composer.py:~194`) — if only mounting holes remain parent-local, the `child_artifact`/`parent_local` split shrinks to one real path; remove `DerivedAttachmentConstraints.parent_local_constraints` and the `parent_local` branch in `derive_attachment_constraints`.
5. Update `connector_outline_sides` population and `_repair_parent_outline` accordingly (connectors are now always leaf-borne).
6. Add a **synthesis-time guard/lint**: warn (or auto-wrap) when a parent sheet carries a non-board-level component, so the invariant is enforced at the source, not patched downstream.

**Deletes/collapses (estimate ~500 LOC out, ~200 in):** `_snap_parent_local` connector branch, `_rotate_component_in_place`, the parent-local constraint branch, the dual rotation handling. **One** connector placement path remains.

**Risks:** manual layout editor references `ManualLayout.parent_local` (`layout_editor/model.py`); provide a migration/deprecation. Existing projects with parent-local connector zones must be auto-wrapped (backward compat). Mounting-hole synthesis must be preserved.

### Lever 2.2 — Centralize coordinate-frame + rotation math (kills convention bugs)

> **STATUS: IMPLEMENTED 2026-06-14 (no-op, corpus-verified).** New module
> `kicraft/autoplacer/brain/geometry.py` owns the single KiCad-CW convention:
> `rotate_vector`, `transform_point`, `bbox_after_rotation`,
> `rotate_component_in_place`. Migrated every ad-hoc site to it —
> `subcircuit_instances._transform_point`/`_rotate_size`/`_rotated_bbox_size`,
> `compose._rotate_component_in_place` (deleted; call site uses `geometry.`),
> `keepout_extract._transform_local_rect`, and the two math-CCW *inverse*
> sites (`parent_adapter._rotated`, `placement_utils._world_artifact_origin`)
> as `rotate_vector(v, -deg)` (provably identical; `math-CCW(θ) ≡
> rotate_vector(·,-θ)`). `tests/test_geometry.py` (18 tests) pins the
> convention incl. **agreement with `pcbnew.SetOrientationDegrees`**; the
> replay corpus confirms leaf placement is byte-identical (true no-op). The
> two inverse sites were found to be a **latent convention bug at 90/270** —
> preserved exactly here and flagged in `lever-2.4-fallback-inventory.md` for a
> separate, parent-corpus-validated fix (likely the Part 3 root cause).
Rotation/transform logic is scattered and uses **inconsistent conventions** — this directly caused a bug in the recent fix (math-CCW vs KiCad CW). Functions found: `subcircuit_instances._transform_point` (KiCad convention, `:830`), `_rotate_size` (`:760`), `_rotated_bbox_size` (`:863`), `parent_adapter._rotated` (`:62`), `compose_subcircuits._rotate_component_in_place` (`:1098`), `placement_solver` rotations, `keepout_extract._transform_local_rect`.

**Proposal:** one module `kicraft/autoplacer/geometry.py` (or extend `types.py`) owning: `rotate_point`, `rotate_size`, `transform_point(origin, rot)`, `rotate_component_in_place`, `bbox_after_rotation` — **all using the single KiCad convention** documented once (`board_angle = local_angle - rotation`; `x' = x·cosθ + y·sinθ; y' = -x·sinθ + y·cosθ`). Replace every ad-hoc rotation with calls to it. Add a property test: `transform_point` agrees with `pcbnew.SetOrientationDegrees` for a pad at (1,0) across 0/90/180/270 (the empirically-verified case already documented in `_transform_point`).

### Lever 2.3 — Centralize edge/anchor/outline math
Anchor + edge-target logic is spread across: `edge_anchor_target_coordinate` (`subcircuit_composer.py:604`), `_compute_local_anchor_offset` (`:542`), `_compute_mounting_hole_anchor` (`:508`), `_constraint_local_rect` (`:1450`), `_resolve_constraint_anchor_positions` (`compose_subcircuits.py:649`), `_snap_parent_local` (`:1126`), `_compute_final_outline` (`:961`), `_repair_parent_outline` (`:2155`), plus `edge_outward_angle`/`opening_board_angle` (`types.py`, added recently).

**Proposal:** a single `edge_attachment` module exposing one tested API: given (component, edge, overhang) → the connector's mouth anchor and the outward rotation; given (constraints, outline) → the final outline and per-ref anchor coordinates. Leaf and (remaining) parent-local snapping both call it. This removes the "snap uses pad centroid but outline uses courtyard mouth" mismatch class of bugs.

### Lever 2.4 — Inventory and remove fallbacks / sloppy patches

> **STATUS: inventory DELIVERED 2026-06-14** — see
> `docs/plans/lever-2.4-fallback-inventory.md` (~98 markers classified P/F/D/K;
> the ~35 in `placement_solver` collapse to "board-containment clamps" that
> should *count + surface* activations rather than silently rescue; plus the
> plan-named items and the new 90/270 convention-bug finding). Removal/promotion
> of individual items is follow-up work.

`placement_solver.py` alone has ~57 fallback/clamp/legacy markers. **Task:** produce a one-page inventory (grep `fallback|best.effort|escape hatch|workaround|legacy|for now|clamp|HACK|XXX`) and for each decide: (a) promote to a real, documented mechanism, (b) replace with a loud failure + diagnostic, or (c) delete if dead. Specific known ones to revisit:
- The opening-direction 3-layer fallback (marker → body-extension → centroid) in `adapter.detect_opening_direction` — keep, but make the chosen layer observable in diagnostics.
- The `>10mm` anchor clamp in `_compute_final_outline` (history: double-rebase) — once frames are centralized (2.2), this clamp may be deletable; verify.
- The keep-all rotation fallback in `_filter_rotations_for_connector_opening` — replace silent keep-all with a logged, surfaced "unsatisfiable" diagnostic (already warns; make it a structured finding).
- Best-effort persistence / promote-dirty-board paths — audit whether they still earn their keep.

### Lever 2.5 — Split the god-files
`compose_subcircuits.py` (~4300) should split along its already-clear seams: `compose/cli.py` (argparse/main), `compose/outline.py` (`_compute_final_outline`, `_repair_parent_outline`), `compose/snap.py` (anchor snapping), `compose/validate.py` (`_validate_parent_geometry`), `compose/stamp.py`. Same for `placement_solver.py` (pinning vs SA vs scoring vs grid). Pure mechanical extraction — do it **after** the logic collapses (2.1–2.3) so we move less code.

### Sequencing
1. **Part 1 replay harness first** (everything else is validated with it).
2. Build a small **fixed corpus**: ~6 committed synthesized workspaces (incl. a stranded-connector case, a flat board, a multi-leaf USB board). A `make replay-corpus` that runs `replay` on each and diffs geometry vs a golden snapshot.
3. **Lever 2.2** (rotation/frame centralization) — lowest risk, removes a bug class, no behavior change intended (corpus diff should be ~empty).
4. **Lever 2.3** (anchor/edge centralization).
5. **Lever 2.1** (parent-only-leaves) — biggest win, biggest blast radius; do it on the back of 2.2/2.3 so there's one anchor/rotation API to call.
6. **Lever 2.4** (fallback inventory) — ongoing, but the inventory is an early deliverable.
7. **Lever 2.5** (file splits) last.

---

## Part 3 — Investigation: "connector not pinned to its assigned edge" (deep-stranding)

### Symptom
In self-eval `20260613T223846Z`, connectors with a correct `edge:` zone ended up **6–9mm inboard** of that edge (run_01/04/08), some not rotated. Initially looked parent-local, but a rebuild showed a **leaf** connector (zone=bottom) stranded 8mm too — so it is **cross-path**, not just the parent-local gap already fixed. This is distinct from the (fixed) 2mm margin-burial: here the connector/leaf is not pulled to the edge at all.

### Method (now unblocked by Parts 1+2)
1. Use **Part 1 `replay`** to get a deterministic repro: find a saved workspace that strands a connector, confirm it reproduces every run.
2. Bisect the placement: dump the connector's intended edge-anchor target vs its actual placed position at each stage — solver output → `_compute_final_outline` → `_snap_parent_local`/leaf-pin → `_repair_parent_outline`. Find the stage where actual diverges from target.
3. With Part 2's single path + centralized anchor math, the divergence should be inspectable in one place.

### Hypotheses to test
- The edge constraint pins the *leaf block* to the edge but the connector sits inboard **within** the leaf (leaf-internal placement put other parts between the connector and the leaf edge) — i.e., the connector isn't the leaf's extremity. (Relates to the earlier `connector_edge_inset_mm` band.)
- The solver satisfies the constraint against the **seed** outline, then `_compute_final_outline` grows the board on that side for unrelated geometry and the connector isn't re-pinned to the final edge.
- For multi-leaf boards, another leaf extends past the connector's leaf on that edge (the connector isn't the board extremity), so the outline edge is defined elsewhere.

### Acceptance
- A regression test (built on the corpus): for every edge-zoned connector, `mouth_to_edge_gap ∈ [−0.1mm (flush), +overhang]` (no burial > ~0.5mm) across the corpus.
- The fix is a **single** mechanism (post Part 2), not another per-path patch.

---

## Non-goals / risks
- Don't change the LLM synthesis stages' behavior (Part 1 deliberately freezes them).
- Routing may not be bit-deterministic (freerouting) — scope determinism guarantees to placement; state this explicitly.
- The parent-only-leaves migration touches the manual layout editor and existing project configs — provide auto-wrap backward-compat and a deprecation note.
- Do each collapse behind the replay corpus diff; a "no-op refactor" that changes any board is a bug.

## Quick reference (confirm before editing)
- Build seam: `_cmd_build` `cli_app.py:2488`; `_layout_route_fab:2573`; `_run_layout:2178`; `_promote_verify_fab:2322`; `_cmd_manual_route:2406` (precedent).
- Parent-local: `extract_parent_local_components` `subcircuit_extractor.py:301`; `_snap_parent_local`/`_rotate_component_in_place` `compose_subcircuits.py:1098–1217`.
- Constraints: `derive_attachment_constraints`/`AttachmentConstraint` `subcircuit_composer.py:~194,235`; `_filter_rotations_for_connector_opening:~433`.
- Rotation/frame: `subcircuit_instances._transform_point:830`, `_rotate_size:760`; `parent_adapter._rotated:62`.
- Edge/anchor: `edge_anchor_target_coordinate` `subcircuit_composer.py:604`; `_compute_final_outline:961`, `_repair_parent_outline:2155` (compose).
- Orientation helpers (already centralized, extend these): `types.edge_outward_angle/opening_board_angle/angles_close`.
- Leaf rule: `hierarchy_parser.py:~510 is_leaf=(len(child_nodes)==0)`.
