> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# Plan v2: connector stranding + stamp-short family — root-cause, not gating

**Status:** Phases 1+2 **MERGED** (2358fd4) & fleet-validated. Phase 3 (Lever
2.1) **DONE 2026-06-14** & fleet-validated (fixture `PARENT_LOCAL_CONN`,
deterministic A/B), MERGED + deployed. Phase 4 (Lever 2.5) **DONE 2026-06-14**:
`compose_subcircuits.py` split 4331→2942 LOC (−32%) into 6 cohesive modules,
corpus-zero-drift. All plan phases complete.

## Progress (2026-06-14)
- **Phase 1 (RC3+RC2) DONE.** `+rot` convention flipped at all three recovery
  sites; `can_overlap_sparse` rewritten to the same-layer-clearance rule
  (fragile per-rect same-layer branch + outline gate deleted). Compose stays
  rc=0 with no seam shorts. Convention unit tests updated to KiCad-CW.
- **Phase 2 (RC1) DONE.** Rotation filter now also requires every edge-zoned
  part to be its leaf's extremity on the zoned side (fixes mouthless switches);
  `edge_zoned_outline_sides` (was `connector_outline_sides`, J-only) registers
  all edge-zoned non-hole refs so `_repair_parent_outline` can't bury them;
  `connector_edge_gap` measures mouth = courtyard ∪ pads. **Fixture result:
  J1/J2/SW1 all flush (SW1 was −9.2mm); the SW1 xfail is now a passing test.**
  Parent corpus golden updated (expected move); leaf placement unchanged.
- **Validation:** full placement/compose unit suite green (only 4 PRE-EXISTING
  unrelated failures remain); parent corpus deterministic across runs.
- **Fleet validation (deterministic A/B, not the noisy synthesis self-eval).**
  Full-synthesis self-eval is dominated by synthesis noise — fab-ready swings
  0/9–6/9 across runs on identical code — so it can't detect a place/route
  regression. Instead I replayed PARENT COMPOSE on the 9 freshly-synthesized
  workspaces' FROZEN routed leaves (where RC1/RC2/RC3 live), my branch vs
  `main` (`scripts/ab_compose.py`):
  - 7/9 compose rc=0 on my branch; the two non-successes — run_08 (rc=2, a FLAT
    board: no leaf subcircuits, compose N/A) and run_09 (rc=1, abort) — fail
    **identically on `main`**. No new failure.
  - run_09 IMPROVED: candidate shorts 8/8/1/5 (mine) vs 16/24/… (main),
    consistent with RC2 holding same-layer leaves apart. **No regression; a net
    improvement on the one stressed board.**
- **Phase 3 (Lever 2.1) DONE 2026-06-14.** Fixture `PARENT_LOCAL_CONN` (=
  `USB_PD_TRIGGER` + a root-level connector `J3` `edge:bottom`, in no leaf →
  parent-local; `scripts/build_parent_local_conn_fixture.py`) was the gate. It
  first proved the premise wrong — a parent-local connector does NOT land flush;
  J3 **stranded ~4 mm** while leaf connectors stayed flush — so Lever 2.1 is a
  **fix**, not just simplification.
  **The fix:** `_wrap_loose_parent_components_as_leaves` wraps each loose
  parent-level non-board component as a single-component leaf so it flows through
  the leaf edge-pin/flush path; the `_snap_parent_local` connector branch is
  deleted (board-level holes keep the generic snap). **Critical sub-bug:** the
  wrapped component must be RE-BASED into its own `(0,0)`-anchored leaf box — it
  still carried absolute seed-PCB coords, so the leaf-frame anchor math didn't
  cancel and the edge anchor landed ~150 mm out of frame (slack fallback →
  stranded by `spacing`). The garbage anchor also polluted the whole outline,
  stranding a neighbour (SW1). After the re-base **all four connectors land
  flush** and the xfail flips to a passing test.
  **Validation:** `USB_PD_TRIGGER` parent golden byte-unchanged (wrap is a no-op
  without a parent-local component); fixture all-flush + deterministic; full unit
  suite no new failures; deterministic fleet A/B (`scripts/ab_compose.py`) 8/9
  IDENTICAL to main and **run_09 IMPROVED (rc=1 abort → rc=0)** — zero
  regressions, one net win. Incidental: `replay_corpus.py` reads a per-fixture
  `parent_compose_spacing_mm` (this board shorts at the 2.0 default) and SKIPs a
  mode with no golden (PARENT_LOCAL_CONN is parent-only).
- **Phase 4 (Lever 2.5, file splits) DONE 2026-06-14.** `compose_subcircuits.py`
  (4331 LOC) split bottom-up into 6 cohesive sibling modules, re-exported to
  preserve the public API + internal references: `_compose_geometry` (bbox/rect
  leaf helpers), `_compose_state` (`CompositionEntry`/`ParentCompositionState`),
  `_compose_validate` (outline repair + geometry validation + render),
  `_compose_stamp` (board stamping + pre-route DRC), `_compose_route`
  (FreeRouting + post-route validation), `_compose_persist` (artifact write).
  Core module → 2942 LOC (orchestrator `_compose_artifacts`, `_search_best_layout`
  — kept since it calls `_compose_artifacts` — parent-local, output formatters,
  CLI). Mechanical/no-op: corpus zero-drift on both fixtures + full unit suite
  unchanged (25 pre-existing fails only). Dep order was enforced with `symtable`
  (each module imports only what it uses). NOTE: `placement_solver.py` is a
  single ~3800-LOC `PlacementSolver` class — not a mechanical file-split target
  (would need a class/mixin decomposition); left intact.

---

**Original status:** proposed 2026-06-14. Supersedes the *gated* framing in
`place-route-replay-and-codebase-simplification.md` Part 3 / Lever 2.1. That doc
did excellent foundational work (replay harness, geometry centralization,
fallback inventory, the `connector_edge_gap` metric) — keep all of it. What this
v2 changes is the **diagnosis** and therefore the **fix order**.

## What I re-measured this session (the evidence)

Reproduced on the committed `USB_PD_TRIGGER` fixture (frozen leaves, pinned
hash+threads, `scripts/diag_convention.py`):

| Experiment | J1 (left) | J2 (right) | SW1 (top) | compose |
|---|---|---|---|---|
| baseline (`-rot`, main) | **+0.475 flush** | **+0.475 flush** | **−9.23 stranded** | rc 0 |
| flip all 3 recovery sites → `+rot` | n/a | n/a | n/a | **rc 1 (abort: shorts 4/18/1/5)** |
| `+rot` + same-layer-clearance rule | −1.20 | +0.475 | **−8.65 still stranded** | rc 0 |

Three findings overturn the old plan:

1. **The 90/270 convention bug is NOT the stranding root cause.** With the
   convention corrected (`+rot`, all three sites consistent) SW1 stays **−8.65mm
   inboard**. The convention bug is real and worth fixing (it mis-recovers
   rotated blocks' origins and is a latent correctness hole), but it is a
   *secondary* contributor, not the lever the old plan made it.

2. **SW1 is already correctly placed inside its own leaf** — its courtyard top
   is 0.53mm from the leaf's top board edge. It strands on the *parent* because a
   **taller neighbour leaf defines the board's top edge**, and the `edge:top`
   constraint only *attracts* SW1's leaf toward the top region; it does not
   **align** the connector edge to the board extremity. Left/right look fine only
   because J1/J2's leaves happen to BE the left/right extremities. This is the
   old plan's "hypothesis 3", which it noted and then walked past.

3. **The seam-shorts that "gated" the convention fix have a clean root cause.**
   The parent solver lets two leaf blocks pack bbox-adjacent whenever
   `can_overlap_sparse` (a sampled-rect heuristic) calls their copper
   "compatible". The `-rot` bug had been padding spacing and masking it. The
   principled rule is: **same-layer leaves are held at design clearance; only
   opposite-layer leaves may stack.** That deletes the fragile per-rect
   same-layer interlocking allowance (the documented source of seam shorts,
   `placement_solver.py:3182`) instead of patching the push distance.

Net: the old plan's "must untangle a coupled solver convention before anything"
conclusion was an over-estimate born of flipping the recovery *inconsistently*.
The work splits into three independent, individually-validatable root-cause
fixes.

## The three root causes (and the single mechanism each should become)

| # | Root cause | Today | Root-cause fix |
|---|---|---|---|
| RC1 | Edge constraint is *soft* attraction, not *hard* extremity alignment | edge-zoned leaf drifts inboard when a neighbour is the extremity or separation pushes it off | one **edge-flush alignment** mechanism: an `edge:S` leaf's connector mouth is pinned flush (within the overhang policy) to the board's S edge, and defines the board extremity on S |
| RC2 | Same-layer leaves pack within clearance on a heuristic | `can_overlap_sparse` per-rect allowance → stamp seam shorts | same-layer ⇒ enforce clearance; only opposite-layer may stack. Delete the per-rect same-layer branch |
| RC3 | 90/270 origin recovery uses `-rot` (math-CCW) not the true inverse `+rot` | 3 sites carry the bug "consistently" | flip all 3 to `+rot`; the geometry module already documents `+rot` as correct |

RC2 and RC3 are coupled (RC3 alone tightens layouts → RC2's shorts). RC1 is the
actual stranding fix and is the user-visible win.

## Sequencing — each gated by the existing harness

Validation gate for every step: `scripts/replay_corpus.py --mode parent` (golden
diff) + `tests/test_connector_edge_gap.py` (KICRAFT_REPLAY_E2E=1) +
`tests/test_geometry.py`. A behaviour change shows as an *expected, located*
golden move; a no-op refactor must show *zero* drift. **Before merge:** one
`self-eval` skill batch (real $, ~1h) — the corpus is a single board and cannot prove
the 50-family fleet didn't regress.

- **Phase 1 — RC3 + RC2 together (correctness + robustness).**
  Flip the 3 recovery sites to `+rot`; add the same-layer-clearance rule and
  delete the dead per-rect same-layer branch. Outcome: compose stays rc 0 with
  no seam shorts, the convention is correct, ~N LOC of heuristic deleted. Update
  the parent golden (expected move). Edge connectors are NOT yet flush — that's
  Phase 2.

- **Phase 2 — RC1 (the stranding fix), ONE mechanism for both paths.**
  Edge-flush alignment: after placement, an `edge:S` leaf is aligned so its
  connector mouth sits flush/overhang on the board's S edge and is the extremity
  there (neighbours may not protrude past it). Shared by the leaf path and the
  (soon-deleted) parent-local path. Gate: `connector_edge_gap` xfail
  (`test_top_zoned_switch_not_stranded`) flips to **pass**; J1/J2 stay flush.

- **Phase 3 — Lever 2.1 (simplification, now safe).**
  Auto-wrap loose parent-level non-structural components as single-component
  leaves; delete `_snap_parent_local`'s connector branch, `_rotate_component_in_place`,
  and the `parent_local` constraint branch. **One** connector placement path.

- **Phase 4 — Lever 2.5 (mechanical).** Split `compose_subcircuits.py` /
  `placement_solver.py` along their seams now that the duplicate logic is gone.

## Non-goals / guardrails
- Don't touch LLM synthesis (frozen, per Part 1).
- Routing stays best-effort-deterministic (FreeRouting); determinism scoped to placement.
- Opposite-layer stacking (LLUPS-style) must keep working — RC2 only forbids *same-layer* packing.
- No board may *regress* on the `connector_edge_gap` gate or self-eval fab-ready count.
