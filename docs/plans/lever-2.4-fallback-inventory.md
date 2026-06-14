# Lever 2.4 — fallback / clamp / legacy inventory (early deliverable)

Companion to `place-route-replay-and-codebase-simplification.md` (Part 2, Lever
2.4). Grep over the placement/compose subsystem for
`fallback|best.effort|workaround|legacy|for now|clamp|HACK|XXX|FIXME` →
**~98 markers**, concentrated in `placement_solver.py` (35),
`compose_subcircuits.py` (10), `solve_subcircuits.py` (8), `adapter.py` (4),
`subcircuit_composer.py` (3).

Per the plan, each is classified: **(P)** promote to a real, documented
mechanism · **(F)** replace with a loud failure + diagnostic · **(D)** delete if
dead · **(K)** keep — legitimate, just label/observe it.

## Categories (the 35 in placement_solver collapse to a few kinds)

| Kind | ~count | Examples (line) | Disposition |
|---|---|---|---|
| **Board-containment clamps** — "shift pads back inside the board outline" | ~15 | `_hard_clamp_*` (3566, 3740), step-10/12 clamps (1031–1059), per-step displacement clamp (3019) | **K** — physically real (a part outside the board is invalid). BUT: when a clamp *fires repeatedly* it's masking a placement that wants to sprawl past the edge. Action: count clamp activations per solve and surface as a diagnostic (`clamp_saves=N`); a high count is a finding, not a silent rescue. The "3 clamp passes" `WARNING` at 1059 should become a structured finding. |
| **Zone/edge clamps** — pin a coord into a zone rect | ~6 | 383, 789, 1236, 1849, 3704 | **K** — same as above; observe activation. |
| **SMT↔THT geometry clamps** | ~2 | 2919, 3032 | **K**. |

## Plan-named items (revisit individually)

| Item | Where | Disposition |
|---|---|---|
| Opening-direction 3-layer fallback (marker → body-extension → centroid) | `adapter.detect_opening_direction` (282) + the "no detectable opening" fallback (`placement_solver:1150`) | **P** — keep all three layers but make the *chosen* layer observable in diagnostics (which heuristic fired). Today a wrong centroid-guess is indistinguishable from a confident marker read. |
| Connector "assign to nearest edge" fallback | `placement_solver:1444` | **K→P** — fine as a default, but log when it overrides an explicit `edge:` zone (that's a model/extraction gap worth surfacing). |
| Keep-all rotation fallback | `subcircuit_composer._filter_rotations_for_connector_opening` (433) | **F (soft)** — already warns; promote the silent keep-all to a *structured* "unsatisfiable connector orientation" finding so it shows up in the run report, not just stderr. |
| `>10mm` anchor clamp (double-rebase history) | `compose_subcircuits._compute_final_outline` (962) | **D? — recheck after 2.2/2.3.** With frames centralized this clamp may be dead. Needs a parent-level corpus to verify deletion is safe (the leaf-only corpus won't catch it). |
| Best-effort persistence / promote-dirty-board | `_promote_verify_fab` (cli_app), leaf "persist best-effort" | **K** — earns its keep (a failed board must stay visible for inspection); already loud (rc 7/8). No change. |

## NEW finding from Lever 2.2 (convention bug — NOT a fallback, but logged here)

`parent_adapter._rotated` (62) and `placement_utils._world_artifact_origin` (41)
invert a placement to recover a child artifact's instance origin via
`origin = body_pos − rotate_CCW(body_center_offset, rotation)`. The forward
transform of `body_center` is **KiCad CW** (`transform_loaded_artifact` →
`_transform_point`), so the correct inverse subtracts
`rotate_CW(offset, +rotation)`, i.e. `rotate_vector(offset, +rotation)` — they
agree only at rotation ∈ {0, 180}. **At 90/270, a block with an off-origin body
center is recovered to an origin shifted by `(R_cw − R_ccw)·offset`** — a strong
suspect for Part 3's "edge connector landed several mm inboard / unrotated"
stranding. The evidence here is the math + the forward/inverse mismatch, not yet
a failing test (the leaf-only corpus and unit suite don't exercise a rotated
off-origin block through compose; building that case is a Part 3 task).

(NB: the pre-existing failing `test_best_round_to_layout_prefers_routed_board_geometry`
— `body_center` 2.025 vs 2.0 — is a *separate, unrelated* issue: a uniform
+0.025 shift on **both** x and y from the routed board's `-0.025` outline-origin
rebase, i.e. a translation, not this rotation skew. Not corroboration.)

**Disposition: F/fix — but as a separate, validated change.** It is *preserved
exactly* by the Lever 2.2 no-op centralization (now `rotate_vector(v, -deg)` with
the caveat flagged inline at both sites). Fixing it changes the parent board, so
it needs a **parent-level** corpus snapshot (the current leaf-only corpus can't
see it) before/after — the natural first task of Part 3.
