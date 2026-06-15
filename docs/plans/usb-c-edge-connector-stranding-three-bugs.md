# USB-C edge connector stranded 8.8 mm inboard — three-bug root cause + fix plan

**Reference board:** `KC-S8PC37` (`~/.kicraft/projects/1/67`), brief "Esp32 with 5x5 grid of
1515 rgb leds". Build is **fab-ready** (rc=0, 0 shorts / 0 unconnected) yet the USB-C
connector **J1** sits **8.8 mm inboard of the left board edge** with empty FR4 between it
and the edge. The LLM zoned J1 correctly (`component_zones: {"J1": {"edge": "left"}}`); the
autoplacer/composer mis-placed it and no gate caught it.

This document is a self-contained handoff. It records the measured evidence, the **three
distinct bugs** that compound to produce the defect, and a concrete fix plan with
verification steps and coupling risks. Read the "Evidence" section before touching code —
two of the three bugs are non-obvious and one fix has a real regression risk.

Related prior work (memory / docs): `place-route-root-cause-v2.md`,
`kicraft-geometry-centralization-and-convention-bug`, `kicraft-usb-edge-connector-placement`,
`kicraft-parent-outline-anchor-double-rebase`. Bug 3 below is a **surviving instance** of the
exact KiCad-CW-vs-math-CCW convention bug that those efforts centralized into
`geometry.transform_point` — one helper was missed.

---

## TL;DR

| # | Bug | File | Effect |
|---|-----|------|--------|
| 1 | Leaf solve **clobbers** the connector's parent edge zone with a leaf-local nearest-edge guess | `autoplacer/brain/leaf_size_reduction.py` `local_solver_config` (~L47–79) | J1 is solved at the leaf's **top** edge, not its **left**; leaf must then be rotated 90° at compose |
| 2 | The connector edge **anchor** is transformed with the wrong rotation handedness (math-CCW) vs. the real placement (KiCad-CW) | `autoplacer/brain/subcircuit_composer.py` `_transform_local_point` (~L1425) → `_constraint_local_rect` → `_compute_local_anchor_offset` | For the 90°-rotated leaf the left anchor lands **~J1-height (8.3 mm) outboard** of J1's real mouth; `_compute_final_outline` snaps the board's left edge to that phantom anchor → the empty strip. **This is the proximate cause of the visible gap.** |
| 3 | No build-time gate measures connector mouth-to-edge distance | `autoplacer/brain/connector_edge_gap.py` `connector_edge_gaps` is **dead code** (only its unit test calls it) | A buried connector ships fab-ready with at most a warning |

Fix all three. Bug 2 is the one that actually creates the empty FR4; Bug 1 is what forces
the 90° rotation that triggers Bug 2; Bug 3 is why it shipped silently.

---

## Evidence (measured, reproducible)

All coordinates are real mm unless marked "local" (compose frame). Final board:
`~/.kicraft/projects/1/67/generated/ESP32_5X5_RGB/ESP32_5X5_RGB.kicad_pcb`.

### The defect
- Board outline: `x[112.57, 184.43] y[78.66, 131.34]` (71.9 × 52.7 mm).
- J1 (`TYPE-C-31-M-12`) bbox: `x[121.37, 129.76]` → **8.80 mm** from the left edge.
- J1 is the **leftmost object on the entire board** — no footprint and no copper is in the
  8.8 mm strip (leftmost trace is at x=121.82, to J1's *right*).
- Empty margin per side: **LEFT 8.80**, RIGHT 2.18, TOP 3.27, BOTTOM 2.04 mm. The gap is
  **left-only** — exactly the side J1 is zoned to. So this is not uniform packing slack; the
  left edge was deliberately pushed out.

### Bug 1 — leaf solve places J1 at the wrong leaf edge
- `..._autoplacer.json` → `component_zones: {"J1": {"edge": "left"}}` (correct).
- USB POWER leaf's own mini-board
  (`.experiments/subcircuits/bffd8f18-..__3056205546/round_0003_leaf_routed.kicad_pcb`,
  outline 24.1 × 14.4): J1 is **1.27 mm from the leaf's top**, **7.59 mm from its left**.
- `leaf_size_reduction.local_solver_config` (L47–79) iterates every `kind=="connector"` and
  does `local_component_zones[ref] = {"edge": nearest_edge}`, recomputing the nearest leaf-board
  edge and **overwriting** the parent's `left`. J1's nearest leaf edge was `top` → zone
  rewritten `left → top`. The leaf solver (`placement_solver.py`, reads `cfg["component_zones"]`)
  then placed J1 flush at the leaf **top**.
- Confirmed `J1 kind == connector` in the leaf `debug.json`, so the override branch fires.

### Bug 2 — anchor rotation-convention mismatch (the proximate cause)
- Compose places the leaf as a rigid unit at `origin=(-7.19, 41.48)` local, `rotation=90°`
  (from `round_0003/parent_pipeline.json` entry for "USB POWER"). Net effect rotated J1's
  mouth to face board-left — **J1's real mouth ends up at the left, x_local ≈ −6.7**.
- The board's left edge is at `x_local = −15.52` — i.e. **8.8 mm outboard of J1's real mouth.**
- `_compute_final_outline` (`compose_subcircuits.py` L794–828) **snaps** an edge-constrained
  side to the connector anchor `c_val`, only rejecting it if it is more than
  `anchor_slack_mm = spacing_mm + 10` (≈ 11 mm) from the placed geometry. Our error is 8.8 mm —
  **under the threshold** — so it snaps the left edge to the phantom anchor (L804 `return c_val`)
  and bakes in the bare-FR4 strip (the outline repair pass only ever grows, never shrinks).
- The anchor is `placement.origin + _compute_local_anchor_offset(...)`
  (`_resolve_constraint_anchor_positions`, `compose_subcircuits.py` L418+). The offset comes
  from `_constraint_local_rect` → `_transform_rect` → **`_transform_local_point`**
  (`subcircuit_composer.py` L1425), whose 90°/270° branches are **math-CCW**:

  ```python
  # subcircuit_composer.py  _transform_local_point  (WRONG: math CCW)
  if abs(rotation - 90.0)  < 1e-9: return Point(-point.y + origin.x,  point.x + origin.y)  # (x,y)->(-y, x)
  if abs(rotation - 270.0) < 1e-9: return Point( point.y + origin.x, -point.x + origin.y)  # (x,y)->( y,-x)
  ```

  The **ground-truth placement** uses KiCad-CW (`subcircuit_instances.py:816 _transform_point`
  → `geometry.transform_point`):

  ```
  x' = x·cosθ + y·sinθ ;  y' = -x·sinθ + y·cosθ      # 90°: (x,y)->( y,-x)
  ```

  These are **opposite handedness**. Empirical proof (J1 body_center local (10.50, 4.245),
  origin (-7.19, 41.48), 90°; real-frame offset +128.09 x / +74.696 y):

  | transform | predicted J1 real | actual J1 real |
  |---|---|---|
  | anchor's `_transform_local_point` (CCW) | (116.66, **126.68**) | (126.59, **105.67**) |
  | placement KiCad-CW (`geometry.transform_point`) | (125.15, **105.67**) | (126.59, **105.67**) |

  The placement convention matches the real board to **0.00 mm in y**; the anchor convention is
  off by ~21 mm. For the 90° leaf the left anchor is therefore reflected across J1's body and
  lands ~J1-height (≈ 8.3 mm, J1 height = 7.44 mm) outboard — matching the 8.8 mm gap (8.3 mm
  anchor error + ~0.5 mm overhang). At 0°/180° the handedness error does not produce a large
  offset, which is why **only rotated connector leaves strand** (and why Bug 1, which forces the
  90° rotation, is a prerequisite for the visible gap on this board).

  `_transform_point` even documents that this exact "math CCW" bug previously caused intra-leaf
  shorts and was fixed by centralizing into `geometry.transform_point`. `_transform_local_point`
  is the one helper that was **not** migrated.

### Bug 3 — no enforced edge gate
- `connector_edge_gap.connector_edge_gaps()` computes the signed mouth-to-edge gap and flags
  `gap < -0.5 mm` as stranded (for J1 it would report **−8.8 mm**). It is imported **only** by
  `tests/test_connector_edge_gap.py` — never in the build path. `subcircuit_composer.py:488`
  even claims compose defers stranding detection to "the post-compose gate (connector_edge_gap)",
  which never runs.
- The checks that *do* run are insufficient: `misoriented_connectors`
  (`compose_subcircuits.py` ~L1505) is orientation-only and warning-only;
  `_filter_rotations_for_connector_opening`'s give-up branch is warning-only;
  `_edge_zoned_is_leaf_extremity` compares J1 only against **sibling components** (J1 *is* the
  leftmost component, so it passes) and never measures J1 vs. the board edge. DRC (the fab-ready
  gate) has no edge-mount concept.

---

## Fix plan

Implement in this order. Each fix is independently valuable; together they close the class.

### Fix A — Bug 2 (anchor convention). HIGHEST PRIORITY, has coupling risk.

**Goal:** the connector edge anchor must be computed with the **same** rotation convention the
leaf is actually placed with (KiCad-CW, `geometry.transform_point`).

**Change:** in `kicraft/autoplacer/brain/subcircuit_composer.py`, make `_transform_local_point`
delegate to the canonical implementation instead of its hand-rolled math-CCW branches:

```python
from kicraft.autoplacer.brain import geometry  # already importable in this module

def _transform_local_point(point: Point, origin: Point, rotation_deg: float) -> Point:
    return geometry.transform_point(point, origin, rotation_deg)
```

This automatically fixes `_transform_rect`, `_constraint_local_rect`, the `edge_marker` path
(L641), and `_compute_local_anchor_offset` — everything that feeds the outline anchor.

**⚠ Coupling risk (must verify, do not skip):** `_transform_local_point` is **also** used for
**overlap detection** — `_transform_rect` (L1452) feeds `_any_rect_overlap` (L1482–1484) and the
`can_overlap_sparse` blocker path. If overlap/legality logic has been implicitly calibrated
against the buggy CCW behavior (the same trap that made a prior convention flip overlap leaf
bboxes — see `kicraft-geometry-centralization-and-convention-bug`), flipping the convention can
change which placements are considered overlapping. Mitigations:
1. Audit every caller of `_transform_local_point` / `_transform_rect` (only ~6 sites, all in
   `subcircuit_composer.py`). For overlap callers, confirm they pass the **same** `rotation_deg`
   the leaf is placed at — if so, correcting the convention makes overlap geometry *more*
   correct, not less.
2. Run the **deterministic A/B compose harness** `scripts/ab_compose.py` and the parent
   determinism corpus (`scripts/replay_corpus.py`, parent mode) before/after. Expect: the
   connector boards change (strand → flush), non-connector boards byte-identical. Any
   *non-connector* board that changes overlap/placement is a coupling regression to investigate.
3. Keep the `PYTHONHASHSEED` / thread pins the replay harness already uses for determinism.

**Expected result on KC-S8PC37 alone:** even with Bug 1 unfixed, the left anchor moves to J1's
real mouth (~−6.7 local) → board left edge ≈ J1 − overhang → J1 lands ~flush. Verify by
rebuilding (below); LEFT margin should drop from 8.80 mm to ≈ overhang (~0.5 mm).

### Fix B — Bug 1 (leaf zone clobber). Removes the burial at the source.

**Change:** in `kicraft/autoplacer/brain/leaf_size_reduction.py` `local_solver_config`, do not
overwrite a connector that already carries an explicit parent edge zone; only derive nearest-edge
as a fallback for parent-**unzoned** connectors:

```python
parent_zones = base_cfg.get("component_zones", {}) or {}
for ref, comp in extraction.local_state.components.items():
    if comp.kind != "connector":
        continue
    parent_edge = (parent_zones.get(ref) or {}).get("edge")
    if parent_edge in ("left", "right", "top", "bottom"):
        # Parent zone is authoritative for an off-board connector: it states which
        # FINAL-board edge the connector mates at. Keep it so the leaf solver places
        # the connector at that leaf edge and compose mounts it flush. (local_component_zones
        # already carries the copied parent spec, so this is effectively "do not clobber".)
        local_component_zones[ref] = {"edge": parent_edge}
        continue
    # ...existing nearest-edge derivation for parent-unzoned connectors...
    local_component_zones[ref] = {"edge": nearest_edge}
```

Rationale: the override's own comment justifies nearest-edge for **non-connector** alignment
groups (parallel batteries) where the parent's axis hint may not match the reduced leaf board —
but it is applied inside the **connector** branch, where the parent edge is precisely what we
must honor. With Fix B, J1 is solved at the leaf's left edge as its extremity; the leaf needs
little/no rotation, sidestepping Bug 2 for this board and making the whole edge-mount pipeline's
"connector is at its leaf's zoned edge" invariant hold.

Strictly safer: parent-unzoned connectors keep today's behavior.

### Fix C — Bug 3 (wire the gate) + tighten the outline clamp. Defense-in-depth.

1. **Enforce `connector_edge_gaps`.** After compose+route, call
   `connector_edge_gaps(parent_board_path, component_zones)` and turn `stranded(gaps)` into a
   real round-rejection reason (surface into `routed_validation.rejection_reasons` /
   `geometry_validation`), so the hierarchical round search rejects a stranded placement and
   retries instead of shipping it. Natural home: alongside `geometry_validation` in
   `compose_subcircuits.py` (post-`_compute_final_outline`, or post-route on the stamped board).
   The metric already exists and is unit-tested (`tests/test_connector_edge_gap.py`); this is
   wiring + a rejection path, not new geometry.
2. **Tighten `_compute_final_outline`'s anchor sanity clamp.** `anchor_slack_mm = spacing_mm + 10`
   (~11 mm) is far too loose — a *flush* connector anchor sits within ~1–2 mm of the placed
   geometry edge on its side. Reduce the edge-side slack (e.g. to `spacing_mm + 2`, or compare
   against the connector's overhang) so an anchor more than a couple mm from geometry falls back
   to `geometry ± spacing` and prints the existing `[outline] … ignoring anchor` diagnostic. This
   would have caught the 8.8 mm error directly. (Note: this is a backstop; with Fix A the anchor
   is correct and the clamp won't trigger.)

---

## Verification

1. **Unit:** existing `tests/test_connector_edge_gap.py` covers the gap math. Add a regression
   that composes a connector leaf at 90°/270° and asserts the final board's mouth-to-edge gap is
   within `[-inboard_tol, +max_overhang]` (would fail pre-Fix-A, pass after).
2. **A/B determinism:** `scripts/ab_compose.py` + `scripts/replay_corpus.py` (parent mode)
   before/after Fix A. Connector boards change strand→flush; non-connector boards byte-identical.
   Investigate any non-connector delta (coupling).
3. **Rebuild KC-S8PC37 end to end** (`~/.kicraft/projects/1/67`) and re-measure:
   ```
   python - <<'PY'
   import pcbnew
   b = pcbnew.LoadBoard(".../generated/ESP32_5X5_RGB/ESP32_5X5_RGB.kicad_pcb")
   e = b.GetBoardEdgesBoundingBox(); bx0 = e.GetLeft()/1e6
   j1 = next(f for f in b.GetFootprints() if f.GetReference()=="J1")
   print("J1 left gap:", j1.GetBoundingBox(False,False).GetLeft()/1e6 - bx0)  # expect ~ -0.5..+0.5
   PY
   ```
   Expect J1 left gap to drop from **8.80 mm** to ≈ overhang (`connector_edge_overhang_mm`=0.5).
   Re-run `connector_edge_gaps` on the rebuilt board: J1 gap should be ~0 and `ok=True`.
4. **No DRC regression:** rebuilt board must remain 0 shorts / 0 unconnected.
5. **Self-eval sweep** (a few connector-bearing briefs) to confirm no new stranding/overlap and
   stable scores.

## Notes / gotchas
- Don't reason about the rotation by hand — conventions bite. Validate against the **real board
  position** (the table above is the template: predict J1 with both conventions, compare to the
  placed footprint).
- The fab-ready gate is DRC-only; it will keep passing buried connectors until Fix C lands. Until
  then a board can be `fab_ready` with an unmateable port.
- `_transform_local_point` flip is one line but the audit is the work. Treat Fix A as
  "1-line change + full A/B determinism pass", not a drive-by.

---

## Implementation outcome (2026-06-15)

All three fixes implemented. Verified deterministically ($0, no LLM).

**Fix A — `subcircuit_composer._transform_local_point` now delegates to
`geometry.transform_point`** (added `from . import geometry`). Confirmed it now equals the
KiCad-CW placement convention for every rotation, and matches the plan's empirical J1 table
(local→world (125.15, 105.68), real J1 y = 105.67).
- *Coupling audit result:* the `_any_rect_overlap` callers are reached only for the
  opposite-layer **stacking** case (the same-side-copper-commit check short-circuits first,
  geometry-independently), and the two conventions are **identical at 0°/180°** — they differ
  only at 90°/270°. So the blast radius is narrow. The deterministic corpus
  (`scripts/replay_corpus.py --mode parent`) confirms **zero** placement change from this edit:
  `PARENT_LOCAL_CONN` matches golden; `USB_PD_TRIGGER` is byte-identical **before vs. after**
  the edit (verified by stashing). The `USB_PD_TRIGGER` "DRIFT vs golden" the corpus prints is
  **pre-existing on `main`** (stale committed golden) — it reproduces identically with the edit
  stashed, so it is not attributable to this change and the golden should be refreshed separately.

**Fix B — `leaf_size_reduction.local_solver_config`** now `continue`s past the nearest-edge
override for any connector that carries an explicit parent `edge` zone (left/right/top/bottom);
parent-unzoned connectors keep today's nearest-edge fallback. Not exercised by the frozen-leaf
corpus (it runs at leaf-solve time), but it is a strictly-narrowing, parent-zone-honoring change.

**Fix C2 — `_compute_final_outline` anchor clamp `spacing_mm + 10` → `spacing_mm + 2`.** Unit
test (`test_compute_final_outline_rejects_far_outboard_edge_anchor`) proves an 8.3 mm-outboard
phantom anchor is now ignored (falls back to geometry − spacing) while a flush anchor is still
honored exactly.

**Fix C1 — connector edge-gap gate wired into the build path** as a **pre-route early-bail** on
the stamped board (stranding is a placement property; measuring pre-route saves ~200 s), mirroring
the `stamp_shorts` guard. A genuinely inboard connector becomes a `connector_stranded:<ref>@…mm`
rejection so the autoexperiment search retries another placement.
- *Deviation from the plan, deliberate:* the multi-round search has **no best-effort promotion**
  for parent compose (`autoexperiment.py` discards a rejected round; if *all* rounds are rejected
  the build yields no parent artifact). To avoid converting borderline placements into
  whole-build failures, the gate is **config-gated** (`enforce_connector_edge_gap`, default on),
  uses a **generous inboard tolerance** (`connector_edge_inboard_tol_mm`, default 1.0 mm vs. the
  metric's 0.5 mm), rejects **inboard stranding only** (not the rarer excessive-overhang case),
  and is wrapped so a pcbnew hiccup can never invent a new failure mode.
- Confirmed on the real defect: `connector_edge_gaps` reports `J1 edge=left gap=-8.88mm ok=False`
  on `KC-S8PC37`'s shipped board, so the gate would have rejected it.

**Tests:** new `test_transform_local_point_*` (convention guard) +
`test_compute_final_outline_rejects_far_outboard_edge_anchor` (clamp). Full edited-subsystem
sweep: 193 passed, 0 new failures (4 failures pre-exist on `main`, unrelated). E2E connector
fixtures (`KICRAFT_REPLAY_E2E=1`): 14/14 — J1/J2/SW1/J3 stay flush.

**Remaining (needs spend):** a full end-to-end LLM rebuild of `KC-S8PC37` to watch J1 go
8.8 mm → flush is the only confirmation not coverable for free (its `.experiments/` frozen leaves
were already cleaned, so a deterministic re-compose isn't possible). The math proof + clamp test
+ gate-on-real-board cover the mechanism; the rebuild would be the live confirmation.
