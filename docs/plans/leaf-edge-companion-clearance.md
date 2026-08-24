> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# Leaf-level edge-companion clearance (keep parts off a connector-pinned edge)

**Status:** PLACEMENT FIXED + verified; routed confirmation OUTSTANDING (capped
replay didn't finish routing this dense board). Two parts: (a) the §3 leaf clamp
keeps R8 behind J1's pads; (b) a parent pass (`_ensure_edge_blocks_extremal`)
makes the edge-zoned block the board extremity so J1 stays flush. Verified in ONE
clean run: R8 0.30→10.82 mm, J1 mouth gap +0.48 mm (flush, the leftmost part), no
stranding, compose succeeds (placeable parent). Routing: `fast` replay left
`unconnected_nets` (single-pass effort); `good` replay hit a 595 s timeout
(`parent_route=fail`) — both are route-completeness/effort issues on a dense
board that was never fab-ready, NOT placement regressions (compose produced a
parent in every run; no stranding). Needs a proper routed A/B (baseline `main`
vs this change, uncapped) before merge. Compose IS rigid (§6 was a cross-run
contamination ERROR — see §7→§8). §8 has the working solution; §2 the first wrong
diagnosis.
**Goal (user, explicit):** *prevent components from being placed too close to a board edge that has an edge-pinned connector on its leaf — solved at the LEAF level.*
**Origin:** KC-S8PC37 (`~/.kicraft/projects/1/67`, "ESP32 + 5x5 RGB LEDs"). Build failed verify with `illegal_routed_geometry`; the sole trigger was a `copper_edge_clearance` violation on **R8** (USB-C CC2 pull-down): its GND pad copper sat **0.300 mm** from the board edge vs the **0.381 mm** rule (`min_copper_edge_clearance`, `kicraft/design/synthesis/kicad_pro.py:31`). The other two issues from that build (antenna keepout, starved thermal) are already fixed+merged+deployed (`71a88b2`). This doc is **only** about the edge-clearance / companion-placement class. `copper_edge_clearance` is the trigger in **9/9** corpus `illegal_routed_geometry` rejections (R8, R2, R1, D1, R4, R5, U4) — systematic, not a footprint bug.

---

## 1. Verified mechanism (measured on the board)

The parent board edge for an edge-zoned connector is **not** a function of the leaf's outline. Compose **re-derives** it from the connector's anchor:

- `kicraft/cli/compose_subcircuits.py::_compute_final_outline()` (~723) calls `constraint_aware_outline(..., constrained_ref_world_anchors=anchor_positions)`. It uses the connector **anchor**, not the leaf `Edge.Cuts`.
- The anchor is computed in `kicraft/autoplacer/brain/subcircuit_composer.py`:
  - `derive_attachment_constraints()` (~306–328): a "mouthed" connector (USB-C, `opening_direction` set) gets `inward_keep_in_mm=0`, `outward_overhang_mm = connector_edge_overhang_mm` (default **0.5**).
  - `_compute_local_anchor_offset()` (~595–654): anchor = the connector's edge-facing line — a `"PCB Edge"` fp marker (`_find_edge_reference`, `edge_reference_points`, ~1705) if present, else the **courtyard/bbox edge** on the constrained side (`_constraint_local_rect`).
  - `constraint_aware_outline()` (~2215–2229): `outline_edge = anchor − inward + outward` ⇒ for a left edge, `board_edge.x = anchor.x + 0.5`.

**Measured on KC-S8PC37 (`parent_routed`):**

| feature | x (mm) | note |
|---|---|---|
| J1 silk/shell bbox left | 117.78 | metal shell overhangs the board |
| **board left edge** | **120.38** | = courtyard anchor 119.88 **+ 0.5** overhang |
| **J1 leftmost PAD copper** | **121.73** | 1.35 mm **inboard** of the board edge |
| **R8 left pad copper** | **120.40** | ≈ at the board edge — and **outboard of J1's own pads** |

**The actual bug:** R8 is parked in the connector's **courtyard apron** — between J1's pads (121.73) and the board edge (120.38). The connector is correctly inset at leaf-solve (`connector_edge_inset_mm`, default 1.0); nothing keeps a companion from drifting into the apron *outboard of the connector's pads*. Intra-block geometry is preserved leaf→parent, so R8's apron position becomes the edge violation.

---

## 2. Why the first attempt (a leaf-bbox keepout band) was wrong — DO NOT repeat

The reverted `_add_edge_keepout_bands()` pass added a `KeepoutRect` relative to `self.state.board_outline` and let Step 9.2 push companions inboard. It failed because:

1. **`self.state.board_outline` during solve is the SEED envelope** (set at extraction in `subcircuit_extractor.py` ~249, never updated; the tight outline is recomputed *post*-solve in `leaf_routing._outline_around_geometry`, ~89). A band in the seed frame corresponds to nothing in the final geometry.
2. **The leaf `Edge.Cuts` does not feed the parent edge** — compose re-derives from the connector anchor (§1). So "reconcile the leaf outline / overhang" (the original framing) targets something compose ignores.
3. **Leaf round-selection** picks best by `PlacementScore` (`types.py` ~390; `edge_compliance` weight only 0.10, no hard "connector is the extremity" term), so a compliant round can be discarded.
4. The original "overhang mismatch (0.43 vs 2.58 mm)" framing compared the leaf tight-bbox edge against the connector's silk/shell bbox — **unrelated references**. The real edge is `courtyard anchor + 0.5`.

Conclusion: the constraint must be **connector-relative**, enforced as a **hard, final** step, and must **not** rely on `board_outline` or the leaf outline.

---

## 3. Plan: make the connector's pads the outboard limit for its companions

**Principle:** on a side carrying an edge-pinned connector, **no companion's copper may extend more outboard (toward that edge) than the connector group's outboard-most pad face on that side.** Because the parent edge is drawn at the connector's courtyard anchor + 0.5 mm — i.e. *outboard* of the connector's pads — a companion kept behind the connector's pads is automatically clear of the parent edge (R8: 120.40 → ≥121.73 ⇒ 1.35 mm clearance ≫ 0.381). Uses only `Component.pads` (already in the leaf solver) — **no marker/courtyard plumbing**, no `board_outline` dependency, immune to round-selection.

**Steps (all in `kicraft/autoplacer/brain/placement_solver.py` + one config key):**

1. **Config** `connector_edge_companion_clearance_mm` (default `0.5`) in `kicraft/autoplacer/config.py`. `0` disables. Leaf-only (skip `kind in {"mounting_hole","subcircuit"}` so it never fires on parent block placement).
2. **`_edge_pinned_groups: dict[side, [connector_ref]]`** built in `_pin_edge_components()` from the existing `edge_groups` (explicit `component_zones[ref]["edge"]` + the `kind=="connector"` nearest-edge fallback), excluding mounting holes / subcircuit blocks. The connector is pinned+locked at Step 1, so its pads are fixed before companions move.
3. **A hard final step** (after Step 13 pinned-restore + its overlap re-resolve — NOT a mid-pipeline band): for each zoned side, take the connector group's outboard-most **pad** face on that side; push every unlocked, non-connector component whose outboard pad face is beyond it inboard to `connector_pad_face ± clearance`. Then `_resolve_overlaps` and repeat, bounded ~3× (the proven Step 9.2 loop shape). Connector-relative ⇒ seed-outline-proof; final+hard ⇒ round-selection-proof.
4. Leave compose, the outline math, and the leaf `Edge.Cuts` recompute untouched.

**Edge cases to encode:** multiple connectors on a side → outboard-most pad among them; a card-edge connector whose pads sit *at* the edge → "inboard of pads" gives little clearance (note it; optional future hardening = also clamp to the courtyard anchor, which needs the marker/courtyard plumbing — defer until a card-edge case appears); a companion legitimately beside the connector at the same depth gets pushed inboard (acceptable).

---

## 4. Verification harness

Deterministic, no LLM. `replay --project` writes IN-PLACE into the project's `generated/.../.experiments` (snapshot first if you care). Use `--quality fast` to re-solve leaves; `good` runs `--parents-only` (frozen leaves, no leaf re-solve).

> ⚠️ **CRITICAL — never compare artifacts across separate replay runs.** Each `replay` regenerates `.experiments` (new leaf uuids, fresh `solved_layout.json`, fresh promoted parent) and leaf placement is seed/state-dependent, so a leaf from run N vs a parent from run M is meaningless. Measure the leaf AND the parent in ONE script immediately after a SINGLE replay. (Verified: within one run, compose stamps leaves perfectly rigidly — Kabsch RMS 0.00 mm. A cross-run comparison once produced a bogus "non-rigid / parent re-places leaves" RMS ~7 mm and sent this investigation down a wrong path.) See memory `kicraft-replay-cross-run-contamination`.

```bash
PY=/home/kicraft/KiCraft/.venv/bin/python
$PY -m kicraft.design.cli_app replay \
  --project /home/kicraft/.kicraft/projects/1/67/generated/ESP32_5X5_RGB \
  --quality fast --no-route /tmp/kc_replay
# measure R8 + J1 vs board edge on the composed parent:
$PY - <<'EOF'
import pcbnew
b=pcbnew.LoadBoard("/home/kicraft/.kicraft/projects/1/67/generated/ESP32_5X5_RGB/ESP32_5X5_RGB.kicad_pcb")
bb=b.GetBoardEdgesBoundingBox(); L,T,R,Bo=(pcbnew.ToMM(bb.GetX()),pcbnew.ToMM(bb.GetY()),pcbnew.ToMM(bb.GetRight()),pcbnew.ToMM(bb.GetBottom()))
for fp in b.GetFootprints():
    if fp.GetReferenceAsString() in ("R8","J1"):
        g=min(min(pcbnew.ToMM(p.GetBoundingBox().GetX())-L, R-pcbnew.ToMM(p.GetBoundingBox().GetRight()),
                  pcbnew.ToMM(p.GetBoundingBox().GetY())-T, Bo-pcbnew.ToMM(p.GetBoundingBox().GetBottom())) for p in fp.Pads())
        print(fp.GetReferenceAsString(), round(g,3), "mm to edge")
EOF
```

Target: **R8 ≥ ~0.4 mm (expect ~1.35 mm = behind J1's pads)**, J1 unmoved. Then route (`--quality good`, drop `--no-route`) and confirm `copper_edge_clearance == 0`:
```bash
/usr/bin/kicad-cli pcb drc --format json --severity-error --output /tmp/drc.json \
  /home/kicraft/.kicraft/projects/1/67/generated/ESP32_5X5_RGB/ESP32_5X5_RGB.kicad_pcb
```
Run `python scripts/replay_corpus.py` (leaf + parent) to catch stranding regressions. Add a `PlacementSolver` unit test: connector + a companion placed outboard of its pads → companion pushed behind the pads, locked connector unmoved.

The fresh USB_POWER leaf is the `*/leaf_pre_freerouting.kicad_pcb` whose footprints include both `J1` and `R8` (pick by mtime — replay rewrites leaf dirs each run).

---

## 5. Regression risk

The connector-edge area has produced repeated stranding regressions (see `docs/plans/usb-c-edge-connector-stranding-three-bugs.md`, `place-route-root-cause-v2.md`, and the connector-stranding memory entries). This plan is lower-risk than the reverted approach: it's connector-pad-relative, additive as a final clamp, and touches only `placement_solver.py` + one config key — no change to compose, the outline math, or `Edge.Cuts`. **A/B `scripts/replay_corpus.py` and a few real boards before merge.** Confirm the edge connector stays the leaf extremity and never strands inboard, and that pushing a companion inboard doesn't create an unresolved overlap (the bounded re-resolve loop must converge).

---

## 6. IMPLEMENTED — verified at the leaf level, but compose discards it (2026-06-15)

The §3 plan is implemented (`config.py` key; `placement_solver._clamp_companions_inboard_of_connectors` + `_edge_pinned_groups`; the call placed in `solve_subcircuits._solve_one_round` AFTER passive-ordering + legality-repair — NOT inside `solve()`, because those two post-`solve()` passes re-place companions and undo a clamp done earlier). Unit tests pass.

**Confirmed working at the leaf level** (KC-S8PC37 USB_POWER leaf, after the clamp):
- `solve()` exit: R8 pad-left 8.49 mm, J1 pad-left 7.99 mm → R8 **behind** J1's pads. ✓
- `leaf_pre_freerouting.kicad_pcb`: R8 pad x[8.49,11.34], J1 pad x[2.85,9.39] → R8 **+5.6 mm inboard** of J1. ✓
- `solved_layout.json` (what compose consumes): J1 pos (5.66,10.86), R8 pos (9.35,21.49) → R8 **+3.69 mm** in x of J1 (inboard). ✓

**But the composed parent puts R8 back outboard of J1:**
- parent: J1 pos (128.06,91.15) rot −90, R8 pos (125.09,98.80) → R8 **−2.97 mm** in x of J1 (OUTBOARD), 0.325 mm from the board edge. ✗

**Why this is a compose bug, not a placement bug:** J1's rotation is −90 on BOTH the leaf and the parent, so compose placed the leaf at rotation 0 — a pure translation should preserve R8's position relative to J1. It does not: R8's offset-from-J1 goes from (3.69, 10.63) in `solved_layout` to (−2.97, 7.65) on the parent. That is **non-rigid** (magnitude 11.25 mm → 8.21 mm), so compose is not applying one rigid leaf→parent transform to all of the leaf's components — R8 is repositioned relative to J1 somewhere in the stamp/compose path. Until that is understood, NO leaf-level placement can stick: the leaf already places R8 correctly and compose moves it back.

**Next investigation (compose side):** find where the parent gets R8's position. Trace `kicraft/cli/compose_subcircuits.py` + `_compose_stamp.py` + `subcircuit_composer.py` stamping: does it consume `solved_layout.json` components, `leaf_pre_freerouting.kicad_pcb`, or re-derive/re-place? Diff a single leaf's component coords (esp. R8 vs J1) at: (a) `solved_layout.json`, (b) the stamped `parent_pre_freerouting`, (c) the promoted parent. The step where R8's offset-from-J1 changes is the bug. Suspects: a per-component (not per-block) transform, a frame/anchor mismatch for companions vs the constrained connector, or the leaf being re-extracted/re-laid at parent time. This is the same connector-edge/anchor/transform area as `usb-c-edge-connector-stranding-three-bugs.md` and `place-route-root-cause-v2.md`.

**Open decision:** keep the (leaf-correct, config-gated) clamp while the compose transform is fixed, or revert it until then. It currently perturbs leaf placement without fixing the parent. `connector_edge_companion_clearance_mm: 0` disables it.

---

## 7. CORRECTION of §6, and the real blocker (2026-06-15, second pass)

**§6 was WRONG — cross-run contamination.** §6 claimed compose places leaf components non-rigidly ("compose discards the leaf placement", Kabsch RMS ~7 mm). That was an artifact of comparing a leaf from one `replay` run against a parent from a *different* run (`replay` regenerates `.experiments` + the promoted parent each invocation). Measured properly — leaf and parent from ONE clean run — **compose stamps the leaf perfectly rigidly: Kabsch RMS 0.00 mm, 0°, identical per-component rotations.** Leaves ARE frozen, exactly as designed; there is no parent-local re-placement of leaf components. (Lesson recorded in memory `kicraft-replay-cross-run-contamination`; see the ⚠️ in §4.)

**So the §3 clamp is the right idea and it DOES work for R8:** in a clean run the leaf clamp keeps R8 behind J1's pads and compose carries it through rigidly — **R8 went from 0.30 mm to ~6.9 mm off the edge, no part under the 0.381 mm rule.** Implemented in `solve_subcircuits._solve_one_round` (after passive-ordering + legality-repair) + `placement_solver._clamp_companions_inboard_of_connectors` + `_edge_pinned_groups` + `connector_edge_companion_clearance_mm`.

**But it introduced a NEW, intermittent regression: connector stranding (rc6).** Routing bails pre-route with `connector_stranded:J1@-1.32mm(left)` on some runs. Root cause is a pre-existing coupling the clamp perturbs into firing:
- The parent board edge is the union of ALL parts' mouths on that side, not J1's. Measured (clean run): the left edge was defined by **R4** (a non-USB 470 Ω resistor, mouth −0.02 mm) and **U4** (level shifter), with **J1 0.31 mm inboard** of it. J1 is edge-*zoned* but not guaranteed to be the edge *extremity* — other leaves' parts sit at/over the same edge.
- `connector_edge_gap.connector_edge_gaps` (gate at `compose_subcircuits.py:~2888`, `inboard_tol` ~1.0) flags J1 when its mouth (courtyard∪pads) is > tol inboard of the board edge. Moving USB_POWER companions inboard shifts parent placement enough that J1's inboard gap crosses the tol on some runs (−0.31 mm here, −1.32 mm in the routed run — run-variable).

So the edge-clearance fix is entangled with: (a) the parent edge being defined by non-connector parts of OTHER leaves crowding the connector's zoned edge, and (b) the stranding gate. Reverted the clamp (kept this doc) — it's not shippable as a standalone leaf change because it trades an intermittent rc7 (R8 edge clearance) for an intermittent rc6 (J1 stranding).

**Real fix scope (for a dedicated pass, single-run discipline mandatory):** make the edge-zoned connector the guaranteed *extremity* of its board edge — keep OTHER leaves' parts from crowding to/over a connector-zoned edge (so the board edge is drawn at J1's mouth, J1 flush, companions inboard). That's a parent-placement + outline + stranding-gate problem, not a leaf-only clamp. Options to explore: a parent-level edge band reserving the connector's zoned edge for the connector; or compute the board edge on a connector-zoned side from the connector's mouth alone (not the union with other leaves) and push everything else inboard of it. Verify R8 clear AND J1 flush AND no other part stranded, all in ONE run, then route + DRC, then `scripts/replay_corpus.py`.

---

## 8. Working solution (2026-06-15, third pass) — two parts

The edge-clearance/stranding problem needs BOTH a leaf fix and a parent fix, because the two failure modes live in different frames:

1. **Leaf clamp** — `kicraft/cli/solve_subcircuits._solve_one_round` (after passive-ordering + legality-repair) calls `placement_solver._clamp_companions_inboard_of_connectors`, which uses `_edge_pinned_groups` (set in `_pin_edge_components`) + the config key `connector_edge_companion_clearance_mm` (0.5). It pushes ordinary parts behind the connector's outboard-most PAD face on its zoned side. Compose stamps the leaf rigidly (§7), so this carries to the parent. → R8 0.30 → 10.82 mm off the edge.

2. **Parent block extremity** — `compose_subcircuits._ensure_edge_blocks_extremal` (called after `_slide_constrained_to_cluster`, gated by `connector_edge_block_extremity`=True) shifts each edge-zoned **block** OUTBOARD until it is the board extremity on its zoned side, so the connector defines the board edge and stays flush instead of another leaf's block edging past it and stranding the connector. Outboard-only → no overlaps, only a tiny board growth. → J1 mouth gap +0.48 mm (flush, the leftmost part); no `connector_stranded`.

**Why both:** the leaf fix alone strands the connector intermittently — removing R8 from J1's outboard side unmasks the fact that another leaf's block can sit a hair more outboard than J1's block, redefining the bbox-union board edge. The parent fix makes the connector's block the guaranteed extremity. (Verified together in one clean run; see Status.)

**Unit tests:** `tests/test_placement_keepout.py` (companion clamp: pushes behind pads, noop when inboard) and `tests/test_compose_unified.py` (`_ensure_edge_blocks_extremal`: shifts to extremity, noop when already extremity). 45 tests pass.

**Remaining before merge:** a routed A/B (uncapped) — baseline `main` vs this change at `good` quality — to confirm the dense parent still routes at least as well as baseline (it was never fab-ready; capped replay here left it `unconnected`/`not_routed`, which appears route-effort-bound, not placement-caused, since compose succeeds every run). Then `scripts/replay_corpus.py` for cross-design regressions, then route + `kicad-cli drc` for `copper_edge_clearance == 0`.
