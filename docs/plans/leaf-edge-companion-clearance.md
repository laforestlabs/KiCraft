# Leaf-level edge-companion clearance (keep parts off a connector-pinned edge)

**Status:** open investigation. Handoff for a follow-up agent.
**Goal (user, explicit):** *prevent components from being placed within ~1 mm of a board edge that has an edge-pinned connector on its leaf — solved at the LEAF level.*
**Origin:** KC-S8PC37 (`~/.kicraft/projects/1/67`, "ESP32 with 5x5 grid of 1515 RGB LEDs"). Build failed the verify gate with `illegal_routed_geometry`; the sole trigger was a `copper_edge_clearance` violation on **R8** (the USB-C CC2 pull-down): its GND pad copper sat **0.300 mm** from the board edge vs the **0.381 mm** rule (`min_copper_edge_clearance` in `kicraft/design/synthesis/kicad_pro.py:31`).

This was the third of three issues found in that build. The other two (antenna-keepout intrusions; starved thermal) are **fixed and shipped** on branch `fix/gnd-pour-antenna-keepout-and-solid-thermal` (commit `d8dfe8b`). This doc is **only** about the edge-clearance / companion-placement issue, which is NOT yet fixed.

---

## 1. The failing geometry (measured)

Cross-run scan: `copper_edge_clearance` is the trigger in **9/9** `illegal_routed_geometry` rejections in the corpus, on a different ordinary part each time (R8, R2, R1, D1, R4, R5, U4) — i.e. a systematic "companion part too close to a connector edge" class, not a footprint bug.

For KC-S8PC37, R8 lives on the **USB_POWER** leaf with **J1** (USB-C, `usb-c-16p:USB-C_SMD-TYPE-C-31-M-12_1`). Measured on the re-solved leaf vs the composed parent (all mm):

| | leaf left edge | J1 mouth (outboard face) | J1 inboard face | R8 left face |
|---|---|---|---|---|
| **LEAF** (`leaf_pre_freerouting`) | -0.03 | -0.46 (overhang 0.43) | 10.03 | **2.17** (2.2 mm clear of leaf edge) |
| **PARENT** (`parent_routed`) | 120.38 | 117.80 (overhang **2.58**) | 128.30 | **120.40** (≈ at the board edge) |

The smoking gun: **R8 is 2.2 mm clear of the *leaf* edge, but ≈0 mm from the *parent* edge.** The connector overhang grew from **0.43 mm (leaf)** to **2.58 mm (parent)**. Compose draws the parent edge at J1's *PCB-edge anchor* (≈ shell mouth + 2.58 mm shell overhang); R8 was placed only 2.2 mm inboard of the leaf edge, which is *outboard* of that deeper parent anchor → R8 lands on the board edge.

**Conclusion: a keep-out measured from the loose leaf-bbox edge does NOT translate to parent-edge clearance.** The leaf and compose disagree on where the connector's board edge is.

---

## 2. Why the naive leaf-keepout band failed (what was tried & reverted)

Implemented and then **reverted** (kept the tree clean; see git history of this branch if you want the diff): a `_add_edge_keepout_bands()` pass in `PlacementSolver` that, for each board side carrying an edge-pinned connector, added a `KeepoutRect` strip and let the existing Step 9.2 `_resolve_keepout_rects` push companions inboard.

Two iterations, both verified ineffective by replay (R8 stayed ~0):

1. **Fixed 1 mm strip from the leaf bbox edge** — R8 was already 2.2 mm clear of the leaf edge, so the band never touched it. The 1 mm leaf margin doesn't survive the 2.58 mm parent overhang anyway.
2. **Strip reaching the connector's inboard face + margin** (so companions land inboard of the whole connector) — better intent, but:
   - It fired during **parent** block placement too, where the edge-pinned "part" is a whole leaf *block* (~19 mm deep) → reserved a band clamped to 25 % of the board, distorting the parent. (Patched by excluding `kind == "subcircuit"`, but…)
   - On the actual USB_POWER leaf, R8 *still* wasn't pushed. Suspected causes (unconfirmed — needs instrumentation): the band is built from `self.state.board_outline` at Step 1.1 which may be a **seed** outline, not the final tight leaf outline; and/or the leaf solver runs multiple rounds and **picks the best by a score that doesn't penalize "companion outboard of the connector anchor,"** so a compliant round is discarded.

The deeper problem under both: the band was expressed relative to the **leaf bbox edge**, but the quantity that matters is distance from the **connector's PCB-edge anchor** — the exact line compose will turn into the board edge.

---

## 3. The mechanism, by file:line

- **Leaf solve entry:** `kicraft/autoplacer/brain/placement_solver.py` → `PlacementSolver.solve()` (~line 660). Edge connectors pinned in `_pin_edge_components()` (~1167); `edge_groups: dict[side, [ref]]` built ~1433–1459 (explicit `component_zones[ref]["edge"]` + the `kind=="connector"` nearest-edge fallback). Keep-out push in Step 9.2 `_resolve_keepout_rects` (~2338) → `_push_out_of_rect` (~2255; skips `locked` and `owner_ref`, prefers an on-board exit). `KeepoutRect` is `types.py:298` (`tl, br, owner_ref, source`).
- **Leaf gets the connector's edge zone:** `kicraft/autoplacer/brain/leaf_size_reduction.py` → `local_solver_config()` (~line 17). A connector with an explicit parent edge keeps it (line ~44); an unzoned connector gets a leaf-local **nearest-edge** zone (line ~76). So at leaf-solve J1 *is* edge-pinned — good — but see the overhang mismatch below.
- **Connector overhang (the mismatch):** `connector_edge_overhang_mm` / `connector_edge_inset_mm` (`config.py` ~102–108, defaults 2.5 inset / 0.5 overhang). The leaf placed J1 with ~0.43 mm overhang; compose draws the edge with ~2.58 mm. Find where compose derives the connector PCB-edge anchor and reconcile it with what the leaf uses.
- **Parent outline drawing (where the edge is finalized):** `kicraft/cli/compose_subcircuits.py` → `_compute_final_outline()` (line 723). For an **edge-constrained** side, `_resolve_min/_resolve_max` (lines ~799–820) **pin the outline to the connector anchor `c_val` and refuse to expand past it** (else the connector un-flushes); an anchor >`spacing+2 mm` from geometry is rejected as a transform bug. `edge_zoned_outline_sides` computed at line ~1710. Connector edge-gap gate: `enforce_connector_edge_gap` / `connector_edge_gaps` (~2857) from `kicraft/autoplacer/brain/connector_edge_gap.py`.

**Why a compose-only fix is hard:** at compose R8 is inside a *routed* leaf block — you can't nudge R8 alone (breaks leaf routing), can't move the whole block (un-flushes J1), and can't redraw the outline outboard (buries the USB-C mouth). Hence the user's call: fix it at the **leaf** level so R8 is never placed outboard of the connector's PCB-edge anchor in the first place.

---

## 4. Proposed leaf-level approach (to pursue)

Make the leaf keep companions inboard of the **connector's PCB-edge anchor** — the same reference compose will use — not the loose leaf bbox edge. Concretely:

1. **Compute the per-side connector anchor at leaf-solve** the same way compose does (connector mouth ± `connector_edge_overhang_mm`/inset). This is the single most important fix: the leaf must reserve clearance from *that* line, not the leaf bbox edge. Reconcile the overhang so the leaf edge and the compose anchor coincide (the 0.43 vs 2.58 mm gap is the bug).
2. **Reserve the strip from that anchor to anchor + (clearance)** and push unlocked companions inboard of it (reuse `_push_out_of_rect`; the locked connector is exempt). Equivalently: enforce "the edge connector is the leaf extremity on its zoned side — nothing non-connector may be more outboard than `anchor + min_copper_edge_clearance`."
3. **Make it a hard constraint, not a soft score term**, or add a score penalty for companion-outboard-of-anchor — otherwise multi-round leaf selection can discard the compliant placement (suspected in iteration 2).
4. **Verify the band is built from the FINAL leaf outline**, not a seed `board_outline` captured at Step 1.1 (re-derive after the constraint-aware outline is known, or express the band relative to the connector's placed position rather than the outline).

Keep it config-gated (e.g. reuse a `connector_edge_companion_clearance_mm`, default ~1.0) and **leaf-only** (exclude `kind in ("mounting_hole","subcircuit")`).

Leave `min_copper_edge_clearance` at 0.381 mm — the point of this work is to satisfy it by placement, not relax it.

---

## 5. Verification harness

Deterministic, no LLM cost. Replay re-solves leaves in place (writes into the project's `generated/.../.experiments`, so snapshot first if you care):

```bash
PY=/home/kicraft/KiCraft/.venv/bin/python
# placement-only (fast, deterministic) — re-solves leaves + composes, no routing:
$PY -m kicraft.design.cli_app replay \
  --project /home/kicraft/.kicraft/projects/1/67/generated/ESP32_5X5_RGB \
  --quality fast --no-route /tmp/kc_replay

# Measure R8 vs the board edge on the composed parent:
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

Target: **R8 ≥ ~0.4 mm (ideally ≥1 mm) from the board edge**, J1 still flush/overhanging. Then route (`--quality good`, drop `--no-route`) and confirm `copper_edge_clearance == 0` via:
```bash
/usr/bin/kicad-cli pcb drc --format json --severity-error --output /tmp/drc.json \
  /home/kicraft/.kicraft/projects/1/67/generated/ESP32_5X5_RGB/ESP32_5X5_RGB.kicad_pcb
```
Also run the deterministic-placement corpus gate to catch regressions: `python scripts/replay_corpus.py` (leaf + parent modes).

**Useful debug facts:** the fresh USB_POWER leaf is the `*/leaf_pre_freerouting.kicad_pcb` whose footprints include both `J1` and `R8` (pick by mtime — replay rewrites leaf dirs each run). `quality good` replays run `--parents-only` (frozen leaves, no leaf re-solve) — use `quality fast` to actually exercise leaf placement.

---

## 6. Regression risk

This is the connector-edge / compose-outline area that has produced **repeated stranding regressions** (see `docs/plans/usb-c-edge-connector-stranding-three-bugs.md`, `place-route-root-cause-v2.md`, and the project memory entries on connector stranding). Pushing companions inboard can interact with: `edge_zoned_outline_sides`, `connector_edge_gap`, the anchor-slack clamp in `_compute_final_outline` (~797, `spacing+2`), and leaf size-reduction. **A/B every change against the replay corpus** (`scripts/replay_corpus.py`) and a few real boards before merging. Confirm the edge connector stays the leaf extremity and never strands inboard.
