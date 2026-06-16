# HANDOFF — LED-array work: tasks 4 (turn-hop routing) & 5 (adaptive decaps)

**Status as of 2026-06-16.** The first wave of LED-matrix array fixes is **merged to
`main` (`d3aadcf`) and deployed** (web + build-worker restarted). This doc hands off the
two remaining tasks. Read the whole thing before touching code — there is one harness
gotcha that will waste hours if you miss it.

Related memory: `kicraft-array-matrix-fixes.md` (the broader array saga, KC-NESCCB /
KC-FFFADA / KC-BUCJZ4 / KC-3QRJMX). User principle that governs everything here:
**NO FALLBACKS — fail loudly, never silently degrade** (`kicraft-no-fallbacks-fail-loudly`,
`kicraft-no-fallback-previews`).

---

## 0. The part under test

WS2812B-1313-V6 addressable RGB LED ("1313"/"1515" family), LCSC **C52941388**, now vendored
at `kicraft/parts_library/ws2812b-1515/`. Footprint `LED-SMD_4P-L1.3-W1.3-P0.80_WS2812E-1313`.

**Pinout / pad geometry (rotation 0, KiCad Y-down) — memorize this, both tasks depend on it:**

| pad | function | offset from body centre (mm) | corner |
|-----|----------|------------------------------|--------|
| 1 | VDD (+5V) | (-0.4, -0.4) | top-left |
| 2 | **DOUT**  | (-0.4, +0.4) | bottom-left |
| 3 | GND       | (+0.4, +0.4) | bottom-right |
| 4 | **DIN**   | (+0.4, -0.4) | top-right |

**DIN and DOUT sit on opposite *diagonal* corners.** This is the whole reason orientation
and routing are hard: no single rotation makes a DOUT→DIN hop purely horizontal.

Test boards:
- **KC-3QRJMX** (`projects/1/73`) — 5×5, *bulk* caps only (C1/C2 in the POWER INPUT leaf),
  no per-LED decaps. The clean case. Builds fab-ready. Use this for **task 4**.
- **KC-BUCJZ4** (`projects/1/72`) — 5×5 with a **per-LED decap each** (C2–C26). Currently
  rc6. Use this for **task 5**.

---

## 1. What is already DONE (merged `d3aadcf`) — do NOT redo

All in `kicraft/autoplacer/brain/array_placement.py` unless noted.

1. **Uniform per-row rotation** (`_orient_array_grid`, replaced the old `_orient_chain`).
   Computes the part's *intrinsic* DOUT corner (normalizes out incoming rotation) and rotates
   every member **to an absolute target**: even rows = R, serpentine odd rows = R+180,
   non-serpentine = single R. Result: **≤2 rotations, exactly one per row**, regardless of how
   members arrive. Driven by `ArraySpec.serpentine` (user wants serpentine to be a real choice:
   serpentine ⇒ flip every other row; non-serpentine ⇒ all same rotation + longer return hops).

2. **`place_array_leaves` fully_handled bug.** A *pure* array leaf (only the grid, no other
   parts) returned `fully_handled=False` (falsy empty `remaining` list) ⇒ `solve()` ran
   force/SA on the locked grid ⇒ SA refine rotated members + legalizer scattered the grid at
   tight pitch. Fixed: empty `remaining` ⇒ `return (placed, True)`, skip force/SA.

3. **Tight-pitch legality** (`leaf_geometry.repair_leaf_placement_legality`). `array_member`
   and `locked` are runtime flags that **do not survive a board serialize/reload**, so the
   legality re-check saw the tight 3 mm grid as a wall of overlaps and the legalizer scattered
   it. Fixed by re-establishing both flags from `cfg["arrays"]` before legalizing.

4. **`ArraySpec.pitch_mm` schema doc** (`kicraft/design/models.py`). Added `Field(description=…)`
   so the BOM JSON schema tells the model to set the requested pitch (e.g. "3 mm pitch" → 3.0).
   Was a bare code comment, invisible to the LLM ⇒ flaky pitch.

5. **3D model** — `ws2812b-1515` vendored (STEP+WRL), added via
   `kicraft add-part --from-lcsc C52941388 --into vendored --name ws2812b-1515`. `stage_3d_models`
   auto-copies it on build.

**Validated** end-to-end on the real build path: 5×5 grid at **exactly 3.00 mm**, **2 rotations
one-per-row**, legal (no `illegal_pre_stamp`), accepted, force/SA skipped. Tests in
`tests/test_array_placement.py` (16 pass). The 4 failing tests in the wider suite
(`test_build_zero_leaf`, `test_solve_subcircuits_layout_persistence`) are **pre-existing** —
they fail with this work stashed too.

---

## 2. ⚠️ HARNESS GOTCHA — `replay --project` does NOT feed `arrays`

`kicraft replay --project <dir>` reads the placement config **from the .kicad_pcb, not from
`<stem>_autoplacer.json`** (see the comment at `cli_app.py:2864` — autoplacer.json is "read only
by the UI panels, not the placer"). The `arrays` hint lives only in autoplacer.json. So under
`replay --project` the array leaf gets **`arrays=0`**, never grids, force/SA scatters it, and the
build *best-effort-promotes the scattered board as "fab-ready."* This is a silent-wrong-validation
that will make you think your code is broken when it isn't.

**Do NOT validate array work with `replay --project`.** Instead run the leaf solve directly with
`--config autoplacer.json` (this is what the real web build does):

```bash
WORK=/tmp/arr; rm -rf "$WORK"
cp -r /home/kicraft/.kicraft/projects/1/73/generated/1515_RGB_MATRIX "$WORK"
rm -rf "$WORK/.experiments" "$WORK/fab"; rm -f "$WORK"/*_fab_*.zip
# (optional) force a pitch into the frozen config:
python -c "import json;p='$WORK/1515_RGB_MATRIX_autoplacer.json';d=json.load(open(p));[a.__setitem__('pitch_mm',3.0) for a in d['arrays']];json.dump(d,open(p,'w'))"

.venv/bin/python kicraft/cli/solve_subcircuits.py \
  "$WORK/1515_RGB_MATRIX.kicad_sch" --pcb "$WORK/1515_RGB_MATRIX.kicad_pcb" \
  --config "$WORK/1515_RGB_MATRIX_autoplacer.json" --rounds 1 --seed 0 \
  --route --only "LED MATRIX"
```

Then inspect `"$WORK"/.experiments/subcircuits/<uuid>/leaf_routed.kicad_pcb` with pcbnew. Look
for `Grid-placed N array member(s); ... skipping force/SA` in stdout and confirm no
`illegal_pre_stamp*` files. (Pure placement check: drop `--route`, much faster.)

**Optional pre-req task: make `replay --project` faithful** by loading `autoplacer.json`'s
`arrays` into the placer config. This is a legitimate no-fallback fix (a scattered array board
should never silently "pass") and would let you use replay normally. Trace from `_cmd_replay`
→ `_layout_route_fab` (`kicraft/design/cli_app.py`) to where the autoexperiment/leaf-solve
`--config` is chosen.

---

## 3. TASK 4 — stamp serpentine row-turn data hops (board: KC-3QRJMX)

**Goal:** make the inter-LED data net **100 % algorithmic / repeating** — kicraft routes every
hop, freerouting routes none of the data.

**Current state.** `array_router.array_daisy_chain_specs`
(`kicraft/autoplacer/brain/array_router.py`) already stamps the **in-row** hops as clean locked
pad-to-pad ties (reusing `breakout_stubs.add_breakout_stubs`). Its foreign-pad guard **drops any
tie whose straight path crosses other copper** — which is exactly the **serpentine row-turn
hops** (last LED of a row → first LED of the next). Those get handed to freerouting, which routes
them as multi-segment via-laden paths (the "non-repeating" routing the user complained about).
On the old measured board: 18 in-row hops stamped clean, 6 turn hops + all power = freerouting,
25 vias.

**Geometry after the new uniform rotation (serpentine, R_even/R_odd = 0/180 or similar):**
- In-row hops ≈ 2.34 mm short diagonals in the inter-LED gap — already stamp fine.
- Turn hop = the last LED of a row and the first of the next are in the **same column, adjacent
  rows** (serpentine reverses the fill). A straight vertical tie between their DOUT/DIN is ~pitch
  long but **crosses the intervening LED's +5V pad** ⇒ guard drops it. Turns alternate edges:
  row0→1 at the right edge, row1→2 at the left edge, etc. (the turn LED is always in the row-end
  column, i.e. against a board edge — so there is always an edge channel to route into).

**What to build.** Extend `array_daisy_chain_specs` (or add a sibling) to detect turn hops
(consecutive chain members in different rows: `(i+1) % cols == 0` in chain order) and emit a
`BreakoutSpec` with an explicit **L/Z waypoint path into the adjacent board-edge channel**
instead of a straight tie:
`DOUT → out to channel_x (just past the row-end column's pads, clamped inside the board inner
box) → along the channel to the target row's y → back in to DIN`. Such a path clears all foreign
pads, so `add_breakout_stubs`'s guard keeps it.

**Key references in `breakout_stubs.py`:** `BreakoutSpec` (waypoints/near_xy), `add_breakout_stubs`
(stamps + guards), `_rect_perimeter_path` / `_nearest_on_rect` (route-around-a-rect helpers you
can mirror), `_board_inner_box_mm` (the clamp — off-board locked copper hangs freerouting 1.9.0),
`_pads_bbox_mm`, `_segment_clears_obstacles`. Wiring point: `leaf_routing.py:560-572`.

**Gotchas:** the turn route must stay inside `_board_inner_box_mm` (clamp `channel_x`); if the
board is too tight for an edge channel, fail loudly / log a skip (do NOT silently hand back to
freerouting without saying so). Validate with the §2 command (with `--route`) and confirm
`leaf_routed.kicad_pcb` has the turn nets (`DAISY_04,05,09,10,14,15`) as 1-2 clean locked
segments with **0 vias on the data net**, and `array-router`-owned coverage = all 24 hops.

**Alternative worth considering:** route turn hops on B.Cu with 2 vias each (via down at DOUT,
straight across the back, via up at DIN). Fewer pad-collision worries (LEDs are all F.Cu SMD),
fully repeating, but needs via support in the stamping path (BreakoutSpec is single-layer today).
The F.Cu edge-channel L-route is probably less invasive.

---

## 4. TASK 5 — adaptive per-LED decap placement (board: KC-BUCJZ4)

**Why the current behaviour is wrong.** `array_placement._place_companion_block` packs the
per-LED decaps in a tall block *below* the grid (electrically pointless — a bunch of caps far
from the LEDs they serve) and on KC-BUCJZ4 the block runs **off the bottom of the leaf outline**
(last cap C26 pad_outside) ⇒ leaf illegal ⇒ array leaf yields no artifact ⇒ parent never routes
⇒ rc6. The sibling KC-NESCCB shows the *overlap* variant of the same block.

**The user's decision (verbatim intent) — implement this adaptive rule:**

1. **DEFAULT: front-side, beside each LED.** Put each decap in the inter-LED gap next to its LED
   on the top layer (1 cap per LED cell, distributed in chain order — they are all VBUS↔GND so
   any 1:1 assignment is electrically equivalent).
2. **If the pitch fits the LED but NOT a cap beside it** (gap = pitch − led_extent < cap_extent +
   clearance): fall back to **either**
   - (a) arrange caps **along the array edge(s)**, **or**
   - (b) **drop most caps**, keeping only **1–2 larger bulk caps**, *when the current draw is low
     enough*.

**Important architectural note:** the placement layer **cannot drop parts** (they are in the
BOM/netlist; deleting them orphans nets). So option 2(b) "drop" is a **synthesis/BOM-stage
decision**, not a placement one. Split the work:
- **Placement side** (`array_placement.py`): default front-side-beside, else edge-rows. This is
  the concrete, unit-testable piece and directly fixes the KC-BUCJZ4 rc6 + KC-NESCCB overlap.
  Replace/augment `_place_companion_block`.
- **Synthesis side** (BOM stage): the "low current ⇒ emit only bulk caps" decision — current draw
  ≈ N_leds × per-LED max (a 25-LED WS2812 string is ~1.5 A at full white, i.e. NOT low; pick a
  sensible threshold). Confirm the cross-layer split with the user before building the synth part.

**Geometry feasibility (3 mm pitch, 1.3 mm LED):** ~1.7 mm gap. A 0402 (1.0×0.5) fits beside on
F.Cu but eats into the data-hop channel — watch interaction with **task 4** (the in-row hops live
in that same gap). Validate decap placement is legal (no `illegal_pre_stamp`) and doesn't block
the data ties.

**Matching a decap to "its" LED:** there is no 1:1 net mapping (all decaps are VBUS↔GND globals).
Just assign decap *k* → the array member at chain index *k* (wrap if counts differ). Keep it
deterministic.

**Files:** `array_placement._place_companion_block` + its caller block (the
`array_colocate_decaps` path, ~line 185-216 region). Tests in `tests/test_array_placement.py`
(see `test_per_led_decaps_colocated_not_scattered` for the existing decap test to evolve).

---

## 5. Validation checklist for whatever you ship

- Run `tests/test_array_placement.py` + `tests/test_kicraft_models.py` (must stay green; the 4
  pre-existing failures elsewhere are not yours).
- Validate on the REAL path via the §2 `solve_subcircuits --config` command — **never** trust
  `replay --project` for arrays.
- KC-3QRJMX (task 4): data net fully kicraft-routed, 0 freerouting data segments, ≤ a couple
  vias, repeating pattern, fab-ready.
- KC-BUCJZ4 (task 5): array leaf places legally (no `illegal_pre_stamp`), composes, parent routes
  (rc7→rc0 ideally), decaps beside/edge per the rule.
- NO FALLBACKS: any path that can't do the right thing must **log/raise**, not silently scatter
  or hand off. (The leaf solve already fails loudly on a `place_array_leaves` exception —
  `solve_subcircuits.py:466` is unwrapped — keep it that way.)
- Commit on a branch, push, merge to `main`, then **deploy**: `bash deploy/restart-web.sh` and
  `bash deploy/restart-build-worker.sh` (both, for pipeline changes).
