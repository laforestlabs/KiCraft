# Self-eval 2026-07-19 batch — analysis + fix plan (V1–V8)

Batch `logs/self_eval/20260719T014949Z`: **16/34 fab-ready** (prev 18/34), mean 71.0 /
median 72.8, grades B:15 C:14 D:5, spend $1.13, wall 27473s (+31% vs prev batch's
20953s). First batch containing T1–T8 (`e839759`..`5ab8334`), the connector-facing
gates (`7c82a8d`), and the GND edge-spine (`eaba0a0`).

## Headline reading

The drop 18→16 is NOT a broad regression: scores went UP (old boards that "passed"
carried real defects the new gates now honestly reject). Per-run flips:

- IMPROVED (not-fab → fab): #11 fpc-breakout, #14 lora, #18 dual-rail (T1–T8 wins).
- REGRESSED (fab → not-fab): #1, #6, #26, #28, #31 — of these only **#26 (infra)** and
  **#31 (gate false positive)** are true regressions; #1/#28 were latent defects the
  new gates now catch (old #1 shipped a 180°-backwards BNC; old #28 shipped 3 of 4
  jacks electrically inert); #6 is routing variance (D_P ×2 unconnected).

## Failure taxonomy (18 non-fab runs, evidence in the batch dir)

| Cluster | Runs | Evidence | Owner fix |
| --- | --- | --- | --- |
| A. Compose hang → SIGKILL | 26 | parent compose round 1 never finished (>1800s; prev batch composed same design in ~127s); killed at harness 2400s. No DSN ever written → stall is pre-freerouting (stamp/candidate search). py-spy profile: see V1. | V1 |
| B. Battery-holder facing false positive | 31 | `connector_misoriented:BT1` — BT-SMD CR2032 holder, mid-board, zoned edge=bottom by synthesis; board otherwise 0/0 clean | V2 |
| C. §9.31/§9.32 fire only at build (no retry) | 28 | wiring committed OK; build 1/5 failed §9.31 (J3/J4/J5 inert) → rc5, model never got the offenders | V3 |
| D. Vendored pin-type defect (easyeda2kicad) | 27 | `dip-switch-3pos` all 6 pins `input` → 3× "Input pin not driven" ERC. Same defect: `screw-terminal-5mm-3p` (3), `usb-micro-b-receptacle-5p` (11) | V4 |
| E. BOM death spiral | 20 | 7 retries burned on: (a) `Mechanical:MountingHole` "exposes no pins" misfire, (b) auto-fetched OLED manifest claims C9900004566 (not in offline catalog) → fetch→reject loop, (c) header MPN confusion | V5 |
| F. §9.29 family gaps (observer cap 50) | 10, 30 | RP2040 passes §9.29 via "has USB connector" but has no BOOTSEL/RUN/SWD; ESP32-C3 has no EN/BOOT despite assumptions claiming them | V6 |
| G. Stranded inset −1.2…−1.8mm | 13 (ANT1 −1.82, J1 −1.72), 30 (J2 −1.37), 01 (J2 −1.20) | the KNOWN KC-YXQ4EC open "flush follow-up" (tol 1.0mm) | V7 |
| H. Identical-leaf mouth flip | 01 | J1+J2 same BNC footprint, both rot=90 → right-end J2 faces inward; board otherwise 0/0 clean. Same defect in the OLD "fab-ready" board — latent, now gated | V8 |
| I. Signal walled-off (C1 v2 family) | 06 (D_P), 09 (SWDIO), 12 (SDA/SCL), 13 (module nets) | known deferred owner: C1 v2 richer pathfinding | deferred |
| J. GND strand persists | 22 (GND×8), 25 (GND×2), parts of 10/30 | edge-spine (`eaba0a0`) did not fire/help here | investigate under V1's profile + C1 v2 |
| K. Illegal geometry / courtyard overlaps | 10, 21, 24 | compose promoted boards with courtyard overlaps + geometry violations; 21 also form-factor 28/32 | investigate (post-V1) |
| L. Known-genre | 29 (circle → RECT FALLBACK), 02 (R-2R leaf unroutable), 21 (form-factor) | ring nesting variance → leaf grid-assignment tuning; R2R → assignment-search tuning | deferred (existing owners) |

Batch-wide: leaf-solve wall time roughly DOUBLED vs the 0717 batch on comparable runs
(#12 321→831s, #32 147→831s, #24 775→1011s, #1 129→242s); median run wall +30%.
run_02 and run_13 hit `[wall-budget]` rescues they didn't need last batch. V1's
profile decides whether this is the same root as cluster A.

## Fixes (priority order)

### V1 — parent placement scorer is quadratic with a huge constant (run_26 hang) [P0]
Reproduced at $0: `compose_subcircuits` replayed on run_26's frozen pinned leaves hangs
exactly like the eval. **py-spy verdict (300s @50Hz, 14925 samples): 99.5% of all time
is under `_blocker_pair_compatible` (placement_utils.py:133), 93% in
`_any_rect_overlap` (subcircuit_composer.py:1540) — pure-Python rect transforms
(`_transform_rect`/`_points_bbox`/`rotate_vector` are the top self-time frames).**
NOT the GND edge-spine, NOT freerouting (no DSN was ever written).

Call-path math: `_score_block_opposite_side` (placement_scorer.py:424) runs
`_blocker_pair_compatible` for ALL C(n,2) pairs on EVERY `score()` — which SA calls
per move (`placement_solver.solve`/`_sa_refine`). Each call → `can_overlap_sparse` →
~10 `_any_rect_overlap` invocations; each is O(|A|×|B|) with rects_b re-transformed
inside the inner loop (≤200×200 after `_coalesce_rects`). 18 leaves ⇒ 153 pairs ⇒
~10⁸ 4-corner transforms per score call ⇒ the >30min cand_00.

Surgical, semantics-preserving fixes (in `subcircuit_composer.py`):
1. Hoist rects_b transforms out of `_any_rect_overlap`'s inner loop (|A|× saving).
2. World-bbox prefilter in `can_overlap_sparse`: transform each blocker set's overall
   bbox first; if every category's bboxes are disjoint there is no copper conflict →
   the pair is trivially compatible (position-dependent, exact, no behavior change).
3. Per-component world-rect cache keyed on (origin, rotation) so SA's single-component
   moves stop re-transforming the other n−1 components' sets.
Verify: repro composes in minutes; replay run_26 build tail to fab-ready; spot-replay
2 unaffected frozen boards (identical verdicts).

**Batch-wide leaf-solve slowdown (2×) is a separate open question** — the leaf path
short-circuits `_blocker_pair_compatible` (blocker_set None), so this scorer is not
the leaf cause. Measured (py-spy over a full run_01 leaf solve): ~87% of small-leaf
solve wall time is IMPORT overhead in short-lived subprocesses — dominated by
`import pcbnew` (~30%) across `_stamp_subcircuit_subprocess.py` and the
`python -c "import pcbnew; LoadBoard(...)"` one-shot probes (freerouting_runner /
_compose_route, all pre-existing). A persistent pcbnew worker (or probe batching)
is a real wall-time lever (~1.5–2s per subprocess × candidates × rounds) but is an
optimization, not a regression fix; whether T2/T3 increased probe COUNT per round
remains the open attribution question for the 2× — needs an invocation-count diff
between the two batches' logs, deferred.

### V2 — exempt non-wire-entry parts from connector_facings [P0, trivial]
`connector_facings` (autoplacer/brain/connector_edge_gap.py) gates every edge-zoned ref
with a detectable opening. Battery holders (BT*/BAT-*) have an opening but no off-board
mating contract. Skip refs whose prefix is BT (IEEE 315 battery class) — the zone stays
(placement bias toward the edge is still right), only the mouth-facing HARD gate skips.
Pinned test: BT1-like zoned holder mid-board → no `misoriented`; screw terminal case
(KC-YJ7Q69) still blocks. Flips run_31 (+1 board).

### V3 — run §9.31 + §9.32 at wiring commit [P1]
Add `check_repeated_block_coverage` + `check_regulator_feedback_vout` to BOTH wiring
check lists in `design/cli_app.py` (~1113 and ~3283) so the model gets the offenders
while it can still retry (they currently first fire in build-time
`collect_validations`). Keeps the build-time run as the safety net.
Test: wiring commit with an inert duplicate jack → rejected with §9.31 offenders.
Should flip run_28 (model wires the other 3 jacks when told).

### V4 — pin-type normalization for vendored parts [P1]
(a) Retype the 3 defective vendored bundles' pins to `passive` (dip-switch-3pos,
screw-terminal-5mm-3p, usb-micro-b-receptacle-5p) + `validate-part --update-hash`
(hash gotcha!). (b) At easyeda2kicad import: normalize pin types on non-IC part
classes (switch/connector/jack/terminal) to `passive`. (c) validate-part lint flags
`input`-typed pins on those classes (extends the KC-MUSEUD lint follow-up).
Flips run_27's ERC (+1 when routing holds).

### V5 — un-jam the BOM retry loop [P1]
(a) `_unresolved_symbols` (design/cli_app.py ~227): a symbol that RESOLVES with zero
pins in the `Mechanical` library is a valid pin-less part (MountingHole) — stop
rejecting it. (b) The manifest-claims gate (~340): when an auto-fetched bundle's
provenance LCSC id is absent from the offline catalog, STRIP the claim (log a note)
instead of failing the commit — orderability of the model's actual pick stays owned by
§9.26. Frees ~4 of run_20's 7 retries for the real sourcing work.

### V6 — §9.29 family-specific programmability [P2]
`check_mcu_programming_access`: "has a USB connector" is not sufficient for
RP2040/ESP32 families. Add family rules at the part-presence half:
- RP2040: require BOOTSEL button/jumper OR SWD header/TPs (USB alone insufficient).
- ESP32 (native USB or UART): require EN/RESET + BOOT buttons, or auto-reset
  (DTR/RTS) circuit, or TPs on EN+IO9/IO0.
Kills the two observer `unprogrammable_mcu` cap-50 gates (grade, not fab count).

### V7 — flush the −1.0…−1.9mm connector inset (KC-YXQ4EC open item) [P2]
Leaf-level: edge-zoned connector courtyards must land flush (gap ≥ −0.5mm) on the leaf
boundary edge that compose will mate to the board edge; today the leaf's canvas
clearance insets them 1.2–1.8mm and the parent gate (tol 1.0mm) rejects. Surgical:
snap-to-edge for the zoned side in leaf placement finalization + keep-out aware.
Improves 13/30 (still gated by their unconnected counts) and future single-header leaves.
**Landed as part of V8's finding #1** — the inset was the same body-offset bug at
1–2mm scale (body_center vs pos in `_connector_edge_x/_y`); pinned by
tests/test_edge_pin_body_offset.py.

### V8 — orient identical-leaf siblings to their assigned edge [P2]
run_01: two byte-identical BNC leaves placed at opposite ends with the SAME rotation —
compose block-rotation never flipped the second sibling to face its zone edge
(`component_zones J1:left, J2:right`). Fix in compose block-rotation: when a leaf's
zoned connector mouth direction is known, the leaf block's rotation at its edge slot
must satisfy mouth==outward (the facing gate's own math, applied at placement time
instead of only at verify time). Flips run_01 (board is otherwise 0/0).

**Implementation findings (2026-07-19):** the deeper chain was reproduced $0 on
run_01's frozen tree. `_filter_rotations_for_connector_opening` COULD NOT satisfy
mouth+extremity at any rotation because the LEAF had packed RV1 outboard of J2's
mouth — and it then gave up entirely (kept all 4 rotations), letting packing pick
the 180°-inward one. Two source fixes landed:

1. **Leaf edge-pin body-offset compensation** (`_connector_edge_x/_y`,
   placement_solver): the flush math used `pos ± width/2`, but `pos` is the
   footprint origin (at the pads) while the body bbox centers on `body_center`.
   Run_01's BNC (13.3mm offset) pinned "flush" with its mouth 13mm inside the
   canvas — the packer filled the phantom gap with RV1. The same mechanism at
   1–2mm offsets IS the V7 inset-mouth family (screw terminals / JST / SMA), so
   this one change owns both. Fresh leaf re-solve verified: J2 becomes the leaf's
   true right extremity, RV1 tucks behind.
2. **Mouth-first fallback** in `_filter_rotations_for_connector_opening`: when
   mouth+extremity is unsatisfiable, narrow to the mouth-correct rotations
   (mateable but possibly stranded) instead of keeping all 4 (possibly backwards).
   Verified on the frozen (stale-leaf) compose: J2 facing flipped misoriented→ok.

## Not fixed here (known owners)
- C1 v2 pathfinding (06/09/12/13 signal walls; 22/25 GND strands if V1's profile
  doesn't implicate the spine's absence) — separate deliverable, unchanged.
- Ring RECT-fallback (29), R-2R leaf (02): leaf grid-assignment tuning owner.
- Form-factor 28/32 + courtyard overlaps (21/24/10): investigate after V1 lands
  (candidate-search demotion interplay suspected).

## Verification strategy
Each fix: pinned unit test + $0 replay of its owning run's frozen workspace via the
build tail (`verify` skill flow). Batch-level: next full self-eval after landing;
expectation ~20–24/34 (V1 +1, V2 +1, V3 +1, V4 +1, V8 +1, V5 possible +1, plus
V1's batch-wide time recovery un-starving budget-edge boards).
