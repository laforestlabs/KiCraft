# Self-eval 2026-07-10 fix plan — batch `20260710T041015Z`

**For the implementing agent.** This plan was produced by running the full 34-brief self-eval
(28 rectangular + 6 new shaped-outline briefs) on branch `placement-streamline` at HEAD
`618344e`, then root-causing every non-fab-ready run with parallel deep-dive investigations.
Every claim below is backed by artifacts in `logs/self_eval/20260710T041015Z/run_NN_<slug>/`
(each has `events.jsonl`, `.kicraft/state.json`, `generated/<stem>/` with `.experiments/`,
`eval/report.json`). Line numbers were verified 2026-07-10; grep for the symbol if they drift.

## 0. Scorecard and the headline diagnosis

Batch: **11/34 fab-ready (32%)** · 12×B / 15×C / 7×D · mean 68.7 / median 70.0 · $1.06 · 7.1 h.
Previous batch `20260706T224451Z` (28 briefs, before 07-09/07-10 commits): **12/28 fab-ready (43%)**.

**This is a regression batch, not a baseline batch.** Six briefs that were fab-ready on 07-06
now fail, and the failures reproduce deterministically on replay (so this is NOT the ~12-pt
score noise floor — these are build-outcome flips):

| Brief | 07-06 | 07-10 | Cause (workstream) |
|---|---|---|---|
| buck-3a | fab-ready | route/infra abort | WS1 (leaf placement regression) |
| led-cc-driver | fab-ready | route/infra abort | WS1 |
| encoder-oled-panel | fab-ready | route/infra abort | WS4 (mounting-hole corner snap, likely exposed by WS1 geometry shift) |
| usb-pd-trigger | fab-ready | not fab-ready (DRC) | WS3 (promoted-artifact desync) + WS6 |
| usb-c-full-breakout | fab-ready | not fab-ready (DRC) | WS3 |
| esp32-s3-sensor | fab-ready | not fab-ready (DRC) | WS6 (USB-C escape stubs) + known FR-edge gotcha |
| proto-shield | build=None (different fail) | route/infra abort | WS5 (form-factor reconcile bug, new feature) |

The two recent default-flips are directly implicated:
- `7d7c900` (07-09) made connectivity-first `leaf_grid_assignment` the DEFAULT leaf placement.
- `618344e` (07-10) enabled `KICRAFT_FORM_FACTOR_ENFORCE` by default.

Failure buckets across the 23 non-fab-ready runs:
7× DRC (unconnected>0, zero true shorts) · 5× synthesis-dead (build never ran) ·
5× route/infra abort (rc=6) · 4× rc=-9 (2400 s harness watchdog kill) ·
1× ERC gate · 1× netlist-faithfulness gate.

Shaped-outline group verdict: **the shape path itself works.** All 6 shapes were captured in
`intent.form_factor` and emitted to `board_outline`; circle/rounded-rect/star built correct
Edge.Cuts. Hexagon/snowman "came out rectangular" is an **eval artifact** (WS9): those builds
died in the leaf phase (before shape stamping) and the eval graded the rectangular synthesis
seed stub. The one real shapes-quality bug is WS8 (star board = 592×563 mm).

---

## Workstreams, priority order

### WS1 (P0) — Leaf placement regression: `leaf_grid_assignment` default produces unroutable leaves

**Symptom.** 3 runs abort with rc=6 `layout/route engine exited 3` (buck-3a, led-cc-driver,
hex-env-sensor): one leaf fails all 12 rounds × 4 canvases with
`leaf_pre_stamp_legality_repair,routing_exception`. Both symptom flavors have one cause:
(a) unrepairable courtyard overlaps after grid assignment (`illegal_pre_stamp`, e.g.
buck-3a `overlap_pairs=['C2:C4']`, led-cc-driver `['R1:L1','LED1:C2']`, hex-env-sensor
`['U1:C1']`); (b) legal but connectivity-pathological placements that starve FreeRouting into
its 120 s timeout ×2 → `FreeRouting produced no SES output after 2 attempts (rc=-1)`
(`kicraft/autoplacer/freerouting_runner.py:1281`, wrapped at
`kicraft/cli/solve_subcircuits.py:571-596`). Replay of the buck-3a leaf shows the assignment
search *degrading* the placement: final score 43.4 with 18 crossovers on a ~20-part leaf vs
65.8 initial. The same mechanism is the main suspect for the 4 rc=-9 kills (WS2) — r2r-dac and
rp2040-min leaves were rejected on *every* round (`illegal_routed_geometry`, unconnected 10–22)
while burning 900–1300 s/round.

**Where.** The default flip: `kicraft/autoplacer/brain/leaf_size_reduction.py:263`
(`local_solver_config`), grid branch in
`kicraft/autoplacer/brain/placement_solver.py:405-470`, search itself in
`kicraft/autoplacer/brain/leaf_grid_assignment.py`.
Context: `docs/plans/placement-reconsider-connectivity-first-handoff.md` (the handoff already
says "needs assignment-search tuning").

**Fix.** Fix at the source — tune the assignment search, don't mask:
1. Score-guard the search: never return an assignment whose connectivity/legality score is
   worse than the input (pre-assignment) placement; the buck-3a replay shows exactly that
   happening (65.8 → 43.4). This is an accept-if-better rule inside the optimizer, not a
   fallback hack.
2. Make the assignment respect courtyard legality during search (the overlap pairs above are
   produced by the grid assignment itself and are unrepairable downstream).
3. If after tuning some leaf class still fails, escalate to the user/plan owner before
   shipping any "fall back to classic SA" — the user explicitly rejects masking fallbacks;
   a fallback here is acceptable only as an explicit, logged, temporary bridge and the
   default flip should be reverted instead if tuning stalls.

**Verify ($0).** Replay the three failing leaves against the frozen round configs (the
route/infra agent replayed them from
`run_15_buck-3a/generated/*/.experiments/subcircuits/9a7f14b5*/`, same for `212f0d3c*` in
run_17 and `5bfa05a2*` in run_32). Then `verify`-skill replay of buck-3a, led-cc-driver,
hex-env-sensor end-to-end → expect accepted leaves and rc∈{0,7}. Finally re-run the previous
batch's fab-ready set: `python -m kicraft.eval.self_eval --only buck-3a,led-cc-driver,usb-pd-trigger,usb-c-full-breakout,esp32-s3-sensor,encoder-oled-panel --no-judge`.

### WS2 (P0) — Autoexperiment has no wall budget and no early-stop for repeated *quality* rejections

**Symptom.** 4 runs SIGKILLed by the harness watchdog at exactly 2400 s of build time
(r2r-dac, rp2040-min, lora-node, snowman-ornament) with **zero final artifacts** — worse than
an honest rc=6/7 partial. None were hung; all were futile-but-legal retry loops. The existing
early-abort (`--unroutable-abort-rounds`, `_STRUCTURAL_UNROUTABLE_REASONS` at
`kicraft/cli/autoexperiment.py:178`, check ~:2620-2637) fires only for *structural* reasons
and only *between* outer rounds, so: quality-rejection loops (`illegal_routed_geometry`,
`routed_drc_rejection`, unconnected>0) retry forever (r2r-dac: 25 FR attempts, 19 no-SES;
rp2040-min: 29/30 routed then rejected, ≈26 min JVM on one leaf); lora-node hit the 600 s
`parent_freerouting_timeout_cap_s` (`kicraft/autoplacer/config.py:305`) in 3 consecutive
rounds; snowman (9 parts!) never finished round 1 because the per-leaf ladder
(12 rounds × 4 canvases, `kicraft/cli/solve_subcircuits.py` ~:1027) has no time bound.

**Fix (all four; they close different holes).**
1. Wall budget in the autoexperiment round loop (~`autoexperiment.py:2590`): `--max-wall-s`;
   before launching round N, if `elapsed + EMA(round_duration) > budget`, finalize
   best-so-far (rc=6/7 **with a board**). Plumb from `_QUALITY_PRESETS` /
   `_run_layout` in `kicraft/design/cli_app.py:3502,3562-3588`, split across leaves-phase and
   parents-phase.
2. Quality-rejection streak abort (alongside `_update_unroutable_streak`,
   `autoexperiment.py:210`): a leaf rejected with an unchanged reason signature and
   non-improving unconnected count for 2 consecutive outer rounds stops being re-solved;
   keep its best artifact and mark the run partial.
3. Per-leaf-solve deadline inside the `solve_subcircuits.py` ladder (config key
   `leaf_solve_max_wall_s`), exiting with the existing structural-failure message so abort
   (2)/(1) can fire. **This is the only fix that catches the snowman case** (round-boundary
   checks never run if round 1 never ends).
4. Parent cap-out early stop: after 2 consecutive rounds where the parent route hits
   `parent_freerouting_timeout_cap_s` (runner: `kicraft/cli/_compose_route.py:44`), stop
   retrying — placement-parameter mutation does not rescue a 600 s router timeout.

Do **NOT** raise the harness `--build-timeout` (2400 s default, `kicraft/eval/self_eval.py:712`)
— it fired correctly; these runs burned 2.7 h of build slots for zero output. Optionally
export the harness budget into the build env so the pipeline self-limits under the watchdog.

**Verify ($0).** Replay r2r-dac / rp2040-min / snowman workspaces with the budgets on: expect
each to exit rc=6/7 within budget with a best-so-far board instead of being killed. WS1 may
independently fix their leaves; verify WS2 with budgets forced low so the early-stops
demonstrably fire.

### WS3 (P0) — Promoted leaf artifact ≠ the board acceptance DRC validated (desync)

**Symptom (new, proven in 2 runs; there is a failing test on this branch that matches).**
The `leaf_routed.kicad_pcb` that compose stamps is not the board that passed acceptance DRC:
- run_05_usb-pd-trigger: accepted `round_0003_leaf_routed.kicad_pcb` had net PG routed
  (5 segments, R1 at (144.62,106.31)); the promoted `leaf_routed.kicad_pcb` — rewritten
  8 min later during the `--parents-only` phase — has **R1 moved ~16 mm and PG's copper
  deleted**, while `renders/routed_drc.json` (unconnected=0) predates the rewrite. Final
  board: PG + CC1 unconnected.
- run_06_usb-c-full-breakout (2 components!): promoted artifact has a 1.0 mm gap in D_N
  (two dangling stubs at (145.60,101.07)) that fresh `kicad-cli pcb drc` flags, while the
  saved acceptance JSON says unconnected=0.
- `pytest tests/test_solve_subcircuits_layout_persistence.py::test_best_round_to_layout_prefers_routed_board_geometry`
  **currently FAILS** on this branch (component at x=2.025 vs routed 2.0) — same class:
  `best_round_to_layout` / the re-base path (`solve_subcircuits.round_to_layout`, see memory:
  leaf re-base to (0,0) happens there) diverges from the routed board geometry.

**Fix.** Make acceptance binding: promote by copying the validated round file verbatim and
assert hash equality, or re-run DRC on the promoted `leaf_routed.kicad_pcb` after any
rewrite/re-base and reject on mismatch. Root-cause the 0.025 offset in
`round_to_layout` (`kicraft/cli/solve_subcircuits.py`) — the failing test is the minimal
repro; start there, then check what the `--parents-only` phase rewrite does to leaf files.
Evidence dirs:
`run_05_usb-pd-trigger/generated/USB_PD_TRIGGER/.experiments/subcircuits/c99cdace-*__c32bbf8fa0/`
and `run_06_usb-c-full-breakout/.../b7043edb-*__baafaca2d9/`.

**Verify.** The failing test goes green; replay run_05/run_06 workspaces → promoted leaf hash
== accepted round hash, PG/D_N routed on the final board.

### WS4 (P1) — Compose: corner-snapped mounting holes stamped onto leaf copper (no collision check)

**Symptom.** encoder-oled-panel: all 3 rounds die with
`candidate-search produced no acceptable placement in K=4 (... shorts=10..16 ...)`
(`kicraft/cli/compose_subcircuits.py:2694-2700`). Every "short" on every candidate is
`PTH pad 1 [<no net>] of H1` vs J1 pads at 0.0000 mm — the corner-pinned mounting hole is
placed exactly on the MAIN_CONNECTOR leaf's header pads.

**Where.** Corner-snap branch of `_pull_constrained_to_cluster` (~`compose_subcircuits.py:705-731`)
and `_snap_parent_local` (:1037): edge slides use `_largest_safe_slide`, corner snaps do no
copper-collision check at all; `parent_keep_in_rects` only protects holes with
`inward_keep_in_mm` and only pushes unlocked components.

**Fix.** Make the corner snap collision-aware: nudge the hole anchor with a
`_largest_safe_slide`-style search (or offset outside the cluster bbox by pad radius +
clearance). The candidate gate itself behaved correctly — do not weaken it.

**Verify ($0).** Replay encoder-oled-panel → compose produces an acceptable candidate, rc∈{0,7}.
(This brief was fab-ready on 07-06; if WS1's geometry shift is what pushed H1 onto J1, WS1 may
also unblock it — fix WS4 anyway, the missing collision check is real.)

### WS5 (P1) — Form-factor reconcile: stacking-header heuristic misses lib footprints → duplicate refs → compose crash

**Symptom.** proto-shield (Arduino Uno format): all rounds die with
`Parent-local component ref 'J4' collides with a child component`
(`kicraft/autoplacer/brain/subcircuit_composer.py:828`). Reconcile added scaffold headers
J4–J7 but failed to replace the LLM's original stacking headers J1/J3 because footprint
`pin-header-female-2-54-1x40:HDR-TH_40P-P2.54-V-F` doesn't match `_is_stacking_header`'s
`("PinHeader_"|"PinSocket_") and "P2.54mm"` KiCad-stock-name heuristic
(`kicraft/form_factors/reconcile.py:50-56`). The header sheet then keeps J1,J3 alongside
J4–J7, the emptied-sheet prune (`compose_subcircuits.py:1246-1257`, fires only when leaf
components ⊆ scaffold refs) doesn't fire, and leaf J4 collides with the scaffold's
parent-local J4.

**Fix (both).**
1. `_is_stacking_header`: detect by pad-pitch geometry / pin count from the footprint itself
   (or at minimum broaden to the vendored lib naming `*-2-54-*` / `P2.54-V`), not stock-name
   substrings.
2. Make the failure loud instead of latent: if, after reconcile, original headers survive on
   a sheet that also received scaffold refs, fail the reconcile step with a clear message
   (or strip scaffold refs per-ref from leaf artifacts instead of the all-or-nothing subset
   prune at `compose_subcircuits.py:1246-1270`).

Note: `618344e` enabled enforcement by default, so this fires on any Arduino-format brief.
The stale comment at `kicraft/design/cli_app.py:2936` still says "default OFF" — update it.

**Verify ($0).** Replay proto-shield → no ref collision, conformance gate runs; re-run
`tests/test_form_factor_reconcile.py` plus a new regression test using the
`pin-header-female-2-54-1x40` footprint.

### WS6 (P1) — Self-eval driver ignores `reconcile_target="bom"` parks → all 5 synthesis deaths

**Symptom.** All 5 build=None runs (nrf52-beacon, dual-rail-supply, can-node, stepper-a4988,
chamfered-badge) died the same way: wiring discovers the BOM lacks a required part (2nd 1 µF,
bootstrap caps, a FET for TERM_EN, FB divider, test points), parks a blocking question with
`reconcile_target="bom"`, and the harness auto-answer
(`kicraft/eval/self_eval.py:202-214`) replies "use sensible engineering defaults; do not ask
further questions" — turning a solvable BOM deficit into an unwinnable wiring task (wiring
cannot add parts). The model then oscillates between unknown-ref rejections
(`kicraft/design/models.py:615`), 9.15 dangling-net rejections, and suppressed re-parks
(`kicraft/server/stage_driver.py:884` blocks a second park once answers are set) until its 5
attempts (`_STAGE_MIN_RETRIES`, `stage_driver.py:646`) burn out. The model converged on every
*winnable* sub-error along the way. **The web path already solves this**: `server/web.py:1979`
re-drives `["bom","wiring"]` once with `_bom_reconcile_instruction` (`web.py:444`). 6/34 runs
hit the class; the only survivor (esp32-s3-sensor) had a deficit wiring could absorb.

**Fix.**
1. Hoist the bom-reconcile re-drive out of `server/web.py` into a shared helper
   (`server/session.py` or `stage_driver`) and call it from both the web path and
   `kicraft/eval/self_eval.py:run_design` (~:220). The eval driver must never plain-answer a
   park whose `reconcile_target` is set — it's the pipeline's note-to-self, not a user question.
2. Let a `reconcile_target` park bypass the second-park suppression at `stage_driver.py:884`
   so wiring can escalate instead of burning attempts.
3. Sharpen the unknown-ref retry feedback (`models.py:615` → `_retry_feedback`,
   `stage_driver.py:672-692`): include the valid BOM ref list and state "wiring cannot add
   parts — park with reconcile_target=bom".

**Verify.** Re-run the 5 dead briefs (`--only nrf52-beacon,dual-rail-supply,can-node,stepper-a4988,chamfered-badge`,
costs real LLM $ but small): expect wiring to commit after one reconcile round; watch that
the re-drive is once-only (no loop).

### WS7 (P1) — Fab gate blind spot: `tracks_crossing` not counted as a short

**Symptom.** rounded-c3-devboard shipped past the shorts=0 gate with a genuine different-net
copper crossing: a 2.54 mm straight GND link between adjacent J2 pads crosses a TXD0 track at
(167.41, 98.84), both F.Cu — KiCad `tracks_crossing` (error severity), but
`_run_kicad_cli_drc` (`kicraft/autoplacer/freerouting_runner.py` ~:1498-1522) counts only
`shorting_items` and 0.000 mm-clearance as shorts. A fab-blocking defect passed the gate.
(Also seen: 8× tracks_crossing on run_30's earlier rounds — the freerouting near-miss class.)

**Fix.** Map `tracks_crossing` into the shorts tally in `_run_kicad_cli_drc` (small class-map
change). Secondarily, find which pass drew that pad-to-pad GND link without a foreign-copper
clearance check (gnd_pour stranded-net repair is the suspect — `kicraft/autoplacer/brain/gnd_pour.py`)
and add the check there; the gate fix alone would have failed this board honestly.

**Verify ($0).** Replay rounded-c3-devboard → run now reports the crossing as a short
(honest fail) or, with the gnd_pour fix, routes clean. Add a unit test with a synthetic
crossing board.

### WS8 (P2) — Shapes: circumscribe can explode board area; `size_mm` is captured but never consumed

**Symptom.** star-ornament shipped **fab-ready at 592×563 mm** (128,146 mm² for 15
footprints, ~60× content area): `_fit_requested_shape`
(`kicraft/cli/_compose_validate.py:199`, called from `kicraft/cli/_compose_stamp.py:143`)
*circumscribes* the shape around the placed rectangle — a low-circularity star (inner_ratio
0.45) around a 47×43 mm rect explodes it. Nothing catches it: `intent.form_factor.size_mm`
(round-led-ring's Ø60 was only honored by luck) is consumed by **no** downstream code, and
the fab gate has no board-size sanity check. heart/snowman (compound family) will do the same.

**Fix.**
1. Consume `size_mm` in both branches of `_fit_requested_shape` as the target dimension:
   if the circumscribed result exceeds it, fail the fit loudly (or shrink-to-fit if content
   allows). Warn on material overshoot even without `size_mm`.
2. Add a fitted-area / content-area ratio cap after circumscribe (loud failure, not silent).
3. Longer-term (separate PR, coordinate with WS1 owners): teach the parent solver the target
   polygon so placement packs *into* the shape (e.g. constrain candidates via
   `PolygonOutline.contains_rect` in the compose candidate scorer) instead of stamping a
   shape around a rectangle.

**Verify ($0).** Replay star-ornament: expect a loud fit failure or a sanely-sized star;
round-led-ring must still pass at ~Ø60.

### WS9 (P2) — Eval: `_find_parent_board` grades the synthesis seed stub as "the built board"

**Symptom.** hex-env-sensor and snowman-ornament were reported as "requested hexagon/snowman
but the board is rectangular". False: their builds died in the leaf phase, the shape was never
stamped, and `kicraft/eval/self_eval.py:_find_parent_board` (:70) globbed
`generated/*/*.kicad_pcb` and graded the rectangular seed stub drawn by
`kicraft/design/synthesis/kicad_pcb_stub.py:_draw_board_outline` (:215). This misattributes
build failures as shape failures (it cost us an investigation lane this batch).

**Fix.** In `_outline_check` (`self_eval.py:78`) / `_find_parent_board`: gate on build outcome
— only classify a *promoted routed parent* (use artifact provenance, `kicraft artifacts`
resolver, not glob); otherwise report `outline_check = {pass: null, reason: "no built parent
(build rc=N)"}` distinct from level-0 "wrong shape". Also fix two provenance holes found
alongside: `_validate_parent_geometry` reports `outline_shape` from `manual_outline` only
(`_compose_validate.py:451-455`) so polygon-path boards log `"rect"`, and
`ParentCompositionState.to_dict` (`kicraft/cli/_compose_state.py:164`) omits
`requested_shape`/`fitted_polygon` — serialize them.

**Verify.** Unit test: rc=6 run dir with seed stub → outline_check reports "no built parent",
not "rectangular". Replay a passing shaped run → unchanged PASS.

### WS10 (P2) — Parts library: two footprint/symbol defect classes

**(a) EP thermal-via `<no net>` pads (blocks every board using these parts).**
run_22: TPS54331DDAR (SOIC-8-EP) and DRV8833PWPR (TSSOP-16-EP) vendored footprints carry
no-net PTH thermal-via pads beside the GND EP pad → intrinsic `solder_mask_bridge` +
`clearance` errors → 3/5 leaves permanently unacceptable every round → compose stamped
never-accepted leaves (secondary bug: compose accepted them silently). Fix at source: tie the
thermal-via pads to the EP pad's net in the vendored footprints; audit the library for the
same pattern (`grep` for PTH pads with no net in *-EP footprints). Remember
`validate-part --update-hash` after every hand-edit (vendored-hash gotcha) or the bundle is
silently dropped.

**(b) Pin electrical types on auto-fetched symbols (run_25's 16 ERC errors → grade cap 45).**
`screw-terminal-5mm-2p:WJ126V-5.0-2P` has both pins `input`; `mcp23017-soic:MCP23017-E_SO`
has all 28 pins `unspecified` → every GPIO line trips `pin_not_driven`. Fix in the
EasyEDA→KiCad importer (`kicraft/parts_library/`): normalize connector/terminal pins to
`passive`; audit existing auto-fetched bundles for `input`/`unspecified` on
connector-category parts.

**Verify ($0).** (a) replay run_22 → MOTOR DRIVER / BUCK leaves acceptable (remaining
failures belong to other WSs). (b) replay run_25 synthesis→ERC → 0 errors.

### WS11 (P2) — Emitter: sheet-pin Y-grid collision silently merges nets (run_11)

**Symptom.** `_emit_sheet_block` (`kicraft/design/synthesis/emitter.py:222-227`) computes
`step = height/(n_pins+1)` then snaps to the 1.27 mm grid; with 24 pins on a 30.48 mm sheet,
step < 1.27 so adjacent sheet pins snap to the SAME position — SIGNAL_12/SIGNAL_13 both at
(130.81, 105.41), their root stubs coincident with two labels → ERC-silent net merge, caught
only by the §9.13 netlist-faithfulness diff (`design/synthesis/validation.py:1743`).

**Fix.** Allocate distinct grid slots (`y + 1.27*(i+1)`) and grow the sheet height to fit
`n_pins`; assert post-snap uniqueness of pin positions (loud failure beats silent merge).

**Verify.** Unit test: 24-pin sheet → 24 distinct pin Ys; replay run_11 synthesis → §9.13 clean.

### WS12 (P3) — Routing completeness quality items (mostly known classes; do after WS1-3 re-baseline)

These are the residual DRC drivers once WS1/WS3 land; several are known/deferred — do not
band-aid them (user preference: fix at source), but the batch quantifies them:
1. **USB-C fine-pitch escape stubs (5/7 DRC runs — the dominant residual class).** External
   nets terminating on TYPE-C-31-M-12 pads A5–A7 (CC1, D±) get ZERO copper: the owning leaf
   gives them no escape stub and the parent router won't enter the pad field. Add leaf-time
   breakout stubs for *external* nets on fine-pitch connector/IC pads
   (`kicraft/autoplacer/brain/breakout_stubs.py`). Attacks run_05/06/12/24/30 without
   touching the deferred C1 repair work.
2. **GND pour islands / F↔B unstitched (run_22, run_26).** Insert same-net stitching vias
   between zone islands before the fab gate (`kicraft/autoplacer/brain/gnd_pour.py`). run_26
   (96×187 mm board at 4.6% utilization for 16 servo-header leaves) also argues for outline
   shrink at compose — separate, coordinate with WS8.3.
3. Known-deferred, report-only (dedupe — no new work): C1 walled-off/completeness repair
   (12 zero-copper inter-leaf nets this batch), Bucket A3 cross-leaf same-edge stranding
   (run_22 J3 at −15.98 mm), freerouting-ignores-DSN-boundary edge wires (run_12's
   copper_edge_clearance).

### WS13 (P3) — Observability gaps that made this investigation harder

1. Persist `rejection_reasons` for leaves with zero successes:
   `_append_failed_rounds_to_debug` silently skips when no prior `debug.json` exists
   (`kicraft/cli/solve_subcircuits.py:1104-1123`) — the actual `routing_exception` text for
   WS1's leaves had to be recovered by replay.
2. Emit a progress event on the no-JSON wiring-attempt path in `drive_stage`
   (`stage_driver.py:861-876`): runs 13/23/27 each silently lost 2 of 5 attempts.
3. Consider a higher wiring token floor for many-net boards (`_STAGE_MIN_TOKENS`,
   `stage_driver.py:665`) — several no-JSON attempts look like truncation.
4. unprogrammable_mcu gate note: run_34's ATtiny402 had UPDI in `no_connect_pins` with no
   programming header — once WS6 lands, check whether the BOM-reconcile round also catches
   missing programming access; if not, it's a wiring-stage validation candidate.

---

## Suggested implementation order & batching

1. **WS3 first** (desync): it has a failing unit test as the repro, it corrupts otherwise-good
   boards, and every later replay depends on trusting promoted artifacts.
2. **WS1** (placement regression) — the biggest fab-ready lever; replay-verifiable at $0.
3. **WS2** (budgets) — independent of WS1; converts any residual pathological run into a
   graded partial.
4. **WS4 + WS5 + WS7** — small, sharply-scoped compose/gate/reconcile fixes, each with a $0 replay.
5. **WS6** (eval driver reconcile) — unblocks the 5 synthesis deaths; touches server/eval only.
6. **WS8-WS11** — quality/correctness singles, each independently verifiable.
7. Re-run the full self-eval (`--no-judge` acceptable for the build-outcome check) and compare
   fab-ready count against **both** batches: target ≥ 20/34 fab-ready. Remember the ~12-pt
   score noise floor: judge-score deltas need N-of-3 medians; build-outcome flips don't.
8. WS12/WS13 as follow-ups after the re-baseline shows what's residual.

## Ground rules (from the project's working preferences)

- Fix at the single point that sets the bad value; no masking gates, no band-aid exemptions.
- Never weaken a gate to make a run pass (WS4's candidate gate and the 2400 s watchdog were
  both correct; the bugs are upstream).
- Verify every fix with a $0 replay of the exact failing run dir before re-running paid evals
  (`verify` skill; run dirs listed per-WS above). Never compare artifacts across separate
  replay runs — measure within one replay.
- Deploy note: pipeline changes need BOTH `deploy/restart-web.sh` and
  `deploy/restart-build-worker.sh`.
- After any hand-edit to a vendored part: `validate-part --update-hash`.
