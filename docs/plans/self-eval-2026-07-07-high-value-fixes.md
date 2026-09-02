> **HISTORICAL — FreeRouting era.** FreeRouting was removed from the codebase on commit `a25e039` (2026-08-22); KiCad Routing Tools is the sole router (`kicraft/autoplacer/kicad_routing_tools.py`). File paths, `freerouting_*` config keys, `*_pre_freerouting.kicad_pcb` artifacts, and routing-behavior claims below describe the removed router — re-verify against the KRT adapter before relying on them.

# Self-eval 2026-07-07 — ranked high-value fixes (implementation plan)

**For the implementing agent.** This plan ranks the pipeline gaps surfaced by the
28-brief self-eval batch at `logs/self_eval/20260706T224451Z/` (first batch run on
main with the KC-9EZE3S fixes `7d17bca`+`56d1fef` deployed). Final baseline that
batch set: **12/28 fab-ready, mean 65.9, median 67.2, grades B×8 C×10 D×10,
0 errored, $1.27 total spend, 4h24m wall** (`summary.json` finished_at
2026-07-07T03:08:48Z).

Ground rules (non-negotiable, from repo history):
- **Fix at the source; no masking gates or post-route band-aids** (see memory
  `kicraft-fix-at-source-no-hacks`). Trace to the single point that sets the bad
  value.
- **Verify with `/verify` replay** (deterministic, $0, no LLM): freeze a failing
  run's workspace, replay through the build tail, read `kicraft artifacts` —
  never glob for boards, never compare artifacts across two separate replays
  (memories `kicraft-replay-command-and-determinism`,
  `kicraft-replay-cross-run-contamination`, `docs/ARTIFACTS.md`).
- Single-run route outcomes are noisy across grade buckets; claim deltas only
  replay-reproduced or N-of-3+ (memory `kicraft-self-eval-2026-06-24-findings`).
- Deploy = `deploy/restart-web.sh` (+ `deploy/restart-build-worker.sh` only for
  place/route changes). Run the full suite first; baseline has ~22 pre-existing
  env-dependent failures (clusters: `test_parent_outline_repair`,
  `test_web_default_project`, `test_build_zero_leaf`) — compare failure *sets*
  against a clean-main run, don't chase them.

---

## FIX 1 — build-timeout kills leak orphaned freerouting JVMs and destroy evidence  [infra/code · cheap · do first]

**Evidence:** 4/28 briefs died `rc=-9` (run_02 r2r-dac, run_07 usb-a-power-splitter,
run_10 rp2040-min, run_27 stepper-a4988 — killed at exactly 40:00 while its
route was still progressing). `kicraft/eval/self_eval.py:257`
`run_build(timeout_s=2400)` arms `threading.Timer(timeout_s, proc.kill)` —
`proc.kill()` SIGKILLs only the direct `cli_app build` child. The
`xvfb-run → java -jar freerouting-1.9.0.jar` grandchildren survive and reparent to
init. Two such JVMs from **July 4** were found still burning a core each on
2026-07-07 (PIDs since dispatched; pattern
`pgrep -fa freerouting` + `PPID 1` + dead `/tmp/tmp*/board.dsn` workspace).
Those two stray JVMs also plausibly slowed *this* batch's routes (two cores gone
→ more builds hitting the 40-min timeout → more orphans: a feedback loop).
Additionally the killed runs' `.kicraft/build.log` are **empty** — the evidence
trail dies with the process.

**Source / fix (three small pieces):**
1. `kicraft/eval/self_eval.py:run_build` — launch the build with
   `start_new_session=True` and kill the whole group:
   `os.killpg(os.getpgid(proc.pid), signal.SIGKILL)` (guard `ProcessLookupError`).
2. Same audit in `kicraft/autoplacer/freerouting_runner.py`: wherever the JVM is
   subprocessed with a timeout, ensure kill-by-group so `cli_app`-level timeouts
   can't strand `xvfb-run`'s child either. (Check the web build worker's job
   timeout path in `kicraft/server/build_worker.py` for the same hole.)
3. Build-log write-through: `cli_app build` should open the per-run
   `.kicraft/build.log` line-buffered / flush per stage line so a SIGKILL leaves
   the partial log (grep how the build tee is implemented — if the harness
   captures stdout and writes at exit, move to incremental writes at the
   producer).

**Verify:** unit-test `run_build` with a fake child that spawns a grandchild
sleeper: after timeout, *neither* survives. Then rerun one rc=-9 brief solo
(`kicraft-eval-batch --limit`-style or replay its frozen workspace) and confirm
`pgrep -f freerouting` is clean after the kill and `build.log` is non-empty.

**Prior-art:** NEW (the timeout itself is by design; the leak+evidence-loss is
the gap). Related: `kicraft-decap-array-grid-colocation` (freerouting 30-min
timeout family).

---

## FIX 2 — why do these routes take >40 min at all?  [investigate before touching params]

**Evidence:** the four rc=-9 runs spent 2400–3554 s in place/route before the kill.
r2r-dac previously routed fine (memory `kicraft-pwrflag-output-driven-net`
run_02 era). Confounder: FIX 1's stray JVMs were stealing 2 cores during this
whole batch, and the harness runs `--build-slots 2` concurrently.

**Fix:** do NOT tune freerouting params yet. After FIX 1 lands and the strays
are gone, replay the four frozen workspaces (`<run>/generated/<stem>` trees are
intact) one at a time with `python -m kicraft.design.cli_app replay --project
<copy> --quality good --seed 0` and wall-clock the route. If they route in
normal time solo → the 40-min timeouts were resource starvation (close as
environmental; consider `--build-slots` vs core count guard in the harness).
If a board genuinely routes >40 min solo → open a real congestion investigation
on that board (use the `kicraft-investigate` skill with the run directory) before changing any timeout.

**Verify:** four solo replays with times logged in the investigation notes.

**VERDICT (2026-07-07, solo replays, ACQUIRED-billed, quality=good seed=0,
zero queue, zero stray JVMs):**

| run | solo build | outcome | verdict |
| --- | --- | --- | --- |
| run_02 r2r-dac | 35.2 min | rc=0 fab-ready | **contention** (batch ran 2 slots on 2 cores + 2 stray JVMs) |
| run_07 usb-a-power-splitter | 93.6 min | still killed | **genuine congestion** — investigate |
| run_10 rp2040-min | 46.5 min | rc=7 | genuine (completes, over 40-min budget) |
| run_27 stepper-a4988 | 58.3 min | rc=6 | genuine — all leaf phase; parent compose insta-fails 0.17 s/round |

Follow-ups: use the `kicraft-investigate` skill on run_07 (worst) and run_27 (leaf-phase
sink + parent-compose failure); consider clamping the harness `--build-slots`
default to `max(1, cores // 6)` (build_slots.py's own sizing) — 2 slots on a
2-core box is guaranteed starvation. Per the ground rule, no timeout changes.

---

## FIX 3 — BOM stage dies to exhaustion with NO board (3/28 total losses)  [prompt/contract + tooling]

**Evidence:** run_18 dual-rail-supply, run_19 relay-quad, run_21 proto-shield:
`stage_status.bom.ok=false` after retry exhaustion; no `generated/` tree at all
(`build=None` in summary — grades there reflect synthesis only). Three distinct
terminal errors:
- run_18: `symbol(s) do not resolve to a pin inventory` (model kept naming
  unresolvable symbols to the end);
- run_19: `footprint(s) do not resolve to a real .kicad_mod` (offender J2);
- run_21: `§9.24 no opposite-edge connectors on one sheet` repeated to
  exhaustion.

These ran ON the new curated-first search code, so reachability alone didn't
save them. A no-board run is a total loss — worse than any DRC-dirty board.

**Source / fix (two independent pieces):**
1. **Suggestions-on-miss for resolve errors** (runs 18/19 class): in
   `kicraft/design/cli_app.py`, where BOM commit validation emits
   `_unresolved_symbols` / `_unresolved_footprints` offenders, append the top-3
   `search_symbols`/`search_footprints` hits for the offending name (the
   curated-first search from `7d17bca` makes these good). Precedent: commit
   `48b6a01` did exactly this for LCSC misses and collapsed BOM retry counts
   (memory `kicraft-pipeline-cost-bom-retries`). Today the model is told "use
   the lookup tools" but not *what the closest real ids are*.
2. **§9.24 winnability audit** (run_21 class): read `events.jsonl` for run_21
   and answer: could the model have satisfied §9.24 *at the BOM stage*, or was
   the opposite-edge conflict baked in by the architecture stage's sheet
   partition (an unwinnable inter-stage contract — the KC-WFFXZ3 shape, memory
   `kicraft-wiring-unwinnable-intersheet-contract`)? If unwinnable → the fix is
   an architecture-stage constraint or a reconcile normalizer, NOT more BOM
   retries. Do not "fix" this by relaxing §9.24.

**Verify:** re-run the three briefs (real LLM, ~$0.15 total) → expect all three
to produce a board (any rc); retry counts in `stage_status.bom.attempts` drop.

**Prior-art:** whack-a-mole family is known (`kicraft-wiring-unwinnable-intersheet-contract`,
`kicraft-reconcile-stage-plan` — PR1 of that plan is still unbuilt and §9.24 may
be another argument for it). The suggestions-on-miss extension is NEW.

---

## FIX 4 — silent substitution: model swaps the asked-for part class (3/28, cap 55)  [library coverage + prompt]

**Evidence (observer gate `silent_substitution`, why-strings in
`<run>/eval/report.json`):**
- run_04 speaker-crossover: brief said *air-core inductor + film capacitors* →
  BOM used a ferrite-core Fastron 11P + ceramics;
- run_17 led-cc-driver: brief said *1 A power LED* → BOM used `Device:LED` on
  an 0805;
- run_20 encoder-oled-panel: brief said *SMT I2C OLED* → BOM used a 1×4 pin
  header.

Common shape: the library/stock has no matching part class, and the model
substitutes silently instead of asking. The observer catches it (grade cap 55
works), but the board is still wrong.

**Source / fix:**
1. **Prompt rule** in `kicraft/server/stage_driver.py:_stage_extra("bom")`:
   when the brief names a specific part class and neither a curated bundle nor
   a faithful stock/LCSC part matches, raise a **clarifying question** (the
   parked-question path already exists) or record the substitution in
   `assumptions` — never silently swap the class. Keep it tight: this must not
   turn every passive into a question (scope to brief-named classes).
2. **Vendor the recurring classes** (now reachable thanks to `7d17bca`):
   a power LED (≥1 A), a 0.96" I2C OLED module, and a small air-core /
   film-cap set — `add-part --from-lcsc <C#> --into vendored`, then
   `refresh_sample_previews.py`, then add `core_blocks.json` rows if they're
   good defaults. Pick in-stock Basic-tier parts (both pools — memory
   `kicraft-bom-never-oos-retail-gate`).

**Verify:** re-run the three briefs → observer gate no longer fires (either the
right part class is used, or an explicit question/assumption exists).

**Prior-art:** NEW as a cluster (the observer gate is recent). Coverage-gap
remedy pattern is established (MCP23017 vendoring `8def8eb`).

---

## FIX 5 — the `unconnected_nets` walled-off routing family is now the #1 fab-blocker  [code · hard · known-deferred C1]

**Evidence:** 5/28 boards routed but not fab-ready on `unconnected_nets` (+
clearance clustered on the same ICs): run_09 stm32-min (`USB_D+`, 1, plus
`connector_stranded:J2@-37.41mm(right)`), run_13 nrf52-beacon (3 nets, U1),
run_14 lora-node (5 nets, U2/U3, + `illegal_routed_geometry`), run_22
esp32-dual-motor (6 motor-driver inputs, U3/U4), run_24 daq-8ch (`D+ D-`, U4).
That's ~18 percentage points of fab-ready rate — the single biggest lever in
the batch.

**Prior-art — READ FIRST, this is a known-deferred cluster:** memories
`kicraft-unconnected-1-cluster-walled-off-signal-power` ("WALLED-OFF routing
(no_clear_path), NOT a last-mm snap; don't band-aid; needs bend/via repair
(deferred C1)") and `kicraft-gnd-island-getcentre-crash-and-walled-off`. The
explicit prior decision: no island-removal band-aids, no masking.

**Fix (scoped):** implement the deferred **C1 bend/via repair** as a design doc
+ prototype, NOT a quick patch: for each unconnected net after freerouting,
attempt a constrained local repair (allow one extra via + off-grid bend through
the blocked corridor) and re-DRC; give up cleanly (keep the honest not-fab-ready
label) rather than force. Start from `kicraft/autoplacer/brain/leaf_routing.py`
/ the post-route repair entry points used by `repair_stranded_power`
(`kicraft-power-plane-strand-no-repair` shows the accepted shape of a
*principled* post-route repair — that one was approved because it repairs
at the router's own abstraction level, not by deleting evidence).
run_09's `connector_stranded` component is the remaining Bucket A3 of
`kicraft-connector-stranding-edge-flush` — check that memory before touching it.

**Verify:** replay the five frozen workspaces (one replay each, measure inside
that replay) → target: ≥3 of 5 reach `unconnected=0` with 0 shorts; NONE may
regress to shorts>0. Then a fresh 9-brief eval batch to check the fab-ready
rate moves.

---

## Quick wins (batch into one small PR)

- **Refdes text height 0.7 mm < 0.8 mm minimum** fires as a DRC warning on
  essentially every board (KC-9EZE3S, run_08 ×5, …). One constant at the point
  where generated footprint/refdes text is stamped (grep `0.7` text height in
  the emitter / board-setup constraints in `design/synthesis/` and
  `autoplacer/`) — either bump text to 0.8 or set the constraint to match.
  Kills the noisiest warning class fleet-wide.
- **Bundle-pin equality warning** (KC-9EZE3S appendix): in
  `_resolve_bom_mpn_sourcing`, when a part's footprint IS a curated bundle and
  the explicit `sourcing_note` C# ≠ the bundle manifest's `sourcing.lcsc`,
  emit a *warning* (not offender) naming both — catches quiet part/footprint
  drift on bundles.
- **`silk_edge_clearance`/`silk_over_copper` on BNC-style edge connectors**:
  cosmetic and expected for barrel-overhang parts; if a cheap clip is available
  (trim silk to the board outline at export), take it; otherwise leave.
- **Push main to origin** — as of this plan local main is ~9 commits ahead
  (2026-07-06 review batches + KC-9EZE3S fixes + email-PR merge). Coordinate
  with whoever holds unpushed work; then plain `git push`.

## Appendix — investigate-only pointers (not yet ranked)

- run_11 fpc-breakout: build label **`netlist faithfulness`** — the routed
  board's netlist diverges from the schematic. Silently-wrong-board class:
  use the `kicraft-investigate` skill on `logs/self_eval/20260706T224451Z/run_11_fpc-breakout`.
- run_23 can-node: 19 ERC errors (`pin_not_connected`×14, `label_dangling`×5) —
  wiring-stage quality on one design; check §2a cross-run scan before calling
  it systematic (`label_dangling` has a known router.py fallback root cause).
- run_28 audio-jack-buffer: `missing_power_pin`×4 — likely the op-amp power
  pins on a passive-heavy sheet; per-design unless it recurs.
- run_08 rs485-terminal: fab-blocked by ONE `courtyards_overlap` (J3 screw
  terminal × C1) — the known connector-pinned residual of
  `kicraft-courtyard-overlap-parent-compose-fix` (+1 run of evidence).
- KC-9EZE3S board-level leftovers (netless BNC mounting posts on the stock
  footprint; duplicate GND pours both on F.Cu with an empty B.Cu): cosmetic /
  quality; only worth a look if the duplicate-pour pattern recurs on new
  boards built post-`b094460`.
