# Self-eval 2026-07-08 — ranked high-value fixes (implementation plan)

> **STATUS 2026-07-08 — FIX 1, FIX 2a, FIX 2b IMPLEMENTED** (uncommitted working tree).
> FIX 3 correctly left to the C1 workstream; the Appendix items are per-design, not
> implemented here.
> - **FIX 1** — `split_cross_sheet_connections` (validation.py) splits a connection
>   whose endpoints span multiple `BomPart.sheet`s into one per-sheet connection,
>   wired in at the wiring-commit before `reconcile_inter_sheet_nets` (cli_app.py).
>   Guard = a cohesion branch added to §9.13 `check_netlist_faithfulness` (a BOM net
>   whose pins scatter across >1 extracted net = a dropped stub). A latent
>   double-PWR_FLAG bug it surfaced was fixed at source in `router._route_power`
>   (flag guarded by `flagged_nets`). **Verified:** re-emitting all 6 frozen
>   `state.json` → `pin_not_connected`+`label_dangling`+`hier_label_mismatch` = 0 on
>   all 6 (ERC clean), §9.13 green.
> - **FIX 2a** — `build_slots.resolve_build_slots()`: self-eval `--build-slots`
>   defaults host-aware (`max(1, cpus//6)` capped at cores) and an explicit value
>   `> cores` is rejected at launch; build worker `max_jobs` capped at cores.
> - **FIX 2b** — autoexperiment early-aborts a structurally-unroutable leaf
>   (`routing_exception` / `leaf_pre_stamp_legality_repair`) as rc6 with the leaf
>   named, instead of retrying to the watchdog wall; `cli_app build` tees stdout to
>   a line-buffered `.kicraft/build.log` (web worker opts out via
>   `KICRAFT_BUILD_LOG=external`). Root cause of the run_27 spiral:
>   `illegal_unrepaired_leaf_placement` on a dense leaf (deep autoplacer, left
>   alone). Full suite: **2446 passed, 0 new failures** vs clean main (15 shared
>   pre-existing env-dependent failures).


**For the implementing agent.** This plan ranks the pipeline gaps surfaced by the
first **34-brief** self-eval batch at `logs/self_eval/20260707T193651Z/` — the
first run of the corpus with the 6 shaped-outline briefs folded in
(`kicraft-corpus-shaped-fold-in`, main `0273560`). Baseline that batch set:
**12/34 fab-ready, mean 64.3, median 69.0, grades B×11 C×12 D×10 F×1, 0 errored,
$1.30 total spend, 5h04m wall** (`summary.json` `wall_s=18246`).

Outcome histogram: **12 rc0 · 9 rc5 (ERC) · 4 rc7 (DRC) · 1 rc6 · 6 rc=−9
(watchdog kill) · 2 no-board**. Shaped subset fared *better* than average
(4/6 fab-ready, 4/6 outline_check pass), so the gaps below are general, not
shape-specific.

Ground rules (non-negotiable, from repo history):
- **Fix at the source; no masking gates or post-route band-aids** (memory
  `kicraft-fix-at-source-no-hacks`). Trace to the single point that sets the bad
  value.
- **Verify with `/verify` replay** (deterministic, $0, no LLM): freeze a failing
  run's workspace, replay the build tail, read `kicraft artifacts` — never glob,
  never compare artifacts across two separate replays (memories
  `kicraft-replay-command-and-determinism`, `kicraft-replay-cross-run-contamination`).
  NOTE: replay re-runs **place/route only** — GAP 1 is a *synthesis-emit* bug, so
  verify it by re-emitting from the frozen `state.json` (recipe in FIX 1), not by
  route replay.
- Single-run route outcomes are noisy across grade buckets; claim deltas only
  replay-reproduced or N-of-3+ (memory `kicraft-self-eval-2026-06-24-findings`).
- Deploy = `deploy/restart-web.sh` + `deploy/restart-build-worker.sh` — FIX 1/2
  touch the build pipeline, so restart **both**. Run the full suite first and
  compare failure *sets* against clean main (there are ~22 pre-existing
  env-dependent failures; don't chase them).

---

## FIX 1 — schematic emitter drops net/sheet-pin label stubs on connector & cross-sheet pins  [code · synthesis · HIGHEST leverage]

**Evidence:** 6/34 briefs died `rc=5` with **co-occurring `label_dangling` +
`pin_not_connected`** on the *same* designs: run_01 rc-lowpass-bnc, run_03
thermocouple-amp, run_05 usb-pd-trigger, run_13 nrf52-beacon, run_22
esp32-dual-motor, run_30 rounded-c3-devboard. This is the batch's **#1
fab-blocker by breadth** (memory `kicraft-erc-emitter-drops-label-stubs`).

**Root cause (proven — code, not model output):** for 4/6 spot-checked, **every**
ERC-unconnected pin and **every** dangling label is *present* in
`state.json bom.connections` as an endpoint / net_name. The wiring model wired
them correctly; the emitter **failed to draw the graphical net/hier-label stub
onto the pin**, so KiCad ERC reports both the orphaned pin and the orphaned label.
- run_01: `J1` header pins 1-4 all in `bom.connections`; label `INPUT` is a
  net_name — yet all flagged.
- run_30: the entire `J2` 2×10 GPIO header (all 20 pins) + 9 GPIO net-labels
  (`UART_TX`, `GPIO_MTMS`, …) present in connections, all dangling.
- run_13 also throws `hier_label_mismatch` → the hierarchical-label / sheet-pin
  path for connector-heavy sheets is implicated. It clusters on **connector/header
  pins and cross-sheet named nets** (the pin→named-net→(other sheet) case).

**Localization recipe (do this first):** confirm the BOM has the connection the
emitted schematic dropped, then find the un-drawn stub.
```
RUN=logs/self_eval/20260707T193651Z/run_01_rc-lowpass-bnc   # then run_30 for the header case
# 1. Show the connections the ERC flagged as unconnected (they ARE in the BOM):
python - "$RUN" <<'PY'
import sys, json
from pathlib import Path
bom = json.loads(next(Path(sys.argv[1]).rglob("state.json")).read_text())["bom"]
for c in bom["connections"]:
    eps = [f"{e['ref']}.{e['pin']}" for e in c["endpoints"]]
    if any(r.startswith("J1.") for r in eps) or c["net_name"] in ("INPUT",):
        print(f"{c.get('sheet','/'):10s} {c['net_name']:8s} <- {eps}")
PY
# 2. Grep the emitted schematic for J1's pins / the INPUT label: is a wire+label
#    stub drawn to each pin, or is the pin bare?
grep -nE 'INPUT|"J1"|hierarchical_label|global_label' "$RUN"/generated/*/*.kicad_sch | head
# The gap is the emitter branch that should draw pin -> short wire ->
# (local net label | hier label | sheet pin) for a named/cross-sheet net.
```
**Source:** `kicraft/design/synthesis/` — the net-label + sheet-pin stub emitter
(skill ERC table points at `router.py` stub+label fallback and
`emitter.py:_emit_root` sheet-pin stubs). The defect is that connector pins wired
only via a *named net* (esp. one crossing a sheet boundary) don't get their
label/hier stub drawn. Trace UP from the missing stub to the branch that decides
"this pin gets a label vs a wire vs nothing."

**Fix + guard:** (1) draw the label/sheet-pin stub for every pin whose net is a
named/cross-sheet net; (2) **guard = a post-emit netlist-parity gate** — parse
the emitted `.kicad_sch` netlist and assert it matches `bom.connections`;
fail synthesis loudly on any pin/net present in the BOM but absent from the
emitted netlist. This is the gate that structurally *cannot* be fooled by the
model (the existing §9.11 net-coverage gate only checks `bom.connections`, which
are already correct — that is why this slipped through).

**Verify:** re-emit the 6 frozen `state.json`s under the fix → expect
`pin_not_connected` + `label_dangling` → 0 on all 6; the new parity gate green.

**Prior-art:** **NEW.** Distinct from `kicraft-sch-label-slide-and-quote-escape`
(that fixed `multiple_net_names` via label-slide — a different ERC class).

---

## FIX 2 — rc=−9 watchdog kills: harness contention + a leaf routing_exception death-spiral  [infra + code]

**Evidence:** 6/34 briefs killed at the 2400 s watchdog (run_09 stm32-min,
run_10 rp2040-min, run_17 led-cc-driver, run_26 servo-driver-16, run_27
stepper-a4988, run_32 hex-env-sensor), all with a **completely missing
`.kicraft/build.log`** (the orphan/empty-log signature). `events.jsonl` splits
them into two distinct sub-causes:

**2a — harness contention (run_09, run_10).** Genuinely slow leaf solves
(`solve_subcircuits_total` ≈ 900–1160 s for a single round) pushed over the cliff
by running **`--parallel 3 --build-slots 2` on a 2-core host**. The prior batch's
FIX 2 verdict already prescribed clamping build-slots to `max(1, cores//6)`
(memory `kicraft-self-eval-2026-07-07-fixes-implemented`) — it was **never
applied**, and this batch ran the contended default again.
- **Source/fix:** clamp the self-eval `--build-slots` default (and the web build
  worker's concurrency) to the host core count via `build_slots.py`'s own sizing;
  add a `build_slots ≤ cores` assertion so a contended config can't silently ship.
- **Verify:** solo-replay run_09/run_10 under the clamp — prior data shows these
  route once uncontended (rp2040 completed in 46.5 min → rc7 solo). Expect
  rc=−9 → rc7/rc0.

**2b — leaf routing_exception death-spiral (run_17, run_26, run_27, run_32).**
`events.jsonl` shows `[solve] error: No accepted routed leaf artifact produced …
after 12 round(s) across 4 canvas attempt(s): leaf_pre_stamp_legality_repair,
routing_exception`, then the autoexperiment keeps mutating params and retrying
until the 2400 s wall — converting a *fast, diagnosable* leaf failure into a slow
rc=−9 with no board and no log. (Cross-run scan: `leaf_routed_artifact_validation`
recurs on 5 designs, `routing_exception` on esp32-s3-sensor too.)
- **Source/fix:** `kicraft/autoplacer/brain/leaf_routing.py` + the autoexperiment
  retry loop (`kicraft/cli/autoexperiment.py`). (i) Root-cause the
  `routing_exception` on these specific leaves; (ii) **early-abort** the
  autoexperiment when a leaf is unroutable after N attempts and surface as **rc6
  with evidence**, not a timeout kill. Also make `cli_app build` write
  `.kicraft/build.log` line-buffered so a SIGKILL leaves partial evidence (the
  FIX-1-of-last-plan write-through never reached the killed runs).
- **Verify:** solo-replay run_27 stepper-a4988; expect a fast rc6 + a non-empty
  `build.log` naming the unroutable leaf, instead of a 40-min rc=−9.

**Prior-art:** **KNOWN, half-fixed.** The orphan-JVM leak is fixed
(`kill_tree`, FIX 1 of `self-eval-2026-07-07`); the slowness/contention and the
leaf-spiral halves are the open follow-ups that plan explicitly deferred.

---

## FIX 3 — parent-route leaves nets unconnected  [known-deferred C1 · no band-aid]

**Evidence:** 7 designs reject on `unconnected_nets` at parent route
(esp32-s3-sensor, dual-rail-supply, can-node, daq-8ch, r2r-dac, rs485-terminal,
chamfered-badge). The nets differ per design (no single tie/pour fixes it).

**Do not band-aid.** This is the standing **C1 "walled-off routing"** family
(memories `kicraft-unconnected-1-cluster-walled-off-signal-power`,
`kicraft-gnd-island-getcentre-crash-and-walled-off`) — the real fix is the C1 v2
selective rip-up / reroute, not a masking gate. Listed here for breadth accounting
only; owned by the C1 workstream.

---

## Appendix — per-design / low-breadth (not ranked)

- **run_20 encoder-oled-panel (F/34.5) + run_21 proto-shield — no board.**
  Synthesis produced nothing buildable; the batch's `unprogrammable_mcu` and
  `silent_substitution` gates each fired once. Investigate per-design.
- **run_25 gpio-expander (rc5) — `pin_not_driven`.** A power/driver ERC, *not*
  the GAP 1 emitter class (§2b of the `kicraft-investigate` skill); MCP23017 rail without a
  PWR_FLAG or an undriven feed pin.
- **usb-a-power-splitter — footprint `annular_width`/`padstack` (1 design).**
  A `.kicad_mod` pad/annular fix if it recurs.
- **U1 clearance on 2 rc7 boards (dual-rail, esp32-s3-sensor)** — different parts,
  so not one footprint; per-design placement clearance.
- **fpc-breakout, lora-node — `netlist faithfulness` (rc5).** Scoring gate, not
  ERC/DRC; check the wiring vs brief per design.

---

### Suggested order
1. **FIX 1** (highest leverage, NEW, self-contained: ~6 boards rc5→clean).
2. **FIX 2a** (nearly free: clamp build-slots — recovers the contention kills).
3. **FIX 2b** (early-abort + build.log write-through — turns rc=−9 into
   diagnosable rc6; then root-cause the leaf `routing_exception`).
4. FIX 3 stays with the C1 workstream.
