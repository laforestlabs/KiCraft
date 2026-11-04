# Routing handoff — two of three live briefs are blocked at the build

Owner-facing handoff for a new session. Written 2026-10-01 on the production box, after the live
cohort `~/.kicraft/debug/cohort-2026-09-30-continuation` closed at 6 of 10 authorized loops.

**Read this first:** the LLM pipeline is not the blocker any more. Briefs 2 and 3 commit *all seven*
stages (intent → functional_spec → architecture → bom → wiring → electrical_review → silk_plan) and
then fail in place-and-route. Brief 1 is stable (fab-ready **and** compliant in its last three
passes). Everything below is about the build tail.

> **2026-10-02 correction — the two claims this handoff was built on are wrong.** Verified against
> the live artifacts (see §8 for the measurements and the fixes). In short:
>
> 1. **Leaves are NOT the problem.** Both briefs accept their leaves: brief 3 scored
>    `leaf_accepted=6/6` in every round, brief 2 `5/6`. The `leafs: accepted=0 total=0` line quoted
>    below is a **completion-status reporting artifact** (the final status falls back to
>    `best_round is None → 0` because the parent never routed, so no round was ever "kept"), not a
>    measurement of the leaf phase. `parent_composed` is not a composition fact either — it is set
>    to `parent_routed` (`kicraft/cli/autoexperiment.py:3233`), so it is `False` whenever routing
>    fails even though the parent board *was* composed.
> 2. **Castellations are not the wall.** `Castellated_Pad_2.54mm` handling exists in several places
>    (contradicting §2/Lead A), and 26 of the 34 edge pads route fine. The parent's 13 opens fail as
>    `blocked_by` **pre-existing (leaf-imported) copper** the run may not rip.
>
> The real proximate cause of the parent failure is **our own 120 s `kicad_routing_tools_timeout_s`
> wall**: the parent router is killed mid-`Final reconciliation`, exactly at 120 s, and the killed
> board is the one with the opens. §8 has the evidence, the fixes and the 5-brief validation.

## 1. The failure signature

Two shapes, both ending in the same refusal:

| shape | what the build tail does | rc |
|---|---|---|
| **no routed parent at all** | the autoexperiment completes with nothing kept | 7 (and rc 6 in the older runs recorded by the previous cohort) |
| **routed but electrically incomplete** | a parent board is produced with nets left open | 7 |

> **2026-10-02:** both rows are the *same* defect seen through two reporting layers, and "nothing
> kept" is misleading. In brief 3 the parent board **is composed and routed**; the route is then
> killed at the 120 s KRT deadline during reconciliation, so `parent_routed=False`, the round is
> never kept, and the *completion* status prints the leaf counts as 0/0. The signature to grep for
> is not "no parent" but `parent_routed: False` **with** a `parent_placed.kicad_pcb` +
> `parent_routed.kicad_pcb` pair on disk and `leaf_accepted > 0` in `experiments.jsonl`.

`rc=7` is the promotion gate in `kicraft/design/cli_app.py` (see "Promotion is unconditional" around
line 6560): it refuses to export a fab package when the promoted board fails no-shorts /
no-unconnected. **The gate is correct** — do not relax it to get a green run.

## 2. Evidence (all measured, paths are on this box)

### Brief 3 — RP2040 motor controller, `KC-SKPY67`, project **1125** (worst and most informative)

Stages: all seven committed. Build job 354: `done rc=7`.

`~/.kicraft/projects/44/1125/generated/RP2040_MOTOR_CONTROLLER/.experiments/run_status.txt`:

```
phase: done            pipeline: hierarchical_subcircuits
progress: 0/3 (0.0%)   best_score: 0.00    kept_count: 0     elapsed: 6m04s
leafs: accepted=0 total=0     composition_status: complete     top_level_status: not_ready
parent_routed: False
```

> **2026-10-02 correction — these are the *completion* status lines, written after the loop, and
> they are lossy.** The per-round record (`experiments.jsonl`, same directory) is the truth: rounds
> 1–2 both recorded `leaf_total: 6, leaf_accepted: 6, parent_composed: false, parent_routed: false`.
> The 0/0 comes from `kicraft/cli/autoexperiment.py:3416` reading `best_round.leaf_total if
> best_round else 0` — and `best_round` is only assigned when a round is **kept** (`:3274`), which
> cannot happen while `parent_route_rc != 0`. So this line is a *consequence* of the routing
> failure, not evidence about the leaf phase.

The build-recovery stage recorded the concrete defect (`state.json` → `stage_status.build_recovery`):

```json
{"ok": false, "failure_kind": "infrastructure", "recovery_attempts": 0, "recovery_max_attempts": 3,
 "recovery_events": [{"action": "none", "outcome": "blocked",
   "reason": "the layout failed without a named component/net to attribute to a design choice",
   "evidence": ["Open connections remain Routing left 13 open connection(s), so the board is not electrically complete.",
                "Unconnected item(s): 13",
                "Nets: MOTOR_IN5, MOTOR_IN6, __KICRAFT_RECIPE__rp2040_minimal__rp2040__QSPI_SD0, __KICRAFT_RECIPE__rp2040_minimal__rp2040…"]}]}
```

**Lead A — the nets that failed are the nets on the board edge.** The parent router's own stdout
(`.experiments/subcircuits/subcircuit__8a5edab282/parent_routed.router.stdout.log`) prints:

```
WARNING: 8 target pad(s) at/over the board outline:
    gpio24 (TP27.1) at (259.74, 107.54) is at/near the board edge (inside the edge keep-out) -- may be unroutable
    gpio7  (TP10.1) at (37.26, 107.54)  … gpio27 (TP30.1) … gpio15 (TP18.1) … gpio28 (TP31.1) … gpio29 (TP32.1)
    MOTOR_IN5 (TP7.1) at (37.26, 99.92) … MOTOR_IN6 (TP8.1) at (37.26, 102.46) -- may be unroutable
```

The pads named there are **castellated test points** (`TP7, TP8, TP10, TP18, TP27, TP30, TP31, TP32`
— the board carries 17 of them). Castellations are *meant* to sit on the outline, so the router's edge
keep-out treats them as unroutable, and the parent run leaves their nets open — `MOTOR_IN5`,
`MOTOR_IN6`, and the RP2040's own QSPI/clock/core nets are exactly the 13 the recovery stage lists.
[INFERENCE: that the two lists are the same nets by cause, not only by name; the mapping is exact for
MOTOR_IN5/IN6 and plausible for the QSPI/XIN/XOUT/DVDD nets, which sit on the flash, crystal and core
supply pins of an RP2040 recipe.]

There is **no castellation-aware handling anywhere in the pipeline** — `grep -rn castellat
kicraft/design/synthesis/*.py kicraft/autoplacer/*.py` returns nothing. This is the first place to look.

> **2026-10-02 correction — Lead A is a misdiagnosis, twice over.**
> 1. **The grep was too narrow.** `grep -rn -i castellat kicraft/` returns 47 hits, and the
>    behaviour is deliberately handled: `kicraft/autoplacer/brain/leaf_geometry.py:399` (castellations
>    are locked out of the out-of-board legalizer), `kicraft/autoplacer/brain/placement_solver.py:863`
>    (they intentionally violate "all copper inside"), `kicraft/cli/_compose_state.py:119`
>    (`fabricated_edge_refs` are exempt from containment), `kicraft/cli/_compose_stamp.py:169`, and
>    the final edge-datum gate `kicraft/design/cli_app.py:6490 _castellation_geometry_violations`.
> 2. **The edge keep-out is not what fails.** The parent router warns about **34** target pads at the
>    outline (all `Castellated_Pad_2.54mm` test points, `descr "Plated half-hole castellation; drill
>    center must lie on Edge.Cuts"`), and **26 of them route**. Only 8 of the 34 are in
>    `failed_single`, and their failure blocks read `blocked_by … no rippable blockers found /
>    preexisting_blockers` — the blocker is *other routed copper*, not the target pad's position. The
>    five nets that fail without appearing in the edge warning at all (`XOUT_RAW`, `QSPI_SD0…SD3`)
>    confirm the warning is informational.
>
> The real parent failure is the 120 s deadline (see §8) and, underneath it, that the main pass
> refuses to rip the leaf-imported copper that blocks those 13 nets.

**Lead B — one leaf failed 20 nets in 3 seconds.** Per-subcircuit router summaries
(`JSON_SUMMARY: {...}` lines in the same logs):

```
leaf …e355e9875c: routed_single [] , failed_single [gpio7, MOTOR_IN4…IN7, gpio9, gpio13, gpio14,
                  gpio24, gpio19…gpio23, gpio25…gpio29, gpio15], successful: 0, failed: 20,
                  total_time: 3.0, total_iterations: 100038, total_vias: 0,
                  rescue: {attempted: 20, recovered: []}
leaf …d8317bfc54: routed ~24 nets (USB-C/USB2 recipe nets) — a leaf CAN route here
parent           : routed_single 31 nets, failed_single [XOUT_RAW, QSPI_SD0…SD3, gpio24, gpio7,
                  gpio27, gpio15, gpio28, gpio29]
```

`total_time: 3.0` with `total_iterations: 100038` and zero vias is *not* our timeout (the bridge uses
`kicad_routing_tools_timeout_s`, default **120 s**, `kicraft/autoplacer/config.py:328`) — so either the
router's own budget/abort fired or the run gave up per net. [INFERENCE — the budget source is inside
KRT, which is not in this repo.]

> **2026-10-02 correction — the cause is in the leaf log itself, and it is not a timeout.**
> The 3 s / 100 038 iterations is the *reconciliation* sub-run tally; the router's own first pass for
> that leaf (same log) routed 24 nets, failed 20, used 1 091 059 iterations and 83 vias. The failing
> nets' blocks say exactly why:
>
> ```
> No route found after 5007 iterations (both directions)
>   FAILED: Could not find route (0.08s)
>   ROUTE FAILED - no rippable blockers found
>   Hint: search exhausted after only 5007 iterations with no rippable blockers - the start/target
>   pads are boxed in by static obstacles (neighboring pads + clearance), not by congestion. ... try a
>   finer grid and/or smaller clearance/track, e.g. --grid-step 0.05 ... --clearance 0.15
>   --track-width 0.15 (current: grid 0.1, clearance 0.153, track 0.2 mm)
>   Hint: the box also includes PROTECTED net(s) 'GND' (locked) within 3mm of the failing endpoint(s).
> ```
>
> So the leaf's 20 nets are *escape* failures at grid 0.1 mm / track 0.2 mm, boxed in by neighbouring
> pads and **locked GND** copper KRT will never rip. The leaf was then rejected by artifact
> validation (`illegal_routed_geometry`) and by the `no_unconnected` acceptance gate — not by the
> wall deadline (`leaf_solve_deadline` is a slowness note appended alongside; the effective deadline
> that round was ~104 s, not 3 s). `--grid-step` is a real KRT knob (`py_router/route.py:4406`) that
> **this bridge never passes**, so the router always runs at its 0.1 mm default.

### Brief 2 — ESP32-S3 indicator controller, `KC-WWYS3W`, project **1121**

Stages: all seven committed. Build job 351: `done rc=7`. `.experiments/run_status.txt`:
`parent_routed: False`, `top_level_status: not_ready`, `leaf_total: 0`, `kept_count: 0`,
`elapsed: 3m29s` — this is the **"nothing kept at all"** shape: no leaf was accepted, so no parent was
even composed. Contrast with brief 3, where leaves routed and only the parent's tail nets failed.

> **2026-10-02 correction — same reporting artifact as brief 3; brief 2's leaves ARE accepted.**
> `experiments.jsonl` for project 1121 records all three rounds with `leaf_total: 6,
> leaf_accepted: 5, parent_composed: false, parent_routed: false`. So briefs 2 and 3 are *not*
> different defects: both accept their leaves and both fail to route the parent. The `leaf_total: 0`
> in `run_status.txt` is the post-loop fallback described above.

### Brief 3, earlier attempt — project **1116**

Same signature, `parent_routed: False`, `elapsed 8m11s`, `leaf_total: 0`, `kept_count: 0`.

### Historical baseline (previous cohort, `~/.kicraft/debug/cohort-20260927-diverse`)

Routing has been the standing wall for a long time: across 15 passes of 5 briefs,
`routing_completed` was `0,1,0,0,0,0,1,0,1,1` (≤ 4 of 50 brief-passes routed) and
`compliant_fab_ready` was **0/5 in every pass**. Two boards did route and export (projects 1018 and
1136/1073/1041/1066 class) but were non-compliant for other reasons — that cohort's note that
"a successful exporter is not a correct board" still applies.

## 3. Where the code is

| concern | file |
|---|---|
| KRT bridge: command build, timeout, NPTH keep-out stamping, JSON summary parsing | `kicraft/autoplacer/kicad_routing_tools.py` (`route_with_kicad_routing_tools` line 561, `_krt_command` 247, `_stage_npth_keepouts` 524, `_stamp_npth_keepouts` 362, `_strip_npth_keepouts` 494, `_krt_json_summaries` 315) |
| config/env: `kicad_routing_tools_timeout_s`, `KICRAFT_KICAD_ROUTING_TOOLS_PYTHON`, `KICRAFT_KICAD_ROUTING_TOOLS_PATH` | `kicraft/autoplacer/config.py` (326-328) |
| placement (what puts the castellated pads where they are) | `kicraft/design/synthesis/placement.py`, `autoplacer.py`, `board_features.py`, `footprint_library.py` |
| board emission / net propagation to the PCB | `kicraft/design/synthesis/kicad_pcb_stub.py`, `emitter.py` |
| rc semantics + promotion/verify gate | `kicraft/design/cli_app.py` (~6560 for the rc 7 comment, `return 7`) |
| autoexperiment status the UI reads | `run_status.json` / `.txt` under `generated/<STEM>/.experiments/` |
| build worker (single serial slot) | `kicraft/server/build_worker.py`, `logs/kicraft_build_worker.log` |

KRT itself is pinned outside the repo: `/home/kicraft/KiCadRoutingTools` at commit
`3ceb773722bea67aa3685e7ee430c0c0d17ef38d` (2026-08-11), venv `/home/kicraft/krt-venv/bin/python`.

**Additions from the 2026-10-02 session (§8)** — the places a next session actually needs:

| concern | file |
|---|---|
| round record / completion status: `parent_composed`, `leaf_total`, tier | `kicraft/cli/autoexperiment.py` (`_parent_snapshot_composition`, `_parent_was_composed`, `_write_live_status`, the `:3416`-area completion block) |
| "is this leaf accepted?": the leaf gate vs the persisted routing verdict | `kicraft/cli/autoexperiment.py` (`_leaf_gate_accepted`, `_leaf_acceptance_config_for`, `_accepted_leaf_artifacts`); the gate itself is `kicraft/autoplacer/brain/leaf_acceptance.py` (`_gate_no_unconnected`, `split_unconnected_nets`) |
| why a pinned leaf's `validation.accepted` is not the gate's verdict | `kicraft/autoplacer/brain/pins.py` (pin copies a `round_NNNN_*` snapshot over the canonical `solved_layout.json`) |
| `validate_routed_board` does **not** judge unconnected nets (by design; each consumer applies its own policy) | `kicraft/autoplacer/routing_board.py` (`validate_routed_board`, rejection reasons near the end) |
| pre-route same-net tie copper (the clearance violation of §8.3.3) | `kicraft/autoplacer/brain/breakout_stubs.py` (`_foreign_pad_margins`' `strict_same_fp=False` own-footprint leniency), consumed by `brain/leaf_routing.py` (`auto_power_tie_specs`) |
| the leaf phase's wall budget vs the per-invocation router cap | `kicraft/design/cli_app.py` (`KICRAFT_BUILD_MAX_WALL_S` → 0.65/0.30 split, `KICRAFT_LEAF_SOLVE_MAX_WALL_S`), `kicraft/cli/_round_scheduler.py` (`_rescue_plan`, `plan_next` wall-budget stop) |
| compose's candidate gate (shorts-only by design) and the parent route invocation | `kicraft/cli/compose_subcircuits.py` (`_search_best_layout` candidate loop), `kicraft/cli/_compose_route.py` (`_route_parent_board`) |

## 4. How to reproduce and read a run

The live path (what actually produced this evidence), and it needs no provider budget for the build
tail — a run that already committed its stages can be rebuilt:

```bash
# 1. see the six stages' state and the build failure of a finished run
python -c "import json;s=json.load(open('/home/kicraft/.kicraft/projects/44/1125/.kicraft/state.json'));print({k:(v.get('ok') if isinstance(v,dict) else v) for k,v in s['stage_status'].items()})"

# 2. read the layout engine's own account of the run
cat /home/kicraft/.kicraft/projects/44/1125/generated/RP2040_MOTOR_CONTROLLER/.experiments/run_status.txt

# 3. read the router, not our summary of it
cd /home/kicraft/.kicraft/projects/44/1125/generated/RP2040_MOTOR_CONTROLLER/.experiments/subcircuits
tail -40 subcircuit__8a5edab282/parent_routed.router.stdout.log
grep -h JSON_SUMMARY */*.router.stdout.log | tail -3

# 4. rebuild an already-designed board without touching the provider pipeline
#    (see kicraft build --help; the cohort used: kicraft build .kicraft/state.json generated --quality good)
```

To generate a *fresh* failing board, submit the frozen brief through https://kicraft.io (the cohort's
drive account, `~/.kicraft/debug/live-cred.json`); the build queue is one slot, so runs serialize.

Frozen brief for brief 3 (project 1125, board `KC-SKPY67`):

```
An RP2040 motor controller with a ULN2003 darlington array, an 18 V DC input, and motor screw terminals. Use a two-layer stack-up.
```

Frozen brief for brief 2 (project 1121, board `KC-WWYS3W`):

```
An indicator controller with eight outputs, using an ESP32-S3 module, a 5 V header from the host board, an LED matrix, and a power LED. Use a four-layer stack-up.
```

## 5. Acceptance — what "routing done" means here

1. **Brief 3's board builds with `rc=0`** and its fab package exports, with the router reporting no
   failed single-ended nets and no open connections on `MOTOR_IN5/IN6`, the QSPI/clock nets or the
   GPIO nets that currently fail.
2. **Brief 2's board gets a routed parent at all** (`parent_routed: true`, `kept_count > 0`) — today
   no leaf is accepted for it, which is a different defect from brief 3's.
3. Then the cohort's own test becomes meaningful: re-run all three briefs through the live site and
   read `compliance_read.py` (fab-ready **and** compliant). Brief 1 already passes; the remaining gap
   is 2/3.

> **2026-10-02 status against these three (see §8 for the measurements).** Criterion 3's premise is
> broken: the Surprise-me space refuses ~95% of its own briefs at the design contract (§8.6), so it
> cannot be the read-out.
> 1. **Not met.** Brief 3's rebuilt board is now produced and complete but still carries 19 opens, all
>    traced to the leaf escapement wall (§8.7). It needs placement/escape work, not another router
>    knob.
> 2. **Met, exceeded.** Brief 2's rebuilt board exports a full fab package at `rc 0`, verify
>    `shorts=0 unconnected=0` (fresh leaf solve) — §8.8. Its *round* record still says
>    `parent_routed: False` because of a pre-existing clearance violation it did not create (§8.3.3);
>    that disagreement is the next defect to fix.
> 3. **Not measurable as written** — see §8.6.

## 6. What I did NOT check (so the next session does not assume it)

- **KRT internals.** I read our side of the bridge and the router's stdout only. I did not read KRT
  source, so I cannot say whether the 3 s / 100 038-iteration leaf run is a per-net timeout, a global
  budget, or an early abort.
- **Whether placement or routing is at fault.** The edge warning is emitted by the router *about
  placement*; a fix could legitimately live in either. I did not measure routability of the same
  placement with the edge pad set inset.
- **Why brief 2 accepts no leaf at all** while brief 3's leaves partly route. Their sizes, stack-ups
  (four-layer vs two-layer) and parts differ; nothing here isolates the variable.
- **Whether the 13 opens include any net that a pipeline-side defect should have prevented.** I
  confirmed the *committed* BOM carries those nets and that the pipeline's own gates pass them; I did
  not trace each net end to end on the PCB.
- Router `stderr` is **empty** in every subcircuit log I checked — the failures are reported on stdout
  as data, not as exceptions.

> **2026-10-02 — four of these five are now checked:**
> - **KRT internals:** read. The 3 s / 100 038-iteration leaf tally is the router's *reconciliation
>   sub-run* over the still-open subset, not a timeout and not the run budget. KRT's `--grid-step`
>   (default 0.1) is a real knob this bridge never passes; `net_rescue` already ladders grid 0.025 +
>   clearance/track/via step-downs and still recovers none of these nets.
> - **Placement vs routing:** placement (escapement). The failing frontier cells are blocked by
>   *neighbouring pads and locked GND*, and the router says so; the same net set fails at every
>   geometry the rescue ladder tries. Insetting the castellations would not help — 26 of the 34 edge
>   pads already route.
> - **Brief 2 vs brief 3:** they are the same shape, not different ones — both accept leaves (5/6 and
>   6/6) and both fail the **parent route**. Brief 2's parent route actually *completes*; it was our
>   custody check that failed it (see §8.2/F3).
> - **Pipeline-side defects among the 13 opens:** yes, one — the router's `Final reconciliation` takes
>   rip authority over pre-existing copper, rips a net, and can leave it un-restored (brief 2:
>   `…matrix_interface__shifted0`). That is open work, not fixed here.

## 7. Do not

- Do not relax the promotion/verify gate to turn `rc=7` into `rc=0`. A board with 13 open connections
  is not a board.
- Do not treat a routed export as success on its own: the previous cohort's routed boards were
  non-compliant in size, part count and part identity. Route completion is the *entry* condition, and
  `compliance_read.py` in the cohort directory is the check to read afterwards.
- Do not re-open the pipeline patches in this session's commits (see `git log 688f7a9..bd5d5cc`) unless
  a build failure is traced to one of them; each carries the measurement behind it, and brief 1's three
  consecutive compliant passes say they hold.

---

## 8. 2026-10-02 validation: measured root causes, fixes, and evidence

Everything below is measured on the live artifacts of projects 1121/1125 unless marked
`[INFERENCE]`. Corrections to §1/§2 are inline above; this section is the mechanism, the fixes,
and what is still open.

### 8.1 The real failure chain

1. **A leaf cannot close its own nets.** The RP2040 CONTROL leaf (`2f574b5d-…__e355e9875c`) ends
   `accepted=False` (`failure_reasons=leaf_solve_deadline,no_unconnected,routing_timeout`, in
   `metadata.json` notes). Its DRC report records **13 unconnected pads** on
   `MOTOR_IN5/IN6, QSPI_SD0…SD3, XOUT_RAW, gpio7/15/24/27/28/29`. The router's own hint:
   > `search exhausted after only 5007 iterations with no rippable blockers - the start/target pads
   > are boxed in by static obstacles (neighboring pads + clearance), not by congestion`

   and the frontier cells are blocked, in the router's words, by **neighbouring pads and locked
   GND**. (Measured later: the failing pads are *not* crowded against other footprints — the tightest
   such gap is 2.478 mm — so "neighbouring pads" means blocked grid cells, i.e. other nets' routed
   copper plus its clearance at 0.1 mm resolution; see §8.7.) KRT's full rescue ladder (grid
   0.025 mm, clearance ladder, track neck-down, via step-down) does not recover them, and neither does
   a completed run given 300 s. This is a **placement/escapement** defect, not the edge keep-out and
   not a timeout.
2. **The incomplete leaf is still composed.** By design (`solve_subcircuits.py:_persist_solution`,
   "A best-effort leaf … must still serialize so it is discoverable, pinnable and composed"), a
   routed-but-gate-rejected leaf is pinned and stamped. So the parent inherits the opens as nets it
   must now route between fixed leaf copper.
3. **The parent cannot close them either, and our deadline hides the outcome.** Parent
   `subcircuit__8a5edab282` routes 31/44 then reports `13 pre-existing net(s) eligible for rip-up`
   and runs KRT's `Final reconciliation`, which takes rip authority over the hinted blockers. Our
   `parent_routed.krt-input.kicad_pcb` is written at 02:28:08 and the SIGTERMed output at 02:30:08 —
   **exactly the 120 s `kicad_routing_tools_timeout_s`** — and the stdout ends mid-reconciliation at
   `Building base obstacle map...`.
4. **Two custody/reporting defects sat on top.** (a) When KRT *finishes*, its cleanup strips
   original-file dead-end/island segments; the bridge's byte-preservation check treated that as
   copper loss and raised, so a **fully routed** parent (brief 2: `routed 16 failed 1`, then
   reconciliation `routed 1 failed 0` → no net left failed) was reported as a route failure. (b) The
   round record and the completion status misreported composition and leaf counts.

### 8.2 Fixes (this session)

| # | defect | fix | file |
|---|---|---|---|
| F1 | `parent_composed` was an alias of `parent_routed`; completion status collapsed to `leaf 0/0` when nothing was kept | derive `parent_composed` from the composer snapshot (`state.entry_count` / `composition.summary`); report the kept round **or the last round run** | `kicraft/cli/autoexperiment.py` (`_parent_snapshot_composition`, `_parent_was_composed`, final `_write_live_status`) |
| F2 | parent route capped at the 120 s **leaf** cap although its phase is allotted ~0.30 × build wall (~486 s); and the leaf cap itself (120 s) was under what a dense leaf needs | new `kicad_routing_tools_parent_timeout_s` (default 420), authoritative for the parent route; `kicad_routing_tools_timeout_s` (leaf/local) raised 120 → **300** (the leaf slice is ~350 s on the production wall) | `kicraft/autoplacer/config.py`, `kicraft/cli/_compose_route.py` |
| F3 | declared unloaded cleanup (dead-end/orphan-island sweep, `output_writer` line) read as copper loss | tolerate a missing-input-copper verdict only when the router **declares** the removals, vias are fully preserved, the counts match, and the loss is ≤ max(20, 10% of input) — recorded in stats; electrical completeness is still proven downstream | `kicraft/autoplacer/kicad_routing_tools.py` |
| F4 | the router's `--grid-step` was never passed, so every route ran at its 0.1 mm default even for fine-pitch escapes | new `kicad_routing_tools_grid_step_mm` (default `None` = router default, no behaviour change), scoped to leaf/local routes; the parent keeps its own `kicad_routing_tools_parent_grid_step_mm` (default `None`) because a leaf-tuned fine grid times the parent route out on a large board (§8.9). Operable from `KICRAFT_KICAD_ROUTING_TOOLS_GRID_STEP_MM` (the build's synthesize step regenerates the per-project `*_autoplacer.json`, so a project-file edit does not survive) | `kicraft/autoplacer/config.py`, `kicraft/autoplacer/kicad_routing_tools.py`, `kicraft/cli/_compose_route.py` |
| F2b | the routing knobs had no environment surface, so the finer grid could not be paired with the longer budget it needs without a code edit | `_optional_env_float` + `KICRAFT_KICAD_ROUTING_TOOLS_TIMEOUT_S` / `KICRAFT_KICAD_ROUTING_TOOLS_PARENT_TIMEOUT_S` (unset → the documented defaults; a typo raises instead of silently disabling) | `kicraft/autoplacer/config.py` |
| F5 | a **pinned** leaf read as accepted even when the leaf gate rejected it: pinning copies a *round* snapshot over the canonical `solved_layout.json`, and a round snapshot carries only `validate_routed_board`'s verdict (which does not judge unconnected nets) | `_accepted_leaf_artifacts` re-derives the gate verdict (pure function of the validation block, thresholds recovered from the artifact's `config_hash`) instead of trusting `validation.accepted` | `kicraft/cli/autoexperiment.py` (`_leaf_gate_accepted`, `_leaf_acceptance_config_for`) |

F3 removes a *spurious* failure at the bridge, and that is all it does. On brief 2's own parent input
it converts `RoutingCopperPreservationError` into a returned, routed board (pruned 1 declared unloaded
segment, vias preserved 92/92) — **necessary but not sufficient** for `rc=0`: that same re-run still
carried 1 open net and the pre-existing clearance violation below, so the promotion gate would still
refuse it. 9101's `rc=0` (§8.8) comes from a fresh leaf solve, not from F3 — F3's own custody note
never fired in that rebuild. F4 changes what is *reachable* (§8.7) but is off by default; F1/F2/F5
change what the run *reports*, how much wall time the parent router gets before its recovery ladder is
cut off, and whether a pinned but gate-rejected leaf counts as accepted.

### 8.3 What FIXED does and does NOT mean

- F2 does **not** make brief 3 close; the 13 nets are geometrically blocked (8.1.1/8.1.3). It removes
  a mis-set budget so the router reports its real verdict instead of being killed mid-run.
- F3 removes a *spurious* failure. Whether the resulting board promotes is then decided by the real
  acceptance gate. On brief 2 the board still carries **1 open net**
  (`…matrix_interface__shifted0`, ripped during reconciliation and not restored) and **1 clearance
  violation that is already present in the composed input** (0.1368 mm vs the 0.153 mm rule) — so the
  routed board is `accepted=False` on `illegal_routed_geometry` + the open net, i.e. still `rc 7`, for
  honest reasons.
- The remaining work named by this validation, in order:
  1. **Leaf escapement** (§8.7) — the RP2040 leaf cannot close 2–20 nets depending on grid; a placement
     or escape-strategy change is needed, not a router knob.
  2. **Reconciliation leaves a ripped pre-existing net open.** Brief 2: the router rips
     `…matrix_interface__shifted0` ("PRE-EXISTING rip … original restored if the reroute fails") and
     the output still has it open. The restore-on-failure path is not holding.
  3. **One leaf-local clearance violation propagates into the parent.** Traced to source: the POWER
     leaf's **pre-route** board (`leaf_placed.kicad_pcb` of project 44/9101) already carries
     `Pad 2 [GND] of U4` vs `Track [VBUS]` at **0.1368 mm vs the 0.153 mm rule**, where U4 is a
     `SOT-23-5_L3.0-W1.7-P0.95` (TLV62569 buck) and the track is a **0.127 mm diagonal tie from pad 1
     (VBUS) to pad 4 (VBUS)**. The tie is stamped by our own `brain/breakout_stubs.py` via
     `leaf_routing.py`'s `auto_power_tie_specs` call, and the module **predicts this exact failure**:
     `_foreign_pad_margins` uses a *collision-only* margin (`half_width + 0.05`) against the source
     footprint's **own** pads unless `strict_same_fp=True`, and its docstring already records the
     signature — *"a stub grazing a same-footprint GND pad at 0.05 mm is a hard error, the KC-UXASHQ
     U1.5-vs-U1.6 signature"* — together with the reason it is not the default (*dense rows would lose
     every escape*). **Measured, so the fix is low-risk:** re-running `perimeter_tie_specs` for U4's
     VBUS pads with `strict_same_fp=True` (monkeypatched `_foreign_pad_margins`, no code change) still
     emits the tie — it routes the *perimeter walk* around the footprint instead of the diagonal:
     `(87.5,18.15) → (85.95,18.15) → (85.95,22.65) → (90.1,22.65) → (90.1,21.35)`. So connectivity is
     preserved and the clearance violation disappears; what remains to decide is only whether the
     retained leniency is still wanted for the *dense-row* case (`array_router`/`gnd_pour` consumers
     share this guard), which is why this was documented rather than flipped unilaterally. Then the
     leaf gate correctly rejects the leaf (`illegal_routed_geometry`), the pipeline composes it
     **best-effort** (documented behaviour), and the parent inherits the violation and is rejected for
     it — which is what keeps brief 2's *round* record at `parent_routed: False` even when the board
     routes and the build exports. Second open question: the composer's stamp DRC does not surface
     such a violation anyway (project 1125 recorded `{"clearance": 0, "copper_edge_clearance":
     34..37, "ran": true}` on its stamped parent), so the two clearance checks disagree.

### 8.4 One-command reproductions (no provider spend)

```bash
set -a && . ./.env && set +a          # KRT path + venv

# (a) Parent-route budget: the live run was SIGTERMed at 120 s mid-reconciliation.
#     Run the bridge on the ARCHIVED parent input with a long deadline and read the tally.
.venv/bin/python - <<'PY'
import sys; sys.path.insert(0, ".")
from kicraft.autoplacer.config import DEFAULT_CONFIG
from kicraft.autoplacer.kicad_routing_tools import route_with_kicad_routing_tools
src = ("/home/kicraft/.kicraft/projects/44/1125/generated/RP2040_MOTOR_CONTROLLER"
       "/.experiments/subcircuits/subcircuit__8a5edab282/parent_routed.krt-input.kicad_pcb")
cfg = dict(DEFAULT_CONFIG)
cfg["kicad_routing_tools_parent_timeout_s"] = 600   # the live run's cap was 120
cfg["pcb_path"] = src
st = route_with_kicad_routing_tools(src, "/tmp/probe_parent.kicad_pcb", cfg)
print("elapsed_s", st["elapsed_s"], "failed_nets", st["failed_nets"])
print(st["json_summaries"][-1]["failed_single"])
PY
# 600 s -> completes in ~200 s and still reports 13 opens (geometric, not budget)

# (b) Leaf escapement: the router's own diagnosis, and what --grid-step does.
L=~/.kicraft/projects/44/1125/generated/RP2040_MOTOR_CONTROLLER/.experiments/subcircuits/2f574b5d-33dc-4625-8dfe-d54053f71f15__e355e9875c
grep -m2 "pads are boxed in" $L/leaf_routed.router.stdout.log
# then set kicad_routing_tools_grid_step_mm = 0.05 and re-run the bridge on
# $L/leaf_routed.krt-input.kicad_pcb with a matching timeout: 20 failed -> 3.

# (c) Brief 2's parent route COMPLETED; our custody check is what failed it.
tail -3 ~/.kicraft/projects/44/1121/generated/ESP32_INDICATOR_CONTROLLER/.experiments/\
subcircuits/subcircuit__8a5edab282/parent_routed.router.stdout.log

# (d) Gate verdicts straight from the artifacts (the pin copies a round snapshot
#     over these, so the persisted `accepted` is NOT the gate's verdict).
.venv/bin/python - <<'PY'
import json, sys; sys.path.insert(0, ".")
from kicraft.autoplacer.brain.leaf_acceptance import acceptance_config_from_dict, evaluate_leaf_acceptance
from kicraft.autoplacer.config import DEFAULT_CONFIG
base = ("/home/kicraft/.kicraft/projects/44/1125/generated/RP2040_MOTOR_CONTROLLER"
        "/.experiments/subcircuits/2f574b5d-33dc-4625-8dfe-d54053f71f15__e355e9875c/")
v = json.load(open(base + "solved_layout.json"))["validation"]
print("drc.unconnected:", (v.get("drc") or {}).get("unconnected"))
print("gate:", evaluate_leaf_acceptance(v, v.get("anchor_summary") or {},
                                        acceptance_config_from_dict(DEFAULT_CONFIG)))
PY

# (e) End-to-end routing tail without the provider: rebuild a committed design in a copy.
cp -a ~/.kicraft/projects/44/1121 /tmp/1121-copy       # then rewrite the abs paths in
#    .kicraft/state.json to /tmp/1121-copy, purge generated/*/.experiments + pins.json, and:
cd /tmp/1121-copy && KICRAFT_BUILD_MAX_WALL_S=1620 \
  /home/kicraft/KiCraft/.venv/bin/python -m kicraft.design.cli_app build .kicraft/state.json generated --quality good
```

### 8.5 Tests

New/changed (all green; 235 tests across the touched subsystems pass):

- `tests/test_kicad_routing_tools.py`
  - `test_krt_route_accepts_declared_unloaded_input_prune` / `test_krt_route_rejects_undeclared_missing_input_copper` (F3)
  - `test_parent_route_uses_parent_timeout_budget` (F2)
  - `test_krt_command_passes_configured_grid_step` + `test_default_config_leaves_grid_step_to_the_router` (F4)
  - the pre-existing `test_krt_route_rejects_missing_input_copper` and
    `test_krt_route_timeout_behavior_is_unchanged` still pin the unchanged paths.
- `tests/test_autoexperiment_scoring.py`
  - `TestParentComposedReporting` (F1) and `TestLeafGateVerdict` (F5).
- Live 20-brief "Surprise me" cohort: `scripts/routing_surprise_cohort.py --count N`
  (or `--corpus N` for the frozen corpus), writing
  `~/.kicraft/debug/routing-surprise-<stamp>/summary.json`.

### 8.6 Cohort result (2026-10-02)

`scripts/routing_surprise_cohort.py` drove **20 freshly drawn "Surprise me" briefs** through the real
design stages and then the build, in three batches (5 + 5 + 10; run dirs under
`~/.kicraft/debug/routing-surprise-*`). Result: **19/20 were refused by a deterministic design
contract before the build; 1 reached the build and built `rc 0` fab-ready.**

| batch | seeds | design | build |
|---|---|---|---|
| A (5) | 2120717267, 1839490469, 758515152, 1145577041, 1688449119 | 0 committed | — |
| B (5) | 781647449, 1702462408, 1058319766, 1024581158, 168859774 | 0 committed | — |
| C (10) | includes 1666054899 | **1 committed** | **rc 0** |

The one that got through — seed **1666054899**, "A 2S Li-ion battery pack to 12 V converter with
reverse-polarity protection, an 8-pin 0.1 inch header, and a power LED. Use a four-layer stack-up." —
produced `leaf 3/3 accepted`, `parent_composed: true`, `parent_routed: true`, `kept_count: 1`,
`top_level_status: ready`, and a complete fab package
(`2S_12V_CONVERTER_fab_20261002.zip`: Gerbers + drill + CPL + BOM + STEP + 3D), verify
`shorts=0 unconnected=0 courtyard=0 keepout=0 traces=127 components=13/13`. That is the routing tail
working end-to-end on a random surprise brief with F1–F5 in place.

The refusals are deterministic contracts, not provider/transport errors — e.g. seed 2120717267:
`physical-obligation-unfulfilled=['battery:li-ion-battery-pack: requires 1 real battery-connector,
found 0; the unit emitted battery_connector=b2b-xh-a-lf-sn:B2B-XH-A…']`. Batch C's cost was **$0.143**
for 10 briefs, so the whole session spent well under $1 on provider calls.

Three controls say this is not a harness artifact:
1. The same harness commits a **corpus** brief (`rc-lowpass-bnc`: intent → functional_spec →
   architecture → bom → wiring all `ok`, 88 s, $0.0067).
2. Re-running seed 2120717267 with the **production instruction** (`instruction=None`, what
   `web._run_design` passes) still fails (`declared-interface-unrealized=['battery: declared interface
   needs one identified hardware owner']`).
3. Passing the live site's **`core_defaults`** registry rows (50 rows) still fails, now at
   **architecture**.
4. Every one of the 20 runs recorded `questions: 0` — no clarifying question was ever parked, so no
   auto-answered park is involved either. The failure kinds are `contract_rejected`,
   `commit_rejected`, `unit_repair_exhausted` and `invalid_schema`: deterministic refusals after the
   repair budget, not provider or transport errors.

The same harness on five **frozen corpus** briefs (`--corpus 5`) committed 2, refused 1 with the same
`commit_rejected`/`unit_repair_exhausted` kinds, and its one build that was not killed by the harness
timer (see §8.8's caveat) exported `rc 0` (`corpus02`, R-2R DAC: `leaf 4/4`, `parent_routed: true`,
`kept_count: 2`). So the frozen corpus is not 100% either, but it is far above the surprise space.

So the plan's premise ("the LLM pipeline is not the blocker any more") holds for briefs 1–3 but the
Surprise-me space as it stands refuses ~95% of its own briefs *before* any routing. This is an *open
axis* rather than necessarily a regression: the 2026-09-24 surprise-me session already measured
`control:stock **0/6**` committed on two frozen surprise-space refusals
(`docs/plans/live-surprise-me-round2-2026-09-24.md` §5).

Consequence for this session's goal: the surprise set barely exercises the routing tail, so the tail
was additionally validated on the **frozen routing-failure boards** (§8.8) and, as a surrogate, on
five frozen corpus briefs (`scripts/routing_surprise_cohort.py --corpus 5`).

### 8.7 Leaf escapement is the binding constraint (measured)

The RP2040 CONTROL leaf of project 1125 is the hard wall, and it is **geometry**, not budget. Router
input `2f574b5d-…__e355e9875c/leaf_routed.krt-input.kicad_pcb`, 43 single-ended nets, board clearance
0.153 mm, Default track 0.2 mm; one KRT invocation per row:

| grid / track | main pass | after rescue + reconciliation | wall |
|---|---|---|---|
| 0.1 / 0.2 (today's default) | 24 routed, **20 failed** | 20 failed | 71 s |
| 0.05 / 0.2 | 38 routed, 4 failed | 4 failed (`QSPI_SD0, gpio22, gpio25, gpio15`) | >420 s |
| 0.05 / 0.15 | 41 routed, 2 failed | 2 failed (`gpio21, gpio24`) | 120 s |
| 0.025 / 0.15 | 40 routed, 3 failed | 2 failed (`gpio23, gpio28`) | 288 s |

Two conclusions:
1. The router's own hint is right and this bridge ignored it (`--grid-step` was never passed) — F4 now
   exposes it. Through the pipeline bridge itself, the same leaf input with
   `kicad_routing_tools_grid_step_mm = 0.05` reports **39 routed / 3 failed** (`gpio14, gpio25,
   gpio27`) and the command carries `--grid-step 0.05`, i.e. 20 opens → 3. That turns a hopeless leaf
   into a nearly-routed one and gives the leaf search a real gradient for the first time.
2. Even the router's recommended fine geometry does **not** close this leaf (2–4 nets remain), so
   finer routing alone cannot make brief 3 fab-ready.

What the wall is *not*: pad crowding. Measured on the current leaf's placed board, the tightest
pad-to-**other-footprint**-pad gap on the failing nets is **2.478 mm** (137 pads; next-tightest 2.700,
2.747, 2.850 mm) — three orders of magnitude above the 0.153 mm rule. The router's "neighboring pads"
phrase describes blocked *grid cells* (other nets' already-routed copper plus its clearance ring at
0.1 mm resolution), not components sitting on top of each other. That is exactly why a finer grid
helps so much and why the remaining 2–4 nets are stubborn: they are decided by copper already on the
board, so what those nets need is a **placement** change (an escape channel reserved around the
fine-pitch part) or an escape-strategy change (via-in-pad / smaller-via-at-pad), not another router
knob. The leaf's own DRC/validation after a completed 300 s run still leaves 15–16 nets open
(`leaf_routed.router.stdout.log`, project 44/9102), so budget is not the binding constraint either.

Note the asymmetry that makes the finer grid non-default: the failing nets are decided in the *main*
pass (7–12 s at 0.05 vs 3 s at 0.1), while the extra wall time is the rescue grinding on the last
stubborn nets — the bridge-level 0.05 run took **365 s**, so the fine grid must be paired with a
matching `kicad_routing_tools_timeout_s` (the 120 s leaf default would SIGTERM it mid-recovery). A
global 0.05 would pay 4x cells on every board to chase nets that are still left open.

### 8.8 Frozen-board rebuild (routing tail, no provider spend)

Because the surprise set never reaches the build (§8.6), the routing tail was measured by rebuilding
the two frozen routing-failure designs in **copies** (original evidence untouched):

- `~/.kicraft/projects/44/9101` ← copy of 1121 (ESP32-S3 indicator, brief 2)
- `~/.kicraft/projects/44/9102` ← copy of 1125 (RP2040 motor controller, brief 3)

Both were copied with every absolute project path rewritten, `.experiments/` and `pins.json` purged,
and built with `kicraft build .kicraft/state.json generated --quality good` under
`KICRAFT_BUILD_MAX_WALL_S=1620` (the production web/worker budget) and the production `.env`. This is
the plan's §4 rebuild path, so no provider budget is consumed and the run is reproducible.

Status: both rebuilds ran to completion. Results:

**9101 ← brief 2 (ESP32-S3 indicator): `rc 0`, fab package exported.**
`[build] 3/5 promoted routed parent -> ESP32_INDICATOR_CONTROLLER.kicad_pcb`, verify
`shorts=0 unconnected=0 courtyard=0 keepout=0 traces=901 components=30/30`, and
`ESP32_INDICATOR_CONTROLLER_fab_20261002.zip` (Gerbers + drill + CPL + BOM + STEP + 3D) written.
Caveat: this success comes from a **fresh leaf solve** (a different placement draw from the original
run), not from F3 — the rebuild never hit the custody path (`declared unloaded input segment` count
in its build log: 0) and its parent routes took 94 s / 76 s, under either cap. What it does show is
that brief 2 is not structurally blocked. Its rounds still report `parent_routed: False` with
electrically complete boards (`unconnected=0`) rejected only on the **pre-existing clearance
violation** of §8.3.3 — the round gate and the build gate disagree, and the run reports
"Best score 0.00 / kept_count 0" while exporting a fab package.

**9102 ← brief 3 (RP2040 motor controller): `rc 6` → `rc 7` after the leaf-cap fix.**
First attempt (leaf cap 120 s, §8.2/F2): the RP2040 CONTROL leaf hit `router deadline … timed out
after 120s` in **every** round and canvas attempt, so it persisted **no artifact at all**; the parent
compose aborted with
`error: compose aborted -- child subcircuit(s) produced no solved artifact (their solve failed):
RP2040 CONTROL`, and the build ended **rc 6 with no board**.
After raising the cap to 300 s (and with F5 in place), the same rebuild gives: **zero** leaf router
deadlines, `leaf_total 6 / leaf_accepted 5` (the leaf is now in the set and correctly gate-rejected),
`parent_composed: True`, `parent_routed: False`, and a complete routed parent board —
`[build] 3/5 promoted routed parent`, verify `shorts=0 unconnected=19`, refused by the promotion gate
(**rc 7**). The residual 19 opens are exactly the leaf escapement wall of §8.7.

So the fixes moved brief 3 from *"no board to inspect"* to *"a complete board whose 19 opens are
explained"*, and left brief 2 exporting a fab package. Neither reaches "routing done" by §5's
definition for brief 3 — that still needs the placement/escape work of §8.7.

Live F1 evidence from the rebuilds: 9101's completion status reports `leafs: accepted=5 total=6` (the
original run reported `0 total=0`) and `composition_status: parent_failed` when composition failed,
where the original always said `complete`.

Harness caveat: `kicraft/eval/self_eval.run_build`'s watchdog is a plain `threading.Timer` started at
`Popen`, so it **includes host build-slot queue time** unlike the web worker's (which restarts at
`ACQUIRED_MARKER`). Corpus-build 01 in the surrogate cohort was SIGKILLed at that timer (−9) after
queueing behind a rebuild; later corpus builds, running with the slot free, complete normally.

### 8.9 Does the finer grid (+ budget) close brief 3? Measured: no

The F4 knob was exercised on the same brief-3 copy (`44/9102`) through the real build, with the
project's `.experiments` purged for a clean leaf solve and the env surface of §8.2/F2b:

```bash
KICRAFT_KICAD_ROUTING_TOOLS_GRID_STEP_MM=0.05 \
KICRAFT_KICAD_ROUTING_TOOLS_TIMEOUT_S=480 \
KICRAFT_KICAD_ROUTING_TOOLS_PARENT_TIMEOUT_S=600 \
KICRAFT_BUILD_MAX_WALL_S=3600 \
  .venv/bin/python -m kicraft.design.cli_app build .kicraft/state.json generated --quality good
```

The override demonstrably reached the solver: every one of the six leaves' `solved_layout.json`
records `config_hash` → `{"kicad_routing_tools_grid_step_mm": 0.05,
"kicad_routing_tools_timeout_s": 480.0}` (this is also how to verify a knob actually applied — the
artifact carries the solve's whole config).

Result: round 1 still reports `leafs=5/6` — the same single leaf has no *accepted* artifact even with
the finer grid and a 480 s budget, and the run spends its rescue round on that leaf. It does now
**persist** a best-effort artifact (0 router deadlines; all six leaves have a `solved_layout.json`),
so the compose no longer aborts on "no solved artifact". The parent phase then runs for **634 s** and
the parent router's stdout is **truncated mid-route** — the fine grid made the *parent* route exceed
even the 600 s cap I gave it (4x the cells on a 210 mm board) — and the build ends **rc 6, no routed
parent**.

So §8.7's conclusion 2 holds end-to-end, and it is why F4 ships the knob **leaf-scoped** with a
separate `kicad_routing_tools_parent_grid_step_mm`: a leaf-tuned fine grid must not be inherited by
the parent route. What the experiment proves is that the finer grid substantially reduces a leaf's
open nets (20 → 3 on the probed input) but does not close it, and that the remaining nets are a
placement/escape-strategy problem, not a grid or budget one.

Operational note: the build's step 1 re-synthesizes the per-project `*_autoplacer.json`, so a knob set
there is overwritten on every build — use the env surface (or `DEFAULT_CONFIG`) to change routing
behaviour for a run.
