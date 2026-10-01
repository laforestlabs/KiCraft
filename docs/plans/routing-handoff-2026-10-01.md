# Routing handoff — two of three live briefs are blocked at the build

Owner-facing handoff for a new session. Written 2026-10-01 on the production box, after the live
cohort `~/.kicraft/debug/cohort-2026-09-30-continuation` closed at 6 of 10 authorized loops.

**Read this first:** the LLM pipeline is not the blocker any more. Briefs 2 and 3 commit *all seven*
stages (intent → functional_spec → architecture → bom → wiring → electrical_review → silk_plan) and
then fail in place-and-route. Brief 1 is stable (fab-ready **and** compliant in its last three
passes). Everything below is about the build tail.

## 1. The failure signature

Two shapes, both ending in the same refusal:

| shape | what the build tail does | rc |
|---|---|---|
| **no routed parent at all** | the autoexperiment completes with nothing kept | 7 (and rc 6 in the older runs recorded by the previous cohort) |
| **routed but electrically incomplete** | a parent board is produced with nets left open | 7 |

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

### Brief 2 — ESP32-S3 indicator controller, `KC-WWYS3W`, project **1121**

Stages: all seven committed. Build job 351: `done rc=7`. `.experiments/run_status.txt`:
`parent_routed: False`, `top_level_status: not_ready`, `leaf_total: 0`, `kept_count: 0`,
`elapsed: 3m29s` — this is the **"nothing kept at all"** shape: no leaf was accepted, so no parent was
even composed. Contrast with brief 3, where leaves routed and only the parent's tail nets failed.

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

## 7. Do not

- Do not relax the promotion/verify gate to turn `rc=7` into `rc=0`. A board with 13 open connections
  is not a board.
- Do not treat a routed export as success on its own: the previous cohort's routed boards were
  non-compliant in size, part count and part identity. Route completion is the *entry* condition, and
  `compliance_read.py` in the cohort directory is the check to read afterwards.
- Do not re-open the pipeline patches in this session's commits (see `git log 688f7a9..bd5d5cc`) unless
  a build failure is traced to one of them; each carries the measurement behind it, and brief 1's three
  consecutive compliant passes say they hold.
