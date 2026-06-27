# Self-eval 2026-06-27 — most-impactful fixes (implementation plan)

**Audience:** an implementing agent. **Source run:** `logs/self_eval/20260627T032049Z`
(28-brief `BENCHMARK_PROMPTS` corpus, design=`deepseek-v4-flash`, judge=`minimax-m3`,
spend ≈ $0.99 + ≈ $0.10 re-grade).

This plan is built on the **judge-independent build reality** (parent-board
`_verify_routed_board` verdicts + KiCad DRC on the promoted boards), *not* the LLM
letter grades — grades measure synthesis quality and are a separate axis from
fab-readiness (a board can grade **B** and still be not-fab-ready).

---

## 0. Already landed in this session (do not re-do — verify + commit)

**Judge truncation fix.** 16/28 runs originally scored `final=None` because the
Class-J judge (`minimax-m3`, a reasoning model) was called with `max_tokens=1600`
and burned the whole budget on reasoning tokens before emitting the JSON answer
(`_extract_json` → `"no JSON object found in reply"`). Identical failure mode the
review gate already fixed (`review_max_tokens 3000→24000`); the judge never got
the same treatment.

Changes (uncommitted on `main` as of writing):
- `kicraft/server/config.py` — new `Settings.eval_judge_max_tokens: int = 24000`
  (+ `KICRAFT_EVAL_JUDGE_MAX_TOKENS` env, parsed in `from_env`).
- `kicraft/eval/judge.py` — `grade_class_j` default `max_tokens` 1600 → 24000.
- `kicraft/eval/run_web.py` — `evaluate_project(..., judge_max_tokens=)` threaded
  to `grade_class_j`; `main` reads the Settings knob.
- `kicraft/eval/self_eval.py` — `evaluate_one(..., judge_max_tokens=)` threaded;
  both call sites pass `s.eval_judge_max_tokens`.

Re-grading the 16 dropped runs with the fix recovered **every** grade (16/16,
$0.072), giving a complete **28/28** scoreboard: **mean 75.6 / median 77.8**,
grades **B×19, C×7, D×2** (was a biased 12-run subsample at 77.2/78.8). The
successful judge call is also *cheaper* (~$0.005 vs ~$0.0075, no wasted 2nd
attempt). `tests/test_web_self_eval.py` green. `summary.json`/`summary.md` were
regenerated in place from the recovered per-run reports (batch spend stays $0.99;
the $0.072 re-grade is a separate recovery action).

**Action:** commit these four files. Consider a regression test that asserts
`grade_class_j` is invoked with a ≥8k `max_tokens` from the self-eval/web paths
(the existing tests use a `FakeClient` and don't pin the budget).

---

## 1. The build reality (what to actually fix)

**8/28 fab-ready (29%).** The 20 not-fab-ready boards cluster into 5 root causes.
Designs usually have **more than one** blocker, so "designs touched" overlaps —
impact is "this is *a* blocker here", not "fixing this alone ships the board."

| Cluster | Designs touched | Code-side? | Fix confidence |
| --- | --- | --- | --- |
| **A. Connector stranding + its stranded high-speed nets** | ~9 (#2,5,6,8,9,13,14,22,25) | yes | medium (recurring family) |
| **B. Parent GND-pour islands** | 4 (#9,22,24,26) | yes | **high** (clean, contained) |
| **C. `courtyards_overlap`** | 6 (#1,4,6,8,12,23) | yes | medium (residual after KC-59PTZA) |
| **D. Route/infra fail — no routable parent** | 3 (#10,17,20) | partial | low (hardest, long tail) |
| **E. Synthesis honesty + schematic** | 3 (#4,11,27) | yes | **high** (deterministic, cheap) |

Recommended order: **B → E → A → C → D** (B and E are clean high-confidence wins;
A and C are the highest *count* but riskier and touch load-bearing placement/route
code; D is the long tail). Rationale per cluster below.

---

## 2. Cluster B — Parent GND-pour islands  *(do first: highest confidence)*

**Symptom.** Parent verdict `unconnected=N` where the open "nets" are
**GND-zone ↔ GND-zone** "missing connection" pairs, not signal nets.

**Evidence.** `run_26_servo-driver-16` parent: **8/8** unconnected are
`Zone [GND] on B.Cu ↔ Zone [GND] on B.Cu` / `B.Cu ↔ F.Cu` (KiCad DRC
`unconnected_items`, all `Zone [GND]`). Also #24 (1/1), #22 (1/5), #09 (1/3).
The GND copper pour fractured into disconnected islands with **no stitching via
tying an island back to the main GND region.**

**Root cause / where.** `kicraft/autoplacer/brain/gnd_pour.py` *already* has the
machinery — geometric union-find over a net's pads/vias/tracks/fill islands
(`gnd_pour.py:295`, `:349`) and an "island with no path to the main cluster… stamp
a direct same-net track" repair (`gnd_pour.py:417-422`). Yet run_26's **parent**
board ships 8 islands. So the island-repair is **not running (or not converging)
on the composed parent pour** — most likely it runs at leaf scope, and the parent
re-pour (after rigid leaf stamping + parent routing carves the plane) re-fragments
GND without a second island-repair pass.

**Approach.**
1. Reproduce: replay `run_26` (see §7) and dump the parent pour islands. Confirm
   the union-find repair is leaf-only and the parent path skips it.
2. Run the existing island union-find + stitch-track/via repair **on the parent
   board after the final parent re-pour**, looping until a single connected GND
   region remains (cap iterations; log if it can't converge).
3. Prefer a **stitching via** between overlapping B.Cu/F.Cu islands; fall back to
   the same-net track stamp for same-layer islands with a collision check against
   routed copper (memory warns thermal vias have been stamped *through* routed
   B.Cu — `kicraft-autopin-safety-net-misfire`).

**Acceptance.** `run_26` parent `unconnected` 8→0; #24/#22/#09 GND-island
component →0. No new `shorts`. `tests/test_gnd_pour*.py` green; add a parent-level
fixture with two GND islands and assert one connected region post-repair.

**Impact.** Removes the single largest open-count board (#26: 8) outright and the
GND component of 3 more. #26 has *no other blocker* → **+1 fab-ready immediately**.

---

## 3. Cluster E — Synthesis honesty + schematic  *(cheap, deterministic, do early)*

Three independent, small, high-confidence fixes:

**E1 — `silent_substitution` (#4 speaker-crossover, capped D/55).** The brief asked
for *binding-post terminals*; synthesis substituted `screw-terminal-5mm-2p`
(WJ126V-5.0-2P), labeled it a "binding-post substitute" in the sourcing note, and
**never surfaced it as an open question** for user confirmation. This is a real
honesty defect (the observer gate caught it). Fix at the BOM/part-resolution stage
in `kicraft/design/synthesis/`: when a resolved part is a *class substitution* of
what the brief named (not just a value pick), emit an `open_question` /
assumption rather than silently committing. Aligns with the user's standing
"surface substitutions, don't mask" preference.

**E2 — ERC error (#27 stepper-a4988).** 1 ERC error caps the grade at 45 and the
board never builds. Localize via `/kicraft-investigate` (×100-corrected coords) →
likely a pin-type / driven-net issue in the schematic emitter
(`design/synthesis/`). Compare against the prior ERC fixes
(`kicraft-switch-pin-not-driven-erc`).

**E3 — netlist faithfulness (#11 fpc-breakout).** §9.13 reports **1 unexpected net
merge** (a silent short introduced at wiring). Root-cause the merge in the wiring
stage / netlist gate; this is the same family as `kicraft-sch-label-slide-and-quote-escape`
(label slide merging nets).

**Impact.** 3 designs, all with cheap localized fixes; E1 is also a credibility win
(the product silently swapped a user-named connector).

---

## 4. Cluster A — Connector stranding + stranded high-speed nets  *(highest count, riskier)*

**Symptom.** Edge connectors placed **inboard** (negative mm from the edge) →
`connector_stranded:<ref>@-N.NNmm(<edge>)` + `illegal_routed_geometry`, **and** the
connector's own pins left unrouted.

**Key evidence — stranding and the signal-opens are the SAME root cause.**
`run_06_usb-c-full-breakout`: stranded `J2/J4/J5/J6 @ ~-2.56mm(right)` **and** the
open nets are `CC1, D_P, D_N, SSTX_P` — i.e. those very connectors' pins. A buried
edge connector can't route its high-speed/CC pins. The USB/diff-pair opens seen
across #05 (CC2), #09 (USB_DP/USB_DM), #12 (D+), #22 (USB_DP) are the same story.

**Per-design:** #02 `J1@-6.59(left)`, #05 `SW1@-1.14(top)`, #06 four J's right,
#13 `E1@-11.18(top)`, #25 `J1(left)/J2(right)/J3(bottom)`.

**Where.** This is a long, partially-solved family — read the memory chain first
(`kicraft-connector-stranding-edge-flush`, `-transform-local-point`,
`-root-cause-v2`, `kicraft-connector-pad-edge-clearance`). Load-bearing code:
- `kicraft/autoplacer/brain/placement_solver.py` `_pin_edge_components` (grows the
  leaf so an edge group sits at the leaf extremity).
- `kicraft/cli/compose_subcircuits.py` / `subcircuit_composer.py` — edge-zoned part
  must be a leaf **extremity** and re-based into the leaf box; the rigid stamp must
  land it flush.
- `_verify_routed_board` — the strict `connector_stranded` / `unconnected==0` gate.
- `connector_edge_*` config knobs.

**Approach (surgical — do NOT rewrite the placer).**
1. Pick `run_06` (4 connectors, richest) + `run_25` (3 edges) as the harness.
   Replay, then measure each connector's final offset from the parent edge.
2. The negative-mm offsets say the connector is being stamped *inboard of* the leaf
   box edge it was supposed to define. Trace UP to the single point that sets the
   inboard position (memory `kicraft-fix-at-source-no-hacks`: fix at source, no
   post-route nudge). Likely the leaf edge-extremity guarantee isn't holding for a
   **multi-connector same-edge group** (#06 has four on one edge).
3. Once flush, the CC/D_P/D_N opens should resolve because the pins reach the edge
   and the router can complete them — verify both the geometry gate **and** the
   `unconnected` count drop together.

**Acceptance.** #06 stranded→0 and `CC1/D_P/D_N/SSTX_P` routed; #02/#05/#13/#25
stranded→0. Add the existing flush-connector assertion to the new fixtures.

**Impact.** Largest design count; also clears the USB/diff-pair open sub-cluster.
But several of these boards have a *second* blocker (e.g. #06 also `courtyard×2`),
so pair with Cluster C to actually ship them.

---

## 5. Cluster C — `courtyards_overlap`  *(residual after KC-59PTZA)*

**Symptom.** `reasons=['courtyards_overlap']`, `courtyard=N` in the parent verdict.

**Evidence.** #01 `J1 (BNC) ↔ RV1 (trim pot)` overlap is **intra-leaf** (leaf DRC
`[courtyards_overlap] J1 / RV1`), so the leaf solver's final pass didn't separate
them. Also #04, #06 (×2), #08, #12, #23.

**Where.** The KC-59PTZA fix is present — Step 16 "Final courtyard-separation
legalization" (`placement_solver.py:1096`) and the same-side-pair handling
(`:3631`, with a `# Leave the stack intact` branch at `:3693`). The residual cases
are exactly the ones Step 16 declines to move: an **edge-pinned connector** (J1 is
pinned to the board edge) vs a neighbor (RV1) — separating them fights the
edge-pin constraint, so the stack is left intact and ships overlapping. Plus
**inter-leaf** courtyard overlaps introduced at rigid compose stamping that the
leaf-scoped Step 16 can't see (#06/#23 have ×2).

**Approach.**
1. Reproduce #01. Determine why Step 16's same-side branch leaves J1/RV1 stacked —
   confirm the edge-pin lock is the blocker, then allow Step 16 to slide the
   *non-pinned* member (RV1) along/away while the pinned member (J1) holds the edge.
2. Add a **parent/compose-level** courtyard-separation pass for inter-leaf overlaps
   (#06/#23) — the leaf pass cannot fix overlaps created by stamping two leaves
   adjacent. Read `kicraft-courtyard-overlap-placement-fix` (the Step-16 +
   minor-clip-gate design) before extending.

**Acceptance.** #01 courtyard 1→0; #06/#23 courtyard 2→0 without re-stranding the
connectors fixed in Cluster A (the two interact — verify together).

---

## 6. Cluster D — Route/infra fail (no routable parent)  *(long tail, lowest ROI)*

#10 `rp2040-min` (QFN-56) — "layout engine produced no routed parent board";
#17 `led-cc-driver` — route/infra failed; #20 `encoder-oled-panel` — `rc=1`.
Memory already flags `rp2040-min`, `encoder-oled-panel`, `proto-shield` as boards
that historically never route (`kicraft-tuning-framework`). These are dense/fine-
pitch parent-compose-route failures in `cli/autoexperiment.py` + the parent route
budget. **Defer** unless B/E/A/C don't move the fab-ready count enough — diagnose
each with `/kicraft-investigate` first; they are likely individual model/route
issues, not one shared bug.

---

## 7. Verification protocol (mandatory)

- **Reproduce with replay, not a fresh design** — deterministic, $0, no LLM:
  `kicraft replay --project logs/self_eval/20260627T032049Z/run_NN_…` (re-runs
  place+route only). See `kicraft-replay-command-and-determinism`.
- **Never compare artifacts across separate replay runs** — replay rewrites
  in-place and regenerates `.experiments`/parent each time; cross-run leaf-vs-parent
  diffs are a known false signal. Measure leaf+parent in ONE script after ONE
  replay (`kicraft-replay-cross-run-contamination`).
- **Fix at the source, no masking gates / post-route band-aids**
  (`kicraft-fix-at-source-no-hacks`). Trace up to the single point that sets the bad
  value.
- **Find the board honestly** with `kicraft artifacts`, not glob (`docs/ARTIFACTS.md`).
- **Guard against the test blind spot:** route tests mock the router, so they miss
  parent-route/pour regressions (`kicraft-parent-route-clear-zones-regression`).
  Add fixtures that exercise the real parent board where feasible.
- Re-run the relevant `tests/test_*` and, for a corpus check, a scoped
  `python -m kicraft.eval.self_eval --only <slug>` (now that the judge is fixed).

---

## Appendix — per-design blocker map (parent verdicts)

| # | brief | build verdict | blocker cluster(s) |
| --- | --- | --- | --- |
| 1 | rc-lowpass-bnc | courtyard=1 | C (J1↔RV1 intra-leaf) |
| 2 | r2r-dac | strand J1@-6.59 + illegal geom | A |
| 4 | speaker-crossover | courtyard=1 + gate | C + **E1** (silent_substitution) |
| 5 | usb-pd-trigger | strand SW1 + CC2 open | A |
| 6 | usb-c-full-breakout | courtyard×2 + 4 strand + CC1/D_P/D_N/SSTX_P open | A + C |
| 8 | rs485-terminal | short=1 + courtyard + 3 open + illegal geom | A + C (+short) |
| 9 | stm32-min | USB_DP/DM open + 1 GND island + illegal geom | A + B |
| 10 | rp2040-min | route/infra fail (no parent) | D |
| 11 | fpc-breakout | netlist faithfulness (1 net merge) | **E3** |
| 12 | esp32-s3-sensor | courtyard=1 + 1 open (D+) | C |
| 13 | nrf52-beacon | strand E1@-11.18 + X2_IN/RESET open | A |
| 14 | lora-node | +3V3 / SENSOR_SIG open | A (signal) |
| 17 | led-cc-driver | route/infra fail | D |
| 20 | encoder-oled-panel | rc=1 route fail | D |
| 22 | esp32-dual-motor | 5 open (USB_DP, motor sigs, 1 GND) + illegal geom | A + B |
| 23 | can-node | courtyard×2 + 2 open | C |
| 24 | daq-8ch | 1 open (GND island) | B |
| 25 | gpio-expander | strand J1/J2/J3 + illegal geom | A |
| 26 | servo-driver-16 | **8 open — all GND islands** | **B** |
| 27 | stepper-a4988 | 1 ERC error | **E2** |

Fab-ready (8): #3 thermocouple-amp, #7 usb-a-power-splitter, #15 buck-3a,
#16 highside-switch, #18 dual-rail-supply, #19 relay-quad, #21 proto-shield,
#28 audio-jack-buffer.
