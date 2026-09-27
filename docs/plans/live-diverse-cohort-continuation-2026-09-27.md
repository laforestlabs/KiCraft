# Continue the diverse live cohort: five more loops

## Owner authorization and progress policy

Owner, 2026-09-27:

> commit and push changes so far then write a plan to continue for another 5 loops in a new session. this is the type of slow progress i was talking about. lets continue to incrementally make the pipeline robust so that we march towards 5/5 fab ready

Run **five additional fix-and-rerun loops, numbered 6–10**, in the next session. Keep the same five diverse briefs, verbatim and in the same order. Do not generate a replacement cohort. The ultimate target is **5/5 compliant fab-ready**.

**Upstream progress counts.** Moving from 1/5 to 5/5 through architecture, getting more parts units accepted, removing a repeated false rejection, and exposing a later genuine defect are meaningful gains. Do not stop this continuation merely because routing remains flat. The previous session's decision not to extend on that basis is superseded by this authorization.

Complete the five additional loops unless all five genuinely succeed, a hard spend/quota/provider/production-safety boundary prevents proceeding, or a material design/capability decision genuinely requires the owner. A stalled brief is not a reason to discard it, replace it with an easier brief, or abandon actionable fixes on the other four. Do not silently expand unsupported hardware capabilities or weaken gates. Do not authorize passes beyond 10 without the owner.

This document is a handoff, not a claim that loops 6–10 have already run.

## Starting point

- Branch: `simplify/bom-wiring-pipeline`, tracking `origin/simplify/bom-wiring-pipeline`.
- Implementation checkpoint: **`859e383`**, pushed to origin. The handoff is committed separately after this checkpoint.
- Production checkout: `/home/kicraft/KiCraft`; live service is `https://kicraft.io`.
- Cohort directory: `/home/kicraft/.kicraft/debug/cohort-20260927-diverse/`.
- Read `cohort.json`, `selection.json`, `scorecard.json`, `comparisons.json`, `findings.json`, `pass_5.json`, `source-state.json`, and `run-log.md` there before changing code.
- All five final runs stopped at BOM. No projects were running at the previous handoff; recheck before any restart.
- These local artifacts are on the production machine, not in Git. Do not assume they are present in a fresh remote checkout. Do not recreate historical results if artifacts are unavailable.
- The discarded seeds63–67 cohort is not a baseline or a source of substitute briefs.

### Previous measured outcomes

| Pass | Projects | Reached BOM | Completed routing | Compliant fab-ready |
|---|---|---:|---:|---:|
| 1 | 1011–1015 | 1/5 | 0/5 | 0/5 |
| 2 | 1016–1020 | 4/5 | 1/5 | 0/5 |
| 3 | 1021–1025 | 4/5 | 0/5 | 0/5 |
| 4 | 1026–1030 | 4/5 | 0/5 | 0/5 |
| 5 | 1031–1035 | 5/5 | 0/5 | 0/5 |

Project1018 routed with zero shorts and zero unconnected nets and exported a fab package. It is **not a compliant success**: approximately99×163mm against a100×100mm limit, sixteen rather than four relays, and no proper coil-driver/flyback hardware. The quantity fix subsequently produced four relays in live pass4. Do not conflate a successful exporter with a correct board.

## Frozen briefs: use exactly these texts

The seed and original project id in `cohort.json` identify the first pass, not the latest run. Record fresh ids in every new pass file.

### 1. Power — seed1790522530 — latest1031 / KC-MY9RS9

```text
An 18 V DC input to 3.3 V converter with reverse-polarity protection, an 8-pin 0.1 inch header, and a power LED. Add an enable jumper.
```

### 2. Controller — seed1790522531 — latest1032 / KC-YN7Z4A

```text
An ESP32-C3 module controller board powered from a 5 V header from the host board, with a 4-pin JST connector for I/O, a UART header, and a power LED. Add a reset button.
```

### 3. Actuator — seed1790522532 — latest1033 / KC-GA9RJJ

```text
An ESP32-C3 module relay board with four relay outputs on screw terminals, a 2S Li-ion battery pack, and a status LED. Keep it under 100 x 100 mm and use common, easily sourced parts.
```

### 4. LED driver — seed1790522533 — latest1034 / KC-MNDV4C

```text
An STM32G0 LED driver board for a WS2812 addressable LED strip, powered from a 5 V header from the host board, with an I2C header and an 8-pin 0.1 inch header. Make the input reverse-polarity protected.
```

### 5. Analog — seed1790522534 — latest1035 / KC-68X3CA

```text
A precision current-sense conditioning board: LM358, an 18 V DC input, three push buttons, and a status LED, under 100 x 100 mm. Add an enable jumper.
```

## Existing fixes to preserve

1. `architecture_intent.py:derive_architecture`: external-source signals such as `edge:SENSE_INPUT -> opamp.input` use the existing connector path while retaining all peers and the actual pin direction. Frozen replay passes; analog architecture committed in passes2–5.
2. USB supply feedback distinguishes an ambiguous choice between declared `VBUS` and `+5V` from absence of a supply. No supply-selection or voltage gate was weakened. Controller eventually committed architecture in passes4–5; do not attribute every stochastic improvement to this wording change.
3. `ArchitectureIntent._obligations_are_owned_once`: replicated sheets no longer each inherit a board-wide minimum when enough distinct sheets already require a matching physical instance. Four local relay units now ask for one each, not four each. `relay-four-instance-gate-smoke.txt` proves whole-board gate9.42 rejects four total before normalization and accepts them after. This is not an exact-upper-count gate.
4. `part_identity.py:_FAMILY_MEMBERS`: `STM32G0` recognizes only the already-reviewed `STM32G030F6P6`. No new package or prefix inference. Deterministic controller ownership changed false→true; pass5 stopped at the header before exercising the MCU BOM unit, so full live verification remains open.
5. `kicraft-debug` skill preflights diversity before spending. The random generator still samples families with replacement; distinct seeds do not ensure distinct designs. For this continuation, diversity is already established: reuse the frozen cohort.

Prior verification:94 architecture tests;247 architecture/work-unit/electrical-invariant tests after quantity repair;418 identity/work-unit/electrical/recipe tests after family recognition. These are historical results, not a substitute for verifying each new change.

## Candidate work order for loops6–10

One coherent root cause per deployed pass; reorder using current evidence. Every pass reruns all five. Do not implement five unrelated patches first and lose attribution.

### Loop6: contact count versus component count

- Evidence: project1034's `aux_header` was rejected with `requires 8 real pin-header, found 1` for one requested eight-pin header.
- Trace the mistaken count back through intent, functional specification and architecture. A contact count belongs to connector geometry; it is not eight physical connectors.
- Starting points: `kicraft/server/reconciliation.py`, quantity interpretation in stage contracts, `models.py` quantity representation, and the consumers in `stage_work_units.py` and synthesis validation.
- Existing prior-art record: `~/.kicraft/debug/cohort-2026-09-27/regex-jev-audit.json`, entryA1 (quantity shortlist interpretation). It is a lead, not a verified fix.
- Prefer typed distinctions and the existing Jev interpretation path for prose ambiguity, not a new collection of description regexes. Preserve actual plural counts: eight headers still means eight headers; two eight-pin headers means two components with eight contacts each.
- Acceptance: one correct eight-contact header is not refused for lacking seven extra headers, while a two-contact part still fails an eight-contact requirement. Then observe whether the LED design reaches and passes its MCU unit.

### Loop7: enable-jumper representation and identity

- Evidence: projects1016/1020/1025/1030/1031/1035 repeatedly fail the enable-jumper requirement. This affects both power and analog briefs.
- Starting points: `stage_work_units.py:_normalize_bom_optional_metadata`, `_normalize_curated_group_identities`, `_bundled_reviewed_record`, `_group_has_physical_feature`; `part_identity.py` physical inventory and trusted hardware witnesses; gate9.42 in `design/synthesis/validation.py`.
- Distinguish separate problems: nonexistent KiCad footprint; invented MPN; a valid bundle MPN containing spaces being erased; no reviewed representation of the actual header/shunt assembly. Fixing one does not prove the others solved.
- A bare two-pin header is not evidence that a removable shunt exists. Neither a regulator nor an arbitrary real part can satisfy a jumper merely because the class is uncovered. Preserve the existing negative regulator-as-jumper regression test.
- Use an existing reviewed realization when sufficient. If additional hardware review is required, identify the exact part/accessory, source its datasheet and contact/package evidence, and surface the capability decision; never silently declare every header a jumper.
- Acceptance: the exact requested enable function has actual, sourceable hardware and correct connections; both affected briefs get a fresh live test. Do not count omission of the jumper as progress.

### Loop8: connector geometry before architecture locks the part

- Evidence: projects1027/1032 specify four contacts but select `B2B-PH-K-S`, whose symbol has only pins1–2; parts selection cannot satisfy declared pins3–4.
- Trace where the wrong two-contact identity is selected or normalized. Keep contact-count compatibility upstream of the irreversible architecture choice.
- Select a compatible already-reviewed four-contact part only if it satisfies the brief's connector requirements. Never fabricate pins3–4, permit an unrelated footprint, or silently substitute a stricter named family.
- Acceptance: all four declared contacts resolve to actual pins; the controller proceeds beyond its connector BOM unit. Wrong-size and incompatible-family counterexamples stay rejected.

### Loop9: repeated physical-class failures in the relay design

- Evidence: project1028 rejects `wireless-module`; project1033 rejects `esp32-c3-module` plus incorrect relay realizations. The design already has reviewed ESP32-C3 hardware, but not every emitted class spelling maps correctly.
- Determine whether each failure is a vocabulary mismatch for correct hardware or genuinely wrong hardware. Inspect the actual group, reviewed identity, package and pin map before changing interpretation.
- Preserve the four-total relay fix. Do not allow a screw terminal to stand in for a relay, or allow a12V coil part to replace a5V coil part on name similarity.
- Acceptance: the existing reviewed MCU module and the correct relay devices satisfy their real demands; incorrect parts still fail. Check coil voltage, driver topology and suppression before claiming electrical success.

### Loop10: next earliest blocker, with honest build acceptance

- Choose from the remaining earliest failures after loops6–9 rather than forcing a cosmetic routing change.
- Any board reaching wiring/build needs a brief audit before it can be counted. Carry forward the project1018 defects: board-size maximum not enforced, relay coils wired without suitable driver/flyback protection, and source/voltage/component-count correctness.
- If working on those safety gaps, change one coherent cause at a time and record unresolved siblings. Put electrical drive checks at the earliest stage that has the necessary circuit evidence; carry board-size constraints to placement and enforce them at final verification. Do not merely relabel a bad export.
- Read `skill://verify` before any place/route/promote implementation change and use its frozen-workspace verification procedure.
- Finish with all five live outcomes and a concrete next checkpoint toward5/5, even if the result is still pre-routing. Do not start pass11 without further authorization.

## Execution contract for the next session

1. Read `skill://kicraft-debug` and `skill://kicraft-investigate`, then this plan and the local cohort artifacts. Initialize separate work items for loops6,7,8,9,10. Preserve baselines for every file touched; do not commit other people's unrelated changes.
2. Check current branch/deployed source, active projects/build jobs, drive-account quota and configured spend limits. Never interrupt someone else's production run to deploy a cohort fix.
3. Reproduce the selected defect on current code using the saved input. Use `triage run`, `triage stages` and `triage audits` for each run; do not mistake historical retry messages for its final failure. Offline compiler/gate exercises diagnose deterministic defects; live stage replays diagnose provider/prompt behavior. N-of-3 before claiming a stochastic fix is reliable.
4. Patch at the source, preserve hard safety checks, and add focused behavioral regression coverage where warranted. Exercise the actual changed path. Document the evidence and what remains unproven. Read existing patterns rather than introducing a second interpreter or compatibility shim.
5. With no cohort design in flight, deploy through `./deploy/deploy-production.sh`. Require web HTTP200 and build-worker ready. Re-login afterward. Do not use systemctl; these production services are detached processes.
6. Submit all five briefs through the browser on `https://kicraft.io`, using `New design`, exact text replacement and `Design`. Assert the textbox value before submission and the database brief afterward. `fill` previously appended text in this UI: focus, Control-down/A/Control-up, Backspace, then type if needed. Start all five before waiting. Assert exactly five matching project rows; never click `Surprise me` for a rerun.
7. Wait for all five terminal outcomes. Track saved stages and accepted/failed parts units while running; post progress notes on material transitions. Do not rely on the page alone, and do not treat a long-running row as abandoned before investigating the worker.
8. Save the pass and compare all five before selecting the next repair. Move immediately to the next authorized loop; a flat routing count alone is not a stop condition.

Account: `ui-drive@kicraft-test.dev`, id44. Credential file: `~/.kicraft/debug/live-cred.json`, mode600. Never copy credentials or `.env` into the plan, logs or Git. The previous handoff had3/20 weekly quota slots consumed; recheck each pass, especially as more runs succeed. Do not alter quota or project status to evade admission limits.

### Budget

Observed production limits were $0.60 per project, $20 daily and $250 total; read the actual configuration and honor any tighter values. Do not raise limits or change providers/models to obtain a pass.

Five more passes permit25 live designs: reserve at most **$15 for this continuation, including any paid diagnostic replays**. Also retain the existing $30 cohort ceiling. Recorded spend so far is $0.451856105 for this cohort; the separately discarded cohort cost $0.099166060. Account for diagnostics and excluded/accidental launches too. Check remaining allowance before each paid operation and before admitting a full pass; never discard costs from the denominator. A cap/quota stop is a blocker report, not a successful completion.

## Durable records and success criteria

Do not overwrite `pass_1.json` through `pass_5.json`, the original frozen cohort or old final report. Append `pass_6.json` through `pass_10.json`, per-pass launch records, findings and run-log entries; write a separate `continuation-summary.md` and machine-readable continuation summary. Record source commit/tree fingerprints, provider/profile, production deployment verification, actual project ids, costs and wall times. A null historical `projects.pipeline` value does not establish which configuration a new process uses.

For every brief, report both latest outcome and best observed outcome:

- last committed stage;
- accepted/failed BOM and wiring units, with concrete remaining error;
- build started, routing reached, and routing completed as separate facts;
- shorts, unconnected nets, warnings and fatal findings from the final promoted board;
- compliant fab-ready only after exported artifacts, clean required checks, and brief/electrical audit;
- cost and any regression relative to the previous pass and the best prior state.

Distinguish stochastic stage regression from an established patch-caused regression. A new safety check correctly rejecting an already-invalid board is not loss of a genuine fab-ready design. Conversely, do not hide a real regression behind aggregate gains. Refine/revert a causal regression before accepting the fix.

Completion report: five additional loops executed (or exact hard blocker), per-loop five-brief table, repairs with deterministic/live proof separated, unresolved safety and capability gaps, cumulative spend, source commits, and `Session changes: +N lines added, -M lines removed.` The next action should target the earliest remaining generalizable blocker, not promise5/5 without evidence.

## New-session launch instruction

> Read `docs/plans/live-diverse-cohort-continuation-2026-09-27.md` and activate the live cohort debugger. Continue the same frozen diverse cohort for passes6–10, fixing one evidenced root cause per pass and rerunning all five unchanged briefs through kicraft.io. Slow architecture/BOM/wiring progress counts; do not stop merely because routing is flat. Preserve safety gates, budgets, prior results and honest brief-compliance audits. Work toward5/5 compliant fab-ready and report each pass.
