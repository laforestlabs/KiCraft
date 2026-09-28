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

## Continuation execution record

Pass 6 repair: the quantity interpreter now explicitly distinguishes separate
components from contacts per connector and retains property counts as typed
quantitative minima. The existing connector geometry compiler consumes those
minima; BOM validation checks compiler-declared connector contacts, including
unused contacts, rather than accepting any header of the same broad class.
Actual plural component counts retain their original minimum.

Evidence is saved under the frozen cohort directory in `pass6-quantity-probe.json`,
`pass6-quantity-final-probe.json`, and `pass6-header-smoke.json`. The real decision
service misread the recorded header subject in 3/3 baseline calls and retained
eight contacts as a property in 3/3 final calls. The frozen BOM exercise rejects
the erroneous eight-component demand, accepts one eight-contact header after
the typed correction, and replaces an undersized two-contact proposal with the
compiler's eight-contact realization. These probes do not establish live cohort
completion; fresh pass outcomes are recorded separately.

Pass 6 live: projects 1036–1040, $0.087949610 project spend; power, controller,
and analog reached BOM, while relay and LED stopped at architecture. No build
started. LED's generated `LED_OUTPUT REGULATOR` sheet name violates the canonical
schema; this independently reproduced compiler defect is the pass 7 target.
Neither earlier stop establishes a regression caused by the contact-count repair.

Pass 7 repair: generate regulator sheet display names using the canonical
uppercase alphanumeric/space alphabet, while retaining underscore-separated
stems and original electrical rail names. Both generated-rail and missing-source
completion paths use the corrected names consistently for sheet ownership.
The before/after smoke changed `Sheet` validation failure into an accepted
`LED OUTPUT REGULATOR` sheet with the converter bound to that same sheet.

The enable-jumper review found no runtime-reviewed removable shunt assembly.
Samtec's SNT-100-BK-G is a concrete accessory candidate (2.54 mm pitch,
0.64 mm square posts, 4.32 mm minimum insertion depth; manufacturer source:
https://suddendocs.samtec.com/catalog_english/snt.pdf). This is not a declaration
of supported hardware. Registration and assembly representation require the
owner's capability decision; bare headers and solder bridges are not silently
accepted as removable shunts. Other actionable cohort repairs continue.

Pass 7 live: projects 1041–1045, $0.080264455 project spend; 4/5 reached BOM.
LED reached the MCU unit and now fails BOM commit on programming access (9.29).
Power 1041 routed/exported with zero shorts and unconnected nets, but is **not
compliant**: SW1 is a momentary button instead of an enable jumper; C1/C2 on
the 18 V protected input were sourced as C19702, rated 10 V. Neither defect was
introduced by the sheet-name repair; both remain unresolved safety gaps.

Pass 8 repair: recognize the already-reviewed ESP32-C3-MINI-1-N4 as an
`esp32-c3-module`, and map the broad `wireless-module` demand to the existing
`wifi-module` feature. No new part, package, pin, or voltage support is added.
The actual 9.42 gate rejected that exact reviewed module for both spellings
before this change and accepts it afterward. STM32/regulator and bare-radio-IC
counterexamples must remain rejected; an ESP32-S3 is not an ESP32-C3.

Pass 8 live: projects 1046–1050, $0.085327255 project spend; 2/5 reached BOM,
no build. Controller moved past its connector to reset-button binding; LED
accepted four BOM units then failed declared MCU selectors. Module terminology
is proven by direct 9.42 replay, not by a completed live BOM gate. Power and
analog stopped earlier at architecture; no causal regression established.

Pass 9 repair: validate model-declared interfaces against an exact reviewed
part's installed symbol during architecture derivation. Resolve unique
published pin names to their physical numbers, but reject absent/ambiguous
selectors before locking an impossible BOM contract. Unavailable symbols
remain subject to the existing downstream gates; no part/family substitution.
The frozen 1042 connector claim (pins 1–4 on B2B-PH-K-S) compiled before and
now fails `declared_port_unknown_contact`, naming the real pins 1–2.

Pass 9 live: projects 1051–1055, $0.130891715 project spend; 3/5 reached BOM.
Relay 1053 committed all five stages, accepted ten BOM and ten wiring units,
routed and exported. Final promoted geometry is 242.821703 × 99.919471 mm;
zero shorts/unconnected/courtyard/keepout failures do not make it compliant.
It has sixteen relays and no proper discrete coil-driver/flyback hardware.
ERC reports 206 warnings (zero errors); power 1041 had 50 warnings.

Pass 10 repair: the final saved relay authoring response omitted optional
replication-group hints. Four distinct sheets still require four physical
instances, but the old normalization charged the shared minimum four to
every sheet unless they shared a replication group. Use the actual distinct
owning sheets as the proof, independently of replication hints. Retain the
board-level quantity, local physical-presence checks, and all insufficient-
instance/unrelated-class checks. This is not an exact-upper-count gate.
Frozen-response replay removes eight false physical-count errors (relays and
terminals); three real declared-interface mismatches remain on the artificial
four-relay subset. It is not an electrical-success claim. Capacitor voltage
qualification, jumper hardware, driver/flyback topology and outline enforcement
remain separate unresolved safety work.

Completed pass 10: projects 1056–1060, $0.090969390 project spend; 3/5 reached
BOM. Relay 1058 accepted 11 BOM and 11 wiring units, has exactly four relays,
and routed/exported with zero shorts/unconnected. Final promoted geometry is
132.437823 × 68.554839 mm, so it still violates the 100 × 100 mm limit.
Its 5 V coil positive is tied to 7.4 V VBAT, with the return wired directly
to the MCU and no proper driver/flyback circuit. ERC has 135 warnings.
LED 1059 accepted all five BOM units but fails commit on programming access.
Controller 1057 fails reset-button identity/binding; its two-contact JST still
contradicts the retained four-contact quantitative obligation. The early pin
selector guard therefore does not complete connector-geometry acceptance.

All five additional loops executed; no pass 11 started. Three exported boards
in this continuation, **zero compliant fab-ready**. Final verification:
551 focused tests passed; every deployment verified HTTP 200 and worker ready;
final HTTP 200, no active projects/build jobs, account quota 6/20 used.
Continuation project spend is $0.475402425; conservative ledger charge,
including diagnostics and untagged decision calls, is $0.517238244.
No source commit was created: base 405f45bd plus the deployed working-tree
fingerprints recorded with each pass.

Full results, unit IDs, timings, regressions, source fingerprints and final
board audits: `~/.kicraft/debug/cohort-20260927-diverse/continuation-summary.md`
and its JSON companion. Next authorized work should enforce the retained
contact-count obligation for model-declared interfaces before locking a
two-contact JST; reset identity/binding and MCU programming access are other
upstream blockers. Electrical source ratings, coil drive/suppression, removable
shunt capability and final outline limits remain mandatory before fab credit.

## Additional authorization: passes 11–15

Owner: “continue up to 5 more loops”. Continue the same frozen five briefs,
in order, for passes 11–15. Stop early for genuine 5/5 compliant success or
a hard production/provider/spend/quota boundary, not flat routing. Preserve
all pass 1–10 records and the prior continuation report.

Starting checkpoint: base 405f45bd plus the deployed changes from passes
6–10. No active projects/build jobs; drive quota 6/20 used. Starting spend
ledger total $140.97817630648. Reserve at most another $15 including
diagnostics, while retaining the $30 cohort ceiling and the prior charged
$1.068260409 including the discarded cohort. Do not alter production limits.

Pass 11 targets the retained four-contact JST requirement accepting the
two-contact B2B-PH-K-S. Exact replay of project 1057's requirement currently
compiles pins 1–2 while retaining `JST connector pins = 4`; pin existence
alone is not sufficient. Later repairs follow the latest five-run evidence.

Pass 11 repair: the reviewed symbol's physical contact inventory must meet
the retained connector contact count, not just the subset wired by the model.
The existing count interpretation is shared with header sizing and restricted
to count/contact units, so pin voltage/current cannot become a contact count.
Frozen 1057 replay accepts the two-contact part before and rejects it after;
unused contacts on a real four-contact part remain legal. No substitution or
new hardware support. Verification: 392 focused tests passed.

Pass 11: projects 1061–1065, $0.105448620, zero successful exports.
All five committed architecture. Controller 1062 still rejects the actual
reset pushbutton; LED 1064 lacks SWD access; power/analog lack a removable
jumper. Relay 1063 completed routing but build rc7 refused the stranded
antenna. Its 119.05335 × 65.484543 mm outline exceeds the brief, and the
four relay coils still lack drivers/flyback. No routing failure is hidden.

Pass 12 targets only the reset-button vocabulary mismatch. Stored
`pushbutton` did not intersect demanded `momentary-button` despite the
existing demand aliases defining those as equivalent. Normalize the four
existing true button synonyms on record load and share their mapping with
demand lookup. Do not reverse one-way alternatives such as wireless/Wi-Fi,
header/socket, or status-LED/package. Actual K2-1109DF-E4SW-04 replay changes
from refused to accepted as a pushbutton, remains refused as a jumper.
Pin-owner and wiring checks are unchanged. Verification: 473 tests passed.

Pass 12: projects 1066–1070, $0.072961935. Controller stopped before BOM,
so the naming repair remains runtime-replay verified, not live validated.
Power 1066 exported (zero shorts/opens, 54 ERC warnings), but is not compliant:
SW_Push replaced the requested jumper, and exported C1/C2 are C19702
10 V capacitors on 18 V VIN_PROTECTED. Other four failed before wiring.

Pass 13 closes the observed adjustment-mechanism bypass: `switch-input`
may implement momentary-button adjustment, not a jumper, rotary switch or
potentiometer. The existing architecture contract reports the refusal;
no saved state or library capability changes. Exact 1066 replay now refuses
the substitution. Six behavior regressions cover incompatible mechanisms
and genuine momentary contacts. Verification: 479 tests passed.

Pass 13: projects 1071–1075, $0.102081795. Relay 1073 exported but is
124.189822 × 75.396635 mm, with 181 ERC warnings and incorrect relay pin
functions. The other four stopped before wiring. Power refuses the missing
jumper; controller's purported four-contact JST is again a two-contact B2B.

Correction to the earlier relay analysis: the existing symbol's drawn coil
joins pins 1/4, common is 5, NC is 3 and NO is 2. The footprint matches the
manufacturer's C-form bottom-view diagram (C35449 catalog datasheet, page 2).
Generated 1063/1073 instead call pins 1/2 the coil. The earlier direct-MCU-coil
description trusted those model claims; actual GPIOs reach the NO contact.
Driver/flyback omissions remain, but adding them alone cannot fix the pin map.
Primary evidence is retained as `srd-c35449-datasheet.pdf` in the cohort record.

Pass 14 extends existing §9.38 to the catalog rating of a selected capacitor
across two typed DC rails. Compare terminal voltage difference (including
negative rails), not either rail alone; cache ratings, never voltage stress.
A selected C# without a verified rating is refused on that known DC span.
Unselected/unpriced capacitors and untyped signal/AC stress remain outside
this check; no full voltage-certification claim. Sourcing selection is not
silently changed. Exact 1066 replay now rejects C1/C2 at 10 V on 18 V;
an in-memory substitution to catalog-proven 25 V C96446 passes. Saved boards
are untouched. Existing wiring-commit and synthesis callers use the same gate.
Verification: 486 tests passed.

Pass 14: projects 1076–1080, $0.097381400, no builds. Jumper, reset-owner,
relay-driver-owner and MCU programming-access refusals remain; analog stopped
at an unwired header. All failed before wiring, so capacitor protection has
saved-state replay evidence but no live-path validation in this pass.

Pass 15 corrects the existing SRD relay library's missing pin functions:
COIL_A=1, COIL_B=4, COM=5, NC=3, NO=2, all physically passive contacts.
These are the existing symbol drawing and footprint's contacts, cross-checked
against the manufacturer's C-form bottom view; no pin numbers, footprint,
part identity or supported topology changes. Reviewed port facts now carry
the same map and a working primary datasheet URL. The non-polarized coil's
A/B labels do not impose a supply polarity. Before, named coil/common
selectors fail; after, actual architecture derivation resolves them to 1/4/5.
Verification: 491 tests passed; KiCad CLI renders the corrected symbol,
including a subsequent label-size-only adjustment to avoid overlap.
This exposes hardware facts; it does not claim a new relay-driver/flyback
validator or prove that every numeric model-authored pin claim is correct.

Pass 15: projects 1081–1085, $0.112605370, no builds. Power still lacks
a real enable jumper. Controller stopped on an invented `jst-ph-4p` symbol
library; its blank project cost is recovered separately from ten attributed
ledger charges ($0.016621885), without altering the saved project.
The four-contact guard also refused the two-contact JST earlier in this run.
Relay committed its BOM and accepted seven wiring units, but the final
consistency check refused twelve contradictions. Initial wiring used the
correct contact map; later repairs still conflict with wrong architecture.
The LED board still claims unavailable MCU pin names; analog still leaves
requested hardware unattached to an implementation.

Five-loop authorization exhausted: 25 runs, three routed boards, two exports,
zero compliant boards. The exported power and relay designs remain unsafe;
no saved board was repaired or declared fabrication-ready. Final behavioral
test batch: 491 passed; the later symbol-label-only change was visually
verified by KiCad SVG export. Provider/profile, briefs and spend caps stayed
unchanged. No pass 16 was launched.

Recorded run costs total $0.490479120. The conservative global ledger increase
is $0.526571131, including diagnostic/other charges during this continuation,
below the $15 authorization. Combined with the prior recorded cohort spend,
the conservative cumulative figure is $1.594831540, below $30. Final production
health: HTTP 200, no active projects/build jobs, account quota 8/20 used.

Next checkpoint, before another live loop: reconcile declared relay pin
functions against reviewed physical contacts before accepting architecture;
then review coil drivers/flyback and the 100 mm outline requirement. Separate
remaining gaps are the removable-shunt assembly, reset/JST hardware ownership,
STM32 programming access/contact mapping, and analog obligation ownership.
The missing-symbol exception and blank project-cost field are recorded
separately; they were not hidden by a fallback. New capability work requires
its normal hardware review, not a safety waiver.

Final evidence is in the existing cohort directory:
`passes11-15-summary.{md,json}`, `passes11-15-scorecard.json`,
`passes11-15-board-audits.json`, `passes11-15-verification.json`,
and per-pass launch, triage and comparison records. Prior pass 1–10 summaries
are preserved. Stage-depth comparisons describe observed execution, not
causal repair claims or hardware quality.
