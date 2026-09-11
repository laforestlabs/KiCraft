# Evidence-driven five-stage recovery

Status: implementation in progress; no acceptance or deployment claimed.

## Central execution loop

The controlling workflow is **fresh live verification → analyze saved results → update owning code → fresh live verification**. Continue this loop until a full fresh campaign commits **34/34** designs, then require the canonical independent deploy canary to do the same. Unit tests and frozen fixtures support a fix; they never replace live verification or constitute completion.

Each round retains its directory, source fingerprint, provider/profile, limits, paid spend, and every terminal failure. Analyze events, rejected work units and state, identify the earliest concrete mechanism, implement its general fix, verify it offline, then run a new affected live round. Once affected failures clear, expand to the complete 34-brief campaign. A full-campaign failure feeds the same loop, not an unchanged retry or a reduced acceptance target. Infrastructure or budget blockers are reported explicitly without declaring this objective complete.

## Critical review of the continuation plan

The five-stage scope, strict electrical gates, typed-only deterministic completion, retained siblings, finite spend, and pre-deploy gate are retained. The continuation document is not itself evidence that its proposed mechanisms are correct.

1. Its six-run report proves 1/6 committed, not that pin allocations are lost at a particular seam. Reconstruct ownership before changing it. A model-owned pin inventory is not proof that a selector topology works.
2. The reported campaign cost is wrong: summary total $0.0822 excludes two exception records whose messages report $0.0648 and $0.0635 spent. Comparing cost per success before fixing failure accounting would be misleading.
3. A named project budget already exists (`Settings.project_llm_budget_usd`, default $0.10). Reuse it; do not introduce a competing evaluation setting. Test $0.15 through configuration before changing the production default. Admission of the next call is not evidence that the board will complete.
4. Reserving all future mandatory calls upstream assumes the BOM and wiring ownership plans already exist. They do not. Do not fabricate reservations or starve required architecture repair to protect hypothetical work. Preserve conservative call preflight and repeated-failure termination.
5. Requiring every previously green cohort to repeat after any unrelated edit conflates local regression evidence with final source consistency. Use focused affected cohorts during diagnosis; the two final full campaigns must use identical frozen source and production settings.
6. Two fresh successes are a repeatability check, not a statistical reliability guarantee. Do not retry unchanged failures until lucky and report only the green pair. Retain every failed campaign and its spend.
7. Current canary trusts run count and `design_committed`. It does not independently prove corpus identity, exactly five successful stage events, completed report, or source stability. HEAD alone does not identify the inherited dirty source tree. Strengthen the evidence gate before treating 34/34 as deploy authority.
8. Do not auto-commit the inherited patch as a claimed reviewed unit. Preserve it and review affected contracts; commits must not scoop up unrelated production work.

## Implementation order and ownership

### Typed architecture contracts

Reconstruct CAN missing-port/signal-as-power failures and MCU recipe contracts. Complete only from uniquely matching typed peer evidence. Missing required ports remain blocking at architecture, with owner and required fields in correction feedback. Keep serialization recovery bounded and self-sufficient. No guessed bus nets or global collection-limit increase.

### Deterministic wiring ownership

Trace saved selectable-PD and ESP32-C3 states through the actual recipe/lowerer/planner. Prevent USB sink lowering from capturing a typed PD controller requirement. Carry valid recipe allocations exactly once; reject deterministic endpoint gaps before dispatch. Do not equate arbitrary model-owned pins with a verified selectable-PDO implementation.

### Budget and reporting

Preserve strict project/global/daily admission. Recover paid failed-run costs and terminal stage attribution through existing ledger/events rather than parse formatted errors as the authoritative source. Report total campaign spend per successful commit, including failures. Test the observed $0.0648 + $0.0384 boundary at $0.10/$0.15. Run same-code live comparison before adopting a larger production default.

### Independent acceptance

Canonical deploy must validate completed report, exact expected corpus and brief identities, one run each, all five committed stage states and successful stage events, no terminal errors/reused stage answers, finite configured cap, and unchanged source provenance. Diagnostic subsets are explicitly subsets, never production acceptance.

## Verification decision gates

1. Integrate independent changes, then run focused behavioral regressions and the stable Python suite once.
2. Run fresh affected live briefs/cohorts, no build/judge/resume, bounded production provider calls. Record every result. Fix the earliest reproducible mechanism, not the corpus slug.
3. Repeat affected cohorts to check stochastic behavior. Exercise all six original cohorts; retain their membership from the continuation plan.
4. Only a fresh complete pre-deploy 34/34 report on frozen source authorizes canonical deploy. The script must produce its own independent fresh 34/34 canary before restart; then verify HTTP 200 and worker readiness.
5. A provider outage, unavailable electrical specification, or spend ceiling is an explicit blocker, never permission to weaken gates, invent electrical meaning, or claim completion. Preserve diagnostics and finish all reachable offline work.

## Evidence log

- Starting report: `logs/self_eval/recovery_serializer_pass1_20260911T0400Z`: 1/6; architecture invalid_schema 1, wiring commit_rejected 2, unattributed budget exceptions 2.
- Inherited patch scope: 24 tracked files, 2451 additions / 468 deletions. Preserved as starting work, not claimed as this implementation.
- Deployment: not attempted.
- Reconstructed failures: selectable PD was reduced to a USB connector plus Rd resistors; C3 BOM had only AP1117 and capacitors, no MCU. Existing allocator expansion already carries valid GPIO/native-USB assignments. Fixed ownership selection rather than rewriting allocator plumbing.
- Historical failure-report CLI now attributes both budget exceptions to BOM and reports total paid spend **$0.210507**, failed-run spend **$0.201787**, cost per committed design **$0.210507**. Original artifacts remain untouched.
- The old plan's STM32 admission arithmetic is incorrect: $0.0648 + $0.0384 = **$0.1032**, not $0.1022. Both refused calls were in BOM, so neither was proven to be the final mandatory call.
- Existing production profile inspected without exposing secrets: `deepseek/deepseek-v4-flash-0731`, provider `open-inference/fp8`, project $0.10, daily $20, total $250, kill switch off. Defaults and `.env` remain unchanged pending live measurement.

### Live round 1 — completed, failed

- Directory: `logs/self_eval/evidence_loop_round1_20260911T024012Z`.
- Source fingerprint: `sha256:a74f9754008c061735adc4dfba4cc91aa27a48bbe1d966ba6bd3a0bf9a87dba8`.
- Result: **0/6**, paid spend **$0.045339**, no budget refusals. Do not retry unchanged or raise the cap to address these failures.
- USB breakout: normalization mistook the header for a receptacle, destroyed D+/D- polarity in generated port keys, and BOM-only fallback left fixed receptacle pin mapping to the model.
- CAN: architecture passed; deterministic two-pin input used a nonexistent `PhoenixContact_...` footprint rather than the installed `TerminalBlock_Phoenix_...` footprint.
- PD: controller-less BOM was accepted because a power-input requirement lacked an exact-part field and its ID did not contain the literal word controller. Selector sheet carried CC contracts but only a two-pin output header survived. No fixed-PDO substitute is accepted.
- STM32: a boot-button BOM unit saw whole-board context and returned MCU/decoupling arrays. Schema advertised 64 groups while the stream enforced 21; seven bounded calls did not repair it.
- C3/badge: respectively unavailable/reserved GPIO0–21 exposure, and missing explicit MCU supply/application bindings. These remain genuine source-stage failures, not reasons to prune contracts or guess supply pins.
- Offline evidence: stable suite **3236 passed, 15 skipped, 1 failed**; the old follower fixture shorted input/output/VCC and has been corrected to distinct AIN/AOUT contracts. Focused new fixture-construction failures were corrected and passed.

Round 2 changes target the above mechanisms: local BOM context and matching schema bounds, repeated-signature termination, typed controller coverage, polarity-safe connector normalization, precise hardware ownership, a verified-pin passive USB2 breakout lowerer with isolated shell, and explicit architecture supply/application guidance. The next run is fresh, not resumed.

### Live round 2 — completed, failed

- Directory: `logs/self_eval/evidence_loop_round2_20260911`.
- Frozen source: `sha256:21ed408d82776eaeaa2d7a7117fca5f681def81cd768098f66cad7979ee6d23e`; unchanged start/end, no resumed runs.
- Result: **1/6**, paid spend **$0.023514**, failed-run spend **$0.022877**. USB breakout committed all five stages ($0.000637); no budget refusals.
- STM32 advanced through BOM ($0.002073), then failed singleton wiring. BOOT/RESET had no typed requirements and acquired whole-board passive groups without buttons. This is still an ownership/source-stage defect, not permission to join arbitrary singleton nets.
- CAN passed BOM but deterministic endpoint proof rejected `MCU:CAN_TX`, `MCU:CAN_RX`, and `CAN TRANSCEIVER:CAN_TERM`. The MCU requirement had no application requests; the transceiver has no declared CAN_TERM port.
- Badge passed architecture with `mcu_present=false` and no requirements despite an explicitly named ATtiny1614 MCU sheet. It then repeated the nonexistent `Battery:CR2032` footprint. The Boolean/prose ownership bypass must be closed.
- PD controller coverage correctly rejected a connector/passive-only first candidate; later candidates invented TPS25730 identities and overflowed serialization. Its separately resolved USB 5V sink also conflicts with external CC contracts. Implement a datasheet-verified selectable controller/selector block, not a fixed-PDO substitute.
- C3 again exceeded the architecture collection bound. Its retry invented GPIO0–130 and routed USB ID/CC to the MCU. Prompt code also slices reference JSON at 24,000 characters, a latent invalid-JSON defect as the catalogue grows; measured C3 recipe metadata is near character 6,589, so truncation is **not** established as this C3 failure's cause. Preserve complete compact JSON without claiming it repairs the observed range hallucination.
- Stable suite: **3257 passed, 15 skipped, 1 failed**. The failure pinned incidental BOM retry counts in a telemetry-copy test; that obsolete test and a pure model-field-copy test were removed rather than repinned.

Round 3 keeps the cap unchanged. It removes the redundant full recipe catalogue from BOM prompts (architecture owns recipe choice), preserves complete architecture reference JSON, corrects the protected-part instruction, adds explicit passive CC breakout ownership, and addresses verified MCU/peripheral and primitive component contracts. Acceptance permits an earlier recovered stage failure but still requires exactly one fresh successful commit per stage, ordered successes, and no later terminal failure.

Round 3 electrical evidence:

- New `ch224k-pd-selectable@1` uses the WCH datasheet's CFG1 resistor mode: 6.8kΩ → 9V, 24kΩ → 12V, open → 20V; CFG2/CFG3 are NC. The verified SS13D07VG4 common pad 2 selects pads 1/3/4. It owns controller and selector, exposes CC1/CC2/VBUS/GND, and adds neither receptacle nor competing Rd resistors. Selection is made **unpowered**; open transitions request 20V and all load-side circuitry must tolerate that voltage. Source PDO support is still required. The fixed-9V recipe is unchanged.
- The actual MYOUNG BS-07-A1BJ001 CR2032 holder is distinct from welded-cell footprints; its installed footprint and manufacturer drawing agree on pad 1 positive / pad 2 negative.
- CAN review found another inherited electrical defect: SN65HVD230 pin 5 was grounded despite TI SLOS346O §7 specifying it as a VCC/2 reference **output**. It is now explicitly NC when unused, not shorted to ground.

### Live round 3 — completed, failed

- Directory: `logs/self_eval/evidence_loop_round3_20260911`.
- Frozen source: `sha256:94de8427ab0659aed5b046160cb91133e570e0705e26c07279fdcddbc863b615`; unchanged start/end, no resumed runs.
- Result: **1/6**, paid **$0.027384**, failed-run **$0.023637**. USB breakout again committed all five stages; this is not yet a green six-brief cohort.
- PD failed functional specification before reaching the new recipe: four/five blocks followed by repeated and Cartesian-product connections across every signal type. First duplicate edges occur before the 128-item cap; reject duplicates at their source rather than spending until the generic cap.
- STM32 architecture truncated, then drifted to RP2040/A4988 and lacked a valid 3.3V supply. CAN omitted named transceiver ownership. C3 emitted long collections and ultimately failed to bind its declared VCC_3V3 supply. These are not budget refusals.
- Badge exposed an earlier bypass: its verbatim goal names ATtiny1614, but canonical intent `named_parts` contained only CR2032. The architecture could then use generic MCU/LED/MOTOR labels with a false MCU flag and no requirements; BOM repeated nonexistent battery identities. Fix exact mixed-case token recognition before downstream ownership.
- Stable suite on this source: **3299 passed, 15 skipped**. Integration regressions fixed actual NC serialization and fixed op-amp port allocation, and removed two catalogue-snapshot tests that pinned incidental global registry contents.

Round 4 adds compound functional-flow uniqueness, consistent provider/stream collection bounds, exact named-part preservation, and narrowly scoped MCU reference choices. It also adds actual ATtiny1614 PTC capability from Microchip DS40002204A Table 5-1: physical pins 2/3/4/5/8/9, not arbitrary GPIO or UPDI. None of these changes authorizes deployment without fresh full-corpus evidence.

Physical review also corrected an inherited SOIC14 pin-map error: ATtiny1614 PA0/UPDI is pin 10, PA1/PA2/PA3 are 11/12/13, not the previous reversed mapping. The installed KiCad symbol and the primary datasheet agree. Existing fixed allocations remain reserved; touch-constrained requests are placed before broad GPIO consumers so feasible LED/touch combinations are not falsely rejected. ATtiny402/412 are not given unverified touch capability.

Round 4 integration: **527 focused tests passed**. A direct smoke execution on the preserved badge intent restores ATtiny1614; replaying the two captured PD streams aborts at duplicate connections 95 and 76 instead of the 128-item ceiling. Exact STM32/C3/ATtiny intents retain their one matching MCU recipe and all non-MCU alternatives, reducing catalogue JSON from 21,621 characters to 15,311/15,084/14,950 respectively. This scopes reference alternatives, not component selection.

Composite recipe normalization now preserves absent physical `exact_part` rather than storing a recipe ID as an MPN; repeated canonical normalization retains actual CH224K CC ownership. Implicit supply completion is narrowly 3.3V-only and rejects contradictory typed voltages; explicit supply bindings still have the pre-existing absence of a recipe-wide voltage-range contract. These changes do not constitute general electrical-safety proof.

### Live round 4 — completed; apparent pass is functionally incomplete

- Directory: `logs/self_eval/evidence_loop_round4_20260911`.
- Frozen source: `sha256:b0cf7007189297f72105e61bd7408312618b75ba1a860f8bd89004491665d7a0`; unchanged start/end, no resumed runs.
- Stage counters: **1/6**, paid **$0.014449**, failed-run **$0.013846**. Stable suite: **3343 passed, 15 skipped**.
- The sole counter-green USB breakout is **not acceptance evidence**. Both round3 and round4 contain only the USB receptacle plus two sink Rd resistors, with no header or exposed CC/SBU nets. The explicit BREAKOUT_HEADER functional block vanished after both blocks were merged onto one sheet. The legacy block-sheet gate checks only zero sheets; recipe membership then suppresses all remaining BOM work on that sheet.
- PD now reaches BOM, but an empty-requirements architecture passes with one generic 2-pin header as a declared controller placeholder. Wiring correctly refuses zero connections. This is missing functional ownership, not a need to relax wiring.
- STM32 now remains the right MCU but asks for external OSC_IN/OSC_OUT despite the recipe already owning its crystal; LED and button contracts remain unbound. CAN, C3, and badge still emit architectures without a valid owning MCU sheet/requirement. Badge's exact ATtiny1614 identity is now preserved and correctly blocks omission.

Round 5 closes the state contract rather than adding another prose-only rule:

1. Add explicit `CircuitRequirement.functional_blocks` membership against committed FunctionalSpec block names. A merged sheet is not proof that every function is implemented. Reject missing/unknown owners at architecture commit and use the same ownership relation for cross-sheet connection checks.
2. Require nonempty requirements and block membership in the provider's architecture schema; hide server-derived resolution fields from the provider. Preserve standalone primitive requirements without pretending they represent a complete FunctionalSpec.
3. Never suppress BOM work merely because a recipe touches a sheet; reject incomplete functional ownership and require real typed controller/header/switch implementations. Remove token-overlap claims that a recipe owns unrelated requirements.
4. Expose actual recipe-owned support roles/values so architecture can keep an already-owned crystal inside its MCU recipe instead of inventing external oscillator endpoints.
5. Correct the independently proved four-pad crystal mismatch in STM32/RP2040 recipes: existing Device:Crystal pins1/2 do not represent the selected 3225 four-pad package. Abracon's primary top-view diagram proves signal1/3 and ground2/4; Raspberry Pi's reference design confirms the RP2040 circuit. This is terminal ownership, not a claim of general oscillator qualification.

The full34/canonical34 deployment boundary remains unchanged. Counter-green designs missing committed functions do not qualify.

Round 5 integration proof: **466 focused tests passed**. Direct execution against the preserved round3 USB, round4 USB, and round4 PD states now rejects them at the architecture ownership gate and before BOM planning can silently return complete. A positive passive-breakout smoke preserves two distinct BOM units and produces the real receptacle plus an 8-pin header exposing VBUS/GND/D+/D−/CC1/CC2/SBU1/SBU2, with no sink Rd. This verifies the USB2 High-Speed contract; it does not claim USB3 SuperSpeed contacts on the 16-pin receptacle.

The architecture prompt and semantic classifier also stopped treating `category: power` alone as proof that a physical load connector or battery holder is merely a net. Bare ground/supply sheets still reject. Crystal regressions verify the active terminals remain separate from the grounded case and preserve the RP2040 series-damping path. Obsolete prompt-prose and source-document-count assertions were removed rather than re-pinned; historical live evidence remains unchanged. No existing changelog convention was found, so this requested plan and the stage contract prompt carry the update.

The first round5 full-suite run exposed 13 failures from the shared CLI fixture's omitted requirements (3356 tests passed). The fixture now declares its two existing functional owners; **all 27 CLI tests pass**, including commit, pinless mechanical, and invalid-identity paths. This changes test inputs to the new contract, not the gate. A fresh six-brief round5 and the complete stable rerun use the now-frozen source.

### Live round 5 — completed, failed

- Directory: `logs/self_eval/evidence_loop_round5_20260911`.
- Frozen source: `sha256:ae3de92d94d35dec199c9f566f1c3720a2325bfed46091affa1ebf80c040ce8c`; unchanged start/end, fresh directory, no resumed answers.
- Result: **0/6**, paid and failed-run spend **$0.021749**, 1578.5 seconds. Five architecture failures and one wiring failure; no budget refusal. The complete stable rerun passed **3369 tests, 15 skipped**.
- PD invented human-readable functional-block names instead of the committed identifiers. The new ownership gate correctly rejected them; the provider schema still advertised arbitrary strings instead of the known upstream set.
- USB preserved both functional requirements, but their ports contained direction words rather than actual net names. Its sheet-scoped BOM units retain requirement IDs; inspection corrected an initial inference from their names. The physical-identity matcher allowed a broad family match to override exact_part, so USB4085-GF-A became TYPE-C-31-M-12 without a substitution. A prose-based deterministic fallback also replaced the declared 10-pin header with an 8-pin header. Wiring then rejected missing and invented receptacle pins.
- STM32 functional specification parked twice for questions after receiving noninteractive defaults. The semantic-repair branch unconditionally parks blocking questions, unlike the primary response branch.
- CAN/C3/STM32/badge still show malformed or runaway architecture serialization. Badge's first range object added undeclared `start_sheet`, `end_sheet`, and `signal_type`, then repeated increasingly long `net_class_...` keys. The stream guard checks top-level array bounds but does not reject forbidden nested object properties.
- A direct lowerer reproduction independently returned a generic 1x02 header despite `pin_count: 20`, and despite an explicitly different header MPN. Unsupported constraints must decline deterministic compilation, not disappear.

Round 6 targets these observed boundaries: exact upstream ownership enums; universal endpoint-to-requirement net bindings; unambiguous catalogue definition names; exact-part precedence over broad family matching; no prose-only fallback overriding typed contracts; consistent primary/repair question disposition; and early rejection of schema-forbidden object keys with existing paid-partial-output accounting. These changes do not claim to cure every provider serialization failure. Keep the same profile, budgets, retry ceilings, fresh-campaign loop, and full34/canonical34 acceptance boundary.

Round 6 integration proof: the first focused run passed **608 tests** and exposed three integration-test failures. Two new fixtures/assertions were corrected; the obsolete test requiring prose-inferred hardware was removed rather than re-pinned. The affected rerun passed **132 tests**; Ruff passed on all 14 touched Python files. Direct execution against the recorded badge stream now stops at its first forbidden nested property after **1,019 characters**, before the runaway suffix. Actual USB state replay rejects the wrong receptacle MPN and unbound endpoints, and declines both typed prose-fallback candidates. Unsupported header pin-count/exact-part probes decline; a supported two-pin header still compiles. The complete stable suite and next fresh six-brief campaign use the now-frozen source.

### Live round 6 — completed, failed; controlled profile comparison

- Flash directory: `logs/self_eval/evidence_loop_round6_20260911`.
- Frozen source: `sha256:82bc7e9a2158a3e811fd9810a6d31250642fca9cedfe39d36ea61317da3966a8`; unchanged start/end. Complete suite: **3397 passed, 15 skipped**.
- Result: **0/6**, recorded spend **$0.019161**, 400.8 seconds. Four architecture failures, two BOM failures, no final budget refusal. STM32 no longer reparks for questions. Stream guards stop invalid collections/properties, but cannot make incorrect architectural choices valid.
- USB now commits explicit net bindings, but invented primitive parameters/family names remain model-owned; BOM puts a footprint identifier in the symbol field. PD uses model-owned controller/selector requirements and repeats an unavailable resistor despite stock diagnostics. STM32 omits LDO input/output bindings; CAN omits controller application bindings; C3 still supplies direction words as net names; badge omits its named MCU and then emits forbidden fields. The complete stage specification and catalogue are present in the actual prompt assembly; absent references are not established as the cause.
- Controlled alternative: `logs/self_eval/evidence_loop_round6_pro_20260911`, same exact frozen fingerprint, fresh upstream calls, configured pro/alibaba route, unchanged $0.10/$20/$250 caps. Result **0/6**, recorded spend **$0.040422**, 28.1 seconds. All six architecture calls were refused before dispatch: ceilings $0.0958–$0.0995 no longer fit after upstream spend. This establishes budget-policy incompatibility, not pro architecture quality. No production route or `.env` changed.

Round 7 adds an explicit `KICRAFT_STAGE_OUTPUT_LIMITS` JSON map which can only tighten existing normal and recovery ceilings. Defaults stay unchanged; the campaign manifest records overrides. Test the configured alternative with architecture8192/BOM4096/wiring8192 while preserving the $0.10 cap, all complete-slot checks, and fixed retry budgets. A truncated response still fails; this is not permission to abbreviate a design or claim reduced-corpus acceptance.

Round 7 policy verification: **261 focused tests passed** and Ruff passed. Replaying all six exact pro admission failures against temporary spend ledgers reproduces the original refusals. Tightening only the architecture output ceiling to8192 reduces their ceilings to **$0.05988628–$0.06366476**; prior spend plus those ceilings is **$0.06480252–$0.07280117**, below the unchanged $0.10 limit. Tests exercise actual client admission and charging for both normal and recovery limits. This proves request affordability, not successful architecture generation. No throwaway files remain; the next campaign and stable suite use frozen source.

### Live round 7 — admitted, but provider schema rejected

- Directory: `logs/self_eval/evidence_loop_round7_pro_20260911`; frozen source `sha256:29149d506d8b49b28896664d131ec68a5c7c4dd8b26a9c7da23944cb7199b977`, unchanged.
- Result: **0/6**, recorded spend **$0.058855**, 46.2 seconds. Every architecture request passed budget admission but returned HTTP400 before generation. Upstream intent/functional-spec requests succeeded. Complete stable suite: **3403 passed, 15 skipped**.
- A real architecture diagnostic on a temporary copy of the saved C3 upstream state captured Alibaba's error: array `uniqueItems` is unsupported. Two rejected diagnostic calls cost $0; no saved campaign artifacts were changed. The error was an SSE-framed string inside OpenRouter error metadata, not a nested ordinary JSON error.
- Round 8 removes only the provider-only `uniqueItems` declaration. Canonical `CircuitRequirement` still rejects duplicate or empty functional owners; committed-block enums and required nonempty membership remain. The obsolete schema-key assertion was removed, not inverted.
- Verification: **144 focused tests passed**, including canonical duplicate-owner rejection. The same real provider request is now accepted and generated output, costing **$0.030331026**, but its candidate failed `invalid_schema`. This proves the HTTP incompatibility is fixed, not architecture quality or cohort acceptance.

Next compare fresh six-brief campaigns on the same frozen source: production flash defaults and the explicit bounded pro policy. Their output limits differ and are recorded; this compares deployable route/policy combinations, not an isolated model-quality variable. The $0.10/$20/$250 monetary caps and full34/canonical34 boundary remain unchanged. No production configuration or services have changed.

### Live round 8 — failed; campaign attribution collision discovered

- Directories: `logs/self_eval/evidence_loop_round8_flash_20260911` and `logs/self_eval/evidence_loop_round8_pro_20260911`.
- Both used frozen source `sha256:cfa34b6de21238b5e259f2a0f3a34f678db52dcb40289c90942a51a95624608d`, unchanged start/end, fresh upstream calls. Both finished **0/6**. Flash took1032.8 seconds; pro took136.6 seconds. Stable suite: **3403 passed, 15 skipped**.
- Their recorded totals, **$0.147113** and **$0.258823**, are **not valid per-campaign spend totals and must not be added**. The first three briefs started in the same second and `evaluate_one` assigned identical `p<stem>-<integer timestamp>` spend IDs across directories. Project admission and exact-run spend queries therefore mixed the two profiles. Budget-comparison conclusions from this round are invalid.
- The collision is independently reproduced offline through the actual evaluation adapter and SpendGuard: the second of two otherwise independent $0.06 calls is refused against the first campaign's $0.10 project scope, and both records report that same $0.06. Fresh invocations need collision-resistant identities, and scoring must use exact run IDs rather than the shared directory-name prefix.
- Pro's STM32 candidate uses a legitimate optional `shield: SHIELD` port. USB normalization compares against an incomplete handwritten four-port set, calls shield CC/SBU exposure, and tells the model to replace its active sink. The verified power-sink and USB2-device recipes both explicitly support shield. Derive the accepted interface from those recipe definitions, preserving the real CC/SBU checks.
- Pro's badge emits a supply-only MCU despite declared LED/touch endpoints. The actual ATtiny1614 summary has `allocatable_gpios: []` and no capability inventory, while its allocator's eligible-pin metadata declares output/input/touch and other peripheral capabilities. Expose that allocation information and distinguish dynamic application bindings from fixed external ports; the allocator remains authoritative.
- Pro PD reaches BOM with the correct selectable recipe, but unnecessarily binds SBU1/SBU2 on the passive receptacle without any peer. Wiring correctly rejects those singleton nets. Clarify that unused optional connector terminals are omitted, whereas requested breakout signals require a real header peer.
- Flash CAN's second stream becomes irreparably invalid at character3034, concatenating objects without a comma, but emits42814 characters. Extend the existing incremental guard to recognize JSON grammar errors and ambiguous duplicate keys; retain paid-partial accounting, valid-prefix tolerance and fixed retry ceilings.

Round 9 implements those source boundaries in independent ownership slices, then runs integration proof before repeating fresh calls. No scope, monetary limit, acceptance gate, production configuration or service state is relaxed.

Round 8 charge recovery: querying the ledger by both exact run ID and model separates the colliding profiles. Nonoverlapping ledger-attributed totals are **flash $0.023330835**, **pro $0.256183181**, combined **$0.279514016**. These are ledger records, including any estimated partial charges, not an independent provider billing statement. Original campaign summaries remain untouched.

Round 9 integration proof:

- The first focused run passed **503 tests** and found two errors in the new USB test fixture (a dictionary was passed to an API requiring `RecipeSelection`). The fixture was corrected; both affected files now pass **60 tests**. Ruff passes on all14 touched Python files.
- Direct execution of the original frozen-clock financial reproduction now admits both independent $0.06 calls under separate $0.10 project scopes. Exact-run scoring tests exclude same-prefix siblings and history; no prefix fallback occurs when an exact identity was requested.
- The actual saved STM32 USB candidate now retains the active USB2 device circuit, its four shield-pad connections, and both CC pull-downs. The external CC/SBU rejection remains covered.
- The two actual CAN streams now stop at their first irreparable syntax errors, offsets946 and3034, rather than consuming1046 and42814 characters. Stream regressions cover both client paths, paid partials, chunk boundaries, duplicates and accepted wrappers.
- A direct allocator smoke uses the newly advertised ATtiny1614 capacities to allocate six distinct LED outputs and two distinct reviewed touch pins. The capability inventory reflects existing allocator rules; this change is not a new datasheet qualification of every advertised peripheral.
- Temporary spend ledgers were removed automatically; no throwaway scripts were added to the checkout. The preserved CAN prefix is a permanent regression fixture.

For the next bounded pro comparison, tighten architecture to6144 tokens (BOM4096/wiring8192 unchanged). All12 pro architecture calls in round8 produced at most3218 output tokens; the smaller ceiling leaves more admission room without increasing any monetary limit or retry count. Complete-slot and truncation rejection remain mandatory. Production flash defaults remain unchanged; the full suite and both fresh campaigns will use the next frozen source.

### Live round 9 — failed; independent attribution verified

- Directories: `logs/self_eval/evidence_loop_round9_flash_20260911` and `logs/self_eval/evidence_loop_round9_pro_20260911`.
- Both used frozen source `sha256:091354f2581f61809c194ee2ade4b95d9f14d0320fab005798ce91e97f37be1e`, unchanged start/end, and disjoint fresh run IDs. Both finished **0/6**. Flash cost **$0.021007** in383.6 seconds; bounded pro cost **$0.373858** in182.5 seconds. Stable suite: **3525 passed, 15 skipped**.
- Flash still failed on collection limits, architecture contracts, and BOM identities. Pro committed four architectures but three designs subsequently hit honest budget admission refusals; CAN exhausted BOM identity repair, C3 failed architecture contracts, and the badge falsely lacked a named CR2032 owner.
- The badge's actual holder requirement lowers successfully to the supported CR2032 holder. Named-part coverage does not recognize that compiled primitive identity. Its actual `input_touch0`/`input_touch1` bindings also allocate to ordinary input pins despite typed direct touch-pad peers; port-name inference alone is insufficient.
- STM32's complete MCU sheet is incorrectly treated as a detached support island because its function mentions the crystal. Fragmentation must identify a separate support-only domain, not any support keyword.
- CAN requests nonexistent `Connector:DB9_Female`. The installed real identifier is `Connector:DE9_Socket`, whose catalogue metadata explicitly says DB9, DSUB, female/socket. Candidate retrieval currently loses that evidence before ranking.
- Schema repair receives a shortened error string but loses structured evidence and the valid rejected draft. Preserve the complete diagnostic and the draft for schema correction; retain intentional draft removal for malformed JSON and clean-slate recovery.

### Round 10 — repair evidence and measured request ceilings

Implement compiled primitive ownership, typed direct-peer touch constraints, support-only fragmentation classification, and real catalogue candidate ranking. Keep canonical ownership, allocator eligibility, physical identity, net-boundary, truncation, and fixed retry gates intact.

The first centralized focused run passed **317 tests** and found **six failures**: four resolver/fixture cases and two candidate-ranking cases. These require correction before fresh paid verification; this is not acceptance evidence.

The next explicit pro policy is architecture6144/BOM1024/wiring2048. Round9's exact, isolated ledger records contain15 architecture calls (maximum3571 output tokens) and10 BOM calls (maximum223). Across48 recorded curated Pro wiring calls, the maximum is499 output tokens. BOM1024 and wiring2048 retain more than four times those observed maxima while reducing unused admission reservations. These observations do not guarantee every future response fits; an incomplete response still fails normally. Monetary caps remain $0.10/$20/$250, and production configuration remains unchanged.

After integration proof, run fresh six-brief verification twice, then a fresh full34, then the independent canonical deployment canary34. No failed or resumed campaign is substituted for either acceptance run.

Round10 integration proof:

- Corrected the four resolver/fixture regressions and the candidate-ranking regressions without removing their physical gates. The next focused run passed135 tests and exposed one remaining DB9 suggestion defect: `RJ9` was treated as a nine-contact structural match. Preserving represented family/contact evidence removes that telephone connector; the failing regression now passes. Actual DB9 suggestions are the installed `DE9_Socket` variants. Cold lookup took0.381 seconds, warm lookup0.052 seconds.
- An offline execution of the real stage driver, using the first two captured C3 candidate objects and stopping before provider dispatch, preserves complete `missing_recipe_port_contract` and `unavailable_recipe_gpio` diagnostics. A valid rejected draft is retained for schema repair; malformed JSON and clean-slate drafts are not replayed. The capture helper had concatenated responses across a clean-slate boundary; the smoke fixture separated complete JSON objects without modifying live evidence.
- The actual badge candidate now normalizes, expands six LED outputs, binds its touch electrodes to reviewed physical pins2/3, and compiles the real holder with pin1 VBAT/pin2 GND. All three captured STM32 MCU domains avoid false fragmentation; a separate decoupling-only domain still gets `repair_required`.
- Ruff passes all13 touched Python files. Two wording/prompt-copy-only tests and incidental wording assertions were removed; bounded-recovery tests remain. Disposable smoke scripts and their duplicated input were removed.
- The broader recursive command `.venv/bin/python -m pytest -q` reports **3769 passed, 15 skipped, 1 xfailed, 2 failed**. Both failures are real full-chain loadtest replays using an obsolete fixture without functional ownership. Prior stable-slice totals are not current full-suite acceptance. Migrate that fixture; do not skip those tests or loosen ownership.

### Live round 10 — one complete design; acceptance still failed

- Directory: `logs/self_eval/evidence_loop_round10_pro_20260911`; fresh frozen source `sha256:f69833049cc257bee45ea429f48a7477d8b392d12d2fbd7d199dfb977d2e5e9f`, unchanged start/end. Explicit policy: architecture6144/BOM1024/wiring2048, pro/alibaba, unchanged monetary caps.
- Result: **1/6**, **$0.384398**, 176.1 seconds. PD committed all five stages for **$0.048945**. The diagnostic acceptance command correctly rejected the campaign; no deployment occurred.
- USB breakout committed architecture/BOM but exhausted budget after aggregate wiring rejected missing SuperSpeed contacts and dangling header nets. It had selected USB2-only hardware and assigned TX/RX conductors to USB2 data pins.
- STM32's first candidate correctly binds USB shield to GND; the resolver wrongly classifies it as a signal in `power_nets`. Subsequent repairs fail a different MCU reset contract.
- C3's first candidate explicitly disables native USB and binds only MCU power ports. The resolver nevertheless invents native USB bindings by scanning remote nets, then rejects those invented bindings. A later candidate also exposes eager connector inference: unrelated global +5V/VBUS aliases are resolved before an explicit port can disambiguate them.
- CAN stops on duplicate net identity `CANH`, at item6 of a collection whose limit is128. Recovery incorrectly describes a collection-size failure and never identifies the duplicate. Two conservative partial charges consume **$0.05467408**; the next architecture call is correctly refused.
- Badge commits architecture but exhausts budget in BOM. Its accepted draft includes a deterministic holder plus a second holder in a unit that owns a regulator requirement. Investigate this ownership gap rather than treating budget refusal as the only problem.

### Round 11 — local topology and physical contracts

Scope optional recipe-port inference to actual ownership, preserve explicit bindings, and represent reviewed shield-to-ground semantics. Resolve connector aliases locally and only for missing ports. Distinguish duplicate-identity recovery from collection overflow, report the actual repeated values, and explain that one shared net has one record containing every participant; uniqueness remains enforced.

Extend the existing passive USB lowerer only with verified SuperSpeed-capable hardware and contact mapping. A separate direct reproduction also proves header lowering depends on JSON object order: sorting keys changes NET2 from pin2 to pin9 and NET10 from pin10 to pin2. Bind explicit `pinN` keys to physical pinN instead. Correct the replay fixture and investigate work-unit coverage before another fresh campaign. No cap, retry, physical gate, or acceptance boundary is relaxed.

### Round 11 — integrated contract repairs, live proof pending

- Recipe inference now uses owning-sheet endpoints and compatible directions, preserves explicit bindings, and cannot acquire an existing remote-only bus. USB shield-to-ground is allowed only on the two reviewed shield ports; signal-to-power and missing ownership still block.
- USB connector completion preserves explicit bindings and resolves signal nets locally. Only unambiguous global ground/supply fallbacks remain.
- Updated the five-stage USB-PD replay fixture to its actual fixed-9V CH224K resistor-strapping implementation and explicit output-terminal ownership. The two previously failing replay regressions passed the first focused R11 run.
- The first focused R11 run reported **339 passed, 2 failed**. One fixture still used obsolete arbitrary header keys; it now declares physical `pin1..pin9`. The frozen CAN case is still rejected, now for missing owned CANH/CANL ports rather than inferring those ports from globally mislabeled power nets. Separate explicit signal-to-power regressions remain.
- Direct lowering smoke exercised the actual R10 SuperSpeed requirement: **Amphenol 12401610E4#2A**, four TX/RX pairs, all **25 logical symbol/footprint contacts** matched, and D_P/D_N retained. Manufacturer drawing: https://cdn.amphenol-cs.com/media/wysiwyg/files/drawing/c12401610_c.pdf . Its drawing specifies 24 signal/power contacts plus shell stakes; logical shell pad S1 is repeated physically.
- Header smoke rejects the actual invalid `rows: 16` contract rather than silently changing it. An explicitly corrected single-row contract retains all **16 physical pin numbers** through sorted-JSON serialization. Architecture guidance now distinguishes row count from pin count; BOM guidance no longer suggests a USB2-only bundle for SuperSpeed.
- Direct streaming smoke stops duplicate CANH identity at 2 records against a 128-record ceiling and returns duplicate-specific feedback naming CANH and requiring one record with all participants. It no longer misreports this as a size overflow.
- Sibling primitive protection uses existing physical-feature evidence, including different-MPN battery holders. A second holder or renamed passive cannot satisfy the outstanding regulator function, and empty direct-battery-rail units remain invalid.
- A new architecture check rejects shared typed signal bindings only when explicit functional signal ownership proves a cross-sheet relationship with missing endpoints. It runs after legitimate peer completion and sheet folding. Same-sheet signals and unrelated or ambiguous same-named local nets remain untouched. **Filtering undeclared bindings into NC was rejected**: it would erase declared electrical intent.
- An exact immutable-pin singleton violation now requests architecture reconciliation immediately; it no longer spends a repair call on a sibling model unit. Model-owned singleton failures retain their existing bounded repair path.
- Integrated focused run: **474 passed, 1 failed**; the remaining failure was another obsolete arbitrary-key header fixture, now migrated to `pin1/pin2`. Follow-up boundary/work-unit regressions: **133 passed**. The new boundary guard initially misclassified implicit GND; a failing reproduction proved this, and the guard now shares the resolver's implicit-GND convention without reclassifying arbitrary signal names.
- Formatting/lint: **13 files checked, all Ruff checks passed**; changed follow-up files also pass. Complete recursive Python suite: **3819 passed, 15 skipped, 1 xfailed** (365.75s; output artifact 312). Temporary manufacturer PDF/raster files were removed after physical verification; the primary-source URL and verified mappings remain recorded.
- Next measurement: fresh six-brief Pro cohort at the unchanged $0.10 cap, plus a same-source STM32/badge comparison at $0.15 using process-local configuration only. This tests the inherited budget proposal rather than adopting it by assumption. Production `.env`, daily/lifetime ceilings, and deployment remain untouched.

### Live round 11 — budget admission helps; stage success exposed a false green

- Frozen source for both comparisons: `sha256:85e031ac9ebf160ad3bfd3a82a2aa18926310bb035ac315353e000c293377f8e`. Both output directories were fresh, start/end fingerprints matched, and run UUIDs were disjoint. Pro/alibaba and output ceilings remained architecture6144/BOM1024/wiring2048; only the second process's per-project budget changed.
- Base directory `logs/self_eval/evidence_loop_round11_pro_20260911`: **0/6**, **$0.310335**, 170.8 seconds, at **$0.10/project**. The acceptance command correctly failed.
- PD failed deterministic BOM sourcing because curated placeholder metadata produced MPN `N/A`. A separate direct reproduction found the deeper defect: normalized `N/A` becomes `na`, which matched “signal termination” and replaced a real resistor with a castellated pad. This is physical identity corruption, not merely an inconvenient sourcing error.
- USB breakout stopped at four functional connections against a 128-entry ceiling: the repeated identity was `(USB_C_RECEPTACLE, HEADER_BREAKOUT, other)`. Distinct physical signals must remain represented, but same-type signals between the same directed block pair belong in one functional graph edge.
- STM32 committed unsupported switch policies `pull_up`/`pull_down`; the deterministic lowerer accepts implementation policy `internal`/`external`, separately from active level. The unadvertised finite contract forced an unnecessary model BOM attempt and its unavailable switch choice.
- CAN reached wiring but could not admit the next bounded request. C3 alternated between a missing fixed UART/handshake contract, over-allocation of generic GPIOs, and an invented range containing reserved GPIO9. Badge could not admit another architecture request after an invalid UPDI binding.
- Comparison directory `logs/self_eval/evidence_loop_round11_pro_budget15_20260911`: **2/2 five-stage commits**, **$0.198797**, 134.2 seconds, at process-local **$0.15/project**. STM32 cost **$0.100421**; badge cost **$0.098376**. This demonstrates admission on these trajectories, not repeatability or a causal guarantee under stochastic provider output.
- **Badge is electrically invalid and does not count as an accepted design.** Its six Device:LED parts put pin2 (anode) on GND and pin1 (cathode) toward the MCU, with six 0-ohm links. Its typed passive design also declares rail3V, LED Vf3V, and target1mA: no resistor voltage headroom. Five committed stages were insufficient evidence; physical correctness must remain load-bearing.
- Production `.env`, global monetary ceilings, and services remain unchanged. No deployment occurred.

### Round 12 — reject invalid contracts before spending on immutable downstream work

- Curated MPN indexing excludes existing absent-metadata sentinels; curated metadata passes through the same optional-metadata normalizer as provider groups. The original “1k signal termination” smoke now retains `Device:R` and its 0603 footprint with no fabricated MPN. Real WJ126V terminal identity remains intact; generic castellated pads retain absent manufacturer identity.
- Registered lowerers now publish their finite parameter choices and required keys, and architecture validates that same contract before recipe resolution. Switch implementation policy and actuation polarity are distinct. Header row count is restricted to physically supported values rather than silently repaired.
- ESP32 module recipes expose their existing fixed UART programming pins without spending application GPIOs. Complete, direction-compatible typed bridge ownership can enable the reviewed two-NPN automatic-reset circuit. Missing, contradictory, ambiguous, remote-only, aliased, and explicitly disabled contracts remain blocking.
- C3 smoke through the real architecture normalizer rejects the original UART contradiction. An explicitly corrected counterfactual retains **all ten application GPIOs**, routes module pin31 TX/pin30 RX correctly, and produces physical-level `DTR RTS → EN BOOT` states **00→11, 11→11, 10→01, 01→10**. GPIO8 shares its application conductor with the required 10k boot pullup; EN retains its 10k/1uF RC.
- Primary hardware evidence: [Espressif boot-mode requirements](https://docs.espressif.com/projects/esptool/en/latest/esp32c3/advanced-topics/boot-mode-selection.html), [reference schematic and truth table](https://dl.espressif.com/dl/schematics/esp32_devkitc_v4-sch-20180607a.pdf), and [onsemi MMBT3904L datasheet](https://www.onsemi.com/pdf/datasheet/mmbt3904lt1-d.pdf). The selected SOT-23 transistor uses B1/E2/C3; cross-coupled emitters, not ground-connected emitters, establish the reference truth table.
- Functional-edge recovery now distinguishes graph identity from individual physical signals. Direct streaming smoke still rejects a duplicate at two entries against the unchanged128 ceiling; an explicitly aggregated edge preserves both SS_TX1 and SS_RX1.
- Passive LED parameters require finite values and positive voltage headroom; BOM rejects an owned LED-current-limiter unit without a proven positive resistor. The scalar lowerer now connects actual symbol A2 toward drive and K1 toward ground.
- New physical check **9.36** runs in CLI validation, wiring commit, and synthesis. It checks typed LED polarity and proven positive series resistance, including zero-ohm bypasses, without joining unrelated same-named local nets. It is a topology check, not a quantitative brightness or full analog-circuit certification.
- Running the real CLI against the untouched saved badge now exits **3** and identifies **all six reversed LEDs**. A separate architecture smoke rejects its zero-headroom contract. A polarity-only counterfactual still fails for six zero-ohm current paths; positive resistors pass the topology-only check, but do not make the original white-LED headroom valid.
- Initial integrated regressions reported **597 passed, 12 failed, 4 skipped**. Failures were newly added fixtures missing required `inter_sheet_nets`, incidental exact diagnostic-dictionary assertions, and a terminal test incorrectly expecting absence of its real MPN. These were corrected without relaxing production validation. The follow-up reported **240 passed, 2 failed, 4 skipped**; the remaining two incidental dictionary assertions were removed in favor of their actual rejection/evidence contract. Complete recursive verification is running before the next fresh campaign.
- Final integrated proof: **3859 passed, 15 skipped, 1 xfailed** in371.30 seconds (artifact357); all14 touched Python files pass Ruff formatting and lint checks. R12 smoke scripts ran in disposable subprocess memory and did not modify saved campaign artifacts. Next measurement is a fresh six-brief Pro campaign at process-local **$0.15/project**; this is still an experiment, not a production configuration change.

### Live round 12 — deterministic success, four remaining contract failures

- Directory: `logs/self_eval/evidence_loop_round12_pro_budget15_20260911`; fresh frozen source `sha256:d3481ec7652bd1eba6fa92424236bf813c5e98bfbd298aebfade5a07322aba94`, unchanged start/end. Pro/alibaba, architecture6144/BOM1024/wiring2048, process-local$0.15/project.
- Result: **2/6**, **$0.279932**, 167.6 seconds. PD cost **$0.027760** and STM32 **$0.027137**; both used **zero provider calls for BOM and wiring**. PD contains the actual switch-selectable CH224K recipe with SS13D07VG4, 6.8k/24k selections and a real WJ126V terminal. Its saved artifact passed real CLI validation before the next source edits.
- The acceptance command correctly rejected USB breakout, CAN, C3 and badge. There was no budget refusal in this round; the four failures are not evidence for another cap increase.
- USB breakout retained all declared signal endpoints and compiled all five units, but the lowerer invented a private shield net for the SuperSpeed symbol's single logical S1 pin. The singleton gate correctly rejected that orphan. This was compiler-created wiring, not a missing user signal.
- CAN's generic `power-input` requirement claims5V→3.3V but declares only an output and ground. Its model BOM then selects a globally protected registered AMS1117 implementation without typed architecture ownership. Badge again leaves an independent `power-distribution` requirement beside its already compiled holder; the model tries to satisfy it with another holder. The ownership fences are correct; these contracts must be repaired upstream.
- C3's three decoded architecture objects have different failures. Objects1/3 explicitly connect identically named device-relative UART TX/TX and RX/RX instead of crossing outputs to inputs. Object2 omits those bindings and receives correct peer completion, then fails on reserved GPIO9. All three guessed ranges contain GPIO9; object3 also omits GPIO7 from its header. The final summary alone does not explain the intermediate object.

### Round 13 — physical ownership and fixed-interface obligations

- Unbound USB shell pins now receive explicit no-connect ownership instead of a fabricated private signal net. Explicit shield bindings, including GND, remain bound exactly as declared. Updated regressions retain full symbol/footprint contact coverage.
- Direct smoke rebuilt wiring from the untouched R12 breakout BOM through all **five deterministic work units**, canonical normalization and the real CLI validator. It passed with **zero provider calls**, all **sixteen declared nets** intact and **J1.S1 explicitly NC**. The temporary copy was removed automatically; saved campaign artifacts were not changed.
- Shared typed-power diagnostics now reject the evidenced missing conversion contract and distribution-only physical obligation at architecture and persisted-BOM planning boundaries. They preserve registered-part protection and require a real component/port implementation or a proven existing physical owner; no automatic empty-unit acceptance or speculative functional fold was added.
- UART feedback names actual MCU and bridge bindings and the exact required crossing. A separate resolver defect treated `interfaces=['uart']` as a request for another allocatable UART even when fixed ports already satisfied it; only the proven fixed obligation is removed from allocation work, not from the declared architecture.
- Initial integrated regressions: **541 passed, 6 failed**. The six failures were a new local fixture missing required `inter_sheet_nets`; fixing that fixture leaves the targeted prompt/semantic suite at **83 passed**. Ruff passed all10 then-touched Python files. Final full-suite verification is pending the electrical correction below.
- **Additional physical defect found during C3 verification:** CH340C's existing recipe always wires V3 as a private bypass node, despite claiming3.3V operation. Official WCH datasheet v3D §5.1 requires V3 tied to VCC at3.3V and only a100nF bypass at5V. §6.2 specifies non-USB VOH≥VCC−0.6V, hence≥4.4V at5V. The frozen candidate powers CH340C at5V and connects UART directly to the3.3V ESP32-C3. This is not an acceptable positive counterfactual.
- Source: [manufacturer download page](https://www.wch-ic.com/downloads/CH340DS1_PDF.html), [original PDF download](https://www.wch-ic.com/download/file?id=79), §§5.1,6.2,6.3,7.6–7.7. Implement both reviewed CH340C supply modes and reject unproven direct power-domain crossings before the next live measurement. Do not fix this by relabeling rails or silently adding a translator.
- CH340C now has a finite numeric `supply_voltage` contract (3.3 or5.0) and a narrowly registered mode-dependent expander. Direct smoke verifies actual symbol pin4 V3, pin16 VCC and both bypass arrangements in both modes; UART pins2/3 remain unchanged. Architecture derives/checks the mode from authoritative bound-rail voltage rather than treating the recipe default as proof.
- Direct known CH340C/ESP32 UART connections require the reviewed common3.3V supply and ground. Different domains are not silently rebound; distinct translator-side conductors are not misclassified as a direct wire. The physical check follows identified devices and conductors, not merely role labels or separate-sheet layout.
- The actual C3 smoke first exposed another evidence-loss boundary: `resolve_architecture_recipes` retained both UART and supply failures, but `apply_architecture_recipe_resolution` raised only its first diagnostic. The exception now carries every blocking diagnostic, and architecture normalization serializes a structured batch when multiple contracts fail. Single-contract errors retain their natural single-record representation.
- Re-running the actual C3 counterfactual now exposes both original UART and supply failures. Repairing only UART/GPIO still fails on the unsafe power domain. Explicitly crossing UART bindings, fixing the GPIO range to0..8 plus10, declaring those GPIO owners, and changing the bridge's supply/endpoints to the common3.3V rail succeeds. Both declared UART interface annotations remain; all ten GPIOs and physical CH340C2→C3-30 / C3-31→CH340C3 connections are retained, with V3/VCC tied correctly.
- Actual frozen CAN and badge states now fail at both architecture normalization and persisted BOM planning. Explicit physical repairs succeed without losing any functional ownership: CAN selects the AMS1117 implementation plus a separate input connector; badge retains one holder and adds an explicitly declared100nF conditioning capacitor. These are counterfactual proofs, not edits to or accepted reruns of the saved campaigns.
- The next focused run reported **393 passed, 1 failed**: the new translator fixture omitted VBUS from its declared power nets. After correcting that fixture and the aggregate-error boundary, all **209 recipe/prompt regressions pass**. All12 touched Python files pass final Ruff checks. Full recursive verification is running. Temporary WCH PDF and browser tab were removed/released after verification.
- Next live measurement will cover **all34 curated briefs**, not only the six serialization examples. This broadens failure discovery while preserving the requirement for two fresh successful cohort proofs, a complete fresh predeploy acceptance, and the canonical independent deployment canary. Production configuration and services remain unchanged.
- Final recursive verification: **3882 passed, 15 skipped, 1 xfailed**,376.43s (`artifact://403`). Global ledger before the next campaign: today$2.423088/$20; lifetime$88.700635/$250; kill switch off. No monetary ceiling was raised.
- First R13 launch stopped before provider dispatch because the explicit Pro profile conflicted with production's legacy `KICRAFT_MODEL` setting. Relaunched with the full consistent process-local Pro model/provider/price configuration; no production configuration was edited. The actual startup banner confirms all34 briefs, Pro, no judge, parallel3 in `logs/self_eval/evidence_loop_round13_pro_budget15_all34_20260911`.

### Live round 13 — full diagnostic corpus, failed

- Directory: `logs/self_eval/evidence_loop_round13_pro_budget15_all34_20260911`; **7/34** five-stage commits, **$2.342762**,1082.5s. Fresh directory, no resume, source unchanged at `sha256:75fa464a48a872f2e97d2502f788c8a6c0994b831a1635e959f2e7e3ed9c1e12`. The independent acceptance command correctly exits1. No deploy or production configuration change.
- Commits: RC lowpass, R2R DAC, selectable PD, full USB-C breakout, STM32 minimum, RP2040 minimum and badge. These are diagnostic successes, not completed cohort repeatability or full acceptance.
- Source was kept frozen while inspecting completed failures. Read-only investigations cover interface bindings, exact part ownership, crossover aggregate repair and current-limiter identity. The table below records terminal signatures, not unproved shared root causes.

| Signature | Briefs | Owning boundary / next investigation |
|---|---|---|
| Nonexistent two-row header symbol | thermocouple-amp | Lowerer emits `Conn_02x04`; installed KiCad requires `_Odd_Even`. Real lookup also proves2x1 uniquely uses the unsuffixed name. |
| Connector cardinality silently changed | led-cc-driver | Three-terminal lowerer becomes curated two-terminal WJ126V; direct execution reproduces3→2 physical pins. Fix identity normalization, not the later unexpected-pin gate. |
| FPC physical obligation/sourcing rejection | fpc-breakout | Composite `fpc-header-breakout` is used for only the FPC sheet; its model unit is required to contain a pin header. A claimed LCSC identifier is also unverified. |
| Repeated missing implementation | rs485-terminal, nrf52-beacon, lora-node, gpio-expander, hex-env-sensor | Compare actual selected parts against typed identity/physical-feature gates before deciding omission versus false mismatch. |
| I2C controller port mismatch | esp32-s3-sensor, encoder-oled-panel, daq-8ch | Repeated missing `sda`; inspect actual request/requirement port keys and declared peer evidence. Encoder subsequently hits collection limits. |
| Nonexistent model-selected symbols | usb-a-power-splitter, highside-switch-10a | TPS2065DBV and Q_PMOS_GDS do not resolve; inspect whether available correction choices preserve the real electrical/pin contract. |
| Bounded BOM serialization failure | buck-3a, dual-rail-supply | Inspect individual1024-token replies before changing output policy; do not confuse legitimate size with runaway generation. |
| Aggregate repair / admission exhaustion | speaker-crossover, audio-jack-buffer, esp32-dual-motor | Trace exact aggregate offenders and preserved siblings; the last case refuses a$0.0625 call ceiling after$0.0920 spent. No cap increase. |
| Named part lacks recognized owner | relay-quad | Repeated ULN2003 ownership diagnostic; inspect actual architecture candidates. |
| Post-merge architecture sheet mismatch | proto-shield | Wiring merge leaves a requirement on removed `PROTOTYPING AREA`; trace authoritative sheet/requirement mutation together. |
| Incompatible switch obligation | can-node | Declares `switch-input` on CANH/GND with120-ohm parameter; do not waive the physical gate or accept a bus-to-ground button as selectable differential termination. |
| Deterministic component classified protected | servo-driver-16 | BOM lowerer unit r003 is rejected as model-authored; inspect source/provenance and sibling ownership. |
| Missing recipe physical controls | stepper-a4988, round-led-ring | A4988 EN/microstep endpoints and WS2812 terminal `data_out` need real ownership semantics, never deletion of requested signals. |
| UART endpoint-direction contradiction | rounded-c3-devboard | Inspect bindings separately from direction annotations; no5V-domain bypass or guessed rewiring. |
| Real orderability gate | star-ornament, snowman-ornament | ATTINY402-SSN reports retail stock47. Investigate proven equivalent ordering variants/availability without weakening§9.26 or silently substituting a different MCU. |

### Round 14 — checkpoint commit `dfc1582`, pushed

- Full suite **3951 passed, 15 skipped, 1 xfailed** (383.4s); all touched focused tests pass (427 across six files). Pushed to `origin/simplify/bom-wiring-pipeline`.
- Verified repairs landed and integrated: exact directional device/order-code + family membership authority (`kicraft/design/part_identity.py`); fixed-interface key aliases resolved without guessed nets; unique family-selected composites own real MPNs; programming-role UART reaches crossing diagnostics; typed passive connectors no longer inferred as ICs; equal-voltage rails require typed directional ownership; BOM semantic repair routes to owning units; real `_Odd_Even` two-row symbols; connector gender selects real `PinSocket` footprints; reserved `NC` contacts become no-connects; connector cardinality and explicit MPN survive curated normalization; standard-form-factor reconcile preserves prototyping hardware/ownership.

## Continuation plan

1. **Finish two remaining source repairs** (both evidenced, not yet implemented):
   - `fpc-breakout`: separate the `fpc-header-breakout` composite so an FPC-sheet model unit is not required to contain a pin header, and validate real FH12 footprints/sourcing without weakening identity gates (`_required_physical_feature` / `_validate_bom_unit_sourcing`).
   - `route_work_unit_ids` (`kicraft/server/stage_work_units.py`): case/separator-normalize evidence↔sheet matching so semantic repair regenerates only the owning unit, not every accepted sibling (speaker-crossover exhaustion).
2. **Verify offline**: focused regressions for the two fixes, then the full stable suite once.
3. **Fresh live loop**: run affected live briefs (`fpc-breakout`, `speaker-crossover`, and any remaining R13 signature), no build/judge/resume, bounded production provider calls. Fix the earliest reproducible mechanism; never retry unchanged or raise the cap.
4. **Cohort repeatability**: two consecutive fresh passes per cohort (serialization, identity, recipe/budget, boundary, ownership, regression sentinels), same frozen source.
5. **Complete pre-deploy campaign**: one fresh 34/34 five-stage commit on frozen source.
6. **Canonical deploy**: `./deploy/deploy-production.sh` — its independent fresh canary also reports 34/34; then verify HTTP 200 and `[build-worker] ready`.
7. **Record** campaign costs and deployment health evidence in this log.

Blocker policy unchanged: provider outage, unavailable electrical specification, or spend ceiling is an explicit blocker, never permission to weaken gates, invent electrical meaning, or claim completion. Production configuration and services remain unchanged.

## Final handoff (2026-09-11)

- Remaining source repairs landed and pushed: commit `c3416da` separates FPC connector ownership from the generic header feature (`fpc-header-breakout` now requires an FPC contact, `Connector_FFC-FPC` footprint) and scopes `route_work_unit_ids` case/separator-insensitively so semantic repair targets the owning unit. Full suite **3953 passed, 15 skipped, 1 xfailed**.
- A fresh R14 full-corpus campaign was launched at `logs/self_eval/evidence_loop_round14_pro_budget15_all34_20260911` and stopped at operator request (exit 143) before completion; it is diagnostic only and does not count toward acceptance.
- **Operator decision**: restart production services now and test live; do not run the deploy canary or any further verification this session.

## To continue (operator live test)

1. If live testing surfaces a concrete failure, classify its earliest mechanism against the R13 signature table and fix at the owning layer (architecture / resolver / BOM / wiring / budget), then re-run the affected brief fresh (no build/judge/resume).
2. Otherwise proceed through the six cohorts (two consecutive fresh passes each), then one fresh complete 34/34 pre-deploy campaign on frozen source.
3. Only a fresh 34/34 (plus the canonical `deploy/verify-design-canary.sh` all-34 during `deploy-production.sh`) authorizes claiming completion. Do not weaken gates, retry unchanged failures, or raise the spend cap to reach it.
