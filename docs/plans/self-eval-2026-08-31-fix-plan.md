# Self-eval 2026-08-31 — completion, routing, and electrical-quality plan

**Status:** revised after artifact/code review
**Source batch:** `/home/kicraft/.kicraft/self_eval/20260831T022210Z`  
**Code:** `ced1c68a5c0c4b7cf8b1fce3330b401c2d58b470` on `simplify/bom-wiring-pipeline`  
**Models:** design `deepseek/deepseek-v4-flash-0731`; judge `minimax/minimax-m3`  
**Rubric:** version 2

## Decision

Fix deterministic build failures first, then stage completion, then electrical quality. Do not tune the rubric, weaken correctness gates, or special-case corpus slugs.

The control-path work from the 2026-08-25 plan succeeded: all 34 briefs were graded, spend fell to $0.6455, and 23 designs reached build. The bottleneck has moved downstream:

1. eight of the 23 builds failed, six through two repeated router/placer defects;
2. eleven designs stopped before build, with six sharing the same dangling-net failure family;
3. only 15 boards were fab-ready, and the judge found recurring functional omissions even in several clean builds.

The fastest honest lift is therefore the deterministic six-board routing cohort, not more LLM retries. Stage and electrical work follows with frozen-state A/B gates because this batch used one stochastic sample per brief.

## Critical review and revised execution

The original ordering is directionally correct but is not an executable change
set. It combines three different classes of work—deterministic defects,
stochastic generation experiments, and product-quality policy—under one release
gate. Those classes need different evidence and must not be implemented as one
unreviewable patch.

Two premises also need correction:

1. `internal_net_count == 0` does **not** mean a leaf has no locally routable
   net. `route_local_subcircuit` already uses the authoritative condition:
   any local net with at least two pads is routable. The failed GPIO connector
   leaf had eight external signal anchors plus a shared external GND net, and
   its artifact contained six shorts and sixteen courtyard overlaps. Skipping
   routing would preserve an illegal placement and strand the shared GND
   connection; it is not a valid fix.
2. The servo failure and connector-bank failures are geometry defects, but the
   source batch does not prove the creating stage from the summary alone.
   They require frozen replay and object-level DRC evidence before changing
   composer or repair policy.

Revised execution:

1. **Implement now:** floor impossible grid pitches to actual member extents,
   pack genuine `ArraySpec` footprints before the overlap postcondition, and
   validate BOM footprints through the exact resolver and
   `pcbnew.FootprintLoad` seam used by synthesis. These causes are deterministic
   and directly established by frozen artifacts.
2. **Diagnose, then fix:** replay the remaining connector-bank and servo
   workspaces. Correct the earliest geometry stage that creates each overlap;
   do not treat `internal_net_count` as the routing predicate and do not exempt
   DRC.
3. **Run as controlled experiments:** topology-aware correction prompts,
   mechanical-intent propagation, provider retry classification, electrical
   blockers, and silk labeling remain separate changes. Each needs its focused
   fixture/frozen A/B gate before adoption. The single stochastic batch is
   evidence for hypotheses, not enough evidence to enable new hard blockers.

The implementation gate for this revision is therefore the focused deterministic
tests plus frozen no-LLM replays of the affected route/build workspaces. The
campaign-wide spend and score gates remain release criteria, not unit-level
acceptance criteria.

## Batch result

| Metric | Result |
|---|---:|
| briefs / graded | 34 / 34 |
| mean / median | 68.4 / 70.5 |
| grades | B:12, C:13, D:8, F:1 |
| design completed and entered build | 23 / 34 |
| fab-ready | 15 / 34 |
| fab-ready among completed designs | 15 / 23 (65.2%) |
| build failures after completed design | 8 / 23 |
| pre-build design failures | 11 / 34 |
| spend | $0.6455 |
| wall time | 10,941.1 s |

The weakest archetypes align with failure concentration:

- `hi_pin_hierarchical`: 0/3 fab-ready, mean 56.7;
- `connector_dense_io`: 0/4 fab-ready, mean 68.4;
- `power_thermal`: 1/4 fab-ready, mean 57.6.

Judge dimensions also show quality debt beyond pipeline completion:

| Dimension | Mean level | Runs at level 0–2 |
|---|---:|---:|
| board self-description | 1.12 | 26/34 |
| convergence efficiency | 1.97 | 18/34 |
| electrical soundness | 2.15 | 24/34 |
| computing-error cleanliness | 2.59 | 12/34 |
| part-selection quality | 2.65 | 10/34 |
| pipeline completion | 2.97 | 11/34 |

## Failure inventory

### A. Pre-build design failures — 11 runs

| Failure family | Count | Runs | Evidence |
|---|---:|---|---|
| dangling single-pin signal nets (§9.15) | 6 | `rs485-terminal`, `rp2040-min`, `esp32-s3-sensor`, `led-cc-driver`, `esp32-dual-motor`, `can-node` | 1–40 singleton nets remained after correction retries; examples include `DE`, `RE`, `SWCLK`, `SWD`, USB series-resistor far sides, buck support pins, and unused ESP32 GPIOs |
| spec-named part / symbol resolution (§9.33) | 1 | `buck-3a` | TPS5430 named in the brief but omitted without a substitution record; selected replacement symbol did not resolve |
| zero-pin mechanical symbol wired as pin 1 (§9.10) | 1 | `encoder-oled-panel` | four `Mechanical:MountingHole` parts were assigned nonexistent pin `1` despite a known empty pin inventory |
| two-terminal self-short (§9.17) | 1 | `daq-8ch` | `R8` and `R9` had both terminals on `GND` |
| provider failure | 1 | `stepper-a4988` | BOM call ended as `provider error`; no deterministic design defect was established |
| electrical graph used for a mechanical feature | 1 | `star-ornament` | functional spec emitted isolated `HANG_HOLE` as an electrical block; the stage failed before architecture, and the resulting design had no programming path |

The gates behaved correctly. They prevented invalid circuits from reaching synthesis. The fix must improve generation and correction; accepting singleton nets, fake pins, or self-shorted passives is prohibited.

### B. Build failures — 8 runs

| Failure family | Count | Runs | Evidence |
|---|---:|---|---|
| co-located genuine array grids | 4 | `r2r-dac`, `usb-pd-trigger`, `stm32-min`, `chamfered-badge` | every `ArraySpec` starts at `x0,y0 = px,py`; `_assert_grids_disjoint` rejected resistor/resistor, capacitor/resistor, or LED/resistor pairs in all three rounds |
| undersized and co-located connector arrays | 2 | `gpio-expander`, `audio-jack-buffer` | terminal/jack leaves requested pitches smaller than the real bodies, then placed multiple arrays at one origin; local routing correctly exposed the resulting illegal geometry |
| selected footprint present as a file but not loadable | 1 | `highside-switch-10a` | `pcbnew.FootprintLoad` returned `None` for `aod4184a:TO-252-2_L6.6-W6.1-P4.57-LS9.9-TL-CW`; BOM commit and synthesis used inconsistent validity checks |
| routed DRC failure | 1 | `servo-driver-16` | routed with zero shorts and zero unconnected nets, then failed for seven courtyard overlaps plus illegal routed geometry |

The first six failures share one repeated deterministic mechanism: impossible or
co-located array geometry. Extra search rounds cannot make overlapping bodies legal.

### C. Electrical and usability defects

The judge findings are not all equally severe. Separate non-functional circuits from production-hardening warnings.

**Non-functional or materially wrong topology:**

- `r2r-dac`: incomplete 8-bit R-2R ladder;
- `speaker-crossover`: connector pin groups shorted together and high-pass topology wrong;
- `usb-pd-trigger`: output connector absent from the shown wiring;
- `usb-c-full-breakout`: no CC role resistors, so attachment/VBUS behavior is undefined;
- `nrf52-beacon`: bare-SoC support network incomplete, including RF matching and required clock/power support;
- `dual-rail-supply`: no circuit actually generates the negative rail;
- `stepper-a4988`: missing current-sense, reset/sleep support, and supply decoupling;
- `proto-shield`: claimed prototyping power distribution is not implemented.

**Recurring production-hardening gaps:**

- exposed USB, CAN, RS-485, audio, screw-terminal, touch, and power interfaces lack protection;
- MCU/regulator/driver support networks omit decoupling, bulk capacitance, boot straps, or transient handling;
- part ratings and thermal/current headroom are not recorded consistently;
- connector and rail labels are weak: board-self-description scored 0–2 on 26/34 runs.

Do not make every protection omission a hard blocker. A missing rail, current limit, programming path, or mandatory family support circuit can block; optional protection and margin findings should remain warnings unless a deterministic contract proves damage or non-function.

## Scope

### In scope

- multi-array placement with disjoint deterministic origins;
- correct handling of leaves with no locally routable nets;
- BOM-time footprint loadability using the synthesis resolver;
- frozen replay of the servo DRC failure before changing geometry repair;
- wiring correction for singleton nets, fake pins, and self-shorted passives;
- functional-spec treatment of board-only mechanical requirements;
- evidence-backed family/topology checks and electrical-review remediation;
- connector/rail labeling after functional correctness is restored;
- focused replay and full-batch acceptance gates.

### Out of scope

- weakening §9.10, §9.15, §9.17, §9.33, ERC, route, or fab gates;
- auto-connecting an ambiguous singleton pin to a guessed net;
- accepting a missing footprint and hoping synthesis finds a fallback;
- allowing illegal copper or courtyard overlap because a board is otherwise routed;
- adding retries or raising token caps without a measured frozen-state improvement;
- rubric reweighting, judge-prompt tuning, or brief-specific code paths;
- requiring optional ESD/protection on every possible connector as a universal hard gate.

## Phase 1 — recover the six deterministic route failures

### 1.1 Pack multiple array grids instead of co-locating them

#### Change

1. Place each ring/grid at its existing local origin, compute its occupied
   member bbox, then translate every array after the first onto one deterministic
   horizontal shelf.
2. Keep at least `placement_clearance_mm` between array bboxes. Preserve member
   order, serpentine/ring orientation, and every legal explicit pitch/radius.
3. Floor an explicit grid pitch only when it is smaller than the member body
   extent plus `array_gap_mm`; an impossible pitch is not a valid constraint.
4. With one array, retain the existing per-member companion placement. With
   multiple arrays and no ownership metadata, use the existing perimeter fallback
   around the union; do not invent companion ownership from reference order.
5. Rebase the complete packed cluster once, after arrays and companions.
6. Keep `_assert_grids_disjoint` as a strict postcondition.
7. Do not drop genuine resistor, LED, capacitor, or mixed-function arrays.

#### Files

- `kicraft/autoplacer/brain/array_placement.py`
- `kicraft/design/synthesis/array_decaps.py` only if ownership metadata is required
- `tests/test_array_placement.py`
- `tests/test_array_decaps_synth.py`

#### Acceptance

- two grid arrays, grid+ring, unequal pitches, and companion-decaps produce
  disjoint occupied bboxes;
- legal single-array geometry is unchanged; an undersized pitch is expanded to
  legal spacing;
- member ordering and chain routing semantics do not change;
- frozen place/route replay succeeds for the array-failure cohort without
  suppressing a route or DRC verdict.

### 1.2 Route locally shared external nets; fix their array geometry

#### Change

1. Keep `route_local_subcircuit`'s existing authoritative predicate: a net is
   locally routable when it has at least two pads on the leaf, regardless of
   whether extraction classifies the net as internal or external.
2. Do not change `_stamp_trivial_leaf`; it already handles the true zero-routable
   case without invoking the router.
3. Treat the GPIO terminal leaves as undersized multi-array placement failures.
   Their 5.0 mm requested pitch is smaller than the 8.09 mm terminal body, and
   both arrays were co-located. Apply the legal-pitch floor and shelf packing.
4. Preserve every interface anchor and route the shared external GND net.
5. Keep placement, courtyard, short, pad-containment, and final DRC gates active.

#### Files

- `kicraft/autoplacer/brain/array_placement.py`
- `tests/test_array_placement.py`

#### Acceptance

- a true zero-routable-net leaf still stamps without router invocation;
- the frozen GPIO terminal leaf retains all anchors, has legal terminal spacing,
  and routes its shared GND net;
- frozen replay promotes all leaves and reaches parent composition for
  `gpio-expander` and `audio-jack-buffer`;
- no DRC rule is ignored and no interface net disappears.

### Phase 1 gate

Replay the six frozen workspaces through place/route only. Required result: six routed parents, no shorts, no unconnected nets, and honest fab verdicts. Do not run a paid design stage to validate deterministic placement code.

## Phase 2 — close synthesis and final-DRC gaps

### 2.1 Use the synthesis footprint resolver at BOM commit

#### Change

1. Replace existence-only footprint acceptance with a loadability check that uses the same library resolution and `pcbnew.FootprintLoad` path as `write_empty_pcb`.
2. Report library, footprint name, resolved directory, and exact failure to the BOM correction attempt.
3. Require the model to select a verified alternative or an explicit supported substitution. Never silently rewrite a package.
4. Remove duplicate `FootprintNotFoundError` semantics if the two current classes can share one resolver exception without creating an import cycle.

#### Files

- `kicraft/design/synthesis/footprint_library.py`
- `kicraft/design/synthesis/kicad_pcb_stub.py`
- BOM validation in `kicraft/design/cli_app.py`
- `tests/test_footprint_search.py`
- `tests/test_symbol_footprint_pin_gate.py`

#### Acceptance

- the exact missing `aod4184a` footprint fails during BOM commit, not after build starts;
- a valid bundled footprint and a valid stock footprint both load through the same check used by synthesis;
- frozen pre-BOM replay of `highside-switch-10a` either commits a loadable part or terminates honestly at BOM; it never reaches an rc=1 traceback.

### 2.2 Diagnose before repairing the servo board

#### Change

1. Replay the frozen synthesized `servo-driver-16` workspace with the real build tail and retain the seven courtyard pairs plus the illegal-geometry objects.
2. Classify each violation as leaf-internal, replicated-leaf, parent-compose, connector-edge, or routed-copper repair.
3. Fix the earliest stage that creates each violation. Do not add a final-verdict exception.
4. Add a regression fixture for the smallest geometry that reproduces each actual mechanism; avoid a 35-component golden-board test when a two-block geometry test suffices.

#### Likely files, selected only after replay evidence

- `kicraft/autoplacer/brain/subcircuit_composer.py`
- `kicraft/autoplacer/brain/geometry_repair.py`
- `kicraft/autoplacer/brain/leaf_geometry.py`
- focused composer/geometry tests

#### Acceptance

- frozen replay has zero shorts, zero unconnected nets, zero courtyard overlaps, no illegal routed geometry, and a fab-ready verdict;
- the repaired board still contains all 35 components and all 789 pre-fix traces are not assumed to survive—connectivity and DRC are authoritative.

#### Frozen replay result

The legal-pitch and multi-array changes remove six of the seven servo courtyard
violations. The remaining pair is `J16`/`J2`, both single-connector child
artifacts; it first appears during parent composition. The routed board remains
honestly rejected with `shorts=0`, `unconnected=0`, `courtyard=1`, and all 35
components present. This is a distinct locked/edge-constrained block-composition
mechanism, not an array-placement regression, and must not be hidden by the
array patch or a fab-verdict exception.

## Phase 3 — improve stage completion without weakening gates

### 3.1 Make wiring correction topology-aware

#### Change

1. For §9.15 feedback, include the offending endpoint's pin function, current sheet, declared inter-sheet nets, and any adjacent two-terminal part. Tell the correction pass which valid choices exist: complete the intended path, declare the cross-sheet net, or mark a truly unused pin `no_connect`.
2. For §9.17, provide the full two-terminal part assignment and require one atomic correction: preserve one source-side net, create/use the load-side net, and update the connected load endpoint in the same response.
3. For zero-pin mechanical symbols, omit them from the required wiring inventory and reject invented assignments at schema normalization with explicit `this part has no electrical pins` feedback.
4. Detect an unchanged normalized rejection signature. After two identical signatures, perform at most one clean-slate correction from the committed BOM plus the latest concrete feedback. Then terminate honestly; do not loop.
5. Keep the one-record-per-real-pin response contract and every current validation gate.

#### Files

- `kicraft/server/stage_prompts.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/server/stage_driver.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/design/synthesis/validation.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_stage_driver_prompt_examples.py`
- `tests/test_kicraft_validation.py`

#### Acceptance

- synthetic singleton, series-part, unused-GPIO, inter-sheet, self-short, and zero-pin-mechanical cases produce precise correction feedback;
- no test accepts a singleton signal, fake mechanical pin, or self-shorted passive;
- replay the eight frozen wiring states (`rs485-terminal`, `rp2040-min`, `esp32-s3-sensor`, `led-cc-driver`, `esp32-dual-motor`, `can-node`, `encoder-oled-panel`, `daq-8ch`) for at least three repeats;
- adoption gate: at least 20/24 valid commits, no accepted §9 defect, and no increase in mean attempts or cost versus control. If the gate misses, retain only feedback/telemetry improvements proven neutral.

### 3.2 Keep mechanical constraints out of the electrical graph

#### Change

1. State explicitly in the functional-spec prompt that mounting holes, hang holes, fiducials, logos, and outline features belong in intent constraints/form-factor data, not as functional electrical blocks.
2. On an isolated block rejection, distinguish a likely board-only mechanical feature from a disconnected electrical function and give the appropriate correction.
3. Preserve the strict disconnected-electrical-block gate. Do not exempt blocks by corpus name or silently connect them.
4. Ensure the constraint survives to architecture/BOM placement requirements so the physical feature is not lost when removed from the electrical graph.

#### Files

- `kicraft/server/stage_prompts.py`
- functional-spec commit validation in `kicraft/design/cli_app.py`
- intent/form-factor propagation code if board-feature constraints are currently dropped
- functional-spec and prompt example tests

#### Acceptance

- a hang hole is retained as a board requirement without becoming an electrical sheet;
- an isolated amplifier, power, or interface block still fails;
- frozen pre-functional-spec replay of `star-ornament` reaches BOM and includes a real first-flash path for the ATtiny402 before it can pass electrical review.

### 3.3 Separate provider incidents from design defects

#### Change

1. Preserve provider class/status/finish reason for terminal stage failures instead of flattening them to `provider error`.
2. Retry only documented transient classes within the existing budget; deterministic 4xx/schema failures are terminal.
3. Add a resumable single-run command or documented invocation that reuses the frozen pre-stage workspace and does not rerun completed stages.

#### Acceptance

- `stepper-a4988` evidence identifies the provider failure class;
- a frozen BOM resume either commits or reports a specific terminal provider condition;
- no retry budget or output cap increases.

## Phase 4 — raise electrical soundness with contracts, not rubric tuning

### 4.1 Turn proven non-functional families into deterministic or corroborated blockers

Start with defect classes demonstrated by this batch:

- switched regulator output existence, feedback/current-limit network, and mandatory support parts;
- dual-rail designs must contain a real negative-rail source;
- R-2R ladder cardinality/topology;
- USB-C role declaration through valid CC networks;
- MCU first-flash path and mandatory family support networks;
- stepper/motor-driver current-sense and enable/reset defaults;
- requested output/input connector must be present and connected;
- multi-pin connector symbols may not collapse signal and return conductors onto one net unless the symbol pin functions prove they are duplicate contacts.

#### Change

1. Encode checks deterministically only where pin functions, named intent, and BOM/netlist data prove the contract. Put these beside existing §9 family checks in `kicraft/design/synthesis/validation.py`.
2. For topology that cannot be proved deterministically, extend `electrical_review.py` categories and require existing multi-pass corroboration before a blocker can trigger.
3. Feed a corroborated blocker into one bounded BOM/wiring reconcile cycle with exact refs/nets and the suggested repair. Re-run deterministic validation and review after the correction.
4. Keep optional protection, preferred termination, margin, and documentation findings as warnings unless the design's stated voltage/current/environment makes them mandatory.

#### Files

- `kicraft/design/synthesis/validation.py`
- `kicraft/design/synthesis/electrical_review.py`
- review orchestration in the server/build pipeline
- `tests/test_electrical_review.py`
- `tests/test_electrical_review_gate.py`
- focused family-contract tests in `tests/test_kicraft_validation.py`

#### Acceptance

- fixtures for the eight non-functional examples above fail before fab export;
- corrected fixtures pass without slug-specific logic;
- optional ESD/reverse-polarity omissions remain visible warnings, not universal blockers;
- frozen digest review has no new false blocker on the 15 fab-ready boards; any changed severity is manually tied to a proved contract.

### 4.2 Make production-hardening findings actionable

#### Change

1. Normalize warning categories for exposed-interface protection, decoupling/bulk support, rating/thermal headroom, and grounding/isolation.
2. Persist refs, nets, source evidence, and a concrete suggested change for each warning.
3. Surface warnings in the final report and project UI. Do not automatically add parts where voltage, bandwidth, capacitance, or safety requirements are ambiguous.

#### Acceptance

- warnings cite actual refs/nets from the committed design;
- repeated findings aggregate by category in self-eval output;
- a user can distinguish `functional blocker` from `production-hardening warning` without reading judge prose.

## Phase 5 — improve board self-description after functionality

### Change

1. Audit the 15 fab-ready boards at actual rendered output, starting with the level-0/1 cohort: `thermocouple-amp`, `usb-c-full-breakout`, `fpc-breakout`, `proto-shield`, `round-led-ring`, `hex-env-sensor`, and `snowman-ornament`.
2. Ensure the deterministic legend contains project identity/revision and every externally actionable connector has concise signal, power-polarity, or channel labels.
3. Derive labels from committed nets and connector pin groups. Do not ask the model to invent labels after routing.
4. Run silk legality after label placement; move or omit a label rather than creating copper/courtyard/edge violations.

### Files

- `kicraft/autoplacer/hardware/silk_legend.py`
- `kicraft/autoplacer/hardware/silk_refdes.py`
- connector-label generation/placement code
- `tests/test_silk_legend_placer.py`
- `tests/test_silk_check.py`
- `tests/test_silk_geometry.py`

### Acceptance

- rendered boards identify the board and external connector functions without the schematic;
- no added silk DRC, pad overlap, or edge clipping;
- the frozen fab-ready cohort improves board-self-description without changing electrical or routing verdicts.

## Verification sequence

Use frozen artifacts before any paid full batch.

1. **Deterministic unit/behavior tests:** focused array, leaf-routing, footprint, stage-feedback, validation, electrical-review, and silk suites.
2. **Route replay cohort:** six Phase 1 workspaces; require six routed parents and honest fab verdicts.
3. **Synthesis/DRC cohort:** `highside-switch-10a` and `servo-driver-16` through the real build tail.
4. **Frozen stage cohort:** eight wiring failures ×3, plus `star-ornament`, `buck-3a`, and `stepper-a4988`; compare commit rate, attempts, cost, and terminal gate family.
5. **Frozen electrical digest cohort:** all 34 reports; inspect every new blocker and warning category before enabling remediation.
6. **Full self-eval:** same corpus, model, judge, rubric, `parallel=3`, and `build_slots=1`.

## Full-batch release gates

- 34/34 valid judge verdicts;
- at least 30/34 designs reach build;
- zero rc=1 synthesis tracebacks from a footprint accepted at BOM;
- zero multi-array co-location failures;
- zero rejected zero-internal-net leaves caused by unnecessary local routing;
- at least 27/34 fab-ready, with every non-fab result carrying a specific evidence-backed reason;
- zero accepted §9.10, §9.15, §9.17, or §9.33 defect;
- no known non-functional topology exported as fab-ready;
- no increase in false electrical blockers on the frozen 15-board fab-ready cohort;
- total spend no more than $0.90 unless a measured completion gain justifies and documents the delta;
- compare single-run score movement descriptively only; require at least three repeats before claiming model-quality regression or improvement.

## Implementation order

1. Multi-array packing and zero-local-net leaf handling.
2. Footprint resolver parity and evidence-driven servo repair.
3. Wiring/mechanical/provider completion work with frozen A/Bs.
4. Electrical contracts and bounded review remediation.
5. Board labeling.
6. Full self-eval only after all focused gates pass.

Keep the source batch immutable. Use `summary.json`, per-run `eval/report.json`, `events.jsonl`, `.kicraft/build.log`, and `.experiments` artifacts as evidence; do not infer causes from the grade table alone.
