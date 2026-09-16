# Self-evaluation 2026-09-15 — delivery correctness and completion recovery

**Status:** revised 2026-09-16 after critical review; plan only, implementation paused at the user's request. No pipeline fixes applied.  
**Campaign:** `logs/self_eval/20260915T132650Z`  
**Revision:** `b4b8be5a87945801972d35f5153d56322134f3dd`  
**Models:** designer `openai/gpt-5.6-luna` through OpenAI; judge `minimax/minimax-m3`.  
**Rubric:** v2, `b2f9009995cc44d0c3c6fdc933fcc632d692e3536c8db367fb22b430a153ad2b`.

## Decision

Make correct designs **constructible and independently checkable**, not merely easier to reject. Execute five milestones: (1) explicit, feasible acceptance contracts for all 34 briefs; (2) reviewed reference inputs proving deterministic realizability; (3) shared contract closure followed by live generation; (4) routing as each reference design becomes available; (5) fresh, repeated, artifact-checked fulfillment. Preserve the existing intent-shaped compiler, recipes, lowerers, and work-unit ownership rather than introducing a parallel design framework.

The headline is **11/34 fabrication-gate passes, not 11 verified working designs**. At least six of those eleven have artifact-confirmed functional or requested-part defects: RC/BNC, crossover, USB-PD trigger, constant-current LED driver, dual-rail supply, and audio-jack buffer. The remaining five are not certified electrically correct by this investigation.

The starting point is **29 known non-fulfillments and five incompletely verified candidates**: 21 pre-build failures, two routing failures, and six accepted-wrong circuits. Turning an unsafe acceptance into an honest refusal is safety progress, but not completion progress. A milestone closes only on its positive implementation evidence as well as its negative counterexamples.

Do not widen retries, raise spend ceilings, weaken ERC/DRC, merge nets heuristically, relabel placeholders as components, or hand-fix these generated boards. The campaign is the frozen witness for generalizable pipeline changes.

## 1. Deployment verification and experiment identity

At the 2026-09-15 campaign launch, the runtime source was already deployed; no restart was necessary. The process and health observations below are historical campaign evidence, not a new deployment check:

- Clean checkout before launch. HEAD's final commit changed two plan documents only.
- Latest tracked runtime-source change: `architecture_intent.py`, 04:04:30 UTC. `.env` predates both service starts.
- Web PID **656388**, started 04:12:50 UTC; worker PID **656425**, started 04:12:52 UTC. Both run from `/home/kicraft/KiCraft`; the venv installation is editable and points at this checkout.
- Worker log ends with `[build-worker] ready (max 2 concurrent build(s))`.
- Local HTTP and public `https://kicraft.io/` returned **200**; public TLS verification passed, IP `5.78.233.146`. Chromium rendered the actual public landing page. The same PIDs and HTTP/TLS health were checked again after the batch.
- This establishes deployment/runtime availability, not success of every design. The headless evaluation below exercises the design/build machinery, not an authenticated browser project-creation flow.

The batch ran **13:27:29–15:03:09 UTC**, **5,740 s / 95 min 40 s**. Canonical invocation:

```bash
.venv/bin/python -m kicraft.eval.self_eval --out /home/kicraft/KiCraft/logs/self_eval/20260915T132650Z
```

All 34 default benchmark briefs, one sample each, no resume/reuse, full events, judge enabled, three concurrent briefs, one host-selected batch build slot. Production caps were unchanged: $0.60/project, $20/day, $250 total. No additional provider-backed investigative replays were run.

A first launcher preflight inherited the supervisor's obsolete `flash` profile and exited before any brief/provider call. The actual launch removed inherited KiCraft/provider overrides and loaded production `.env`; the manifest confirms **Luna/live**, not replay/mock. The initial preflight line remains in `run.log` for provenance.

Start/end source fingerprints match:
`sha256:27ff25dc92ebde690dc70f478ea0a9c308e2fb4c12734340863ee77718afcad4`.
The process exited **0**; `finished_at`, the 34 unique corpus identities, and `source_unchanged=true` were checked. Exit 0 means the harness completed, not that its designs passed.

## 2. Scorecard

| Metric | Result |
|---|---:|
| Selected / evaluated / graded | 34 / 34 / 34 |
| All five design stages committed | 13/34 (38.2%) |
| Stopped before build | 21/34 |
| Fabrication-gate passes | 11/34 (32.4%); 11/13 builds (84.6%) |
| Route failures | 2, both rc6 |
| Mean / median grade score | 57.6 / 55.0 |
| Grade distribution | A:0, B:6, C:9, D:19, F:0 |
| Scores below 60 | 19 |
| Harness exceptions (`n_errored`) | 0 |
| Stage subprocess crashes | 1: prototype shield wiring commit |
| Recorded semantic-clean / design fab-safe flags | 6 / 12; not functional certification |
| Shaped outlines | 2/2 built shapes passed; 4/6 briefs never built |
| Design / judge spend | $0.608656 / $0.050065 |
| Total recorded campaign spend | **$0.658721** |
| Spend on unsuccessful designs, including their judges | $0.392139 (59.5% of total) |

Terminal stages: **architecture 12, BOM 5, wiring 4**. Mechanisms: `contract_rejected` 12, `unit_repair_exhausted` 4, `commit_rejected` 3, `architecture_reconciliation_required` 1, `commit_process_failed` 1. No terminal provider, transport, or budget refusal was recorded. Architecture was the largest recorded design-spend category ($0.324625); additional retry budget is not the first remedy.

### Grade interpretation

- Raw observer gates: `silent_substitution` **4** (#1, #4, #16, #28); `unprogrammable_mcu` **7** (#9, #13, #23, #30, #31, #33, #34). All seven MCU-gated runs stopped before build. They are not seven delivered unprogrammable boards. #16 likewise has no manufactured substitution to inspect.
- No `erc_errors` or `synthesis_broken` rubric gates fired. This does not mean no crash: #21's stage-commit traceback was returned as a design failure and was not reflected as a synthesis crash in grading.
- #21 received **B/76** despite never building; #10 received **B/83.5** despite rc6. Rubric verdicts are not delivery verdicts.
- Judge electrical-soundness mean is **1.38/4**, part-selection **1.59/4**, board self-description **0.79/4**. These identify review areas, but truncated evidence prevents treating every rationale as a defect.
- The earlier design-only canary in `architecture-constructive-slot-2026-09-14.md` §10.11 also committed 13/34, on `154fa79`, before the later supply fix. This is context, **not a controlled regression comparison**: source and mode differ, and each brief has only one stochastic sample. The August full-batch grades also used a different model.

## 3. Ranked pipeline gaps

### GAP 1 — fabrication success can conceal missing or electrically wrong implementation [P0, gate/contract]

**Evidence:** six distinct rc0 designs, all on this revision. Verified from committed state, the delivered USB-PD PCB, existing deterministic predicates, and the component datasheet—not just judge text.

| Run | Confirmed defect | Earliest responsible boundary |
|---|---|---|
| #1 `rc-lowpass-bnc` | J1/J2 are 1x02 pin sockets, not BNCs; R1 is a fixed 10 kΩ resistor, not a trim pot. Material substitution questions remain unanswered and nonblocking. | Preserve requested component class from intent into architecture requirements and physical BOM realization. |
| #4 `speaker-crossover` | Pin sockets replace binding posts. Recorded 8 Ω / 2.5 kHz low-pass uses **0.51 µH**, while L = R/(2πf) requires **509 µH ≈ 0.51 mH**. Shipped value implies ≈2.50 MHz. | Architecture/BOM numeric design contract; unit-aware component calculation. |
| #5 `usb-pd-trigger` | J2 USB power pads are on `VBUS`; output J1.1 and controller-feed R3/R4 are on separate `VBUS_NEGOTIATED`. No component connects the input rail to the feed/output. Confirmed with `pcbnew.LoadBoard` on the promoted PCB. | Architecture power connectivity across connector and PD recipe, before expansion. |
| #17 `led-cc-driver` | TPS54331 U1.1 **BOOT** and U1.8 **PH** share `SW_NODE`; C3 is SW_NODE-to-GND instead of between BOOT and PH. | Device-specific BOM/wiring contract. |
| #18 `dual-rail-supply` | U2 is `Connector_Generic:Conn_01x06`, explicitly recorded as an unresolved converter placeholder, yet the build is rc0 with a fab ZIP. Output-filter capacitors are on different nets from the converter outputs without a connecting series element. | BOM requirement implementation and power-path verification. |
| #28 `audio-jack-buffer` | The four requested 3.5 mm jacks became two 1x08 pin sockets; substitution ledger is empty. `intent.named_parts=[]` despite the connector constraint. | Typed interface capture/realization, not optional named-part prose alone. |

The [TPS54331 datasheet](https://www.ti.com/lit/ds/symlink/tps54331.pdf), Table 5-1, requires a 0.1 µF capacitor between BOOT and PH and states that inadequate bootstrap voltage forces the high-side MOSFET off. This is a functional defect, not a cosmetic DRC finding.

**Source / missed detection:**

- `stage_work_units.py::_validate_bom_unit` / `_requirement_needs_controller` checks only MCU, certain PD families, or identities containing `controller`. A requirement with role `regulator`, family `dual-output-dc-dc-converter`, and no exact MPN does not require real converter hardware. The current predicate returns false on #18's frozen requirement.
- `architecture_intent.py::derive_architecture`, recipe port bindings, and BOM work-unit realization preserve legal local nets without proving the intended energy path. A connected output connector is not evidence of a powered output.
- `lowering.py::_connector` and `_rc_filter` correctly implement the **wrong selected requirement** (header, fixed RC). The fix starts at contract selection; do not teach a header lowerer to guess that a label means BNC.
- `synthesis/validation.py::check_named_part_substitutions` (§9.23) and `cli_app.py::_cmd_stage_commit` intentionally create nonblocking material questions. §9.33 exact-part accountability cannot enforce a component class that was never captured as an exact/typed obligation.
- ERC/DRC verify pin types and geometric/net consistency, not converter existence, transfer-function math, bootstrap topology, or requested connector class.

**Fix:** strengthen the existing implementation contract: every functional requirement must own verified hardware or remain explicitly unresolved; an acknowledged placeholder may support inspection, never successful delivery. Carry physical connector/adjustability obligations into requirements. Derive quantitative values from typed units and known topology. Pair each narrow device/power-path invariant with a reviewed implementation that satisfies it, and supply the model the complete failed obligation. Rejection alone does not close the recovery work.

**Guard / verification:** frozen #18 must fail BOM implementation before build; #5 must reject the unpowered output and pass only with one real input-to-controller/output path; #17 must reject BOOT=PH and accept a bootstrap capacitor spanning distinct pins; #4 must calculate the correct inductance order of magnitude; #1/#28 must contain the requested physical parts or an explicitly accepted substitution that does not pretend to satisfy the original ask. Verify resulting pins, nets, parts, and delivered files, not merely stage return codes. No post-route rewiring or brief-slug exceptions.

**Prior art:** known named-part advisory weakness (`self-eval-2026-06-27-fixes.md`, E1) and September typed-accountability work; fresh residuals in the current compiler/units. The supply split, bootstrap fault, converter placeholder, and unit-scale error are new observations here. Their artifacts are confirmed; no proposed fix or N-of-3 provider improvement has been verified.

### GAP 2 — declared intent, recipe interfaces, and BOM ownership disagree [P1, completion]

**Evidence:** all **21 pre-build failures**. Twelve architecture refusals; five BOM failures; four wiring failures. This is the dominant completion bottleneck, but several refusals correctly prevent unsafe or unsupported designs. The grouped failure inventory below distinguishes coverage from model mistakes.

**Sources:** `architecture_intent.py::_catalog`, `_supply_port`, `derive_architecture`; `recipes/resolver.py`; `stage_work_units.py::plan_stage_work_units`, `_group_has_physical_feature`, `_group_implements_controller`, `_validate_bom_unit`; `part_identity.py`; `stage_runtime.py::_drive_work_unit_stage`; `form_factors/reconcile.py::reconcile_standard_form_factor`.

**Measured contract mismatches:**

- `_catalog` retains ground/supply-looking lowerer keys but ignores the model's declared interface and other functional keys. Status-LED `drive/gnd` can acquire an incompatible `vdd` or fail supply binding entirely. #7, #31, #33 show this refusal family; #12 loses deterministic LED wiring and then repeats a real singleton mistake.
- A single implicit global `GND` conflicts with #8's isolated logic/field domains. Prefix/suffix supply-name heuristics are also insufficient for an isolator or multi-supply driver. This is a representation/design-contract problem, not permission to merge grounds.
- #9 repeats `RESET_BUTTON` onto an already bound `NRST`; #7/#23/#25/#27 have other conflicting signal/rail port assignments. Preserve the one-port/one-net refusal where the candidate is contradictory. Some duplicate logical-edge declarations may be coalescible, but that requires explicit equivalence—not guessing from similar names.
- #11's 24-pin FPC exceeds the existing 10–19-pin composite lowerer range. Its parts-tool-recommended FPC bundle is rejected by `_group_has_physical_feature`, which accepts only stock connector library prefixes. The current predicate returns false on the actual bundle. Its first sourcing failure also names an LCSC identity absent from the offline catalog.
- #13: the supplied `NRF52840-QIAA-R` bundle fails the reviewed `nrf52840` identity relation, which accepts only `NRF52840-QIAA-R7`. #14: the STM32L0 family accepts `STM32L031K6T6` but rejects the candidate `STM32L072CZU6`; feedback does not name the accepted set. Both results were reproduced with the current identity function. Package/carrier equivalence must be vendor-reviewed, not inferred by prefix.
- #21's generic shield architecture has no explicitly owned stacking interface. Reconciliation raises `ValueError("standard form factor has no explicitly owned stacking interface")`, escaping stage-commit as a traceback. Calling the real reconciler on an in-memory copy of its frozen state reproduces it without changing the run.

**Fix:** finish the existing interface/ownership contract across architecture compilation, part discovery, unit planning, and validation. Persist the relevant interface claims rather than retaining only their requirement identities; complete the lowerer vocabulary and report unsupported configurations before silently demoting them to model-owned units. Use reviewed concrete parts, pin inventories, and multi-domain supply declarations. Preserve existing bounds and safety refusals. Split the work into the independently verifiable milestones in §5 rather than one large retry/prompt rewrite.

**Guard / verification:** exact saved lowerer, identity, and FPC counterexamples must change at the owning boundary; negative package/pin/domain cases must remain rejected. Reconcile the shield's true stacking ownership upstream and report unsupported ownership structurally, with no partial state mutation or unhandled exception. Then run each affected stage N-of-3 against copied frozen state, followed by fresh full-design cohorts to prove the failure did not merely move downstream.

**Prior art:** `architecture-constructive-slot-2026-09-14.md` §10.11 explicitly defers lowerer `declared_ports`; the September 9/11 recipe/unit plans already list FPC, identity, and singleton-ownership gaps. These are live residuals, not evidence that deleting the old architecture bookkeeping should be reversed. Secondary research is saved with exact per-run evidence; speculative net-coalescing/name-based ground proposals are **not** approved by this plan.

### GAP 3 — evaluation is not yet a reliable delivery-quality gate [P1, measurement]

**Evidence / source:**

1. `eval/run_web.py::build_run_digest` slices pretty-printed state at **16,000 characters**, without an explicit omission manifest. In #2, the real 100 nF C1 is connected between U1's +5 V and GND, but neither C1 nor `100nF` appears in the rebuilt digest. The judge claims missing decoupling. #17/#29 show further absence-based rationales that cannot be accepted without complete BOM evidence.
2. The web path calls `run_post_wiring_review` and authors silkscreen after wiring. `eval/self_eval.py::run_design/evaluate_one` runs five stages then build/judge directly. This batch therefore omits the web electrical-review/rewire lifecycle and silk authoring; its empty review/silk state is not proof those live-web features failed.
3. `_render_md` flags only exceptions, observer gates, or scores <60. It omits some unbuilt and rc6 designs from “Needs attention”, including B/76 #21 and B/83.5 #10. `_stage_failure_attribution` records empty `failure_codes` for these architecture refusals even though diagnostic events contain actionable codes.
4. All seven `unprogrammable_mcu` observer flags are on designs that never built. Preserve the raw rubric evidence but do not present them as shipped-board findings. The shield's commit crash is not included in `n_errored=0` or the synthesis-crash gate.

**Fix:** use bounded structured judge and electrical-review digests with guaranteed brief/requirements, complete relevant pin/part/net facts, terminal status, and explicit missing/omitted sections. Neither model may infer absence from omitted evidence. Share the existing post-wiring lifecycle between web and batch rather than implementing a second review path, but do not mistake advisory review parity for a fulfillment gate. Keep **generation**, **fabrication**, **software-verified fulfillment**, and **physical validation** distinct; ensure every unsuccessful full-build run appears in attention lists and preserve nested stage diagnostics/crash attribution.

**Guard / verification:** rebuild #2's digest and prove C1 plus its supply connections are visible; use #21 and #10 to verify honest terminal verdicts and attention membership; on no-BOM fixtures, suppress unsupported delivered-MCU claims. Test a paid one-brief parity smoke before the next full campaign and account separately for review, silk, designer, and judge spend. Preserve this campaign and its raw rubric scores; do not silently regrade history.

**Prior art:** `run_web.py` already has dedicated silk, regulator-math, substitution-ledger, and programming-path lines to repair earlier truncation failures. This is the remaining general evidence-loss class, not a request for a larger blind text slice or new scoring weights.


## 4. Remaining routing, sourcing, and quality work

### Routing: two honest rc6 failures, not verified regressions

| Run | Frozen evidence | Next source investigation |
|---|---|---|
| #6 `usb-c-full-breakout` | Both leaves accepted. Parent round 1 has zero pre-route stamp violations but post-route acceptance fails with 2 unconnected nets (`TX2N`, `TX2P`), 551 clearance and 16 annular-width findings; clearance references cluster on J2. Later rounds do not produce an accepted parent. Promoted provenance is **partial**, not routed. | Replay a copied workspace with its original rule files. Localize J2 footprint/rule compatibility and escape geometry; inspect routing output before modifying placement or rules. Owners to inspect: `cli/_compose_route.py`, `autoplacer/kicad_routing_tools.py`, and the actual connector footprint. |
| #10 `rp2040-min` | Leaf solve terminates after four attempts across three canvases with `leaf_solve_deadline,routing_exception`; compose never runs. Successful sibling leaves are retained. | Recover the failed leaf's router exception and reproduce its escape/deadline behavior. Inspect `cli/solve_subcircuits.py` and `autoplacer/brain/subcircuit_solver.py`; distinguish geometric infeasibility from exhausted routing time. |

Do **not** infer copper outside the board from `illegal_routed_geometry`, or infer a missing router installation from the human label `route/infra failed`. This batch successfully routed eleven other boards. Do not “fix” the USB example by inspecting its partial promoted preview as if it were the failed routed artifact.

These failure families overlap existing fine-pitch/escape plans. The creating defect is **not yet replay-verified**. No routing change is approved from the single-run results alone. Use N-of-3 frozen replays under original `quality=good`, fixed seeds, original `.kicad_pro`/autoplacer configuration, and a cold workspace that retains any required pre-promote seed. A successful reroute alone does not demonstrate a general fix. Measure original acceptance failures and keep all DRC constraints intact.

### Sourcing and library portability

- #34 `snowman-ornament`: BOM §9.26 rejects `ATTINY402-SSN` / C1339884 and reports retail stock 40. This is an **honest sourcing-policy rejection**, not a malformed reply or provider problem. Refresh assembly and storefront evidence, then select a reviewed stocked compatible implementation with a substitution record. Do not lower the stock rule or treat an annotation as fulfillment.
- #11's failed FPC candidate claimed C9900010611, absent from the offline catalog. Review/re-vendor the bundle identity and sourcing; do not make its fabricated-looking metadata authoritative merely because it lives in a library.
- Full `triage` audits were run for all 34. No Pass-A catalog suspects or Pass-B MPN-mismatch/Pass-C fabricated-LCSC tags were found in the **committed BOM audit population**. This does not contradict #11: its rejected unit candidate never became a committed BOM.
- Home-fetched dependencies remain in #17 (`smd1812p050tf`), #18 (`sd05c-tvs`), and #19 (`pc817c-s`, `uln2003`, `srd-05vdc-sl-c`). Vendor verified symbols/footprints/sourcing into the repository and replay from a clean parts environment. These are portability risks, not evidence that the corresponding files were missing during this batch.
- Existing standard-library parts reported without a custom manifest are not automatically fabricated parts.

### Quality caveats to retain, not blindly turn into blockers

#2's decoupling exists; #1's input and output should be separated by an RC network, and #2 is a DAC rather than a direct connector adapter. §9.22 currently keys on the word “breakout” and direct co-occurrence of connectors in one connection entry, so its zero-bridge warning does not prove these signal-processing circuits are disconnected. #4's actual film-capacitor value still triggers a token-level “Film capacitors” warning. Therefore **do not turn every material question or §9.22/§9.23 warning into a hard gate**. Replace ambiguous text matching with the required functional/physical contract before enforcing it.

Circle #29 and hexagon #32 passed deterministic outline-family checks. The other four shaped briefs did not build; they are completion failures, not measured outline regressions. Outline shape does not certify LED arrangement, radio performance, or electrical correctness. Programming facts explicitly pass for built #22 and #29; do not re-label their native USB/UPDI paths as missing from judge prose.

### Critical-review additions (2026-09-16)

These are additional specification/source findings, not changes to the frozen campaign's grades or failure attribution.

- **#15 has an unresolved operating-range conflict.** The original brief in `kicraft/tuning/benchmark.py` requests a **5 V to 3.3 V, 3 A TPS5430 buck**. The [TI TPS5430 datasheet](https://www.ti.com/lit/ds/symlink/tps5430.pdf), §5.3, specifies **5.5–36 V input**. A direct 5 V-input TPS5430 buck is outside its specified operating range. Inspecting the rejected BOM remains useful, but cannot resolve this specification conflict. A change of input voltage or reviewed device requires explicit approval and a versioned acceptance contract; neither has been approved. Do not claim that a correct refusal fulfills the original brief.
- **Deterministic parts are not proof of correct intent or composition.** Running the existing `kicraft.eval.recipe_coverage.analyze_recipe_coverage` over saved rc0 BOM states found #1 has 4/4 lowerer parts, #5 has seven recipe plus two lowerer parts, and #28 has ten recipe plus three lowerer parts. These three known-wrong deliveries already have entirely deterministic **part** provenance. This is not a claim about complete wiring provenance. More recipe coverage alone cannot fix lost physical requirements or disconnected cross-block power paths.
- **The existing interface source needs completion and persistence, not replacement.** `architecture_intent.py::_catalog` already selects recipe, lowerer, or declared interfaces. Its lowerer branch keeps supply/ground-looking keys with an otherwise open vocabulary; the full `declared_ports` claim does not survive as a canonical per-requirement interface. `stage_work_units.py::plan_stage_work_units` can route a failed lowering into a sheet-scoped model unit. Milestone 3 must close these seams without restoring the deleted bookkeeping layer.
- **Review parity does not enforce delivery correctness.** `synthesis/electrical_review.py` explicitly caps electrically sound requested-part substitutions at warning, and `build_design_digest` also blindly truncates at 14,000 characters. `server/web.py` re-drives only wiring once; missing physical parts require an earlier owning-stage repair. `run_post_wiring_review` is fail-soft and can finish with unresolved findings. Reuse this lifecycle for parity, not as the sole fulfillment oracle.
- **An emitted diagnostic is not necessarily an enforced gate.** The build tail in `design/cli_app.py` enforces persisted semantic `fab_gate` diagnostics only when raw `KICRAFT_STAGE_SEMANTICS` equals `enforce`; `Settings.stage_semantics` defaults to `repair`. Name the enforcement boundary and exercised execution mode for each new mandatory invariant. Do not blanket-enable ambiguous existing advisory checks.
- **Headless defaults are not consent.** `server/stage_runtime.py::NONINTERACTIVE_DEFAULTS_INSTRUCTION` asks the model to default without questions. Model-selected defaults or auto-answers cannot authorize a material substitution. The acceptance contract must contain any human-approved deviation before it can count toward an approved-contract result.

## 5. Constructive delivery milestones and acceptance

**Execution status:** all five milestones are planned, not implemented. This revision changes only the plan. No provider campaign, reference build, source change, or deployment is authorized by this document update.

**Unit of progress:** “this requirement has a discoverable, constructible, independently checked implementation that survives a fresh full design,” not “this error string disappeared.”

| Milestone | Deliverable | Completion evidence |
|---|---|---|
| 1. Explicit, feasible acceptance contracts | Versioned obligations and feasibility disposition for all 34 original briefs | Every mandatory requirement has an evidence check; conflicts and consent are explicit |
| 2. Reviewed deterministic reference inputs | A realizable reference circuit for each approved feasible contract | The real compiler/lowering/validation/synthesis path produces the required hardware without model luck |
| 3. Shared contract closure and live generation | Constructive fixes at the existing owning boundaries | Bad witnesses reject; valid implementations and fresh upstream generation satisfy the same obligations |
| 4. Early routing and delivered geometry | Legally routed reference and generated boards as coverage becomes available | Fresh accepted routed artifacts under original manufacturing rules, not partial previews |
| 5. Repeated artifact-checked fulfillment | Honest reporting, lifecycle parity, and repeatable release evidence | All 34 declared contracts fulfilled in fresh complete campaigns; no unsafe acceptance or omitted obligation |

Milestone 1 defines the acceptance contract before a cohort is repaired. Milestones 2 and 3 advance per cohort; milestone 4 starts as soon as each reference is available, not after all LLM failures disappear. Measurement work needed by milestone 5 starts alongside milestone 1. Use one integration owner for shared compiler/unit mutations; independent library work and read-only route diagnosis may proceed concurrently. Do not run overlapping project-wide validation across partial edits.

### Milestone 1 — establish 34 explicit, feasible acceptance contracts

**Own:** `kicraft/tuning/benchmark.py`, the existing evaluation/acceptance machinery, original briefs, and reviewed device/physical metadata. Contract expectations belong to evaluation; production behavior must not branch on benchmark slug, index, archetype, or exact brief text.

For each original brief, capture:

1. Stable corpus/brief identity and contract version, with mandatory functional, quantitative, physical, and mechanical obligations. Expectations must be independently reviewed against the **original brief**, not derived only from generated state that may already have dropped an obligation.
2. Feasibility disposition: reviewed feasible, specification conflict, or not yet reviewed. Record operating limits, applicable manufacturer evidence, and unresolved engineering choices. Do not label the other 33 feasible merely because only #15 currently has a documented range conflict.
3. Explicit assumptions and human-approved substitutions, including who approved them and which requirement changes. An unanswered question, a substitution ledger entry, or a model default is not approval.
4. For each obligation, a concrete check, owning implementation boundary, evidence source, and result: pass, fail, or unverified. Checks must identify actual part/package/pin/net/geometry or computed electrical facts; prose claims and component labels are not proof. Missing evidence cannot yield pass.
5. The reviewed recipe/lowerer/part realization or the named coverage gap. An operating-range conflict is distinct from an unavailable part or an unsupported compiler interface.

All contracts also require real, appropriately rated and sourceable parts, correct symbol/footprint/pin mappings, electrical operating limits, complete required connections, faithful exported artifacts, and the existing applicable ERC/DRC/programming/geometry gates. Do not add unstated numerical targets merely to make a test easy. If the brief leaves an engineering parameter open, record and review the selected assumption before checking its consequences.

#### Original-brief obligation inventory

This is the starting inventory for the executable contracts, **not** a declaration that feasibility or implementation has been verified. The common checks above apply to every row. Preserve every row, including the five rc0 designs not yet fully audited.

| # | Brief | Mandatory fulfillment evidence beyond a stage/build pass |
|---|---|---|
| 1 | `rc-lowpass-bnc` | Two physical BNC connectors; a cutoff-setting trim pot in the real passive RC path; adjustable response consistent with the declared range; no MCU. |
| 2 | `r2r-dac` | Eight distinct logic inputs on a header; correctly weighted R-2R network; op-amp-buffered analog output within declared supply/load limits; no MCU. |
| 3 | `thermocouple-amp` | MAX31855 realization with reviewed K-type input/support circuitry; screw-terminal input; complete SPI header and valid supply/interface levels. |
| 4 | `speaker-crossover` | Two-way passive topology; real air-core inductor, film capacitors, and binding posts; unit-aware response against explicitly recorded load/crossover assumptions; no active parts. |
| 5 | `usb-pd-trigger` | Physical USB-C PD input; reviewed switch configurations for 9 V, 12 V, and 20 V; powered controller and real output power path in each mode; suitably rated components. |
| 6 | `usb-c-full-breakout` | Real USB-C receptacle; requested VBUS/GND, CC, SBU, and high-speed contacts mapped to 0.1-inch headers without unintended omissions or shorts; legal fine-pitch escape. |
| 7 | `usb-a-power-splitter` | USB-C input and two physical USB-A power outputs; independent per-port current limiting and meaningful status-LED circuits; supported input/output power budget. |
| 8 | `rs485-terminal` | MAX485 with real signal and power isolation as required by the chosen implementation; separate reference domains; A/B/GND screw terminals; correctly operating DE/RE jumper. |
| 9 | `stm32-min` | STM32F103 in LQFP-48; USB-C, 8 MHz crystal, accessible SWD, and boot/reset buttons with reviewed support circuitry and valid pin bindings. |
| 10 | `rp2040-min` | RP2040 QFN-56, QSPI flash, USB-C, 12 MHz crystal, and usable programming path; actual castellated GPIO geometry, not ordinary interior pads. |
| 11 | `fpc-breakout` | Real 24-contact, 0.5 mm-pitch FPC/FFC connector; all 24 distinct contacts mapped to the 0.1-inch header row; exactly owned connector/header hardware. |
| 12 | `esp32-s3-sensor` | ESP32-S3 and BME280 temperature/humidity/pressure interface; physical USB-C and usable programming access; correctly connected, limited status LED. |
| 13 | `nrf52-beacon` | Reviewed nRF52840 package/order code, RF support and onboard chip antenna/keepout; coin-cell holder, user button, and usable programming path. |
| 14 | `lora-node` | SX1276 module, real SMA antenna connector/RF path, reviewed STM32L0 implementation, screw-terminal sensor interface, and programming/power support. |
| 15 | `buck-3a` | Original request is 5 V input, 3.3 V/3 A output, TPS5430, screw terminals, and thermal-via copper. Resolve the TPS5430 5.5 V minimum-input conflict before claiming a compliant direct buck implementation. |
| 16 | `highside-switch-10a` | Logic-controlled P-channel high-side MOSFET path and screw terminals rated for the declared 10 A conditions; gate-drive limits, conductor/thermal sizing, and real thermal geometry without a fictitious signal. |
| 17 | `led-cc-driver` | USB-C-powered 1 A constant-current path for the declared single-LED operating envelope; current-setting/feedback calculation, valid switcher support topology, heatsink copper; no MCU. |
| 18 | `dual-rail-supply` | Real DC-DC conversion from 24 V to both +12 V and −12 V; actual screw-terminal outputs and connected output filters; ratings checked against recorded load assumptions. |
| 19 | `relay-quad` | Four through-hole relays, SMT ULN2003 drive, genuinely opto-isolated input domains, reviewed coil/flyback paths, and screw-terminal contact outputs. |
| 20 | `encoder-oled-panel` | Through-hole rotary encoder with push button, actual SMT I2C OLED, three additional buttons, four physical mounting holes, and complete independent interfaces. |
| 21 | `proto-shield` | Canonical Arduino Uno shield geometry and pin mapping; real stacking through-hole headers, usable prototyping area, and SMT 3.3 V regulator. |
| 22 | `esp32-dual-motor` | ESP32-S3, two DRV8833 devices and their usable motor channels, buck conversion from the declared 2S battery envelope, motor screw terminals, and programming access. |
| 23 | `can-node` | STM32, SN65HVD230, DB9 with reviewed pin mapping, and genuinely switchable CAN termination; supported supply/logic levels and programming access. |
| 24 | `daq-8ch` | MCU plus two ADS1115 devices with compatible I2C addressing; all eight distinct analog inputs on screw terminals; valid input range and USB-C/power/programming implementation. |
| 25 | `gpio-expander` | MCP23017 with all sixteen GPIOs reaching screw terminals; physically chainable I2C header/interface and valid address/pull-up/supply choices. |
| 26 | `servo-driver-16` | PCA9685 and sixteen independently mapped signal/power/ground servo headers at the board edge; power screw terminal and declared external-load distribution budget. |
| 27 | `stepper-a4988` | A4988 support/current-setting circuit, working microstep-select DIP switches, motor connector, and 12 V screw terminal; valid pin capabilities and thermal/current envelope. |
| 28 | `audio-jack-buffer` | Four physical 3.5 mm jacks along the edge; complete declared channel mapping into unity-gain op-amp buffering with valid supply/signal range; no MCU. |
| 29 | `round-led-ring` | Round 60 mm outline; twelve WS2812B LEDs evenly spaced in a circle, complete data chain, ATtiny412/programming path, JST-PH power connection, and no edge connectors. |
| 30 | `rounded-c3-devboard` | Rounded-corner outline; ESP32-C3 with USB-C on one edge and a real 2x10 0.1-inch GPIO header on the opposite edge; usable mapped GPIO/programming access. |
| 31 | `chamfered-badge` | Chamfered outline; ATtiny1614, six 0805 LEDs, CR2032 holder, and two real capacitive-touch pads; valid current budget and touch-capable pin assignments. |
| 32 | `hex-env-sensor` | Hexagonal outline; BME280 temperature/humidity/pressure interface, real compatible Qwiic connector/interface, and working power LED. |
| 33 | `star-ornament` | Star outline with five warm-white LEDs at the points, ATtiny402, CR2032 holder, and top hang hole; valid LED/battery budget and programming access. |
| 34 | `snowman-ornament` | Three stacked snowman sections with warm-white LEDs in each, ATtiny402 and CR2032 power; reviewed orderable hardware, current budget, and programming access. |

**#15 decision gate:** obtain explicit approval to change the input specification or select a reviewed 5 V-capable device, or retain the original case as specification-conflicted. No option is selected by this plan. A corrected corpus needs a new version and baseline; keep the original 34 identities and report its unresolved case as non-fulfillment. “34/34 correctly handled” and “34/34 delivered” are different results. The delivery goal cannot be claimed by dropping the case or counting its refusal.

**Acceptance:** all 34 identities have reviewed obligation/evidence mappings and a feasibility disposition; every conflict has a recorded resolution or remains visibly blocking. The contract makes lost connector class, quantity, adjustability, isolated domain, and operating-range constraints observable independently of generated state. The original #15 conflict is explicit, with no self-granted consent.

### Milestone 2 — prove deterministic realizability with reviewed reference inputs

**Own:** existing architecture-intent compilation, recipes/lowerers, parts library, work-unit validators, synthesis, and reference fixtures for the milestone-1 contracts.

1. For every approved feasible contract, create an independently reviewed structured reference input at the relevant existing design boundaries. Exercise the real compiler, resolver, ownership, normalization, and commit validators. Reference candidates for genuinely model-owned units must pass those same boundaries; do not bypass a gate or mock a passing production result.
2. Resolve exact physical parts, packages, pin inventories, supported parameters, and ownership before treating a reference as realizable. Reuse and extend reviewed assets. Make part discovery return choices that the corresponding identity/physical validators can actually accept.
3. Pair every negative witness with a positive implementation. Hardening an obligation without supplying its satisfying implementation closes only a safety task. The six GAP-1 references must implement the requested circuit, not just stop earlier.
4. Validate original obligations against parts, pins, nets, unit-aware calculations, and synthesized/exported KiCad evidence. A correct BOOT/PH capacitor alone does not establish 1 A regulation; a PD power path alone does not establish all selector modes; an outline alone does not establish LED placement.
5. Make references portable: vendor reviewed home-fetched dependencies, preserve sourcing provenance, and exercise them from a clean parts environment. Unsupported identity/package/stock cases remain explicit failures.
6. Send each realized reference to milestone 4 immediately. Track unimplemented, electrically checked, synthesized, routed, and artifact-fulfilled references separately so synthesis cannot masquerade as delivery.

Fixtures are test inputs and engineering evidence, **not** board templates selected by benchmark slug in production. Recipe correctness and composed-board correctness need separate checks. No provider success is inferred from a no-LLM reference.

**Acceptance:** every approved feasible contract has a reviewed reference that the real deterministic path can realize with its required hardware and electrical facts. Missing coverage is named and blocks closure. The reference's required routed/geometry evidence is completed through milestone 4; an unbuilt reference does not count toward 34/34.

### Milestone 3 — close shared contracts, then prove fresh live generation

**Own:** `kicraft/design/architecture_intent.py`, `lowering.py`, `recipes/{models,resolver,registry}.py`, `part_identity.py`, `synthesis/validation.py`, `kicraft/server/{stage_contracts,stage_work_units,stage_runtime}.py`, stage-commit integration, and `kicraft/form_factors/reconcile.py`.

#### Preserve obligations and make interfaces executable throughout

- Carry requested connector class, adjustability, quantity, conversion behavior, and quantitative limits into owned requirements. Preserve the link to the original obligation; a class in `assumptions` or an empty `intent.named_parts` list cannot erase it.
- Complete `_catalog` and the lowerer interface definitions, and persist the relevant per-requirement interface claim for BOM/wiring verification against real pin inventory. Do not add a second vocabulary or restore the deleted explicit bookkeeping layer.
- Reject unsupported ports/parameters at the owning boundary with the accepted concrete alternatives. Distinguish novel model-owned circuitry from a supposedly supported lowerer that failed; do not silently turn compiler coverage defects into model-repair work.
- Give multi-supply and isolated parts explicit per-port rail/reference-domain bindings. Preserve one-port/one-net; only identical endpoint/conductor evidence can justify coalescing. Do not collapse grounds by name or normalize #27's conflicting A4988 capabilities into a pass.
- Include **#3 MAX31855** in the declared-interface cohort, alongside #7/#8/#9/#20/#23/#25/#26/#27/#30/#31/#33. Uncurated hardware requires a complete discoverable contract and verified eventual realization, not only more repair prose.

#### Pair implementation accountability with constructive coverage

- Close #18's regulator loophole in the existing requirement-implementation predicates. An unnamed converter still needs converter hardware. Track actual fulfillment through BOM/wiring; do not equate “not handled by a registered recipe” with “unfulfilled” when reviewed model-owned hardware can satisfy the requirement.
- Provide reviewed physical realizations for BNCs, binding posts, audio jacks, trim pots, and required converter functions. A tightened matcher must land with a valid satisfying implementation, not only a larger rejection list.
- Check source-to-load behavior across recipe boundaries (#5/#18) using reviewed component transfer behavior and reference domains. Arbitrary connectivity through a capacitor, control pin, or unrelated rail is not a power path; a PD negotiator is not an independent voltage source.
- Use typed units and supported-topology calculations for #4; check device-specific bootstrap/support topology and the full current-regulation contract for #17. Check reviewed operating ranges against actual typed inputs, including #15's conflict, without silently changing voltage or device.
- Add a valid 24-pin, 0.5 mm FPC realization and explicit connector/header ownership (#11), using physical metadata rather than stock-library prefixes or an unjustified numeric footprint-range extension.
- Align NRF52840 bundle/order codes and STM32L0 discoverable members with manufacturer-reviewed identity data (#13/#14). Preserve explicit candidate MPNs and reject wrong packages/unrelated devices; no permissive prefix matching.
- Resolve #34 through current stock evidence and reviewed compatible hardware under the milestone-1 consent policy. Do not lower sourcing requirements or repeat identical deterministic BOM attempts.

#### Repair at the owning stage and retain accepted work

- For #12, route the repeated `D1/R9` singleton to its model-owned wiring unit. Do not label it an immutable architecture defect. Preserve correct architecture/BOM and accepted sibling units, with bounded repair.
- For #16, keep thermal features in actual physical geometry unless a real electrical terminal owns them; no invented `THERMAL_PAD` signal.
- For #24, preserve all eight analog channels through unit ownership and aggregate inter-sheet realization.
- For #21, establish true stacking-interface ownership before reconciliation. Unsupported migration must return a structured failure transactionally; catching the exception and omitting headers is not a solution.
- Keep repair local to the stage that can change the failed obligation. A wiring-only re-drive cannot add a missing BNC, jack, or converter. Discovery/contract contradictions need correction at their source, not more identical provider attempts.

#### Make mandatory invariants genuinely enforceable

For each new invariant, specify the authoritative typed inputs, diagnostic code, constructive satisfying path, owning hard commit/build boundary, and exercised production configuration. Do not stop at a persisted `fab_gate` flag that the selected execution mode ignores. Align settings interpretation where needed, but do not switch every existing advisory heuristic to hard enforcement. Missing/unmeasured evidence is not a positive fulfillment result.

**Acceptance:** deterministic saved candidates reproduce the old defects and the corrected implementation satisfies the paired positive references. Package, pin, domain, ownership, and sourcing negatives remain rejected. Then run N-of-3 live frozen-stage comparisons for changed stochastic behavior and fresh full-design cohorts from the original brief. Record first-draft acceptance, requirement retention, calls, spend, terminal defects, and artifact fulfillment. All six GAP-1 designs must have positive circuit evidence; six earlier refusals alone cannot close this milestone. Previously correct affected behavior must remain correct.

### Milestone 4 — route early and verify delivered physical requirements

**Own:** the placement/escape/router boundary identified by replay; relevant physical-library assets and existing promote/export/geometry checks.

Only 13 original designs reached routing. The other 21 are **untested routing demand**, not presumed successes. Route each milestone-2 reference and each newly recovered live design as it becomes available, rather than deferring all physical feasibility until the final campaign.

1. Diagnose #6 and #10 using copied frozen workspaces and the exact evidence in §4. Recover #10's router exception; localize #6's connector/rule/escape failure before choosing a source fix.
2. Preserve original `quality=good`, preset seeds, `.kicad_pro`, autoplacer rules, companion files, and required pre-promote seed. Use cold copied workspaces and repeated comparable replays. Do not treat a partial promoted preview as routed evidence.
3. Fix the demonstrated creating defect. No weaker clearance/annular/keepout rules, longer timeout by default, post-route circuit rewiring, or manual generated-board repair. Keep footprint vendoring and layout changes separately attributable.
4. Verify a fresh accepted routed parent, final ERC/DRC/manufacturing acceptance, no missing required parts or opens, and the milestone-1 physical obligations against the delivered artifact.
5. Measure geometry the brief actually requests: castellations, stacking headers, thermal copper/vias, connector edges, holes, LED count/spacing, and silhouette/diameter where specified. An outline-family pass alone is insufficient.

**Acceptance:** all approved feasible references and recovered live designs route legally under the declared rules and meet their physical contracts. The two historical rc6 witnesses have repeatable positive routing evidence after a demonstrated fix. Every newly exposed routing failure remains on the delivery failure list until resolved; milestone 3 cannot conceal it with a five-stage commit.

### Milestone 5 — fresh, repeated, artifact-checked fulfillment

**Own:** `kicraft/eval/{self_eval,run_web,metrics_web,design_acceptance}.py`, the existing electrical-review/silkscreen lifecycle, deterministic artifact checks, and release verification.

#### Measurement and lifecycle work starts with milestone 1

1. Report generation, fabrication, software-verified fulfillment, and physical/bench validation separately. Preserve rubric grades as advisory quality evidence, not delivery verdicts. Only a fabrication pass **and** every mandatory obligation positively verified counts as a software-fulfilled delivery.
2. Distinguish original-contract fulfillment from fulfillment under a separately approved revised contract. Keep specification conflict, coverage/sourcing failure, execution failure, and unverified evidence visible; none counts as a fulfilled original design. Maintain the original denominator and corpus identities.
3. Include every pre-build failure and non-rc0 full-build result in “Needs attention”, irrespective of grade. Do not mislabel an intentionally unbuilt successful design-only run as a build failure or as a delivered board.
4. Preserve actionable nested terminal diagnostic codes and distinguish harness exceptions, stage subprocess crashes, and build failures. Do not retain stale rejected-attempt codes after recovery. Seven pre-build MCU flags remain incomplete-design findings, not assertions about shipped hardware.
5. Replace blind truncation in **both** judge and electrical-review digests with structured bounded evidence and explicit omissions. Guarantee relevant brief/obligation, part, pin, net, and terminal facts; an omitted section cannot justify an absence claim.
6. Share the existing web post-wiring review/repair and silk-authoring lifecycle with batch, with explicit execution mode and separately accounted designer/review/silk/judge spend. Review remains supplementary: advisory or fail-soft review completion is not deterministic fulfillment proof.

**Measurement acceptance:** #2's C1 and its supply endpoints are visible; #10/#21 are not presented as delivery-ready; all 23 non-rc0 frozen cases appear in full-build attention output; stage crash attribution is preserved. A one-brief paid parity smoke exercises the shared lifecycle before the next full campaign. Rebuilt reports go to a new location; preserve historical raw scores and artifacts.

#### Release gate

1. After integration, run the targeted behavior-level counterexamples and positive references, then the applicable shared validation once. Mutating a required connector class, deleting a required channel, breaking a power path, or shorting isolated domains must fail the relevant artifact check; do not test only field forwarding or prompt wording.
2. Use N-of-3 frozen stage/route replays for diagnosis, not a reliability certificate. Frozen-stage success must survive fresh upstream generation; reference fixtures do not substitute for live provider evidence.
3. Resolve milestone-1 specification decisions before selecting the release corpus version. Run all 34 contracts in a **fresh directory**, no resume/reused stage answers, judge enabled, declared production profile/model/rules, unchanged spend ceilings, and equal start/end source fingerprints. Preserve this campaign as the historical baseline; any approved corpus correction establishes a separately identified baseline.
4. Require 34/34 software-fulfilled deliveries, zero accepted circuits violating mandatory obligations, no missing/unverified mandatory checks, and no unsupported consent. A correct refusal remains a non-delivery and blocks this release target.
5. After the first clean campaign, require **two additional fresh complete campaigns** under the same declared conditions: 102/102 fulfilled runs across three campaigns, not the best sample per brief. Predeclare routing seeds and retain every outcome. This is an operational gate, not proof of an arbitrary population reliability bound: even independent 99%-reliable designs yield only about a 71% chance of all 34 succeeding.
6. Exercise small, independently reviewed variations of supported parameters, connector counts, and interface compositions to check generalization. Keep these additional cases separate from the original 34 denominator; production code must not recognize benchmark wording.
7. After acceptance, deploy using `deploy/deploy-production.sh`; verify both services, public HTTPS, and a real-provider design smoke. A healthy deploy is necessary but does not replace the delivery gate.

**Meaning of success:** 34/34 **software-verified fulfillment** under the declared contract and operating assumptions. Circuit invariants, calculations, and ERC/DRC do not certify measured RF performance, thermal margins, or protocol behavior on manufactured hardware. Require suitable simulation or representative hardware tests before making those stronger claims; record physical validation separately.

### Future verification commands for the implementation owner

These commands are instructions for later implementation, not work performed by this plan revision. Use the same declared production profile and preserve the frozen campaign.

```bash
# Evidence readers, no provider spend:
.venv/bin/python -m kicraft.cli.triage run logs/self_eval/20260915T132650Z/run_18_dual-rail-supply
.venv/bin/python -m kicraft.cli.triage stages logs/self_eval/20260915T132650Z/run_21_proto-shield
.venv/bin/python -m kicraft.cli.triage scan logs/self_eval/20260915T132650Z

# Live stage check after source changes; run N-of-3 in fresh replay workspaces:
.venv/bin/python -m kicraft.server.stage_driver replay --state logs/self_eval/20260915T132650Z/run_13_nrf52-beacon/.kicraft/state.json --stage bom --budget 0.25

# No-LLM route check, only after copying the complete generated project:
.venv/bin/python -m kicraft.design.cli_app replay --project /tmp/COPIED_PROJECT --quality good --seed 0
.venv/bin/python -m kicraft.design.cli_app artifacts --project /tmp/COPIED_PROJECT
```

Replace `/tmp/COPIED_PROJECT` with an actual scratch copy retaining the required companion files and seed. Never replay destructively in the baseline workspace. A successful invocation alone does not satisfy the milestones: inspect the positive circuit and delivered-artifact obligations.

## 6. Complete per-brief disposition

Raw grades and verdicts below are unchanged. “Fab-ready” means build rc0, not a functional certification. All unbuilt and failed builds are included, including those omitted by the generated report's attention list.


| # | Brief | Grade / score | Rubric verdict | Build | Cost | Disposition |
|---|---|---|---|---|---:|---|
| 1 | `rc-lowpass-bnc` | D / 55 | NOT-READY | fab-ready | $0.008761 | BNCs/trim pot replaced by pin sockets/fixed resistor; GAP 1. |
| 2 | `r2r-dac` | B / 79.5 | SHIP-WITH-FIXES | fab-ready | $0.010628 | Fabrication pass; missing-decoupling judge claim disproven (C1 exists). |
| 3 | `thermocouple-amp` | D / 42.5 | NOT-READY | not built | $0.013959 | Architecture: uncurated MAX31855 requirement lacks declared interface. |
| 4 | `speaker-crossover` | D / 55 | NOT-READY | fab-ready | $0.023941 | Pin sockets replace binding posts; 0.51 µH versus required ~509 µH. |
| 5 | `usb-pd-trigger` | B / 85 | SHIP-WITH-FIXES | fab-ready | $0.014194 | USB input VBUS disconnected from controller/output VBUS_NEGOTIATED. |
| 6 | `usb-c-full-breakout` | C / 73 | REWORK | route/infra failed | $0.011503 | rc6: parent legality/escape failure; TX2N/TX2P opens; partial preview. |
| 7 | `usb-a-power-splitter` | D / 40 | NOT-READY | not built | $0.014711 | Architecture: lowerer supply vocabulary and conflicting power/LED bindings. |
| 8 | `rs485-terminal` | D / 40 | NOT-READY | not built | $0.016031 | Architecture: isolated power/ground domains and conflicting DE/RE bindings. |
| 9 | `stm32-min` | D / 44 | NOT-READY | not built | $0.012220 | Architecture: RESET_BUTTON repeats already-bound NRST. |
| 10 | `rp2040-min` | B / 83.5 | SHIP-WITH-FIXES | route/infra failed | $0.008883 | rc6: RP2040 leaf deadline/routing exception; compose never runs. |
| 11 | `fpc-breakout` | C / 61.5 | REWORK | not built | $0.023382 | BOM: 24-pin FPC coverage, physical-feature identity and sourcing conflicts. |
| 12 | `esp32-s3-sensor` | C / 62 | REWORK | not built | $0.031080 | Wiring: repeated STATUS_LED_A singleton; model-owned defect misattributed upstream. |
| 13 | `nrf52-beacon` | D / 50 | NOT-READY | not built | $0.020609 | BOM: NRF52840 bundle order code rejected by reviewed identity table. |
| 14 | `lora-node` | C / 63 | REWORK | not built | $0.027412 | BOM: STM32L0 candidate outside the reviewed accepted member set. |
| 15 | `buck-3a` | D / 46.5 | NOT-READY | not built | $0.021860 | BOM: converter requirement implementation refused; inspect exact TPS5430 candidate. |
| 16 | `highside-switch-10a` | D / 49.5 | NOT-READY | not built | $0.036429 | Wiring: unowned THERMAL_PAD inter-sheet endpoint (§9.14). |
| 17 | `led-cc-driver` | C / 66.5 | REWORK | fab-ready | $0.029754 | BOOT and PH shorted on TPS54331; bootstrap capacitor connected to ground. |
| 18 | `dual-rail-supply` | C / 66.5 | REWORK | fab-ready | $0.025456 | Connector impersonates a converter; power outputs also lack real filter paths. |
| 19 | `relay-quad` | C / 73.5 | REWORK | fab-ready | $0.081607 | Fabrication pass; home-fetched relay/opto/driver dependencies; review quality evidence. |
| 20 | `encoder-oled-panel` | D / 50 | NOT-READY | not built | $0.017611 | Architecture: encoder has no declared supply port; button-port reuse. |
| 21 | `proto-shield` | B / 76 | SHIP-WITH-FIXES | not built | $0.019084 | Wiring commit subprocess crashes on missing owned stacking interface. |
| 22 | `esp32-dual-motor` | B / 82 | SHIP-WITH-FIXES | fab-ready | $0.010724 | Fabrication pass; deterministic native-USB programming path passes. |
| 23 | `can-node` | D / 41.5 | NOT-READY | not built | $0.012951 | Architecture: TERM_ENABLE conflicts with CAN_H binding. |
| 24 | `daq-8ch` | C / 63.5 | REWORK | not built | $0.030443 | Wiring: six missing analog-input sheet endpoints (§9.14). |
| 25 | `gpio-expander` | D / 44 | NOT-READY | not built | $0.016466 | Architecture: supply/connector-port conflicts after named-family correction. |
| 26 | `servo-driver-16` | D / 41.5 | NOT-READY | not built | $0.014288 | Architecture: unknown supply/unbound ports, then PCA9685-based ownership refusal. |
| 27 | `stepper-a4988` | D / 40 | NOT-READY | not built | $0.013765 | Architecture: A4988 pin reuse and unsupported/unbound interface assignments. |
| 28 | `audio-jack-buffer` | D / 55 | NOT-READY | fab-ready | $0.012188 | Requested 3.5 mm jacks replaced by pin sockets; empty substitution ledger. |
| 29 | `round-led-ring` | C / 74.5 | REWORK | fab-ready | $0.009843 | Circle outline passes; deterministic UPDI path passes; judge evidence needs correction. |
| 30 | `rounded-c3-devboard` | D / 41.5 | NOT-READY | not built | $0.013578 | Architecture: header binding conflict then unsatisfied GPIO capability. |
| 31 | `chamfered-badge` | D / 45 | NOT-READY | not built | $0.014759 | Architecture: status-LED supply/lowerer vocabulary and port conflicts. |
| 32 | `hex-env-sensor` | B / 78.5 | SHIP-WITH-FIXES | fab-ready | $0.019100 | Hexagon outline passes; no additional hard defect established by this audit. |
| 33 | `star-ornament` | D / 40 | NOT-READY | not built | $0.013344 | Architecture: status-LED supply/lowerer vocabulary and port conflicts. |
| 34 | `snowman-ornament` | D / 50 | NOT-READY | not built | $0.008157 | BOM §9.26: ATTINY402-SSN orderability rejection (retail stock 40 reported). |


## 7. Evidence and confidence

All paths below are relative to the campaign root unless stated otherwise:

- [Raw summary](../../logs/self_eval/20260915T132650Z/summary.md) and [machine-readable summary](../../logs/self_eval/20260915T132650Z/summary.json): all 34 scores, costs, terminal stages, gates, and outline checks.
- [Campaign manifest](../../logs/self_eval/20260915T132650Z/campaign_manifest.json): frozen corpus/revision/model/budget identity; [deployment verification](../../logs/self_eval/20260915T132650Z/deployment_verification.json): live-service and source provenance.
- [Run log](../../logs/self_eval/20260915T132650Z/run.log): initial launcher preflight, actual batch start, per-brief outcomes, shield traceback, and completion banner.
- [All-run triage](../../logs/self_eval/20260915T132650Z/triage_runs.json), [cross-run scan](../../logs/self_eval/20260915T132650Z/triage_scan.json), and [all-run audits](../../logs/self_eval/20260915T132650Z/triage_audits.json): generated using `kicraft.cli.triage` readers for every completed run. Round-level rejected artifacts can coexist with a successful final build; final `build_done` and provenance take precedence.
- [Analytical rollup](../../logs/self_eval/20260915T132650Z/analysis_summary.json): reconciled counts, gate populations, audit flags, and dimension means.
- [Quality reproductions](../../logs/self_eval/20260915T132650Z/quality_reproduction.json): current advisory checks, truncated-digest witness, crossover math, promoted USB-PD pad/net inspection, converter predicate, and datasheet-backed bootstrap evidence.
- [Stage reproductions](../../logs/self_eval/20260915T132650Z/stage_reproduction.json): current FPC predicate, exact identity comparisons, and shield reconciliation exception on an in-memory state copy.
- [Architecture research](../../logs/self_eval/20260915T132650Z/architecture_cohort_analysis.json) and [unit-stage research](../../logs/self_eval/20260915T132650Z/unit_stage_cohort_analysis.json): secondary code/artifact analysis for the enumerated early cohorts, with limitations and speculative proposals retained. They are supporting notes, not approved changes; this plan's safer constraints supersede any suggested name-based rail/ground coalescing or unreviewed identity broadening.
- Each `run_NN_<slug>/` retains `brief.txt`, `events.jsonl`, `.kicraft/state.json`, and `eval/report.json`. Runs reaching build additionally retain `.kicraft/build.log`, synthesis/ERC evidence, generated KiCad projects, routing experiments, and any exported package.

**Campaign verification (2026-09-15):** all 34 briefs finished and were judged under unchanged source; every run received triage/audits; source/runtime deployment and public availability were checked; selected deterministic contract failures and circuit defects were reproduced/inspected as described. The source campaign was not repaired or replayed in place.

**Critical-review evidence:** the §4 additions come from current-source inspection, read-only use of the existing part-provenance analyzer on the saved BOM states, the original #15 brief, and TI's linked operating-range specification. They do not establish a new provider, routing, or physical-validation result.

**Not verified:** implementation of these milestones, comparative N-of-3 model improvement, root cause of the two route failures, complete electrical correctness of the other five rc0 designs, all-34 feasibility, or authenticated web end-to-end parity. Implementation is paused; this revision changes the plan only. No source/test/configuration changes, paid runs, builds, or deployment are part of it.
