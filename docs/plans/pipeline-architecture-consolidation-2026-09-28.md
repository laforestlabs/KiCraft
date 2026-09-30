# Pipeline architecture consolidation — 2026-09-28

**Status:** implementation in progress, not released. The original baseline was interrupted for a reproduced deterministic safety defect. The operator approved prerequisite repair and independent verification before freezing a new baseline; those safety gains are excluded from later consolidation gains. See §§13–14.

**Decision:** make KiCraft a constructive design-and-repair pipeline, not a sequence of independently authored documents that must each survive rejection. Reuse the existing circuit compiler, parts library, state, and PCB backend. Consolidate or remove stage boundaries and repair paths when they force the model to restate facts or prevent correction at the source.

**Objective:** more original requests produce complete, software-verified boards within the existing spend/time limits. A smaller supported set is an implementation starting point, **not the primary fix or permission to narrow the product promise**. Reliability requires both broad enough construction coverage and recovery when an otherwise reasonable choice fails.

## 1. Evidence and limits

The original investigation used `kicraft.cli.triage scan`, `run`, `stages`, and `audits`. Its historical census is retained, not remeasured by this review:

- Of 2,354 historical runs with stage events, 1,913 ended on an uncommitted stage (about 81%). Architecture contract rejection accounted for 659 runs; BOM unit-repair exhaustion for 248; BOM commit rejection for 117.
- Separately, 281 runs had layout/ERC artifacts, of which triage classified 188 as clean/fab-ready. This selected population excludes most early failures; it is not an end-to-end success denominator or proof of functional correctness.

This review ran read-only `triage run`, `stages`, and `audits` on both recent witnesses:

- `1/1090` committed intent and functional specification, then failed architecture after five attempts. Recorded diagnostics moved through USB supply/port contracts, unsupported CP2102N coverage, and missing ownership of the named CP2102N. No build ran. These are several different constraints, not evidence that every attempt repeated one identical error.
- `1/1089` committed architecture and had no failing BOM work units in the triage summary, but aggregate gate 9.42 rejected unproven physical pin-header realization for the RP2040 requirement. No build ran. This does **not** establish that the library lacks a header; inspect which requirement owns it, which unit emits it, and what the aggregate check recognizes.
- Neither audit has a committed BOM to certify. The audit reports stale catalog data; its mechanical keyword heuristic is not a verified mechanical defect.

Current source establishes these starting points:

| Observed implementation | Consequence for this plan |
|---|---|
| `stage_contracts.py::validate_obligation_retention` requires functional specification to repeat committed obligation rows; architecture restores the top-level rows but still assigns them to requirements. | Stop model-copying immutable facts; retain the real choice of who implements them and independently check fulfillment. |
| `architecture_intent.py`, recipe resolution, and `stage_work_units.py` already derive circuitry and partition deterministic/model-owned work. | Complete and simplify this path; do not introduce a second circuit representation or resolver. |
| `web.py` and `eval/self_eval.py` already invoke `cli_app.py::run_post_wiring_lifecycle`. | Review parity is partly implemented, not a new workstream. Verify actual execution-policy parity. |
| `cli_app.py::run_post_wiring_review` can re-drive wiring once and remains fail-soft. | A completed review stage is not a correctness certificate; a missing component cannot always be fixed by wiring. |

These are mixed historical, experimental, and repeated runs, not today's production success rate. The witnesses were read, not replay-reproduced on current code. Source observations identify architectural seams; they do not prove a new regression or its yield impact.

**Working hypothesis [INFERENCE]:** too many model-authored consistency obligations, incomplete construction, and recovery at the wrong owner make reasonable designs fragile. Earlier rejection alone cannot fix that. Nor does fully deterministic part provenance prove correct composition: the [September 15 review](self-eval-2026-09-15-remediation-plan.md) documented wrong delivered circuits with entirely deterministic parts.

## 2. Relationship to existing work

Reuse, do not restart:

- [Constructive architecture slot](architecture-constructive-slot-2026-09-14.md): derived bookkeeping belongs to the compiler. Its scoped historical results are not a current broad reliability baseline.
- [Recipe and deterministic-lowering expansion](recipe-and-deterministic-lowering-expansion-2026-09-09.md): common circuits are compiled, reviewed assets, with exact identities and owned pins.
- [Self-eval remediation](self-eval-2026-09-15-remediation-plan.md), especially milestones 1–4: original obligations, independently reviewed references, constructive coverage, and delivered-artifact evidence.
- [Design-yield recovery plan](design-yield-recovery-2026-09-20-options-1-4-plan.md), especially **§12 Results**, not only its proposal: compiler-known rail/contact derivation, unique pin-name resolution, persisted commit diagnostics, advisories, and a manually selected legacy pipeline are recorded as implemented. Its promised paid yield improvements were not measured there.

This plan supersedes conflicting sequencing here: recovery-policy changes and elimination of redundant stage work are in scope; a supported envelope alone is not the product outcome. Preserve the earlier plans' unresolved correctness obligations and the original 34-brief denominator. Historical recorded results are evidence, not a substitute for checking current behavior.

### Critical decisions from this review

1. **Construct before rejecting.** Distinguish an impossible electrical requirement from missing compiler metadata, a recoverable choice, and an unverified part. Each needs a different action.
2. **Remove consistency work, not correctness checks.** The model should choose a circuit, not repeatedly copy obligation rows, exact identities, port inventories, nets, and ownership tables. Delete checks made redundant by construction; preserve independent electrical and delivered-artifact checks.
3. **Make adaptation an explicit runtime contract.** “Local repair” without upstream correction, progress criteria, or build feedback is insufficient. Replace wasteful repair branches within the current caps; do not merely add retries.
4. **Prove compositions, not just blocks.** Reviewed components can still form an unpowered, overloaded, unprogrammable, or physically incomplete board.
5. **Do not freeze the five model calls as architecture.** Keep useful user checkpoints; derive stages that contain only consequences. Merge decision steps where their split creates no independent engineering value.
6. **Measure breadth and delivery together.** Report original-request completion, supported-path reliability, and coverage separately. More early refusals or a high pass rate on a tiny subset cannot close this plan.

## 3. Scope and invariants

### In scope

1. Constructive feasibility, including composition of reviewed blocks and physical interfaces.
2. One resolved circuit as the authority for implementation ownership and derived artifacts.
3. Fewer independent model decisions and elimination of redundant stage payloads/checks.
4. Bounded local repair and upstream backtracking, including attributable build feedback.
5. Honest defaulting, clarification, substitution consent, and partial-output labeling.
6. Fresh end-to-end measurement of both request coverage and correct delivery.

### Non-goals

- A new router, general-purpose constraint language, multi-agent orchestration framework, or parallel design database.
- Arbitrary part/topology coverage before delivering the first reliable increment.
- More total attempts, larger prompts, or a model migration as the primary change.
- Relaxing electrical/manufacturing rules, silently changing requirements, or accepting placeholders.
- Benchmark-slug-specific production templates or special cases.

Existing stage structure and placement policy are **not sacred**. Targeted changes to partitioning, pin assignment, placement constraints, or allowed board geometry are in scope when a reproduced failure justifies them. Replace the backend only if separate evidence shows it is necessary; this plan does not assume that.

### Invariants

- Preserve the original brief's obligations, quantities, exact parts, electrical limits, and mechanical constraints. Record explicit consent for material substitutions or specification changes.
- Every implementation requirement and physical obligation has one accountable owner. Shared hardware must have an explicit supported sharing contract; it cannot be counted twice by accident.
- Unsupported, unmeasured, and review-required are not verified deliveries. A failed search is not proof that a circuit is physically impossible.
- Logical realizability is not a guarantee of routability, manufacturability, or electrical performance. Software-verified fulfillment is not hardware qualification or certification.
- Existing valid outputs must not become invalid merely because the internal representation changes.
- Reuse existing state, requirement identities, recipe provenance, and diagnostics. Do not introduce a parallel design database or competing capability vocabulary.
- Default reversible unspecified details from reviewed engineering rules and record the resulting operating limits. Ask only for contradictory requirements, safety-relevant unknowns without a defensible bound, or material choices the user must own. An assumption or automatic answer is not substitution consent.

### What may block completion

Audit the existing gates by the property they establish, not by their historical severity:

| Finding | Required treatment |
|---|---|
| Wrong voltage/domain, short, missing requested hardware, invalid package/pins, unmeasured mandatory fabrication check | Repair the cause or block verified delivery; never downgrade to improve yield. |
| Compiler-derivable identity, net, ownership link, or duplicate representation | Construct/canonicalize it at its owner; remove the redundant model field and associated rejection path. Ambiguous ownership remains a real design decision. |
| Advisory styling, unsupported inference from prose, optional review unavailable | Record the limitation; do not create a new hard stop by default. Missing evidence for a **mandatory** property still prevents a verified label. |
| Compiler contradicts its advertised realization | Internal implementation failure; do not blame the user or ask the model to rewrite protected circuitry. |

Every changed gate needs a correct positive circuit and an unsafe/incorrect counterexample. Removing a bookkeeping gate is simplification only when its invariant follows from construction; independent exported-artifact checks remain.

## 4. Target boundary

The LLM interprets the brief and proposes genuine circuit choices, especially novel topology or ambiguous implementation ownership. It does not own copies of facts already accepted.

The compiler resolves known choices, expands circuits, allocates resources, derives support components and physical endpoints, and returns concrete conflicts plus feasible alternatives. Prefer a deterministic ranked choice among reviewed equivalents when the brief leaves that choice open; use the model when an engineering decision remains.

Target flow:

```text
Original brief + preserved requirements + explicit user decisions
    -> proposed circuit choices
    -> compile: expand, bind, allocate, check composed circuit
        -> attributable conflict -> smallest responsible choice -> recompile
        -> resolved circuit -> synthesis/ERC -> placement/routing -> checks/export
                                   |                  |
                                   +-- bounded owner-directed correction --+
    -> original-requirement fulfillment assessment + honestly labeled artifacts
```

Clarification or a capability limitation is a terminal/paused outcome only when the allowed recovery cannot resolve it. Never keep spending on a known immutable gap.

“Resolved circuit” names the existing architecture, resolution/provenance, and derived BOM/wiring state—not a new model-authored IR. Separate immutable user requirements from revisable implementation choices. Canonical requirement rows are carried once by the compiler; model output references their IDs and selects owners rather than copying whole rows.

For covered compositions, **architecture acceptance must exercise real expansion and aggregate realization**, not merely confirm that a recipe name exists. Reuse the same normalization/validation functions that the later commit uses. Do not persist partial state while probing a choice. Electrical-domain compatibility, actual input-to-load power paths, programming access, resource capacity, and physical endpoints must agree across blocks.

Target for an accepted fully covered design: **zero model calls to author BOM or wiring**. Those stages become deterministic construction/checkpoints, including connections between blocks, not just their internals. Optional review and silkscreen calls are reported separately and cannot hide generation calls. For uncovered circuitry, constrain model work to the explicit unresolved portion.

Keep useful UI checkpoints, not compulsory model invocations. Derive functional-specification content that merely restates intent; merge it with design proposal where it adds no independent decision. Update all readers and evaluation definitions with the cutover; do not satisfy old stage counters by writing fake successful events.

## 5. Phase A — freeze evidence and choose the first deliverable

**Owners:** existing triage/evaluation tooling, recipe/lowerer registry, reviewed parts metadata, and original-brief obligation contracts.

1. Inventory the relevant linked-plan results and current code: implemented/exercised, implemented/unverified, missing, or obsolete. Scope this inventory to the first construction seam; do not make a repository-wide redesign a prerequisite to a board.
2. Freeze revision and working-tree fingerprint, effective non-secret configuration, model/provider, pipeline selection, parts data, quality settings, and spend/time limits. Separate production, evaluation, replay, repeated attempts, and legacy-pipeline results.
3. Establish a fresh full-generation baseline on the unchanged 34 briefs. Independently classify specification conflicts and missing capabilities **before** judging the new implementation. Keep every submitted brief in the overall denominator.
4. Select the first useful compositions from reviewed capabilities and actual demand, not only past successes. Include power conversion, MCU programming, an external bus/sensor, real connector realization, and repeated instances across the development set. Record voltage/current, exact variants, resource and geometry bounds.
5. Freeze development cases, held-out **combinations and boundary values**, and invalid/unsupported cases. Include the two witnesses for diagnosis, without declaring their missing coverage solved. Reserve independent expected results; generated requirements are not their own correctness oracle.
6. Record first-pass and recovered deliveries separately, calls, cost, elapsed time, terminal cause, and user intervention. Define a numeric minimum improvement on the original corpus and a coverage floor before implementation; neither can be met by changing dispositions after failures.

**Acceptance:** current baseline, fixed denominators, a versioned construction envelope, and the first end-to-end slice with explicit positive evidence to obtain. Breadth must grow with reliable construction. Narrowing the advertised product requires a separate explicit decision, not an automatic consequence of this envelope.

## 6. Phase B — make feasibility constructive and early

**Primary owners:** `kicraft/design/architecture_intent.py`, `kicraft/design/recipes/resolver.py`, existing recipe/lowerer metadata, and `kicraft/server/stage_contracts.py`.

1. Reuse the capability descriptions already exposed by `stage_reference.py::architecture_reference_extras`, recipe summaries, and lowerer summaries. Fill omissions from the executable contracts; do not create a separately curated compatibility matrix.
2. Support three honest realization modes within the same pipeline:
   - **Reviewed block:** exact versioned implementation with known parts, pins, electrical limits, and support circuit.
   - **Reviewed composition:** several such blocks/primitives plus checked binding rules. A new combination does not need a bespoke whole-board recipe. Check shared power, bus loading/address conflicts, pin capacity, returns, and physical connectors where applicable.
   - **Unreviewed remainder:** real part/pin/footprint evidence and explicit unresolved engineering claims. Constrain model work to this remainder; keep its output review-required until the relevant evidence is independently established.
3. Check exact user-requested parts early, but distinguish unsupported **selection** from unsupported **request**. If an unconstrained model choice cannot be built, select another feasible reviewed implementation automatically. Never replace a user-required CP2102N or change a requested connector type merely because another implementation is easier.
4. Offer compile/resolve feedback during proposal, using the same construction path as acceptance. Complete unambiguous implied support and connections; return the remaining conflicts together with the IDs and choices that can resolve them. Avoid a ladder that discovers one bookkeeping error per paid call.
5. Establish physical ownership before work-unit partitioning. Exercise expanded parts, wiring, library assets, and aggregate realization before calling the covered circuit resolved. A requirement split across reviewed blocks must name their supported composition, not masquerade as a single unrelated component.
6. A library gap should lead to concrete coverage work: verified part data, manufacturer reference circuit, reviewed implementation, then composed/exported evidence. Fetching a footprint or obtaining an LLM-approved circuit alone does not promote it to reviewed status. Do not silently fall through from a protected recipe to arbitrary model wiring.

**Acceptance:** positive compositions actually construct; incompatible ones name the violated constraint. A recoverable model-selected device does not cause an immediate user-facing refusal. An immutable exact-part gap stops before futile downstream calls. No original requirement disappears, and a working circuit is not rejected solely for lacking a whole-board template.

## 7. Phase C — consolidate resolved-design ownership

**Primary owners:** existing architecture/state models, recipe expansion and pin allocation, `stage_contracts.py`, `stage_work_units.py`, and synthesis validation.

1. Map authoritative producers/consumers for requirement IDs, choices, parts, pins, nets, interfaces, physical features, substitutions, and original-brief links. Extend existing state only for missing evidence.
2. Carry canonical source obligations automatically through derived stages. The model selects an owner by ID only where ambiguous; a unique, typed, capacity-checked match can be assigned deterministically. Validate that every required obligation is realized—not that the model copied its wording.
3. Generate BOM and wiring from the same expansion and ownership records. Preserve independent fulfillment expectations derived from the original brief; sharing implementation data must not make validation a tautology.
4. Check aggregate capacity/distinctness, shared-resource contracts, and repeated-instance isolation. A logical signal at the board boundary requires its requested physical contact; a recipe's internal programming hardware does not automatically fulfill a separate user connector.
5. Use existing work-unit dependencies and draft fingerprints for invalidation. An upstream choice change invalidates its dependent expansion, allocations, wiring, layout, and evidence. Keep siblings only when their inputs remain valid. A prior board is a stale preview until rebuilt and reverified against the new accepted state.
6. Probe revisions without mutating the last accepted state; adopt only a validated revision with its dependent state invalidated. Reuse the current stage-store/commit mechanism rather than introducing a second transaction database.

### Required deletion/cutover list

- Model-authored copies of immutable obligations and already-derived interfaces/identities.
- BOM/wiring model units for fully resolved compositions, including redundant compiler-to-model repair fallbacks.
- Duplicate ownership derivations and detectors that only check deleted bookkeeping fields.
- Stage-specific recovery branches replaced by the shared owner-directed policy in D.
- Obsolete prompt instructions, schema fields, stage-count assumptions, and aliases associated with those paths.

Migrate all affected readers/writers and update historical-state loading explicitly. Preserve original saved projects/artifacts. Do not keep two live contract paths indefinitely or automatically fall back to the legacy pipeline.

**Acceptance:** consistent ownership from choice through exported pins/parts; no copying-only provider calls; repeated instances remain separate; upstream edits invalidate dependent work without losing valid siblings. Record what was removed as well as what was added. A new schema wrapped around the old repair loops does not pass.

## 8. Phase D — make recovery bounded and adaptive

**Primary owners:** `stage_runtime.py`, `stage_work_units.py`, `session.py`, `stage_pipeline.py`, and the existing web/build handoff; recipes/lowerers own deterministic circuit repair.

Reuse the current unit routing, validated draft store, rejection signatures, candidate fingerprints, and BOM reconciliation machinery. Current source has wiring-to-BOM reconciliation and a web-only ERC recovery pass (`web.py`); it does not yet provide a shared general build-to-design recovery policy. Consolidate those seams rather than nesting another retry loop around them.

### Failure-to-action policy

| Failure evidence | Action and owner |
|---|---|
| Missing derivable metadata/support circuit in a reviewed implementation | Fix/complete deterministic construction; never ask the model to recreate protected pins or parts. An implementation defect is not “unsupported user request.” |
| Model chooses an unavailable implementation but the requirement permits alternatives | Resolve a different compatible reviewed choice, invalidate dependents, and recompile. No user question for a choice the user never constrained. |
| Local model-authored connection/parameter is wrong | Repair only that owned decision using the concrete violated constraint, then revalidate its affected circuit. |
| Aggregate/BOM/wiring failure caused by upstream ownership, topology, or resource allocation | Backtrack to the owning choice through the existing session flow; do not keep redrafting a downstream unit forbidden to change it. |
| ERC or electrical review names a concrete design defect | Attribute it to construction, BOM, wiring, or topology. Review is evidence to investigate, not permission to rewrite all wiring or an automatic hard gate on model opinion. |
| Placement/routing failure with attributable congestion, escape, or partition constraint | First use the existing placement/router search. When exhausted, change an allowed placement, assignment, or partition choice and rebuild the affected layout. Change outline/stack-up/device only within stated user freedom. Do not relax DRC or claim routing exhaustion proves physical impossibility. |
| Required exact part, electrical range, or geometry has no safe implementation within allowed choices | State the specific limitation; request consent only for a material alternative, or expose explicitly review-required output. Do not loop on the same gap. |
| Provider/transport, tooling, budget, or process failure | Preserve accepted work and report/retry under the existing infrastructure policy. Do not change the circuit to cure an outage or reset the run budget. |

### Recovery rules

1. Keep the existing hard project spend/time limits. Account for **all** nested repair, reconciliation, review-directed edits, and rebuilds under that run. Replace wasteful attempts, not stack new allowances. Reserve enough remaining budget for the chosen recovery and verification; stop honestly if it cannot fit.
2. Select the smallest choice that can actually change the failing fact. Feed stable diagnostic IDs, affected requirement/part/pin IDs, measured evidence, and allowed edits. Reuse existing diagnostic carriers; do not invent a competing exception/state hierarchy.
3. Repair from the last useful candidate and preserve unaffected work. A clean-slate escape is appropriate for unusable representation or a justified topology change, not automatically because a nearly-correct circuit used its serialization allowance.
4. Use existing defect scores, signatures, and fingerprints, but distinguish **repair progress** from **alternative exploration**. A repair must remove its target defect without losing requirements or breaking accepted invariants. Fewer messages, different prose, or a changed gate ID alone is not progress.
5. A new upstream alternative may temporarily invalidate dependent work; it is allowed only as a bounded, constraint-relevant, previously untried choice. Record choice and failure identity to prevent A→B→A oscillation. An unchanged candidate/failure stops or escalates; it never receives another identical request.
6. A deterministic implementation failure may trigger another reviewed implementation if the brief permits one. It may not trigger arbitrary model repair of the broken implementation. Keep the defect attributable even if an alternative completes.
7. Share the same policy across web, headless generation, and evaluation. The build worker returns evidence; the existing session owner decides whether to revise the design and enqueue a new build. Do not make the worker a second autonomous designer.

**Acceptance:** exercise local repair, upstream backtracking, allowed implementation replacement, and build feedback. Valid siblings survive; protected pins stay protected; repeated/oscillating choices terminate; caps cannot reset through re-entry. Demonstrate at least one case that fails without the justified recovery and exports correctly with it, plus a real contradiction that remains blocked. Measure recovery-policy changes separately from coverage additions where possible.

## 9. Phase E — expose the actual product promise

**Owners:** existing web project/stage status surfaces and export labeling. Locate the current UI/state integration before implementation rather than creating a separate support-status store.

1. Show what is known to be constructible, which assumptions apply, and any immutable limitation before substantial downstream spend. Update that explanation as composition/build evidence arrives; do not promise a finished board on parts coverage alone.
2. Continue automatically through safe defaults and allowed implementation alternatives. Explain the actual action: e.g. “reassigning the output to an available GPIO” or “adding the required programming connector,” rather than asking the user to fix an internal contract.
3. Preserve the distinction between **user-required** parts/limits and **model-selected** ones. Require explicit approval to change the former. Normalize a symbol alias or ordering suffix automatically only with reviewed evidence that the requested device, package, ratings, and relevant capabilities are unchanged. Same-family membership is insufficient: a memory-size change is not an equivalent spelling.
4. Persist original request, accepted deviations, assumptions, and realization evidence through existing state/provenance. An architecture assumption or substitution ledger records a change; it is not proof the user consented.
5. Distinguish software-verified complete export, generated but review-required output, failed/partial preview, awaiting clarification, and capability limitation. A downloadable PCB or a completed review stage does not by itself justify “verified.” Partial or unsafe artifacts must not be presented as fabrication-ready.
6. Explain limitations in circuit terms and offer only checked alternatives. Keep existing projects/artifacts intact; a new envelope must not silently rewrite briefs or upgrade historical assurance.

**Acceptance:** exercise the actual web UI with a straightforward delivery, automatically recovered delivery, exact-part limitation, clarification, exhausted build, and review-required case. Reopen the project and confirm status, accepted deviations, stale-preview labeling, and evidence agree with persisted state. A refusal is correctly handled but never counted as delivered.

## 10. Phase F — verify the product outcome and release

Use existing stage-driver, triage, self-eval, replay, and production deployment tooling. Read the relevant runtime skills before execution. Do not create a second evaluation harness.

### Required evidence

- Deterministic before/after witnesses for changed contracts, paired with correct implementations and unsafe counterexamples. Keep behavioral regressions for ownership, quantities, variants, domains, progress/stall transitions, consent, and dependency invalidation—not snapshots of error wording or source code.
- Frozen-stage N-of-3 replay for changed stochastic behavior. Diagnose the stage without claiming end-to-end success from its commits.
- Fresh full runs from original briefs: at least three independent trials per brief on the original 34, plus the held-out supported compositions. Freeze trial counts before either arm. No successful cached stage reuse, manual candidate editing, or unrecorded clarifications in the fresh-generation measure.
- Real construction/build/export of each reviewed reference as soon as it exists. References establish feasibility, not LLM generation yield. Preserve complete project rules and provenance for separate build-tail replays; a copied bare board is not equivalent evidence.
- Independent assessment against the original brief, including numeric operating limits, requested hardware, power and signal paths, programming access, mechanical constraints, synthesis/ERC, layout/DRC, and complete fabrication exports. Inspect actual exported pins/nets/parts; neither deterministic provenance nor an LLM judge's grade suffices.
- Web/headless/evaluation policy parity, including BOM reconciliation, post-wiring review, **build feedback**, and user-consent handling. Current shared post-wiring review does not establish that parity: the web has a separate ERC recovery branch.

### Scorecard and proposed release gate

Report these separately:

| Measure | Denominator / meaning |
|---|---|
| Original-request delivery | All submitted attempts on the unchanged corpus; unsupported, provider-failed, budget-exhausted, and clarification-paused attempts remain non-deliveries. |
| Construction coverage | Requests whose required circuit can be independently constructed within the declared bounds; measured before the run, not assigned only to winners. |
| Supported-path reliability | Correct complete exports / all attempts inside the frozen envelope, separately for development and held-out compositions. |
| Recovery effectiveness | Recovered correct deliveries, failed recoveries, extra calls/builds/cost, and terminal cause. |
| Breadth | Distinct original briefs correctly fulfilled and per-family/per-brief outcomes, not only aggregate attempt yield. |
| Assurance | Generated artifacts, fabrication-gate passes, software-verified fulfillment, and physical validation are different counts. |

Freeze the numeric breadth/improvement targets in A; do not tune them after implementation:

1. Every supported brief has at least one independently reviewed correct end-to-end delivery; no capability has only fixtures or stage commits.
2. At least **90% of fresh supported attempts** produce software-verified complete exports on both development and held-out sets. Report counts and uncertainty; three trials on a few briefs do not establish a universal 90% service guarantee.
3. The original 34-brief corpus meets the preregistered **positive delivery improvement and coverage floor**, in both attempt rate and distinct briefs. Merely reporting a flat/worse broad result beside an improved narrow score does not pass. Investigate per-brief regressions; historical unsafe outputs are not successes to restore.
4. Zero observed safety-critical or silent requirement-loss deliveries. All negative cases receive their expected disposition; false refusals of supported cases count as failures. A provider outage is separately attributed but is still a user-visible non-delivery.
5. No increase in hard per-run spend/time limits. Report total spend—including failed runs, review, and repair—per correct delivery, plus first-pass/recovered latency.
6. Fully covered compositions require no BOM/wiring authoring calls. The cutover removes the corresponding duplicated payloads and recovery branches; both simplification and delivery gains must be demonstrated.

Failure of the gate returns work to its owning phase. Do not change denominators, weaken checks, or repeat trials until only a favorable batch remains.

### Production rollout

- First verify offline/reference paths and isolated fresh runs. This is the production host: no service restart during plan review and no overlapping evaluation campaigns. Use the existing build-slot controls, conservative concurrency, and a declared campaign spend cap; measure away from production if resources cannot be isolated.
- Deploy the exercised increment only after its affected-case, broad-regression, and actual UI checks pass, using `deploy/deploy-production.sh` and its health verification. Do not describe an increment as completion of this plan until the full release gate passes.
- Preserve a rollback revision and state-schema compatibility strategy before deployment. No automatic legacy-pipeline fallback or rollback by relaxed electrical/fabrication gates.
- Monitor the same completed-board denominator after release. HTTP health and worker readiness establish service availability, not design reliability.

## 11. Sequencing and stop conditions

A freezes the baseline and first slice. B–E are workstreams **inside each vertical slice**, not four prerequisites to beginning construction. Exercise F throughout. Keep one integration owner for state/ownership/recovery changes; independent reviewed-part additions may proceed separately against that contract.

### First slices, in order

1. **One covered composition reaches export without downstream circuit invention.** Use an independently reviewed MCU + power + external-interface design. Trace the `1/1089` ownership failure from requested physical part through expansion and gate 9.42; do not assume either the library or gate is wrong. Land the smallest general construction/ownership correction, prove real BOM/wiring/synthesis/layout/export, and measure a fresh design from its original brief. No product-wide disposition redesign first.
2. **Remove restatement and prove composition breadth.** Replace copied obligation rows with compiler-carried facts and explicit owner choices. Extend the existing recipe edges/companions and pin allocation where reviewed inter-block wiring is missing; do not add another DSL. Exercise repeated blocks, power-domain boundaries, and held-out combinations. Remove unnecessary model stages/units with their readers and tests.
3. **Recover a failing choice without restarting the design.** Prove local correction, one attributable upstream replan, and exhausted/repeated-error termination. Compare recovery policy on frozen inputs, then fresh generation. Do not reward changed diagnostics without a board.
4. **Close the build-to-design loop and expand coverage by failed demand.** Reproduce an actual build constraint, repair its owning allowable choice, and get a clean rebuilt export. Exercise the same path through web and evaluation. Add requested devices/support circuits from reviewed evidence, then repeat the broad corpus and held-out gate.

Each slice includes its positive construction, negative safety boundary, necessary persisted/UI behavior, deletion of superseded code, and delivery measurement. A limitation fix may close a safety defect without completing a delivery slice; record the distinction.

Stop and reassess when:

- the reference cannot construct/build: fix the deterministic implementation before spending on generation;
- the design needs another representation or independently maintained truth table: remove duplication first;
- yield improves only because requirements, hard cases, or infrastructure failures left the denominator;
- a policy change spends more on the same constraint or erases useful accepted work;
- a narrower supported set scores well but the original requests still almost never produce boards.

The response to the last condition is more useful construction/composition coverage or a different simplification—not silently declaring the remaining requests out of scope.

## 12. Definition of done

This plan is complete only when **more original requests reliably reach correct exported boards**, the frozen supported envelope meets the release gate, recoverable failures trigger bounded owner-directed correction, and downstream stages no longer independently reinvent resolved circuitry. The actual UI must distinguish fulfillment, limitations, assumptions, consent, and review-required output.

Update existing architecture/product documentation and the changelog with the exercised behavior. Remove obsolete contract/repair paths and temporary measurement scaffolding after proof. Preserve reusable evidence and behavioral regressions under existing conventions.

Not done: better prompts, more passing unit tests, earlier rejection alone, five committed stages without a board, success on a reduced denominator, a new schema with the old runtime underneath, or a broad “adaptive” loop that just spends longer before failing.

## 13. Execution preregistration — 2026-09-28

Implementation requested; release is not yet established. The operator authorized
a **$150 aggregate provider-spend ceiling** for implementation and evaluation.
This does not raise the existing $0.60 project, $20 daily, or $250 shared lifetime
ceilings. Failed attempts, reviews, replays, and held-out runs count against the
campaign allowance. Production services must not be restarted before the release
checks pass.

Before runtime edits or fresh generation, freeze these targets:

- Three separate fresh 34-brief campaigns per arm: **102 baseline attempts and
  102 implementation attempts**, using the original corpus and no approved-device
  variant. Retain every refusal, clarification, infrastructure failure, and
  exhausted attempt in the denominator. No rerunning a failed trial for selection.
- Minimum original-request improvement: **11 additional software-verified
  deliveries out of 102** (more than ten percentage points), and **four additional
  distinct fulfilled briefs** versus baseline, capped at 34 distinct briefs.
- Construction-coverage floor: **18 of the original 34 briefs**, independently
  constructed within recorded limits. A reference-stage commit is not a complete
  construction/export witness and cannot establish this floor.
- The existing 90% development and held-out supported-path delivery gates, zero
  observed unsafe/requirement-loss deliveries, unchanged per-run caps, and zero
  covered BOM/wiring authoring calls remain mandatory.
- Development priorities: MCU/power/external-interface composition, then repeated
  devices and resource/domain boundaries. Held-out combinations and their numeric
  limits must be frozen before implementation; no case is supported merely
  because its parts have recipes.

Starting source revision: `b40ee8f9661bac070f97ab8a47bdae2a363acd9e`.
The initial working tree had no tracked changes; this plan was its sole
untracked file. Existing evaluator contracts classify 18 briefs as
`reviewed_feasible`, 15 as `not_yet_reviewed`, and the original `buck-3a` as
`specification_conflict`. Those are recorded compiler-boundary classifications,
not a newly measured full-build coverage result.

The existing full-release verifier expects three separate fresh campaigns.
The evaluator's `--campaign-budget-usd` currently accepts only an external corpus;
it must not be passed to the original-corpus command as if it enforced a cap.
Resolve campaign-wide accounting and production isolation before paid execution.

### Initial seam inventory and evidence

| Seam | Current classification | Evidence |
|---|---|---|
| Architecture-derived nets, ports, recipe bindings | Implemented; broad delivery unverified | `architecture_intent.py::derive_architecture`; September 14 results are stage evidence, not current export yield. |
| Typed lowerers and protected recipe expansion | Implemented; composed/exported coverage must be measured | `stage_work_units.py::deterministic_bom_candidate`, `deterministic_wiring_candidate`, recipe registry. |
| Unique physical-owner completion | Implemented but incomplete | On frozen `1/1089`, the current normalizer returns owners `i2c_header` and `rp2040` even though its reviewed-candidate predicate returns only `i2c_header`. |
| Independent physical realization | Implemented; correctly rejects the witness | Fresh `triage run/stages/audits 1/1089`: BOM gate 9.42 rejects both local and aggregate RP2040-sheet header claims; no build exists. |
| Immutable exact-part limitation | Detected, but repeated repair remains | Fresh `triage run/stages/audits 1/1090`: five architecture calls; CP2102N coverage failure followed by missing ownership; no build exists. |
| Canonical obligation copying | Still model work in functional specification and requirement rows | `stage_contracts.py::validate_obligation_retention`, provider schema, and stage prompts. |
| Shared post-wiring lifecycle | Implemented; not build-feedback parity | Web and evaluation invoke `run_post_wiring_lifecycle`; web retains a separate ERC-driven wiring recovery. |
| Original-corpus release evaluation | Needs cutover to this plan's scorecard | Existing `design_acceptance --release` requires all 34 to fulfill, including the known original TPS5430 contradiction. Preserve its per-artifact checks; do not mistake that old all-success rule for this plan's supported-path/improvement gate. |

Reusable preflight evidence is under
`logs/self_eval/pipeline_consolidation_20260928/`: `preflight.json` (original
briefs, independent contracts, configuration and source fingerprint),
`data_identity.json` (catalog checksum and shared-spend headroom),
`construction_cases.json` (development, held-out and negative expectations),
and `ownership_before.json` (executed unchanged ownership normalizer).
The catalog was reported 15 days old by triage; it was not refreshed.

The source fingerprint is
`sha256:3cc411e7daa24e88f59036f168c3cfc4daea1a1829238d0d4461ca8b9245e30c`.
The configuration selects `openai/gpt-6-luna` via `openai`, with
`minimax/minimax-m3` as judge. Baseline execution uses `--parallel 1`,
`--build-slots 1`, original contracts, judge enabled, and the existing 2400 s
build timeout. At preregistration, the shared ledger had spent $141.57635:
the unchanged $250 ceiling leaves $108.42365, less than the authorized $150.
Production spending also consumes that headroom.

## 14. Implementation record

### Safety prerequisites and baseline interruption

The original unchanged-source campaign's checkpoint contains 9 completed records
of the scheduled 34, $0.140786 recorded cost, and 2 fabrication-gate passes.
It is **not** a completed baseline, and a fabrication pass is not proof of
electrical fulfillment. Its manifest, attempted runs, and explicit interruption
receipt remain under `baseline_r1/`; do not resume it or selectively retain
successful attempts for comparison.

The reviewed CAN reference exported with clean ERC, zero shorts/unconnected
items, and a fabrication archive, but its actual TPS54331 PH net had no catch
diode. TI SLVS839H identifies this device as non-synchronous. The exported
511k/100k EN divider also cannot establish the 1.35 V maximum enable threshold
from USB 5 V without treating typical internal pull-up current as a guarantee.
This is an implementation failure, not an impossible user request.

Prerequisite changes:

- TPS54331 construction now includes reviewed SS34, actual K→PH and A→GND,
  and the manufacturer's documented floating-enable mode as an explicit EN
  no-connect. Synchronous buck implementations are unchanged.
- Existing §9.37 independently checks required rectifier endpoints and reviewed
  current/reverse-voltage bounds; an equivalent sufficiently rated reviewed
  Schottky is accepted. Existing §9.38 checks static enable bias and absolute
  maximum, without substituting typical pull-up current for a guaranteed bound.
- Compiler-generated regulator requirement IDs now normalize punctuation and
  numeric prefixes, preserve the schema length bound, and avoid collisions.
  Original requirement IDs and electrical net names are not renamed.
- The prior incorrect “all bucks are synchronous” statement in the September 24
  implementation record is corrected. No PH→GND transfer-graph shortcut is added.

Verification: 329 recipe, semantic, and electrical-invariant tests passed.
Running the new validators directly against the **previously exported** CAN
state rejects both its missing rectifier and old enable bias. The fresh reference
completed synthesis, routing and fabrication export: 37/37 components, zero
shorts/unconnected/courtyard/keepout violations, 638 traces and 96 vias.
Independent PCB-pad inspection confirms D1 K→U3 PH, D1 A→U3 GND, floating
EN, USB VBUS→VIN, inductor output→MCU/transceiver supplies, and CAN H/L/GND
on DB9 contacts 7/2/3. The interrupted first build is retained separately from
the completed build; neither is a paid fresh-generation trial.

That inspection exposed a separate measurement defect: the electrical evaluator
required library nicknames that saved KiCad footprints omit, while inventory
evaluation already handled this correctly. Both now use the existing exact-item
matcher; conflicting present namespaces and different packages still fail.
41 artifact/product acceptance tests passed, and the same delivered board now
resolves 37 footprints rather than zero. This observation fix is part of the
new baseline, not a claimed consolidation gain.

`can_exported_graph_verified.json` preserves the actual graph and current
assessment. Fabrication passes, but software fulfillment remains **unverified**:
the switchable-termination observation is not implemented, and the frozen
sourcing policy records unverified retail stock. Complete required connections
therefore also remain unverified. These are not silently promoted to deliveries.

The SS34's reviewed nominal 3 A / 40 V limits cover the metadata's 3 A / 28 V
bounds. Generic inductor saturation, capacitor ripple/derating, and thermal
qualification remain unproven; the retained component-bound review requirement
must not be described as certification.

Separate observed issue reserved for the architecture-authority cutover:
declared-load completion currently replaces a requested rail voltage with the
nearest reviewed converter output. The new baseline must retain and count this
behavior honestly; consolidation must not silently change original limits.

### Phase E — assurance, consent and export labels (2026-09-29)

Implemented as one **derived** reading, not a new store: `web._project_assurance`
computes every project's label from its own persisted evidence (committed slots,
`stage_status` including the recovery history, review findings, `.kicraft` build
gate/synthesis summaries, and any independent verification recorded for the run)
and a reopen derives the same answer. Surfaces: the workspace summary (label +
one-line statement + folded evidence/assumptions/limitations/actions/deviations),
the project rows, the Fab inspector and the export/download controls.

Labels implemented: software-verified complete export; fabrication-ready but
independently unverified; generated with recorded gaps (review required);
recorded capability limitation; incomplete/failed or **stale preview**; awaiting
clarification; work in progress. Promotion to verified requires a CURRENT,
artifact-hash-bound verification (`eval/product_acceptance.json` or filled
`eval/acceptance_evidence.json`); a downloadable package, parts coverage or a
completed review stage cannot produce it, and a recorded success whose artifacts,
delivered selection or brief no longer match is reported stale and never counted.

Consent: a `bom.substitutions` ledger entry (or an auto-defaulted answer) is
never treated as consent. A change to a part the user required needs a recorded
answer that names it; otherwise the export is labelled review-required and the
change is listed as having no recorded approval. Model-selected substitutions and
automatic defaults are reported as actions KiCraft took.

Staleness: browser artifacts that the current accepted design no longer matches
(the real `invalidate_downstream` path) are labelled a stale preview, kept on
disk as history, and their download is withheld. `stale` is registered as a
presentation status in `activity.PRESENTATION_STATUSES`.

Evidence: `tests/test_web_assurance.py` (19 tests) covers the six acceptance
scenarios, stale preview, ledger-vs-recorded-answer consent, a real
`evaluate_product` audit that promotes and then goes stale when an artifact
changes, and the actual page (honest export label; withheld stale download). 112
neighbouring web/presentation tests still pass.

### Phase B/C — constructive feasibility and obligation ownership (2026-09-29)

**What was removed (model-authored copies of immutable facts).** A requirement no longer
copies the user's typed obligation rows. `kicraft/design/architecture_intent.py::IntentRequirement`
dropped `obligations: list[RequirementObligation]` for `obligation_ids: list[str]` — the
`original_obligation_id` values the requirement implements. The provider schema
(`stage_contracts.build_stage_response_contract`) no longer publishes a requirement-level
`obligations` property for the architecture stage nor a slot-level one for `functional_spec`,
and constrains every `obligation_ids` entry with an `enum` of the committed ids, so an invented
id cannot be emitted. The prompts, worked examples, stage reference and
`.agents/skills/kicraft/stages/{architecture,functional_spec}.md` say the same thing. The
compiler writes the rows instead: `restore_source_obligations` fills `FunctionalSpec.obligations`
and the architecture's top-level list from the committed intent/spec, and `derive_architecture`
resolves each requirement's ids to those rows (contact-count checks, prototyping-area derivation
and every downstream reader use the resolved rows).

**What checks remain, and on what.** `validate_obligation_retention` now verifies *ownership*, not
copying: an owner-requiring committed row no requirement names → `source_obligation_not_retained`
(with the committed row as evidence and the uniquely-provable candidate ids); a named id no
committed stage carries → `unknown_source_obligation`. It reads either shape (provider ids or the
canonical resolved rows), so the same check serves the provider path and the manual slot path
(`cli_app._apply_slot`, which now also restores the rows). `attach_uniquely_provable_physical_obligations`
still assigns the unique reviewed physical owner automatically, by id.

**Historical input.** `_fold_legacy_requirement_obligations` reads a pre-cutover intent-shaped
draft that still carries rows on the requirement (reference transcripts, frozen states) for its
ids instead of refusing it; the canonical shape is untouched.

**Constructive acceptance.** Architecture acceptance probes the *actual* covered composition
(`stage_contracts._validate_constructed_architecture` → `stage_work_units.probe_architecture_construction`:
deterministic BOM + wiring expansion through the production compiler, then footprint/symbol
inventory, symbol↔footprint pin mismatch and composed-wiring checks). The probe is
side-effect-free and runs at **commit**, after the clarification path: a missing external-load
current still parks as one question, and only a covered circuit that actually refuses to compose
is rejected. `cli_app` architecture commit runs the same probe against the real project root, so
a manual slot edit cannot commit an invalid covered circuit either.

**Two defects the cutover exposed (both fixed).**
`architecture_intent._adopt_carrier_families` and `stage_contracts._requirement_proves_physical_obligation`
read the requirement's physical classes from its own row list; with ids they silently saw nothing,
so a demanded `jst-xh-connector` class was not mapped onto the reviewed carrier family and a
lowerer-proven unique owner became invisible. Both now resolve ids against the payload's
committed rows (and still read legacy rows); the lowering probe projects the requirement onto
`CircuitRequirement` fields, since provider-only keys (`obligation_ids`, `supply`, `ties`) are
rejected by that model.

**Evidence.** `can-node` reference replay commits all five stages with `bom attempts=0 tools=0`
and `wiring attempts=0` (zero model calls to author BOM or wiring for a covered composition), and
the freshly built `can_constructive_reference` export completed synthesis + placement + routing +
fabrication export: 37/37 components, 0 shorts, 0 unconnected, 0 courtyard/keepout violations,
675 traces, 102 vias, Gerber/drill/CPL/BOM/STEP/3D package written. The ownership refusals,
the id→row resolution, the functional-spec restore and the legacy fold were each exercised
directly against the real helpers.

### Phase D — bounded, owner-directed recovery (2026-09-29)

The web-only ERC wiring retry is gone. One shared policy
(`kicraft/server/session.py::run_build_recovery`, `classify_build_failure`, `read_build_evidence`,
`_BUILD_OWNER_BY_GATE`, `_owner_choice_fingerprint`) now serves the web build, headless
generation (`stage_pipeline.run_pipeline`, the `stage_driver` CLI) and batch self-evaluation. The
build worker returns deterministic evidence only; the session owner decides and performs the
revision: wiring-owned gate → wiring only (siblings retained), BOM-owned → bom+wiring,
architecture-owned → architecture+bom+wiring (intent/spec preserved). Tenant actions are
`repair_wiring | backtrack_bom | backtrack_architecture | try_reviewed_alternative | none`; a
permitted reviewed alternative needs an availability-class gate or an attributable place/route
constraint, a model-selected part, and no user-pinned part or limit. Unchanged, repeated or
oscillating (owner,choice) fingerprints terminate with `recovery_repeated`,
`recovery_stalled`/`recovery_oscillation`, and compiler-owned gates (§9.37–§9.41) are reported as
`compiler_defect` rather than handed to the model. All nested repair, review-directed edits and
rebuilds bill to the same run and stay under the existing per-project/daily/lifetime ceilings —
no new attempts were added and no cap was raised. Durable history rides the existing
`ConversationState.stage_status['build_recovery']` carrier (`StageStatus.recovery_*` plus a typed
`BuildRecoveryEvent` reusing `StageDiagnostic`), so a reopen reads the same answer.

Review-directed repair is owner-scoped (`cli_app._review_repair_scope`/`_call_rewire`): a blocker
is attributed to bom, architecture or wiring instead of rewriting all wiring. Self-eval reports a
separate `recovery` section (status counts, recovered deliveries, attempts, extra designer spend)
so recovery gains are measured apart from coverage gains.

### Regression evidence: review-status replay (2026-09-29)

`design_acceptance.verify_reference_replay()` replays all 34 reviewed reference rows through the
five stages with a recorded transcript (`KICRAFT_LLM_MODE=replay`, $0, no provider). Run on the
frozen safety-prerequisite revision and on this candidate, the error sets are **identical, 10
slugs each**; two rows simply surface the same defect one stage earlier, which is what the
constructive acceptance is for:

| Row | frozen revision | candidate |
|---|---|---|
| `r2r-dac` | refused at wiring (9.38 `U1.3` `+3V3` outside the reviewed AP63205 VIN range) | refused at architecture, same defect |
| `proto-shield` | refused at bom (`uno_analog` declared interface unrealized) | refused at architecture, same defect |
| `highside-switch-10a`, `usb-a-power-splitter`, `rs485-terminal`, `led-cc-driver`, `dual-rail-supply`, `stepper-a4988`, `esp32-s3-sensor`, `esp32-dual-motor` | refused at the same stage with the same evidence | unchanged |

No reference row that committed at the frozen revision is refused by this candidate. The two
earlier refusals are the recorded defects themselves (`r2r-dac` is the known declared-load rail
substitution; `proto-shield` the unrealized declared interface), not new rejections.

Full unit suite on the candidate: 4793 passed, 15 skipped, 1 xfailed, plus one **pre-existing,
unrelated** failure — `tests/parts_library/test_vendored_bundles_load.py` reports a stale
`content_hash` in the shipped `srd-05vdc-relay` bundle, which reproduces identically at the frozen
revision and in every runtime (`skipping 1 broken parts: ['vendored:srd-05vdc-sl-c']`).

### Phase F measurement — status

Preregistered target: three fresh 34-brief campaigns per arm, `--parallel 1 --build-slots 1`,
original corpus and contracts, judge on, 2400 s build timeout. The baseline arm is running from the
frozen safety-prerequisite checkout (`baseline_safety_r1`). The candidate revision measured by the
implementation arm is commit `20f4147` on `simplify/bom-wiring-pipeline` (the frozen baseline arm
stays at `b40ee8f` plus the safety prerequisites in its own checkout). The remaining campaigns are
queued (one at a time — the host has two cores and two concurrent routing builds would trip the
build timeout) by `logs/self_eval/pipeline_consolidation_20260928/run_campaigns.sh`:
`consolidation_r1`, `baseline_safety_r2`, `consolidation_r2`, `baseline_safety_r3`,
`consolidation_r3`. The supervisor waits for the in-flight baseline campaign and then runs each
campaign to completion, recording to `campaign_supervisor.log`.

A three-brief live smoke on the candidate (`consolidation_smoke`) exercised the whole cutover
against the real provider: all stages ran on the new contract, and it produced no fabrication
delivery (graded mean 47.2, 0/3 fab-ready) for reasons unrelated to this change — two
architecture contract refusals (`can-node` repeated a `conflicting_port_binding` on `db9.can_h`,
already `CAN_H`) and one wiring 9.42 declared-interface failure on the relay quartet. The release
gate is **not** met: the campaigns must complete and be reported separately by arm, and no
increment is deployed on the strength of unit tests or reference replay alone.



### Release measurement — final (2026-09-30, revision `9317cc4`)

Three fresh 34-brief campaigns per arm, same corpus, same settings, one campaign at a time on the
same host. The baseline arm runs the frozen safety-prerequisite checkout, launched with its cwd in
that checkout **and an asserted `import kicraft` path** (an earlier launch from this checkout
silently imported the candidate, which invalidated every comparison before `baseline_safety_r3`;
those numbers are withdrawn).

| Arm | Campaigns | Fab-ready exports | Rate | Distinct briefs |
|---|---|---|---|---|
| baseline (frozen) | `baseline_safety_r3`, `r4`, `r5` | **15 / 102** | 14.7% | 10 |
| candidate `9317cc4` | `candidate_final3_r1`, `r2`, `r3` | **26 / 102** | 25.5% | 13 |

Per campaign: candidate 7, 11, 8 against baseline 6, 5, 4.

- **Deliveries: +11 per 102** — exactly the preregistered minimum improvement, and the candidate
  rate is 1.7x the baseline's.
- **Distinct briefs: +3** (`can-node`, `hex-env-sensor`, `highside-switch-10a`,
  `rounded-c3-devboard`, `servo-driver-16` gained; `star-ornament`, `thermocouple-amp` lost, each
  of which is 1/3 on the baseline with different failure reasons per candidate attempt, i.e.
  sampling) — **one short of the preregistered +4**.
- **Held-out pair: 0 boards in 3 rounds x 3 trials.** Every round reached later than the last
  (paper trail below); the last round's RP2040 attempt cleared intent, spec, architecture and BOM
  and failed at the wiring commit on a draft-invented dangling `QSPI_CS` net, which is a draft
  defect the pipeline may not rewrite. Reported as a **capability limitation**, not a delivery.

Deterministic evidence (independent of sampling):
- the seven preregistered negative boundaries all produce their expected disposition (4/7 before
  the I2C/analog gate work);
- uncovered demanded-class occurrences over 2 228 saved states: 14.2% -> ~6%;
- `complete_unused_published_ports` changes 11 of 270 recorded committed BOMs, all gains, none
  losing a wired pin — and removed a silent defect where four ADS1115 ALERT pins had been tied
  together by one promoted one-pin net;
- the obligation-copy refusal, the deliberation audit's false catalogue findings and the
  family-mismatch refusal no longer occur at all in the candidate arm.

**Verdict against the preregistered gate: partly met.** The delivery target is met exactly; the
distinct-brief target is one short; the held-out supported compositions are not delivered; the
construction-coverage floor (18 of 34) has not been measured against the final revision. Nothing is
deployed on this evidence, and the held-out pair must be labelled as a limitation wherever the
product claims coverage.
