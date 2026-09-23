# Plan: >90% delivery yield on unseen, reasonable PCB briefs

## Decision

Replace “34/34 briefs eventually passed” as the product objective with:

> More than 90% of independently sampled, reasonable few-sentence briefs produce a requirement-complete, electrically reviewed, fabrication-ready design in one normal product run, without engineer intervention and within a fixed cost/time policy.

This is a better product goal, not an easier engineering target. A finite regression corpus can be memorized; unseen briefs expose missing circuit coverage, bad composition, ambiguous intent, and supply variability. Keep the original 34 as regression cases. Do not make clearing their last outliers a prerequisite for working on generalization.

The user subsequently authorized implementation and selected a **60-brief synthetic development baseline with a $5 aggregate allowance**, preserving production per-project/global caps. This does not authorize additional paid validation or independent-release campaigns. No production deployment is part of the measurement changes below.

## Evidence and limits

The supplied handoff reports 17/34 first-pass successes and 21/34 ever-passed coverage. These are different metrics; neither estimates unseen-brief yield. It identifies repeated support-network deficits despite three reconcile passes, port/identity mismatches, sourcing failures, and physical constraints discovered after routing.

The repository has useful foundations rather than a blank slate:

- `kicraft/eval/self_eval.py` selects built-in briefs and records frozen campaign manifests, source fingerprints, repeated runs, pipeline identity, and lifecycle evidence. Extend this machinery; do not create a second runner.
- Its summary currently counts `build_rc == 0` as `fab_ready`, separately from `semantic_clean` and `fab_safe`. That headline is not by itself the proposed product-success predicate.
- `kicraft/eval/design_acceptance.py` already supports evidence-backed obligations, but its corpus selection and release verifier are tied to the known corpus; `verify_release_campaigns` requires the 34 briefs across three campaigns. General brief acceptance needs a separate explicit mode, not a silent reinterpretation of that contract.
- The handoff describes a stronger historical pinned engine plus current-tree manufacturing/completion machinery. Do not switch engines on architectural preference alone.

Primary evidence: [previous handoff](design-yield-34-of-34-plan-2026-09-22.md), `kicraft/eval/self_eval.py`, and `kicraft/eval/design_acceptance.py`. Implementation locations below are starting points, not claims that every suggested capability already exists.

## 1. Define the denominator before optimizing

### What “reasonable” means

Start with ordinary prototype boards represented by expected users: sensor and MCU boards, interfaces/adapters, LED and display controllers, low-voltage power conversion, modest motor control, and analog signal conditioning. Include mixtures of these functions, not just isolated examples.

Freeze a capability envelope before sampling: manufacturing limits, voltage/current limits, layer/process constraints, supported packaging, and system complexity. Base those limits on the intended product and manufacturer, not on what the current implementation happens to pass. Publish the exact values in the campaign manifest before running.

Safety-critical, mains/high-energy, custom RF, and high-speed memory designs need an explicit product decision; they are not silently excluded after failing. Out-of-envelope requests are reported separately. In-envelope refusals, unsupported components, provider errors, budget exhaustion, and timeouts count as failures. A high refusal rate must never manufacture a high yield number.

An aggregate >90% claim is meaningful only for a declared distribution. It cannot mean a guarantee over every conceivable reasonable brief or >90% within every circuit family. Report family-level results so the average cannot conceal a systematically unsupported family.

### One normal product run

One submitted brief, using the same engine, parts sources, stages, recovery policy, and fabrication tail as production. Internal bounded recovery is allowed because users experience it as one job. Freeze its limits before measuring; restarting failed jobs until one passes is not allowed in the headline.

For this goal, run without human clarification: use documented safe defaults for nonessential omissions. If critical requirements cannot safely be inferred, stop honestly and count the brief as unsuccessful. Track clarification-assisted completion separately. Do not use the harness's first suggested answer if it discards or weakens an explicit requirement.

### Product-success predicate

Every successful run must have all of:

1. Explicit requested functions and numerical/mechanical constraints fulfilled, or an explicitly permitted substitution. No dropped features or quietly changed board requirements.
2. Artifact-backed electrical checks: supply compatibility, required support circuits, programming/boot path where relevant, and domain-specific operating limits. An LLM grade is advisory, not proof.
3. Passing ERC under the documented legitimate-exclusion policy, zero DRC errors, zero unconnected nets, and passing mechanical/antenna/connector checks.
4. A verified, internally consistent Gerber/drill package and associated BOM/assembly artifacts where the requested delivery includes assembly.
5. Completion without manual editing and within frozen per-job cost/time/recovery limits.

Missing evidence means not demonstrated, not passed. Freeze expected obligations independently before each evaluation, and keep evaluator-only obligations out of the design input. Production still derives its own typed requirements from the brief. Check both against the original brief; a generator and checker agreeing on the same misunderstanding is not success.

This is a design-release standard, not proof of working hardware. Physically build and bench-check representative reusable circuits as a separate confidence program; do not describe DRC-clean boards as experimentally functional.

## 2. Build an evaluation that measures generalization

### Three roles, not one ever-growing training set

- **Known regressions:** the existing 34 and saved failure workspaces. Cheap targeted replay and periodic full runs detect regressions. Their ever-passed count is diagnostic only.
- **Development distribution:** initially 60 new briefs from independent authors or consented, de-identified user requests, deliberately spanning circuit families, composition, part choice, density, outlines, and ambiguity. Run one frozen baseline. These become visible engineering material.
- **Sealed release sample:** 200 fresh, independent briefs sampled only after the candidate and evaluation policy are frozen. Draw from the declared target mix, independently of known results. No tuning on this sample.

Use a separate fresh validation batch during development when the visible set appears strong. Once inspected, a validation or release set is development material, not an unseen holdout for the next candidate.

Do not generate hundreds of paraphrases of the 34 and call them independent. Include genuinely new functional combinations and parameter ranges: channel count, voltage/current, connector arrangement, programming interface, board shape and available area, and interchangeable versus explicitly named parts. Correlated paraphrases are robustness probes, not additional independent evidence.

Use natural short briefs, not prompts written to match internal recipe names. Review feasibility and expected obligations before execution without reference to the generator's result. If synthetic briefs are needed initially, report synthetic yield as such; do not equate it to actual user traffic.

### Runner changes

Extend the existing runner and acceptance machinery with an immutable external brief manifest containing brief text/hash, provenance, eligibility, family tags, and separately stored evaluation obligations. Preserve source/config/model/pipeline identity, budget policy, stock timestamps, artifact hashes, and failure evidence. Resume must recover the same external manifest rather than intersect it with the built-in corpus.

Add the conjunctive `product_success` metric without conflating it with build success, semantic cleanliness, or design commitment. Record the actual engine-owned reconcile count rather than a counter that can stay zero across native-owned recovery. Give every unsuccessful submission a first blocking failure category, while retaining downstream evidence.

### Release acceptance

Target **at least 188/200 successful independent briefs**, under one frozen production-equivalent configuration. For a fixed-size binomial experiment this makes the exact one-sided 95% lower confidence limit exceed 90%. Merely observing 181/200 does not provide that evidence.

The threshold was checked using exact binomial tails at p=0.90: P(X >= 187) = 0.056562; P(X >= 188) = 0.032047. The calculation assumes independently sampled briefs from the declared distribution; it does not justify counting correlated variants or repeated attempts as independent.

Do not peek and stop early, drop failed runs, or combine changing engine versions. Repeated certification attempts need a predeclared multiple-testing/confidence policy; otherwise report each campaign without claiming that repeated attempts preserve the original confidence guarantee. After launch, report all eligible traffic and periodically audit a fresh sample for distribution drift.

## 3. Change circuit generation from invention to constrained composition

The highest-leverage hypothesis from the handoff is that models are being asked to repeatedly rediscover mandatory circuit details. Prove or revise this priority using the new baseline, rather than extrapolating a promised yield increase from six known cases.

### A. Own support circuitry as reviewed data

Extend existing reviewed identities, recipes, and lowering; do not build a competing recipe framework. A reviewed functional block should own:

- Exact symbol/footprint/pin identity and electrical port types.
- Supply range, loading limits, parameter-dependent component values and ratings.
- Required decoupling, bias, boot/reset, clocks, protection, charge pumps, and programming connections as applicable.
- Placement relationships such as decoupling proximity and antenna keepouts.
- Supported parameter ranges and explicit rejection outside them.
- Datasheet provenance and artifact-backed reference examples.

The model chooses and parameterizes blocks; deterministic machinery instantiates their required parts and connections. Completion must be idempotent and must not silently replace user-selected parts or duplicate existing support networks.

Start with high-frequency missing capabilities in the baseline. The handoff's MCU support, LED bypass, motor-driver charge pumps, and op-amp biasing are strong initial candidates, not an exhaustive catalog. A library of 34 complete reference boards will not meet this goal; parameterized blocks and their composition can.

**Acceptance:** a failure reproduction is fixed; a new parameterization and a new composition work through the real pipeline; an out-of-range case fails honestly. Test the actual electrical behavior/invariant, not a hardcoded brief name.

### B. Make block interfaces electrically typed

Resolve semantic ports to reviewed physical pins for exact identities. Carry voltage domain, direction, bus role, multiplicity, current demand, and required series/termination topology into composition. Verify compatibility before accepting a connection.

Repair from structured terminal/net evidence, not prose alone. Do not turn the old handoff's series-part suggestion into a universal “all resistors/diodes must use distinct nets” rule: enforce intended electrical roles and valid exceptions instead of a component-name heuristic.

Sourcing alternatives must preserve the required function and reviewed topology. An explicit part request remains a constraint unless substitution was permitted. Apply the requested purchasing/assembly stock policy explicitly rather than quietly changing it to get a pass.

**Acceptance:** previously failing interface cases and unseen compatible combinations pass; wrong voltage domains, missing required terminals, and genuinely bypassed series networks fail before PCB generation.

### C. Make physical feasibility a generation constraint

Move existing edge-mating, antenna, keepout, grouping, and outline requirements into candidate generation/scoring before expensive routing. Keep terminal checks as independent backstops. Estimate area, escape difficulty, current-path requirements, and routing feasibility before committing to an architecture.

If outline and layer count are unconstrained, choose conservative documented defaults. If specified, honor them; do not enlarge the board, add layers, or move a required connector to pass silently.

**Acceptance:** frozen failing layouts pass the real build tail after the fix, and unseen connector/outline combinations demonstrate that the change is not reference-specific.

## 4. Make recovery causal and bounded

Preserve existing production limits initially. Each repair must name a concrete violated constraint and change the relevant state. Detect repeated identical deficits and stop instead of spending the same reconcile budget on identical requests.

Use deterministic completion for reviewed missing support parts; structured wiring repair for topology mistakes; placement reconsideration for geometric failures; reviewed alternatives for genuine sourcing failures. Recovery is not permission to remove requirements.

Do not raise attempt counts or switch models until telemetry shows that the relevant failure is actually stochastic and the change improves end-to-end yield within the same cost/time policy. Measure recovery yield and cost separately from no-repair yield, while counting bounded internal recovery as part of the single product run.

## 5. Keep one measured production path

Use the handoff's current-tree completion around the historically stronger pinned engine as the initial low-risk landing strategy, conditional on confirming that it is still the effective deployed path when implementation starts. Keep the pinned checkout unchanged. Reuse canonical reviewed data and lowering; avoid two independent electrical truth sources.

This is not a commitment to keep a two-tree system permanently. A move to the current engine is a separate experiment on the same unseen validation distribution and budget policy. Migrate only on evidence of equal or better product yield and no hidden requirement/safety regressions. Consolidate into one engine when that condition is met; do not run an architecture rewrite in front of the reliability work.

## 6. Execution order and exit criteria

| Milestone | Work | Exit evidence |
|---|---|---|
| M0: honest measurement | Freeze envelope, distribution and job policy; external manifests; conjunctive success; actual retry accounting | A new brief enters the normal runner and reaches an artifact-backed verdict; incomplete requirements cannot count as success |
| M1: unseen baseline | Run 60 development briefs under one config, preserving every result and ledger delta | Per-family yield, first-blocker Pareto, missing-capability inventory, cost/latency distribution |
| M2: highest-frequency causes | Implement support ownership, typed composition and physical feasibility in the order the baseline warrants | Each fix has a reproduction, saved-workspace replay where applicable, and successful unseen variations; terminal gates unchanged |
| M3: fresh validation | Run a fresh development-validation sample; test common ambiguous and composite briefs | High yield is not confined to the visible 60 or original 34; no unexplained family-wide failures; cost/time within policy |
| M4: release evidence | Freeze candidate; independently sample and run the 200-brief release experiment | At least 188/200 product successes, complete failure accounting and artifact audit, valid sampling/confidence protocol |
| M5: sustain | Monitor real eligible requests and periodically sample fresh audits | Yield, requirements fidelity and cost remain acceptable as request mix, models and stock change |

Within M2, prioritize measured failure frequency × expected generalization benefit / implementation risk. Fix a cheap genuine validator false positive, such as the mechanical-only sheet case, when encountered, but do not let rare known-corpus outliers displace the dominant unseen failure classes. Do not forecast cumulative percentage gains by adding overlapping failure categories.

Verification sequence for a source change: targeted reproduction; deterministic reference/replay proof; real-provider targeted unseen examples; then broader campaign after edits settle. Use the existing frozen-tail verification workflow for place/route changes. Saved-state replay alone cannot prove improved brief-to-design behavior.

Before each paid campaign, estimate total cost from measured attempts including failures, set authorized campaign/day limits, and reserve production capacity. A cap-aborted campaign is incomplete evidence, not a smaller successful denominator. Deploy through the canonical production script and record the effective engine/config; do not mutate services or pins during a measurement.

## Immediate next action

Implement M0, then run M1 with an explicit spend allowance. Do not begin by rerunning only the remaining 13 or adding more reconcile attempts. Those actions improve knowledge of the old corpus, not the new objective.

The core bet is concrete: **broad reviewed circuit blocks + checked composition + early physical feasibility**, judged on fresh user-like briefs. Keep or replace that bet according to the unseen baseline, not according to progress toward 34/34.

## M1 baseline result (60 synthetic briefs, one frozen configuration)

`logs/self_eval/m1_general_development_2026-09-22_run2`: `campaign_valid=True`, `campaign_complete=True`, `source_unchanged=True`, 04:37–11:21, spend **$0.708572**.

**Product success: 0/60. No brief reached the build stage** — every run died in design, so raw fabrication yield is 0 by construction, not by DRC.

First-blocker Pareto: `invalid_json` 39, `commit_rejected` 13, `truncated_json` 8. First failing stage: `bom` 46, `functional_spec` 11, `architecture` 3. Retry classes across all runs: `no JSON in reply` 83, slot-validation 66, `isolated block(s) with no connections` 18, self-loop 4.

Two root causes were found and fixed in the pinned engine (user-approved amendment):

1. **The `bom` tool loop never converged.** `bom` is the only stage with tools. Its thrash cutoff required the *same* call signature three times, so on unseen briefs (where every lookup differs) `force_final` never fired and `tool_choice` stayed `"auto"` through the final round. The loop then returned prose — "Let me search…" — and the stage failed with `no JSON in reply`. Evidence: median reply at failure was ~4 KB of prose, far below the 16384-token round cap, so this was never truncation. Fix: hard-stop tools on the final round (`final_response = force_final or last_round`), which is exactly what the current checkout already does — the pinned revision simply predates it.
2. **`functional_spec` could not represent a mechanical block.** `BlockCategory` had no `mechanical` value, and the sanity gate rejected any block with no connection, so a brief asking for mounting holes failed the stage twice over (18 isolated + 12 category rejections). Fix: add the category, exempt mechanical blocks from the isolation requirement, and document the value in the stage prompt. `FunctionalBlock.category` has no other consumers, so the value is inert downstream; pin-less `Mechanical:MountingHole*` BOM parts are already supported by the symbol check and the composer.

Verification: re-driving the `bom` stage on the six briefs that previously failed there now commits **6/6** (previously 0/46 across the baseline); retries now surface real, self-resolving gates (`9.33` spec-named parts, `9.25` capacitor polarity, unresolved footprints/symbols). Re-driving the annunciator's `functional_spec` commits with `MOUNTING_HOLES` category `mechanical`.

Still unproven at the time of writing: whether a fully designed board now passes the downstream gates and product acceptance. The mounting-hole caveat originally recorded here is withdrawn — see "Mounting holes are already realized" below. The next baseline measures the post-fix distribution.

### JSON reliability (the dominant class) — third fix

The in-loop hard stop alone was insufficient at scale: a re-run still failed 4/4 briefs at `bom` with `invalid_json`/`truncated_json`. Diagnosis showed two distinct causes, both now addressed:

3. **The model wrapped the slot** — a retry reported `BOM parts Field required [input_value={'bom_slot': {...}}]`, i.e. it invented a key named after the slot instead of emitting the slot object at the top level. The three places that demand a final answer (the in-loop final round, the post-loop forced final, and the serialization retry) now state the exact shape and forbid a wrapper key or prose.
4. **Prose instead of JSON.** The single tool-free serialization recovery — the last line of defence, since its budget is shared across the whole drive — now requests provider JSON mode (`response_format={"type": "json_object"}`), falling back to an unconstrained retry if the provider rejects the field, so the recovery can never be lost to an unsupported parameter. The maintained current checkout already enforces structured output this way (`stage_runtime.py` response contracts); the pinned revision had no `response_format` support at all.

5. **Self-inflicted crash in the recovery template.** Adding the JSON-shape examples to `_SERIALIZATION_RETRY_MSG` introduced unescaped braces (`{"parts": ...}`) into a string consumed by `str.format`, so `.format` raised on every parse failure and the legacy process died before the recovery call — `legacy_protocol_error` with no result packet. The aborted fourth run failed 6/6 briefs this way. Evidence: the `bom` retry event is emitted immediately before `.format` runs, and the ledger showed zero `serialization` calls; the earlier run had clean packets (`truncated_json`) on the same path. Fixed by escaping the braces, with a legacy regression test (`tests/test_stage_driver_retry.py`) that renders the template and asserts the literal JSON examples survive.

6. **Unbounded answerless stream stalls.** The `bom` stage's large tool-loop prompts (~44k input tokens) intermittently left a provider stream open with no answer: 19 minutes with no content, twice. `requests`' read timeout does not bound this because SSE keepalives keep satisfying the read, and the existing reasoning-loop breaker only fires when deltas arrive. Production design runs use this same client, so an unbounded stall hangs a real design job. Fix: bound the phase before any answer content or tool-call arguments exist (`2 × request_timeout_s`, min 240 s) and raise `Timeout`, which the existing stream-retry and transport-failure paths already handle. Once content streams, generation is bounded by `max_tokens` instead, so long legitimate answers are unaffected. Two legacy tests pin both directions.

Verification: the three briefs that failed at `bom` in the aborted re-run now commit **3/3**, with replies of the form `{"parts":[...]}` — no wrapper, no prose — and remaining retries are real, self-resolving gates (`9.33` spec-named parts, unresolved footprints/symbols).

Deliberately NOT changed: the serialization recovery budget stays at its documented one-per-drive invariant, and the strict guard's admission lock still spans the call. Evidence no longer demands widening the first, and an attempt to narrow the second correctly failed the race regression that pins the one-of-two admission property — so it was reverted rather than forced through.

### Mounting holes are already realized (no fix needed)

The earlier caveat that a requested hole count was not threaded into the layout is **withdrawn**. Once `functional_spec` can express a `mechanical` block, the BOM stage emits the matching `Mechanical:MountingHole*` parts on its own, and the engine already supports those pin-less parts. Verified from committed BOMs: the greenhouse brief (4 requested) produced a `MOUNTING_HOLES` block with `count: 4` and **4** mounting-hole parts; the annunciator produced **4** parts; the irrigation brief, which requests none, produced **zero**. No layout-seam change is required.

### Cost and latency profile (measured, all campaigns)

1000 campaign calls consumed **17.6M input tokens** (median 18.6k, p90 34.8k, max 48.3k) and 767k output tokens. The `bom` tool loop accounts for 647 of those calls at median 23.7k input tokens, because each of its 6 rounds re-sends the whole conversation, including 16 tool results truncated at 4000 characters each. This is the cost/latency driver rather than a yield driver, so it is deferred until yield is measured; it is the leading candidate for bounding next.

### The `bom` stage is the whole bottleneck — recovery budget widened

Run7 (all fixes except this one, 7.4 h, $0.841317, `campaign_valid=True`) still scored **0/60**, but the shape of the failure moved decisively: `functional_spec` failures fell 11 → 2 and `architecture` 3 → 1, while **`bom` failed 57/60**. The remaining failures are `invalid_json` 48, `truncated_json` 10, `commit_rejected` 2 — i.e. one stage is now the entire bottleneck.

Diagnosis (per-call evidence): the model *does* produce JSON — attempts returned ~8 KB of `{"parts":[...]}` — and the surviving failures are (a) prose replies with no JSON and (b) schema-shape errors (`BOM parts.N.ref Field required`, the model inventing `designator`/`id`). The decisive defect was structural: the tool-free serialization recovery is the only path that re-asks for JSON without tools, and its budget was **shared across the whole drive**, so after the first parse failure attempts 2–5 had no recovery at all. Run7 logged 101 "no JSON in reply" retries while that single recovery was already spent.

The failure is also demonstrably stochastic — the same brief both committed `bom` in one run (3 attempts) and failed in another with identical code and input.

Fix: `Settings.serialization_retries` 1 → 3 (documented in `config.py`). The mechanism is unchanged — still tool-free, still the fixed cap, never a repeated tool loop, never a dynamically grown cap — only the number of chances per drive. This meets the plan's bar for widening recovery ("do not raise attempt counts until telemetry shows the relevant failure is actually stochastic").

Measured effect: re-driving run7's `bom`-failed briefs with the widened budget commits **3/4** (previously 0/57 across the baseline). Deliberately NOT done: normalising the model's invented `designator`/`id` keys into `ref`/`symbol`, which would paper over a prompt-compliance failure rather than fix it.

### Provider-enforced JSON on the tool-free answer paths

Run8 (widened recovery, 1 h 14 m before being stopped) confirmed the recovery now *runs* but still failed 3/3 briefs at `bom`. Per-call evidence showed why: the model emits tiny prose (3–134 chars), or runs away (one recovery produced **97,479 characters**, truncated at the 32,768-token cap — which also explains the multi-minute stalls), while the tool-loop context grows 19.9k → 24.7k → 31.1k → 35.5k → 40.6k tokens per attempt, reaching **71k input tokens**.

The provider was verified to accept both `response_format: {"type": "json_schema", strict: true}` and `{"type": "json_object"}` (HTTP 200, and the strict schema returned exactly the requested shape). This is the maintained current checkout's own mechanism (`stage_runtime.py` response contracts); the pinned revision had no `response_format` support. An earlier attempt to use it was reverted after a 19-minute stall — that attribution was **wrong**: stalls occur with and without the parameter (run6/run7 stalled too), so it was re-applied.

Applied only where it cannot conflict with tools: the post-loop forced final (with the inert `tools` key dropped) and the tool-free serialization recovery, the latter falling back to an unconstrained retry if the provider rejects the parameter so the recovery can never be lost to an unsupported field. Measured on briefs that were 0/57: 3/4 then 3/3 now commit `bom`.

## Implementation status

The external-corpus path now extends `kicraft.eval.self_eval`; the existing 34-brief mode and its release contract remain separate.

- `external_briefs.py` validates immutable brief/obligation bundles, finite policy bounds, provenance, eligibility, unique identifiers, and the full denominator.
- `corpora/general-development-2026-09-22/` contains 60 synthetic briefs across six equally represented families and 297 evaluator-only obligations. Corpus hash: `sha256:5520cf9926136d0eb28e4ea3679c30b3b1c27c1d7cddbd0df1f9088ac327494c`.
- The frozen envelope is non-safety-critical prototype electronics, at most 24 V DC, 3 A board current, 100 × 100 mm, and two or four layers; modules are permitted. The per-job policy is $0.60, 3,600 seconds, 12 park rounds, and a 2,400-second build timeout.
- External runs retain the original brief, never apply known-corpus rewrites, and do not silently answer blocking questions. Native-owned BOM reconcile passes and harness-owned provider retries are recorded separately.
- Resume restores the saved bundle, preserves failed/interrupted attempts, and checks source/native-engine/configuration identity. Successful records bind the exact audit digest, frozen obligations/policy, successful verdict, and selection-relevant artifact inventory; adding a conflicting board or replacing a passing audit cannot preserve success. Missing attempts and budget refusals remain in the denominator.
- `product_acceptance.py` requires common artifact-backed obligations plus the independently frozen brief obligations. It reconciles canonical BOM terminals against actual delivered footprint identities, pads, and net groups before proving connectivity/programming. It runs a fresh JSON DRC on the canonical delivered board, including excluded violations, and hashes the board, rule/project files, schematics, reports, state, and fabrication package.
- Fabrication export uses a fresh staging directory, excluding stale files. A companion receipt binds the PCB, archive, and every member by SHA-256. Certification checks Gerber-job layer/outline coverage and nonempty Gerber/drill outputs, rather than trusting filenames or `build_rc == 0`.
- Evaluation-only native budget admission reserves a conservative bounded cost before **each actual HTTP POST**, including native header retries, under a shared ledger lock. Discarded/lost streams and missing provider-cost receipts retain durable uncertain exposure; a later successful retry settles only its own reservation. Unbounded/nonstandard request shapes are refused. Campaign admission counts exposure against the approved allowance. Production behavior outside this opt-in mode is unchanged.
- `--audit-campaign` performs an offline denominator/artifact audit without provider calls.

Validate the corpus without spending:

```bash
.venv/bin/python -m kicraft.eval.self_eval \
  --brief-manifest kicraft/eval/corpora/general-development-2026-09-22/briefs.json \
  --obligations kicraft/eval/corpora/general-development-2026-09-22/obligations.json \
  --validate-manifest
```

The approved development baseline command, with a fresh output directory:

```bash
.venv/bin/python -u -m kicraft.eval.self_eval \
  --brief-manifest kicraft/eval/corpora/general-development-2026-09-22/briefs.json \
  --obligations kicraft/eval/corpora/general-development-2026-09-22/obligations.json \
  --campaign-budget-usd 5 --parallel 3 --build-slots 1 --no-judge \
  --out logs/self_eval/general_development_baseline
```

Current limitations are explicit: many new function-specific obligations lack generic extractors and therefore remain **unverified**, never passed by inference. This baseline measures both generator failures and evidence-coverage gaps; raw fabrication yield must be reported separately from demonstrated product success. Synthetic development results cannot certify independent-user yield. Fresh validation and a sealed independent 200-brief release sample still require their own inputs and spend authorization.

Runtime verification exercised KiCad 9.0.9 on a copied historical routed PCB: fresh export and independent DRC passed; a stale drill file was excluded, and changing the PCB invalidated the fabrication receipt. This is software/artifact verification, not a hardware bench test.

The initial calibration attempt at `logs/self_eval/general_development_baseline` was stopped after 2m57s when final review found header-retry exposure release and incomplete connection/audit proofs. Three briefs had started; none completed. Its immutable manifest, partial workspaces, and `interruption.json` are preserved. The observed ledger delta was $0.002813108 and retained active exposure was $0.0376448. These are included in the original $5 allowance; they are not discarded or described as a completed baseline. The corrected campaign must retain the original absolute ledger ceiling of $138.80079697348.

After these corrections, 128 targeted tests passed. A real `pcbnew` terminal-graph smoke passed with all expected nets present and withheld the connection gate after removing one required pad net. A fault-injected transport exercised the pinned native `_open_stream` retry implementation without provider spend: two POSTs left reservation states `uncertain` and `settled`, respectively; an unbounded multiple-completion request was refused before dispatch.

### Measurement-defect found by the second attempt (self-inflicted, fixed)

The second attempt (`logs/self_eval/m1_general_development_2026-09-22`, stopped after 9m16s) failed all three briefs with `legacy_protocol_error` after `architecture`. This was **not** a generator defect: the evaluation-only strict-budget adapter had grown a whitelist of admissible request fields, and the pinned tool-loop sends `parallel_tool_calls` (`KiCraft-legacy/kicraft/server/client.py:390`). The whitelist rejected the first tool-using stage (`bom`), and `stage_driver` deliberately lets `BudgetExceeded` escape, so the runner died before printing its result packet and the harness recorded a protocol error instead of the true cause.

Two fixes, both in the evaluation adapter only:

- The field whitelist is gone. The serialized request already bounds prompt cost for any provider field; only genuine billed-completion multipliers (`n`, `best_of`) are refused, so a legitimate provider field can no longer silently kill a run.
- `main()` now catches `BudgetExceeded`/`KillSwitchEngaged` and returns a structured failed result stamped `budget_exceeded`/`kill_switch`, so a guard refusal is reported as the first blocking category rather than as a broken protocol.

Verification: the strict guard now admits the pinned tool-loop body and still refuses `n=3` (probe regression). Re-driving the failed workspace's `bom` stage against a copy reproduced the fix at the real seam — the stage that previously died with zero calls committed with 45 parts over 13 model calls.

### Where M2 fixes can actually land (constraint verified before implementation)

The baseline's designs are built by the **pinned legacy engine**, and the plan forbids changing that checkout. Verified: legacy-designed `state.json` carries no `recipe_selections`, and its BOM parts have no recipe fields. The current checkout's reviewed recipe registry (`kicraft/design/recipes/`) therefore does **not** participate in baseline designs; nor does `server/stage_contracts._expand_bom_groups`, which is gated on current-tree design ownership. Legacy `synthesize` consumes `state.bom`/`state.architecture` from the build snapshot and runs its own determinism passes.

Consequently the current-tree recipe registry cannot change a legacy-run design; the only current-tree surface is a completion/repair pass over `build_state.json` before `native_build_runner._synthesize_native` delegates to legacy synthesis. Existing deterministic completion is narrow: the legacy `apply_deterministic_bom_adds` provisions only regex-parseable R/C/L passives named in the model's own deficit note.

The plan's original "keep the pinned checkout unchanged" rule therefore made every design-stage failure unfixable, because the M1 baseline's dominant failures are all design-stage. The user approved amending that rule to allow **minimal, evidence-backed fixes in the pinned tree**. Scope discipline for those fixes: each must be tied to a measured failure class, must not change the pipeline's architecture, must be mirrored in the current checkout where the two diverge (so the trees do not become two truth sources), and must be re-measured. The two M1 fixes above follow that rule; the tool-loop fix was in fact a port of code the current checkout already had.
