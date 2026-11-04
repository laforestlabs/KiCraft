# Autonomous surprise-me product-yield improvement

Status: execution plan for the next session. The user requests autonomous work for **at least eight hours without intervention**. Do not ask for decisions this plan, the repository, or existing evidence can resolve.

## 1. Objective and definition of progress

Increase the number of unchanged, randomly generated **surprise-me briefs that reach a complete, fabrication-ready delivered product**. The unit of progress is another original brief satisfied by its actual final artifacts—not another passing test, stage commit, constructed BOM, route-only replay, or repaired report.

Maintain two separate outcomes:

1. **Physical fabrication pass:** fresh routed board, required references present, accepted selected leaves under existing accounting, final verification passing against the actual board/rules, and fresh matching fabrication exports.
2. **Complete-product pass:** physical pass **plus every original brief obligation** supported by artifact-bound evidence through the existing product acceptance evaluator. Missing evidence remains unverified.

Report physical-only successes separately. Do not imply generated/executed firmware, measured analog performance, thermal qualification, aggregate GPIO capability, or manufactured-hardware certification from static/native construction evidence.

This plan changes the next work priority from individual fixture repairs to **population-level yield improvement**. It supersedes the previous handoff's ordering of remaining gaps, not its safety constraints or retained evidence. An unrelated red RP2040 diagnostic must not monopolize the session or become a prerequisite for investigating broad architecture/BOM blockers.

## 2. Autonomous operating contract

### Minimum duration

- At startup record `started_at_utc` and `not_before_finish_utc = started_at_utc + 8 hours`; retain an elapsed-time reference. Do not reset the clock after compaction or recovery.
- Continue meaningful investigation, implementation, runtime verification, artifact auditing and regression work until that time. Builds and other necessary external waits count toward elapsed wall time.
- Do not stop after a plan, a first fix, a green unit suite, a stage commit, or a phase boundary. When one useful task finishes, take the next evidence-ranked task.
- Do not manufacture activity, sleep to fill time, repeatedly run unchanged experiments, or spend merely to occupy eight hours. If a goal is reached early, use the remaining time for independent obligation audits, affected-family controls, plausible consumer-visible regression cases and verified generalization.
- Safety and genuine external unavailability override the duration requirement. If all safe actionable work is exhausted, record exactly what is blocked and what was tried; do not invent results or bypass controls. Infrastructure/session survival cannot be guaranteed by a prompt. Preserve enough durable state to resume the same work rather than restart it.
- At the eight-hour boundary, finish an in-flight bounded verification/audit rather than abandon it. Do not start another long paid/build campaign merely to extend the session. Deliver a complete evidence-based report.

### No-intervention decisions

Choose conservative defaults using repository conventions. Handle normal technical choices independently. If a material decision requires authority not granted here, do not pause awaiting user input: leave that action blocked, explain the risk in the journal, and continue independent authorized work. Never treat user silence as approval to deploy, alter secrets, change protected inputs, increase spending, or relax product acceptance.

### Durable checkpoints

Use one new, nonexisting UTC-suffixed directory under the development root below. Keep a small `session_progress.json` or Markdown journal and evidence index there; this is bookkeeping, **not a new evaluation runner/framework**.

Record after each significant experiment or source change, and roughly every 30–60 minutes during longer work:

- Start/minimum-finish times, current hypothesis and prioritized remaining work.
- Candidate snapshot/fingerprint, source-owning changes and verification commands/results.
- Original brief IDs/hashes, immutable input hashes, budgets, profile and evaluator version.
- Active process/job IDs, output paths, completed trials and occupied provider/build slots.
- All failures, costs, blockers and the next concrete source operation to investigate.

After recovery, read this journal and inspect existing jobs before launching anything. Never duplicate an active provider/build job. Use completion notifications or blocking waits, not polling loops.

## 3. Production, custody, concurrency and spend

The checkout is the production host. Follow the loaded `AGENTS.md` and matching skills (`self-eval`, `kicraft-investigate`, and `verify` before their corresponding work).

- No deployment, service restart, `.env` edit, shared-cap increase, production queue mutation, router replacement, clearance reduction, unauthorized via-in-pad, acceptance-policy weakening or hand-fixing individual boards to count them as successes.
- Preserve original briefs, manifests, projects, states, requirement/consent records, seeds, rules and evidence. Never replay in place, overwrite retained candidates, substitute easier briefs, rewrite exact requested identities, or exclude failures from denominators.
- Inspect campaign evidence **only** beneath:
  `logs/self_eval/surprise_completion/20261010T204651Z/development/`.
  Do not broadly search the campaign root. Do not inspect sealed/quarantined holdouts or `ReviewPowerRatingProof`.
- **The sealed release set stays closed throughout this session.** No release run is authorized by reaching a promising selected diagnostic.
- At most **one provider job and one physical-build job**. Inspect occupancy before acquiring either slot; do not terminate unrelated production jobs. Offline research/edits may proceed while a frozen candidate runs. Delegate independent substantial code slices, with one integration owner; do not run sibling builds/provider calls.
- Use `/home/kicraft/KiCraft/.venv/bin/python`, native KiCad, and the production-pinned KRT router. Keep the current production model/provider profile and lower existing caps.
- Provider ceiling per end-to-end brief/run: **at most $0.60**, including that run's internal retries; respect any lower cap. Targeted repeated diagnostics each need their own retained identity and cost; never combine them into a best-of-N success.
- Conservative session provider ceiling: **$12 total**, also bounded by the lower remaining production daily/lifetime/project allowances. Record spend from existing guards/reports. Do not reset guards, raise limits or wait for a budget reset to evade this ceiling. If insufficient for a proposed cohort, do not launch it: continue safe offline/native/source work and label the population measurement blocked.
- Keep documented execution limits unchanged: `max_duration_s=3600`, `build_timeout_s=1800`, `max_park_rounds=12`, and `KICRAFT_BUILD_MAX_WALL_S=1620`.
- Run provider/physical measurements from unique frozen source snapshots, with snapshot `cwd` and `PYTHONPATH`. Symlink `.venv` and `.env`; never copy secrets. Assert original-before == original-after == frozen source fingerprint at freeze. Hash original input custody before/after each replay. Stop further jobs on a custody mismatch; do not restore over outside changes.
- Mutable external catalog data is not covered by a source fingerprint. Record the catalog snapshot/version/hash used where available. Do not stop production catalog maintenance or pretend source freeze also freezes the catalog. Disclose catalog changes that invalidate a paired comparison.

## 4. Starting evidence: facts, not current yield

Read `docs/handoff-surprise-completion-2026-10-12.md`, especially its November continuation and qualification results. Do not rerun old failures merely to confirm already-observed results.

Retained population audit:
`development/regrade_policy_v2/regrade_summary_verified.json`.
It records a valid **historical 0/60 complete-product result**, not today's code's yield.

| Historical outcome | Briefs |
| --- | ---: |
| Architecture failure | 27 |
| BOM failure | 19 |
| Wiring failure | 1 |
| Reached a build | 13 |

Nine recorded build rc0. Forty-seven never reached a build. The largest retained terminal diagnostic bucket is `declared_interface_unrealized` (18 briefs); other substantial buckets include `missing_recipe_requirement` (7) and `unsupported_lowerer_contract` (6). These are **historical labels, not established current bugs**. Unverified capability counts are not counts of missing circuit implementations.

Latest checkpoint:
`development/qualification_resume_20261102T021800Z/qualification_summary.json`.
Reviewed runtime snapshot was `/tmp/kicraft-power-candidate-v4f8erou`, fingerprint
`sha256:ff86480830f1dc779f530688f2853f5442dcb0f769d01889724f66c2ba959f56`.
The checkout subsequently received documentation-only updates. Inspect current source rather than assuming it is unchanged.

- 1,064 tests in 15 isolated suites; 186 additional tests; native switch/RGB/edge proofs passed.
- Reviewed logger architecture: 2/3 commits, with writable storage, four exact TEMT6000 owners/ADC paths, buttons, Qwiic and WCH-Link conductors in deterministic construction. Not routed logger products.
- Reviewed battery architecture: 0/3; pin-capability and offline-catalog refusals. Do not suppress them.
- ESP32 fixture 9101: 3/3 physically built/custody-verified, **complete-product unverified**. Finish interpreting original obligations and sourceability rather than assuming rc0 is success.
- RP2040 fixture 9102: 0/3 clean, selected leaf has three opens and parent ten opens. Retain it as a routing regression witness; do not automatically repeat six long physical trials or make it the session's central task.
- Native castellation primitive now has the proper KiCad property, but old replay seeds retain old native pad properties. This is separate from RP2040's routing opens.

Existing original fixtures are `/home/kicraft/.kicraft/projects/44/9101` and `/home/kicraft/.kicraft/projects/44/9102`. Treat them as immutable evidence. Some throwaway `/tmp` drivers were archived and removed: reusable driver evidence is under the checkpoint's `native_witnesses/drivers/`. Read any remaining `/tmp` script before execution. Do not assume old temporary workspaces still exist.

## 5. Execution order: diagnose breadth, repair source, prove products

### Phase A — establish the live, source-owned failure Pareto

1. Validate the retained development manifest, brief identities, distribution, evaluator contracts and custody using the existing evaluation machinery. Do not generate replacement briefs or build a second harness.
2. Group the 60 originals by **first blocking stage and concrete diagnostic**, counting unique affected briefs. Start with `declared_interface_unrealized`; then investigate the next largest still-live bucket.
3. Inspect bounded retained inputs/candidates, current source and deterministic construction. Use production triage readers; do not recreate their artifact interpretation in ad hoc parsers.
4. Classify each candidate issue as:
   - Real missing reusable circuit/lowerer/recipe implementation.
   - Incorrect observer or producer/consumer contract mismatch.
   - Unavailable exact part/sourceability inconsistency.
   - Correct rejection of an infeasible provider composition.
   - Stale/already-fixed historical defect.
5. Reproduce an unverified cause through the cheapest relevant real path. Use offline construction/native evidence for deterministic code; a capped stage replay only where live provider behavior is necessary. Retain all attempts.
6. Rank **concrete current defects** by affected original briefs, generality, likelihood of unlocking a complete product and verification cost. Identify the first source-owning operation and the gate that should detect it. A diagnostic frequency alone is not a fix specification.

Deliverable: a small ranked live-blocker table with affected IDs, source ownership, reproduction and expected product-level consequence. Move on if the historical largest bucket is already fixed; do not weaken its gate to recreate an apparent win.

### Phase B — implement the highest-leverage reusable repair

Work on one coherent source-owned repair at a time. Prefer existing compiler/lowerer/recipe/allocator and library patterns. Do not launch another orchestration framework, add general retries/telemetry, or invent a second interface convention.

Potential work areas must be selected by Phase A evidence, not assumed:

- Bind declared interfaces to the actual connector contacts, MCU capabilities and powered circuit that implement them.
- Compose only concurrently feasible pin/power assignments; preserve exact part and voltage limits.
- Bring recipe/core defaults and sourceability evidence into agreement without silently substituting protected identities.
- Implement a genuinely missing reviewed reusable circuit instead of adding an alias that bypasses topology proof.

Before editing, reproduce the specific defect where practical. Add deterministic behavior regressions for plausible user-visible failure modes, including a broken-circuit negative. Do not add source-text, wording or mock-forwarding tests. Run focused checks and **exercise the changed path**: tests alone do not prove the behavior. For electrical/geometry changes, use saved/reloaded native KiCad evidence where appropriate. Update affected callers/docs and remove throwaway scaffolding after preserving proof.

Do not hand-edit retained boards/BOMs/states to make a fixture pass. A model-output defect is not permission to relax the correct guard.

### Phase C — prove end-to-end wins and detect regressions

For the first candidate, preregister a small engineering verification set **before new outcome inspection**:

- Three affected originals from the chosen live bucket, or all if fewer exist.
- Three nonaffected controls spanning different relevant families/stages.
- Select deterministically from the original manifest order within those strata; record why each is included. Do not choose the easiest/winners after results arrive.

This is a deliberately targeted diagnostic set, **not an estimate of random-brief yield**. Keep membership stable across the comparison. Add further affected originals only as an explicitly named additional set; retain all earlier failures.

Freeze source before runtime measurements. Use `kicraft.eval.self_eval` and `product_acceptance.evaluate_product` with the original manifest/contracts and existing noninteractive defaults. Drive each original from its unchanged brief through all design stages and the deterministic build, not just its previously failing stage. Any frozen-state stage/route replay remains a diagnostic, not an end-to-end product attempt.

For each attempt retain:

- Original brief/hash and candidate/evaluator identity.
- First blocking stage/source and all failures/retries/costs.
- Actual final artifact paths and freshness.
- Persisted same-selected-round leaf acceptance and measured local opens under existing accounting; never reconstruct acceptance from `validation.accepted` or treat missing measurements as zero.
- Parent/final DRC, required refs, mechanical and interface conformance.
- Final board/rules hashes, fresh export receipt, ZIP membership and member hashes; explain any post-routing processing before final verification/export.
- Every original product obligation, exact identity/count, functional electrical path and sourceability evidence. Missing support stays unverified.

Run both `triage run` and `triage audits`, including on rc0. Use the existing product evaluator; do not replace its acceptance policy with a hand-written boolean checklist. A passing physical tail with absent original-brief evidence is **physically built, product-unverified**.

If reliable retained before-fix evidence exists, retain and label its source/conditions. Otherwise obtain a matched bounded before/after witness when needed; never describe historical outputs under a different profile/catalog/evaluator as a controlled comparison. If an observer repair changes evidence interpretation, evaluate old and new artifacts under the same declared evaluator and distinguish **newly recognized existing success** from **newly generated product success**.

Record net product wins and regressions on the same verification set. A candidate with only stage-level improvement has not met this phase's product objective. Investigate its next first blocker; don't celebrate the stage and stop.

### Phase D — repeat by measured yield leverage

After the first repair's end-to-end outcomes, update the ranking using observed blockers. Work down the largest remaining reusable cause, with affected controls and end-to-end proof for each candidate.

Do not keep rolling identical failed provider/routing attempts, choose best-of-N winners, or spend the session on a one-off board without evidence of broader leverage. A routing gap can become highest priority if current population evidence shows it is now the limiting shared stage.

Use the already-built ESP32 fixture to finish an artifact-only original-obligation audit in a bounded side task; this may resolve an existing unknown, but cannot establish fresh random-brief generation yield by itself.

## 6. When a full development measurement is justified

Do not spend on a full 60 merely because unit tests or stage replays pass. Consider it after:

- The candidate repairs a reproduced shared cause rather than lowering acceptance.
- The preregistered affected/control set has genuine complete-product wins and no unexplained safety/control regression.
- Required offline/native checks pass and integrated source is frozen.
- The unchanged external manifest/contracts are validated and the remaining session/production budgets support the run.

An unrelated known failing brief need not be silently excluded or fixed first: it stays in the full denominator. This is the point of measuring the original distribution rather than perfecting selected witnesses.

Run the entire unchanged development cohort using the existing runner, `--parallel 1 --build-slots 1 --no-judge`, production profile and limits from section 3. Determine the exact existing manifest/CLI arguments from source/docs; do not guess flags or substitute the curated examples for the registered surprise-me corpus. Keep all first attempts. Mark interrupted/pending work honestly and use the runner's supported resume semantics—never regenerate finished failures as replacement winners.

Report full-cohort outcomes only after completion and independent artifact-bound audit. Report both physical fabrication rate and complete-product rate, per-family denominators and paired wins/losses where the evidence permits comparison. No extrapolation from the targeted set, no new population score from selected diagnostics, and no success from missing evidence.

Development exit remains **57/60 complete products**. The later once-only independent sealed release requirement remains **188/200**. The sealed set is not authorized for this session even if development reaches its threshold; preserve that decision for a subsequent release action.

## 7. Failure handling without user intervention

- **Provider/budget unavailable:** preserve exact failures and remaining allowance; do not change providers/models/caps. Continue deterministic reproduction, source repair, native checks, retained-artifact audits and regression work. Label paid/full-yield measurement blocked if it cannot run.
- **Router unavailable:** diagnose configuration against the pinned installation; do not switch routers or raise limits. Continue other reachable work. Do not call it a code pass.
- **Unrelated production job/catalog change:** leave it untouched. Avoid competing for occupied slots. Pin or identify read-only external evidence where supported, or disclose that the comparison is not controlled.
- **Custody mismatch:** stop further jobs using that input; preserve new evidence and the exact mismatch. Never restore originals over outside changes. Continue independent authorized work if safe.
- **Missing replay input/script:** do not substitute a winning workspace. Recover the recorded procedure from documentation/archived drivers in a disposable location; if the original input is unavailable, count that attempt as custody-blocked, not success.
- **Missing product proof:** keep it unverified and identify its owning observer or implementation gap. Never invent measurements, firmware or sourceability.

## 8. Session deliverables

Before finishing, complete all reachable implementation/verification work, preserve runtime evidence and update the durable handoff. Do not leave stub implementations, fake fallbacks or jobs without ownership.

The terminal report and saved evidence must contain:

1. Actual start/end/elapsed time and whether the eight-hour minimum was met; any safety exception explicitly explained.
2. Ranked live failure Pareto versus the historical starting picture, with stale causes distinguished.
3. Source changes and the real behavior/runtime proofs for each.
4. Preregistered diagnostic membership; every attempt, cost, product win and regression. Separate stage progress, physical pass and complete-product pass.
5. Full 60-brief score **only if actually completed and independently audited**; otherwise say not measured or precisely blocked. Preserve the historical baseline label.
6. Frozen source/evaluator/input/catalog provenance, artifact/export custody and original-before/after hashes.
7. Remaining highest-leverage source-owned blocker and its exact next verification—not a catch-all future promise.
8. No deployment/secret/acceptance changes, sealed-set status, active-job disposition and a precise continuation point.

Use the same evidence-first standard even if no new products succeed. An honest zero plus a reproduced source defect is more useful than an inflated fab-ready claim, but continue the autonomous repair-and-proof loop while safe actionable work remains.

## 9. Next-session kickoff prompt

```text
Execute docs/plan-surprise-yield-autonomous-2026-11-02.md as the approved plan. This is an execution request, not a request for another plan.

Work autonomously for at least eight hours of elapsed wall time without needing input from me. Record the start time and eight-hour minimum-finish time in durable progress evidence immediately. Do not stop at a phase boundary, after one repair, after passing tests, or after a stage commit. Continue the evidence-ranked repair-and-verification loop while safe actionable work remains. Never idle or spend just to fill time. Genuine safety or external blockers override the duration requirement; exhaust independent safe work and document the exact blocker.

The goal is more unchanged surprise-me briefs delivered as complete, independently audited, fabrication-ready products. Start by establishing which historical architecture/BOM failure buckets are still live, beginning with declared_interface_unrealized. Repair the highest-leverage concrete shared cause, then carry preregistered affected briefs and cross-family controls through the entire pipeline. Do not make the RP2040 fixture the default centre of the session.

Follow the plan's production/custody protections, unchanged acceptance rules, frozen-source measurements, one provider job and one physical-build job, at most $0.60 per run including internal retries, $12 session provider ceiling and any lower existing limits. Do not deploy, restart services, edit .env, raise caps, mutate originals, or inspect sealed/quarantined evidence.

Make normal engineering decisions from repository evidence without asking me. Keep all failures and costs; no best-of-N winners or easier replacement briefs. Tests, stage commits and route-only replays are diagnostics, not product successes. Save durable progress throughout and recover existing jobs rather than duplicating them.

Before finishing, complete in-flight bounded verification, preserve artifacts, update the handoff, and report elapsed time, source changes, audited product wins/regressions, all denominators, spend and the next source-owned blocker. Report a population yield only if the full unchanged cohort actually completed and was independently audited.
```
