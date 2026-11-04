# Surprise-me yield: disk recovery, complete boards, matched measurement

Status: replacement execution plan for one autonomous session of **at least eight elapsed hours**. This document replaces the previous session's work ordering, not its production protections, original acceptance rules, retained failures or release thresholds. Execute it; do not spend the session writing another plan.

## 1. The outcome and the mistake this plan must prevent

The goal is a higher percentage of **unchanged Surprise me briefs delivered as complete, independently audited, fabrication-ready boards**. A compiler fix, passing tests, stage commit, native circuit, routed preview, build rc0, or newly favorable report is not that outcome.

The previous session retained 123 first-attempt records, 122 completed attempts, one interruption and zero complete products. Its full G campaign is invalid/incomplete. Its latest I0 source has no new paid end-to-end measurement. Those facts are starting evidence, **not a current before/after yield baseline**.

This session must close the entire chain for real briefs, not fix successive isolated gates indefinitely. Its execution controls are:

- **Disk reclamation first**, before source snapshots, provider calls or physical builds.
- One complete, reusable vertical repair at a time: viable parts → original function → actual connections → routing → fresh exports → independent audit.
- A fixed six-family pilot, deadlines for obtaining a real product, and mandatory strategy changes after zero-product results.
- A fresh, fixed-denominator before/after comparison, not comparisons of incompatible historical runs.
- No expansion to the full cohort before actual pilot product wins. No bypass of that gate because tests pass or the clock is running.

### Success and reporting contract

1. **Pilot evidence:** new complete products on the preregistered pilot, compared with its fresh baseline. Label this selected development evidence, never random-population yield.
2. **Primary demonstration:** both frozen arms finish all 60 unchanged development briefs, receive independent artifact-bound audits under the same acceptance semantics, and the candidate has more complete products than baseline. Let B and C be their complete-product counts; report B/60, C/60 and **100 × (C − B) / 60 percentage points**, plus paired gains and losses.
3. Target **at least three net additional complete products across at least two families**. The minimum demonstrated improvement is a positive net full-cohort count; the target is not permission to redefine failure as success.
4. Report physical-fabrication-only passes separately. All original product obligations must pass to count a complete product; missing evidence stays unverified. Do not claim energized/thermal/firmware/manufactured-hardware performance from static artifacts.
5. A full 60-brief development comparison is an empirical result on that fixed cohort, not proof of a population probability or an independent release result. Do not infer a population yield from the pilot or an incomplete cohort. The existing development/release qualification thresholds remain 57/60 and 188/200; this session does not authorize opening the sealed release set.
6. If no product gain is demonstrated, the first sentence of the final report must say so. Never headline tests or source changes instead.

## 2. First execution phase: reclaim disk space and prevent another ENOSPC

Record the session start and eight-hour minimum-finish time in a new durable development evidence directory **before cleanup**. The clock includes this phase; do not restart it later. Inspect and recover existing owned jobs rather than duplicating them.

At plan preparation, the production root had approximately **5.28 GB free and 94% used**. Recheck; this is not adequate justification to copy another large workspace. The previous session filled the production filesystem while copying a source snapshot and interrupted G57. Its later recovery does not establish that production users were unaffected.

### Reclaim, rather than merely record, space

1. Capture filesystem identity, byte-level free space, active-job ownership and a **metadata-only size inventory** of candidate locations. Do not inspect sealed/quarantined evidence while finding large files. Campaign evidence access remains confined to the development root in section 8.
2. Make a protected-path list: live checkout and environments, `.env`, production databases/catalog, active job directories, original projects, every retained measured run, failed/interrupted trial, final export, and source snapshot required by a retained receipt. These are not cleanup candidates.
3. Identify genuinely reclaimable material: verified inactive and unmeasured partial copies, session-owned orphan scratch outputs already preserved elsewhere, and reproducible package download/wheel caches whose eviction does not uninstall or change running environments. Establish ownership, disposability and absence of active readers/writers before deletion. No age-only or wildcard cleanup, production log truncation, database vacuum, environment deletion, or deletion of unrelated code/data.
4. Actually remove authorized disposable material. Record each exact path, reason it is disposable, allocated bytes and measured free space before/after. Do not confuse apparent duplicate size with physical bytes reclaimed, especially for hardlinks.
5. If duplicate immutable files remain between **own frozen snapshots**, lossless deduplication is allowed only with content-hash equality and complete source-fingerprint checks before/after. Never link mutable live-checkout files into frozen snapshots. Do not edit any linked snapshot in place. Do not copy everything first and deduplicate after filling the disk.
6. Do not delete measured snapshots/evidence to achieve a numeric free-space target. If another already-authorized persistent filesystem is available, preserving complete, hash-verified artifacts there may be considered without changing originals or losing recorded retrieval paths. Do not provision storage, spend money, or change system mounts without authority.

### Admission gate after reclamation

- Initial admission requires **at least 8 GiB free** and `free_bytes >= 4 GiB + 2 × estimated_remaining_write_bytes`. The **4 GiB production headroom is untouchable**. Include the next snapshot, all in-flight work, planned measurement artifacts and audit/export scratch space in the estimate, using actual allocated bytes and observed prior build sizes. At subsequent admission boundaries, enforce the remaining-write formula; the initial 8 GiB entry target is not an excuse to let a later job consume the 4 GiB floor.
- Do not start paid work, a snapshot copy or a physical build until reclamation is recorded and its storage reservation passes. Recheck at existing job/brief admission boundaries and before large copies/builds/exports. Use an existing preflight where available; if absent, add the smallest source-owned check at the existing admission point, not a monitoring framework.
- Capacity for the full paired measurement must be assessed before launching either full arm. Never plan to discard failed artifacts later to make the campaign fit.
- When headroom is insufficient, stop **new admission**; preserve active bounded work and its reserved output space. Do not fill production storage and hope to recover afterward. A disk/custody fault blocks further paid runs until resolved.
- If safe reclamation cannot satisfy the reserve, record the exact shortfall and exhausted safe candidates. Continue reachable low-write source analysis/repair, but label physical/yield measurement storage-blocked. Do not silently lower the floor, erase evidence, or wait idly.

Deliverable: `disk_reclamation.json` with actual reclaimed bytes, protected paths, capacity/reservation calculation and the continuing admission rule. A disk-usage observation alone does not complete this phase; if nothing is safely reclaimable, explicitly record that blocker instead of claiming cleanup.

## 3. Establish immutable experiment identities, not another broad investigation

Read these existing entry points; do not repeat all the historical investigation:

- `docs/handoff-surprise-completion-2026-10-12.md`, November execution/closeout sections.
- `logs/self_eval/surprise_completion/20261010T204651Z/development/autonomous_yield_20261102T140800Z/authoritative_evidence_index.json`.
- That directory's `denominators_and_outcomes.json`, `final_source_and_input_custody.json`, `final_cost_reconciliation.json`, and `all_retained_rc0_physical_obligation_statuses.json`.
- Original `development/bundle/manifest.json`, `obligations.json` and `review.json` through the existing validators.

After disk admission passes:

1. Freeze **B0**, the actual starting generator/compiler source. Reuse the retained I0 snapshot only if its fingerprint and the current source match and its integrity is verified; otherwise record the actual new baseline. Do not label H/G outcomes as B0.
2. Record the original manifest/brief/obligation hashes, generator and evaluator identity, production profile, router identity, limits, catalog identity and ledger/exposure start. Freeze acceptance interpretation **E0** for both comparison arms. Evaluator-only obligations must never enter the generation prompt.
3. Bind both arms to the same catalog data. Use an existing supported read-only, consistent snapshot/backup mechanism if available and covered by the disk reservation. Never refresh or edit the production catalog, stop its maintenance, or improvise a partial database copy. If only the live catalog can be used, record before/after identities and invalidate a controlled comparison if it changes materially.
4. Preregister full primary membership: all original 60, no exclusions or replacements. Family denominators are power 12, controller 14, actuator 8, analog 12, sensors 6, display/LED 8.
5. Preregister pilot membership and attempt order below before new paid outcomes. Every baseline/candidate trial gets a distinct identity. Keep all outcomes and costs.

### Fixed pilot: first original in each family

These are selected mechanically from original manifest order, not by new success:

| Original index | Slug | Family | Purpose |
| ---: | --- | --- | --- |
| 1 | surprise-1483430243 | power | USB-C 5 V → 5 V, reverse protection, four terminals, one-edge placement |
| 2 | surprise-1997580059 | controller | RP2040 + CH340C, 2S input, JST and RGB sentinel; not the session centre |
| 3 | surprise-1437981489 | actuator | ESP32-C3 module, two relays, 24 V input and reset |
| 4 | surprise-449192858 | analog | MCP6002 humidity conditioning, 5 V input, terminals and reset |
| 7 | surprise-82012717 | sensors | ESP32-S3, 18 V input, buttons and four layers |
| 8 | surprise-185438647 | display_led | CH32V003, four outputs, four-digit display, 24 V and four layers |

Run one fresh full-pipeline B0 pilot baseline per brief. These are baseline measurements, not reruns intended to reconfirm historical errors. Pilot membership remains six even when a brief is difficult. Do not replace the controller sentinel with an easier board. Label sentinels as controls only where dependency analysis actually establishes they are nonaffected.

## 4. Repair a whole product-producing path

Time-box initial evidence consolidation to approximately 30 minutes after disk admission, not hours of recreating the old Pareto. While the single B0 provider/build slot is occupied, independent source research can proceed. Freeze B0 first; do not modify it.

### Starting repair priority: sourceable, electrically valid complete power/core composition

The last H run had no builds: four originals hit unavailable canonical AP63205/XAL7070/ESP32-S3/UMW choices before meaningful build verification. Separate earlier wiring/identity repairs did not remove these later blockers. The retained power brief reached physical acceptance only in F and still lacked output-voltage/conversion/reverse-protection proof; an additional USB source could bypass its PMOS protection. Those are concrete end-to-end obstacles, not reasons to relax audit gates.

Construct a **product closure table** for the fixed pilot. For each original obligation, record the existing implementation, required parts, catalog availability, remaining producer/consumer gaps, build risks and exact artifact proof. The first rejected gate is not the whole table. Before another supplier-default swap, inspect the downstream parts and functional obligations it would expose.

Choose one coherent reusable implementation with an evidenced path to completing at least two pilot families. Start with canonical power/core composition unless current B0 evidence falsifies that ranking. The repair must:

- Resolve an entire required support circuit to real available parts, including controller/regulator, inductor, capacitors with effective-voltage ratings, protection, feedback, LEDs, connectors and programming interfaces—not just the first missing C-number.
- For **generic/family-only** requests, select a reviewed capable available implementation/variant; distinguish this from substituting an exact protected identity. Explicit required MPNs/manufacturers stay exact, and unsupported ones remain counted failures.
- Fix the source choosing/advertising the impossible default and its deterministic implementation together. Filtering an unavailable part out of a prompt without providing a valid supported path is not a product unlock.
- Respect the actual input range, output voltage/current, simultaneous pin capabilities, feedback/compensation, reverse paths, fuse/active-protection behavior and thermal/current limits. Manufacturer data and actual parts/topology must support the claims.
- For 5 V → 5 V, do not choose a buck that cannot meet its required output at the actual input. Conversely, do not replace a converter with a wire/fuse and declare conversion proven. Satisfy the unchanged original contract and existing evaluator with the actual circuit.
- Eliminate all unintended source paths around required protection. Trace every physical power input and output, not only the intended net label.
- Treat unresolved PTC evidence as a **blocked candidate implementation**, not as proof all source work is blocked. Investigate a genuinely qualified implementation satisfying the same original requirement. Do not remove required protection, invent geometry, rewrite UMW into ST under the same bundle, refresh the catalog, or guess ratings.
- Use existing typed requirements, lowerers/recipes, ownership, allocator and validation conventions. Deterministically publish canonical support wiring/ports for supported circuits rather than repeatedly asking the model to rediscover them. No brief-ID/seed-specific branches, golden-output injection, hand-edited run states, new orchestration framework, retry expansion or limit increase.

Before a paid candidate run, prove the entire chosen support BOM resolves in the fixed catalog and exercise its real compile/synthesis/native path. Save/reload the actual KiCad witness and reject a meaningful broken-pad/protection/binding negative. Unknown downstream obligations stay explicitly open; do not equate a partial native graph with a finished product.

Use a fresh full-pipeline candidate pilot to find what still prevents delivery. Keep ownership of the **same end-to-end brief closure** until it passes or a documented external/safety barrier makes that implementation unavailable. Do not switch to an unrelated low-level defect just because the current test suite is green. Do not retry an unchanged source/unchanged failure hoping the model chooses better parts.

## 5. Outcome deadlines and compulsory changes of approach

These are execution controls, not permission to falsify success or abandon bounded jobs:

| Session checkpoint | Required outcome or corrective action |
| --- | --- |
| First 45 minutes | Disk reclamation/reservation resolved, or exact safety blocker recorded; no blind source copy |
| By 90 minutes | Baseline identities/pilot fixed and first full candidate brief launched, unless an explicit safety/external prerequisite blocks it |
| By hour 3 | First independently audited **new complete product**, not stage progress; if zero, enact the zero-product escalation below |
| By about hour 4 | Target two new pilot products in two families; freeze the measurement candidate only after its expansion gate passes |
| Remaining window | Finish the fixed before/after measurement and audits; do not spend it polishing unrelated code or creating more candidate snapshots |
| At/after hour 8 | Finish admitted bounded verification, preserve evidence and report; do not abandon an active measurement or launch a new long campaign just to extend time |

### Zero-product escalation

After two changed-source candidate pilot rounds with zero product wins, or at hour 3 with none, **stop the current repair strategy**. Do not launch a third nearly identical round, another full cohort, or a broader green-test sweep.

Write a short decision receipt: every pilot failure, the whole remaining obligation chain, what the attempted repairs failed to address, and the next materially different source operation. Then:

1. Inspect the actual first failure and the product closure table together; distinguish a library/supported-composition gap from model serialization or a truthful hardware refusal.
2. If component qualification is blocked, move to another reusable implementation/family within the **same fixed pilot**, preserving failed cases in its denominator. Do not spend the session searching one PTC indefinitely.
3. If the model cannot emit a known supported composition, repair the existing typed producer/consumer contract or deterministic canonical construction; do not increase correction rounds or inflate collections. Obtain the actual bounded failure input before claiming a parser defect.
4. If a real board exists but evidence is missing, independently determine whether the circuit is actually correct. Fix a proven observer defect only with native intact/broken witnesses and common before/after reevaluation. Never turn unsupported measurements into assertions.
5. For stochastic prompt/contract claims, preregister a bounded N-of-3 diagnostic where required by the investigation skill, report all three, and never select an any-success winner. These diagnostics are not primary cohort trials.

A missed deadline requires a changed operation and journal entry, not relabeling another test as progress. If no complete products exist, the session remains unsuccessful irrespective of how many source fixes land.

## 6. Expansion gate and matched full-cohort measurement

A full run is authorized only when **all** are true:

- The fixed pilot has at least **two baseline-failure → candidate-complete-product gains in two families**, independently audited, with no unexplained candidate loss or safety regression.
- The change is reusable source behavior, not fixture surgery or acceptance weakening.
- Integrated source and the relevant native negative checks pass; the entire measurement generator/compiler candidate **C1** is frozen.
- Input, evaluator, profile, router, catalog, storage and budget identities/reservations pass.
- Observed durations support a bounded full comparison plus independent audits. If prerequisites take longer than planned, do not launch a campaign whose only rationale is reaching the eight-hour boundary.

Pilot outcomes are **not spliced into** primary campaigns. Preregister **two fresh primary arms**, B0 and C1, all 60 each, one first attempt per brief per arm. This avoids unsupported receipt migration, changing pilot denominators or selecting winning retries. All diagnostic rounds remain separately reported.

Use the existing runner, not a new evaluator. Exact options below were checked against its CLI during plan preparation:

```bash
# In EACH respective frozen snapshot, with snapshot PYTHONPATH and production
# profile/unchanged limits. OUT and allowance are distinct per arm, assigned
# in the preregistration and bounded by the shared remaining session guard.
/home/kicraft/KiCraft/.venv/bin/python -m kicraft.eval.self_eval \
  --brief-manifest /home/kicraft/KiCraft/logs/self_eval/surprise_completion/20261010T204651Z/development/bundle/manifest.json \
  --obligations /home/kicraft/KiCraft/logs/self_eval/surprise_completion/20261010T204651Z/development/bundle/obligations.json \
  --validate-manifest

# For execution REMOVE --validate-manifest, retain both input arguments, and add:
# --out <unique-arm-directory> --campaign-budget-usd <reserved-arm-allowance>
# --parallel 1 --build-slots 1 --no-judge --repeats 1
# --contract-version benchmark-original-v1
# --max-park-rounds 12 --build-timeout 1800
# Do not add --only, --limit, --design-only or family filters to a primary arm.
```

Run B0 then C1 sequentially under the one-provider/one-build restriction. Run `--audit-campaign <arm-directory>` and `triage run`, `triage stages`, `triage audits` using the existing readers; audit every apparent success, not only failures. An independent read-only reviewer must reconcile originals, evaluator results and the delivered artifacts; it cannot accept the producer's own claim as proof. No separate provider judging spend is required.

For each counted board require actual required identities/counts/connections, sourceability, powered functional paths, mechanical/interface constraints, selected-round acceptance and opens, final board/rules freshness, existing ERC/DRC gates and fresh matching fabrication exports/ZIP member hashes. Native proofs, missing measurements and rc0 alone cannot satisfy this.

If an observer bug is repaired, lock a single corrected interpretation for **both** arms, reevaluate old and new artifacts consistently, and classify newly recognized old successes separately from newly generated products. Do not change normative acceptance thresholds. A materially changed profile/catalog/contract invalidates a controlled pair; retain results without pretending they are matched.

Do not change C1 during its primary run. Do not rerun failed first attempts as replacements. Recover interrupted jobs with the runner's supported external resume semantics, which preserve failed/interrupted first attempts. An incomplete/invalid arm means **no full-cohort percentage comparison**; report the exact missing/interrupted denominator and any pilot wins without extrapolation.

## 7. Eight-hour autonomy, resources and durable evidence

At startup create a unique, nonexisting UTC-suffixed session directory under the authorized development root. Record `started_at_utc`, `not_before_finish_utc = start + 8h`, elapsed reference, ledger baseline/exposure and recovery/job ownership immediately. Use existing runtime job facilities, not a detached agent that nobody monitors.

Keep a short progress journal and evidence index after every significant change/outcome and at least every 30 minutes during longer work. Its headline fields are **complete products, net new products, paired losses, first blockers and remaining measurement capacity**. Tests and stages are supporting fields. Record source/input/catalog hashes, original IDs, every trial/cost, active job IDs, reserved disk/provider capacity, deadline decisions and exact next operation. Recover the same session and jobs after interruption; never reset its clock or costs.

### Limits

- At most **one provider job and one physical-build job**. Read occupancy before admission; do not kill unrelated production work. Independent substantial source/review slices may run concurrently under one integration owner; sibling provider/build jobs are forbidden.
- **$0.60 maximum per run including internal retries**, and **$12 maximum new session provider spend**, subject to every lower existing daily/lifetime/project/provider guard. Neither these values nor deadlines authorize cap increases.
- Reserve **at least $8 of the session ceiling for the two primary arms and their bounded completion**, with at most $4 available for baseline/candidate pilots and live diagnostics; lower remaining production allowances reduce these figures. Assign nonoverlapping guarded arm allowances before launch. These are spending reservations, not promises all trials cost their maximum. Forecast from observed complete lifecycle charges with a safety factor, not just successful-call costs; do not launch a comparison that existing guards predict cannot finish.
- Include failed, retried, interrupted and unattributed positive charges conservatively. Distinguish old reserved/uncertain exposure from new charges. No guard resets, provider/model changes, budget-reset waiting or purchasing more capacity.
- Keep `max_duration_s=3600`, `build_timeout_s=1800`, `max_park_rounds=12`, and `KICRAFT_BUILD_MAX_WALL_S=1620`. Keep production profile, native KiCad and production-pinned KRT. No extra retries, raised token/collection limits or routing-rule relaxation.

Continue useful repair/audit/regression work for at least eight elapsed hours. A product win early is a reason to generalize and measure, not stop. No sleeping, repeated unchanged checks or spending to fill time. Safety or genuine external unavailability may override duration **only after independent safe work is exhausted and exact blockers are recorded**. A difficult engineering problem is not automatically an external blocker. Do not ask the user for choices the repository/evidence can answer.

## 8. Production and custody boundaries

- This checkout is the production host. Read applicable skills before their work: `self-eval`, `kicraft-investigate`, `kicraft` for circuit/library design, and `verify` for place/route/promote changes.
- No deployment, service restart, `.env` edit, production queue mutation, cap increase, router replacement, clearance reduction, unauthorized via-in-pad or acceptance weakening.
- Campaign evidence access **only** beneath `logs/self_eval/surprise_completion/20261010T204651Z/development/`. Do not broadly search its parent campaign, inspect sealed/quarantined holdouts or `ReviewPowerRatingProof`, or open the sealed release set.
- Originals, briefs, manifests, obligations, seeds, projects, states, consent, rules and all failed/interrupted evidence stay immutable. Scratch copies are for diagnostics, never replacements for original full-pipeline trials. Do not hand-edit a board to count it as a generated success.
- Run measurements from unique frozen source cwd/PYTHONPATH; symlink `.venv`/`.env`, never copy secrets. Verify source/input custody before/after. On outside changes, preserve them and suspend affected admission; do not restore over the user's work.
- Use existing production readers/runner/product evaluator. No second harness, handwritten replacement acceptance boolean, custom yield scorer or golden-board fixture injection.
- Protect all old measured snapshots/artifacts during disk cleanup. Archive and hash owned throwaway drivers before removing scaffolding. A storage guard must not erase failures.

## 9. Closeout deliverables

Finish admitted bounded work and independent audits before yielding; preserve every failure/cost and update the existing handoff. Do not leave source stubs, unowned jobs or a supposedly measured candidate whose code changed mid-run.

The final reply must be understandable without KiCraft internals:

1. First sentence: **whether a measured fab-ready percentage improvement was actually demonstrated**.
2. Before/after complete boards and denominators, percentage-point change and per-family counts **only if both full arms completed and audited**. Otherwise explicitly say full-cohort improvement was not measured; list pilot outcomes separately. Show paired gains/losses and all interruptions.
3. Start/end/elapsed time, actual disk bytes reclaimed, remaining headroom and whether the disk admission rule held.
4. Total conservative new provider spend, all attempted-run denominators and separate old exposure.
5. Briefly name the reusable change that produced the new boards. If none, say zero and explain the failed deadline/pivot decisions rather than presenting green tests as compensation.
6. Link representative complete board/fabrication artifacts, independent audits, frozen source/inputs and the evidence index; give the exact next source-owned blocker.
7. State remaining owned-job disposition and any production/custody incident without claiming unmeasured no-impact.

No plan can guarantee new board successes. This plan makes complete boards and matched measurements the advancement criteria, imposes an early strategy change when they do not materialize, and forbids calling another eight hours of intermediate repairs a yield improvement.
