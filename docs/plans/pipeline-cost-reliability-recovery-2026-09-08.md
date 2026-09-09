# KiCraft pipeline cost and reliability recovery plan

**Status:** proposed recovery plan; implementation not deployed  
**Evidence window:** production spend ledger from 2026-09-02 through 2026-09-08, plus the ESP32-S3/HUB75 stage-debug walkthrough

## Decision

Stop optimizing for “eventually return something” through repeated model calls. Optimize for one attributable candidate, one bounded correction, and an inspectable failure.

During recovery:

1. Do not silently move a run from the flash price class to the pro price class.
2. Do not regenerate a complete BOM or wiring result after a deterministic rejection unless the rejection is both recorded and model-correctable.
3. Preserve every valid work unit and retry only the smallest failed unit.
4. Treat cost, retry count, semantic cleanliness, deterministic commit, and final fab readiness as separate release gates.
5. Keep all electrical, ERC, DRC, and fabrication gates strict. The fix is earlier correctness and bounded execution, not accepting unsafe output.

## Current evidence

### Reliability and cost baseline

Production ledger data since 2026-09-02:

| Metric | Observed |
|---|---:|
| Tracked run IDs | 19 |
| Runs with all five LLM stages recorded successful | 9 |
| Mean summed stage cost per run | $0.080830 |
| Maximum summed stage cost per run | $0.345598 |
| Maximum full spend-ledger cost for one run | $0.428397 across 68 calls |
| BOM runs | 19; 15 successful |
| Mean BOM attempts | 6.21 |
| Costliest BOM failure | 22 attempts, 2,241.956 s, $0.323107 |
| Wiring attempts | 663 |
| Wiring `commit_rejected` attempts | 468, costing $0.669564 |
| Wiring rejections with no diagnostic code | 409 |

Only 9 of 19 tracked runs reached five successful LLM stages. That is before requiring a routed, fabricated-quality board.

### Price-class amplification

Production uses the `flash` design profile. `KICRAFT_PROVIDER_FALLBACK_PROFILE` is unset, so `Settings` defaults the fallback to `pro`.

| Profile | Prompt ceiling per million tokens | Completion ceiling per million tokens | Recent mean cost per tracked attempt |
|---|---:|---:|---:|
| flash | $0.11 | $0.24 | $0.001247 |
| pro | $1.46 | $4.38 | $0.015369 |

The pro profile is 13.3 times the configured prompt price and 18.25 times the completion price. In the recent ledger it cost 12.3 times as much per tracked attempt. Pro represented 105 of 960 tracked attempts but $1.613777 of $2.680189 tracked attempt cost.

The architecture walkthrough reproduced the mechanism twice:

1. `deepseek-v4-flash` returned `provider_rate_limited`.
2. The stage driver automatically switched to `deepseek-v4-pro`.
3. The pro call entered a reasoning loop or returned a question/schema defect.
4. More pro attempts followed.

This is the direct mechanism behind a roughly 10x cost increase.

### Retry amplification

The reviewed architecture artifacts alone recorded at least $0.095208 in provider spend while repeatedly correcting one facet. Recurrent causes were:

- reasoning-loop retries;
- semantic-repair retries;
- a repeated clarification despite prior guidance;
- automatic pro fallback after rate limiting;
- schema rejection of the natural name `5V BUCK`;
- electrical repairs that introduced a different electrical defect;
- false-positive semantic checks that triggered another complete draft.

The final clean artifact still took 168.171 seconds and three provider attempts because its first flash call entered a reasoning loop. Local CPU time was 0.233 seconds, so provider execution—not KiCraft computation—dominated latency.

### Rejection telemetry is incomplete

Of 468 recent wiring `commit_rejected` attempts, 409 have `diagnostic_codes=[]`. The runtime has the deterministic `errors`, `offenders`, and `_commit_signature`, but `stage_attempts` persists only semantic diagnostic codes. The ledger therefore proves the cost but usually cannot identify the gate that caused it.

This attribution gap must be fixed before tuning prompts or increasing retry limits. Otherwise the next change cannot be tied to the rejection it is supposed to eliminate.

## Current walkthrough state

The accepted state still contains only `intent` and `functional_spec`. Architecture was never committed.

The last complete pending architecture candidate is electrically coherent but oversized:

- 20 V / 5 A USB-PD contract;
- 5 V / 6 A synchronous buck;
- separate 3.3 V regulator;
- separate MCU, HUB75, LED-string, and speaker domains.

The user selected a correction to **15 V / 3 A input, 5 V / 6 A buck, and separate 3.3 V buck**. The corresponding provider draft was cancelled when this plan was requested. Do not commit the existing 20 V / 5 A artifact. Resume by producing one reviewed architecture draft with the selected topology.

Local source changes made during the walkthrough are not deployed:

- `.agents/skills/kicraft/stages/architecture.md`
- `kicraft/design/models.py`
- `kicraft/design/stage_semantics.py`
- `kicraft/server/stage_runtime.py`
- `tests/test_kicraft_models.py`
- `tests/test_stage_semantics.py`
- `.kicraft/debug/findings.json` is diagnostic bookkeeping only

Focused verification currently passes:

```text
2 passed: sheet-name model coverage and architecture power/regulator semantics
```

## Phase 0 — stop uncontrolled spend

### Change

1. Set `KICRAFT_PROVIDER_FALLBACK_PROFILE=` in production during recovery. A rate-limited flash stage should return a classified, retryable availability error instead of silently entering the pro price class.
2. Keep the existing per-draft hard budget. Add a project-level LLM budget that is checked before every stage/work-unit call, not only after a stage returns.
3. Emit a user-visible `provider_rate_limited` state with a retry action. Do not relabel it as a design failure.
4. Stop a stage when the same deterministic commit signature repeats. Do not spend the remaining retry budget on an unchanged gate/offender set.
5. For `commit_process_failed`, fail immediately with the subprocess evidence. A model cannot repair an unavailable executable, timeout, or crashed deterministic commit process.

### Files

- `.env` for the temporary production fallback setting; never commit secrets
- `.env.example` for documented recovery behavior
- `kicraft/server/config.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/server/client.py`
- `kicraft/server/spend_guard.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_spend_guard.py`

### Acceptance

- A simulated 429 never changes to a more expensive profile when fallback is disabled.
- The returned failure kind is `provider_rate_limited` and retains the original stage.
- A repeated commit signature performs zero additional provider calls.
- `commit_process_failed` performs zero model correction calls.
- Every call is rejected before dispatch when the remaining project budget cannot cover its configured ceiling.

## Phase 1 — make every paid rejection attributable

### Change

Extend `stage_attempts` with redacted deterministic rejection facts:

- `commit_gate_codes`: stable gate IDs such as `9.14`, `9.15`, and `9.29`;
- `offender_count`;
- `rejection_signature`: hash of ordered gate IDs plus redacted offender identities;
- `provider_profile` and `fallback_reason`;
- `candidate_retained`: whether a valid candidate/work unit survived the attempt.

Populate these fields from `commit_result` and `_commit_signature` for every `commit_rejected` and `commit_process_failed` path. Do not store prompts, responses, customer brief text, or raw part identifiers in the ledger.

Update the cost report to rank:

1. cost by stage and provider profile;
2. cost by failure kind;
3. commit rejection count and cost by gate code;
4. repeated rejection signatures within one run;
5. first-pass, repair-pass, and aggregate-repair success rates by work unit.

### Files

- `kicraft/server/spend_guard.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/cli/web_cost_report.py`
- `kicraft/eval/llm_analysis.py`
- `tests/test_spend_guard.py`
- `tests/test_stage_driver_retry.py`
- cost-report tests adjacent to `web_cost_report.py`

### Acceptance

- Zero new `commit_rejected` rows have an empty gate-code set when `commit_result.errors` contains a gate.
- A report can explain 100% of provider cost by run, stage, profile, call mode, and terminal outcome.
- The report shows the top wiring and BOM rejection signatures without reading production-user content.

## Phase 2 — finish architecture correctness without repair churn

### Already addressed locally

The walkthrough added or tightened deterministic coverage for:

- 5 V external-load demand versus input-source headroom;
- USB-PD's 5 A contract ceiling;
- separate input-contract and regulated-output current parsing;
- 5 V converter output-current headroom;
- duplicate same-voltage rail identity;
- a distinct ESP32-S3 3.3 V regulator topology and sheet;
- natural digit-leading sheet names such as `5V BUCK` and `3V3 REGULATOR`;
- false positives caused by regulator descriptions that mention the MCU.

### Remaining change

1. Add right-sizing guidance: choose the smallest common PD contract with adequate output, onboard, conversion-loss, and transient margin.
2. Diagnose gross input overprovision as a review-required cost/complexity finding, not a fabrication gate. For this fixture, 100 W input for a 30 W regulated rail must be surfaced.
3. Suppress a blocking question when its complete material answer already exists in the instruction and prior-answer set.
4. Ensure the corrected 15 V / 3 A + dual-buck draft retains all sheets, inter-sheet nets, programming strategy, and signal directions.
5. Freeze the failed and corrected candidates as minimized fixtures without provider reasoning or user-identifying content.

### Files

- `.agents/skills/kicraft/stages/architecture.md`
- `kicraft/design/stage_semantics.py`
- `kicraft/server/stage_runtime.py`
- `tests/test_stage_semantics.py`
- `tests/fixtures/stage_reliability/`

### Acceptance

- The frozen original candidate emits the expected stable diagnostic codes.
- The corrected 15 V / 3 A candidate emits none of those codes.
- One fresh provider draft reaches review in at most two calls, with no repeated question.
- No architecture call falls back to pro during the recovery canary.

## Phase 3 — stop BOM from regenerating expensive failures

BOM is the current cost center. Recent spend records show $1.746025 across BOM calls, including $1.348436 on pro. The costliest run failed after 22 BOM attempts and five work units.

### Change

1. Complete Phase 1 first and rank BOM failures by deterministic gate/signature.
2. Preserve every schema-valid, commit-valid sheet work unit. Retry only the failed sheet; never regenerate accepted sheets during aggregate repair.
3. Carry the exact failed gate and bounded offender set into the repair prompt. If the signature does not shrink after one repair, stop and retain the candidate.
4. Do not redraft on `commit_process_failed`; return the process failure.
5. Implement the remaining cost reductions already identified in `stage-resource-telemetry-and-bom-cost.md`:
   - pre-filter the parts-library block by architecture categories;
   - collapse lookup plus vendoring into one resolve-and-bundle operation;
   - drive vendoring priorities from observed misses.
6. Cache identical prompt-state/tool-resolution inputs by content hash within a run. Reusing a work unit must not call the provider again.
7. Require BOM support for the selected architecture before wiring: PD controller, 5 V / 6 A buck, 3.3 V buck, protection, programming straps, level shifters, and audio amplifier must each have explicit roles and ratings.

### Files

- `kicraft/server/stage_runtime.py`
- `kicraft/server/stage_work_units.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/design/cli_app.py`
- parts resolver and bundling modules identified by the top failure signatures
- `tests/test_stage_work_units.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_kicraft_stage_cli.py`

### Acceptance

- Mean provider calls per BOM work unit are at most 1.5 on the fixed canary.
- No accepted work unit is regenerated.
- Aggregate repair runs at most once.
- Repeating a rejection signature stops the stage.
- `commit_process_failed` costs no follow-up provider call.
- BOM p95 LLM cost is below $0.02 on the recovery canary.

## Phase 4 — remove wiring's commit-rejection loop

Wiring has 468 recent commit rejections; 409 cannot currently be attributed. Do not tune the wiring prompt before Phase 1 exposes the actual gate distribution.

### Change

1. Rank wiring rejections by gate code, offender count, work unit, and repeated signature.
2. Fix the highest-cost signature first and replay the frozen candidate before changing another class.
3. Make recipe-owned support wiring deterministic. The model should map only project-specific pins and nets.
4. Preserve valid sheet wiring and repair only failed nets/pins. Never regenerate the entire board wiring after one sheet-local rejection.
5. Resolve BOM insufficiency through the explicit BOM reconcile path; wiring must not invent refs or consume retries attempting to do so.
6. Convert stable graph invariants—programming reachability, special-pin treatment, inter-sheet realization, dangling signal nets, and no-connect coverage—into pre-commit diagnostics using the same implementation as commit.
7. Stop after one non-improving repair signature.

### Files

- `kicraft/server/stage_runtime.py`
- `kicraft/server/stage_work_units.py`
- `kicraft/design/stage_semantics.py`
- `kicraft/design/synthesis/validation.py`
- `kicraft/design/recipes.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_stage_work_units.py`
- `tests/test_kicraft_validation.py`

### Acceptance

- Zero unexplained wiring commit rejections.
- At least 90% of wiring work units commit on their first candidate in the fixed canary.
- Wiring retries never regenerate accepted sheets.
- BOM reconciliation performs one explicit stage transition, not repeated wiring calls.
- Wiring p95 LLM cost is below $0.015 on the recovery canary.

## Phase 5 — eliminate reasoning-loop latency

Architecture repeatedly spent 168–238 seconds despite less than 0.3 local CPU-seconds.

### Change

1. Use the existing reasoning-loop telemetry to compare reasoning-disabled first calls against the current policy on the same frozen briefs.
2. For stages where reasoning does not improve deterministic acceptance, disable it on the first call rather than paying for a known loop and then retrying without reasoning.
3. Keep one bounded serialization recovery only for truncated or malformed JSON.
4. Separate provider wall-stall, output-token exhaustion, and repeated-reasoning signatures in telemetry.
5. Never combine a reasoning retry, schema retry, semantic repair, and expensive model fallback into an unbounded chain. One call budget owns all recovery modes.

### Files

- `kicraft/server/config.py`
- `kicraft/server/client.py`
- `kicraft/server/stage_runtime.py`
- `tests/test_client_provider.py`
- `tests/test_stage_driver_retry.py`
- `kicraft/eval/llm_analysis.py`

### Acceptance

- Architecture p95 provider wall time is below 60 seconds on the fixed canary.
- Reasoning-loop incidence is below 2%.
- A loop consumes at most one paid response and never triggers an expensive-profile fallback.
- Correctness and commit rate do not regress versus the frozen baseline.

## Phase 6 — end-to-end release gate

Use one fixed corpus throughout implementation. Do not change briefs between phases.

### Required measurements

For each run record:

- stage and work-unit provider calls;
- flash/pro profile and fallback reason;
- input/output/cached tokens and cost;
- schema and semantic diagnostics;
- commit gate codes and rejection signatures;
- reused versus regenerated work units;
- stage wall/CPU time;
- deterministic build result;
- ERC, DRC, unrouted count, connectivity, and fab-ready verdict.

### Recovery gate

Before production rollout:

- at least 90% of the fixed canary completes all five LLM stages;
- at least 80% reaches honest fab-ready output;
- p95 total LLM cost is at most $0.05 and no run exceeds $0.10;
- zero run silently changes to a model above the configured price class;
- zero commit rejection lacks an attributable gate/signature;
- no repeated rejection signature triggers another provider call;
- every failed run retains an inspectable candidate and one exact terminal cause.

After this recovery gate holds, run the larger statistical campaign already specified in `llm-stage-reliability-implementation-2026-09-02.md`. Do not claim >99% reliability from the small recovery canary.

## Implementation order

1. Phase 0 cost containment.
2. Phase 1 rejection attribution.
3. Finish and validate the architecture changes already in the working tree.
4. Phase 3 BOM, driven by newly visible rejection signatures.
5. Phase 4 wiring, driven by newly visible rejection signatures.
6. Phase 5 reasoning policy.
7. Phase 6 frozen replay, live canary, and deterministic build verification.
8. Deploy only after the recovery gate passes.

Do not combine provider policy, BOM repair, and wiring repair in one rollback unit. Their failure modes and verification evidence differ.

## Production rollout

1. Save the pre-rollout ledger report and configuration snapshot without secrets.
2. Deploy the smallest completed phase.
3. Install the editable server/design package if dependency metadata changed.
4. Restart with `deploy/restart-web.sh` and `deploy/restart-build-worker.sh`; do not use `systemctl`.
5. Verify `http://127.0.0.1:8080/` returns 200.
6. Verify `logs/kicraft_build_worker.log` ends with `[build-worker] ready`.
7. Run the fixed canary and compare cost, attempts, rejection signatures, stage completion, and fab-ready rate with the saved baseline.
8. Roll back the phase if any hard fabrication gate weakens or p95 cost/latency regresses.

## Relationship to existing plans

This plan does not replace the electrical/fabrication scope in `llm-stage-reliability-implementation-2026-09-02.md`. It adds the missing production recovery order and hard cost/rejection gates demonstrated by the current ledger.

It incorporates the still-relevant remaining work from `stage-resource-telemetry-and-bom-cost.md`, especially parts-block pre-filtering and combined resolution/vendoring. Actual production data now makes those BOM reductions urgent rather than optional.

`reasoning-loop-breaker.md` supplied detection and retry behavior; the current ledger shows detection alone is insufficient. The remaining decision is whether reasoning should be disabled on the first call for structured stages that repeatedly loop.
