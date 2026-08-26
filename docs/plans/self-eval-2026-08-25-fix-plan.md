# Self-eval 2026-08-25 — reliability recovery plan

**Status:** proposed
**Source batch:** `/home/kicraft/.kicraft/self_eval/20260825T033602Z`
**Comparison batch:** `logs/self_eval/20260823T133726Z`
**Models:** design `deepseek/deepseek-v4-flash`; judge `minimax/minimax-m3`
**Rubric:** version 2

## Decision

Fix evaluator/model-call reliability before changing routing, electrical gates, prompts for individual briefs, or the rubric.

The batch does not show a broad scoring regression among comparable graded runs. It shows three narrower control-path failures:

1. the global reasoning-loop breaker is aborting ordinary judge reasoning and withholding four grades;
2. seven BOM calls ignore the new 500-part contract, emit 651–854 part records, and truncate twice;
3. every terminal wiring failure follows a reasoning-loop abort, after which correction retries repeatedly preserve the same invalid topology.

Routing is not the current bottleneck. Every design that completed reached a built board: 13/14 were fab-ready and 1/14 was DRC-failing. Do not spend this fix wave on placement/routing.

## Batch result

| Metric | 2026-08-23 | 2026-08-25 | Delta |
|---|---:|---:|---:|
| briefs | 34 | 34 | — |
| design completed | 19 | 14 | −5 |
| graded | 34 | 30 | −4 |
| fab-ready | 17 | 13 | −4 |
| fab-ready among completed designs | 89.5% | 92.9% | +3.4 pp |
| mean score | 62.4 | 61.4 | −1.0 |
| median score | 72.0 | 61.0 | −11.0 |
| total cost | $1.6419 | $2.0478 | +$0.4059 / +24.7% |
| design cost | $1.3388 | $1.0791 | −$0.2597 |
| judge cost | $0.3031 | $0.9687 | +$0.6656 |
| wall time | 10,479.5 s | 8,844.3 s | −15.6% |

The headline median is composition-sensitive because four grades are missing and five fewer designs completed. Across the 30 briefs graded in both batches, the mean per-brief delta is −1.1 points and the median delta is **+0.5**. Single-run changes remain noisy: eight comparable briefs moved up by at least 10 points and ten moved down by at least 10. Do not tune the rubric or a brief-specific prompt from these one-sample score deltas.

## Evidence and root causes

### A. The reasoning-loop guard is globally scoped

`kicraft/server/client.py:_stream` invokes `_reasoning_loop(...)` for every streamed call. The guard was designed for design stages, but it also runs for `phase=eval_judge` and `phase=electrical_review`.

The spend ledger recorded **61** loop aborts during this batch:

| phase / stage | aborts |
|---|---:|
| BOM tool/final calls | 20 |
| wiring | 17 |
| eval judge | 16 |
| architecture | 6 |
| electrical review | 2 |

Most aborts occur at 16,385–16,392 reasoning characters: the configured provider-independent 4,096-token ceiling multiplied by four. The 16 judge aborts are not evidence of a repeated design-stage loop. The repository explicitly documents that `minimax-m3` commonly uses 10k–23k reasoning tokens before its JSON verdict, while the generic client guard cuts it near 4k tokens.

Four briefs lost their grade because both judge attempts returned no answer after loop aborts:

- `usb-pd-trigger` — judge cost $0.112320;
- `led-cc-driver` — judge cost $0.115650;
- `esp32-dual-motor` — judge cost $0.095400;
- `daq-8ch` — judge cost $0.091260.

This explains the spend regression: judge cost rose from $0.3031 to $0.9687 and consumed 47.3% of total batch spend. Raising `eval_judge_max_tokens` cannot fix a separate client-side abort threshold.

### B. BOM degeneration remains terminal despite the parse-side bound

Twelve runs failed at BOM. Seven ended as `truncated JSON at the output token limit`; three ended as `no JSON in reply`; two failed programming/part-resolution gates.

The seven truncated serialization responses were not legitimate large boards:

| brief | final response chars | emitted `ref` fields before truncation |
|---|---:|---:|
| stm32-min | 82,360 | 679 |
| rp2040-min | 97,036 | 789 |
| nrf52-beacon | 93,868 | 795 |
| esp32-dual-motor | 93,582 | 810 |
| can-node | 82,871 | 651 |
| daq-8ch | 93,835 | 854 |
| stepper-a4988 | 98,137 | 814 |

The tails are long runs of invented passives such as sequential 10 kΩ resistors or 100 nF capacitors. No response reached `placement_hints`; each was cut inside `parts`.

`docs/plans/bom-emission-bounds.md` correctly added a canonical 500-total/450-per-sheet contract and a parse-side cardinality gate. It also states its remaining limitation: truncated JSON cannot reach that gate and no in-stream content breaker exists. The full corpus now proves that limitation is common, not hypothetical: 7/34 runs crossed the bound without being interrupted.

The serialization retry is reasoning-disabled but retains deterministic temperature 0.0. In these seven runs it commonly reproduced a larger version of the same degenerate sequence. A fixed larger token cap would only buy more bad output.

### C. Wiring recovery is correlated with terminal invalid topology

Eight runs failed at wiring:

- multi-net pin shorts: `r2r-dac`, `rs485-terminal`, `esp32-s3-sensor`, `dual-rail-supply`, `audio-jack-buffer`;
- two-terminal self-short: `lora-node`;
- dangling signal nets: `encoder-oled-panel`, `proto-shield`.

All eight first triggered `reasoning loop detected — retrying with reasoning disabled`. Sixteen runs total triggered the wiring loop recovery; eight committed and eight failed. No wiring run without a loop abort ended in a terminal wiring failure in this batch.

The terminal cases then repeated the same offender family through correction attempts. Examples:

- `r2r-dac`: the same four multi-net resistor pins persisted through four commit retries;
- `audio-jack-buffer`: the same twelve multi-net connector/codec pins persisted through four retries;
- `lora-node`: the same two-terminal resistor self-short persisted through three retries;
- `proto-shield`: three repeated multi-net-pin failures changed into a dangling `CE` net on the last attempt.

The deterministic gates are working: they prevent electrical shorts from reaching synthesis. Weakening §9.15, §9.17, or §9.19 would convert honest failures into bad boards and is prohibited.

### D. Routing and outline handling are downstream casualties, not the priority

The fresh batch produced 13 fab-ready boards and one DRC-failing board from 14 completed designs. There were no completed designs that failed to enter the build. The absolute fab-ready count fell because six more designs died before build, not because the router regressed.

Likewise, shaped-outline output fell from 4/6 to 3/6 fab-ready because `round-led-ring`, `rounded-c3-devboard`, and `snowman-ornament` died at BOM. This batch provides no basis for an outline or KRT change.

## Scope

### In scope

- per-call reasoning-loop policies and explicit abort reasons;
- reliable Class-J completion with bounded judge reasoning;
- in-stream enforcement of configured collection bounds;
- a non-deterministic, bounded serialization recovery after detected degeneration;
- measured wiring-policy and retry-history A/Bs on the frozen failing cohort;
- telemetry that separates hard ceiling, repetition, and wall-stall aborts.

### Out of scope

- raising output caps;
- accepting or salvaging truncated JSON;
- weakening BOM/wiring/electrical gates;
- automatic choice of a “correct” net when a pin appears on two nets;
- router, placer, board-outline, or DRC changes;
- rubric reweighting or brief-specific score tuning;
- making `minimax-m3` the normal design model;
- special-casing any of the 34 slugs.

## Phase 1 — make reasoning guards call-specific

### Change

1. In `kicraft/server/client.py`, replace the implicit settings-wide reasoning guard with an explicit per-call policy passed through `chat` and `chat_with_tools` to `_stream` as internal control data. The policy must carry:
   - hard reasoning-token ceiling;
   - repetition detection enabled/disabled and its window/threshold;
   - wall-stall ceiling;
   - a stable policy name for telemetry.
2. Keep the existing design-stage policy as the default supplied by `stage_driver.drive_stage`; do not infer policy from free-form `meta.phase` strings.
3. Give `grade_class_j` a judge policy sized for its documented 10k–23k reasoning range. Preserve a finite hard ceiling, but disable the design-stage repetition heuristic unless a judge-specific canary proves it has no false positives.
4. Give electrical review its own policy derived from review settings rather than the design-stage 4,096-token fallback.
5. Record `loop_abort_reason` (`hard_ceiling`, `repetition`, or `wall_stall`) and `reasoning_policy_name` in spend metadata. Keep `loop_detected` for compatibility.
6. In `kicraft/eval/judge.py`, treat a synthetic loop abort as a distinct retry defect. The second attempt remains bounded and fail-closed; never report it as generic “no JSON object found” when the client knows the abort reason.

### Files

- `kicraft/server/client.py`
- `kicraft/server/config.py`
- `kicraft/server/stage_driver.py`
- `kicraft/eval/judge.py`
- `kicraft/eval/run_web.py`
- `tests/test_client_provider.py`
- `tests/test_stage_driver_retry.py`
- judge tests following the existing `kicraft/eval/judge.py` test location

### Acceptance

- A synthetic design reasoning stream still aborts at the design ceiling and retries exactly once.
- A synthetic 10k–23k-token judge reasoning stream followed by valid JSON is not aborted by the design policy.
- Judge/review calls always retain a finite ceiling.
- Hard-ceiling, repetition, and wall-stall fixtures produce distinct telemetry.
- Re-grade the 34 frozen digests without rerunning design/build. All 34 receive valid Class-J verdicts; no grade is withheld because of the 4,096-token design ceiling.
- Frozen-digest judge spend returns near the prior batch: target ≤$0.45 total, with no relaxation of verdict validation.

## Phase 2 — stop BOM overflow in-stream and make recovery escape determinism

### Change

1. Add a generic streaming collection guard driven by the existing immutable `CollectionBound` policy. It must lex JSON incrementally across arbitrary chunk boundaries, ignore braces inside strings/escapes, locate the configured top-level array, and count direct members without retaining another full copy of the response.
2. Pass the guard from `stage_driver` into the client for slot-producing responses only. Reset it for every provider call/tool round. The HTTP client owns interruption; it must close the stream once the next item would exceed the configured bound.
3. Return a synthetic `finish_reason="collection_limit"` with the field, observed lower-bound count, configured total, emitted content chars, and cost. Do not classify it as transport failure or truncated JSON.
4. Route the first `collection_limit` through the one existing serialization-recovery budget. State the exact observed overflow and canonical limits. A second overflow is terminal `collection_limit`; never increase caps or invoke another tool loop.
5. Run serialization recovery at an explicit escape temperature (candidate 0.4), not the same deterministic 0.0 that reproduced all seven runaway sequences. Reasoning remains disabled and the fixed serialization token cap remains unchanged.
6. Preserve the parse-side §9.35 cardinality gate. The stream guard is a cost/recoverability boundary; the parsed-object gate remains the correctness authority.
7. Preserve legitimate array designs. The guard enforces the existing 500-total/450-per-sheet policy, not an ad hoc smaller global limit.

### Files

- `kicraft/server/client.py`
- `kicraft/server/stage_driver.py`
- `kicraft/server/config.py` only if the recovery temperature becomes a named policy value
- `kicraft/server/spend_guard.py` for the new terminal classification
- `kicraft/design/models.py` for the stable failure-kind vocabulary
- `tests/test_client_provider.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_stage_resource_telemetry.py`
- existing collection-bound tests in `tests/test_kicraft_validation.py` and `tests/test_kicraft_stage_cli.py`

### Acceptance

- Chunk-split, escaped-string, nested-object, empty-array, tool-call, and malformed-JSON fixtures cannot miscount direct collection members.
- A 501st BOM part aborts before its full object is buffered; spend and telemetry include the partial response.
- A documented 400-part LED-array-style BOM remains accepted.
- A parseable 501-part response still fails §9.35 if it bypasses streaming through a mock/non-streaming client.
- Replay the frozen pre-BOM state for the seven truncated briefs with the current path and the candidate recovery path. Candidate target: zero terminal `truncated_json`, zero responses over the stream bound, and no cap increase.
- If temperature 0.4 does not improve valid commits, ship the stream safety boundary but do not claim a completion fix; run a separate model canary rather than adding retries.

## Phase 3 — measure wiring reasoning and reset stalled corrections

Do not guess that “more reasoning” or “less reasoning” is better. The batch supports a controlled frozen-state comparison.

### Experiment

Replay the 16 pre-wiring workspaces that triggered a wiring loop abort:

- control: current 2,048-token reasoning policy plus loop recovery;
- candidate: reasoning disabled from the first wiring call;
- same model, temperature, prompt, commit gates, retry budget, and spend ceiling.

Measure committed count, first-pass commit count, terminal gate family, attempts, reasoning/content characters, cost, and wall time. Run at least three repeats for the eight terminal-failure states before treating a per-brief flip as real.

### Conditional implementation

Adopt reasoning-disabled wiring only if it improves cohort completion without increasing accepted electrical defects. Otherwise retain bounded reasoning.

Independently, add correction no-progress handling in `stage_driver`:

1. Normalize the ordered gate IDs plus offender identities from each commit rejection.
2. If the same signature occurs twice, stop appending another full invalid slot to history.
3. Spend at most one remaining retry from pristine base messages plus the latest concrete feedback, reasoning disabled, and escape temperature 0.4.
4. If the signature persists, terminate honestly as `commit_rejected`. Do not add retries and do not auto-delete nets.

### Files

- `kicraft/server/config.py`
- `kicraft/server/stage_driver.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_kicraft_validation.py` only for unchanged-gate regression coverage

### Acceptance

- The replay report names both arms and retains every §9 failure; no invalid wiring is committed.
- Adoption gate for upfront reasoning disable: at least 12/16 committed in the frozen cohort and no worse terminal correctness signature than control. If the gate misses, keep the current initial policy.
- Repeated identical rejection signatures trigger exactly one clean-slate correction, then terminate; attempts never exceed the existing budget.
- Existing park/reconcile behavior is unchanged.

## Phase 4 — rebaseline only after reliability is restored

1. Run focused deterministic suites once after implementation.
2. Re-grade the frozen 34 digests to verify the judge independently of design noise.
3. Run the seven-brief BOM cohort and 16-brief wiring cohort before paying for a full batch.
4. Run one full 34-brief self-eval with the same design model, judge, rubric, parallelism, and build slots.
5. Compare reliability and cost to both source batches. Treat score movement as descriptive unless a ≥3-repeat cohort supports it.

### Full-batch gates

- 34/34 judge verdicts valid;
- zero terminal `reasoning_loop` caused by applying a design policy to judge/review;
- zero terminal BOM `truncated_json` from collection overflow;
- design completion at least 19/34 (the 2026-08-23 baseline);
- all completed designs enter build;
- no reduction in conditional fab-ready rate below the 2026-08-23 89.5% baseline;
- total cost ≤$1.75, with judge cost ≤$0.45;
- no gate suppression and no partial/truncated slot acceptance.

## Follow-up triage after the control-path fixes

Only after Phase 4 should the remaining honest design defects be ranked. Current examples are:

- MCU programming access: `nrf52-beacon`, `rounded-c3-devboard`, `snowman-ornament`;
- silent substitution: `dual-rail-supply`;
- unresolved symbols/footprints: `snowman-ornament`;
- persistent multi-net/dangling-net topology failures from the wiring cohort.

These may justify a later general prompt, part-library, or gate-feedback change. They do not justify slug-specific repairs, and the current batch cannot separate model variance from a product-level defect until the reasoning and serialization control paths are stable.

## Verification commands

Focused deterministic suite:

```bash
.venv/bin/python -m pytest -q \
  tests/test_client_provider.py \
  tests/test_stage_driver_retry.py \
  tests/test_stage_resource_telemetry.py \
  tests/test_kicraft_validation.py \
  tests/test_kicraft_stage_cli.py
```

Full evaluation, after cohort gates pass:

```bash
.venv/bin/python -m kicraft.eval.self_eval
```

Keep the new batch directory immutable. Compare from `summary.json`, per-run `eval/report.json`, `events.jsonl`, and the spend ledger; do not infer root cause from `summary.md` scores alone.
