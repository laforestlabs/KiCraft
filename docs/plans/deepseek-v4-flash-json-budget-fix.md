# DeepSeek V4 Flash JSON-Budget Fix

## Goal

Prevent reasoning-heavy design-stage calls from exhausting the completion budget before emitting the required slot JSON.

The fix must preserve:

- bounded per-call spend and latency;
- deterministic schema validation and commit gates;
- existing BOM tool-loop behavior;
- compatibility with models that do not expose a reasoning channel;
- explicit failure when a complete, valid slot cannot be produced.

## Evidence and diagnosis

The comparison is between:

- baseline: `logs/self_eval/20260728T120442Z`, model `deepseek/deepseek-v4-flash`;
- candidate: `logs/self_eval/20260805T213639Z`, model `deepseek/deepseek-v4-flash-0731`.

Both used 34 briefs, `minimax/minimax-m3`, and rubric version 2. The candidate had 17 design-stage failures versus 3 in the baseline. Fifteen candidate failures were recorded as `no JSON in reply`; most ended with `finish_reason=length` after long reasoning streams.

The relevant implementation is:

- `kicraft/server/client.py`: streams reasoning/content separately and sends `max_tokens`;
- `kicraft/server/stage_driver.py`: passes explicit per-stage caps, parses `res["text"] or res.get("reasoning")`, and currently responds to truncation by doubling the cap;
- `kicraft/server/config.py`: exposes global token and spend settings;
- `tests/test_stage_driver_retry.py` and `tests/test_client_provider.py`: existing retry and client-contract coverage.

The main failure is not solved by simply raising the same cap. The new model spends the larger completion budget on reasoning, leaving no complete JSON answer.

## Decision

Do **not** remove `max_tokens` in production.

Implement bounded, separate policies for:

1. reasoning budget;
2. answer/output budget;
3. retry mode after truncation.

When a response is length-terminated or has invalid JSON, switch to one bounded serialization retry with reasoning disabled or minimized. Do not repeatedly double the cap without changing the response policy.

## Options considered

### Option A: remove `max_tokens`

Do not implement as the production fix.

Reasons:

- `stage_driver.py` passes explicit stage caps, so removing only `KICRAFT_MAX_TOKENS_PER_CALL` does not remove the design-stage limit;
- an unbounded call makes per-run cost and latency unpredictable;
- BOM has multiple tool rounds and BOM/wiring have elevated retry counts;
- a provider may apply its own default cap;
- more reasoning still does not guarantee a final JSON object.

This may be used as a short diagnostic canary only, never as the normal runtime policy.

### Option B: raise the existing global cap

Useful as a diagnostic, but insufficient as the final fix. The candidate already reached 16k/32k output caps in several stages while still failing to emit JSON.

### Option C: stage/model-specific reasoning and answer budgets

Recommended. Add a policy layer that keeps the answer budget bounded while limiting reasoning separately. This is the primary fix.

### Option D: reasoning-light serialization retry

Recommended with Option C. A truncated attempt should be retried with reasoning disabled/minimized and a compact-output instruction, rather than merely receiving a larger cap.

### Option E: provider JSON-schema response format

Recommended as a second layer where the selected provider supports it. Retain a textual JSON fallback because provider support is not uniform.

### Option F: split analysis and serialization into separate calls

Defer unless the bounded-reasoning approach is insufficient. It can separate thinking from serialization, but doubles calls and complicates state handling.

### Option G: rollback or route structured stages to the old model/provider

Keep as an operational rollback option during rollout. It is not the permanent fix.

## Implementation plan

### 1. Add usage and truncation telemetry

Files:

- `kicraft/server/client.py`
- the spend-ledger metadata path used by the client/stage driver
- relevant telemetry tests

Changes:

- return usage details from `chat()` and `chat_with_tools()` instead of returning an empty `usage` object;
- preserve `finish_reason`;
- record requested `max_tokens`;
- record the selected reasoning policy;
- record prompt/completion token counts;
- record reasoning/content token counts when the provider supplies them;
- distinguish `finish_reason=length`, invalid JSON after a normal stop, and transport/provider errors.

Do not add raw model reasoning to spend metadata. Existing transcript persistence remains the source for detailed debugging.

Acceptance:

- each stage ledger record identifies the cap, reasoning policy, token usage, and termination reason;
- existing clients that do not return usage continue to work with null/zero-safe handling.

### 2. Add model/stage response policies

Files:

- `kicraft/server/config.py`
- `kicraft/server/stage_driver.py`
- `.env.example` if new environment overrides are exposed

Introduce a policy abstraction rather than another global integer. The policy should contain at least:

```python
StagePolicy(
    answer_max_tokens=...,
    reasoning=...,
)
```

Select a policy by model/stage, with safe defaults for models that do not support reasoning controls.

Initial pilot values should be configurable rather than hard-coded as an irreversible decision:

| Stage | Reasoning policy | Initial answer-cap range |
|---|---|---:|
| `intent` | disabled or low effort | 8k |
| `functional_spec` | disabled or 1–2k | 8k |
| `architecture` | disabled or 2–4k | 12–16k |
| `bom` | low/2–4k | 32k |
| `wiring` | low/2–4k | 32k |

The implementation must test provider semantics for `reasoning={"max_tokens": ...}`. If a provider does not support that form, use a supported low-effort setting or disable reasoning for the serialization-critical stage.

Preserve the existing higher retry floors for BOM and wiring. Make the existing `_STAGE_MIN_TOKENS` policy-driven instead of allowing retry behavior to be the only budget mechanism.

### 3. Propagate reasoning policy through all client paths

Files:

- `kicraft/server/client.py`
- `kicraft/server/stage_driver.py`
- `tests/test_client_provider.py`
- `tests/test_client_tool_loop.py`

`client.chat()` already accepts `reasoning`; ensure the stage driver passes it. Extend `chat_with_tools()` to accept and include the same policy on every round, including the forced final JSON round.

Do not change behavior for callers that omit a reasoning policy.

Verify that the provider payload contains:

- bounded `max_tokens`;
- reasoning policy when configured;
- no internal `_meta` fields sent to the provider.

### 4. Replace cap-only truncation retry

File:

- `kicraft/server/stage_driver.py`

Current behavior doubles `cur_max_tokens` after `finish_reason=length`. Replace it with a bounded response-mode transition:

1. normal stage policy;
2. one serialization retry with reasoning disabled/minimized, compact JSON instructions, and no unnecessary prior reasoning transcript;
3. one final bounded serialization attempt if required;
4. explicit `truncated_json` failure if no valid slot commits.

The serialization retry must:

- say `Output ONLY the complete slot JSON`;
- prohibit markdown and explanatory prose;
- use compact single-line entries where the stage already supports them;
- preserve all valid entries after a commit rejection;
- retain the existing BOM reconcile path and wiring offender feedback.

Do not accept an incomplete JSON prefix or silently repair/drop missing BOM parts, nets, or pins.

Keep retry costs bounded. A retry must not reset into an unbounded loop or grow past the configured provider-safe maximum.

### 5. Add structured output where supported

Files:

- `kicraft/server/client.py`
- `kicraft/server/stage_driver.py`
- provider/client tests

Investigate `response_format` with a JSON schema generated from the existing stage models. Use it for stages/providers that support it. Retain the current textual parser as fallback.

Structured output is additive; it does not replace the reasoning budget because a model can still fail to reach its answer section before the cap.

### 6. Add tests

#### Stage-driver tests

Extend `tests/test_stage_driver_retry.py` with deterministic fake clients for:

- long reasoning plus empty content plus `finish_reason="length"`, followed by valid JSON;
- valid JSON only after the serialization fallback;
- repeated truncation terminating with `truncated_json`;
- per-stage policy selection;
- model-specific policy override;
- no unbounded cap growth;
- preservation of existing BOM reconcile and wiring retry behavior.

#### Client tests

Extend `tests/test_client_provider.py` and `tests/test_client_tool_loop.py` with:

- reasoning policy propagation for plain calls;
- reasoning policy propagation for tool calls and final tool round;
- usage/finish metadata propagation;
- compatibility when reasoning is absent;
- bounded payload values.

#### Regression fixtures

Add compact deterministic fixtures based on the observed failure modes:

- architecture response that reasons until the cap;
- BOM/tool-loop response that reaches the cap before final JSON;
- wiring response that emits a partial list before truncation.

Do not make tests depend on OpenRouter or live model output.

### 7. Run a pinned-provider canary

Before a full self-eval, run the new model against representative briefs:

- one architecture-heavy case;
- one BOM/tool-heavy case;
- one wiring-heavy case;
- one simple/shaped case.

Pin the provider during the canary so model behavior and provider behavior are not conflated.

Compare at least:

- design-stage completion;
- `no JSON` count;
- `finish_reason=length` count;
- reasoning/output token usage;
- per-run cost and latency;
- number of retries;
- schema/commit failures.

### 8. Full-batch rollout

Run the full 34-brief self-eval with the same judge and rubric as the comparison batch.

Targets:

- zero `no JSON in reply` failures caused by truncation;
- design completion at least the old baseline's 31/34;
- no unbounded calls or retry loops;
- spend increase explicitly measured and accepted;
- separate reporting of model-stage failures versus place/route failures.

Keep the old model/provider available as a rollback until the full batch meets these targets.

## Observability requirements

The saved run/ledger data should make these queries possible:

- Which stage/model/provider hit `finish_reason=length`?
- What answer cap and reasoning policy were active?
- Did the response contain any answer tokens?
- How many retries were consumed?
- Did the serialization fallback succeed?
- What did the failure cost?

Add a concise stage failure label such as:

- `truncated_json`
- `invalid_json`
- `schema_rejected`
- `provider_error`
- `commit_rejected`

This is preferable to collapsing all parser failures into `no JSON in reply`.

## Non-goals

- Do not weaken schema validation.
- Do not accept partial BOM/wiring JSON.
- Do not silently discard reasoning or content needed for debugging.
- Do not remove global daily/total spend guards.
- Do not attribute place/route regressions to the model without a pinned-provider replay.

## Definition of done

The implementation is complete when:

1. the new model can complete representative intent, architecture, BOM, and wiring stages without exhausting the answer budget;
2. truncation triggers a bounded reasoning-light serialization retry;
3. all client paths, including BOM tools, propagate the policy;
4. telemetry records caps, usage, policies, and finish reasons;
5. deterministic unit tests cover truncation, fallback, propagation, and bounded retries;
6. the pinned-provider canary passes with no truncation-induced JSON failures;
7. the full 34-brief batch reaches at least 31/34 design-complete runs or documents a non-truncation blocker.
