# Reasoning Loop Breaker

## Goal

Stop design-stage LLM calls from burning hundreds of thousands of tokens in a
stuck reasoning loop (e.g. `KC-VWW5X7`: the `functional_spec` stage re-derived
the same "should this be an assumption?" decision verbatim for thousands of
tokens and never emitted the slot JSON). The fix must:

- **prevent** the loop from starting;
- **recognize** it in-flight (mid-stream, not after the completion ends);
- **break out** of it (abort + retry with a loop-proof policy), or fail with an
  honest `reasoning_loop` label.

It must preserve the existing bounded spend/latency, schema/commit gates, BOM
tool-loop, and mock/load-test surface.

## Root cause

1. **Reasoning is unbounded.** `drive_stage` calls `client.chat()` with no
   `reasoning=` arg; `chat()` only forwards it when truthy. For DeepSeek-served
   models OpenRouter's `max_tokens` bounds the *answer* channel only — the
   reasoning channel is uncapped. A `KICRAFT_MAX_TOKENS_PER_CALL=4096` run can
   therefore emit "hundreds of thousands" of reasoning tokens.
2. **Greedy decoding reproduces the loop.** `design_temperature` defaults to
   `0.0`; a deterministic loop repeats identically every retry.
3. **The only defense is post-hoc.** `_json_failure_recovery` acts on
   `finish_reason=="length"` *after* the provider finally cuts the stream off,
   which is exactly the spend the user pays for.

The prior `deepseek-v4-flash-json-budget-fix.md` plan applied reasoning budgets
only to the review/judge path; the design stages still run reasoning unbounded,
greedily, with a reactive-only handler.

## Design (three layers)

### Layer 1 — Prevent

Pass an explicit reasoning policy on design-stage calls. Add
`Settings.design_reasoning(stage)` mirroring `review_reasoning()`:

- `intent` / `functional_spec` → `{"enabled": False}` (small, serialization-
  critical, the observed loop site).
- `architecture` / `bom` / `wiring` → `{"max_tokens": design_reasoning_tokens}`
  (default 2048); `0` disables for all stages.

Thread it through `drive_stage` into `chat` and `chat_with_tools` exactly like
the review path already does.

**Provider-compat gate:** the bakeoff already found models that 400 on
`reasoning={"max_tokens": N}` and accept only `{"effort": …}`. A pinned canary
must confirm which form `deepseek/deepseek-v4-flash` accepts. Layer 2 is
provider-independent and is the guaranteed breaker even if the policy is ignored.

### Layer 2 — Recognize (in-stream, in `_stream`)

`client.py:_stream` is the only place tokens arrive; detection lives there.
Three signals feed one abort path, all gated on "no answer content yet":

1. **Hard ceiling** — `reasoning_chars > reasoning_max_tokens * 4` (default
   4096 tokens ≈ 16k chars). Turns an unbounded loop into a bounded event.
2. **Verbatim repetition** — the trailing `reasoning_repeat_window` (256) chars
   appear ≥ `reasoning_repeat_threshold` (3) times in a bounded recent buffer.
   Catches the verbatim-block loop before the ceiling.
3. **Wall-clock stall** — reasoning flowing, no content, past
   `request_timeout_s` (120s). Independent of provider-reported token counts.

On abort: `break` out of `iter_lines` (NOT the network-retry path), close the
response, set `finish_reason="reasoning_loop"` + `loop_detected=True`, and
record an *estimated* cost from partial chars (no final usage chunk arrives, so
`record()` must not under-count — use `estimate_cost`).

### Layer 3 — Break out (stage driver)

- `chat` / `chat_with_tools` propagate `loop_detected` + the synthetic
  `finish_reason="reasoning_loop"`.
- In `drive_stage`, a `loop_detected` branch (before `_extract_json`) retries
  with reasoning disabled, temperature raised (`+0.4`), and the existing
  `_REASONING_LOOP_RETRY_MSG`. Bounded by `_MAX_LOOP_RETRIES = 1`; a second
  loop fails with an explicit `reasoning_loop` error (not `no JSON in reply`).
- In `chat_with_tools`, a loop in any tool round sets `force_final` + disables
  reasoning, so BOM can't loop during part resolution.

## Config additions (`config.py` + `.env.example`)

| Setting | Default | Purpose |
|---|---|---|
| `design_reasoning_tokens` | `2048` | reasoning budget for arch/bom/wiring (0 = off everywhere); intent/functional_spec always off |
| `reasoning_max_tokens` | `4096` | hard per-call reasoning ceiling (signal 1) |
| `reasoning_repeat_window` | `256` | repetition fingerprint window (signal 2) |
| `reasoning_repeat_threshold` | `3` | repetition fingerprint count (signal 2) |
| (reuse) `request_timeout_s` | `120` | content-stall timer (signal 3) |

All env-tunable with the above safe defaults; no live-model dependency.

## Telemetry

- New failure label `reasoning_loop` (add to the `truncated_json` /
  `invalid_json` / … vocabulary).
- Record `reasoning_chars` and `loop_detected` in the spend-ledger `rec_meta`
  and in `stage_status` (`_stamp_stage_status` gains an `error` field).
- Emit a `retry` progress event with a loop-specific message so the Thinking
  window shows "reasoning loop detected — retrying with reasoning disabled"
  instead of freezing mid-stream.

## Tests (deterministic, no OpenRouter)

- `tests/test_client_provider.py` — fake `_FakeResp`/chunk harness: reasoning-
  only stream past ceiling → aborts `reasoning_loop` + cost recorded; repeated
  256-char window → early abort; content-stall timer; content-present stream
  must NOT abort; `loop_detected` propagates through `chat()`.
- `tests/test_stage_driver_retry.py` — fake client returning
  `finish_reason="reasoning_loop"` then valid JSON → commits with reasoning
  disabled; second loop → fails `reasoning_loop`; per-stage `design_reasoning`
  selection; `chat_with_tools` loop → forced final.
- Regression fixture: replay the `KC-VWW5X7` functional_spec block and assert
  abort within the ceiling.

## Rollout / verification

1. Pinned-provider canary on the 4 brief classes: confirm the reasoning-control
   form, measure `finish_reason=length` / `reasoning_loop` counts, cost, latency.
2. Full 34-brief self-eval: zero `reasoning_loop` / truncation failures, ≥
   old-baseline design completion, no unbounded calls.
3. Rollback knobs: `KICRAFT_DESIGN_REASONING_TOKENS` + `KICRAFT_MODEL`.

## Non-goals

- Don't weaken schema validation or accept partial JSON.
- Don't remove the daily/total spend guards.
- Don't route design stages to a pricier model as the primary fix — Layer 2
  makes the cheap model safe; a model swap is a fallback, not the plan.
