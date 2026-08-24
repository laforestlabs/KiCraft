# BOM-stage programming and JSON reliability plan

**Status:** reviewed, implementation-ready plan
**Source run:** `KC-7FVTPW` (`/home/kicraft/.kicraft/projects/1/743`)
**Scope:** fix design-stage response contracts, JSON recovery, and the BOM programming prompt. Do not edit or repair the failed board.

## Problem

`KC-7FVTPW` never reached schematic synthesis, ERC, placement, or routing. The BOM stage consumed five attempts and ended with `no JSON in reply`.

The first parseable BOM had three model defects:

1. The ESP32-S3 had no BOOT plus EN/RESET recovery path.
2. The architecture named `ESP32-WROOM-32`, but the BOM selected `ESP32-S3-MINI-1-N8` without a `bom.substitutions` entry.
3. Q1 used nonexistent `Device:Q_NPN_BEC` instead of a resolvable `Transistor_BJT` symbol.

The existing commit gates correctly rejected all three. Subsequent completions were malformed or unfinished. Two defects then prevented useful recovery:

- `kicraft/server/stage_driver.py::_json_failure_recovery()` treats parse failure as another ordinary stage attempt. A truncated answer doubles `cur_max_tokens` up to 32,768, while other malformed output retries with the same policy. It has no explicit, tool-free serialization state.
- `.claude/skills/kicraft/stages/bom.md` says native-USB ESP32 designs need “nothing extra ... beyond the MCU itself,” contradicting the §9.29 BOOT/EN/RESET or strap-access gate.

Cross-run evidence is broad enough to justify a pipeline fix: at least 23 of 222 scanned run artifacts have design-stage `no JSON in reply` outcomes, and 14 have the ESP32 strap-gate failure.

## Review corrections to the original plan

The implementation must account for these repository facts:

1. **Reasoning controls already exist.** `Settings.design_reasoning()` disables reasoning for `intent` and `functional_spec` and caps the other design stages at `design_reasoning_tokens` (currently 2,048). This work should extend that path, not create a parallel policy system.
2. **Reasoning-loop recovery already exists.** The client emits `finish_reason="reasoning_loop"`; the stage driver retries once with reasoning disabled. Empty content with `finish_reason="length"` is a separate provider-exhaustion signature and must not be mislabeled `invalid_json`.
3. **Usage is currently discarded.** `_stream()` receives provider usage but returns only `(message, cost)`, and `chat()` currently returns `usage: {}`. Metadata work therefore requires changing the internal completion result, not only forwarding existing fields.
4. **`stage_runs` cannot store a failure class.** `SpendGuard.record_stage()` and its SQLite table have no error/metadata column. Persisting terminal classifications requires a backward-compatible schema migration and report/test updates.
5. **A serialization retry must be tool-free in code, not only in prose.** Re-entering `chat_with_tools()` still permits more tool rounds and reasoning. Parse recovery must call `chat()` with no tools and reasoning disabled.
6. **Retry counters currently mix failure types.** Commit corrections, clarification handling, reasoning-loop recovery, and parse recovery all consume the same `for attempt` budget. The new state machine must state which retries are available and report actual calls rather than the configured maximum.
7. **The replay harness already creates a fresh temporary workspace.** Verification should use `drive_replay`/the `replay` command directly; no manual copying or target mutation is needed.

## Goals

- Keep schema, symbol, footprint, pin, stock, substitution, and programming-path gates hard.
- Give each design-stage request a finite output cap and a provider-compatible reasoning policy.
- Use at most one tool-free, reasoning-disabled serialization retry for a parse failure.
- Distinguish terminal `truncated_json`, `invalid_json`, `reasoning_loop`, `commit_rejected`, and provider/transport failures.
- Make the BOM prompt agree with §9.29 before the first BOM attempt.
- Preserve bounded spend, BOM tool use, commit-correction behavior, and clients/mocks that do not expose reasoning controls or usage.

## Non-goals

- Do not weaken §9.29, §9.33, symbol, footprint, pin, or stock gates.
- Do not salvage a JSON prefix, accept a partial slot, or silently drop fields or list entries.
- Do not add downstream synthesis, wiring, or review masks for BOM defects.
- Do not change or replay in place `/home/kicraft/.kicraft/projects/1/743`.
- Do not add a symbol-library workaround for Q1; early symbol rejection is correct.
- Do not redesign transport retry policy or catch `BudgetExceeded` as a model failure.

## Design

### Failure taxonomy

Use one stable field, `failure_kind`, in returned stage results, `stage_status[stage]`, and `stage_runs` metadata:

| `failure_kind` | Meaning | Eligible recovery |
|---|---|---|
| `reasoning_loop` | client in-stream loop detector aborted twice, or an empty `length` completion repeated after reasoning was disabled | one existing reasoning-disabled retry only |
| `truncated_json` | answer content exists, `finish_reason="length"`, and no complete JSON object parses | one serialization retry |
| `invalid_json` | nonempty answer ends normally but is not one complete JSON object, or answer is empty after a normal stop | one serialization retry |
| `commit_rejected` | JSON parsed, but `_commit()` rejected the slot | existing commit-correction attempts |
| `provider_error` | provider HTTP/API failure after transport retries | none in JSON recovery |
| `transport_error` | stream/network failure after client transport retries | none in JSON recovery |

Keep the human-readable `error` field for UI compatibility, but derive it from `failure_kind`; do not use free-form error strings as the durable classification. `BudgetExceeded` remains a budget failure owned by the existing guard/caller path.

A parser finding a complete object followed by non-whitespace is `invalid_json`, not success. The current `_extract_json()` behavior must be tested before changing it; tighten it only if it accepts trailing prose or a second object.

### Retry state machine

Separate three bounded recovery paths:

1. **Reasoning recovery:** preserve the existing one-retry limit. Disable reasoning and use the existing anti-loop instruction. An empty `finish_reason="length"` follows this path even if the client loop detector did not fire.
2. **Serialization recovery:** after the first `truncated_json` or `invalid_json`, make exactly one plain `client.chat()` call:
   - system prompt remains the stage prompt;
   - user content remains the pristine stage task/state/extras plus a short serialization instruction;
   - do not resend the BOM tool transcript;
   - do not expose tools;
   - pass `reasoning={"enabled": false}`;
   - require one compact, complete slot object with no markdown or prose;
   - use the policy’s fixed serialization cap; never double it dynamically.
3. **Commit correction:** when parsing succeeds and `_commit()` rejects, preserve the current lean retry containing the complete candidate and `_retry_feedback()` offender aggregation. BOM reconcile and wiring unknown-reference behavior must remain unchanged. This is not serialization recovery and may use the normal stage/tool policy.

Serialization recovery is available once per `drive_stage()` invocation, not once per commit-correction attempt. A commit-correction response that is malformed may consume that one serialization retry. If serialization produces parseable but invalid slot JSON, `_commit()` still gets the result and may use remaining commit-correction attempts.

Track actual provider calls and each recovery kind independently. Terminal `attempts` must reflect calls made, not always `max_retries + 1`.

### Response policy

Extend the existing settings path with one immutable stage policy value containing:

- normal `max_tokens`;
- normal reasoning payload (`None`, disabled, or bounded);
- serialization `max_tokens`;
- serialization retry count (fixed at one for design stages).

Keep `Settings.design_reasoning(stage)` as the compatibility source for reasoning. `_stage_max_tokens()` and existing caller-provided higher caps remain supported for normal calls. Do not add a second global output-cap environment variable that overrides every stage.

Initial defaults should preserve known-good normal limits rather than immediately raising every call:

| Stage | Normal output floor | Normal reasoning | Serialization cap |
|---|---:|---|---:|
| `intent`, `functional_spec` | caller default (currently 4,096) | disabled | 8,192 |
| `architecture` | caller default | existing bounded setting | 16,384 |
| `bom` | 16,384 | existing bounded setting | 32,768 |
| `wiring` | 8,192 | existing bounded setting | 32,768 |

Every request, including every BOM tool round and the forced final round, must carry a finite `max_tokens`. Providers or legacy mocks without reasoning support receive no unsupported field on normal calls; serialization mode still requests disabled reasoning where supported, with omission as the compatibility fallback.

## Implementation

### 1. Add policy and explicit recovery state

**Owners:** `kicraft/server/config.py`, `kicraft/server/stage_driver.py`

1. Introduce the stage response policy using the existing `design_reasoning()` compatibility path and token floors.
2. Replace `_json_failure_recovery()`’s tuple/cap-doubling API with classification plus an explicit recovery decision.
3. Refactor `drive_stage()` so reasoning recovery, the single serialization call, and commit corrections have separate counters.
4. Route serialization through `client.chat()` even for BOM; never through `chat_with_tools()`.
5. Preserve `_lean_retry()` for commit rejection. For serialization, rebuild from `base_messages` plus the serialization instruction; do not echo an incomplete response.
6. Report actual call count and terminal `failure_kind`. Preserve `reply_head`, rounds, tool-call count, cost, wall time, and CPU time where available.
7. Do not catch `BudgetExceeded`. Catch only the client’s known exhausted provider/transport exception families around each completion call, stamp and return a failed stage result with `provider_error` or `transport_error`, and never send those failures through JSON recovery. Unknown programming exceptions continue to propagate.

**Invariant:** no path can turn one parse failure into repeated tool loops or dynamic cap growth.

### 2. Return and persist completion metadata

**Owners:** `kicraft/server/client.py`, `kicraft/server/spend_guard.py`, `kicraft/cli/web_cost_report.py`

1. Change the private `_stream()` result so `chat()` and `chat_with_tools()` can return:
   - `finish_reason`;
   - content/reasoning presence or character counts;
   - provider usage fields when supplied;
   - requested `max_tokens`;
   - selected reasoning policy;
   - provider name and loop signal.
2. Keep all new fields null-safe for mocks and legacy providers. Do not infer exact token counts from characters except the existing spend-guard fallback after an aborted stream.
3. Apply the same normal reasoning policy and finite cap to every tool round and forced final round. The separate serialization call always disables reasoning.
4. Record requested cap, reasoning policy, finish reason, and usage in per-call spend metadata. Never persist raw reasoning.
5. Continue stripping `_meta` and `_meta_ctx` from provider payloads.
6. Add a nullable `failure_kind` column to `stage_runs`. Migrate existing SQLite databases with `PRAGMA table_info` plus `ALTER TABLE`; `CREATE TABLE IF NOT EXISTS` alone does not update production databases. Update `web_cost_report.load_stage_runs()` to select it while remaining compatible with ledgers that predate both the table and the column.
7. Extend `SpendGuard.record_stage()` and `_record_stage_ledger()` to write the classification. Old rows remain readable as unclassified.

### 3. Align the BOM programming contract with §9.29

**Owner:** `.claude/skills/kicraft/stages/bom.md`

Replace the native-USB paragraph with this contract:

- Native USB means **no USB-UART bridge**; it does not mean no programming-support parts.
- For ESP32-S3/C3/S2/C6, follow the architecture decision and include one complete recovery/programming mechanism:
  - BOOT and EN/RESET buttons or jumpers;
  - labeled test pads reaching the required boot strap and EN/reset signals; or
  - another architecture-approved reset/strap mechanism.
- A USB connector alone is insufficient for first download and recovery.
- Include required pull components from the selected module/datasheet. Place controls on the MCU sheet and include supporting passives in the MCU `ic_groups` entry where applicable.
- For classic ESP32, retain the selected USB-UART bridge plus DTR/RTS auto-reset network.

`architecture.md` remains the decision point. BOM must implement that decision, and `wiring.md` must connect it rather than reopen it as a user question.

### 4. Add deterministic regression coverage

**Owners:** `tests/test_stage_driver_retry.py`, `tests/test_client_provider.py`, `tests/test_client_tool_loop.py`, `tests/test_stage_driver_replay.py`, `tests/test_spend_guard.py`, `tests/test_stage_resource_telemetry.py`, and a focused prompt test if no existing prompt-contract file fits

Add fake-client tests for these observable contracts:

- empty `finish_reason="length"` takes the reasoning-recovery path, disables reasoning, and never becomes `invalid_json`;
- truncated answer content triggers exactly one plain, tool-free serialization call with reasoning disabled and the fixed cap;
- normal-stop malformed or empty content triggers the same single serialization call and terminates as `invalid_json` if still malformed;
- a second truncated serialization result terminates as `truncated_json`;
- no cap is dynamically doubled and every normal/tool/final/serialization request is finite;
- serialization is available only once across later commit-correction responses;
- parseable serialization output still reaches `_commit()`;
- commit-rejection feedback retains the full candidate, offenders, BOM reconcile behavior, and wiring valid-ref escape hatch;
- actual provider-call counts and `failure_kind` reach the returned result and `stage_status`;
- policy selection preserves caller cap overrides and settings/mocks without `design_reasoning()`;
- usage, provider, finish reason, requested cap, and policy flow through plain calls, tool rounds, and forced final rounds without leaking `_meta_ctx`;
- an existing `stage_runs` SQLite database migrates and accepts classified and legacy/unclassified rows;
- provider/transport failures are not sent through JSON recovery;
- the source replay state remains byte-for-byte untouched.

Replace `test_truncated_json_recovery_still_doubles_budget`; the new assertion is a transition to one fixed-cap serialization call. Retain the existing reasoning-loop tests.

Add a prompt assertion that `build_system("bom")` contains BOOT plus EN/RESET or strap-test-pad requirements and explicitly says native USB omits the bridge, not the recovery mechanism.

## Verification

All replay runs must use the existing frozen-state harness. It creates a new `kc-replay-*` workspace and copies the state; the source file remains untouched.

1. Run focused deterministic tests, including whichever spend/report test owns the schema migration:

   ```bash
   .venv/bin/python -m pytest -q \
     tests/test_stage_driver_retry.py \
     tests/test_client_provider.py \
     tests/test_client_tool_loop.py \
     tests/test_stage_driver_replay.py \
     tests/test_spend_guard.py \
     tests/test_stage_resource_telemetry.py \
     tests/test_triage_cli.py
   ```

2. Run the prompt/schema example tests that exercise `build_system("bom")` and the BOM slot models.

3. Replay the frozen target BOM stage three times:

   ```bash
   .venv/bin/python -m kicraft.server.stage_driver replay \
     --state /home/kicraft/.kicraft/projects/1/743/.kicraft/state.json \
     --stage bom --budget 0.25
   ```

   Before and after, compare the source state file checksum. Each command must report its temporary workspace so successful BOMs can be inspected without touching the source.

4. For JSON recovery, require all three runs to avoid terminal `truncated_json` and `invalid_json`. A `commit_rejected` is acceptable evidence only when it names a real BOM gate failure and is not mislabeled as parsing failure; it does not count as a successful replay.

5. For the prompt fix, each committed replay BOM must contain the architecture-approved programming mechanism and pass §9.29.

6. Run the no-build canary against four representative frozen briefs: large/tool-heavy BOM, native-USB ESP32, wiring-heavy, and simple. Use three repetitions for any reliability claim.

7. Only after the canary passes, run the fixed self-eval cohort and compare:
   - stage completion rate and failure-kind distribution;
   - finish reasons plus reasoning/content usage;
   - provider-call count, tool rounds, wall time, and spend;
   - §9.29 false negatives and false positives.

## Acceptance criteria

- A parse failure causes at most one fixed-cap, tool-free, reasoning-disabled serialization call.
- Empty reasoning exhaustion, truncated answer JSON, invalid JSON, commit rejection, and provider/transport failures remain distinguishable in results, state, and ledger data.
- No partial BOM or JSON prefix is committed.
- Normal calls, BOM tool rounds, forced final rounds, and serialization calls all have finite caps.
- Existing commit-correction, BOM reconcile, and wiring unknown-reference behavior remains covered and passing.
- Native-USB ESP32 BOM and architecture prompts state the same programming/recovery contract.
- Three fresh target replays commit valid BOMs without terminal serialization failures; stochastic gate failures remain explicit rather than being reclassified.
- Q1-like symbol hallucinations, silent substitutions, missing programming paths, and stock failures still reject the BOM.
- Production ledger migration preserves old rows and records new classifications.
- Spend remains bounded by `SpendGuard`, the stage retry limits, and fixed policy ceilings.

## Rollout and rollback

1. Land deterministic policy/recovery tests and the backward-compatible ledger migration before changing production behavior.
2. Deploy the BOM prompt and serialization path together; either change alone leaves the failure mode partly intact.
3. Run the four-brief canary, then the fixed self-eval cohort. Compare completion, classifications, call counts, latency, and spend.
4. Rollback may disable the new serialization retry or reasoning controls independently, but must not restore dynamic cap doubling or weaken commit gates.
5. Use `deploy/restart-web.sh` and `deploy/restart-build-worker.sh` for production restart; verify HTTP 200 and build-worker readiness per repository operations guidance.

## Prior art

- `docs/plans/deepseek-v4-flash-json-budget-fix.md`: identifies reasoning/output-budget exhaustion and proposes bounded policies plus serialization recovery.
- `docs/plans/codebase-review-2026-07-19.md §7.3`: identifies missing proactive ESP32 BOOT/EN guidance.
- `7a8b81d`: added ESP32 strap guidance to `architecture.md`; the BOM prompt remains contradictory.
- `a48563d` and the 2026-07-27 self-eval fix wave: established the hard §9.33 substitution ledger and related synthesis gates. This plan does not weaken them.
