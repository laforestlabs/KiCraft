# KC-2PFPVD — break the §9.15/§9.20 offender ping-pong, escalate a stalled wiring stage, +50% attempt budget

**Status:** revised plan (investigation re-verified 2026-09-03 against production source and `783/events.jsonl`)
**Board:** `KC-2PFPVD` = `~/.kicraft/projects/1/783` (web run, 2026-09-02 23:19 UTC, wiring `commit_rejected` after 5 provider calls; build never started)
**Brief class:** third identical-brief death: KC-JMSMVE (1/765, 8/27), KC-VKUT5H (1/780, 9/02 11:43), KC-2PFPVD (1/783, 9/02 23:19) — USB-C PD 5V → ESP32-S3-WROOM-1-N16R8 + HUB75 + LED-string + speaker.
**Code at run:** post-`7c1f212` (§9.20 signal-assignment gate) + `5be28d5` (bounded continuation) + `ddbc667` (§9.15 series enrichment) — the full KC-VKUT5H fix plan was live.

## Verified diagnosis

The gates worked: §9.15 converged 18→18→2→0 dangling nets, and §9.20 correctly refused every wrong-pin USB candidate. The correction text did not.

1. On attempts 1–3, `_dangling_net_context` (`validation.py:1246`) listed every non-resistor endpoint on the series part's other net. For `USB_DP_MCU`, that included U3 TXD0 and other GPIOs which `_check_known_signal_assignments` (`validation.py:1864`) simultaneously prohibited from carrying any accepted D+ net name. Its instruction to move an intended endpoint therefore included invalid choices.
2. On attempts 4–5, §9.15 was clean; only §9.20 remained. `_check_known_signal_assignments` adds `(swap the two)` whenever the required target pin has any differently named net. It does not test whether both names match the same signal regex. The model followed that instruction exactly: attempt 4 put TXD0 on `USB_DP` and IO20 on `USB_DP_MCU`; attempt 5 exchanged only those names. The two rejection signatures are identical because `_commit_rejection_signature` keys them to gate `9.20` and offender pin U3.37, not net spelling.
3. The current retry state machine matters to the remedy. `next_attempt` gives an exact repeated signature one pristine escape, then terminates if the pristine response repeats it. A general “last two signatures overlap” escalation would be a second, competing stall detector and is undefined for improving supersets such as attempts 2→3. Escalation must reuse the existing exact-repeat/clean-slate transition.
4. The current spend guards check already-recorded spend before a call. They intentionally permit at most one completion to overshoot a ceiling; they cannot prove that a proposed pro completion “would cross” the cap. The implementation and acceptance language must preserve that actual contract rather than promise predictive refusal.
5. Retries use `base_messages + previous assistant response + latest correction`, not an accumulated transcript. There is no attempt-8 history compactor to inspect.
6. Scope is wiring only. Board 783's BOM committed successfully on attempt 4, and this incident supplies no evidence for increasing or escalating BOM retries.

The true corrective operation is: keep the connector/conditioning side and MCU side on opposite sides of R9/R10; keep IO20/IO19 on the MCU-side D+/D− nets; remove TXD0/RXD0 and arbitrary GPIO endpoints from every USB-named net; only connect those pins to a real, separately named function if the design actually uses them. The validator must not tell the model to merge both sides of a series part or invent a console net.

## Workstreams

### W1 — make §9.15 and §9.20 guidance jointly satisfiable

**Target:** `kicraft/design/synthesis/validation.py`

1. Extract one private matcher which, for a resolved part and net name, returns the matching `_SignalAssignment` signal tuple and whether a pin function satisfies its required function. Use it from both checks; do not duplicate the family/net/function rules.
2. In `_check_known_signal_assignments`, retain the existing swap advice only when the target pin's current net does **not** match the same `sig_re`. This preserves the useful D+/D− polarity swap and USB-vs-HUB75 swap cases.
3. When `cur` and `net` both match the same `sig_re`, emit removal/topology advice instead of swap advice:
   - keep the uniquely resolved required-function pin on its existing USB net;
   - remove the reported wrong-function pin from all names for that USB signal;
   - keep any proven series-part terminals on different nets;
   - connect the removed pin to a separately named functional net only if another real endpoint requires that function; otherwise mark it `no_connect`.
   Do not call this “deduping” the two USB net names: connector-side and MCU-side names may be intentionally distinct across a series resistor.
4. `_dangling_net_context` currently receives pin inventories but not each ref's symbol/value identity, so it cannot apply a family matcher correctly. Build a `ref -> "<symbol> <value>"` index once in `check_no_dangling_signal_nets`, pass it into the context helper, and in the series branch filter each candidate `(ref, pin)` only when that candidate's identity and the related net match a known fixed signal and the candidate pin function violates it. Preserve connector/passive candidates and fail open when the part identity or pin function is unresolved.
5. If filtering removed candidates, append identity-safe guidance naming the required function (for example IO20 for D+) and saying rejected candidates cannot carry either accepted name variant. Preserve the invariant at `validation.py:1195`: contextual text must not add `REF.PIN` or `REF pin N` tokens that change `_offender_identity`.

**Tests:** `tests/test_kicraft_validation.py`

- Add a focused series topology matching board 783: R9 separates `USB_DP`/`USB_DP_MCU`; IO20 and TXD0 are on the connector-side name; the MCU-side resistor terminal dangles. Assert TXD0 is absent from candidate endpoints, IO20 remains, and the instruction keeps R9's terminals on different nets.
- Add the attempt-4/5 §9.20 state: TXD0 is on one accepted D+ variant and IO20 on the other. Assert the offender says to remove TXD0, does not say `swap the two`, does not tell the model to merge the net variants, and retains one canonical offender identity.
- Keep `test_a2_swapped_polarity_fails`: opposite D+/D− assignments must still say `swap the two`.
- Add fail-open coverage for an unresolved candidate part/pin so candidate filtering cannot silently hide the only useful endpoint.

**Acceptance:** the exact attempt-4/5 name permutation receives one stable, satisfiable correction; valid series topology and genuine D+/D− swaps retain their existing behavior.

### W2 — use the stronger profile for the existing pristine escape

**Targets:** `kicraft/server/config.py`, `kicraft/server/client.py`, `kicraft/server/stage_runtime.py`, `tests/test_config.py`, `tests/test_client_provider.py`, and `tests/test_stage_driver_retry.py`. `stage_pipeline.py` needs no second budget path: the escalated client must share the original guard.

**Trigger:** the next wiring call is the one pristine escape already armed by `next_attempt` after two adjacent, exactly equal `_commit_rejection_signature` values, and that next call is provider attempt 3 or later. Do not add offender-set overlap, substring similarity, or a second clean-slate allowance.

**Behavior:**

1. Add `Settings.escalation_profile`, resolved from `KICRAFT_ESCALATION_PROFILE`; default `pro`, empty disables it, and any non-empty value must name a `DESIGN_PROFILES` entry. If it equals the active design profile, treat escalation as disabled.
2. Add `CappedOpenRouterClient.with_design_profile(profile_name)`: clone the active `Settings` with the selected profile's model, provider order, and price caps, then return a client reusing the **same guard object**. It must not call `Settings.from_env()` again and must not create a second `SpendGuard`/run-budget snapshot. Scripted clients used by retry tests implement the same method while sharing their response queue and guard; stage runtime treats a missing method as escalation unavailable rather than constructing a live client.
3. At the existing exact-repeat transition, build the same pristine `_lean_retry(None, feedback)` call, reasoning-disabled as today, but route that call through the escalated client. Once escalated, keep that client for any remaining bounded continuation calls in this stage. There is still exactly one pristine escape and one outer provider-call budget.
4. Use the active client consistently for provider calls, `_record_attempt_facts`, model metadata, and cost accumulation. The final stage result remains one stage ledger record through the shared guard.
5. Emit:
   `{"kind":"escalation","stage":"wiring","from":<model>,"to":<model>,"attempt":<1-based upcoming provider call>,"reason":"repeated_commit_signature"}`
   exactly once, immediately before the first pro call. Add `model` to wiring `retry` events; `stage_start` currently reports only the initial model and retry events currently report none.
6. If the shared guard refuses the pro call, preserve the existing `BudgetExceeded` terminal behavior and ledger facts. Do not fall back to flash after a cap refusal: that would make another call through the same exhausted guard. Document the existing one-completion overshoot bound; do not claim predictive “would cross cap” enforcement.

**Tests:** `tests/test_stage_driver_retry.py`, `tests/test_config.py`, and `tests/test_client_provider.py`

- A wiring sequence `A, A` escalates the upcoming pristine third call exactly once; the third call has a pristine transcript, reasoning disabled, pro model/provider settings, and the original guard by object identity.
- A changed signature does not escalate; non-wiring stages do not escalate; empty/same-profile configuration does not escalate.
- A pro pristine response equal to the arming signature terminates exactly as the flash pristine response does; a changed response continues on pro without a second escalation.
- Progress events contain one escalation event with the upcoming 1-based attempt and retry events identify the model that produced each rejection.
- Budget refusal on the escalated call records the normal terminal budget failure and makes no fallback call.

**Acceptance:** escalation is a model-route change inside the proven bounded retry state machine, not a new retry path; all calls remain under the original global/per-run guard and call budget.

### W3 — increase wiring's attempt ceiling from 5 to 8

**Target:** `kicraft/server/stage_runtime.py:138` and retry tests.

- Change only wiring: `_STAGE_MIN_RETRIES = {"wiring": 7, "bom": 4}`. Because the loop is `range(max_retries + 1)`, the default wiring ceiling becomes 8 provider attempts: `ceil(5 × 1.5) = 8`.
- Keep `provider_call_budget = max_retries + 2`. It remains one slot above the normal outer loop solely for the nested serialization-recovery call; it is not a promise that every rejection sequence receives 9 calls.
- Do not change exact-repeat termination. Extra attempts are available to changing/progressing offender signatures, not to unbounded identical churn.
- No transcript-compaction change is needed: `_lean_retry` carries only the base prompt, immediately previous assistant JSON, and latest correction.

**Tests:** pin `_stage_max_retries("wiring", 2) == 7`, `_stage_max_retries("bom", 2) == 4`, and the higher caller-default behavior. Drive eight **distinct** commit-rejection signatures and assert terminal `attempts == 8`; a repeated-signature test must still stop after the single escalated pristine escape rather than artificially consuming all eight.

## Verification and rollout

1. Run the focused tests:
   `.venv/bin/python -m pytest tests/test_kicraft_validation.py tests/test_stage_driver_retry.py tests/test_config.py tests/test_client_provider.py`
   Then run the project formatter on touched Python files.
2. **W1 deterministic proof, no LLM:** construct the frozen attempt-4/5 connection state in the validation test helpers and run both `check_family_wiring_contracts` and `check_no_dangling_signal_nets`. Save/assert the exact offender properties above; do not depend on parsing streamed `answer_delta` fragments at test time.
3. **W2 deterministic proof, no spend:** scripted client tests collect the `progress` callback and prove route, guard identity, event shape, one-shot behavior, and terminal behavior.
4. **Live replay, capped:** run three independent wiring replays from the frozen pre-wiring state:
   `.venv/bin/python -m kicraft.server.stage_driver replay --state ~/.kicraft/projects/1/783/.kicraft/state.json --stage wiring --max-retries 7 --budget 0.25`
   The CLI does not currently persist replay progress events, so verify the route from `stage_attempts.model` in the spend ledger, or invoke `drive_replay(..., progress=collector.append)` in a short operator script. Do not claim an escalation event exists in the temp workspace unless a progress writer was explicitly supplied.
5. Require at least 2/3 replay commits before deployment. For every commit, inspect `bom.connections`: R9/R10 separate connector-side and MCU-side nets; USB D+/D− reach IO20/IO19; TXD0/RXD0 are absent from USB names; HUB75 does not consume IO19/20. A gate-clean but electrically wrong commit fails acceptance.
6. Deploy only after the replay bar: `deploy/restart-web.sh`, `deploy/restart-build-worker.sh`; verify HTTP 200 and a trailing `[build-worker] ready` as required by `AGENTS.md`.
7. Run the verbatim 783 brief once through the web UI, then run `.venv/bin/python -m kicraft.cli.triage run KC-<new>` and `triage audits`. The live acceptance bar is wiring committed, escalation/retry model attribution visible in `events.jsonl`, and build reaching layout (`rc > 5`), not judge score.
8. After at least five subsequent ESP32-S3 wiring runs, use `triage scan` to check that §9.15/§9.20 repeated-signature attempt exhaustion trends to zero. This is post-deploy monitoring, not a substitute for the replay gate.

## Non-goals

- No gate loosening or deterministic netlist normalizer.
- No whole-run profile switch and no BOM escalation/retry increase without separate evidence.
- No second stall detector, second clean-slate escape, or fallback call after budget refusal.
- No invented console/debug net: removed UART pins are wired only when the design has a real endpoint, otherwise `no_connect`.
- `rt9013-33` home-fetch and unrelated shaped/form-factor findings remain separate work.
