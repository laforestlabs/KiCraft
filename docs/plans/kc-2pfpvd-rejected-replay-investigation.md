# KC-2PFPVD — investigate post-fallback wiring commit rejections

**Status:** investigation plan based on the 2026-09-03 post-fallback replay batch

**Frozen source:** `~/.kicraft/projects/1/783/.kicraft/state.json`

**Goal:** explain why two provider-successful wiring replays still converged to `commit_rejected`, then make the smallest validator/guidance correction supported by captured attempt evidence. This work does not relax a gate, add another stall detector, or increase retry budgets again.

## Verified evidence

Three post-fallback replays ran with `--max-retries 7 --budget 0.25`.

1. `/tmp/kc-replay-nv4g3n0j` committed on provider attempt 4 for `$0.023066`.
   - Attempts 1–2: flash on DeepInfra, commit rejected.
   - Attempt 3: flash route returned HTTP 429 after its internal transport retries.
   - Attempt 4: the one provider fallback used pro on Alibaba and committed.
   - The committed USB topology is correct: R9 separates `USB_DP`/`USB_DP_MCU`; R10 separates `USB_DN`/`USB_DN_MCU`; U3 IO20/IO19 carry D+/D−; U3 TXD0/RXD0 are `no_connect`; HUB75 does not consume IO19/IO20.

2. `/tmp/kc-replay-6iwf_4ad` failed after 6 provider attempts for `$0.029698`.
   - Attempts 1–3: flash on DeepInfra, commit rejected.
   - Attempt 4: the existing repeated-signature pristine escape escalated to pro on Alibaba.
   - Attempts 4–6: pro, commit rejected.
   - The terminal rejection contained §9.17 (R11, 10k, both terminals on `GND`), §9.20 (U3 TXD0 on `USB_DP_MCU`), and three §9.15 dangling nets including `HUB75_CLK`.
   - Because the pristine escape occurred at attempt 4, attempts 2 and 3 had exactly equal `_commit_rejection_signature` values. Termination at attempt 6, below the eight-attempt outer ceiling, is consistent with a post-escape exact repeat, but the CLI did not persist enough detail to identify the complete attempt-5/6 signatures.

3. `/tmp/kc-replay-cf1cq91s` failed after 7 provider attempts for `$0.031234`.
   - Attempts 1–4: flash on DeepInfra, commit rejected.
   - Attempt 5: repeated-signature pristine escape escalated to pro on Alibaba.
   - Attempts 5–7: pro, commit rejected.
   - The terminal rejection contained three §9.15 dangling nets. One visible offender was R12 (330R): `LED_DATA` had only one R12 terminal while the other terminal shared `LED_DATA_OUT` with J3.
   - Because the pristine escape occurred at attempt 5, attempts 3 and 4 had exactly equal signatures. Termination at attempt 7 is again consistent with a post-escape exact repeat.

No failed post-fallback attempt was provider-rate-limited. These are correction-convergence failures, not evidence that provider fallback failed.

## Evidence gap

The replay CLI prints only the final stage result. It does not persist its `progress` stream, normalized candidate slots, full commit results, or rejection signatures. Failed replay workspaces retain the unchanged frozen state plus aggregate `stage_status`; the spend ledger retains route/cost/outcome facts but not candidate topology or offenders. Terminal console output is truncated.

Therefore the exact defect transitions cannot be reconstructed honestly from these two workspaces. Instrumentation and traced replays come before another guidance change.

## W1 — add opt-in replay attempt tracing

**Targets:** `kicraft/server/stage_driver.py`, `kicraft/server/stage_pipeline.py`, `kicraft/server/stage_runtime.py`, and focused retry/replay tests.

1. Add an opt-in replay trace path, for example `replay --trace-jsonl <path>`. No trace is written by ordinary production/web runs.
2. Thread a dedicated attempt observer through `drive_replay` and `drive_stage`; do not overload production progress events with full BOM/wiring payloads.
3. After response normalization and after each commit attempt, write one JSONL record containing:
   - 1-based provider attempt;
   - call mode (`normal`, `clean_slate`, or serialization recovery);
   - active model, design profile, and provider attribution when available;
   - normalized candidate slot (for wiring: `connections`, `no_connect_pins`, and questions);
   - full commit result (`errors`, complete offenders, and `offenders_total`);
   - `_commit_rejection_signature`;
   - whether the response armed/used the pristine escape, escalated, or used provider fallback.
4. Exclude prompts, API keys, headers, hidden reasoning, and raw provider bodies. The normalized design slot and deterministic commit output are sufficient.
5. Flush each record before the next paid call so a crash or budget refusal does not erase earlier attempts. Print the trace path in the replay footer.
6. Pin with scripted tests: exact record count/order, candidate-to-rejection association, route attribution, one clean-slate transition, terminal repeat, and no trace when the option is absent.

**Acceptance:** one failed scripted replay yields a self-contained trace from which every signature transition and topology edit can be reproduced without reading terminal output or the spend ledger.

## W2 — capture and classify three traced board-783 replays

1. Run three independent frozen replays with the same cap:
   `.venv/bin/python -m kicraft.server.stage_driver replay --state ~/.kicraft/projects/1/783/.kicraft/state.json --stage wiring --max-retries 7 --budget 0.25 --trace-jsonl <unique-path>`.
2. Build an attempt table per replay: model/provider, call mode, gate IDs, canonical offender identities, signature equality with the prior attempt, and commit result.
3. Diff each normalized candidate against the prior candidate by `(sheet, net_name, ref, pin)` and `no_connect_pins`. Separate actual topology movement from net renaming.
4. For every repeated signature, identify which prior correction sentence the next candidate followed and why that operation left or recreated the same canonical defect.
5. Confirm that provider fallback and commit escalation remain orthogonal: a 429 may change the active route once, while only an exact repeated commit signature arms the one pristine escape.

**Acceptance:** each failed replay has an evidence-backed causal chain from validator text to candidate edit to next rejection. Do not implement a topology fix from terminal excerpts alone.

## W3 — test the three leading hypotheses

### H1 — generic §9.15 series advice assumes the wrong direction

The R12 terminal offender exposes a likely asymmetry in `_dangling_net_context`: when one resistor terminal dangles, the current text tells the model to move a load/destination endpoint from the other terminal's net onto the dangling net. That was correct for the original USB case only because IO20 had been placed on the connector side. For a conventional LED series resistor, J3 may already be the correct load on `LED_DATA_OUT`; the missing endpoint may instead be the MCU/inter-sheet source on `LED_DATA`. Moving J3 would merely move the dangling condition across R12.

Trace assertions:

- Determine whether J3 repeatedly moved between `LED_DATA` and `LED_DATA_OUT` or whether the MCU source disappeared.
- Check the frozen architecture direction for `LED_DATA` (`MCU: output`, `ADDRESSABLE LED OUTPUT: input`) against the candidate's local endpoints.
- Verify whether the offender incorrectly described a declared `LED_DATA` inter-sheet net as undeclared. If so, reproduce that separately before changing series guidance; exact-name architecture coverage should exempt a valid one-pin local stub.

Candidate fix only if confirmed:

- Stop unconditionally calling endpoints on the populated side the destination to move.
- Use proven architecture direction and resolvable pin electrical/function data when they identify source versus load.
- Otherwise list both sides and require two endpoints per side without prescribing which non-resistor endpoint moves.
- Preserve fixed-signal filtering and the invariant that contextual labels cannot change `_offender_identity`.

### H2 — §9.17 correction treats a shunt resistor as a series resistor

R11 is 10k on the HUB75 sheet and is plausibly a pull-up/pull-down, while R9/R10 and R12 are low-value series parts. The current §9.17 retry note prescribes a three-item source/series/destination split for every self-shorted two-terminal part. That operation is wrong for a pull resistor whose valid topology is one signal terminal plus one rail terminal.

Trace assertions:

- Identify R11's intended signal from the candidate immediately before it became `GND`/`GND`.
- Check whether the model followed the generic §9.17 series instruction and invented/moved a local net.
- Determine whether R11 repeatedly oscillated between `GND`/`GND`, a dangling signal terminal, and a valid HUB75 control-net pull-down.

Candidate fix only if confirmed:

- Keep the deterministic self-short rejection.
- Make correction text topology-neutral unless role evidence is conclusive.
- For one rail side plus a control/data context, explain the valid shunt pattern: one terminal on the signal and the other on the rail; never create a second series-path net.
- Retain the existing series-path wording only for proven conditioning/series contexts.

### H3 — multi-gate feedback repairs one defect by recreating another

The second failure ended with §9.15, §9.17, and §9.20 simultaneously. The current retry message concatenates every gate's advice, including generic §9.15/§9.17 notes after detailed offenders. The model may obey mutually incompatible generic operations after following the precise USB removal instruction.

Trace assertions:

- Compare the order of offender/generic notes with the exact changes to R9/R10, R11, R12, U3, and HUB75 nets.
- Check whether the §9.20 same-signal removal instruction was present or whether IO20 was absent from every accepted D+ variant, which legitimately selects the ordinary move instruction.
- Identify any correction pair that directs the same endpoint to two different nets.

Candidate fix only if confirmed:

- Suppress a generic retry note when an offender already contains more specific topology guidance for the same part/net.
- Do not reorder or drop unrelated offenders.
- Keep one complete-slot response contract and one exact-repeat state machine.

## W4 — deterministic reproduction before fixing

For every confirmed failure mechanism:

1. Extract the smallest normalized candidate plus frozen architecture/BOM identity needed to reproduce its exact validator output.
2. Add a regression that fails on the captured candidate and asserts the unsafe or contradictory correction phrase.
3. Implement the smallest source correction.
4. Assert both sides:
   - the captured bad candidate now receives satisfiable guidance;
   - existing valid USB series topology, D+/D− polarity swaps, unresolved-pin fail-open behavior, inter-sheet stubs, and generic non-series dangling nets remain unchanged.
5. Add a scripted retry test using the captured candidate sequence where practical; the corrected guidance must change the repeated signature or commit, not merely change prose.

## Verification and rollout gate

1. Run focused tests:
   `.venv/bin/python -m pytest tests/test_kicraft_validation.py tests/test_stage_driver_retry.py tests/test_client_provider.py`.
2. Format touched Python files.
3. Run the three traced frozen replays again with independent `$0.25` caps.
4. Require at least 2/3 commits. For every commit inspect:
   - R9/R10 keep connector and MCU nets separate;
   - IO20/IO19 carry USB D+/D−;
   - TXD0/RXD0 are absent from USB nets;
   - HUB75 excludes IO19/IO20;
   - R11 has one signal side and one intended rail side, not a self-short;
   - R12 has a complete two-net series path from the MCU/inter-sheet source to J3.
5. A gate-clean but electrically wrong candidate fails acceptance.
6. Only after the 2/3 bar, restart with `deploy/restart-web.sh` and `deploy/restart-build-worker.sh`; verify HTTP 200 and trailing `[build-worker] ready`.
7. Then run the verbatim brief once through the web UI and use triage to verify wiring committed, route/retry attribution is present, and the build reaches layout (`rc > 5`).

## Non-goals

- No gate loosening, deterministic netlist normalizer, or hard-coded board-783 netlist.
- No new overlap/similarity stall detector, second pristine escape, or further retry-budget increase.
- No unrestricted OpenRouter fallback and no fallback after budget refusal.
- No guidance change based only on truncated terminal output; traced candidate evidence is mandatory.
- No production deployment before the existing 2/3 electrically correct replay bar.
