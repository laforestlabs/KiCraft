# Canary self-eval and LLM reliability analysis

**Status:** implementation-ready operational plan  
**Primary target:** the deployed main-designer path after the stage-contract/runtime changes  
**Reference batch:** `/home/kicraft/.kicraft/self_eval/20260825T033602Z`  
**Production evidence root:** `/home/kicraft/.kicraft/projects`  
**Spend ledger:** `/home/kicraft/.kicraft/spend_ledger.db`

## Decision

Run one deliberately diagnostic brief from each of the nine benchmark archetypes, once, through the real self-eval path with full events and the normal judge. Run sequentially with one build slot. Then perform a deterministic, evidence-first analysis of design-stage behavior before discussing placement, routing, fab readiness, or aggregate rubric grade.

This is a canary, not a reliability estimate. One run per archetype can prove a blocker still exists; it cannot prove a noisy model is reliable. Do not claim a model-quality improvement from nine single observations. If the canary clears the LLM stop gates below, a separate repeated campaign may follow.

## Goals

1. Exercise every `BENCHMARK_PROMPTS` archetype through the same `run_session` and build tail used by production.
2. Revisit exact briefs that witnessed the recent LLM failure classes: missing JSON, output truncation, wiring shorts, dangling nets, repeated-output degeneration, and programming-access omissions.
3. Verify the new schema-bound stage runtime, compact BOM path, architecture-sheet enum, serialization recovery, and typed wiring-patch correction under live provider behavior.
4. Compare the canary with the exact same slugs in the 2026-08-25 batch and with recent production-stage telemetry.
5. Produce a concise report that separates LLM-stage defects from deterministic build/place/route defects.

## Non-goals

- Do not change prompts, retry limits, token limits, model, provider order, price caps, gates, or build settings during the canary.
- Do not silently switch from Flash to Pro after a failure.
- Do not weaken §9 commit checks or classify a rejected design as a success because a partial board exists.
- Do not use aggregate A–F grade or fab readiness as the primary LLM verdict.
- Do not restart production services merely to run the batch.
- Do not include raw user briefs, answers, reasoning, or tool results from production projects in the written report. Production reporting is aggregate plus project/run IDs and redacted failure signatures.

## Canary cohort

The corpus has nine archetypes in `kicraft/tuning/benchmark.py`. Select these exact slugs:

| Archetype | Slug | Recent witness |
|---|---|---|
| `single_passive` | `r2r-dac` | Wiring exhausted correction on §9.19 multi-net pins. |
| `usb_c_connector` | `usb-a-power-splitter` | Terminal `no JSON in reply`. |
| `fine_pitch` | `stm32-min` | Truncated JSON at the output token limit. |
| `rf_antenna` | `nrf52-beacon` | Truncated JSON at the output token limit. |
| `power_thermal` | `dual-rail-supply` | Wiring exhausted correction on §9.19 multi-net pins. |
| `mixed_tht_smt` | `encoder-oled-panel` | Wiring exhausted correction on §9.15 dangling nets. |
| `hi_pin_hierarchical` | `can-node` | Truncated JSON at the output token limit. |
| `connector_dense_io` | `servo-driver-16` | Terminal `no JSON in reply`. |
| `shaped_outline` | `round-led-ring` | Terminal `no JSON in reply`; also exercises repeated BOM members and a deterministic outline grade. |

This cohort is intentionally adversarial. In the reference batch it was **0/9 design-complete**, cost **$0.295744** in design calls, and split evenly across three terminal classes:

- 3 `no JSON in reply`;
- 3 truncated JSON failures;
- 3 deterministic wiring commit failures (§9.19 twice, §9.15 once).

Keep the cohort fixed even if a different brief looks easier after launch. Changing a slug destroys the exact comparison.

## Phase 0 — freeze identity and safety prerequisites

Before any paid call:

1. Confirm the checkout is the deployed commit and the working tree has no uncommitted runtime/config changes.
2. Record, without secrets:
   - UTC start time and git commit;
   - resolved design profile;
   - exact dated design model ID, judge model ID, provider order, price ceilings, reasoning policy, serialization limits, and collection bounds;
   - current spend-ledger totals and remaining daily/total headroom;
   - host CPU count and resolved build-slot count.
3. Refuse to proceed if the design model is an unversioned/mutable alias. The campaign manifest must contain the exact resolved model returned by settings and the self-eval summary must match it.
4. Confirm no production build is currently running before taking the sole build slot. Do not kill or delay a user build for the canary.
5. Estimate the campaign envelope from the exact reference cohort. The reference design cost was $0.295744; reserve at least $0.50 for design calls plus judge headroom. Stop if current global ceilings cannot absorb that without risking production traffic.
6. Run the existing capability smoke checks for the actual designer role and judge role. For the default Flash profile, use:

```bash
OUT=/home/kicraft/.kicraft/self_eval/$(date -u +%Y%m%dT%H%M%SZ)_llm_canary
mkdir -p "$OUT"
.venv/bin/python -m kicraft.cli.model_preflight \
  --role flash --out "$OUT/preflight-designer.json"
.venv/bin/python -m kicraft.cli.model_preflight \
  --role judge --out "$OUT/preflight-judge.json"
```

If a different explicit profile is deployed, use its matching preflight role; do not pretend Flash metadata validates Pro. Stop on missing model, incompatible provider, rejected `response_format`, tool incompatibility, price-cap mismatch, or failed schema smoke response.

Write a sanitized `campaign.json` into `$OUT` containing the frozen identity/configuration and preflight artifact paths. Never copy `.env` or API keys.

## Phase 1 — run the nine-brief canary

Use the real batch entry point. Keep full events; do not pass `--lean-events` or `--no-judge`. Use one thread and one build slot so provider traces, spend, CPU, and build behavior remain attributable and production overshoot is bounded to one in-flight call.

```bash
SLUGS='r2r-dac,usb-a-power-splitter,stm32-min,nrf52-beacon,dual-rail-supply,encoder-oled-panel,can-node,servo-driver-16,round-led-ring'

.venv/bin/python -m kicraft.eval.self_eval \
  --only "$SLUGS" \
  --out "$OUT" \
  --repeats 1 \
  --parallel 1 \
  --build-slots 1 \
  --build-timeout 2400
```

Operational rules:

- Preserve stdout/stderr in `$OUT/canary.log` while retaining terminal visibility, e.g. through `tee`.
- Let `summary.json` checkpoint after each brief. Do not manually edit a run directory.
- If the process is interrupted, resume the same directory and same cohort:

```bash
.venv/bin/python -m kicraft.eval.self_eval \
  --resume "$OUT" --only "$SLUGS" \
  --repeats 1 --parallel 1 --build-slots 1 --build-timeout 2400
```

- `--resume` may rerun errored/missing records according to the harness contract. It must not be used to cherry-pick a better stochastic result after a completed failure.
- A spend-guard, provider, transport, capability, or campaign-integrity failure stops the batch. Preserve the partial summary and analyze it as an operational failure; do not raise ceilings mid-run.
- A design-stage failure in one brief does not corrupt later independent briefs. Allow the fixed cohort to finish unless the failure proves a shared operational prerequisite is broken, such as every structured response being rejected by the provider.

## Phase 2 — validate batch integrity before analysis

Require all of the following before calling the campaign complete:

1. `$OUT/summary.json` and `$OUT/summary.md` exist.
2. The summary contains exactly nine distinct selected slugs, `repeats == 1`, `parallel == 1`, `build_slots == 1`, full events enabled, and zero missing/duplicate records.
3. Every run directory contains `brief.txt`, `events.jsonl`, and `.kicraft/state.json`; design-complete runs also have the expected generated/build artifacts.
4. The summary's `design_model`, judge model, rubric version, and campaign settings equal the frozen `campaign.json` values.
5. Spend-ledger rows for the campaign reconcile with per-run design costs within normal rounding. Report discrepancies; never silently substitute one total for another.
6. The selected cohort in the produced summary exactly matches the reference cohort. Compare by slug, not by corpus index.

If integrity fails, the result is `INVALID_CAMPAIGN`, not an LLM regression or improvement.

## Phase 3 — LLM-specific analysis

Perform this analysis deterministically from artifacts. The judge may grade design quality, but it must not classify provider/runtime failure modes.

### Data sources

For each canary run, read:

- `$OUT/summary.json` run record;
- `<run>/events.jsonl` full provider/stage/tool trace;
- `<run>/.kicraft/state.json`, especially `stage_status`, committed architecture, BOM, wiring, questions, and artifacts;
- `<run>/eval/report.json` for rubric/gate context;
- matching `stage_runs` rows from `/home/kicraft/.kicraft/spend_ledger.db`.

Comparison sources:

- exact matching slug in `/home/kicraft/.kicraft/self_eval/20260825T033602Z/summary.json`;
- the last 25 production projects by `.kicraft/state.json` modification time under `/home/kicraft/.kicraft/projects`, plus the known unknown-sheet witnesses `1/701`, `1/748`, `1/749`, and `1/754` even if outside that window;
- production `events.jsonl` and `stage_runs` rows where available.

Production evidence is read-only. Report only aggregate counts, project IDs, stage names, gate IDs, normalized offender identities, costs, and timings. Do not quote customer text or model reasoning.

### Classification precedence

Classify each terminal design outcome once, using this order:

1. `operational`: spend guard, transport, provider HTTP, capability/preflight, or missing artifacts;
2. `reasoning`: `reasoning_loop`, empty length completion, hard reasoning ceiling, or repeated reasoning abort;
3. `serialization`: `collection_limit`, `truncated_json`, `invalid_json`, literal `no JSON in reply`, or schema-bound recovery failure;
4. `schema_contract`: `invalid_schema`, malformed question payload, response-contract rejection, or BOM contract preparation failure;
5. `commit_contract`: parseable candidate rejected until terminal; record every §9 gate ID and normalized offender signature;
6. `question_or_reconcile`: blocking user question or wiring-to-BOM reconciliation that parks the stage;
7. `design_complete`: all five design stages committed.

Keep deterministic build failures in a separate `build_outcome` field. A routing rc6/rc7 must not be attributed to the LLM unless the report identifies an upstream committed design defect with evidence.

### Required analyses

#### A. Structured-output and recovery behavior

For every stage and slug, report:

- provider calls attempted, normal calls, serialization-recovery calls, and commit-correction/patch calls;
- terminal `failure_kind` and finish/retry evidence available in events;
- first-pass parse/commit, recovery success, or terminal path;
- stage cost, wall time, prompt/output/reasoning/cache tokens where recorded;
- whether any full-slot call was made without the prepared stage response contract.

The last item must be evidence-backed. If current events/ledger do not expose response-format metadata, mark it `not observable`; do not infer it from successful parsing. Use the preflight and code contract as supporting evidence, not as proof of a particular live call.

#### B. BOM-specific failures

Report per BOM run:

- rounds, tool calls, normalized MPN lookup counts, repeated/cached tool signatures, cap hits, and resolution-ledger reuse;
- emitted collection count and compact-run expanded count;
- ordinary `parts` versus expanded repeated parts;
- schema/recovery calls and their costs;
- unresolved or guessed symbol/footprint IDs and any stock/retail rejection;
- exact architecture sheet names and every `parts[].sheet`/expanded `part_runs[].sheet` reference.

Explicitly test the repaired incident class:

- zero canary candidates or committed parts may reference a sheet outside committed `architecture.sheets[].name`;
- the known typos from project `1/754` remain impossible under the generated enum;
- no fuzzy matching, aliasing, or autocorrection is credited as a pass.

Compare with production projects `1/701`, `1/748`, `1/749`, and `1/754` to show whether the unknown-sheet class disappeared after the contract change.

#### C. Wiring-specific failures

For each wiring run, report:

- first-call outcome and whether it entered typed patch mode;
- full-slot calls versus patch calls;
- rejection signatures by gate and offender identity;
- offender count progression across attempts;
- whether unrelated valid endpoints remained stable across patches;
- clean-slate transition count;
- final gate result and total wiring cost.

Group terminal commit failures into the recent defect families:

- §9.19 one pin assigned to multiple nets;
- §9.17 two-terminal self-short;
- §9.15 dangling signal net;
- unknown ref/pin/net and coverage failures;
- rail names used as component refs;
- missing BOM part requiring wiring-to-BOM reconciliation.

A patch is progress only when the deterministic offender set shrinks or changes to a distinct defect. Rewriting the board and oscillating between shorts and dangling nets is `no_progress`, even if each raw reply differs.

#### D. Programming and architecture completeness

Report:

- architecture power-net versus sheet-name confusion;
- native-USB versus USB-UART decisions;
- BOOT/RESET, SWD, UPDI, or test-pad access committed for every MCU;
- §9.29 programming-access rejection or silent omission;
- BOM-to-architecture and wiring-to-BOM reference integrity.

Use recent witnesses such as `rounded-c3-devboard` and `snowman-ornament` as comparison context even though they are not in the nine-run canary. Do not expand the paid cohort during analysis.

#### E. Questions, retries, and no-progress spend

Report per stage:

- blocking questions, auto-answers, reconcile parks, and resume rounds;
- actual provider attempts, not configured maximum retries;
- repeated rejection signatures and whether the one clean-slate escape terminated correctly;
- cost spent on terminal failures;
- cost per design-complete project;
- median and p90 stage wall time/cost where the denominator is large enough; for nine single runs, label quantiles descriptive rather than statistically stable.

Flag:

- any repeated semantic defect after the clean-slate escape;
- any stage whose failure cost exceeds one normal plus one recovery call without reaching a distinct parseable commit defect;
- any BOM round count above 6 or wiring provider-call count above 5;
- any hidden model/provider change within a project.

## Phase 4 — comparison report

Write two artifacts beside the batch summary:

- `$OUT/llm_analysis.json`: machine-readable facts and classifications;
- `$OUT/llm_analysis.md`: evidence-first operator report.

`llm_analysis.json` must contain:

- campaign identity/config/preflight result;
- cohort and integrity verdict;
- one row per slug with archetype, design/build outcomes, failed stage, classification, failure kind, gate signatures, calls, attempts, rounds, tools, cost, wall time, and artifact paths;
- per-stage and per-archetype aggregates;
- BOM, wiring, programming, and question/reconcile diagnostics;
- exact-slug baseline deltas;
- recent-production aggregate and the four unknown-sheet witness outcomes;
- explicit `not_observable` fields rather than guessed values;
- stop-gate verdicts and overall recommendation.

`llm_analysis.md` must use this order:

1. **Verdict:** `PASS`, `PASS_WITH_LLM_FINDINGS`, `FAIL_LLM`, or `INVALID_CAMPAIGN`;
2. **Identity:** commit, dated model, provider/profile, judge, policy, cost;
3. **Canary table:** nine rows, one per archetype;
4. **Recent failure-class deltas:** serialization, schema, BOM sheets/tools, wiring patches/gates, programming, questions;
5. **Production comparison:** aggregate only, with project IDs as evidence pointers;
6. **Build outcomes:** clearly separated from LLM outcomes;
7. **Recommendation:** proceed to repeated campaign, hold deployment, pin/reroute provider, or repair a named contract;
8. **Reproduction:** exact command and artifact paths.

For each rate, show numerator and denominator. Do not write “improved” from a percentage without the matching baseline count.

## Stop gates and acceptance

### Campaign validity

- 9/9 selected records present with matching frozen identity;
- designer and judge preflights pass;
- no hidden model/provider switch;
- costs reconcile and stay within the predeclared campaign envelope;
- all required full-fidelity artifacts exist.

### LLM canary pass

All must hold:

- 9/9 complete all five design stages;
- zero `no JSON in reply`, `truncated_json`, `invalid_json`, `collection_limit`, or `reasoning_loop` terminals;
- zero terminal `invalid_schema` or response-contract failures;
- zero unknown architecture-sheet references in raw candidates that reached local decoding or in committed state;
- BOM tool loops never exceed 6 rounds and wiring never exceeds 5 provider calls;
- zero terminal repeated-signature/no-progress wiring oscillations;
- no invalid candidate bypasses deterministic commit gates;
- exact dated model/profile/provider identity is stable across every run.

Build/fab readiness is reported but is not an LLM canary stop gate unless evidence ties the failure to a committed LLM design defect. A design-complete board that later fails routing is not a structured-output regression.

### Verdict mapping

- `PASS`: campaign valid, all LLM gates pass, no new LLM defect family.
- `PASS_WITH_LLM_FINDINGS`: 9/9 design-complete and no safety/serialization stop gate, but retries/cost/gate corrections show a nonterminal regression worth follow-up.
- `FAIL_LLM`: any design-stage terminal, model drift, unknown-sheet candidate, schema/serialization terminal, invalid commit, or repeated no-progress exhaustion.
- `INVALID_CAMPAIGN`: prerequisites, artifacts, identity, or accounting are not trustworthy.

A `PASS` or `PASS_WITH_LLM_FINDINGS` authorizes a separate three-repeat reliability campaign over the same fixed cohort. It does not by itself authorize a model migration. A `FAIL_LLM` report must name the exact stage, transition, contract/gate, and artifact evidence before proposing code or prompt changes.

## Implementation checklist for the executing agent

1. Read `AGENTS.md`, `kicraft/eval/self_eval.py`, `kicraft/tuning/benchmark.py`, `kicraft/server/stage_runtime.py`, and the two 2026-08-25 LLM reliability plans before execution.
2. Freeze campaign identity and perform designer/judge preflight.
3. Create the output directory and sanitized campaign manifest.
4. Run the exact nine-slug cohort sequentially with full events and one build slot.
5. Resume only interrupted/errored-missing records; never reroll a completed failure.
6. Validate campaign integrity.
7. Analyze canary, exact-slug baseline, recent production window, and unknown-sheet witnesses read-only.
8. Write `llm_analysis.json` and `llm_analysis.md` with the required schema/sections.
9. Verify every numeric claim against its source artifact and show denominators.
10. Deliver the verdict, artifact paths, total spend, and the next concrete action. Do not deploy, restart, or alter model policy as part of this plan.
