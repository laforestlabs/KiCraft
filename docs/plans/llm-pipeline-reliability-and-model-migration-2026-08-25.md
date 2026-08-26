# LLM pipeline reliability recovery and model migration — 2026-08-25

**Status:** proposed
**Primary evidence:** `docs/plans/live-llm-reliability-verification-2026-08-25.md`
**Canary:** `logs/live_verification/20260825T114515Z/`
**Current live roles:** design `deepseek/deepseek-v4-flash`; electrical review and Class-J fallback `minimax/minimax-m3`

## Decision

Recover the stage contracts and retry behavior first, then compare the two dated DeepSeek V4 revisions on frozen inputs. Do not deploy another mutable model alias, silently switch models inside a run, raise token limits, add retries, or weaken an electrical gate.

Production target:

- **Main designer, default cost profile:** `deepseek/deepseek-v4-flash-0731`.
- **Main designer, explicit quality profile:** `deepseek/deepseek-v4-pro-0813`.
- **Electrical reviewer candidate:** `deepseek/deepseek-v4-pro-0813`, promoted only if it passes the existing labeled electrical-review corpus. The older Pro revision had 100% blocker recall but unacceptable 57% clean false-blocking, so a model-name upgrade alone is not evidence of suitability.
- **Class-J judge candidate:** `deepseek/deepseek-v4-pro-0813`, promoted independently on frozen digests. Judge and electrical-review decisions remain separate settings and policies even if they resolve to the same model.

The operator chooses Flash or Pro before a design run. Every stage and spend row records the dated model and provider. A failed Flash run may be re-run explicitly with Pro, but the pipeline must not fail over between models inside one stage: hidden failover makes cost, reproducibility, and defect attribution unknowable.

Official model records:

- Flash 0731: <https://openrouter.ai/deepseek/deepseek-v4-flash-0731>
- Pro 0813: <https://openrouter.ai/deepseek/deepseek-v4-pro-0813>

OpenRouter currently lists Flash at $0.035/M input and $0.10/M output at its cheapest endpoint, and Pro at $0.66/M input and $1.98/M output at its base endpoint. Provider prices and capabilities vary; these are discovery data, not production routing guarantees.

## What the canary proved

The safety work is correct but completion recovery is not:

| Path | Evidence | Conclusion |
|---|---|---|
| Class-J judge | 4/4 valid, no reasoning abort, $0.015227 | Keep the call-specific `eval_judge` guard. |
| BOM collection guard | 0 truncations and 0 oversized accepted slots | Keep both streaming and parsed-object bounds. |
| BOM completion | 4/7 replay commits; fresh BOM failures cost $0.056442 and $0.078910 | The model still degenerates or emits no JSON; the recovery re-solves the entire BOM and repeats the failure. |
| Wiring safety | No rejected topology committed; five failures ended as typed `commit_rejected` | Keep §9 gates and the five-call ceiling. |
| Wiring completion | 3/8 commits; all eight first calls hit the reasoning hard ceiling | The current policy predictably buys a failed reasoning call before useful wiring begins. |
| Fresh chain | 1/3 through wiring | The control paths are not reliable enough for a full paid batch. |

Three BOM replays (`rp2040-min`, `nrf52-beacon`, `daq-8ch`) stopped on item 451 of one sheet after emitting 50–57 kB. These were not legitimate 451-part sheets. The successful replay BOMs contained 24–31 parts. The three failed replays alone cost $0.248162.

The wiring retries show correction progress, but whole-slot regeneration oscillates: a short is removed, a dangling net appears, and a later full rewrite reintroduces a short. After the initial reasoning abort, up to four more provider calls rewrite already-correct pins rather than repairing only the rejected endpoints.

Cost regression is also real. The 2026-08-05 dated-Flash batch cost $1.0523 total; the 2026-08-23 mutable-alias batch cost $1.6419. Median design cost rose from $0.0215 to $0.0451. The 2026-08-25 source batch then spent $0.9687 on judges because the wrong reasoning guard aborted normal judge work. That judge defect is fixed; BOM and wiring still spend heavily on failed recovery.

## Root causes to fix

1. **Mutable model identity.** `.env` and the default `Settings.model` use `deepseek/deepseek-v4-flash`, while a prior batch explicitly used `deepseek/deepseek-v4-flash-0731`. An alias can move without a code or configuration change.
2. **Routing policy is tied to the old cheap tier.** The design route defaults to `novita/fp8,siliconflow/fp8,streamlake` with $0.18/M input and $0.35/M output caps. The new Flash and Pro endpoints differ in price, cache behavior, and structured-output support. Pro cannot route under the current design price caps.
3. **Plain-text slot serialization.** Stage prompts request JSON, but final replies are not constrained by a provider JSON schema. The fresh LED matrix exhausted three BOM calls and returned no JSON.
4. **Repeated BOMs are serialized one component at a time.** A valid 400-part array and a degenerate 800-passive sequence look equally expensive until the stream counter stops them.
5. **Serialization recovery discards useful resolution state.** It correctly drops the huge tool transcript and rejected partial draft, but then asks a reasoning-disabled model with no tools to reconstruct the complete BOM from the original state. That reproduces collection degeneration.
6. **Wiring starts with a policy that failed 8/8 relevant canaries.** Each replay first spends a reasoning-enabled call that reaches the hard ceiling, then retries with reasoning disabled.
7. **Wiring correction regenerates the whole netlist.** Deterministic feedback is endpoint-specific, but the model returns every connection again. Unrelated valid wiring churns between attempts.
8. **Budgets cap total damage, not no-progress spend.** The daily ledger remains essential, but it does not stop a stage as soon as the same semantic defect or serialization degeneration repeats.

## Phase 1 — pin role models and make routing preflightable

### Change

1. Change the design default and deployed `KICRAFT_MODEL` to the dated Flash ID. Keep `Settings.model` as the single main-designer selector; do not introduce a second overlapping design-model setting.
2. Set `KICRAFT_REVIEW_MODEL` and `KICRAFT_EVAL_JUDGE_MODEL` explicitly. Remove the implicit judge fallback to the review model from production configuration; code may retain a safe fallback for local use, but campaign manifests must show the resolved role model.
3. Add an admin model preflight that checks each configured role against OpenRouter endpoint metadata before a paid campaign:
   - exact model ID exists;
   - selected providers serve that exact revision;
   - streaming, tools, `response_format`, and reasoning controls required by the role are supported;
   - an endpoint survives the configured price caps;
   - one bounded schema-response smoke call succeeds.
4. Re-benchmark provider order separately for Flash and Pro. Do not carry the old alias's provider order forward by assumption. Prefer a verified provider with stable JSON/tool behavior over the nominal cheapest endpoint; then set a hard price ceiling immediately above the chosen route, not `0.0`.
5. Add an explicit design profile selector in deployment/config tooling, not in stage logic:
   - `flash`: dated Flash model plus its verified provider order and price caps;
   - `pro`: dated Pro model plus its verified provider order and price caps.
   Both resolve into the existing `Settings` fields before client construction.
6. Stamp model ID, provider tag, profile, response policy name, prompt/output/reasoning tokens, cache-read tokens, cost, and finish reason into stage ledger rows and campaign manifests.
7. Fail startup/preflight on an internally inconsistent known profile, such as Pro with the Flash price ceiling. Never relax the cap automatically.

### Files

- `kicraft/server/config.py`
- `kicraft/server/client.py`
- `kicraft/server/stage_driver.py`
- `kicraft/eval/self_eval.py`
- `kicraft/eval/run_web.py`
- `.env.example`
- deployment configuration that supplies the live environment
- provider-benchmark/admin command tests following existing CLI conventions

### Acceptance

- Default and example configuration contain no unversioned DeepSeek V4 alias.
- Flash and Pro profiles each resolve to one dated model and a finite provider price ceiling.
- A Pro selection cannot silently reuse an incompatible Flash cap or route.
- Two runs with the same profile expose the same model ID in every stage, review, judge, ledger, and manifest record.
- A provider missing tools or schema output fails preflight before the stage cohort starts.

## Phase 2 — make stage output schema-bound and compact

### Change

1. Thread an optional `response_format` through `CappedOpenRouterClient.chat` and `chat_with_tools`. Build it from the existing Pydantic slot models, with a separate wiring response schema. Use it on every final slot-producing response, including serialization recovery.
2. Keep tool arguments and final slot output distinct. BOM tool rounds remain ordinary tool calls; the final assistant response must satisfy the slot response schema.
3. Add a **stage-only compact BOM emission shape** for repeated identical components. It contains:
   - ordinary `parts` for heterogeneous parts;
   - bounded `part_runs` with a reference prefix/range or explicit reference list plus one shared value/symbol/footprint/sheet payload;
   - the existing placement/array metadata.
4. Expand `part_runs` deterministically before Pydantic BOM validation and commit. Check total count, per-sheet count, duplicate refs, malformed ranges, and conflicts with ordinary parts before allocating or expanding the canonical list. Persist only the existing canonical `BOM.parts` representation; no downstream schema changes.
5. Preserve the canonical 500-total/450-per-sheet gates. A compact declaration of 501 total or 451 on one sheet fails before expansion. A legitimate 400-member array remains expressible in a small response.
6. Have the BOM executor retain a small, deterministic resolution ledger: requested part, accepted LCSC ID, exact symbol, exact footprint, and source tool. On serialization recovery, send the pristine project state plus this bounded ledger. Do not send the full tool transcript or accept/salvage the stopped draft.
7. Use one schema-bound, reasoning-disabled recovery call at the existing fixed cap. If it repeats `collection_limit` or invalid schema, terminate. No cap growth and no second recovery.
8. Add per-call output metrics for emitted collection count and compact-run expanded count so cost regressions cannot hide behind a successful commit.

### Files

- `kicraft/server/client.py`
- `kicraft/server/stage_driver.py`
- `kicraft/server/config.py`
- `kicraft/design/models.py` or a stage-response-only model beside `stage_driver`
- `tests/test_client_provider.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_kicraft_validation.py`
- `tests/test_kicraft_stage_cli.py`

### Acceptance

- Every non-tool final stage response requests a JSON schema; unsupported routes are rejected by preflight rather than downgraded to unconstrained text.
- The fresh LED-matrix BOM cannot end as `no JSON in reply` without first producing a typed provider/schema failure.
- A 400-part repeated array round-trips through compact emission into the unchanged canonical BOM.
- 501 total, 451 on one sheet, overlapping ranges, duplicate refs, and invalid reference sequences fail before expansion/commit.
- Replaying the seven BOM states yields zero `truncated_json`, zero oversized accepted slots, and at least 6/7 commits in each promoted designer arm.
- No failed BOM replay costs more than one normal attempt plus one bounded recovery unless it reached a parseable slot and is spending the existing commit-correction budget on distinct deterministic defects.

## Phase 3 — repair wiring deltas instead of rewriting the board

### Change

1. Disable reasoning on the first wiring call for both initial designer candidates. The canary's relevant current-model cohort hit the hard ceiling 8/8 times. Re-enable it only for a model/profile that wins a repeated wiring-policy experiment.
2. Keep the first wiring response schema-bound and full-slot. Once it parses, retain it as the candidate topology even if commit rejects it.
3. Convert deterministic commit feedback into a small correction request whose response is a typed patch, not another whole wiring slot. Patch operations may add/remove an endpoint, set a pin's net, add/remove a connection, or mark/unmark a no-connect. Each operation names the expected current value so stale or contradictory patches fail closed.
4. Apply the patch to a copy of the last candidate, then run the complete existing commit gates. The LLM never chooses a net outside this reviewed patch, and no invalid candidate is persisted.
5. Preserve already-valid endpoints across correction calls. Send only the relevant sheet, offender pins, pinouts, architecture nets, and current connections touching those pins.
6. Normalize rejection signatures by gate ID and offender identity. One repeated signature gets one clean-slate, reasoning-disabled escape correction; persistence after that is terminal `commit_rejected`.
7. Keep the existing five-provider-call maximum. Patch calls count against it. Do not add retries. Stop earlier when no progress is proven.
8. Preserve question parking and wiring-to-BOM reconciliation unchanged; a genuine missing-part condition still re-drives BOM rather than being patched into a no-connect.

### Files

- `kicraft/server/stage_driver.py`
- stage-response models used by wiring
- `kicraft/server/config.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_kicraft_validation.py`
- wiring reconciliation/park tests

### Acceptance

- Synthetic correction changes only the named endpoints; unrelated valid wiring is byte-for-byte stable after canonical sorting.
- Stale preconditions, unknown refs/pins/nets, duplicate endpoint assignments, and malformed patches fail without mutating the retained candidate.
- Every patched candidate passes the complete §9 gate set before commit.
- The eight frozen terminal wiring states commit at least 6/8 per promoted arm, with zero rejected topology committed and no run over five provider calls.
- Identical-signature no-progress terminates early; distinct, shrinking offender sets may continue within the same budget.
- Wiring cost per successful replay and cost wasted on terminal failures are both reported.

## Phase 4 — qualify the new role models independently

Do not combine model selection with a single end-to-end score. A designer, electrical reviewer, and Class-J judge have different correctness contracts.

### Main designer bakeoff

Run dated Flash and dated Pro against identical frozen workspaces after Phases 1–3:

- seven pre-BOM overflow states;
- eight terminal pre-wiring states;
- the three fresh briefs;
- three repeats per state/profile, with identical prompts, gates, retry budgets, and provider pins.

Report completion, first-pass completion, calls, failure kinds, deterministic gate signatures, wall time, prompt/output/reasoning/cache tokens, provider, and cost per successful project. Promote Flash as the default only if it meets all reliability gates and restores a competitive cost envelope. Keep Pro as an explicit quality profile; make it the default only if its completion/quality gain justifies its measured cost rather than its advertised capability.

Designer promotion gates:

- BOM: at least 5/7 commits in each repeat and at least 18/21 across all three repeats, with no truncation or accepted overflow;
- wiring: at least 5/8 commits in each repeat and at least 18/24 across all three repeats, with zero invalid commits and a maximum of five calls per run;
- fresh: at least 2/3 complete through wiring in each repeat and at least 8/9 across all three repeats;
- zero untyped terminal failures;
- Flash median design cost at or below the 2026-08-05 $0.0215 baseline and p90 at or below $0.05 after provider tuning;
- Pro has an explicit observed per-project budget and is never selected under Flash's dollar ceiling.

### Electrical reviewer bakeoff

Replay the frozen, hand-labeled corpus from `docs/electrical_review_model_bakeoff.md` with Pro 0813 at reasoning off/medium before changing the production gate. Keep the same digest, labels, severity mapping, corroboration, wall ceiling, and parser.

Promotion gates are at least as strong as the incumbent winner:

- 100% recall on real blocker designs;
- clean false-block rate no worse than 14%;
- warning-board false-block rate no worse than 20%;
- 100% valid structured verdicts;
- no call exceeds the finite review wall/reasoning ceiling;
- mean cost and p90 latency reported.

If Pro misses a gate, retain Minimax M3 for electrical review. The request to modernize models does not justify shipping a reviewer already known to risk false fabrication blocks.

### Class-J judge bakeoff

Re-grade all 34 frozen digests with Pro 0813 for three repeats. This is digest-only: no design/build calls.

Promotion gates:

- 102/102 valid verdicts;
- every row uses `eval_judge`, never the design reasoning policy;
- zero hard-ceiling, repetition, wall-stall, or schema aborts;
- all rubric dimensions present and legal;
- repeat stability reported per dimension and grade band;
- total and per-digest cost reported under a dedicated judge campaign ceiling.

Do not use agreement with a DeepSeek-authored design as the only quality signal. Retain deterministic Class-C dimensions, compare against the existing frozen Minimax verdicts, and manually adjudicate every material disagreement on the labeled reviewer subset.

## Phase 5 — staged rollout and stop rules

1. Land deterministic schema/patch/config tests.
2. Run model/provider preflight for both designer profiles and the two new role candidates.
3. Run BOM, wiring, reviewer, and judge cohorts independently. Stop only the failing branch; do not pay for downstream experiments whose prerequisite failed.
4. Run the three fresh briefs for the winning designer profile.
5. Run one 34-brief batch with dated Flash. Run Pro only as a separate campaign, never mixed into the same aggregate.
6. Promote the dated Flash profile to production if the full-batch gates pass. Expose Pro as an explicit operator-selected profile after its own gates pass.
7. Promote reviewer and judge independently. A reviewer miss does not block the designer migration; a judge miss does not affect production design/build.
8. Keep the old campaign directories and source-state hashes immutable. Write a machine-readable manifest with model IDs, providers, profile, code revision, policies, caps, and corpus hashes.

Full-batch gates:

- 34/34 Class-J verdicts valid;
- design completion at least 19/34, with a target above the 2026-08-23 baseline rather than accepting the canary's regression;
- zero terminal collection overflow/truncation;
- zero untyped failures;
- all completed designs enter build;
- conditional fab-ready rate at least 89.5%;
- no gate suppression, partial-slot acceptance, hidden model fallback, cap growth, or retry increase;
- Flash total design spend no higher than $1.00 and total judge spend no higher than $0.45;
- stage p50/p90 cost and failure-waste cost improve versus both the 2026-08-23 batch and the 2026-08-25 canary.

Immediate stop conditions:

- source-state hash changes;
- a rejected BOM/wiring slot is committed;
- selected provider/model differs from the manifest;
- a route exceeds its configured price cap;
- less than the campaign reserve remains;
- two identical uncontrolled degeneration signatures appear after the bounded recovery.

## Verification

Deterministic tests, run once after implementation:

```bash
.venv/bin/python -m pytest -q \
  tests/test_client_provider.py \
  tests/test_stage_driver_retry.py \
  tests/test_stage_resource_telemetry.py \
  tests/test_kicraft_validation.py \
  tests/test_kicraft_stage_cli.py \
  tests/test_web_self_eval.py \
  tests/test_self_eval.py
```

Required behavioral evidence before a full batch:

1. Provider/model preflight artifacts for Flash, Pro, reviewer, and judge.
2. Machine-readable BOM replay report with expanded counts and schema outcomes.
3. Machine-readable wiring replay report with patch operations and rejection-signature progression.
4. Electrical-review labeled-corpus report.
5. Three-repeat frozen-digest judge report.
6. Fresh-brief report with exact stage costs and terminal classifications.

Only after all prerequisite gates pass:

```bash
.venv/bin/python -m kicraft.eval.self_eval
```

## Non-goals

- No router, placer, DRC, outline, rubric-weight, or brief-specific tuning in this recovery.
- No larger token caps or retry budgets.
- No salvage of truncated JSON.
- No weakening 500/450 BOM bounds, §9 wiring gates, electrical-review severity rules, or spend ceilings.
- No automatic net selection, part deletion, or model failover.
- No claim that a newer model is better until its role-specific frozen corpus passes.
