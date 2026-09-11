# LLM-stage 34/34 continuation plan

**Status:** execution handoff; continue iterating until acceptance is proven  
**Branch:** `simplify/bom-wiring-pipeline`  
**Starting worktree:** `69fe05c` plus the uncommitted recovery implementation listed below  
**Predecessor:** `docs/plans/llm-stage-34-of-34-recovery-2026-09-10.md`  
**Latest live evidence:** `logs/self_eval/recovery_serializer_pass1_20260911T0400Z`  
**Scope:** exactly five committed design stages: `intent -> functional_spec -> architecture -> bom -> wiring`. Build, placement, routing, judging, and rubric scores are outside the acceptance result.

## Mission and non-negotiable stopping condition

Continue the diagnose/fix/verify loop until KiCraft produces **two fresh complete 34/34 reports** under the production provider/profile and finite budgets:

1. a fresh pre-deploy 34-brief campaign;
2. the independent fresh 34-brief canary run performed by `./deploy/deploy-production.sh`.

Do not stop at a unit-test pass, a frozen replay, a green subset, a process exit code, or a plausible explanation. Do not deploy or restart production before the first complete 34/34 campaign is proven. If a run fails, classify the concrete mechanism, fix the earliest owning pipeline layer, and repeat the smallest affected cohort before attempting another complete campaign.

## Current evidence and honest starting point

The predecessor baseline at `logs/self_eval/design34_fresh_20260910T090000Z` committed **2/34** designs. The recovery implementation materially improved attribution, deterministic ownership, unit validation, boundary enforcement, and wiring repair, but its first fresh serializer cohort did **not** satisfy acceptance:

| Brief | Latest result | Immediate signature |
|---|---|---|
| `usb-c-full-breakout` | five stages committed | current positive sentinel |
| `usb-pd-trigger` | wiring failed | §9.14: `SEL_1`, `SEL_2`, `SEL_3` enter `USB PD TRIGGER` without a carrying pin |
| `can-node` | architecture failed | recipe port completion is unstable; a clean-slate candidate omitted `CAN_TX`, `CAN_RX`, and `+3V3`, while another placed `CANH`/`CANL` in `power_nets` |
| `rounded-c3-devboard` | wiring failed | §9.14: USB data and valid GPIO contracts have no carrying pins on the ESP32-C3 sheet |
| `stm32-min` | budget refused before completion | spent `$0.0648` of `$0.10`; next conservative call ceiling was `$0.0384` |
| `chamfered-badge` | budget refused before completion | spent `$0.0635` of `$0.10`; next conservative call ceiling was `$0.0373` |

Latest cohort summary: `logs/self_eval/recovery_serializer_pass1_20260911T0400Z/summary.json`. It reports `design_committed == 1` of 6. The stable Python suite after the latest edits reports **3214 passed, 15 skipped** via `.venv/bin/pytest -q tests/test_*.py`.

The worktree is intentionally not clean. It contains the predecessor implementation plus follow-up fixes in:

- `kicraft/design/cli_app.py`
- `kicraft/design/recipes/{pin_allocator,resolver,wave_a_mcus,wave_c_interfaces}.py`
- `kicraft/design/synthesis/validation.py`
- `kicraft/eval/{self_eval,stage_failure_report}.py`
- `kicraft/server/{spend_guard,stage_contracts,stage_runtime,stage_work_units}.py`
- the corresponding tests and `tests/fixtures/stage_reliability/recovery_baseline_20260910.json`

Treat these changes as the starting implementation. Review them as a coherent patch; do not reset or overwrite them. Before changing an exported symbol, use LSP references. Preserve unrelated production-box work.

## Decisions

### Correctness before cost minimization

A failed cheap board has zero value. Optimize cost per **successful five-stage commit**, not raw provider spend. Retain finite per-call, per-stage, per-project, daily, and total ceilings, but permit a modest per-project increase when the candidate is making novel progress.

Adopt this budget policy unless fresh measurements justify a smaller value:

- cohort and complete-campaign per-project ceiling: **`$0.15`**;
- single-stage investigation replay ceiling: **up to `$0.25`**, as already supported by `stage_driver replay`;
- no increase to output-token or generic retry counts merely to replay the same failure;
- no automatic ceiling escalation inside a run;
- record `spent`, next-call ceiling, attempts, work-unit IDs, retained siblings, and terminal signature for every budget refusal.

`$0.15` is deliberately modest: the two observed refusals needed approximately `$0.1022` and `$0.1008` to admit their next bounded calls. The extra headroom should admit one useful terminal call without turning repeated identical retries into an open-ended loop. If a design spends more than `$0.10`, its report must show why each additional call was novel: new schema correction, new owning unit, or changed aggregate signature.

### Broader pipeline changes are allowed

Do not constrain the implementation to prompt tweaks. Change the pipeline when evidence shows the model is being asked to reproduce facts already owned by typed state, recipes, symbol pinouts, or commit gates. Preferred order:

1. deterministic contract completion from existing typed evidence;
2. smaller stage/work-unit schema and prompt context;
3. source-stage semantic detection with one focused repair;
4. localized model generation;
5. budget increase only after the preceding options are exhausted.

A broader refactor is acceptable when it removes an entire recurring failure family. It must have a clean cutover, migrate every caller, delete the obsolete path, and retain strict commit gates.

## Invariants

1. **No corpus exceptions.** Never branch on slug, corpus index, archetype label, or verbatim brief text.
2. **No invented electrical meaning.** Do not guess a pin, net, part, voltage, current, package, or substitution. Deterministic completion requires typed recipe, requirement, symbol, connector-port, or architecture evidence.
3. **No weaker gates.** §9.10, §9.11, §9.14, §9.15, §9.26, §9.27, §9.33, schema validation, ERC, and fabrication checks keep their safety meaning.
4. **Origin-stage repair.** Architecture omissions are repaired or rejected in architecture; BOM identity defects in their BOM unit; pin ownership and assignment defects in their wiring unit.
5. **Preserve accepted siblings.** A rejected unit cannot cause unrelated accepted units to be regenerated.
6. **Exactly-once ownership.** Before provider dispatch, every BOM pin must be owned by exactly one locked recipe/lowerer assignment or one wiring unit.
7. **Fixed recipes remain honest.** Do not apply `ch224k-pd-trigger@1` to a selectable-PDO requirement; its verified contract owns one fixed 9 V configuration only.
8. **Fresh evidence only.** A cohort proof uses a new output directory, real provider calls, no `--resume`, and no reused stage answers.
9. **Production safety.** This checkout serves `kicraft.io`; use the canonical deploy script only after pre-deploy 34/34.

## Iteration engine

Every iteration follows the same loop. Do not batch unrelated speculative changes.

### 1. Capture and classify

For each failed run:

- read `summary.json`, `.kicraft/state.json`, and `events.jsonl`;
- reconstruct the exact failing response/work unit when the candidate was not committed;
- record `failed_stage`, `failure_kind`, stable gate codes, schema error, collection field/count, response size, attempt count, work-unit ID, accepted sibling count, cost, and next-call ceiling;
- run `python -m kicraft.eval.stage_failure_report <batch-dir>` and update the mechanism matrix;
- deduplicate against the predecessor plan and current implementation.

Classify the failure into exactly one earliest actionable family:

- serializer/transport;
- schema/alias normalization;
- architecture semantic contract;
- recipe/lowerer selection or ownership;
- BOM identity/sourcing/pad compatibility;
- BOM/wiring stage boundary;
- wiring unit inventory or assignment;
- aggregate commit routing;
- budget scheduler/accounting;
- infrastructure/provider outage.

### 2. Prove the mechanism cheaply

Before editing, create or update the smallest durable reproduction:

- pure unit test for normalizers, ownership, response policy, budgeting, and routing;
- frozen response/work-unit fixture for schema and commit-gate failures;
- `stage_driver replay --stage <stage> --budget 0.25` for a live prompt/guardrail change;
- no full corpus run to test a single hypothesis.

A test must assert consumer-visible behavior: the exact selected recipe, committed net contract, owned pin set, retained sibling, failure code, or budget admission/refusal. Do not assert source text or incidental prompt wording unless the wording is itself the provider contract.

### 3. Fix the earliest owner

Use the decision tree below. Prefer one mechanism-level change over multiple brief-specific patches.

### 4. Verify in layers

1. focused unit/frozen tests;
2. stable suite: `.venv/bin/pytest -q tests/test_*.py`;
3. one affected live brief until its exact terminal mechanism is gone;
4. the full affected cohort once;
5. the identical cohort a second fresh time;
6. only then promote that cohort to green.

When code changes during a running provider batch, mark that batch invalidated. It may provide diagnostic evidence but cannot count toward acceptance.

### 5. Maintain a failure queue

After every cohort, write a compact table into this plan or an adjacent existing status section:

| signature | breadth | latest run | owner | deterministic reproduction | next change | state |
|---|---:|---|---|---|---|---|

Rank by: blocks more briefs, occurs later (wasted spend), repeats across fresh runs, then cost. Do not chase a one-off stochastic wording defect ahead of a systematic ownership or contract failure.

## Workstream A — architecture contract completion and bounded serialization

### Problem

Fresh failures show that schema-valid architecture output can still omit required recipe ports or misclassify signal nets as power. Large numbered families can also reach the stream collection guard before deterministic recipe pin limits can prune them.

### Required changes

1. **Expose conditional requirements in the architecture contract.** Once a requirement resolves uniquely to a recipe, derive a compact list of required external ports and allowed allocatable pins for the architecture repair prompt. Do not ask the model to rediscover recipe pins from prose.
2. **Add a pre-commit recipe-port semantic diagnostic.** Report missing ports with the owning requirement, selected recipe, sheet, and available architecture nets. One repair prompt must request only the missing architecture contracts, not a whole unrelated redesign.
3. **Complete ports only from typed peers.** For example, a transceiver `canh`/`canl` port may bind to a connector requirement's explicitly declared `CANH`/`CANL` port values. If the connector has no typed bus ports, reject and repair architecture; do not synthesize a guessed net.
4. **Separate signals from power.** Architecture validation must reject known recipe signal ports such as CANH/CANL when they appear only in `power_nets`. The repair should move them into the explicit inter-sheet contract, preserving their names.
5. **Bound numbered families before streaming overflow.** The system prompt must require `inter_sheet_net_ranges` for sequential families and state the selected device's valid GPIO range. If the provider still emits a repeated numeric family beyond physical pins, stop once and perform the existing clean-slate compact call; do not raise the global 128-net limit.
6. **Make compact recovery self-sufficient.** The clean-slate response must preserve all required architecture fields and upstream decisions. Add fixtures where the first response overflows `inter_sheet_nets` and the recovery uses one range.
7. **Do not hide semantic omissions as warning-only exhaustion.** If the final architecture candidate lacks a selected recipe's required external contract, architecture must fail with the stable owning diagnostic rather than commit and defer an impossible §9.14 repair to wiring.

### Initial targets

- `can-node`: transceiver controller-side nets, differential bus nets, and +3V3 must all be present and typed before BOM.
- `rounded-c3-devboard`: architecture may expose only ESP32-C3 recipe-valid GPIOs; sequential GPIO output must use ranges or a bounded explicit list.
- `stm32-min`: reduce architecture retry count by turning crystal/programming/native-USB facts into selected-recipe-owned state instead of repeated prose repair.

### Acceptance

- Frozen candidates reproduce the previous missing-port and collection-limit signatures before the fix and produce schema-valid, semantically complete architecture afterward.
- A missing peer connector port remains a blocking architecture diagnostic; no guessed bus net appears.
- Two fresh serialization-cohort runs commit all six briefs under the chosen `$0.15` ceiling.

## Workstream B — deterministic ownership reaches wiring

### Problem

`usb-pd-trigger` and `rounded-c3-devboard` reached wiring with architecture contracts that no owning pin could carry. Either the BOM omitted the implementing device or recipe/lowerer pin allocations did not enter locked wiring ownership.

### Required changes

1. **Assert realizability before BOM commit.** For each recipe/lowerer-owned inter-sheet endpoint, prove that the typed expansion exposes a pin allocation or explicit port binding capable of carrying it. This is not full wiring coverage; it validates only deterministic ownership.
2. **Keep selectable PD model-owned until a verified recipe exists.** The fixed-9V CH224K recipe cannot claim a switch-selectable 9/12/20 V requirement. The USB-C sink lowerer must not intercept a sheet whose typed requirement is a PD trigger.
3. **Make the selectable-PD unit complete.** Its BOM unit must include the controller and selection hardware. Its wiring unit must see `SEL_1..SEL_3`, the exact controller symbol pins, selection component pins, CC nets, supply nets, and the full expected pin inventory. If no verified selectable topology can be expressed, architecture must ask a material question rather than commit a fixed-PDO substitute.
4. **Carry MCU pin allocations into locked wiring.** `RecipeSelection.pin_allocations` is the single source for recipe-valid GPIO and optional native-USB pins. The wiring plan must exclude those pins from model units and merge their connections exactly once.
5. **Check connection ownership before dispatch.** An inter-sheet endpoint on a recipe sheet with neither a locked recipe assignment nor a model-owned expected pin is a local planning error. Fail before spending on a wiring call.
6. **Use exact expected assignment sets.** Unit repair receives the complete expected set plus rejected set and replaces the unit once. Aggregate §9.14 routes to the unit or deterministic selection that owns the missing endpoint.

### Initial targets

- `usb-pd-trigger`: `SEL_1..SEL_3` must terminate on real controller/selector pins or be rejected in architecture/BOM before wiring.
- `rounded-c3-devboard`: `USB_D_P`, `USB_D_N`, and every retained GPIO must come from ESP32-C3 recipe pin allocations; the 2x10 header owns its remaining pins/no-connects deterministically.

### Acceptance

- Unit tests prove every retained GPIO/USB/selector endpoint has exactly one carrying pin before provider dispatch.
- Frozen wiring candidates fail locally when a deterministic selection omits an endpoint.
- Both affected briefs commit wiring twice in fresh runs without weakening §9.14 or §9.15.

## Workstream C — budget scheduler and cost per success

### Problem

The current `$0.10` project ceiling refuses a potentially useful terminal call after approximately `$0.064`. The scheduler is conservative by one whole call ceiling, and expensive upstream retries consume budget needed by wiring.

### Required changes

1. Add a named evaluation/production design-stage budget setting of **`$0.15`**; do not hard-code it in the corpus runner.
2. Keep the preflight invariant: no call begins if its conservative ceiling would exceed the project cap.
3. Reserve budget by remaining mandatory model-owned work units. Before an upstream optional repair, calculate whether admitting it would leave enough ceiling for all uncompleted mandatory units.
4. Prefer deterministic/accepted work before reserving provider calls. Recalculate after every accepted unit and never reserve for recipe/lowerer-owned units.
5. Stop repeated signatures before spending extra budget. The extra `$0.05` is for one novel correction or terminal owning unit, not for a third byte-equivalent retry.
6. Report cost per committed design and stage-level cost. Cohort summaries must distinguish budget refusal, provider error, and content failure.
7. Compare `$0.10` and `$0.15` on `stm32-min` and `chamfered-badge` with identical code. Accept the higher default only if it increases five-stage commit rate without increasing repeated-signature calls.

### Acceptance

- A synthetic scheduler test reproduces the `$0.0648 + $0.0384` refusal at `$0.10` and admission at `$0.15`.
- The call is still refused when it would exceed `$0.15`.
- `stm32-min` and `chamfered-badge` each commit twice; reports show no duplicate-signature retries funded by the larger cap.

## Workstream D — cohort-by-cohort completion

Use new output directories and real production settings. During diagnosis keep full events; counted acceptance runs may use `--lean-events` as specified by the predecessor plan. Always add `--no-judge` and `--design-only`.

### Cohorts

1. **Serialization / current blockers**  
   `usb-pd-trigger,usb-c-full-breakout,stm32-min,can-node,rounded-c3-devboard,chamfered-badge`
2. **Part identity/accountability**  
   `speaker-crossover,rs485-terminal,buck-3a,dual-rail-supply,relay-quad,proto-shield,gpio-expander,servo-driver-16,stepper-a4988,audio-jack-buffer,round-led-ring,star-ornament,lora-node,hex-env-sensor`
3. **Recipe/budget protocol**  
   `rp2040-min,esp32-s3-sensor,nrf52-beacon,encoder-oled-panel,esp32-dual-motor,daq-8ch`
4. **BOM/wiring boundary**  
   `usb-a-power-splitter,led-cc-driver,snowman-ornament`
5. **Wiring ownership**  
   `rc-lowpass-bnc,thermocouple-amp,highside-switch-10a`
6. **Regression sentinels**  
   `r2r-dac,fpc-breakout`

### Counted command template

```bash
.venv/bin/python -m kicraft.eval.self_eval \
  --out logs/self_eval/<cohort>_pass<N>_<UTC> \
  --design-only --no-judge --lean-events --parallel 3 \
  --only <comma-separated-cohort>
```

Use the configured `$0.15` per-project ceiling. A run counts only when its `summary.json` exists and every selected brief has:

- `design_committed is true`;
- exactly five `stage_done` events with `ok: true` for the required stages;
- no reused/cached stage answer from another batch;
- no hidden budget/provider error;
- no source edit after that batch process started.

Require **two consecutive fresh passes** for each cohort. If pass 2 fails, invalidate the pair, reopen its mechanism, and obtain two new consecutive passes after the fix.

## Workstream E — full 34/34 acceptance

Only begin after all six cohorts have two consecutive passes on the same source revision.

### Pre-deploy campaign

```bash
.venv/bin/python -m kicraft.eval.self_eval \
  --out logs/self_eval/design34_acceptance_<UTC> \
  --design-only --no-judge --lean-events --parallel 3
```

Validate with the report module and an independent structured check. Required:

- `n == 34`;
- `design_committed == 34`;
- every run committed all five required stages;
- no omitted, duplicated, or substituted brief;
- no `--only`, `--limit`, or `--resume`;
- fresh output directory;
- spend within global/daily ceilings and the configured per-project `$0.15` ceiling.

If this campaign is less than 34/34, do **not** rerun it unchanged. Classify failures, return to the smallest affected cohort, require two consecutive cohort passes on the new code, then run a new complete campaign.

### Canonical production gate

After the pre-deploy campaign is exactly 34/34:

```bash
./deploy/deploy-production.sh
```

Do not use `systemctl` or direct process kills. The deploy script must run its own independent fresh all-34 canary and pass before restart.

After deploy:

```bash
curl -sf http://127.0.0.1:8080/
```

Then verify `logs/kicraft_build_worker.log` ends with `[build-worker] ready`. Read both complete campaign reports and record their directories, commit SHA, total spend, and exact 34/34 result.

## Tests and evidence that must remain

Keep tests only for durable contracts:

- exact stage/failure/work-unit attribution;
- unit-specific serialization bounds and one clean-slate retry;
- concrete schema errors in events/ledger;
- sheet alias collision rejection;
- compact numbered architecture ranges;
- typed exact-part propagation and substitution accountability;
- symbol/footprint/pad validation inside owning BOM units;
- recipe/lowerer vs model ownership exclusion;
- exact duplicate pruning and conflicting duplicate rejection;
- BOM partial-boundary behavior and full wiring gates;
- exactly-once wiring pin ownership;
- aggregate offender-to-unit routing;
- `$0.10` vs `$0.15` budget admission/refusal;
- report classification of all 34 baseline runs.

Do not add tests that assert prompt prose, field copies, implementation wiring, or tautological non-empty output. Use live cohorts for provider behavior.

## Commit discipline

Keep commits mechanism-sized and bisectable:

1. `Complete recipe-required architecture contracts`
2. `Carry deterministic port ownership into wiring`
3. `Keep selectable PD topology model-owned`
4. `Reserve budget for remaining model units`
5. `Converge identity and sourcing cohorts`
6. `Converge boundary and wiring cohorts`
7. `Prove fresh 34-of-34 stage completion`

Before the first commit, review and include the existing uncommitted predecessor implementation rather than discarding it. Never commit `logs/self_eval/*`, generated projects, replay workspaces, provider response dumps, secrets, or binary artifacts. The minimized JSON fixture and report module are source artifacts and should be committed.

## Cleanup

After the smoke/campaign proof, not before:

- remove throwaway scripts and temporary replay workspaces;
- retain only minimized secret-free regression fixtures;
- leave live run directories untracked as evidence or remove only directories created by this effort when storage cleanup is required;
- update the predecessor and this plan with final commit SHA, cohort directories, both 34/34 directories, spend, and deployment health evidence;
- update an existing changelog only if the repository already requires one; do not create a new changelog.

## Completion checklist

- [ ] Review and commit the existing recovery worktree without dropping user changes.
- [ ] Implement typed architecture recipe-port completeness and signal/power validation.
- [ ] Make selectable-PD and MCU GPIO/USB wiring ownership realizable before dispatch.
- [ ] Add configurable `$0.15` per-project design-stage ceiling with strict preflight.
- [ ] Prove the serialization cohort twice consecutively.
- [ ] Prove the identity cohort twice consecutively.
- [ ] Prove the recipe/budget cohort twice consecutively.
- [ ] Prove the BOM/wiring boundary cohort twice consecutively.
- [ ] Prove the wiring ownership cohort twice consecutively.
- [ ] Prove regression sentinels twice consecutively.
- [ ] Run one fresh complete pre-deploy campaign with `34/34` five-stage commits.
- [ ] Run `./deploy/deploy-production.sh`; its independent fresh canary also reports `34/34`.
- [ ] Verify HTTP 200 and `[build-worker] ready` after canonical deployment.
- [ ] Record exact commit SHA, report directories, spend, and health evidence.
- [ ] Remove throwaway artifacts and update plan status.

**Completion is only the checked state above. Continue iterating until every box is satisfied.**
