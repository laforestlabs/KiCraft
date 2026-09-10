# LLM-stage 34/34 recovery plan

**Status:** implementation handoff  
**Baseline commit:** `8c4d058` on `simplify/bom-wiring-pipeline`  
**Fresh baseline:** `/home/kicraft/KiCraft/logs/self_eval/design34_fresh_20260910T090000Z`  
**Model/profile:** `deepseek/deepseek-v4-flash-0731` / production `flash` profile  
**Scope:** `intent -> functional_spec -> architecture -> bom -> wiring`; build, placement, routing, judging, and rubric scores are not acceptance criteria for this plan.

## Decision

Reach 34/34 by reducing what the model can get wrong and repairing only the smallest rejected unit. Do not weaken schema, sourcing, pin, net, ERC, or fabrication gates; do not special-case corpus slugs; do not raise token, retry, or project-spend limits as a substitute for fixing generation.

The fresh baseline is **2/34** five-stage commits. Only `r2r-dac` and `fpc-breakout` passed. The dominant blocker is BOM generation: 23 runs stopped at BOM, five at architecture, three at wiring, and one at functional spec.

The implementation order is intentional:

1. Freeze the exact failed candidates and make every rejection attributable.
2. Repair serializer/schema/collection failures before electrical content work.
3. Move resolvable part identity and recipe ownership out of model output.
4. Fix BOM-stage ownership so wiring-only gates run after wiring exists.
5. Repair wiring at the owning unit, without redrafting accepted work.
6. Run cohort canaries, then two entirely fresh 34-brief acceptance runs.

## Baseline and failure inventory

Authoritative result: `summary.json` in the fresh baseline directory reports `design_committed=2`, `n=34`, spend `$0.402`, and wall time approximately 1 h 54 min.

| Terminal stage | Count | Briefs |
|---|---:|---|
| passed | 2 | `r2r-dac`, `fpc-breakout` |
| functional_spec | 1 | `usb-pd-trigger` |
| architecture | 5 | `usb-c-full-breakout`, `stm32-min`, `can-node`, `rounded-c3-devboard`, `chamfered-badge` |
| bom | 23 | `speaker-crossover`, `usb-a-power-splitter`, `rs485-terminal`, `rp2040-min`, `esp32-s3-sensor`, `nrf52-beacon`, `lora-node`, `buck-3a`, `led-cc-driver`, `dual-rail-supply`, `relay-quad`, `encoder-oled-panel`, `proto-shield`, `esp32-dual-motor`, `daq-8ch`, `gpio-expander`, `servo-driver-16`, `stepper-a4988`, `audio-jack-buffer`, `round-led-ring`, `hex-env-sensor`, `star-ornament`, `snowman-ornament` |
| wiring | 3 | `rc-lowpass-bnc`, `thermocouple-amp`, `highside-switch-10a` |

Observed failure families overlap; counts below assign each run to its first actionable family.

| Failure family | Count | Briefs | Representative evidence |
|---|---:|---|---|
| BOM symbol/footprint/pad validity | 10 | `speaker-crossover`, `dual-rail-supply`, `relay-quad`, `proto-shield`, `gpio-expander`, `servo-driver-16`, `stepper-a4988`, `audio-jack-buffer`, `round-led-ring`, `star-ornament` | Hallucinated stock footprint names, a footprint library used as a symbol, and incompatible symbol pin/footprint pad numbering reached aggregate BOM commit. |
| BOM work-unit/recipe/budget protocol | 6 | `rp2040-min`, `esp32-s3-sensor`, `nrf52-beacon`, `encoder-oled-panel`, `esp32-dual-motor`, `daq-8ch` | `collection_limit`, provider-call budget exhaustion, `recipe-duplicate`, or `model_authored_protected_identity` exhausted a unit rather than pruning deterministic ownership. |
| Architecture serialization/schema | 5 | architecture cohort above | Four repeated `invalid_schema` terminals and one repeated `collection_limit`; no candidate committed. |
| Wiring pin/net ownership | 3 | wiring cohort above | Repeated duplicate/missing/unexpected pin assignments, or aggregate dangling `GATE_CTRL`. |
| BOM connection/net completeness | 3 | `usb-a-power-splitter`, `led-cc-driver`, `snowman-ornament` | Full §9.11/§9.14 coverage ran during BOM because deterministic connections existed while model-authored groups were intentionally not wired yet. |
| BOM orderability/MPN validity | 2 | `rs485-terminal`, `buck-3a` | Literal `MPN: "N/A"` reached §9.26 instead of a concrete stocked identity or an explicit unresolved-part decision. |
| BOM named-part accountability | 2 | `lora-node`, `hex-env-sensor` | Architecture named core defaults such as `ME6211C33`, but BOM neither shipped that part nor recorded a substitution. |
| Functional-spec serialization | 1 | `usb-pd-trigger` | Repeated `collection_limit`. |

Do not infer fixes only from this table. For every listed run, the implementing agent must use its immutable `.kicraft/state.json` and `events.jsonl` to identify the exact attempt, candidate, work unit, collection field, commit result, and owning refs.

## Invariants and non-goals

1. **No benchmark exceptions.** No branch on slug, prompt index, archetype label, or verbatim brief text.
2. **No weaker gates.** §9.10, §9.11, §9.14, §9.15, §9.26, §9.27, §9.33, schema validation, ERC, and downstream fabrication gates retain their safety meaning.
3. **Stage ownership is strict.** BOM proves component identities and deterministic fragments. Wiring proves complete pin/net coverage. A BOM containing some deterministic connections must not be mistaken for a complete wired design.
4. **Lossless normalization only.** Case, sentinel-null, exact duplicate rows, and alias normalization are allowed only when meaning is unambiguous. Never guess a net, part, package, voltage, current, or substitution.
5. **Preserve accepted work.** A rejected unit must not regenerate accepted sibling units. An aggregate rejection must be routed to its owning unit(s).
6. **Bounded calls.** Do not increase retry counts, output caps, or spend ceilings until a frozen replay proves the current bound cannot represent a valid minimal response.
7. **Real-provider proof.** Unit tests and frozen replays prove code paths; only fresh, non-resumed provider runs prove the 34/34 requirement.
8. **Acceptance means exactly five committed LLM stages.** A build or judge result is deliberately out of scope and must be reported separately.

## Phase 0 — freeze evidence and add an attributable cohort report

### Change

1. Create minimized fixtures from the 32 failed run directories. Each fixture keeps only committed upstream state, the rejected response/candidate or work unit, the structured rejection, and expected stage/failure family. Exclude reasoning text, credentials, generated boards, and unrelated event traffic.
2. Preserve at least one fixture for every distinct mechanism:
   - functional-spec collection overflow;
   - architecture invalid schema and collection overflow;
   - unresolved footprint;
   - unresolved symbol;
   - symbol/pad mismatch;
   - `N/A` MPN;
   - named-part omission;
   - recipe duplicate;
   - protected-identity collision;
   - BOM-time partial-wiring coverage;
   - duplicate/missing wiring pins;
   - aggregate dangling signal net.
3. Extend the self-eval summary/report with `failed_stage`, `failure_kind`, stable commit gate codes, and work-unit ID where available. The summary must not require parsing a human error string to count failure classes.
4. Add a small analysis command/module that reads a completed self-eval directory and prints the stage/family matrix above. It must be read-only and operate without provider calls.

### Files

- `kicraft/eval/self_eval.py`
- `kicraft/eval/llm_analysis.py` or a narrowly scoped adjacent report module
- `kicraft/server/stage_runtime.py`
- `kicraft/server/spend_guard.py` if attempt rows still omit gate/work-unit attribution
- `tests/fixtures/stage_reliability/` or the existing fixture convention nearest stage-driver tests
- `tests/test_self_eval.py`
- `tests/test_stage_driver_retry.py`

### Acceptance

- The fresh baseline is classified deterministically as 1 functional-spec, 5 architecture, 23 BOM, 3 wiring, and 2 passed.
- Every `commit_rejected` attempt with structured `errors` records at least one stable gate/failure code.
- Every work-unit terminal records its exact unit ID and whether accepted siblings were retained.
- Fixtures contain no secrets, raw reasoning, or generated binary artifacts.

## Phase 1 — make structured response recovery converge

### 1.1 Diagnose exact schema defects before changing normalization

For each of the five architecture failures and the functional-spec failure, reconstruct each response from `answer_delta` events and run it through `decode_stage_response` / `_normalize_stage_response` without committing. Record the exact `StageSchemaError`, collection field/count, finish reason, response-format mode, and response size. The current event summary's generic `provider response did not satisfy...` is insufficient.

Separate three cases:

1. provider rejected or failed native response-format enforcement;
2. valid JSON failed local Pydantic/schema validation;
3. the streaming collection guard stopped an otherwise recoverable response.

### 1.2 Fix only demonstrated serializer defects

1. In `stage_runtime.py`, ensure recovery prompts use the **unit-specific** response policy. The work-unit path currently formats its first serialization retry from `policy.collection_bounds`; it must use `response_policy_for_unit(unit).collection_bounds`, matching the actual stream guard.
2. Persist the concrete schema error in attempt telemetry and retry events, redacted only where needed. Do not collapse it to `invalid_schema` before building the next correction prompt.
3. Apply existing alias/case normalization before Pydantic validation only where it is one-to-one and contract-owned:
   - canonical uppercase architecture sheet names;
   - canonical uppercase underscore stems;
   - known sheet endpoint aliases;
   - omission of null optional fields.
   Reject collisions after normalization rather than silently merging sheets.
4. If native `response_format` is the source of repeated provider-side failures, use the existing plain compact-JSON serialization recovery for that attempt. Do not disable local schema validation.
5. Correct collection bounds only if the frozen valid target cannot fit. Prefer compact range fields and per-unit bounds over a larger global collection or token cap.
6. When the same schema or collection signature repeats, replace the response with a clean-slate compact serialization call once; then stop with the exact terminal signature. Do not spend generic retries replaying identical instructions.

### Files

- `kicraft/server/stage_runtime.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/server/stage_prompts.py`
- `kicraft/server/client.py` only if the collection guard or response-format seam is proven causal
- `kicraft/server/config.py` only if a frozen valid response proves a bound incorrect
- `.agents/skills/kicraft/stages/functional_spec.md`
- `.agents/skills/kicraft/stages/architecture.md`
- `tests/test_stage_driver_retry.py`
- `tests/test_stage_driver_prompt_examples.py`
- `tests/test_kicraft_models.py`

### Acceptance

- All six frozen functional-spec/architecture failures decode or recover to schema-valid candidates under existing finite token and call budgets.
- No normalization converts two distinct sheet names/stems into one.
- A repeated invalid-schema signature does not consume the full generic retry budget.
- Two fresh cohort runs pass: `usb-pd-trigger,usb-c-full-breakout,stm32-min,can-node,rounded-c3-devboard,chamfered-badge`.

## Phase 2 — validate and resolve BOM identity inside each work unit

Aggregate BOM commit is too late to discover a hallucinated library identifier. By then the model may have produced many accepted siblings and aggregate repair has little context about the specific part selection.

### 2.1 Use the synthesis resolver before accepting a unit

1. Thread `extras`/project-root resolution context through `_validate_bom_unit`; today `validate_unit_candidate` receives `extras` but drops it on the BOM path.
2. Resolve every emitted symbol and footprint through the same tiered resolver and `pcbnew.FootprintLoad` seam used by BOM commit/synthesis.
3. Check symbol pin inventory and footprint pad compatibility (§9.27) before marking the unit accepted.
4. Return a compact structured defect containing group ID, bad field, rejected identifier, and a bounded list of real candidates already available in the curated parts table or stock KiCad library.
5. Repair only that unit. Accepted sibling units remain in `DraftStore`.

### 2.2 Reduce free-form identity generation

1. Prefer exact bundle/core-default rows supplied by `stage-prep`; represent the choice by stable bundle/default identity and derive symbol, footprint, MPN, and sourcing metadata deterministically.
2. Add deterministic registry entries for recurring generic physical parts that already have clear stock-KiCad identities: common passives, standard headers, common battery holders, relays, and terminal blocks. Registry entries must resolve through the real KiCad installation and pass pad compatibility at startup/test time.
3. For non-registry ICs/connectors, require `resolve_and_bundle` or an exact curated row. Do not let the model synthesize a plausible library string after a lookup miss.
4. Treat case-insensitive sentinel values `N/A`, `NA`, `none`, `unknown`, and empty strings as absent optional metadata at decode time. Absence does not satisfy sourcing: a purchasable part still needs a real identity or a material unresolved-part question.
5. Feed §9.26 stock results into the same unit repair. A dry/unresolved part must not force regeneration of other sheets.

### 2.3 Replace prose-derived named-part accountability with typed ownership

1. Propagate exact architecture part decisions through `CircuitRequirement.exact_part`, recipe selections, or another existing typed identity field—not by mining assumptions/topology prose.
2. A core-default selection must deterministically become either the shipped BOM identity or a typed substitution entry.
3. `check_spec_named_mpn_substitutions` remains strict for user-named and typed architecture-selected parts, but must not invent requirements from incidental prose tokens.
4. Preserve the substitution ledger: never auto-accept a materially different part.

### Files

- `kicraft/server/stage_work_units.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/server/stage_bom_tools.py`
- `kicraft/server/stage_prompts.py`
- `kicraft/design/cli_app.py`
- `kicraft/design/models.py` only if an existing typed field cannot carry the decision
- `kicraft/design/synthesis/validation.py`
- `kicraft/parts_library/` registries/resolvers
- `.agents/skills/kicraft/stages/architecture.md`
- `.agents/skills/kicraft/stages/bom.md`
- `tests/test_stage_work_units.py`
- `tests/test_kicraft_stage_cli.py`
- existing resolver/parts-library tests

### Acceptance

- Every frozen bad symbol, footprint, and pin/pad pair is rejected within its owning unit before aggregate merge.
- A repaired identity reuses accepted sibling units and performs no provider calls for those siblings.
- `N/A` cannot reach §9.26 as an MPN.
- User-named parts still require the exact part or an explicit substitution.
- Two fresh cohort runs pass: `speaker-crossover,rs485-terminal,buck-3a,dual-rail-supply,relay-quad,proto-shield,gpio-expander,servo-driver-16,stepper-a4988,audio-jack-buffer,round-led-ring,star-ornament,lora-node,hex-env-sensor`.

## Phase 3 — make deterministic recipe/lowerer ownership authoritative

### Change

1. Fix work-unit planning so recipe/lowerer-owned requirements and protected identities do not receive a competing model unit.
2. When a model emits an exact duplicate of a deterministic recipe group on the same sheet, prune the duplicate before unit validation and retain the recipe group. Only exact semantic identity may be pruned; a conflicting value/package remains a rejection.
3. When a unit emits only protected sibling identities it does not own, remove those groups and decide from the plan whether the unit is already satisfied:
   - if recipe/lowerer output owns its requirement, accept deterministic output with no model group;
   - otherwise reject `missing-requirement-implementation` with the exact owned requirement.
4. Prevent `recipe-duplicate` and `model_authored_protected_identity` from entering repeated repair loops with instructions that cannot change ownership. Ownership conflicts are fixed deterministically or terminate after one attributable rejection.
5. Derive BOM and wiring from one typed lowerer artifact. Do not independently ask the model to reproduce its pins.
6. Recalculate provider-call budget from planned **model-owned** units, excluding deterministic units. Keep the absolute project budget unchanged.

### Files

- `kicraft/server/stage_work_units.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/design/lowering.py`
- deterministic recipe registry modules
- `tests/test_stage_work_units.py`
- `tests/test_stage_driver_retry.py`
- lowerer/recipe tests adjacent to their implementation

### Acceptance

- The frozen `encoder-oled-panel` recipe duplicate is resolved without a provider repair call.
- The frozen `esp32-dual-motor` protected-identity collision preserves recipe-owned ESP32 support and retains the unit's genuinely owned motor/power groups.
- `rp2040-min`, `nrf52-beacon`, `esp32-s3-sensor`, and `daq-8ch` stay within the existing per-project spend budget because deterministic and accepted units are not redrafted.
- Two fresh cohort runs pass: `rp2040-min,esp32-s3-sensor,nrf52-beacon,encoder-oled-panel,esp32-dual-motor,daq-8ch`.

## Phase 4 — enforce the BOM/wiring stage boundary

The baseline's `usb-a-power-splitter`, `led-cc-driver`, and `snowman-ornament` failures show a deterministic ownership bug: because lowerer/recipe connections are present in the BOM, the current commit path runs complete-design §9.11/§9.14 coverage while model-authored parts are still intentionally unwired.

### Change

1. At BOM commit, validate deterministic recipe/lowerer connections only against the pins and inter-sheet endpoints those artifacts own.
2. Do not run full-design net coverage, connectivity, ERC, or inter-sheet endpoint completeness until the wiring stage has merged all model and deterministic pin assignments.
3. Keep all full-design checks unchanged at wiring commit. This is a stage-boundary correction, not a weaker gate.
4. Store connection provenance (`lowerer`, `recipe`, or wiring work-unit ID) through merge so partial checks can use explicit ownership rather than infer it from non-empty `connections`.
5. Add a mixed candidate fixture: one fully wired lowerer-owned group plus one unwired model-owned IC/support group. BOM must commit; wiring must fail until the second group is completely assigned.

### Files

- `kicraft/design/cli_app.py`
- `kicraft/design/models.py` if connection provenance is not already representable
- `kicraft/server/stage_contracts.py`
- `kicraft/server/stage_work_units.py`
- `kicraft/design/synthesis/validation.py`
- `tests/test_kicraft_stage_cli.py`
- `tests/test_stage_work_units.py`
- `tests/test_architecture_intersheet.py`

### Acceptance

- Mixed deterministic/model BOM candidates no longer fail full-design §9.11/§9.14 at BOM commit.
- The identical candidate without completed wiring still fails those gates at wiring commit.
- Two fresh cohort runs pass: `usb-a-power-splitter,led-cc-driver,snowman-ornament`.

## Phase 5 — make wiring units mechanically complete and repairable

### Change

1. Assert when planning units that every BOM pin is owned by exactly one source: locked recipe/lowerer assignment or one wiring work unit. Fail locally before any provider call on overlaps or gaps.
2. Partition model-owned wiring by the smallest independent sheet/subgraph that can be validated with complete context. Do not combine unrelated connectors and analog networks merely to reduce call count.
3. Include only the unit's refs, exact symbol pin inventory, locked/excluded pins, sheet-local nets, and relevant inter-sheet endpoints in its prompt. Exclude sibling refs that the unit cannot emit.
4. Normalize exact duplicate assignments only when ref, pin, net/no-connect state are identical. Conflicting duplicate assignments remain hard failures.
5. On missing/unexpected/unknown pins, return the complete expected pin set and the rejected assignment set, then request one full unit replacement. Do not ask for a patch.
6. Route aggregate wiring rejections back through `ref_to_unit` / connection provenance. For `highside-switch-10a`, the `GATE_CTRL` dangling-net offender must redrive only the unit that owns `R1` and its counterpart endpoint.
7. If a singleton signal has no unambiguous endpoint in committed architecture/BOM, do not guess. Return an attributable architecture/BOM reconciliation requirement rather than consuming repeated wiring attempts.
8. Expand deterministic lowerers only for circuit families with a typed, fully testable pin/net transformation. Do not encode the corpus brief as a special case.

### Files

- `kicraft/server/stage_work_units.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/design/lowering.py`
- `.agents/skills/kicraft/stages/wiring.md`
- `tests/test_stage_work_units.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_architecture_intersheet.py`

### Acceptance

- Planning proves exactly-once ownership for every BOM pin before provider dispatch.
- The frozen RC and thermocouple candidates receive complete, non-overlapping unit pin inventories and converge without duplicate/unexpected refs.
- Aggregate `GATE_CTRL` repair redrives only its owner and preserves accepted sibling units.
- Two fresh cohort runs pass: `rc-lowpass-bnc,thermocouple-amp,highside-switch-10a`.

## Phase 6 — staged campaign acceptance and production gate

### Cohort ladder

Run each cohort with a new output directory, real production provider/profile, no judge, and no resume:

```bash
.venv/bin/python -m kicraft.eval.self_eval \
  --out logs/self_eval/<new-dir> \
  --design-only --no-judge --lean-events --parallel 3 \
  --only <comma-separated-cohort>
```

A cohort is green only if every selected run has `design_committed is true` and exactly five successful stage statuses. A process exit code of zero is not sufficient; read the completed `summary.json`.

Run cohorts in this order:

1. serialization: six functional-spec/architecture failures;
2. part identity/accountability: fourteen runs;
3. recipe/budget protocol: six runs;
4. BOM/wiring boundary: three runs;
5. wiring ownership: three runs;
6. regression sentinels: `r2r-dac,fpc-breakout`.

Require two consecutive fresh passes for each changed cohort before the full campaign. A failed second pass means the fix is stochastic and is not accepted.

### Full acceptance

Run the complete corpus with no `--only`, `--limit`, or `--resume`:

```bash
.venv/bin/python -m kicraft.eval.self_eval \
  --out logs/self_eval/design34_acceptance_<UTC> \
  --design-only --no-judge --lean-events --parallel 3
```

Then run the canonical production gate, which performs another fresh all-34 campaign before restart:

```bash
./deploy/deploy-production.sh
```

### Required result

For both complete campaigns:

- `n == 34`;
- `design_committed == 34`;
- every run has `design_committed is true`;
- every run records successful `intent`, `functional_spec`, `architecture`, `bom`, and `wiring` statuses;
- no cache/reuse from an earlier campaign directory;
- no brief is omitted or replaced;
- no stage failure is masked by process exit code, summary wording, or build status;
- total spend remains within configured production ceilings.

After the production gate passes, verify the existing operational contract: web HTTP 200 and the build-worker log ends in `[build-worker] ready`.

## Implementation sequencing and commit boundaries

Keep changes reviewable and bisectable:

1. `Record attributable stage/work-unit failures` — Phase 0 only.
2. `Converge structured stage serialization` — Phase 1 plus frozen fixtures.
3. `Validate BOM identities inside work units` — Phase 2.1/2.2.
4. `Propagate typed selected-part accountability` — Phase 2.3.
5. `Make deterministic recipe ownership authoritative` — Phase 3.
6. `Separate BOM partial wiring from wiring completeness` — Phase 4.
7. `Repair wiring by exact owning unit` — Phase 5.
8. `Prove fresh 34-of-34 stage completion` — campaign artifacts/report only; do not commit generated run directories.

Run focused tests after each commit; run the relevant real-provider cohort only after its focused deterministic checks pass. Do not run the full 34 after every small edit.

## Handoff checklist

- [ ] Start from commit `8c4d058` or later on `simplify/bom-wiring-pipeline`.
- [ ] Treat the fresh baseline directory as immutable evidence.
- [ ] Complete Phase 0 classification before changing retry/bound behavior.
- [ ] Keep gates strict and fixes generic by schema, part family, ownership, or circuit lowerer.
- [ ] Preserve accepted work units across every localized repair.
- [ ] Record every cohort output directory and exact commit SHA.
- [ ] Do not claim completion from unit tests, cached runs, subset canaries, or exit code alone.
- [ ] Completion is two fresh complete reports with 34/34 five-stage commits, the second obtained through the canonical deployment gate.
