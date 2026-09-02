# LLM stage reliability — implementation plan

**Status:** proposed for implementation
**Date:** 2026-09-02
**Primary evidence:** `/home/kicraft/.kicraft/self_eval/interactive-rp2040-20260901`
**Latest batch:** `/home/kicraft/.kicraft/self_eval/20260901T131941Z`
**Pipeline:** `intent -> functional_spec -> architecture -> bom -> wiring`

## Decision

Make stage availability and stage correctness separate, durable outcomes. Advisory semantic defects must never stop a user's run. High-confidence repair findings get one bounded best-effort correction; if that call fails, commit the best schema-valid candidate and preserve the finding. Deterministic electrical or fabrication invariants remain strict, but they block fabrication promotion rather than erasing the inspectable design.

Do not try to reach >99% by increasing token caps or generic retries. Reduce the model's responsibility: it chooses intent, topology, and project-specific mappings; deterministic code expands repetitive structures, instantiates verified reference circuits, wires recipe-owned pins, and proves graph invariants.

## Evidence and problem statement

The live `rp2040-min` run reported all five design stages successful:

| Stage | Runtime status | Attempts | Cost | Wall time | Artifact assessment |
|---|---:|---:|---:|---:|---|
| intent | OK | 1 | $0.000198 | 2.730 s | explicit parts and constraints misclassified |
| functional_spec | OK | 1 | $0.000330 | 13.801 s | premature LDO/ESD/component decisions |
| architecture | OK | 2 | $0.001764 | 109.310 s | invalid sheet decomposition and incomplete VBUS graph |
| bom | OK | 2 | $0.040388 | 430.660 s | fake castellations, weak SWD, contradictory support plan |
| wiring | OK | 1 | $0.002050 | 53.850 s | ineffective BOOTSEL and incorrect special-pin handling |
| **total** | **5/5 OK** | **7** | **$0.044730** | **609.494 s** | **not fabrication-safe** |

Synthesis reported ERC clean. The real build accepted 7/7 leaves but produced no routed parent. The LLM defects precede and are distinct from that router result.

Specific witnesses this plan must close:

1. `intent.constraints=[]` and `intent.named_parts=[]` despite RP2040, QFN-56, QSPI flash, USB-C, 12 MHz crystal, and castellated GPIO in the brief.
2. Functional spec committed to an LDO and ESD protection despite its abstract-stage contract.
3. Architecture emitted a `POWER 5V` sheet although power blocks are nets, separated crystal/castellated support into independent physical domains, omitted `LDO 3V3` from the VBUS endpoints, and declared USB/GPIO directions as one-way.
4. Architecture's successful second attempt persisted only `attempts=2`; the first rejection is not durable without an attached event sink.
5. BOM represented 30 castellated pads as purchasable vertical `PinHeader_1x01_P2.54mm` components and chose a 1x03 SWD header although the stage contract names 2x5 or 1x4 access.
6. BOM claimed four QSPI pull-ups; committed wiring used three QSPI pulls plus pulls on RUN, SWDIO, and SWCLK.
7. Wiring created a `BOOTSEL` net containing only the switch contacts; it never reached QSPI CS. The existing RP2040 checker accepts any switch or any wired SWD pin.
8. `U3.19`, confirmed by the generated symbol as RP2040 `TESTEN`, was marked no-connect.
9. The latest 34-brief batch completed only 5 designs; 23 failures collapsed to the non-actionable terminal label `provider error`.

## Target outcome model

### Stage status

Extend `kicraft.design.models.StageStatus` without changing the meaning of existing states during migration:

```python
class StageDiagnostic(BaseModel):
    code: str
    severity: Literal["advisory", "repair_required", "fab_gate"]
    message: str
    evidence: list[str] = []
    detector_version: int
    attempt: int | None = None

class StageStatus(BaseModel):
    ok: bool                         # compatibility: candidate committed
    provider_ok: bool | None = None
    schema_ok: bool | None = None
    semantic_clean: bool | None = None
    repair_required: bool = False
    fab_safe: bool | None = None
    repair_attempted: bool = False
    repair_adopted: bool = False
    diagnostics: list[StageDiagnostic] = []
    # existing cost/timing/failure fields remain
```

Definitions:

- `provider_ok`: at least one provider response completed without a terminal provider/transport error.
- `schema_ok`: at least one response satisfied the stage response schema.
- `ok`: a candidate committed. Keep this field for old project-state readers.
- `semantic_clean`: no deterministic semantic diagnostics remain on the committed candidate.
- `repair_required`: committed candidate remains usable but contains a high-confidence defect downstream should see.
- `fab_safe`: `None` before relevant electrical/mechanical checks, then `True`/`False` from deterministic gates.

Do not overload `failure_kind` with semantic findings. `failure_kind` remains the terminal operational/schema/commit classification; semantic diagnostics have stable codes.

### User-visible status

Extend `derive_stage_statuses` and `StageTabs` to support:

- `done`: committed and semantically clean;
- `warning`: committed with advisory or unresolved repair diagnostics;
- `failed`: no candidate committed;
- existing `pending`, `parked`, and live `active` remain.

A warning must not prevent `remaining_stages` from advancing. A `fab_gate` finding keeps the design inspectable but makes the fabrication phase failed/invalid until a re-drive clears it.

### Severity policy

| Severity | Example | User run | Repair | Fabrication |
|---|---|---|---|---|
| advisory | intent classification, copied goal | continue | none or optional | unaffected |
| repair_required | missing architecture endpoint, premature topology, contradictory BOM plan | continue | one bounded focused correction | unresolved result is visibly warned |
| fab_gate | ineffective programming path, wrong mechanical implementation, required special pin incorrect | design remains inspectable | deterministic repair/re-drive | no fab-ready promotion |

## Invariants and non-goals

1. Preserve the original brief as canonical input to every stage.
2. Never mutate a candidate with uncertain regex guesses. Deterministic normalization is permitted only when the transformation is lossless and contract-owned.
3. Never add an LLM judge to stage execution.
4. Never weaken existing pin, net, stock, ERC, DRC, routing, or fabrication gates.
5. Never treat a rubric grade as the LLM-stage reliability signal.
6. Do not special-case benchmark slugs. RP2040 behavior is selected by part family/recipe identity.
7. Do not persist raw production briefs, model reasoning, or full responses in aggregate telemetry.
8. Keep every provider call bounded. Semantic repair gets at most one additional call.
9. A repair failure must not discard the first schema-valid candidate.
10. Build/place/route reliability is reported separately from LLM-stage reliability.

---

## Phase 0 — freeze evidence and contracts

### Change

1. Add minimized fixtures derived from the live workspace, one at each stage boundary. Keep only brief, committed upstream state, model candidate, and expected diagnostics; do not commit generated boards or provider reasoning.
2. Add the exact malformed live candidates as regression fixtures:
   - empty intent classification;
   - functional spec with premature `LDO_3V3`/ESD;
   - seven-sheet architecture with missing LDO VBUS endpoint;
   - BOM with 30 vertical-header castellation placeholders;
   - wiring with isolated BOOTSEL and `TESTEN` no-connect.
3. Record expected facts rather than exact prose.
4. Freeze response-policy versions in campaign manifests. Any schema change below increments the affected policy version.

### Files

- `tests/fixtures/stage_reliability/rp2040_brief.json` (new)
- `tests/fixtures/stage_reliability/rp2040_intent_candidate.json` (new)
- `tests/fixtures/stage_reliability/rp2040_functional_spec_candidate.json` (new)
- `tests/fixtures/stage_reliability/rp2040_architecture_candidate.json` (new)
- `tests/fixtures/stage_reliability/rp2040_bom_candidate.json` (new)
- `tests/fixtures/stage_reliability/rp2040_wiring_candidate.json` (new)
- `tests/test_stage_semantics.py` (new)

### Acceptance

- Every observed failure above is represented by a deterministic fixture and stable expected diagnostic code.
- Fixtures contain no provider reasoning, credentials, production-user text, or generated binary artifacts.
- Current behavior is captured before implementation so each later phase proves a delta.

---

## Phase 1 — durable outcomes, attempts, and provider failures

### 1.1 Extend stage status and event contracts

#### Change

1. Add `StageDiagnostic` and the new status dimensions to `StageStatus`.
2. Extend `stamp_stage_status` and `finalize_stage` to write them atomically.
3. Add a `stage_diagnostic` event containing only stage, code, severity, detector version, normalized evidence, and attempt.
4. Include `stage_diagnostic` in self-eval's structural event allowlist.
5. Add the same fields to caller-visible stage results.
6. Keep old state files readable: absent fields mean unknown, not clean.

#### Files

- `kicraft/design/models.py`
- `kicraft/server/stage_state_io.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/eval/self_eval.py`
- `kicraft/server/session.py`
- `kicraft/server/stagetabs.py`
- `tests/test_stage_status.py`
- `tests/test_stage_resource_telemetry.py`
- `tests/test_self_eval.py`
- `tests/test_stagetabs_helpers.py`

#### Acceptance

- A committed stage with diagnostics restores as `warning`, advances to the next stage, and keeps `ok=true`.
- A legacy committed stage with no new fields restores as `done`, not warning.
- A fab-gate diagnostic survives state load/dump and makes only the fabrication phase invalid.
- Self-eval full and lean event modes retain structural diagnostic events.

### 1.2 Persist every attempt, including recovered failures

#### Change

1. Add a `stage_attempts` table to the spend ledger rather than packing attempt history into `stage_runs`.
2. Store one row per provider call:
   - timestamp, run ID, stage, attempt number, call mode (`normal`, `serialization`, `semantic_repair`, `clean_slate`);
   - model, selected provider when observable, finish reason;
   - outcome (`candidate`, `question`, `invalid_schema`, `commit_rejected`, provider failure family, etc.);
   - HTTP status/error code/request ID when safely observable;
   - wall time, input/output token counts, cost;
   - normalized diagnostic/gate codes.
3. Do not store messages, raw response text, reasoning, tool output, or brief.
4. Record commit-rejection signatures even when a later attempt succeeds.
5. Extend `web_cost_report.load_stage_runs` or add `load_stage_attempts`; do not infer recovered failures from final `stage_runs.failure_kind`.

#### Files

- `kicraft/server/spend_guard.py`
- `kicraft/server/client.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/cli/web_cost_report.py`
- `tests/test_stage_resource_telemetry.py`
- `tests/test_client_provider.py`
- `tests/test_web_cost_report.py`

#### Acceptance

- The live architecture shape—attempt 1 rejected, attempt 2 committed—produces two attempt rows and one successful stage row.
- Ledger failure cannot fail a user design.
- Attempt rows contain none of `_FORBIDDEN_EVENT_FIELDS` used by `llm_analysis`.
- Concurrent runs are attributable by exact run ID and stage.

### 1.3 Replace generic provider errors

#### Change

1. Classify provider exceptions before `stage_runtime` catches them. At minimum distinguish:
   - `provider_rate_limited` (429);
   - `provider_upstream_5xx`;
   - `provider_auth` (401/403);
   - `provider_request_rejected` (other 4xx);
   - `provider_response_format_rejected` when the response-format capability is named;
   - `provider_capability_rejected` for tool/reasoning/schema capability failures;
   - `transport_timeout`, `transport_connection`, and `transport_stream_interrupted`;
   - `provider_unknown` only when no safer classification exists.
2. Preserve HTTP status, OpenRouter/provider error code, selected upstream provider, and request ID when supplied. Redact response body to an allowlisted error-code/message fragment; never store echoed prompt content.
3. Map the detailed kind to a concise user-safe error message. Do not expose internal routing or credentials in the UI.
4. Update terminal classification sets and reports; remove the bare `provider_error` bucket for new rows.

#### Files

- `kicraft/server/client.py`
- `kicraft/server/stage_runtime.py`
- `kicraft/server/spend_guard.py`
- `kicraft/eval/llm_analysis.py`
- `tests/test_client_provider.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_stage_resource_telemetry.py`

#### Acceptance

- Fixtures for 429, 500, 401, unsupported response format, timeout, and interrupted stream produce distinct stable failure kinds.
- The latest-batch analysis can no longer collapse a new campaign into an unexplained `provider error` count.
- No exception detail contains request payloads or secrets.

---

## Phase 2 — deterministic semantic diagnostics and bounded repair

### 2.1 Add a pure semantic-diagnostics engine

#### Change

Create a stage-independent interface:

```python
def diagnose_stage(
    stage: str,
    *,
    brief: str,
    upstream_state: dict,
    candidate: dict,
) -> list[StageDiagnostic]: ...
```

Rules are pure, deterministic, versioned, and side-effect free. Put shared token normalization in one place. Reuse/generalize `_spec_named_tokens` rather than creating a second MPN recognizer.

Do not put semantic diagnostics in Pydantic validators: legitimate empty fields and multiple valid designs make them unsuitable schema errors.

#### Files

- `kicraft/design/stage_semantics.py` (new)
- `kicraft/design/synthesis/validation.py` (export/generalize named-token helper only)
- `tests/test_stage_semantics.py`

### 2.2 Implement intent diagnostics

#### Initial codes

- `intent_named_part_omitted`: a conservative MPN/family token found in the brief is absent from `named_parts`.
- `intent_constraints_empty`: constraints are empty despite explicit package, quantity, voltage/frequency/unit, inclusion/exclusion, interface, or mechanical wording.
- `intent_unclassified_copy`: goal substantially copies the brief while both classification fields remain empty.

#### Rules

1. Compare normalized tokens; accept family-preserving forms such as punctuation/case variations.
2. Treat legitimate vague briefs as clean.
3. Evidence contains only normalized matched tokens, not the full brief.
4. These findings are advisory by default. `intent_constraints_empty` may request repair when two or more high-confidence explicit facts are present.

#### Acceptance

- The RP2040 live candidate reports omitted `RP2040` and explicit-constraint findings.
- “A small sensor board” may retain empty lists without warning.
- No detector invents a named part or modifies the slot.

### 2.3 Implement functional-spec diagnostics

#### Initial codes

- `functional_spec_premature_topology`: unrequested topology/part choice such as LDO, buck, boost, flyback, or a specific protection IC appears before architecture.
- `functional_spec_unrequested_feature`: a feature absent from brief/intent is introduced without a corresponding `(defaulted)` assumption.
- `functional_spec_nonfunctional_block`: a pure rail, ground, mechanical hole, or component-level support item is emitted as an independent functional block.
- `functional_spec_unrecorded_assumption`: output relies on a default not represented in assumptions.

Use a deliberately narrow topology vocabulary and existing electrical categories. False positives are more damaging than missed advisory findings.

#### Acceptance

- The live LDO/ESD candidate produces stable findings.
- A brief that explicitly asks for an LDO or ESD protection does not trigger those findings.
- Explicit user-named RP2040 remains permitted in a functional block.

### 2.4 Integrate one best-effort semantic repair

#### Change

1. Diagnose each schema-valid candidate before commit.
2. Retain the first schema-valid candidate in memory.
3. For advisory-only results, commit immediately and persist diagnostics.
4. For `repair_required`, make at most one additional, tool-free focused repair call using:
   - the same response schema;
   - normalized diagnostic codes/evidence;
   - an instruction to preserve valid content and avoid new assumptions.
5. Diagnose the repaired candidate. Adopt it only when:
   - it is schema-valid;
   - it has fewer `repair_required`/`fab_gate` findings;
   - it passes ordinary commit gates.
6. If the repair call/provider/schema/commit fails, commit the retained original candidate if it passes ordinary commit gates. Preserve `repair_attempted=true`, `repair_adopted=false`, and both attempt outcomes.
7. Semantic repair has a separate budget of one call and never recursively repairs itself.
8. Do not use BOM tools during semantic repair. BOM tool-dependent hard corrections stay in the existing normal correction/reconcile flow.

#### Files

- `kicraft/server/stage_runtime.py`
- `kicraft/server/stage_prompts.py`
- `kicraft/server/config.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_stage_semantic_repair.py` (new)

#### Acceptance

- Broken -> clean repair commits clean candidate.
- Broken -> still broken repair commits the lower-severity candidate.
- Broken -> provider failure commits the first valid candidate and advances.
- Semantic repair never exceeds one extra call.
- Existing serialization, clean-slate, commit-correction, and BOM reconcile budgets remain independently bounded.

### 2.5 Tighten prompts and examples

#### Change

1. Intent: require explicit classification of named parts and hard requirements; include bad/valid contrasting examples.
2. Functional spec: prohibit premature topology and require every introduced default in assumptions.
3. Architecture: emphasize physical-domain sheets, power nets as nets, bidirectional USB/GPIO, and programming decisions.
4. BOM: prohibit placeholder footprints and BOM entries for board-fabricated features.
5. Wiring: state that programming-path labels do not prove reachability.
6. Keep examples validated against Pydantic models.

#### Files

- `.claude/skills/kicraft/stages/intent.md`
- `.claude/skills/kicraft/stages/functional_spec.md`
- `.claude/skills/kicraft/stages/architecture.md`
- `.claude/skills/kicraft/stages/bom.md`
- `.claude/skills/kicraft/stages/wiring.md`
- `kicraft/server/stage_prompts.py`
- `tests/test_stage_driver_prompt_examples.py`

#### Acceptance

- All worked examples validate against the current response schemas.
- Prompt-only A/B results are reported separately from deterministic-linter results.

---

## Phase 3 — architecture compactness, physical domains, and graph checks

### 3.1 Add compact inter-sheet net ranges

#### Change

1. Add a model-facing compact range form for repetitive nets; keep canonical `Architecture.inter_sheet_nets` fully expanded after response normalization.
2. Support only unambiguous numeric ranges initially:

```json
{
  "name_pattern": "GPIO{n}",
  "start": 0,
  "end": 29,
  "endpoints": [
    {"sheet": "MCU RP2040", "direction": "bidirectional"},
    {"sheet": "CASTELLATED IO", "direction": "bidirectional"}
  ]
}
```

3. Reject overlapping generated names, invalid ranges, and expansion beyond existing collection bounds.
4. Expand ranges in `stage_contracts` before canonical model validation. Downstream code continues to consume ordinary `InterSheetNet` objects.
5. Increment the architecture response-policy version.

#### Files

- `kicraft/server/stage_contracts.py`
- `kicraft/design/models.py` only if canonical provenance is retained
- `kicraft/eval/self_eval.py` response-policy manifest metadata
- `tests/test_stage_contracts.py` or new `tests/test_architecture_contract.py`
- `tests/test_stage_driver_prompt_examples.py`

#### Acceptance

- GPIO0..GPIO29 expands deterministically to 30 canonical nets.
- Duplicate/overlapping ranges fail schema normalization before commit.
- Wiring and synthesis require no range-aware branches.

### 3.2 Add architecture semantic checks

#### Codes/checks

- `architecture_power_block_as_sheet`: a power-only FS block was emitted as a sheet.
- `architecture_missing_power_endpoint`: a power consumer/producer topology is not represented on the corresponding rail net.
- `architecture_wrong_signal_direction`: known bidirectional protocols or GPIO are declared one-way.
- `architecture_fragmented_physical_domain`: trivial connector, crystal, passive support, or board feature is split from the IC it directly supports.
- `architecture_distinct_ics_merged`: multiple non-trivial IC domains are collapsed onto one from-scratch sheet.
- `architecture_programming_decision_incomplete`: MCU present without an explicit recipe/access choice.

#### Implementation

1. Add pure checks beside existing `check_every_block_has_sheet` and `check_fs_connections_mapped`.
2. Use functional block category, topology text, protocol, and known recipe metadata. Fail open when the physical domain is ambiguous.
3. Aggregate all architecture diagnostics in one repair message.
4. Keep schema-valid original candidate available if repair fails.
5. Do not silently delete sheets or guess missing electrical endpoints. Compact-range expansion is deterministic; physical redesign is not.

#### Files

- `kicraft/design/stage_semantics.py`
- `kicraft/design/synthesis/validation.py`
- `kicraft/design/cli_app.py`
- `tests/test_stage_semantics.py`
- `tests/test_cross_stage_validation.py`

#### Acceptance

- The live seven-sheet architecture identifies the power sheet, fragmented clock/castellation domains, missing LDO VBUS endpoint, and USB/GPIO direction issues in one pass.
- Valid multi-IC designs retain separate sheets.
- Library-leaf boundaries are never rewritten by these diagnostics.

---

## Phase 4 — verified design recipes and deterministic wiring

### 4.1 Add a recipe registry

#### Design

Introduce versioned, vendored circuit recipes. A recipe is not a parts bundle: a parts bundle describes one physical component; a circuit recipe describes a verified set of component roles, parameters, nets, pin assignments, and invariants.

Proposed canonical models:

```python
class RecipeSelection(BaseModel):
    recipe: str                  # e.g. "rp2040-minimal@1"
    instance: str                # stable within architecture
    sheets: dict[str, str]       # recipe domain -> Architecture Sheet.name
    parameters: dict[str, JsonScalar]

class RecipePartSpec(BaseModel):
    role: str
    group: RecipeComponentGroup  # design-layer model; never import server stage contracts

class RecipePinSpec(BaseModel):
    role: str
    pin: str
    net: str
```

`RecipeComponentGroup` is the recipe-owned analogue of the model-facing
`BomComponentGroup`. Keep it in `kicraft.design.recipes.models` (or move the
shared group fields into `kicraft.design.models`) so the design layer never
imports `kicraft.server.stage_contracts`.

Add `recipe_selections` to the architecture response/canonical model. Keep selection explicit and inspectable; do not infer a recipe from an arbitrary BOM after the fact.

#### Registry behavior

1. Resolve exact recipe name/version; no fuzzy aliasing.
2. Validate parameters and required sheet-role mapping.
3. Expand recipe parts before model-emitted BOM groups so reference allocation is deterministic.
4. Tag canonical parts with recipe ID, instance, and role.
5. Expose recipe summaries in architecture/BOM stage extras, bounded like core defaults.
6. A recipe selection takes precedence over model-invented support parts for its owned roles. Duplicate role/function parts fail with a precise diagnostic.
7. Recipe definitions are source-controlled Python/data, reviewed like validation contracts.

#### Files

- `kicraft/design/recipes/__init__.py` (new)
- `kicraft/design/recipes/models.py` (new)
- `kicraft/design/recipes/registry.py` (new)
- `kicraft/design/models.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/server/stage_prompts.py`
- `kicraft/server/stage_runtime.py`
- `tests/test_design_recipes.py` (new)

#### Acceptance

- Unknown recipe/version/parameter/sheet role fails before provider-driven BOM correction.
- Recipe expansion is deterministic and idempotent.
- Model BOM groups cannot replace or duplicate locked recipe roles.
- Canonical parts retain enough recipe provenance for wiring and diagnostics.

### 4.2 Move recipe-owned wiring out of the LLM

#### Change

1. At wiring prep, derive all recipe-owned pin assignments from recipe role mappings and canonical recipe-tagged refs.
2. Remove locked pins from `extras.symbol_pinouts`; tell the model exactly which refs/pins remain project-owned.
3. Normalize the model's remaining assignments, then merge locked recipe assignments.
4. Reject any model attempt to overwrite a recipe-owned pin.
5. Run ordinary full-board pin coverage and electrical checks over the merged canonical BOM.
6. Persist recipe wiring provenance for diagnostics, not as a second netlist representation.

#### Files

- `kicraft/design/recipes/registry.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/design/cli_app.py` stage-prep wiring path
- `kicraft/server/stage_prompts.py`
- `tests/test_recipe_wiring.py` (new)
- `tests/test_stage_driver_retry.py`

#### Acceptance

- A recipe-owned RP2040 support network requires zero LLM pin assignments.
- The model only maps project GPIO/external signals.
- Full `check_pin_existence`, `check_net_coverage`, family contracts, and ERC still run after merging.
- Re-driving wiring replaces only model-owned assignments while deterministically recreating recipe assignments.

### 4.3 Implement `rp2040-minimal@1`

#### Recipe scope

The recipe must be based on the RP2040 datasheet/reference design and checked into source with citations in code comments. It owns:

- RP2040 QFN-56 symbol/footprint;
- required 3V3/1V1/ADC/USB decoupling and regulator-support connections;
- `TESTEN` handling;
- 12 MHz crystal/load network;
- QSPI flash and required pulls/series elements;
- USB D+/D- termination and CC device-role resistors when USB-C is selected;
- RUN handling;
- one explicit programming/recovery option:
  - functional BOOTSEL path to QSPI CS and GND, or
  - complete SWD access with SWDIO, SWCLK, GND, and VTref;
- recipe-owned local nets and inter-sheet QSPI/USB/power endpoints.

Parameters should be bounded and explicit, for example `usb=true`, `programming="bootsel+swd"`, `flash_mbit=16`, `gpio_exposed=30`. Do not allow arbitrary part IDs through parameters; part substitution uses versioned recipe updates.

#### Files

- `kicraft/design/recipes/rp2040_minimal.py` (new)
- appropriate reviewed component bundles under `kicraft/parts_library/` if any required part is missing
- `tests/test_rp2040_recipe.py` (new)
- `.claude/skills/kicraft/stages/architecture.md`
- `.claude/skills/kicraft/stages/bom.md`

#### Acceptance

- Recipe-expanded state passes all commit checks and `kicad-cli` ERC without LLM-authored support wiring.
- `TESTEN` has the documented required connection.
- BOOTSEL graph activation reaches QSPI CS and GND; an isolated switch fails.
- SWD reaches an access connector with VTref and GND when selected.
- USB series/CC network matches the cited reference design.
- Frozen wiring replay cannot recreate the live isolated-BOOTSEL or `TESTEN`-NC defects.

### 4.4 Replace label heuristics with programming graph reachability

#### Change

1. Fix `_rp2040_boot_problem`: `any switch` is not evidence. Prove either:
   - both SWDIO and SWCLK reach the same external programming connector/test interface, with GND and VTref present as required by the selected access recipe; or
   - a BOOTSEL switching path connects the required QSPI-CS node to GND.
2. Extend `check_mcu_programming_access` reachability beyond UPDI to RP2040, STM32/SWD, and ESP32 strap/reset paths where pin roles are known.
3. Share graph utilities between advisory §9.21, hard §9.29, and `mcu_programming_facts` so the eval verdict cannot disagree with commit validation.
4. Fail open only when the part family/pinout is unknown; recipe-selected families are always knowable and therefore strict.

#### Files

- `kicraft/design/synthesis/validation.py`
- `tests/test_programming_access.py`
- `tests/test_substitution_ledger.py` if `mcu_programming_facts` output changes
- `kicraft/eval/llm_analysis.py`

#### Acceptance

- Isolated BOOTSEL switch fails.
- One merely wired SWD pin fails.
- SWD pins connected only to pull resistors fail.
- Valid BOOTSEL and valid SWD paths pass.
- BOM part presence alone never proves wiring reachability.

---

## Phase 5 — board-fabricated features and real castellations

### 5.1 Model non-assembly board features explicitly

#### Change

1. Add `assembly: bool = True` to canonical `BomPart`. Keep `assembly=false` internal to deterministic recipes; do not initially expose it in the model-facing BOM schema.
2. Mark fabrication-only footprints with KiCad's exclude-from-BOM and exclude-from-position-file attributes when they are stamped. `_write_bom_csv` must also filter `assembly=false` as defense in depth; verify the `kicad-cli pcb export pos` CPL omits them.
3. Retain fabrication-only refs in schematic, PCB, routing, ERC, DRC, and net coverage.
4. Add a canonical edge-interface record to BOM/placement metadata containing refs, side assignment, pitch, and edge behavior.
5. Reject model-authored sourcing notes/MPNs on fabrication-only refs.

#### Files

- `kicraft/design/models.py`
- `kicraft/design/synthesis/fab_export.py`
- `kicraft/design/synthesis/autoplacer.py`
- `kicraft/design/synthesis/kicad_pcb_stub.py`
- `tests/test_fab_export.py`
- `tests/test_autoplacer.py`
- `tests/test_kicad_pcb_stub.py`

#### Acceptance

- Fabrication-only pads are routed and validated but absent from assembly BOM/CPL.
- Existing ordinary parts export unchanged.
- No placeholder part can claim an LCSC code for a board-fabricated feature.

### 5.2 Implement castellated edge interfaces

#### Change

1. Add a reviewed custom castellation footprint with plated half-hole/pad geometry and a generic one-pin schematic symbol association.
2. The RP2040 recipe emits two deterministic banks for 30 GPIOs unless architecture explicitly chooses another valid distribution. Do not emit one unconstrained 1x30 vertical-header row.
3. Assign banks to opposite board edges, preserve logical GPIO ordering, and expose power/GND pads when the selected module interface requires them.
4. Emit castellation bank members into `autoplacer.json` as an ordered edge array with an exact edge-datum constraint. Extend the existing array/edge placement contracts; do not patch the pinned external router.
5. Teach single-board and parent composition that declared castellation copper must intersect its assigned outline while its footprint body remains constrained. This is a separate constraint from connector mouth overhang; do not reuse the generic connector exception.
6. Add a post-placement geometric gate proving:
   - each declared castellation intersects exactly one board edge;
   - drill/pad geometry is legal;
   - pitch/order match the declared interface;
   - no pad is stranded fully inside or outside the outline.
7. Keep DRC active. Generate the narrow KiCad clearance exception required for the declared plated half-holes, scoped to their exact refs/pads and edge interaction. Do not blanket-waive copper-edge, courtyard, or containment errors for castellation-like footprints.

#### Files

- new reviewed bundle under `kicraft/parts_library/castellated-pad-2p54/`
- `kicraft/design/recipes/rp2040_minimal.py`
- `kicraft/design/synthesis/autoplacer.py`
- `kicraft/design/synthesis/validation.py`
- `kicraft/autoplacer/brain/array_placement.py`
- `kicraft/autoplacer/brain/placement_solver.py`
- `kicraft/autoplacer/brain/subcircuit_composer.py`
- `kicraft/autoplacer/routing_board.py`
- `kicraft/cli/_compose_validate.py`
- `tests/test_castellated_interface.py` (new)
- focused existing autoplacer/compose tests for edge arrays, parent outline, and DRC classification

#### Acceptance

- Rendered PCB shows true plated edge castellations, not vertical headers.
- Gerber/drill outputs contain the expected edge-intersecting plated holes/pads.
- Assembly exports omit castellation pseudo-components.
- A misplaced interior castellation and a one-row 30-header placeholder both fail deterministic validation.
- Frozen `rp2040-min` build reaches routing with the requested mechanical interface represented honestly.

---

## Phase 6 — reliability evaluation and statistical release gates

### 6.1 Extend deterministic analysis

#### Change

1. Increment `llm_analysis.SCHEMA_VERSION`.
2. Load `stage_attempts` and stage diagnostics by exact run ID.
3. Report per stage:
   - provider/transport completion;
   - schema-valid first pass;
   - commit first pass;
   - semantic-clean first pass;
   - semantic-clean after bounded repair;
   - unresolved advisory/repair/fab-gate counts by code;
   - attempts, latency, tokens, tools, and cost;
   - user-continuation rate.
4. Keep build outcomes in a separate section.
5. Treat missing telemetry as `not_observable`, never as a pass.
6. Add a privacy integrity check prohibiting brief/reasoning/answer/raw-candidate fields in attempt/diagnostic reports.

#### Files

- `kicraft/eval/llm_analysis.py`
- `kicraft/cli/web_cost_report.py`
- `tests/test_llm_analysis.py`
- `tests/test_stage_resource_telemetry.py`

#### Acceptance

- Recovered failures remain visible in aggregates.
- A stage with `ok=true, semantic_clean=false` is not counted as semantically successful.
- Provider failures, semantic failures, and build failures use separate denominators.

### 6.2 Add a frozen stage-reliability campaign

#### Change

1. Add a runner that replays individual stages from frozen upstream states through the real provider/runtime path.
2. Build a committed labeled corpus of at least 306 independently varied cases per stage:
   - all benchmark archetypes;
   - meaningful brief paraphrases;
   - historical failure fixtures;
   - legitimate empty/negative cases;
   - complex MCU, power, connector, array, hierarchy, and shaped-board cases.
3. Store expected facts/diagnostic outcomes, not exact prose or exact component ordering unless the recipe owns it.
4. Do not generate paraphrases during the measured campaign; campaign inputs are immutable and hashed.
5. Pin checkout, model, provider order, response-policy versions, detector versions, recipe versions, and spend envelope in a manifest.
6. Run downstream stages from frozen valid upstream states so early failures do not shrink downstream denominators.

#### Files

- `kicraft/eval/stage_reliability.py` (new)
- `kicraft/eval/stage_reliability_corpus.json` (new; or equivalent versioned data split by stage)
- `tests/test_stage_reliability.py` (new)
- `pyproject.toml` console entry if a dedicated command is useful

#### Acceptance

- Resume/checkpoint behavior mirrors self-eval and preserves exact case identity.
- Every stage has the full measured denominator even when another stage/provider case fails.
- Campaign integrity rejects changed inputs, duplicates, missing attempts, model/provider drift, or detector/recipe version drift.

### 6.3 Implement statistical gates

For a claim that a stage's true success rate exceeds 99% with a one-sided 95% lower confidence bound, require at least 299 independent observations with zero failures. Use 306 as the campaign floor. For the zero-failure gate, the exact lower bound is `0.05 ** (1 / n)`; no new statistics dependency is needed.

Required release gates per stage:

1. `n >= 299` valid independent cases;
2. zero terminal provider/transport failures for the provider-availability claim;
3. zero schema/commit failures for the commit-availability claim;
4. zero remaining labeled semantic defects after bounded repair for the semantic claim;
5. zero unlogged defects found by corpus expectations;
6. zero deterministic fab-gate escapes;
7. 100% user continuation for advisory-only findings;
8. p95 cost and wall time no worse than the frozen baseline by more than 10%, unless an explicitly approved correctness tradeoff documents the increase.

Do not combine these into one flattering percentage. A stage may pass semantic reliability and fail provider availability, or vice versa.

### 6.4 Rollout modes

Add one temporary deployment setting:

```text
KICRAFT_STAGE_SEMANTICS=observe|repair|enforce
```

- `observe`: diagnose and persist only; output is unchanged.
- `repair`: enable one best-effort semantic repair; unresolved findings warn.
- `enforce`: enable deterministic fabrication gates and verified recipes for supported families.

Rollout:

1. Deploy `observe`; inspect aggregate codes and manually review a redacted sample for false positives.
2. Run the 306-case frozen campaign.
3. Enable `repair`; repeat the same campaign and compare cost/latency/cleanliness.
4. Enable RP2040 recipe/castellation gates in `enforce`; replay the live fixture and RP2040 corpus.
5. Run a fixed live canary, then the full self-eval corpus.
6. Promote only if every release gate passes.
7. After the observation window, remove the temporary mode and obsolete branches; retain additive state/ledger migrations for historical records.

#### Files

- `kicraft/server/config.py`
- `kicraft/server/stage_runtime.py`
- deploy environment documentation/configuration already used by production
- `tests/test_config.py`
- campaign plan/report under `docs/plans/` only when the measured run exists

---

## Phase 7 — UI, reporting, and cleanup

### Change

1. Show yellow stage status for non-blocking semantic findings; show stable code and concise evidence in the stage inspector.
2. Keep detailed provider classification admin-only.
3. Add admin aggregates by stage, code, model, provider, detector version, recipe version, first-pass/repaired state, cost, and latency.
4. Update self-eval summary to show:
   - design committed;
   - semantic clean;
   - fab safe;
   - build/fab-ready outcome separately.
5. Remove any report logic that equates `stage_status.ok` with semantic success.
6. Update stage docs and CLI help for compact ranges, recipes, partial model-owned wiring, diagnostics, and warning semantics.
7. Delete temporary compatibility code after rollout; do not leave two permanent outcome systems.

### Files

- `kicraft/server/stagetabs.py`
- `kicraft/server/routes_admin.py`
- `kicraft/server/session.py`
- `kicraft/eval/self_eval.py`
- `kicraft/eval/llm_analysis.py`
- `kicraft/cli/web_cost_report.py`
- `.claude/skills/kicraft/stages/*.md`
- affected web/eval tests

### Acceptance

- Reopened projects reproduce live warning/fab-gate status from state alone.
- Admin reports can explain every failed/recovered provider attempt without raw customer content.
- Self-eval cannot report five semantically defective stages as five clean successes.

---

## Cross-phase test matrix

| Behavior | Unit | Runtime integration | Frozen replay | Live campaign |
|---|---|---|---|---|
| status dimensions/migration | required | required | — | — |
| attempt/provider classification | required | required | required | required |
| intent/FS diagnostics | required | required | required | required |
| bounded semantic repair fallback | required | required | required | required |
| compact architecture expansion | required | required | required | required |
| physical-domain/power graph checks | required | required | required | required |
| recipe expansion/provenance | required | required | required | required |
| recipe-owned wiring merge | required | required | required | required |
| RP2040 programming graph | required | required | required | required |
| real castellation geometry | required | build smoke | route/build replay | selected canary |
| >99% claim | — | — | 306 cases/stage | corroborating canary |

## Implementation order and merge boundaries

Implement as reviewable changes in this order:

1. Phase 0 fixtures.
2. Phase 1 status/attempt/provider telemetry.
3. Phase 2 pure diagnostics, then runtime repair integration, then prompts.
4. Phase 3 compact architecture contract and graph diagnostics.
5. Phase 4 recipe substrate, recipe wiring, RP2040 recipe, programming graph.
6. Phase 5 fabrication-only features and castellation geometry.
7. Phase 6 analyzer/campaign/gates.
8. Phase 7 UI/reporting and removal of temporary compatibility paths.

Do not combine recipe infrastructure, PCB castellation geometry, and telemetry migrations in one patch. Their acceptance evidence is different and a single rollback boundary would be unsafe.

## Final acceptance

The work is complete only when all of the following are true:

1. The frozen live RP2040 candidates produce the expected diagnostic codes.
2. Advisory failures never stop downstream stages.
3. A failed semantic repair commits the retained schema-valid candidate and records the repair failure.
4. Recovered provider/schema/commit failures remain durably attributable.
5. New runs never end with an unexplained generic `provider error` when status/code metadata exists.
6. The corrected RP2040 design uses a verified recipe, functional programming graph, correct special-pin handling, and real castellated board geometry.
7. Recipe-owned support wiring is deterministic; the LLM maps only project-specific pins.
8. Full pin/net/ERC/DRC/fabrication gates remain active.
9. A 306-case frozen campaign per stage meets the stated availability, semantic, privacy, cost, and latency gates.
10. A fixed live canary corroborates the frozen result without model/provider drift.
11. Self-eval and the UI distinguish committed, semantic-clean, fab-safe, and fab-ready outcomes.
12. Production rollout uses the real restart scripts from `AGENTS.md`, followed by HTTP and build-worker readiness verification; no `systemctl` path is introduced.
