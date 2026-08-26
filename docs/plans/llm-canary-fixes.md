# LLM canary fixes

**Status:** superseded as an implementation plan by
`bom-wiring-resimplification.md`. Keep this document as canary evidence and a
catalog of observed failure shapes; do not implement its phased fixes before the
resimplification exploration chooses the new BOM and wiring contracts.

## Context

The fixed nine-slug production canary at
`/home/kicraft/.kicraft/self_eval/20260826T052354Z_llm_canary` produced a valid
`FAIL_LLM` verdict:

- 2/9 design-complete;
- 6/9 terminal wiring `commit_contract` failures;
- 1/9 terminal BOM `serialization` failure (`collection_limit`);
- $0.338351 cohort spend and $0.338400 including paid preflights;
- zero candidate or committed unknown-sheet violations.

Reports:

- `/home/kicraft/.kicraft/self_eval/20260826T052354Z_llm_canary/llm_analysis.json`
- `/home/kicraft/.kicraft/self_eval/20260826T052354Z_llm_canary/llm_analysis.md`
- `/home/kicraft/.kicraft/self_eval/20260826T052354Z_llm_canary/canary.log`

This plan fixes deterministic contract failures before considering any model or
policy change. Do not resume the completed canary; a resume would cherry-pick
stochastic results.

## Problem

The canary exposed two root defect families, not a general model-selection
problem.

| Family | Evidence | Impact |
|---|---|---|
| Typed wiring-patch contract is too permissive at generation and too strict at application | Six wiring failures used five attempts each, but zero patch replies decoded successfully. Replies used architecture sheet names as `endpoint.ref`, attempted duplicate connections, or used `add_endpoint` for already-assigned pins. | The correction budget was spent rejecting patch syntax and preconditions rather than correcting §9.10, §9.11, §9.15, §9.17, and §9.19 defects. |
| BOM cardinality controls permit implausible passive explosions | `nrf52-beacon` committed 321 parts, including 314 capacitors and 303 identical 1 µF capacitors. `dual-rail-supply` emitted 328 ordinary parts, then its recovery reached 451 parts on one sheet and terminated at `collection_limit`. | Excessive cost, serialization failure, and electrically implausible committed BOMs. |

Supporting findings:

- `r2r-dac` proves typed patching can work: one decoded patch, three operations,
  unrelated endpoints stable, design completed.
- `usb-a-power-splitter` completed first-pass wiring.
- No candidate or committed unknown-sheet violations: retain the closed
  sheet-name contract unchanged.
- Provider, model, profile, and response-policy attribution was exact. No
  routing or model fix is indicated.
- The baseline improved from 0/9 to 2/9, but seven terminal failures still
  require `FAIL_LLM`.

## Decision

Fix deterministic contracts before changing prompts, model, reasoning policy,
retries, or token limits.

Priority:

1. Make every wiring patch structurally expressible and schema-constrained.
2. Stop BOM explosions before they consume a full response.
3. Preserve exact diagnostics for locally rejected patch replies.
4. Replay the canary failures without paid stochastic rerolls.
5. Run a fresh paid canary only after deterministic regression coverage passes.

## Non-goals

- Do not change the design or judge model.
- Do not change provider order, price caps, temperature, reasoning policy,
  retry counts, or token caps.
- Do not weaken §9 electrical commit gates.
- Do not auto-delete or silently reduce model-selected components.
- Do not rerun or resume the completed canary to obtain a better sample.
- Do not deploy or restart production services as part of implementation.

## Phase 1 — Freeze minimal regression reproductions

### Files

- `tests/test_stage_driver_retry.py`
- stage-contract and validation test modules following existing conventions

### Changes

Add minimized fixtures derived from the benchmark canary, not full raw
transcripts. Cover these exact failure shapes:

1. A patch uses a sheet name such as `USB INPUT` as `endpoint.ref`.
2. A patch attempts `add_connection` for an existing `[sheet, net]` key.
3. A patch uses `add_endpoint` when the endpoint is already assigned and should
   use `set_pin_net`.
4. A full wiring candidate contains duplicate `[sheet, net]` connection rows.
5. A BOM contains 303 identical passives on a sheet with one active IC.
6. Serialization recovery emits 451 ordinary parts instead of `part_runs`.

Fixtures must contain only structural fields:

- operation type;
- refs, pins, sheets, and nets;
- gate IDs and normalized offenders;
- collection counts;
- no reasoning, answer text, or tool output.

### Acceptance

- Each fixture fails against the current implementation for the observed
  reason.
- Tests identify the root boundary: provider-schema rejection, patch
  precondition failure, duplicate-key initialization failure, or BOM
  cardinality gate.

## Phase 2 — Constrain wiring patches at provider-schema time

### Files

- `kicraft/server/stage_wiring_patch.py`
- `kicraft/server/stage_runtime.py`
- `tests/test_stage_driver_retry.py`

### Changes

### 2.1 Build a contextual patch response schema

Change `wiring_patch_response_format()` to accept the current patch constraints,
conceptually:

```python
wiring_patch_response_format(
    *,
    addable_refs: set[str],
    removable_refs: set[str],
    allowed_sheets: set[str],
    allowed_nets: set[str],
    existing_nets: set[str],
) -> dict
```

Apply enums to the generated JSON schema:

- `add_endpoint.ref`, `set_pin_net.ref`, and
  `mark_no_connect.endpoint.ref`: committed BOM refs only;
- removal preconditions: refs present in the current candidate, including an
  invalid or unknown ref that must be removed;
- `sheet`: exact committed architecture sheet names;
- new `net`: allowed architecture and current nets;
- `expected_net`: current candidate nets plus `null`.

Do not use one global ref enum. Removal must remain capable of deleting a bad
candidate endpoint, while additions must never introduce one.

### 2.2 Resolve constraints before the provider call

In `stage_runtime.drive_stage`:

1. Compute `wiring_patch_constraints` before each patch call.
2. Build the contextual response format from those exact constraints.
3. Pass the same immutable constraint set to `decode_stage_response` and
   `apply_wiring_patch`.

Invariant: the provider schema and local validator describe the same allowed
operation space.

### 2.3 Distinguish additive and subtractive validation

In `apply_wiring_patch`:

- additions and reassignment destinations require committed BOM refs and valid
  pins;
- removal operations may target an endpoint demonstrably present in the
  candidate, even when that endpoint caused an unknown-ref rejection;
- new connections remain fully strict;
- removal preconditions remain exact and stale-safe.

Never reinterpret `add_endpoint` as `set_pin_net`. Return a typed precondition
error so the next constrained correction must choose the correct operation.

### Acceptance

- A sheet name cannot be emitted as `endpoint.ref` under the contextual response
  schema.
- A candidate containing an unknown endpoint can still be repaired by removing
  it.
- New endpoints cannot use unknown refs, pins, sheets, or nets.
- Stale preconditions still fail without mutation.
- The existing seven typed operations remain the public contract.

## Phase 3 — Normalize duplicate wiring connection keys

### Files

- `kicraft/server/stage_contracts.py`
- `kicraft/server/stage_wiring_patch.py`
- relevant wiring contract tests

### Changes

Add a deterministic canonicalization step for full wiring candidates:

- group connections by `(sheet, net_name)`;
- merge endpoints for identical keys;
- deduplicate identical endpoints;
- preserve stable ordering;
- do not merge different nets;
- do not resolve pins assigned to multiple nets.

Apply it after local response decoding and before the candidate is retained for
commit or patch correction.

Record the number of coalesced connection rows in `candidate_decoded`, for
example:

```json
{
  "coalesced_connection_rows": 2
}
```

This removes a representational duplicate while leaving electrical defects
visible to §9.17 and §9.19.

### Acceptance

- Duplicate same-net rows no longer make `apply_wiring_patch` impossible to
  initialize.
- Multi-net assignments remain rejected.
- Two-terminal self-shorts remain rejected.
- Canonicalization is idempotent and order-stable.
- No unrelated endpoint assignment changes.

## Phase 4 — Make patch rejection and progress observable

### Files

- `kicraft/server/stage_runtime.py`
- `kicraft/eval/self_eval.py`
- `kicraft/eval/llm_analysis.py`
- `tests/test_stage_driver_retry.py`
- `tests/test_llm_analysis.py`

### Changes

When patch parsing or application fails, emit a privacy-safe structural event
before retrying:

```json
{
  "kind": "wiring_patch_rejected",
  "stage": "wiring",
  "attempt": 3,
  "failure_code": "unknown_ref | stale_precondition | duplicate_connection | already_assigned | invalid_patch_schema",
  "target_endpoint_count": 2,
  "clean_slate_next": false
}
```

Do not include raw Pydantic errors or the full patch payload.

Analyzer changes:

- count billed patch calls from ledger `response_policy_name`;
- separately count patch replies billed, decoded, applied, and locally rejected;
- include patch rejection codes in progress analysis;
- treat repeated identical local rejection codes after clean slate as
  `no_progress`;
- retain `unrelated_endpoints_stable` only for successfully applied patches.

### Acceptance

For each wiring attempt:

```text
billed patch calls
= decoded/applied patch calls
+ locally rejected patch calls
+ provider/transport failures
```

No failed patch attempt disappears from the report.

## Phase 5 — Separate ordinary BOM items from compact repetitions

### Files

- `kicraft/server/config.py`
- `kicraft/server/stage_contracts.py`
- `kicraft/server/stage_prompts.py`
- collection-guard tests
- stage retry tests

### Changes

### 5.1 Lower only the raw ordinary-parts bound

Keep the expanded canonical limits:

- 500 total parts;
- 450 expanded parts per sheet.

Add tighter serialized representation limits:

- `parts`: initially 128 ordinary items total and 96 per sheet;
- `part_runs`: initially 64 run declarations;
- each run may still expand within the existing 450-reference bound.

The constants must be named and tested rather than embedded independently in
prompts, schemas, and guards.

This preserves legitimate arrays while preventing hundreds of repeated
passives from being serialized individually.

### 5.2 Put raw limits in both schema and stream guard

Ensure the same limits exist in:

- `BomStageResponse` JSON schema;
- `STAGE_COLLECTION_BOUNDS`;
- normalization validation;
- recovery guidance.

Do not leave a mismatch where the schema permits 500 raw parts but the stream
guard stops earlier.

### 5.3 Make collection recovery explicitly compact

For a `parts` collection limit, include structural recovery data:

- observed raw item count;
- offending sheet when known;
- raw `parts` maximum;
- expanded maximum;
- instruction to convert identical members to `part_runs`.

Keep the existing single, tool-free, reasoning-disabled recovery call and fixed
token cap.

### Acceptance

- A 200-LED array represented with `part_runs` expands and commits.
- The same array serialized as 200 ordinary repeated rows is rejected early.
- `dual-rail-supply`-shaped runaway output stops at the raw bound, not item 451.
- Recovery cannot exceed the same raw schema and stream limits.

## Phase 6 — Add an implausible repeated-passive commit gate

Lower raw serialization limits alone would allow the 303-capacitor error to be
encoded efficiently. A semantic gate is also required.

### Files

- `kicraft/design/synthesis/validation.py`
- BOM commit/check orchestration in `kicraft/design/cli_app.py`
- validation tests

### Changes

Add a conservative repeated-passive sanity check grouped by:

```text
(sheet, symbol, footprint, value)
```

Flag a repeated passive group when its count is implausible relative to the
sheet's active component population. Initial rule:

```text
allowed identical passives = max(32, 2 × non-passive parts on the sheet)
```

Refine it with explicit array evidence:

- allow counts justified by a declared `ArraySpec`;
- allow one-per-array-member decoupling when the relationship is explicit;
- allow an explicit functional-spec block count;
- otherwise reject with a deterministic gate and offenders summarized by group
  and count.

Example rejection:

```text
repeated passive runaway: 303 × 1uF capacitors on MCU NRF52840,
but the sheet has 1 non-passive device and no declared array
```

Do not auto-delete or silently reduce components.

### Acceptance

- The nRF fixture with 303 identical 1 µF capacitors fails before wiring.
- Twelve LEDs plus twelve decouplers passes.
- A declared 200-member LED array plus 200 decouplers passes.
- Ordinary resistor ladders and connector pull networks below the floor pass.
- Gate output contains counts and normalized group identity, not the full BOM.

## Phase 7 — Improve correction context without changing policy

### Files

- `kicraft/server/stage_wiring_patch.py`
- patch-message tests

### Changes

Extend the structural patch context with:

- each offender endpoint's current assignment set;
- exact allowed pins for offender refs;
- existing connection keys;
- whether the required operation is structurally add, remove, reassign, or
  remove-and-recreate connection;
- explicit statement that `add_endpoint` requires no current assignment and
  `set_pin_net` requires one exact current assignment.

These are deterministic hints derived from candidate state and gate output, not
electrical guesses.

Do not change:

- system-wide prompts outside the patch correction message;
- model or provider;
- temperatures;
- retry count;
- token caps;
- commit gates.

### Acceptance

Scripted replies for the canary's three patch-error families converge within
the existing five-call wiring budget.

## Phase 8 — Verification

### Deterministic coverage

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_stage_driver_retry.py \
  tests/test_llm_analysis.py \
  tests/test_self_eval.py \
  tests/test_client_provider.py \
  tests/test_stage_resource_telemetry.py
```

Required contracts:

- contextual patch-schema enums;
- unknown-ref removal versus strict addition;
- duplicate connection canonicalization;
- stale-precondition immutability;
- patch rejection telemetry;
- billed/decoded/rejected call reconciliation;
- raw BOM versus expanded BOM bounds;
- repeated-passive semantic gate;
- privacy sentinel exclusion.

### Artifact replay

Before another paid campaign, reconstruct minimized provider replies from these
six failed benchmark runs and replay them through `decode_stage_response` and
commit correction:

- `stm32-min`;
- `nrf52-beacon`;
- `encoder-oled-panel`;
- `can-node`;
- `servo-driver-16`;
- `round-led-ring`.

Expected:

- no reply is trapped solely because a patch used a sheet as a ref;
- duplicate connection rows are canonicalized;
- every rejected patch attempt is observable;
- electrically wrong patches still fail the original §9 gates.

Replay BOM cardinality for:

- `nrf52-beacon`;
- `dual-rail-supply`.

Expected:

- runaway responses fail cheaply and deterministically;
- legitimate compact arrays remain supported.

### Fresh paid canary

Only after deterministic replay passes, run a new directory through
`kicraft-llm-canary`. Do not resume the completed failing batch.

Success criteria for the next canary:

- valid campaign integrity;
- exact nine fixed slugs;
- zero terminal `invalid_patch_schema`, duplicate-key initialization, or
  unobservable patch failures;
- zero candidate or committed unknown-sheet violations;
- zero implausible repeated-passive groups;
- zero BOM raw-part collection exhaustion;
- any remaining wiring failure must be an actual electrical gate failure after
  at least one successfully decoded constrained patch, not a patch-contract
  trap.

A 9/9 result is desirable but is not the acceptance criterion for these fixes.
The immediate contract is narrower: remove deterministic correction dead ends
without weakening electrical gates or increasing budgets.
