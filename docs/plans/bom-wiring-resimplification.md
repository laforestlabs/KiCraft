# BOM and wiring resimplification

**Status:** superseded by the clean-cutover plan below; implementation authorized
on `simplify/bom-wiring-pipeline`.

## Critical analysis

The original exploration identifies the correct problem and the correct target
mental model, but it is not an executable plan:

- It postpones the production decision through six evidence phases even though
  the existing code already demonstrates the two representation splits.
- The proposed BOM example does not define how ref-bearing fields such as
  arrays, placement hints, IC groups, and zones survive deterministic reference
  generation. Keeping them unchanged would make the model predict generated
  refs and recreate the same fragility.
- The proposed wiring shape is sound, but the plan does not name the runtime
  state machine and telemetry that must disappear with the patch DSL.
- Validation ownership cannot be solved only inside the response translators.
  The commit monolith still owns electrical gates and deterministic
  normalizations. Moving those gates during the schema cutover would combine two
  high-risk changes and weaken proof.
- Offline translators followed by a later clean cutover would intentionally add
  a second temporary path. The production canonical BOM and wiring shapes
  already provide the translator boundary, so a direct response-contract
  cutover is smaller.

## Chosen clean cutover

Keep the durable canonical `BOM` unchanged so synthesis, placement, archived
projects, and electrical commit gates do not need migration. Replace only the
model-facing contracts and normalize once before commit:

1. **BOM response:** one non-empty `groups` list. Each group supplies an id,
   reference prefix, quantity, sheet, and shared part identity. KiCraft assigns
   sequential refs per prefix. Arrays name one complete group; KiCraft derives
   their refs. Optional canonical placement metadata that requires model-chosen
   refs is omitted and defaults deterministically.
2. **Wiring response:** one `pins` list. Each `(ref, pin)` occurs once and has
   exactly one final state: `net` or `no_connect: true`. KiCraft derives sheets
   from the canonical BOM and groups connected pins by `(sheet, net)`.
3. **Correction:** every retry returns a complete replacement under the same
   stage schema. Delete the seven-operation wiring patch union, patch
   preconditions, patch-only prompts, alternate response format, and
   patch-specific runtime state.
4. **Validation:** response models own shape and cardinality; the two pure
   normalizers own reference expansion and connection grouping; the existing
   atomic stage commit continues to own real-part, pin, coverage, and electrical
   safety gates.
5. **Observability:** retain call/collection/correction events, but delete
   representation-specific patch telemetry. Record only expanded component
   count for the group-to-canonical boundary.
6. **Cutover:** update prompts, response policy names, analysis tooling, and
   tests in the same branch. No aliases, legacy response schemas, or runtime dual
   path.

Acceptance:

- BOM v2 rejects `parts` and `part_runs`; wiring v2 rejects `connections`,
  `no_connect_pins`, and patch operations.
- Group expansion is deterministic across repeated prefixes and preserves total
  and per-sheet limits.
- Wiring normalization rejects duplicate or unknown endpoints and derives
  canonical connection sheets without model input.
- A rejected wiring candidate receives the ordinary wiring v2 response contract
  on correction.
- Existing canonical commit gates still accept valid normalized candidates and
  reject unsafe ones before durable state changes.

## Why this exists

The BOM and wiring stages have become too difficult to explain, change, and debug.
The immediate canary failures are real, but adding more schemas, patch operations,
limits, recovery modes, and telemetry to the current design would deepen the
problem.

The previous tactical plan, `llm-canary-fixes.md`, remains useful as failure
evidence. It is not the implementation plan.

This work steps back and answers a more important question:

> What is the smallest BOM-and-wiring pipeline that can still produce safe,
> buildable KiCad designs?

## The intended mental model

A new contributor should be able to understand the two stages as follows:

1. **BOM:** choose the components and how many of each are needed.
2. **Wiring:** assign every component pin to one net or mark it intentionally
   unconnected.
3. **KiCraft:** derive references, sheets, connection rows, and other redundant
   structures wherever it can do so deterministically.
4. **Validation:** reject electrically unsafe or impossible results before they
   become durable state.

Anything more complicated must justify its existence with a concrete design that
cannot be represented this way.

## Working hypotheses

These are starting points to test, not decisions to defend.

### BOM: one group-first representation

Today the model can emit both individual `parts` and compact `part_runs`, after
which KiCraft expands and validates them through several overlapping limits.

Prototype one representation in which every BOM entry describes a component type
and quantity. A quantity of one represents an ordinary unique component. KiCraft
generates the individual references deterministically.

A rough example:

```json
{
  "groups": [
    {
      "id": "mcu",
      "sheet": "CONTROLLER",
      "part": {"symbol": "...", "footprint": "...", "value": "..."},
      "quantity": 1,
      "reference_prefix": "U"
    },
    {
      "id": "led_array",
      "sheet": "DISPLAY",
      "part": {"symbol": "...", "footprint": "...", "value": "..."},
      "quantity": 200,
      "reference_prefix": "D"
    }
  ]
}
```

The exact schema is an exploration output. The important property is that there
is no ordinary-versus-run dual path.

### Wiring: final pin assignments, not a seven-operation patch language

Prototype a wiring response as final pin state:

```json
{
  "pins": [
    {"ref": "U1", "pin": "1", "net": "+3V3"},
    {"ref": "U1", "pin": "2", "net": "GND"},
    {"ref": "U1", "pin": "3", "no_connect": true}
  ]
}
```

KiCraft already knows the sheet for each component from the BOM. It should derive
`NetConnection` rows by grouping pin assignments by `(sheet, net)`. The model
should not repeat sheet names in endpoint records or manage duplicate connection
rows.

Start correction with a complete replacement response using this same schema. If
measured response size makes full replacement impractical, test a single
endpoint-replacement shape. Do not recreate the current union of seven operation
types.

### Validation: three boundaries

Every rule should belong to one of three places:

1. **Response shape:** required fields, basic types, and finite collection sizes.
2. **Deterministic normalization:** reference generation, quantity expansion,
   sheet derivation, and connection grouping.
3. **Electrical commit checks:** real part, pin, net, coverage, and safety
   invariants.

A rule with two owners is a duplication candidate. A correction mechanism that
exists only because two representations disagree is a deletion candidate.

## Invariants that simplification may not weaken

The exploration must preserve these outcomes, even if their implementation
changes:

### BOM

- Every committed component has one unique valid reference.
- Every component belongs to a declared architecture sheet.
- Symbols and footprints resolve.
- Expanded quantities have explicit total and per-sheet limits.
- Arrays and repeated blocks remain expressible.
- Implausible designs are rejected, not silently trimmed.

### Wiring

- Every endpoint refers to a committed component and a real pin.
- A pin has at most one final assignment.
- A pin cannot be both connected and no-connect.
- Inter-sheet architecture nets are realized correctly.
- Required pins are either connected or deliberately no-connect.
- Existing electrical §9 gates remain effective until a simpler equivalent is
  proven against the same failures.

### Pipeline

- Invalid candidates never mutate durable state.
- Retry and provider-call budgets remain finite.
- No model, provider, price, temperature, or token-policy change is part of this
  exploration.

## Phase 1 — Draw the current system

Produce two short maps, one for BOM and one for wiring. Each map follows one model
response through:

```text
provider response
→ schema decode
→ normalization
→ commit checks
→ durable state
→ retry or success
```

For each step record:

- owning function and file;
- input and output shape;
- whether it changes data;
- every retry or recovery branch;
- every limit applied;
- every validator and whether another layer repeats it.

Also produce a deletion inventory covering:

- the `parts` / `part_runs` split;
- wiring patch operation models;
- contextual patch messages and constraints;
- duplicate-connection repair;
- serialization-specific recovery;
- prose parsing used to authorize refs, pins, sheets, or nets;
- telemetry that exists only to explain those mechanisms.

**Deliverable:** two diagrams, two source-of-truth tables, and a ranked deletion
inventory. Do not propose new production abstractions during this phase.

## Phase 2 — Reduce the contract on paper

Define the minimum information the model must supply and everything KiCraft can
derive.

Answer these questions explicitly:

1. Must the model choose references, or only prefixes and quantities?
2. Which repeated structures come from the brief or functional specification
   rather than the BOM response?
3. Can sheet membership be derived for any BOM groups?
4. Must the model emit connection rows, or only final pin assignments?
5. Which net names are architecture-owned, and which may be sheet-local?
6. Which current commit checks defend electrical safety, and which only defend a
   complicated representation?

Compare no more than three alternatives for each stage. Select the smallest
contract that can represent the frozen corpus. Prefer deleting model-supplied
fields over adding validation for them.

**Deliverable:** one page per stage containing the proposed response shape, the
canonical committed shape, and the deterministic transformation between them.

## Phase 3 — Build offline translators

Implement test-only pure functions outside the production stage path:

```text
simple BOM response → current canonical BOM
simple pin assignments → current canonical wiring
```

Use frozen fixtures from:

- the two successful canary designs;
- the six wiring failures;
- the two BOM explosion shapes;
- at least one large legitimate array;
- at least one dense MCU design;
- repeated functional blocks.

The translators must prove that the simpler input can still feed the existing
commit gates. They must not call an LLM, write project state, or add compatibility
logic to production.

**Decision gate:** stop if a required design cannot be represented without adding
substantial stage-specific exceptions. Document the missing information before
changing the proposed contract.

## Phase 4 — Test correction without the patch DSL

Replay the frozen wiring failures through two correction strategies, in order:

1. full replacement using the simple final-pin schema;
2. only if needed, replacement records for the rejected endpoints using that
   same schema.

Measure:

- serialized response size;
- whether each original defect is expressible;
- number of local rejection modes;
- whether unrelated final assignments change;
- number of correction-specific branches required.

The preferred strategy is the first one that fits the existing call and token
budgets. Simplicity wins unless the measurements show it cannot work.

**Decision gate:** the replacement design must eliminate the seven-operation
patch union, stale operation preconditions, and duplicate connection-key
initialization failures.

## Phase 5 — Collapse validation ownership

Classify every BOM and wiring validator from Phase 1 as:

- keep at response shape;
- keep in normalization;
- keep as an electrical commit check;
- delete as duplicate or representation-only.

Give each surviving invariant one authoritative implementation and one stable,
structured error. Retry guidance should consume those structured errors directly;
it must not parse human-readable diagnostic prose.

Do not weaken a §9 gate merely because it is inconvenient. Replace or remove it
only when the simpler representation makes the invalid state impossible or an
equivalent check is proven against its regression fixture.

**Deliverable:** an old-to-new validator table and a concrete deletion list by
symbol and file.

## Phase 6 — Choose the production cutover

Only after the offline evidence passes, write a separate implementation plan for
a clean cutover. It must specify:

- final schemas;
- state migration, if durable state changes;
- all callers and tests to migrate;
- obsolete code to delete in the same change;
- deterministic replay coverage;
- one fresh paid canary after local replay passes.

Do not preserve the old BOM or patch contracts through aliases or indefinite dual
paths. If old durable projects require migration, migrate their data at one
explicit boundary and keep the runtime single-path.

## Success criteria for the exploration

The exploration succeeds when all of the following are true:

- BOM can be explained as component groups expanded deterministically.
- Wiring can be explained as final pin assignments grouped deterministically.
- One response representation exists per stage.
- Correction reuses the wiring representation instead of a separate patch
  language.
- Every surviving invariant has one owner.
- Historical successful designs remain representable.
- Historical runaway and invalid-wiring fixtures still fail safely.
- The proposed cutover deletes more stage-specific machinery than it adds.
- A production implementation plan can fit on a few pages without requiring a
  glossary of recovery modes.

## Non-goals

- Do not implement another tactical layer on the current patch contract.
- Do not tune the model to compensate for a complicated representation.
- Do not weaken electrical safety gates.
- Do not silently delete model-selected parts or wiring.
- Do not run a paid canary during exploration.
- Do not deploy or restart production services.
- Do not redesign functional specification, architecture, placement, or routing
  unless the exploration proves a specific missing upstream fact forces it.
