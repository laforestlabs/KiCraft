# BOM emission bounds — bounded-collection contract for design-stage output

**Status:** implemented and live-verified 2026-08-25
**Source run:** `KC-KN6KDR` (`/home/kicraft/.kicraft/projects/1/745`, 2026-08-24)
**Investigation:** pipeline-gap report for `KC-KN6KDR` (GAP 1: "BOM emission unbounded — runaway output is terminal by design")
**Scope:** bound countable design-stage output at the contract, the serialization retry, and commit validation. One generic mechanism; BOM is the first consumer. Do not edit or repair the failed board.

## Problem

The BOM stage of `KC-KN6KDR` emitted a degenerate 818-part BOM (811 identical 10 kΩ
`Device:R` resistors R1–R811, everything on one sheet) that exceeded the output cap
twice and terminated the build:

| attempt | cap | emitted | result |
|---|---|---|---|
| normal | 16,384 tokens | 330 refs, truncated mid-`parts` | `truncated_json` |
| serialization retry | 32,768 tokens | 818 refs, truncated mid-entry | terminal `truncated_json` |

The build died at stage 4/5 with no schematic, ERC, or layout. The model never called a
tool; it went straight into runaway emission and spent ~40k output tokens on garbage.

Three pipeline facts make this outcome *guaranteed* for any runaway emitter:

1. **No gate can see truncated output.** Every existing BOM gate (pin existence, net
   coverage, polarity, sheet references, two-terminal self-short — `cli_app.py` commit
   checks, defined in `kicraft/design/synthesis/validation.py`) requires a *parsed* BOM.
   Truncated JSON never parses, so nothing ever validates it.
2. **The serialization retry is size-blind.** `_SERIALIZATION_RETRY_MSG`
   (`kicraft/server/stage_driver.py`) says "compact … fits the output budget" — no
   numeric bound, no feedback about how big the failed attempt was. The retry re-emits
   from the pristine task and runs away again at the larger cap.
3. **The contract has no bound.** `.agents/skills/kicraft/stages/bom.md` (119 lines)
   never states a part-count limit. The pipeline *already* accepts per-sheet caps
   downstream (`DENSE_SHEET_ROUTABLE_MAX = 15` routable parts per sheet in
   `sheet_partition.py`) — it just never applies that philosophy at the stage that
   emits the parts.

The commit `39ca3a9` fixed truncation *recovery* (one fixed-cap, tool-free retry) but
deliberately leaves a second truncation terminal. This run executed on that fixed code
and behaved exactly per its spec. The residual gap is that nothing bounds emission, so
a runaway emitter is terminal by design. Raising caps is not a fix: a runaway beats any
finite cap, and the 32,768 ceiling already cost 324 s and ~40k garbage tokens.

## Critical review findings

The original proposal had five correctness problems:

1. **The proposed 250-total / 80-per-sheet limits contradicted the existing array
   contract.** `bom.md` explicitly supports a 200-part LED matrix, and its decoupling
   rules can make that a 400-part single-sheet BOM. `DENSE_SHEET_ROUTABLE_MAX` is not
   evidence for a BOM limit: arrays bypass the general-purpose placer precisely because
   they are intentionally dense. The limits are therefore raised to 500 total and 450
   per sheet. These still reject the 818-part witness while preserving the largest
   concrete design documented by the current contract.
2. **The proposed policy type was false.** `group_key` is a string, not an integer, so
   `dict[str, dict[str, dict[str, int]]]` could not describe the shown value. The
   implementation uses an explicit `CollectionBound` value object and immutable tuples
   in `StageResponsePolicy`.
3. **Hard-coding the same numbers in both `config.py` and `bom.md` created drift.**
   `build_system()` now derives the first-attempt bounded-output paragraph from the
   policy table. The retry formatter consumes the same policy object. The static stage
   document keeps only collection semantics; numeric policy has one source of truth.
4. **The original guarantee was impossible without an in-stream answer breaker.** A
   model can ignore a prompt and hit both token caps; truncated JSON cannot safely
   provide an exact item count. This change makes bounds explicit, makes a retry
   materially informed, and rejects parseable oversize output. A second truncation
   remains a loud terminal `truncated_json`, not cardinality feedback. No claim is made
   that prompt text mechanically prevents the second cap.
5. **The first implementation did not enforce its “before sourcing/build work”
   criterion.** It evaluated the size check inside an aggregate, then continued through
   footprint, symbol, pin, and catalog resolution before returning the rejection.
   Cardinality is now a true preflight: an oversized BOM returns immediately, before
   normalization or any identity/sourcing resolver.

## Goals

- Give every configured countable design-stage collection a numeric bound in the
  first-attempt system contract.
- Make the serialization retry carry that bound plus the prior attempt's measured
  character size.
- Add a deterministic, parse-side cardinality gate so an oversized-but-parseable BOM
  is rejected through the existing commit-correction loop.
- Keep the mechanism generic and single-source: one policy table, one instruction
  formatter, and one validation helper. BOM is the first consumer.
- Preserve existing response caps, retry counts, and behavior for stages without a
  collection policy.

## Non-goals

- **No cap raising.** Token caps continue to bound cost; collection bounds constrain
  parseable content.
- **No prefix salvage.** Never commit or repair a truncated JSON prefix.
- **No in-stream answer repetition detector.** This change does not pretend that prompt
  instructions enforce a byte ceiling.
- **No judge/LLM gate.** Cardinality is deterministic.
- **No per-design repair for `KC-KN6KDR`.**
- **No weakening of existing BOM gates.**

## Design

One policy is consumed at three enforcement points:

```
first attempt   -> build_system() injects the configured numeric bounds
parse failure   -> serialization retry quotes the bounds and prior character count
parse success   -> BOM commit rejects cardinality violations with offender feedback
```

### Bound numbers

| collection | bound | rationale |
|---|---|---|
| BOM `parts`, total | **500** | Preserves the documented 200-member array plus one decoupler per member and ordinary support circuitry; rejects the 818-part witness. |
| BOM `parts`, per sheet | **450** | Preserves that concrete single-sheet array case; rejects the 818-on-one-sheet witness. This is intentionally unrelated to the non-array placer threshold. |

These are degeneration guards, not a statement that a 500-part response is guaranteed
to fit a provider token cap. A legitimate design beyond these current product limits
must be split or the policy deliberately revised with a new concrete workload.

### Typed policy table

`kicraft/server/config.py` defines:

```python
@dataclass(frozen=True)
class CollectionBound:
    field: str
    total: int
    per_group: int | None = None
    group_key: str | None = None

STAGE_COLLECTION_BOUNDS = {
    "bom": (CollectionBound("parts", total=500,
                            per_group=450, group_key="sheet"),),
}
```

`StageResponsePolicy.collection_bounds` is an immutable tuple. Settings copy the
configured stage tuple into the per-drive policy. Legacy/mock clients use the same
table fallback. Stages without an entry receive an empty tuple.

### Shared instruction formatter

`stage_driver.py` formats a policy tuple into a deterministic sentence. `build_system()`
injects it into the first-attempt system prompt; serialization recovery appends the same
sentence and `len(raw)`:

```text
Your previous reply was not a single complete JSON object (the prior reply was about
N characters and was truncated or malformed), so nothing was committed. The `parts`
collection must contain at most 500 items total and at most 450 items per `sheet`. ...
```

This avoids duplicating numbers in `.agents/skills/kicraft/stages/bom.md`. For a stage
without bounds, the only retry-message change is the newly reported prior size.

### Validation gate

`kicraft/design/synthesis/validation.py` adds:

```python
def check_collection_bounds(
    field: str,
    items,
    *,
    total: int,
    per_group: int | None = None,
    group_key: Callable | None = None,
) -> CheckResult:
```

Total overflow and every over-limit group are returned as offenders, sorted
deterministically by descending count and then group name. `check_bom_size()` reads the
canonical BOM policy and supplies `part.sheet` as the group function. `cli_app.py`
runs this as a preflight before BOM normalization, identity resolution, or sourcing;
an oversized parseable BOM is rejected immediately with exact offender counts.

## Implementation

1. **`kicraft/server/config.py`** — add `CollectionBound`,
   `STAGE_COLLECTION_BOUNDS`, and the immutable policy field; populate it in both real
   settings and mock/legacy fallback paths.
2. **`kicraft/server/stage_driver.py`** — add the shared formatter, inject it through
   `build_system()`, and format the serialization retry with the prior character count.
   Keep serialization budget, fixed token cap, tool-free `chat()`, and reasoning-off
   behavior unchanged.
3. **`kicraft/design/synthesis/validation.py` + `kicraft/design/cli_app.py`** — add
   the generic helper and BOM wrapper, then enforce it as a short-circuiting BOM commit
   preflight before normalization, identity resolution, or sourcing.
4. **Tests** — cover policy propagation/fallback, first-attempt prompt injection,
   bounded and unbounded retry messages, total and per-sheet rejection, deterministic
   offender text, and the BOM commit path.

## Verification

Focused deterministic suite (no LLM, no network):

```bash
.venv/bin/python -m pytest -q \
  tests/test_stage_driver_retry.py \
  tests/test_kicraft_validation.py \
  tests/test_stage_driver_prompt_examples.py \
  tests/test_kicraft_stage_cli.py
```

Result: **148 passed, 4 skipped** in 104.97 s.

The frozen witness was replayed live three times on 2026-08-25 with a $0.25
per-run ceiling:

```python
from kicraft.server.stage_driver import drive_replay
drive_replay("/home/kicraft/.kicraft/projects/1/745/.kicraft/state.json",
             "bom", budget_usd=0.25)
```

| run | workspace | attempts | tool calls | parts | max sheet | cost | terminal result |
|---|---|---:|---:|---:|---:|---:|---|
| 1 | `/tmp/kc-replay-jb0fzdnf` | 4 | 17 | 57 | 9 | $0.038463 | committed |
| 2 | `/tmp/kc-replay-izfy9k7i` | 6 | 3 | 60 | 9 | $0.046536 | committed |
| 3 | `/tmp/kc-replay-ja56lco1` | 3 | 0 | 72 | 12 | $0.026969 | committed |

Total paid cost: **$0.111968**. All three frozen-witness replays committed a BOM.
None triggered §9.35, `truncated_json`, `invalid_json`, or the original 818-identical-
resistor signature. The largest result contained 72 parts; the densest sheet contained
12. This N=3 live sample exercises the real provider, tool loop, commit correction,
identity checks, and sourcing path under the $0.25 per-run ceiling.

## Success criteria

- The first attempt and serialization retry quote the same canonical numeric policy.
- Parseable BOMs above either limit are rejected before sourcing/build work, with exact
  total or sheet counts in commit feedback.
- A retry quotes the prior reply's measured character length.
- Stages without a collection policy retain their caps and retry semantics and receive
  no collection-bound sentence.
- The documented 200-member array plus one support part per member remains below the
  configured limits.
