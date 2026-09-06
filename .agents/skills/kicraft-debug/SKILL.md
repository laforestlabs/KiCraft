---
name: kicraft-debug
description: Inspect, debug, or walk through KiCraft's real provider-backed LLM design stages with explicit review before commit. Activate only when the user explicitly asks to debug, inspect, review, or step through KiCraft LLM stage decisions; ordinary PCB design requests use the kicraft skill.
compatibility: Requires the KiCraft server and design extras, OpenRouter configuration, and an agent capable of reading files, writing temporary files, and running local commands.
---

# KiCraft stage debugger

Use the production KiCraft stage driver, paused immediately before durable commit. Guide the user through one stage candidate at a time. Do not emulate a stage, draft slot JSON yourself, or substitute the active agent's own model for the configured production provider.

## Invariants

- Never use `cd`. The current working directory is the project workspace for the entire session.
- Read `.kicraft/state.json` at the start of every turn. An absent file means no stage has been accepted yet.
- Never hand-edit `.kicraft/state.json`. Only `kicraft-stage-debug debug-commit` may accept a reviewed candidate.
- Process exactly one stage at a time in canonical order: `intent`, `functional_spec`, `architecture`, `bom`, `wiring`.
- Reuse the sibling `../kicraft/stages/<stage>.md` files as the canonical electrical and schema contract. Read the relevant specification; never restate or maintain a second contract here.
- A pending artifact is `.kicraft/debug/<stage>.json`. It contains the complete current candidate and forensic trace. A redraft replaces it atomically.
- Keep `.kicraft/state.json` byte-for-byte unchanged during input review, drafting, questions, candidate review, corrections, and commit rejection.
- The default hard provider budget is $0.25 per draft. State that before the first provider call. Change it only on an explicit user request.

## Resume selection

At the start of a turn, read state and inspect `.kicraft/debug/<stage>.json` only for the current stage. Resume a pending artifact with status `needs_review` or `needs_input`. Otherwise choose the first incomplete stage in canonical order. Wiring is incomplete when `bom.connections` is empty. After accepting a stage, immediately prepare the next incomplete stage and present its input checkpoint in the same turn.

## State machine

### 1. Input checkpoint

Before spending provider budget:

1. Run side-effect-free `kicraft stage-prep <stage>`.
2. Read `../kicraft/stages/<stage>.md` relative to this `SKILL.md`.
3. Explain concisely:
   - reviewed upstream facts;
   - relevant `extras` supplied by stage prep;
   - decisions this stage is allowed to make;
   - decisions forbidden as premature or owned by another stage.
4. Request one focused confirmation or correction, then wait. Treat an unambiguous request to proceed, continue, or drive toward completion as confirmation; do not repeat an already confirmed checkpoint. Do not call `debug-draft` before that confirmation.

### 2. Draft

Only after the user confirms or corrects the input checkpoint:

1. Write the project brief to `/tmp/kicraft_debug_brief.txt`.
2. When guidance exists, write it verbatim to `/tmp/kicraft_debug_instruction.txt`.
3. When answering model questions, write a JSON list of `{text, answer}` objects to `/tmp/kicraft_debug_answers.json`.
4. Run:

```text
kicraft-stage-debug debug-draft --workspace . --stage <stage> \
  --brief-file /tmp/kicraft_debug_brief.txt \
  [--instruction-file /tmp/kicraft_debug_instruction.txt] \
  [--answers-file /tmp/kicraft_debug_answers.json] \
  --budget 0.25
```

5. Read `.kicraft/debug/<stage>.json`. Never infer candidate details from compact stdout.

If the command reports a provider/config failure, report the actual prerequisite or error. Never fall back to another LLM.

### Progress discipline

- Every user turn must advance the current stage through the next actionable state transition: prepare, draft, review, repair, redraft, or commit. Never answer with status or workflow explanation alone.
- Continue automatically through successful commands and pipeline repairs until reaching a required user review or decision boundary.
- At each boundary, state the reviewed decision in plain language and end with the exact next action the user can approve. Do not create extra pauses between a result and its review.
- Preserve constant oversight: never auto-accept a stage, skip a review facet, answer a material design question for the user, or spend provider budget before the input checkpoint is confirmed.


### 3. Guided candidate review

Do not dump prompt, raw response, tool trace, or full candidate JSON by default. Present one facet, include diagnostic `code` and `evidence` beside the affected decision, then stop and wait before the next facet.

Review facets:

- `intent`
  1. goal, explicit constraints, named parts;
  2. assumptions/defaults and form factor.
- `functional_spec`
  1. functional blocks and counts;
  2. flows and system boundaries.
- `architecture`
  1. topologies, rails, protocols;
  2. physical sheets, library/recipe use, replication boundaries;
  3. inter-sheet nets and programming strategy.
- `bom`: one sheet at a time. For each sheet review roles, ratings, stock/provenance, quantities, support parts, substitutions, and arrays.
- `wiring`: one sheet at a time. For each sheet review power, programming, feedback, decoupling, series nets, and no-connects; finish with whole-board pin/net coverage.

### 4. Pipeline gap repair loop

Treat unexpected omissions, invalid decisions, repeated questions, and misleading diagnostics as possible production-pipeline gaps, not merely candidate feedback.

1. Maintain `.kicraft/debug/findings.json` as the running list for the debug run. For each gap record the stage and facet, observed failure and diagnostic evidence, suspected root cause, proposed production patch, status (`observed`, `patching`, `validated`, or `unresolved`), and validation evidence. This file is diagnostic bookkeeping only; never copy its contents into `.kicraft/state.json`.
2. Decide whether the gap is candidate-specific or generalizable. Inspect the pending artifact trace and the production prompt, schema, validator, and stage code as needed. Explain the gap in plain language.
3. For a generalizable gap, patch the production source on the fly and add or update the smallest focused regression test that reproduces it. Never patch the pending candidate JSON.
4. Re-run that test, then run a fresh `debug-draft` for the same stage with the same brief, answers, budget, and accepted upstream state. Do not add an instruction that merely tells the provider to avoid the failure; the unchanged input is required to validate the pipeline patch.
5. Compare the replacement artifact with the recorded failure. Mark the finding `validated` only when the focused test passes and the original failure is absent from the fresh candidate. Otherwise keep it open, record the evidence, and continue diagnosis.
6. Resume guided review at the facet that exposed the gap and call out any other candidate facets changed by the redraft.

Candidate-specific preferences still use the feedback flow below. A pipeline repair does not grant commit permission and must leave `.kicraft/state.json` byte-for-byte unchanged.

At the end of the debug run, present the findings list with each patch and its validation result before handing control back to the ordinary `kicraft` skill.


### 5. Questions and feedback

If artifact status is `needs_input`, explain each question and its stage consequence, then wait. Put the user's answer in `--answers-file` and produce a fresh complete draft.

A wiring question with `reconcile_target: "bom"` is a visible BOM repair escalation, not an ordinary answer and not an automatic mutation. Explain the concrete missing BOM support, switch to the BOM input checkpoint, use the wiring question text as the BOM repair instruction, review and accept that BOM candidate explicitly, then return to a fresh wiring draft.

Any user correction requires a new complete `debug-draft` with `--instruction-file`; never patch the pending slot. Restart review at the changed facet and call out every other facet that changed. The words “continue” and “looks plausible”, and ordinary feedback, are not commit permission.

### 6. Acceptance

Only explicit `accept`, `approve`, or `commit this stage` permits a commit.

1. Write a factual one-paragraph summary of the accepted candidate to `/tmp/kicraft_debug_history.txt`.
2. Run:

```text
kicraft-stage-debug debug-commit --workspace . --stage <stage> \
  --history-message-file /tmp/kicraft_debug_history.txt
```

3. Report `invalidated_stages` from stdout.
4. Re-read `.kicraft/state.json` and verify the accepted stage is present.
5. If another stage remains, immediately run its input checkpoint and present it in the same turn. After wiring acceptance, proceed to the completion boundary.

If deterministic commit rejects the candidate, show the exact `errors` and `offenders`, connect them to the reviewed facet they invalidate, and wait for guidance. Never silently retry, auto-correct, or commit a replacement.

## Forensic requests

Reveal only the requested pending-artifact field:

- `show raw input` → `result.debug_context.prompt_state` and `extras`;
- `show exact prompt` → `result.debug_context.base_messages`;
- `show raw response` → `result.debug_context.raw_response`;
- `show tool trace` → artifact `events` filtered to tool/retry/serialization/reasoning events;
- `show candidate JSON` → `result.slot`.

Showing forensic data never changes state.

## Completion boundary

After wiring acceptance, report that all five LLM stages are committed. Do not synthesize or build in this skill. Hand control back to the ordinary `kicraft` skill for the deterministic build.
