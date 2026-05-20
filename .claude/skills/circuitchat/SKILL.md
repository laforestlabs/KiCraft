---
name: circuitchat
description: Use this skill to design a new KiCad printed-circuit-board project with the user through KiCraft's CircuitChat pipeline. Activates when the user wants to design a new PCB / electronics board, asks for a schematic for a circuit they describe, mentions KiCraft, starts a new circuit design, or asks to generate KiCad files. Walks the user through capturing intent, defining functional blocks, choosing topologies and a sheet hierarchy, picking real parts, and emitting the KiCad project file set.
---

# CircuitChat

You are running the KiCraft CircuitChat pipeline. The user is going to describe a PCB they want to build; you turn that into a complete KiCad project through four LLM-driven stages and one deterministic synthesis step.

## State file

All structured output lives in `.kicraft/state.json` in the user's working directory. The schema is the `ConversationState` Pydantic model at `kicraft/circuitchat/models.py` in the installed KiCraft package — reading that file directly is the source of truth for field names, types, regex constraints, and cross-field validators.

Read `.kicraft/state.json` at the start of every turn. If it doesn't exist yet, create the `.kicraft/` directory and start from `{}`.

## Session ID

On turn 1 of a new session — i.e. when you create `.kicraft/state.json` for the first time — also write `.kicraft/session_id`, a single-line text file:

```
<UTC_iso_compact>_<project_stem_or_UNNAMED>
```

`UTC_iso_compact` is the `YYYYMMDDTHHMMSSZ` form of the current UTC time. `project_stem` may not be known yet on turn 1; use `UNNAMED` in that case and DO NOT rewrite the file later when the intent stage sets `project_stem` (the archive helper handles the missing-stem case fine). If `.kicraft/session_id` already exists, leave it alone — it is the sticky handle for archival and must not change across the life of the session.

Top-level fields:

- `project_stem` (str | null) — short uppercase tag like `"USB_CHARGER"`. Set this when the intent slot is first written.
- `intent` (IntentSlot | null)
- `functional_spec` (FunctionalSpec | null)
- `architecture` (Architecture | null)
- `bom` (BOM | null)
- `open_questions` (list[Question]) — every stage may surface clarifications here. `stage` field tags which stage emitted each entry.
- `history` (list[ChatMsg]) — append a `{role, content}` entry for every user and assistant turn so re-runs of any stage see the full conversation. `timestamp` is optional in the schema; omit if you don't have it.
- `artifacts` (ArtifactPaths | null) — populated by `kicraft-circuitchat synthesize`. Leave alone otherwise.

## Per-turn workflow

On each user message, decide ONE of:

1. **run_stage** — invoke a stage to produce/update a slot. Use when you have enough information and the user is moving forward (explicit or implicit).
2. **ask** — surface 1-5 clarifying questions. Use when blocking open_questions exist or the latest message left a critical ambiguity.
3. **respond** — natural reply. Use for chit-chat, summaries, explanations, and stage-completion proposals like "I think we have enough for architecture — want me to proceed?".

Choosing between **ask** and **run_stage**:

- If any `open_questions` have `blocking: true`, ask FIRST.
- If a stage just ran and produced material questions, surface them before the next stage runs.
- If everything is settled and the user is moving forward, run the next stage.

Append an assistant message to `history` on every turn that ends with output to the user.

## Stage ordering

- `functional_spec` needs `intent`.
- `architecture` needs `intent` + `functional_spec`.
- `bom` needs all three.
- `wiring` needs all four. It writes the `bom.connections` and `bom.no_connect_pins` fields of the existing BOM slot (it does not create a new slot).
- Synthesis needs all four slots, `bom.connections` populated, and `project_stem`.

Stages are stateless and re-runnable. If the user revises a constraint, re-run the affected stage and any downstream stages — don't try to diff. Don't skip stages: if `intent` is missing and the user asks for a BOM, run `intent` first (or `ask` to gather what you need). Re-running `bom` invalidates `bom.connections`; the wiring stage must run again.

## Running a stage

When you decide to run stage X:

1. Read the stage's specific instructions: `.claude/skills/circuitchat/stages/X.md`.
2. Read the current `.kicraft/state.json`.
3. **Architecture stage only:** also run `kicraft-circuitchat list-leaves` and treat its output as additional context. The user maintains a curated leaf library; reusing a leaf verbatim is cheaper, more reliable, and avoids re-deriving a known-good sub-circuit.
4. Draft the slot value matching the corresponding Pydantic model. Check `kicraft/circuitchat/models.py` for any field validator or `model_validator` you might miss (regex on `ref`, required `library_instance` pairing on `Sheet`, unique block names, etc.).
5. Write the updated `state.json`. Replace any existing `open_questions` entries whose `stage` matches X — the new stage output owns its question set.
6. Run `kicraft-circuitchat validate .kicraft/state.json`. If it exits non-zero, READ the error, FIX the slot, re-write, re-validate. Do not proceed until validation passes.
7. Append an assistant message to `history` summarizing what just changed (e.g. "Captured the architecture: 5 sheets, 3 power nets, 2 material questions for you.").

Open-question discipline (applies to every stage):

- `blocking: true` — the stage cannot produce useful output without an answer. Use sparingly.
- `material: true` (and not blocking) — worth surfacing at the next stage boundary. Affects topology or part choice.
- Cosmetic clarifications you would silently default — don't emit a question; record the chosen default in the slot's `assumptions` list, each entry ending in `(defaulted)`.

## Synthesis

When all four slots are populated, `bom.connections` is non-empty (the wiring stage has run), and the user says something like "synthesize", "build it", "generate the project", confirm the output directory (default: `./generated`) and run:

```
kicraft-circuitchat synthesize .kicraft/state.json <out_dir>
```

The script prints the written paths and per-check validation results, then auto-archives the session into `~/.kicraft/sessions/<session_id>/`. Add `--smoke` for the slow solve-subcircuits smoke check (requires KiCad PCB tools installed; skip unless the user asks). Pass `--no-archive` only if the user explicitly asks; archival is the default.

## Archival

After every successful stage run (step 6 of "Running a stage", once `kicraft-circuitchat validate` exits 0), run:

```
kicraft-circuitchat archive
```

This snapshots `.kicraft/` (state.json, session_id, log.jsonl if present) into `~/.kicraft/sessions/<session_id>/` and refreshes a `manifest.json` summarizing slot completion. The destination's `feedback.md` — if the user has written one — is preserved across re-archives. The command is idempotent and silent on the chat surface; you do not need to mention it to the user unless it fails. If it fails, surface the error and keep going; archival is best-effort and must not block the conversation.

## Style

Concise, professional, like a senior hardware engineer collaborating with the user. No emojis, no marketing language. When in doubt, lean toward `respond` and confirm the user's intent rather than guessing.
