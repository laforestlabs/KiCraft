---
name: kicraft
description: Design a new KiCad printed-circuit-board project through KiCraft's five design stages and deterministic build. Use when the user wants to design a PCB, create a schematic or KiCad project, or continue an existing KiCraft design. Do not use for explicit stage-driver debugging; use kicraft-debug instead.
compatibility: Requires the KiCraft design CLI, Python 3.12, and an agent capable of reading files, writing temporary files, and running local commands.
---

# KiCraft

Turn the user's PCB brief into a fab-ready KiCad project through five LLM-authored design stages followed by the deterministic KiCraft build. The active LLM performs the interview and stage drafting; KiCraft owns schema validation, state mutation, synthesis, placement, routing, verification, and fab export.

## Invariants

- Keep the project workspace as the current working directory. Never change it with `cd`.
- Read `.kicraft/state.json` at the start of every turn. An absent file means no stage has been committed.
- Never create or hand-edit `.kicraft/state.json`. Only `kicraft stage-commit` may mutate design state.
- Process one stage at a time in canonical order: `intent`, `functional_spec`, `architecture`, `bom`, `wiring`.
- Read the current stage contract from this skill's `stages/<stage>.md`. Those files are the canonical electrical and schema instructions.
- Do not read KiCad symbol or footprint library files directly. `stage-prep` supplies bounded reference data and fails explicitly when a required symbol cannot be resolved.
- A stage correction is a complete replacement slot committed through the same gate, never a direct state patch.

## State ownership

Structured design data lives in `.kicraft/state.json` and validates as `ConversationState` from `kicraft/design/models.py`.

- `project_stem`: uppercase project identifier set with intent.
- `intent`: goals, explicit constraints, named parts, expertise, defaults, form factor.
- `functional_spec`: abstract functional blocks and flows.
- `architecture`: topologies, rails, protocols, sheets, and inter-sheet nets.
- `bom`: real parts and physical organization.
- `bom.connections` and `bom.no_connect_pins`: owned by wiring.
- `open_questions`: stage-tagged clarifications.
- `history`: appended by `stage-commit`; never edit it directly.
- `artifacts`: produced by deterministic build steps.

## Turn workflow

Choose exactly one action per user turn:

1. **Ask**: surface blocking or material questions when the next stage cannot safely default them.
2. **Draft and commit one stage**: when its inputs are sufficient and the user wants to proceed.
3. **Respond**: explain state, summarize decisions, or discuss a proposed next step without mutating state.
4. **Build**: only after all five stages are complete and the user asks to generate the board.

Do not commit a later stage while an earlier stage is incomplete. A re-committed upstream stage invalidates its downstream stages; regenerate them in canonical order.

## Draft and commit a stage

For stage `<stage>`:

1. Run:

   ```text
   kicraft stage-prep <stage>
   ```

   Parse its `state` and `extras`. Relevant extras include architecture library leaves, BOM part/catalog data, and wiring symbol pinouts.

2. Read `stages/<stage>.md` relative to this `SKILL.md`.

3. Draft one complete slot JSON object using:
   - the user's brief and current-turn corrections;
   - committed upstream state;
   - `stage-prep` extras;
   - the stage contract and model validators.

4. Write the slot to `/tmp/kicraft_stage_<stage>.json`.

5. If the stage needs user-visible clarifications, write a JSON list of Question objects to `/tmp/kicraft_questions_<stage>.json`. Use blocking questions sparingly. Cosmetic defaults belong in the slot's assumptions and end with `(defaulted)`.

6. Commit:

   ```text
   kicraft stage-commit <stage> \
     --slot-file /tmp/kicraft_stage_<stage>.json \
     [--questions-file /tmp/kicraft_questions_<stage>.json] \
     --history-message "<factual one-paragraph summary>" \
     [--project-stem <STEM>]
   ```

   `--project-stem` is intent-only.

7. If commit returns `ok: false`, read its exact `errors` and `offenders`, correct the complete slot, and retry at most twice. If it still fails, report the gate evidence and wait for user guidance.

8. Re-read `.kicraft/state.json`, report the committed result concisely, and stop before drafting the next stage.

## Question discipline

- `blocking: true`: no useful candidate can be produced without an answer.
- `material: true`, non-blocking: worth surfacing at the next boundary because it affects topology, part choice, or manufacturability.
- Minor choices: choose a safe default and record it in `assumptions` with `(defaulted)`.

If `open_questions` contains a blocking question for the current stage, ask it before drafting. Feed the user's answer into the next complete stage replacement; do not edit the question or state directly.

## Build to fab

When intent, functional spec, architecture, BOM, and wiring are complete and the user requests the board, run:

```text
kicraft build .kicraft/state.json <out_dir> --quality good
```

Default `<out_dir>` to `./generated` unless the user chose another path. `good` is the normal quality; use `fast`, `draft`, or `best` only when requested or when a documented retry calls for it.

A board is complete only when `build` exits 0. Exit 0 means routed, zero shorts, zero unconnected items, and fab outputs written.

- Exit 5: synthesis/ERC failure. Correct the owning design stage, then rebuild.
- Exit 6: no legal routed board. Retry once at `--quality best`; if it repeats, report the placement/routing failure.
- Exit 7: routed but not fab-ready. Report exact shorts and unconnected counts. A near miss with zero shorts may be retried once at `--quality best`.

Never describe a schematic-only `synthesize` output as placed, routed, or fab-ready.

## Portable tool requirements

This skill assumes only generic agent capabilities:

- read repository and project files;
- write `/tmp/kicraft_*` files;
- execute `kicraft stage-prep`, `kicraft stage-commit`, and `kicraft build`;
- present questions and command results to the user.

It does not require slash commands, a vendor-specific delegation tool, vendor-specific permission syntax, or a particular LLM provider.
