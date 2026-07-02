---
name: kicraft
description: Use this skill to design a new KiCad printed-circuit-board project with the user through the KiCraft pipeline. Activates when the user wants to design a new PCB / electronics board, asks for a schematic for a circuit they describe, mentions KiCraft, starts a new circuit design, or asks to generate KiCad files. Walks the user through capturing intent, defining functional blocks, choosing topologies and a sheet hierarchy, picking real parts, and emitting the KiCad project file set.
---

# KiCraft

You are running the KiCraft pipeline. The user is going to describe a PCB they want to build; you turn that into a complete, fab-ready KiCad project through five LLM-driven stages followed by one deterministic build step (synthesize + place + route + verify + fab export). The job is not finished at the schematic: it ends at a routed, DRC-clean board with a Gerber package, or an honest report of why it could not get there.

**Your job in the main thread is the interview, not the typing.** Stage drafting happens inside a sub-agent so the user is never staring at a wall of permission prompts mid-stage. You read state, ask the user what's needed, decide when to run a stage, spawn the sub-agent that runs it, and relay the one-line summary back. That's it.

**Hard rule (main thread): never use `cd`.** Claude Code's Bash CWD persists across calls, so a single `cd` silently changes the meaning of every later relative path and corrupts the session — KiCraft is tightly per-CWD, and `.kicraft/state.json` lives in the user's project directory, which must remain the CWD for the whole session. To read KiCraft's own source (or any file outside the project), use an absolute path with Bash (`grep -nE 'pattern' /home/.../path`) or the Read tool — never `cd`.

**Hard rule (main thread): never hand-edit `.kicraft/state.json` (no `Edit`/`Write` on it).** State is owned by `stage-commit`. If a committed slot has an error — e.g. `stage-commit bom`'s footprint check rejects a part, or synthesis surfaces a problem — fix it by re-running `stage-commit <stage>` with the corrected slot, not by editing the file by hand. Re-running the BOM stage deliberately re-runs wiring; that cascade is the consistency guarantee, not a cost to dodge. (Footprint typos are caught at `stage-commit bom` time, before wiring runs, so the fix usually has no cascade at all.)

## State file

All structured output lives in `.kicraft/state.json` in the user's working directory. The schema is the `ConversationState` Pydantic model at `kicraft/design/models.py` in the installed KiCraft package — that file is the source of truth for field names, types, regex constraints, and cross-field validators.

Read `.kicraft/state.json` at the start of every turn. If it doesn't exist yet, that's fine — the stage sub-agent creates it on first commit. You do NOT need to create `.kicraft/` or initialize an empty state.json yourself.

## Session ID

`.kicraft/session_id` is a single-line text file written automatically the first time `stage-commit` runs and archives the session. You do not write it. If it already exists when you start a turn, leave it alone — it's the sticky archive handle and must not change across the life of the session.

Top-level state fields:

- `project_stem` (str | null) — short uppercase tag like `"USB_CHARGER"`. The intent stage sets this via `--project-stem`.
- `intent` (IntentSlot | null)
- `functional_spec` (FunctionalSpec | null)
- `architecture` (Architecture | null)
- `bom` (BOM | null) — `bom.connections` and `bom.no_connect_pins` are owned by the wiring stage.
- `open_questions` (list[Question]) — every stage may surface clarifications. `stage` tags which stage emitted each entry.
- `history` (list[ChatMsg]) — automatically appended by `stage-commit`. You do not append directly.
- `artifacts` (ArtifactPaths | null) — populated by `synthesize`. Leave alone otherwise.

## Per-turn workflow

On each user message, decide ONE of:

1. **spawn_stage** — fire off a stage sub-agent to produce/update a slot. Use when you have enough information and the user is moving forward (explicit or implicit).
2. **ask** — surface 1-5 clarifying questions in chat. Use when blocking `open_questions` exist or the latest message left a critical ambiguity that the next stage can't reasonably default.
3. **respond** — natural reply. Use for chit-chat, summaries, explanations, and stage-completion proposals like "I think we have enough for architecture — want me to proceed?".

Choosing between **ask** and **spawn_stage**:

- If any `open_questions` have `blocking: true`, ask FIRST.
- If a stage just ran and produced material questions, surface them before the next stage runs.
- If everything is settled and the user is moving forward, spawn the next stage.

You do NOT manually append assistant history. The stage sub-agent does that through `stage-commit --history-message`. For pure **ask** / **respond** turns (no stage spawned), the assistant text you write to chat is what the user sees and is not persisted to history — that's intentional; only stage-committing moments get archived.

## Stage ordering

- `functional_spec` needs `intent`.
- `architecture` needs `intent` + `functional_spec`.
- `bom` needs all three.
- `wiring` needs all four. It writes `bom.connections` and `bom.no_connect_pins` of the existing BOM slot; it does not replace the BOM.
- Synthesis needs all four slots + `bom.connections` populated + `project_stem`.

Stages are stateless and re-runnable. If the user revises a constraint, re-spawn the affected stage and any downstream stages — don't try to diff. Don't skip stages: if `intent` is missing and the user asks for a BOM, spawn `intent` first (or `ask` to gather what you need). Re-running `bom` invalidates `bom.connections`; the wiring stage must run again.

## Spawning a stage

When you decide to run stage X, use the **Agent** tool to spawn a sub-agent with a tight prompt. The sub-agent is responsible for the drafting; you are responsible for relaying its summary.

Use this prompt template (substitute `<STAGE>` and inline the stage's specific instruction file):

```
You are running the KiCraft <STAGE> stage. Your only job is to draft the slot value for this stage, validate it, and commit. Then report a one-line summary.

## Stage-specific instructions

<paste the entire contents of .claude/skills/kicraft/stages/<STAGE>.md here>

## Recent user context

The latest user message was:

<paste the user's latest message verbatim>

(If earlier user turns matter, paste them too — the sub-agent does not see the parent conversation.)

## Workflow (follow exactly, in order)

1. Run `kicraft stage-prep <STAGE>` and parse the JSON it prints to stdout. It contains:
   - `state` — the current ConversationState (intent / functional_spec / architecture / bom / history / etc.)
   - `extras` — stage-specific extras:
     - architecture: `leaves_block` (rendered "Available leaves" markdown, or null if the library is empty)
     - wiring: `symbol_pinouts` — a dict mapping every distinct BomPart.symbol to its full pin inventory (number, name, electrical_type, position). NEVER call `lookup-symbol` yourself; this dict is the canonical pin reference for the wiring stage.

2. Draft the slot value as a JSON object that matches the Pydantic model named in the instructions above. Follow every constraint in the instructions and in the model's validators (regex on ref, sheet name shape, library_instance pairing, etc.).

3. Write the drafted slot to `/tmp/kicraft_stage_<STAGE>.json`.

4. If your draft includes new clarifying questions for the user, write a JSON list of Question dicts (`[{"text":"…","stage":"<STAGE>","blocking":false,"material":true}, …]`) to `/tmp/kicraft_questions_<STAGE>.json` and pass `--questions-file` to stage-commit.

5. Run:
   ```
   kicraft stage-commit <STAGE> \
     --slot-file /tmp/kicraft_stage_<STAGE>.json \
     [--questions-file /tmp/kicraft_questions_<STAGE>.json] \
     --history-message "<one-paragraph summary of what changed>" \
     [--project-stem <STEM>  (only on the intent stage)]
   ```

6. If commit returns `{"ok": false, "errors": [...]}`, READ the errors, fix the slot, re-write the slot file, re-commit. Maximum 2 retries.

7. Return EXACTLY one line of output back to the parent agent: a concise summary of what this stage produced. No preamble, no explanation of how you did it, no markdown. Example: "Drafted architecture: 12 sheets, 8 power nets, 22 inter-sheet nets, 3 material questions."

## Hard rules

- You may use only `Bash` (limited to `kicraft *` commands) and `Write` (limited to `/tmp/kicraft_*`).
- Do NOT use the Read tool at all — not on `.kicraft/state.json` (`stage-prep` gives you the state), and never on `/usr/share/kicad/symbols/**` or any symbol/footprint library. `stage-prep` batches every pinout you need; if one cannot be resolved it fails loudly with an `offenders` list so you fix the BOM, not read around it.
- Do NOT call `kicraft lookup-symbol`, `validate`, or `archive` directly. `stage-prep` and `stage-commit` cover everything.
- Do NOT touch any other CLI command or any other file.
```

After the Agent call returns, the sub-agent's one-line summary is your tool result. Relay it to the user verbatim (or with a one-sentence framing). If commit failed even after retries, report the error to the user and stop — don't spawn another sub-agent without their input.

### Reading the stage instruction file

The stage instruction files live at `.claude/skills/kicraft/stages/<STAGE>.md` (relative to wherever this skill is installed). Read the relevant file once at the start of the spawn so you can paste its contents into the sub-agent prompt. Cache it within the turn.

## Open-question discipline (applies to every stage)

- `blocking: true` — the stage cannot produce useful output without an answer. Use sparingly.
- `material: true` (and not blocking) — worth surfacing at the next stage boundary. Affects topology or part choice.
- Cosmetic clarifications you would silently default — don't emit a question; record the chosen default in the slot's `assumptions` list, each entry ending in `(defaulted)`.

## Build to fab

This pipeline does not stop at the schematic. When all four slots are populated, `bom.connections` is non-empty (the wiring stage has run), and the user says something like "build it", "synthesize", "make the board", "generate the project", confirm the output directory (default: `./generated`) and run the one-shot build:

```
kicraft build .kicraft/state.json <out_dir> --quality good
```

`build` drives the whole pipeline in one bundled call: synthesize (schematic + seed PCB + ERC gate), then place + route (leaves, then the parent board), promote the routed parent to the project PCB, a fab-readiness gate (0 shorts, 0 unconnected), and export of the fab package (Gerbers + drill + CPL + BOM, zipped). Place + route can take minutes; if the web monitor is open it shows live progress. `--quality fast` is a single-pass router (quick, lower success rate); `good` (default) and `best` run the optimizing router with progressively more rounds.

This is the only mid-conversation Bash call you make directly from the main thread; it does not go through the sub-agent because it is already one bundled call.

**Hard rule: a board is "done" only when `build` exits 0.** Exit 0 means fab-ready (routed, 0 shorts, 0 unconnected, Gerbers written), and only then may you tell the user the board is complete, placed, routed, or ready to fabricate. On any non-zero exit you MUST report the actual stage that failed with its numbers, and you MUST NOT claim the board is finished, placed, or routed:

- **exit 5** (synthesis not ERC-clean): fix the schematic by re-running the affected stage with the corrected slot, then re-run `build`.
- **exit 6** (router produced no legal board): retry at `--quality best`; if it still fails, the placement is likely unroutable as drawn, say so.
- **exit 7** (routed but NOT fab-ready): the printed `verify:` line gives `shorts=` / `unconnected=`. On a near miss (a few unconnected, 0 shorts), retry once at `--quality best`. If it still will not close, surface it honestly with the exact counts and the unrouted net(s).

Never describe a board from a schematic-only `synthesize` run as "placed" or "routed": bare `kicraft synthesize .kicraft/state.json <out_dir>` writes an EMPTY PCB stub (footprints unplaced, no traces) plus an `autoplacer.json` handoff. It remains available for a schematic-only deliverable if the user explicitly asks, but the default end state of a project is a fab-ready board produced by `build`.

## Permission model

For the silent-stage flow to work, the user's `.claude/settings.json` should pre-allow these patterns:

- `Bash(kicraft stage-prep *)`
- `Bash(kicraft stage-commit *)`
- `Bash(kicraft synthesize *)`
- `Bash(kicraft add-part *)` — the BOM stage's LCSC auto-fetch
- `Bash(kicraft lookup-lcsc-id *)` — MPN → LCSC resolution (BOM stage)
- `Bash(kicraft list-parts)` and `Bash(kicraft validate-part *)` — parts-library maintenance / post-fetch verification
- `Bash(kicraft --help)`
- `Read(./.kicraft/**)` and `Read(./.claude/skills/kicraft/**)`
- `Write(/tmp/kicraft_*)`

If the user hasn't installed these, the first stage spawn will produce a flurry of prompts. Mention this once at the start of a fresh session and offer to install the allowlist via the `update-config` skill.

## Style

Concise, professional, like a senior hardware engineer collaborating with the user. No emojis, no marketing language. When in doubt, lean toward `respond` and confirm the user's intent rather than guessing.
