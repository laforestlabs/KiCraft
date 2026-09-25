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

## Say it plainly

The owner reads every report. The pipeline's internal vocabulary is not an explanation.

- **Never** explain with an internal word: "facet", "slot", "candidate", "obligation",
  "component class", "semantic repair", "diagnostic", "lowerer", "recipe", "payload". Name
  the concrete thing instead -- the sentence in the brief, the part, the voltage, the pin,
  the connector, the line of the answer.
- Quote the machine's own words in backticks when they are the evidence, then explain them
  in the same sentence: ``the checker refused it -- `intent_quantity_subject_unbound`, "the
  count never says what it is counting"``. A quoted code is evidence, never the explanation.
- Every claim about what the machine did carries the line that shows it; every claim about
  money or saved state carries the figure.
- A worked example beats a rule. "Your brief asks for two JST-XH connectors; the answer does
  count two of them, but the parts library has no such part, so nothing can be placed" is a
  report. "The demanded class has no reviewed carrier" is not.
- Before sending, delete every sentence that would need a second sentence to be understood.
- If a review step must be named, name it by its subject in plain words: "the goal and the
  must-haves", "the guesses it made", "the parts on the power sheet" -- never "facet 2".

## Modes

Two ways to run. The owner's words choose; never assume.

- **Interactive** (the default): the owner is watching. Prepare each step, draft, show the review
  one piece at a time, and save a step only on an explicit "accept", "approve" or "commit".
- **Loop** (unattended): the owner asks for it -- "auto", "unattended", "run it in a loop",
  "drive it to a finished board", or a wrapper says so. Then:
  1. never wait for the owner, and never ask a question the pipeline's own default can answer
     (see *Default, then analyse*);
  2. run each step's review as a self-check against its checklist, and keep only what looks wrong
     or surprising;
  3. save a step as soon as its checks are clean and the deterministic save succeeds -- for this
     mode, launching it is the owner's acceptance;
  4. keep going through steps, repairs and redrafts until one of the documented stop conditions;
  5. finish with the end-of-run summary and the machine-readable last line.

Nothing else changes between modes: the same provider, the same caps, the same workspace rules,
the same plain-language reporting, and the same running record of findings.

## Default, then analyse

An unknown the design can absorb is never a reason to stop. Default it the way the live product
would, say so, and work out what the default costs -- in the same turn, without asking.

1. **Default it.** Use the default the live product applies: the stage contract's own rule, the
   pipeline's deterministic completion, or the conventional engineering value. Record it where
   the design records guesses -- the stage's `assumptions`, ending `(defaulted)`.
2. **Analyse that choice immediately**, in this shape:
   - what the default assumes, in one plain sentence;
   - what it would take for the default to be wrong (the number, the part, the load);
   - which later step breaks if it is wrong, and how loudly -- a refused step, a part the library
     cannot rate, or a board that builds and works but is bigger, hotter or costlier;
   - how to settle it cheaply later: one question to the owner, one datasheet, one measurement.
3. **Record it** in the run's findings file under `assumptions_taken`: stage, the assumption, the
   analysis, and a status (`taken`, `overridden`, `settled`). The end-of-run summary hands the
   owner the short list of guesses worth a second look.
4. **Stop and ask only** when the design cannot absorb the unknown: two stated facts contradict
   each other, or the choice is one a later step cannot undo (battery versus USB power, one board
   versus two, a shape the outline is cut from).

Worked example -- the shape wanted, from the owner's own brief of 2026-09-25 (a 5 V to 3.3 V
converter that never says what the 3.3 V output feeds):

- Wrong: stop and ask how much current the output needs.
- Right: default to the production reading -- a low-current logic rail that the board supplies --
  and record "The board supplies the 3.3 V output load; current unspecified, treated as under
  100 mA (defaulted)". Then analyse: at under 100 mA a single small linear regulator is legal and
  quiet; if the real load were 1 A, the parts step would pick a regulator the load cooks, the
  board could still build and pass every check, and the only giveaway is heat and a converter
  rated below its load. Settle it with one question at the end of the run, or by measuring the
  load. Record that, carry on, and put it in the end-of-run list.

## Keep the owner oriented

A run that works silently for an hour is a run the owner cannot trust. Say where things stand
often, in plain words, and never let a long stretch pass unaccounted for.

- **After every step change of state** -- a draft came back, a complaint was found, a repair was
  attempted, a step was saved, a build started or finished -- post one short note: what just
  happened, what it means, and what happens next.
- **Keep each note to a few lines**: the fact, the consequence, the next action. No walkthrough, no
  restatement of the plan, no internal vocabulary (see *Say it plainly*).
- **Say the direction, not just the event**: "the parts step refused the connector we invented, so I
  am adding a real reviewed part and will draft again" beats "draft failed".
- **In loop mode this matters more, not less.** Loops run unattended, so the notes are the only
  account of the run while it is happening: write one per step boundary and per repair into the run
  log as the run goes, keep the running record (`findings.json`, `assumptions_taken`) current, and
  still finish with the end-of-run summary. Never batch a whole run into one report at the end.
- **Flag trouble early**: the first sign of a failing step, an unexpected refusal, a cost that
  looks wrong, or a decision you had to guess gets its own note, before you work around it.

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
4. Request one focused confirmation or correction, then wait. Treat an unambiguous request to proceed, continue, or drive toward completion as confirmation; do not repeat an already confirmed checkpoint. Do not call `debug-draft` before that confirmation. In loop mode there is no wait: do steps 1-3 in the same turn and draft immediately -- launching the mode is the confirmation.

### 2. Draft

Only after the user confirms or corrects the input checkpoint (or immediately, in loop mode):

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
- At each boundary, state the reviewed decision in plain language and end with the exact next action the user can approve. Do not create extra pauses between a result and its review. In loop mode there is no boundary pause: keep the run moving, and let the stop conditions and the end-of-run summary do the reporting.
- Preserve constant oversight: never auto-accept a stage outside loop mode, skip a piece of the review, answer a material design question for the user outside *Default, then analyse*, or spend provider budget before the input checkpoint is done.


### 3. Guided review, one piece at a time

Do not dump prompt, raw response, tool trace, or full answer JSON by default. Walk the answer
in the pieces listed below, **one piece per turn**: say that piece in plain words, with the
exact lines that matter and the machine's own complaint beside them, then stop and wait before
the next piece.

Pieces of the review, in order:

- `intent`
  1. the goal, the must-haves it wrote down, and the parts it named;
  2. the guesses it made on the owner's behalf, and the board outline.
- `functional_spec`
  1. the functional blocks and their counts;
  2. the flows between them and where the system boundary sits.
- `architecture`
  1. the topologies, the supply rails, the protocols;
  2. the physical sheets, the reusable library blocks, the repeated groups;
  3. the nets that cross sheets and how the board gets programmed.
- `bom`: one sheet at a time. For each sheet: the parts and their jobs, the ratings, where the
  stock and sourcing facts come from, the quantities, the support parts, the substitutions, and
  any repeated arrays.
- `wiring`: one sheet at a time. For each sheet: power, programming, feedback, decoupling, the
  series components, and the pins deliberately left unconnected; finish with a whole-board
  check that every pin and net is covered.

In loop mode, walk every piece as a self-check: keep only what is wrong, missing or surprising,
and record it (the findings file, or `assumptions_taken`). Do not print the piece-by-piece
walkthrough or stop between pieces -- but do post the short progress note each piece boundary owes
the owner (see *Keep the owner oriented*), and carry everything the owner must act on into the
end-of-run summary.

### 4. Pipeline gap repair loop

Treat unexpected omissions, invalid decisions, repeated questions, and misleading diagnostics as possible production-pipeline gaps, not merely candidate feedback.

1. Maintain `.kicraft/debug/findings.json` as the running list for the debug run. For each gap record the stage, the piece of the answer it showed up in, the observed failure and diagnostic evidence, suspected root cause, proposed production patch, status (`observed`, `patching`, `validated`, or `unresolved`), and validation evidence. The same file carries `assumptions_taken`: every default the run took on the owner's behalf, with its analysis and status (see *Default, then analyse*). This file is diagnostic bookkeeping only; never copy its contents into `.kicraft/state.json`.
2. Decide whether the gap is candidate-specific or generalizable. Inspect the pending artifact trace and the production prompt, schema, validator, and stage code as needed. Explain the gap in plain language.
3. For a generalizable gap, patch the production source on the fly and add or update the smallest focused regression test that reproduces it. Never patch the pending candidate JSON.
4. Re-run that test, then run a fresh `debug-draft` for the same stage with the same brief, answers, budget, and accepted upstream state. Do not add an instruction that merely tells the provider to avoid the failure; the unchanged input is required to validate the pipeline patch.
5. Compare the replacement artifact with the recorded failure. Mark the finding `validated` only when the focused test passes and the original failure is absent from the fresh candidate. Otherwise keep it open, record the evidence, and continue diagnosis.
6. Resume the review at the piece that exposed the gap, and call out every other piece of the answer the redraft changed.

Candidate-specific preferences still use the feedback flow below. A pipeline repair does not grant commit permission and must leave `.kicraft/state.json` byte-for-byte unchanged.

At the end of the debug run, present the findings list with each patch and its validation result before handing control back to the ordinary `kicraft` skill.


### 5. Questions and feedback

If artifact status is `needs_input`, explain each question and its stage consequence, then wait. Put the user's answer in `--answers-file` and produce a fresh complete draft. In loop mode, answer it yourself wherever a documented default exists -- the stage contract's rule, the pipeline's completion, or the conventional value -- recorded through *Default, then analyse*, and keep going. Stop only for a question with no defensible default.

A wiring question with `reconcile_target: "bom"` is a visible BOM repair escalation, not an ordinary answer and not an automatic mutation. Explain the concrete missing BOM support, switch to the BOM input checkpoint, use the wiring question text as the BOM repair instruction, review and accept that BOM candidate explicitly, then return to a fresh wiring draft.

Any user correction requires a new complete `debug-draft` with `--instruction-file`; never patch the pending slot. Restart the review at the piece that changed and call out every other piece the redraft moved. The words “continue” and “looks plausible”, and ordinary feedback, are not commit permission.

### 6. Acceptance

Only explicit `accept`, `approve`, or `commit this stage` permits a commit. In loop mode, launching the loop is that acceptance: save as soon as the piece checks are clean and the deterministic save succeeds, report the result in one line, and prepare the next step immediately.

1. Write a factual one-paragraph summary of the accepted candidate to `/tmp/kicraft_debug_history.txt`.
2. Run:

```text
kicraft-stage-debug debug-commit --workspace . --stage <stage> \
  --history-message-file /tmp/kicraft_debug_history.txt
```

3. Report `invalidated_stages` from stdout.
4. Re-read `.kicraft/state.json` and verify the accepted stage is present.
5. If another stage remains, immediately run its input checkpoint and present it in the same turn. After wiring acceptance, proceed to the completion boundary.

If deterministic commit rejects the candidate, show the exact `errors` and `offenders`, connect them to the piece of the review they invalidate, and wait for guidance. Never silently retry, auto-correct, or commit a replacement.

## Running in a loop

For repeated unattended runs -- one board per brief, started by a wrapper, read afterwards by a
human:

- **One workspace per run.** A fresh directory per design
  (`~/.kicraft/debug/<what-it-is>-<seed>-<date>`), never a reused one, so any run can be read
  later without another run's history mixed into it.
- **Caps are hard.** The per-draft cap and the per-run cap are absolute. On reaching the run cap,
  stop cleanly, save nothing more, and report what is already saved. Never raise a cap to finish.
- **The only stop conditions** (anything else is a reason to default, repair and continue):
  1. a provider or configuration failure -- report the actual error and never fall back to
     another model;
  2. a save the deterministic gate refuses after two complete redrafts, with the exact complaints
     quoted;
  3. a repair escalation the run cannot justify -- a wiring question demanding a parts change the
     saved design gives it no basis for;
  4. the run cap.
- **Where the run writes itself down**, all inside the workspace: the per-step answer files under
  `.kicraft/debug/<stage>.json`, the running record of gaps in `.kicraft/debug/findings.json`
  (with `assumptions_taken` for every default), and the saved design in `.kicraft/state.json`.
  Nothing about the run may exist only in the chat.
- **End-of-run summary** (one screen, plain words): each step saved or stopped, the cost, the
  guesses taken with the ones worth a second look, the gaps found with their patch status, and
  the first thing a human should check.
- **Last line, machine-readable**, so a wrapper can iterate without parsing prose:

```text
{"run": "<workspace path>", "stem": "<board name>", "stages": {"intent": "committed", ...},
 "cost_usd": <n>, "assumptions_taken": <n>, "findings": {"validated": <n>, "open": <n>},
 "stopped": "<reason, or null>"}
```

- **Exit status**: 0 only when every step is saved; nonzero when the run stopped early. A stopped
  run is never reported as finished.
- **The build is not this skill's job** (see the completion boundary). A loop wrapper chains it
  after a fully saved run: `kicraft build .kicraft/state.json generated --quality good`, and reads
  that command's exit status as the run's final verdict.

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
