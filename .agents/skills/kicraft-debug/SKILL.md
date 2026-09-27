---
name: kicraft-debug
description: Inspect, debug, or walk through KiCraft's real provider-backed LLM design stages with explicit review before commit, and run the unattended live cohort (five `Surprise me` briefs through kicraft.io, repair the pipeline gaps they expose, then rerun all five so a fix is tested against every brief and not only the one it targets). Activate only when the user explicitly asks to debug, inspect, review, or step through KiCraft LLM stage decisions, or to drive the live cohort; ordinary PCB design requests use the kicraft skill.
compatibility: Requires the KiCraft server and design extras, OpenRouter configuration, and an agent capable of reading files, writing temporary files, and running local commands. Cohort mode additionally needs the production box (its live accounts database, spend ledger and web log), a browser session that can sign in to https://kicraft.io, and permission to restart the production services and spend provider budget.
---

# KiCraft stage debugger

Use the production KiCraft stage driver, paused immediately before durable commit. Guide the user through one stage candidate at a time. Do not emulate a stage, draft slot JSON yourself, or substitute the active agent's own model for the configured production provider.

## Invariants

These invariants govern the workspace modes (interactive and loop). Cohort mode has no project
workspace and never drafts a stage itself; see *Running the live cohort* for its own rules.

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

Three ways to run. The owner's words choose; never assume. Naming the live product -- "kicraft.io",
"Surprise me", "the live cohort", "five briefs", "run them, fix it, run them again" -- means **cohort**
mode. Naming a stage, a project, or a workspace means **interactive** or **loop**.

- **Interactive** (the default): the owner is watching. Prepare each step, draft, show the review
  one piece at a time, and save a step only on an explicit "accept", "approve" or "commit".
- **Loop** (unattended, one design in a workspace): the owner asks for it -- "auto", "unattended", "run
  it in a loop", "drive it to a finished board", or a wrapper says so. Then:
  1. never wait for the owner, and never ask a question the pipeline's own default can answer
     (see *Default, then analyse*);
  2. run each step's review as a self-check against its checklist, and keep only what looks wrong
     or surprising;
  3. save a step as soon as its checks are clean and the deterministic save succeeds -- for this
     mode, launching it is the owner's acceptance;
  4. keep going through steps, repairs and redrafts until one of the documented stop conditions;
  5. finish with the end-of-run summary and the machine-readable last line.
- **Cohort** (unattended, five live briefs through the real product): run five `Surprise me` briefs
  live on kicraft.io, repair the pipeline gaps they expose at the source, then rerun all five -- the
  other four briefs are the guard against a fix that trades one board for another. See *Running the
  live cohort*. Cohort mode is the one mode that uses no project workspace and never drafts with
  `kicraft-stage-debug`; the workspace invariants at the top scope to the other two.

Nothing else changes between modes: the same provider, the same caps, the same plain-language
reporting, and the same running record of findings.

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
- **In loop mode this matters more, not less** -- and most of all in cohort mode, where five runs are
  in flight and the owner sees only what you write. Loops run unattended, so the notes are the only
  account of the run while it is happening: write one per step boundary and per repair into the run
  log as the run goes, keep the running record (`findings.json`, `assumptions_taken`) current, and
  still finish with the end-of-run summary. Never batch a whole run into one report at the end.
- **Flag trouble early**: the first sign of a failing step, an unexpected refusal, a cost that
  looks wrong, or a decision you had to guess gets its own note, before you work around it.

## Session change totals (all modes)

Every debug session MUST end with a summary containing
**`Session changes: +N lines added, -M lines removed.`**
This applies to interactive, loop, and cohort modes, including failed, stopped,
and read-only sessions. Report `+0` and `-0` when no changes remain.

- Before the first edit, preserve the starting contents of each file you touch and
  record which files are new. Keep this baseline across commits and session resumes.
- Measure the final diff against that baseline using `git diff --numstat` or an
  equivalent content comparison. Sum additions and removals separately across the
  session's source, tests, documentation, and other maintained project files.
- Include both committed and uncommitted session changes, new text files, and
  deleted text files. Exclude pre-existing or unrelated edits by others, generated
  board/run artifacts, and temporary probes. Report binary changes separately.
- These are start-to-finish counts, not editing churn: reverted edits contribute
  zero. Do not substitute a diff against the current `HEAD` or estimate the totals.
- Include `"changes": {"lines_added": N, "lines_removed": M}` in every
  machine-readable end summary as well.

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

## Running one design in a loop

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
  the first thing a human should check, and the session's lines added and removed
  (see *Session change totals*).
- **Last line, machine-readable**, so a wrapper can iterate without parsing prose:

```text
{"run": "<workspace path>", "stem": "<board name>", "stages": {"intent": "committed", ...},
 "cost_usd": <n>, "assumptions_taken": <n>, "findings": {"validated": <n>, "open": <n>},
 "changes": {"lines_added": <n>, "lines_removed": <n>},
 "stopped": "<reason, or null>"}
```

- **Exit status**: 0 only when every step is saved; nonzero when the run stopped early. A stopped
  run is never reported as finished.
- **The build is not this skill's job** (see the completion boundary). A loop wrapper chains it
  after a fully saved run: `kicraft build .kicraft/state.json generated --quality good`, and reads
  that command's exit status as the run's final verdict.

## Running the live cohort: five briefs, then fix, then rerun all five

The owner's words for this mode are the product's own -- "run five `Surprise me` briefs live on
kicraft.io", "auto through the site", "run them, fix it, run them again". Launching it is the
acceptance for what it does, exactly as launching loop mode is; everything else in this file still
applies except where this section says otherwise.

**Why five, and why all five again.** One brief can only show that a fix helps the case it targets.
The other four are the guard: a change that repairs one brief and breaks another is an unintended
consequence, and the cheapest moment to see it is the pass that introduced it. A fix is therefore
never validated on its own brief -- it is validated on the whole cohort, in the same pass, and a fix
that trades one brief for another is not accepted.

**"Live" is not negotiable.** The five briefs go through the public product: a browser session on
`https://kicraft.io`, the composer’s `Design` handler, the running web process and build worker, the
real provider budget and the real per-run cap. Generate and diversity-check the briefs before
launching them, as described below; `Surprise me` starts spending immediately and cannot preflight
a cohort. Never substitute `kicraft-stage-debug`, `kicraft.server.stage_driver run`, or the self-eval
batch for a pass: those tools are for diagnosis *between* passes, never for the passes themselves.

### Cohort 1. Surfaces and account

- Site: `https://kicraft.io` (Caddy in front of NiceGUI on `127.0.0.1:8080`). Read and drive it in a
  browser tab. A frozen or closed tab does not stop a run -- reopen it from the project page.
- Account: the drive account `ui-drive@kicraft-test.dev` (id 44, Max tier, verified) is a real row in
  `~/.kicraft/accounts.db`, so the cohort spends and meters the way a user's does. Keep its credential
  in `~/.kicraft/debug/live-cred.json` (mode 600, never in the repo); a session that created it may have
  left a copy at `/tmp/ui-cred.json`, which is readable as `{"email": ..., "pw": ...}`. If neither
  exists, set a fresh password with `kicraft-accounts reset-password ui-drive@kicraft-test.dev --set`
  and store it. The shared site password, if a fresh browser profile asks for it, is
  `KICRAFT_ACCESS_PASSWORD` in `.env`.
- Quota: Max is 20 designs per trailing 7 days, and only `running`, `ok` and `awaiting_input` rows
  consume a slot -- a `failed` run frees its slot. Five live briefs per pass is therefore cheap in
  quota; check the account's quota before the first pass and before every rerun, and never start a
  pass that cannot finish inside the window.
- Where each run writes itself down: `~/.kicraft/projects/<user_id>/<project_id>/.kicraft/state.json`
  (`stage_status[].diagnostics` carries the gate sentence), `~/.kicraft/accounts.db` (`projects` for
  status and cost, `build_jobs` for the build), `~/.kicraft/spend_ledger.db` (`stage_runs`, keyed by
  `run_id` `p<project_id>-…`, for per-stage attempts, wall time and cost), and `logs/kicraft_web.log`
  (`[surprise] seed=N brief='…'` on every click, plus one line per stage).

### Cohort 2. A pass: five diverse briefs, started through the live composer

1. **Select before spending.** `generate_brief(seed)` in `kicraft.server.examples` chooses a family
   independently for each seed, with replacement. Different seeds guarantee neither different
   families nor different text. Five consecutive `Surprise me` clicks are NOT a diverse cohort.
   Generate candidates without starting designs, using the product's generator unchanged. Recover
   the family from the generator's current selection logic (currently the first
   `random.Random(seed).choice(BRIEF_TEMPLATES)`); do not infer it from keyword regexes.
   - Freeze five distinct families, at most one of each: controller, sensors, display_led, power,
     analog, actuator. Take the first eligible candidate for each new family in seed order.
   - Also inspect the actual functions: reject near-duplicates differing only in output count,
     supply, MCU, connector, size, or stack-up. Different family labels alone are not sufficient.
   - Reject text already used by this drive account or a previous/discarded cohort. Use a fresh
     recorded seed range; do not reset or edit the product's persistent seed counter.
   - Do not filter for expected success, supported parts, simplicity, or convenient debugging.
     Diversity is the selection criterion, not a way to remove difficult briefs.
   - Save every considered seed, family, verbatim brief and acceptance/rejection reason in
     `selection.json`, and show the five chosen functions before launching. Freeze the accepted
     seed/text/family list in `cohort.json`. A replacement cohort starts its pass count at one;
     retain the discarded cohort's results and costs separately.
2. Sign in at `https://kicraft.io/login`; wait for the composer to be live. For each frozen brief:
   - use `New design` to detach from the previous run without stopping it;
   - replace the composer text with the exact generated brief; assert the DOM value equals it
     before pressing `Design` (a fill that appends silently changes the experiment);
   - press `Design` and record its board code and project id. This is the same live start handler
     that `Surprise me` invokes after generating its text.
   Start all five without waiting for their designs to finish; the build queue owns concurrency.
3. Assert exactly five new project rows, each with the corresponding frozen brief verbatim.
   A lost click is not a smaller cohort: recover the missing frozen submission, never pad with a
   newly drawn brief. If text was corrupted, preserve the excluded run and its cost, then restart
   the full pass with the same frozen five.
4. Wait for all five to reach a terminal status (`ok` or `failed`), watching the database rather than
   the page -- one `projects.status` row per brief, plus `build_jobs` and `stage_runs` for the finished
   ones:

```bash
# Confirm the five frozen submissions and their live statuses.
.venv/bin/python -c "
import sqlite3; a=sqlite3.connect('/home/kicraft/.kicraft/accounts.db')
print(a.execute('select id,status,board_code,cost_usd from projects where id between ? and ?', (<first>, <last>)).fetchall())"
```

   Post one short note per status change (*Keep the owner oriented*). A long `running` row is not stuck
   until the orphan reaper says so (`[reap] project <id> …` in the web log).
5. For each finished brief, capture the **outcome triple**: the terminal status; the last stage reached
   with its `failure_kind`; and the gate sentence from `stage_status[].diagnostics` (the run panel
   shows the same words, and for a `contract_rejected` failure the sentence names the thing that
   refused it). A brief that built is not fab-ready until its build job exits 0 and the fab outputs are
   in the generated tree.

### Cohort 3. Write the pass down before touching the source

One directory per cohort, `~/.kicraft/debug/cohort-<date>/`, holding:

- `cohort.json` -- the frozen five: seed, generator family, brief text verbatim, board code, project
  id, and the pipeline each row used (`projects.pipeline`); `selection.json` records diversity
  preflight and rejected draws. Backend switches and brief substitutions cannot masquerade as fixes.
- `pass_<n>.json` -- each brief's outcome triple, attempts, wall time and cost, and whether it reached
  fab-ready;
- `findings.json` -- one entry per gap: the briefs that showed it, the stage, the observed failure and
  its gate sentence, the suspected root cause, the production patch, the focused regression test, the
  status (`observed`, `patching`, `validated`, `unresolved`), and the before/after evidence for **every**
  brief in the cohort;
- `run-log.md` -- the short notes, in order, as the passes happened.

Nothing about the cohort may exist only in the chat; the pass record is what the comparison reads.

### Cohort 4. Fix, restart, rerun all five

1. **Triage.** For each failing brief, separate the generalizable pipeline gap (a contract, prompt,
   resolver or validator defect a patch can fix) from the capability request (a brief asking for
   something the curated library has no reviewed representation for -- a topology, a part class, a
   stack-up). Patch the first; record and surface the second for the owner's call, and never widen the
   library silently to make a brief pass.
2. **Patch at the source, with the smallest focused test.** Change production code, and add or update
   the test that reproduces the failure. Never patch a run's saved state, and never "fix" a failure by
   telling the model in an instruction to avoid it -- the rerun must show the unchanged brief passing
   on its own.
3. **One root cause per pass.** A pass can only attribute what it changed. When several fixes must land
   together they must share one class, and the pass still reruns all five as a set.
4. **Restart the product, so the rerun exercises the patch.** Nothing in a running web process changes
   under it. With no cohort run in flight -- a restart kills the design stages mid-flight, while a
   queued build survives -- use the canonical path:

```bash
./deploy/deploy-production.sh      # restarts both services, verifies web 200 + [build-worker] ready
```

   Re-login afterwards, because a restart drops sessions, and record the working-tree state in the pass
   record: these are uncommitted-tree fixes, and the restart runs what the tree holds.
5. **Rerun all five, same briefs, same order.** A fresh `Surprise me` click would draw five *different*
   briefs and void the comparison -- so `Surprise me` is not used at all in a rerun. Instead, put each
   recorded brief back into the composer verbatim and press `Design`, with the same `New design` between
   briefs and the same five-row assertion as in cohort 2: the composer's own handler does exactly what a
   `Surprise me` click does after it sets the text (compose the brief, then `start()`), so the entry
   point, the auto-defaulted questions, the caps and the queues are identical. Do all five, then wait
   and capture as in cohort 2.
6. **Compare every brief, not just the target.** Order the outcomes
   `intent < functional_spec < architecture < bom < wiring < build < fab-ready`, and mark each brief
   `improved`, `unchanged` or `regressed` against the previous pass. Anything that moved backwards -- an
   earlier stage now failing, a new failure kind on a brief that was already failing, a built board that
   no longer builds -- is an unintended consequence of the fix. Revert or refine it and say so, with both
   briefs' evidence.
7. **Iterate.** Repeat from cohort 2 with the cohort unchanged until all five are fab-ready or a stop
   condition below fires. State the pass cap before pass 1: the product's per-run budget bounds each
   brief, and when the owner sets no number of passes, default to at most three fix passes and say so in
   the first note.

### Cohort 5. Cohort stop conditions

The loop's stop conditions above still hold, plus these for a cohort:

1. all five briefs reach fab-ready -- the loop is done;
2. a brief whose failure reproduces identically twice with no generalizable cause found -- record the
   evidence, mark the finding `unresolved`, and move on rather than guessing;
3. a brief that needs a capability the curated library cannot express -- the owner's call, not a silent
   change;
4. the cohort's spend cap (state it before pass 1) -- on reaching it, stop cleanly and report what is
   saved;
5. a fix that has caused a regression twice, after a revert -- stop and hand the owner both passes,
   rather than iterating on a change that cannot be made to hold.

Never raise a cap, never widen the cohort, never drop a brief from the rerun to make the pass look
better, and never change a brief's text between passes -- a cohort whose five briefs changed proves
nothing about the four that were supposed to guard the fix.

### Cohort 6. End of cohort

One screen, in plain words: the five briefs and what each produced, the passes run, every fix with what
it targeted and what the guard briefs did, the guesses and capability requests awaiting the owner, and
the first thing a human should check, and the session's lines added and removed
(see *Session change totals*). Then the machine-readable last line, so a wrapper
can iterate without parsing prose:

```text
{"cohort": "<dir>", "passes": <n>, "briefs": [{"seed": <n>, "board": "KC-XXXXXX", "stem": "...", "outcome": "..."}],
 "fab_ready": <n>/5, "regressions": <n>, "findings": {"validated": <n>, "open": <n>},
 "changes": {"lines_added": <n>, "lines_removed": <n>},
 "cost_usd": <n>, "stopped": "<reason, or null>"}
```

Exit status 0 only when all five are fab-ready. A cohort that stopped early is never reported as
finished.

## Forensic requests

Reveal only the requested pending-artifact field:

- `show raw input` → `result.debug_context.prompt_state` and `extras`;
- `show exact prompt` → `result.debug_context.base_messages`;
- `show raw response` → `result.debug_context.raw_response`;
- `show tool trace` → artifact `events` filtered to tool/retry/serialization/reasoning events;
- `show candidate JSON` → `result.slot`.

Showing forensic data never changes state.

## Completion boundary

This boundary is the workspace modes' (interactive and loop). In cohort mode the live product authors,
builds and exports the boards itself, and the cohort's verdict is the five briefs' own fab-readiness
(see *Cohort 6*); nothing is handed to another skill for it.

After wiring acceptance, report that all five LLM stages are committed. Do not synthesize or build in this skill. Hand control back to the ordinary `kicraft` skill for the deterministic build.
