# KiCraft skill-eval — workflow

The end-to-end process for running, scoring, and reporting one KiCraft eval.
Read `README.md` first for the three-way split and the role definitions.

A run has two agents: a **subject** (under test) and an **observer** (evaluator),
in **separate** `claude` sessions. The observer drives this workflow.

```
preflight ─▶ launch subject ─▶ watch live ─▶ harvest ─▶ score (C) ─▶ grade (J) ─▶ finalize ─▶ report
```

## Phase 0 — Preflight (observer)

1. **Rubric integrity:** `bin/rubric_hash.py check` must print OK. If it fails, the
   rubric was edited without re-stamping — fix that before scoring anything, or all
   scores are meaningless.
2. **Choose the target** (what's under test) and record it:
   - **release** (default, true-user sim): pipx `kicraft` + global
     `~/.claude/skills/kicraft`. Catches install/packaging bugs.
   - **dev** (verify an unreleased fix): repo `.venv` CLI + working-tree skill. Sync
     the skill to global and `diff -rq` to confirm no drift, so you grade the
     version you think you're grading.
3. **Pick a scenario** from `scenarios/` and read it fully — the opening prompt,
   user-script, traps, and known design pitfalls are your grading checklist.
4. **Clean workspace:** an empty dir outside the repo, e.g.
   `~/kicraft-eval/workspaces/<S0x>-<stamp>/`. No prior `.kicraft/` or `generated/`.

## Phase 1 — Launch the subject (subject session)

Follow `templates/subject_brief.md`. In short: a fresh `claude` in the workspace,
paste the scenario's opening prompt **verbatim**, answer clarifications only from
the scenario's **user-script** (in character, minimal), and let the run reach
synthesis or a clean stop on its own. **Do not coach it and do not push it past
errors** — how it handles an error is part of the test. Record the session UUID.

## Phase 2 — Watch live (observer)

Arm a Monitor on the workspace (see `observer_prompt.md` for a sample loop) and
keep a **timeline** (time | event | notes). Capture, as they happen:

- each **stage commit** (read the freshly-written slot; sanity-check it against the
  stage spec);
- **`generated/`** first appearance and the synthesis result;
- each new **`.claude/settings.local.json`** entry — a persisted permission prompt;
  this is the permission **floor** (it misses single-shot "Yes" prompts, so report
  it as "at least N");
- **spec-violation tells**: a `cd` in the subject's main thread, an `Edit`/`Write`
  on `state.json`, a sub-agent using `Read`, or a stock `Sensor_*`/`MCU_*`/`RF_*`
  symbol in the BOM (possible silent substitution).

Watching is for capturing what the artifacts won't show later; the heavy grading
happens post-hoc from the transcript.

## Phase 3 — Harvest (observer)

When the subject finishes:

```
bin/harvest_run.py --workspace <ws> --scenario <S0x> \
    --target-mode <release|dev> --skill-dir <skill dir under test>
```

This copies `.kicraft/`, `generated/`, `settings.local.json`, finds the subject's
transcript (`~/.claude/projects/<mangled-ws>/*.jsonl`), and stamps `run.json`
(target mode, skill `sha256`, CLI path, session ids). Output: `~/kicraft-eval/runs/<run-id>/`.

## Phase 4 — Deterministic score, Class C (observer)

```
bin/score_run.py score <run-dir> --scenario <S0x>
```

Writes `<run-dir>/report.json` with the five **Class-C** dimensions + script gates
scored, and prints the metrics block. Class-J levels are left `null`. If the
transcript was missing, latency/convergence/friction come back `partial` — note it.

## Phase 5 — Judgment grade, Class J (observer)

The observer's real work. Read the transcript + artifacts and grade the five
Class-J dimensions on the rubric anchors, **with evidence for each level**:
`spec_compliance`, `intent_fidelity`, `electrical_soundness` (the gotcha layer —
walk the checklist), `part_selection_quality`, `failure_honesty`. Also evaluate the
observer-detectable gates (`silent_substitution`, `unprogrammable_mcu`,
`state_corruption`) and add any that hold to `report.json.gates.triggered`. Edit the
Class-J `level`s into `report.json`.

## Phase 6 — Finalize + report (observer)

```
bin/score_run.py finalize <run-dir>/report.json
```

Computes `weighted = Σ(weight·level/4)`, applies the lowest triggered gate cap, and
assigns grade + verdict — by code, not mental math. Then copy
`templates/RUN_REPORT.md` into the run record and fill it: scorecard + metrics
(must match `report.json`), findings table, timeline, per-stage grading, the
electrical-design review, and the tiered fix-plan.

---

## The error taxonomy (what we're hunting)

Every finding is tagged **Class** (who can see it) and **Severity** (how bad).

**Class C — computing / deterministic** (machine-detectable, scored by the script):
- pipeline didn't complete; synthesize crashed / produced no files
- ERC errors; failed synthesis checks (`synthesis_check.json`)
- CLI non-zero exits, tracebacks, `ModuleNotFound` / env-isolation failures
- unresolvable footprint/symbol caught (or missed) at commit
- excess permission prompts; thrash (error-driven re-commits/aborts); stalls

**Class J — judgment / strategy / intent** (needs the observer reading the run):
- **spec violations** — state hand-edit, sub-agent `Read`, `cd`, silent substitution
- **intent infidelity** — a stated constraint dropped or contradicted
- **bad electrical design** — ground loops, missing decoupling/protection, **no MCU
  programming path**, marginal thermal/rail sizing, wrong polarity. *ERC-clean is
  not the same as correct.*
- **bad part selection** — inferior substitution, library bundle bypassed, wrong
  ratings/package
- **dishonest finish** — "looks healthy, isn't"; stale `open_questions`; grinding on
  a failure instead of surfacing it

**Severity:** **P0** ship-blocker · **P1** real bug / spec violation · **P2**
UX/quality · **P3** minor / upstream.

The fix-plan in the report turns these findings into ordered, pick-up-cold work
items (Symptom / Where / Concretely / Acceptance) — the same structure that made
the prototype `fix_plan.md` directly actionable.

## When the rubric changes

Edit `rubric.yaml`, bump `meta.version`, `bin/rubric_hash.py compute --write`,
update `RUBRIC.md` (stamp + changelog), `bin/rubric_hash.py check`. Scores from
before the change keep their old hash and are **not** comparable to new ones — start
a fresh cohort.
