# KiCraft skill-eval

A repeatable way to **test the KiCraft skill the way a user actually runs it**,
score each run against a versioned rubric, and hand an implementation agent a
concrete fix-plan. This is the formalization of the ad-hoc `tests/manual-runs/`
prototype (`observer_notes.md` / `MONITOR_REPORT.md` / `fix_plan.md`).

It evaluates the **agentic skill** (the LLM-driven design conversation), not the
deterministic Python — and it scores the *agent's run*, which is a different thing
from `kicraft/scoring/` (that scores *PCB-layout* quality).

## The three-way split

The thing that made the old prototype painful was running everything inside the
repo. This separates three concerns:

1. **Harness** — *this directory*, versioned in the repo. The reusable assets:
   the workflow, the rubric, the scenario library, the report templates, and the
   scripts. Reviewed in PRs; evolves with the skill.
2. **Workspace** — a throwaway dir **outside the repo** (`~/kicraft-eval/workspaces/<id>/`)
   where the subject `claude` session actually runs `/kicraft`, exactly like a
   real user's project. Disposable. No KiCraft source nearby.
3. **Run record** — the harvested evidence + the written report for one run, also
   **outside the repo** (`~/kicraft-eval/runs/<id>/`). Disposable; only a report is
   copied back into the repo if it motivates a fix.

Running the subject externally is the point: it catches packaging/install bugs the
editable dev install hides, and prevents the subject from wandering into the source
tree (a real failure mode in the old runs).

## Three roles

- **Subject** — fresh `claude` session under test. Knows nothing about the eval.
  → `templates/subject_brief.md`
- **Observer** — separate session that watches the subject, scores it, writes the
  report. Does the Class-J judgment. → `templates/observer_prompt.md`
- **Implementation agent** — picks up the report's fix-plan cold.

## Two-tier scoring

One weighted **0–100** score (+ hard-fail gates) per run, from two halves:

- **Class C — deterministic** (`bin/score_run.py`): latency, #user-questions,
  re-commits/aborts, ERC errors/warnings, failed synthesis checks, permission
  floor. Reproducible; the baseline you trend.
- **Class J — judgment** (observer agent): grounding/ground-loops, part selection,
  electrical soundness, intent fidelity, spec compliance. The gotcha layer.

The scorer also records per-session **token usage + estimated cost** (parsed from
the transcript into `report.json` `metrics.token_usage`). It is an observability
metric, not a scored dimension, so it never affects the score or the rubric hash.

The score is stamped with the **rubric content hash**, so scores are only ever
compared within one rubric version. See `RUBRIC.md`.

## Quick start

```bash
REPO=/home/jason/Documents/SW_projects/KiCraft
PY=$REPO/.venv/bin/python

# 0. sanity: rubric hash is consistent
$PY $REPO/tests/skill-eval/bin/rubric_hash.py check

# 1. run a subject in a clean external workspace (see templates/subject_brief.md)
mkdir -p ~/kicraft-eval/workspaces/S02-run1 && cd ~/kicraft-eval/workspaces/S02-run1
claude            # paste scenarios/S02 opening prompt, follow its user-script

# 2. harvest + deterministic score (see templates/observer_prompt.md)
$PY $REPO/tests/skill-eval/bin/harvest_run.py --workspace ~/kicraft-eval/workspaces/S02-run1 \
    --scenario S02 --target-mode release --skill-dir ~/.claude/skills/kicraft
$PY $REPO/tests/skill-eval/bin/score_run.py score ~/kicraft-eval/runs/S02-<stamp> --scenario S02

# 3. observer grades Class-J in report.json, then:
$PY $REPO/tests/skill-eval/bin/score_run.py finalize ~/kicraft-eval/runs/S02-<stamp>/report.json
# ...and fills templates/RUN_REPORT.md into the run record.
```

Full process: **`WORKFLOW.md`**.

## File map

```
WORKFLOW.md            step-by-step process + the error taxonomy
RUBRIC.md              human-readable rubric (mirror of rubric.yaml)
rubric.yaml            CANONICAL scoring contract (hashed)
scenarios/             the use-case library (S01..S05) + schema
templates/
  subject_brief.md     how to launch a subject
  observer_prompt.md   how to boot an observer
  RUN_REPORT.md        the report template
  report.schema.json   shape of report.json
bin/
  rubric_hash.py       compute/check the rubric hash
  score_run.py         Class-C scorer + finalize
  harvest_run.py       workspace -> run record + provenance
  fixtures/            scorer regression fixtures
```

## Conventions

- Run scripts with the repo venv (`$REPO/.venv/bin/python`) — it has PyYAML + pcbnew.
- Nothing the harness produces should land inside the repo. Workspaces and run
  records live under `~/kicraft-eval/`. (Add that path to your shell, not to git.)
- One run = one scenario = one rubric hash. Don't compare across hashes.
