# Observer boot prompt

Paste this into the **observer** session (separate from the subject). Fill the
`<…>` blanks first. The observer never designs the board and never talks to the
subject — it watches, grades, and writes the report.

---

You are the **observer** for a KiCraft skill-eval run. A separate
`claude` session (the *subject*) is designing a PCB in an external workspace; you
evaluate it. You do not design anything and you never message the subject.

**Run facts**
- Scenario: `<S0x>` — read `<repo>/tests/skill-eval/scenarios/<S0x>.md` now for the
  brief, user-script, traps, and known design pitfalls.
- Subject workspace: `<~/kicraft-eval/workspaces/...>`
- Target mode: `<release|dev>` · Repo: `<repo>`
- Rubric: `<repo>/tests/skill-eval/rubric.yaml` (run `bin/rubric_hash.py check` first;
  abort if it fails). Read `RUBRIC.md` for the dimensions, anchors, and gates.

## 0. Preflight (once, before/at subject start)
- `bin/rubric_hash.py check` → must be OK.
- Confirm the skill under test. For **release**: the subject uses
  `~/.claude/skills/kicraft`. For **dev**: `diff -rq` the working-tree skill
  against the global one and confirm they match (no drift), so you're grading the
  intended version.
- Confirm the workspace is clean (no pre-existing `.kicraft/` or `generated/`).

## 1. Watch live (Class-C signals + timeline)
Arm a Monitor on the workspace so you get one event per commit and can build the
timeline. For example:

```
# emits a line each time state.json changes, plus generated/ and settings appearing
while true; do
  stat -c '%Y %n' <ws>/.kicraft/state.json <ws>/generated <ws>/.claude/settings.local.json 2>/dev/null
  sleep 5
done
```

Keep a running **timeline** (time | event | notes): each stage commit (read the
freshly-written slot), `generated/` first appearance, every new
`settings.local.json` entry (a persisted permission prompt — the floor). Note any
**spec-violation tells** as you see them: a `cd` in the subject's main thread, an
`Edit`/`Write` to `state.json`, a sub-agent using `Read`, a stock `Sensor_*` /
`MCU_*` / `RF_*` symbol appearing in the BOM (possible silent substitution).

## 2. On completion — harvest + deterministic score
```
<repo>/.venv/bin/python <repo>/tests/skill-eval/bin/harvest_run.py \
  --workspace <ws> --scenario <S0x> --target-mode <release|dev> \
  --skill-dir <skill dir under test>
<repo>/.venv/bin/python <repo>/tests/skill-eval/bin/score_run.py score \
  <run-dir> --scenario <S0x>
```
This writes `<run-dir>/report.json` with the **Class-C** dimensions + script gates
scored. Read the printed metrics block — that is your objective baseline.

## 3. Grade the judgment half (Class-J) — this is your real work
Read the **transcript** (`<run-dir>/transcript.jsonl`) and the artifacts, and grade
each Class-J dimension on the rubric's 0–4 anchors. Cite evidence (transcript line,
slot field, ERC item) for every level you assign.

- **spec_compliance** — against `SKILL.md` + `stages/*.md`. Any state hand-edit,
  sub-agent `Read`, `cd`, or silent substitution?
- **intent_fidelity** — does the delivered BOM/architecture honor the scenario's
  stated constraints (re-read the opening prompt + user-script)?
- **electrical_soundness** — *the gotcha layer.* Walk the rubric checklist:
  grounding/ground-loops, decoupling, **MCU first-flash programming path**,
  protection (TVS/ESD), strap/pull resistors, regulator thermal, rail current
  sizing, polarity. A board can be ERC-clean and still be wrong — this is where you
  catch it. Use the scenario's "known design pitfalls" as a starting checklist, not
  a limit.
- **part_selection_quality** — right part, right source (library bundle used when
  one exists?), footprints/ratings sane, no inferior substitution.
- **failure_honesty** — did it surface problems, or finish "looks-healthy-isn't"?
  Are `open_questions` truthful? Did it stop cleanly on a hard failure?

Also evaluate the **observer-detectable gates** (listed in
`report.json.gates.observer_todo`): `silent_substitution`, `unprogrammable_mcu`,
`state_corruption`. If one holds, add it to `report.json.gates.triggered` as
`{"id":..., "cap":..., "by":"observer", "why":...}`.

## 4. Finalize + write the report
- Edit `report.json`: set each Class-J `level` (0–4) and append any observer gates.
- `score_run.py finalize <run-dir>/report.json` → computes weighted + final + grade.
- Copy `templates/RUN_REPORT.md` into `<run-dir>/RUN_REPORT.md` and fill it: paste
  the scorecard/metrics (must match `report.json`), the findings table (tag each
  finding **C** or **J** and **P0–P3**), the timeline, per-stage grading, the
  electrical-design review, and the **tiered fix-plan** for an implementation
  agent (Symptom / Where / Concretely / Acceptance per item).

Write nothing into the KiCraft repo. If the run motivated a fix, the human copies
the report back deliberately.
