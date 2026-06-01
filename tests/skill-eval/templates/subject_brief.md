# Launching a subject session

The **subject** is a fresh `claude` session that runs KiCraft as a real user
would. It is the thing under test; it must know nothing about this eval harness,
the rubric, or that it is being observed. Do not coach it.

## 1. Pick the target (what's under test)

| mode | CLI | skill | use when |
|---|---|---|---|
| **release** (default) | pipx `kicraft` | global `~/.claude/skills/kicraft` | simulating a real user; catches packaging/install bugs |
| **dev** | repo `.venv/bin/kicraft` | working-tree `.claude/skills/kicraft` synced to global | verifying an unreleased fix |

For **dev** mode, sync the skill so the global copy matches the working tree, and
record it:

```
diff -rq <repo>/.claude/skills/kicraft ~/.claude/skills/kicraft   # confirm/sync
```

The observer records the resulting `skill_sha256` via `harvest_run.py --skill-dir`.

## 2. Make a clean external workspace

**Never run the subject inside the KiCraft repo** (it masks install bugs and the
subject can wander into the source tree). Use a throwaway dir outside it:

```
mkdir -p ~/kicraft-eval/workspaces/<S0x>-<UTCstamp>
cd       ~/kicraft-eval/workspaces/<S0x>-<UTCstamp>
```

This dir must contain no KiCraft source and no prior `.kicraft/`.

## 3. Launch and drive

```
claude
```

Paste the scenario's **opening prompt verbatim** (from `scenarios/<S0x>.md`). Then
follow the scenario's **user-script**: when the subject asks a clarifying
question, answer with the script's canned answer for that fork — and nothing more.
If the subject asks something the script doesn't anticipate, answer minimally and
in character (a real user of the stated expertise level), and note the
off-script question for the observer. Let the run proceed to synthesis (or to a
clean stop) on its own; do not nudge it past errors.

Record the **session UUID** (top of the `claude` session, or the newest file in
`~/.claude/projects/<mangled-workspace-path>/`) so the observer can grab the
transcript.

## 4. When it ends

Hand off to the observer:

```
<repo>/.venv/bin/python <repo>/tests/skill-eval/bin/harvest_run.py \
  --workspace ~/kicraft-eval/workspaces/<S0x>-<UTCstamp> \
  --scenario <S0x> --target-mode <release|dev> \
  --skill-dir <skill dir under test>
```

Then the observer scores and writes the report (see `observer_prompt.md`).
