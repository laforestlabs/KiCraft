# Launching a subject session

The **subject** is a fresh Agent Skills-compatible session that runs KiCraft as a real user would. It is the system under test and must know nothing about the eval harness, rubric, or observer. Do not coach it.

## 1. Pick the target

| mode | CLI | skill | use when |
|---|---|---|---|
| **release** | packaged/pipx `kicraft` | installed `.agents/skills/kicraft` | simulate a real user and catch installation defects |
| **dev** | repository `.venv/bin/kicraft` | working-tree `.agents/skills/kicraft` copied to the subject runtime's discovery root | verify an unreleased change |

For dev mode, compare the installed skill directory with `<repo>/.agents/skills/kicraft` before starting. The observer records the tested directory's `skill_sha256` through `harvest_run.py --skill-dir`.

## 2. Make a clean external workspace

Never run the subject inside the KiCraft repository; that masks installation defects and exposes source files a real user would not have. Use an empty external directory:

```text
~/kicraft-eval/workspaces/<S0x>-<UTCstamp>
```

It must contain no KiCraft source, `.kicraft/`, or prior `generated/` directory.

## 3. Launch and drive

Start any Agent Skills-compatible coding agent in the workspace. Confirm it discovers the `kicraft` skill, then paste the scenario's opening prompt verbatim from `scenarios/<S0x>.md`.

Follow only the scenario's user script. When the subject asks a clarification, answer with the canned answer for that fork and nothing more. If the script does not anticipate a question, answer minimally in character and record the off-script question. Let the run proceed to build or a clean stop without nudging it past errors.

If the runtime can export a JSONL transcript containing assistant usage/tool records, save its path for `harvest_run.py --transcript`. Transcript capture is optional; artifact grading still works without it.

## 4. Hand off

```text
<repo>/.venv/bin/python <repo>/tests/skill-eval/bin/harvest_run.py \
  --workspace ~/kicraft-eval/workspaces/<S0x>-<UTCstamp> \
  --scenario <S0x> --target-mode <release|dev> \
  --skill-dir <skill dir under test> \
  [--transcript <agent transcript.jsonl>]
```

The observer then scores the run and writes the report using `observer_prompt.md`.
