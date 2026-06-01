---
description: Launch the KiCraft Experiment Manager GUI for a project — auto-resolves the .kicad_pro directory, uses the repo venv, runs in the background, and reports the URL.
argument-hint: "[project path or name] (optional)"
---

Launch the KiCraft Experiment Manager (NiceGUI app, `python -m kicraft.gui`) for a project. The requested project is: `$ARGUMENTS` (may be empty).

Key fact that makes this work: the GUI picks its project by walking **upward** from the current working directory for the first `*.kicad_pro` file (`kicraft/gui/state.py:_project_root`). So you must launch it with cwd set to the directory that *directly contains* the `.kicad_pro`. For KiCraft output that file is nested, e.g. `tests/manual-runs/<proj>/generated/<NAME>/<NAME>.kicad_pro` — not at the folder the user usually names. Resolve it, don't assume.

Follow this runbook. Print a short note before each step (per the user's global preference for visible reasoning), keep tool fan-out small.

## 1. Resolve repo, venv, and the project's `.kicad_pro` directory

Run this single block (it prints the distinct project directories plus the venv python path):

```bash
REPO=$(git rev-parse --show-toplevel); PY="$REPO/.venv/bin/python"
ARG="$ARGUMENTS"
search() { find "$1" -maxdepth 4 -name '*.kicad_pro' 2>/dev/null; }
CANDS=""
if [ -n "$ARG" ]; then
  for base in "$ARG" "$REPO/$ARG" "$REPO/tests/manual-runs/$ARG"; do
    [ -e "$base" ] && CANDS="$CANDS"$'\n'"$(search "$base")"
  done
  if [ -z "$(echo "$CANDS" | tr -d '[:space:]')" ]; then
    CANDS=$(find "$REPO/tests/manual-runs" -maxdepth 4 -name '*.kicad_pro' -path "*$ARG*" 2>/dev/null)
  fi
else
  CANDS=$(search "$REPO/tests/manual-runs")
fi
echo "=== project dirs (each contains a .kicad_pro) ==="
echo "$CANDS" | sed '/^$/d' | xargs -r -n1 dirname | sort -u
echo "PY=$PY"
echo "REPO=$REPO"
```

Then decide based on how many distinct directories printed:

- **Exactly one** → that is `PROJECT_DIR`. Proceed.
- **More than one** → use `AskUserQuestion` to let the user pick which one to launch (label each by its `.kicad_pro` stem and its path under `tests/manual-runs/`). The user chose "always let me pick," so when no argument was given you will normally land here.
- **Zero** → the project wasn't found. List what *is* available by running `find "$REPO/tests/manual-runs" -maxdepth 4 -name '*.kicad_pro'`, show the user, and ask which to launch (or to pass a path).

Capture the chosen absolute `PROJECT_DIR` and the printed `PY` — you'll paste them literally into later commands (shell variables do **not** persist between Bash calls).

## 2. Pre-flight: venv + port 8080

```bash
test -x "<PY>" && echo "venv ok: <PY>" || echo "MISSING venv python at <PY>"
ss -ltn 2>/dev/null | grep -q ':8080 ' && echo "PORT 8080 IN USE" || echo "port 8080 free"
```

- If the venv python is missing, stop and tell the user (the repo `.venv` isn't set up).
- If **port 8080 is in use**, a GUI is already running. Do **not** silently kill it. Tell the user it's already up at http://localhost:8080 and ask via `AskUserQuestion` whether to (a) keep the running one, or (b) stop it and relaunch with the selected project. Only if they choose (b), run `pkill -f 'kicraft\.gui'`, wait ~1s, then continue.

## 3. Launch in the background

Use a project-specific log file. Run this with **run_in_background: true**:

```bash
cd "<PROJECT_DIR>" && "<PY>" -m kicraft.gui > "/tmp/kicraft_gui_$(basename "<PROJECT_DIR>").log" 2>&1
```

## 4. Confirm readiness and report

Poll the log (do not sleep in the foreground excessively — a short bounded loop is fine):

```bash
LOG="/tmp/kicraft_gui_$(basename "<PROJECT_DIR>").log"
for i in $(seq 1 30); do
  grep -qiE "NiceGUI ready|Uvicorn running|http://localhost:8080" "$LOG" 2>/dev/null && break
  grep -qiE "Traceback|Error|Address already in use" "$LOG" 2>/dev/null && break
  sleep 0.5
done
cat "$LOG"
```

- On success (`NiceGUI ready ...`), tell the user the GUI is live at **http://localhost:8080** and which project it loaded (the `.kicad_pro` stem).
- On error, show the relevant log tail and diagnose (common cases: port already bound, missing dependency, no `.kicad_pro` in the launch dir).

Keep the final message short: the URL, the loaded project, and the background task id so the user can stop it later.
