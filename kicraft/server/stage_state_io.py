"""Durable stage state and argv-only design CLI boundaries."""
from __future__ import annotations

import datetime as dt
import json
import re
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from kicraft.fsutil import atomic_write_text

# The repo venv has no `kicraft` console script; cli_app.py has a __main__ guard.
KICRAFT = [sys.executable, "-m", "kicraft.design.cli_app"]

def run_design_cli(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    # Tag part-query telemetry from the web path so part-query-report can split
    # hosted vs offline usage (query_log reads $KICRAFT_CALLER). Honors an
    # explicit override if the environment already set one.
    env = {**os.environ, "KICRAFT_CALLER": os.environ.get("KICRAFT_CALLER", "web")}
    return subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=(str(cwd) if cwd else None)
    )


def prepare_stage(stage: str, state_path, workspace: Path) -> subprocess.CompletedProcess:
    return run_design_cli(KICRAFT + ["stage-prep", stage, str(state_path)], workspace)

def _fallback_stem(brief: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", brief.upper())[:3]
    return ("_".join(words)[:32]) or "PROJECT"

def commit_stage(stage, slot, state_path, brief, project_stem=None, workspace=None) -> tuple[bool, dict]:
    sf = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False)
    json.dump(slot, sf)
    sf.close()
    # Positionals (stage, state) BEFORE options: Python 3.12's argparse won't bind a
    # trailing optional positional that follows an option (e.g. --slot-file).
    cmd = KICRAFT + ["stage-commit", stage, str(state_path), "--slot-file", sf.name, "--no-archive"]
    if stage == "intent":
        cmd += ["--project-stem", project_stem or _fallback_stem(brief)]
    proc = run_design_cli(cmd, workspace)
    Path(sf.name).unlink(missing_ok=True)
    try:
        out = json.loads(proc.stdout)
    except json.JSONDecodeError:
        out = {"ok": False, "errors": [proc.stdout.strip() or proc.stderr.strip()]}
    return (proc.returncode == 0 and bool(out.get("ok"))), out


def stamp_stage_status(
    state_path,
    stage: str,
    ok: bool,
    *,
    cost_usd=None,
    attempts=None,
    rounds=None,
    tool_calls=None,
    wall_s=None,
    cpu_s=None,
    error=None,
    failure_kind=None,
) -> None:
    """Record a stage's durable outcome in state.json's stage_status block (a real
    ConversationState field, so the CLI's load/validate/dump round-trip preserves
    it). This is what lets a reopened project restore its pipeline progress
    without the ephemeral event stream. wall_s/cpu_s/rounds/tool_calls fill the
    prior measurement gap: how long a stage took, how much child CPU it burned,
    and how many tool rounds it cost (the written ledger records the same for the
    cross-run report). failure_kind is the terminal classification of a failed
    stage (one of reasoning_loop / truncated_json / invalid_json /
    commit_rejected / provider_error / transport_error). Tolerates a missing
    state.json (a first-stage failure before any commit). Atomic write: the web
    render timer reads this file concurrently."""
    p = Path(state_path)
    try:
        sj = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sj = {}
    entry: dict = {"ok": bool(ok), "finished_at": dt.datetime.now(dt.timezone.utc).isoformat()}
    if cost_usd is not None:
        entry["cost_usd"] = round(float(cost_usd), 6)
    if attempts is not None:
        entry["attempts"] = int(attempts)
    if wall_s is not None:
        entry["wall_s"] = round(float(wall_s), 3)
    if cpu_s is not None:
        entry["cpu_s"] = round(float(cpu_s), 3)
    if rounds is not None:
        entry["rounds"] = int(rounds)
    if tool_calls is not None:
        entry["tool_calls"] = int(tool_calls)
    if error is not None:
        entry["error"] = str(error)
    if failure_kind is not None:
        entry["failure_kind"] = str(failure_kind)
    block = sj.get("stage_status") or {}
    block[stage] = entry
    sj["stage_status"] = block
    p.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(p, json.dumps(sj, indent=2) + "\n")

def committed_bom_refs(state_path) -> list[str]:
    """Refs the committed BOM already contains -- the only refs wiring may use."""
    try:
        sj = json.loads(Path(state_path).read_text(encoding="utf-8"))
        parts = (sj.get("bom") or {}).get("parts") or []
        return sorted(str(p.get("ref")) for p in parts if isinstance(p, dict) and p.get("ref"))
    except (OSError, json.JSONDecodeError, AttributeError):
        return []

def attach_questions(state_path, stage: str, questions: list[dict]) -> None:
    """Write the stage's clarifying questions into state.json's open_questions
    (replacing any prior ones for this stage), so a reopened/parked project shows
    them. Tolerates a not-yet-created state.json (a first-stage question)."""
    try:
        sj = json.loads(Path(state_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        sj = {}
    kept = [q for q in (sj.get("open_questions") or []) if q.get("stage") != stage]
    sj["open_questions"] = kept + list(questions)
    Path(state_path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(state_path, json.dumps(sj, indent=2) + "\n")
