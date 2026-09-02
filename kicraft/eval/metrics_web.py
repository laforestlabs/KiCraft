"""Build the Class-C metrics dict ``m`` from a finished web design project.

The web app runs the whole pipeline server-side and persists, per project under
``projects_dir/<uid>/<pid>/``:

  * ``state.json`` + the ``kicraft/`` tree (slots, history, open_questions, bom),
  * ``generated/<stem>/`` with the KiCad files, ``synthesis_check.json``, ERC reports,
  * ``events.jsonl`` (the design event stream).

Token/cost lives in the SQLite spend ledger keyed by ``run_id`` (``p<pid>-<ts>``).

This collector reuses the shared artifact parsers (so pipeline-completion and
ERC/synthesis cleanliness are scored identically to the offline harness) and
synthesises the ``transcript`` sub-dict from ``events.jsonl`` so the convergence
and computing-cleanliness scorers run unchanged. Optional agent-runtime
permission evidence has no web analog and is reported as zero excess.
"""
from __future__ import annotations

import json
from pathlib import Path

from .artifacts import (
    _find_glob,
    _find_one,
    analyze_state,
    count_generated,
    parse_erc,
    parse_synthesis_check,
)
from .scoring import _parse_ts, compute_latency_min

# build-log lines carrying these markers count as a synthesis-blocking crash,
# matching the harness transcript crash definition (kept in lockstep on purpose).
_CRASH_MARKERS = ("Traceback (most recent call last)", "ModuleNotFoundError")


def analyze_events(events_path: Path) -> dict:
    """Reduce ``events.jsonl`` to the same shape the harness transcript parser
    produces, so the shared scorers consume it without changes.

    Mapping (web event stream -> run-trace signals):
      retry event        -> one error-driven re-commit  (failed_commits)
      question event     -> one clarifying-question turn (ask_questions)
      stage_done ok:True  -> a committed slot            (stage_commit_calls)
      build_start present -> synthesis was attempted     (synth_attempts)
      build_log traceback -> a synthesis-blocking crash  (crashes)
    Event records carry no timestamps, so latency is computed elsewhere.
    """
    events_path = Path(events_path)
    if not events_path.exists():
        return {"present": False}
    retries = questions = stage_commits = crashes = 0
    build_started = False
    for line in events_path.read_text(errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = ev.get("kind")
        if kind == "retry":
            retries += 1
        elif kind == "question":
            questions += 1
        elif kind == "stage_done" and ev.get("ok"):
            stage_commits += 1
        elif kind == "build_start":
            build_started = True
        elif kind == "build_log":
            text = ev.get("text") or ""
            if any(mark in text for mark in _CRASH_MARKERS):
                crashes += 1
    return {
        "present": True,
        "path": str(events_path),
        "first_ts": None, "last_ts": None, "synth_ts": None,
        "stage_commit_calls": stage_commits,
        "failed_commits": retries,
        "ask_questions": questions,
        "synth_attempts": 1 if build_started else 0,
        "crashes": crashes,
    }


def _token_usage_from_ledger(ledger_path, run_id_prefix: str | None) -> dict | None:
    """Sum the project's billed token usage from the spend ledger, grouped by the
    project's run_id prefix (``p<pid>-``). Observability only; never scored, so a
    missing ledger or import degrades to None like the harness token summary."""
    if not ledger_path or not Path(ledger_path).exists():
        return None
    try:
        from kicraft.cli.web_cost_report import load_rows
    except ImportError:
        return None
    try:
        rows = load_rows(str(ledger_path))
    except Exception:
        return None
    inp = out = 0
    cost = 0.0
    turns = 0
    by_model: dict[str, int] = {}
    for r in rows:
        meta = r.get("meta") or {}
        rid = meta.get("run_id") or ""
        if run_id_prefix and not rid.startswith(run_id_prefix):
            continue
        inp += int(r.get("input_tokens") or 0)
        out += int(r.get("output_tokens") or 0)
        cost += float(r.get("cost_usd") or 0.0)
        turns += 1
        model = r.get("model") or "?"
        by_model[model] = by_model.get(model, 0) + 1
    if turns == 0:
        return None
    return {
        "input_tokens": inp, "output_tokens": out, "total_tokens": inp + out,
        "turns": turns, "estimated_cost_usd": round(cost, 4),
        "cost_known": True, "by_model": by_model,
    }


def collect_web_metrics(project_dir, *, ledger_path=None, run_id_prefix=None,
                        started_at: str | None = None,
                        finished_at: str | None = None) -> dict:
    """Build the ``m`` metrics dict for a finished web project directory.

    Same shape as ``score_run.collect_metrics`` so the shared Class-C scorers and
    script gates run unchanged. ``started_at``/``finished_at`` (the project row's
    ISO timestamps) give a precise latency; without them latency falls back to the
    state-history -> synth-checked_at heuristic (flagged approximate). Token usage
    is pulled from the ledger when ``ledger_path`` is given (prefix defaults to
    ``p<project_dir name>-``).
    """
    pd = Path(project_dir)
    state = analyze_state(_find_one(pd, "state.json"))
    synth = parse_synthesis_check(_find_one(pd, "synthesis_check.json"))
    erc = parse_erc(_find_glob(pd, "*_erc.rpt"))
    generated = count_generated(pd)
    transcript = analyze_events(pd / "events.jsonl")

    latency = None
    if started_at and finished_at:
        a, b = _parse_ts(started_at), _parse_ts(finished_at)
        if a and b and b >= a:
            latency = (round((b - a).total_seconds() / 60, 1), False)
    if latency is None:
        latency = compute_latency_min(transcript, state, synth)

    if run_id_prefix is None:
        run_id_prefix = f"p{pd.name}-"
    token_usage = _token_usage_from_ledger(ledger_path, run_id_prefix)

    return {
        "state": state, "synth": synth, "erc": erc, "generated": generated,
        # No permission prompts server-side: zero floor, zero excess.
        "perm": {"present": False, "count": 0, "excess": 0, "entries": []},
        "transcript": transcript, "latency": latency,
        "token_usage": token_usage,
        # Deterministic question band for web/self-eval runs. With None here,
        # score_friction's q_state stayed permanently "unknown" and (with
        # excess pinned to 0 above) the function collapsed to a constant
        # level 3 -- 6% of every grade was dead weight dressed as evaluated
        # output, and no Class-J dimension actually judges question count
        # (2026-07-19 review §8.1). The self-eval briefs are curated to be
        # answerable without clarification, so 0-2 questions is in-band; a
        # run interrogating a complete brief IS friction worth scoring down.
        "expected_question_band": (0, 2),
        "run_meta": {}, "scenario": None, "target_mode": "web",
    }
