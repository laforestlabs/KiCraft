"""Append-only telemetry for parts-library queries.

Every BOM-stage part query (lookup_lcsc_id, add_part, search/lookup of a symbol
or footprint, list_parts) appends one JSON line here, so we can later see:

- which curated bundles are actually used (popularity -> polish / 3D-model
  candidates), and
- which queries miss the library and fall back to LCSC or stock KiCad
  (add-to-library candidates that would cut BOM cost and add consistency).

Both the hosted web executor and the offline Claude Code skill drive the same
``kicraft`` CLI, so recording at the CLI-handler layer captures every part
query on a machine. The log is per-machine at ``~/.kicraft/part_queries.jsonl``
(override with ``$KICRAFT_QUERY_LOG``), next to the web spend ledger. It stores
part identifiers, search keywords, outcomes, and the project stem only, not
full designs.

Writing is strictly best-effort: telemetry must never break a tool call, so
every failure here is swallowed. Reading skips malformed lines.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from pathlib import Path
from typing import Iterator

ENV_PATH = "KICRAFT_QUERY_LOG"   # override the log file location
ENV_RUN_ID = "KICRAFT_RUN_ID"    # correlate a design run (set by the web driver)
ENV_CALLER = "KICRAFT_CALLER"    # free-form origin tag, e.g. "web" / "cli"


def log_path() -> Path:
    override = os.environ.get(ENV_PATH)
    if override:
        return Path(override).expanduser()
    return Path.home() / ".kicraft" / "part_queries.jsonl"


def _now_iso() -> str:
    return (
        _dt.datetime.now(_dt.timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def record(tool: str, *, outcome: str | None = None, query: str | None = None,
           **fields) -> None:
    """Append one query event as a JSON line. Best-effort: never raises.

    ``tool`` is the query verb (e.g. ``"lookup_lcsc_id"``). ``outcome`` is a
    short classifier (``"hit"`` / ``"miss"`` / ``"fetched"`` / ...). Extra
    ``fields`` (e.g. ``lcsc=...``, ``library_name=...``, ``n_matches=...``) are
    merged in; ``None`` values are dropped to keep lines compact. Caller context
    (project = cwd basename, run_id, caller) is attached from the environment.
    """
    try:
        event: dict = {"ts": _now_iso(), "tool": tool}
        if outcome is not None:
            event["outcome"] = outcome
        if query is not None:
            event["query"] = query
        event.update({k: v for k, v in fields.items() if v is not None})
        run_id = os.environ.get(ENV_RUN_ID)
        if run_id:
            event["run_id"] = run_id
        caller = os.environ.get(ENV_CALLER)
        if caller:
            event["caller"] = caller
        try:
            event["project"] = Path.cwd().name
        except OSError:
            pass
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        # Mode "a" => O_APPEND: a single small write is effectively atomic on
        # POSIX, so concurrent CLI subprocesses (the web service shells out per
        # tool call) interleave cleanly without a lock.
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, separators=(",", ":")) + "\n")
    except Exception:
        # Telemetry must not interfere with the actual command.
        pass


def read_events(path: Path | str | None = None) -> Iterator[dict]:
    """Yield events from the log oldest-first, skipping malformed lines.

    Yields nothing if the log does not exist yet.
    """
    p = Path(path) if path else log_path()
    if not p.is_file():
        return
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                yield obj


__all__ = ["ENV_PATH", "ENV_RUN_ID", "ENV_CALLER", "log_path", "record", "read_events"]
