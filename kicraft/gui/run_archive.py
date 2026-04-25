"""Disk-based experiment-run archive.

Replaces the ``experiments.db`` SQLite layer. Each completed autoexperiment
run is archived to ``.experiments/runs/<run_id>/`` with the per-round JSONs
that the GUI's analysis page consumes. The current ``.experiments/`` files
remain as the live monitor's working set; this module never touches them.

Schema (per archived run):

  .experiments/runs/<run_id>/
    summary.json          # final hierarchical_summary.json contents
    experiments.jsonl     # per-round events log
    rounds/round_NNNN.json

The disk is the canonical store; nothing else needs to be in sync. CLI runs
and GUI runs are both visible to the analysis page through this single path.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunSummary:
    """A completed run, derived from on-disk archive contents."""

    run_id: str
    name: str = ""
    status: str = "done"  # done | error | partial
    pcb_file: str = ""
    total_rounds: int = 0
    completed_rounds: int = 0
    best_score: float = 0.0
    started_at: str = ""
    completed_at: str = ""
    archive_dir: Path = field(default_factory=Path)

    @property
    def id(self) -> str:
        # Compatibility shim with the prior ``state.db.get_experiments()``
        # API. Callers that iterated and read ``.id`` keep working.
        return self.run_id


def runs_root(experiments_dir: str | Path) -> Path:
    return Path(experiments_dir) / "runs"


def list_runs(experiments_dir: str | Path) -> list[RunSummary]:
    """Return all archived runs, newest first."""
    root = runs_root(experiments_dir)
    if not root.exists():
        return []
    summaries: list[RunSummary] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        summary = _read_run_summary(child)
        if summary is not None:
            summaries.append(summary)
    summaries.sort(key=lambda s: s.started_at or s.run_id, reverse=True)
    return summaries


def get_run(experiments_dir: str | Path, run_id: str) -> RunSummary | None:
    root = runs_root(experiments_dir) / run_id
    if not root.is_dir():
        return None
    return _read_run_summary(root)


def load_run_rounds(
    experiments_dir: str | Path,
    run_id: str,
) -> list[dict[str, Any]]:
    """Return the per-round dicts for an archived run, ordered by round_num.

    Reads ``runs/<run_id>/rounds/round_*.json`` directly. Falls back to
    parsing ``runs/<run_id>/experiments.jsonl`` when individual round files
    are missing (older archives, or aborted runs whose detail JSONs failed
    to write but the JSONL did).
    """
    archive = runs_root(experiments_dir) / run_id
    if not archive.is_dir():
        return []

    rounds_dir = archive / "rounds"
    rounds: list[dict[str, Any]] = []
    if rounds_dir.is_dir():
        for f in sorted(rounds_dir.iterdir()):
            if f.is_file() and f.suffix == ".json":
                try:
                    rounds.append(json.loads(f.read_text(encoding="utf-8")))
                except (OSError, json.JSONDecodeError):
                    continue

    if not rounds:
        jsonl = archive / "experiments.jsonl"
        if jsonl.is_file():
            try:
                with jsonl.open(encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            rounds.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except OSError:
                pass

    rounds.sort(key=lambda r: int(r.get("round_num", r.get("round", 0)) or 0))
    return rounds


def load_live_rounds(experiments_dir: str | Path) -> list[dict[str, Any]]:
    """Return the rounds for the currently-running (or most-recent) live run.

    Reads from ``.experiments/experiments.jsonl`` directly. Used by the
    monitor page; this module is the single place where the GUI looks for
    round data so stale-cache bugs across run boundaries can't recur.
    """
    jsonl = Path(experiments_dir) / "experiments.jsonl"
    if not jsonl.is_file():
        return []
    rounds: list[dict[str, Any]] = []
    try:
        with jsonl.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rounds.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    rounds.sort(key=lambda r: int(r.get("round_num", r.get("round", 0)) or 0))
    return rounds


def _read_run_summary(archive_dir: Path) -> RunSummary | None:
    summary_path = archive_dir / "summary.json"
    payload: dict[str, Any] = {}
    if summary_path.is_file():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            payload = {}

    rounds_dir = archive_dir / "rounds"
    completed = 0
    best_score = 0.0
    if rounds_dir.is_dir():
        for f in rounds_dir.iterdir():
            if f.suffix != ".json":
                continue
            completed += 1
            try:
                rec = json.loads(f.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            try:
                s = float(rec.get("score", 0) or 0)
            except (TypeError, ValueError):
                s = 0.0
            if s > best_score:
                best_score = s

    if completed == 0 and not payload:
        return None

    run_id = archive_dir.name
    pcb_file = str(payload.get("pcb_file", "") or "")
    started_at = str(payload.get("started_at", "") or "")
    completed_at = str(payload.get("completed_at", "") or "")
    name = _format_name(run_id, started_at, pcb_file)

    payload_best = payload.get("best_score")
    try:
        if isinstance(payload_best, (int, float)):
            best_score = max(best_score, float(payload_best))
    except (TypeError, ValueError):
        pass

    return RunSummary(
        run_id=run_id,
        name=name,
        status=str(payload.get("status", "done") or "done"),
        pcb_file=pcb_file,
        total_rounds=int(payload.get("rounds_requested", completed) or completed),
        completed_rounds=completed,
        best_score=best_score,
        started_at=started_at,
        completed_at=completed_at,
        archive_dir=archive_dir,
    )


def _format_name(run_id: str, started_at: str, pcb_file: str) -> str:
    pretty = started_at or run_id
    pcb_tail = Path(pcb_file).stem if pcb_file else ""
    if pcb_tail:
        return f"{pretty} ({pcb_tail})"
    return pretty
