"""SQLite results store for load-test runs (mirrors kicraft/tuning/store.py).

Three tables: ``runs`` (one per scenario invocation), ``samples`` (the 1 Hz
host/process/queue time series from metrics.py), and ``events`` (per-request /
per-build latency + outcome points). Append-only writes under WAL, so the 1 Hz
sampler thread, the harness, and the /admin/loadtest dashboard (a separate
process) never block each other. Connection-per-op, like accounts.py, so the
store is safe to share across the harness's worker threads.
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id       TEXT PRIMARY KEY,
    scenario     TEXT,
    started_at   REAL,
    finished_at  REAL,
    params_json  TEXT,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS samples (
    run_id        TEXT NOT NULL,
    ts            REAL NOT NULL,
    cpu_pct       REAL,
    mem_used_mb   REAL,
    mem_pct       REAL,
    loadavg       REAL,
    disk_free_mb  REAL,
    web_rss_mb    REAL,
    worker_rss_mb REAL,
    queue_depth   INTEGER,
    queue_running INTEGER,
    wal_bytes     INTEGER,
    lock_ms       REAL
);
CREATE TABLE IF NOT EXISTS events (
    run_id     TEXT NOT NULL,
    ts         REAL NOT NULL,
    kind       TEXT,
    latency_ms REAL,
    rc         INTEGER,
    detail     TEXT
);
CREATE INDEX IF NOT EXISTS idx_samples_run ON samples(run_id, ts);
CREATE INDEX IF NOT EXISTS idx_events_run ON events(run_id, ts, kind);
"""

_SAMPLE_COLS = (
    "ts", "cpu_pct", "mem_used_mb", "mem_pct", "loadavg", "disk_free_mb",
    "web_rss_mb", "worker_rss_mb", "queue_depth", "queue_running", "wal_bytes", "lock_ms",
)


def default_store_path() -> Path:
    """Load-test DB location -- sibling of self_eval/, independent of Settings so
    it never needs an OPENROUTER_API_KEY."""
    root = os.environ.get("KICRAFT_LOADTEST_DIR", "").strip()
    base = Path(root) if root else Path.home() / ".kicraft" / "loadtest"
    return base / "loadtest.db"


class LoadResultStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as db:
            db.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=30.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        db.execute("PRAGMA synchronous=NORMAL")
        return db

    # --- runs --------------------------------------------------------------
    def start_run(self, run_id: str, scenario: str, params: dict | None = None,
                  started_at: float | None = None) -> None:
        with self._conn() as db:
            db.execute(
                "INSERT OR REPLACE INTO runs (run_id, scenario, started_at, params_json) "
                "VALUES (?, ?, ?, ?)",
                (run_id, scenario, started_at if started_at is not None else time.time(),
                 json.dumps(params or {})),
            )

    def finish_run(self, run_id: str, summary: dict | None = None,
                   finished_at: float | None = None) -> None:
        with self._conn() as db:
            db.execute(
                "UPDATE runs SET finished_at=?, summary_json=? WHERE run_id=?",
                (finished_at if finished_at is not None else time.time(),
                 json.dumps(summary or {}), run_id),
            )

    def get_run(self, run_id: str) -> dict | None:
        with self._conn() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
        return _run_row(row) if row else None

    def list_runs(self, limit: int = 100) -> list[dict]:
        with self._conn() as db:
            rows = db.execute(
                "SELECT * FROM runs ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [_run_row(r) for r in rows]

    # --- samples -----------------------------------------------------------
    def add_sample(self, run_id: str, sample: dict) -> None:
        cols = ", ".join(("run_id", *_SAMPLE_COLS))
        ph = ", ".join("?" for _ in range(len(_SAMPLE_COLS) + 1))
        vals = (run_id, *(sample.get(c) for c in _SAMPLE_COLS))
        with self._conn() as db:
            db.execute(f"INSERT INTO samples ({cols}) VALUES ({ph})", vals)

    def samples_for(self, run_id: str) -> list[dict]:
        with self._conn() as db:
            rows = db.execute(
                "SELECT * FROM samples WHERE run_id=? ORDER BY ts", (run_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    # --- events ------------------------------------------------------------
    def add_event(self, run_id: str, kind: str, *, latency_ms: float | None = None,
                  rc: int | None = None, detail: str | None = None,
                  ts: float | None = None) -> None:
        with self._conn() as db:
            db.execute(
                "INSERT INTO events (run_id, ts, kind, latency_ms, rc, detail) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, ts if ts is not None else time.time(), kind, latency_ms, rc, detail),
            )

    def events_for(self, run_id: str, kind: str | None = None) -> list[dict]:
        with self._conn() as db:
            if kind is None:
                rows = db.execute(
                    "SELECT * FROM events WHERE run_id=? ORDER BY ts", (run_id,)).fetchall()
            else:
                rows = db.execute(
                    "SELECT * FROM events WHERE run_id=? AND kind=? ORDER BY ts",
                    (run_id, kind)).fetchall()
        return [dict(r) for r in rows]

    def latency_summary(self, run_id: str, kind: str | None = None) -> dict:
        """p50/p95/p99/max/mean over event latencies, plus a non-zero rc count."""
        rows = self.events_for(run_id, kind)
        lats = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
        errs = sum(1 for r in rows if r["rc"] not in (None, 0))
        return {**_quantiles(lats), "n": len(rows), "errors": errs}


def _run_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["params"] = json.loads(d.pop("params_json") or "{}")
    d["summary"] = json.loads(d.pop("summary_json") or "{}")
    return d


def _quantiles(sorted_vals: list[float]) -> dict:
    """p50/p95/p99/max/mean over an already-sorted list (nearest-rank)."""
    n = len(sorted_vals)
    if not n:
        return {"p50": None, "p95": None, "p99": None, "max": None, "mean": None}

    def pct(p: float) -> float:
        # nearest-rank: index = ceil(p/100 * n) - 1, clamped
        idx = max(0, min(n - 1, int(p / 100.0 * n + 0.999999) - 1))
        return sorted_vals[idx]

    return {"p50": pct(50), "p95": pct(95), "p99": pct(99),
            "max": sorted_vals[-1], "mean": sum(sorted_vals) / n}
