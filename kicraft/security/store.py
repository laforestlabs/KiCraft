"""SQLite store for security-scan results (one row per finding).

A finding is keyed by a stable ``fingerprint`` (tool+rule+location+message) so
re-running a scan updates ``last_seen`` and PRESERVES a human's acknowledged
status instead of resurfacing it as new. Mirrors the connection-per-op / WAL
shape of the rest of the codebase; surfaced on /admin/security.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS scans (
    scan_id      TEXT PRIMARY KEY,
    tool         TEXT,
    started_at   REAL,
    finished_at  REAL,
    status       TEXT,
    summary_json TEXT
);
CREATE TABLE IF NOT EXISTS findings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    fingerprint TEXT UNIQUE,
    tool        TEXT,
    severity    TEXT,
    rule        TEXT,
    location    TEXT,
    message     TEXT,
    status      TEXT DEFAULT 'open',
    first_seen  REAL,
    last_seen   REAL
);
CREATE INDEX IF NOT EXISTS idx_findings_tool ON findings(tool, status);
"""

# Coarse severity ordering for sorting / charts.
SEVERITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4, "unknown": 5}


def default_store_path() -> Path:
    root = os.environ.get("KICRAFT_SECURITY_DIR", "").strip()
    base = Path(root) if root else Path.home() / ".kicraft" / "security"
    return base / "security.db"


def fingerprint(tool: str, rule: str, location: str, message: str) -> str:
    blob = "|".join((tool or "", rule or "", location or "", (message or "")[:200]))
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class SecurityResultStore:
    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path else default_store_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as db:
            db.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        db = sqlite3.connect(str(self.path), timeout=30.0)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA journal_mode=WAL")
        return db

    # --- scans -------------------------------------------------------------
    def start_scan(self, scan_id: str, tool: str) -> None:
        with self._conn() as db:
            db.execute(
                "INSERT OR REPLACE INTO scans (scan_id, tool, started_at, status) "
                "VALUES (?, ?, ?, 'running')", (scan_id, tool, time.time()))

    def finish_scan(self, scan_id: str, status: str, summary: dict | None = None) -> None:
        with self._conn() as db:
            db.execute(
                "UPDATE scans SET finished_at=?, status=?, summary_json=? WHERE scan_id=?",
                (time.time(), status, json.dumps(summary or {}), scan_id))

    def list_scans(self, limit: int = 100) -> list[dict]:
        with self._conn() as db:
            rows = db.execute("SELECT * FROM scans ORDER BY started_at DESC LIMIT ?",
                              (limit,)).fetchall()
        return [_scan_row(r) for r in rows]

    # --- findings ----------------------------------------------------------
    def record_finding(self, *, tool: str, severity: str, rule: str, location: str,
                       message: str) -> str:
        """Upsert a finding by fingerprint, preserving an existing ack status."""
        fp = fingerprint(tool, rule, location, message)
        now = time.time()
        sev = (severity or "unknown").lower()
        with self._conn() as db:
            db.execute(
                "INSERT INTO findings (fingerprint, tool, severity, rule, location, "
                "message, status, first_seen, last_seen) "
                "VALUES (?, ?, ?, ?, ?, ?, 'open', ?, ?) "
                "ON CONFLICT(fingerprint) DO UPDATE SET last_seen=excluded.last_seen, "
                "severity=excluded.severity, message=excluded.message",
                (fp, tool, sev, rule, location, message, now, now))
        return fp

    def set_status(self, finding_id: int, status: str) -> None:
        with self._conn() as db:
            db.execute("UPDATE findings SET status=? WHERE id=?", (status, finding_id))

    def list_findings(self, tool: str | None = None, status: str | None = None) -> list[dict]:
        q = "SELECT * FROM findings"
        clauses, params = [], []
        if tool:
            clauses.append("tool=?"); params.append(tool)
        if status:
            clauses.append("status=?"); params.append(status)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        with self._conn() as db:
            rows = db.execute(q, params).fetchall()
        return sorted((dict(r) for r in rows),
                      key=lambda f: (SEVERITY_ORDER.get(f["severity"], 9), f["tool"]))

    def severity_counts(self, status: str | None = "open") -> dict:
        rows = self.list_findings(status=status)
        out: dict[str, int] = {}
        for r in rows:
            out[r["severity"]] = out.get(r["severity"], 0) + 1
        return out

    def status_counts(self) -> dict:
        with self._conn() as db:
            rows = db.execute(
                "SELECT status, COUNT(*) c FROM findings GROUP BY status").fetchall()
        return {r["status"]: r["c"] for r in rows}


def _scan_row(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["summary"] = json.loads(d.pop("summary_json") or "{}")
    return d
