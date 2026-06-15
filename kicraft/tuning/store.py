"""sqlite results store: cache + checkpoint for the tuning loop.

Keyed by (config_hash, board, seed, mode) so an interrupted run resumes for
free: before evaluating a tuple, look it up; only a cache miss spawns the
(expensive) place+route subprocess. This also lets K grow incrementally — add a
seed, the prior seeds are already cached.

``config_hash`` is a stable hash of the canonicalized overlay (rounded floats,
sorted keys), so two numerically-equal candidates collide and share cache.
Pure stdlib (``sqlite3``); safe for concurrent readers/writers via WAL.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any, Sequence

from kicraft.tuning.evaluate import EvalResult

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evals (
    config_hash     TEXT NOT NULL,
    board           TEXT NOT NULL,
    seed            INTEGER NOT NULL,
    mode            TEXT NOT NULL,
    rc              INTEGER,
    fab_ready       INTEGER,
    shorts          INTEGER,
    unconnected     INTEGER,
    drc_total       INTEGER,
    traces          INTEGER,
    vias            INTEGER,
    total_length_mm REAL,
    wall_s          REAL,
    error           TEXT,
    created_at      TEXT DEFAULT (datetime('now')),
    PRIMARY KEY (config_hash, board, seed, mode)
);
CREATE TABLE IF NOT EXISTS configs (
    config_hash TEXT PRIMARY KEY,
    overlay_json TEXT NOT NULL,
    source      TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS generations (
    run_id        TEXT NOT NULL,
    gen           INTEGER NOT NULL,
    config_hash   TEXT NOT NULL,
    scalarization TEXT,
    j             REAL,
    is_train      INTEGER,
    fab_ready_rate REAL,
    mean_drc      REAL,
    mean_wall_s   REAL,
    created_at    TEXT DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_evals_cfg ON evals(config_hash, mode);
CREATE INDEX IF NOT EXISTS idx_gen_run ON generations(run_id, gen);
"""

_EVAL_COLS = (
    "config_hash", "board", "seed", "mode", "rc", "fab_ready", "shorts",
    "unconnected", "drc_total", "traces", "vias", "total_length_mm", "wall_s",
    "error",
)


def canonical_overlay(overlay: dict) -> dict:
    """Round floats / sort sets so numerically-equal overlays serialize equal."""
    out: dict[str, Any] = {}
    for k, v in overlay.items():
        if isinstance(v, float):
            out[k] = round(v, 4)
        elif isinstance(v, set):
            out[k] = sorted(v)
        else:
            out[k] = v
    return out


def config_hash(overlay: dict) -> str:
    blob = json.dumps(canonical_overlay(overlay), sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


class Store:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(self.path), timeout=30.0)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()

    # --- evals -------------------------------------------------------------
    def record(self, result: EvalResult) -> None:
        row = result.as_row()
        row["fab_ready"] = int(bool(row["fab_ready"]))
        cols = ", ".join(_EVAL_COLS)
        ph = ", ".join("?" for _ in _EVAL_COLS)
        self._db.execute(
            f"INSERT OR REPLACE INTO evals ({cols}) VALUES ({ph})",
            tuple(row[c] for c in _EVAL_COLS),
        )
        self._db.commit()

    def record_many(self, results: Sequence[EvalResult]) -> None:
        for r in results:
            self.record(r)

    def lookup(self, cfg_hash: str, board: str, seed: int, mode: str) -> EvalResult | None:
        cur = self._db.execute(
            "SELECT * FROM evals WHERE config_hash=? AND board=? AND seed=? AND mode=?",
            (cfg_hash, board, seed, mode),
        )
        row = cur.fetchone()
        return _row_to_result(row) if row else None

    def results_for(self, cfg_hash: str, mode: str | None = None) -> list[EvalResult]:
        if mode is None:
            cur = self._db.execute(
                "SELECT * FROM evals WHERE config_hash=?", (cfg_hash,)
            )
        else:
            cur = self._db.execute(
                "SELECT * FROM evals WHERE config_hash=? AND mode=?", (cfg_hash, mode)
            )
        return [_row_to_result(r) for r in cur.fetchall()]

    # --- configs -----------------------------------------------------------
    def upsert_config(self, cfg_hash: str, overlay: dict, source: str = "") -> None:
        self._db.execute(
            "INSERT OR IGNORE INTO configs (config_hash, overlay_json, source) "
            "VALUES (?, ?, ?)",
            (cfg_hash, json.dumps(canonical_overlay(overlay), sort_keys=True), source),
        )
        self._db.commit()

    def get_overlay(self, cfg_hash: str) -> dict | None:
        cur = self._db.execute(
            "SELECT overlay_json FROM configs WHERE config_hash=?", (cfg_hash,)
        )
        row = cur.fetchone()
        return json.loads(row["overlay_json"]) if row else None

    # --- generations log ---------------------------------------------------
    def record_generation(
        self, run_id: str, gen: int, cfg_hash: str, *, scalarization: str,
        j: float, is_train: bool, fab_ready_rate: float, mean_drc: float,
        mean_wall_s: float,
    ) -> None:
        self._db.execute(
            "INSERT INTO generations (run_id, gen, config_hash, scalarization, j, "
            "is_train, fab_ready_rate, mean_drc, mean_wall_s) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (run_id, gen, cfg_hash, scalarization, j, int(is_train),
             fab_ready_rate, mean_drc, mean_wall_s),
        )
        self._db.commit()

    def all_evaluated_hashes(self, mode: str) -> list[str]:
        cur = self._db.execute(
            "SELECT DISTINCT config_hash FROM evals WHERE mode=?", (mode,)
        )
        return [r["config_hash"] for r in cur.fetchall()]

    def close(self) -> None:
        self._db.close()


def _row_to_result(row: sqlite3.Row) -> EvalResult:
    return EvalResult(
        config_hash=row["config_hash"], board=row["board"], seed=row["seed"],
        mode=row["mode"], rc=row["rc"], fab_ready=bool(row["fab_ready"]),
        shorts=row["shorts"], unconnected=row["unconnected"],
        drc_total=row["drc_total"], traces=row["traces"], vias=row["vias"],
        total_length_mm=row["total_length_mm"], wall_s=row["wall_s"],
        error=row["error"] or "",
    )
