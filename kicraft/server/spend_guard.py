"""The spend gate: the single enforcement point for model cost (plan B0).

Every model call runs `preflight()` (refuse if over a ceiling or kill-switched),
then spends, then `record()` the actual cost. Ceilings are checked against a
persistent SQLite ledger, so the limit survives restarts and is shared across
worker processes. Combined with a bounded `max_tokens` per call, the worst case
is a single small overshoot past a ceiling, never an unbounded bill.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from .config import Settings


class BudgetExceeded(RuntimeError):
    """Raised by `preflight()` when a spend ceiling has been reached."""


class KillSwitchEngaged(RuntimeError):
    """Raised by `preflight()` when KICRAFT_KILL_SWITCH is set."""


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _today_start_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT00:00:00+00:00")


class SpendGuard:
    def __init__(self, settings: Settings):
        self.s = settings
        self.path = Path(settings.ledger_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._conn() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS spend ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "ts TEXT NOT NULL,"
                "model TEXT,"
                "input_tokens INTEGER,"
                "output_tokens INTEGER,"
                "cost_usd REAL NOT NULL,"
                "meta TEXT)"
            )

    def _sum(self, where: str = "", params: tuple = ()) -> float:
        with self._conn() as conn:
            row = conn.execute(
                f"SELECT COALESCE(SUM(cost_usd), 0) FROM spend {where}", params
            ).fetchone()
        return float(row[0] or 0.0)

    def spent_today(self) -> float:
        return self._sum("WHERE ts >= ?", (_today_start_iso(),))

    def spent_total(self) -> float:
        return self._sum()

    def status(self) -> dict:
        day, total = self.spent_today(), self.spent_total()
        return {
            "spent_today_usd": round(day, 6),
            "spent_total_usd": round(total, 6),
            "daily_ceiling_usd": self.s.daily_usd_ceiling,
            "total_ceiling_usd": self.s.total_usd_ceiling,
            "daily_remaining_usd": round(self.s.daily_usd_ceiling - day, 6),
            "total_remaining_usd": round(self.s.total_usd_ceiling - total, 6),
            "kill_switch": self.s.kill_switch,
        }

    def preflight(self) -> None:
        """Refuse before spending if kill-switched or a ceiling is already reached."""
        if self.s.kill_switch:
            raise KillSwitchEngaged("KICRAFT_KILL_SWITCH is engaged; refusing all model calls.")
        total = self.spent_total()
        if total >= self.s.total_usd_ceiling:
            raise BudgetExceeded(
                f"total spend ${total:.4f} has reached the ceiling "
                f"${self.s.total_usd_ceiling:.2f}; refusing.")
        day = self.spent_today()
        if day >= self.s.daily_usd_ceiling:
            raise BudgetExceeded(
                f"today's spend ${day:.4f} has reached the daily ceiling "
                f"${self.s.daily_usd_ceiling:.2f}; refusing.")

    def record(self, model: str, input_tokens, output_tokens, cost_usd: float,
               meta="") -> None:
        """Append one billed call. `meta` may be a bare phase string (legacy) or a
        dict of structured context (run_id/stage/attempt/provider/cached_tokens/
        finish_reason); a dict is stored as a compact JSON blob so the cost report
        can attribute spend per run/stage/provider. Old bare-string rows still
        parse (the report treats them as {"phase": <str>})."""
        meta_str = meta if isinstance(meta, str) else json.dumps(meta, sort_keys=True, default=str)
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO spend (ts, model, input_tokens, output_tokens, cost_usd, meta) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (_utcnow_iso(), model, int(input_tokens or 0), int(output_tokens or 0),
                 float(cost_usd or 0.0), meta_str),
            )
