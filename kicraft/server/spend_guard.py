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
            # One row per completed pipeline stage (design or build): the durable
            # per-stage resource record. cost_usd duplicates the summed LLM spend
            # (present for the report's side-by-side view) but is NOT summed into
            # the spend ceiling (that lives in `spend`). wall_s/cpu_s are the
            # gap metrics: a stage's wall-clock duration and child-CPU seconds,
            # captured by the stage driver around the LLM tool loop + subprocess
            # tool calls.
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stage_runs ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "ts TEXT NOT NULL,"
                "run_id TEXT,"
                "stage TEXT NOT NULL,"
                "ok INTEGER,"
                "attempts INTEGER,"
                "rounds INTEGER,"
                "tool_calls INTEGER,"
                "wall_s REAL,"
                "cpu_s REAL,"
                "cost_usd REAL,"
                "failure_kind TEXT,"
                "emitted_collection_count INTEGER,"
                "compact_run_expanded_count INTEGER,"
                "wiring_patch_operations INTEGER)"
            )
            # Backward-compatible migration: ledgers created before failure_kind
            # existed keep their rows and gain the column via ALTER TABLE --
            # CREATE TABLE IF NOT EXISTS alone cannot add a column to an
            # existing table, and the production ledger predates this field.
            cols = {r[1] for r in conn.execute("PRAGMA table_info(stage_runs)")}
            if "failure_kind" not in cols:
                conn.execute("ALTER TABLE stage_runs ADD COLUMN failure_kind TEXT")
            for column in (
                "emitted_collection_count INTEGER",
                "compact_run_expanded_count INTEGER",
                "wiring_patch_operations INTEGER",
            ):
                name = column.split()[0]
                if name not in cols:
                    conn.execute(f"ALTER TABLE stage_runs ADD COLUMN {column}")

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

    def spent_for_project(self, project_id) -> float:
        """The total cost attributed to one web project: the sum of every model call
        tagged run_id='p<project_id>-<ts>' in `meta`. A project can span several runs
        (initial build, ERC recovery, a later reopen/continue) that all share the
        'p<id>-' prefix, so this is the project's true incremental spend -- NOT the
        global running total (`spent_total`), which is what every project's cost_usd
        used to be stamped with. Legacy bare-string meta rows carry no run_id and are
        skipped by json_valid()."""
        if project_id is None:
            return 0.0
        return self._sum(
            "WHERE json_valid(meta) AND json_extract(meta, '$.run_id') LIKE ?",
            (f"p{int(project_id)}-%",),
        )

    def spent_by_day(self, days: int = 30) -> list[tuple[str, float]]:
        """(YYYY-MM-DD, cost) for the trailing `days`, summing EVERY ledger call
        (project + non-project, e.g. eval/judge/smoketest). This is the true site
        spend and matches the OpenRouter dashboard; contrast
        AccountStore.spend_per_day, which counts only project-attributed spend.
        ts is ISO-8601 UTC, so substr(ts,1,10) slices to a calendar day."""
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).date().isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT substr(ts, 1, 10) AS d, COALESCE(SUM(cost_usd), 0) AS c "
                "FROM spend WHERE substr(ts, 1, 10) >= ? GROUP BY d ORDER BY d",
                (cutoff,),
            ).fetchall()
        return [(r[0], float(r[1] or 0.0)) for r in rows]

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
                f"${self.s.total_usd_ceiling:.2f}; refusing."
            )
        day = self.spent_today()
        if day >= self.s.daily_usd_ceiling:
            raise BudgetExceeded(
                f"today's spend ${day:.4f} has reached the daily ceiling "
                f"${self.s.daily_usd_ceiling:.2f}; refusing."
            )

    def record(self, model: str, input_tokens, output_tokens, cost_usd: float, meta="") -> None:
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
                (
                    _utcnow_iso(),
                    model,
                    int(input_tokens or 0),
                    int(output_tokens or 0),
                    float(cost_usd or 0.0),
                    meta_str,
                ),
            )

    def record_stage(
        self,
        *,
        run_id: str | None,
        stage: str,
        ok: bool,
        attempts: int | None,
        rounds: int | None,
        tool_calls: int | None,
        wall_s: float | None,
        cpu_s: float | None,
        cost_usd: float,
        failure_kind: str | None = None,
        emitted_collection_count: int | None = None,
        compact_run_expanded_count: int | None = None,
        wiring_patch_operations: int | None = None,
    ) -> None:
        """Append one completed stage to ``stage_runs`` — the durable per-stage
        resource record. ``wall_s``/``cpu_s`` are the gap metrics: a stage's
        wall-clock duration and child-CPU seconds (LLM latency + subprocess tool
        calls). ``cost_usd`` mirrors the summed LLM spend for the side-by-side
        report; it is intentionally NOT added to the ``spend`` ceiling (the
        per-call rows there already enforce it). ``failure_kind`` is the terminal
        classification of a failed stage (collection_limit / reasoning_loop /
        truncated_json / invalid_json / commit_rejected / provider_error /
        transport_error);
        None for a committed stage or a legacy row.

        Note: ``cpu_s`` comes from RUSAGE_CHILDREN, which is per-process, so it
        is only trustworthy when designs run serially — concurrent stages in the
        same web process cross-contaminate each other's child-CPU delta (see
        stage_driver._child_cpu_s). ``wall_s`` is unaffected."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO stage_runs (ts, run_id, stage, ok, attempts, rounds, "
                "tool_calls, wall_s, cpu_s, cost_usd, failure_kind, "
                "emitted_collection_count, compact_run_expanded_count, "
                "wiring_patch_operations) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    _utcnow_iso(),
                    run_id,
                    stage,
                    int(bool(ok)),
                    int(attempts) if attempts is not None else None,
                    int(rounds) if rounds is not None else None,
                    int(tool_calls) if tool_calls is not None else None,
                    float(wall_s) if wall_s is not None else None,
                    float(cpu_s) if cpu_s is not None else None,
                    float(cost_usd or 0.0),
                    str(failure_kind) if failure_kind is not None else None,
                    int(emitted_collection_count) if emitted_collection_count is not None else None,
                    int(compact_run_expanded_count)
                    if compact_run_expanded_count is not None
                    else None,
                    int(wiring_patch_operations) if wiring_patch_operations is not None else None,
                ),
            )
