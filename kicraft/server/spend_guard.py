"""The spend gate: the single enforcement point for model cost (plan B0).

Every model call runs `preflight()` with its configured call-cost ceiling
(refuse if the remaining global/project budget cannot cover it or when
kill-switched), then spends, then `record()` stores actual cost. Ceilings use a
persistent SQLite ledger shared across worker processes, so bounded calls do not
cross a configured budget merely because accounting happens after dispatch.
"""

from __future__ import annotations

import datetime as dt
import json
import sqlite3
from pathlib import Path

from .config import Settings


class BudgetExceeded(RuntimeError):
    """Raised by `preflight()` when a spend ceiling has been reached."""

    def __init__(
        self,
        message: str,
        *,
        scope: str | None = None,
        spent_usd: float | None = None,
        limit_usd: float | None = None,
        call_ceiling_usd: float | None = None,
        run_id: str | None = None,
    ):
        super().__init__(message)
        self.scope = scope
        self.spent_usd = spent_usd
        self.limit_usd = limit_usd
        self.call_ceiling_usd = call_ceiling_usd
        self.run_id = run_id


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
                "expanded_component_count INTEGER,"
                "work_units INTEGER,"
                "reused_work_units INTEGER,"
                "aggregate_repair_rounds INTEGER)"
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
                "expanded_component_count INTEGER",
                "work_units INTEGER",
                "reused_work_units INTEGER",
                "aggregate_repair_rounds INTEGER",
            ):
                name = column.split()[0]
                if name not in cols:
                    conn.execute(f"ALTER TABLE stage_runs ADD COLUMN {column}")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS stage_attempts ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "ts TEXT NOT NULL,"
                "run_id TEXT,"
                "stage TEXT NOT NULL,"
                "attempt INTEGER NOT NULL,"
                "call_mode TEXT NOT NULL,"
                "model TEXT,"
                "provider TEXT,"
                "finish_reason TEXT,"
                "outcome TEXT NOT NULL,"
                "http_status INTEGER,"
                "error_code TEXT,"
                "request_id TEXT,"
                "wall_s REAL,"
                "input_tokens INTEGER,"
                "output_tokens INTEGER,"
                "cost_usd REAL,"
                "diagnostic_codes TEXT,"
                "unit_id TEXT,"
                "unit_attempt INTEGER,"
                "aggregate_round INTEGER,"
                "commit_gate_codes TEXT,"
                "offender_count INTEGER,"
                "rejection_signature TEXT,"
                "provider_profile TEXT,"
                "fallback_reason TEXT,"
                "candidate_retained INTEGER,"
                "reasoning_failure_kind TEXT,"
                "schema_error TEXT,"
                "failure_detail TEXT,"
                "response_chars INTEGER,"
                "response_format_mode TEXT)"
            )
            attempt_cols = {row[1] for row in conn.execute("PRAGMA table_info(stage_attempts)")}
            for column in (
                "unit_id TEXT",
                "unit_attempt INTEGER",
                "aggregate_round INTEGER",
                "commit_gate_codes TEXT",
                "offender_count INTEGER",
                "rejection_signature TEXT",
                "provider_profile TEXT",
                "fallback_reason TEXT",
                "candidate_retained INTEGER",
                "reasoning_failure_kind TEXT",
                "schema_error TEXT",
                "failure_detail TEXT",
                "response_chars INTEGER",
                "response_format_mode TEXT",
            ):
                name = column.split()[0]
                if name not in attempt_cols:
                    conn.execute(f"ALTER TABLE stage_attempts ADD COLUMN {column}")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS stage_attempts_run_stage "
                "ON stage_attempts(run_id, stage, attempt)"
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

    def spent_for_run(self, run_id: str | None) -> float:
        """Actual provider cost attributed to one exact pipeline run."""
        if not run_id:
            return 0.0
        return self._sum(
            "WHERE json_valid(meta) AND json_extract(meta, '$.run_id') = ?",
            (str(run_id),),
        )

    def spent_by_stage_for_run(self, run_id: str) -> dict[str, float]:
        """Sum billed calls, including failed attempts, without stage-run duplicates."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT json_extract(meta, '$.stage'), SUM(cost_usd) FROM spend "
                "WHERE json_valid(meta) AND json_extract(meta, '$.run_id') = ? "
                "GROUP BY json_extract(meta, '$.stage')",
                (run_id,),
            ).fetchall()
        return {str(stage or "unknown"): float(cost) for stage, cost in rows}

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

    def stage_attempt_aggregates(self, days: int = 30) -> list[dict]:
        """Redacted provider/semantic attempt aggregates for admin diagnosis."""
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT stage,model,provider,outcome,error_code,unit_id,"
                "COUNT(*) AS attempts,AVG(wall_s) AS avg_wall_s,"
                "SUM(cost_usd) AS cost_usd "
                "FROM stage_attempts WHERE ts >= ? "
                "GROUP BY stage,model,provider,outcome,error_code,unit_id "
                "ORDER BY attempts DESC,stage,model,provider,outcome,unit_id",
                (cutoff,),
            ).fetchall()
        return [
            {
                "stage": row[0],
                "model": row[1],
                "provider": row[2],
                "outcome": row[3],
                "error_code": row[4],
                **({"unit_id": row[5]} if row[5] is not None else {}),
                "attempts": int(row[6]),
                "avg_wall_s": float(row[7]) if row[7] is not None else None,
                "cost_usd": float(row[8] or 0.0),
            }
            for row in rows
        ]

    def stage_diagnostic_aggregates(self, days: int = 30) -> list[dict]:
        cutoff = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT a.stage,j.value,COUNT(*) FROM stage_attempts AS a,"
                "json_each(COALESCE(a.diagnostic_codes,'[]')) AS j "
                "WHERE a.ts >= ? GROUP BY a.stage,j.value ORDER BY COUNT(*) DESC,a.stage,j.value",
                (cutoff,),
            ).fetchall()
        return [{"stage": row[0], "code": row[1], "attempts": int(row[2])} for row in rows]

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
            "project_llm_budget_usd": float(getattr(self.s, "project_llm_budget_usd", 0.0) or 0.0),
        }

    def preflight(
        self,
        call_ceiling_usd: float = 0.0,
        run_id: str | None = None,
    ) -> None:
        """Reserve enough remaining budget for one configured bounded call."""
        if self.s.kill_switch:
            raise KillSwitchEngaged("KICRAFT_KILL_SWITCH is engaged; refusing all model calls.")
        reserve = max(0.0, float(call_ceiling_usd or 0.0))
        total = self.spent_total()
        if total >= self.s.total_usd_ceiling or total + reserve > self.s.total_usd_ceiling:
            raise BudgetExceeded(
                f"total remaining budget cannot cover call ceiling ${reserve:.4f} "
                f"(spent ${total:.4f} of ${self.s.total_usd_ceiling:.2f}); refusing.",
                scope="total",
                spent_usd=total,
                limit_usd=self.s.total_usd_ceiling,
                call_ceiling_usd=reserve,
                run_id=run_id,
            )
        day = self.spent_today()
        if day >= self.s.daily_usd_ceiling or day + reserve > self.s.daily_usd_ceiling:
            raise BudgetExceeded(
                f"daily remaining budget cannot cover call ceiling ${reserve:.4f} "
                f"(spent ${day:.4f} of ${self.s.daily_usd_ceiling:.2f}); refusing.",
                scope="daily",
                spent_usd=day,
                limit_usd=self.s.daily_usd_ceiling,
                call_ceiling_usd=reserve,
                run_id=run_id,
            )
        project_budget = float(getattr(self.s, "project_llm_budget_usd", 0.0) or 0.0)
        if run_id and project_budget > 0:
            run_spend = self.spent_for_run(run_id)
            if run_spend + reserve > project_budget:
                raise BudgetExceeded(
                    f"project run {run_id} remaining budget cannot cover call ceiling "
                    f"${reserve:.4f} (spent ${run_spend:.4f} of "
                    f"${project_budget:.2f}); refusing.",
                    scope="project",
                    spent_usd=run_spend,
                    limit_usd=project_budget,
                    call_ceiling_usd=reserve,
                    run_id=run_id,
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
        expanded_component_count: int | None = None,
        work_units: int | None = None,
        reused_work_units: int | None = None,
        aggregate_repair_rounds: int | None = None,
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
        stage_runtime._child_cpu_s). ``wall_s`` is unaffected."""
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO stage_runs (ts, run_id, stage, ok, attempts, rounds, "
                "tool_calls, wall_s, cpu_s, cost_usd, failure_kind, "
                "emitted_collection_count, expanded_component_count, work_units, "
                "reused_work_units, aggregate_repair_rounds) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
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
                    int(expanded_component_count) if expanded_component_count is not None else None,
                    int(work_units) if work_units is not None else None,
                    int(reused_work_units) if reused_work_units is not None else None,
                    int(aggregate_repair_rounds) if aggregate_repair_rounds is not None else None,
                ),
            )

    def record_stage_attempt(
        self,
        *,
        run_id: str | None,
        stage: str,
        attempt: int,
        call_mode: str,
        outcome: str,
        model: str | None = None,
        provider: str | None = None,
        finish_reason: str | None = None,
        http_status: int | None = None,
        error_code: str | None = None,
        request_id: str | None = None,
        wall_s: float | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cost_usd: float | None = None,
        diagnostic_codes=(),
        unit_id: str | None = None,
        unit_attempt: int | None = None,
        aggregate_round: int | None = None,
        commit_gate_codes=(),
        offender_count: int | None = None,
        rejection_signature: str | None = None,
        provider_profile: str | None = None,
        fallback_reason: str | None = None,
        candidate_retained: bool | None = None,
        reasoning_failure_kind: str | None = None,
        schema_error: str | None = None,
        failure_detail: str | None = None,
        response_chars: int | None = None,
        response_format_mode: str | None = None,
    ) -> None:
        """Append redacted per-call facts; never stores prompts or responses."""
        codes = sorted({str(code) for code in diagnostic_codes if code})
        gate_codes = list(dict.fromkeys(str(code) for code in commit_gate_codes if code))
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO stage_attempts (ts,run_id,stage,attempt,call_mode,"
                "model,provider,finish_reason,outcome,http_status,error_code,"
                "request_id,wall_s,input_tokens,output_tokens,cost_usd,diagnostic_codes,"
                "unit_id,unit_attempt,aggregate_round,commit_gate_codes,offender_count,"
                "rejection_signature,provider_profile,fallback_reason,candidate_retained,"
                "reasoning_failure_kind,schema_error,failure_detail,response_chars,"
                "response_format_mode)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    _utcnow_iso(),
                    run_id,
                    stage,
                    int(attempt),
                    str(call_mode),
                    model,
                    provider,
                    finish_reason,
                    str(outcome),
                    http_status,
                    error_code,
                    request_id,
                    wall_s,
                    input_tokens,
                    output_tokens,
                    float(cost_usd or 0.0),
                    json.dumps(codes, separators=(",", ":")),
                    unit_id,
                    int(unit_attempt) if unit_attempt is not None else None,
                    int(aggregate_round) if aggregate_round is not None else None,
                    json.dumps(gate_codes, separators=(",", ":")),
                    int(offender_count) if offender_count is not None else None,
                    rejection_signature,
                    provider_profile,
                    fallback_reason,
                    int(candidate_retained) if candidate_retained is not None else None,
                    reasoning_failure_kind,
                    schema_error,
                    failure_detail,
                    int(response_chars) if response_chars is not None else None,
                    response_format_mode,
                ),
            )
