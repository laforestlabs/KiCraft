"""Read ledger spend separately from durable uncertain evaluation exposure."""

from __future__ import annotations

from pathlib import Path
import sqlite3


def strict_budget_exposure_status(ledger_path, *, run_id: str | None = None) -> dict:
    """Report reservations without treating missing usage receipts as billed spend."""
    path = Path(ledger_path).resolve()
    run_where = (
        (" WHERE json_valid(meta) AND json_extract(meta, '$.run_id') = ?", (run_id,))
        if run_id
        else ("", ())
    )
    exposure_where = (" AND run_id = ?", (run_id,)) if run_id else ("", ())
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=30) as conn:
        actual = float(
            conn.execute(
                "SELECT COALESCE(SUM(cost_usd), 0) FROM spend" + run_where[0], run_where[1]
            ).fetchone()[0]
        )
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='strict_budget_exposure'"
        ).fetchone()
        rows = (
            conn.execute(
                "SELECT state, COALESCE(SUM(ceiling_usd), 0) FROM strict_budget_exposure "
                "WHERE state IN ('reserved','uncertain')" + exposure_where[0] + " GROUP BY state",
                exposure_where[1],
            ).fetchall()
            if exists
            else []
        )
    exposure = {str(state): float(cost) for state, cost in rows}
    reserved = exposure.get("reserved", 0.0)
    uncertain = exposure.get("uncertain", 0.0)
    return {
        "ledger_spend_usd": actual,
        "reserved_exposure_usd": reserved,
        "uncertain_exposure_usd": uncertain,
        "active_exposure_usd": reserved + uncertain,
    }
