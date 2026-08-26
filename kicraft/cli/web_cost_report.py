#!/usr/bin/env python3
"""Summarize the web server's real OpenRouter spend from its spend ledger.

This is the web counterpart to ``kicraft/cli/token_report.py`` (which prices
Claude Code transcripts for the offline skill path). The web app records the
*actual* billed cost of every model call into the SQLite spend ledger
(``~/.kicraft/spend_ledger.db``) via ``kicraft.server.spend_guard``. Newer rows
carry a structured JSON ``meta`` (run_id / stage / attempt / provider /
cached_tokens / finish_reason); older rows carry a bare phase string and are
still summarized (grouped by a time-gap fallback).

It answers the questions that matter for keeping the hosted service cheap:

  * what did each design run cost, broken down by stage?
  * is prompt caching actually engaging (cache hit-rate > 0)?
  * which backend served each call, and did any call exceed a price ceiling
    (the non-deterministic provider-routing spike that pinning is meant to kill)?

Pure stdlib (sqlite3 + json), so it has no kicraft/pcbnew/pydantic imports.

    web-cost-report                         # ~/.kicraft/spend_ledger.db, by run
    web-cost-report /path/to/ledger.db --by stage
    web-cost-report --since 2026-06-04 --json
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DEFAULT_LEDGER = Path.home() / ".kicraft" / "spend_ledger.db"
# A call whose implied input price exceeds this ($/Mtok) is flagged as a routing
# spike: deepseek-v4-flash's cheap backends bill ~0.02-0.13/Mtok, so anything
# well above that means OpenRouter routed to a premium (and non-caching) backend.
DEFAULT_SPIKE_THRESHOLD = 0.50
_GAP_SECONDS = 300  # legacy rows (no run_id): a >5min gap starts a new run


def _parse_meta(meta) -> dict:
    """A ledger ``meta`` cell -> dict. JSON object as-is; bare string -> {phase}."""
    if isinstance(meta, str) and meta[:1] == "{":
        try:
            d = json.loads(meta)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
    return {"phase": meta or ""}


def load_rows(db_path, since=None) -> list[dict]:
    """Rows from the ledger as dicts with the meta pre-parsed, oldest first."""
    conn = sqlite3.connect(str(db_path))
    try:
        q = "SELECT id, ts, model, input_tokens, output_tokens, cost_usd, meta FROM spend"
        params: tuple = ()
        if since:
            q += " WHERE ts >= ?"
            params = (since,)
        q += " ORDER BY ts"
        cur = conn.execute(q, params)
        rows = []
        for rid, ts, model, intok, outtok, cost, meta in cur.fetchall():
            m = _parse_meta(meta)
            rows.append(
                {
                    "id": rid,
                    "ts": ts,
                    "model": model,
                    "input_tokens": int(intok or 0),
                    "output_tokens": int(outtok or 0),
                    "cost_usd": float(cost or 0.0),
                    "meta": m,
                    "cached_tokens": int(m.get("cached_tokens") or 0),
                    "provider": m.get("provider") or "?",
                    "stage": m.get("stage") or m.get("phase") or "?",
                }
            )
        return rows
    finally:
        conn.close()


def _run_key(row, fallback_idx) -> str:
    rid = row["meta"].get("run_id")
    return str(rid) if rid else fallback_idx


def _assign_runs(rows) -> None:
    """Set row['run'] from meta.run_id, or a time-gap cluster id for legacy rows.

    Legacy rows (no run_id) are grouped by a >5min gap and share the cluster's
    START timestamp as the label, so one run is not split into many."""
    prev_ts = None
    legacy_n = 0
    legacy_label = "legacy#0"
    for r in rows:
        ts = datetime.fromisoformat(r["ts"])
        if r["meta"].get("run_id"):
            r["run"] = str(r["meta"]["run_id"])
        else:
            if prev_ts is None or (ts - prev_ts).total_seconds() > _GAP_SECONDS:
                legacy_n += 1
                legacy_label = f"legacy#{legacy_n} ({r['ts'][:16]})"
            r["run"] = legacy_label
        prev_ts = ts


def _bucket():
    return {
        "calls": 0,
        "input": 0,
        "output": 0,
        "cached": 0,
        "cost": 0.0,
        "spikes": 0,
        "truncations": 0,
    }


def _add(b, r, spike_threshold):
    b["calls"] += 1
    b["input"] += r["input_tokens"]
    b["output"] += r["output_tokens"]
    b["cached"] += r["cached_tokens"]
    b["cost"] += r["cost_usd"]
    if (
        r["input_tokens"]
        and (r["cost_usd"] / r["input_tokens"] * 1e6) > spike_threshold
        and r["output_tokens"] < 400
    ):
        b["spikes"] += 1
    if r["meta"].get("finish_reason") == "length":
        b["truncations"] += 1


def summarize(rows, spike_threshold=DEFAULT_SPIKE_THRESHOLD) -> dict:
    """Aggregate the ledger rows into all-time totals + per-run/stage/provider."""
    _assign_runs(rows)
    runs: dict = defaultdict(_bucket)
    run_stage: dict = defaultdict(lambda: defaultdict(_bucket))
    providers: dict = defaultdict(_bucket)
    total = _bucket()
    for r in rows:
        _add(total, r, spike_threshold)
        _add(runs[r["run"]], r, spike_threshold)
        _add(run_stage[r["run"]][r["stage"]], r, spike_threshold)
        _add(providers[r["provider"]], r, spike_threshold)
    return {
        "total": total,
        "runs": dict(runs),
        "run_stage": {k: dict(v) for k, v in run_stage.items()},
        "providers": dict(providers),
        "spike_threshold": spike_threshold,
        "n_rows": len(rows),
    }


def _hit_rate(b) -> float:
    return (b["cached"] / b["input"] * 100.0) if b["input"] else 0.0


def format_report(summary, by="run") -> str:
    t = summary["total"]
    out = []
    out.append("=" * 72)
    out.append("  KiCraft web spend  ({} calls, {} runs)".format(t["calls"], len(summary["runs"])))
    out.append("=" * 72)
    out.append("  Total cost     ${:.4f}".format(t["cost"]))
    out.append(
        "  Input tokens   {:>14,}   (cache hit-rate {:.1f}%)".format(t["input"], _hit_rate(t))
    )
    out.append("  Output tokens  {:>14,}".format(t["output"]))
    out.append(
        "  Price spikes   {:>14}   (calls > ${:.2f}/Mtok input)".format(
            t["spikes"], summary["spike_threshold"]
        )
    )
    out.append("  Truncations    {:>14}   (output hit the token cap)".format(t["truncations"]))

    if by in ("provider", "all"):
        out.append("  " + "-" * 68)
        out.append("  By provider:")
        for prov, b in sorted(summary["providers"].items(), key=lambda kv: -kv[1]["cost"]):
            out.append(
                "    {:<18} ${:>8.4f}  {:>4} calls  hit {:>5.1f}%  spikes {}".format(
                    str(prov)[:18], b["cost"], b["calls"], _hit_rate(b), b["spikes"]
                )
            )

    if by in ("run", "stage", "all"):
        out.append("  " + "-" * 68)
        out.append("  By run:")
        for run, b in sorted(summary["runs"].items(), key=lambda kv: -kv[1]["cost"]):
            out.append(
                "    {:<26} ${:>8.4f}  {:>4} calls  hit {:>5.1f}%  spk {}  trunc {}".format(
                    str(run)[:26],
                    b["cost"],
                    b["calls"],
                    _hit_rate(b),
                    b["spikes"],
                    b["truncations"],
                )
            )
            if by in ("stage", "all"):
                for stage, sb in sorted(
                    summary["run_stage"][run].items(), key=lambda kv: -kv[1]["cost"]
                ):
                    out.append(
                        "        {:<22} ${:>8.4f}  {:>3} calls  in {:>8,}  hit {:>5.1f}%".format(
                            str(stage)[:22], sb["cost"], sb["calls"], sb["input"], _hit_rate(sb)
                        )
                    )
    out.append("")
    return "\n".join(out)


def load_stage_runs(db_path, since=None) -> list[dict]:
    """Per-stage resource rows from the ledger's ``stage_runs`` table (a stage's
    wall_s/cpu_s/rounds/tool_calls/cost, one row per completed stage). Older
    ledgers predate the table -> returns []; ledgers that predate the
    ``failure_kind`` column return rows with ``failure_kind`` unset (legacy
    rows stay readable as unclassified)."""
    conn = sqlite3.connect(str(db_path))
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(stage_runs)")}
        if not cols:  # no stage_runs table on this ledger
            return []
        optional = [
            name
            for name in (
                "failure_kind",
                "emitted_collection_count",
                "expanded_component_count",
            )
            if name in cols
        ]
        base = [
            "ts",
            "run_id",
            "stage",
            "ok",
            "attempts",
            "rounds",
            "tool_calls",
            "wall_s",
            "cpu_s",
            "cost_usd",
        ]
        q = f"SELECT {', '.join(base + optional)} FROM stage_runs"
        params: tuple = ()
        if since:
            q += " WHERE ts >= ?"
            params = (since,)
        q += " ORDER BY ts"
        rows = []
        for values in conn.execute(q, params).fetchall():
            row = dict(zip(base + optional, values))
            rows.append(
                {
                    **row,
                    "ok": bool(row["ok"]),
                    "failure_kind": row.get("failure_kind"),
                    "emitted_collection_count": row.get("emitted_collection_count"),
                    "expanded_component_count": row.get("expanded_component_count"),
                }
            )
        return rows
    except sqlite3.OperationalError:  # no stage_runs table on this ledger
        return []
    finally:
        conn.close()


def summarize_stage_runs(rows) -> dict:
    """Aggregate stage_runs into per-stage {n, wall_s, cpu_s, cost, rounds,
    tool_calls, attempts, fails, failure_kinds}. This is the resource breakdown
    that lets you see which stages burn wall time vs CPU vs LLM tokens side by
    side, plus the terminal failure-kind distribution (why stages fail:
    invalid_json vs truncated_json vs commit_rejected vs ...). Legacy rows
    without a failure_kind contribute to fails but not to any kind bucket."""
    agg: dict[str, dict] = {}
    for r in rows:
        s = agg.setdefault(
            r["stage"],
            {
                "n": 0,
                "wall_s": 0.0,
                "cpu_s": 0.0,
                "cost": 0.0,
                "rounds": 0,
                "tool_calls": 0,
                "attempts": 0,
                "fails": 0,
                "failure_kinds": {},
            },
        )
        s["n"] += 1
        s["wall_s"] += r["wall_s"] or 0.0
        s["cpu_s"] += r["cpu_s"] or 0.0
        s["cost"] += r["cost_usd"] or 0.0
        s["rounds"] += r["rounds"] or 0
        s["tool_calls"] += r["tool_calls"] or 0
        s["attempts"] += r["attempts"] or 0
        if not r["ok"]:
            s["fails"] += 1
            kind = r.get("failure_kind")
            if kind:
                s["failure_kinds"][kind] = s["failure_kinds"].get(kind, 0) + 1
    return agg


def format_stage_runs(rows) -> str:
    agg = summarize_stage_runs(rows)
    if not agg:
        return ""
    out = [
        "",
        "  " + "-" * 68,
        "  By stage (wall / CPU / LLM cost / tool rounds, all runs):",
        "    {:<14} {:>4} {:>9} {:>8} {:>9} {:>6} {:>6} {:>4}".format(
            "stage", "n", "wall_s", "cpu_s", "cost", "rounds", "tools", "fail"
        ),
    ]
    for stage, s in sorted(agg.items(), key=lambda kv: -kv[1]["wall_s"]):
        cpu_pct = (s["cpu_s"] / s["wall_s"] * 100.0) if s["wall_s"] else 0.0
        out.append(
            "    {:<14} {:>4} {:>9.1f} {:>8.2f} {:>9.4f} {:>6} {:>6} {:>3}".format(
                str(stage)[:14],
                s["n"],
                s["wall_s"],
                s["cpu_s"],
                s["cost"],
                s["rounds"] or "-",
                s["tool_calls"] or "-",
                s["fails"],
            )
        )
        out.append(
            "      {:<14} cpu/wall {:.0f}%  mean wall {:.1f}s".format(
                "", cpu_pct, (s["wall_s"] / s["n"]) if s["n"] else 0.0
            )
        )
        kinds = s["failure_kinds"]
        if kinds:
            out.append(
                "      {:<14} failure kinds: {}".format(
                    "", ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
                )
            )
    out.append(
        "    note: cpu_s is process-global (RUSAGE_CHILDREN) and reliable "
        "only when designs run serially; wall_s is always accurate."
    )
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Summarize the web server's OpenRouter spend ledger by "
        "run / stage / provider, with cache hit-rate and price-spike flags."
    )
    ap.add_argument(
        "ledger",
        nargs="?",
        default=str(DEFAULT_LEDGER),
        help=f"path to spend_ledger.db (default: {DEFAULT_LEDGER})",
    )
    ap.add_argument(
        "--by",
        choices=["run", "stage", "provider", "all"],
        default="run",
        help="grouping detail (default: run)",
    )
    ap.add_argument(
        "--since",
        metavar="ISO_TS",
        help="only count calls at/after this ISO timestamp (e.g. 2026-06-04)",
    )
    ap.add_argument(
        "--spike-threshold",
        type=float,
        default=DEFAULT_SPIKE_THRESHOLD,
        help="flag calls whose input $/Mtok exceeds this (default 0.50)",
    )
    ap.add_argument("--json", action="store_true", help="emit the summary as JSON")
    args = ap.parse_args(argv)

    if not os.path.isfile(args.ledger):
        print(f"error: ledger not found: {args.ledger}", file=sys.stderr)
        return 2
    rows = load_rows(args.ledger, since=args.since)
    if not rows:
        print(
            "no spend rows found" + (f" since {args.since}" if args.since else ""), file=sys.stderr
        )
        return 0
    summary = summarize(rows, spike_threshold=args.spike_threshold)
    stage_runs = load_stage_runs(args.ledger, since=args.since)
    summary["stage_runs"] = summarize_stage_runs(stage_runs)
    if args.json:
        print(json.dumps(summary, indent=2, default=str))
    else:
        print(format_report(summary, by=args.by))
        extra = format_stage_runs(stage_runs)
        if extra:
            print(extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())
