"""Tests for kicraft/loadtest/store.py (LoadResultStore)."""
from __future__ import annotations

from kicraft.loadtest.store import LoadResultStore, _quantiles


def _store(tmp_path):
    return LoadResultStore(tmp_path / "loadtest.db")


def test_run_lifecycle_round_trips(tmp_path):
    s = _store(tmp_path)
    s.start_run("r1", "build-storm", {"n": 12, "slots": 2}, started_at=100.0)
    run = s.get_run("r1")
    assert run["scenario"] == "build-storm"
    assert run["params"] == {"n": 12, "slots": 2}
    assert run["finished_at"] is None
    s.finish_run("r1", {"ok": True, "p95": 1234}, finished_at=200.0)
    run = s.get_run("r1")
    assert run["finished_at"] == 200.0
    assert run["summary"]["p95"] == 1234


def test_list_runs_newest_first(tmp_path):
    s = _store(tmp_path)
    s.start_run("old", "pipeline", started_at=1.0)
    s.start_run("new", "pipeline", started_at=2.0)
    assert [r["run_id"] for r in s.list_runs()] == ["new", "old"]


def test_samples_round_trip_and_order(tmp_path):
    s = _store(tmp_path)
    s.start_run("r", "x")
    s.add_sample("r", {"ts": 2.0, "cpu_pct": 50.0, "queue_depth": 3})
    s.add_sample("r", {"ts": 1.0, "cpu_pct": 10.0, "queue_depth": 1})
    rows = s.samples_for("r")
    assert [row["ts"] for row in rows] == [1.0, 2.0]  # ordered by ts
    assert rows[1]["cpu_pct"] == 50.0
    # unset sample columns are stored as NULL, not an error
    assert rows[0]["wal_bytes"] is None


def test_events_and_latency_summary(tmp_path):
    s = _store(tmp_path)
    s.start_run("r", "x")
    for i, ms in enumerate([10, 20, 30, 40, 100]):
        s.add_event("r", "build", latency_ms=ms, rc=0, ts=float(i))
    s.add_event("r", "build", latency_ms=None, rc=1)  # a failure, no latency
    summ = s.latency_summary("r", "build")
    assert summ["n"] == 6 and summ["errors"] == 1
    assert summ["max"] == 100 and summ["p50"] == 30
    # kind filter isolates event types
    s.add_event("r", "http", latency_ms=5, rc=0)
    assert s.latency_summary("r", "http")["n"] == 1


def test_quantiles_nearest_rank():
    vals = list(range(1, 101))  # 1..100 sorted
    q = _quantiles(vals)
    assert q["p50"] == 50 and q["p95"] == 95 and q["p99"] == 99 and q["max"] == 100
    assert _quantiles([])["p95"] is None
    assert _quantiles([7])["p99"] == 7
