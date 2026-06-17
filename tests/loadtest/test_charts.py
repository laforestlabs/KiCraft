"""Tests for the pure ECharts option builders (no NiceGUI needed)."""
from __future__ import annotations

from kicraft.loadtest import charts as lc
from kicraft.security import charts as sc


def _samples():
    return [
        {"ts": 100.0, "cpu_pct": 10, "mem_pct": 20, "loadavg": 0.5,
         "queue_depth": 5, "queue_running": 1, "disk_free_mb": 9000},
        {"ts": 101.0, "cpu_pct": 80, "mem_pct": 25, "loadavg": 1.5,
         "queue_depth": 3, "queue_running": 2, "disk_free_mb": 8900},
    ]


def test_host_chart_has_three_series_and_relative_x():
    opt = lc.host_chart(_samples())
    assert [s["name"] for s in opt["series"]] == ["cpu%", "mem%", "loadavg"]
    assert opt["xAxis"]["data"] == [0.0, 1.0]  # relative seconds from first sample
    assert opt["series"][0]["data"] == [10, 80]


def test_queue_and_disk_charts():
    opt = lc.queue_chart(_samples())
    assert [s["name"] for s in opt["series"]] == ["queued", "running"]
    assert opt["series"][1]["data"] == [1, 2]
    disk = lc.disk_chart(_samples())
    assert disk["series"][0]["data"] == [9000, 8900]


def test_latency_bar_and_outcome_pie_buildstorm_shape():
    summary = {"build": {"p50": 30, "p95": 35, "p99": 36, "max": 40},
               "ok": 10, "nonzero_rc": 1, "failed": 0}
    bar = lc.latency_bar(summary)
    assert bar["xAxis"]["data"] == ["p50", "p95", "p99", "max"]
    assert bar["series"][0]["data"] == [30, 35, 36, 40]
    pie = lc.outcome_pie(summary)
    names = {d["name"] for d in pie["series"][0]["data"]}
    assert "ok" in names and "failed" not in names  # zero slices dropped


def test_outcome_pie_pipeline_shape():
    pie = lc.outcome_pie({"design_ok": 8, "build_ok": 7, "errors": 1})
    names = {d["name"] for d in pie["series"][0]["data"]}
    assert names == {"design ok", "build ok", "errors"}


def test_slot_sweep_bar():
    summaries = [{"slots": 1, "build": {"p95": 60}},
                 {"slots": 2, "build": {"p95": 35}}]
    opt = lc.slot_sweep_bar(summaries)
    assert opt["xAxis"]["data"] == ["1", "2"]
    assert opt["series"][0]["data"] == [60, 35]


def test_empty_samples_do_not_crash():
    assert lc.host_chart([])["xAxis"]["data"] == []
    assert lc.latency_bar({})["series"][0]["data"] == [0, 0, 0, 0]


# --- security charts ---------------------------------------------------------
def test_severity_bar_orders_critical_first():
    opt = sc.severity_bar({"low": 155, "high": 4, "critical": 2})
    assert opt["xAxis"]["data"] == ["critical", "high", "low"]  # info/medium absent
    # each bar carries its own severity color
    assert opt["series"][0]["data"][0]["itemStyle"]["color"] == "#ef4444"


def test_status_pie_drops_zero_slices():
    opt = sc.status_pie({"open": 12, "acknowledged": 0})
    assert [d["name"] for d in opt["series"][0]["data"]] == ["open"]
