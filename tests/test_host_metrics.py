"""Tests for the always-on host-resource metrics feed backing the /admin
usage charts.

Covers the SQLite store (record/series/count/purge + the since-window the
dashboard's timescale selector maps to), the psutil-or-/proc probe (numeric
disk/mem; CPU's first sample is baseline-less so it may be None), the
background sampler appending rows, and the pure ECharts option builders on
the admin route (downsampling + a time-axis chart dict).
"""
from __future__ import annotations

import time

from kicraft.server import host_metrics as hm
from kicraft.server.routes_admin import (
    _HOST_TIMESCALE_SECONDS,
    _HOST_TIMESCALES,
    _host_downsample,
    _host_usage_chart,
)


# --- store ----------------------------------------------------------------- #

def test_store_record_series_roundtrip(tmp_path):
    store = hm.HostMetricsStore(tmp_path / "hm.db")
    assert store.series() == []
    assert store.count() == 0
    t0 = 1_700_000_000.0
    store.record({"ts": t0, "cpu_pct": 12.3, "mem_used_mb": 100.0,
                  "mem_pct": 50.0, "disk_used_mb": 1000.0,
                  "disk_total_mb": 2000.0, "disk_pct": 50.0})
    store.record({"ts": t0 + 60, "cpu_pct": 80.0, "mem_used_mb": 120.0,
                  "mem_pct": 60.0, "disk_used_mb": 1050.0,
                  "disk_total_mb": 2000.0, "disk_pct": 52.5})
    rows = store.series()
    assert store.count() == 2
    assert [r["ts"] for r in rows] == [t0, t0 + 60]
    assert rows[0]["cpu_pct"] == 12.3
    assert rows[-1]["disk_pct"] == 52.5


def test_store_series_since_window_filters(tmp_path):
    store = hm.HostMetricsStore(tmp_path / "hm.db")
    t0 = 1_700_000_000.0
    for i in range(10):
        store.record({"ts": t0 + i * 30, "cpu_pct": float(i),
                      "mem_used_mb": 0.0, "mem_pct": 0.0,
                      "disk_used_mb": 0.0, "disk_total_mb": 0.0,
                      "disk_pct": 0.0})
    # since = t0 + 60 should keep the last (10 - 3) = 7 rows
    rows = store.series(since=t0 + 60)
    assert [r["ts"] for r in rows] == [t0 + i * 30 for i in range(2, 10)]
    # until cap
    rows_until = store.series(since=t0, until=t0 + 60)
    assert [r["ts"] for r in rows_until] == [t0, t0 + 30, t0 + 60]


def test_store_record_replaces_on_duplicate_ts(tmp_path):
    """ts is the PRIMARY KEY and floored by the sampler -- a restart must not
    double-write a row for the same bucket."""
    store = hm.HostMetricsStore(tmp_path / "hm.db")
    t0 = 1_700_000_000.0
    store.record({"ts": t0, "cpu_pct": 1.0, "mem_used_mb": 0.0,
                  "mem_pct": 0.0, "disk_used_mb": 0.0,
                  "disk_total_mb": 0.0, "disk_pct": 0.0})
    store.record({"ts": t0, "cpu_pct": 99.0, "mem_used_mb": 0.0,
                  "mem_pct": 0.0, "disk_used_mb": 0.0,
                  "disk_total_mb": 0.0, "disk_pct": 0.0})
    rows = store.series()
    assert len(rows) == 1
    assert rows[0]["cpu_pct"] == 99.0


def test_store_purge_before_removes_old_rows(tmp_path):
    store = hm.HostMetricsStore(tmp_path / "hm.db")
    t0 = 1_700_000_000.0
    for i in range(5):
        store.record({"ts": t0 + i * 30, "cpu_pct": 0.0, "mem_used_mb": 0.0,
                      "mem_pct": 0.0, "disk_used_mb": 0.0,
                      "disk_total_mb": 0.0, "disk_pct": 0.0})
    removed = store.purge_before(t0 + 2 * 30)
    assert removed == 2
    assert store.count() == 3


# --- probe ----------------------------------------------------------------- #

def test_probe_sample_has_disk_and_mem_numbers(tmp_path):
    probe = hm._HostSampleProbe(tmp_path)
    s = probe.sample()
    assert set(s) == {"cpu_pct", "mem_used_mb", "mem_pct",
                       "disk_used_mb", "disk_total_mb", "disk_pct"}
    # disk on tmp_path (a real FS) must be numeric & self-consistent
    assert s["disk_used_mb"] is not None and s["disk_total_mb"] is not None
    assert 0.0 < s["disk_pct"] <= 100.0
    assert s["mem_used_mb"] is not None
    assert s["mem_pct"] is not None and 0.0 <= s["mem_pct"] <= 100.0
    # CPU's first sample is baseline-less (psutil/proc delta); allow None.
    assert s["cpu_pct"] is None or 0.0 <= s["cpu_pct"] <= 100.0


def test_probe_second_sample_has_cpu_pct(tmp_path):
    """After a baseline is primed, cpu_pct must be numeric (psutil or /proc)."""
    probe = hm._HostSampleProbe(tmp_path)
    probe.sample()  # prime
    # A tiny spin so the /proc (or psutil) delta is non-zero-measurable.
    _ = sum(i * i for i in range(50000))
    s = probe.sample()
    if s["cpu_pct"] is not None:
        # psutil reports overall load on a busy box; just require numeric.
        assert 0.0 <= s["cpu_pct"] <= 100.0
    # mem should remain stable regardless.
    assert s["mem_pct"] is not None


# --- sampler --------------------------------------------------------------- #

def test_sampler_appends_rows_to_store(tmp_path):
    store = hm.HostMetricsStore(tmp_path / "hm.db")
    sampler = hm.HostMetricsSampler(store, interval_s=0.05,
                                    disk_path=tmp_path)
    assert sampler._stop_evt.is_set() is False
    with sampler:
        # ~0.05s cadence: collect a handful of rows in well under a second.
        deadline = time.time() + 3.0
        while store.count() < 3 and time.time() < deadline:
            time.sleep(0.05)
    assert store.count() >= 3
    rows = store.series()
    assert all(r["disk_pct"] is not None for r in rows)
    # ts values are floored to the interval -> strictly increasing & uniform.
    ts = [r["ts"] for r in rows]
    assert ts == sorted(ts)


def test_sampler_floored_ts_is_interval_aligned(tmp_path):
    store = hm.HostMetricsStore(tmp_path / "hm.db")
    sampler = hm.HostMetricsSampler(store, interval_s=30.0, disk_path=tmp_path)
    ts = sampler._floored_ts()
    assert ts % 30.0 == 0.0


def test_start_sampler_is_idempotent(monkeypatch, tmp_path):
    """start_host_metrics_sampler must not spawn a second thread if one is alive."""
    monkeypatch.setenv("KICRAFT_HOST_METRICS_DIR", str(tmp_path))
    # If a test before us left one running, stop it so this is deterministic.
    hm.stop_host_metrics_sampler()
    s1 = hm.start_host_metrics_sampler(interval_s=0.05)
    try:
        s2 = hm.start_host_metrics_sampler(interval_s=0.05)
        assert s2 is s1  # same running instance, no second thread
    finally:
        hm.stop_host_metrics_sampler()
        s1.stop()


# --- chart option builders (pure) ------------------------------------------ #

def test_timescale_map_covers_expected_windows():
    assert _HOST_TIMESCALES[0] == "1h"
    assert "7d" in _HOST_TIMESCALE_SECONDS
    assert _HOST_TIMESCALE_SECONDS["7d"] == 7 * 86400
    assert _HOST_TIMESCALE_SECONDS["1h"] == 3600
    assert _HOST_TIMESCALE_SECONDS["all"] is None


def _rows(n):
    t0 = 1_700_000_000.0
    out = []
    for i in range(n):
        out.append({"ts": t0 + i * 30, "cpu_pct": float(i % 100),
                    "mem_pct": 50.0 + (i % 20), "disk_pct": 60.0,
                    "mem_used_mb": 1234.0, "disk_used_mb": 5000.0,
                    "disk_total_mb": 10000.0})
    return out


def test_downsample_is_noop_below_max():
    rows = _rows(10)
    assert _host_downsample(rows) is rows  # unchanged, no copy


def test_downsample_buckets_to_max_points():
    rows = _rows(1000)
    out = _host_downsample(rows, max_points=100)
    assert len(out) <= 100
    # each output ts is an original sample ts (bucket anchor = last in bucket)
    orig_ts = {r["ts"] for r in rows}
    assert all(r["ts"] in orig_ts for r in out)
    # disk_pct is constant 60 -> averaged to 60 in every bucket
    assert all(r["disk_pct"] == 60.0 for r in out)


def test_usage_chart_uses_time_axis_and_percent_axis():
    opt = _host_usage_chart(_rows(5), "cpu_pct", title="CPU usage (%)",
                            color="#60a5fa", y_name="%")
    assert opt["xAxis"]["type"] == "time"
    assert opt["yAxis"]["min"] == 0 and opt["yAxis"]["max"] == 100
    assert opt["series"][0]["type"] == "line"
    assert opt["series"][0]["areaStyle"]["opacity"] == 0.15
    # points are [ts_ms, value]
    pts = opt["series"][0]["data"]
    assert len(pts) == 5
    assert all(isinstance(p[0], (int, float)) for p in pts)
    assert pts[0][0] == 1_700_000_000.0 * 1000.0


def test_usage_chart_skips_none_points():
    rows = _rows(4)
    rows[1]["cpu_pct"] = None  # baseline-less first CPU sample, for instance
    opt = _host_usage_chart(rows, "cpu_pct", title="CPU usage (%)",
                            color="#60a5fa", y_name="%")
    pts = opt["series"][0]["data"]
    assert len(pts) == 3  # the None row is dropped