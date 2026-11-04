"""Tests for kicraft/loadtest/metrics.py (host/process sampler).

These exercise the /proc fallback when psutil is absent (the state on a fresh box
before the `loadtest` extra is installed) and the psutil path when present.
"""
from __future__ import annotations

import os
import sqlite3
import threading
import time
from pathlib import Path

from kicraft.loadtest import metrics
from kicraft.loadtest.store import LoadResultStore


def test_host_probe_reports_numeric_host_metrics(tmp_path):
    probe = metrics._HostProbe(tmp_path)
    s1 = probe.sample()
    time.sleep(0.05)
    s2 = probe.sample()  # second sample has a cpu delta baseline
    for key in ("mem_used_mb", "mem_pct", "loadavg", "disk_free_mb"):
        assert isinstance(s2[key], (int, float)), key
    assert s2["disk_free_mb"] > 0
    assert s2["cpu_pct"] is None or 0.0 <= s2["cpu_pct"] <= 100.0
    assert s1 is not s2


def test_rss_mb_for_current_process_is_positive():
    assert metrics.rss_mb([os.getpid()]) > 0
    assert metrics.rss_mb([]) == 0
    assert metrics.rss_mb([2 ** 31 - 1]) == 0  # nonexistent pid -> skipped


def test_find_pids_locates_this_process():
    # the pytest process cmdline contains 'python' or 'pytest'; match the exe dir
    found = metrics.find_pids(Path(os.sys.executable).name)
    assert os.getpid() in found


def test_wal_bytes_sums_existing_files(tmp_path):
    a = tmp_path / "a-wal"
    a.write_bytes(b"x" * 100)
    assert metrics.wal_bytes([a, tmp_path / "missing-wal"]) == 100
    assert metrics.wal_bytes([]) == 0


def test_lock_latency_probe(tmp_path):
    db = tmp_path / "x.db"
    assert metrics.lock_latency_ms(db) is None  # missing db
    sqlite3.connect(str(db)).execute("CREATE TABLE t(x)")
    ms = metrics.lock_latency_ms(db)
    assert isinstance(ms, float) and ms >= 0.0


def test_sampler_writes_rows_with_injected_probes(tmp_path):
    store = LoadResultStore(tmp_path / "loadtest.db")
    store.start_run("run1", "test")
    depth = {"n": 5}
    sampler = metrics.MetricsSampler(
        store, "run1", interval_s=0.1, disk_path=tmp_path,
        web_pids_probe=lambda: [os.getpid()],
        queue_probe=lambda: (depth["n"], 2),
    )
    with sampler:
        time.sleep(0.45)
    rows = store.samples_for("run1")
    assert len(rows) >= 2
    assert [r["ts"] for r in rows] == sorted(r["ts"] for r in rows)  # monotonic
    assert rows[-1]["queue_depth"] == 5 and rows[-1]["queue_running"] == 2
    assert rows[-1]["web_rss_mb"] > 0


def test_sampler_recovers_from_transient_store_failure(tmp_path, monkeypatch):
    store = LoadResultStore(tmp_path / "loadtest.db")
    store.start_run("recover", "test")
    save_sample = store.add_sample
    persisted = threading.Event()
    attempts = 0

    def fail_once_then_persist(run_id, sample):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise sqlite3.OperationalError("temporary lock")
        save_sample(run_id, sample)
        persisted.set()

    monkeypatch.setattr(store, "add_sample", fail_once_then_persist)
    with metrics.MetricsSampler(
        store, "recover", interval_s=0.01, disk_path=tmp_path,
        queue_probe=lambda: (7, 1),
    ):
        assert persisted.wait(timeout=3)
    rows = store.samples_for("recover")
    assert attempts >= 2
    assert rows[0]["queue_depth"] == 7
    assert rows[0]["queue_running"] == 1



