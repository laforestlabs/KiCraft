"""Tests for kicraft/loadtest/buildstorm.py.

Drives the REAL build queue + BuildWorker with a fast fake replay command, so the
queue mechanics (FIFO claim, slot cap, finish) are exercised without a real
place+route. No prod DB is touched (throwaway AccountStore on tmp).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from kicraft.loadtest import buildstorm
from kicraft.server.accounts import AccountStore

# A fake "replay": exits 0 quickly, ignores the workspace contents.
_FAST_OK = [sys.executable, "-c", "import time; time.sleep(0.05)"]
_FAST_FAIL = [sys.executable, "-c", "raise SystemExit(3)"]


def _store(tmp_path) -> AccountStore:
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


def _fake_source(tmp_path) -> Path:
    """A minimal synthesized-looking workspace: .kicraft/state.json + generated/STEM."""
    ws = tmp_path / "src"
    (ws / ".kicraft").mkdir(parents=True)
    (ws / ".kicraft" / "state.json").write_text(json.dumps({"project_stem": "DEMO"}))
    (ws / "generated" / "DEMO").mkdir(parents=True)
    return ws


def test_detect_stem_and_replay_command(tmp_path):
    src = _fake_source(tmp_path)
    assert buildstorm.detect_stem(src) == "DEMO"
    cmd = buildstorm.replay_command("DEMO", route=False)
    assert "replay" in cmd and "generated/DEMO" in cmd and "--no-route" in cmd
    assert "--route" in buildstorm.replay_command("DEMO", route=True)


def test_detect_stem_rejects_ambiguous_generated(tmp_path):
    ws = tmp_path / "src"
    (ws / "generated" / "A").mkdir(parents=True)
    (ws / "generated" / "B").mkdir(parents=True)
    with pytest.raises(ValueError):
        buildstorm.detect_stem(ws)


def test_stage_workspaces_makes_independent_copies(tmp_path):
    src = _fake_source(tmp_path)
    wss = buildstorm.stage_workspaces(src, 3, tmp_path / "work")
    assert len(wss) == 3
    for ws in wss:
        assert (ws / ".kicraft" / "state.json").is_file()
        assert (ws / "generated" / "DEMO").is_dir()
    # mutating one copy must not affect another
    (wss[0] / "marker").write_text("x")
    assert not (wss[1] / "marker").exists()


def test_storm_drains_queue_without_exceeding_slots(tmp_path):
    store = _store(tmp_path)
    src = _fake_source(tmp_path)
    wss = buildstorm.stage_workspaces(src, 6, tmp_path / "work")
    ticks = []
    summary = buildstorm.run_storm(
        store, wss, command=_FAST_OK, max_jobs=2, timeout_s=30, poll_s=0.02,
        on_tick=lambda running, depth: ticks.append(running))
    assert summary["n"] == 6
    assert summary["ok"] == 6 and summary["failed"] == 0
    assert summary["max_running"] <= 2  # the slot cap is never violated
    assert max(ticks) <= 2
    # every job recorded a wait and a wall time
    assert all(j["wait_s"] is not None and j["wall_s"] is not None for j in summary["jobs"])
    assert summary["build"]["p95"] is not None


def test_storm_counts_nonzero_rc_builds(tmp_path):
    store = _store(tmp_path)
    src = _fake_source(tmp_path)
    wss = buildstorm.stage_workspaces(src, 3, tmp_path / "work")
    summary = buildstorm.run_storm(store, wss, command=_FAST_FAIL, max_jobs=2,
                                   timeout_s=30, poll_s=0.02)
    # a nonzero exit is a completed build (status done, rc!=0), not a failed job
    assert summary["ok"] == 0
    assert summary["nonzero_rc"] == 3
    assert summary["failed"] == 0


def test_storm_aborts_cleanly_on_tick_signal(tmp_path):
    """An on_tick that raises (the abort-file path) stops the storm cleanly and
    kills in-flight builds, returning aborted=True instead of orphaning processes."""
    store = _store(tmp_path)
    src = _fake_source(tmp_path)
    wss = buildstorm.stage_workspaces(src, 4, tmp_path / "work")
    slow = [sys.executable, "-c", "import time; time.sleep(30)"]
    calls = {"n": 0}

    def abort_after_first(running, depth):
        calls["n"] += 1
        if calls["n"] >= 1:
            raise KeyboardInterrupt("abort file appeared")

    summary = buildstorm.run_storm(store, wss, command=slow, max_jobs=2,
                                   timeout_s=30, poll_s=0.05, on_tick=abort_after_first)
    assert summary["aborted"] is True
    # in-flight builds were killed (not left running to time out)
    assert store.count_running_builds() == 0


def test_sweep_slots_runs_each_setting(tmp_path):
    src = _fake_source(tmp_path)
    monkey_cmd = _FAST_OK

    def store_factory(slots):
        return _store(tmp_path / f"s{slots}")

    # patch replay_command so the sweep uses the fast fake instead of real replay
    orig = buildstorm.replay_command
    buildstorm.replay_command = lambda *a, **k: monkey_cmd
    try:
        results = buildstorm.sweep_slots(
            store_factory, src, 4, [1, 2], work_root=tmp_path / "sweep",
            route=False, timeout_s=30)
    finally:
        buildstorm.replay_command = orig
    assert [r["slots"] for r in results] == [1, 2]
    assert all(r["ok"] == 4 for r in results)
    assert results[0]["max_running"] <= 1 and results[1]["max_running"] <= 2
