"""Tests for kicraft/loadtest/pipeline_load.py.

The orchestration (concurrency, build gate, event recording) is tested with fast
fakes; a KiCad-guarded test proves the real mock pipeline drives stage-prep/commit
under concurrency at $0.
"""
from __future__ import annotations

import shutil
import threading
import time
from pathlib import Path

import pytest

from kicraft.loadtest import pipeline_load
from kicraft.loadtest.mockllm import load_transcript
from kicraft.loadtest.store import LoadResultStore

_FIXTURE = (Path(__file__).resolve().parents[2] / "kicraft" / "loadtest"
            / "fixtures" / "transcript_usb_pd_trigger.json")


def test_orchestration_respects_build_slot_cap(tmp_path):
    live = {"n": 0, "max": 0}
    lock = threading.Lock()

    def fake_design(ws, brief, transcript):
        return {"status": "ok"}

    def fake_build(ws, timeout_s):
        with lock:
            live["n"] += 1
            live["max"] = max(live["max"], live["n"])
        time.sleep(0.05)
        with lock:
            live["n"] -= 1
        return {"rc": 0, "wall_s": 0.05}

    store = LoadResultStore(tmp_path / "l.db")
    store.start_run("r", "pipeline")
    summary = pipeline_load.run_pipeline_load(
        ["b"] * 6, parallel=4, build_slots=2, transcript={}, work_root=tmp_path / "w",
        design_fn=fake_design, build_fn=fake_build, store=store, run_id="r")
    assert summary["design_ok"] == 6 and summary["build_ok"] == 6
    assert live["max"] <= 2  # build-slot gate honored even with parallel=4
    # events recorded for the dashboard
    assert store.latency_summary("r", "design")["n"] == 6
    assert store.latency_summary("r", "build")["n"] == 6


def test_failed_design_skips_build(tmp_path):
    def fake_design(ws, brief, transcript):
        return {"status": "failed"}

    summary = pipeline_load.run_pipeline_load(
        ["b"] * 3, parallel=2, build_slots=2, transcript={}, work_root=tmp_path / "w",
        design_fn=fake_design, build_fn=lambda ws, t: pytest.fail("build must not run"))
    assert summary["design_ok"] == 0 and summary["build_ok"] == 0


def test_one_brief_blowing_up_does_not_sink_the_run(tmp_path):
    def flaky_design(ws, brief, transcript):
        if brief == "boom":
            raise RuntimeError("kaboom")
        return {"status": "ok"}

    summary = pipeline_load.run_pipeline_load(
        ["ok", "boom", "ok"], parallel=2, build_slots=1, transcript={},
        work_root=tmp_path / "w", do_build=False, design_fn=flaky_design)
    assert summary["errors"] == 1 and summary["design_ok"] == 2


def _kicad_available() -> bool:
    if shutil.which("kicad-cli") is None:
        return False
    try:
        import pcbnew  # noqa: F401
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _kicad_available(), reason="kicad-cli / pcbnew not available")
def test_real_mock_pipeline_runs_concurrently_at_zero_cost(tmp_path, monkeypatch):
    """Two concurrent mock designs both commit every stage, no spend -- the
    concurrency proof for the design pipeline (build skipped: it is the heavy,
    separately-tested part)."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    transcript = load_transcript(_FIXTURE)
    summary = pipeline_load.run_pipeline_load(
        ["a usb-c pd trigger"] * 2, parallel=2, build_slots=1, transcript=transcript,
        work_root=tmp_path / "w", do_build=False)
    assert summary["design_ok"] == 2 and summary["errors"] == 0
