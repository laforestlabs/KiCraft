"""Full-pipeline load: run the whole LLM design pipeline at $0 under concurrency.

Mirrors the self_eval parallel driver shape (ThreadPoolExecutor + a build-slot
BoundedSemaphore gate) but drives each brief with a MockClient, so stage_driver,
the subprocess stage-prep/commit calls, and -- optionally -- a real synthesize +
place+route build all run concurrently without spend. This is what proves the
daemon-thread model + queue + SQLite hold up under load.

The design step uses the mock (free); the optional build step runs the real
``kicraft build`` (deterministic synth + place+route, no LLM), gated so no more
than ``build_slots`` run at once. Both the design and build callables are
injectable so tests drive the orchestration with fast fakes.
"""
from __future__ import annotations

import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .mockllm import MockClient

_BUILD_CMD = [sys.executable, "-m", "kicraft.design.cli_app", "build",
              ".kicraft/state.json", "generated", "--no-archive"]


def _default_design(ws: Path, brief: str, transcript: dict) -> dict:
    """Drive the design stages with a per-thread MockClient (clients are not
    thread-safe to share). The pipeline imports (pydantic-backed) are deferred so
    this module is importable in a minimal env for the fake-fn orchestration tests."""
    from kicraft.server.session import run_session
    from kicraft.server.stage_driver import DESIGN_STAGES
    client = MockClient(transcript=transcript)
    return run_session(ws, brief, DESIGN_STAGES, client=client)


def _default_build(ws: Path, timeout_s: float) -> dict:
    t0 = time.time()
    proc = subprocess.run(_BUILD_CMD, cwd=str(ws), capture_output=True, text=True,
                          timeout=timeout_s)
    return {"rc": proc.returncode, "wall_s": round(time.time() - t0, 2)}


def run_pipeline_load(briefs, *, parallel: int = 3, build_slots: int = 2,
                      transcript: dict | None = None, work_root: Path,
                      do_build: bool = True, build_timeout_s: float = 1800.0,
                      store=None, run_id: str | None = None,
                      design_fn=None, build_fn=None) -> dict:
    """Run ``briefs`` through the design pipeline (mock) + optional build, with at
    most ``parallel`` designs and ``build_slots`` concurrent builds.

    Returns a summary with per-brief design status + build rc/wall. If ``store``
    and ``run_id`` are given, each brief's design and build latencies are recorded
    as events for the dashboard.
    """
    design_fn = design_fn or _default_design
    build_fn = build_fn or _default_build
    work_root = Path(work_root)
    work_root.mkdir(parents=True, exist_ok=True)
    build_gate = threading.BoundedSemaphore(max(1, build_slots))
    records: list[dict] = []
    rec_lock = threading.Lock()

    def _one(idx: int, brief: str) -> dict:
        ws = work_root / f"design_{idx:03d}"
        ws.mkdir(parents=True, exist_ok=True)
        rec = {"idx": idx, "brief": brief[:80], "design_status": None,
               "design_s": None, "build_rc": None, "build_s": None, "error": None}
        try:
            t0 = time.time()
            res = design_fn(ws, brief, transcript)
            rec["design_status"] = res.get("status")
            rec["design_s"] = round(time.time() - t0, 2)
            if store and run_id:
                store.add_event(run_id, "design", latency_ms=rec["design_s"] * 1000.0,
                                rc=0 if res.get("status") == "ok" else 1, detail=brief[:120])
            if do_build and res.get("status") == "ok":
                with build_gate:
                    b = build_fn(ws, build_timeout_s)
                rec["build_rc"], rec["build_s"] = b.get("rc"), b.get("wall_s")
                if store and run_id:
                    store.add_event(run_id, "build",
                                    latency_ms=(rec["build_s"] or 0) * 1000.0,
                                    rc=rec["build_rc"], detail=brief[:120])
        except Exception as e:  # a single brief blowing up must not sink the run
            rec["error"] = f"{type(e).__name__}: {e}"
        with rec_lock:
            records.append(rec)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        list(ex.map(lambda p: _one(*p), list(enumerate(briefs))))

    records.sort(key=lambda r: r["idx"])
    return {
        "n": len(briefs),
        "parallel": parallel,
        "build_slots": build_slots,
        "design_ok": sum(1 for r in records if r["design_status"] == "ok"),
        "build_ok": sum(1 for r in records if r["build_rc"] == 0),
        "errors": sum(1 for r in records if r["error"]),
        "wall_total_s": round(time.time() - t0, 2),
        "records": records,
    }
