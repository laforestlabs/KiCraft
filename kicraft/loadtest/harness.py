"""Scenario orchestrator: own a LoadResultStore run + a live metrics sampler.

A scenario records one ``runs`` row, samples host/process/queue metrics at the
chosen cadence for its whole duration, runs the load, and writes a summary back to
the run. Everything is reconstructable afterward and chartable live on
/admin/loadtest. The harness is the single seam the CLI (__main__.py) and the
dashboard launcher call.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from . import buildstorm, metrics, pipeline_load
from .store import LoadResultStore, default_store_path


def make_run_id(scenario: str) -> str:
    return f"{scenario}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"


def _kicraft_home() -> Path:
    return Path(os.environ.get("KICRAFT_WORK_DIR",
                               str(Path.home() / ".kicraft" / "work"))).parent


def default_wal_paths() -> list[Path]:
    home = _kicraft_home()
    return [home / "accounts.db-wal", home / "spend_ledger.db-wal"]


def _aborted(abort_file: str | Path | None) -> bool:
    return bool(abort_file) and Path(abort_file).exists()


def _sampler(store: LoadResultStore, run_id: str, *, interval_s: float,
             disk_path, queue_probe=None, lock_db=None) -> metrics.MetricsSampler:
    return metrics.MetricsSampler(
        store, run_id, interval_s=interval_s, disk_path=disk_path,
        web_pids_probe=lambda: metrics.find_pids("kicraft.server.web"),
        worker_pids_probe=lambda: metrics.find_pids("kicraft.server.build_worker"),
        queue_probe=queue_probe, wal_paths=default_wal_paths(), lock_db=lock_db)


def run_build_storm(*, source: Path, n: int, slots: int, route: bool = False,
                    store_path: Path | None = None, work_root: Path | None = None,
                    interval_s: float = 1.0, timeout_s: float = 1800.0,
                    abort_file: str | Path | None = None) -> dict:
    """Compute-tier storm with live metrics. Uses a throwaway AccountStore for the
    queue so the prod accounts DB is never touched."""
    from kicraft.server.accounts import AccountStore

    run_id = make_run_id("build-storm")
    store = LoadResultStore(store_path)
    work_root = Path(work_root or (Path.home() / ".kicraft" / "loadtest_work" / run_id))
    work_root.mkdir(parents=True, exist_ok=True)
    queue_db = work_root / "queue.db"
    acct = AccountStore(queue_db, work_root / "projects")

    stem = buildstorm.detect_stem(source)
    command = buildstorm.replay_command(stem, route=route)
    workspaces = buildstorm.stage_workspaces(source, n, work_root / "ws")

    store.start_run(run_id, "build-storm",
                    {"n": n, "slots": slots, "route": route, "stem": stem})
    sampler = _sampler(store, run_id, interval_s=interval_s, disk_path=work_root,
                       queue_probe=lambda: (acct.build_queue_position(0)[1],
                                            acct.count_running_builds()),
                       lock_db=queue_db)

    def on_tick(running, depth):
        if _aborted(abort_file):
            raise KeyboardInterrupt("aborted via dashboard")

    sampler.start()
    try:
        summary = buildstorm.run_storm(acct, workspaces, command=command,
                                       max_jobs=slots, timeout_s=timeout_s,
                                       on_tick=on_tick)
    finally:
        sampler.stop()
    for job in summary["jobs"]:
        if job["wall_s"] is not None:
            store.add_event(run_id, "build", latency_ms=job["wall_s"] * 1000.0,
                            rc=job["rc"], detail=f"job {job['id']}")
    store.finish_run(run_id, summary)
    return {"run_id": run_id, "summary": summary, "store": str(store.path)}


def run_pipeline(*, briefs, parallel: int = 3, build_slots: int = 2,
                 transcript: dict, do_build: bool = True,
                 store_path: Path | None = None, work_root: Path | None = None,
                 interval_s: float = 1.0, build_timeout_s: float = 1800.0) -> dict:
    """Full-pipeline load (mock design + optional real build) with live metrics."""
    run_id = make_run_id("pipeline")
    store = LoadResultStore(store_path)
    work_root = Path(work_root or (Path.home() / ".kicraft" / "loadtest_work" / run_id))
    store.start_run(run_id, "pipeline",
                    {"n": len(briefs), "parallel": parallel, "build_slots": build_slots,
                     "do_build": do_build})
    sampler = _sampler(store, run_id, interval_s=interval_s, disk_path=work_root)
    sampler.start()
    try:
        summary = pipeline_load.run_pipeline_load(
            briefs, parallel=parallel, build_slots=build_slots, transcript=transcript,
            work_root=work_root, do_build=do_build, build_timeout_s=build_timeout_s,
            store=store, run_id=run_id)
    finally:
        sampler.stop()
    store.finish_run(run_id, summary)
    return {"run_id": run_id, "summary": summary, "store": str(store.path)}


def store_for_dashboard(store_path: Path | None = None) -> LoadResultStore:
    """Open the load store the dashboard reads (default location)."""
    return LoadResultStore(store_path or default_store_path())
