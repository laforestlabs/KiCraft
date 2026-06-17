"""Compute-tier build storm: saturate the build queue + flock slots at $0.

The place+route build is the real scaling bottleneck (CPU-bound FreeRouting +
kicad-cli, gated by a host-wide flock; see kicraft/build_slots.py). This module
stages N copies of an already-synthesized workspace, enqueues them as
``loadtest_replay`` build_jobs, and drives a real ``BuildWorker`` whose command
is ``kicraft replay`` -- deterministic, no LLM, $0 -- so we exercise the actual
queue + slot machinery and find the saturation knee as KICRAFT_BUILD_SLOTS grows.

Nothing here touches the prod accounts DB: callers pass a throwaway AccountStore.
The build command is injectable, so tests drive the queue with a fast fake replay.
"""
from __future__ import annotations

import datetime as dt
import shutil
import sys
import time
from pathlib import Path

from kicraft.server.build_worker import BuildWorker, _kill_build

from .store import _quantiles

# A distinct job kind so a real prod worker (which only knows build/manual_route)
# would never pick these up by mistake (deploy-skew safety, build_worker.py:71).
LOADTEST_KIND = "loadtest_replay"


def detect_stem(workspace: Path) -> str:
    """The synthesized project stem = the single subdir under ``generated/``."""
    gen = Path(workspace) / "generated"
    subs = [d for d in gen.iterdir() if d.is_dir()] if gen.is_dir() else []
    if len(subs) != 1:
        raise ValueError(
            f"expected exactly one synthesized project under {gen} (found {len(subs)})")
    return subs[0].name


def replay_command(stem: str, *, route: bool = True, seed: int = 0,
                   quality: str = "fast") -> list[str]:
    """The per-job ``kicraft replay`` command (run with cwd = the job workspace).

    ``route=False`` is the fast placement-only storm (no FreeRouting) for shaking
    out queue/slot mechanics quickly; ``route=True`` is the heavy, realistic load.
    """
    cmd = [sys.executable, "-m", "kicraft.design.cli_app", "replay",
           "--project", f"generated/{stem}", "--quality", quality,
           "--seed", str(seed), "--no-fab"]
    cmd.append("--route" if route else "--no-route")
    return cmd


def stage_workspaces(source: Path, n: int, dest_root: Path) -> list[Path]:
    """Copy a synthesized workspace ``source`` into ``n`` sibling work dirs.

    Each copy gets its own ``.kicraft/state.json`` + ``generated/`` tree, so the
    builds are independent (a build mutates its workspace in place).
    """
    source = Path(source)
    if not (source / ".kicraft" / "state.json").is_file():
        raise ValueError(f"{source} is not a synthesized workspace (.kicraft/state.json)")
    dest_root = Path(dest_root)
    dest_root.mkdir(parents=True, exist_ok=True)
    out = []
    for i in range(n):
        dst = dest_root / f"storm_{i:03d}"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(source, dst)
        out.append(dst)
    return out


def _parse(ts: str | None) -> dt.datetime | None:
    if not ts:
        return None
    try:
        return dt.datetime.fromisoformat(ts)
    except ValueError:
        return None


def _job_stat(job) -> dict:
    created, started, finished = (_parse(job.created_at), _parse(job.started_at),
                                  _parse(job.finished_at))
    wait_s = (started - created).total_seconds() if created and started else None
    wall_s = (finished - started).total_seconds() if started and finished else None
    return {"id": job.id, "status": job.status, "rc": job.rc,
            "wait_s": wait_s, "wall_s": wall_s}


def run_storm(store, workspaces, *, command: list[str], max_jobs: int,
              kind: str = LOADTEST_KIND, timeout_s: float = 1800.0,
              poll_s: float = 0.1, on_tick=None) -> dict:
    """Enqueue one job per workspace and drive a BuildWorker until the queue drains.

    Returns a summary: per-job wait/wall stats, the max concurrency actually
    observed (must never exceed ``max_jobs``), and wait/build latency quantiles.
    ``on_tick(running, depth)`` is called each poll for live metrics wiring.
    """
    ids = [store.enqueue_build(workspace=str(ws), kind=kind) for ws in workspaces]
    worker = BuildWorker(store, commands={kind: list(command)}, max_jobs=max_jobs,
                         timeout_s=timeout_s, poll_s=poll_s)
    t0 = time.time()
    max_running = 0
    aborted = False
    deadline = t0 + timeout_s + 60.0
    try:
        while time.time() < deadline:
            worker.run_once()
            running = store.count_running_builds()
            depth = store.build_queue_position(0)[1]  # id<0 -> ahead=0, depth=all queued
            max_running = max(max_running, running)
            if on_tick:
                on_tick(running, depth)  # may raise (e.g. abort file appeared)
            jobs = [store.get_build_job(i) for i in ids]
            if all(j is not None and j.status in ("done", "failed") for j in jobs):
                break
            time.sleep(poll_s)
    except (KeyboardInterrupt, RuntimeError):
        # Abort: kill in-flight build subprocesses so no `kicraft replay` orphans
        # (they run in their own session via start_new_session=True). A just-claimed
        # job registers its proc a beat after its thread starts, so poll-and-kill over
        # a short grace window to catch late registrations.
        aborted = True
        worker.stop.set()
        grace = time.time() + 5.0
        while time.time() < grace and any(t.is_alive() for t in worker._threads):
            for proc in list(worker._procs.values()):
                _kill_build(proc)
            time.sleep(0.05)
    for t in worker._threads:
        t.join(timeout=timeout_s)

    stats = [_job_stat(store.get_build_job(i)) for i in ids]
    waits = sorted(s["wait_s"] for s in stats if s["wait_s"] is not None)
    walls = sorted(s["wall_s"] for s in stats if s["wall_s"] is not None)
    return {
        "slots": max_jobs,
        "n": len(ids),
        "aborted": aborted,
        "ok": sum(1 for s in stats if s["status"] == "done" and s["rc"] == 0),
        "nonzero_rc": sum(1 for s in stats if s["status"] == "done" and s["rc"] not in (0, None)),
        "failed": sum(1 for s in stats if s["status"] == "failed"),
        "max_running": max_running,
        "wall_total_s": round(time.time() - t0, 2),
        "wait": _quantiles(waits),
        "build": _quantiles(walls),
        "jobs": stats,
    }


def sweep_slots(store_factory, source: Path, n: int, slots_list, *, work_root: Path,
                route: bool = False, timeout_s: float = 1800.0,
                on_run=None) -> list[dict]:
    """Run the storm once per slot count to find the saturation knee.

    ``store_factory(slots)`` returns a fresh throwaway AccountStore (so the queue
    starts empty each sweep step). Returns one summary dict per slot setting.
    """
    stem = detect_stem(source)
    command = replay_command(stem, route=route)
    out = []
    for slots in slots_list:
        store = store_factory(slots)
        workspaces = stage_workspaces(source, n, work_root / f"slots_{slots}")
        summary = run_storm(store, workspaces, command=command, max_jobs=slots,
                            timeout_s=timeout_s)
        out.append(summary)
        if on_run:
            on_run(summary)
    return out
