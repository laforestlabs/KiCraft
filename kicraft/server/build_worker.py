"""Standalone build worker: claims queued build_jobs rows and executes them.

Run as its own service (deploy/kicraft-build-worker.service) so deterministic
builds survive `kicraft-web` restarts: the web app enqueues a row and tails the
job's log file; this process claims the row, runs `kicraft build` in the job's
workspace, streams the output to that log, and records the exit code. The web
app keeps a fallback path that runs builds in-process whenever no worker has
heartbeated recently (see AccountStore.build_worker_alive), so a deploy without
this unit behaves exactly as before the queue existed.

Workspaces must live on a path both services can see (KICRAFT_WORK_DIR, default
~/.kicraft/work). /tmp does NOT qualify on the box: systemd PrivateTmp gives
each service a private namespace.

Concurrency: up to build_slots.slot_count() jobs at once; each build also takes
a host-wide flock slot inside `kicraft build` itself, which is what actually
protects the host from manual/eval builds running alongside.

The 30-minute build timeout starts counting at the ACQUIRED_MARKER line, not at
process start, so time a build spends queued behind a slot is never billed
against it.

On SIGTERM the worker aborts its running builds and requeues them (or fails
them once they have burned their claim attempts), so a worker deploy is safe:
jobs go back to the queue head instead of sticking in 'running' forever.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

from kicraft.build_slots import ACQUIRED_MARKER, slot_count

from .accounts import AccountStore, BuildJob

# Mirrors the web worker's build invocation (relative paths against the job's
# workspace, no archive sweep).
_BUILD_CMD = [sys.executable, "-m", "kicraft.design.cli_app",
              "build", ".kicraft/state.json", "generated", "--no-archive"]

# Route + promote a saved manual layout (kicraft manual-route); enqueued by the
# web layout editor's "Route this layout".
_MANUAL_ROUTE_CMD = [sys.executable, "-m", "kicraft.design.cli_app",
                     "manual-route", ".kicraft/state.json", "generated"]

_MAX_ATTEMPTS = 2  # claims a job may burn before it is failed instead of requeued


def _log(msg: str) -> None:
    print(f"[build-worker] {msg}", flush=True)


class BuildWorker:
    """The claim-and-execute loop. Separated from main() so tests can drive it
    with a throwaway store and a fake build command."""

    def __init__(self, store: AccountStore, *, build_cmd: list[str] | None = None,
                 timeout_s: float = 1800.0, poll_s: float = 2.0,
                 max_jobs: int | None = None,
                 commands: dict[str, list[str]] | None = None):
        self.store = store
        self.build_cmd = list(build_cmd or _BUILD_CMD)
        # Command per job kind. A kind absent here FAILS the job rather
        # than falling back to 'build' (deploy-skew safety: an old worker
        # must never run the wrong command on a job from a newer web).
        self.commands = (
            {k: list(v) for k, v in commands.items()} if commands is not None
            else {"build": self.build_cmd, "manual_route": list(_MANUAL_ROUTE_CMD)}
        )
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.max_jobs = max_jobs if max_jobs is not None else max(1, slot_count())
        self.stop = threading.Event()
        self._lock = threading.Lock()
        self._procs: dict[int, subprocess.Popen] = {}  # job id -> live build
        self._threads: list[threading.Thread] = []

    # ---- lifecycle -----------------------------------------------------------
    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: self.stop.set())
        signal.signal(signal.SIGINT, lambda *_: self.stop.set())
        recovered = self.store.requeue_stale_builds(max_attempts=_MAX_ATTEMPTS)
        if recovered:
            _log(f"recovered {recovered} stale job(s) from a dead predecessor")
        _log(f"ready (max {self.max_jobs} concurrent build(s))")
        while not self.stop.is_set():
            self.store.beat_build_worker()
            self.run_once()
            self.stop.wait(self.poll_s)
        self._shutdown()

    def run_once(self) -> bool:
        """Claim and start at most one job; True if one was started."""
        self._threads = [t for t in self._threads if t.is_alive()]
        if len(self._threads) >= self.max_jobs:
            return False
        job = self.store.claim_next_build(f"pid:{os.getpid()}")
        if job is None:
            return False
        t = threading.Thread(target=self._run_job, args=(job,), daemon=True)
        self._threads.append(t)
        t.start()
        return True

    def _shutdown(self) -> None:
        """Abort running builds and put their jobs back on the queue (or fail the
        ones already on their last claim, so a crash-looping deploy cannot bounce
        one bad job forever)."""
        with self._lock:
            live = dict(self._procs)
        for job_id, proc in live.items():
            # Requeue/fail BEFORE killing: the kill wakes the _run_job reader,
            # whose finish_build('done', rc=-9) would otherwise race this
            # thread and stamp the row terminal, silently losing the retry.
            # Once the row has left 'running', that finalize no-ops (guarded
            # UPDATE), so the order closes the race.
            job = self.store.get_build_job(job_id)
            if job is not None and job.attempts >= _MAX_ATTEMPTS:
                self.store.finish_build(job_id, rc=None, status="failed")
                _log(f"job {job_id}: aborted by shutdown on final attempt -> failed")
            else:
                self.store.requeue_build(job_id)
                _log(f"job {job_id}: aborted by shutdown -> requeued")
            _kill_build(proc)
        for t in self._threads:
            t.join(timeout=10)
        _log("stopped")

    # ---- one job --------------------------------------------------------------
    def _run_job(self, job: BuildJob) -> None:
        """Crash barrier around one job: any unexpected exception (a non-UTF-8
        byte in the build output, a sqlite error, ...) must finalize the row.
        Without this the thread dies, the row wedges in 'running' with a live
        claimant pid, and requeue_stale_builds skips it forever."""
        try:
            self._execute_job(job)
        except Exception:  # noqa: BLE001 -- see docstring
            _log(f"job {job.id}: crashed:\n{traceback.format_exc()}")
            with self._lock:
                proc = self._procs.pop(job.id, None)
            if proc is not None:
                _kill_build(proc)
            cur = self.store.get_build_job(job.id)
            if cur is not None and cur.status == "running":
                if cur.attempts >= _MAX_ATTEMPTS:
                    self.store.finish_build(job.id, rc=None, status="failed")
                    _log(f"job {job.id}: crashed on final attempt -> failed")
                else:
                    self.store.requeue_build(job.id)
                    _log(f"job {job.id}: crashed -> requeued")

    def _execute_job(self, job: BuildJob) -> None:
        ws = Path(job.workspace)
        log_path = Path(job.log_path or (ws / ".kicraft" / "build.log"))
        if not (ws / ".kicraft" / "state.json").is_file():
            _log(f"job {job.id}: workspace gone ({ws}) -> failed")
            self.store.finish_build(job.id, rc=None, status="failed")
            return
        kind = getattr(job, "kind", "build") or "build"
        cmd_base = self.commands.get(kind)
        if cmd_base is None:
            _log(f"job {job.id}: unknown job kind {kind!r} -> failed")
            self.store.finish_build(job.id, rc=None, status="failed")
            return
        _log(f"job {job.id}: {kind} in {ws}")
        log_path.parent.mkdir(parents=True, exist_ok=True)
        env = {**os.environ, "PYTHONUNBUFFERED": "1",
               "KICRAFT_CALLER": os.environ.get("KICRAFT_CALLER", "web")}
        cmd = list(cmd_base)
        if kind == "build":
            quality = self.store.build_quality_for_user(job.user_id)
            if quality:
                cmd += ["--quality", quality]
                _log(f"job {job.id}: tier quality override --quality {quality}")
        try:
            with log_path.open("a", encoding="utf-8") as logf:
                # errors="replace": build tools (freerouting JVM, kicad-cli) can
                # emit non-UTF-8 bytes; a strict decode would kill the reader
                # loop mid-build.
                proc = subprocess.Popen(
                    cmd, cwd=str(ws), stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT, text=True, errors="replace",
                    bufsize=1, env=env, start_new_session=True)
                with self._lock:
                    self._procs[job.id] = proc
                # The watchdog enforces the wall clock even when the build goes
                # silent (a hung FreeRouting prints nothing, so a per-line check
                # would never fire and the job would wedge a queue slot forever).
                wd = {"deadline": time.monotonic() + self.timeout_s, "killed": False}

                interval = min(5.0, max(0.05, self.timeout_s / 10.0))

                def watchdog() -> None:
                    while proc.poll() is None:
                        if time.monotonic() > wd["deadline"]:
                            wd["killed"] = True
                            _kill_build(proc)
                            return
                        time.sleep(interval)

                threading.Thread(target=watchdog, daemon=True).start()
                for line in proc.stdout or []:
                    logf.write(line)
                    logf.flush()
                    if ACQUIRED_MARKER in line:
                        # Queued-for-a-slot time is not build time.
                        wd["deadline"] = time.monotonic() + self.timeout_s
                rc = proc.wait()
                if wd["killed"]:
                    logf.write(f"[build exceeded {int(self.timeout_s // 60)}m, "
                               "killed]\n")
                    logf.flush()
        except OSError as e:
            _log(f"job {job.id}: failed to launch build: {e}")
            self.store.finish_build(job.id, rc=None, status="failed")
            return
        finally:
            with self._lock:
                self._procs.pop(job.id, None)
        # A shutdown abort already requeued/failed the row; finish_build's
        # status guard makes this a no-op in that case (no TOCTOU window).
        if self.store.finish_build(job.id, rc=rc, status="done"):
            _log(f"job {job.id}: done rc={rc}")


def _kill_build(proc: subprocess.Popen) -> None:
    """Kill the build and its whole process tree (leaf solvers, FreeRouting JVMs);
    the build runs in its own session, so the process group is ours to kill."""
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _store_from_env() -> AccountStore:
    """The accounts store from the same env/.env the web app uses, WITHOUT
    requiring OPENROUTER_API_KEY (Settings.from_env exits on a missing key, and
    this process never calls a model)."""
    from .config import Settings, load_dotenv
    load_dotenv()
    db = os.environ.get("KICRAFT_USERS_DB", str(Settings.users_db_path))
    projects = os.environ.get("KICRAFT_PROJECTS_DIR", str(Settings.projects_dir))
    return AccountStore(db, projects)


def main() -> int:
    BuildWorker(_store_from_env()).run_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
