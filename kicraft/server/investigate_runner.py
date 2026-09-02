"""Run the portable ``kicraft-investigate`` skill against a support report.

The skill is an Agent Skills procedure with no Python entry point, so the web
service drives the installed ``omp`` agent CLI in print mode::

    omp -p "Use the kicraft-investigate skill to investigate KC-XXXX." --auto-approve

The process runs from the repository root, where OMP discovers
``.agents/skills/kicraft-investigate/SKILL.md``. Its final Markdown response is
stored in ``support_investigations.report_md``. Both the admin support page and
the automatic user-report trigger go through :func:`enqueue_investigation`.

Each run is a real agent session with LLM spend, so a module-level semaphore
bounds concurrency and :func:`enqueue_investigation` de-duplicates queued or
running investigations for the same report. ``--auto-approve`` is deliberate:
the owner-triggered skill uses read-only investigation commands, while a
headless process has no TTY for permission prompts.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
import traceback
from pathlib import Path

from kicraft.proc_tree import kill_tree

from .accounts import AccountStore, SupportReport

# Bound concurrent headless investigations: each saturates an LLM session and
# costs real money, and they are admin/single-user triggered (never a flood).
_MAX_CONCURRENT = max(1, int(os.environ.get("KICRAFT_INVESTIGATE_CONCURRENCY", "2")))
_slots = threading.Semaphore(_MAX_CONCURRENT)

# Wall-clock ceiling for one investigation (the skill replays place+route and
# reads several runs; generous, but never unbounded -> no wedged session).
_TIMEOUT_S = float(os.environ.get("KICRAFT_INVESTIGATE_TIMEOUT_S", "1800"))

# Explicit model pin so headless spend is deliberate, not whatever the CLI's
# session default happens to be. Override via KICRAFT_INVESTIGATE_MODEL.
_DEFAULT_MODEL = "opencode-go/deepseek-v4-flash:high"


def _log(msg: str) -> None:
    print(f"[investigate] {msg}", flush=True)


def _omp_bin() -> str | None:
    """The ``omp`` CLI, from an explicit override or PATH. None if absent so
    the runner can fail the row with a clear message instead of crashing."""
    return os.environ.get("KICRAFT_OMP_BIN") or shutil.which("omp")


def _repo_dir() -> Path:
    """Repository root containing the project-level Agent Skills directory.

    Overridable via KICRAFT_REPO_DIR; defaults to this package's repository root.
    """
    env = os.environ.get("KICRAFT_REPO_DIR")
    return Path(env) if env else Path(__file__).resolve().parents[2]


def _resolve_target(store: AccountStore, report: SupportReport) -> str | None:
    """Target described to the skill: board code first, then project path."""
    if report.board_code:
        return report.board_code
    if report.project_id is not None:
        p = store.get_project(report.project_id)
        if p and p.dir_path:
            return p.dir_path
    return None


def enqueue_investigation(store: AccountStore, report: SupportReport, *,
                          log_dir: Path | None = None,
                          runner=None) -> int | None:
    """De-dup, create the queued row, and spawn the headless run in a daemon
    thread. Returns the investigation id, or None if there is nothing to
    investigate or a run is already queued/running for this report. ``runner``
    is a test seam standing in for :func:`run_investigation` (so tests never
    invoke ``omp``)."""
    # Hard gate: never spawn a REAL headless `omp` from inside pytest.
    # An unstubbed test path spawned one phantom investigation per suite run
    # (real LLM-side spend, 30-min watchdog, orphaned to PID 1 -- 46
    # phantoms between 2026-07-12 and 2026-07-20). The `runner=` seam stays
    # the way tests exercise this function.
    if runner is None and os.environ.get("PYTEST_CURRENT_TEST"):
        return None
    target = _resolve_target(store, report)
    if target is None:
        return None
    if store.active_investigation_exists(report.id):
        return None
    log_path = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = str(log_dir / f"investigate_report_{report.id}.log")
    inv_id = store.create_investigation(
        report_id=report.id, board_code=report.board_code, log_path=log_path)
    run = runner or run_investigation
    threading.Thread(target=run, args=(store, inv_id, target), daemon=True).start()
    return inv_id


def run_investigation(store: AccountStore, inv_id: int, target: str) -> None:
    """Execute one investigation: shell out to ``omp -p`` and store the
    Markdown report. Crash-safe -- any failure finalizes the row so it never
    wedges 'running'."""
    if not store.start_investigation(inv_id):
        return  # a duplicate runner won the guarded queued->running transition
    with _slots:
        try:
            rc, out = _run_omp(store, inv_id, target)
            status = "done" if rc == 0 else "failed"
            store.finish_investigation(inv_id, rc=rc, report_md=out, status=status)
            _log(f"inv {inv_id}: {status} rc={rc} ({len(out or '')} chars)")
        except Exception:  # noqa: BLE001 -- must finalize the row (see docstring)
            _log(f"inv {inv_id}: crashed:\n{traceback.format_exc()}")
            store.finish_investigation(
                inv_id, rc=None, status="failed",
                report_md="Investigation crashed before completing; "
                          "see the server logs.")


def _run_omp(store: AccountStore, inv_id: int, target: str):
    """Run the CLI, tee stdout to the row's log, enforce a wall-clock watchdog.
    Returns (rc, captured_markdown)."""
    omp = _omp_bin()
    if omp is None:
        return None, ("The `omp` CLI is not installed or not on PATH for the "
                      "web service user, so the investigate skill cannot run "
                      "headlessly. Install Oh My Pi (omp) or set KICRAFT_OMP_BIN, "
                      "then retry.")
    inv = store.get_investigation(inv_id)
    log_path = Path(inv.log_path) if inv and inv.log_path else None
    repo = _repo_dir()
    prompt = f"Use the kicraft-investigate skill to investigate {target}."
    cmd = [omp, "-p", prompt, "--auto-approve"]
    model = os.environ.get("KICRAFT_INVESTIGATE_MODEL") or _DEFAULT_MODEL
    cmd += ["--model", model]
    _log(f"inv {inv_id}: kicraft-investigate {target} (cwd={repo})")
    chunks: list[str] = []
    logf = log_path.open("a", encoding="utf-8") if log_path else None
    try:
        proc = subprocess.Popen(
            cmd, cwd=str(repo), stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors="replace",
            bufsize=1,
            # The skill's headless section keys off this: budget inside the
            # watchdog (skip dense replays, mark findings PLAUSIBLE) and put
            # the whole report in the final message (only the final assistant
            # message survives omp -p).
            env={**os.environ, "KICRAFT_INVESTIGATE_HEADLESS": "1"},
            start_new_session=True)
        # A silent hang (a wedged tool call) prints nothing, so a per-line check
        # would never fire; the watchdog enforces the wall clock regardless.
        wd = {"deadline": time.monotonic() + _TIMEOUT_S, "killed": False}

        def watchdog() -> None:
            while proc.poll() is None:
                if time.monotonic() > wd["deadline"]:
                    wd["killed"] = True
                    _kill(proc)
                    return
                time.sleep(5.0)

        threading.Thread(target=watchdog, daemon=True).start()
        for line in proc.stdout or []:
            chunks.append(line)
            if logf:
                logf.write(line)
                logf.flush()
        rc = proc.wait()
        out = "".join(chunks).strip()
        if wd["killed"]:
            note = f"\n\n[investigation exceeded {int(_TIMEOUT_S // 60)}m, killed]"
            out += note
            if logf:
                logf.write(note + "\n")
            rc = rc if rc not in (0, None) else -9
        return rc, (out or "(the investigate skill produced no output)")
    finally:
        if logf:
            logf.close()


def _kill(proc: subprocess.Popen) -> None:
    """Kill the CLI and its whole process tree (the bash/python tool calls it
    spawns), mirroring the build worker's tree-kill so a timeout leaves no
    orphans."""
    kill_tree(proc.pid)
    try:
        proc.kill()
    except OSError:
        pass
