"""kill_tree must reap the WHOLE build tree, including grandchildren that
detached into their own session — the FreeRouting-JVM orphan pattern of
self-eval-2026-07-07 FIX 1 (proc.kill() on the direct child left xvfb-run/java
reparented to init, burning a core for days)."""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from kicraft.proc_tree import descendants, kill_tree

pytestmark = pytest.mark.skipif(
    not Path("/proc").is_dir(), reason="kill_tree's tree walk needs /proc"
)

# A child that spawns a session-detached grandchild sleeper (mimicking
# freerouting_runner's start_new_session JVM), prints the grandchild pid,
# then sleeps as the "build".
_CHILD_SRC = """
import subprocess, sys, time
g = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"],
                     start_new_session=True)
print(g.pid, flush=True)
time.sleep(600)
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _reap(proc: subprocess.Popen) -> None:
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass


def test_kill_tree_reaps_session_detached_grandchild():
    proc = subprocess.Popen([sys.executable, "-c", _CHILD_SRC],
                            stdout=subprocess.PIPE, text=True,
                            start_new_session=True)
    try:
        grandchild = int(proc.stdout.readline())  # type: ignore[union-attr]
        assert _alive(proc.pid) and _alive(grandchild)
        # The grandchild is in a DIFFERENT session/group: the pre-fix
        # killpg-only approach provably misses it.
        assert os.getpgid(grandchild) != os.getpgid(proc.pid)
        assert grandchild in descendants(proc.pid)

        kill_tree(proc.pid)
        _reap(proc)  # child must be reaped by us before os.kill(pid, 0) is honest
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and _alive(grandchild):
            time.sleep(0.05)
        assert not _alive(proc.pid)
        assert not _alive(grandchild)
    finally:
        kill_tree(proc.pid)
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except OSError:
            pass
        _reap(proc)


def test_kill_tree_survives_already_dead_pid():
    proc = subprocess.Popen([sys.executable, "-c", "pass"])
    proc.wait()
    kill_tree(proc.pid)  # must not raise
