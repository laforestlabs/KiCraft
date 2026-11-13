"""Tests for E2E Finding 4: a pcbnew subprocess that finishes its work and
saves the board, then crashes at interpreter teardown (pcbnew/wx static
destructor SIGSEGV, returncode -11), must NOT be treated as a failed
operation -- the output is already intact.

This is the mechanism behind the preserve-copper parent route reporting a
failure even though the routed board + DRC json were written: the
preserve-only ``_unlock_traces`` (and ``import_ses``) subprocess saved the
board, then segfaulted at teardown, and ``_retry_pcbnew_run`` raised on the
non-zero returncode. These tests don't require pcbnew -- they exercise the
returncode/sentinel logic directly.
"""
from __future__ import annotations

import subprocess
import time
import sys

import pytest

from kicraft.autoplacer import routing_board as fr

SENTINEL = fr._PCBNEW_OK_SENTINEL


def _cmd(body: str) -> list[str]:
    return [sys.executable, "-c", body]




def test_crash_before_sentinel_still_fails():
    # Segfault BEFORE emitting the sentinel = a real mid-work crash. Must raise.
    body = (
        "import os, signal\n"
        "os.kill(os.getpid(), signal.SIGSEGV)\n"
        f"print({SENTINEL!r})\n"
    )
    with pytest.raises(RuntimeError, match="pcbnew subprocess failed"):
        fr._retry_pcbnew_run(_cmd(body))


def test_normal_error_exit_still_fails():
    # A normal non-zero exit (Python exception) has positive rc and no
    # sentinel -- must still raise.
    body = "raise SystemExit(1)\n"
    with pytest.raises(RuntimeError, match="pcbnew subprocess failed"):
        fr._retry_pcbnew_run(_cmd(body))








def test_deadline_kills_process_group_even_after_success_sentinel(tmp_path):
    marker = tmp_path / "child-survived"
    ready = tmp_path / "child-ready"
    child = (
        "import time\n"
        "from pathlib import Path\n"
        f"Path({str(ready)!r}).write_text('ready')\n"
        f"time.sleep(1.2); Path({str(marker)!r}).write_text('alive')\n"
    )
    body = (
        "import subprocess, sys, time\n"
        "from pathlib import Path\n"
        f"subprocess.Popen([sys.executable, '-c', {child!r}])\n"
        f"while not Path({str(ready)!r}).exists(): time.sleep(0.01)\n"
        f"print({SENTINEL!r}, flush=True)\n"
        "time.sleep(10)\n"
    )
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        fr._retry_pcbnew_run(_cmd(body), deadline=time.monotonic() + 0.6)
    assert SENTINEL in caught.value.stdout
    assert ready.exists()
    time.sleep(1.3)
    assert not marker.exists()


def test_deadline_is_shared_across_load_retries_and_backoff(tmp_path):
    attempts = tmp_path / "attempts"
    body = (
        "import sys, time\n"
        "from pathlib import Path\n"
        f"p = Path({str(attempts)!r})\n"
        "with p.open('a') as f: f.write('x')\n"
        "time.sleep(0.13)\n"
        "print('Failed to load board: transient', file=sys.stderr, flush=True)\n"
        "raise SystemExit(1)\n"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        fr._retry_pcbnew_run(_cmd(body), deadline=started + 0.4)
    assert 1 <= len(attempts.read_text()) <= 2
    assert time.monotonic() - started < 1.5


