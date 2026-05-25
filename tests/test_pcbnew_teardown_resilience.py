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

import sys

import pytest

from kicraft.autoplacer import freerouting_runner as fr

SENTINEL = fr._PCBNEW_OK_SENTINEL


def _cmd(body: str) -> list[str]:
    return [sys.executable, "-c", body]


def test_teardown_segfault_after_sentinel_is_success():
    # Print sentinel (flushed), then die by SIGSEGV during atexit -- i.e.
    # after the work completed, mimicking a wx static-destructor teardown
    # crash. _retry_pcbnew_run must return normally (no exception).
    body = (
        "import os, signal, atexit, sys\n"
        "atexit.register(lambda: os.kill(os.getpid(), signal.SIGSEGV))\n"
        f"print({SENTINEL!r}); sys.stdout.flush()\n"
    )
    fr._retry_pcbnew_run(_cmd(body))  # must not raise


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


def test_signal_death_without_sentinel_fails():
    # Signal death (rc<0) but no sentinel printed -- e.g. crash during work
    # before any save. Must raise, not be mistaken for a teardown crash.
    body = "import os, signal\nos.kill(os.getpid(), signal.SIGSEGV)\n"
    with pytest.raises(RuntimeError, match="pcbnew subprocess failed"):
        fr._retry_pcbnew_run(_cmd(body))


def test_clean_exit_is_success():
    body = f"print({SENTINEL!r})\n"
    fr._retry_pcbnew_run(_cmd(body))  # rc 0 -> returns


def test_run_pcbnew_script_appends_sentinel_and_survives_teardown_crash():
    # End-to-end through _run_pcbnew_script: the body does its "work" (no
    # explicit sentinel), _run_pcbnew_script appends the sentinel, and a
    # teardown SIGSEGV registered by the body fires after it. Must not raise.
    script = (
        "import os, signal, atexit\n"
        "atexit.register(lambda: os.kill(os.getpid(), signal.SIGSEGV))\n"
        "# (pretend board.Save happened here)\n"
    )
    fr._run_pcbnew_script(script)  # must not raise
