"""Build-slot DoS: a malicious brief that produces a giant board could monopolize
the single build slot. Verify the cross-process host gate never double-books and is
recoverable (the 30-min watchdog + FIFO drain bound starvation; tested in the
build-storm harness)."""
from __future__ import annotations

import os
import subprocess
import sys
import time

from kicraft import build_slots

# A holder subprocess: acquire the (only) slot, signal ACQUIRED, hold until STOP.
_HOLDER = """
import os, sys, time
os.environ["KICRAFT_BUILD_SLOTS"] = "1"
os.environ["KICRAFT_BUILD_SLOTS_DIR"] = sys.argv[1]
from kicraft import build_slots
acq, stop = sys.argv[2], sys.argv[3]
with build_slots.build_slot():
    open(acq, "w").close()
    while not os.path.exists(stop):
        time.sleep(0.02)
"""


def test_slot_gate_never_double_books_across_processes(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "1")
    slots_dir = tmp_path / "slots"
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS_DIR", str(slots_dir))
    assert build_slots.slot_count() == 1

    acq, stop = tmp_path / "acquired", tmp_path / "stop"
    holder = subprocess.Popen([sys.executable, "-c", _HOLDER, str(slots_dir),
                               str(acq), str(stop)])
    try:
        # wait for the holder to take the only slot
        for _ in range(200):
            if acq.exists():
                break
            time.sleep(0.02)
        assert acq.exists(), "holder process never acquired the slot"

        # while held by another process, this process CANNOT also get it (no double-book)
        d = build_slots.slots_dir()
        d.mkdir(parents=True, exist_ok=True)
        assert build_slots._try_acquire(1, d) is None
    finally:
        stop.write_text("go")  # release the holder
        holder.wait(timeout=10)

    # holder gone -> the slot is recoverable (flock released on exit)
    got = build_slots._try_acquire(1, build_slots.slots_dir())
    assert got is not None, "slot should be free once the holder exits"
    index, fd = got
    import fcntl
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def test_gating_disabled_is_a_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "0")  # explicit off (tests/throwaway hosts)
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS_DIR", str(tmp_path / "slots"))
    with build_slots.build_slot() as idx:
        assert idx is None  # no gating, yields immediately


def test_default_slot_sizing_bounds_concurrency(monkeypatch):
    """Default sizing is max(1, cpus//6): each build fans to ~6 workers, so the
    host is never oversubscribed by the default. (DoS mitigation rationale.)"""
    monkeypatch.delenv("KICRAFT_BUILD_SLOTS", raising=False)
    assert build_slots.slot_count() >= 1
    assert build_slots.slot_count() <= max(1, (os.cpu_count() or 1))
