"""Cross-process build-slot gate bounding concurrent place+route runs per host.

One `kicraft build` deliberately saturates the machine during place+route:
autoexperiment fans out to min(cpu_count, 6) leaf-solver processes
(kicraft/cli/autoexperiment.py), each launching a FreeRouting JVM with no heap
cap. Two unmetered builds therefore double-book every core and can OOM a small
host. This module bounds that with N flock'd lockfiles shared by every build
on the host (web runs, the build worker, admin self-eval batches, manual CLI
runs). flock is released by the kernel when the holding process dies, so a
crashed build can never leak a slot.

Sizing (KICRAFT_BUILD_SLOTS): unset -> max(1, cpu_count // 6), because each
build internally runs up to 6 workers, so slots * 6 ~ cores; an explicit 0
disables gating (useful in tests and on throwaway machines).

Callers that enforce wall-clock build timeouts should measure from the
ACQUIRED_MARKER line, not process start, so time spent queued for a slot is
never billed against the build (see kicraft.server.build_worker).
"""
from __future__ import annotations

import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

try:
    import fcntl
except ImportError:  # non-POSIX host: no flock, gate becomes a no-op
    fcntl = None  # type: ignore[assignment]

# Stable prefixes for log-line matching; the acquired line resets build timeouts.
WAITING_MARKER = "[build] waiting for a free build slot"
ACQUIRED_MARKER = "[build] build slot acquired"

_POLL_S = 2.0
_REECHO_S = 30.0  # repeat the waiting line so a queued build's log shows liveness


def slot_count() -> int:
    """Slots on this host: KICRAFT_BUILD_SLOTS, or max(1, cpus // 6); 0 = off."""
    raw = os.environ.get("KICRAFT_BUILD_SLOTS", "").strip()
    if raw:
        try:
            return max(0, int(raw))
        except ValueError:
            pass
    return max(1, (os.cpu_count() or 1) // 6)


def host_cpu_count() -> int:
    """Usable host cores for concurrency sizing (the repo-wide `os.cpu_count()`
    idiom, floored at 1)."""
    return os.cpu_count() or 1


def default_build_slots() -> int:
    """CPU-safe concurrent-build count from the ACTUAL core count: each build
    fans out to up to 6 leaf solvers, so ``max(1, cpus//6)`` keeps
    ``slots * 6 ~ cores``.

    Deliberately NOT ``slot_count()``: that layers the ``KICRAFT_BUILD_SLOTS``
    host-gate override on top of the sizing, and an operator who set the gate to
    2 on a 2-core host (which is what produced the self-eval 2026-07-08 rc=-9
    cluster) would otherwise pull the harness default straight back to the
    over-subscribed value. The host-gate flock still bounds total host
    concurrency on top of this; this is only the harness's own CPU-safe sizing.
    """
    return max(1, host_cpu_count() // 6)


def resolve_build_slots(requested: int | None) -> int:
    """Concurrent-build count for a fan-out harness (self-eval, loadtest),
    clamped so it can never over-subscribe the host.

    ``requested is None`` -> :func:`default_build_slots` (``max(1, cpus//6)``
    from the real core count). An explicit value is honored but must be
    ``<= cores``: a single concurrent build already saturates the CPU, so
    ``build_slots > cores`` guarantees the thrash that turned genuinely-slow
    leaf solves into 2400s watchdog kills (the self-eval 2026-07-08 rc=-9
    cluster ran ``--build-slots 2`` on a 2-core host). Raises ``ValueError``
    rather than silently clamping, so a contended config fails loudly at launch
    instead of shipping.
    """
    cores = host_cpu_count()
    if requested is None:
        return default_build_slots()
    slots = max(1, int(requested))
    if slots > cores:
        raise ValueError(
            f"build_slots={slots} exceeds host core count {cores}: each "
            f"concurrent build saturates the CPU, so this over-subscribes and "
            f"trips the build timeout. Use build_slots <= {cores} "
            f"(host-aware default is {default_build_slots()})."
        )
    return slots


def slots_dir() -> Path:
    """Lockfile directory; must be shared by every process on the host (NOT /tmp,
    which systemd PrivateTmp namespaces per service)."""
    return Path(os.environ.get("KICRAFT_BUILD_SLOTS_DIR",
                               str(Path.home() / ".kicraft" / "build_slots")))


def _try_acquire(n: int, d: Path):
    """Try every slot file once; return (index, open fd) or None."""
    for i in range(n):
        fd = os.open(d / f"slot_{i}.lock", os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            os.close(fd)
            continue
        # Owner breadcrumb for an operator inspecting a busy host; never read back.
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} acquired={time.strftime('%FT%T')}\n".encode())
        return i, fd
    return None


@contextmanager
def build_slot(echo=None):
    """Hold one build slot for the duration of the with-block.

    `echo` (e.g. `print`) receives human-readable status lines, including the
    WAITING_MARKER/ACQUIRED_MARKER lines callers key timeouts off. Yields the
    slot index, or None when gating is disabled.
    """
    n = slot_count()
    if n == 0 or fcntl is None:
        yield None
        return
    say = echo or (lambda _line: None)
    d = slots_dir()
    d.mkdir(parents=True, exist_ok=True)
    got = _try_acquire(n, d)
    if got is None:
        say(f"{WAITING_MARKER} ({n} slot(s), all busy) ...")
        last_echo = time.monotonic()
        while got is None:
            time.sleep(_POLL_S + random.uniform(0, 0.5))
            got = _try_acquire(n, d)
            if got is None and time.monotonic() - last_echo >= _REECHO_S:
                say(f"{WAITING_MARKER} (still queued) ...")
                last_echo = time.monotonic()
    index, fd = got
    say(f"{ACQUIRED_MARKER} ({index + 1}/{n})")
    try:
        yield index
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)
