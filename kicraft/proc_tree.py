"""Kill an entire build process tree, including session-detached descendants.

``os.killpg(os.getpgid(pid), SIGKILL)`` alone is NOT enough for KiCraft builds:
the FreeRouting JVM is launched with ``start_new_session=True``
(``freerouting_runner.run_freerouting``) so its OWN timeout can kill its group
without shooting the caller. The cost is that the JVM escapes the build's
process group, so an outer watchdog (self-eval harness, web build worker) that
kills only the build's group strands the ``xvfb-run -> java`` grandchildren on
init, each burning a core until someone notices
(docs/plans/self-eval-2026-07-07-high-value-fixes.md FIX 1).

The fix: snapshot the PPID tree under the build FIRST (via /proc), then SIGKILL
every process group found in it, then any stragglers by pid. Killing parents
before enumerating would reparent grandchildren to init and lose them from the
PPID walk, so order matters.
"""

from __future__ import annotations

import os
import signal


def _proc_children_map() -> dict[int, list[int]]:
    """``{ppid: [pid, ...]}`` for all live processes, best-effort via /proc.

    Returns ``{}`` on hosts without /proc (the caller then degrades to a plain
    process-group kill, which is what the pre-fix code did).
    """
    children: dict[int, list[int]] = {}
    try:
        entries = os.listdir("/proc")
    except OSError:
        return children
    for name in entries:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as fh:
                stat = fh.read().decode("ascii", "replace")
        except OSError:
            continue  # raced with process exit
        # comm (field 2) can contain spaces and parens, so split from the LAST
        # ')': the remainder is "<state> <ppid> ...".
        try:
            ppid = int(stat.rsplit(")", 1)[1].split()[1])
        except (IndexError, ValueError):
            continue
        children.setdefault(ppid, []).append(int(name))
    return children


def descendants(pid: int) -> list[int]:
    """All live descendants of ``pid`` (children, grandchildren, ...)."""
    tree = _proc_children_map()
    out: list[int] = []
    stack = [pid]
    while stack:
        for child in tree.get(stack.pop(), ()):
            out.append(child)
            stack.append(child)
    return out


def kill_tree(pid: int, sig: int = signal.SIGKILL) -> None:
    """Signal ``pid`` and every descendant, including ones that detached into
    their own session/process group (the FreeRouting JVM case).

    Killing whole process groups (not just pids) also catches members spawned
    between the snapshot and the kill, since children inherit their parent's
    pgid. Our own group is never signalled.
    """
    doomed = [pid] + descendants(pid)
    pgids: set[int] = set()
    for p in doomed:
        try:
            pgids.add(os.getpgid(p))
        except OSError:
            pass
    try:
        pgids.discard(os.getpgid(0))
    except OSError:
        pass
    for pgid in pgids:
        try:
            os.killpg(pgid, sig)
        except OSError:
            pass
    for p in doomed:
        try:
            os.kill(p, sig)
        except OSError:
            pass
