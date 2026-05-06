"""Regression test: per-round artifact snapshots must never copy stale
files from a prior round / prior run into a round that didn't produce
its own output.

The autoexperiment per-round flow (autoexperiment.py around line 2424
and line 2491) snapshots the canonical parent renders and PCB files
into ``round_NNNN/`` so the GUI Monitor tab can show per-round
previews after the canonical files are overwritten by the next round.

The bug this test guards against: when a round fails (compose aborts,
routing fails, gate rejects, etc.) the canonical files in
``subcircuits/<slug>/renders/`` and
``subcircuits/<slug>/parent_*.kicad_pcb`` still hold the LAST
*successful* round's output -- or, if no round has succeeded, an
orphan from a previous run that wasn't fully purged. Without the
freshness gate, the per-round copy step picks those stale files up
and labels them as this round's output.

The gate: each candidate source file's mtime must be >= the round's
wall-clock start timestamp. Any file with an older mtime predates
the round and is rejected.

This test exercises the gate as a pure unit (no pcbnew, no kicad,
no actual rounds) by building the same predicate the autoexperiment
uses inline and asserting its accept/reject contract.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest


def _fresh_for_this_round(p: Path, round_wall_started_at: float) -> bool:
    """Mirrors the inline helper in autoexperiment.py per-round flow.

    The autoexperiment defines this closure inside the round loop
    with ``round_wall_started_at`` captured. Replicated here so the
    contract is testable without spawning the full pipeline."""
    try:
        return p.stat().st_mtime >= round_wall_started_at
    except OSError:
        return False


def test_fresh_gate_accepts_file_modified_after_round_start(tmp_path: Path) -> None:
    """File written AFTER the round started -- this round's output."""
    round_wall_started_at = time.time()
    # Sleep just enough that the file's mtime resolution puts it
    # strictly after round_wall_started_at on platforms where mtime
    # has 1-second granularity (most common case is sub-second, but
    # the gate must hold either way).
    time.sleep(0.05)
    f = tmp_path / "parent_routed.png"
    f.write_bytes(b"fresh PNG data")

    assert _fresh_for_this_round(f, round_wall_started_at), (
        "A render file written after round_start must be accepted as "
        "this round's output. If this assertion fails, the freshness "
        "gate has regressed and failed rounds will display rounds-old "
        "renders as if they were current."
    )


def test_fresh_gate_rejects_file_modified_before_round_start(tmp_path: Path) -> None:
    """File from a prior round / prior run -- mtime predates round_start."""
    f = tmp_path / "parent_routed.png"
    f.write_bytes(b"stale PNG from earlier run")
    # Backdate the mtime so it predates round_start.
    old_mtime = time.time() - 3600  # one hour ago
    import os
    os.utime(f, (old_mtime, old_mtime))

    round_wall_started_at = time.time()
    assert not _fresh_for_this_round(f, round_wall_started_at), (
        "A render file with mtime predating round_start must be "
        "rejected. This is the gate that prevents prior-round / "
        "orphan-run renders from being attributed to a failed round."
    )


def test_fresh_gate_rejects_missing_file(tmp_path: Path) -> None:
    """Source paths that no longer exist must be rejected, not raise."""
    missing = tmp_path / "does_not_exist.png"
    round_wall_started_at = time.time()
    assert not _fresh_for_this_round(missing, round_wall_started_at)


def test_fresh_gate_handles_clock_skew_at_boundary(tmp_path: Path) -> None:
    """File mtime exactly equal to round_start counts as fresh.

    Edge case: if compose succeeded and rendered immediately, mtime
    can land in the same epoch second as round_start. Must accept.
    Test by setting both to the same value explicitly; in practice
    the strict-greater-than-or-equal in the gate handles this."""
    f = tmp_path / "parent_routed.png"
    f.write_bytes(b"boundary case")
    import os

    boundary = time.time()
    os.utime(f, (boundary, boundary))

    assert _fresh_for_this_round(f, boundary), (
        "mtime == round_start should be accepted (>=, not strict >). "
        "Otherwise renders produced in the same epoch second as round "
        "start get rejected and we lose them."
    )


def test_fresh_gate_module_uses_inline_predicate() -> None:
    """The autoexperiment module must define round_wall_started_at and
    use a freshness-gated copy. This test pins the source-level shape
    so a future refactor that drops the gate fails CI loudly.

    Uses ``inspect`` to read the source rather than importing private
    helpers (the predicate is an inline closure in the round loop,
    not a module-level function -- intentional, since it needs the
    round-local timestamp)."""
    import inspect
    from kicraft.cli import autoexperiment

    source = inspect.getsource(autoexperiment)

    # Must define the wall-clock timestamp at round start.
    assert "round_wall_started_at = time.time()" in source, (
        "autoexperiment.py must record round_wall_started_at = "
        "time.time() at the top of each round so per-round snapshot "
        "copies can gate on file mtime."
    )

    # Must define the inline freshness helper.
    assert "_fresh_for_this_round" in source, (
        "autoexperiment.py must define the _fresh_for_this_round "
        "helper used to gate per-round artifact snapshots."
    )

    # Must scrub stale files at round start (belt-and-suspenders).
    assert "parent_routed.png" in source and "stale_path.unlink" in source.replace(
        "_stale_path.unlink", "stale_path.unlink"
    ), (
        "Round directory must scrub pre-existing parent_routed.png / "
        "parent_stamped.png / parent_pipeline.json at round start so "
        "they cannot leak across runs."
    )
