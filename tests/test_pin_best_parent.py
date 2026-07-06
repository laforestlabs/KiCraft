"""The parent phase must ship the SELECTED round's board, not the last-written one.

Regression from KC-WMS7KK (projects/1/565): every parent round overwrites the
shared canonical ``subcircuits/<parent>/parent_routed.kicad_pcb``. Round 2 was
kept (validated clean, copied to ``.experiments/best/``), then discarded round 3
clobbered the canonical scratch with a DRC-dirty board; promotion resolves the
canonical path by recency (``artifact_paths.resolve_parent_board``), so the
build shipped the rejected board and failed verify -- a false rc7 with a
fab-ready board on disk. ``_pin_best_parent`` restores the best round's board
over the canonical file at run end (parent-side twin of the leaf auto-pin).
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from kicraft.cli.artifact_paths import resolve_parent_board
from kicraft.cli.autoexperiment import _pin_best_parent

BEST_BYTES = b"(kicad_pcb ROUND2-BEST)"
LAST_BYTES = b"(kicad_pcb ROUND3-DISCARDED)"


def _project(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A minimal project tree: canonical parent artifact holding the LAST
    round's board, and a best/ pin holding the kept round's board."""
    project_dir = tmp_path / "proj"
    canonical_dir = project_dir / ".experiments" / "subcircuits" / "subcircuit__x"
    canonical_dir.mkdir(parents=True)
    (canonical_dir / "parent_routed.kicad_pcb").write_bytes(LAST_BYTES)
    # The markers _discover_latest_parent_artifact_dir keys on.
    (canonical_dir / "metadata.json").write_text(
        json.dumps(
            {"parent_composition": True, "schema_version": "parent-compose-v1"}
        )
    )
    best_dir = project_dir / ".experiments" / "best"
    best_dir.mkdir(parents=True)
    (best_dir / "parent_routed.kicad_pcb").write_bytes(BEST_BYTES)
    (best_dir / "parent_routed.kicad_pro").write_text("{}")
    return project_dir, best_dir, canonical_dir


def test_pin_restores_best_round_board_over_last_round(tmp_path: Path) -> None:
    project_dir, best_dir, canonical_dir = _project(tmp_path)
    best_round = SimpleNamespace(round_num=2, parent_routed=True)

    _pin_best_parent(project_dir, best_dir, best_round)

    canonical = canonical_dir / "parent_routed.kicad_pcb"
    assert canonical.read_bytes() == BEST_BYTES
    # Sibling project file rides along so KiCad opens the board cleanly.
    assert (canonical_dir / "parent_routed.kicad_pro").is_file()
    # What promotion resolves is now the selected board.
    resolved = resolve_parent_board(project_dir, kind="routed")
    assert resolved is not None
    assert resolved.read_bytes() == BEST_BYTES


def test_pin_noop_without_kept_routed_round(tmp_path: Path) -> None:
    project_dir, best_dir, canonical_dir = _project(tmp_path)
    canonical = canonical_dir / "parent_routed.kicad_pcb"

    _pin_best_parent(project_dir, best_dir, None)
    assert canonical.read_bytes() == LAST_BYTES

    # A kept round that never routed a parent must not resurrect a stale
    # best/ board from an earlier invocation.
    _pin_best_parent(
        project_dir, best_dir, SimpleNamespace(round_num=1, parent_routed=False)
    )
    assert canonical.read_bytes() == LAST_BYTES


def test_pin_noop_when_best_board_missing(tmp_path: Path) -> None:
    project_dir, best_dir, canonical_dir = _project(tmp_path)
    (best_dir / "parent_routed.kicad_pcb").unlink()

    _pin_best_parent(
        project_dir, best_dir, SimpleNamespace(round_num=2, parent_routed=True)
    )
    assert (canonical_dir / "parent_routed.kicad_pcb").read_bytes() == LAST_BYTES
