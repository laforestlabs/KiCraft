"""A leaf that routes boards but serializes no result must be DETECTED rather
than silently dropped from the parent compose (which strands its components
off-board).

A leaf solve can route its rounds (producing ``round_*_leaf_routed`` boards) but
die before writing ``metadata.json`` / ``solved_layout.json`` -- e.g. when every
round fails the acceptance gate. ``_all_leaf_artifacts`` enumerates leaves by
those JSON files, so such a "board-only" leaf is invisible to auto-pin and the
parent silently drops it. ``_board_only_leaf_dirs`` surfaces them so the run can
fail loudly instead of shipping a board missing a whole leaf.
"""

from __future__ import annotations

from pathlib import Path

from kicraft.cli.autoexperiment import _board_only_leaf_dirs


def _leaf(root: Path, name: str, *, serialized: bool, routed_board: bool) -> Path:
    d = root / ".experiments" / "subcircuits" / name
    d.mkdir(parents=True)
    if serialized:
        (d / "metadata.json").write_text("{}")
        (d / "solved_layout.json").write_text("{}")
    if routed_board:
        (d / "round_0000_leaf_routed.kicad_pcb").write_text("(kicad_pcb)")
    return d


def test_board_only_leaf_is_detected(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    _leaf(proj, "ok__a", serialized=True, routed_board=True)  # normal leaf
    _leaf(proj, "dropped__b", serialized=False, routed_board=True)  # board-only
    _leaf(proj, "empty__c", serialized=False, routed_board=False)  # nothing to pin

    assert [d.name for d in _board_only_leaf_dirs(proj)] == ["dropped__b"]


def test_no_board_only_leaves_when_all_serialized(tmp_path: Path) -> None:
    proj = tmp_path / "proj"
    _leaf(proj, "a__1", serialized=True, routed_board=True)
    _leaf(proj, "b__2", serialized=True, routed_board=True)
    assert _board_only_leaf_dirs(proj) == []


def test_missing_subcircuits_dir_is_empty(tmp_path: Path) -> None:
    assert _board_only_leaf_dirs(tmp_path / "nonexistent") == []
