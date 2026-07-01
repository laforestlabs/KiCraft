"""Regression: a routed-but-gate-rejected leaf must be a best-effort fallback.

The rc=1 "board-only leaf" bug: a leaf whose board routed but was rejected by
validation (freerouting flags the round ``failed`` for residual opens) had
``result.routed == False``, so it was excluded from ``best_routed``, the
no-accepted-round recovery never fired, nothing serialized, and the auto-pin
safety net refused to compose -- dropping the whole block off-board and failing
the build rc=1. ``_round_yielded_routed_board`` fixes eligibility: a round that
stamped a board on disk qualifies, even when flagged ``failed``.
"""

from __future__ import annotations

from types import SimpleNamespace

from kicraft.cli.solve_subcircuits import _round_yielded_routed_board


def _round(routed: bool, board_path=None):
    routing: dict = {}
    if board_path is not None:
        routing["routed_board_path"] = str(board_path)
    return SimpleNamespace(routed=routed, routing=routing)


def test_cleanly_routed_round_qualifies():
    assert _round_yielded_routed_board(_round(routed=True)) is True


def test_failed_round_with_board_on_disk_qualifies(tmp_path):
    # THE FIX: freerouting flagged the round failed (routed=False) but a routed
    # board was stamped -> it is a valid best-effort fallback.
    board = tmp_path / "leaf_routed.kicad_pcb"
    board.write_text("(kicad_pcb)")
    assert _round_yielded_routed_board(_round(routed=False, board_path=board)) is True


def test_failed_round_without_board_is_excluded():
    # Genuine infra failure: no board produced -> not a fallback.
    assert _round_yielded_routed_board(_round(routed=False)) is False


def test_failed_round_with_missing_board_path_is_excluded(tmp_path):
    # A path recorded but no file on disk -> not usable.
    assert _round_yielded_routed_board(
        _round(routed=False, board_path=tmp_path / "gone.kicad_pcb")
    ) is False
