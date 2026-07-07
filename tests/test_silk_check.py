"""SilkCheck — the layout scorer's silkscreen self-description term (Phase 4b).

The pure guard locks silk DRC violation types as MINOR (a legend can never fail a
board); the pcbnew integration builds tiny boards to exercise presence scoring.
"""
import pytest

from kicraft.scoring.drc_check import CRITICAL_TYPES, MAJOR_TYPES, MINOR_TYPES


def test_silk_drc_violations_stay_minor():
    # Phase-4 contract: adding a legend/labels can never fail a build. Silk
    # overlaps are cosmetic — MINOR in the scorer, uncounted by the fab gate.
    for t in ("silk_overlap", "silk_over_copper", "silk_edge_clearance"):
        assert t in MINOR_TYPES
        assert t not in CRITICAL_TYPES
        assert t not in MAJOR_TYPES


pcbnew = pytest.importorskip("pcbnew")

from kicraft.scoring.silk_check import SilkCheck  # noqa: E402


def _board_with_silk(lines):
    board = pcbnew.BOARD()
    for s in lines:
        t = pcbnew.PCB_TEXT(board)
        t.SetText(s)
        t.SetLayer(pcbnew.F_SilkS)
        board.Add(t)
    return board


def test_legend_and_label_scores_full():
    board = _board_with_silk([
        "USB-C PD Trigger",                    # title (content line)
        "KiCraft KC-TEST rev 1.0 2026-01-01",  # maker/legend line
        "IN 9/12/20V",                         # functional label (content line)
    ])
    res = SilkCheck().run(board, {})
    assert res.metrics["legend_present"] is True
    assert res.metrics["content_line_count"] == 2
    assert res.score == 100.0


def test_legend_only_gets_partial_label_credit():
    board = _board_with_silk(["My Board", "KiCraft KC-TEST rev 1.0 2026-01-01"])
    res = SilkCheck().run(board, {})
    assert res.metrics["legend_present"] is True
    assert res.metrics["content_line_count"] == 1  # title only, no label
    assert res.score == 85.0
    assert any(i.severity == "info" for i in res.issues)


def test_no_silk_scores_zero_and_warns():
    board = pcbnew.BOARD()
    res = SilkCheck().run(board, {})
    assert res.metrics["legend_present"] is False
    assert res.score == 0.0
    assert any(i.severity == "warning" for i in res.issues)


def test_dropped_labels_dock_the_score():
    board = _board_with_silk([
        "KiCraft KC-TEST rev 1.0", "IN 9V", "OUT 5V"])
    res = SilkCheck().run(board, {"_silk_dropped": ["usb-in: no clear space"]})
    assert res.metrics["dropped_count"] == 1
    assert res.score == 95.0  # 100 - 5*1


def test_refdes_silk_is_not_counted_as_legend():
    # Footprint reference/value silk lives on the footprint, not board.GetDrawings();
    # a board carrying only refdes text must NOT read as self-describing. Here we
    # add no board-level text at all, standing in for a refdes-only board.
    board = pcbnew.BOARD()
    res = SilkCheck().run(board, {})
    assert res.metrics["legend_present"] is False
