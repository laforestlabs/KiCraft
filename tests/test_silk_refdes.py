"""Tests for hardware.silk_refdes geometric legalization."""
from __future__ import annotations

import os

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.hardware.silk_refdes import legalize_refdes  # noqa: E402

_LIB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kicraft",
    "parts_library",
)
_WROOM = (
    os.path.join(_LIB, "esp32-s3-wroom-1", "esp32-s3-wroom-1.pretty"),
    "WIRELM-SMD_ESP32-S3-WROOM-1",
)
_LED = (
    os.path.join(_LIB, "ws2812b-2020", "ws2812b-2020.pretty"),
    "LED-SMD_4P-L2.0-W2.0-TL_WS2812B-2020",
)


def _build(path, placements):
    """placements: list of ((pretty, name), ref, x, y) -> saved+reloaded board."""
    mm = pcbnew.FromMM
    board = pcbnew.NewBoard(path)
    for (pretty, name), ref, x, y in placements:
        fp = pcbnew.FootprintLoad(pretty, name)
        if fp is None:
            pytest.skip(f"could not load {name}")
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))
        board.Add(fp)
    board.Save(path)
    return pcbnew.LoadBoard(path)


def _fp(board, ref):
    return next(f for f in board.Footprints() if f.GetReferenceAsString() == ref)


def test_oversized_refdes_moved_to_fab(tmp_path):
    # "D147" is ~4 mm wide; the LED courtyard is ~2 mm -> can't fit -> Fab.
    board = _build(str(tmp_path / "b.kicad_pcb"), [(_LED, "D147", 60.0, 60.0)])
    res = legalize_refdes(board)
    assert res["moved_to_fab"] == ["D147"], res
    ref = _fp(board, "D147").Reference()
    assert ref.GetLayerName() == "F.Fab"
    assert ref.IsVisible(), "designator must survive on Fab for assembly"


def test_normal_refdes_kept_on_silk_and_fits_courtyard(tmp_path):
    # "U1" easily fits the WROOM's ~18x25 mm courtyard.
    board = _build(str(tmp_path / "b.kicad_pcb"), [(_WROOM, "U1", 30.0, 30.0)])
    res = legalize_refdes(board)
    assert res["kept"] == ["U1"], res
    fp = _fp(board, "U1")
    ref = fp.Reference()
    assert ref.GetLayerName() == "F.Silkscreen"
    rb = ref.GetBoundingBox()
    cb = fp.GetCourtyard(pcbnew.F_CrtYd).BBox()
    assert rb.GetWidth() <= cb.GetWidth() and rb.GetHeight() <= cb.GetHeight()


def test_adjacent_array_members_both_moved_to_fab(tmp_path):
    # A WS2812B array: 4-char D-numbers overflow their 2 mm parts -> all Fab.
    board = _build(
        str(tmp_path / "b.kicad_pcb"),
        [(_LED, "D100", 30.0, 30.0), (_LED, "D101", 32.0, 30.0)],
    )
    res = legalize_refdes(board)
    assert set(res["moved_to_fab"]) == {"D100", "D101"}, res
    assert all(
        _fp(board, r).Reference().GetLayerName() == "F.Fab" for r in ("D100", "D101")
    )


def test_stamp_subprocesses_wire_in_legalize_refdes():
    """Both stamp subprocesses must call the refdes pass before save."""
    from pathlib import Path

    import kicraft.autoplacer.hardware._stamp_subcircuit_subprocess as leaf
    import kicraft.cli._parent_stamp_subprocess as parent

    for mod in (parent, leaf):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "legalize_refdes(board" in src, (
            f"{mod.__name__} must call legalize_refdes(board, ...) before board.Save()"
        )


def test_invariant_visible_silk_refdes_fit_and_clear_neighbours(tmp_path):
    # Mixed board: every refdes left on silk must fit its courtyard and clear
    # all other courtyards; the rest land on Fab.
    board = _build(
        str(tmp_path / "b.kicad_pcb"),
        [(_WROOM, "U1", 30.0, 30.0), (_LED, "D5", 60.0, 60.0)],
    )
    legalize_refdes(board)
    courts = {
        f.GetReferenceAsString(): f.GetCourtyard(pcbnew.F_CrtYd).BBox()
        for f in board.Footprints()
    }
    for fp in board.Footprints():
        ref = fp.Reference()
        if ref.GetLayerName() != "F.Silkscreen":
            continue
        rb = ref.GetBoundingBox()
        own = courts[fp.GetReferenceAsString()]
        assert rb.GetWidth() <= own.GetWidth() and rb.GetHeight() <= own.GetHeight()
        for other_ref, oc in courts.items():
            if other_ref == fp.GetReferenceAsString():
                continue
            ox = min(rb.GetRight(), oc.GetRight()) - max(rb.GetLeft(), oc.GetLeft())
            oy = min(rb.GetBottom(), oc.GetBottom()) - max(rb.GetTop(), oc.GetTop())
            assert not (ox > 0 and oy > 0), f"{fp.GetReferenceAsString()} silk overlaps {other_ref}"
