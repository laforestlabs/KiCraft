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
    for start, end in [((0, 0), (100, 0)), ((100, 0), (100, 100)),
                       ((100, 100), (0, 100)), ((0, 100), (0, 0))]:
        edge = pcbnew.PCB_SHAPE()
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(pcbnew.VECTOR2I(mm(start[0]), mm(start[1])))
        edge.SetEnd(pcbnew.VECTOR2I(mm(end[0]), mm(end[1])))
        edge.SetLayer(pcbnew.Edge_Cuts)
        board.Add(edge)
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


def test_missing_outline_moves_reference_to_fab(tmp_path, monkeypatch):
    board = _build(str(tmp_path / "b.kicad_pcb"), [(_LED, "D147", 60.0, 60.0)])
    monkeypatch.setattr("kicraft.autoplacer.hardware.silk_refdes._board_polygon", lambda board: None)
    res = legalize_refdes(board)
    assert res["moved_to_fab"] == ["D147"]
    assert _fp(board, "D147").Reference().GetLayerName() == "F.Fab"
    assert _fp(board, "D147").Reference().IsVisible()


def test_reference_is_outside_own_body(tmp_path):
    board = _build(str(tmp_path / "b.kicad_pcb"), [(_WROOM, "U1", 30.0, 30.0)])
    res = legalize_refdes(board)
    assert res["kept"] == ["U1"], res
    fp = _fp(board, "U1")
    rb = fp.Reference().GetBoundingBox()
    cb = fp.GetCourtyard(pcbnew.F_CrtYd).BBox()
    assert rb.GetRight() <= cb.GetLeft() or rb.GetLeft() >= cb.GetRight() or \
        rb.GetBottom() <= cb.GetTop() or rb.GetTop() >= cb.GetBottom()


def test_nearby_references_do_not_overlap(tmp_path):
    board = _build(
        str(tmp_path / "b.kicad_pcb"),
        [(_LED, "D100", 30.0, 30.0), (_LED, "D101", 32.0, 30.0)],
    )
    legalize_refdes(board)
    refs = [f.Reference() for f in board.Footprints()
            if f.Reference().GetLayer() == pcbnew.F_SilkS]
    if len(refs) == 2:
        a, b = (r.GetBoundingBox() for r in refs)
        assert a.GetRight() <= b.GetLeft() or b.GetRight() <= a.GetLeft() or \
            a.GetBottom() <= b.GetTop() or b.GetBottom() <= a.GetTop()





def test_invariant_visible_silk_refdes_clear_all_courtyards(tmp_path):
    # The component's own courtyard is occupied too.
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
        for other_ref, oc in courts.items():
            ox = min(rb.GetRight(), oc.GetRight()) - max(rb.GetLeft(), oc.GetLeft())
            oy = min(rb.GetBottom(), oc.GetBottom()) - max(rb.GetTop(), oc.GetTop())
            assert not (ox > 0 and oy > 0), f"{fp.GetReferenceAsString()} silk overlaps {other_ref}"


def test_concave_outline_rejects_hidden_notch():
    from kicraft.autoplacer.hardware.silk_refdes import _inside_outline
    outline = [(0, 0), (10, 0), (10, 10), (5.2, 10),
               (5.2, 4), (4.8, 4), (4.8, 10), (0, 10)]
    assert not _inside_outline((4, 3, 6, 8), outline, 0.0)


def test_outline_hole_blocks_text_even_when_corners_are_inside_board():
    from kicraft.autoplacer.hardware.silk_refdes import _inside_outline
    outer = [(0, 0), (10, 0), (10, 10), (0, 10)]
    hole = [(4.9, 4.9), (5.1, 4.9), (5.1, 5.1), (4.9, 5.1)]
    assert not _inside_outline((4, 4, 6, 6), outer, 0.2, [hole])
    assert _inside_outline((1, 1, 2, 2), outer, 0.2, [hole])


def test_courtyard_partial_geometry_failure_rejects_instead_of_waiving(tmp_path, monkeypatch):
    from kicraft.autoplacer.brain.leaf_acceptance import (
        LeafAcceptanceConfig,
        _gate_no_gross_courtyard_overlap,
    )
    from kicraft.autoplacer.courtyard_overlap import measure_courtyard_overlaps

    path = str(tmp_path / "courtyards.kicad_pcb")
    board = _build(path, [(_LED, "D1", 30, 30), (_LED, "D2", 32, 30),
                          (_LED, "D3", 60, 60)])
    first = _fp(board, "D1")
    outline = first.GetCourtyard(pcbnew.F_CrtYd).COutline(0)
    xs = [outline.CPoint(i).x for i in range(outline.PointCount())]
    width = pcbnew.ToMM(max(xs) - min(xs))
    _fp(board, "D2").SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(30 + width - 0.1), pcbnew.FromMM(30)))
    board.Save(path)
    measured = measure_courtyard_overlaps(path)
    assert measured is not None and len(measured) == 1
    ok, _ = _gate_no_gross_courtyard_overlap(
        {"board_path": path, "drc": {"courtyard": 1}}, {}, LeafAcceptanceConfig()
    )
    assert ok

    loaded = pcbnew.LoadBoard(path)

    class BrokenFootprint:
        def __init__(self, fp):
            self.fp = fp

        def GetCourtyard(self, layer):
            raise RuntimeError("courtyard geometry unavailable")

    class PartialBoard:
        def GetFootprints(self):
            return [BrokenFootprint(fp) if fp.GetReferenceAsString() == "D3" else fp
                    for fp in loaded.GetFootprints()]

    monkeypatch.setattr(pcbnew, "LoadBoard", lambda _path: PartialBoard())
    assert measure_courtyard_overlaps(path) is None
    ok, detail = _gate_no_gross_courtyard_overlap(
        {"board_path": path, "drc": {"courtyard": 2}}, {}, LeafAcceptanceConfig()
    )
    assert not ok and detail["unmeasurable"]


def test_leaf_stamp_failed_save_rejects_old_valid_board(tmp_path, monkeypatch):
    import json
    import kicraft.autoplacer.hardware._stamp_subcircuit_subprocess as stamp
    from kicraft.autoplacer.hardware.adapter import _atomic_save_board

    path_obj = tmp_path / "leaf.kicad_pcb"
    path = str(path_obj)
    board = _build(path, [(_LED, "D1", 30, 30)])
    original = path_obj.read_bytes()

    class FailedSaveBoard:
        def __getattr__(self, name):
            return getattr(board, name)

        def Save(self, _path):
            return False

    monkeypatch.setattr(pcbnew, "LoadBoard", lambda _path: FailedSaveBoard())
    payload = tmp_path / "payload.json"
    payload.write_text(json.dumps({
        "pcb_path": path, "output_path": path,
        "outline": {"tl_x": 0, "tl_y": 0, "br_x": 50, "br_y": 50},
        "components": [], "traces": [], "vias": [],
        "clear_existing_tracks": True, "clear_existing_zones": True,
        "remove_unmapped_footprints": True,
    }))
    assert stamp.main(["stamp", str(payload)]) != 0
    assert path_obj.read_bytes() == original
    with pytest.raises(RuntimeError, match="failed to save board"):
        _atomic_save_board(FailedSaveBoard(), path)
    assert path_obj.read_bytes() == original
