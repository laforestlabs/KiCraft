"""Leaf boundary constraints, native copper containment, and visible outlines."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from kicraft.autoplacer.brain.leaf_routing import _outline_around_geometry
from kicraft.autoplacer.brain.subcircuit_solver import (
    _build_leaf_silkscreen,
)
from kicraft.autoplacer.brain.types import Component, Layer, Point, SilkscreenElement
from kicraft.autoplacer.hardware.adapter import _extract_leaf_outline_polyline_mm


CFG = {
    "silkscreen_margin_mm": 0.5,
    "silkscreen_corner_radius_mm": 1.0,
    "group_labels": {"U1": "FAKE LEAF"},
}

TOL_MM = 1e-6


def _fake_components() -> dict[str, Component]:
    return {
        "U1": Component(
            ref="U1", value="IC", pos=Point(2.5, 5.0), rotation=0.0,
            layer=Layer.FRONT, width_mm=4.0, height_mm=8.0,
        ),
    }


def _component_bbox(components: dict[str, Component]) -> dict[str, float]:
    boxes = [c.physical_bbox() for c in components.values()]
    return {
        "min_x": min(tl.x for tl, _ in boxes),
        "min_y": min(tl.y for tl, _ in boxes),
        "max_x": max(br.x for _, br in boxes),
        "max_y": max(br.y for _, br in boxes),
    }


def _silk_poly_points(silk_elements) -> list[Point]:
    for el in silk_elements:
        if el.kind == "poly":
            return list(el.points)
    return []


def test_outline_aabb_matches_silk_aabb():
    """The leaf bbox stored in ``state.board_outline`` and the silk poly's
    AABB must agree so any consumer that uses the bbox (parent composer,
    placement scorers) sees the same rectangle the rounded silk fits in.
    """
    components = _fake_components()
    outline = _outline_around_geometry(components, CFG)
    assert outline is not None
    tl, br = outline

    silk_pts = _silk_poly_points(
        _build_leaf_silkscreen(components, _component_bbox(components), extraction=None, config=CFG)
    )
    assert silk_pts
    sx0 = min(p.x for p in silk_pts)
    sy0 = min(p.y for p in silk_pts)
    sx1 = max(p.x for p in silk_pts)
    sy1 = max(p.y for p in silk_pts)

    assert math.isclose(tl.x, sx0, abs_tol=TOL_MM)
    assert math.isclose(tl.y, sy0, abs_tol=TOL_MM)
    assert math.isclose(br.x, sx1, abs_tol=TOL_MM)
    assert math.isclose(br.y, sy1, abs_tol=TOL_MM)


def test_unlabeled_leaf_has_no_outline_polyline():
    """Leaves with no ``group_labels`` match still produce no silk poly --
    legacy behaviour preserved. The Edge.Cuts stamper then falls back to
    a sharp rectangle (its previous shape), which is fine because there's
    no rounded silk to visually disagree with.
    """
    components = _fake_components()
    cfg_no_labels = dict(CFG)
    cfg_no_labels["group_labels"] = {}

    silk_elements = _build_leaf_silkscreen(
        components, _component_bbox(components), extraction=None, config=cfg_no_labels,
    )
    assert silk_elements == []
    assert _extract_leaf_outline_polyline_mm(silk_elements) is None


def test_extractor_returns_none_for_non_poly_silk():
    """The extractor must ignore silk text and any non-F.SilkS polys so
    Edge.Cuts only ever traces the dedicated leaf-outline poly, not e.g.
    a B.SilkS marking or the leaf label text.
    """
    silkscreen = [
        SilkscreenElement(kind="text", layer="F.SilkS", text="LABEL"),
        SilkscreenElement(
            kind="poly",
            layer="B.SilkS",
            points=[Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 1)],
        ),
    ]
    assert _extract_leaf_outline_polyline_mm(silkscreen) is None


@pytest.mark.parametrize("outboard_copper", [False, True])
def test_native_leaf_mating_edge_cannot_grow_around_routed_copper(
    tmp_path, outboard_copper,
):
    pcbnew = pytest.importorskip("pcbnew")
    import shutil

    if not shutil.which("kicad-cli"):
        pytest.skip("kicad-cli unavailable")
    from kicraft.autoplacer.brain.connector_edge_gap import connector_edge_gaps
    from kicraft.autoplacer.brain.leaf_routing import _leaf_mating_edges, _silk_for_leaf
    from kicraft.autoplacer.brain.types import TraceSegment
    from kicraft.autoplacer.hardware.adapter import KiCadAdapter
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc

    path = str(tmp_path / "usb-leaf.kicad_pcb")
    board = pcbnew.NewBoard(path)
    mm = pcbnew.FromMM
    for a, b in zip(((0, 0), (40, 0), (40, 40), (0, 40)),
                    ((40, 0), (40, 40), (0, 40), (0, 0))):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(pcbnew.VECTOR2I(*(mm(v) for v in a)))
        edge.SetEnd(pcbnew.VECTOR2I(*(mm(v) for v in b)))
        edge.SetLayer(pcbnew.Edge_Cuts)
        board.Add(edge)
    library = Path(__file__).parents[1] / (
        "kicraft/parts_library/usb-c-16p/usb-c-16p.pretty"
    )
    fp = pcbnew.FootprintLoad(str(library), "USB-C_SMD-TYPE-C-31-M-12_1")
    fp.SetReference("J1")
    fp.SetPosition(pcbnew.VECTOR2I(mm(20), mm(25)))
    board.Add(fp)
    signal = pcbnew.NETINFO_ITEM(board, "SIG")
    board.Add(signal)
    pad = next(p for p in fp.Pads() if p.GetAttribute() == pcbnew.PAD_ATTRIB_PTH)
    pad.SetNet(signal)
    start = Point(pcbnew.ToMM(pad.GetPosition().x), pcbnew.ToMM(pad.GetPosition().y))
    board.Save(path)
    cfg = {
        **CFG, "component_zones": {"J1": {"edge": "bottom"}},
        "group_labels": {"J1": "USB"}, "connector_edge_overhang_mm": .75,
    }
    adapter = KiCadAdapter(path, config=cfg)
    state = adapter.load()
    state.components["J1"].mating_axis = "in_plane"
    state.components["J1"].opening_direction = 90
    limits = _leaf_mating_edges(path, state.components, cfg)
    end = Point(start.x, limits["bottom"] + 1 if outboard_copper else start.y - .3)
    state.traces = [TraceSegment(start=start, end=end, net="SIG", width_mm=.25,
                                 layer=Layer.FRONT)]
    state.board_outline = _outline_around_geometry(
        state.components, cfg, traces=state.traces, mating_edges=limits,
    )
    state.silkscreen = _silk_for_leaf(
        None, state.components, cfg, traces=state.traces, outline=state.board_outline,
    )
    adapter.stamp_subcircuit_board(
        state, output_path=path, clear_existing_tracks=True,
        clear_existing_zones=True, remove_unmapped_footprints=False,
    )

    gap, = connector_edge_gaps(path, cfg["component_zones"])
    assert .7 <= gap.gap_mm <= .8
    drc = run_kicad_cli_drc(path)
    # Outboard copper must remain an honest failure, not grow the board,
    # disappear, or be silently shifted to make the connector check pass.
    assert (drc["copper_edge_clearance"] > 0) is outboard_copper
