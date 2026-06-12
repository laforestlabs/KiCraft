"""Mounting-hole footprint synthesis: planning + real stamping.

Planning (pure): user holes map onto existing H-refs first (legacy
contract), surplus holes get fresh non-colliding H9xx refs and stock
footprint specs.

Stamping (pcbnew): the parent stamp subprocess loads the stock NPTH
footprint for each ``synthesize_footprints`` entry, places it at the
hole position, refuses ref collisions, and the result carries the
correct drill with no copper pads, excluded from CPL/BOM by the stock
footprint's own attributes.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from kicraft.autoplacer.brain.types import Point
from kicraft.layout_editor.holes import (
    SCREW_TABLE,
    allocate_synth_refs,
    find_stock_mounting_hole_lib,
    plan_mounting_holes,
    screw_spec,
)
from kicraft.layout_editor.model import ManualMountingHole


def _hole(index: int, screw: str = "M3") -> ManualMountingHole:
    return ManualMountingHole(
        index=index, corner=None, inset_mm=5.0,
        pos=Point(5.0 * (index + 1), 5.0), screw=screw,
    )


def test_plan_maps_existing_refs_first_then_synthesizes():
    holes = [_hole(0), _hole(1), _hole(2), _hole(3)]
    mapped, synth = plan_mounting_holes(holes, ["H4", "H86"], {"U1", "R1"})
    # Alphabetical existing-ref order preserved (H4 < H86).
    assert [(h.index, ref) for h, ref in mapped] == [(0, "H4"), (1, "H86")]
    assert [(h.index, ref) for h, ref in synth] == [(2, "H901"), (3, "H902")]


def test_plan_with_no_existing_refs_synthesizes_all():
    holes = [_hole(0), _hole(1)]
    mapped, synth = plan_mounting_holes(holes, [], set())
    assert mapped == []
    assert [ref for _h, ref in synth] == ["H901", "H902"]


def test_synth_refs_avoid_collisions_case_insensitively():
    refs = allocate_synth_refs(3, {"H901", "h902"})
    assert refs == ["H903", "H904", "H905"]


def test_screw_spec_fallback():
    assert screw_spec("M2.5").drill_mm == 2.7
    assert screw_spec("M9").screw == "M3"  # unknown -> default
    assert screw_spec(None).screw == "M3"
    assert screw_spec("M4").fp_name == "MountingHole_4.3mm_M4"


_STOCK_LIB = find_stock_mounting_hole_lib()


@pytest.mark.skipif(_STOCK_LIB is None, reason="stock MountingHole.pretty not found")
def test_every_screw_table_footprint_exists_in_stock_lib():
    for spec in SCREW_TABLE.values():
        path = _STOCK_LIB / f"{spec.fp_name}.kicad_mod"
        assert path.is_file(), f"{spec.fp_name} missing from {_STOCK_LIB}"
        text = path.read_text(encoding="utf-8")
        # NPTH variant: no copper, auto-excluded from CPL/BOM.
        assert "exclude_from_pos_files" in text
        assert "exclude_from_bom" in text


# --- Real stamping through the subprocess -----------------------------------

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.freerouting_runner import _run_pcbnew_script_file  # noqa: E402
from kicraft.cli.compose_subcircuits import _PARENT_STAMP_SCRIPT_PATH  # noqa: E402


def _make_source_board(path: str, with_h_ref: bool = False) -> None:
    board = pcbnew.NewBoard(path)
    mm = pcbnew.FromMM
    for x1, y1, x2, y2 in [(0, 0, 60, 0), (60, 0, 60, 60), (60, 60, 0, 60), (0, 60, 0, 0)]:
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(mm(x1), mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)
    if with_h_ref:
        fp = pcbnew.FootprintLoad(str(_STOCK_LIB), "MountingHole_3.2mm_M3")
        fp.SetReference("H901")
        fp.SetPosition(pcbnew.VECTOR2I(mm(30), mm(30)))
        board.Add(fp)
    board.Save(path)


def _stamp(payload: dict) -> None:
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="stamp_mh_")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f)
        _run_pcbnew_script_file(_PARENT_STAMP_SCRIPT_PATH, tmp)
    finally:
        os.unlink(tmp)


def _base_payload(src: str, out: str) -> dict:
    return {
        "pcb_path": src,
        "output_path": out,
        "components": [],
        "traces": [],
        "vias": [],
        "silkscreen": [],
        "keepouts": [],
        "outline": {"tl_x": 0.0, "tl_y": 0.0, "br_x": 60.0, "br_y": 60.0},
    }


@pytest.mark.skipif(_STOCK_LIB is None, reason="stock MountingHole.pretty not found")
def test_stamp_synthesizes_footprint_at_position_with_correct_drill():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.kicad_pcb")
        out = os.path.join(d, "out.kicad_pcb")
        _make_source_board(src)
        payload = _base_payload(src, out)
        payload["synthesize_footprints"] = [
            {"ref": "H901", "x": 5.0, "y": 5.0,
             "lib_dir": str(_STOCK_LIB), "fp_name": "MountingHole_3.2mm_M3",
             "screw": "M3"},
            {"ref": "H902", "x": 55.0, "y": 55.0,
             "lib_dir": str(_STOCK_LIB), "fp_name": "MountingHole_2.2mm_M2",
             "screw": "M2"},
        ]
        _stamp(payload)

        board = pcbnew.LoadBoard(out)
        fps = {fp.GetReferenceAsString(): fp for fp in board.Footprints()}
        assert "H901" in fps and "H902" in fps

        h901 = fps["H901"]
        pos = h901.GetPosition()
        assert pcbnew.ToMM(pos.x) == pytest.approx(5.0, abs=1e-3)
        assert pcbnew.ToMM(pos.y) == pytest.approx(5.0, abs=1e-3)
        pads = list(h901.Pads())
        assert len(pads) == 1  # the NPTH "pad" carrying the drill
        drill = pads[0].GetDrillSize()
        assert pcbnew.ToMM(drill.x) == pytest.approx(3.2, abs=1e-3)
        # NPTH: hole only, no copper anywhere.
        assert pads[0].GetAttribute() == pcbnew.PAD_ATTRIB_NPTH

        drill_m2 = list(fps["H902"].Pads())[0].GetDrillSize()
        assert pcbnew.ToMM(drill_m2.x) == pytest.approx(2.2, abs=1e-3)


@pytest.mark.skipif(_STOCK_LIB is None, reason="stock MountingHole.pretty not found")
def test_stamp_refuses_ref_collision_with_source_board():
    with tempfile.TemporaryDirectory() as d:
        src = os.path.join(d, "src.kicad_pcb")
        out = os.path.join(d, "out.kicad_pcb")
        _make_source_board(src, with_h_ref=True)  # board already has H901
        payload = _base_payload(src, out)
        payload["synthesize_footprints"] = [
            {"ref": "H901", "x": 5.0, "y": 5.0,
             "lib_dir": str(_STOCK_LIB), "fp_name": "MountingHole_3.2mm_M3",
             "screw": "M3"},
        ]
        with pytest.raises(RuntimeError, match="already"):
            _stamp(payload)
