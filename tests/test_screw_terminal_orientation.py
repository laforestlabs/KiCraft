"""KC-YJ7Q69 regression: 90-degree screw terminals MUST face the board edge.

The 3P screw terminal (CONN-TH_3P-P5.00_WJ126V-5.0-3P, auto-fetched, no
'PCB Edge' marker, body symmetric within 0.2mm around its pad row) shipped
fab_ready with its wire mouth parallel to the zoned edge:

  1. detect_opening_direction returned None (no marker, heuristics below
     threshold), so the leaf placer had no mouth to aim;
  2. _connector_wants_perp_axis misread the 3-pad row as a pin-header bank
     (its screw-terminal exclusion only covered 2-pin parts), keeping the
     mouth-parallel rotation;
  3. every downstream orientation check skipped opening_direction=None and
     the fab gate's connector check is bbox-based (rotation-blind).

These tests pin all three layers of the fix: the vendored+marked footprint
detects 90 (mouth +Y), deep-bodied connectors are excluded from the header
bank heuristic, and the fab-gate facing verdict flags a known-mouth connector
facing the wrong way (and surfaces undetectable ones instead of skipping).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point

CFG_ON = {"connector_perp_orientation": True}

LIB_3P = Path("kicraft/parts_library/screw-terminal-5mm-3p")
FP_3P = "CONN-TH_3P-P5.00_WJ126V-5.0-3P"


def _screw_terminal_3p() -> Component:
    """The WJ126V-5.0-3P as the leaf solver sees it: 3 pads in an x-row at
    5mm pitch, body 15.35 x 8.15 (deep across the row), no detected mouth."""
    pads = [
        Pad(ref="J2", pad_id=str(i + 1), pos=Point(x, 0.0), net=n, layer=Layer.FRONT)
        for i, (x, n) in enumerate([(5.0, "COM"), (0.0, "NC"), (-5.0, "NO")])
    ]
    return Component(
        ref="J2",
        value="WJ126V-5.0-3P",
        pos=Point(0.0, 0.0),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=15.35,
        height_mm=8.15,
        kind="connector",
        pads=pads,
        opening_direction=None,
    )


# --- Layer 2: the placer fallback ---------------------------------------


def test_3p_screw_terminal_is_not_a_header_bank():
    term = _screw_terminal_3p()
    assert not PlacementSolver._connector_wants_perp_axis(term, CFG_ON)


def test_3p_screw_terminal_long_axis_parallel_to_right_edge():
    # Even with no detectable mouth, the fallback must keep the wire face
    # available: long axis parallel to the edge (rot 90 on a vertical edge),
    # never pads-perpendicular (rot 0 = the shipped KC-YJ7Q69 defect).
    term = _screw_terminal_3p()
    assert PlacementSolver._best_rotation_for_edge(term, "right", CFG_ON) == 90.0


def _header_strip(width_mm: float) -> Component:
    pads = [
        Pad(ref="J1", pad_id=str(i + 1), pos=Point(0.0, i * 2.54), net="X",
            layer=Layer.FRONT)
        for i in range(3)
    ]
    return Component(
        ref="J1", value="Conn_1x03", pos=Point(0.0, 0.0), rotation=0.0,
        layer=Layer.FRONT, width_mm=width_mm, height_mm=8.71, kind="connector",
        pads=pads,
    )


def test_shallow_header_bank_still_perpendicular():
    # The KC-8A3US3 bank fix must survive: a bare 1x3 header strip
    # (body ~2.5mm deep) still turns pins-into-the-board.
    assert PlacementSolver._connector_wants_perp_axis(_header_strip(2.5), CFG_ON)


def test_real_courtyard_header_strip_perpendicular():
    # Component width/height are the COURTYARD bbox, and a real
    # PinHeader_1x03_P2.54mm_Vertical loads at 3.63 x 8.71 mm -- NOT the
    # ~2.5 mm bare body the original threshold assumed. The 3.0 mm cut read
    # every real strip as deep-bodied, silently disabling perp packing for
    # its target genre (KC-YXQ4EC: 16x 1x3 strung out 193 mm, GND pour
    # fragmented into 13 islands).
    assert PlacementSolver._connector_wants_perp_axis(_header_strip(3.63), CFG_ON)


# --- Layer 1: the vendored footprint ------------------------------------


def test_detect_opening_direction_real_3p_screw_terminal():
    """The vendored bundle carries the authoritative 'PCB Edge' marker at the
    wire-entry face (+Y at rot 0, verified against the WRL model)."""
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction

    fp = pcbnew.FootprintLoad(str(LIB_3P / f"{LIB_3P.name}.pretty"), FP_3P)
    assert fp is not None
    assert detect_opening_direction(fp) == 90.0
    # Local direction is invariant to board orientation.
    for rot in (90.0, 180.0, 270.0):
        fp.SetOrientationDegrees(rot)
        assert detect_opening_direction(fp) == 90.0


# --- Layer 3: the fab-gate facing verdict --------------------------------


def _make_board(tmp_path: Path, *, rotation: float, strip_marker: bool) -> Path:
    pcbnew = pytest.importorskip("pcbnew")
    board = pcbnew.CreateEmptyBoard()
    fp = pcbnew.FootprintLoad(str(LIB_3P / f"{LIB_3P.name}.pretty"), FP_3P)
    assert fp is not None
    fp.SetReference("J2")
    if strip_marker:
        for item in list(fp.GraphicalItems()):
            try:
                text = item.GetText()
            except Exception:
                continue
            if text and "edge" in text.lower():
                fp.Remove(item)
    fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(160), pcbnew.FromMM(100)))
    fp.SetOrientationDegrees(rotation)
    board.Add(fp)
    rect = pcbnew.PCB_SHAPE(board)
    rect.SetShape(pcbnew.SHAPE_T_RECT)
    rect.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(130), pcbnew.FromMM(90)))
    rect.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(170), pcbnew.FromMM(110)))
    rect.SetLayer(pcbnew.Edge_Cuts)
    board.Add(rect)
    out = tmp_path / "facing_test.kicad_pcb"
    pcbnew.SaveBoard(str(out), board)
    return out


ZONES = {"J2": {"edge": "right"}}


def test_facing_flags_mouth_parallel_to_edge(tmp_path):
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings

    pcb = _make_board(tmp_path, rotation=0.0, strip_marker=False)
    (v,) = connector_facings(str(pcb), ZONES)
    assert v.status == "misoriented"  # mouth +Y (90) vs right outward (0)


def test_facing_accepts_mouth_outward(tmp_path):
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings

    # board_opening = local(90) - rotation(90) = 0 = right-edge outward.
    pcb = _make_board(tmp_path, rotation=90.0, strip_marker=False)
    (v,) = connector_facings(str(pcb), ZONES)
    assert v.status == "ok"


def test_facing_omits_bare_vertical_header_strip(tmp_path):
    # A vertical 1xN pin-header strip mates from above -- it has no mouth to
    # verify, and the docstring promises it is OMITTED. Its mouth bbox
    # (courtyard + pads) measures 3.63 mm, so the old 3.0 mm cut fired the
    # unverifiable warning on every edge-zoned strip (17 per servo board).
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings

    lib = Path("/usr/share/kicad/footprints/Connector_PinHeader_2.54mm.pretty")
    if not lib.is_dir():
        pytest.skip("stock KiCad footprints not installed")
    fp = pcbnew_mod = pytest.importorskip("pcbnew")
    fp = pcbnew_mod.FootprintLoad(str(lib), "PinHeader_1x03_P2.54mm_Vertical")
    assert fp is not None
    board = pcbnew_mod.CreateEmptyBoard()
    fp.SetReference("J9")
    fp.SetPosition(pcbnew_mod.VECTOR2I(pcbnew_mod.FromMM(168), pcbnew_mod.FromMM(100)))
    board.Add(fp)
    rect = pcbnew_mod.PCB_SHAPE(board)
    rect.SetShape(pcbnew_mod.SHAPE_T_RECT)
    rect.SetStart(pcbnew_mod.VECTOR2I(pcbnew_mod.FromMM(130), pcbnew_mod.FromMM(90)))
    rect.SetEnd(pcbnew_mod.VECTOR2I(pcbnew_mod.FromMM(170), pcbnew_mod.FromMM(110)))
    rect.SetLayer(pcbnew_mod.Edge_Cuts)
    board.Add(rect)
    out = tmp_path / "strip.kicad_pcb"
    pcbnew_mod.SaveBoard(str(out), board)
    assert connector_facings(str(out), {"J9": {"edge": "right"}}) == []


def test_facing_surfaces_undetectable_mouth(tmp_path):
    # The pre-fix footprint (no marker): the gate must SAY it cannot verify,
    # not silently skip -- that silence is how KC-YJ7Q69 shipped.
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings

    pcb = _make_board(tmp_path, rotation=0.0, strip_marker=True)
    (v,) = connector_facings(str(pcb), ZONES)
    assert v.status == "unknown_mouth"


def test_fab_gate_blocks_misoriented_connector(tmp_path):
    from kicraft.design.cli_app import _connector_misoriented

    pcb = _make_board(tmp_path, rotation=0.0, strip_marker=False)
    (tmp_path / "X_autoplacer.json").write_text(
        json.dumps({"component_zones": ZONES})
    )
    blocking, warnings = _connector_misoriented(pcb)
    assert len(blocking) == 1 and "connector_misoriented:J2" in blocking[0]
    assert warnings == []


def test_fab_gate_warns_on_unverifiable_connector(tmp_path):
    from kicraft.design.cli_app import _connector_misoriented

    pcb = _make_board(tmp_path, rotation=0.0, strip_marker=True)
    (tmp_path / "X_autoplacer.json").write_text(
        json.dumps({"component_zones": ZONES})
    )
    blocking, warnings = _connector_misoriented(pcb)
    assert blocking == []
    assert len(warnings) == 1 and "J2" in warnings[0]


# --- Layer 4: vendoring lint ---------------------------------------------


def test_validate_part_warns_on_markerless_directional_connector(tmp_path, capsys):
    pytest.importorskip("pcbnew")
    from kicraft.design.cli_app import _cmd_validate_part

    bundle = tmp_path / LIB_3P.name
    shutil.copytree(LIB_3P, bundle)
    mod = bundle / f"{LIB_3P.name}.pretty" / f"{FP_3P}.kicad_mod"
    lines = [
        ln for ln in mod.read_text().splitlines()
        if "PCB Edge" not in ln
    ]
    mod.write_text("\n".join(lines) + "\n")

    import argparse

    rc = _cmd_validate_part(argparse.Namespace(path=str(bundle), update_hash=True))
    assert rc == 0
    assert "no detectable opening" in capsys.readouterr().err


def test_validate_part_quiet_on_marked_connector(capsys):
    pytest.importorskip("pcbnew")
    from kicraft.design.cli_app import _cmd_validate_part

    import argparse

    rc = _cmd_validate_part(
        argparse.Namespace(path=str(LIB_3P), update_hash=False)
    )
    assert rc == 0
    assert "no detectable opening" not in capsys.readouterr().err
