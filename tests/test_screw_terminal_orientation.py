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


def _make_board(
    tmp_path: Path, *, rotation: float, strip_marker: bool, ref: str = "J2"
) -> Path:
    pcbnew = pytest.importorskip("pcbnew")
    board = pcbnew.CreateEmptyBoard()
    fp = pcbnew.FootprintLoad(str(LIB_3P / f"{LIB_3P.name}.pretty"), FP_3P)
    assert fp is not None
    fp.SetReference(ref)
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


def test_facing_skips_battery_holder_refs(tmp_path):
    """A BT* part's edge zone is an ACCESS hint, not a mating contract: a
    coin-cell holder mid-board with its cell-insertion opening pointing
    anywhere is a fine board (self-eval 2026-07-19 run_31 was rejected fab-
    ready on exactly this). Same geometry as the misoriented case above --
    only the ref class changes the verdict."""
    from kicraft.autoplacer.brain.connector_edge_gap import (
        connector_edge_gaps,
        connector_facings,
    )

    pcb = _make_board(tmp_path, rotation=0.0, strip_marker=False, ref="BT1")
    zones = {"BT1": {"edge": "right"}}
    assert connector_facings(str(pcb), zones) == []
    # The flush (stranded) gate skips access-only refs the same way.
    assert connector_edge_gaps(str(pcb), zones) == []


def test_facing_omits_verified_vertical_header_strip(tmp_path):
    """A named vertical header mates along Z despite having no edge mouth."""
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings
    from kicraft.autoplacer.hardware.adapter import connector_mating_evidence

    lib = Path("/usr/share/kicad/footprints/Connector_PinHeader_2.54mm.pretty")
    if not lib.is_dir():
        pytest.skip("stock KiCad footprints not installed")
    pcbnew_mod = pytest.importorskip("pcbnew")
    fp = pcbnew_mod.FootprintLoad(str(lib), "PinHeader_2x05_P2.54mm_Vertical")
    assert fp is not None
    assert connector_mating_evidence(fp) == ("board_normal", None)
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
    out = tmp_path / "vertical_header.kicad_pcb"
    pcbnew_mod.SaveBoard(str(out), board)
    assert connector_facings(str(out), {"J9": {"edge": "right"}}) == []


def test_facing_blocks_unknown_mouth(tmp_path):
    # No vertical identity and no mouth datum must block rather than disappear.
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings

    pcb = _make_board(tmp_path, rotation=0.0, strip_marker=True)
    (v,) = connector_facings(str(pcb), ZONES)
    assert v.status == "unverified_directional"


def test_fab_gate_blocks_misoriented_connector(tmp_path):
    from kicraft.design.cli_app import _connector_misoriented

    pcb = _make_board(tmp_path, rotation=0.0, strip_marker=False)
    (tmp_path / "X_autoplacer.json").write_text(
        json.dumps({"component_zones": ZONES})
    )
    blocking, warnings = _connector_misoriented(pcb)
    assert len(blocking) == 1 and "connector_misoriented:J2" in blocking[0]
    assert warnings == []


def test_fab_gate_blocks_unverifiable_connector(tmp_path):
    from kicraft.design.cli_app import _connector_misoriented

    pcb = _make_board(tmp_path, rotation=0.0, strip_marker=True)
    (tmp_path / "X_autoplacer.json").write_text(
        json.dumps({"component_zones": ZONES})
    )
    blocking, warnings = _connector_misoriented(pcb)
    assert blocking == ["connector_orientation_unmeasured:J2"]
    assert warnings == []


# --- Layer 0: the adapter populates the mouth for edge-zoned refs --------


def test_adapter_detects_mouth_for_edge_zoned_switch(tmp_path):
    """The facings fab gate checks EVERY edge-zoned ref prefix-blind, but the
    adapter only ran mouth detection for kind == "connector" (J*). An
    edge-zoned slide switch (SW*, kind "misc") therefore placed by the
    aspect-ratio coin flip and the gate honestly rejected the bad rolls
    (self-eval 2026-07-20 run_05: connector_misoriented:SW1). The adapter
    must detect the mouth for anything zoned to an edge."""
    pytest.importorskip("pcbnew")
    from kicraft.autoplacer.hardware.adapter import KiCadAdapter

    pcb = _make_board(tmp_path, rotation=0.0, strip_marker=False, ref="SW1")

    zoned = KiCadAdapter(
        str(pcb), {"component_zones": {"SW1": {"edge": "top"}}}
    ).load()
    assert zoned.components["SW1"].kind == "misc"
    assert zoned.components["SW1"].opening_direction == 90.0

    # Unzoned, the same part keeps opening_direction None: mid-board misc
    # parts must not start orienting by body asymmetry.
    unzoned = KiCadAdapter(str(pcb), {}).load()
    assert unzoned.components["SW1"].opening_direction is None


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


# --- KC-DZQ76R: the stock MKDS horizontal terminal family ----------------
#
# The stock Phoenix MKDS-1,5 horizontal terminal has no "PCB Edge" marker and
# its rear courtyard reaches 0.61mm further from the pad row than the front,
# which beat the body-overhang heuristic's 0.5mm threshold: the placer aimed
# the screwdriver mouths INTO the board. The reviewed datum is +Y/90deg.

STOCK_LIB = Path("/usr/share/kicad/footprints/TerminalBlock_Phoenix.pretty")


def _mkds_name(n: int) -> str:
    return (
        f"TerminalBlock_Phoenix_MKDS-1,5-{n}_1x{n:02d}_P5.00mm_Horizontal"
    )


def _stock_terminal(n: int):
    pcbnew = pytest.importorskip("pcbnew")
    if not STOCK_LIB.is_dir():
        pytest.skip("stock TerminalBlock_Phoenix library not installed")
    fp = pcbnew.FootprintLoad(str(STOCK_LIB), _mkds_name(n))
    assert fp is not None, f"{_mkds_name(n)} missing from {STOCK_LIB}"
    return pcbnew, fp


@pytest.mark.parametrize("count", [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_mkds_family_opening_is_the_wire_face(count):
    """Every count the screw-terminal lowerer emits (2..12) reads +Y/90deg.

    The reviewed entry is also the load-time annotation datum, so placement,
    the fab gate and the stamped marker cannot disagree.
    """
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction
    from kicraft.parts_library.footprint_opening import (
        is_horizontal_terminal,
        reviewed_connector_opening,
    )

    pcbnew, fp = _stock_terminal(count)
    name = _mkds_name(count)
    assert is_horizontal_terminal(name)
    opening_deg, (marker_x, marker_y) = reviewed_connector_opening(name)
    assert opening_deg == 90.0
    # Marker sits on the family's front Fab datum (y=+4.6mm), centred between
    # the first and last screw pad (pads at 0, 5.00, ... mm).
    assert marker_y == 4.6
    assert marker_x == (count - 1) * 5.00 / 2
    assert detect_opening_direction(fp) == 90.0
    # Local direction is invariant to board orientation.
    for rot in (90.0, 180.0, 270.0):
        fp.SetOrientationDegrees(rot)
        assert detect_opening_direction(fp) == 90.0


def test_annotate_adds_marker_once_and_preserves_an_explicit_one():
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction
    from kicraft.parts_library.footprint_opening import annotate_connector_opening

    pcbnew, fp = _stock_terminal(4)
    name = _mkds_name(4)
    assert annotate_connector_opening(pcbnew, fp, name) is True
    markers = [
        i for i in fp.GraphicalItems()
        if i.GetLayer() == pcbnew.Dwgs_User and "edge" in i.GetText().lower()
    ]
    assert len(markers) == 1
    # Idempotent: a second call sees the marker and adds nothing.
    assert annotate_connector_opening(pcbnew, fp, name) is False
    assert len([
        i for i in fp.GraphicalItems()
        if i.GetLayer() == pcbnew.Dwgs_User and "edge" in i.GetText().lower()
    ]) == 1

    # An author-declared marker is authoritative: a *different* direction is
    # left exactly as written, not overwritten with the reviewed datum.
    pcbnew2, fp2 = _stock_terminal(4)
    author = pcbnew2.PCB_TEXT(fp2)
    author.SetText("Board Edge")
    author.SetLayer(pcbnew2.Dwgs_User)
    author.SetPosition(
        pcbnew2.VECTOR2I(pcbnew2.FromMM(7.5), pcbnew2.FromMM(-4.6))
    )
    fp2.Add(author)
    assert annotate_connector_opening(pcbnew2, fp2, name) is False
    assert detect_opening_direction(fp2) == 270.0


def test_annotation_rejects_contradictory_markers():
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction
    from kicraft.parts_library.footprint_opening import (
        annotate_connector_opening,
        explicit_edge_marker_direction,
    )

    pcbnew, fp = _stock_terminal(4)
    for y in (4.6, -4.6):
        text = pcbnew.PCB_TEXT(fp)
        text.SetText("PCB Edge")
        text.SetLayer(pcbnew.Dwgs_User)
        text.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(7.5), pcbnew.FromMM(y)))
        fp.Add(text)

    with pytest.raises(ValueError, match="contradictory"):
        explicit_edge_marker_direction(pcbnew, fp)
    with pytest.raises(ValueError, match="contradictory"):
        annotate_connector_opening(pcbnew, fp, _mkds_name(4))
    with pytest.raises(ValueError, match="contradictory"):
        detect_opening_direction(fp)


def test_unreviewed_horizontal_terminal_is_unmeasured():
    """A horizontal terminal whose actual model was never reviewed must NOT be
    guessed at from its body: asymmetry is not evidence of a mouth."""
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction
    from kicraft.parts_library.footprint_opening import (
        is_horizontal_terminal,
        reviewed_connector_opening,
    )

    pcbnew, _ = _stock_terminal(2)  # 5.00mm family (reviewed)
    unknown = "TerminalBlock_Phoenix_MKDS-1,5-2-5.08_1x02_P5.08mm_Horizontal"
    assert is_horizontal_terminal(unknown)
    assert reviewed_connector_opening(unknown) is None
    other = pcbnew.FootprintLoad(str(STOCK_LIB), unknown)
    assert other is not None
    assert detect_opening_direction(other) is None


def test_vertical_terminal_is_not_a_directional_terminal():
    from kicraft.autoplacer.hardware.adapter import detect_opening_direction
    from kicraft.parts_library.footprint_opening import is_horizontal_terminal

    vertical = (
        "TerminalBlock_Phoenix_PTSM-0,5-2-2,5-V-SMD_1x02-1MP_P2.50mm_Vertical"
    )
    assert not is_horizontal_terminal(vertical)
    assert not is_horizontal_terminal("Connector_PinHeader_2.54mm:PinHeader_1x04_P2.54mm_Vertical")
    assert not is_horizontal_terminal("Battery_Holder_Keystone_3001")
    pcbnew = pytest.importorskip("pcbnew")
    if not STOCK_LIB.is_dir():
        pytest.skip("stock TerminalBlock_Phoenix library not installed")
    fp = pcbnew.FootprintLoad(str(STOCK_LIB), vertical)
    assert fp is not None
    # Vertical terminals keep whatever the geometric heuristics say (they are
    # not gated on a family datum); the point is only that they are not
    # *classified* as directional terminals.
    detect_opening_direction(fp)


def _mkds_board(
    tmp_path: Path,
    *,
    count: int,
    rotation: float,
    edge: str,
    back: bool = False,
    ref: str = "J2",
    name: str | None = None,
) -> Path:
    """Real board: one MKDS terminal, an outline, and its edge zone."""
    pcbnew = pytest.importorskip("pcbnew")
    if not STOCK_LIB.is_dir():
        pytest.skip("stock TerminalBlock_Phoenix library not installed")
    fp_name = name or _mkds_name(count)
    fp = pcbnew.FootprintLoad(str(STOCK_LIB), fp_name)
    assert fp is not None, f"{fp_name} missing from {STOCK_LIB}"
    board = pcbnew.CreateEmptyBoard()
    fp.SetReference(ref)
    fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(150), pcbnew.FromMM(100)))
    board.Add(fp)
    if back:
        fp.Flip(fp.GetPosition(), False)
    fp.SetOrientationDegrees(rotation)
    rect = pcbnew.PCB_SHAPE(board)
    rect.SetShape(pcbnew.SHAPE_T_RECT)
    rect.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(130), pcbnew.FromMM(80)))
    rect.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(170), pcbnew.FromMM(120)))
    rect.SetLayer(pcbnew.Edge_Cuts)
    board.Add(rect)
    out = tmp_path / f"mkds_{count}_{int(rotation)}_{edge}_{int(back)}.kicad_pcb"
    pcbnew.SaveBoard(str(out), board)
    return out


def test_mkds_terminal_faces_its_zoned_edge(tmp_path):
    """rot 0 puts the +Y wire mouth out the bottom edge; rot 180 buries it."""
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings

    zones = {"J2": {"edge": "bottom"}}
    ok = _mkds_board(tmp_path, count=4, rotation=0.0, edge="bottom")
    (v,) = connector_facings(str(ok), zones)
    assert v.status == "ok" and v.opening_board_deg == 90.0

    buried = _mkds_board(tmp_path, count=4, rotation=180.0, edge="bottom")
    (v,) = connector_facings(str(buried), zones)
    assert v.status == "misoriented" and v.opening_board_deg == 270.0


def test_mkds_terminal_back_side_uses_the_shared_convention(tmp_path):
    """Flipping to B.Cu mirrors local X and restores rotation 180, so the +Y
    mouth ends up pointing at board -Y: it can face a TOP edge, not a bottom
    one. Asserted through the same edge_outward_angle/opening_board_angle pair
    the placer solves with -- no second transform formula."""
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings

    back_top = _mkds_board(tmp_path, count=2, rotation=180.0, edge="top", back=True)
    (v,) = connector_facings(str(back_top), {"J2": {"edge": "top"}})
    assert v.status == "ok"

    back_bottom = _mkds_board(
        tmp_path, count=2, rotation=180.0, edge="bottom", back=True
    )
    (v,) = connector_facings(str(back_bottom), {"J2": {"edge": "bottom"}})
    assert v.status == "misoriented"


def test_facing_blocks_unmeasured_horizontal_terminal(tmp_path):
    """A horizontal terminal with neither marker nor reviewed datum is a
    blocking 'unmeasured' verdict, not a warning: the gate cannot certify a
    direction, so it must not certify the board."""
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings
    from kicraft.design.cli_app import _connector_misoriented

    pcb = _mkds_board(
        tmp_path, count=2, rotation=0.0, edge="bottom",
        name="TerminalBlock_Phoenix_MKDS-1,5-2-5.08_1x02_P5.08mm_Horizontal",
    )
    (v,) = connector_facings(str(pcb), {"J2": {"edge": "bottom"}})
    assert v.status == "unverified_directional"

    (tmp_path / "X_autoplacer.json").write_text(
        json.dumps({"component_zones": {"J2": {"edge": "bottom"}}})
    )
    blocking, warnings = _connector_misoriented(pcb)
    assert blocking == ["connector_orientation_unmeasured:J2"]
    assert warnings == []


def test_facing_blocks_zoned_ref_missing_from_board(tmp_path):
    """A zone whose ref is not on the board cannot be verified either."""
    from kicraft.autoplacer.brain.connector_edge_gap import connector_facings
    from kicraft.design.cli_app import _connector_misoriented

    pcb = _mkds_board(tmp_path, count=2, rotation=0.0, edge="bottom")
    (v,) = connector_facings(str(pcb), {"J9": {"edge": "bottom"}})
    assert v.status == "unverified_directional"

    (tmp_path / "X_autoplacer.json").write_text(
        json.dumps({"component_zones": {"J9": {"edge": "bottom"}}})
    )
    blocking, _ = _connector_misoriented(pcb)
    assert blocking == ["connector_orientation_unmeasured:J9"]


def test_fab_gate_reports_measurement_failure_as_unmeasured(tmp_path, monkeypatch):
    """A failure to measure is blocking and named, never a silent success."""
    from kicraft.design import cli_app

    pcb = _mkds_board(tmp_path, count=2, rotation=0.0, edge="bottom")
    (tmp_path / "X_autoplacer.json").write_text(
        json.dumps({"component_zones": {"J2": {"edge": "bottom"}}})
    )

    import kicraft.autoplacer.brain.connector_edge_gap as gap

    def _boom(*args, **kwargs):
        raise RuntimeError("pcbnew exploded")

    monkeypatch.setattr(gap, "connector_facings", _boom)
    blocking, warnings = cli_app._connector_misoriented(pcb)
    assert blocking == ["connector_orientation_unmeasured:RuntimeError"]
    assert warnings == []


def test_directional_edge_candidate_uses_mating_evidence():
    """Compose accepts only verified board-normal mouthless connectors."""
    from kicraft.autoplacer.brain.types import Component, Layer, Pad, Point
    from kicraft.cli.compose_subcircuits import _directional_edge_candidate
    from kicraft.parts_library.footprint_opening import is_board_normal_header

    def _comp(*, ref, opening, mating_axis):
        return Component(
            ref=ref, value="X", pos=Point(0.0, 0.0), rotation=0.0,
            layer=Layer.FRONT, width_mm=12.7, height_mm=5.0, kind="connector",
            is_through_hole=True,
            pads=[Pad(ref=ref, pad_id="1", pos=Point(0.0, 0.0), net="A",
                      layer=Layer.FRONT)],
            opening_direction=opening,
            mating_axis=mating_axis,
        )

    # Frozen USB J6 mechanical identity: HDR-TH plus its -V- mounting variant.
    assert is_board_normal_header("HDR-TH_10P-P2.54-V-M-R2-C5-S2.54")
    assert not _directional_edge_candidate(
        _comp(ref="J6", opening=None, mating_axis="board_normal")
    )
    # A known horizontal connector keeps its measured mouth and is checked for
    # outward rotation by the compose/final gates.
    assert not _directional_edge_candidate(
        _comp(ref="J2", opening=90.0, mating_axis="in_plane")
    )
    # Unknown physical identity is never treated as vertical merely because
    # the 2D body lacks a detectable opening.
    assert _directional_edge_candidate(
        _comp(ref="JX", opening=None, mating_axis="unknown")
    )


def test_mating_axis_survives_leaf_artifact_transform():
    """Compose must receive the same vertical evidence extraction recorded."""
    from kicraft.autoplacer.brain.subcircuit_artifacts import serialize_component
    from kicraft.autoplacer.brain.subcircuit_instances import (
        _component_from_dict,
        _transform_component,
    )
    from kicraft.autoplacer.brain.types import Component, Layer, Point

    header = Component(
        ref="J6", value="2x05 header", pos=Point(1.0, 2.0), rotation=0.0,
        layer=Layer.FRONT, width_mm=12.7, height_mm=5.0, kind="connector",
        is_through_hole=True, mating_axis="board_normal",
    )
    restored = _component_from_dict(serialize_component(header))
    transformed = _transform_component(restored, Point(10.0, 20.0), 90.0)
    assert restored.mating_axis == transformed.mating_axis == "board_normal"

