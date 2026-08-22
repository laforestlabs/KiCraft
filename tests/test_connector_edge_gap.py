"""Part 3 acceptance metric: connector_edge_gap.

Fast unit tests pin the outward-gap arithmetic for every edge (flush / overhang
/ inboard). The gated integration test composes the committed fixture and
records the live measurement -- it is the harness that found the real
stranding: J1/J2 land flush, but the top-zoned switch SW1 is ~9mm INBOARD (the
90/270 convention bug, proven + documented on parent_adapter._rotated; its fix
is gated on parent-stamp robustness). When that fix lands, SW1 flips to flush
and the xfail here is removed.
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from kicraft.autoplacer.brain.connector_edge_gap import (
    EdgeGap,
    _access_only_connector,
    _prog_debug_header,
    connector_edge_gaps,
    edge_gap_mm,
    stranded,
)


class _FakeFP:
    """Minimal footprint stub exposing the two text getters the debug-header
    classifier reads (board Value + library FPID)."""

    def __init__(self, value: str = "", fpid: str = ""):
        self._value = value
        self._fpid = fpid

    def GetValue(self) -> str:  # noqa: N802 (pcbnew API name)
        return self._value

    def GetFPIDAsString(self) -> str:  # noqa: N802 (pcbnew API name)
        return self._fpid


@pytest.mark.parametrize(
    "value, fpid, expected",
    [
        # Debug / programming headers -> access-only (skip mating gates).
        ("SWD 10-pin", "Connector:PinHeader_2x05_P2.54mm_Vertical", True),
        ("Cortex Debug", "x:Conn_ARM_JTAG_SWD_10", True),
        ("JTAG", "x", True),
        ("ICSP header", "x", True),
        ("UPDI", "x", True),
        ("", "Connector:Conn_ARM_JTAG_SWD_10", True),  # debug-ness in the FPID
        # Real off-board mating connectors and unrelated parts -> NOT exempt.
        ("USB-C", "x:USB_C_Receptacle", False),
        ("Barrel jack", "x:BarrelJack", False),
        ("Screw terminal", "x:screw-terminal-5mm-3p", False),
        ("100nF", "Device:C", False),
        ("Tactile switch", "kh-6x6x5h-stm:KH-6X6X5H-STM", False),
        ("Conn_01x04", "Connector:PinHeader_1x04", False),  # generic I/O header
    ],
)
def test_prog_debug_header_classifier(value, fpid, expected):
    assert _prog_debug_header(_FakeFP(value, fpid)) is expected


def test_access_only_connector_covers_coincell_and_debug_header():
    # Coin cell by ref (existing behavior) ...
    assert _access_only_connector("BT1", _FakeFP("CR2032 holder", "x:BS-CR2032")) is True
    # ... and an SWD debug header by footprint Value (KC-69TGAP), whose generic
    # 2x05 footprint is indistinguishable from an ordinary I/O header.
    assert _access_only_connector("J1", _FakeFP("SWD 10-pin", "x:PinHeader_2x05")) is True
    # A real edge connector stays subject to the stranding/facing gates.
    assert _access_only_connector("J2", _FakeFP("USB-C", "x:USB_C_Receptacle")) is False


def test_prog_debug_header_survives_missing_getters():
    class _Bare:
        pass

    # Never raises even when the footprint lacks the getters entirely.
    assert _prog_debug_header(_Bare()) is False

FIXTURE = (
    Path(__file__).parent / "fixtures" / "replay_workspace" / "USB_PD_TRIGGER"
)
# Same board + a parent-local connector J3 (in no leaf, edge:bottom) -- the only
# fixture that exercises _snap_parent_local's connector branch (Lever 2.1).
PARENT_LOCAL_FIXTURE = (
    Path(__file__).parent / "fixtures" / "replay_workspace" / "PARENT_LOCAL_CONN"
)
# board (x0,y0,x1,y1) = left,top,right,bottom in KiCad Y-down
BOARD = (0.0, 0.0, 20.0, 10.0)


@pytest.mark.parametrize(
    "edge, court, expected",
    [
        # flush: courtyard edge exactly on the board edge
        ("left", (0.0, 2.0, 5.0, 8.0), 0.0),
        ("right", (15.0, 2.0, 20.0, 8.0), 0.0),
        ("top", (2.0, 0.0, 8.0, 5.0), 0.0),
        ("bottom", (2.0, 5.0, 8.0, 10.0), 0.0),
        # overhang: courtyard past the board edge (positive)
        ("left", (-1.5, 2.0, 5.0, 8.0), 1.5),
        ("right", (15.0, 2.0, 22.0, 8.0), 2.0),
        # inboard / stranded: courtyard pulled in from the edge (negative)
        ("top", (2.0, 3.0, 8.0, 6.0), -3.0),
        ("bottom", (2.0, 4.0, 8.0, 9.0), -1.0),
    ],
)
def test_edge_gap_arithmetic(edge, court, expected):
    assert edge_gap_mm(edge, BOARD, court) == pytest.approx(expected)


def test_edge_gap_rejects_bad_edge():
    with pytest.raises(ValueError):
        edge_gap_mm("middle", BOARD, BOARD)


def test_stranded_filters_failures():
    gaps = [
        EdgeGap("J1", "left", 0.4, True),
        EdgeGap("SW1", "top", -9.2, False),
        EdgeGap("J2", "right", 0.4, True),
    ]
    assert [g.ref for g in stranded(gaps)] == ["SW1"]


# ---- gated integration: live measurement on the committed fixture -----------


def _compose_and_measure(tmp_path: Path, fixture: Path = FIXTURE) -> dict[str, EdgeGap]:
    stem = fixture.name  # dir name == project stem for both fixtures
    cfg = json.loads(
        (fixture / f"{stem}_autoplacer.json").read_text(encoding="utf-8")
    )
    # Some fixtures (an extra edge connector packs tighter) record their own
    # parent-compose clearance so the gate composes them as they were frozen.
    spacing = str(cfg.get("parent_compose_spacing_mm", 2.0))
    dest = tmp_path / stem
    shutil.copytree(fixture, dest)
    real = str(dest.resolve())
    for jf in (dest / ".experiments").rglob("*.json"):
        t = jf.read_text(encoding="utf-8")
        if "__KICRAFT_PROJECT_DIR__" in t:
            jf.write_text(t.replace("__KICRAFT_PROJECT_DIR__", real), encoding="utf-8")
    for p in glob.glob(str(dest / ".experiments" / "subcircuits"
                            / "subcircuit__*" / "parent_placed.kicad_pcb")):
        os.remove(p)
    env = {**os.environ, "PYTHONHASHSEED": "0", "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    rc = subprocess.run(
        [sys.executable, "-m", "kicraft.cli.compose_subcircuits",
         "--project", str(dest), "--parent", stem,
         "--pcb", str(dest / f"{stem}.kicad_pcb"),
         "--spacing-mm", spacing, "--stamp", "--seed", "0"],
        cwd=str(Path(__file__).resolve().parent.parent), env=env,
    ).returncode
    assert rc == 0, f"compose exited {rc}"
    board = sorted(glob.glob(str(dest / ".experiments" / "subcircuits"
                                 / "subcircuit__*" / "parent_placed.kicad_pcb")))[-1]
    return {g.ref: g for g in connector_edge_gaps(board, cfg["component_zones"])}


@pytest.mark.skipif(
    not os.environ.get("KICRAFT_REPLAY_E2E"),
    reason="set KICRAFT_REPLAY_E2E=1 to run (slow; spawns compose)",
)
@pytest.mark.skipif(
    not (FIXTURE / ".experiments").is_dir(), reason="frozen-leaf fixture missing"
)
def _bnc_board_with_edge_at(tmp_path, edge_x_mm: float):
    """One fixed BNC at (100,100) rot=-90 (mouth left, marker line at
    board-x 100-9.5=90.5) on a board whose LEFT edge is at ``edge_x_mm``."""
    pcbnew = pytest.importorskip("pcbnew")
    board = pcbnew.NewBoard(str(tmp_path / "bnc_gap.kicad_pcb"))
    fp = pcbnew.FootprintLoad(
        "kicraft/parts_library/bnc-pcb-jack/bnc-pcb-jack.pretty",
        "ANT-TH_KH-BNC50-3511",
    )
    fp.SetReference("J1")
    fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(100), pcbnew.FromMM(100)))
    fp.SetOrientationDegrees(-90)
    board.Add(fp)
    pts = [(edge_x_mm, 80), (140, 80), (140, 120), (edge_x_mm, 120), (edge_x_mm, 80)]
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(x1), pcbnew.FromMM(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(x2), pcbnew.FromMM(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetWidth(pcbnew.FromMM(0.1))
        board.Add(seg)
    path = tmp_path / "bnc_gap.kicad_pcb"
    board.Save(str(path))
    return path


def test_marker_datum_barrel_overhang_ok(tmp_path):
    """A marker-carrying barrel connector is measured from its 'Board Edge'
    marker, not the courtyard tip: with the edge AT the marker line (barrel
    overhanging ~21mm) the gap is ~0 and the gate passes -- the old
    courtyard-tip datum read this correct board as 21mm 'absurd overhang'."""
    path = _bnc_board_with_edge_at(tmp_path, 90.5)
    gaps = connector_edge_gaps(str(path), {"J1": {"edge": "left"}})
    assert len(gaps) == 1
    g = gaps[0]
    assert abs(g.gap_mm) < 0.6, g
    assert g.ok, g


def test_marker_datum_buried_barrel_stranded(tmp_path):
    """With the edge out at the barrel TIP (the outline-swallowed-the-barrel
    failure KC-DVA3UP shipped), the marker sits ~21mm inboard -> stranded.
    The old courtyard-tip datum called this unusable board flush/ok."""
    path = _bnc_board_with_edge_at(tmp_path, 69.0)
    gaps = connector_edge_gaps(str(path), {"J1": {"edge": "left"}})
    assert len(gaps) == 1
    g = gaps[0]
    assert g.gap_mm < -5.0, g
    assert not g.ok, g


def test_edge_connectors_flush_on_fixture(tmp_path):
    pytest.importorskip("pcbnew")
    gaps = _compose_and_measure(tmp_path)
    # The USB-C edge connectors land flush (the metric's positive case).
    assert gaps["J1"].ok, gaps["J1"]
    assert gaps["J2"].ok, gaps["J2"]


@pytest.mark.skipif(
    not os.environ.get("KICRAFT_REPLAY_E2E"),
    reason="set KICRAFT_REPLAY_E2E=1 to run (slow; spawns compose)",
)
@pytest.mark.skipif(
    not (FIXTURE / ".experiments").is_dir(), reason="frozen-leaf fixture missing"
)
def test_top_zoned_switch_not_stranded(tmp_path):
    """SW1 (top-zoned switch) lands flush. Was -9.2mm buried -- fixed by the
    RC1/RC2/RC3 chain: +rot convention, same-layer clearance, the rotation
    extremity constraint (a mouthless switch must still be its leaf's top
    extremity), and registering non-connector edge-zoned parts in
    edge_zoned_outline_sides so _repair_parent_outline doesn't bury them under
    breathing-room margin. (plan v2: docs/plans/place-route-root-cause-v2.md)"""
    pytest.importorskip("pcbnew")
    gaps = _compose_and_measure(tmp_path)
    assert gaps["SW1"].ok, gaps["SW1"]


# ---- parent-local connector (Lever 2.1): the only fixture exercising the -----
# ---- _snap_parent_local connector branch the simplification will delete. -----


@pytest.mark.skipif(
    not os.environ.get("KICRAFT_REPLAY_E2E"),
    reason="set KICRAFT_REPLAY_E2E=1 to run (slow; spawns compose)",
)
@pytest.mark.skipif(
    not (PARENT_LOCAL_FIXTURE / ".experiments").is_dir(),
    reason="parent-local-connector fixture missing",
)
def test_parent_local_fixture_leaf_connectors_flush(tmp_path):
    """Positive control: on the parent-local-connector fixture the LEAF
    connectors (J1/J2/SW1, each inside a subcircuit) still land flush. Proves
    the fixture is sane and isolates the parent-local J3 as the only stranded
    connector (the next test)."""
    pytest.importorskip("pcbnew")
    gaps = _compose_and_measure(tmp_path, PARENT_LOCAL_FIXTURE)
    assert gaps["J1"].ok, gaps["J1"]
    assert gaps["J2"].ok, gaps["J2"]
    assert gaps["SW1"].ok, gaps["SW1"]


@pytest.mark.skipif(
    not os.environ.get("KICRAFT_REPLAY_E2E"),
    reason="set KICRAFT_REPLAY_E2E=1 to run (slow; spawns compose)",
)
@pytest.mark.skipif(
    not (PARENT_LOCAL_FIXTURE / ".experiments").is_dir(),
    reason="parent-local-connector fixture missing",
)
def test_parent_local_connector_not_stranded(tmp_path):
    """A parent-local edge connector (J3, in no leaf, edge:bottom) lands flush
    on its board edge like a leaf connector. Was ~4mm stranded (the parent-local
    snap pinned it to the pre-repair outline behind a taller leaf); fixed by
    Lever 2.1 -- compose auto-wraps a loose parent-level connector as a
    single-component leaf so the solver edge-pins it as the board extremity.
    See docs/plans/place-route-root-cause-v2.md."""
    pytest.importorskip("pcbnew")
    gaps = _compose_and_measure(tmp_path, PARENT_LOCAL_FIXTURE)
    assert gaps["J3"].ok, gaps["J3"]
