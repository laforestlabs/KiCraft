"""Regression: antenna / mounting-hole keep-outs must survive into the DSN.

Two keep-out sources must reach FreeRouting so it won't route through them:

1. **Footprint-internal** rule-areas (e.g. the ESP32 antenna keep-out baked
   into the library footprint by Fix 0). These belong to the *footprint*, not
   the board, so ``board.Zones()`` never contained them and
   ``freerouting_runner.clear_zones`` (which iterates ``board.Zones()``) cannot
   strip them. ``ExportSpecctraDSN`` emits them in the component's local frame.

2. **Board-level** rule-areas (the parent stamp's mounting-hole / injected
   antenna keep-outs). These DO live on the board, so they only reach the DSN
   if they are still present at export time. The parent route preserves them by
   setting ``freerouting_clear_zones=False`` (compose_subcircuits) — *not*
   because any helper preserves rule areas. (An earlier comment claimed a
   non-existent ``freerouting_runner.strip_zones()`` did this; see test 3 for
   the real semantics.)

These tests pin the empirically-verified behavior so a future change to
``clear_zones``, to the parent-route config, or to KiCad's DSN export gets
caught instead of silently routing copper through an antenna.
"""
from __future__ import annotations

import os
import re
import tempfile

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer import freerouting_runner as fr  # noqa: E402

_S3_PRETTY = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "kicraft",
    "parts_library",
    "esp32-s3-wroom-1",
    "esp32-s3-wroom-1.pretty",
)
_KEEPOUT_RE = re.compile(r"\(keepout\b")


def _board_with_outline(path: str):
    """A 60x60 board with an Edge.Cuts boundary (DSN export needs one)."""
    mm = pcbnew.FromMM
    board = pcbnew.NewBoard(path)
    for x1, y1, x2, y2 in [(0, 0, 60, 0), (60, 0, 60, 60), (60, 60, 0, 60), (0, 60, 0, 0)]:
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(mm(x1), mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)
    return board


def _count_dsn_keepouts(board, pcb_path: str) -> int:
    board.BuildConnectivity()
    board.Save(pcb_path)
    dsn_path = pcb_path[:-10] + ".dsn"
    fr.export_dsn(pcb_path, dsn_path)
    return len(_KEEPOUT_RE.findall(open(dsn_path, encoding="utf-8").read()))


def _place(board, pretty: str, name: str, ref: str, x: float, y: float):
    fp = pcbnew.FootprintLoad(pretty, name)
    if fp is None:
        pytest.skip(f"could not load {name}")
    fp.SetReference(ref)
    fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y)))
    board.Add(fp)
    return fp


def test_export_dsn_emits_footprint_internal_keepout():
    """The ESP32 antenna keep-out (footprint-internal) must reach the DSN.

    Differential on the same footprint: count DSN keep-outs with the antenna
    zone present vs. stripped. The delta isolates the antenna zone as the
    source and controls for any keep-outs KiCad derives from pad/hole geometry
    (a plain count would conflate the two). The zone spans F.Cu+B.Cu, so it
    contributes 2 keep-out regions.
    """
    with tempfile.TemporaryDirectory() as d:
        with_pcb = os.path.join(d, "with_board.kicad_pcb")
        with_board = _board_with_outline(with_pcb)
        with_fp = _place(with_board, _S3_PRETTY, "WIRELM-SMD_ESP32-S3-WROOM-1", "U1", 30, 35)
        assert len(list(with_fp.Zones())) >= 1, "WROOM footprint lost its antenna keep-out"
        keepouts_with = _count_dsn_keepouts(with_board, with_pcb)

        without_pcb = os.path.join(d, "without_board.kicad_pcb")
        without_board = _board_with_outline(without_pcb)
        without_fp = _place(without_board, _S3_PRETTY, "WIRELM-SMD_ESP32-S3-WROOM-1", "U1", 30, 35)
        for z in list(without_fp.Zones()):  # strip the antenna zone
            without_fp.Remove(z)
        keepouts_without = _count_dsn_keepouts(without_board, without_pcb)

    assert keepouts_with - keepouts_without >= 2, (
        f"the WROOM antenna keep-out must reach the DSN on F.Cu and B.Cu: "
        f"with-zone={keepouts_with}, without-zone={keepouts_without} "
        f"(delta {keepouts_with - keepouts_without}, expected >=2)"
    )


def test_export_dsn_emits_board_level_rule_area_keepout():
    """A parent-stamped board-level rule-area keep-out must reach the DSN."""
    mm = pcbnew.FromMM
    with tempfile.TemporaryDirectory() as d:
        pcb = os.path.join(d, "stamp_board.kicad_pcb")
        board = _board_with_outline(pcb)
        # mirror _parent_stamp_subprocess: one rule-area zone per copper layer
        for layer in (pcbnew.F_Cu, pcbnew.B_Cu):
            z = pcbnew.ZONE(board)
            z.SetLayer(layer)
            z.SetIsRuleArea(True)
            z.SetDoNotAllowTracks(True)
            z.SetDoNotAllowVias(True)
            z.SetDoNotAllowCopperPour(True)
            o = z.Outline()
            o.NewOutline()
            for x, y in [(2, 2), (12, 2), (12, 12), (2, 12)]:
                o.Append(mm(x), mm(y))
            board.Add(z)
        keepouts = _count_dsn_keepouts(board, pcb)
    assert keepouts >= 2, (
        f"parent-stamped board-level keep-out did not reach the DSN "
        f"(got {keepouts}); FreeRouting would route through mounting holes / "
        f"injected antenna zones"
    )


def test_clear_zones_removes_board_level_but_preserves_footprint_internal():
    """Documents the real keep-out survival mechanism.

    ``clear_zones`` iterates ``board.Zones()`` and removes every board-level
    zone (rule areas included) — so the parent route must NOT call it (it sets
    ``freerouting_clear_zones=False``). Footprint-internal zones are untouched
    because they are not board zones, which is why Fix 0's antenna keep-out is
    robust on every route path.
    """
    mm = pcbnew.FromMM
    with tempfile.TemporaryDirectory() as d:
        pcb = os.path.join(d, "mixed_board.kicad_pcb")
        board = _board_with_outline(pcb)
        # board-level rule area
        z = pcbnew.ZONE(board)
        z.SetLayer(pcbnew.F_Cu)
        z.SetIsRuleArea(True)
        z.SetDoNotAllowTracks(True)
        o = z.Outline()
        o.NewOutline()
        for x, y in [(2, 2), (12, 2), (12, 12), (2, 12)]:
            o.Append(mm(x), mm(y))
        board.Add(z)
        # footprint with an internal keep-out
        fp = _place(board, _S3_PRETTY, "WIRELM-SMD_ESP32-S3-WROOM-1", "U1", 30, 35)
        fp_zones_before = len(list(fp.Zones()))
        assert fp_zones_before >= 1
        board.BuildConnectivity()
        board.Save(pcb)

        fr.clear_zones(pcb)

        reloaded = pcbnew.LoadBoard(pcb)
        board_zones_after = len(list(reloaded.Zones()))
        fp_after = next(iter(reloaded.Footprints()))
        fp_zones_after = len(list(fp_after.Zones()))

    assert board_zones_after == 0, (
        f"clear_zones should strip all board-level zones, {board_zones_after} remain"
    )
    assert fp_zones_after == fp_zones_before, (
        f"clear_zones must not touch footprint-internal zones: "
        f"{fp_zones_before} -> {fp_zones_after}"
    )
