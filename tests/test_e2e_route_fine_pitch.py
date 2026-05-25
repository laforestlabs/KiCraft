"""Opt-in end-to-end routing test (E2E pipeline findings 1/3/4).

This is the route-level guard the findings were found by hand: it builds a real
board containing the vendored fine-pitch USB-C connector plus passives, then
drives the *actual* FreeRouting pipeline (DSN export -> FreeRouting jar -> SES
import -> DRC) and asserts the board routes cleanly.

It exercises:
  * Finding 1a -- the board has an Edge.Cuts outline enclosing all parts.
  * Finding 3  -- fine-pitch clearance auto-lowering lets the autorouter escape
                  the USB-C pad field (without it: an unroutable net + clearance
                  violations).
  * Finding 4  -- the route completes and is accepted (a pcbnew teardown SIGSEGV
                  after a successful save no longer reports the route as failed).

It is SLOW (a real FreeRouting run, ~1-3 min) and needs external tooling, so it
is opt-in:

    KICRAFT_RUN_E2E=1 .venv/bin/python -m pytest tests/test_e2e_route_fine_pitch.py -v

Without KICRAFT_RUN_E2E set, or if pcbnew / the FreeRouting jar / kicad-cli are
unavailable, it skips.
"""
from __future__ import annotations

import os
import shutil
import tempfile

import pytest

pytestmark = [pytest.mark.slow, pytest.mark.e2e]

_USB_C_PRETTY = "kicraft/parts_library/usb-c-16p/usb-c-16p.pretty"
_USB_C_NAME = "USB-C_SMD-TYPE-C-31-M-12_1"


def _require_e2e():
    if not os.environ.get("KICRAFT_RUN_E2E"):
        pytest.skip("opt-in E2E: set KICRAFT_RUN_E2E=1 to run")
    from kicraft.autoplacer.config import DEFAULT_CONFIG

    jar = DEFAULT_CONFIG.get("freerouting_jar", "")
    if not jar or not os.path.isfile(jar):
        pytest.skip(f"FreeRouting jar not found at {jar!r}")
    if shutil.which("kicad-cli") is None:
        pytest.skip("kicad-cli not on PATH")
    if not os.path.isdir(_USB_C_PRETTY):
        pytest.skip("vendored usb-c footprint unavailable")


def _build_fine_pitch_board(pcbnew, path: str) -> None:
    """USB-C connector + 2 resistors, wired so nets must escape the dense
    USB-C pad field, with an Edge.Cuts outline enclosing everything."""
    board = pcbnew.NewBoard(path)

    j1 = pcbnew.FootprintLoad(_USB_C_PRETTY, _USB_C_NAME)
    j1.SetReference("J1")
    j1.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(50), pcbnew.FromMM(50)))
    board.Add(j1)

    res_pretty = None
    for cand in (
        "/usr/share/kicad/footprints/Resistor_SMD.pretty",
        "/usr/share/kicad/modules/Resistor_SMD.pretty",
    ):
        if os.path.isdir(cand):
            res_pretty = cand
            break
    if res_pretty is None:
        pytest.skip("stock Resistor_SMD.pretty not found")

    resistors = []
    for i, ref in enumerate(("R1", "R2")):
        r = pcbnew.FootprintLoad(res_pretty, "R_0805_2012Metric")
        r.SetReference(ref)
        r.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(56 + i * 4), pcbnew.FromMM(45)))
        board.Add(r)
        resistors.append(r)

    def add_net(name):
        n = pcbnew.NETINFO_ITEM(board, name)
        board.Add(n)
        return n.GetNetCode()

    # Two different-net signals taken off adjacent (0.1mm-gap) USB-C pads --
    # the escape that fails under a 0.2mm clearance rule -- plus a shared GND.
    j1_pads = list(j1.Pads())
    cc1, cc2, gnd = add_net("CC1"), add_net("CC2"), add_net("GND")
    j1_pads[0].SetNetCode(cc1)
    j1_pads[1].SetNetCode(cc2)
    resistors[0].FindPadByNumber("1").SetNetCode(cc1)
    resistors[0].FindPadByNumber("2").SetNetCode(gnd)
    resistors[1].FindPadByNumber("1").SetNetCode(cc2)
    resistors[1].FindPadByNumber("2").SetNetCode(gnd)

    # Edge.Cuts enclosing all parts + margin (Finding 1a style).
    bbox = None
    for fp in board.GetFootprints():
        fb = fp.GetBoundingBox()
        bbox = fb if bbox is None else (bbox.Merge(fb) or bbox)
    m = pcbnew.FromMM(4.0)
    rect = pcbnew.PCB_SHAPE(board)
    rect.SetShape(pcbnew.SHAPE_T_RECT)
    rect.SetStart(pcbnew.VECTOR2I(bbox.GetLeft() - m, bbox.GetTop() - m))
    rect.SetEnd(pcbnew.VECTOR2I(bbox.GetRight() + m, bbox.GetBottom() + m))
    rect.SetLayer(pcbnew.Edge_Cuts)
    rect.SetWidth(pcbnew.FromMM(0.1))
    board.Add(rect)

    board.BuildConnectivity()
    board.Save(path)


def test_fine_pitch_board_routes_clean():
    _require_e2e()
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.config import DEFAULT_CONFIG
    from kicraft.autoplacer.freerouting_runner import (
        min_intra_footprint_pad_gap_mm,
        route_with_freerouting,
        validate_routed_board,
    )

    d = tempfile.mkdtemp()
    pcb = os.path.join(d, "fine_pitch.kicad_pcb")
    routed = os.path.join(d, "fine_pitch_routed.kicad_pcb")
    _build_fine_pitch_board(pcbnew, pcb)

    # Sanity: the board really is fine-pitch (auto-lowering should engage).
    gap = min_intra_footprint_pad_gap_mm(pcb)
    assert gap is not None and gap < 0.2, f"expected fine pitch, got {gap}"

    cfg = {**DEFAULT_CONFIG, "freerouting_timeout_s": 240}
    stats = route_with_freerouting(
        kicad_pcb_path=pcb,
        output_path=routed,
        jar_path=cfg["freerouting_jar"],
        config=cfg,
    )

    assert os.path.isfile(routed), "routed board was not written"
    validation = validate_routed_board(routed, cfg=cfg)

    # The route must complete and be accepted. Any clearance violations must be
    # footprint-internal (the connector's own pad field), i.e. waived -- not a
    # routing failure.
    drc = validation.get("drc", {})
    assert drc.get("unconnected", 0) == 0, (
        f"unrouted nets remain: {drc.get('unconnected')} "
        f"(stats={stats.get('unrouted')})"
    )
    assert validation["accepted"], (
        f"routed board rejected: {validation.get('rejection_reasons')} "
        f"drc={ {k: v for k, v in drc.items() if isinstance(v, int)} }"
    )
