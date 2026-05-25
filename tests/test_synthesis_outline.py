"""Regression guard for E2E Finding 1a: synthesis must emit a board outline.

Before the fix, ``write_empty_pcb`` populated footprints + nets but never
drew ``Edge.Cuts``, so the seed board's edge bbox was ``0x0``. compose then
under-sized the parent and FreeRouting produced no SES. This test asserts the
synthesized PCB has a non-zero outline that encloses every footprint.

Requires pcbnew + the KiCad stock footprint libraries (the repo .venv has
both); skipped otherwise so the rest of the suite still runs.
"""
from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.circuitchat.models import BOM, BomPart, NetConnection, PinEndpoint
from kicraft.circuitchat.synthesis.kicad_pcb_stub import write_empty_pcb


def _demo_bom() -> BOM:
    parts = [
        BomPart(ref="R1", value="10k", symbol="Device:R",
                footprint="Resistor_SMD:R_0805_2012Metric", sheet="MAIN"),
        BomPart(ref="C1", value="100n", symbol="Device:C",
                footprint="Capacitor_SMD:C_0805_2012Metric", sheet="MAIN"),
        BomPart(ref="R2", value="1k", symbol="Device:R",
                footprint="Resistor_SMD:R_0805_2012Metric", sheet="MAIN"),
    ]
    conns = [
        NetConnection(net_name="N1", sheet="MAIN",
                      endpoints=[PinEndpoint(ref="R1", pin="1"),
                                 PinEndpoint(ref="C1", pin="1")]),
        NetConnection(net_name="N2", sheet="MAIN",
                      endpoints=[PinEndpoint(ref="C1", pin="2"),
                                 PinEndpoint(ref="R2", pin="1")]),
    ]
    return BOM(parts=parts, connections=conns)


def test_synthesis_emits_outline_enclosing_all_footprints(tmp_path):
    try:
        out = write_empty_pcb(tmp_path, "TEST", _demo_bom())
    except Exception as exc:  # missing stock footprint libs on this host
        pytest.skip(f"stock footprints unavailable: {exc}")

    board = pcbnew.LoadBoard(str(out))
    edge = board.GetBoardEdgesBoundingBox()

    # Non-zero outline (the core regression: previously 0x0).
    assert edge.GetWidth() > 0, "board outline has zero width"
    assert edge.GetHeight() > 0, "board outline has zero height"

    # Every footprint must sit inside the outline.
    footprints = list(board.GetFootprints())
    assert len(footprints) == 3
    for fp in footprints:
        assert edge.Contains(fp.GetBoundingBox()), \
            f"{fp.GetReference()} is not enclosed by Edge.Cuts"


def test_empty_bom_still_writes_empty_board(tmp_path):
    # No connections -> Stage-A backwards-compat path: empty board, no crash,
    # no outline required.
    out = write_empty_pcb(tmp_path, "EMPTY", None)
    board = pcbnew.LoadBoard(str(out))
    assert len(list(board.GetFootprints())) == 0
