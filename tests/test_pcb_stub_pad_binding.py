"""Seed-PCB stub must fail loudly when a wired endpoint matches no pad.

Regression source (KC-V8YWN8): Q1 used Device:Q_NPN whose literal pin numbers
are B/C/E; the SOT-23 footprint has pads 1/2/3, so the netting loop's
``.get(ep.pin, ())`` silently no-opped and every Q1 pad stayed netless — dead
copper invisible to ERC (schematic self-consistent) and DRC (no ratsnest from
netless pads). The stub now raises PadBindingError naming every unbound
endpoint instead.
"""
from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.design.models import BOM, BomPart, NetConnection, PinEndpoint
from kicraft.design.synthesis.kicad_pcb_stub import (
    PadBindingError,
    write_empty_pcb,
)


def _bom(pin_for_r1: str) -> BOM:
    parts = [
        BomPart(ref="R1", value="10k", symbol="Device:R",
                footprint="Resistor_SMD:R_0805_2012Metric", sheet="MAIN"),
        BomPart(ref="C1", value="100n", symbol="Device:C",
                footprint="Capacitor_SMD:C_0805_2012Metric", sheet="MAIN"),
    ]
    conns = [
        NetConnection(net_name="N1", sheet="MAIN",
                      endpoints=[PinEndpoint(ref="R1", pin=pin_for_r1),
                                 PinEndpoint(ref="C1", pin="1")]),
    ]
    return BOM(parts=parts, connections=conns)


def test_unbindable_endpoint_raises_pad_binding_error(tmp_path):
    # Pin "B" on a resistor footprint (pads 1/2) is the Q_NPN-class miss.
    try:
        with pytest.raises(PadBindingError) as exc:
            write_empty_pcb(tmp_path, "BAD", _bom(pin_for_r1="B"))
    except pytest.skip.Exception:  # pragma: no cover
        raise
    except Exception as e:  # missing stock footprint libs on this host
        pytest.skip(f"stock footprints unavailable: {e}")
    msg = str(exc.value)
    assert "R1.B" in msg
    assert "N1" in msg
    assert "dead copper" in msg


def test_bindable_endpoints_still_net_the_pads(tmp_path):
    try:
        out = write_empty_pcb(tmp_path, "GOOD", _bom(pin_for_r1="1"))
    except Exception as e:  # missing stock footprint libs on this host
        pytest.skip(f"stock footprints unavailable: {e}")
    board = pcbnew.LoadBoard(str(out))
    nets_by_ref = {}
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.GetNumber() == "1":
                nets_by_ref[fp.GetReference()] = pad.GetNetname()
    assert nets_by_ref.get("R1") == "N1"
    assert nets_by_ref.get("C1") == "N1"
