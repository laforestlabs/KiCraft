"""Back-side parts: BomPart.side -> autoplacer.json component_layers -> Component.layer.

"A header on the back side" must actually land on B.Cu. Synthesis records the
side intent as a `component_layers` map in autoplacer.json (replay-safe -- read
from the solver config, not re-synthesised); the adapter reads it at board load
and sets Component.layer = BACK, and the existing stamp path flips the footprint.
These tests pin each hop.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from kicraft.design.models import BOM, Architecture, BomPart, Sheet
from kicraft.design.synthesis.autoplacer import write_autoplacer_json


def _part(ref: str, sheet: str = "MCU", side=None) -> BomPart:
    return BomPart(
        ref=ref, value="x", sheet=sheet, symbol="Device:R",
        footprint="Resistor_SMD:R_0402_1005Metric", side=side,
    )


def _arch() -> Architecture:
    return Architecture(
        sheets=[Sheet(name="MCU", stem="MCU", function="microcontroller")],
        power_nets=["+3V3", "GND"], inter_sheet_nets=[],
    )


def test_bompart_side_defaults_and_validates():
    assert _part("U1").side is None
    assert _part("U1", side="front").side == "front"
    assert _part("J1", side="back").side == "back"
    with pytest.raises(ValidationError):
        _part("J1", side="bottom")  # only front | back | None allowed


def test_write_autoplacer_emits_component_layers_for_back_parts(tmp_path):
    bom = BOM(parts=[_part("U1"), _part("J1", side="back")])
    cfg = json.loads(write_autoplacer_json(tmp_path, "WIDGET", _arch(), bom).read_text())
    assert cfg["component_layers"] == {"J1": "back"}


def test_write_autoplacer_omits_component_layers_when_all_front(tmp_path):
    bom = BOM(parts=[_part("U1"), _part("J1", side="front")])
    cfg = json.loads(write_autoplacer_json(tmp_path, "WIDGET", _arch(), bom).read_text())
    assert "component_layers" not in cfg


def test_adapter_honors_component_layers_override():
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.brain.types import Layer
    from kicraft.autoplacer.hardware.adapter import KiCadAdapter

    mm = pcbnew.FromMM
    with tempfile.TemporaryDirectory() as d:
        path = str(Path(d) / "b.kicad_pcb")
        board = pcbnew.NewBoard(path)
        for ref, x in (("U1", 20), ("U2", 40)):
            fp = pcbnew.FOOTPRINT(board)
            fp.SetReference(ref)
            fp.SetValue("x")
            fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(20)))
            board.Add(fp)
        board.BuildConnectivity()
        board.Save(path)

        # U1 overridden to back; U2 (no entry) stays front. The seed footprints
        # are NOT flipped on disk -- the override is layer-only at load, and the
        # stamp path owns the actual geometry flip downstream.
        st = KiCadAdapter(path, {"component_layers": {"U1": "back"}}).load()
        assert st.components["U1"].layer == Layer.BACK
        assert st.components["U2"].layer == Layer.FRONT
