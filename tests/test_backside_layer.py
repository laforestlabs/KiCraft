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


def test_assign_layers_keeps_edge_connectors_on_front():
    """A large THT edge-mating connector must NOT be auto-flipped to B.Cu.

    ``_assign_layers`` sends large THT parts to the back so they don't block
    front-side routing. But an edge-zoned connector defines a board edge from
    the front: flipping it mirrors its pad X and swaps left<->right in
    ``edge_outward_angle``, inverting the opening the compose rotation filter
    solves for, which strands the connector inboard (the run_07 USB-C
    signature). Such parts stay on F.Cu; a plain large THT part still flips.
    """
    from kicraft.autoplacer.brain.placement_solver import PlacementSolver
    from kicraft.autoplacer.brain.types import BoardState, Component, Layer, Point

    def _big_tht(ref, kind="misc", opening=None):
        # 100 mm² courtyard > the 50 mm² tht_backside threshold, PTH pads.
        return Component(
            ref=ref, value="x", pos=Point(20.0, 20.0), rotation=0.0,
            layer=Layer.FRONT, width_mm=10.0, height_mm=10.0,
            kind=kind, is_through_hole=True, opening_direction=opening,
        )

    comps = {
        "J1": _big_tht("J1", kind="misc"),            # edge-zoned via cfg
        "SW1": _big_tht("SW1", kind="connector"),     # kind == connector
        "J2": _big_tht("J2", opening=0.0),            # carries opening_direction
        "BT1": _big_tht("BT1", kind="misc"),          # plain large THT -> back
    }
    cfg = {"component_zones": {"J1": {"edge": "right"}}}
    solver = PlacementSolver(BoardState(), cfg, seed=0)
    solver._assign_layers(comps)

    assert comps["J1"].layer == Layer.FRONT   # edge zone -> stays front
    assert comps["SW1"].layer == Layer.FRONT  # connector kind -> stays front
    assert comps["J2"].layer == Layer.FRONT   # opening_direction -> stays front
    assert comps["BT1"].layer == Layer.BACK   # plain large THT still flips
