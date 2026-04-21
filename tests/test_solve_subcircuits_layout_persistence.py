from __future__ import annotations

from dataclasses import dataclass

import pytest

from kicraft.autoplacer.brain.hierarchy_parser import HierarchyNode
from kicraft.autoplacer.brain.subcircuit_extractor import ExtractedSubcircuitBoard, NetPartition
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    PlacementScore,
    Point,
    SolveRoundResult,
    SubCircuitDefinition,
    SubCircuitId,
)
from kicraft.cli.solve_subcircuits import SolvedLeafSubcircuit


def _make_component(ref: str, *, layer: Layer, pos_x: float, body_x: float) -> Component:
    return Component(
        ref=ref,
        value=ref,
        pos=Point(pos_x, 5.0),
        rotation=90.0,
        layer=layer,
        width_mm=10.0,
        height_mm=4.0,
        kind="connector",
        is_through_hole=True,
        body_center=Point(body_x, 5.0),
    )


def _make_solved_leaf(round_result: SolveRoundResult) -> SolvedLeafSubcircuit:
    definition = SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name="USB_INPUT",
            instance_path="/USB_INPUT",
            sheet_file="usb_input.kicad_sch",
        ),
        schematic_path="/tmp/project/usb_input.kicad_sch",
        component_refs=["J1"],
        ports=[],
        is_leaf=True,
    )
    extraction = ExtractedSubcircuitBoard(
        subcircuit=definition,
        full_state=BoardState(),
        local_state=BoardState(board_outline=(Point(0.0, 0.0), Point(25.0, 20.0))),
        component_refs=["J1"],
        interface_ports=[],
        net_partition=NetPartition(),
    )
    return SolvedLeafSubcircuit(
        node=HierarchyNode(definition=definition),
        extraction=extraction,
        best_round=round_result,
    )


def test_best_round_to_layout_prefers_routed_board_geometry(monkeypatch: pytest.MonkeyPatch):
    stale_component = _make_component("J1", layer=Layer.FRONT, pos_x=6.0, body_x=10.0)
    actual_component = _make_component("J1", layer=Layer.BACK, pos_x=6.0, body_x=2.0)
    routed_state = BoardState(
        components={"J1": actual_component},
        board_outline=(Point(-0.025, -0.025), Point(28.275, 20.87)),
    )

    @dataclass
    class FakeAdapter:
        pcb_path: str
        config: dict[str, object]

        def load(self) -> BoardState:
            return routed_state

    monkeypatch.setattr("kicraft.cli.solve_subcircuits.KiCadAdapter", FakeAdapter)

    round_result = SolveRoundResult(
        round_index=0,
        seed=1,
        score=42.0,
        placement=PlacementScore(),
        components={"J1": stale_component},
        routing={"routed_board_path": "/tmp/fake_leaf_routed.kicad_pcb"},
    )

    solved = _make_solved_leaf(round_result)
    layout = solved.best_round_to_layout(cfg={})

    assert layout.components["J1"].layer == Layer.BACK
    assert layout.components["J1"].body_center is not None
    assert layout.components["J1"].body_center.x == 2.0
    assert layout.bounding_box[0] == pytest.approx(28.3)
    assert layout.bounding_box[1] == pytest.approx(20.895)


def test_best_round_to_layout_falls_back_without_routed_board():
    stale_component = _make_component("J1", layer=Layer.FRONT, pos_x=6.0, body_x=10.0)
    round_result = SolveRoundResult(
        round_index=0,
        seed=1,
        score=42.0,
        placement=PlacementScore(),
        components={"J1": stale_component},
        routing={},
    )

    solved = _make_solved_leaf(round_result)
    layout = solved.best_round_to_layout(cfg={})

    assert layout.components["J1"].layer == Layer.FRONT
    assert layout.components["J1"].body_center is not None
    assert layout.components["J1"].body_center.x == 10.0
    assert layout.bounding_box == (25.0, 20.0)
