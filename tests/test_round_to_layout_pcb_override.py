"""Regression test for the per-round solved_layout / PCB consistency bug.

``SolvedLeafSubcircuit.round_to_layout`` is called twice in
``solve_subcircuits._persist_solution``:

  1. For the WINNING round, to materialise the canonical
     ``solved_layout.json``. At this moment ``leaf_routed.kicad_pcb``
     (the canonical path) is in the winner's state, so dereferencing
     ``round_result.routing["routed_board_path"]`` works.

  2. Once per accepted round, to write per-round
     ``round_NNNN_solved_layout.json`` snapshots that the GUI uses for
     "pin this round's geometry" workflows. By the time this loop runs,
     the canonical PCB has been overwritten many times by subsequent
     rounds + size_reduction; ``round_result.routing["routed_board_path"]``
     still names the canonical path, but its CONTENT is no longer the
     content from ``round_result``'s solve.

This test locks the contract: when called with
``routed_board_path_override=<round_NNNN_leaf_routed.kicad_pcb>``,
``round_to_layout`` must read components from the override path and
ignore ``round_result.routing["routed_board_path"]``. Without the
override, paired ``round_NNNN_solved_layout.json`` and
``round_NNNN_leaf_routed.kicad_pcb`` describe different positions, and
``pin_best_leaves`` promotes a layout/PCB pair that disagree -- which
is what produced the "J3 off by 1.9mm" failure in compose.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kicraft.autoplacer.brain.hierarchy_parser import HierarchyNode
from kicraft.autoplacer.brain.subcircuit_extractor import (
    ExtractedSubcircuitBoard,
    NetPartition,
)
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Pad,
    PlacementScore,
    Point,
    SolveRoundResult,
    SubCircuitDefinition,
    SubCircuitId,
)
from kicraft.cli import solve_subcircuits as solve_module
from kicraft.cli.solve_subcircuits import SolvedLeafSubcircuit


CANONICAL_PCB = "/tmp/test_round_layout/leaf_routed.kicad_pcb"
ROUND_PCB = "/tmp/test_round_layout/round_0001_leaf_routed.kicad_pcb"


def _board_state(j3_pos_x: float, j3_pad1_x: float) -> BoardState:
    j3 = Component(
        ref="J3",
        value="HEADER",
        pos=Point(j3_pos_x, 4.32),
        rotation=90.0,
        layer=Layer.FRONT,
        width_mm=8.71,
        height_mm=3.63,
        body_center=Point(j3_pos_x + 2.54, 4.32),
        kind="connector",
        is_through_hole=True,
    )
    j3.pads = [
        Pad(
            ref="J3",
            pad_id=str(i + 1),
            pos=Point(j3_pad1_x + i * 2.54, 4.32),
            net=f"NET{i + 1}",
            layer=Layer.FRONT,
            size_mm=Point(1.7, 1.7),
        )
        for i in range(3)
    ]
    return BoardState(
        components={"J3": j3},
        nets={},
        traces=[],
        vias=[],
        board_outline=(Point(0.0, 0.0), Point(60.0, 30.0)),
    )


def _make_solved_leaf() -> SolvedLeafSubcircuit:
    definition = SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name="LDO_3.3V",
            instance_path="/LDO_3.3V",
            sheet_file="ldo.kicad_sch",
        ),
        schematic_path="/tmp/test_round_layout/ldo.kicad_sch",
        component_refs=["J3"],
        ports=[],
        is_leaf=True,
    )
    extraction = ExtractedSubcircuitBoard(
        subcircuit=definition,
        full_state=BoardState(),
        local_state=BoardState(
            board_outline=(Point(0.0, 0.0), Point(60.0, 30.0))
        ),
        component_refs=["J3"],
        interface_ports=[],
        net_partition=NetPartition(),
    )
    routing = {
        "router": "freerouting",
        "routed_board_path": CANONICAL_PCB,
        "validation": {"accepted": True},
        "_trace_segments": [],
        "_via_objects": [],
        "traces": 0,
        "vias": 0,
    }
    round_result = SolveRoundResult(
        round_index=1,
        seed=0,
        score=80.0,
        placement=PlacementScore(),
        components={},
        routing=routing,
        routed=True,
    )
    return SolvedLeafSubcircuit(
        node=HierarchyNode(definition=definition),
        extraction=extraction,
        best_round=round_result,
        all_rounds=[round_result],
    )


@pytest.fixture
def patched_load_board_state(monkeypatch: pytest.MonkeyPatch):
    """Substitute _load_board_state with a path-keyed lookup.

    Mimics the real-world scenario: canonical PCB ends up in round 2's
    state by the time per-round snapshots are written; round_0001's
    snapshot file holds round 1's state.
    """
    canonical_state = _board_state(j3_pos_x=35.12, j3_pad1_x=35.12)
    round_state = _board_state(j3_pos_x=33.22, j3_pad1_x=33.22)

    def fake_load(path: Path | str, _cfg) -> BoardState:
        path_str = str(path)
        if path_str == CANONICAL_PCB:
            return canonical_state
        if path_str == ROUND_PCB:
            return round_state
        raise AssertionError(f"unexpected path requested: {path_str}")

    monkeypatch.setattr(solve_module, "_load_board_state", fake_load)
    return canonical_state, round_state


def test_round_to_layout_default_reads_canonical(patched_load_board_state):
    """Without override, the routed_board_path on the round_result is used.

    This is the correct behaviour for the WINNING round at solve time --
    canonical is in the winner's state.
    """
    leaf = _make_solved_leaf()
    layout = leaf.round_to_layout(leaf.best_round, cfg=None)

    j3 = layout.components["J3"]
    assert j3.pos.x == pytest.approx(35.12), (
        "default path must read the canonical PCB"
    )
    assert j3.pads[0].pos.x == pytest.approx(35.12)


def test_round_to_layout_override_reads_round_snapshot(patched_load_board_state):
    """With override, the per-round PCB snapshot is used.

    This is the correct behaviour for per-round snapshot writing.
    Without the override, all per-round layouts would describe the
    canonical's final state (round 2 here), so a pin operation that
    promotes round 1 would land mismatched JSON + PCB.
    """
    leaf = _make_solved_leaf()
    layout = leaf.round_to_layout(
        leaf.best_round,
        cfg=None,
        routed_board_path_override=ROUND_PCB,
    )

    j3 = layout.components["J3"]
    assert j3.pos.x == pytest.approx(33.22), (
        "override must redirect the read to the per-round snapshot"
    )
    assert j3.pads[0].pos.x == pytest.approx(33.22)


def test_override_takes_precedence_over_routing_path(patched_load_board_state):
    """The override wins even when routing.routed_board_path also points
    somewhere -- the canonical is irrelevant once the caller asks for a
    specific snapshot.
    """
    leaf = _make_solved_leaf()
    # routing.routed_board_path still names the canonical; the override
    # must redirect the read regardless.
    layout = leaf.round_to_layout(
        leaf.best_round,
        cfg=None,
        routed_board_path_override=ROUND_PCB,
    )
    assert layout.components["J3"].pos.x == pytest.approx(33.22)
