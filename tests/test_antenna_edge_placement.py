from __future__ import annotations

import copy
from dataclasses import replace
import os

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain import geometry
from kicraft.autoplacer.brain.leaf_routing import _outline_around_geometry
from kicraft.autoplacer.brain.antenna_edge import verify_antenna_edges
from kicraft.autoplacer.brain.placement_solver import (
    PlacementSolver,
    antenna_anchor_offset,
    antenna_faces_edge,
)
from kicraft.autoplacer.brain.subcircuit_artifacts import serialize_antenna_edge_intent
from kicraft.autoplacer.brain.subcircuit_instances import _parse_antenna_edge_intents
from kicraft.autoplacer.brain.types import Point
from kicraft.autoplacer.config import DEFAULT_CONFIG
from kicraft.autoplacer.hardware.adapter import KiCadAdapter
from kicraft.autoplacer.hardware.keepout_extract import extract_antenna_edge_intents

_LIB = os.path.join(os.path.dirname(__file__), "..", "kicraft", "parts_library")
_PRETTY = os.path.join(_LIB, "esp32-s3-wroom-1", "esp32-s3-wroom-1.pretty")
_NAME = "WIRELM-SMD_ESP32-S3-WROOM-1"


def _board_path(tmp_path, *, rotation=0.0):
    path = str(tmp_path / f"antenna-{rotation}.kicad_pcb")
    board = pcbnew.NewBoard(path)
    fp = pcbnew.FootprintLoad(_PRETTY, _NAME)
    if fp is None:
        pytest.skip("ESP32-S3-WROOM footprint unavailable")
    fp.SetReference("U1")
    fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(30), pcbnew.FromMM(25)))
    fp.SetOrientationDegrees(rotation)
    board.Add(fp)
    board.Save(path)
    return path


@pytest.mark.parametrize("rotation", [0.0, 90.0, 180.0, 270.0])
def test_named_rule_area_recovers_stable_local_antenna_semantics(tmp_path, rotation):
    board = pcbnew.LoadBoard(_board_path(tmp_path, rotation=rotation))
    result = extract_antenna_edge_intents(board, DEFAULT_CONFIG)

    assert result.diagnostics == ()
    assert len(result.intents) == 1
    intent = result.intents[0]
    assert intent.source == "footprint_rule_area"
    assert intent.source_id == "antenna_keepout"
    assert intent.local_direction == "top"
    assert intent.local_anchor_mm == pytest.approx(-16.4, abs=0.2)

def _add_rect_outline(board, left=0.0, top=0.0, right=60.0, bottom=50.0):
    points = [(left, top), (right, top), (right, bottom), (left, bottom)]
    for start, end in zip(points, points[1:] + points[:1]):
        shape = pcbnew.PCB_SHAPE(board)
        shape.SetShape(pcbnew.SHAPE_T_SEGMENT)
        shape.SetLayer(pcbnew.Edge_Cuts)
        shape.SetStart(pcbnew.VECTOR2I(pcbnew.FromMM(start[0]), pcbnew.FromMM(start[1])))
        shape.SetEnd(pcbnew.VECTOR2I(pcbnew.FromMM(end[0]), pcbnew.FromMM(end[1])))
        board.Add(shape)


def test_nonsemantic_rule_area_name_does_not_infer_antenna(tmp_path):
    board = pcbnew.LoadBoard(_board_path(tmp_path))
    zone = list(board.Footprints()[0].Zones())[0]
    zone.SetZoneName("mounting_keepout")
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["antenna_keepouts"] = {}

    result = extract_antenna_edge_intents(board, cfg)

    assert result.intents == ()
    assert result.diagnostics == ()


def test_kill_switch_preserves_only_explicit_edge_contract(tmp_path):
    board = pcbnew.LoadBoard(_board_path(tmp_path))
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["antenna_edge_pin_enabled"] = False
    assert extract_antenna_edge_intents(board, cfg).intents == ()
    cfg["component_zones"] = {"U1": {"edge": "right"}}
    intents = extract_antenna_edge_intents(board, cfg).intents
    assert len(intents) == 1
    assert intents[0].target_edge == "right"
    assert intents[0].explicit_edge


def test_locked_manual_placement_is_not_reinterpreted_as_inferred(tmp_path):
    path = _board_path(tmp_path)
    board = pcbnew.LoadBoard(path)
    board.Footprints()[0].SetLocked(True)
    board.Save(path)
    state = KiCadAdapter(path, copy.deepcopy(DEFAULT_CONFIG)).load()
    assert state.antenna_edge_intents == []





@pytest.mark.parametrize("edge", ["left", "right", "top", "bottom"])
def test_solver_orients_and_flushes_antenna_support_line(tmp_path, edge):
    path = _board_path(tmp_path)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["antenna_default_edge"] = edge
    cfg["edge_jitter_mm"] = 0.0
    cfg["unlock_all_footprints"] = False
    state = KiCadAdapter(path, cfg).load()
    state.board_outline = (Point(0.0, 0.0), Point(60.0, 50.0))
    original_cfg = copy.deepcopy(cfg)
    solver = PlacementSolver(state, cfg, seed=0)
    components = copy.deepcopy(state.components)

    solver._pin_edge_components(components)

    assert cfg == original_cfg
    assert solver.antenna_edge_conflicts == []
    intent = state.antenna_edge_intents[0]
    component = components["U1"]
    assert antenna_faces_edge(intent, component)
    offset = antenna_anchor_offset(intent, component)
    anchor = Point(component.pos.x + offset.x, component.pos.y + offset.y)
    expected = {
        "left": 0.0,
        "right": 60.0,
        "top": 0.0,
        "bottom": 50.0,
    }[edge]
    actual = anchor.x if edge in ("left", "right") else anchor.y
    assert actual == pytest.approx(expected, abs=1e-6)
    assert solver._pinned_rotations["U1"] == component.rotation

def test_full_solver_preserves_antenna_rotation_and_anchor(tmp_path):
    path = _board_path(tmp_path)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["antenna_default_edge"] = "top"
    cfg["max_placement_iterations"] = 10
    state = KiCadAdapter(path, cfg).load()
    state.board_outline = (Point(0.0, 0.0), Point(60.0, 50.0))
    solver = PlacementSolver(state, cfg, seed=0)

    components = solver.solve(max_iterations=10)

    intent = state.antenna_edge_intents[0]
    component = components["U1"]
    assert antenna_faces_edge(intent, component)
    offset = antenna_anchor_offset(intent, component)
    assert component.pos.y + offset.y == pytest.approx(0.0, abs=1e-6)


def test_explicit_incompatible_rotation_is_a_visible_conflict(tmp_path):
    path = _board_path(tmp_path)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["component_zones"] = {"U1": {"edge": "top", "rotation": 180.0}}
    state = KiCadAdapter(path, cfg).load()
    state.board_outline = (Point(0.0, 0.0), Point(60.0, 50.0))
    solver = PlacementSolver(state, cfg, seed=0)

    solver._pin_edge_components(copy.deepcopy(state.components))

    assert solver.antenna_edge_conflicts == [
        "antenna_edge_orientation_conflict:U1"
    ]

def test_leaf_outline_uses_antenna_support_line_not_generic_margin(tmp_path):
    path = _board_path(tmp_path)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    state = KiCadAdapter(path, cfg).load()
    state.board_outline = (Point(0.0, 0.0), Point(60.0, 50.0))
    solver = PlacementSolver(state, cfg, seed=0)
    components = copy.deepcopy(state.components)
    solver._pin_edge_components(components)
    intent = state.antenna_edge_intents[0]

    outline = _outline_around_geometry(
        components, cfg, antenna_edge_intents=[intent]
    )

    assert outline is not None
    assert outline[0].y == pytest.approx(0.0, abs=1e-6)


def test_multiple_antennas_pack_keepout_polygons_without_overlap(tmp_path):
    path = _board_path(tmp_path)
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["antenna_default_edge"] = "top"
    cfg["edge_jitter_mm"] = 0.0
    state = KiCadAdapter(path, cfg).load()
    state.board_outline = (Point(0.0, 0.0), Point(80.0, 50.0))
    second = copy.deepcopy(state.components["U1"])
    second.ref = "U2"
    for pad in second.pads:
        pad.ref = "U2"
    state.components["U2"] = second
    state.antenna_edge_intents.append(
        replace(state.antenna_edge_intents[0], owner_ref="U2")
    )
    solver = PlacementSolver(state, cfg, seed=0)
    components = copy.deepcopy(state.components)

    solver._pin_edge_components(components)

    spans = []
    for intent in state.antenna_edge_intents:
        component = components[intent.owner_ref]
        xs = [
            component.pos.x
            + geometry.rotate_vector(point, component.rotation).x
            for point in intent.local_polygon
        ]
        spans.append((min(xs), max(xs)))
    spans.sort()
    assert spans[0][1] + cfg["connector_gap_mm"] <= spans[1][0] + 1e-6

def test_intent_round_trip_and_old_artifact_default(tmp_path):
    board = pcbnew.LoadBoard(_board_path(tmp_path))
    intent = extract_antenna_edge_intents(board, DEFAULT_CONFIG).intents[0]
    row = serialize_antenna_edge_intent(intent)

    loaded = _parse_antenna_edge_intents({"antenna_edge_intents": [row]})

    assert loaded == [intent]
    assert _parse_antenna_edge_intents({}) == []
    with pytest.raises(ValueError, match="invalid antenna edge intent"):
        _parse_antenna_edge_intents(
            {"antenna_edge_intents": [{**row, "target_edge": "diagonal"}]}
        )


def test_final_verifier_accepts_flush_outward_antenna_and_rejects_rotation(tmp_path):
    path = _board_path(tmp_path)
    board = pcbnew.LoadBoard(path)
    fp = board.Footprints()[0]
    fp.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(30.0), pcbnew.FromMM(16.4)))
    _add_rect_outline(board)
    board.Save(path)
    intent = extract_antenna_edge_intents(board, DEFAULT_CONFIG).intents[0]

    verdicts, violations = verify_antenna_edges(path, [intent])

    assert violations == []
    assert verdicts[0].gap_mm == pytest.approx(0.0, abs=0.05)
    board = pcbnew.LoadBoard(path)
    board.Footprints()[0].SetOrientationDegrees(180.0)
    board.Save(path)
    _, violations = verify_antenna_edges(path, [intent])
    assert any(item.startswith("antenna_misoriented:U1") for item in violations)
