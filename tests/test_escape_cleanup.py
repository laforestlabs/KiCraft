import hashlib
import json
from pathlib import Path
import shutil

import pytest

from kicraft.autoplacer.brain.copper_accounting import (
    attach_signal_escape_ownership,
    attach_stamped_escape_ownership,
)
from kicraft.autoplacer.brain.escape_cleanup import audit_dangling_copper, cleanup_signal_escapes
from kicraft.autoplacer.brain.subcircuit_artifacts import serialize_trace
from kicraft.autoplacer.brain.subcircuit_instances import _parse_traces, _transform_trace
from kicraft.autoplacer.brain.types import Layer, Point, TraceSegment


def test_escape_ownership_survives_geometry_and_reimport():
    trace = TraceSegment(Point(1, 2), Point(2, 2), Layer.FRONT, "DATA", 0.2)
    copper = {"traces": [trace], "vias": []}
    summary = {"owned_signal_escapes": [{"owner": "compiler.signal_escape", "ref": "U1",
        "pad": "1", "items": [{"uuid": "original", "kind": "track", "net": "DATA",
        "layer": "F.Cu", "start_mm": [1, 2], "end_mm": [2, 2], "width_mm": 0.2}]}]}
    assert attach_signal_escape_ownership(copper, summary)["bound"] == 1
    restored = _parse_traces({"traces": [serialize_trace(trace)]})[0]
    moved = _transform_trace(restored, Point(10, 10), 90)
    receipt = {"traces": [{"start_x": moved.start.x, "start_y": moved.start.y,
                           "end_x": moved.end.x, "end_y": moved.end.y,
                           "net_name": moved.net, "layer": "F.Cu", "width": moved.width_mm,
                           "escape_ownership": moved.escape_ownership}], "vias": []}
    imported = TraceSegment(moved.start, moved.end, moved.layer, moved.net, moved.width_mm)
    assert attach_stamped_escape_ownership({"traces": [imported], "vias": []}, receipt)["bound"] == 1
    assert imported.escape_ownership["source_uuid"] == "original"


def test_ambiguous_geometry_never_claims_existing_copper():
    traces = [TraceSegment(Point(1, 2), Point(2, 2), Layer.FRONT, "DATA", 0.2)
              for _ in range(2)]
    summary = {"owned_signal_escapes": [{"owner": "compiler.signal_escape", "ref": "U1",
        "pad": "1", "items": [{"uuid": "generated", "kind": "track", "net": "DATA",
        "layer": "F.Cu", "start_mm": [1, 2], "end_mm": [2, 2], "width_mm": 0.2}]}]}
    result = attach_signal_escape_ownership({"traces": traces}, summary)
    assert result["unresolved_source_uuids"] == ["generated"]
    assert all(trace.escape_ownership is None for trace in traces)


def test_dangling_copper_requires_board_bound_specific_evidence(tmp_path):
    board = tmp_path / "board.kicad_pcb"
    board.write_bytes(b"board")
    violation = {"type": "via_dangling", "items": [{"uuid": "via-1"}]}
    drc = {"violations": [violation]}
    assert not audit_dangling_copper(board, {}, [])['accepted']
    assert not audit_dangling_copper(board, drc, [])['accepted']
    proof = {"uuid": "via-1", "board_sha256": hashlib.sha256(b"board").hexdigest(),
             "justification": "verified grounded stitching via"}
    assert audit_dangling_copper(board, drc, [proof])["accepted"]
    board.write_bytes(b"changed board")
    assert not audit_dangling_copper(board, drc, [proof])["accepted"]


def test_stale_ownership_receipt_never_changes_board(tmp_path):
    source = (Path(__file__).parent / "fixtures" / "replay_workspace"
              / "PARENT_LOCAL_CONN" / "PARENT_LOCAL_CONN.kicad_pcb")
    board = tmp_path / "board.kicad_pcb"
    shutil.copyfile(source, board)
    before = board.read_bytes()
    ownership = tmp_path / "ownership.json"
    ownership.write_text(json.dumps({"version": 1, "stamped_board_sha256": "original",
                                     "routed_board_sha256": "stale", "traces": [], "vias": []}))
    with pytest.raises(RuntimeError):
        cleanup_signal_escapes(board, ownership, tmp_path / "cleanup.json")
    assert board.read_bytes() == before


@pytest.mark.parametrize("route_layer", [None, "F.Cu", "B.Cu"])
def test_cleanup_removes_only_unused_owned_branch_and_preserves_ground(tmp_path, route_layer):
    pcbnew = pytest.importorskip("pcbnew")
    board = pcbnew.BOARD()
    nets = {}
    for name in ("DATA", "GND"):
        nets[name] = pcbnew.NETINFO_ITEM(board, name)
        board.Add(nets[name])

    def point(x, y):
        return pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))

    def pad(ref, x, layer):
        footprint = pcbnew.FOOTPRINT(board)
        footprint.SetReference(ref)
        board.Add(footprint)
        item = pcbnew.PAD(footprint)
        item.SetNumber("1")
        item.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        item.SetShape(pcbnew.PAD_SHAPE_RECT)
        item.SetSize(point(1, 1))
        layers = pcbnew.LSET()
        layers.AddLayer(layer)
        item.SetLayerSet(layers)
        item.SetNet(nets["DATA"])
        footprint.Add(item)
        item.SetPosition(point(x, 5))

    def trace(start, end, layer):
        item = pcbnew.PCB_TRACK(board)
        item.SetStart(point(*start))
        item.SetEnd(point(*end))
        item.SetLayer(layer)
        item.SetWidth(pcbnew.FromMM(0.2))
        item.SetNet(nets["DATA"])
        board.Add(item)
        return item

    def via(x, y, net):
        item = pcbnew.PCB_VIA(board)
        item.SetPosition(point(x, y))
        item.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        item.SetDrill(pcbnew.FromMM(0.3))
        try:
            item.SetWidth(pcbnew.FromMM(0.6))
        except TypeError:
            item.SetWidth(pcbnew.F_Cu, pcbnew.FromMM(0.6))
        item.SetNet(nets[net])
        board.Add(item)
        return item

    pad("U1", 5, pcbnew.F_Cu)
    escape = trace((5, 5), (8, 5), pcbnew.F_Cu)
    escape_via = via(8, 5, "DATA")
    ground = via(15, 15, "GND")
    if route_layer is not None:
        layer = board.GetLayerID(route_layer)
        pad("U2", 10, layer)
        trace((8, 5), (10, 5), layer)
    path = tmp_path / "board.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    before_ids = {item.m_Uuid.AsString() for item in board.GetTracks()}
    owned_ids = {escape.m_Uuid.AsString(), escape_via.m_Uuid.AsString()}
    owner = {"owner": "compiler.signal_escape", "ref": "U1", "pad": "1",
             "branch_id": "U1.1", "branch_item_count": "2"}
    ownership = tmp_path / "ownership.json"
    ownership.write_text(json.dumps({
        "version": 1, "stamped_board_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "routed_board_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "traces": [{"start_x": 5, "start_y": 5, "end_x": 8, "end_y": 5,
                    "width": 0.2, "layer": "F.Cu", "net_name": "DATA",
                    "escape_ownership": {**owner, "source_uuid": escape.m_Uuid.AsString()}}],
        "vias": [{"x": 8, "y": 5, "size": 0.6, "drill": 0.3, "net_name": "DATA",
                  "escape_ownership": {**owner, "source_uuid": escape_via.m_Uuid.AsString()}}],
    }))
    receipt = cleanup_signal_escapes(path, ownership, tmp_path / "cleanup.json")
    after = pcbnew.LoadBoard(str(path))
    after_ids = {item.m_Uuid.AsString() for item in after.GetTracks()}
    assert receipt["accepted"]
    assert ground.m_Uuid.AsString() in after_ids
    expected_removed = (owned_ids if route_layer is None
                        else {escape_via.m_Uuid.AsString()} if route_layer == "F.Cu"
                        else set())
    assert after_ids == before_ids - expected_removed
    assert {item["native_uuid"] for row in receipt["removed"]
            for item in row["items"]} == expected_removed
    for row in receipt["removed"]:
        proof = row["proof"]
        assert proof["partition_before"] == proof["partition_after"]
