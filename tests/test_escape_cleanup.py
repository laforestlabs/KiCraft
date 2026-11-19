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
@pytest.mark.parametrize("launch", [False, True])
def test_cleanup_removes_only_unused_owned_branch_and_preserves_ground(tmp_path, route_layer, launch):
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
    launch_track = trace((8, 5), (8, 8), pcbnew.B_Cu) if launch else None
    if launch:
        outline = pcbnew.PCB_SHAPE(board)
        outline.SetShape(pcbnew.SHAPE_T_RECT)
        outline.SetStart(point(0, 0))
        outline.SetEnd(point(25, 20))
        outline.SetLayer(pcbnew.Edge_Cuts)
        board.Add(outline)
    if route_layer is not None:
        layer = board.GetLayerID(route_layer)
        pad("U2", 10, layer)
        trace((8, 5), (10, 5), layer)
    path = tmp_path / "board.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    before_ids = {item.m_Uuid.AsString() for item in board.GetTracks()}
    owned_ids = {escape.m_Uuid.AsString(), escape_via.m_Uuid.AsString()}
    if launch:
        owned_ids.add(launch_track.m_Uuid.AsString())
    owner = {"owner": "compiler.signal_escape", "ref": "U1", "pad": "1",
             "branch_id": "U1.1", "branch_item_count": "3" if launch else "2"}
    ownership = tmp_path / "ownership.json"
    traces = [{"start_x": 5, "start_y": 5, "end_x": 8, "end_y": 5,
               "width": 0.2, "layer": "F.Cu", "net_name": "DATA",
               "escape_ownership": {**owner, "source_uuid": escape.m_Uuid.AsString(),
                                    "branch_item_index": "0"}}]
    if launch:
        traces.append({"start_x": 8, "start_y": 5, "end_x": 8, "end_y": 8,
                       "width": 0.2, "layer": "B.Cu", "net_name": "DATA",
                       "escape_ownership": {**owner, "source_uuid": launch_track.m_Uuid.AsString(),
                                            "branch_item_index": "2"}})
    ownership.write_text(json.dumps({
        "version": 1, "stamped_board_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "routed_board_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "prune_terminal_tracks": launch, "require_native_drc": launch,
        "traces": traces,
        "vias": [{"x": 8, "y": 5, "size": 0.6, "drill": 0.3, "net_name": "DATA",
                  "escape_ownership": {**owner, "source_uuid": escape_via.m_Uuid.AsString(),
                                       "branch_item_index": "1"}}],
    }))
    receipt = cleanup_signal_escapes(path, ownership, tmp_path / "cleanup.json")
    after = pcbnew.LoadBoard(str(path))
    after_ids = {item.m_Uuid.AsString() for item in after.GetTracks()}
    assert receipt["accepted"]
    assert ground.m_Uuid.AsString() in after_ids
    expected_removed = (owned_ids if route_layer is None
                        else {escape_via.m_Uuid.AsString()} if route_layer == "F.Cu"
                        else set())
    if launch:
        expected_removed.add(launch_track.m_Uuid.AsString())
    assert after_ids == before_ids - expected_removed
    assert {item["native_uuid"] for row in receipt["removed"]
            for item in row["items"]} == expected_removed
    for row in receipt["removed"]:
        proof = row["proof"]
        assert proof["partition_before"] == proof["partition_after"]


@pytest.mark.parametrize("kind", [
    "track", "overlap-track", "via", "protected-via", "cascade", "restamped", "restamped-via",
    "track-far-fill", "track-touching-fill", "via-touching-fill", "via-empty-fill",
    "after-owned-cleanup", "stale-owned-cleanup",
])
def test_router_delta_prunes_only_native_drc_named_new_unprotected_copper(tmp_path, kind):
    from kicraft.autoplacer.brain.escape_cleanup import _report_drc, prune_router_spurs

    pcbnew = pytest.importorskip("pcbnew")
    if not shutil.which("kicad-cli"):
        pytest.skip("native DRC requires kicad-cli")
    board = pcbnew.CreateEmptyBoard()
    point = lambda x, y: pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))
    net = pcbnew.NETINFO_ITEM(board, "DATA")
    board.Add(net)
    outline = pcbnew.PCB_SHAPE(board)
    outline.SetShape(pcbnew.SHAPE_T_RECT)
    outline.SetStart(point(0, 0))
    outline.SetEnd(point(30, 25))
    outline.SetLayer(pcbnew.Edge_Cuts)
    board.Add(outline)
    contacts = [("J1", 5, 10, net), ("J2", 20, 10, net)]
    if kind in {"after-owned-cleanup", "stale-owned-cleanup"}:
        contacts.append(("J5", 8, 10, net))
    if kind == "overlap-track":
        other_net = pcbnew.NETINFO_ITEM(board, "OTHER")
        board.Add(other_net)
        contacts.extend([("J3", 5, 17, other_net), ("J4", 20, 17, other_net)])
    for ref, x, y, circuit in contacts:
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetNumber("1")
        pad.SetShape(pcbnew.PAD_SHAPE_RECT)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetSize(point(1, 1))
        pad.SetPosition(point(x, y))
        pad.SetNet(circuit)
        fp.Add(pad)

    def trace(start, end, circuit=net, layer=pcbnew.F_Cu):
        item = pcbnew.PCB_TRACK(board)
        item.SetLayer(layer)
        item.SetStart(point(*start))
        item.SetEnd(point(*end))
        item.SetWidth(pcbnew.FromMM(0.2))
        item.SetNet(circuit)
        board.Add(item)
        return item

    def via(x, y):
        item = pcbnew.PCB_VIA(board)
        item.SetPosition(point(x, y))
        item.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        item.SetDrill(pcbnew.FromMM(0.3))
        try:
            item.SetWidth(pcbnew.FromMM(0.6))
        except TypeError:
            item.SetWidth(pcbnew.F_Cu, pcbnew.FromMM(0.6))
        item.SetNet(net)
        board.Add(item)
        return item

    trace((5, 10), (20, 10))
    inherited = trace((9, 10), (9, 12)).m_Uuid.AsString()
    if kind == "overlap-track":
        # Offset router junctions share copper with the inherited long trunk,
        # while KiCad still reports the jogs' unsplit trunk ends as dangling.
        for y, circuit in ((10, net), (17, other_net)):
            if y == 17:
                trace((5, y), (20, y), circuit)
            trace((5, y + 0.035), (12, y + 0.035), circuit)
            trace((12, y + 0.035), (20, y + 0.035), circuit)
    after_owned_cleanup = kind in {"after-owned-cleanup", "stale-owned-cleanup"}
    if after_owned_cleanup:
        owned_track = trace((5, 10), (8, 10))
        owned_via = via(8, 10)
    source = tmp_path / "posted.kicad_pcb"
    assert pcbnew.SaveBoard(str(source), board)
    if after_owned_cleanup:
        added = [trace((8, 10), (8, 12))]
    elif kind in {"track", "track-far-fill", "track-touching-fill", "restamped"}:
        added = [trace((12, 10), (12, 12))]
    elif kind == "overlap-track":
        from kicraft.autoplacer.hardware.local_contract import _copper_graph, _position

        added = [trace((12.023624, y), (12, y + 0.035), circuit)
                 for y, circuit in ((10, net), (17, other_net))]
        for item in added:
            graph, _, _ = _copper_graph(board, item.GetNetname())
            assert all(len(graph[(_position(end), pcbnew.F_Cu)]) > 1
                       for end in (item.GetStart(), item.GetEnd()))
    else:
        added = [via(12, 10)]
        if kind == "cascade":
            added.append(trace((12, 10), (12, 12), layer=pcbnew.B_Cu))
    ids = {item.m_Uuid.AsString() for item in added}
    fill = None
    if kind.endswith("-fill"):
        bounds = {
            "track-far-fill": (18, 8, 22, 13),
            "track-touching-fill": (11, 9, 13, 11),
            "via-touching-fill": (11, 9, 13, 11),
            "via-empty-fill": (22, 15, 25, 20),
        }
        x1, y1, x2, y2 = bounds[kind]
        zone = pcbnew.ZONE(board)
        zone.SetLayer(pcbnew.F_Cu)
        zone.SetNet(net)
        zone.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
        zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
        zone.Outline().NewOutline()
        for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
            q = point(x, y)
            zone.Outline().Append(q.x, q.y)
        board.Add(zone)
        if kind == "via-empty-fill":
            keepout = pcbnew.ZONE(board)
            keepout.SetLayer(pcbnew.F_Cu)
            keepout.SetIsRuleArea(True)
            keepout.SetDoNotAllowCopperPour(True)
            keepout.SetDoNotAllowTracks(False)
            keepout.SetDoNotAllowVias(False)
            keepout.SetDoNotAllowPads(False)
            keepout.Outline().NewOutline()
            for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                q = point(x, y)
                keepout.Outline().Append(q.x, q.y)
            board.Add(keepout)
        from kicraft.autoplacer.routing_board import run_pcbnew_script

        unfilled = tmp_path / "zone-input.kicad_pcb"
        assert pcbnew.SaveBoard(str(unfilled), board)
        run_pcbnew_script(
            "import pcbnew\n"
            f"board = pcbnew.LoadBoard({str(unfilled)!r})\n"
            "board.BuildConnectivity()\n"
            "filler = pcbnew.ZONE_FILLER(board)\nfiller.Fill(board.Zones())\n"
            f"board.Save({str(unfilled)!r})\n"
        )
        board = pcbnew.LoadBoard(str(unfilled))
        zone = next(zone for zone in board.Zones() if not zone.GetIsRuleArea())
        polygons = zone.GetFilledPolysList(pcbnew.F_Cu)
        assert bool(polygons.OutlineCount()) == (kind != "via-empty-fill")
        fill = polygons.Format()
    routed = tmp_path / "routed.kicad_pcb"
    assert pcbnew.SaveBoard(str(routed), board)
    if fill is not None:
        report = _report_drc(routed, tmp_path / "original_drc.json")
        assert any(
            row["type"] in {"track_dangling", "via_dangling"}
            and any(item["uuid"] in ids for item in row["items"])
            for row in report["violations"]
        )
    input_cleanup = None
    if after_owned_cleanup:
        owner = {"owner": "compiler.signal_escape", "ref": "J1", "pad": "1",
                 "branch_id": "J1.1", "branch_item_count": "2"}
        ownership = tmp_path / "ownership.json"
        ownership.write_text(json.dumps({
            "version": 1,
            "prune_terminal_tracks": True,
            "stamped_board_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "routed_board_sha256": hashlib.sha256(routed.read_bytes()).hexdigest(),
            "traces": [{"start_x": 5, "start_y": 10, "end_x": 8, "end_y": 10,
                        "width": 0.2, "layer": "F.Cu", "net_name": "DATA",
                        "escape_ownership": {**owner, "source_uuid": owned_track.m_Uuid.AsString(),
                                             "branch_item_index": "0"}}],
            "vias": [{"x": 8, "y": 10, "size": 0.6, "drill": 0.3, "net_name": "DATA",
                      "escape_ownership": {**owner, "source_uuid": owned_via.m_Uuid.AsString(),
                                           "branch_item_index": "1"}}],
        }))
        input_cleanup = cleanup_signal_escapes(routed, ownership, tmp_path / "cleanup.json")
        assert input_cleanup["accepted"] and input_cleanup["native_readback"]
        assert {row["native_uuid"] for branch in input_cleanup["removed"] for row in branch["items"]} == {
            owned_via.m_Uuid.AsString(),
        }
        if kind == "stale-owned-cleanup":
            input_cleanup["stamped_board_sha256"] = "0" * 64
    if kind in {"restamped", "restamped-via"}:
        from kicraft.autoplacer.brain.escape_cleanup import (
            _bind_router_origin_frame, _capture_router_origin, _finish_router_origin_copper,
            _prepare_router_origin_frame,
        )

        origin = json.loads(json.dumps(_capture_router_origin(source, routed)))
        delta = Point(123.35, 82.461)
        prepared = _prepare_router_origin_frame(origin, routed, delta)
        board = pcbnew.LoadBoard(str(routed))
        shift = point(delta.x, delta.y)
        for item in [*board.GetFootprints(), *board.GetDrawings()]:
            item.Move(shift)
        old = list(board.GetTracks())
        replacements = []
        identities = {}
        for item in old:
            if isinstance(item, pcbnew.PCB_VIA):
                copy = pcbnew.PCB_VIA(board)
                copy.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
                copy.SetPosition(point(
                    pcbnew.ToMM(item.GetPosition().x) + delta.x,
                    pcbnew.ToMM(item.GetPosition().y) + delta.y,
                ))
                copy.SetWidth(pcbnew.FromMM(pcbnew.ToMM(item.GetWidth(pcbnew.F_Cu))))
                copy.SetDrill(pcbnew.FromMM(pcbnew.ToMM(item.GetDrillValue())))
            else:
                copy = pcbnew.PCB_TRACK(board)
                copy.SetLayer(item.GetLayer())
                for setter, position in ((copy.SetStart, item.GetStart()), (copy.SetEnd, item.GetEnd())):
                    setter(point(pcbnew.ToMM(position.x) + delta.x, pcbnew.ToMM(position.y) + delta.y))
                copy.SetWidth(pcbnew.FromMM(pcbnew.ToMM(item.GetWidth())))
            copy.SetNetCode(item.GetNetCode())
            copy.SetLocked(item.IsLocked())
            identities[item.m_Uuid.AsString()] = copy.m_Uuid.AsString()
            replacements.append(copy)
        for item in old:
            board.Remove(item)
        for copy in replacements:
            board.Add(copy)
        assert pcbnew.SaveBoard(str(routed), board)
        bound = _bind_router_origin_frame(prepared, routed)
        ids = {identities[uid] for uid in ids}
        inherited = identities[inherited]
        assert set(bound["items"]) == ids
        # Finisher copper must never inherit the raw router's deletion scope.
        board = pcbnew.LoadBoard(str(routed))
        finisher = trace(
            (15 + delta.x, 10 + delta.y), (15 + delta.x, 12 + delta.y),
            circuit=board.FindNet("DATA"),
        ).m_Uuid.AsString()
        assert pcbnew.SaveBoard(str(routed), board)
        proof = _finish_router_origin_copper(bound, routed, {})
        assert proof["accepted"] and proof["native_readback"], proof
        final = pcbnew.LoadBoard(str(routed))
        all_ids = {item.m_Uuid.AsString() for item in final.GetTracks()}
        assert inherited in all_ids and finisher in all_ids
        assert not ids & all_ids
        assert {row["uuid"] for row in proof["removed"]} == ids
        return
    original = routed.read_bytes()
    proof = prune_router_spurs(
        source, routed, {}, protected_nets={"DATA"} if kind == "protected-via" else None,
        input_cleanup=input_cleanup,
    )
    if kind == "stale-owned-cleanup":
        assert not proof["accepted"] and routed.read_bytes() == original
        return
    assert proof["accepted"], proof
    final = pcbnew.LoadBoard(str(routed))
    all_ids = {t.m_Uuid.AsString() for t in final.GetTracks()}
    assert inherited in all_ids
    if fill is not None:
        [saved_zone] = [zone for zone in final.Zones() if not zone.GetIsRuleArea()]
        assert saved_zone.GetFilledPolysList(pcbnew.F_Cu).Format() == fill
    if kind in {"protected-via", "track-touching-fill", "via-touching-fill"}:
        assert not proof["removed"] and routed.read_bytes() == original
    else:
        assert proof["native_readback"]
        assert {row["uuid"] for row in proof["removed"]} == ids
        assert not ids & all_ids


def test_cleanup_retains_later_same_net_zone_after_other_branch_removal(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec, add_breakout_stubs
    from kicraft.autoplacer.brain.copper_accounting import signal_escape_geometry_receipt
    from kicraft.autoplacer.routing_board import import_routed_copper

    path = tmp_path / "board.kicad_pcb"
    board = pcbnew.CreateEmptyBoard()
    point = lambda x, y: pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))
    outline = pcbnew.PCB_SHAPE(board)
    outline.SetShape(pcbnew.SHAPE_T_RECT)
    outline.SetStart(point(0, 0))
    outline.SetEnd(point(30, 25))
    outline.SetLayer(pcbnew.Edge_Cuts)
    board.Add(outline)

    specs = []
    for index, netname in enumerate(("FIRST", "ZONED", "LAST"), 1):
        net = pcbnew.NETINFO_ITEM(board, netname)
        board.Add(net)
        fp = pcbnew.FOOTPRINT(board)
        ref = f"U{index}"
        fp.SetReference(ref)
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetNumber("1")
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetSize(point(1, 1))
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNet(net)
        fp.Add(pad)
        pad.SetPosition(point(5, 5 * index))
        specs.append(BreakoutSpec(ref=ref, pad="1", waypoints=[(8, 5 * index)], via_at_end=True))
        if netname == "ZONED":
            zone = pcbnew.ZONE(board)
            zone.SetLayer(pcbnew.F_Cu)
            zone.SetNet(net)
            polygon = zone.Outline()
            polygon.NewOutline()
            for x, y in ((19, 7), (23, 7), (23, 12), (19, 12)):
                pos = point(x, y)
                polygon.Append(pos.x, pos.y)
            board.Add(zone)
    assert pcbnew.SaveBoard(str(path), board)

    summary = add_breakout_stubs(str(path), specs, signal_escape_owner=True)
    assert summary["stubs"] == summary["vias"] == 3
    copper = import_routed_copper(str(path))
    assert attach_signal_escape_ownership(copper, summary)["bound"] == 6
    records = signal_escape_geometry_receipt(copper)
    for name in ("traces", "vias"):
        records[name].sort(key=lambda row: row["escape_ownership"]["ref"])
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    ownership = tmp_path / "ownership.json"
    ownership.write_text(json.dumps({
        "version": 1, "stamped_board_sha256": before, "routed_board_sha256": before, **records,
    }))
    proof = cleanup_signal_escapes(path, ownership, tmp_path / "cleanup.json")
    assert proof["accepted"]
    assert proof["native_readback"]
    assert [row["net"] for row in proof["removed"]] == ["FIRST", "LAST"]
    assert [row["reason"] for row in proof["retained"]] == ["same_net_zone_not_in_graph"]
    final = pcbnew.LoadBoard(str(path))
    assert {item.GetNetname() for item in final.GetTracks()} == {"ZONED"}


def test_private_cleanup_preserves_native_pad_landing_for_corner_contact(tmp_path):
    pcbnew = pytest.importorskip("pcbnew")
    if not shutil.which("kicad-cli"):
        pytest.skip("native DRC requires kicad-cli")
    from kicraft.autoplacer.brain.escape_cleanup import _report_drc

    board = pcbnew.CreateEmptyBoard()
    point = lambda x, y: pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))
    net = pcbnew.NETINFO_ITEM(board, "BOOTSTRAP")
    board.Add(net)
    outline = pcbnew.PCB_SHAPE(board)
    outline.SetShape(pcbnew.SHAPE_T_RECT)
    outline.SetStart(point(0, 0))
    outline.SetEnd(point(15, 15))
    outline.SetLayer(pcbnew.Edge_Cuts)
    board.Add(outline)
    for ref, xy, size in (("U1", (5, 5), (2.2, 0.5)), ("C1", (10, 10), (0.9, 0.95))):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetNumber("1")
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetShape(pcbnew.PAD_SHAPE_RECT)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetSize(point(*size))
        pad.SetPosition(point(*xy))
        pad.SetNet(net)
        fp.Add(pad)

    def trace(start, end):
        item = pcbnew.PCB_TRACK(board)
        item.SetStart(point(*start))
        item.SetEnd(point(*end))
        item.SetLayer(pcbnew.F_Cu)
        item.SetWidth(pcbnew.FromMM(0.127))
        item.SetNet(net)
        board.Add(item)
        return item

    ends = [(5, 5), (6.3265, 5), (6.3365, 5), (6.743, 5)]
    owned = [trace(a, b) for a, b in zip(ends, ends[1:])]
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(point(*ends[-1]))
    via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    via.SetDrill(pcbnew.FromMM(0.3))
    try:
        via.SetWidth(pcbnew.FromMM(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, pcbnew.FromMM(0.6))
    via.SetNet(net)
    board.Add(via)
    # The routed diagonal grazes the pad corner, but its free endpoint lands
    # on the compiler escape. Removing that entire escape disconnects the
    # endpoint even though finite copper still connects the two pads.
    routed = trace((6.325, 4.995001), (5.825, 5.495001))
    trace((5.825, 5.495001), (10, 10))
    path = tmp_path / "corner.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    owner = {"owner": "compiler.signal_escape", "ref": "U1", "pad": "1",
             "branch_id": "U1.1", "branch_item_count": "4"}
    rows = [
        {"start_x": a[0], "start_y": a[1], "end_x": b[0], "end_y": b[1],
         "width": 0.127, "layer": "F.Cu", "net_name": "BOOTSTRAP",
         "escape_ownership": {**owner, "source_uuid": item.m_Uuid.AsString(),
                              "branch_item_index": str(i)}}
        for i, (a, b, item) in enumerate(zip(ends, ends[1:], owned))
    ]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    ownership = tmp_path / "ownership.json"
    ownership.write_text(json.dumps({
        "version": 1, "prune_terminal_tracks": True, "require_native_drc": True,
        "stamped_board_sha256": sha, "routed_board_sha256": sha, "traces": rows,
        "vias": [{"x": ends[-1][0], "y": ends[-1][1], "size": 0.6, "drill": 0.3,
                  "net_name": "BOOTSTRAP",
                  "escape_ownership": {**owner, "source_uuid": via.m_Uuid.AsString(),
                                       "branch_item_index": "3"}}],
    }))
    receipt = cleanup_signal_escapes(path, ownership, tmp_path / "cleanup.json")
    assert receipt["accepted"] and receipt["native_readback"], receipt
    final = pcbnew.LoadBoard(str(path))
    remaining = {item.m_Uuid.AsString() for item in final.GetTracks()}
    assert {owned[0].m_Uuid.AsString(), routed.m_Uuid.AsString()} <= remaining
    assert not {item.m_Uuid.AsString() for item in [*owned[1:], via]} & remaining
    drc = _report_drc(path, tmp_path / "drc.json")
    assert not drc["unconnected_items"]
    assert not [row for row in drc["violations"] if row["type"] in {"track_dangling", "via_dangling"}]
