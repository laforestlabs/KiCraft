"""USB contract checks use native board copper and native polygon fills."""
from __future__ import annotations

import hashlib
import json

import pytest

pcbnew = pytest.importorskip("pcbnew")
from kicraft.autoplacer.hardware.usb_contract import audit_usb_routing, _paired_geometry, _return_vias


def pt(x, y):
    return pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))


def fixture_board(tmp_path, *, negative_end=10):
    path = tmp_path / "usb.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    for net in ("USB_DP", "USB_DM", "GND"):
        board.Add(pcbnew.NETINFO_ITEM(board, net))
    for ref, x in (("J1", 0), ("U1", 10)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        board.Add(fp)
        for number, y, net in (("1", 0, "USB_DP"), ("2", 0.4, "USB_DM")):
            pad = pcbnew.PAD(fp)
            pad.SetNumber(number)
            pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
            pad.SetSize(pt(0.2, 0.2))
            pad.SetPosition(pt(x, y))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNet(board.GetNetInfo().GetNetItem(net))
            fp.Add(pad)
    for y, net, end in ((0, "USB_DP", 10), (0.4, "USB_DM", negative_end)):
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(pt(0, y))
        track.SetEnd(pt(end, y))
        track.SetWidth(pcbnew.FromMM(0.2))
        track.SetLayer(pcbnew.F_Cu)
        track.SetNet(board.GetNetInfo().GetNetItem(net))
        board.Add(track)
    board.Save(str(path))
    return board, path


def contract():
    endpoint = lambda ref, pin: {"ref": ref, "pin": pin}
    return {"id": "usb", "reference_net": "GND", "impedance_ohm": 90,
            "impedance_tolerance": 0.1, "max_skew_mm": 0.5,
            "max_layer_transitions": 2, "segments": [{"id": "data",
            "positive_net": "USB_DP", "negative_net": "USB_DM",
            "positive_from": [endpoint("J1", "1")], "positive_to": [endpoint("U1", "1")],
            "negative_from": [endpoint("J1", "2")], "negative_to": [endpoint("U1", "2")]}]}


def rectangle(x1, y1, x2, y2):
    shape = pcbnew.SHAPE_POLY_SET()
    shape.NewOutline()
    for x, y in ((x1,y1), (x2,y1), (x2,y2), (x1,y2)):
        shape.Append(pcbnew.FromMM(x), pcbnew.FromMM(y))
    return shape


def test_missing_approval_and_swapped_terminal_are_failures(tmp_path):
    board, _ = fixture_board(tmp_path)
    spec = contract()
    spec["segments"][0]["positive_to"] = [{"ref": "U1", "pin": "2"}]
    receipt = audit_usb_routing(board, [spec])
    assert not receipt["ok"]
    assert any("fabricator-approved" in error for error in receipt["errors"])
    assert any("wrong polarity" in error for error in receipt["errors"])


def test_open_route_is_not_inferred_from_declared_nets(tmp_path):
    board, _ = fixture_board(tmp_path, negative_end=8)
    receipt = audit_usb_routing(board, [contract()])
    assert any("not copper-connected" in error for error in receipt["errors"])
    assert receipt["contracts"][0]["length_mm"]["negative"] is None
    assert receipt["contracts"][0]["skew_mm"] is None


def test_unproved_contact_does_not_discard_measured_path_length(tmp_path, monkeypatch):
    from kicraft.autoplacer.hardware import usb_contract
    board, _ = fixture_board(tmp_path)
    original = usb_contract._path_geometry
    calls = 0
    def geometry(path):
        nonlocal calls
        calls += 1
        tracks, transitions, errors = original(path)
        if calls == 2:
            errors.append("route includes an unproved finite copper contact")
        return tracks, transitions, errors
    monkeypatch.setattr(usb_contract, "_path_geometry", geometry)
    receipt = audit_usb_routing(board, [contract()])
    row = receipt["contracts"][0]
    assert any("unproved finite copper contact" in error for error in receipt["errors"])
    assert row["length_mm"]["positive"] == pytest.approx(10)
    assert row["length_mm"]["negative"] == pytest.approx(10)
    assert row["skew_mm"] == pytest.approx(0)


def test_chain_boundary_must_cross_expected_component(tmp_path):
    board, _ = fixture_board(tmp_path)
    spec = contract()
    segment = spec["segments"][0]
    following = dict(segment)
    following["id"] = "next"
    # An unrelated connector pad cannot represent a reviewed ESD/series path.
    following["positive_from"] = [{"ref": "J1", "pin": "1"}]
    spec["segments"].append(following)
    receipt = audit_usb_routing(board, [spec])
    assert any("component chain boundary" in error for error in receipt["errors"])


def test_native_corridor_tracks_splits_and_filled_holes():
    iu = pcbnew.FromMM
    width = iu(0.2)
    p = [((iu(0),iu(0)), (iu(5),iu(0)), pcbnew.F_Cu, width),
         ((iu(5),iu(0)), (iu(10),iu(0)), pcbnew.F_Cu, width)]
    n = [((iu(0),iu(0.4)), (iu(10),iu(0.4)), pcbnew.F_Cu, width)]
    approval = {"geometry_by_layer": {"F.Cu": {"width_mm": 0.2, "gap_mm": 0.2}}}
    plane = rectangle(-2,-2,12,2)
    spans, _ = _paired_geometry(p, n, [(pcbnew.B_Cu, plane)], approval)
    assert len(spans) == 2
    void = rectangle(4,-0.1,5,0.1)
    plane.BooleanSubtract(void)
    with pytest.raises(ValueError, match="reference plane"):
        _paired_geometry(p, n, [(pcbnew.B_Cu, plane)], approval)
    n[0] = ((iu(0),iu(0.7)), (iu(10),iu(0.7)), pcbnew.F_Cu, width)
    with pytest.raises(ValueError, match="parallel pair corridor"):
        _paired_geometry(p, n, [(pcbnew.B_Cu, rectangle(-2,-2,12,2))], approval)


def test_native_skew_rejects_long_separate_corridor(tmp_path, review_geometry):
    board, _ = fixture_board(tmp_path)
    for track in list(board.GetTracks()):
        if track.GetNetname() == "USB_DM":
            board.Remove(track)
    for start, end in (((0, 0.4), (0, 2.4)), ((0, 2.4), (10, 2.4)),
                       ((10, 2.4), (10, 0.4))):
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(pt(*start))
        track.SetEnd(pt(*end))
        track.SetWidth(pcbnew.FromMM(0.2))
        track.SetLayer(pcbnew.F_Cu)
        track.SetNet(board.GetNetInfo().GetNetItem("USB_DM"))
        board.Add(track)
    receipt = audit_usb_routing(board, [contract()])
    assert any("skew" in error for error in receipt["errors"])
    spec = contract()
    spec["max_skew_mm"] = 10
    review = audit_usb_routing(board, [spec], {"usb_geometry_mode": "engineering_review"})
    assert not review["geometry_ok"]
    assert any("skew" in error for error in review["geometry_errors"])


def test_final_gate_binds_failure_to_promoted_board(tmp_path, monkeypatch):
    from kicraft.design.cli_app import _verify_routed_board
    from kicraft.autoplacer import routing_board
    _, path = fixture_board(tmp_path)
    monkeypatch.setattr(routing_board, "validate_routed_board",
                        lambda *args, **kwargs: {"accepted": True, "drc": {}})
    result = _verify_routed_board(path, physical_config={"usb_routing_contracts": [contract()]})
    assert not result["fab_acceptable"]
    receipt = json.loads(path.with_suffix(".usb-routing.receipt.json").read_text())
    assert not receipt["ok"]
    assert receipt["board_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_native_pair_with_45_degree_bend_and_miter():
    iu = pcbnew.FromMM
    width = iu(0.2)
    p = [((iu(0), iu(0)), (iu(5), iu(0)), pcbnew.F_Cu, width),
         ((iu(5), iu(0)), (iu(10), iu(5)), pcbnew.F_Cu, width)]
    n = [((iu(0), iu(0.4)), (iu(4.834315), iu(0.4)), pcbnew.F_Cu, width),
         ((iu(4.834315), iu(0.4)), (iu(9.834315), iu(5.4)), pcbnew.F_Cu, width)]
    approval = {"geometry_by_layer": {"F.Cu": {"width_mm": 0.2, "gap_mm": 0.2}}}
    spans, _ = _paired_geometry(p, n, [(pcbnew.B_Cu, rectangle(-2,-2,12,8))], approval)
    assert len(spans) == 2


def test_return_transition_needs_grounded_native_via(tmp_path):
    board, _ = fixture_board(tmp_path)
    signal = (pcbnew.FromMM(3), pcbnew.FromMM(0))
    transition = [(signal, signal, pcbnew.F_Cu, pcbnew.B_Cu)]
    regions = [(pcbnew.F_Cu, rectangle(-1,-1,12,2)),
               (pcbnew.B_Cu, rectangle(-1,-1,12,2))]
    approval = {"max_return_via_distance_mm": 1.0}
    with pytest.raises(ValueError, match="no nearby grounded via"):
        _return_vias(board, transition, regions, "GND", approval,
                     {pcbnew.F_Cu: {0}, pcbnew.B_Cu: {1}})
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pt(3, 0.5))
    via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    via.SetDrill(pcbnew.FromMM(0.2))
    try:
        via.SetWidth(pcbnew.FromMM(0.4))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, pcbnew.FromMM(0.4))
    via.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Add(via)
    assert len(_return_vias(board, transition, regions, "GND", approval,
                            {pcbnew.F_Cu: {0}, pcbnew.B_Cu: {1}})) == 1


@pytest.fixture
def review_geometry(monkeypatch):
    from kicraft.autoplacer.hardware import usb_contract

    calculated = {
        "provenance": "calculated_unconfirmed",
        "fabrication_approved": False,
        "source": "test calculation",
        "stackup_sha256": "test-stackup",
        "stackup": {},
        "geometry_by_layer": {
            layer: {"width_mm": 0.2, "gap_mm": 0.2, "estimated_ohm": 90}
            for layer in ("F.Cu", "B.Cu")
        },
        "max_return_via_distance_mm": 1.0,
        "assumptions": [],
    }
    monkeypatch.setattr(usb_contract, "calculate_usb_review_geometry",
                        lambda board, config, contract, nets: calculated)
    plane = rectangle(-2, -2, 12, 2)
    monkeypatch.setattr(usb_contract, "_ground_regions",
                        lambda board, net: [(pcbnew.B_Cu, plane)])
    return plane


def test_review_geometry_pass_is_not_fabrication_approval(tmp_path, review_geometry):
    board, _ = fixture_board(tmp_path)
    receipt = audit_usb_routing(board, [contract()], {"usb_geometry_mode": "engineering_review"})
    row = receipt["contracts"][0]
    assert receipt["geometry_ok"] and not receipt["geometry_errors"]
    assert row["geometry_ok"] and row["segments"][0]["paired_spans"]
    assert row["review_geometry"]["provenance"] == "calculated_unconfirmed"
    assert row["review_geometry"]["fabrication_approved"] is False
    assert not receipt["fabrication_approved"] and not row["fabrication_approved"]
    assert "impedance_approval" not in row
    assert not receipt["ok"] and not row["ok"]


def test_review_never_applies_validated_approval_to_calculated_geometry(
        tmp_path, review_geometry, monkeypatch):
    from kicraft.autoplacer.hardware import usb_contract

    board, _ = fixture_board(tmp_path)
    layers = [{"name": "test"}]
    stackup_hash = hashlib.sha256(
        json.dumps(layers, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    approved_geometry = {
        layer: {"width_mm": 0.2, "gap_mm": 0.2}
        for layer in ("F.Cu", "B.Cu")
    }
    approval = {"source": "fabricator", "stackup_sha256": stackup_hash,
                "geometry_by_layer": approved_geometry}
    monkeypatch.setattr(usb_contract, "preflight_kicad_routing_tools", lambda config: {})
    monkeypatch.setattr(usb_contract, "_usb_stackup", lambda *args: {
        "layers": layers, "planned_geometry": approved_geometry})
    receipt = audit_usb_routing(board, [contract()], {
        "usb_geometry_mode": "engineering_review",
        "usb_impedance_fabrication_approval": approval,
    })
    row = receipt["contracts"][0]
    assert receipt["geometry_ok"] and row["geometry_ok"]
    assert not receipt["fabrication_approved"] and not row["fabrication_approved"]
    assert not receipt["ok"] and not row["ok"]
    assert "impedance_approval" not in row
    assert any("fabrication release blocked" in error for error in receipt["errors"])
    assert not receipt["geometry_errors"]


def test_review_rejects_actual_track_width(tmp_path, review_geometry):
    board, _ = fixture_board(tmp_path)
    next(iter(board.GetTracks())).SetWidth(pcbnew.FromMM(0.3))
    receipt = audit_usb_routing(board, [contract()], {"usb_geometry_mode": "engineering_review"})
    assert not receipt["geometry_ok"] and not receipt["ok"]
    assert any("track width" in error for error in receipt["geometry_errors"])
    assert not receipt["fabrication_approved"]


def test_review_rejects_unpaired_native_corridor(tmp_path, review_geometry):
    board, _ = fixture_board(tmp_path)
    for footprint in board.GetFootprints():
        for pad in footprint.Pads():
            if pad.GetNetname() == "USB_DM":
                position = pad.GetPosition()
                pad.SetPosition(pcbnew.VECTOR2I(position.x, pcbnew.FromMM(0.7)))
    for track in board.GetTracks():
        if track.GetNetname() == "USB_DM":
            track.SetStart(pt(0, 0.7))
            track.SetEnd(pt(10, 0.7))
    receipt = audit_usb_routing(board, [contract()], {"usb_geometry_mode": "engineering_review"})
    assert not receipt["geometry_ok"] and not receipt["ok"]
    assert any("pair corridor" in error for error in receipt["geometry_errors"])


def test_review_rejects_reference_void(tmp_path, review_geometry):
    board, _ = fixture_board(tmp_path)
    review_geometry.BooleanSubtract(rectangle(4, -0.1, 5, 0.1))
    receipt = audit_usb_routing(board, [contract()], {"usb_geometry_mode": "engineering_review"})
    assert not receipt["geometry_ok"] and not receipt["ok"]
    assert any("reference plane" in error for error in receipt["geometry_errors"])


def test_review_rejects_open_and_broken_chain(tmp_path, review_geometry):
    board, _ = fixture_board(tmp_path, negative_end=8)
    spec = contract()
    following = dict(spec["segments"][0])
    following["id"] = "next"
    following["positive_from"] = [{"ref": "J1", "pin": "1"}]
    spec["segments"].append(following)
    receipt = audit_usb_routing(board, [spec], {"usb_geometry_mode": "engineering_review"})
    assert not receipt["geometry_ok"] and not receipt["ok"]
    assert any("not copper-connected" in error for error in receipt["geometry_errors"])
    assert any("component boundary" in error for error in receipt["geometry_errors"])


@pytest.fixture
def bounded_geometry(review_geometry, monkeypatch):
    from kicraft.autoplacer.hardware import usb_contract

    geometry = {
        "provenance": "calculated_unconfirmed", "fabrication_approved": False,
        "geometry_by_layer": {
            layer: {"width_mm": 0.2, "gap_mm": 0.2}
            for layer in ("F.Cu", "B.Cu")
        },
        "max_return_via_distance_mm": 1.0,
        "launch_policy": {"mode": "bounded_uncontrolled", "width_mm": 0.1,
                          "gap_mm": 0.2, "max_length_mm": 0.8,
                          "provenance": "uncontrolled_launch"},
    }
    monkeypatch.setattr(usb_contract, "calculate_usb_review_geometry",
                        lambda *args, **kwargs: geometry)
    return geometry


def _split_usb_tracks(board, pieces):
    for track in list(board.GetTracks()):
        board.Remove(track)
    for y, net in ((0, "USB_DP"), (0.4, "USB_DM")):
        for start, end, width in pieces:
            track = pcbnew.PCB_TRACK(board)
            track.SetStart(pt(start, y))
            track.SetEnd(pt(end, y))
            track.SetWidth(pcbnew.FromMM(width))
            track.SetLayer(pcbnew.F_Cu)
            track.SetNet(board.GetNetInfo().GetNetItem(net))
            board.Add(track)


def _bounded_config():
    return {"usb_geometry_mode": "engineering_review",
            "usb_review_allow_bounded_launches": True}


def test_bounded_launches_record_actual_copper_without_fabrication_approval(
        tmp_path, bounded_geometry):
    board, _ = fixture_board(tmp_path)
    _split_usb_tracks(board, [(0, 0.5, 0.1), (0.5, 9.5, 0.2), (9.5, 10, 0.1)])
    receipt = audit_usb_routing(board, [contract()], _bounded_config())
    row = receipt["contracts"][0]
    assert row["geometry_ok"], row["geometry_errors"]
    assert len(row["segments"][0]["uncontrolled_spans"]) == 4
    assert all(not span["impedance_qualified"]
               for span in row["segments"][0]["uncontrolled_spans"])
    assert row["segments"][0]["paired_spans"]
    assert not receipt["fabrication_approved"] and not receipt["ok"]


@pytest.mark.parametrize("pieces", [
    [(0, 10, 0.1)],
    [(0, 4, 0.2), (4, 4.5, 0.1), (4.5, 10, 0.2)],
])
def test_bounded_launches_reject_long_or_middle_narrow_copper(
        tmp_path, bounded_geometry, pieces):
    board, _ = fixture_board(tmp_path)
    _split_usb_tracks(board, pieces)
    receipt = audit_usb_routing(board, [contract()], _bounded_config())
    assert not receipt["geometry_ok"]
    assert any("bounded contiguous" in error for error in receipt["geometry_errors"])


@pytest.mark.parametrize("middle", [False, True])
def test_alias_branch_is_reported_not_hidden_by_shortest_path(tmp_path, bounded_geometry, middle):
    board, _ = fixture_board(tmp_path)
    pieces = ([(0, 0.5, 0.1), (0.5, 4, 0.2), (4, 4.5, 0.1), (4.5, 10, 0.2)]
              if middle else [(0, 0.5, 0.1), (0.5, 10, 0.2)])
    _split_usb_tracks(board, pieces)
    fp = board.FindFootprintByReference("J1")
    pad = pcbnew.PAD(fp)
    pad.SetNumber("3")
    pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
    pad.SetSize(pt(0.2, 0.2))
    pad.SetPosition(pt(-0.3, 0))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
    pad.SetLayerSet(pcbnew.PAD.SMDMask())
    pad.SetNet(board.GetNetInfo().GetNetItem("USB_DP"))
    fp.Add(pad)
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(pt(-0.3, 0))
    track.SetEnd(pt(0, 0))
    track.SetWidth(pcbnew.FromMM(0.1))
    track.SetLayer(pcbnew.F_Cu)
    track.SetNet(board.GetNetInfo().GetNetItem("USB_DP"))
    board.Add(track)
    spec = contract()
    spec["segments"][0]["positive_from"].append({"ref": "J1", "pin": "3"})
    row = audit_usb_routing(board, [spec], _bounded_config())["contracts"][0]
    if middle:
        assert not row["geometry_ok"]
        assert any("bounded contiguous" in error for error in row["geometry_errors"])
        return
    assert row["geometry_ok"], row["geometry_errors"]
    spans = row["segments"][0]["uncontrolled_spans"]
    assert len(spans) == 3
    assert any(span["kind"] == "contact_join" and span["length_mm"] == pytest.approx(0.3)
               for span in spans)

@pytest.mark.parametrize("excursion,profile_delta,accepted", [
    (0.85, 0, True), (1.4, 0, False), (0.85, 0.01, False),
])
def test_alias_join_is_bounded_by_native_contact_copper(
        tmp_path, bounded_geometry, monkeypatch, excursion, profile_delta, accepted):
    from kicraft.autoplacer.hardware import usb_contract
    board, _ = fixture_board(tmp_path)
    for dimensions in bounded_geometry["geometry_by_layer"].values():
        dimensions["width_mm"] = 0.23
        dimensions["gap_mm"] = 0.17
    policy = bounded_geometry["launch_policy"]
    policy.update(width_mm=0.17, gap_mm=0.17, max_length_mm=0.74)
    _split_usb_tracks(board, [(0, 0.5, 0.17), (0.5, 10, 0.23)])
    # Run away from a real fine-pitch contact row; long contact pads must not
    # overlap the opposite net merely to isolate the alias-bound assertion.
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            position = pad.GetPosition()
            pad.SetPosition(pt(pcbnew.ToMM(position.y), pcbnew.ToMM(position.x)))
    for track in board.GetTracks():
        start, end = track.GetStart(), track.GetEnd()
        track.SetStart(pt(pcbnew.ToMM(start.y), pcbnew.ToMM(start.x)))
        track.SetEnd(pt(pcbnew.ToMM(end.y), pcbnew.ToMM(end.x)))
    monkeypatch.setattr(usb_contract, "_ground_regions",
                        lambda *a: [(pcbnew.B_Cu, rectangle(-5, -5, 15, 15))])
    fp = board.FindFootprintByReference("J1")
    for primary_pin, alias_pin, x, primary_x, net in (
            ("1", "3", -1, 0, "USB_DP"), ("2", "4", 1.4, 0.4, "USB_DM")):
        original = next(p for p in fp.Pads() if p.GetNumber() == primary_pin)
        original.SetShape(pcbnew.PAD_SHAPE_RECT)
        original.SetSize(pt(0.2, 1.55))
        alias = pcbnew.PAD(fp)
        alias.SetNumber(alias_pin)
        alias.SetShape(pcbnew.PAD_SHAPE_RECT)
        alias.SetSize(pt(0.2, 1.55))
        alias.SetPosition(pt(x, 0))
        alias.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        alias.SetLayerSet(pcbnew.PAD.SMDMask())
        alias.SetNet(board.GetNetInfo().GetNetItem(net))
        fp.Add(alias)
        for start, end in (((x, 0), (x, -excursion)),
                           ((x, -excursion), (primary_x, -excursion)),
                           ((primary_x, -excursion), (primary_x, 0))):
            track = pcbnew.PCB_TRACK(board)
            track.SetStart(pt(*start))
            track.SetEnd(pt(*end))
            track.SetWidth(pcbnew.FromMM(0.17))
            track.SetLayer(pcbnew.F_Cu)
            track.SetNet(board.GetNetInfo().GetNetItem(net))
            board.Add(track)
    spec = contract()
    spec["segments"][0]["positive_from"].append({"ref": "J1", "pin": "3"})
    spec["segments"][0]["negative_from"].append({"ref": "J1", "pin": "4"})
    policy["contact_join_bounds"] = [{
        "segment_id": "data", "polarity": "positive", "side": "from",
        "ref": "J1", "pins": ["1", "3"], "max_length_mm": 3.23 + profile_delta,
        "envelope_mm": [-1.44, -1.115, 0.44, 1.115],
    }, {
        "segment_id": "data", "polarity": "negative", "side": "from",
        "ref": "J1", "pins": ["2", "4"], "max_length_mm": 3.23,
        "envelope_mm": [-0.04, -1.115, 1.84, 1.115],
    }]
    row = audit_usb_routing(board, [spec], _bounded_config())["contracts"][0]
    if not accepted:
        assert not row["geometry_ok"]
        assert any("pad-envelope" in error or "profile disagrees" in error
                   for error in row["geometry_errors"])
        return
    assert row["geometry_ok"], row["geometry_errors"]
    segment = row["segments"][0]
    contact = segment["contact_join_bounds"][0]
    assert contact["max_length_mm"] == pytest.approx(3.23, abs=1e-6)
    assert contact["envelope_mm"] == pytest.approx([-1.44, -1.115, 0.44, 1.115], abs=1e-6)
    narrow_length = sum(pcbnew.ToMM(track.GetLength()) for track in board.GetTracks()
                        if track.GetWidth() == pcbnew.FromMM(0.17))
    assert sum(span["length_mm"] for span in segment["uncontrolled_spans"]) == pytest.approx(
        narrow_length, abs=1e-6,
    )
    assert not row["fabrication_approved"]


@pytest.mark.parametrize("config", [
    {"usb_review_allow_bounded_launches": True},
    {"usb_geometry_mode": "engineering_review", "usb_review_allow_bounded_launches": "true"},
])
def test_bounded_launch_option_cannot_leak_into_fabrication(tmp_path, bounded_geometry, config):
    board, _ = fixture_board(tmp_path)
    receipt = audit_usb_routing(board, [contract()], config)
    assert not receipt["geometry_ok"]
    assert any("strict boolean" in error for error in receipt["geometry_errors"])


def test_orientation_skew_uses_corresponding_aliases(monkeypatch):
    from kicraft.autoplacer.hardware import usb_contract

    segment = contract()["segments"][0]
    segment["positive_from"].append({"ref": "J1", "pin": "3"})
    segment["negative_from"].append({"ref": "J1", "pin": "4"})
    monkeypatch.setattr(usb_contract, "_endpoint_nodes",
                        lambda board, pins, endpoint, net: [endpoint["pin"]])
    distances = {"1": 5.0, "2": 5.4, "3": 5.4, "4": 5.0}
    monkeypatch.setattr(usb_contract, "_shortest",
                        lambda graph, sources, targets: (distances[sources[0]], []))
    cache = {"USB_DP": ({}, {}, []), "USB_DM": ({}, {}, [])}
    orientations = usb_contract._orientation_lengths(None, [segment], cache, [])
    assert orientations == [{"positive": 5.0, "negative": 5.4},
                            {"positive": 5.4, "negative": 5.0}]
    assert max(abs(row["positive"] - row["negative"]) for row in orientations) == pytest.approx(0.4)
