"""Native power-path checks on real KiCad tracks, vias and pad shapes."""
from __future__ import annotations

import pytest
import copy
import shutil

pcbnew = pytest.importorskip("pcbnew")
from kicraft.autoplacer.hardware.local_contract import (
    _contact_width, _copper_graph, _shortest_qualified,
    _track_via_annular_contact, audit_local_copper,
)


def _point(x, y):
    return pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))


def _board(tmp_path):
    board = pcbnew.NewBoard(str(tmp_path / "local.kicad_pcb"))
    board.Add(pcbnew.NETINFO_ITEM(board, "PWR"))
    return board


def _pad(board, ref, xy, *, pth=False, diameter=0.4, number="1", net="PWR"):
    fp = board.FindFootprintByReference(ref)
    if fp is None:
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        board.Add(fp)
    pad = pcbnew.PAD(fp)
    pad.SetNumber(number)
    pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
    pad.SetSize(_point(diameter, diameter))
    pad.SetPosition(_point(*xy))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH if pth else pcbnew.PAD_ATTRIB_SMD)
    pad.SetLayerSet(pcbnew.PAD.PTHMask() if pth else pcbnew.PAD.SMDMask())
    if pth:
        pad.SetDrillSize(_point(0.2, 0.2))
    pad.SetNet(board.GetNetInfo().GetNetItem(net))
    fp.Add(pad)
    return pad


def _track(board, start, end, *, width=0.5, layer=None, net="PWR"):
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(_point(*start))
    track.SetEnd(_point(*end))
    track.SetWidth(pcbnew.FromMM(width))
    track.SetLayer(pcbnew.F_Cu if layer is None else layer)
    track.SetNet(board.GetNetInfo().GetNetItem(net))
    board.Add(track)


def _via(board, xy, *, net="PWR"):
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(_point(*xy))
    via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    via.SetDrill(pcbnew.FromMM(0.3))
    via.SetNet(board.GetNetInfo().GetNetItem(net))
    try:
        via.SetWidth(pcbnew.FromMM(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, pcbnew.FromMM(0.6))
    board.Add(via)


def _spec(*, route_anchor="L1", limit=25):
    return {"owner_ref": "U1", "bypass_ref": "C1", "power_pins": ["1", "1"],
            "route_anchor_ref": route_anchor, "route_anchor_pin": "1",
            "max_route_length_mm": limit, "min_route_width_mm": 0.5}


def test_current_path_uses_inductor_not_nearby_feedback(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0))
    _pad(board, "U1", (8, 0))
    _pad(board, "C1", (9, 0))
    for a, b in [((0, 0), (0, 10)), ((0, 10), (8, 10)),
                 ((8, 10), (8, 0)), ((8, 0), (9, 0))]:
        _track(board, a, b)
    assert any("exceeds" in e for e in audit_local_copper(board, [_spec(limit=10)]))
    assert not audit_local_copper(board, [_spec(route_anchor="U1", limit=10)])


def test_t_junction_and_via_on_track_interior_connect(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0))
    _pad(board, "C1", (8, 3), pth=True)
    _track(board, (0, 0), (8, 0))
    _track(board, (4, 0), (4, 1))  # T junction on an unsplit bus
    _via(board, (4, 0))             # also lies on the bus interior
    _track(board, (4, 0), (4, 3), layer=pcbnew.B_Cu)
    _track(board, (4, 3), (8, 3), layer=pcbnew.B_Cu)
    assert not audit_local_copper(board, [_spec()])


def test_plated_pad_connects_front_and_back_copper(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0))
    _pad(board, "J1", (4, 0), pth=True)
    _pad(board, "C1", (8, 0), pth=True)
    _track(board, (0, 0), (4, 0))
    _track(board, (4, 0), (8, 0), layer=pcbnew.B_Cu)
    assert not audit_local_copper(board, [_spec()])


def test_rounded_pad_bbox_corner_does_not_bridge_copper(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0))
    _pad(board, "J1", (3, 0), diameter=2.0)
    _pad(board, "C1", (6, 0))
    _track(board, (0, 0), (3.9, 0.9))
    _track(board, (3.9, -0.9), (6, 0))
    assert any("no qualified" in e for e in audit_local_copper(board, [_spec()]))


def test_split_narrow_escape_is_limited_as_a_whole(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0), diameter=0.1)
    _pad(board, "C1", (4, 1), diameter=0.1)
    _track(board, (0, 0), (1, 0), width=0.2)
    _track(board, (1, 0), (1, 1), width=0.2)
    _track(board, (1, 1), (4, 1), width=0.5)
    errors = audit_local_copper(board, [_spec(), {"current_carrying_net": "PWR"}])
    assert any("no qualified" in e for e in errors)
    assert any("narrow copper" in e for e in errors)


def test_unanchored_narrow_midroute_is_rejected(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0))
    _pad(board, "C1", (8, 0))
    _track(board, (0, 0), (3, 0))
    _track(board, (3, 0), (3.5, 0), width=0.2)
    _track(board, (3.5, 0), (8, 0))
    errors = audit_local_copper(board, [_spec(), {"current_carrying_net": "PWR"}])
    assert any("no qualified" in e for e in errors)
    assert any("narrow copper" in e for e in errors)


@pytest.mark.parametrize("separation,passes", [(2.3, True), (4.0, False)])
def test_native_pad_boundaries_measure_only_exposed_narrow_neck(tmp_path, separation, passes):
    board = _board(tmp_path)
    sot = _pad(board, "U1", (0, 0))
    sot.SetShape(pcbnew.PAD_SHAPE_RECT)
    sot.SetSize(_point(0.55, 1.2))
    coil = _pad(board, "C1", (0, -separation))
    coil.SetShape(pcbnew.PAD_SHAPE_RECT)
    coil.SetSize(_point(2.424, 3.62))
    coil.SetOrientationDegrees(-90)
    _track(board, (0, 0), (0, -separation), width=0.2)
    spec = {"owner_ref": "U1", "bypass_ref": "C1",
            "power_pins": ["1", "1"], "max_route_length_mm": 10,
            "min_route_width_mm": 0.5}
    errors = audit_local_copper(board, [spec, {"current_carrying_net": "PWR"}])
    if passes:
        # 2.3 - 0.6 - 1.212 = 0.488 mm of exposed centreline.
        assert not errors
    else:
        assert any("no qualified" in e for e in errors)
        assert any("narrow copper" in e for e in errors)


def test_pad_covered_edges_keep_physical_length_without_extending_neck(tmp_path):
    board = _board(tmp_path)
    _pad(board, "U1", (0, 0), diameter=2)
    _pad(board, "C1", (2, 0), diameter=0.2)
    _track(board, (0, 0), (2, 0), width=0.2)
    graph, _, _ = _copper_graph(board, "PWR")
    covered = [(length, width) for edges in graph.values()
               for _, length, width in edges if width == -2]
    assert any(length > 0.9 for length, _ in covered)
    # Direct graph traversal still counts the covered distance in route length.
    a, b, c = "a", "b", "c"
    path = {a: [(b, 1.0, -2.0)], b: [(a, 1.0, -2.0), (c, 0.5, 0.2)],
            c: [(b, 0.5, 0.2)]}
    assert _shortest_qualified(path, [a], [c], 0.5, [a]) == 1.5
    spec = {"owner_ref": "U1", "bypass_ref": "C1",
            "power_pins": ["1", "1"], "max_route_length_mm": 1,
            "min_route_width_mm": 0.5}
    assert any("exceeds" in e for e in audit_local_copper(board, [spec]))
    spec["max_route_length_mm"] = 2.1
    assert not audit_local_copper(board, [spec, {"current_carrying_net": "PWR"}])


def test_side_contact_only_counts_exposed_part_as_narrow(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0), diameter=4)
    _pad(board, "C1", (5, 0))
    _track(board, (0, 0), (2.1, 0))
    _track(board, (2.1, 0), (5, 0))
    assert not audit_local_copper(board, [_spec(), {"current_carrying_net": "PWR"}])


@pytest.mark.parametrize("separation", [0.49, 0.5])
def test_shallow_and_tangent_endcaps_do_not_bypass_narrow_track(tmp_path, separation):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0))
    _pad(board, "C1", (8, 0))
    _track(board, (0, 0), (3, 0))
    _track(board, (3, 0), (3 + separation, 0), width=0.2)
    _track(board, (3 + separation, 0), (8, 0))
    errors = audit_local_copper(board, [_spec(), {"current_carrying_net": "PWR"}])
    assert any("no qualified" in e for e in errors)
    assert any("narrow copper" in e for e in errors)
    width = _contact_width(pcbnew.FromMM(0.5), pcbnew.FromMM(0.5),
                           pcbnew.FromMM(separation))
    if separation == 0.49:
        assert 0.099 < pcbnew.ToMM(width) < 0.1
    else:
        assert width == 0


def test_narrow_escape_does_not_reset_at_via(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0), diameter=0.1)
    _pad(board, "C1", (3, 0), pth=True, diameter=0.4)
    _track(board, (0, 0), (1, 0), width=0.2)
    _via(board, (1, 0))
    _track(board, (1, 0), (2, 0), width=0.2, layer=pcbnew.B_Cu)
    _track(board, (2, 0), (3, 0), layer=pcbnew.B_Cu)
    errors = audit_local_copper(board, [_spec(), {"current_carrying_net": "PWR"}])
    assert any("no qualified" in e for e in errors)
    assert any("narrow copper" in e for e in errors)


def test_drilled_pad_hole_does_not_create_trace_join(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (-0.5, 0), diameter=0.1)
    _pad(board, "C1", (0.5, 0), diameter=0.1)
    pad = _pad(board, "J1", (0, 0), pth=True, diameter=2)
    pad.SetDrillSize(_point(1.8, 1.8))
    # Copper purportedly entirely in a drill cannot join the pads.
    _track(board, (-0.5, 0), (0.5, 0), width=0.2)
    assert any("no qualified" in e for e in audit_local_copper(board, [_spec()]))


def test_custom_pad_geometry_fails_closed(tmp_path):
    board = _board(tmp_path)
    pad = _pad(board, "L1", (0, 0))
    pad.SetShape(pcbnew.PAD_SHAPE_CUSTOM)
    _pad(board, "C1", (2, 0))
    _track(board, (0, 0), (2, 0))
    with pytest.raises(ValueError, match="unsupported custom pad"):
        audit_local_copper(board, [_spec()])

def test_offset_via_contact_does_not_claim_unproved_current_width(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0))
    _pad(board, "C1", (8, 3), pth=True)
    _track(board, (0, 0), (8, 0))
    _via(board, (4, 0.1))
    _track(board, (4, 0.1), (4, 3), layer=pcbnew.B_Cu)
    _track(board, (4, 3), (8, 3), layer=pcbnew.B_Cu)
    assert any("no qualified" in e for e in audit_local_copper(board, [_spec()]))


@pytest.mark.parametrize("blocked", [False, True])
def test_leaf_closes_shallow_via_landings_only_through_guarded_copper(tmp_path, blocked):
    from kicraft.autoplacer.brain.leaf_routing import _finish_leaf_current_landings

    board = _board(tmp_path)
    _pad(board, "L1", (5, 5))
    _pad(board, "C1", (13, 8), pth=True)
    _track(board, (5, 5), (9, 5))
    _via(board, (9.1, 5.4))
    _track(board, (9.1, 5.4), (9.1, 8), layer=pcbnew.B_Cu)
    _track(board, (9.1, 8), (13, 8), layer=pcbnew.B_Cu)
    if blocked:
        board.Add(pcbnew.NETINFO_ITEM(board, "OTHER"))
        _pad(board, "X1", (9.35, 5.2), diameter=0.1, net="OTHER")
    spec = {**_spec(route_anchor="L1"), "owner_ref": "L1"}
    assert any("no qualified" in error for error in audit_local_copper(board, [spec]))
    path = tmp_path / "landing.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    cfg = {"local_bypasses": [spec]}

    _finish_leaf_current_landings(path, cfg)
    finished = pcbnew.LoadBoard(str(path))
    errors = audit_local_copper(finished, [spec])
    if blocked:
        assert any("no qualified" in error for error in errors)
    else:
        assert not errors
        qualified = path.read_bytes()
        _finish_leaf_current_landings(path, cfg)
        assert path.read_bytes() == qualified


@pytest.mark.parametrize("angle", [0, 45, 90, 180])
@pytest.mark.parametrize("reverse", [False, True])
def test_offset_via_annulus_carries_full_bootstrap_track(tmp_path, angle, reverse):
    import math

    board = _board(tmp_path)
    radians = math.radians(angle)
    def rotate(x, y):
        return (10 + x * math.cos(radians) - y * math.sin(radians),
                10 + x * math.sin(radians) + y * math.cos(radians))
    start, end, center, target = [
        rotate(*p) for p in [(0, 0), (2, 0), (2.1, 0.1), (4, 0.1)]
    ]
    _pad(board, "L1", start)
    _pad(board, "C1", target, pth=True)
    _track(board, *( (end, start) if reverse else (start, end)), width=0.2)
    _via(board, center)
    _track(board, center, target, width=0.2, layer=pcbnew.B_Cu)
    contract = {**_spec(limit=6), "min_route_width_mm": 0.17}
    assert not audit_local_copper(board, [contract])


@pytest.mark.parametrize("start,end", [
    ((-0.04, 0), (0.04, 0)),  # Entire segment inside drill.
    ((-1, 0.4), (1, 0.4)),  # Outer edge tangency.
    ((0.3, 0), (1, 0)),  # Endpoint-only annular tangency.
    ((-1, 0.25), (1, 0.25)),  # Positive but sub-width annular overlap.
])
def test_offset_via_requires_full_width_annular_section(start, end):
    native = lambda p: tuple(pcbnew.FromMM(v) for v in p)
    assert _track_via_annular_contact(
        native(start), native(end), pcbnew.FromMM(0.2), (0, 0),
        pcbnew.FromMM(0.6), pcbnew.FromMM(0.3),
    ) is None


def test_offset_overlapping_tracks_are_connected_but_separate_tracks_are_not(tmp_path):
    board = _board(tmp_path)
    _pad(board, "L1", (0, 0))
    _pad(board, "C1", (8, 0))
    _track(board, (0, 0), (4, 0), width=0.6)
    _track(board, (4.2, 0.1), (8, 0), width=0.6)
    assert not audit_local_copper(board, [_spec()])
    separate = _board(tmp_path)
    _pad(separate, "L1", (0, 0))
    _pad(separate, "C1", (8, 0))
    _track(separate, (0, 0), (4, 0))
    _track(separate, (4.6, 0.2), (8, 0))
    assert any("no qualified" in e for e in audit_local_copper(separate, [_spec()]))


def test_floating_bootstrap_checks_both_pin_pairs_and_routed_lengths(tmp_path):
    board = _board(tmp_path)
    board.Add(pcbnew.NETINFO_ITEM(board, "BST"))
    _pad(board, "U1", (0, 0), number="1", net="BST")
    _pad(board, "U1", (0, 2), number="2")
    _pad(board, "C1", (2, 0), number="1", net="BST")
    _pad(board, "C1", (2, 2), number="2")
    _track(board, (0, 0), (2, 0), width=0.2, net="BST")
    _track(board, (0, 2), (0, 6), width=0.2)
    _track(board, (0, 6), (2, 6), width=0.2)
    _track(board, (2, 6), (2, 2), width=0.2)
    contract = {"owner_ref": "U1", "bypass_ref": "C1",
                "pin_pairs": [["1", "1"], ["2", "2"]],
                "max_route_length_mm": 6.0, "min_route_width_mm": 0.17}
    errors = audit_local_copper(board, [contract])
    assert len(errors) == 1 and "exceeds" in errors[0] and "U1.2/C1.2" in errors[0]


def test_local_input_hf_pad_proximity_does_not_excuse_long_power_route(tmp_path):
    board = _board(tmp_path)
    _pad(board, "U1", (0, 0))
    _pad(board, "C1", (2, 0))
    for a, b in [((0, 0), (0, 4)), ((0, 4), (2, 4)), ((2, 4), (2, 0))]:
        _track(board, a, b)
    contract = {"owner_ref": "U1", "bypass_ref": "C1", "power_pins": ["1", "1"],
                "max_route_length_mm": 6.0, "min_route_width_mm": 0.5}
    assert any("exceeds" in e for e in audit_local_copper(board, [contract]))


@pytest.mark.parametrize("route,expected", [
    ("good", None),
    ("long", "exceeds"),
    ("neck", "narrow copper"),
])
def test_leaf_current_gate_audits_native_routed_board(tmp_path, route, expected):
    from kicraft.autoplacer.brain.leaf_acceptance import evaluate_leaf_acceptance
    from kicraft.autoplacer.brain.leaf_routing import _validate_leaf_current_contracts
    from kicraft.autoplacer.routing_board import validate_routed_board

    if not shutil.which("kicad-cli"):
        pytest.skip("kicad-cli required for native DRC")
    board = _board(tmp_path)
    _pad(board, "U1", (10, 10))
    _pad(board, "C1", (18, 10))
    if route == "long":
        for a, b in [((10, 10), (10, 20)), ((10, 20), (18, 20)),
                     ((18, 20), (18, 10))]:
            _track(board, a, b)
    elif route == "neck":
        _track(board, (10, 10), (13, 10))
        _track(board, (13, 10), (15, 10), width=0.2)
        _track(board, (15, 10), (18, 10))
    else:
        _track(board, (10, 10), (18, 10))
    for a, b in [((0, 0), (30, 0)), ((30, 0), (30, 30)),
                 ((30, 30), (0, 30)), ((0, 30), (0, 0))]:
        edge = pcbnew.PCB_SHAPE()
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(_point(*a))
        edge.SetEnd(_point(*b))
        edge.SetLayer(pcbnew.Edge_Cuts)
        edge.SetWidth(pcbnew.FromMM(0.05))
        board.Add(edge)
    path = tmp_path / "routed.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    spec = _spec(route_anchor="U1", limit=12)
    spec["current_carrying_net"] = "PWR"
    cfg = {"local_bypasses": [spec]}
    original = copy.deepcopy(cfg)
    validation = validate_routed_board(str(path))
    assert validation["accepted"], validation
    assert validation["drc"]["shorts"] == 0
    assert validation["drc"]["unconnected"] == 0
    _validate_leaf_current_contracts(path, cfg, validation)
    assert cfg == original
    result = evaluate_leaf_acceptance(validation, {})
    if expected:
        assert not validation["accepted"]
        assert "current_contract" in validation["rejection_reasons"]
        assert any(expected in error for error in validation["current_contract_errors"])
        assert not result.accepted
        assert "current_contract" in result.rejection_reasons
    else:
        assert validation["current_contract_errors"] == []
        assert result.accepted


@pytest.mark.parametrize("members,missing_pin,expected", [
    ((), False, None),
    (("U1",), False, "split across leaves"),
    (("C1",), False, "split across leaves"),
    (("U1", "C1"), True, "missing or ambiguous native pad"),
])
def test_leaf_current_projection_handles_native_membership(tmp_path, members, missing_pin, expected):
    from kicraft.autoplacer.brain.leaf_routing import _validate_leaf_current_contracts

    board = _board(tmp_path)
    for index, ref in enumerate(members):
        _pad(board, ref, (10 + index * 8, 10), number="2" if missing_pin else "1")
    path = tmp_path / "leaf.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    validation = {"accepted": True, "rejection_reasons": []}
    _validate_leaf_current_contracts(
        path, {"local_bypasses": [_spec(route_anchor="U1")]}, validation
    )
    assert validation["accepted"] is (expected is None)
    if expected:
        assert any(expected in error for error in validation["current_contract_errors"])
    else:
        assert validation["current_contract_errors"] == []


def _ground_fixture(tmp_path):
    board = _board(tmp_path)
    board.Add(pcbnew.NETINFO_ITEM(board, "GND"))
    _pad(board, "U1", (5, 5), number="10", net="GND")
    _pad(board, "C1", (15, 5), number="2", net="GND")
    for a, b in zip(((0, 0), (20, 0), (20, 10), (0, 10)),
                    ((20, 0), (20, 10), (0, 10), (0, 0))):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(_point(*a))
        edge.SetEnd(_point(*b))
        edge.SetLayer(pcbnew.Edge_Cuts)
        board.Add(edge)
    spec = {"owner_ref": "U1", "bypass_ref": "C1",
            "ground_return": {"owner_ref": "U1", "owner_pin": "10", "member_pin": "2",
                              "net": "GND", "max_via_distance_mm": 3.0}}
    return board, spec


def _ground_zone(board, layer, corners):
    zone = pcbnew.ZONE(board)
    zone.SetNet(board.GetNetInfo().GetNetItem("GND"))
    zone.SetLayer(layer)
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    chain = pcbnew.SHAPE_LINE_CHAIN()
    for x, y in corners:
        chain.Append(pcbnew.FromMM(x), pcbnew.FromMM(y))
    chain.SetClosed(True)
    zone.Outline().AddOutline(chain)
    board.Add(zone)


def _fill(board):
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())


def test_ground_return_requires_filled_shared_region(tmp_path):
    board, spec = _ground_fixture(tmp_path)
    assert any("ground return" in e for e in audit_local_copper(board, [spec]))
    _ground_zone(board, pcbnew.F_Cu, [(3, 3), (7, 3), (7, 7), (3, 7)])
    _fill(board)
    assert any("ground return" in e for e in audit_local_copper(board, [spec]))


def test_ground_return_rejects_separate_same_net_islands(tmp_path):
    board, spec = _ground_fixture(tmp_path)
    _ground_zone(board, pcbnew.F_Cu, [(3, 3), (7, 3), (7, 7), (3, 7)])
    _ground_zone(board, pcbnew.F_Cu, [(13, 3), (17, 3), (17, 7), (13, 7)])
    _fill(board)
    assert any("ground return" in e for e in audit_local_copper(board, [spec]))


def test_ground_return_accepts_direct_shared_plane_without_vias(tmp_path):
    board, spec = _ground_fixture(tmp_path)
    _ground_zone(board, pcbnew.F_Cu, [(2, 2), (18, 2), (18, 8), (2, 8)])
    _fill(board)
    assert not audit_local_copper(board, [spec])


@pytest.mark.parametrize("shared", [False, True])
def test_leaf_gate_checks_native_ground_return(tmp_path, shared):
    from kicraft.autoplacer.brain.leaf_routing import _validate_leaf_current_contracts

    board, spec = _ground_fixture(tmp_path)
    if shared:
        _ground_zone(board, pcbnew.F_Cu, [(2, 2), (18, 2), (18, 8), (2, 8)])
    else:
        _ground_zone(board, pcbnew.F_Cu, [(3, 3), (7, 3), (7, 7), (3, 7)])
        _ground_zone(board, pcbnew.F_Cu, [(13, 3), (17, 3), (17, 7), (13, 7)])
    _fill(board)
    path = tmp_path / "leaf-ground.kicad_pcb"
    pcbnew.SaveBoard(str(path), board)
    cfg = {"local_bypasses": [spec]}
    original = copy.deepcopy(cfg)
    validation = {"accepted": True, "rejection_reasons": []}
    _validate_leaf_current_contracts(path, cfg, validation)
    assert cfg == original
    assert validation["accepted"] is shared
    if shared:
        assert validation["current_contract_errors"] == []
    else:
        assert "current_contract" in validation["rejection_reasons"]
        assert any("no shared filled ground region" in error
                   for error in validation["current_contract_errors"])


def test_ground_return_accepts_locally_bonded_vias_to_opposite_plane(tmp_path):
    board, spec = _ground_fixture(tmp_path)
    for x in (5, 15):
        _via(board, (x + 1, 5), net="GND")
        _track(board, (x, 5), (x + 1, 5), width=0.25, net="GND")
    _ground_zone(board, pcbnew.B_Cu, [(2, 2), (18, 2), (18, 8), (2, 8)])
    _fill(board)
    assert not audit_local_copper(board, [spec])


@pytest.mark.parametrize("bond", ["shared_plane", "local_vias"])
def test_output_ground_return_uses_converter_not_inductor(tmp_path, bond):
    board, spec = _ground_fixture(tmp_path)
    list(board.FindFootprintByReference("U1").Pads())[0].SetNumber("4")
    _pad(board, "L1", (11, 5), number="2")
    _pad(board, "C1", (14, 5), number="1")
    _track(board, (11, 5), (14, 5))
    spec.update(owner_ref="L1", pin_pairs=[["2", "1"]],
                max_route_length_mm=15.0, min_route_width_mm=0.5)
    spec["ground_return"]["owner_pin"] = "4"
    if bond == "shared_plane":
        _ground_zone(board, pcbnew.F_Cu, [(2, 2), (18, 2), (18, 8), (2, 8)])
    else:
        for x in (5, 15):
            _via(board, (x + 1, 5), net="GND")
            _track(board, (x, 5), (x + 1, 5), width=0.25, net="GND")
        _ground_zone(board, pcbnew.B_Cu, [(2, 2), (18, 2), (18, 8), (2, 8)])
    _fill(board)
    assert not audit_local_copper(board, [spec])
    spec["ground_return"]["owner_ref"] = "L1"
    assert any("ground return" in e for e in audit_local_copper(board, [spec]))
    spec["ground_return"]["owner_ref"] = "C1"
    assert any("ground return" in e for e in audit_local_copper(board, [spec]))
    del spec["ground_return"]["owner_ref"]
    assert any("ground return" in e for e in audit_local_copper(board, [spec]))
    spec["ground_return"]["owner_ref"] = "U1"
    del spec["ground_return"]["owner_pin"]
    assert any("ground return" in e for e in audit_local_copper(board, [spec]))


def test_ground_return_rejects_near_unbonded_via_and_wrong_net(tmp_path):
    board, spec = _ground_fixture(tmp_path)
    for x in (5, 15):
        _via(board, (x + 1, 5), net="GND")
    _track(board, (5, 5), (6, 5), width=0.25, net="GND")
    _ground_zone(board, pcbnew.B_Cu, [(2, 2), (18, 2), (18, 8), (2, 8)])
    _fill(board)
    assert any("ground return" in e for e in audit_local_copper(board, [spec]))
    spec["ground_return"]["net"] = "PWR"
    assert any("net mismatch" in e for e in audit_local_copper(board, [spec]))


def test_ground_return_rejects_long_local_via_detour(tmp_path):
    board, spec = _ground_fixture(tmp_path)
    for x in (5, 15):
        _via(board, (x + 1, 5), net="GND")
    _track(board, (5, 5), (6, 5), width=0.25, net="GND")
    for a, b in [((15, 5), (15, 8)), ((15, 8), (16, 8)), ((16, 8), (16, 5))]:
        _track(board, a, b, width=0.25, net="GND")
    _ground_zone(board, pcbnew.B_Cu, [(2, 2), (18, 2), (18, 8), (2, 8)])
    _fill(board)
    assert any("ground return" in e for e in audit_local_copper(board, [spec]))


def test_via_width_requests_native_layer_first():
    from kicraft.autoplacer.hardware.local_contract import _via_width

    class LayeredVia:
        def GetWidth(self, layer):
            assert layer == pcbnew.B_Cu
            return 42

    assert _via_width(LayeredVia(), pcbnew.B_Cu) == 42


def test_ground_return_accepts_local_filled_islands_to_common_back_plane(tmp_path):
    board, spec = _ground_fixture(tmp_path)
    for x in (5, 15):
        _via(board, (x + 1, 5), net="GND")
        _ground_zone(board, pcbnew.F_Cu,
                     [(x - 1, 4), (x + 2, 4), (x + 2, 6), (x - 1, 6)])
    _ground_zone(board, pcbnew.B_Cu, [(2, 2), (18, 2), (18, 8), (2, 8)])
    _fill(board)
    assert not audit_local_copper(board, [spec])


def test_ground_filled_bond_rejects_hole_detour():
    from kicraft.autoplacer.hardware.local_contract import _filled_straight_bond

    region = pcbnew.SHAPE_POLY_SET()
    region.NewOutline()
    for x, y in ((0, 0), (4, 0), (4, 4), (0, 4)):
        region.Append(pcbnew.FromMM(x), pcbnew.FromMM(y))
    hole = pcbnew.SHAPE_POLY_SET()
    hole.NewOutline()
    for x, y in ((1.9, 1), (2.1, 1), (2.1, 3), (1.9, 3)):
        hole.Append(pcbnew.FromMM(x), pcbnew.FromMM(y))
    region.BooleanSubtract(hole)
    assert not _filled_straight_bond(
        region, (pcbnew.FromMM(1), pcbnew.FromMM(2)),
        (pcbnew.FromMM(3), pcbnew.FromMM(2)))


@pytest.mark.parametrize("planner", ["local", "global"])
def test_ground_reservations_ignore_equivalent_copper_item_order(tmp_path, planner):
    """Identical physical copper must leave identical space for signal routes."""
    from kicraft.autoplacer.hardware.ground_reservations import (
        reserve_connected_ground, reserve_local_ground_planes,
    )

    reservations = []
    links = [((5, 5), (6, 4)), ((5, 5), (6, 6)),
             ((15, 5), (14, 4)), ((15, 5), (14, 6))]
    for reverse in (False, True):
        board, spec = _ground_fixture(tmp_path)
        for start, end in reversed(links) if reverse else links:
            _track(board, start, end, width=.25, net="GND")
            _via(board, end, net="GND")
        for left, right in ((3, 8), (12, 17)):
            _ground_zone(board, pcbnew.F_Cu,
                         [(left, 3), (right, 3), (right, 7), (left, 7)])
        _ground_zone(board, pcbnew.B_Cu, [(2, 2), (18, 2), (18, 8), (2, 8)])
        _fill(board)
        assert not audit_local_copper(board, [spec])
        if planner == "local":
            reserve_local_ground_planes(
                board, [spec], "order-independent", router_clearance_mm=.17,
            )
        else:
            reserve_connected_ground(
                board, "GND", "order-independent", width_mm=.17,
                router_clearance_mm=.17,
            )
        masks = {layer: pcbnew.SHAPE_POLY_SET()
                 for layer in (pcbnew.F_Cu, pcbnew.B_Cu)}
        for zone in board.Zones():
            if zone.GetZoneName() == "order-independent":
                masks[zone.GetLayer()].BooleanAdd(zone.Outline())
        reservations.append(masks)
        _fill(board)
        assert not audit_local_copper(board, [spec])

    for layer in (pcbnew.F_Cu, pcbnew.B_Cu):
        difference = pcbnew.SHAPE_POLY_SET(reservations[0][layer])
        difference.BooleanXor(reservations[1][layer])
        assert difference.OutlineCount() == 0


def test_ground_forest_uses_short_front_bond_instead_of_long_back_detour(tmp_path):
    from kicraft.autoplacer.hardware.ground_reservations import reserve_connected_ground

    board, spec = _ground_fixture(tmp_path)
    for start, end in (((5, 5), (3, 5)), ((15, 5), (17, 5))):
        _track(board, start, end, width=.25, net="GND")
        _via(board, end, net="GND")
    for layer in (pcbnew.F_Cu, pcbnew.B_Cu):
        _ground_zone(board, layer, [(2, 2), (18, 2), (18, 8), (2, 8)])
    _fill(board)
    assert not audit_local_copper(board, [spec])

    reserve_connected_ground(
        board, "GND", "short-physical-bond", width_mm=.17,
        router_clearance_mm=.17,
    )

    # A legal back-layer signal corridor must not be blocked by the longer
    # plane alternative when a shorter native front bond already exists.
    corridor = pcbnew.SEG(_point(10, 2), _point(10, 8))
    masks = [zone for zone in board.Zones()
             if zone.GetZoneName() == "short-physical-bond"]
    assert all(not zone.Outline().Collide(corridor, pcbnew.FromMM(.27))
               for zone in masks if zone.IsOnLayer(pcbnew.B_Cu))
    assert any(zone.Outline().Collide(corridor, 0)
               for zone in masks if zone.IsOnLayer(pcbnew.F_Cu))
    _fill(board)
    assert not audit_local_copper(board, [spec])
