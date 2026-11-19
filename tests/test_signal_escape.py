"""Regression coverage for SMD signal-field and cross-leaf pre-escapes."""
from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.breakout_stubs import (  # noqa: E402
    STAMP_CLEARANCE_GUARD_MM,
    add_breakout_stubs,
)
from kicraft.autoplacer.brain.signal_escape import signal_escape_specs  # noqa: E402
from kicraft.autoplacer.fab_profile import fab_floors, fanout_via  # noqa: E402


_mm = pcbnew.FromMM


def _dense_pad_field(path, *, pitch_mm=0.4, pad_width_mm=0.2, rotation=0, face="front"):
    board = pcbnew.NewBoard(path)
    for name in ("+3V3", "GND", "GPIO0", "GPIO1", "GPIO2", "GPIO3", "GPIO4", "GPIO5"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("U1")
    fp.SetPosition(pcbnew.VECTOR2I(_mm(10.0), _mm(10.0)))
    if face == "back":
        fp.SetLayer(pcbnew.B_Cu)
    board.Add(fp)
    for index, net in enumerate(
        ("+3V3", "GPIO0", "GPIO1", "GND", "GPIO2", "GPIO3", "GPIO4", "GPIO5"), 1
    ):
        pad = pcbnew.PAD(fp)
        pad.SetNumber(str(index))
        pad.SetSize(pcbnew.VECTOR2I(_mm(pad_width_mm), _mm(0.875)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(9.0 + pitch_mm * index), _mm(12.0)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        back = face == "back" or (face == "mixed" and index % 2 == 0)
        layers = pcbnew.LSET()
        for layer in ((pcbnew.B_Cu, pcbnew.B_Paste, pcbnew.B_Mask) if back
                      else (pcbnew.F_Cu, pcbnew.F_Paste, pcbnew.F_Mask)):
            layers.AddLayer(layer)
        pad.SetLayerSet(layers)
        fp.Add(pad)
        pad.SetNet(board.GetNetInfo().GetNetItem(net))
    fp.SetOrientationDegrees(rotation)
    board.Save(path)


@pytest.mark.parametrize("rotation", (0, 90, 37))
def test_dense_signal_pads_reserve_spaced_strict_dogbones_but_plane_pads_do_not(tmp_path, rotation):
    path = str(tmp_path / "dense.kicad_pcb")
    _dense_pad_field(path, rotation=rotation)

    specs = signal_escape_specs(pcbnew.LoadBoard(path))

    via_size_mm, _ = fanout_via({})
    floors = fab_floors({})
    required_lane_mm = via_size_mm + floors["track_mm"] + 2 * (
        floors["clearance_mm"] + STAMP_CLEARANCE_GUARD_MM
    )
    # Every signal must escape, including across the footprint-anchor side
    # boundary. Dropping alternate pads traps them behind the locked dog-bones.
    assert {spec.pad for spec in specs} == {"2", "3", "5", "6", "7", "8"}
    stamped = add_breakout_stubs(path, specs)
    assert stamped["stubs"] == len(specs), stamped
    vias = [
        via
        for via in pcbnew.LoadBoard(path).GetTracks()
        if isinstance(via, pcbnew.PCB_VIA)
    ]
    assert len(vias) == len(specs)
    assert {via.GetNetname() for via in vias} == {f"GPIO{index}" for index in range(6)}
    assert all(
        (pcbnew.ToMM(first.GetPosition().x) - pcbnew.ToMM(second.GetPosition().x)) ** 2
        + (pcbnew.ToMM(first.GetPosition().y) - pcbnew.ToMM(second.GetPosition().y)) ** 2
        >= required_lane_mm**2
        for index, first in enumerate(vias)
        for second in vias[index + 1:]
    )
    assert all(spec.via_at_end for spec in specs)
    assert all(spec.strict_same_footprint for spec in specs)


@pytest.mark.parametrize("face", ("front", "back", "mixed"))
@pytest.mark.parametrize("rotation", (0, 37, 90))
def test_saved_pad_faces_get_connected_locked_escape_copper(tmp_path, face, rotation):
    path = str(tmp_path / "faces.kicad_pcb")
    _dense_pad_field(path, face=face, rotation=rotation)
    board = pcbnew.LoadBoard(path)
    pads = {pad.GetNumber(): pad for fp in board.GetFootprints() for pad in fp.Pads()}
    specs = signal_escape_specs(board)
    assert {spec.pad for spec in specs} == {"2", "3", "5", "6", "7", "8"}
    for spec in specs:
        pad = pads[spec.pad]
        expected = pcbnew.B_Cu if face == "back" or (face == "mixed" and int(spec.pad) % 2 == 0) else pcbnew.F_Cu
        assert pad.IsOnLayer(expected)
        assert not pad.IsOnLayer(pcbnew.F_Cu if expected == pcbnew.B_Cu else pcbnew.B_Cu)
        assert spec.layer == ("B.Cu" if expected == pcbnew.B_Cu else "F.Cu")

    result = add_breakout_stubs(path, specs)
    assert result["stubs"] == len(specs), result
    board = pcbnew.LoadBoard(path)
    pads = {pad.GetNumber(): pad for fp in board.GetFootprints() for pad in fp.Pads()}
    tracks = list(board.GetTracks())
    board.BuildConnectivity()
    connectivity = board.GetConnectivity()
    for spec in specs:
        pad = pads[spec.pad]
        layer = pcbnew.B_Cu if spec.layer == "B.Cu" else pcbnew.F_Cu
        copper = [track for track in tracks if track.GetNetname() == pad.GetNetname()]
        segments = [track for track in copper if not isinstance(track, pcbnew.PCB_VIA)]
        vias = [track for track in copper if isinstance(track, pcbnew.PCB_VIA)]
        assert segments and len(vias) == 1
        assert all(track.IsLocked() and track.GetLayer() == layer for track in segments)
        assert vias[0].IsLocked()
        reached = set()
        pending = list(connectivity.GetConnectedTracks(pad))
        while pending:
            item = pending.pop()
            uid = item.m_Uuid.AsString()
            if uid in reached:
                continue
            reached.add(uid)
            pending.extend(connectivity.GetConnectedTracks(item))
        # Saved positions may differ by 1 IU after a right-angle transform.
        # Native copper connectivity, not exact center equality, is the contract.
        assert {track.m_Uuid.AsString() for track in copper} <= reached


@pytest.mark.parametrize("face", ("front", "back"))
@pytest.mark.parametrize("rotation", (0, 37))
def test_obstructed_face_escape_is_not_partially_stamped(tmp_path, face, rotation):
    path = str(tmp_path / "obstructed.kicad_pcb")
    _dense_pad_field(path, face=face, rotation=rotation)
    board = pcbnew.LoadBoard(path)
    spec = next(s for s in signal_escape_specs(board) if s.pad == "2")
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("BLOCK")
    board.Add(fp)
    pad = pcbnew.PAD(fp)
    pad.SetNumber("1")
    pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
    layers = pcbnew.LSET()
    layers.AddLayer(pcbnew.B_Cu if face == "back" else pcbnew.F_Cu)
    pad.SetLayerSet(layers)
    pad.SetSize(pcbnew.VECTOR2I(_mm(0.3), _mm(0.3)))
    x, y = spec.waypoints[0]
    pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
    fp.Add(pad)
    pad.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Save(path)

    result = add_breakout_stubs(path, [spec])
    assert result["stubs"] == result["segments"] == result["vias"] == 0
    assert any("waypoint_crosses_pad" in reason for reason in result["skipped"])
    assert not list(pcbnew.LoadBoard(path).GetTracks())

@pytest.mark.parametrize("face", ("front", "back"))
@pytest.mark.parametrize("rotation", (0, 37))
def test_internal_wider_pitch_signal_field_keeps_native_via_landings(tmp_path, face, rotation):
    from kicraft.autoplacer.brain.signal_escape import unlanded_interface_components

    path = str(tmp_path / "internal-outputs.kicad_pcb")
    _dense_pad_field(
        path, pitch_mm=0.65, pad_width_mm=0.45, rotation=rotation, face=face,
    )
    # No external-net hint: local IC outputs need access before wide power
    # routes too. Verify actual pad-to-via connectivity, not just planned specs.
    board = pcbnew.LoadBoard(path)
    specs = signal_escape_specs(board)
    result = add_breakout_stubs(path, specs)
    assert not result["skipped"], result
    assert not unlanded_interface_components(
        pcbnew.LoadBoard(path), {f"GPIO{index}" for index in range(6)},
    )


def test_two_ounce_escape_planning_and_stamping_enforce_actual_width(tmp_path):
    from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec

    path = str(tmp_path / "heavy.kicad_pcb")
    _dense_pad_field(path, pitch_mm=0.5)
    cfg = {"thermal_requirements": [{"outer_copper_oz": 2}],
           "fab_capability": {"min_track_mm": 0.1, "min_clearance_mm": 0.1}}
    board = pcbnew.LoadBoard(path)
    planned = signal_escape_specs(board, cfg)
    assert planned and all(spec.width_mm == 0.17 for spec in planned)
    assert all((spec.via_size_mm, spec.via_drill_mm) == (0.36, 0.15) for spec in planned)
    fp = next(iter(board.GetFootprints()))
    for pad in list(fp.Pads()):
        if pad.GetNumber() != "2":
            fp.Remove(pad)
    board.Save(path)
    spec = BreakoutSpec(ref="U1", pad="2", waypoints=[(10.0, 14.0)], width_mm=0.10)
    result = add_breakout_stubs(path, [spec], cfg=cfg)
    assert result["stubs"] == 1, result
    tracks = [t for t in pcbnew.LoadBoard(path).GetTracks() if not isinstance(t, pcbnew.PCB_VIA)]
    assert tracks and all(pcbnew.ToMM(track.GetWidth()) >= 0.17 for track in tracks)


def test_dense_dogbone_uses_short_clear_exit_before_nearby_land(tmp_path):
    """A long planned stagger is blocked although a legal short via exists."""
    path = str(tmp_path / "short-exit.kicad_pcb")
    board = pcbnew.NewBoard(path)
    for name in ("SIG", "POWER"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("U1")
    fp.SetPosition(pcbnew.VECTOR2I(_mm(10), _mm(10)))
    board.Add(fp)
    # Reproduce the 0.5 mm-pitch pad row and adjacent 0603 land geometry.
    for index in range(8):
        pad = pcbnew.PAD(fp)
        pad.SetNumber(str(index + 1))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetShape(pcbnew.PAD_SHAPE_RECT)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.24), _mm(0.58)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(8.25 + index * 0.5), _mm(8.0875)))
        fp.Add(pad)
        if index == 4:
            pad.SetNet(board.GetNetInfo().GetNetItem("SIG"))
    cap = pcbnew.FOOTPRINT(board)
    cap.SetReference("C1")
    board.Add(cap)
    obstacle = pcbnew.PAD(cap)
    obstacle.SetNumber("1")
    obstacle.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
    obstacle.SetShape(pcbnew.PAD_SHAPE_RECT)
    obstacle.SetLayerSet(pcbnew.PAD.SMDMask())
    obstacle.SetSize(pcbnew.VECTOR2I(_mm(0.9), _mm(0.95)))
    obstacle.SetPosition(pcbnew.VECTOR2I(_mm(10.35), _mm(6.4875)))
    cap.Add(obstacle)
    obstacle.SetNet(board.GetNetInfo().GetNetItem("POWER"))
    board.Save(path)

    cfg = {"thermal_requirements": [{"outer_copper_oz": 2}],
           "hole_to_hole_min_mm": 0.2, "hole_clearance_min_mm": 0.2}
    specs = signal_escape_specs(pcbnew.LoadBoard(path), cfg)
    spec = next(spec for spec in specs if spec.ref == "U1" and spec.pad == "5")
    result = add_breakout_stubs(path, [spec], cfg=cfg)
    assert result["stubs"] == result["vias"] == 1, result
    routed = pcbnew.LoadBoard(path)
    via = next(t for t in routed.GetTracks() if isinstance(t, pcbnew.PCB_VIA))
    assert pcbnew.ToMM(via.GetWidth(pcbnew.F_Cu)) == pytest.approx(0.36)
    assert pcbnew.ToMM(via.GetDrillValue()) == pytest.approx(0.15)
    assert pcbnew.ToMM(via.GetPosition().y) > 7.3
    from kicraft.autoplacer.brain.breakout_stubs import _foreign_pad_margins
    src = next(p for f in routed.GetFootprints() if f.GetReferenceAsString() == "U1"
               for p in f.Pads() if p.GetNumber() == "5")
    _, obstacles = _foreign_pad_margins(
        routed, src, floor_mm=0.17, half_width_mm=0.085,
        layer_id=pcbnew.F_Cu, strict_same_fp=True, hole_clearance_mm=0.2,
    )
    assert not any(p.HitTest(via.GetPosition(), margin + _mm(0.095))
                   for p, margin in obstacles)


def test_off_board_planned_dogbone_uses_guarded_short_exit(tmp_path):
    from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec

    path = str(tmp_path / "edge-exit.kicad_pcb")
    _interface_board(path)
    board = pcbnew.LoadBoard(path)
    for a, b in (((0, 0), (40, 0)), ((40, 0), (40, 11.6)),
                 ((40, 11.6), (0, 11.6)), ((0, 11.6), (0, 0))):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetStart(pcbnew.VECTOR2I(*(_mm(v) for v in a)))
        edge.SetEnd(pcbnew.VECTOR2I(*(_mm(v) for v in b)))
        edge.SetLayer(pcbnew.Edge_Cuts)
        board.Add(edge)
    board.Save(path)
    spec = BreakoutSpec(
        ref="U1", pad="1", waypoints=[(10, 12)], width_mm=0.17,
        via_at_end=True, via_size_mm=0.36, via_drill_mm=0.15,
        strict_same_footprint=True, short_fanout_fallback=True,
    )
    result = add_breakout_stubs(path, [spec])
    assert result["stubs"] == result["vias"] == 1, result
    board = pcbnew.LoadBoard(path)
    via = next(t for t in board.GetTracks() if isinstance(t, pcbnew.PCB_VIA))
    assert 10 < pcbnew.ToMM(via.GetPosition().y) < 11.1


def test_existing_native_via_is_read_without_layer_assertions(tmp_path, capfd):
    from kicraft.autoplacer.brain.unconnected_repair import _copper_obstacles

    path = str(tmp_path / "existing-via.kicad_pcb")
    _dense_pad_field(path)
    board = pcbnew.LoadBoard(path)
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(20), _mm(20)))
    via.SetWidth(_mm(0.36))
    via.SetDrill(_mm(0.15))
    via.SetNet(board.GetNetInfo().GetNetItem("GPIO0"))
    board.Add(via)
    board.Save(path)
    obstacles = _copper_obstacles(pcbnew.LoadBoard(path))
    assert any(row[0] == "*" and row[1] == "GPIO0"
               and row[-1] == pytest.approx(0.18) for row in obstacles)
    specs = signal_escape_specs(pcbnew.LoadBoard(path))
    add_breakout_stubs(path, specs[:1])
    assert "GetWidth" not in capfd.readouterr().err


def _interface_board(path):
    board = pcbnew.NewBoard(path)
    for name in ("SIG", "GND", "+3V3", "LOCAL"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))
    for ref, x, net in (("U1", 10, "SIG"), ("U2", 15, "SIG"),
                        ("U3", 20, "SIG"), ("U4", 25, "GND"),
                        ("U5", 30, "+3V3"), ("U6", 35, "LOCAL")):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(9)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetNumber("1")
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.9), _mm(1.2)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(10)))
        fp.Add(pad)
        pad.SetNet(board.GetNetInfo().GetNetItem(net))
    board.Save(path)


def test_finished_interface_retry_adds_only_missing_landings(tmp_path):
    from types import SimpleNamespace
    from kicraft.autoplacer.brain.leaf_routing import _finish_interface_landings
    from kicraft.autoplacer.brain.signal_escape import unlanded_interface_components

    path = tmp_path / "finish-interface.kicad_pcb"
    _interface_board(str(path))
    extraction = SimpleNamespace(external_net_names=["SIG"])
    board = pcbnew.LoadBoard(str(path))
    specs = signal_escape_specs(board, external_net_names={"SIG"})
    add_breakout_stubs(str(path), [next(s for s in specs if s.ref == "U1")])
    before = pcbnew.LoadBoard(str(path))
    assert len(unlanded_interface_components(before, {"SIG"})) == 2
    sidecar = path.with_suffix(".kicad_dru")
    sidecar.write_bytes(b"(version 1)\n")

    result = _finish_interface_landings(path, extraction, {})
    assert result["stubs"] == result["vias"] == 2, result
    assert unlanded_interface_components(pcbnew.LoadBoard(str(path)), {"SIG"}) == []
    assert sidecar.read_bytes() == b"(version 1)\n"
    assert _finish_interface_landings(path, extraction, {})["stubs"] == 0


def test_wide_interface_signal_pads_are_planned_and_stamped(tmp_path):
    path = str(tmp_path / "interface.kicad_pcb")
    _interface_board(path)
    board = pcbnew.LoadBoard(path)
    assert signal_escape_specs(board) == []
    specs = signal_escape_specs(board, external_net_names={"SIG", "GND", "+3V3"})
    assert {spec.ref for spec in specs} == {"U1", "U2", "U3"}
    result = add_breakout_stubs(path, specs)
    assert result["stubs"] == result["vias"] == 3, result


def test_wide_interface_obstruction_is_reported_without_partial_copper(tmp_path):
    path = str(tmp_path / "blocked-interface.kicad_pcb")
    _interface_board(path)
    board = pcbnew.LoadBoard(path)
    spec = next(s for s in signal_escape_specs(board, external_net_names={"SIG"})
                if s.ref == "U1")
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("OBSTACLE")
    board.Add(fp)
    obstacle = pcbnew.PAD(fp)
    obstacle.SetNumber("1")
    obstacle.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
    obstacle.SetLayerSet(pcbnew.PAD.SMDMask())
    obstacle.SetSize(pcbnew.VECTOR2I(_mm(0.4), _mm(0.4)))
    obstacle.SetPosition(pcbnew.VECTOR2I(*(_mm(v) for v in spec.waypoints[0])))
    fp.Add(obstacle)
    obstacle.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Save(path)
    result = add_breakout_stubs(path, [spec])
    assert result["stubs"] == result["vias"] == 0
    assert result["skipped"]
    assert not list(pcbnew.LoadBoard(path).GetTracks())


def test_interface_landing_is_per_native_copper_component_not_net(tmp_path):
    from kicraft.autoplacer.brain.signal_escape import unlanded_interface_components
    from kicraft.autoplacer.brain.leaf_acceptance import (
        _gate_no_unconnected, acceptance_config_from_dict, split_unconnected_nets,
    )

    path = str(tmp_path / "components.kicad_pcb")
    _interface_board(path)
    board = pcbnew.LoadBoard(path)
    net = board.GetNetInfo().GetNetItem("SIG")
    def segment(a, b):
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(pcbnew.VECTOR2I(*(_mm(v) for v in a)))
        track.SetEnd(pcbnew.VECTOR2I(*(_mm(v) for v in b)))
        track.SetWidth(_mm(0.2))
        track.SetLayer(pcbnew.F_Cu)
        track.SetNet(net)
        board.Add(track)
    # U1 and U2 share a connected via; U3 only has a short pad-contact nub.
    segment((10, 10), (15, 10))
    segment((15, 10), (15, 11))
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(15), _mm(11)))
    via.SetWidth(_mm(0.36))
    via.SetDrill(_mm(0.15))
    via.SetNet(net)
    board.Add(via)
    segment((20, 10.55), (20, 10.59))
    board.Save(path)
    missing = unlanded_interface_components(pcbnew.LoadBoard(path), {"SIG"})
    assert missing == [{"net": "SIG", "pads": ["U3.1"]}]
    validation = {
        "drc": {"unconnected_nets": ["SIG"]},
        "interface_port_names": ["SIG"],
        "interface_unlanded_nets": [row["net"] for row in missing],
    }
    assert split_unconnected_nets(validation, acceptance_config_from_dict({}))[0] == ["SIG"]
    assert not _gate_no_unconnected(validation, {}, acceptance_config_from_dict({}))[0]
    validation["interface_unlanded_nets"] = []
    assert split_unconnected_nets(validation, acceptance_config_from_dict({}))[2] == ["SIG"]

    board = pcbnew.LoadBoard(path)
    pad = next(iter(board.FindFootprintByReference("U3").Pads()))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
    pad.SetLayerSet(pcbnew.PAD.PTHMask())
    pad.SetDrillSize(pcbnew.VECTOR2I(_mm(0.4), _mm(0.4)))
    board.Save(path)
    assert unlanded_interface_components(pcbnew.LoadBoard(path), {"SIG"}) == []


def test_canonical_power_alias_does_not_get_signal_width_fanout(tmp_path):
    path = str(tmp_path / "power-alias.kicad_pcb")
    _interface_board(path)
    board = pcbnew.LoadBoard(path)
    rail = pcbnew.NETINFO_ITEM(board, "V5")
    board.Add(rail)
    pad = next(iter(board.FindFootprintByReference("U1").Pads()))
    pad.SetNet(rail)
    specs = signal_escape_specs(
        board, {"power_nets": ["V5"]}, external_net_names={"V5", "SIG"},
    )
    assert {spec.ref for spec in specs} == {"U2", "U3"}


def test_bare_singleton_interface_is_rejected_without_a_local_drc_open(tmp_path):
    import shutil
    from types import SimpleNamespace
    from kicraft.autoplacer.brain.leaf_routing import _audit_interface_landings
    from kicraft.autoplacer.brain.leaf_acceptance import (
        _gate_no_unconnected, acceptance_config_from_dict,
    )
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc

    if shutil.which("kicad-cli") is None:
        pytest.skip("native singleton DRC requires kicad-cli")
    path = tmp_path / "singleton.kicad_pcb"
    _interface_board(str(path))
    board = pcbnew.LoadBoard(str(path))
    for footprint in list(board.GetFootprints()):
        if footprint.GetReferenceAsString() != "U6":
            board.Remove(footprint)
    board.Save(str(path))
    drc = run_kicad_cli_drc(str(path))
    assert drc["ran"] and drc["unconnected"] == 0
    validation = {"drc": drc, "interface_port_names": ["LOCAL"]}
    _audit_interface_landings(
        path, SimpleNamespace(external_net_names=["LOCAL"]), validation, {},
    )
    assert validation["interface_landing_audit"]["unlanded_components"] == [
        {"net": "LOCAL", "pads": ["U6.1"]},
    ]
    assert not _gate_no_unconnected(validation, {}, acceptance_config_from_dict({}))[0]



@pytest.mark.parametrize("face", (pcbnew.F_Cu, pcbnew.B_Cu))
def test_interface_ground_bonds_wait_for_signal_routes_and_finish_connected(tmp_path, face):
    from types import SimpleNamespace

    from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec
    from kicraft.autoplacer.brain.gnd_pour import add_gnd_pour_and_thermal_vias
    from kicraft.autoplacer.brain.leaf_routing import _leaf_gnd_pre_escape_specs
    from kicraft.autoplacer.hardware.ground_reservations import reserve_connected_ground
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc

    path = str(tmp_path / "interface-ground.kicad_pcb")
    board = pcbnew.NewBoard(path)
    for net in ("SIG", "GND", "V5"):
        board.Add(pcbnew.NETINFO_ITEM(board, net))
    for start, end in (((0, 0), (22, 0)), ((22, 0), (22, 22)),
                       ((22, 22), (0, 22)), ((0, 22), (0, 0))):
        edge = pcbnew.PCB_SHAPE()
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetLayer(pcbnew.Edge_Cuts)
        edge.SetStart(pcbnew.VECTOR2I(*map(_mm, start)))
        edge.SetEnd(pcbnew.VECTOR2I(*map(_mm, end)))
        edge.SetWidth(_mm(0.05))
        board.Add(edge)

    def footprint(ref, origin, pads, layer=pcbnew.F_Cu):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetLayer(layer)
        fp.SetPosition(pcbnew.VECTOR2I(*map(_mm, origin)))
        board.Add(fp)
        for number, net, xy in pads:
            pad = pcbnew.PAD(fp)
            pad.SetNumber(number)
            pad.SetSize(pcbnew.VECTOR2I(_mm(0.3), _mm(0.9)))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            layers = pcbnew.LSET()
            for copper_layer in ((pcbnew.F_Cu, pcbnew.F_Paste, pcbnew.F_Mask)
                                 if layer == pcbnew.F_Cu
                                 else (pcbnew.B_Cu, pcbnew.B_Paste, pcbnew.B_Mask)):
                layers.AddLayer(copper_layer)
            pad.SetLayerSet(layers)
            pad.SetPosition(pcbnew.VECTOR2I(*map(_mm, xy)))
            fp.Add(pad)
            pad.SetNet(board.GetNetInfo().GetNetItem(net))
        return fp

    footprint("U1", (10, 8), (("1", "SIG", (10, 10)), ("2", "GND", (10.65, 10))), face)
    # A power-only interface must not defer this ordinary ground bond.
    footprint("U2", (15, 15), (("1", "GND", (15, 16)), ("2", "V5", (16.5, 16))))
    footprint("U3", (6, 15), (("1", "GND", (6, 16)),))
    target = footprint("J1", (18, 11), (("1", "SIG", (18, 11)),))
    pad = next(iter(target.Pads()))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
    pad.SetLayerSet(pcbnew.PAD.PTHMask())
    pad.SetSize(pcbnew.VECTOR2I(_mm(0.9), _mm(0.9)))
    pad.SetDrillSize(pcbnew.VECTOR2I(_mm(0.4), _mm(0.4)))
    board.Save(path)
    cfg = {"power_nets": ["V5"]}
    layer = "F.Cu" if face == pcbnew.F_Cu else "B.Cu"
    initial = add_breakout_stubs(path, [
        BreakoutSpec(ref="U1", pad="1", waypoints=[(10, 11)], layer=layer,
                     via_at_end=True, via_size_mm=0.36, via_drill_mm=0.15),
        BreakoutSpec(ref="U3", pad="1", waypoints=[(6, 17.5)], via_at_end=True),
    ], cfg=cfg)
    assert initial["stubs"] == 2, initial
    board = pcbnew.LoadBoard(path)
    protected_ids = {item.m_Uuid.AsString() for item in board.GetTracks()}
    specs = _leaf_gnd_pre_escape_specs(
        board, SimpleNamespace(external_net_names={"SIG", "GND", "V5"}),
        cfg, {("U3", "1")},
    )
    add_breakout_stubs(path, specs, cfg=cfg)
    board = pcbnew.LoadBoard(path)
    assert protected_ids <= {item.m_Uuid.AsString() for item in board.GetTracks()}
    board.BuildConnectivity()
    ordinary = next(pad for pad in board.FindFootprintByReference("U2").Pads()
                    if pad.GetNumber() == "1")
    pending = list(board.GetConnectivity().GetConnectedTracks(ordinary))
    visited = set()
    ordinary_via = False
    while pending:
        item = pending.pop()
        uid = item.m_Uuid.AsString()
        if uid in visited:
            continue
        visited.add(uid)
        ordinary_via |= item.Type() == pcbnew.PCB_VIA_T
        pending.extend(board.GetConnectivity().GetConnectedTracks(item))
    assert ordinary_via, "power-only interfaces retain their ordinary early bond"
    # The interface's optional plane bond must not cage this actual signal exit.
    result = add_breakout_stubs(path, [
        BreakoutSpec(ref="track", pad="signal-exit", start_xy=(10, 11), net="SIG",
                     waypoints=[(18, 11)],
                     layer="B.Cu" if face == pcbnew.F_Cu else "F.Cu"),
    ], cfg=cfg)
    assert result["stubs"] == 1, result
    add_gnd_pour_and_thermal_vias(path, cfg)
    board = pcbnew.LoadBoard(path)
    board.BuildConnectivity()
    source = next(pad for pad in board.FindFootprintByReference("U1").Pads()
                  if pad.GetNumber() == "1")
    target = next(iter(board.FindFootprintByReference("J1").Pads()))
    pending = [source]
    reached = set()
    connectivity = board.GetConnectivity()
    while pending:
        item = pending.pop()
        uid = item.m_Uuid.AsString()
        if uid in reached:
            continue
        reached.add(uid)
        pending.extend(connectivity.GetConnectedTracks(item))
        pending.extend(connectivity.GetConnectedPads(item))
    assert target.m_Uuid.AsString() in reached
    proof = reserve_connected_ground(
        board, "GND", "test finished interface returns",
        width_mm=fab_floors(cfg)["track_mm"],
        router_clearance_mm=fab_floors(cfg)["clearance_mm"],
    )
    assert proof["pads"] == 3
    drc = run_kicad_cli_drc(path)
    assert drc["unconnected"] == drc["shorts"] == drc["clearance"] == 0, drc


def test_declared_usb_pair_cannot_receive_independent_dogbone_vias(tmp_path):
    path = str(tmp_path / "usb-field.kicad_pcb")
    _dense_pad_field(path)
    contract = {
        "id": "usb", "reference_net": "GND", "segments": [{
            "id": "connector-device", "positive_net": "GPIO0", "negative_net": "GPIO1",
            "positive_from": [{"ref": "U1", "pin": "2"}],
            "positive_to": [{"ref": "J1", "pin": "1"}],
            "negative_from": [{"ref": "U1", "pin": "3"}],
            "negative_to": [{"ref": "J1", "pin": "2"}],
        }],
    }
    board = pcbnew.LoadBoard(path)
    specs = signal_escape_specs(board, {"usb_routing_contracts": [contract]})
    assert {spec.pad for spec in specs} == {"5", "6", "7", "8"}
    add_breakout_stubs(path, specs)
    assert all(track.GetNetname() not in {"GPIO0", "GPIO1"}
               for track in pcbnew.LoadBoard(path).GetTracks())


_RP2040_PORTS = {
    "2": "SPI_MISO", "3": "SPI_CS", "4": "SPI_SCLK", "5": "SPI_MOSI",
    "6": "I2C_SDA", "7": "I2C_SCL",
}


def _rp2040_exit_board(tmp_path):
    import json
    from pathlib import Path

    from kicraft.design.synthesis.kicad_pro import DEFAULT_NETCLASS, DEFAULT_RULES

    library = Path("/usr/share/kicad/footprints/Package_DFN_QFN.pretty")
    item = "QFN-56-1EP_7x7mm_P0.4mm_EP3.2x3.2mm"
    if not (library / f"{item}.kicad_mod").is_file():
        pytest.skip("the native KiCad RP2040 QFN footprint is not installed")
    path = tmp_path / "rp2040-interface.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    for name in (*_RP2040_PORTS.values(), "+3V3"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))
    fp = pcbnew.FootprintLoad(str(library), item)
    fp.SetReference("U4")
    fp.SetOrientationDegrees(-90)
    fp.SetPosition(pcbnew.VECTOR2I(_mm(146.4375), _mm(106.095224)))
    board.Add(fp)
    for pad in fp.Pads():
        net = "+3V3" if pad.GetNumber() == "1" else _RP2040_PORTS.get(pad.GetNumber())
        if net:
            pad.SetNet(board.GetNetInfo().GetNetItem(net))
    board.Save(str(path))
    path.with_suffix(".kicad_pro").write_text(json.dumps({
        "board": {"design_settings": {"rules": DEFAULT_RULES}},
        "net_settings": {"classes": [DEFAULT_NETCLASS]},
    }))
    return path


def test_rp2040_dense_singletons_reserve_concurrent_native_launches(tmp_path):
    from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec, _seg_seg_dist_mm
    from kicraft.autoplacer.brain.signal_escape import unlanded_interface_components

    path = _rp2040_exit_board(tmp_path)
    nets = set(_RP2040_PORTS.values())
    board = pcbnew.LoadBoard(str(path))
    local = signal_escape_specs(board)
    assert len(local) == 6 and all(spec.via_exit is None for spec in local)
    specs = signal_escape_specs(board, external_net_names=nets)
    assert {spec.pad for spec in specs} == set(_RP2040_PORTS)
    stamped = add_breakout_stubs(str(path), specs, signal_escape_owner=True)
    assert stamped["stubs"] == stamped["vias"] == len(nets), stamped

    board = pcbnew.LoadBoard(str(path))
    assert unlanded_interface_components(board, nets) == []
    from types import SimpleNamespace
    from kicraft.autoplacer.brain.leaf_routing import _audit_interface_landings

    validation = {}
    _audit_interface_landings(
        path, SimpleNamespace(external_net_names=nets), validation,
        {"power_nets": {"+3V3", "GND"}},
    )
    assert validation["interface_unlanded_nets"] == [], validation
    branches = {branch["pad"]: branch["items"] for branch in stamped["owned_signal_escapes"]}
    exits = []
    for pad_number, net in _RP2040_PORTS.items():
        items = branches[pad_number]
        via = next(item for item in items if item["kind"] == "via")
        exit_track = next(item for item in items if item.get("layer") == "B.Cu")
        assert exit_track["net"] == net
        assert exit_track["width_mm"] == pytest.approx(0.2)
        assert exit_track["start_mm"] == via["position_mm"]
        exits.append(exit_track)
    for index, first in enumerate(exits):
        for second in exits[index + 1:]:
            assert _seg_seg_dist_mm(
                first["start_mm"], first["end_mm"], second["start_mm"], second["end_mm"],
            ) >= 0.2 + 0.153

    # Parent copper cannot cross another signal's reserved public launch.
    reserved = next(item for item in exits if item["net"] == "SPI_CS")
    start, end = reserved["start_mm"], reserved["end_mm"]
    middle = tuple((a + b) / 2 for a, b in zip(start, end, strict=True))
    crossing = BreakoutSpec(
        ref="crossing", pad="1", net="I2C_SDA", start_xy=middle,
        waypoints=[end], width_mm=0.127, layer="B.Cu",
    )
    assert add_breakout_stubs(str(path), [crossing])["skipped"] == [
        "crossing.1:conflicts_with_stamped_stub",
    ]
    # A native via left connected by itself must no longer pass the audit.
    launch = next(track for track in board.GetTracks() if track.GetLayer() == pcbnew.B_Cu
                  and track.GetNetname() == "SPI_CS" and track.Type() == pcbnew.PCB_TRACE_T)
    board.Remove(launch)
    board.Save(str(path))
    assert unlanded_interface_components(pcbnew.LoadBoard(str(path)), nets) == [
        {"net": "SPI_CS", "pads": ["U4.3"], "reason": "missing_reserved_opposite_face_launch"},
    ]


def test_rp2040_opposite_face_crossing_blocks_entire_launch(tmp_path):
    path = _rp2040_exit_board(tmp_path)
    board = pcbnew.LoadBoard(str(path))
    from kicraft.autoplacer.brain.breakout_stubs import _seg_seg_dist_mm

    specs = signal_escape_specs(board, external_net_names=set(_RP2040_PORTS.values()))
    chip_select = next(spec for spec in specs if spec.pad == "3")
    end_x, end_y = chip_select.via_exit[0]
    start, end = (end_x - 0.25, end_y), (end_x + 0.25, end_y)
    tip = chip_select.waypoints[-1]
    assert _seg_seg_dist_mm(tip, tip, start, end) > 0.36 / 2 + 0.127 / 2 + 0.153
    crossing = pcbnew.PCB_TRACK(board)
    # Block only the opposite-face launch, not the via or source-face dogbone.
    crossing.SetStart(pcbnew.VECTOR2I(_mm(start[0]), _mm(start[1])))
    crossing.SetEnd(pcbnew.VECTOR2I(_mm(end[0]), _mm(end[1])))
    crossing.SetLayer(pcbnew.B_Cu)
    crossing.SetWidth(_mm(0.127))
    crossing.SetNet(board.GetNetInfo().GetNetItem("I2C_SDA"))
    crossing.SetLocked(True)
    board.Add(crossing)
    board.Save(str(path))

    result = add_breakout_stubs(str(path), [chip_select], signal_escape_owner=True)
    assert result["stubs"] == result["vias"] == result["segments"] == 0
    assert result["owned_signal_escapes"] == []
    assert {track.m_Uuid.AsString() for track in pcbnew.LoadBoard(str(path)).GetTracks()} == {
        crossing.m_Uuid.AsString(),
    }


def test_dense_interface_ports_admit_parent_via_envelope(tmp_path):
    """The normal parent via must fit without moving inherited fanout."""
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc

    path = _rp2040_exit_board(tmp_path)
    board = pcbnew.LoadBoard(str(path))
    specs = signal_escape_specs(board, external_net_names=set(_RP2040_PORTS.values()))
    stamped = add_breakout_stubs(str(path), specs, signal_escape_owner=True)
    assert not stamped["skipped"], stamped
    board = pcbnew.LoadBoard(str(path))
    for via in board.GetTracks():
        if isinstance(via, pcbnew.PCB_VIA):
            netclass = via.GetNet().GetNetClassSlow()
            via.SetWidth(netclass.GetViaDiameter())
            via.SetDrill(netclass.GetViaDrill())
    assert pcbnew.SaveBoard(str(path), board)
    drc = run_kicad_cli_drc(str(path))
    assert drc["shorts"] == drc["clearance"] == 0, drc


@pytest.mark.parametrize("side", (pcbnew.F_Cu, pcbnew.B_Cu))
@pytest.mark.parametrize("terminal_kind", ("track", "via", "through", "boundary", "unadvertised"))
def test_source_owned_landing_canonicalization(tmp_path, side, terminal_kind):
    import hashlib
    import shutil

    from kicraft.autoplacer.brain.copper_accounting import attach_signal_escape_ownership
    from kicraft.autoplacer.brain.escape_cleanup import _drc_item_key, _report_drc
    from kicraft.autoplacer.brain.leaf_routing import _update_escape_items
    from kicraft.autoplacer.brain.signal_escape import _escape_track_record, _finish_owned_escape_landings
    from kicraft.autoplacer.fab_profile import stamp_fab_floors_into_pro
    from kicraft.autoplacer.routing_board import import_routed_copper

    if not shutil.which("kicad-cli"):
        pytest.skip("KiCad CLI not installed")
    path = tmp_path / "source.kicad_pcb"
    project = path.with_suffix(".kicad_pro")
    project.write_text("{}")
    board = pcbnew.BOARD()
    net = pcbnew.NETINFO_ITEM(board, "SIG")
    board.Add(net)

    def point(x, y):
        return pcbnew.VECTOR2I(round(x * 1e6), round(y * 1e6))

    def pad(ref, at, layer):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        board.Add(fp)
        item = pcbnew.PAD(fp)
        item.SetNumber("1")
        item.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        item.SetShape(pcbnew.PAD_SHAPE_RECT)
        item.SetSize(point(0.6, 0.6))
        layers = pcbnew.LSET()
        layers.AddLayer(layer)
        item.SetLayerSet(layers)
        item.SetNet(net)
        fp.Add(item)
        item.SetPosition(point(*at))

    def trace(start, end, layer, width, locked=False):
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(point(*start))
        track.SetEnd(point(*end))
        track.SetLayer(layer)
        track.SetWidth(round(width * 1e6))
        track.SetNet(net)
        track.SetLocked(locked)
        board.Add(track)
        return track

    outline = pcbnew.PCB_SHAPE(board)
    outline.SetShape(pcbnew.SHAPE_T_RECT)
    outline.SetStart(point(1, 1))
    outline.SetEnd(point(13, 9))
    outline.SetLayer(pcbnew.Edge_Cuts)
    board.Add(outline)
    launch = (9, 5)
    pad("U1", (10, 5), side)
    predecessor = trace((10, 5), launch, side, 0.127, locked=True)
    if terminal_kind == "track":
        terminal = trace(launch, (8.593499, 5), side, 0.127, locked=True)
        owned = _escape_track_record(board, terminal)
        contact, destination = (8.833573, 5.003088), (8.033573, 5.803088)
        route_side = side
    else:
        terminal = pcbnew.PCB_VIA(board)
        terminal.SetPosition(point(*launch))
        terminal.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        terminal.SetWidth(_mm(0.36))
        terminal.SetDrill(_mm(0.15))
        terminal.SetNet(net)
        terminal.SetLocked(True)
        board.Add(terminal)
        owned = {
            "uuid": terminal.m_Uuid.AsString(), "kind": "via", "net": "SIG",
            "position_mm": launch, "size_mm": 0.36, "drill_mm": 0.15,
        }
        contact, destination = (8.960927, 4.821234), (7.860927, 3.721233)
        route_side = side
        if terminal_kind == "through":
            contact = launch
            route_side = pcbnew.B_Cu if side == pcbnew.F_Cu else pcbnew.F_Cu
    pad("U2", destination, route_side)
    router = trace(contact, destination, route_side, 0.2)
    board.Save(str(path))
    stamp_fab_floors_into_pro(project)
    summary = {"owned_signal_escapes": [{
        "owner": "compiler.signal_escape", "ref": "U1", "pad": "1",
        "items": [_escape_track_record(board, predecessor), owned],
    }]}

    original = path.read_bytes()
    assert _finish_owned_escape_landings(path, (summary,), set(), {}) == {}
    assert path.read_bytes() == original
    assert _finish_owned_escape_landings(
        path, (summary,), {"SIG"}, {"fab_capability": {"min_track_mm": 0.17}},
    ) == {}
    contract = {"owner_ref": "U1", "bypass_ref": "U2", "power_pins": ["1", "1"],
                "max_route_length_mm": 10}
    assert _finish_owned_escape_landings(path, (summary,), {"SIG"}, {"local_bypasses": [contract]}) == {}
    assert path.read_bytes() == original
    before = _report_drc(path, tmp_path / "before.json")
    uid = owned["uuid"]
    if terminal_kind in {"boundary", "unadvertised"}:
        import json
        from kicraft.autoplacer.brain.escape_cleanup import (
            cleanup_signal_escapes, finish_boundary_escape_landings,
        )

        owner = {"owner": "compiler.signal_escape", "ref": "U1", "pad": "1",
                 "branch_id": "U1.1", "branch_item_count": "2"}
        predecessor_id = predecessor.m_Uuid.AsString()
        ownership = tmp_path / "ownership.json"
        sha = hashlib.sha256(original).hexdigest()
        ownership.write_text(json.dumps({
            "version": 1, "stamped_board_sha256": sha, "routed_board_sha256": sha,
            **({"interface_net_names": ["SIG"]} if terminal_kind == "boundary" else {}),
            "boundary_landing_constraints": {},
            "traces": [{
                "start_x": 10, "start_y": 5, "end_x": 9, "end_y": 5,
                "width": 0.127, "layer": "F.Cu" if side == pcbnew.F_Cu else "B.Cu",
                "net_name": "SIG", "native_uuid": predecessor_id,
                "escape_ownership": {**owner, "source_uuid": predecessor_id, "branch_item_index": "0"},
            }],
            "vias": [{
                "x": 9, "y": 5, "size": 0.36, "drill": 0.15, "net_name": "SIG",
                "native_uuid": uid,
                "escape_ownership": {**owner, "source_uuid": uid, "branch_item_index": "1"},
            }],
        }))
        cleanup = cleanup_signal_escapes(path, ownership, tmp_path / "cleanup.json")
        assert cleanup["accepted"] and not cleanup["removed"]
        result = finish_boundary_escape_landings(path, ownership, cleanup)
        if terminal_kind == "unadvertised":
            assert result == {} and path.read_bytes() == original
            return
    else:
        result = _finish_owned_escape_landings(path, (summary,), {"SIG"}, {})
    if terminal_kind == "through":
        assert result == {} and path.read_bytes() == original
        return
    warning_type = "track_dangling" if terminal_kind == "track" else "via_dangling"
    assert any(row["type"] == warning_type and row["items"][0]["uuid"] == uid
               for row in before["violations"])
    assert result["accepted"] and result["native_readback"]
    assert result["board_before_sha256"] == hashlib.sha256(original).hexdigest()
    assert result["board_after_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["retired"] == {
        uid: {key: list(value) if isinstance(value, tuple) else value for key, value in owned.items()},
    }
    corrected = result["replacements"][uid]
    assert tuple(corrected["start_mm"]) == launch and tuple(corrected["end_mm"]) == contact
    assert corrected["width_mm"] == 0.127
    saved = pcbnew.LoadBoard(str(path))
    assert {item.m_Uuid.AsString() for item in saved.GetTracks()} == {
        router.m_Uuid.AsString(), predecessor.m_Uuid.AsString(), corrected["uuid"],
    }
    after = _report_drc(path, tmp_path / "after.json")
    assert not before["unconnected_items"] and not after["unconnected_items"]
    assert set(map(_drc_item_key, after["violations"])) == {
        _drc_item_key(row) for row in before["violations"]
        if row["type"] != warning_type or row["items"][0]["uuid"] != uid
    }
    copper = import_routed_copper(str(path))
    updated = _update_escape_items(summary, set(), result["replacements"])
    assert attach_signal_escape_ownership(copper, updated) == {"bound": 2, "unresolved_source_uuids": []}
