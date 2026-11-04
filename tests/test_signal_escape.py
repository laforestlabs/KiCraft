"""Regression coverage for dense fine-pitch signal pre-escape planning."""
from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.breakout_stubs import (  # noqa: E402
    STAMP_CLEARANCE_GUARD_MM,
    add_breakout_stubs,
)
from kicraft.autoplacer.brain.signal_escape import fine_pitch_signal_escape_specs  # noqa: E402
from kicraft.autoplacer.fab_profile import fab_floors, fanout_via  # noqa: E402


_mm = pcbnew.FromMM


def _dense_pad_field(path, *, pitch_mm=0.4, rotation=0, face="front"):
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
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.2), _mm(0.875)))
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

    specs = fine_pitch_signal_escape_specs(pcbnew.LoadBoard(path))

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
    specs = fine_pitch_signal_escape_specs(board)
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
    spec = next(s for s in fine_pitch_signal_escape_specs(board) if s.pad == "2")
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

def test_wide_pitch_narrow_pads_do_not_reserve_dogbones(tmp_path):
    path = str(tmp_path / "wide.kicad_pcb")
    _dense_pad_field(path, pitch_mm=0.61)

    assert fine_pitch_signal_escape_specs(pcbnew.LoadBoard(path)) == []
