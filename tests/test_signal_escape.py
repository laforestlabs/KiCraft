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


def _dense_pad_field(path, *, pitch_mm=0.4, rotation=0):
    board = pcbnew.NewBoard(path)
    for name in ("+3V3", "GND", "GPIO0", "GPIO1", "GPIO2", "GPIO3", "GPIO4", "GPIO5"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("U1")
    fp.SetPosition(pcbnew.VECTOR2I(_mm(10.0), _mm(10.0)))
    board.Add(fp)
    for index, net in enumerate(
        ("+3V3", "GPIO0", "GPIO1", "GND", "GPIO2", "GPIO3", "GPIO4", "GPIO5"), 1
    ):
        pad = pcbnew.PAD(fp)
        pad.SetNumber(str(index))
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.2), _mm(0.875)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(9.0 + pitch_mm * index), _mm(12.0)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNet(board.GetNetInfo().GetNetItem(net))
        fp.Add(pad)
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


def test_wide_pitch_narrow_pads_do_not_reserve_dogbones(tmp_path):
    path = str(tmp_path / "wide.kicad_pcb")
    _dense_pad_field(path, pitch_mm=0.61)

    assert fine_pitch_signal_escape_specs(pcbnew.LoadBoard(path)) == []
