"""GND edge-spine repair (kicraft.autoplacer.brain.gnd_pour.stamp_gnd_edge_spine).

A bank of connectors along one board edge is the canonical GND-strand shape
(KC-YXQ4EC: sixteen 1x3 servo headers -> 13 disconnected B.Cu pour islands):
each header's signal fan fences the pour between neighbours, so no straight
tie can reconnect the plane. The spine chains the bank's PTH GND pads through
the pads-to-edge corridor, pad to pad, one guarded link at a time.
"""

from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.gnd_pour import (  # noqa: E402
    _collect_net_clusters,
    stamp_gnd_edge_spine,
)

_mm = pcbnew.FromMM


def _outline(board, w=40, h=40):
    corners = [(0, 0), (w, 0), (w, h), (0, h), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)


def _pth_connector(board, ref, x, y, net):
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference(ref)
    board.Add(fp)
    pad = pcbnew.PAD(fp)
    pad.SetSize(pcbnew.VECTOR2I(_mm(1.7), _mm(1.7)))
    pad.SetDrillSize(pcbnew.VECTOR2I(_mm(1.0), _mm(1.0)))
    pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
    pad.SetLayerSet(pcbnew.PAD.PTHMask())
    pad.SetNumber("1")
    pad.SetNet(net)
    fp.Add(pad)
    return fp


def _bank_board(path, n=3, edge_x=38.0):
    board = pcbnew.NewBoard(path)
    board.Add(pcbnew.NETINFO_ITEM(board, "GND"))
    gnd = board.GetNetInfo().GetNetItem("GND")
    _outline(board)
    for i in range(n):
        _pth_connector(board, f"J{i + 1}", edge_x, 10.0 + 10.0 * i, gnd)
    board.Save(path)
    return path


def test_spine_chains_edge_bank(tmp_path):
    path = str(tmp_path / "bank.kicad_pcb")
    _bank_board(path, n=3)
    zones = {f"J{i}": {"edge": "right"} for i in (1, 2, 3)}
    s = stamp_gnd_edge_spine(path, {"component_zones": zones})
    # 3 pads chain with 2 pad-to-pad links; every link landed.
    assert s["stubs"] == 2
    assert s["edges"]["right"]["skipped"] == []
    board = pcbnew.LoadBoard(path)
    gnd_code = board.GetNetInfo().GetNetItem("GND").GetNetCode()
    spine_tracks = [
        t for t in board.GetTracks()
        if t.GetNetCode() == gnd_code and pcbnew.ToMM(t.GetStart().x) > 38.0
    ]
    assert spine_tracks, "spine rail must run outboard of the pad column"
    # The chain must make the three pads ONE electrical cluster.
    clusters, _ = _collect_net_clusters(board, "GND")
    assert len(clusters) == 1


def test_spine_needs_two_connectors(tmp_path):
    path = str(tmp_path / "single.kicad_pcb")
    _bank_board(path, n=1)
    s = stamp_gnd_edge_spine(path, {"component_zones": {"J1": {"edge": "right"}}})
    assert s["stubs"] == 0


def test_spine_ignores_unzoned_and_far_connectors(tmp_path):
    path = str(tmp_path / "far.kicad_pcb")
    board = pcbnew.NewBoard(path)
    board.Add(pcbnew.NETINFO_ITEM(board, "GND"))
    gnd = board.GetNetInfo().GetNetItem("GND")
    _outline(board)
    # Two zoned connectors but 20 mm inboard of their claimed edge: the
    # max-inset sanity guard must refuse to run a rail across the board.
    for i in (1, 2):
        _pth_connector(board, f"J{i}", 18.0, 10.0 * i, gnd)
    board.Save(path)
    s = stamp_gnd_edge_spine(
        path, {"component_zones": {"J1": {"edge": "right"}, "J2": {"edge": "right"}}}
    )
    assert s["stubs"] == 0
    assert s["edges"]["right"]["skipped"] == "pads_not_near_edge"


def test_collect_net_clusters_sees_mid_track_fill_contact(tmp_path):
    """A fill island touching a track mid-run (not at an endpoint) is the
    same electrical cluster: endpoint-only probing split it into phantoms
    and made every island repair mis-target."""
    path = str(tmp_path / "midspan.kicad_pcb")
    board = pcbnew.NewBoard(path)
    board.Add(pcbnew.NETINFO_ITEM(board, "GND"))
    gnd = board.GetNetInfo().GetNetItem("GND")
    _outline(board)
    # A GND fill strip in the middle; a GND track crossing it whose
    # endpoints both lie OUTSIDE the strip.
    zone = pcbnew.ZONE(board)
    zone.SetNet(gnd)
    zone.SetLayer(pcbnew.F_Cu)
    pts = [(16, 5), (24, 5), (24, 35), (16, 35)]
    chain = pcbnew.SHAPE_LINE_CHAIN()
    for x, y in pts:
        chain.Append(_mm(x), _mm(y))
    chain.SetClosed(True)
    zone.Outline().AddOutline(chain)
    board.Add(zone)
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(pcbnew.VECTOR2I(_mm(4), _mm(20)))
    track.SetEnd(pcbnew.VECTOR2I(_mm(36), _mm(20)))
    track.SetWidth(_mm(0.25))
    track.SetLayer(pcbnew.F_Cu)
    track.SetNet(gnd)
    board.Add(track)
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(path)
    clusters, _ = _collect_net_clusters(board, "GND")
    assert len(clusters) == 1
