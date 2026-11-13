"""Unit tests for the C1 signal unconnected repair's DRC-report parsing and
path candidates (the geometric layer needs no pcbnew)."""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.unconnected_repair import (
    _parse_unconnected_edges,
    _path_clear,
    _seg_seg_dist_mm,
)

# Real shapes from run_09 / run_13 of logs/self_eval/20260706T224451Z.
_REPORT = """\
** Found 3 unconnected pads **
[unconnected_items]: Missing connection between items
    Local override; error
    @(123.5788 mm, 114.2167 mm): Pad A6 [USB_D+] of J1 on F.Cu
    @(142.4313 mm, 102.4121 mm): Pad 33 [USB_D+] of U2 on F.Cu
[unconnected_items]: Missing connection between items
    Local override; error
    @(147.5492 mm, 106.3944 mm): Track [BUTTON_GPIO] on F.Cu, length 1.1857 mm
    @(144.7332 mm, 118.3694 mm): Pad AC9 [BUTTON_GPIO] of U1 on F.Cu
[unconnected_items]: Missing connection between items
    Local override; error
    @(145.7332 mm, 118.3695 mm): Track [nRESET] on F.Cu, length 0.2500 mm
    @(147.6699 mm, 132.1643 mm): PTH pad 5 [nRESET] of J1
[shorting_items]: Shorting items
    @(1.0 mm, 1.0 mm): Track [A] on F.Cu
    @(1.0 mm, 1.0 mm): Track [B] on F.Cu
"""


def test_parses_pad_track_and_pth_endpoints():
    edges = _parse_unconnected_edges(_REPORT)
    assert len(edges) == 3  # the shorting_items block is not an edge

    usb = edges[0]
    assert (usb[0].net, usb[0].ref, usb[0].pad) == ("USB_D+", "J1", "A6")
    assert usb[0].layers == {"F.Cu"}
    assert (usb[1].ref, usb[1].pad) == ("U2", "33")

    button = edges[1]
    assert button[0].ref is None and button[0].pad is None  # track endpoint
    assert button[0].xy == (147.5492, 106.3944)
    assert (button[1].ref, button[1].pad) == ("U1", "AC9")

    nreset = edges[2]
    assert (nreset[1].ref, nreset[1].pad) == ("J1", "5")
    assert nreset[1].layers == {"F.Cu", "B.Cu"}  # PTH: no layer suffix



def test_seg_seg_dist_crossing_parallel_and_apart():
    # Proper crossing -> 0.
    assert _seg_seg_dist_mm((0, 0), (10, 0), (5, -5), (5, 5)) == 0.0
    # Parallel, 2mm apart.
    assert _seg_seg_dist_mm((0, 0), (10, 0), (0, 2), (10, 2)) == pytest.approx(2.0)
    # Endpoint-to-endpoint.
    assert _seg_seg_dist_mm((0, 0), (1, 0), (4, 4), (9, 4)) == pytest.approx(5.0)


def test_path_clear_prunes_crossing_and_forgives_endpoints():
    # One foreign F.Cu track crossing the corridor at x=10.
    obstacles = [("F.Cu", "GND", 10.0, -20.0, 10.0, 20.0, 0.25)]
    kw = dict(layer="F.Cu", net="SIG", margin_mm=0.23, src=(0.0, 0.0), tgt=(20.0, 0.0))
    # Straight path crosses it -> pruned.
    assert not _path_clear(obstacles, [(0.0, 0.0), (20.0, 0.0)], **kw)
    # A detour around its end (track spans y +/-20) stays clear.
    assert _path_clear(
        obstacles, [(0.0, 0.0), (5.0, 25.0), (15.0, 25.0), (20.0, 0.0)], **kw
    )
    # Same-net copper is never an obstacle.
    assert _path_clear(
        [("F.Cu", "SIG", 10.0, -20.0, 10.0, 20.0, 0.25)],
        [(0.0, 0.0), (20.0, 0.0)], **kw,
    )
    # Other-layer copper is not an obstacle (no via in the chain model).
    assert _path_clear(
        [("B.Cu", "GND", 10.0, -20.0, 10.0, 20.0, 0.25)],
        [(0.0, 0.0), (20.0, 0.0)], **kw,
    )
    # An obstacle hugging the TARGET pad is forgiven (endpoint carve-out):
    # the tie must be allowed to approach copper at both ends.
    assert _path_clear(
        [("F.Cu", "GND", 19.5, -0.5, 19.5, 0.5, 0.25)],
        [(0.0, 0.0), (20.0, 0.0)], **kw,
    )


def _empty_board_with_outline(pcbnew, path: str, w_mm: float, h_mm: float):
    mm = pcbnew.FromMM
    board = pcbnew.NewBoard(path)
    corners = [(0, 0), (w_mm, 0), (w_mm, h_mm), (0, h_mm), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(mm(x1), mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)
    return board


def test_gap_cap_scales_to_board_diagonal(tmp_path):
    """A fixed 60mm cap skipped every castellated fan-out edge (62-86mm on
    run_10) as gap_over_cap; the default now scales to the board diagonal.
    An explicit signal_repair_max_mm still wins, small boards keep the 60mm
    floor."""
    pcbnew = pytest.importorskip("pcbnew")
    from kicraft.autoplacer.brain.unconnected_repair import _gap_cap_mm

    small = _empty_board_with_outline(
        pcbnew, str(tmp_path / "small.kicad_pcb"), 30.0, 20.0
    )
    assert _gap_cap_mm(small, {}) == pytest.approx(60.0)

    big = _empty_board_with_outline(
        pcbnew, str(tmp_path / "big.kicad_pcb"), 100.0, 70.0
    )
    assert _gap_cap_mm(big, {}) == pytest.approx((100**2 + 70**2) ** 0.5, rel=0.02)

    assert _gap_cap_mm(big, {"signal_repair_max_mm": 45.0}) == 45.0


def test_repair_ties_track_to_track_edge(tmp_path):
    """C1 v2 track-endpoint anchors: an unconnected edge whose BOTH endpoints
    are dangling track stubs (run_10's USB_DN/XIN signature) was skipped
    no_pad_anchor under the pads-only rule; it must now tie from the bare
    track coordinate."""
    import shutil as _shutil

    pcbnew = pytest.importorskip("pcbnew")
    if not _shutil.which("kicad-cli"):
        pytest.skip("kicad-cli unavailable (repair edges come from real DRC)")
    from kicraft.autoplacer.brain.unconnected_repair import (
        repair_unconnected_signals,
    )

    mm = pcbnew.FromMM
    path = str(tmp_path / "tracks.kicad_pcb")
    board = _empty_board_with_outline(pcbnew, path, 30.0, 30.0)
    board.Add(pcbnew.NETINFO_ITEM(board, "SIG"))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    for ref, x in (("J1", 5.0), ("J2", 25.0)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(15.0)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(mm(1.0), mm(1.0)))
        pad.SetPosition(pcbnew.VECTOR2I(mm(x), mm(15.0)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber("1")
        pad.SetNet(net("SIG"))
        fp.Add(pad)

    # Each pad carries a stub toward the middle; the closest pair of items
    # across the two clusters is the two bare TRACK ends at x=12 / x=18, so
    # the DRC edge is track<->track.
    for x1, x2 in ((5.0, 12.0), (25.0, 18.0)):
        t = pcbnew.PCB_TRACK(board)
        t.SetStart(pcbnew.VECTOR2I(mm(x1), mm(15.0)))
        t.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(15.0)))
        t.SetWidth(mm(0.25))
        t.SetLayer(pcbnew.F_Cu)
        t.SetNet(net("SIG"))
        board.Add(t)
    board.Save(path)

    s = repair_unconnected_signals(path, {})

    assert s["edges"] == 1
    assert not any("no_pad_anchor" in sk for sk in s["skipped"]), s
    assert s["tied"] == 1, f"track<->track edge not tied: {s}"


def test_repair_ties_dogleg_around_partial_wall(tmp_path):
    """POSITIVE CONTROL for the candidate pre-filter: an edge that IS
    winnable (straight corridor walled by a foreign track, wide detour open)
    must still get tied. Guards against the pre-filter over-pruning -- on
    genuinely-blocked boards it returns 0 survivors (20260713 probe), and
    this test is the proof that winnable candidates survive the screen and
    reach the stamping guards."""
    import shutil as _shutil

    pcbnew = pytest.importorskip("pcbnew")
    if not _shutil.which("kicad-cli"):
        pytest.skip("kicad-cli unavailable (repair edges come from real DRC)")
    from kicraft.autoplacer.brain.unconnected_repair import (
        repair_unconnected_signals,
    )

    mm = pcbnew.FromMM
    path = str(tmp_path / "wall.kicad_pcb")
    board = pcbnew.NewBoard(path)
    for name in ("SIG", "GND"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    corners = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(mm(x1), mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    for ref, x in (("J1", 5.0), ("J2", 20.0)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(15.0)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(mm(1.0), mm(1.0)))
        pad.SetPosition(pcbnew.VECTOR2I(mm(x), mm(15.0)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber("1")
        pad.SetNet(net("SIG"))
        fp.Add(pad)

    # Foreign wall: a GND track across the straight corridor (y 10..20 at
    # x=12) -- straight and small-offset doglegs collide, the +/-7.5mm
    # dogleg clears it with >2mm to spare.
    wall = pcbnew.PCB_TRACK(board)
    wall.SetStart(pcbnew.VECTOR2I(mm(12.0), mm(10.0)))
    wall.SetEnd(pcbnew.VECTOR2I(mm(12.0), mm(20.0)))
    wall.SetWidth(mm(0.5))
    wall.SetLayer(pcbnew.F_Cu)
    wall.SetNet(net("GND"))
    board.Add(wall)
    board.Save(path)

    s = repair_unconnected_signals(path, {})

    assert s["edges"] == 1
    assert s["tied"] == 1, f"winnable edge not tied: {s}"
    assert s["skipped"] == []
    assert s["pruned"] > 0, "the walled straight/near candidates must be pruned"


def test_repair_reuses_connected_vias_behind_reported_front_track_ends(tmp_path):
    """A front-only wall must be crossed through real existing back access."""
    import shutil

    pcbnew = pytest.importorskip("pcbnew")
    if not shutil.which("kicad-cli"):
        pytest.skip("kicad-cli unavailable")
    from kicraft.autoplacer.brain.unconnected_repair import repair_unconnected_signals
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc

    mm = pcbnew.FromMM
    path = str(tmp_path / "existing-vias.kicad_pcb")
    board = _empty_board_with_outline(pcbnew, path, 20.0, 20.0)
    net = pcbnew.NETINFO_ITEM(board, "SIG")
    board.Add(net)
    for ref, x, end_x, via_x in (("J1", 3, 8, 5), ("J2", 17, 12, 15)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(10)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(mm(1), mm(1)))
        pad.SetPosition(pcbnew.VECTOR2I(mm(x), mm(10)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber("1")
        pad.SetNet(net)
        fp.Add(pad)
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(pcbnew.VECTOR2I(mm(x), mm(10)))
        track.SetEnd(pcbnew.VECTOR2I(mm(end_x), mm(10)))
        track.SetWidth(mm(.25))
        track.SetLayer(pcbnew.F_Cu)
        track.SetNet(net)
        board.Add(track)
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(mm(via_x), mm(10)))
        via.SetWidth(mm(.6))
        via.SetDrill(mm(.3))
        via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        via.SetNet(net)
        board.Add(via)

    wall_net = pcbnew.NETINFO_ITEM(board, "GND")
    board.Add(wall_net)
    wall = pcbnew.PCB_TRACK(board)
    wall.SetLayer(pcbnew.F_Cu)
    wall.SetStart(pcbnew.VECTOR2I(mm(10), mm(1)))
    wall.SetEnd(pcbnew.VECTOR2I(mm(10), mm(19)))
    wall.SetWidth(mm(.5))
    wall.SetNet(wall_net)
    board.Add(wall)
    before_vias = {t.m_Uuid.AsString() for t in board.GetTracks()
                   if t.Type() == pcbnew.PCB_VIA_T}
    board.Save(path)
    initial = run_kicad_cli_drc(path)
    edges = _parse_unconnected_edges(initial["report_text"])
    assert len(edges) == 1
    assert all(e.ref is None and e.layers == {"F.Cu"} for e in edges[0])

    result = repair_unconnected_signals(path, {"signal_repair_max_attempts": 6})

    assert result["tied"] == 1, result
    final = run_kicad_cli_drc(path)
    for count in ("unconnected", "shorts", "clearance", "items_not_allowed",
                  "tracks_crossing", "track_width", "annular_width", "padstack",
                  "copper_edge_clearance"):
        assert final[count] == 0, (count, final)
    routed = pcbnew.LoadBoard(path)
    assert {t.m_Uuid.AsString() for t in routed.GetTracks()
            if t.Type() == pcbnew.PCB_VIA_T} == before_vias
    assert any(t.Type() == pcbnew.PCB_TRACE_T and t.GetLayer() == pcbnew.B_Cu
               for t in routed.GetTracks())


@pytest.mark.parametrize("wall_kind", ["copper", "keepout"])
def test_repair_crosses_alternating_face_walls_with_new_vias(tmp_path, wall_kind):
    """Neither face spans the gap; the legal path needs two internal changes."""
    import shutil

    pcbnew = pytest.importorskip("pcbnew")
    if not shutil.which("kicad-cli"):
        pytest.skip("kicad-cli unavailable")
    from kicraft.autoplacer.brain.unconnected_repair import repair_unconnected_signals
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc

    mm = pcbnew.FromMM
    path = str(tmp_path / "alternating-walls.kicad_pcb")
    board = _empty_board_with_outline(pcbnew, path, 30, 30)
    signal = pcbnew.NETINFO_ITEM(board, "SIG")
    ground = pcbnew.NETINFO_ITEM(board, "GND")
    board.Add(signal)
    board.Add(ground)

    def track(net, layer, start, end):
        item = pcbnew.PCB_TRACK(board)
        item.SetStart(pcbnew.VECTOR2I(*(mm(v) for v in start)))
        item.SetEnd(pcbnew.VECTOR2I(*(mm(v) for v in end)))
        item.SetWidth(mm(.25))
        item.SetLayer(layer)
        item.SetNet(net)
        board.Add(item)

    def via(net, x, y):
        item = pcbnew.PCB_VIA(board)
        item.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))
        item.SetWidth(mm(.6))
        item.SetDrill(mm(.3))
        item.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        item.SetNet(net)
        board.Add(item)

    for ref, x, via_x in (("J1", 2, 4), ("J2", 22, 20)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(10)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(mm(1), mm(1)))
        pad.SetPosition(pcbnew.VECTOR2I(mm(x), mm(10)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber("1")
        pad.SetNet(signal)
        fp.Add(pad)
        track(signal, pcbnew.F_Cu, (x, 10), (via_x, 10))
        via(signal, via_x, 10)
    for x, layer in ((8, pcbnew.B_Cu), (12, pcbnew.F_Cu), (16, pcbnew.B_Cu)):
        if wall_kind == "copper":
            track(ground, layer, (x, .9), (x, 29.1))
        else:
            zone = pcbnew.ZONE(board)
            zone.SetIsRuleArea(True)
            zone.SetLayer(layer)
            zone.SetDoNotAllowTracks(True)
            zone.SetDoNotAllowVias(True)
            zone.Outline().NewOutline()
            for px, py in ((x - .125, .9), (x + .125, .9),
                           (x + .125, 29.1), (x - .125, 29.1)):
                zone.Outline().Append(mm(px), mm(py))
            board.Add(zone)
    if wall_kind == "copper":
        track(ground, pcbnew.F_Cu, (8, .9), (16, .9))
        via(ground, 8, .9)
        via(ground, 16, .9)
    original = {item.m_Uuid.AsString() for item in board.GetTracks()}
    board.Save(path)
    initial = run_kicad_cli_drc(path)
    assert initial["unconnected"] == 1
    assert initial["shorts"] == initial["clearance"] == 0
    assert initial["copper_edge_clearance"] == 0

    result = repair_unconnected_signals(path)

    assert result["tied"] == 1, result
    final = run_kicad_cli_drc(path)
    for key in ("unconnected", "shorts", "clearance", "items_not_allowed",
                "tracks_crossing", "track_width", "annular_width", "padstack",
                "copper_edge_clearance"):
        assert final[key] == 0, (key, final)
    routed = pcbnew.LoadBoard(path)
    assert original <= {item.m_Uuid.AsString() for item in routed.GetTracks()}


@pytest.mark.parametrize("pad_attribute", ["SMD", "CONN"])
def test_mesh_repair_stamps_outside_same_net_pad_mask(tmp_path, pad_attribute):
    """A usable via window must exclude the land's actual mask opening."""
    import shutil

    pcbnew = pytest.importorskip("pcbnew")
    if not shutil.which("kicad-cli"):
        pytest.skip("kicad-cli unavailable")
    from kicraft.autoplacer.brain.breakout_stubs import add_breakout_stubs
    from kicraft.autoplacer.brain.signal_mesh import _mesh_repair_specs
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc

    mm = pcbnew.FromMM
    path = str(tmp_path / "mask-window.kicad_pcb")
    board = _empty_board_with_outline(pcbnew, path, 10, 10)
    net = pcbnew.NETINFO_ITEM(board, "SIG")
    board.Add(net)

    for ref, x, layer in (("J1", 5, pcbnew.F_Cu), ("J2", 8, pcbnew.B_Cu)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(5)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetNumber("1")
        pad.SetShape(pcbnew.PAD_SHAPE_RECT)
        pad.SetSize(pcbnew.VECTOR2I(mm(.2 if ref == "J1" else 1), mm(.2 if ref == "J1" else 1)))
        pad.SetPosition(pcbnew.VECTOR2I(mm(x), mm(5)))
        pad.SetAttribute(getattr(pcbnew, "PAD_ATTRIB_" + pad_attribute))
        layers = pcbnew.LSET()
        layers.AddLayer(layer)
        layers.AddLayer(pcbnew.F_Mask if layer == pcbnew.F_Cu else pcbnew.B_Mask)
        pad.SetLayerSet(layers)
        if ref == "J1":
            pad.SetLocalSolderMaskMargin(mm(.2))
            land = pad
        pad.SetNet(net)
        fp.Add(pad)

    # The four via-only rule areas leave a small square window.  A via at
    # (4.4, 5) clears the land's copper but intersects its expanded mask;
    # narrow legal strips still exist between that opening and the rule areas.
    for x1, y1, x2, y2 in (
        (0, 0, 3.8, 10), (6.2, 0, 10, 10),
        (3.8, 0, 6.2, 3.8), (3.8, 6.2, 6.2, 10),
    ):
        zone = pcbnew.ZONE(board)
        zone.SetIsRuleArea(True)
        zone.SetDoNotAllowVias(True)
        zone.SetDoNotAllowTracks(False)
        zone.SetDoNotAllowPads(False)
        layers = pcbnew.LSET()
        layers.AddLayer(pcbnew.F_Cu)
        layers.AddLayer(pcbnew.B_Cu)
        zone.SetLayerSet(layers)
        zone.Outline().NewOutline()
        for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
            zone.Outline().Append(mm(x), mm(y))
        board.Add(zone)
    probe = pcbnew.VECTOR2I(mm(4.4), mm(5))
    assert not land.HitTest(probe, mm(.4))
    assert land.HitTest(probe, mm(.6))
    board.Save(path)

    cfg = {"escape_via_size_mm": .8, "escape_via_drill_mm": .4,
           "fab_capability": {"min_track_mm": .25}}
    specs = _mesh_repair_specs(
        board, "SIG", [((5, 5), frozenset({"F.Cu"}))],
        [((8, 5), frozenset({"B.Cu"}))], cfg,
    )
    assert len(specs) == 2 and sum(spec.via_at_end for spec in specs) == 1
    stamped = add_breakout_stubs(path, specs, cfg=cfg)
    assert stamped["stubs"] == 2 and stamped["vias"] == 1, stamped

    routed = pcbnew.LoadBoard(path)
    via = next(item for item in routed.GetTracks() if item.GetClass() == "PCB_VIA")
    final_land = next(p for fp in routed.GetFootprints() if fp.GetReference() == "J1"
                      for p in fp.Pads())
    assert not final_land.HitTest(via.GetPosition(), mm(.6))
    drc = run_kicad_cli_drc(path)
    for key in ("unconnected", "shorts", "clearance", "items_not_allowed",
                "tracks_crossing", "track_width", "annular_width", "padstack",
                "copper_edge_clearance"):
        assert drc[key] == 0, (key, drc)


def test_power_mesh_detours_foreign_front_slot_hole_halo(tmp_path):
    """A back-face full-width tie must also clear a front land's drilled slot."""
    import shutil
    from dataclasses import replace

    pcbnew = pytest.importorskip("pcbnew")
    if not shutil.which("kicad-cli"):
        pytest.skip("native DRC is required")
    from kicraft.autoplacer.brain.breakout_stubs import add_breakout_stubs
    from kicraft.autoplacer.brain.signal_mesh import _mesh_repair_specs
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc

    mm = pcbnew.FromMM
    path = tmp_path / "slot-halo.kicad_pcb"
    direct_path = tmp_path / "slot-direct.kicad_pcb"
    board = _empty_board_with_outline(pcbnew, str(path), 10, 10)
    power = pcbnew.NETINFO_ITEM(board, "RAIL_A")
    ground = pcbnew.NETINFO_ITEM(board, "GND")
    board.Add(power)
    board.Add(ground)
    for ref, x in (("J1", 2), ("J2", 8)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(6.6)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetNumber("1")
        pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
        pad.SetSize(pcbnew.VECTOR2I(mm(1), mm(1)))
        pad.SetPosition(fp.GetPosition())
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        layers = pcbnew.LSET()
        layers.AddLayer(pcbnew.B_Cu)
        layers.AddLayer(pcbnew.B_Mask)
        pad.SetLayerSet(layers)
        pad.SetNet(power)
        fp.Add(pad)

    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("J3")
    fp.SetPosition(pcbnew.VECTOR2I(mm(5), mm(5)))
    board.Add(fp)
    pad = pcbnew.PAD(fp)
    pad.SetNumber("1")
    pad.SetShape(pcbnew.PAD_SHAPE_OVAL)
    pad.SetSize(pcbnew.VECTOR2I(mm(1.32), mm(2.12)))
    pad.SetDrillShape(pcbnew.PAD_DRILL_SHAPE_OBLONG)
    pad.SetDrillSize(pcbnew.VECTOR2I(mm(.8), mm(1.6)))
    pad.SetPosition(fp.GetPosition())
    pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH)
    layers = pcbnew.LSET()
    layers.AddLayer(pcbnew.F_Cu)
    layers.AddLayer(pcbnew.F_Mask)
    pad.SetLayerSet(layers)
    pad.SetNet(ground)
    fp.Add(pad)
    board.Save(str(path))
    board.Save(str(direct_path))

    cfg = {
        "fab_capability": {"min_track_mm": .5, "min_clearance_mm": .17},
        "hole_clearance_min_mm": .25,
        "escape_via_size_mm": .8,
        "escape_via_drill_mm": .4,
    }
    specs = _mesh_repair_specs(
        board, "RAIL_A", [((2, 6.6), frozenset({"B.Cu"}))],
        [((8, 6.6), frozenset({"B.Cu"}))], cfg,
    )
    assert len(specs) == 1 and specs[0].layer == "B.Cu"
    assert not specs[0].via_at_end

    # The straight line clears the old pair-clearance bbox and drill circle,
    # but not the authoritative pad-overhang guard. Exercise the real
    # stamper on a separate copy so this control cannot change the test board.
    direct = replace(specs[0], waypoints=[(8, 6.6)])
    rejected = add_breakout_stubs(str(direct_path), [direct], cfg=cfg)
    assert rejected["stubs"] == 0, rejected
    assert any("waypoint_crosses_pad" in reason for reason in rejected["skipped"]), rejected

    stamped = add_breakout_stubs(str(path), specs, cfg=cfg)
    assert stamped["stubs"] == 1 and not stamped["skipped"], stamped
    routed = pcbnew.LoadBoard(str(path))
    traces = list(routed.GetTracks())
    assert traces and all(track.GetNetname() == "RAIL_A"
                          and track.GetLayer() == pcbnew.B_Cu
                          and track.GetWidth() == mm(.5) for track in traces)
    drc = run_kicad_cli_drc(str(path))
    for key in ("unconnected", "shorts", "clearance", "items_not_allowed",
                "tracks_crossing", "track_width", "copper_edge_clearance"):
        assert drc[key] == 0, (key, drc)


def test_power_mesh_uses_sibling_class_geometry_and_preserves_copper(tmp_path):
    """An opposite-face power gap requires a full-class via, not signal floors."""
    import copy
    import json
    import os
    import shutil
    from pathlib import Path

    pcbnew = pytest.importorskip("pcbnew")
    router_value = os.environ.get("KICRAFT_KICAD_ROUTING_TOOLS_PATH")
    python_value = os.environ.get("KICRAFT_KICAD_ROUTING_TOOLS_PYTHON")
    if not shutil.which("kicad-cli") or not router_value or not python_value:
        pytest.skip("native DRC and explicitly configured pinned routing runtime are required")
    router = Path(router_value)
    router_python = Path(python_value)
    from kicraft.autoplacer.brain.unconnected_repair import (
        repair_unconnected_power, repair_unconnected_signals,
    )
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc
    from kicraft.design.synthesis.kicad_pro import DEFAULT_NETCLASS, POWER_NETCLASS

    mm = pcbnew.FromMM
    path = tmp_path / "power-mesh.kicad_pcb"
    board = _empty_board_with_outline(pcbnew, str(path), 10, 10)
    power = pcbnew.NETINFO_ITEM(board, "RAIL_A")
    ground = pcbnew.NETINFO_ITEM(board, "GND")
    board.Add(power)
    board.Add(ground)
    for ref, x, layer in (("J1", 3, pcbnew.F_Cu), ("J2", 7, pcbnew.B_Cu)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(5)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetNumber("1")
        pad.SetSize(pcbnew.VECTOR2I(mm(1.2), mm(1.2)))
        pad.SetPosition(pcbnew.VECTOR2I(mm(x), mm(5)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        layers = pcbnew.LSET()
        layers.AddLayer(layer)
        layers.AddLayer(pcbnew.F_Mask if layer == pcbnew.F_Cu else pcbnew.B_Mask)
        pad.SetLayerSet(layers)
        pad.SetNet(power)
        fp.Add(pad)
    # Existing unrelated ground copper is part of the input, not a repair target.
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(pcbnew.VECTOR2I(mm(2), mm(2)))
    track.SetEnd(pcbnew.VECTOR2I(mm(8), mm(2)))
    track.SetWidth(mm(.5))
    track.SetLayer(pcbnew.F_Cu)
    track.SetNet(ground)
    board.Add(track)
    original = {item.m_Uuid.AsString() for item in board.GetTracks()}
    board.Save(str(path))
    pro = path.with_suffix(".kicad_pro")
    pro.write_text(json.dumps({
        "board": {"design_settings": {"meta": {"version": 2}}},
        "meta": {"filename": pro.name, "version": 1},
        "net_settings": {
            "classes": [dict(DEFAULT_NETCLASS), dict(
                POWER_NETCLASS, track_width=.5, via_diameter=.8, via_drill=.4,
            )],
            "meta": {"version": 3},
            "net_colors": None,
            "netclass_assignments": None,
            "netclass_patterns": [{"netclass": "Power", "pattern": "RAIL_A"}],
        },
    }))
    cfg = {
        "power_nets": {"RAIL_A", "GND"},
        "signal_repair_extra_nets": ["RAIL_A"],
        "kicad_routing_tools_path": str(router),
        "kicad_routing_tools_python": str(router_python),
        "fab_capability": {"min_track_mm": .17, "min_clearance_mm": .17},
        "escape_via_size_mm": .36,
        "escape_via_drill_mm": .15,
    }
    original_cfg = copy.deepcopy(cfg)
    original_pro = pro.read_bytes()
    initial = run_kicad_cli_drc(str(path))
    assert initial["unconnected"] == 1, initial
    untouched = path.read_bytes()
    signal = repair_unconnected_signals(str(path), cfg)
    assert signal["tied"] == 0 and path.read_bytes() == untouched, signal

    result = repair_unconnected_power(str(path), cfg)

    assert result["edges"] == result["tied"] == 1, result
    assert cfg == original_cfg and pro.read_bytes() == original_pro
    routed = pcbnew.LoadBoard(str(path))
    items = list(routed.GetTracks())
    assert original <= {item.m_Uuid.AsString() for item in items}
    ground_after = next(item for item in items if item.m_Uuid.AsString() in original)
    assert ground_after.GetNetname() == "GND"
    assert ground_after.GetStart() == pcbnew.VECTOR2I(mm(2), mm(2))
    assert ground_after.GetEnd() == pcbnew.VECTOR2I(mm(8), mm(2))
    assert ground_after.GetWidth() == mm(.5) and ground_after.GetLayer() == pcbnew.F_Cu
    added = [item for item in items if item.m_Uuid.AsString() not in original]
    assert added and {item.GetNetname() for item in added} == {"RAIL_A"}
    vias = [item for item in added if item.Type() == pcbnew.PCB_VIA_T]
    traces = [item for item in added if item.Type() == pcbnew.PCB_TRACE_T]
    assert vias and traces
    assert all(item.GetWidth() >= mm(.5) for item in traces)
    assert all(item.GetWidth(pcbnew.F_Cu) >= mm(.8) and item.GetDrillValue() >= mm(.4)
               for item in vias)
    final = run_kicad_cli_drc(str(path))
    for key in ("unconnected", "shorts", "clearance", "items_not_allowed",
                "tracks_crossing", "track_width", "annular_width", "padstack",
                "copper_edge_clearance"):
        assert final[key] == 0, (key, final)




def test_signal_wrapper_restores_board_when_redrc_times_out(tmp_path, monkeypatch):
    from kicraft.autoplacer import routing_board
    from kicraft.autoplacer.brain.unconnected_repair_pass import attempt_unconnected_repair

    board = tmp_path / "board.kicad_pcb"
    board.write_text("original")
    monkeypatch.setattr(routing_board, "run_pcbnew_script", lambda _: board.write_text("modified"))
    monkeypatch.setattr(
        routing_board, "validate_routed_board",
        lambda *a, **k: {"drc": {"unconnected": 0, "shorts": 0, "timed_out": True}},
    )
    result = attempt_unconnected_repair(
        board, {"signal_unconnected_repair_enabled": True,
                "power_strand_repair_enabled": False}, {"drc": {"unconnected": 1, "shorts": 0}}
    )
    assert board.read_text() == "original"
    assert result["unconnected_repair"]["reverted"]
    assert not result["unconnected_repair"]["kept"]


def test_signal_repair_subprocess_stamps_two_ounce_width(tmp_path):
    import shutil
    from kicraft.autoplacer.brain.unconnected_repair_pass import attempt_unconnected_repair

    pcbnew = pytest.importorskip("pcbnew")
    if not shutil.which("kicad-cli"):
        pytest.skip("native DRC is required to locate the real unconnected pads")
    mm = pcbnew.FromMM
    path = str(tmp_path / "repair.kicad_pcb")
    board = _empty_board_with_outline(pcbnew, path, 30.0, 30.0)
    net = pcbnew.NETINFO_ITEM(board, "SIG")
    board.Add(net)
    for ref, x in (("J1", 5.0), ("J2", 20.0)):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(15.0)))
        board.Add(fp)
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(mm(1.0), mm(1.0)))
        pad.SetPosition(pcbnew.VECTOR2I(mm(x), mm(15.0)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber("1")
        pad.SetNet(net)
        fp.Add(pad)
    board.Save(path)
    cfg = {"thermal_requirements": [{"outer_copper_oz": 2.0}],
           "fab_capability": {"min_track_mm": 0.1, "min_clearance_mm": 0.1},
           "signal_unconnected_repair_enabled": True,
           "power_strand_repair_enabled": False}
    result = attempt_unconnected_repair(
        path, cfg, {"drc": {"unconnected": 1, "shorts": 0}}
    )
    assert result["unconnected_repair"]["kept"], result
    copper = [t for t in pcbnew.LoadBoard(path).GetTracks() if not isinstance(t, pcbnew.PCB_VIA)]
    assert copper and all(pcbnew.ToMM(t.GetWidth()) >= 0.17 for t in copper)


def test_signal_repair_leaves_canonical_power_open_for_final_gate(tmp_path):
    """A real signal tie cannot turn a separate power open into signal copper."""
    import shutil

    pcbnew = pytest.importorskip("pcbnew")
    if not shutil.which("kicad-cli"):
        pytest.skip("native DRC is required")
    from kicraft.autoplacer.brain.unconnected_repair_pass import attempt_unconnected_repair
    from kicraft.autoplacer.routing_board import validate_routed_board
    from kicraft.design.cli_app import _verify_routed_board

    mm = pcbnew.FromMM
    path = tmp_path / "mixed.kicad_pcb"
    board = _empty_board_with_outline(pcbnew, str(path), 30.0, 30.0)
    for name, y in (("SIG", 10.0), ("RAIL_A", 20.0)):
        net = pcbnew.NETINFO_ITEM(board, name)
        board.Add(net)
        for index, x in enumerate((5.0, 20.0)):
            fp = pcbnew.FOOTPRINT(board)
            fp.SetReference(f"J{1 + index + (2 if name == 'RAIL_A' else 0)}")
            fp.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))
            board.Add(fp)
            pad = pcbnew.PAD(fp)
            pad.SetSize(pcbnew.VECTOR2I(mm(1.0), mm(1.0)))
            pad.SetPosition(pcbnew.VECTOR2I(mm(x), mm(y)))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNumber("1")
            pad.SetNet(net)
            fp.Add(pad)
    board.Save(str(path))

    # A set mirrors the loaded generated configuration; even an explicit
    # extra-net request must not override canonical power classification.
    cfg = {
        "signal_unconnected_repair_enabled": True,
        "power_strand_repair_enabled": False,
        "power_nets": {"RAIL_A"},
        "signal_repair_extra_nets": ["RAIL_A"],
        "signal_repair_escape_mm": [0.0],
        "thermal_requirements": [{"outer_copper_oz": 2.0}],
        "fab_capability": {"min_track_mm": 0.1, "min_clearance_mm": 0.1},
    }
    before = validate_routed_board(str(path), cfg=cfg)
    assert before["drc"]["unconnected"] == 2
    result = attempt_unconnected_repair(path, cfg, before)
    assert result["unconnected_repair"]["kept"], result
    assert result["drc"]["unconnected"] == 1
    assert result["drc"]["unconnected_nets"] == ["RAIL_A"]
    tracks = list(pcbnew.LoadBoard(str(path)).GetTracks())
    assert tracks and {track.GetNetname() for track in tracks} == {"SIG"}
    gate = _verify_routed_board(path)
    assert not gate["ok"] and not gate["fab_acceptable"]
    assert gate["unconnected"] == 1
    assert gate["unconnected_nets"] == ["RAIL_A"]
