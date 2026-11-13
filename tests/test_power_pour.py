"""Tests for the power-plane pour (kicraft.autoplacer.brain.gnd_pour).

A power rail (e.g. VBUS) is poured as a solid plane on the layer opposite GND so
paired connector power pads the autorouter can't thread together connect through
copper. Solid (not thermal) pad connection is required -- thermal-relief spokes
need a gap wider than a dense connector's pad pitch and never form.
"""

from __future__ import annotations

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.gnd_pour import (  # noqa: E402
    _board_edge_geometry,
    _detect_power_nets,
    _via_clears_board_edge,
    pour_power_planes,
)

_mm = pcbnew.FromMM


def _board(path, pad_nets):
    """Build a board: rectangular outline + one footprint whose pads carry the
    given (number -> net name) mapping. Returns the saved+reloaded board path."""
    board = pcbnew.NewBoard(path)
    for name in {n for n in pad_nets.values() if n}:
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    # 20x20 mm Edge.Cuts rectangle so GetBoardEdgesBoundingBox is non-empty.
    corners = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("J1")
    board.Add(fp)
    x = 5
    for num, netname in pad_nets.items():
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(_mm(1), _mm(1)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(10)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber(num)
        if netname:
            pad.SetNet(net(netname))
        fp.Add(pad)
        x += 3
    board.Save(path)
    return path


def test_gnd_via_keeps_its_copper_clear_of_board_edges_and_cutouts(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "GND"})
    board = pcbnew.LoadBoard(path)
    # Opposite winding from the outer outline makes this an internal Edge.Cuts
    # cutout. Its bounding box still contains the legal copper surrounding it.
    cutout = [(8.0, 8.0), (8.0, 12.0), (12.0, 12.0), (12.0, 8.0), (8.0, 8.0)]
    for (x1, y1), (x2, y2) in zip(cutout, cutout[1:]):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        edge.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        edge.SetLayer(pcbnew.Edge_Cuts)
        board.Add(edge)
    board.Save(path)

    edge_geometry = _board_edge_geometry(pcbnew.LoadBoard(path))
    assert edge_geometry is not None
    via_radius_iu = _mm(0.3)
    assert not _via_clears_board_edge(
        _mm(0.2), _mm(10.0), edge_geometry=edge_geometry, via_radius_iu=via_radius_iu
    )
    # (7.7, 10) is inside the outer bbox and board, but its via barrel reaches
    # the internal cutout boundary.
    assert not _via_clears_board_edge(
        _mm(7.7), _mm(10.0), edge_geometry=edge_geometry, via_radius_iu=via_radius_iu
    )
    assert _via_clears_board_edge(
        _mm(2.0), _mm(10.0), edge_geometry=edge_geometry, via_radius_iu=via_radius_iu
    )




def test_detect_power_nets_ranks_by_pad_count_and_excludes_gnd(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "VBUS", "2": "VBUS", "3": "GND", "4": "USB_D+"})
    board = pcbnew.LoadBoard(path)
    # GND excluded; VBUS (2 pads) ranks above any tie; max_nets default 1.
    assert _detect_power_nets(board, {"gnd_zone_net": "GND"}) == ["VBUS"]


def test_detect_power_nets_explicit_override(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "VBUS", "2": "+3V3"})
    board = pcbnew.LoadBoard(path)
    cfg = {"gnd_zone_net": "GND", "power_plane_nets": ["+3V3"]}
    assert _detect_power_nets(board, cfg) == ["+3V3"]


def test_detect_power_nets_max_nets(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "VBUS", "2": "VBUS", "3": "+3V3"})
    board = pcbnew.LoadBoard(path)
    cfg = {"gnd_zone_net": "GND", "power_plane_max_nets": 2}
    assert set(_detect_power_nets(board, cfg)) == {"VBUS", "+3V3"}


def test_pour_creates_solid_power_zone(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "VBUS", "2": "VBUS", "3": "GND"})
    res = pour_power_planes(path, {"gnd_zone_net": "GND"}, layers=("F.Cu",))
    assert res["nets"] == ["VBUS"]
    assert res["zones"] == 1

    board = pcbnew.LoadBoard(path)
    vbus_zones = [
        z
        for z in board.Zones()
        if z.GetNetname() == "VBUS" and z.GetLayer() == pcbnew.F_Cu
    ]
    assert len(vbus_zones) == 1
    z = vbus_zones[0]
    assert z.GetPadConnection() == pcbnew.ZONE_CONNECTION_FULL
    assert z.GetAssignedPriority() == 1
    # No GND zone is created by the power pour (that is the GND pour's job).
    assert not [zz for zz in board.Zones() if zz.GetNetname() == "GND"]


def test_pour_disabled_is_noop(tmp_path):
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "VBUS", "2": "VBUS"})
    res = pour_power_planes(path, {"power_plane_enabled": False})
    assert res == {"nets": [], "zones": 0}



def test_power_pour_preserves_native_ground_witness_and_rebuilds_plan(tmp_path):
    import hashlib
    import shutil

    from kicraft.autoplacer.brain.gnd_pour import pour_gnd_planes
    from kicraft.autoplacer.hardware.local_contract import _ground_regions

    path = str(tmp_path / "routed.kicad_pcb")
    _board(path, {"1": "GND", "2": "GND", "3": "VBUS", "4": "VBUS"})
    cfg = {"power_plane_nets": ["VBUS"], "zone_clearance_mm": 0.3}
    pour_gnd_planes(path, cfg, layers=("F.Cu",))

    def plan_at(y, name):
        staged = tmp_path / f"{name}.kicad_pcb"
        shutil.copyfile(path, staged)
        board = pcbnew.LoadBoard(str(staged))
        for layer, tag, x1, y1, x2, y2 in (
            (pcbnew.F_Cu, name, 4.8, y - 0.35, 8.2, y + 0.35),
            (pcbnew.B_Cu, name, 15.5, 15.5, 16.5, 16.5),
            (pcbnew.F_Cu, "authored", 15.5, 4.5, 16.5, 5.5),
        ):
            area = pcbnew.ZONE(board)
            area.SetLayer(layer)
            area.SetIsRuleArea(True)
            area.SetDoNotAllowTracks(True)
            area.SetDoNotAllowVias(True)
            area.SetDoNotAllowCopperPour(False)
            area.SetZoneName(tag)
            area.Outline().NewOutline()
            for x, yy in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
                area.Outline().Append(_mm(x), _mm(yy))
            board.Add(area)
        board.Save(str(staged))
        return {"board_path": str(staged), "zone_name": name,
                "sha256": hashlib.sha256(staged.read_bytes()).hexdigest()}

    first = plan_at(10, "first")
    sidecars = [tmp_path / "routed.kicad_pro", tmp_path / "routed.kicad_dru"]
    before_rules = [p.read_bytes() if p.is_file() else None for p in sidecars]
    config = {**cfg, "_parent_ground_reservation_plan": first}
    result = pour_power_planes(path, config)
    assert result == {"nets": ["VBUS"], "zones": 1}
    assert [p.read_bytes() if p.is_file() else None for p in sidecars] == before_rules
    board = pcbnew.LoadBoard(path)
    power = next(z for z in board.Zones() if z.GetNetname() == "VBUS")
    point = lambda x, y: pcbnew.VECTOR2I(_mm(x), _mm(y))
    assert not power.Outline().Contains(point(6.5, 10))
    assert power.Outline().Contains(point(16, 16))  # Other layer is not subtracted.
    assert power.Outline().Contains(point(16, 5))  # Untagged authored area is not subtracted.
    assert any(z.GetFilledPolysList(pcbnew.F_Cu).Contains(point(6.5, 12))
               for z in board.Zones() if z.GetNetname() == "VBUS")
    regions = _ground_regions(board, "GND")
    corridor = pcbnew.SHAPE_POLY_SET()
    corridor.NewOutline()
    for x, y in ((5.2, 9.88), (7.8, 9.88), (7.8, 10.12), (5.2, 10.12)):
        corridor.Append(_mm(x), _mm(y))
    # The entire 0.24 mm section, not merely a point or nominal outline,
    # remains in one actual filled ground island.
    assert any(
        (lambda remainder: (remainder.BooleanSubtract(region),
                            abs(remainder.Area()) == 0)[1])(pcbnew.SHAPE_POLY_SET(corridor))
        for layer, region in regions if layer == pcbnew.F_Cu
    )
    pour_power_planes(path, config)
    second = plan_at(12, "second")
    pour_power_planes(path, {**cfg, "_parent_ground_reservation_plan": second})
    board = pcbnew.LoadBoard(path)
    power = next(z for z in board.Zones() if z.GetNetname() == "VBUS")
    assert power.Outline().Contains(point(6.5, 10))
    assert not power.Outline().Contains(point(6.5, 12))

    with pytest.raises(ValueError, match="hash mismatch"):
        pour_power_planes(path, {**cfg, "_parent_ground_reservation_plan":
                                 {**first, "sha256": "bad"}})
    with pytest.raises(ValueError, match="missing or invalid"):
        pour_power_planes(path, {**cfg, "_parent_ground_reservation_plan":
                                 {**first, "zone_name": "missing"}})


def test_power_pour_preserves_source_face_ground_via_landings(tmp_path):
    import hashlib
    import shutil

    from kicraft.autoplacer.brain.gnd_pour import pour_gnd_planes
    from kicraft.autoplacer.hardware.ground_reservations import reserve_local_ground_planes
    from kicraft.autoplacer.hardware.local_contract import audit_local_copper

    path = str(tmp_path / "landings.kicad_pcb")
    _board(path, {"1": "GND", "2": "GND", "3": "VBUS", "4": "VBUS"})
    board = pcbnew.LoadBoard(path)
    for x in (5, 8):
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(12)))
        via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
        via.SetWidth(_mm(1))
        via.SetDrill(_mm(0.4))
        via.SetNet(board.FindNet("GND"))
        board.Add(via)
    board.Save(path)
    cfg = {"power_plane_nets": ["VBUS"], "zone_clearance_mm": 0.3}
    pour_gnd_planes(path, cfg, layers=("F.Cu", "B.Cu"))
    spec = {"owner_ref": "J1", "bypass_ref": "J1",
            "ground_return": {"net": "GND", "owner_ref": "J1", "owner_pin": "1",
                              "member_pin": "2", "max_via_distance_mm": 3}}
    assert audit_local_copper(pcbnew.LoadBoard(path), [spec]) == []
    staged = tmp_path / "landing-plan.kicad_pcb"
    shutil.copyfile(path, staged)
    board = pcbnew.LoadBoard(str(staged))
    name = "ground landing reservations"
    reserve_local_ground_planes(board, [spec], name, router_clearance_mm=0.17)
    board.Save(str(staged))
    plan = {"board_path": str(staged), "zone_name": name,
            "sha256": hashlib.sha256(staged.read_bytes()).hexdigest()}

    pour_power_planes(path, {**cfg, "_parent_ground_reservation_plan": plan})

    # The B.Cu shared plane alone cannot substitute for the F.Cu pad-to-via
    # landings: there are deliberately no fixed tracks carrying those bonds.
    assert audit_local_copper(pcbnew.LoadBoard(path), [spec]) == []

# ---------------------------------------------------------------------------
# GND thermal-via stitcher: collision guard + small-pad escape stitching
# ---------------------------------------------------------------------------

from kicraft.autoplacer.brain.gnd_pour import add_gnd_pour_and_thermal_vias  # noqa: E402


def _sot23_board(path, *, blocker_track=False, thermal=False):
    """Small ordinary GND pad and a larger GND pad, optionally reviewed EP."""
    board = pcbnew.NewBoard(path)
    for name in ("GND", "VIN", "VOUT", "SIG"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    corners = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    def add_fp(ref, pads):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        if ref == "U2" and thermal:
            fp.SetValue("ESP32-S3-MINI-1U-N4R2")
            fp.SetFPID(pcbnew.LIB_ID("reviewed", "BULETM-SMD_ESPRESSIF_ESP32-S3-MINI-1U-N8"))
        board.Add(fp)
        for num, (netname, x, y, w, h) in pads.items():
            pad = pcbnew.PAD(fp)
            if ref == "U2" and thermal and num == "1":
                w = h = 1.2
            pad.SetSize(pcbnew.VECTOR2I(_mm(w), _mm(h)))
            pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNumber("61" if ref == "U2" and thermal and num == "1" else num)
            if netname:
                pad.SetNet(net(netname))
            fp.Add(pad)

    add_fp("U1", {
        "1": ("VIN", 8.5, 10.0, 0.5, 0.5),
        "2": ("GND", 10.0, 10.0, 0.5, 0.5),   # too small for an in-pad via
        "3": ("VOUT", 11.5, 10.0, 0.5, 0.5),
    })
    add_fp("U2", {str(i): (("GND" if i == 1 else "SIG"), 13.0 + 2 * i, 15.0, 1.3, 1.3)
                  for i in range(1, 7)})
    if blocker_track:
        t = pcbnew.PCB_TRACK(board)
        t.SetStart(pcbnew.VECTOR2I(_mm(13.0), _mm(15.0)))
        t.SetEnd(pcbnew.VECTOR2I(_mm(19.0), _mm(15.0)))
        t.SetWidth(_mm(0.3))
        t.SetLayer(pcbnew.B_Cu)
        t.SetNet(net("SIG"))
        board.Add(t)
    board.Save(path)
    return path


def _gnd_vias(path):
    board = pcbnew.LoadBoard(path)
    return [t for t in board.GetTracks()
            if isinstance(t, pcbnew.PCB_VIA) and t.GetNetname() == "GND"]


def test_small_ic_gnd_pad_is_escape_stitched(tmp_path):
    # The post-connector-fix rc7 signature (run_03 U1.5 / run_05 U2.2): a
    # SOT-23-class regulator's lone GND pad floats as an F.Cu pour island
    # because the old stitcher required >= 6 pads AND an in-pad via fit.
    # It must now get a short escape stub with a via at the tip.
    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path)
    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["escape_stitched"] >= 1, res
    assert res["thermal_vias_added"] == 0, res
    vias = _gnd_vias(path)
    assert len(vias) >= 1
    # The escape via is OFF the small pad (10,10), not drilled through it.
    assert all(
        (pcbnew.ToMM(v.GetPosition().x), pcbnew.ToMM(v.GetPosition().y))
        != (pytest.approx(10.0), pytest.approx(10.0))
        for v in vias
    )

def test_reviewed_module_thermal_land_accepts_in_pad_via(tmp_path):
    path = str(tmp_path / "thermal.kicad_pcb")
    _sot23_board(path, thermal=True)
    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["thermal_vias_added"] == 1, res
    board = pcbnew.LoadBoard(path)
    fp = next(fp for fp in board.GetFootprints() if fp.GetReference() == "U2")
    pad = next(p for p in fp.Pads() if p.GetNumber() == "61")
    assert any(pad.HitTest(v.GetPosition()) for v in _gnd_vias(path))

@pytest.mark.parametrize("value", ["ESP32-C3-MINI-1-H4X", "ESP32-C3-MINI-1-N4"])
def test_reviewed_c3_exposed_land_accepts_in_pad_via(tmp_path, value):
    path = str(tmp_path / "c3.kicad_pcb")
    _sot23_board(path, thermal=True)
    board = pcbnew.LoadBoard(path)
    fp = next(fp for fp in board.GetFootprints() if fp.GetReference() == "U2")
    fp.SetValue(value)
    fp.SetFPID(pcbnew.LIB_ID("reviewed", "WIFIM-SMD_ESP32-C3-MINI-1"))
    pad = next(p for p in fp.Pads() if p.GetNumber() == "61")
    pad.SetNumber("49")
    pad.SetSize(pcbnew.VECTOR2I(_mm(1.45), _mm(1.45)))
    assert ("U2", "49") not in {(s.ref, s.pad) for s in gnd_escape_specs(board)}
    board.Save(path)
    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["thermal_vias_added"] == 1, res
    board = pcbnew.LoadBoard(path)
    pad = next(p for fp in board.GetFootprints() if fp.GetReference() == "U2"
               for p in fp.Pads() if p.GetNumber() == "49")
    assert any(pad.HitTest(v.GetPosition()) for v in _gnd_vias(path))


def test_c3_ordinary_land_is_not_authorized_by_size_and_name(tmp_path):
    path = str(tmp_path / "c3-ordinary.kicad_pcb")
    _sot23_board(path, thermal=True)
    board = pcbnew.LoadBoard(path)
    fp = next(fp for fp in board.GetFootprints() if fp.GetReference() == "U2")
    fp.SetValue("ESP32-C3-MINI-1-H4X")
    fp.SetFPID(pcbnew.LIB_ID("reviewed", "WIFIM-SMD_ESP32-C3-MINI-1"))
    pad = next(p for p in fp.Pads() if p.GetNumber() == "61")
    pad.SetNumber("48")
    pad.SetSize(pcbnew.VECTOR2I(_mm(1.45), _mm(1.45)))
    assert ("U2", "48") in {(s.ref, s.pad) for s in gnd_escape_specs(board)}
    board.Save(path)
    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["thermal_vias_added"] == 0, res
    board = pcbnew.LoadBoard(path)
    pad = next(p for fp in board.GetFootprints() if fp.GetReference() == "U2"
               for p in fp.Pads() if p.GetNumber() == "48")
    assert all(not pad.HitTest(v.GetPosition()) for v in _gnd_vias(path))

def test_thermal_permission_is_bound_to_reviewed_identity_and_land(tmp_path):
    path = str(tmp_path / "identity.kicad_pcb")
    _sot23_board(path, thermal=True)
    board = pcbnew.LoadBoard(path)
    fp = next(fp for fp in board.GetFootprints() if fp.GetReference() == "U2")
    assert ("U2", "61") not in {(s.ref, s.pad) for s in gnd_escape_specs(board)}
    fp.SetValue("unreviewed module")
    assert ("U2", "61") in {(s.ref, s.pad) for s in gnd_escape_specs(board)}
    board.Save(path)
    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["thermal_vias_added"] == 0, res
    board = pcbnew.LoadBoard(path)
    pad = next(p for fp in board.GetFootprints() if fp.GetReference() == "U2"
               for p in fp.Pads() if p.GetNumber() == "61")
    assert all(not pad.HitTest(v.GetPosition()) for v in _gnd_vias(path))



def test_thermal_via_not_stamped_through_foreign_track(tmp_path):
    # Regression for the IP2368-bank incident: _add_via had no collision check
    # and stamped GND vias straight through routed B.Cu tracks (7 shorts). A
    # foreign B.Cu track under U2's GND pad must block that via.
    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path, blocker_track=True, thermal=True)
    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["thermal_vias_blocked"] >= 1, res
    blocked_at = (15.0, 15.0)  # U2 pad 1 centre, right on the SIG track
    for v in _gnd_vias(path):
        pos = (pcbnew.ToMM(v.GetPosition().x), pcbnew.ToMM(v.GetPosition().y))
        assert pos != (pytest.approx(blocked_at[0]), pytest.approx(blocked_at[1]))


# ---------------------------------------------------------------------------
# GND strand repair: stranded clusters tied back to the main plane
# ---------------------------------------------------------------------------

from kicraft.autoplacer.brain.gnd_pour import repair_stranded_gnd  # noqa: E402


def _stranded_board(path, *, block_both_layers=False):
    """Main GND cluster (U2 pad + via on it) at (8,10); a stranded 2-pad
    connector GND pin J7.2 at (16,10) with nothing nearby -- the run_03 shape.
    Optionally copper walls on BOTH layers between them."""
    board = pcbnew.NewBoard(path)
    for name in ("GND", "5V", "SIG"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    corners = [(0, 0), (24, 0), (24, 20), (0, 20), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    def add_fp(ref, pads):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        board.Add(fp)
        for num, (netname, x, y) in pads.items():
            pad = pcbnew.PAD(fp)
            pad.SetSize(pcbnew.VECTOR2I(_mm(1.0), _mm(1.0)))
            pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNumber(num)
            pad.SetNet(net(netname))
            fp.Add(pad)

    add_fp("U2", {"1": ("GND", 8.0, 10.0), "2": ("SIG", 8.0, 13.0),
                  "3": ("5V", 8.0, 7.0)})
    add_fp("J7", {"1": ("5V", 16.0, 7.0), "2": ("GND", 16.0, 10.0)})
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(8.0), _mm(10.0)))
    via.SetDrill(_mm(0.3))
    try:
        via.SetWidth(_mm(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, _mm(0.6))
    via.SetNet(net("GND"))
    board.Add(via)
    if block_both_layers:
        for layer in (pcbnew.F_Cu, pcbnew.B_Cu):
            t = pcbnew.PCB_TRACK(board)
            t.SetStart(pcbnew.VECTOR2I(_mm(12.0), _mm(2.0)))
            t.SetEnd(pcbnew.VECTOR2I(_mm(12.0), _mm(18.0)))
            t.SetWidth(_mm(0.3))
            t.SetLayer(layer)
            t.SetNet(net("SIG"))
            board.Add(t)
    board.Save(path)
    return path


def test_stranded_gnd_pad_is_tied_back_to_main_cluster(tmp_path):
    # run_03 J7.2 regression: a 2-pad THT/SMD connector GND pin with no plane
    # reach, no via, no shield-tie mate. The repair pass must tie it straight
    # back to the main GND cluster and report it.
    path = str(tmp_path / "b.kicad_pcb")
    _stranded_board(path)
    res = repair_stranded_gnd(path, {"gnd_zone_net": "GND"})
    assert res["stranded"] == 1 and res["tied"] == 1, res
    board = pcbnew.LoadBoard(path)
    gnd_tracks = [t for t in board.GetTracks()
                  if not isinstance(t, pcbnew.PCB_VIA) and t.GetNetname() == "GND"]
    assert len(gnd_tracks) == 1
    xs = sorted([pcbnew.ToMM(gnd_tracks[0].GetStart().x),
                 pcbnew.ToMM(gnd_tracks[0].GetEnd().x)])
    assert xs == [pytest.approx(8.0), pytest.approx(16.0)]


def test_stranded_gnd_skipped_when_both_layers_blocked(tmp_path):
    # A foreign wall on BOTH layers: the tie must be skipped (board no worse),
    # never stamped across the foreign copper.
    path = str(tmp_path / "b.kicad_pcb")
    _stranded_board(path, block_both_layers=True)
    res = repair_stranded_gnd(path, {"gnd_zone_net": "GND"})
    assert res["tied"] == 0, res
    assert any("no_clear_path" in s for s in res["skipped"]), res
    board = pcbnew.LoadBoard(path)
    assert not [t for t in board.GetTracks()
                if not isinstance(t, pcbnew.PCB_VIA) and t.GetNetname() == "GND"]


def test_strand_repair_noop_when_single_cluster(tmp_path):
    # Everything already connected -> nothing stamped.
    path = str(tmp_path / "b.kicad_pcb")
    _stranded_board(path)
    board = pcbnew.LoadBoard(path)
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(pcbnew.VECTOR2I(_mm(8.0), _mm(10.0)))
    t.SetEnd(pcbnew.VECTOR2I(_mm(16.0), _mm(10.0)))
    t.SetWidth(_mm(0.3))
    t.SetLayer(pcbnew.F_Cu)
    t.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Add(t)
    board.Save(path)
    res = repair_stranded_gnd(path, {"gnd_zone_net": "GND"})
    assert res["stranded"] == 0 and res["tied"] == 0, res


def test_stranded_power_pad_is_tied_back_to_main_cluster(tmp_path):
    # KC-Z57JEZ regression shape: a power rail (5V) split into two clusters
    # (U2.3 vs J7.1) with no track between them. The generalized repair must
    # tie them exactly like the GND pass does.
    from kicraft.autoplacer.brain.gnd_pour import repair_stranded_power

    path = str(tmp_path / "b.kicad_pcb")
    _stranded_board(path)
    res = repair_stranded_power(path, ["5V"], {"gnd_zone_net": "GND"})
    assert res["nets"] == ["5V"], res
    assert res["stranded"] == 1 and res["tied"] == 1, res
    board = pcbnew.LoadBoard(path)
    pwr_tracks = [t for t in board.GetTracks()
                  if not isinstance(t, pcbnew.PCB_VIA) and t.GetNetname() == "5V"]
    assert len(pwr_tracks) == 1
    xs = sorted([pcbnew.ToMM(pwr_tracks[0].GetStart().x),
                 pcbnew.ToMM(pwr_tracks[0].GetEnd().x)])
    assert xs == [pytest.approx(8.0), pytest.approx(16.0)]


def test_power_strand_repair_disabled_is_noop(tmp_path):
    from kicraft.autoplacer.brain.gnd_pour import repair_stranded_power

    path = str(tmp_path / "b.kicad_pcb")
    _stranded_board(path)
    res = repair_stranded_power(
        path, ["5V"],
        {"gnd_zone_net": "GND", "power_strand_repair_enabled": False})
    assert res == {"nets": [], "stranded": 0, "tied": 0, "skipped": [],
                   "unresolved": 0}
    board = pcbnew.LoadBoard(path)
    assert not [t for t in board.GetTracks()
                if not isinstance(t, pcbnew.PCB_VIA) and t.GetNetname() == "5V"]


def _same_number_pads_board(path, *, stranded_layer=pcbnew.F_Cu):
    """run_01/run_03 ESP32-module shape: U2 carries TWO pads both numbered
    "GND". The first-in-iteration one sits ON the main cluster (with a via);
    the far one is stranded. ``_find_pad``'s first-match used to tie the
    already-connected pad and report success while the strand stayed."""
    board = pcbnew.NewBoard(path)
    for name in ("GND", "SIG"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    corners = [(0, 0), (30, 0), (30, 20), (0, 20), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    def _smd_mask(layer):
        # Build the single-side SMD layer set explicitly. Passing
        # pcbnew.PAD.SMDMask().FlipStandardLayers() straight into SetLayerSet
        # segfaults in the KiCad 9 binding (FlipStandardLayers hands back a
        # temporary LSET proxy that SetLayerSet then reads after it is freed);
        # AddLayer onto a held LSET is stable.
        ls = pcbnew.LSET()
        cu, paste, mask = ((pcbnew.F_Cu, pcbnew.F_Paste, pcbnew.F_Mask)
                           if layer == pcbnew.F_Cu
                           else (pcbnew.B_Cu, pcbnew.B_Paste, pcbnew.B_Mask))
        for L in (cu, paste, mask):
            ls.AddLayer(L)
        return ls

    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("U2")
    board.Add(fp)
    for x, y, layer in ((8.0, 10.0, pcbnew.F_Cu), (22.0, 10.0, stranded_layer)):
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(_mm(1.0), _mm(1.0)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(_smd_mask(layer))
        pad.SetNumber("GND")
        pad.SetNet(net("GND"))
        fp.Add(pad)
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(8.0), _mm(10.0)))
    via.SetDrill(_mm(0.3))
    try:
        via.SetWidth(_mm(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, _mm(0.6))
    via.SetNet(net("GND"))
    board.Add(via)
    board.Save(path)
    return path


def test_strand_repair_ties_the_stranded_same_number_pad(tmp_path):
    # run_01 ESP32-S3 regression: 13 module pads all numbered "GND"; the
    # repair must tie the STRANDED one, not the first number-match (which is
    # already on the main plane -- a zero-length no-op that reported "tied").
    path = str(tmp_path / "b.kicad_pcb")
    _same_number_pads_board(path)
    res = repair_stranded_gnd(path, {"gnd_zone_net": "GND"})
    assert res["stranded"] == 1 and res["tied"] == 1, res
    assert res["unresolved"] == 0, res
    board = pcbnew.LoadBoard(path)
    gnd_tracks = [t for t in board.GetTracks()
                  if not isinstance(t, pcbnew.PCB_VIA) and t.GetNetname() == "GND"]
    assert len(gnd_tracks) == 1
    xs = sorted([pcbnew.ToMM(gnd_tracks[0].GetStart().x),
                 pcbnew.ToMM(gnd_tracks[0].GetEnd().x)])
    assert xs == [pytest.approx(8.0), pytest.approx(22.0)]


def test_strand_repair_skips_layer_unreachable_tie(tmp_path):
    # A B.Cu-only stranded pad with no via bridge cannot legally tie to an
    # F.Cu-only target: the old code stamped the F.Cu track anyway (dead
    # copper starting under a B.Cu pad) and claimed success. Now it must
    # skip, stamp nothing, and report the strand as unresolved.
    path = str(tmp_path / "b.kicad_pcb")
    _same_number_pads_board(path, stranded_layer=pcbnew.B_Cu)
    board = pcbnew.LoadBoard(path)
    for t in list(board.GetTracks()):  # drop the via: F.Cu pad+nothing main
        board.Remove(t)
    board.Save(path)
    res = repair_stranded_gnd(path, {"gnd_zone_net": "GND"})
    assert res["tied"] == 0, res
    assert res["unresolved"] == 1, res
    assert any("no_clear_path" in s for s in res["skipped"]), res
    board = pcbnew.LoadBoard(path)
    assert not [t for t in board.GetTracks()
                if not isinstance(t, pcbnew.PCB_VIA) and t.GetNetname() == "GND"]


def test_strand_repair_via_bridge_allows_far_layer_tie(tmp_path):
    # The escape-stitch shape: an F.Cu stranded pad WITH a via at its centre
    # may tie on B.Cu through that via when the target only has B.Cu copper.
    path = str(tmp_path / "b.kicad_pcb")
    _same_number_pads_board(path, stranded_layer=pcbnew.B_Cu)
    board = pcbnew.LoadBoard(path)
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(22.0), _mm(10.0)))
    via.SetDrill(_mm(0.3))
    try:
        via.SetWidth(_mm(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, _mm(0.6))
    via.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Add(via)
    board.Save(path)
    res = repair_stranded_gnd(path, {"gnd_zone_net": "GND"})
    assert res["tied"] == 1 and res["unresolved"] == 0, res


def test_power_strand_repair_autodetects_poured_rails(tmp_path):
    # nets=None must fall back to the same detection pour_power_planes uses
    # (most-padded power net, GND excluded) so the parent caller that pours
    # then repairs in separate subprocesses agrees with the pour.
    from kicraft.autoplacer.brain.gnd_pour import repair_stranded_power

    path = str(tmp_path / "b.kicad_pcb")
    _stranded_board(path)
    res = repair_stranded_power(path, None, {"gnd_zone_net": "GND"})
    assert res["nets"] == ["5V"], res
    assert res["tied"] == 1, res


# ---------------------------------------------------------------------------
# Netclass pair clearance + hole-to-hole + shape-independent stitching
# ---------------------------------------------------------------------------

import json  # noqa: E402
import shutil  # noqa: E402

from kicraft.autoplacer.brain.gnd_pour import gnd_escape_specs  # noqa: E402
from kicraft.autoplacer.brain.breakout_stubs import add_breakout_stubs  # noqa: E402

# Default(0.15) / Power(0.30) with GND assigned to Power -- the shape every
# generated board ships in its sibling .kicad_pro.
_NET_SETTINGS = {
    "classes": [
        {"bus_width": 12, "clearance": 0.15, "diff_pair_gap": 0.25,
         "diff_pair_via_gap": 0.25, "diff_pair_width": 0.2, "line_style": 0,
         "microvia_diameter": 0.3, "microvia_drill": 0.1, "name": "Default",
         "pcb_color": "rgba(0, 0, 0, 0.000)", "priority": 2147483647,
         "schematic_color": "rgba(0, 0, 0, 0.000)", "track_width": 0.2,
         "via_diameter": 0.6, "via_drill": 0.3, "wire_width": 6},
        {"bus_width": 12, "clearance": 0.3, "diff_pair_gap": 0.25,
         "diff_pair_via_gap": 0.25, "diff_pair_width": 0.2, "line_style": 0,
         "microvia_diameter": 0.3, "microvia_drill": 0.1, "name": "Power",
         "pcb_color": "rgba(0, 0, 0, 0.000)", "priority": 0,
         "schematic_color": "rgba(0, 0, 0, 0.000)", "track_width": 0.5,
         "via_diameter": 0.8, "via_drill": 0.4, "wire_width": 6},
    ],
    "meta": {"version": 4},
    "net_colors": None,
    "netclass_assignments": None,
    "netclass_patterns": [{"netclass": "Power", "pattern": "GND"}],
}


def _as_project_board(build_path: str, case_path: str) -> str:
    """Copy a built board to a FRESH path with a netclass-bearing sibling
    .kicad_pro. pcbnew caches one project per board path in-process, so the
    case path must never have been opened before -- loading the copy is the
    only way a test board resolves netclass clearances like pipeline boards."""
    shutil.copy(build_path, case_path)
    pro = case_path[: -len(".kicad_pcb")] + ".kicad_pro"
    with open(pro, "w", encoding="utf-8") as fh:
        json.dump({"meta": {"filename": pro.rsplit("/", 1)[-1], "version": 3},
                   "net_settings": _NET_SETTINGS}, fh)
    return case_path


def test_thermal_via_respects_netclass_pair_clearance(tmp_path):
    # A SIG (Default, 0.15) B.Cu track 0.6 mm from the GND pad centre: the
    # old flat 0.153 floor margin (0.528 mm) let the via land 0.225 mm from
    # the track -- a Power-netclass (0.30) DRC error, the KC-UXASHQ escape-via
    # signature. With pair clearance the margin is 0.675 mm -> blocked.
    build = str(tmp_path / "build.kicad_pcb")
    _sot23_board(build, thermal=True)
    board = pcbnew.LoadBoard(build)
    t = pcbnew.PCB_TRACK(board)
    t.SetStart(pcbnew.VECTOR2I(_mm(12.0), _mm(15.6)))
    t.SetEnd(pcbnew.VECTOR2I(_mm(18.0), _mm(15.6)))
    t.SetWidth(_mm(0.15))
    t.SetLayer(pcbnew.B_Cu)
    t.SetNet(board.GetNetInfo().GetNetItem("SIG"))
    board.Add(t)
    board.Save(build)

    # Control (no project netclasses -> 0.153 floor): the via lands.
    ctl = str(tmp_path / "control.kicad_pcb")
    shutil.copy(build, ctl)
    res = add_gnd_pour_and_thermal_vias(ctl, {"gnd_zone_net": "GND"})
    assert any(
        (pcbnew.ToMM(v.GetPosition().x), pcbnew.ToMM(v.GetPosition().y))
        == (pytest.approx(15.0), pytest.approx(15.0))
        for v in _gnd_vias(ctl)
    ), res

    # With Power=0.30 netclasses the same via is blocked.
    case = _as_project_board(build, str(tmp_path / "case.kicad_pcb"))
    res = add_gnd_pour_and_thermal_vias(case, {"gnd_zone_net": "GND"})
    assert res["thermal_vias_blocked"] >= 1, res
    assert not any(
        (pcbnew.ToMM(v.GetPosition().x), pcbnew.ToMM(v.GetPosition().y))
        == (pytest.approx(15.0), pytest.approx(15.0))
        for v in _gnd_vias(case)
    )


def test_thermal_via_blocked_by_hole_to_hole(tmp_path):
    # A GND via 0.4 mm from the GND pad centre is same-net copper (the old
    # guard allowed it) but its drilled hole is 0.4 mm from where the in-pad
    # via would drill -- inside hole-to-hole minimum. Must be blocked.
    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path, thermal=True)
    board = pcbnew.LoadBoard(path)
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(15.4), _mm(15.0)))
    via.SetDrill(_mm(0.3))
    try:
        via.SetWidth(_mm(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, _mm(0.6))
    via.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Add(via)
    board.Save(path)

    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["thermal_vias_blocked"] >= 1, res
    assert not any(
        (pcbnew.ToMM(v.GetPosition().x), pcbnew.ToMM(v.GetPosition().y))
        == (pytest.approx(15.0), pytest.approx(15.0))
        for v in _gnd_vias(path)
    )


def test_two_pad_passive_via_fitting_gnd_pad_is_stitched(tmp_path):
    # A two-pad capacitor needs an off-land dogbone, not a drilled solder land.
    path = str(tmp_path / "b.kicad_pcb")
    board = pcbnew.NewBoard(path)
    for name in ("GND", "+3V3"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))
    corners = [(0, 0), (20, 0), (20, 20), (0, 20), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("C2")
    board.Add(fp)
    for num, netname, x in (("1", "+3V3", 8.0), ("2", "GND", 6.0)):
        pad = pcbnew.PAD(fp)
        pad.SetSize(pcbnew.VECTOR2I(_mm(1.0), _mm(1.45)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(8.0)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber(num)
        pad.SetNet(board.GetNetInfo().GetNetItem(netname))
        fp.Add(pad)
    board.Save(path)

    res = add_gnd_pour_and_thermal_vias(
        path, {"gnd_zone_net": "GND", "gnd_escape_length_mm": 1.5}
    )
    assert res["escape_stitched"] == 1, res
    board = pcbnew.LoadBoard(path)
    pad = next(p for fp in board.GetFootprints() for p in fp.Pads() if p.GetNumber() == "2")
    via = next(t for t in board.GetTracks() if isinstance(t, pcbnew.PCB_VIA))
    assert not pad.HitTest(via.GetPosition(), _mm(0.4))
    track = next(t for t in board.GetTracks() if t.GetClass() == "PCB_TRACK")
    assert pad.HitTest(track.GetStart())
    assert track.GetEnd() == via.GetPosition()
    assert track.GetNetname() == via.GetNetname() == "GND"

def _bottom_ground_pad_board(path):
    _board(path, {"1": "GND"})
    board = pcbnew.LoadBoard(path)
    pad = next(p for fp in board.GetFootprints() for p in fp.Pads())
    layers = pcbnew.LSET()
    for layer in (pcbnew.B_Cu, pcbnew.B_Mask, pcbnew.B_Paste):
        layers.AddLayer(layer)
    pad.SetLayerSet(layers)
    board.Save(path)


def _assert_bottom_pad_connected_to_via(path):
    board = pcbnew.LoadBoard(path)
    pad = next(p for fp in board.GetFootprints() for p in fp.Pads())
    assert pad.IsOnLayer(pcbnew.B_Cu) and not pad.IsOnLayer(pcbnew.F_Cu)
    vias = [t for t in board.GetTracks()
            if t.GetClass() == "PCB_VIA" and t.GetNetname() == "GND"]
    tracks = [t for t in board.GetTracks()
              if t.GetClass() == "PCB_TRACK" and t.GetLayer() == pcbnew.B_Cu
              and t.GetNetname() == "GND"]
    assert any(
        pad.HitTest(track.GetStart())
        and any(track.GetEnd() == via.GetPosition() and
                not pad.HitTest(via.GetPosition(), _mm(0.4)) for via in vias)
        for track in tracks
    )


@pytest.mark.parametrize("pre_route", [True, False])
def test_bottom_smd_ground_gets_same_face_connected_dogbone(tmp_path, pre_route):
    path = str(tmp_path / "bottom.kicad_pcb")
    _bottom_ground_pad_board(path)
    if pre_route:
        board = pcbnew.LoadBoard(path)
        specs = gnd_escape_specs(board)
        del board
        res = add_breakout_stubs(path, specs)
        assert res["vias"] == 1, res
    else:
        res = add_gnd_pour_and_thermal_vias(path)
        assert res["escape_stitched"] == 1, res
    _assert_bottom_pad_connected_to_via(path)


def test_wrong_face_track_does_not_count_as_bottom_pad_bond(tmp_path):
    path = str(tmp_path / "wrong-face.kicad_pcb")
    _bottom_ground_pad_board(path)
    board = pcbnew.LoadBoard(path)
    gnd = board.GetNetInfo().GetNetItem("GND")
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(6), _mm(10)))
    via.SetDrill(_mm(0.3))
    via.SetWidth(_mm(0.6))
    via.SetNet(gnd)
    board.Add(via)
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(pcbnew.VECTOR2I(_mm(5), _mm(10)))
    track.SetEnd(via.GetPosition())
    track.SetWidth(_mm(0.2))
    track.SetLayer(pcbnew.F_Cu)
    track.SetNet(gnd)
    board.Add(track)
    board.Save(path)
    res = add_gnd_pour_and_thermal_vias(path)
    assert res["escape_stitched"] == 1, res
    _assert_bottom_pad_connected_to_via(path)


def test_escape_skipped_when_pre_route_via_already_bonds(tmp_path):
    # A GND via 0.9 mm from U1's small GND pad (a pre-route gnd_escape_specs
    # stub tip) already bonds it to the plane: the post-route pass must not
    # stamp a SECOND escape stub for the same pad.
    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path)
    board = pcbnew.LoadBoard(path)
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(10.9), _mm(10.0)))
    via.SetDrill(_mm(0.3))
    try:
        via.SetWidth(_mm(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, _mm(0.6))
    via.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Add(via)
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(pcbnew.VECTOR2I(_mm(10), _mm(10)))
    track.SetEnd(via.GetPosition())
    track.SetWidth(_mm(0.2))
    track.SetLayer(pcbnew.F_Cu)
    track.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Add(track)
    board.Save(path)

    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert not any(
        t.GetClass() == "PCB_TRACK" and t.GetStart() == pcbnew.VECTOR2I(_mm(10), _mm(10))
        and t.GetEnd() != pcbnew.VECTOR2I(_mm(10.9), _mm(10))
        for t in pcbnew.LoadBoard(path).GetTracks()
    ), res


def test_ground_return_finisher_does_not_accept_remote_existing_via(tmp_path, monkeypatch):
    from kicraft.autoplacer.brain import breakout_stubs

    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path)
    board = pcbnew.LoadBoard(path)
    net = board.GetNetInfo().GetNetItem("GND")
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(10), _mm(14)))
    via.SetDrill(_mm(0.3))
    try:
        via.SetWidth(_mm(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, _mm(0.6))
    via.SetNet(net)
    board.Add(via)
    track = pcbnew.PCB_TRACK(board)
    track.SetStart(pcbnew.VECTOR2I(_mm(10), _mm(10)))
    track.SetEnd(via.GetPosition())
    track.SetWidth(_mm(0.2))
    track.SetLayer(pcbnew.F_Cu)
    track.SetNet(net)
    board.Add(track)
    board.Save(path)
    captured = []
    def capture(path, specs, cfg):
        captured.extend(specs)
        return {"vias": 0, "skipped": []}
    monkeypatch.setattr(breakout_stubs, "add_breakout_stubs", capture)
    contract = {"owner_ref": "U1", "bypass_ref": "U2",
                "ground_return": {"owner_ref": "U1", "owner_pin": "2", "member_pin": "1",
                                  "net": "GND", "max_via_distance_mm": 3.0}}
    add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND",
                                        "local_bypasses": [contract]})
    assert any(s.ref == "U1" and s.pad == "2" and s.length_mm <= 3.0 for s in captured)


def test_ground_finisher_prioritizes_explicit_owner_and_bypass_not_power_owner(tmp_path, monkeypatch):
    from kicraft.autoplacer.brain import breakout_stubs

    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path)
    captured = []
    def capture(path, specs, cfg):
        captured.extend(specs)
        return {"vias": 0, "skipped": []}
    monkeypatch.setattr(breakout_stubs, "add_breakout_stubs", capture)
    contract = {"owner_ref": "L1", "bypass_ref": "U2",
                "ground_return": {"owner_ref": "U1", "owner_pin": "2", "member_pin": "1",
                                  "net": "GND", "max_via_distance_mm": 0.7}}
    add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND",
                                        "local_bypasses": [contract]})
    assert {(s.ref, str(s.pad)) for s in captured[:2]} == {("U1", "2"), ("U2", "1")}
    assert all(s.length_mm <= 0.7 for s in captured[:2])
    assert not any(s.ref == "L1" for s in captured)


@pytest.mark.parametrize("missing", ["owner_ref", "owner_pin"])
def test_ground_finisher_rejects_missing_ground_owner(tmp_path, missing):
    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path)
    ground = {"owner_ref": "U1", "owner_pin": "2", "member_pin": "1",
              "net": "GND", "max_via_distance_mm": 3.0}
    del ground[missing]
    with pytest.raises(ValueError, match="explicit owner"):
        add_gnd_pour_and_thermal_vias(
            path, {"gnd_zone_net": "GND", "local_bypasses": [
                {"owner_ref": "L1", "bypass_ref": "U2", "ground_return": ground}]})


def test_gnd_escape_specs_targets_ordinary_gnd_pads(tmp_path):
    # Size alone does not authorize an in-pad via.
    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path)
    board = pcbnew.LoadBoard(path)
    specs = gnd_escape_specs(board, {"gnd_zone_net": "GND"})
    assert {(s.ref, s.pad) for s in specs} == {("U1", "2"), ("U2", "1")}
    assert all(s.via_at_end and s.via_outside_smd_pads for s in specs)
    assert gnd_escape_specs(board, {"gnd_zone_net": "GND",
                                    "gnd_pre_escape": False}) == []


# ---------------------------------------------------------------------------
# Footprint rule-area keep-out guard + solid GND pad connection (KC-S8PC37):
# the ESP32-S3-MINI ships an antenna_keepout rule area inside its .kicad_mod;
# the GND finisher stamped 16 escape stubs + 14 vias straight into it (30
# items_not_allowed), and the thermal GND zone starved spokes on the module's
# dense pin ring.
# ---------------------------------------------------------------------------

from kicraft.autoplacer.brain.gnd_pour import pour_gnd_planes  # noqa: E402


def _module_keepout_board(path):
    """A module-shaped U1: a dense row of 0.5 mm GND pads (0.85 mm pitch -- too
    small for an in-pad via, too tight for thermal spokes) with an embedded
    antenna_keepout rule area just below the row, plus one large GND pad on U2
    far from the keep-out that must still get its in-pad stitch via."""
    board = pcbnew.NewBoard(path)
    for name in ("GND", "SIG"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))

    def net(n):
        return board.GetNetInfo().GetNetItem(n)

    corners = [(0, 0), (30, 0), (30, 30), (0, 30), (0, 0)]
    for (x1, y1), (x2, y2) in zip(corners, corners[1:]):
        seg = pcbnew.PCB_SHAPE(board)
        seg.SetShape(pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(_mm(x1), _mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(_mm(x2), _mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        board.Add(seg)

    u1 = pcbnew.FOOTPRINT(board)
    u1.SetReference("U1")
    board.Add(u1)
    x = 10.0
    for i in range(12):
        pad = pcbnew.PAD(u1)
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.5), _mm(0.5)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(10.0)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNumber(str(i + 1))
        pad.SetNet(net("GND"))
        u1.Add(pad)
        x += 0.85
    # Antenna keep-out embedded in the footprint, just below the GND pad row.
    ko = pcbnew.ZONE(u1)
    ko.SetIsRuleArea(True)
    ko.SetZoneName("antenna_keepout")
    ko.SetDoNotAllowTracks(True)
    ko.SetDoNotAllowVias(True)
    ko.SetDoNotAllowCopperPour(True)
    ko.SetDoNotAllowPads(True)
    ls = pcbnew.LSET()
    ls.AddLayer(pcbnew.F_Cu)
    ls.AddLayer(pcbnew.B_Cu)
    ko.SetLayerSet(ls)
    outline = ko.Outline()
    outline.NewOutline()
    for px, py in ((9.5, 10.6), (20.0, 10.6), (20.0, 14.0), (9.5, 14.0)):
        outline.Append(_mm(px), _mm(py))
    u1.Add(ko)

    # An ordinary GND land far from the keepout still gets an off-pad escape.
    u2 = pcbnew.FOOTPRINT(board)
    u2.SetReference("U2")
    board.Add(u2)
    pad = pcbnew.PAD(u2)
    pad.SetSize(pcbnew.VECTOR2I(_mm(1.2), _mm(1.2)))
    pad.SetPosition(pcbnew.VECTOR2I(_mm(25.0), _mm(25.0)))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
    pad.SetLayerSet(pcbnew.PAD.SMDMask())
    pad.SetNumber("1")
    pad.SetNet(net("GND"))
    u2.Add(pad)

    board.Save(path)
    return path


def test_gnd_finisher_keeps_copper_out_of_footprint_keepout(tmp_path):
    # KC-S8PC37: every escape stub + thermal via the finisher would stamp into
    # U1's antenna_keepout must be skipped, while the far U2 GND pad still
    # stitches (the guard is targeted, not a blanket disable).
    path = str(tmp_path / "b.kicad_pcb")
    _module_keepout_board(path)
    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})

    board = pcbnew.LoadBoard(path)
    ko = next(z for fp in board.GetFootprints() for z in fp.Zones()
              if z.GetIsRuleArea())
    poly = ko.Outline()

    def _inside(item):
        p = item.GetPosition()
        return poly.Contains(pcbnew.VECTOR2I(int(p.x), int(p.y)))

    gnd_in_keepout = [t for t in board.GetTracks()
                      if t.GetNetname() == "GND" and _inside(t)]
    assert gnd_in_keepout == [], gnd_in_keepout
    # The far U2 pad is unaffected by the keep-out and still gets a dogbone.
    assert any(
        isinstance(t, pcbnew.PCB_VIA) and t.GetNetname() == "GND"
        and abs(pcbnew.ToMM(t.GetPosition().x) - 25.0) <= 1.25
        and abs(pcbnew.ToMM(t.GetPosition().y) - 25.0) <= 1.25
        for t in board.GetTracks()
    ), res


def test_gnd_pour_pad_connection_solid_by_default_thermal_when_configured(tmp_path):
    # The starved_thermal fix: GND pours connect pads solidly by default (no
    # spokes to starve on a dense pin ring), and "thermal" restores relief.
    path = str(tmp_path / "b.kicad_pcb")
    _board(path, {"1": "GND", "2": "GND"})
    pour_gnd_planes(path, {"gnd_zone_net": "GND"}, layers=("F.Cu",))
    board = pcbnew.LoadBoard(path)
    z = next(z for z in board.Zones()
             if z.GetNetname() == "GND" and not z.GetIsRuleArea())
    assert z.GetPadConnection() == pcbnew.ZONE_CONNECTION_FULL

    path2 = str(tmp_path / "b2.kicad_pcb")
    _board(path2, {"1": "GND", "2": "GND"})
    pour_gnd_planes(path2, {"gnd_zone_net": "GND",
                            "gnd_plane_pad_connection": "thermal"},
                    layers=("F.Cu",))
    board2 = pcbnew.LoadBoard(path2)
    z2 = next(z for z in board2.Zones()
              if z.GetNetname() == "GND" and not z.GetIsRuleArea())
    assert z2.GetPadConnection() == pcbnew.ZONE_CONNECTION_THERMAL


def _fine_pitch_thermal_board(path, *, via_positions=((10.2, 11.8), (10.2, 8.2))):
    """Two 0.5 mm-pitch ground pins beside an exposed ground land."""
    _board(path, {"1": "GND", "2": "SIG"})
    board = pcbnew.LoadBoard(path)
    fp = next(iter(board.GetFootprints()))
    fp.SetReference("U8")
    for pad in list(fp.Pads()):
        fp.Remove(pad)
    gnd = board.GetNetInfo().GetNetItem("GND")
    sig = board.GetNetInfo().GetNetItem("SIG")
    for number, y, net in (
        ("7", 11.25, sig), ("8", 10.75, gnd), ("9", 10.25, sig),
        ("10", 9.75, sig), ("11", 9.25, gnd), ("12", 8.75, sig),
    ):
        pad = pcbnew.PAD(fp)
        pad.SetNumber(number)
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetShape(pcbnew.PAD_SHAPE_RECT)
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.58), _mm(0.24)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(11.9125), _mm(y)))
        pad.SetNet(net)
        fp.Add(pad)
    ep = pcbnew.PAD(fp)
    ep.SetNumber("25")
    ep.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
    ep.SetLayerSet(pcbnew.PAD.SMDMask())
    ep.SetShape(pcbnew.PAD_SHAPE_RECT)
    ep.SetSize(pcbnew.VECTOR2I(_mm(2.7), _mm(2.7)))
    ep.SetPosition(pcbnew.VECTOR2I(_mm(10), _mm(10)))
    ep.SetNet(gnd)
    fp.Add(ep)
    for y, inner_y in ((10.75, 10.48), (9.25, 9.52)):
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(pcbnew.VECTOR2I(_mm(11.6225), _mm(y)))
        track.SetEnd(pcbnew.VECTOR2I(_mm(11.34), _mm(inner_y)))
        track.SetWidth(_mm(0.17))
        track.SetLayer(pcbnew.F_Cu)
        track.SetNet(gnd)
        board.Add(track)
    for x, y in via_positions:
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
        via.SetWidth(_mm(0.6))
        via.SetDrill(_mm(0.3))
        via.SetNet(gnd)
        board.Add(via)
    board.Save(path)


def test_thermal_ground_fine_pitch_zone_uses_2oz_fab_floor(tmp_path):
    from kicraft.autoplacer.brain.gnd_pour import audit_thermal_ground
    from kicraft.autoplacer.config import DEFAULT_CONFIG

    requirement = {"owner_ref": "U8", "ground_pins": ["8", "11", "25"],
                   "outer_copper_oz": 2.0, "local_radius_mm": 3.0,
                   "minimum_local_ground_area_mm2": 4.0}
    cfg = {**DEFAULT_CONFIG, "gnd_zone_net": "GND", "thermal_requirements": [requirement]}
    path = str(tmp_path / "thermal_fine_pitch.kicad_pcb")
    _fine_pitch_thermal_board(path)
    # The old 0.30 mm local clearance starves the narrow same-net pad even
    # though 0.17 mm is the enforced 2 oz fabrication clearance.
    pour_gnd_planes(path, {**cfg, "zone_clearance_mm": 0.3},
                    layers=("F.Cu", "B.Cu"))
    with pytest.raises(ValueError, match="no direct ground plane bond"):
        audit_thermal_ground(path, [requirement])

    # The same existing zones must adopt the resolved default, not retain
    # their old 0.30 mm clearance after regeneration.
    pour_gnd_planes(path, cfg, layers=("F.Cu", "B.Cu"))
    board = pcbnew.LoadBoard(path)
    assert all(z.GetLocalClearance() == _mm(0.17) for z in board.Zones())
    result = audit_thermal_ground(path, [requirement])
    assert {row["pad"] for row in result["pads"]} == {"8", "11", "25"}
    assert all(row["connected_vias"] and
               min(row["local_filled_area_mm2"].values()) >= 4.0
               for row in result["pads"])


def test_thermal_ground_rejects_missing_and_isolated_planes_natively(tmp_path):
    from kicraft.autoplacer.brain.gnd_pour import audit_thermal_ground

    requirement = {"owner_ref": "U8", "ground_pins": ["8", "11", "25"],
                   "outer_copper_oz": 2.0}
    cfg = {"gnd_zone_net": "GND", "thermal_requirements": [requirement]}
    path = str(tmp_path / "no_plane.kicad_pcb")
    _fine_pitch_thermal_board(path)
    with pytest.raises(ValueError, match="no direct ground plane bond"):
        audit_thermal_ground(path, [requirement])
    pour_gnd_planes(path, cfg, layers=("F.Cu",))
    with pytest.raises(ValueError, match="both faces"):
        audit_thermal_ground(path, [requirement])

    isolated = str(tmp_path / "isolated_nearby_planes.kicad_pcb")
    _fine_pitch_thermal_board(isolated, via_positions=((2.0, 2.0),))
    pour_gnd_planes(isolated, cfg, layers=("F.Cu", "B.Cu"))
    with pytest.raises(ValueError, match="both faces"):
        audit_thermal_ground(isolated, [requirement])


def _plane_stitch_board(path, *, blocked=False, hole_position=(5.8, 10)):
    _board(path, {"1": "GND"})
    board = pcbnew.LoadBoard(path)
    owner = next(iter(board.GetFootprints()))
    owner.SetReference("U8")
    gnd = board.GetNetInfo().GetNetItem("GND")

    # A remote plated anchor retains the B.Cu fill without supplying a local
    # via. An NPTH beside the thermal land makes the plane contain a real hole.
    anchor = pcbnew.FOOTPRINT(board)
    anchor.SetReference("J2")
    board.Add(anchor)
    for number, x, y, plated in (("1", 2, 2, True), ("2", *hole_position, False)):
        pad = pcbnew.PAD(anchor)
        pad.SetNumber(number)
        pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.8), _mm(0.8)))
        pad.SetDrillSize(pcbnew.VECTOR2I(_mm(0.4), _mm(0.4)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_PTH if plated else pcbnew.PAD_ATTRIB_NPTH)
        pad.SetLayerSet(pcbnew.PAD.PTHMask())
        if plated:
            pad.SetNet(gnd)
        anchor.Add(pad)

    # No radial dogbone can leave the pad, but copper pours remain permitted.
    # The negative case prohibits vias throughout the local thermal radius.
    ko = pcbnew.ZONE(board)
    ko.SetIsRuleArea(True)
    ko.SetDoNotAllowTracks(True)
    ko.SetDoNotAllowVias(True)
    ko.SetDoNotAllowCopperPour(False)
    layers = pcbnew.LSET()
    layers.AddLayer(pcbnew.F_Cu)
    layers.AddLayer(pcbnew.B_Cu)
    ko.SetLayerSet(layers)
    outline = ko.Outline()
    outline.NewOutline()
    half = 3.5 if blocked else 0.8
    for x, y in ((5-half, 10-half), (5+half, 10-half),
                 (5+half, 10+half), (5-half, 10+half)):
        outline.Append(_mm(x), _mm(y))
    board.Add(ko)
    board.Save(path)


@pytest.mark.parametrize("blocked", [False, True])
def test_thermal_plane_stitch_uses_native_overlap_and_guards(tmp_path, blocked):
    from kicraft.autoplacer.brain.gnd_pour import audit_thermal_ground

    path = str(tmp_path / "plane_stitch.kicad_pcb")
    requirement = {"owner_ref": "U8", "ground_pins": ["1"],
                   "local_radius_mm": 3.0, "minimum_local_ground_area_mm2": 4.0}
    cfg = {"gnd_zone_net": "GND", "thermal_requirements": [
        requirement, {"owner_ref": "ABSENT", "ground_pins": ["1"]}]}
    _plane_stitch_board(path, blocked=blocked)
    pour_gnd_planes(path, cfg, layers=("F.Cu", "B.Cu"))
    with pytest.raises(ValueError, match="both faces"):
        audit_thermal_ground(path, [requirement])

    result = add_gnd_pour_and_thermal_vias(path, cfg)
    if blocked:
        assert result["plane_stitches_added"] == 0
        with pytest.raises(ValueError, match="both faces"):
            audit_thermal_ground(path, [requirement])
        return

    assert result["plane_stitches_added"] == 1
    assert result["escape_stitched"] == 0
    board = pcbnew.LoadBoard(path)
    pad = next(p for fp in board.GetFootprints() if fp.GetReference() == "U8"
               for p in fp.Pads())
    vias = _gnd_vias(path)
    assert len(vias) == 1 and not pad.HitTest(vias[0].GetPosition())
    assert audit_thermal_ground(path, [requirement])["pads"][0]["connected_vias"] == 1
    before = [(t.GetClass(), t.GetPosition().x, t.GetPosition().y)
              for t in board.GetTracks()]
    again = add_gnd_pour_and_thermal_vias(path, cfg)
    assert again["plane_stitches_added"] == 0
    assert [(t.GetClass(), t.GetPosition().x, t.GetPosition().y)
            for t in pcbnew.LoadBoard(path).GetTracks()] == before

def test_parent_ground_repair_reconstructs_late_thermal_bond(tmp_path):
    from kicraft.autoplacer.brain.gnd_pour import (
        audit_thermal_ground, repair_parent_gnd_islands,
    )

    path = str(tmp_path / "late-thermal.kicad_pcb")
    requirement = {"owner_ref": "U8", "ground_pins": ["1"],
                   "local_radius_mm": 3.0, "minimum_local_ground_area_mm2": 4.0}
    cfg = {"gnd_zone_net": "GND", "thermal_requirements": [requirement],
           "gnd_edge_spine_enabled": False}
    _plane_stitch_board(path)
    pour_gnd_planes(path, cfg, layers=("F.Cu", "B.Cu"))
    with pytest.raises(ValueError, match="both faces"):
        audit_thermal_ground(path, [requirement])
    result = repair_parent_gnd_islands(path, cfg)
    assert result["thermal_stitch"]["plane_stitches_added"] == 1
    assert audit_thermal_ground(path, [requirement])["pads"][0]["connected_vias"] >= 1



def test_staged_thermal_stitch_respects_configured_hole_copper_clearance(tmp_path):
    from kicraft.autoplacer.brain.gnd_pour import audit_thermal_ground
    from kicraft.autoplacer.kicad_routing_tools import (
        _stamp_local_ground_keepouts, _strip_router_keepouts,
    )

    path = tmp_path / "hole-clearance.kicad_pcb"
    _plane_stitch_board(str(path), hole_position=(4, 8.5))
    requirement = {"owner_ref": "U8", "ground_pins": ["1"],
                   "local_radius_mm": 3, "minimum_local_ground_area_mm2": 4}
    cfg = {"gnd_zone_net": "GND", "thermal_requirements": [requirement],
           "hole_clearance_min_mm": 1.2, "via_size_mm": 0.6, "via_drill_mm": 0.3}
    name = "configured hole clearance"
    _stamp_local_ground_keepouts(path, cfg, name, router_clearance_mm=0.17)
    _strip_router_keepouts(path, name)
    board = pcbnew.LoadBoard(str(path))
    hole = next(p for fp in board.GetFootprints() if fp.GetReference() == "J2"
                for p in fp.Pads() if p.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH)
    for via in _gnd_vias(str(path)):
        delta = via.GetPosition() - hole.GetPosition()
        minimum = _mm(1.2) + via.GetWidth(pcbnew.F_Cu) // 2 + hole.GetDrillSize().x // 2
        assert delta.x ** 2 + delta.y ** 2 >= minimum ** 2
    assert audit_thermal_ground(str(path), [requirement])["pads"][0]["connected_vias"] == 1


@pytest.mark.parametrize("state", ["absent", "inadequate", "qualified"])
def test_leaf_thermal_gate_uses_owned_native_geometry(tmp_path, state):
    from kicraft.autoplacer.brain.leaf_acceptance import evaluate_leaf_acceptance
    from kicraft.autoplacer.brain.leaf_routing import _validate_leaf_thermal_requirements

    path = str(tmp_path / "leaf-thermal.kicad_pcb")
    _plane_stitch_board(path)
    requirement = {"owner_ref": "ABSENT" if state == "absent" else "U8",
                   "ground_pins": ["1"], "local_radius_mm": 3,
                   "minimum_local_ground_area_mm2": 4}
    cfg = {"gnd_zone_net": "GND", "thermal_requirements": [requirement]}
    if state == "qualified":
        add_gnd_pour_and_thermal_vias(path, cfg)
    validation = {"accepted": True, "rejection_reasons": [], "board_exists": True,
                  "drc": {"shorts": 0, "unconnected": 0, "unconnected_nets": []},
                  "track_summary": {"traces": 1, "vias": 1}}
    _validate_leaf_thermal_requirements(path, cfg, validation)
    result = evaluate_leaf_acceptance(validation, {})
    assert validation["accepted"] is (state != "inadequate")
    assert result.gate_results["thermal_contract"]["passed"] is (state != "inadequate")
    assert ("thermal_contract" in result.rejection_reasons) is (state == "inadequate")


def test_late_thermal_zone_reclaims_power_pour_without_changing_tracks(tmp_path):
    from kicraft.autoplacer.brain.gnd_pour import (
        audit_thermal_ground, finish_local_thermal_ground,
    )

    path = str(tmp_path / "competing-plane.kicad_pcb")
    _board(path, {"1": "GND", "2": "VBUS"})
    board = pcbnew.LoadBoard(path)
    fp = next(iter(board.GetFootprints()))
    fp.SetReference("U1")
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(5.8), _mm(10)))
    via.SetWidth(_mm(0.6))
    via.SetDrill(_mm(0.3))
    via.SetNet(board.FindNet("GND"))
    board.Add(via)
    board.Save(path)
    requirement = {"owner_ref": "U1", "ground_pins": ["1"],
                   "local_radius_mm": 3, "minimum_local_ground_area_mm2": 4}
    cfg = {"gnd_zone_net": "GND", "thermal_requirements": [requirement]}
    pour_gnd_planes(path, cfg, layers=("F.Cu", "B.Cu"))
    pour_power_planes(path, cfg, layers=("F.Cu",))
    with pytest.raises(ValueError):
        audit_thermal_ground(path, [requirement])
    before = [(t.m_Uuid.AsString(), t.GetClass()) for t in pcbnew.LoadBoard(path).GetTracks()]
    assert finish_local_thermal_ground(path, cfg) == 1
    result = audit_thermal_ground(path, [requirement])
    assert min(result["pads"][0]["local_filled_area_mm2"].values()) >= 4
    assert [(t.m_Uuid.AsString(), t.GetClass()) for t in pcbnew.LoadBoard(path).GetTracks()] == before
    assert finish_local_thermal_ground(path, cfg) == 0

    def owned():
        return {z.GetZoneName() for z in pcbnew.LoadBoard(path).Zones()
                if z.GetZoneName().startswith("kicraft-thermal-ground:")}

    original = owned()
    assert len(original) == 1
    requirement["local_radius_mm"] = 2.5
    finish_local_thermal_ground(path, cfg)
    assert not original & owned()
    previous = owned()
    board = pcbnew.LoadBoard(path)
    pad = next(p for p in board.FindFootprintByReference("U1").Pads()
               if p.GetNumber() == "1")
    pad.SetPosition(pcbnew.VECTOR2I(_mm(5.2), _mm(10)))
    board.Save(path)
    finish_local_thermal_ground(path, cfg)
    assert not previous & owned()
    finish_local_thermal_ground(path, {**cfg, "thermal_requirements": []})
    assert not owned()
    assert any(z.GetNetname() == "VBUS" for z in pcbnew.LoadBoard(path).Zones())


def test_overlapping_local_thermal_zones_have_distinct_priorities(tmp_path):
    from kicraft.autoplacer.brain.gnd_pour import finish_local_thermal_ground

    path = str(tmp_path / "multiple-thermal-pads.kicad_pcb")
    _board(path, {"1": "GND", "2": "GND", "3": "VBUS"})
    board = pcbnew.LoadBoard(path)
    next(iter(board.GetFootprints())).SetReference("U1")
    board.Save(path)
    cfg = {"gnd_zone_net": "GND", "thermal_requirements": [
        {"owner_ref": "U1", "ground_pins": ["1", "2"], "local_radius_mm": 3,
         "minimum_local_ground_area_mm2": 4}]}
    pour_gnd_planes(path, cfg, layers=("F.Cu", "B.Cu"))
    pour_power_planes(path, cfg, layers=("F.Cu",))
    assert finish_local_thermal_ground(path, cfg) == 2
    zones = [z for z in pcbnew.LoadBoard(path).Zones()
             if z.GetZoneName().startswith("kicraft-thermal-ground:")]
    assert len(zones) == 2
    assert len({z.GetAssignedPriority() for z in zones}) == 2
    assert finish_local_thermal_ground(path, cfg) == 0


def test_ground_anchor_detour_joins_around_foreign_track(tmp_path):
    from kicraft.autoplacer.brain.gnd_pour import (
        _collect_net_clusters, _repair_ground_anchor_detours,
    )

    path = str(tmp_path / "ground-detour.kicad_pcb")
    _board(path, {"1": "GND", "2": "GND"})
    board = pcbnew.LoadBoard(path)
    signal = pcbnew.NETINFO_ITEM(board, "SIG")
    board.Add(signal)
    obstacle = pcbnew.PCB_TRACK(board)
    obstacle.SetStart(pcbnew.VECTOR2I(_mm(6.5), _mm(9.7)))
    obstacle.SetEnd(pcbnew.VECTOR2I(_mm(6.5), _mm(10.3)))
    obstacle.SetWidth(_mm(0.2))
    obstacle.SetLayer(pcbnew.F_Cu)
    obstacle.SetNet(signal)
    board.Add(obstacle)
    obstacle_id = obstacle.m_Uuid.AsString()
    board.Save(path)
    assert len(_collect_net_clusters(board, "GND")[0]) == 2
    assert _repair_ground_anchor_detours(path, {"gnd_zone_net": "GND"}) == 1
    result = pcbnew.LoadBoard(path)
    assert len(_collect_net_clusters(result, "GND")[0]) == 1
    assert any(t.m_Uuid.AsString() == obstacle_id for t in result.GetTracks())


@pytest.mark.parametrize("blocked", [False, True])
def test_parent_ground_repair_stitches_interior_overlap_or_reports_unresolved(tmp_path, blocked):
    from kicraft.autoplacer.brain.gnd_pour import repair_parent_gnd_islands

    path = str(tmp_path / "islands.kicad_pcb")
    _board(path, {"1": "GND", "2": "GND"})
    board = pcbnew.LoadBoard(path)
    net = board.FindNet("GND")
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(5.8), _mm(10)))
    via.SetWidth(_mm(0.6))
    via.SetDrill(_mm(0.3))
    via.SetNet(net)
    board.Add(via)
    for layer, bounds in ((pcbnew.F_Cu, (4, 9, 6.5, 11)),
                          (pcbnew.F_Cu, (7, 9, 9, 11)),
                          (pcbnew.B_Cu, (4, 8, 10, 12))):
        zone = pcbnew.ZONE(board)
        zone.SetLayer(layer)
        zone.SetNet(net)
        zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
        zone.Outline().NewOutline()
        x1, y1, x2, y2 = bounds
        for x, y in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
            zone.Outline().Append(_mm(x), _mm(y))
        board.Add(zone)
    if blocked:
        keepout = pcbnew.ZONE(board)
        keepout.SetLayer(pcbnew.F_Cu)
        keepout.SetIsRuleArea(True)
        keepout.SetDoNotAllowVias(True)
        keepout.SetDoNotAllowTracks(True)
        keepout.Outline().NewOutline()
        for x, y in ((6.5, 8), (10, 8), (10, 12), (6.5, 12)):
            keepout.Outline().Append(_mm(x), _mm(y))
        board.Add(keepout)
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(path)
    summary = repair_parent_gnd_islands(
        path, {"gnd_zone_net": "GND", "gnd_edge_spine_enabled": False,
               "gnd_parent_repair_max_iter": 2})
    assert (summary["unresolved"] == 0) is (not blocked)
    if not blocked:
        assert summary["vias"] >= 1
    else:
        assert not summary["converged"]


def test_thermal_finisher_recreates_both_faces_after_zone_clear(tmp_path):
    from kicraft.autoplacer.brain.gnd_pour import audit_thermal_ground

    path = str(tmp_path / "rebuilt-thermal.kicad_pcb")
    _plane_stitch_board(path)
    requirement = {"owner_ref": "U8", "ground_pins": ["1"],
                   "local_radius_mm": 3, "minimum_local_ground_area_mm2": 4}
    cfg = {"gnd_zone_net": "GND", "thermal_requirements": [requirement]}
    first = add_gnd_pour_and_thermal_vias(path, cfg)
    assert first["plane_stitches_added"] == 1
    before = [(v.GetPosition().x, v.GetPosition().y, v.GetWidth(pcbnew.F_Cu),
               v.GetDrillValue()) for v in _gnd_vias(path)]
    board = pcbnew.LoadBoard(path)
    for zone in list(board.Zones()):
        if not zone.GetIsRuleArea():
            board.Remove(zone)
    board.Save(path)
    rebuilt = add_gnd_pour_and_thermal_vias(path, cfg)
    assert rebuilt["plane_stitches_added"] == 0
    assert [(v.GetPosition().x, v.GetPosition().y, v.GetWidth(pcbnew.F_Cu),
             v.GetDrillValue()) for v in _gnd_vias(path)] == before
    result = audit_thermal_ground(path, [requirement])
    assert result["pads"][0]["connected_vias"] == 1
    assert all(area >= 4 for area in result["pads"][0]["local_filled_area_mm2"].values())


def test_thermal_finisher_restitches_after_signal_splits_front_fill(tmp_path):
    from kicraft.autoplacer.brain.gnd_pour import audit_thermal_ground

    path = str(tmp_path / "split-front.kicad_pcb")
    _board(path, {"1": "GND"})
    board = pcbnew.LoadBoard(path)
    owner = next(iter(board.GetFootprints()))
    owner.SetReference("U1")
    next(iter(owner.Pads())).SetPosition(pcbnew.VECTOR2I(_mm(10), _mm(10)))
    signal = pcbnew.NETINFO_ITEM(board, "SIG")
    board.Add(signal)
    terminals = pcbnew.FOOTPRINT(board)
    terminals.SetReference("J2")
    board.Add(terminals)
    for number, y in (("1", 5), ("2", 15)):
        pad = pcbnew.PAD(terminals)
        pad.SetNumber(number)
        pad.SetSize(pcbnew.VECTOR2I(_mm(0.6), _mm(0.6)))
        pad.SetPosition(pcbnew.VECTOR2I(_mm(11.35), _mm(y)))
        pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
        pad.SetLayerSet(pcbnew.PAD.SMDMask())
        pad.SetNet(signal)
        terminals.Add(pad)
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(pcbnew.VECTOR2I(_mm(12), _mm(10)))
    via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    via.SetWidth(_mm(0.6))
    via.SetDrill(_mm(0.3))
    via.SetNet(board.FindNet("GND"))
    board.Add(via)
    hole = pcbnew.ZONE(board)
    hole.SetLayer(pcbnew.F_Cu)
    hole.SetIsRuleArea(True)
    hole.SetDoNotAllowTracks(False)
    hole.SetDoNotAllowVias(False)
    hole.SetDoNotAllowCopperPour(True)
    hole.Outline().NewOutline()
    for x, y in ((10.8, 9.5), (11.2, 9.5), (11.2, 10.5), (10.8, 10.5)):
        hole.Outline().Append(_mm(x), _mm(y))
    board.Add(hole)
    board.Save(path)
    requirement = {"owner_ref": "U1", "ground_pins": ["1"], "local_radius_mm": 3,
                   "minimum_local_ground_area_mm2": 4, "outer_copper_oz": 2}
    cfg = {"gnd_zone_net": "GND", "gnd_zone_margin_mm": 6,
           "thermal_requirements": [requirement]}
    # The existing via is qualified through fill around the hole, not a trace
    # or a straight pad-to-via corridor. It must not gain an unnecessary stitch.
    assert add_gnd_pour_and_thermal_vias(path, cfg)["plane_stitches_added"] == 0
    assert audit_thermal_ground(path, [requirement])["pads"][0]["connected_vias"] == 1
    board = pcbnew.LoadBoard(path)
    crossing = pcbnew.PCB_TRACK(board)
    crossing.SetStart(pcbnew.VECTOR2I(_mm(11.35), _mm(5)))
    crossing.SetEnd(pcbnew.VECTOR2I(_mm(11.35), _mm(15)))
    crossing.SetWidth(_mm(0.2))
    crossing.SetLayer(pcbnew.F_Cu)
    crossing.SetNet(board.FindNet("SIG"))
    board.Add(crossing)
    crossing_id = crossing.m_Uuid.AsString()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(path)
    with pytest.raises(ValueError, match="both faces"):
        audit_thermal_ground(path, [requirement])
    result = add_gnd_pour_and_thermal_vias(path, cfg)
    assert result["plane_stitches_added"] == 1
    assert audit_thermal_ground(path, [requirement])["pads"][0]["connected_vias"] == 1
    finished = pcbnew.LoadBoard(path)
    retained = next(t for t in finished.GetTracks()
                    if t.m_Uuid.AsString() == crossing_id)
    assert retained.GetNetname() == "SIG" and retained.GetLayer() == pcbnew.F_Cu
    assert retained.GetStart() == crossing.GetStart() and retained.GetEnd() == crossing.GetEnd()
    assert retained.GetWidth() == _mm(0.2)


def test_thermal_audit_does_not_count_disconnected_vias(tmp_path, monkeypatch):
    from kicraft.autoplacer.brain import gnd_pour

    path = str(tmp_path / "thermal.kicad_pcb")
    _sot23_board(path)
    class Fill:
        def UnitSet(self, index):
            return self

        def Area(self):
            return float(pcbnew.FromMM(1)) ** 2 * 10

        def Contains(self, point, index):
            return True

    nodes = [
        {"kind": "pad", "ref": "U1", "num": "2", "xy": (1.0, 1.0),
         "layers": {pcbnew.F_Cu}, "pts": [(1.0, 1.0)]},
        {"kind": "island", "layers": {pcbnew.F_Cu}, "fill": Fill(), "idx": 0},
        {"kind": "island", "layers": {pcbnew.B_Cu}, "fill": Fill(), "idx": 0},
        {"kind": "via", "xy": (2.0, 1.0)},
    ]
    monkeypatch.setattr(gnd_pour, "_collect_net_clusters",
                        lambda board, net: ({0: [0, 1], 2: [2, 3]}, nodes))
    req = [{"owner_ref": "U1", "ground_pins": ["2"]}]
    with pytest.raises(ValueError, match="both faces"):
        gnd_pour.audit_thermal_ground(path, req)
    monkeypatch.setattr(gnd_pour, "_collect_net_clusters",
                        lambda board, net: ({0: [0, 1, 2, 3]}, nodes))
    monkeypatch.setattr(gnd_pour, "_local_island_area", lambda *args: 10.0)
    assert gnd_pour.audit_thermal_ground(path, req)["pads"][0]["connected_vias"] == 1
    with pytest.raises(ValueError, match="pads missing"):
        gnd_pour.audit_thermal_ground(path, [{"owner_ref": "U1", "ground_pins": ["2", "25"]}])
    nodes[3]["xy"] = (20.0, 1.0)
    with pytest.raises(ValueError, match="both faces"):
        gnd_pour.audit_thermal_ground(path, req)
    nodes[3]["xy"] = (2.0, 1.0)
    monkeypatch.setattr(gnd_pour, "_local_island_area", lambda *args: 0.1)
    with pytest.raises(ValueError, match="insufficient local ground copper"):
        gnd_pour.audit_thermal_ground(path, req)


