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
    _detect_power_nets,
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


# ---------------------------------------------------------------------------
# GND thermal-via stitcher: collision guard + small-pad escape stitching
# ---------------------------------------------------------------------------

from kicraft.autoplacer.brain.gnd_pour import add_gnd_pour_and_thermal_vias  # noqa: E402


def _sot23_board(path, *, blocker_track=False):
    """A 3-pad SOT-23-shaped U1 whose GND pad (0.5 mm -- too small for an
    in-pad 0.6 mm via) sits at (10,10), plus a 1.3 mm via-fitting GND pad on a
    6-pad U2 at (15,15). Optionally a foreign B.Cu track right under U2's pad
    (the IP2368-bank shorts shape: a thermal via stamped blind through it)."""
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
        board.Add(fp)
        for num, (netname, x, y, w, h) in pads.items():
            pad = pcbnew.PAD(fp)
            pad.SetSize(pcbnew.VECTOR2I(_mm(w), _mm(h)))
            pad.SetPosition(pcbnew.VECTOR2I(_mm(x), _mm(y)))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNumber(num)
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
    # U2's via-fitting GND pad still gets its in-pad via.
    assert res["thermal_vias_added"] >= 1, res
    vias = _gnd_vias(path)
    assert len(vias) >= 2
    # The escape via is OFF the small pad (10,10), not drilled through it.
    assert all(
        (pcbnew.ToMM(v.GetPosition().x), pcbnew.ToMM(v.GetPosition().y))
        != (pytest.approx(10.0), pytest.approx(10.0))
        for v in vias
    )


def test_thermal_via_not_stamped_through_foreign_track(tmp_path):
    # Regression for the IP2368-bank incident: _add_via had no collision check
    # and stamped GND vias straight through routed B.Cu tracks (7 shorts). A
    # foreign B.Cu track under U2's GND pad must block that via.
    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path, blocker_track=True)
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
    assert res == {"nets": [], "stranded": 0, "tied": 0, "skipped": []}
    board = pcbnew.LoadBoard(path)
    assert not [t for t in board.GetTracks()
                if not isinstance(t, pcbnew.PCB_VIA) and t.GetNetname() == "5V"]


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
    _sot23_board(build)
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
    _sot23_board(path)
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
    # The KC-UXASHQ C2.2 strand: a decoupling cap's 1.0 mm GND pad could host
    # an in-pad via with nothing blocking, but the old multipad>=3 gate
    # skipped 2-pad passives entirely -> B.Cu-only pour never reached it.
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

    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["gnd_pads_stitched"] >= 1, res
    assert any(
        (pcbnew.ToMM(v.GetPosition().x), pcbnew.ToMM(v.GetPosition().y))
        == (pytest.approx(6.0), pytest.approx(8.0))
        for v in _gnd_vias(path)
    )


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
    board.Save(path)

    res = add_gnd_pour_and_thermal_vias(path, {"gnd_zone_net": "GND"})
    assert res["escape_stitched"] == 0, res


def test_gnd_escape_specs_targets_only_small_gnd_pads(tmp_path):
    # Pre-route spec gen: U1's 0.5 mm GND pad (no in-pad via possible) gets a
    # via_at_end escape spec; U2's via-fitting 1.3 mm GND pad does not.
    path = str(tmp_path / "b.kicad_pcb")
    _sot23_board(path)
    board = pcbnew.LoadBoard(path)
    specs = gnd_escape_specs(board, {"gnd_zone_net": "GND"})
    assert [(s.ref, s.pad) for s in specs] == [("U1", "2")]
    assert all(s.via_at_end for s in specs)
    assert gnd_escape_specs(board, {"gnd_zone_net": "GND",
                                    "gnd_pre_escape": False}) == []
