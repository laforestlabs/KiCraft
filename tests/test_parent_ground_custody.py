"""Ground-only custody must not reject a later guarded copper repair."""
from __future__ import annotations

import shutil
from types import SimpleNamespace

import pytest

pcbnew = pytest.importorskip("pcbnew")
pytestmark = pytest.mark.skipif(not shutil.which("kicad-cli"), reason="native DRC requires kicad-cli")

from kicraft.cli._compose_route import _route_parent_board


def _point(x, y):
    return pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))


def _parent(tmp_path, *, illegal_stub):
    path = tmp_path / "parent_stamped.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    for net in ("PWR", "GND"):
        board.Add(pcbnew.NETINFO_ITEM(board, net))
    for a, b in zip(((0, 0), (20, 0), (20, 20), (0, 20)),
                    ((20, 0), (20, 20), (0, 20), (0, 0)), strict=True):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetStart(_point(*a))
        edge.SetEnd(_point(*b))
        edge.SetWidth(pcbnew.FromMM(0.05))
        edge.SetLayer(pcbnew.Edge_Cuts)
        board.Add(edge)
    for ref, x, ground_pin in (("U1", 5, "4"), ("C1", 7, "2")):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(_point(x, 5))
        fp.Reference().SetPosition(_point(x, 8))
        fp.Value().SetPosition(_point(x, 9))
        board.Add(fp)
        for pin, y, net in (("1", 5, "PWR"), (ground_pin, 6, "GND")):
            pad = pcbnew.PAD(fp)
            pad.SetNumber(pin)
            pad.SetShape(pcbnew.PAD_SHAPE_CIRCLE)
            pad.SetSize(_point(0.6, 0.6))
            pad.SetPosition(_point(x, y))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNet(board.GetNetInfo().GetNetItem(net))
            fp.Add(pad)
    segments = [("PWR", (5, 5), (7, 5)), ("GND", (5, 6), (7, 6))]
    if illegal_stub:
        segments.append(("PWR", (5, 5), (5, -2)))
    for net, start, end in segments:
        track = pcbnew.PCB_TRACK(board)
        track.SetStart(_point(*start))
        track.SetEnd(_point(*end))
        track.SetWidth(pcbnew.FromMM(0.5))
        track.SetLayer(pcbnew.F_Cu)
        track.SetNet(board.GetNetInfo().GetNetItem(net))
        board.Add(track)
    via = pcbnew.PCB_VIA(board)
    via.SetPosition(_point(6, 6))
    via.SetLayerPair(pcbnew.F_Cu, pcbnew.B_Cu)
    via.SetDrill(pcbnew.FromMM(0.3))
    try:
        via.SetWidth(pcbnew.FromMM(0.6))
    except TypeError:
        via.SetWidth(pcbnew.F_Cu, pcbnew.FromMM(0.6))
    via.SetNet(board.GetNetInfo().GetNetItem("GND"))
    board.Add(via)
    pcbnew.SaveBoard(str(path), board)
    cfg = {
        "gnd_zone_net": "GND", "power_plane_enabled": False,
        "gnd_strand_repair_enabled": False,
        "signal_unconnected_repair_enabled": False,
        "power_strand_repair_enabled": False,
        "local_bypasses": [{
            "owner_ref": "U1", "bypass_ref": "C1",
            "power_pins": ["1", "1"], "max_distance_mm": 3.0,
            "max_route_length_mm": 6.0, "min_route_width_mm": 0.5,
            "ground_return": {"owner_ref": "U1", "owner_pin": "4",
                              "member_pin": "2", "net": "GND",
                              "max_via_distance_mm": 3.0},
        }],
    }
    return path, cfg


@pytest.fixture
def passthrough_router(monkeypatch):
    # Routing itself is outside this phase-boundary regression; all finishing,
    # copper import, geometry repair and native DRC below are real.
    def route(source, output, cfg):
        shutil.copy2(source, output)
        return {"backend": "fixture"}
    monkeypatch.setattr(
        "kicraft.autoplacer.kicad_routing_tools.route_with_kicad_routing_tools", route)


def test_guarded_geometry_repair_does_not_look_like_ground_copper_loss(tmp_path, passthrough_router):
    stamped, cfg = _parent(tmp_path, illegal_stub=True)
    result = _route_parent_board(stamped, SimpleNamespace(composition=object()), tmp_path, cfg)
    validation = result["validation"]
    assert validation["accepted"], validation
    assert validation["illegal_geometry_repair"]["kept"]
    assert validation["drc"]["unconnected"] == 0
    assert validation["drc"]["shorts"] == 0
    assert all(group["missing_count"] == 0
               for group in validation["ground_finish_custody"].values())
    board = pcbnew.LoadBoard(result["routed_board_path"])
    assert all(item.GetStart().y >= 0 and item.GetEnd().y >= 0
               for item in board.GetTracks() if not isinstance(item, pcbnew.PCB_VIA))


@pytest.mark.parametrize("kind", ["trace", "via"])
def test_destructive_ground_finisher_still_rejected(tmp_path, passthrough_router, monkeypatch, kind):
    from kicraft.autoplacer.brain import gnd_pour

    stamped, cfg = _parent(tmp_path, illegal_stub=False)
    finish = gnd_pour.add_gnd_pour_and_thermal_vias

    def lose_copper(path, config):
        original = pcbnew.LoadBoard(path)
        target_uuid = next(
            item.m_Uuid.AsString() for item in original.GetTracks()
            if isinstance(item, pcbnew.PCB_VIA) == (kind == "via")
        )
        result = finish(path, config)
        board = pcbnew.LoadBoard(path)
        item = next(item for item in board.GetTracks()
                    if item.m_Uuid.AsString() == target_uuid)
        board.Remove(item)
        pcbnew.SaveBoard(path, board)
        return result

    monkeypatch.setattr(gnd_pour, "add_gnd_pour_and_thermal_vias", lose_copper)
    with pytest.raises(RuntimeError, match="ground finishing lost routed copper"):
        _route_parent_board(stamped, SimpleNamespace(composition=object()), tmp_path, cfg)
