"""Required title space must coexist with close-coupled edge interfaces."""
import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.placement_solver import PlacementSolver, local_contract_violations
from kicraft.autoplacer.brain.types import Point
from kicraft.autoplacer.hardware.adapter import KiCadAdapter


def _point(x, y):
    return pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))


@pytest.mark.parametrize("edge,rotation", [("bottom", 0), ("right", 90), ("top", 180), ("left", 270)])
def test_required_usb_title_preserves_three_mm_coupling_at_each_edge(tmp_path, edge, rotation):
    path = tmp_path / "interface.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    for net in ("USB_DP", "USB_DM"):
        board.Add(pcbnew.NETINFO_ITEM(board, net))
    for ref, x, y, width, height, pad_y in (
        ("J1", 10, 12, 4, 8, 9.5), ("U1", 30, 12, 0.8, 0.8, 12),
    ):
        fp = pcbnew.FOOTPRINT(board)
        fp.SetReference(ref)
        fp.SetPosition(_point(x, y))
        board.Add(fp)
        shape = pcbnew.PCB_SHAPE(fp)
        shape.SetShape(pcbnew.SHAPE_T_RECT)
        shape.SetStart(_point(x - width / 2, y - height / 2))
        shape.SetEnd(_point(x + width / 2, y + height / 2))
        shape.SetLayer(pcbnew.F_CrtYd)
        shape.SetWidth(pcbnew.FromMM(0.05))
        fp.Add(shape)
        for pin, dx, net in (("1", -0.25, "USB_DP"), ("2", 0.25, "USB_DM")):
            pad = pcbnew.PAD(fp)
            pad.SetNumber(pin)
            pad.SetShape(pcbnew.PAD_SHAPE_RECT)
            pad.SetSize(_point(0.3, 0.6))
            pad.SetPosition(_point(x + dx, pad_y))
            pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
            pad.SetLayerSet(pcbnew.PAD.SMDMask())
            pad.SetNet(board.GetNetInfo().GetNetItem(net))
            fp.Add(pad)
    board.Save(str(path))
    specs = [{"owner_ref": "J1", "bypass_ref": "U1", "pin_pairs": [["1", "1"], ["2", "2"]],
              "max_distance_mm": 3.0}]
    config = {
        "local_bypasses": specs,
        "silk_label_reservations": {"J1": {"text": "USB PROGRAM\nBOARD POWER SEPARATE", "pin_texts": []}},
        "component_zones": {"J1": {"edge": edge, "rotation": rotation}},
        "edge_margin_mm": 2.0, "connector_edge_inset_mm": 1.0,
        "legality_clearance_mm": 0.2, "courtyard_overlap_min_gap_mm": 0.2,
        "edge_jitter_mm": 0.0,
    }
    state = KiCadAdapter(str(path), config).load()
    state.board_outline = (Point(0, 0), Point(60, 40))
    solver = PlacementSolver(state, config)
    solver._pin_edge_components(state.components)
    pinned = state.components["J1"].pos
    solver._reserve_local_bypasses(state.components)
    assert state.components["J1"].pos == pinned
    assert local_contract_violations(state, specs) == []
    low, high = state.components["J1"].physical_bbox()
    assert low.x >= (1 if edge == "left" else 2) - 1e-9
    assert low.y >= (1 if edge == "top" else 2) - 1e-9
    assert high.x <= 60 - (1 if edge == "right" else 2) + 1e-9
    assert high.y <= 40 - (1 if edge == "bottom" else 2) + 1e-9
    mate_low, mate_high = state.components["U1"].physical_bbox()
    assert 2 <= mate_low.x <= mate_high.x <= 58
    assert 2 <= mate_low.y <= mate_high.y <= 38
    assert (mate_high.x + 0.2 <= low.x + 1e-9 or high.x + 0.2 <= mate_low.x + 1e-9
            or mate_high.y + 0.2 <= low.y + 1e-9 or high.y + 0.2 <= mate_low.y + 1e-9)
