"""PAM8302A supply placement and native ground-return proof."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

pcbnew = pytest.importorskip("pcbnew")

from kicraft.autoplacer.brain.placement_solver import PlacementSolver, local_contract_violations
from kicraft.autoplacer.brain.types import Point
from kicraft.autoplacer.hardware.adapter import KiCadAdapter
from kicraft.autoplacer.hardware.local_contract import audit_local_copper
from kicraft.design.models import BOM, RecipeSelection
from kicraft.design.recipes.registry import expand_recipe
from kicraft.design.synthesis.autoplacer import _recipe_local_bypasses
from kicraft.parts_library.footprint_courtyard import courtyard_pad_clearance_mm


_ROOT = Path(__file__).resolve().parents[1]
_AMP_LIB = _ROOT / "kicraft/parts_library/pam8302a/pam8302a.pretty"
_AMP_NAME = "SOP-8_L4.9-W3.9-P1.27-LS6.0-BL"
_CAP_LIB = Path("/usr/share/kicad/footprints/Capacitor_SMD.pretty")


def _point(x, y):
    return pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y))


def _outline(board):
    corners = [(0, 0), (60, 0), (60, 60), (0, 60)]
    for start, end in zip(corners, corners[1:] + corners[:1]):
        edge = pcbnew.PCB_SHAPE(board)
        edge.SetShape(pcbnew.SHAPE_T_SEGMENT)
        edge.SetLayer(pcbnew.Edge_Cuts)
        edge.SetStart(_point(*start))
        edge.SetEnd(_point(*end))
        board.Add(edge)


def test_pam8302a_joint_caps_require_actual_ground_copper(tmp_path):
    selection = RecipeSelection(
        recipe="pam8302a-pwm-speaker@1", instance="speaker",
        sheets={"interface": "INTERFACE"},
        port_bindings={"vdd": "VDD", "gnd": "GND", "pwm": "PWM"},
    )
    expansion = expand_recipe(selection)
    bom = BOM(parts=expansion.parts, connections=expansion.connections,
              recipe_ownership=[expansion.ownership])
    specs = _recipe_local_bypasses(SimpleNamespace(recipe_selections=[selection]), bom)
    roles = {part.recipe_role: part.ref for part in expansion.parts}
    owner = roles["amplifier"]
    cap_refs = [roles["supply_decoupling"], roles["supply_bulk"]]
    assert len(specs) == 2
    for spec in specs:
        assert spec["owner_ref"] == owner
        assert spec["bypass_ref"] in cap_refs
        assert spec["pin_pairs"] == [["6", "1"]]
        assert spec["max_distance_mm"] == 3.0
        assert spec["ground_return"] == {
            "owner_ref": owner, "owner_pin": "7", "member_pin": "2",
            "net": "GND", "max_via_distance_mm": 3.0,
        }

    path = tmp_path / "audio.kicad_pcb"
    board = pcbnew.NewBoard(str(path))
    for name in ("VDD", "GND"):
        board.Add(pcbnew.NETINFO_ITEM(board, name))
    _outline(board)
    definitions = [
        (owner, _AMP_LIB, _AMP_NAME, (25, 25)),
        (cap_refs[0], _CAP_LIB, "C_0603_1608Metric", (40, 40)),
        (cap_refs[1], _CAP_LIB, "C_1206_3216Metric", (45, 40)),
    ]
    for ref, library, name, position in definitions:
        fp = pcbnew.FootprintLoad(str(library), name)
        assert fp is not None
        fp.SetReference(ref)
        fp.SetPosition(_point(*position))
        fp.InvalidateGeometryCaches()
        if ref == owner:
            assert courtyard_pad_clearance_mm(fp) >= 0.25 - 1e-4
            fp.SetLocked(True)
        board.Add(fp)
        for pad in fp.Pads():
            net = ("VDD" if pad.GetNumber() == "6" else
                   "GND" if pad.GetNumber() == "7" else None) if ref == owner else (
                   "VDD" if pad.GetNumber() == "1" else "GND")
            if net:
                pad.SetNet(board.GetNetInfo().GetNetItem(net))
    pcbnew.SaveBoard(str(path), board)

    adapter = KiCadAdapter(str(path), {"unlock_all_footprints": False})
    state = adapter.load()
    state.board_outline = (Point(0, 0), Point(60, 60))
    solver = PlacementSolver(state, {"local_bypasses": specs, "legality_clearance_mm": 0.2})
    solver._reserve_local_bypasses(state.components)
    assert state.components[owner].pos == Point(25, 25)
    assert local_contract_violations(state, specs) == []
    components = [state.components[ref] for ref in [owner, *cap_refs]]
    owner_pads = {pad.pad_id: pad for pad in components[0].pads}
    for cap in components[1:]:
        pads = {pad.pad_id: pad for pad in cap.pads}
        assert owner_pads["6"].pos.dist(pads["1"].pos) <= 3.0
    for i, component in enumerate(components):
        at, ab = component.bbox(0.1)
        for other in components[i + 1:]:
            bt, bb = other.bbox(0.1)
            assert not (at.x < bb.x and ab.x > bt.x and at.y < bb.y and ab.y > bt.y)

    # Transfer the chosen poses to the real footprints, not copies of wiring
    # metadata: the native audit must fail without copper and pass after fill.
    for component in components:
        fp = board.FindFootprintByReference(component.ref)
        fp.SetPosition(_point(component.pos.x, component.pos.y))
        fp.SetOrientationDegrees(component.rotation)
    assert any("ground return" in error for error in audit_local_copper(board, specs))
    zone = pcbnew.ZONE(board)
    zone.SetNet(board.GetNetInfo().GetNetItem("GND"))
    zone.SetLayer(pcbnew.F_Cu)
    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
    chain = pcbnew.SHAPE_LINE_CHAIN()
    for x, y in ((1, 1), (59, 1), (59, 59), (1, 59)):
        chain.Append(pcbnew.FromMM(x), pcbnew.FromMM(y))
    chain.SetClosed(True)
    zone.Outline().AddOutline(chain)
    board.Add(zone)
    board.BuildConnectivity()
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    assert audit_local_copper(board, specs) == []
