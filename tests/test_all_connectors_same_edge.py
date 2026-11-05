"""Hard all-connector edge constraints survive synthesis, including debug headers."""
import json

import pytest

from kicraft.design.models import Architecture, BOM, BomPart, IntentSlot, PlacementSection
from kicraft.design.synthesis.autoplacer import required_same_edge_connector_refs, write_autoplacer_json


def _part(ref, value, footprint):
    return BomPart(ref=ref, value=value, sheet="MAIN", symbol="Connector_Generic:Conn_01x02", footprint=footprint)


def _arch():
    return Architecture(sheets=[{"name": "MAIN", "stem": "MAIN", "function": "board"}], power_nets=[], inter_sheet_nets=[])


def _bom():
    return BOM(parts=[
        _part("J1", "SWD", "Connector_PinHeader_2.54mm:PinHeader_1x06_P2.54mm_Vertical"),
        _part("J2", "HOST", "Connector_PinSocket_2.54mm:PinSocket_1x02_P2.54mm_Vertical"),
        _part("J3", "FIELD", "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P"),
        _part("P1", "HEADER", "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"),
    ])


def test_same_edge_constraint_includes_internal_recipe_owned_debug_header(tmp_path):
    cfg = json.loads(write_autoplacer_json(tmp_path, "BOARD", _arch(), _bom(),
        intent=IntentSlot(goal="Hall board", constraints=["All connectors located on one edge"])).read_text())
    assert cfg["required_same_edge_connectors"] == {"edge": "bottom", "refs": ["J1", "J2", "J3", "P1"]}
    assert all(cfg["component_zones"][ref]["edge"] == "bottom" for ref in ("J1", "J2", "J3", "P1"))


def test_explicit_edge_is_respected_but_conflicting_user_zones_fail(tmp_path):
    placement = PlacementSection(component_zones={"J1": {"edge": "left"}})
    cfg = json.loads(write_autoplacer_json(tmp_path, "BOARD", _arch(), _bom(),
        placement=placement, intent=IntentSlot(goal="board", constraints=["Put all the connectors on one edge"])).read_text())
    assert cfg["required_same_edge_connectors"]["edge"] == "left"
    assert all(cfg["component_zones"][ref]["edge"] == "left" for ref in ("J1", "J2", "J3", "P1"))
    conflict = PlacementSection(component_zones={"J1": {"edge": "left"}, "J2": {"edge": "right"}})
    with pytest.raises(ValueError, match="conflicts"):
        write_autoplacer_json(tmp_path, "BAD", _arch(), _bom(), placement=conflict,
            intent=IntentSlot(goal="board", constraints=["All connectors on the same edge"]))


def test_same_edge_is_not_inferred_from_absent_or_negated_constraint(tmp_path):
    for constraints in ([], ["Not all connectors on one edge"], ["Four screw terminals"]):
        cfg = json.loads(write_autoplacer_json(tmp_path, "BOARD", _arch(), _bom(),
            intent=IntentSlot(goal="board", constraints=constraints)).read_text())
        assert "required_same_edge_connectors" not in cfg
        assert "J1" not in cfg.get("component_zones", {})
    assert required_same_edge_connector_refs(IntentSlot(goal="board", constraints=["All connectors on one edge"]), BOM(parts=[])) == ()
    assert required_same_edge_connector_refs(None, _bom()) is None
