"""Regression coverage for native AH49E Hall acquisition ownership."""
from copy import deepcopy

import pytest

from kicraft.design.models import RecipeSelection
from kicraft.design.recipes import expand_recipe
from kicraft.design.recipes.resolver import resolve_architecture_recipes


RECIPE = "ah49entr-g1-analog-hall@1"


def _selection(**changes):
    data = dict(recipe=RECIPE, instance="hall1", sheets={"sensor": "HALL"},
                requirement_ids=["hall1"], port_bindings={"vdd": "+3V3", "gnd": "GND", "out": "SENSOR_1_OUT"})
    data.update(changes)
    return RecipeSelection(**data)


def _architecture():
    return {
        "topologies": {"SENSING": "Four linear analog Hall sensors"},
        "rail_voltages": {"+3V3": 3.3, "GND": 0.0, "VBUS": 5.0},
        "power_nets": ["+3V3", "GND", "VBUS"], "mcu_present": False,
        "sheets": [{"name": "HALL", "stem": "HALL", "function": "Hall sensing"}],
        "inter_sheet_nets": [],
        "requirements": [
            {"id": f"hall{i}", "sheet": "HALL", "role": "sensor", "family": "hall-effect",
             "exact_part": "AH49ENTR-G1", "parameters": {},
             "ports": {"vdd": "+3V3", "gnd": "GND", "out": f"SENSOR_{i}_OUT"},
             "declared_interface": {"ports": [
                 {"key": "vdd", "direction": "power", "function": "3.3 V sensor supply", "pin": "1"},
                 {"key": "gnd", "direction": "power", "function": "sensor ground", "pin": "2"},
                 {"key": "out", "direction": "output", "function": "linear analog output", "pin": "3"},
             ]}}
            for i in range(1, 5)
        ],
    }


def test_four_stale_declared_pinmaps_lower_to_native_circuit():
    architecture = _architecture()
    original = deepcopy(architecture)
    result = resolve_architecture_recipes(architecture)
    assert not result.blocking
    assert architecture == original
    for i in range(1, 5):
        selection = next(row for row in result.selections if row.requirement_ids == [f"hall{i}"])
        expansion = expand_recipe(selection)
        sensor = next(p for p in expansion.parts if p.recipe_role == "sensor")
        cap = next(p for p in expansion.parts if p.recipe_role == "decoupling")
        nets = {(ep.ref, ep.pin): connection.net_name for connection in expansion.connections for ep in connection.endpoints}
        assert sensor.mpn == "AH49ENTR-G1"
        assert [nets[sensor.ref, str(pin)] for pin in (1, 2, 3)] == ["+3V3", f"SENSOR_{i}_OUT", "GND"]
        assert [nets[cap.ref, str(pin)] for pin in (1, 2)] == ["+3V3", "GND"]


def test_hall_uses_manufacturer_landpattern_and_native_pin_truth():
    import pcbnew
    from kicraft.design.synthesis.footprint_library import load_footprint
    from kicraft.design.synthesis.symbol_pinout import lookup_pins

    pins = lookup_pins("diodes-ah49entr-g1:AH49ENTR-G1")["pins"]
    assert {pin["number"]: pin["name"] for pin in pins} == {"1": "VCC", "2": "OUT", "3": "GND"}
    fp, _ = load_footprint(pcbnew, "diodes-ah49entr-g1", "SOT-23-3_Diodes_AH49E")
    pads = {p.GetNumber(): (pcbnew.ToMM(p.GetPosition().x), pcbnew.ToMM(p.GetPosition().y),
                            pcbnew.ToMM(p.GetSize().x), pcbnew.ToMM(p.GetSize().y)) for p in fp.Pads()}
    assert pads == {"1": (1.3, 0.95, 1.0, 0.7), "2": (1.3, -0.95, 1.0, 0.7),
                    "3": (-1.3, 0.0, 1.0, 0.7)}


@pytest.mark.parametrize("vdd,rail_voltage", [("VBUS", 5.0), ("MYSTERY", None), ("+3V3", 2.9), ("+3V3", 3.6)])
def test_hall_supply_requires_authoritative_compatible_rail(vdd, rail_voltage):
    payload = _architecture()
    payload["requirements"] = payload["requirements"][:1]
    payload["requirements"][0]["ports"]["vdd"] = vdd
    if rail_voltage is not None:
        payload["rail_voltages"][vdd] = rail_voltage
    result = resolve_architecture_recipes(payload)
    assert result.blocking
    assert not any(row.recipe == RECIPE for row in result.selections)


@pytest.mark.parametrize("port", ["vdd", "gnd"])
def test_expansion_refuses_shorted_analog_output(port):
    ports = {"vdd": "+3V3", "gnd": "GND", "out": "SENSOR_1_OUT"}
    ports["out"] = ports[port]
    with pytest.raises(ValueError, match="distinct"):
        expand_recipe(_selection(port_bindings=ports))


def test_digital_hall_is_not_replaced_by_analog():
    payload = _architecture()
    payload["requirements"] = payload["requirements"][:1]
    sensor = payload["requirements"][0]
    sensor["exact_part"] = None
    sensor["declared_interface"]["ports"][2]["function"] = "digital Hall switch output"
    result = resolve_architecture_recipes(payload)
    assert not any(row.recipe == RECIPE for row in result.selections)


def test_different_exact_hall_ordering_code_is_not_substituted():
    payload = _architecture()
    payload["requirements"] = payload["requirements"][:1]
    payload["requirements"][0]["exact_part"] = "AH49FNTR-G1"
    result = resolve_architecture_recipes(payload)
    assert not any(row.recipe == RECIPE for row in result.selections)


def test_hall_bypass_is_exported_as_owned_physical_proximity(tmp_path):
    import json
    from kicraft.design.models import Architecture, BOM
    from kicraft.design.synthesis.autoplacer import write_autoplacer_json

    selection = _selection()
    expansion = expand_recipe(selection)
    architecture = Architecture(
        sheets=[{"name": "HALL", "stem": "HALL", "function": "Hall sensing"}],
        power_nets=["+3V3", "GND"], inter_sheet_nets=[],
        requirements=[_architecture()["requirements"][0]],
        recipe_selections=[selection],
    )
    bom = BOM(parts=expansion.parts, connections=expansion.connections,
              recipe_ownership=[expansion.ownership])
    cfg = json.loads(write_autoplacer_json(tmp_path, "HALL", architecture, bom).read_text())
    assert cfg["local_bypasses"] == [{
        "owner_ref": "U1", "bypass_ref": "C1", "power_pins": ["1", "1"],
        "ground_pins": [["3"], "2"], "max_distance_mm": 3.0,
    }]
