"""Hardware contract for the TPS26631 USB-PD series entry."""

import pytest
from pydantic import ValidationError

from kicraft.design.recipes import get_recipe
from kicraft.design.recipes.models import RecipeDefinition
from kicraft.design.synthesis.symbol_pinout import lookup_pins


RECIPE = "tps26631-pd-inrush-power-path@1"


def test_entry_is_a_distinct_series_path_with_exact_imported_contacts():
    recipe = get_recipe(RECIPE)
    assert recipe.pass_through and recipe.rated_output_current_a == 1.8
    assert recipe.supply_current_reservations_a == {"input": 0.003}
    assert {p.name: p.supply_voltage_range_v for p in recipe.ports}["input"] == (8.0, 22.0)
    pins = {pin.pin: pin.net for pin in recipe.pins if pin.role == "efuse"}
    assert {k: pins[k] for k in ("1", "2", "5", "6", "7", "8", "9", "10", "11", "15", "17", "18", "25")} == {
        "1": "input", "2": "input", "5": "input", "6": "uvlo", "7": "ovp",
        "8": "gnd", "9": "dvdt", "10": "ilim", "11": "gnd", "15": "pgth",
        "17": "output", "18": "output", "25": "gnd",
    }
    assert pins["16"] == "power_good"
    part = next(p for p in recipe.parts if p.role == "efuse")
    symbol = lookup_pins(part.symbol, all_units=True)
    names = {p["number"]: p["name"] for p in symbol["pins"]}
    assert {k: names[k] for k in ("1", "5", "6", "7", "9", "10", "15", "17", "25")} == {
        "1": "IN", "5": "IN_SYS", "6": "UVLO", "7": "OVP", "9": "dVdT",
        "10": "ILIM", "15": "PGTH", "17": "OUT", "25": "EP",
    }
    assert {p.pin for p in recipe.no_connects if p.role == "efuse"} == {
        "3", "4", "12", "13", "14", "19", "20", "21", "22", "23", "24"
    }


def test_power_good_holds_regulators_until_output_is_ready():
    recipe = get_recipe(RECIPE)
    pins = {(p.role, p.pin): p.net for p in recipe.pins}
    assert pins[("efuse", "15")] == "pgth"
    assert pins[("efuse", "16")] == "power_good"
    assert pins[("pgth_top", "1")] == "output"
    assert pins[("pgth_top", "2")] == pins[("pgth_bottom", "1")] == "pgth"
    assert pins[("pgth_bottom", "2")] == "gnd"
    assert pins[("pg_pullup", "1")] == "output"
    assert pins[("pg_pullup", "2")] == pins[("pg_pulldown", "1")] == "power_good"
    assert pins[("pg_pulldown", "2")] == "gnd"
    values = {p.role: p.value for p in recipe.parts}
    assert [values[k] for k in ("pgth_top", "pgth_bottom", "pg_pullup", "pg_pulldown")] == [
        "68k 1%", "13k 1%", "33k 1%", "33k 1%"
    ]
    # 1.7mA max IC + two 81k ladders + 66k PG divider at 22V.
    assert .0017 + 2 * 22 / (81000 * .99) + 22 / (66000 * .99) < .003


def test_threshold_and_inrush_design_corners():
    recipe = get_recipe(RECIPE)
    parts = {p.role: p for p in recipe.parts}
    assert [parts[k].value for k in ("threshold_top", "threshold_middle", "threshold_bottom", "current_limit")] == [
        "68k 1%", "9.1k 1%", "3.9k 1%", "9k 1%"
    ]
    assert parts["slew_cap"].value.startswith("220nF 50V X7R")
    # Include an additional 10mV at VIN for the 150nA pin leakage terms.
    assert 1.09 * (1 + 68 * .99 / (13 * 1.01)) - .01 > 5.5
    assert 1.224 * (1 + 68 * 1.01 / (13 * .99)) + .01 < 8.0
    assert 1.176 * (1 + (68 + 9.1) * .99 / (3.9 * 1.01)) - .02 > 22.0
    assert 9 * 220e-9 * .7 / (2.225e-6 * 26) > .020
    assert 250e-6 * (2.225e-6 * 26) / (220e-9 * .7) < .095
    # Datasheet 9k endpoints, scaled with the documented 18/R formula.
    assert 1.84 / 1.01 > 1.8
    assert 2.16 / .99 < 3


def test_pass_through_requires_both_power_ports():
    recipe = get_recipe(RECIPE)
    for missing in ("input", "output"):
        data = recipe.model_dump()
        data["ports"] = [p for p in data["ports"] if p["name"] != missing]
        with pytest.raises(ValidationError, match="input and output power ports"):
            RecipeDefinition.model_validate(data)
    data = recipe.model_dump()
    next(p for p in data["ports"] if p["name"] == "output")["direction"] = "input"
    with pytest.raises(ValidationError, match="input and output power ports"):
        RecipeDefinition.model_validate(data)


@pytest.mark.parametrize(("parameter", "value"), [
    ("uvlo_rising_v_nominal", 5.0),
    ("uvlo_rising_v_nominal", 9.0),
    ("ovp_rising_v_nominal", 20.0),
    ("current_limit_a_nominal", 3.5),
    ("max_capacitive_load_uf", 300.0),
])
def test_unsupported_controls_are_rejected(parameter, value):
    from kicraft.design.models import RecipeSelection
    from kicraft.design.recipes.registry import expand_recipe

    selection = RecipeSelection(
        recipe=RECIPE, instance="entry", sheets={"power": "Power"},
        parameters={parameter: value},
        port_bindings={"input": "VBUS_RAW", "output": "VBUS_PROTECTED", "gnd": "GND"},
        requirement_ids=["entry"],
    )
    with pytest.raises(ValueError, match="must be one of"):
        expand_recipe(selection)
