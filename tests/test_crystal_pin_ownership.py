import pytest

from kicraft.design.models import RecipeSelection
from kicraft.design.recipes import expand_recipe, get_recipe
from kicraft.design.synthesis.symbol_pinout import lookup_pins


@pytest.mark.parametrize(
    ("recipe", "crystal_role", "mcu_input", "mcu_output", "series_role"),
    [
        ("stm32f103c8t6-minimal@1", "hse_crystal", "5", "6", None),
        ("rp2040-minimal@2", "crystal", "20", "21", "xout_resistor"),
    ],
)
def test_four_pad_crystal_keeps_oscillator_off_grounded_case(
    recipe, crystal_role, mcu_input, mcu_output, series_role
):
    # Abracon ABM8 top view: resonator 1/3, grounded case 2/4.
    definition = get_recipe(recipe)
    expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe,
            instance="core",
            sheets={role: role.upper() for role in definition.required_sheet_roles},
            port_bindings={"vdd": "BOARD_3V3", "gnd": "BOARD_GND"},
        )
    )
    parts = {part.recipe_role: part for part in expansion.parts}
    crystal = parts[crystal_role]
    mcu = parts["mcu"]
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    assert {str(pin["number"]) for pin in lookup_pins(crystal.symbol)["pins"]} == {
        "1",
        "2",
        "3",
        "4",
    }
    assert nets[crystal.ref, "2"] == nets[crystal.ref, "4"] == "BOARD_GND"
    assert nets[crystal.ref, "1"] == nets[mcu.ref, mcu_input]
    assert len({nets[crystal.ref, "1"], nets[crystal.ref, "3"], "BOARD_GND"}) == 3
    if series_role is None:
        assert nets[crystal.ref, "3"] == nets[mcu.ref, mcu_output]
    else:
        series = parts[series_role]
        assert nets[series.ref, "1"] == nets[mcu.ref, mcu_output]
        assert nets[series.ref, "2"] == nets[crystal.ref, "3"]
