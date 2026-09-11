"""Electrical contracts for the resistor-selected, PD-only CH224K block."""

import pytest

from kicraft.design.models import RecipeSelection
from kicraft.design.recipes.registry import expand_recipe


def _selection(**parameters):
    return RecipeSelection(
        recipe="ch224k-pd-selectable@1",
        instance="pd",
        sheets={"power": "PD"},
        port_bindings={"cc1": "CC1", "cc2": "CC2", "vbus": "VBUS", "gnd": "GND"},
        parameters=parameters,
    )


def _circuit():
    expansion = expand_recipe(_selection())
    parts = {part.recipe_role: part for part in expansion.parts}
    nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    return expansion, parts, nets


@pytest.mark.parametrize(
    "throw,expected_resistance", [("1", "6.8k 1%"), ("3", "24k 1%"), ("4", None)]
)
def test_each_physical_selector_position_presents_datasheet_resistance(throw, expected_resistance):
    expansion, parts, nets = _circuit()
    controller = parts["controller"].ref
    selector = parts["selector"].ref
    # Manufacturer drawing: the long common contact is physical pad 2.
    # Close only the selected mechanical contact, then inspect all resistor
    # paths from CFG1 to ground rather than assuming either resistor is wired.
    common = nets[(selector, "2")]
    assert nets[(controller, "9")] == common
    selected = nets.get((selector, throw))
    reachable = {common, selected} - {None}
    resistances = []
    for part in expansion.parts:
        if part.symbol != "Device:R":
            continue
        ends = {nets[(part.ref, "1")], nets[(part.ref, "2")]}
        if "GND" in ends and ends & reachable:
            resistances.append(part.value)
    assert resistances == ([] if expected_resistance is None else [expected_resistance])
    if throw == "4":
        assert (selector, "4") in {(pin.ref, pin.pin) for pin in expansion.no_connect_pins}


def test_pd_only_cc_has_no_parallel_termination_and_all_package_pins_are_owned():
    expansion, parts, nets = _circuit()
    controller = parts["controller"].ref
    selector = parts["selector"].ref
    assert {key for key, net in nets.items() if net == "CC1"} == {(controller, "7")}
    assert {key for key, net in nets.items() if net == "CC2"} == {(controller, "6")}
    assert nets[(controller, "4")] == nets[(controller, "5")]
    nc = {(pin.ref, pin.pin) for pin in expansion.no_connect_pins}
    assert {(controller, pin) for pin in ("2", "3", "10")} <= nc
    assert nets[(controller, "11")] == "GND"
    assert nets[(selector, "5")] == nets[(selector, "6")] == "GND"
    assert not set(nets) & nc
    assert {pin for ref, pin in set(nets) | nc if ref == controller} == {
        str(i) for i in range(1, 12)
    }
    assert {pin for ref, pin in set(nets) | nc if ref == selector} == {str(i) for i in range(1, 7)}


@pytest.mark.parametrize(
    "parameters",
    [
        {"voltage_options": "5/9/15V"},
        {"selection_mode": "logic"},
        {"selector_part": "SW_SPDT"},
        {"output_voltage": 12},
    ],
)
def test_selectable_recipe_refuses_unsupported_electrical_configuration(parameters):
    with pytest.raises(ValueError):
        expand_recipe(_selection(**parameters))


def test_selectable_recipe_requires_both_external_cc_connections():
    selection = _selection()
    selection.port_bindings.pop("cc2")
    with pytest.raises(ValueError, match="port bindings mismatch"):
        expand_recipe(selection)


def test_fixed_recipe_cannot_claim_selectable_or_twelve_volt_operation():
    selection = RecipeSelection(
        recipe="ch224k-pd-trigger@1",
        instance="fixed",
        sheets={"power": "PD"},
        port_bindings={"vbus": "VBUS", "gnd": "GND"},
        parameters={"output_voltage": 12},
    )
    with pytest.raises(ValueError):
        expand_recipe(selection)
