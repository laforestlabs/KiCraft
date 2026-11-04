"""Consumer-visible regressions for bounded converter power-stage evidence."""
from __future__ import annotations

import pytest

from kicraft.design.models import RecipeSelection
from kicraft.design.recipes import expand_recipe
from kicraft.design.synthesis.symbol_pinout import lookup_pins
from kicraft.eval.converter_current_evidence import qualified_converter_power_stage_currents


def _qualified_recipe(recipe: str, voltage: float):
    expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe,
            instance="regulated",
            sheets={"power": "POWER"},
            port_bindings={"input": "VIN", "output": "VOUT", "gnd": "GND"},
        )
    )
    parts = [part.model_dump(mode="json") for part in expansion.parts]
    pad_nets = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }
    names = {
        (part.ref, str(pin["number"])): str(pin.get("name") or "")
        for part in expansion.parts
        for pin in lookup_pins(part.symbol, all_units=True)["pins"]
    }
    converter = next(part for part in expansion.parts if part.recipe_role == "converter")
    outputs = [{
        "ref": converter.ref,
        "input_net": "VIN",
        "output_net": "VOUT",
        "ground_net": "GND",
        # This is the precondition normally supplied by the board-topology
        # observer, not an architecture rail label.
        "voltage": voltage,
    }]
    return parts, names, pad_nets, outputs

def _bound_source(voltage: float):
    return lambda rail, ground: voltage if (rail, ground) == ("VIN", "GND") else None


@pytest.mark.parametrize(
    ("recipe", "output_voltage", "input_voltage"),
    (
        ("tps54331-adjustable@1", 3.3, 18.0),
        ("tps54331-adjustable@1", 3.3, 24.0),
        ("tps54331-12v-1a@1", 12.0, 18.0),
    ),
)
def test_fxl_max_and_typical_test_points_do_not_prove_a_current_rating(
    recipe, output_voltage, input_voltage
):
    parts, names, pad_nets, outputs = _qualified_recipe(recipe, output_voltage)
    assert qualified_converter_power_stage_currents(
        parts, names, pad_nets, outputs, _bound_source(input_voltage)
    ) == []


@pytest.mark.parametrize(
    ("input_voltage", "expected"),
    ((5.99, None), (6.0, 0.5), (18.0, 0.5), (24.0, 0.5),
     (29.99, 0.5), (30.0, None)),
)
def test_integrated_module_current_is_bounded_by_supported_input(
    input_voltage, expected
):
    parts, names, pad_nets, outputs = _qualified_recipe("ylptec-k7803m-1000r3@1", 3.3)
    facts = qualified_converter_power_stage_currents(
        parts, names, pad_nets, outputs, _bound_source(input_voltage)
    )
    assert [fact["power_stage_current_a"] for fact in facts] == (
        [] if expected is None else [expected]
    )


@pytest.mark.parametrize("role", ("input_cap", "output_cap"))
def test_integrated_module_with_opened_support_is_not_qualified(role):
    parts, names, pad_nets, outputs = _qualified_recipe("ylptec-k7803m-1000r3@1", 3.3)
    cap = next(part for part in parts if part["recipe_role"] == role)
    pad_nets[cap["ref"], "2"] = "FLOATING_RETURN"
    assert qualified_converter_power_stage_currents(
        parts, names, pad_nets, outputs, _bound_source(24.0)
    ) == []
