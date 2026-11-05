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
        ("tps54331-adjustable@2", 3.3, 18.0),
        ("tps54331-adjustable@2", 3.3, 24.0),
        ("tps54331-12v-1a@2", 12.0, 18.0),
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


@pytest.mark.parametrize(("role", "pin"), [
    ("reverse_fet", "1"), ("reverse_fet", "2"), ("reverse_fet", "3"),
    ("reverse_controller", "1"), ("reverse_controller", "3"),
    ("reverse_controller", "5"), ("reverse_controller", "6"),
])
def test_motor_current_evidence_requires_actual_reverse_boundary(role, pin):
    parts, names, pads, outputs = _qualified_recipe("k7803m-3v3-reverse-blocked@1", 3.3)
    facts = qualified_converter_power_stage_currents(parts, names, pads, outputs, _bound_source(24))
    assert [row["power_stage_current_a"] for row in facts] == [0.5]
    part = next(p for p in parts if p["recipe_role"] == role)
    pads[part["ref"], pin] = "BROKEN"
    assert not qualified_converter_power_stage_currents(parts, names, pads, outputs, _bound_source(24))


def test_motor_current_evidence_rejects_parallel_bypass_and_wrong_manufacturer():
    parts, names, pads, outputs = _qualified_recipe("k7803m-3v3-reverse-blocked@1", 3.3)
    converter = next(p for p in parts if p["recipe_role"] == "converter")
    raw = pads[converter["ref"], "3"]
    assert qualified_converter_power_stage_currents(parts, names, pads, outputs, _bound_source(24))
    parts.append({"ref": "R999", "symbol": "Device:R", "value": "0R"})
    pads["R999", "1"] = raw
    pads["R999", "2"] = "VOUT"
    assert not qualified_converter_power_stage_currents(parts, names, pads, outputs, _bound_source(24))
    parts.pop()
    fet = next(p for p in parts if p["recipe_role"] == "reverse_fet")
    fet["sourcing_note"] = "LCSC C999999"
    assert not qualified_converter_power_stage_currents(parts, names, pads, outputs, _bound_source(24))


@pytest.mark.parametrize(("voltage", "accepted"), [
    (5.99, False), (6.0, True), (24.0, True), (29.99, True), (30.0, False),
])
def test_protected_motor_current_retains_module_input_envelope(voltage, accepted):
    parts, names, pads, outputs = _qualified_recipe("k7803m-3v3-reverse-blocked@1", 3.3)
    facts = qualified_converter_power_stage_currents(parts, names, pads, outputs,
                                                      _bound_source(voltage))
    assert bool(facts) is accepted
