from copy import deepcopy

import pytest

from kicraft.design.models import RecipeSelection
from kicraft.design.recipes import expand_recipe
from kicraft.eval.dual_source_power_evidence import observe_dual_source_battery_power


def _fixture():
    expanded = expand_recipe(RecipeSelection(
        recipe="dual-source-battery-power@1", instance="sources",
        sheets={"power": "POWER"},
        port_bindings={"host_input": "HOST", "battery_input": "RAW", "output": "OUT", "gnd": "GND"},
    ))
    parts = [part.model_dump(mode="json") for part in expanded.parts]
    pads = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expanded.connections for endpoint in connection.endpoints
    }
    refs = {}
    for part in parts:
        refs.setdefault(part["recipe_role"], []).append(part["ref"])
    return parts, pads, refs


def _source(part, positive, ground, source_class, minimum, maximum):
    return (source_class, minimum, maximum, positive, ground) in {
        ("host_5v", 4.75, 5.25, "HOST", "GND"),
        ("single_cell_liion", 3.0, 4.2, "RAW", "GND"),
    }


def test_observer_proves_actual_recipe_sources_and_distinct_mux_output():
    parts, pads, _ = _fixture()
    result = observe_dual_source_battery_power(parts, pads, source_voltage=_source)
    assert result is not None
    assert (result.output_net, result.ground_net) == ("OUT", "GND")
    assert (result.minimum_output_voltage_v, result.maximum_output_voltage_v) == (4.675, 5.25)
    assert observe_dual_source_battery_power(parts, pads, source_voltage=lambda *args: False) is None

@pytest.mark.parametrize("left,right", [
    ("HOST", "OUT"), ("HOST", "RAW"), ("OUT", "GND"),
])
def test_zero_ohm_bypass_does_not_prove_isolated_power(left, right):
    parts, pads, refs = _fixture()
    assert observe_dual_source_battery_power(parts, pads, source_voltage=_source) is not None
    bypass = deepcopy(next(part for part in parts if part["ref"] == refs["mux_current_limit"][0]))
    bypass.update(ref="R_BYPASS", value="0R")
    parts.append(bypass)
    pads["R_BYPASS", "1"], pads["R_BYPASS", "2"] = left, right
    assert observe_dual_source_battery_power(parts, pads, source_voltage=_source) is None


def test_series_zero_ohm_bypass_does_not_prove_isolation():
    parts, pads, refs = _fixture()
    template = next(part for part in parts if part["ref"] == refs["mux_current_limit"][0])
    for ref, left, right in (("R_BYPASS1", "HOST", "BRIDGE"), ("R_BYPASS2", "BRIDGE", "OUT")):
        bypass = deepcopy(template)
        bypass.update(ref=ref, value="0R")
        parts.append(bypass)
        pads[ref, "1"], pads[ref, "2"] = left, right
    assert observe_dual_source_battery_power(parts, pads, source_voltage=_source) is None


@pytest.mark.parametrize("role,pin", [
    ("battery_connector", "1"), ("host_connector", "1"),
    ("power_mux", "2"), ("mux_current_limit", "2"),
    ("battery_boost", "15"), ("battery_boost", "21"),
    ("boost_output_cap", "1"), ("boost_comp_cap", "2"),
    ("battery_boost", "2"), ("battery_boost", "3"), ("battery_boost", "10"),
    ("battery_boost", "13"), ("battery_boost", "19"),
    ("host_enable_diode", "2"), ("battery_enable_diode", "2"),
])
def test_open_source_support_or_control_cannot_pass(role, pin):
    parts, pads, refs = _fixture()
    pads[refs[role][0], pin] = "OPEN"
    assert observe_dual_source_battery_power(parts, pads, source_voltage=_source) is None


@pytest.mark.parametrize("role", ["boost_output_cap", "host_enable_diode", "battery_enable_diode"])
def test_reversed_polarized_parts_cannot_pass(role):
    parts, pads, refs = _fixture()
    ref = refs[role][0]
    pads[ref, "1"], pads[ref, "2"] = pads[ref, "2"], pads[ref, "1"]
    assert observe_dual_source_battery_power(parts, pads, source_voltage=_source) is None


@pytest.mark.parametrize("role", ["power_mux", "boost_output_cap", "boost_comp_cap"])
def test_claimed_mpn_with_wrong_delivered_package_cannot_pass(role):
    parts, pads, refs = _fixture()
    damaged = deepcopy(parts)
    next(part for part in damaged if part["ref"] == refs[role][0])["footprint"] = "Resistor_SMD:R_0603_1608Metric"
    assert observe_dual_source_battery_power(damaged, pads, source_voltage=_source) is None
