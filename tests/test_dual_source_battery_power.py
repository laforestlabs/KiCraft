import pytest

from kicraft.design.lowering import lower_requirement
from kicraft.design.models import CircuitRequirement


_PARAMS = {
    "host_voltage_v": 5.0,
    "host_voltage_min_v": 4.75,
    "host_voltage_max_v": 5.25,
    "battery_chemistry": "single-cell-liion",
    "battery_voltage_min_v": 3.0,
    "battery_voltage_max_v": 4.2,
    "load_current_a": 0.5,
    "include_host_connector": True,
}


def _requirement(**updates):
    values = {
        "id": "dual_source",
        "sheet": "POWER",
        "role": "power_input",
        "family": "dual-source-battery-power",
        "parameters": _PARAMS,
        "ports": {
            "host_input": "HOST_5V",
            "battery_input": "BATTERY_RAW",
            "output": "TTL_5V",
            "gnd": "GND",
        },
    }
    values.update(updates)
    return CircuitRequirement(**values)


def _pins(artifact):
    return {(pin.role, pin.pin): pin.net for pin in artifact.pins}


def test_dual_source_power_has_independent_physical_inputs_and_real_battery_path():
    artifact = lower_requirement(_requirement())
    groups = {group.role: group for group in artifact.groups}
    pins = _pins(artifact)

    assert groups["battery_connector"].mpn == "S2B-PH-SM4-TB(LF)(SN)"
    assert groups["battery_boost"].mpn == "TPS61088RHLR"
    assert groups["power_mux"].mpn == "TPS2113APWR"
    assert groups["power_mux"].footprint == "tps2113apwr:TSSOP-8_3x4p4mm_P0.65mm"
    assert pins["host_connector", "1"] == "HOST_5V"
    assert pins["battery_connector", "1"] == "BATTERY_RAW"
    assert pins["host_connector", "1"] != pins["battery_connector", "1"]
    assert pins["battery_boost", "9"] == "BATTERY_RAW"
    assert pins["power_mux", "8"] == "HOST_5V"
    assert pins["power_mux", "6"] != "HOST_5V"
    assert pins["power_mux", "7"] == "TTL_5V"
    assert groups["boost_output_cap"].mpn == "6TPE100MAZB"
    assert groups["boost_output_cap"].quantity == 3


def test_dual_source_power_has_mux_control_and_no_passive_source_tie():
    artifact = lower_requirement(_requirement())
    pins = _pins(artifact)

    assert pins["power_mux", "2"] == pins["mux_enable_pulldown", "1"]
    assert pins["enable_jumper", "1"] == pins["power_mux", "2"]
    disable_high = pins["enable_jumper", "2"]
    assert disable_high == pins["host_enable_diode", "1"] == pins["battery_enable_diode", "1"]
    assert pins["host_enable_diode", "2"] == "HOST_5V"
    assert pins["battery_enable_diode", "2"] == pins["power_mux", "6"]
    assert disable_high not in {"HOST_5V", pins["power_mux", "6"], "TTL_5V", "GND"}
    assert pins["power_mux", "3"] == pins["host_priority_top", "2"]
    assert pins["power_mux", "3"] == pins["host_priority_bottom", "1"]
    assert pins["power_mux", "8"] != pins["power_mux", "6"]
    assert all(
        {pin.net for pin in artifact.pins if pin.role == role} != {"HOST_5V", "BATTERY_RAW"}
        for role in ("boost_inductor", "host_priority_top", "host_priority_bottom")
    )


def test_dual_source_can_bind_one_existing_exact_host_header_without_duplicate():
    artifact = lower_requirement(_requirement(parameters={**_PARAMS, "include_host_connector": False}))
    assert "host_connector" not in {group.role for group in artifact.groups}
    assert all(pin.role != "host_connector" for pin in artifact.pins)
    from kicraft.design.models import RecipeSelection
    from kicraft.design.recipes import expand_recipe

    expanded = expand_recipe(RecipeSelection(
        recipe="dual-source-battery-power@1", instance="sources",
        sheets={"power": "POWER"},
        parameters={"include_host_connector": False},
        port_bindings=_requirement().ports,
    ))
    assert not any(part.recipe_role == "host_connector" for part in expanded.parts)



def test_dual_source_power_refuses_shorting_sources_or_output_at_contract_boundary():
    for bad_ports in (
        {"host_input": "POWER", "battery_input": "POWER", "output": "TTL_5V", "gnd": "GND"},
        {"host_input": "HOST_5V", "battery_input": "BATTERY_RAW", "output": "HOST_5V", "gnd": "GND"},
        {"host_input": "HOST_5V", "battery_input": "BATTERY_RAW", "output": "TTL_5V", "gnd": "TTL_5V"},
    ):
        with pytest.raises(ValueError):
            lower_requirement(_requirement(ports=bad_ports))


def test_dual_source_power_refuses_unreviewed_source_or_unsafe_host_tolerance():
    with pytest.raises(ValueError):
        lower_requirement(_requirement(parameters={**_PARAMS, "battery_chemistry": "cr2032"}))
    with pytest.raises(ValueError):
        lower_requirement(_requirement(parameters={**_PARAMS, "host_voltage_min_v": 4.5}))
    with pytest.raises(ValueError):
        lower_requirement(_requirement(exact_part="TPS2113ADRBR"))
