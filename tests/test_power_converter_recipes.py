"""Focused structural tests for reviewed converter recipes.

These exercise recipe topology before artifact-level electrical observers inspect a routed board.
"""

from __future__ import annotations

import pytest

from kicraft.design.models import RecipeSelection
from kicraft.design.recipes import expand_recipe
from kicraft.design.recipes.registry import get_recipe
from kicraft.design.recipes.wave_b_power import enable_control_max_voltage
from kicraft.design.recipes.resolver import resolve_architecture_recipes


def _endpoint_nets(expansion):
    return {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections
        for endpoint in connection.endpoints
    }


@pytest.mark.parametrize(
    ("quantity", "relation", "value", "unit", "blocked"),
    [
        ("output current", "equal", 1.0, "A", False),
        ("output current", "minimum", 1001, "mA", True),
        ("3.3 V supply current capacity", "minimum", 1.5, "A", True),
        ("output current", "maximum", 2.0, "A", False),
        ("input current", "equal", 2.0, "A", False),
        ("inductor current", "equal", 2.0, "A", False),
        ("output current per channel", "equal", 2.0, "A", False),
    ],
)
def test_recipe_output_current_uses_owned_typed_load_only(quantity, relation, value, unit, blocked):
    requirement = {
        "id": "reg", "sheet": "POWER", "role": "regulator",
        "family": "ap63203-3v3", "exact_part": "AP63203WU-7",
        "ports": {"input": "VIN", "output": "VOUT", "gnd": "GND"},
        "obligations": [{
            "kind": "quantitative", "original_obligation_id": "load-current",
            "quantity": quantity, "relation": relation, "value": value, "unit": unit,
        }],
    }
    architecture = {
        "sheets": [{"name": "POWER", "stem": "POWER", "function": "power"}],
        "power_nets": ["VIN", "VOUT", "GND"],
        "inter_sheet_nets": [],
        "requirements": [requirement],
    }
    result = resolve_architecture_recipes(architecture)
    errors = [d for d in result.blocking if d.code == "recipe_output_current_exceeded"]
    assert bool(errors) is blocked
    if blocked:
        assert errors[0].requirement_id == "reg"
        assert "obligation=load-current" in errors[0].evidence
        assert "rated_output_current_a=1" in errors[0].evidence
        assert not result.selections
    else:
        assert any(row.recipe == "ap63203-3v3@1" for row in result.selections)


@pytest.mark.parametrize(("maximum", "blocked"), [(1.0, False), (1.1, True)])
def test_recipe_output_current_range_uses_upper_requested_load(maximum, blocked):
    architecture = {
        "sheets": [{"name": "POWER", "stem": "POWER", "function": "power"}],
        "power_nets": ["VIN", "VOUT", "GND"],
        "inter_sheet_nets": [],
        "requirements": [{
            "id": "reg", "sheet": "POWER", "role": "regulator",
            "family": "ap63203-3v3",
            "ports": {"input": "VIN", "output": "VOUT", "gnd": "GND"},
            "obligations": [{
                "kind": "quantitative", "original_obligation_id": "load-range",
                "quantity": "load current", "relation": "range",
                "minimum": 0, "maximum": maximum, "unit": "A",
            }],
        }],
    }
    result = resolve_architecture_recipes(architecture)
    assert any(d.code == "recipe_output_current_exceeded" for d in result.blocking) is blocked


@pytest.mark.parametrize(("load_a", "blocked"), [(0.5, False), (0.501, True)])
def test_tps54331_3v3_composition_respects_load_boundary(load_a, blocked):
    architecture = {
        "sheets": [{"name": "POWER", "stem": "POWER", "function": "power"}],
        "power_nets": ["VIN", "VOUT", "GND"],
        "inter_sheet_nets": [],
        "requirements": [{
            "id": "reg", "sheet": "POWER", "role": "regulator",
            "family": "tps54331-adjustable",
            "ports": {"input": "VIN", "output": "VOUT", "gnd": "GND"},
            "obligations": [{
                "kind": "quantitative", "original_obligation_id": "load-current",
                "quantity": "output current", "relation": "minimum",
                "value": load_a, "unit": "A",
            }],
        }],
    }
    result = resolve_architecture_recipes(architecture)
    assert any(d.code == "recipe_output_current_exceeded" for d in result.blocking) is blocked
    if not blocked:
        assert any(s.recipe == "tps54331-adjustable@2" for s in result.selections)


def test_tps54331_3v3_feedback_tolerance_stays_within_c3_supply_range():
    definition = get_recipe("tps54331-adjustable@2")
    parts = {part.role: part for part in definition.parts}
    top = float(parts["feedback_top"].value.split("k", 1)[0])
    bottom = float(parts["feedback_bottom"].value.split("k", 1)[0])
    low = 0.772 * (1 + top * 0.99 / (bottom * 1.01))
    high = 0.828 * (1 + top * 1.01 / (bottom * 0.99))
    assert 3.0 <= low < high <= 3.6

def _assert_tps54331_buck_topology(definition) -> None:
    """Validate topology from pins, not recipe/rail naming prose."""
    pins = {(pin.role, pin.index, pin.pin): pin.net for pin in definition.pins}

    assert pins[("converter", 0, "2")] == "input"
    assert pins[("converter", 0, "7")] == pins[("converter", 0, "9")] == "gnd"
    assert pins[("converter", 0, "8")] == pins[("inductor", 0, "1")] == "switch"
    # Device:D_Schottky pin 1 is cathode, so this is the required SW-cathode/GND-anode
    # freewheel orientation for the non-synchronous buck.
    assert pins[("catch_diode", 0, "1")] == "switch"
    assert pins[("catch_diode", 0, "2")] == "gnd"
    assert pins[("inductor", 0, "2")] == pins[("feedback_top", 0, "1")] == "output"
    assert pins[("feedback_top", 0, "2")] == pins[("feedback_bottom", 0, "1")] == pins[("converter", 0, "5")] == "feedback"
    assert pins[("feedback_bottom", 0, "2")] == "gnd"
    assert pins[("converter", 0, "1")] == pins[("bootstrap_cap", 0, "1")] == "bootstrap"
    assert pins[("bootstrap_cap", 0, "2")] == "switch"
    assert pins[("converter", 0, "6")] == pins[("comp_resistor", 0, "1")] == pins[("comp_hf_capacitor", 0, "1")] == "compensation"
    assert pins[("comp_resistor", 0, "2")] == pins[("comp_capacitor", 0, "1")] == "comp_rc"
    assert pins[("comp_capacitor", 0, "2")] == pins[("comp_hf_capacitor", 0, "2")] == "gnd"
    # Shunt fitted means OFF; the unpopulated shunt leaves EN floating for the IC's
    # internal pull-up.  It is never a VIN-to-EN raw high-voltage strap.
    assert pins[("converter", 0, "3")] == pins[("enable_jumper", 0, "1")] == "enable"
    assert pins[("enable_jumper", 0, "2")] == "gnd"


@pytest.mark.parametrize("recipe", ("tps54331-adjustable@2", "tps54331-12v-1a@2"))
def test_tps54331_recipes_have_complete_buck_and_safe_enable_topology(recipe):
    definition = get_recipe(recipe)
    _assert_tps54331_buck_topology(definition)

    enabled_expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe,
            instance="regulated",
            sheets={"power": "POWER"},
            parameters={"enable_control": True},
            port_bindings={"input": "VIN", "gnd": "GND", "output": "VOUT"},
        )
    )
    nets = _endpoint_nets(enabled_expansion)
    converter = next(part for part in enabled_expansion.parts if part.recipe_role == "converter")
    assert nets[(converter.ref, "2")] == "VIN"
    assert nets[(converter.ref, "7")] == nets[(converter.ref, "9")] == "GND"
    assert nets[(converter.ref, "3")].startswith("__KICRAFT_RECIPE__")

    default_expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe,
            instance="default",
            sheets={"power": "POWER"},
            port_bindings={"input": "VIN", "gnd": "GND", "output": "VOUT"},
        )
    )
    default_converter = next(
        part for part in default_expansion.parts if part.recipe_role == "converter"
    )
    assert (default_converter.ref, "3") in {
        (pin.ref, pin.pin) for pin in default_expansion.no_connect_pins
    }
    assert not any(part.recipe_role == "enable_jumper" for part in default_expansion.parts)


def test_tps54331_12v_feedback_math_is_connected_and_exact_nominally():
    definition = get_recipe("tps54331-12v-1a@2")
    parts = {part.role: part for part in definition.parts}
    r_top = float(parts["feedback_top"].value.split("k", 1)[0]) * 1_000
    r_bottom = float(parts["feedback_bottom"].value.split("k", 1)[0]) * 1_000
    assert 0.8 * (1 + r_top / r_bottom) == pytest.approx(12.0)
    _assert_tps54331_buck_topology(definition)



def test_enable_control_voltage_limits_are_part_specific():
    # AP6320x documents EN as a high-voltage input; TPS54331 does not.
    assert enable_control_max_voltage("ap63203-3v3@1") == 32.0
    assert enable_control_max_voltage("ap63205-5v@2") == 32.0
    assert enable_control_max_voltage("tps54331-adjustable@2") == 6.0
    assert enable_control_max_voltage("tps54331-12v-1a@2") == 6.0
    assert enable_control_max_voltage("lm2577t-adj-12v-boost@1") == 18.0

    assert enable_control_max_voltage("tps61088-2s-12v-1a@1") == 18.0
    assert enable_control_max_voltage("tps61088-host5-12v-1a@1") == 18.0

def _assert_lm2577_fault_limited_boost_topology(definition) -> None:
    """Validate boost, feedback and external fault limiting from connected pins."""
    pins = {(pin.role, pin.index, pin.pin): pin.net for pin in definition.pins}

    assert all(
        pins[("current_limiter", 0, str(pin))] == "protected_input"
        for pin in range(4, 9)
    )
    assert pins[("current_limiter", 0, "17")] == pins[("efuse_ilim", 0, "1")] == "ilim"
    assert pins[("efuse_ilim", 0, "2")] == "gnd"
    assert pins[("current_limiter", 0, "21")] == "gnd"
    assert pins[("converter", 0, "5")] == pins[("inductor", 0, "1")] == "protected_input"
    assert pins[("inductor", 0, "2")] == pins[("converter", 0, "4")] == "switch"
    # Device:D_Schottky pin 2 is anode and pin 1 cathode.
    assert pins[("boost_diode", 0, "2")] == "switch"
    assert pins[("boost_diode", 0, "1")] == pins[("output_cap", 0, "1")] == "output"
    assert pins[("output_cap", 0, "2")] == pins[("converter", 0, "3")] == "gnd"
    assert (
        pins[("feedback_top_a", 0, "1")]
        == pins[("output_cap", 0, "1")]
        == "output"
    )
    assert pins[("feedback_top_a", 0, "2")] == pins[("feedback_top_b", 0, "1")]
    assert pins[("feedback_top_b", 0, "2")] == pins[("feedback_bottom", 0, "1")] == pins[("converter", 0, "2")] == "feedback"
    assert pins[("feedback_bottom", 0, "2")] == "gnd"


@pytest.mark.parametrize(
    "recipe",
    ("lm2577t-adj-12v-boost@1", "lm2577t-adj-2s-12v-boost@1"),
)
def test_lm2577_12v_recipes_are_boost_and_fault_limited(recipe):
    definition = get_recipe(recipe)
    _assert_lm2577_fault_limited_boost_topology(definition)

    expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe,
            instance="boost",
            sheets={"power": "POWER"},
            port_bindings={"input": "SOURCE", "output": "LOAD", "gnd": "RETURN"},
        )
    )
    nets = _endpoint_nets(expansion)
    refs = {part.recipe_role: part.ref for part in expansion.parts}
    # The published source must feed the protection device, not an orphaned
    # private rail that leaves an otherwise complete boost circuit unpowered.
    assert all(nets[refs["current_limiter"], str(pin)] == "SOURCE" for pin in range(9, 14))
    assert nets[refs["input_bulk_cap"], "1"] == "SOURCE"
    assert nets[refs["efuse_input_bypass"], "1"] == "SOURCE"
    assert nets[refs["converter"], "5"] == nets[refs["current_limiter"], "4"]
    assert nets[refs["converter"], "5"] != "SOURCE"




def _assert_tps61088_protected_boost_topology(definition) -> None:
    """Check the actual synchronous boost, feedback and connector-short path."""
    pins = {(pin.role, pin.index, pin.pin): pin.net for pin in definition.pins}

    assert all(pins[("converter", 0, str(pin))] == "switch" for pin in range(4, 8))
    assert pins[("inductor", 0, "1")] == pins[("converter", 0, "9")] == "input"
    assert pins[("inductor", 0, "2")] == "switch"
    assert all(
        pins[("converter", 0, str(pin))] == "boost_output" for pin in range(14, 17)
    )
    assert all(
        pins[("output_limiter", 0, str(pin))] == "boost_output" for pin in range(9, 14)
    )
    assert all(
        pins[("output_limiter", 0, str(pin))] == "output" for pin in range(4, 9)
    )
    assert pins[("output_limiter", 0, "21")] == "gnd"
    assert pins[("converter", 0, "17")] == pins[("feedback_bottom", 0, "1")] == "feedback"
    assert pins[("feedback_top_a", 0, "1")] == "boost_output"
    assert pins[("feedback_top_a", 0, "2")] == pins[("feedback_top_b", 0, "1")]
    assert pins[("feedback_top_b", 0, "2")] == pins[("feedback_bottom", 0, "1")]
    assert pins[("feedback_bottom", 0, "2")] == "gnd"
    assert pins[("converter", 0, "8")] == pins[("bootstrap_cap", 0, "1")] == "bootstrap"
    assert pins[("bootstrap_cap", 0, "2")] == "switch"
    assert pins[("converter", 0, "3")] == pins[("frequency_resistor", 0, "1")]
    assert pins[("frequency_resistor", 0, "2")] == "switch"
    assert pins[("converter", 0, "19")] == pins[("switch_limit", 0, "1")] == "ilim"
    assert pins[("switch_limit", 0, "2")] == "gnd"
    assert pins[("output_limiter", 0, "17")] == pins[("efuse_ilim_resistor", 0, "1")]
    assert pins[("efuse_ilim_resistor", 0, "2")] == "gnd"


@pytest.mark.parametrize(
    "recipe",
    ("tps61088-2s-12v-1a@1", "tps61088-host5-12v-1a@1"),
)
def test_tps61088_recipes_are_feedback_regulated_and_output_short_protected(recipe):
    definition = get_recipe(recipe)
    assert definition.rated_output_current_a == 1.0
    _assert_tps61088_protected_boost_topology(definition)

    parts = {part.role: part for part in definition.parts}
    assert 1.204 * (1 + (499_000 + 3_160) / 56_000) == pytest.approx(12.00044)
    assert parts["switch_limit"].value == "49.9k 1%"
    assert parts["efuse_ilim_resistor"].value == "30.1k 1%"




@pytest.mark.parametrize(
    ("recipe", "external_vcc"),
    (
        ("tps552892-12v-2a@1", False),
        ("tps552892-5v-2a@1", True),
    ),
)
def test_tps552892_expansion_has_delivered_kelvin_sense_and_four_switch_path(
    recipe, external_vcc
):
    """Check the production expansion, including its scoped internal nets."""
    expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe,
            instance="regulated",
            sheets={"power": "POWER"},
            port_bindings={"input": "VIN", "gnd": "GND", "output": "VOUT"},
        )
    )
    nets = _endpoint_nets(expansion)
    parts = {part.recipe_role: part for part in expansion.parts}
    refs = {role: part.ref for role, part in parts.items()}
    output_caps = [
        part for part in expansion.parts if part.recipe_role == "output_cap"
    ]
    input_caps = [
        part for part in expansion.parts if part.recipe_role == "input_cap"
    ]
    converter = refs["converter"]


    assert nets[(converter, "7")] == "VIN"
    assert nets[(converter, "8")] == nets[(refs["inductor"], "1")]
    assert nets[(converter, "10")] == nets[(refs["inductor"], "2")]
    assert nets[(converter, "11")] == nets[(converter, "12")]
    assert nets[(converter, "11")] == nets[(refs["sense_resistor"], "1")]
    assert nets[(converter, "13")] == nets[(refs["sense_resistor"], "2")] == "VOUT"
    assert nets[(refs["feedback_top"], "1")] == "VOUT"
    assert (
        nets[(refs["feedback_top"], "2")]
        == nets[(refs["feedback_bottom"], "1")]
        == nets[(converter, "14")]
    )
    assert nets[(refs["feedback_bottom"], "2")] == "GND"
    assert all(
        nets[(part.ref, "1")] == nets[(converter, "11")]
        and nets[(part.ref, "2")] == "GND"
        for part in output_caps
    )
    connected_converter_pins = {
        endpoint.pin
        for connection in expansion.connections
        for endpoint in connection.endpoints
        if endpoint.ref == converter
    }
    connected_converter_pins.update(
        endpoint.pin for endpoint in expansion.no_connect_pins if endpoint.ref == converter
    )
    assert connected_converter_pins == {str(pin) for pin in range(1, 22)}

    owned_internal_nets = set(expansion.ownership.internal_nets)
    for net in (
        "enable",
        "frequency",
        "sw1",
        "sw2",
        "output_raw",
        "feedback",
        "compensation",
        "comp_rc",
        "boot1",
        "boot2",
    ):
        assert any(scoped.endswith(f"__{net}") for scoped in owned_internal_nets)
    assert (converter, "3") in {
        (endpoint.ref, endpoint.pin) for endpoint in expansion.no_connect_pins
    }
    assert (converter, "4") in {
        (endpoint.ref, endpoint.pin) for endpoint in expansion.no_connect_pins
    }
    assert (converter, "16") in {
        (endpoint.ref, endpoint.pin) for endpoint in expansion.no_connect_pins
    }
    assert all(part.recipe_role != "soft_start" for part in expansion.parts)

    if external_vcc:
        # SLVSH28A table 5-1: EXTVCC low selects externally supplied VCC.
        assert nets[(converter, "18")] == "VIN"
        assert nets[(converter, "21")] == "GND"
    else:
        assert nets[(converter, "18")].endswith("__vcc")
        assert (converter, "21") in {
            (endpoint.ref, endpoint.pin) for endpoint in expansion.no_connect_pins
        }


@pytest.mark.parametrize("recipe", ("tps552892-12v-2a@1", "tps552892-5v-2a@1"))
def test_tps552892_production_expansion_rejects_unbound_delivered_output(recipe):
    with pytest.raises(ValueError, match=r"port bindings mismatch.*output"):
        expand_recipe(
            RecipeSelection(
                recipe=recipe,
                instance="missing-output",
                sheets={"power": "POWER"},
                port_bindings={"input": "VIN", "gnd": "GND"},
            )
        )


def _tps552892_polarity_bom(recipe="tps552892-5v-2a@1"):
    from kicraft.design.models import BOM

    expansion = expand_recipe(
        RecipeSelection(
            recipe=recipe, instance="polarity", sheets={"power": "POWER"},
            port_bindings={"input": "VIN", "gnd": "GND", "output": "VOUT"},
        )
    )
    return BOM(
        parts=expansion.parts, arrays=[], connections=expansion.connections,
        no_connect_pins=expansion.no_connect_pins, recipe_ownership=[expansion.ownership],
    )


@pytest.mark.parametrize("recipe", ("tps552892-5v-2a@1", "tps552892-12v-2a@1"))
def test_tps552892_bias_modes_have_correct_power_pin_polarity(recipe):
    from kicraft.design.synthesis.validation import check_power_pin_polarity

    result = check_power_pin_polarity(_tps552892_polarity_bom(recipe))
    assert result.ok, result.offenders


@pytest.mark.parametrize(
    ("pin", "net"),
    [("18", "GND"), ("7", "GND"), ("17", "VIN"), ("21", "-5V")],
)
def test_tps552892_mode_role_does_not_excuse_reversed_real_supplies(pin, net):
    from kicraft.design.models import BOM
    from kicraft.design.synthesis.validation import check_power_pin_polarity

    bom = _tps552892_polarity_bom()
    converter = next(part for part in bom.parts if part.recipe_role == "converter")
    payload = bom.model_dump(mode="json")
    for connection in payload["connections"]:
        connection["endpoints"] = [
            endpoint for endpoint in connection["endpoints"]
            if (endpoint["ref"], endpoint["pin"]) != (converter.ref, pin)
        ]
    payload["connections"] = [
        connection for connection in payload["connections"] if connection["endpoints"]
    ]
    target = next(
        (connection for connection in payload["connections"] if connection["net_name"] == net),
        None,
    )
    if target is None:
        target = {"sheet": converter.sheet, "net_name": net, "endpoints": []}
        payload["connections"].append(target)
    target["endpoints"].append({"ref": converter.ref, "pin": pin})
    result = check_power_pin_polarity(BOM.model_validate(payload))
    assert not result.ok


@pytest.mark.parametrize(
    ("field", "value"),
    [("mpn", "TPS552892WRONG"), ("footprint", "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")],
)
def test_extvcc_mode_requires_exact_reviewed_device_and_package(field, value):
    from kicraft.design.synthesis.validation import check_power_pin_polarity

    bom = _tps552892_polarity_bom()
    converter = next(part for part in bom.parts if part.recipe_role == "converter")
    setattr(converter, field, value)
    assert not check_power_pin_polarity(bom).ok


def _exact_converter_family(family="k7803m-1000r3", *, load=0.5):
    return {
        "sheets": [{"name": "POWER", "stem": "POWER", "function": "24 V to 3.3 V supply"}],
        "power_nets": ["VIN", "VOUT", "GND"],
        "rail_voltages": {"VIN": 24.0, "VOUT": 3.3},
        "inter_sheet_nets": [],
        "requirements": [{
            "id": "reg", "sheet": "POWER", "role": "regulator", "family": family,
            "parameters": {"output_voltage": 3.3},
            "ports": {"input": "VIN", "gnd": "GND", "output": "VOUT"},
            "obligations": [{
                "kind": "quantitative", "original_obligation_id": "load",
                "quantity": "output current", "relation": "minimum", "value": load, "unit": "A",
            }],
        }],
    }


def test_exact_converter_identity_in_family_owns_its_complete_native_power_graph():
    from kicraft.design.architecture_intent import derive_architecture
    from kicraft.design.recipes.resolver import apply_architecture_recipe_resolution
    from kicraft.server.stage_work_units import probe_architecture_construction

    native = derive_architecture({
        "sheets": [{"name": "POWER", "stem": "POWER", "role": "power", "function": "Supply"}],
        "requirements": [
            {"id": "input", "sheet": "POWER", "role": "power_input",
             "family": "screw-terminal", "ties": {"negative": "GND"}},
            {"id": "reg", "sheet": "POWER", "role": "regulator",
             "family": "k7803m-1000r3", "supply_bindings": {"input": "VIN"},
             "parameters": {"output_voltage": 3.3}},
        ],
        "power": {"rails": {
            "VIN": {"voltage": 24.0, "from": "input.positive"},
            "VOUT": {"voltage": 3.3, "from": "reg.output"},
        }},
        "signals": [],
    })
    resolved = apply_architecture_recipe_resolution(native)
    reg = next(row for row in resolved.requirements if row.id == "reg")
    assert reg.family == "ylptec-k7803m-1000r3"
    assert reg.declared_interface is None
    construction = probe_architecture_construction({"architecture": resolved.model_dump(mode="json")})
    module = next(part for part in construction["parts"] if part.get("recipe_role") == "converter")
    assert module["mpn"] == "K7803M-1000R3"
    pins = {
        endpoint["pin"]: connection["net_name"]
        for connection in construction["connections"]
        for endpoint in connection["endpoints"] if endpoint["ref"] == module["ref"]
    }
    assert pins == {"1": "VIN", "2": "GND", "3": "VOUT"}
    for role, rail in (("input_cap", "VIN"), ("output_cap", "VOUT")):
        cap = next(part for part in construction["parts"] if part.get("recipe_role") == role)
        assert {
            endpoint["pin"]: connection["net_name"]
            for connection in construction["connections"]
            for endpoint in connection["endpoints"] if endpoint["ref"] == cap["ref"]
        } == {"1": rail, "2": "GND"}


def test_exact_converter_family_does_not_expand_prefixes_or_override_user_parts():
    unresolved = resolve_architecture_recipes(_exact_converter_family("k7803m"))
    assert unresolved.unresolved_requirements == ["reg"]
    assert unresolved.selections == []
    conflict = _exact_converter_family()
    conflict["requirements"][0]["exact_part"] = "AP63203WU-7"
    result = resolve_architecture_recipes(conflict, {"named_parts": ["K7803M-1000R3"]})
    assert any(d.code == "conflicting_exact_variant" for d in result.blocking)


def test_exact_converter_family_keeps_the_reviewed_output_current_limit():
    result = resolve_architecture_recipes(_exact_converter_family(load=0.501))
    assert any(d.code == "recipe_output_current_exceeded" for d in result.blocking)
    assert result.selections == []
