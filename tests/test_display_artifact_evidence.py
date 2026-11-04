import pytest

from kicraft.design.lowering import lower_requirement
from kicraft.design.models import CircuitRequirement, RecipeSelection
from kicraft.design.part_identity import reviewed_part
from kicraft.design.recipes import expand_recipe
from kicraft.design.synthesis.symbol_pinout import lookup_pins
from kicraft.eval.application_pin_evidence import reviewed_application_pins
from kicraft.eval.display_artifact_evidence import display_circuit_facts
from kicraft.eval.electrical_artifact_evidence import _functional_facts, _rgb_status_observed


def _expand(recipe, bindings):
    expansion = expand_recipe(RecipeSelection(
        recipe=recipe, instance="display", sheets={"interface": "DISPLAY"},
        port_bindings=bindings,
    ))
    parts = [part.model_dump(mode="json") for part in expansion.parts]
    pads = {
        (endpoint.ref, endpoint.pin): connection.net_name
        for connection in expansion.connections for endpoint in connection.endpoints
    }
    names = {
        (part.ref, str(pin["number"])): str(pin.get("name") or "")
        for part in expansion.parts
        for pin in lookup_pins(part.symbol, all_units=True)["pins"]
    }
    return parts, names, pads


def _mcu_context(count, *, minimum_high=2.0):
    candidates = [
        (identity, pins, record)
        for identity, pins in reviewed_application_pins("output").items()
        if len(pins) >= count
        if (record := reviewed_part(identity)) is not None
        and isinstance(record.operating_limits.get("output_high_min_v_at_3v3"), (int, float))
        and isinstance(record.operating_limits.get("output_low_max_v_at_3v3"), (int, float))
        and record.operating_limits["output_high_min_v_at_3v3"] >= minimum_high
        and (
            isinstance((conditions := record.support_network.get("gpio_output_test_conditions")), dict)
            and conditions.get("supply_v") == 3.3
            and isinstance(conditions.get("source_sink_current_a"), dict)
            and isinstance(conditions["source_sink_current_a"].get("source"), (int, float))
            and "PAD_DRIVER = 3" in str(conditions.get("table") or "")
        )
    ]
    identity, pins, record = candidates[0]
    selected = sorted(pins)[:count]
    part = {"ref": "U_MCU", "mpn": record.identity, "symbol": record.symbol, "footprint": record.footprint}
    nets = {f"GPIO{index}": [f"U_MCU.{pin}"] for index, pin in enumerate(selected)}
    facts = {
        "complete_mcu_supplies": {"U_MCU": {"rail": "+3V3", "ground": "GND"}},
        "complete_mcu_gpio_nets": nets,
    }
    return part, facts


def _voltage(rail, ground):
    return {
        ("+3V3", "GND"): 3.3, ("+5V", "GND"): 5.0, ("VIN", "GND"): 12.0,
        ("+1V8", "GND"): 1.8,
    }.get((rail, ground))


def _facts(parts, names, pads, mcu_facts):
    return display_circuit_facts(parts, names, pads, mcu_facts=mcu_facts, supply_voltage=_voltage)


def test_external_ws2812_requires_a_powered_hct_control_path_and_source_terminator():
    parts, names, pads = _expand("ws2812-external-strip-interface@1", {
        "vdd_5v": "+5V", "gnd": "GND", "data_in": "GPIO0",
    })
    mcu, mcu_facts = _mcu_context(1)
    parts.append(mcu)
    assert _facts(parts, names, pads, mcu_facts)["gates"]["external_ws2812_strip_interface"] == "pass"

    shifter = next(part for part in parts if part.get("mpn") == "SN74HCT245PWR-JSM")
    original_output = pads[shifter["ref"], "18"]
    pads[shifter["ref"], "2"] = "OPEN"
    assert "external_ws2812_strip_interface" not in _facts(parts, names, pads, mcu_facts)["gates"]
    pads[shifter["ref"], "2"] = "GPIO0"
    pads[shifter["ref"], "18"] = "+5V"
    assert "external_ws2812_strip_interface" not in _facts(parts, names, pads, mcu_facts)["gates"]
    pads[shifter["ref"], "18"] = original_output
    mcu_facts["complete_mcu_supplies"]["U_MCU"]["rail"] = "+1V8"
    assert "external_ws2812_strip_interface" not in _facts(parts, names, pads, mcu_facts)["gates"]


def test_four_digit_display_requires_power_current_setting_and_three_translated_controls():
    parts, names, pads = _expand("max7219-four-digit-seven-segment@1", {
        "vdd_5v": "+5V", "gnd": "GND", "din": "GPIO0", "clk": "GPIO1", "load": "GPIO2",
    })
    mcu, mcu_facts = _mcu_context(3)
    parts.append(mcu)
    assert _facts(parts, names, pads, mcu_facts)["gates"]["external_four_digit_seven_segment_display"] == "pass"

    iset = next(part for part in parts if part.get("recipe_role") == "iset")
    iset["value"] = "10k"
    assert "external_four_digit_seven_segment_display" not in _facts(parts, names, pads, mcu_facts)["gates"]
    iset["value"] = "27k"
    next(part for part in parts if part.get("recipe_role") == "decoupling")["value"] = "10uF"
    assert "external_four_digit_seven_segment_display" not in _facts(parts, names, pads, mcu_facts)["gates"]


def test_hub75_matrix_requires_thirteen_distinct_powered_controls():
    bindings = {"vdd_5v": "+5V", "gnd": "GND"}
    bindings.update({name: f"GPIO{index}" for index, name in enumerate((
        "r0", "g0", "b0", "r1", "g1", "b1", "addr_a", "addr_b", "addr_c", "addr_d", "clk", "lat", "oe",
    ))})
    parts, names, pads = _expand("led-matrix-hub75-interface@1", bindings)
    mcu, mcu_facts = _mcu_context(13)
    parts.append(mcu)
    assert _facts(parts, names, pads, mcu_facts)["gates"]["external_led_matrix_function"] == "pass"

    first_shifter = next(part for part in parts if part.get("mpn") == "SN74HCT245PWR-JSM")
    pads[first_shifter["ref"], "2"] = "GPIO1"
    assert "external_led_matrix_function" not in _facts(parts, names, pads, mcu_facts)["gates"]


def _power_led_artifact(*, defect=None):
    parts, names, pads = _expand("al8860-3w-power-led@1", {
        "vin": "VIN", "gnd": "GND", "enable": "GPIO0",
    })
    mcu, mcu_facts = _mcu_context(1, minimum_high=2.5)
    parts.append(mcu)
    terminal = reviewed_part("WJ126V-5.0-02P-14-00A")
    parts.append({"ref": "JIN", "mpn": terminal.identity, "symbol": terminal.symbol, "footprint": terminal.footprint})
    pads["JIN", "1"], pads["JIN", "2"] = "VIN", "GND"
    if defect == "current":
        next(part for part in parts if part.get("recipe_role") == "sense")["value"] = "0.2R"
    if defect == "control":
        driver = next(part for part in parts if part.get("mpn") == "AL8860MP-13")
        pads[driver["ref"], "4"] = "OPEN"
    if defect == "power":
        pads["JIN", "1"] = "OPEN"
    if defect == "overvoltage":
        driver = next(part for part in parts if part.get("mpn") == "AL8860MP-13")
        pads[driver["ref"], "4"] = "VIN"
    if defect == "reversed_diode":
        diode = next(part for part in parts if part.get("mpn") == "SS14")
        anode, cathode = (diode["ref"], "2"), (diode["ref"], "1")
        pads[anode], pads[cathode] = pads[cathode], pads[anode]
    return parts, names, pads, mcu_facts


@pytest.mark.parametrize("defect", [None, "current", "control", "power", "overvoltage", "reversed_diode"])
def test_three_watt_led_driver_requires_current_setting_control_and_real_input(defect):
    parts, names, pads, mcu_facts = _power_led_artifact(defect=defect)
    facts = _facts(parts, names, pads, mcu_facts)
    passed = facts["gates"].get("external_three_watt_power_led_function") == "pass"
    assert passed is (defect is None)
    assert (facts["gates"].get("external_led_driver_function") == "pass") is (defect is None)


@pytest.mark.parametrize("repaired", [None, "first", "second"])
def test_power_led_loop_and_control_must_belong_to_the_same_driver(repaired):
    parts, names, pads, mcu_facts = _power_led_artifact(
        defect=None if repaired == "first" else "control"
    )
    other_parts, other_names, other_pads, _ = _power_led_artifact(
        defect=None if repaired == "second" else "reversed_diode"
    )
    refs = {
        part["ref"]: part["ref"] + "_SECOND"
        for part in other_parts if part["ref"] != "U_MCU"
    }
    parts.extend({**part, "ref": refs[part["ref"]]} for part in other_parts if part["ref"] in refs)
    names.update({(refs[ref], pin): name for (ref, pin), name in other_names.items() if ref in refs})
    pads.update({
        (refs[ref], pin): net if net in {"VIN", "GND", "GPIO0"} else net + "_SECOND"
        for (ref, pin), net in other_pads.items() if ref in refs
    })
    facts = _facts(parts, names, pads, mcu_facts)
    assert (
        facts["gates"].get("external_three_watt_power_led_function") == "pass"
    ) is (repaired is not None)



def _rgb_indicator_artifact(*, defect=None, mpn="APTF1616LSEEZGKQBKC"):
    mcu, mcu_facts = _mcu_context(3, minimum_high=2.64)
    led = reviewed_part(mpn)
    transistor = reviewed_part("MMBT3904")
    parts = [
        mcu,
        {"ref": "D1", "mpn": led.identity, "symbol": led.symbol, "footprint": led.footprint},
    ]
    pads = {
        ("D1", led.port_pins["common_anode"]): "+5V",
        ("D1", led.port_pins["red_cathode"]): "RED_K",
        ("D1", led.port_pins["green_cathode"]): "GREEN_K",
        ("D1", led.port_pins["blue_cathode"]): "BLUE_K",
    }
    current_values = ("1.5k 1%", "820R 1%", "820R 1%")
    for index, (color, current_value) in enumerate(zip(("RED", "GREEN", "BLUE"), current_values, strict=True)):
        base, collector = f"BASE_{color}", f"COLLECTOR_{color}"
        parts.extend((
            {"ref": f"Q{index + 1}", "mpn": transistor.identity, "symbol": transistor.symbol,
             "footprint": transistor.footprint},
            {"ref": f"R{index + 1}", "value": current_value, "symbol": "Device:R",
             "footprint": "Resistor_SMD:R_0603_1608Metric"},
            {"ref": f"R{index + 4}", "value": "4.7k 1%", "symbol": "Device:R",
             "footprint": "Resistor_SMD:R_0603_1608Metric"},
            {"ref": f"R{index + 7}", "value": "100k 1%", "symbol": "Device:R",
             "footprint": "Resistor_SMD:R_0603_1608Metric"},
        ))
        pads.update({
            (f"Q{index + 1}", "1"): base, (f"Q{index + 1}", "2"): "GND",
            (f"Q{index + 1}", "3"): collector,
            (f"R{index + 1}", "1"): f"{color}_K", (f"R{index + 1}", "2"): collector,
            (f"R{index + 4}", "1"): f"GPIO{index}", (f"R{index + 4}", "2"): base,
            (f"R{index + 7}", "1"): base, (f"R{index + 7}", "2"): "GND",
        })
    if defect == "supply":
        pads["D1", led.port_pins["common_anode"]] = "+3V3"
    elif defect == "control":
        pads["R6", "1"] = "GPIO1"
    elif defect == "reference":
        pads["Q3", "2"] = "OTHER_GND"
    elif defect == "current_limit":
        next(part for part in parts if part["ref"] == "R2")["value"] = "1k"
    elif defect == "pinout":
        pads["D1", led.port_pins["common_anode"]] = "RED_K"
    elif defect == "different_mcu":
        other, other_facts = _mcu_context(1, minimum_high=2.64)
        other["ref"] = "U_OTHER"
        parts.append(other)
        endpoint = next(iter(other_facts["complete_mcu_gpio_nets"].values()))[0]
        mcu_facts["complete_mcu_supplies"]["U_OTHER"] = {"rail": "+3V3", "ground": "GND"}
        mcu_facts["complete_mcu_gpio_nets"]["GPIO2"] = [endpoint.replace("U_MCU", "U_OTHER")]
    return parts, {}, pads, mcu_facts


@pytest.mark.parametrize("mpn", ["APTF1616LSEEZGKQBKC", "XL-1615RGBC-RF"])
@pytest.mark.parametrize(
    "defect", [None, "supply", "control", "reference", "current_limit", "pinout", "different_mcu"],
)
def test_rgb_status_indicator_requires_one_powered_mcu_and_three_complete_channel_paths(defect, mpn):
    parts, names, pads, mcu_facts = _rgb_indicator_artifact(defect=defect, mpn=mpn)
    facts = _facts(parts, names, pads, mcu_facts)
    assert (
        facts["gates"].get("external_rgb_status_led_channels") == "pass"
    ) is (defect is None)
    assert (
        facts["channel_counts"].get("rgb_status_led_channel") == 3
    ) is (defect is None)


def test_rgb_lowerer_has_current_limiters_and_deterministic_off_controls():
    artifact = lower_requirement(CircuitRequirement(
        id="status", sheet="MCU", role="user_io", family="rgb-led",
        exact_part="APTF1616LSEEZGKQBKC", parameters={"target_current_ma": 2.0},
        ports={
            "vdd_5v": "+5V", "gnd": "GND", "red": "GPIO0",
            "green": "GPIO1", "blue": "GPIO2",
        },
    ))
    with pytest.raises(ValueError, match="cannot realize"):
        lower_requirement(CircuitRequirement(
            id="status", sheet="MCU", role="user_io", family="rgb-led",
            exact_part="APTF1616LSEEZGQKBKC", parameters={"target_current_ma": 2.0},
            ports={
                "vdd_5v": "+5V", "gnd": "GND", "red": "GPIO0",
                "green": "GPIO1", "blue": "GPIO2",
            },
        ))
    pins = {(pin.role, pin.index, pin.pin): pin.net for pin in artifact.pins}
    assert artifact.lowerer_id == "common-anode-rgb-indicator@2"
    assert pins["rgb_led", 0, "1"] == "+5V"
    assert [pins["rgb_led", 0, pin] for pin in ("2", "3", "4")] == [
        "__KICRAFT_RGB_INDICATOR_RED_K",
        "__KICRAFT_RGB_INDICATOR_GREEN_K",
        "__KICRAFT_RGB_INDICATOR_BLUE_K",
    ]
    for index, gpio in enumerate(("GPIO0", "GPIO1", "GPIO2")):
        base = f"__KICRAFT_RGB_INDICATOR_BASE_{index}"
        assert pins["channel_driver", index, "2"] == "GND"
        assert pins["base_resistor", index, "1"] == gpio
        assert pins["base_resistor", index, "2"] == base
        assert pins["base_pulldown", index, "1"] == base
        assert pins["base_pulldown", index, "2"] == "GND"


@pytest.mark.parametrize("requested", [None, "XL-1615RGBC-RF", "APTF1616LSEEZGKQBKC"])
def test_rgb_lowering_preserves_exact_identity_and_reviewed_channel_wiring(requested):
    artifact = lower_requirement(CircuitRequirement(
        id="rgb", sheet="MCU", role="user_io", family="rgb-led",
        exact_part=requested, parameters={"target_current_ma": 2.0},
        ports={"vdd_5v": "+5V", "gnd": "GND", "red": "GPIO0", "green": "GPIO1", "blue": "GPIO2"},
    ))
    group = next(group for group in artifact.groups if group.role == "rgb_led")
    led = reviewed_part(requested or "XL-1615RGBC-RF")
    assert (group.mpn.casefold(), group.symbol, group.footprint) == (
        led.identity, led.symbol, led.footprint,
    )
    pins = {pin.pin: pin.net for pin in artifact.pins if pin.role == "rgb_led"}
    assert pins[led.port_pins["common_anode"]] == "+5V"
    for color in ("red", "green", "blue"):
        assert pins[led.port_pins[f"{color}_cathode"]] == f"__KICRAFT_RGB_INDICATOR_{color.upper()}_K"


@pytest.mark.parametrize("defect", ["wrong_pair", "open_channel"])
def test_rgb_observer_rejects_unreviewed_pair_and_open_channel(defect):
    parts, names, pads, mcu_facts = _rgb_indicator_artifact(mpn="XL-1615RGBC-RF")
    if defect == "wrong_pair":
        next(part for part in parts if part["ref"] == "D1")["footprint"] = reviewed_part("APTF1616LSEEZGKQBKC").footprint
    else:
        del pads["R3", "2"]
    assert "external_rgb_status_led_channels" not in _facts(parts, names, pads, mcu_facts)["gates"]


@pytest.mark.parametrize("defect", [None, "unpowered", "wrong_pair", "open_channel", "shorted_pins"])
def test_power_rgb_observer_requires_reviewed_identity_and_three_powered_paths(defect):
    led = reviewed_part("XL-1615RGBC-RF")
    transistor = reviewed_part("MMBT3904")
    parts = [{"ref": "D1", "mpn": led.identity, "symbol": led.symbol, "footprint": led.footprint}]
    pads = {("D1", led.port_pins["common_anode"]): "VIN"}
    names = {}
    for index, color in enumerate(("red", "green", "blue")):
        q = f"Q{index + 1}"
        r, base_r = f"R{index + 1}", f"R{index + 4}"
        cathode, collector, base = f"K{index}", f"C{index}", f"B{index}"
        parts.extend((
            {"ref": q, "mpn": transistor.identity, "symbol": transistor.symbol, "footprint": transistor.footprint},
            {"ref": r, "value": "10k", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric"},
            {"ref": base_r, "value": "10k", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric"},
        ))
        pads.update({
            ("D1", led.port_pins[f"{color}_cathode"]): cathode,
            (q, "1"): base, (q, "2"): "GND", (q, "3"): collector,
            (r, "1"): cathode, (r, "2"): collector,
            (base_r, "1"): "VIN", (base_r, "2"): base,
        })
        names.update({(q, "1"): "B", (q, "2"): "E", (q, "3"): "C"})
    if defect == "unpowered":
        pads["D1", led.port_pins["common_anode"]] = "GND"
    elif defect == "wrong_pair":
        parts[0]["footprint"] = reviewed_part("APTF1616LSEEZGKQBKC").footprint
    elif defect == "open_channel":
        del pads["R3", "2"]
    elif defect == "shorted_pins":
        pads["D1", led.port_pins["blue_cathode"]] = pads["D1", led.port_pins["green_cathode"]]
    observed = _rgb_status_observed(
        parts, {("VIN", "GND")},
        [{"input_net": "VIN", "output_net": "OUT", "ground_net": "GND"}],
        names, pads,
    )
    assert observed is (defect is None)


@pytest.mark.parametrize(
    "defect",
    [None, "grounded_base", "open_base", "open_sense", "open_inverter",
     "inverter_return", "unbound_sense", "wrong_output", "open_sense_pulldown",
     "open_cathode", "open_return", "shorted_channels", "wrong_pair",
     "branched_base", "cycle_base", "open_source_fuse", "grounded_common",
     "open_input_contact", "open_output_return", "unbound_source", "undeclared_voltage"],
)
def test_power_rgb_discrete_producer_graph_requires_complete_control_paths(defect):
    requirement = CircuitRequirement(
        id="distribution", sheet="POWER", role="power_input",
        family="fused-power-distribution",
        parameters={
            "channels": 2, "input_voltage_v": 24.0, "source_current_a": 0.5,
            "enable_jumper": False, "reset_button": False, "rgb_status": True,
        },
        ports={
            "input": "SOURCE_POSITIVE", "gnd": "SOURCE_RETURN",
            "output1": "LOAD_1_POSITIVE", "output2": "LOAD_2_POSITIVE",
        },
    )
    artifact = lower_requirement(requirement)
    parts, names, pads, refs = [], {}, {}, {}
    for group in artifact.groups:
        for index in range(group.quantity):
            ref = f"{group.reference_prefix}{len(parts) + 1}"
            refs[group.role, index] = ref
            parts.append({
                "ref": ref, "mpn": group.mpn, "value": group.value,
                "symbol": group.symbol, "footprint": group.footprint,
            })
            if group.role in {"rgb_driver", "rgb_output_sense"}:
                names.update({(ref, "1"): "B", (ref, "2"): "E", (ref, "3"): "C"})
    for pin in artifact.pins:
        pads[refs[pin.role, pin.index], pin.pin] = pin.net
    connector = reviewed_part("PM254V-11-02-H85")
    for index, positive in enumerate(("SOURCE_POSITIVE", "LOAD_1_POSITIVE", "LOAD_2_POSITIVE"), 1):
        ref = f"J{index}"
        parts.append({
            "ref": ref, "mpn": connector.identity, "value": connector.identity,
            "symbol": connector.symbol, "footprint": connector.footprint,
        })
        pads[ref, "1"], pads[ref, "2"] = positive, "SOURCE_RETURN"
    led = refs["rgb_status_led", 0]
    driver = refs["rgb_driver", 0]
    inverter = refs["rgb_output_sense", 0]
    base_resistor = refs["rgb_base_resistor", 2]
    sense_resistor = refs["rgb_sense_base_resistor", 2]
    if defect == "grounded_base":
        pads[driver, "1"] = "SOURCE_RETURN"
    elif defect == "open_base":
        del pads[base_resistor, "2"]
    elif defect == "open_sense":
        del pads[sense_resistor, "2"]
    elif defect == "open_inverter":
        del pads[inverter, "3"]
    elif defect == "inverter_return":
        del pads[inverter, "2"]
    elif defect == "unbound_sense":
        pads[refs["rgb_sense_base_resistor", 0], "1"] = "UNBOUND"
    elif defect == "wrong_output":
        pads[refs["rgb_base_resistor", 5], "1"] = "LOAD_2_POSITIVE"
    elif defect == "open_sense_pulldown":
        del pads[refs["rgb_sense_base_pulldown", 0], "2"]
    elif defect == "open_cathode":
        del pads[refs["rgb_channel_resistor", 0], "2"]
    elif defect == "open_return":
        del pads[driver, "2"]
    elif defect == "shorted_channels":
        pads[led, "3"] = pads[led, "4"]
    elif defect == "wrong_pair":
        next(part for part in parts if part["ref"] == led)["footprint"] = reviewed_part("APTF1616LSEEZGKQBKC").footprint
    elif defect == "open_source_fuse":
        del pads[refs["source_fuse", 0], "2"]
    elif defect == "grounded_common":
        pads[led, "2"] = "SOURCE_RETURN"
    elif defect == "open_input_contact":
        del pads["J1", "1"]
    elif defect == "open_output_return":
        del pads["J2", "2"]
    elif defect == "unbound_source":
        requirement.role = "connector"
    elif defect in {"branched_base", "cycle_base"}:
        branch = "R_EXTRA"
        parts.append({"ref": branch, "value": "10k", "symbol": "Device:R"})
        pads[branch, "1"] = pads[base_resistor, "1"]
        pads[branch, "2"] = (
            "UNBOUND" if defect == "branched_base" else pads[refs["rgb_base_resistor", 0], "1"]
        )
    state = {"architecture": {
        "requirements": [requirement.model_dump(mode="json")],
        "rail_voltages": (
            {"SOURCE_RETURN": 0.0} if defect == "undeclared_voltage"
            else {"SOURCE_POSITIVE": 24.0, "SOURCE_RETURN": 0.0}
        ),
    }}
    facts = _functional_facts(
        parts, names, pads, state,
        {"obligations": [{"id": "rgb_status", "check": {
            "kind": "gate", "gate": "external_rgb_status_led_channels",
        }}]},
    )
    assert (facts["gates"].get("external_rgb_status_led_channels") == "pass") is (defect is None)
    assert "output_voltage_v" not in facts


def test_four_indicator_outputs_require_four_separate_mcu_driven_led_circuits():
    led = reviewed_part("LTST-C190KGKT")
    mcu, mcu_facts = _mcu_context(4)
    parts = [mcu]
    names = {("JGROUND", "1"): "GND"}
    pads = {("JGROUND", "1"): "GND"}
    for index in range(4):
        parts.extend((
            {"ref": f"D{index}", "mpn": led.identity, "symbol": led.symbol, "footprint": led.footprint},
            {"ref": f"R{index}", "value": "1k", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric"},
        ))
        pads[f"R{index}", "1"] = f"GPIO{index}"
        pads[f"R{index}", "2"] = f"LED{index}"
        pads[f"D{index}", "1"] = "GND"
        pads[f"D{index}", "2"] = f"LED{index}"
    facts = _facts(parts, names, pads, mcu_facts)
    assert facts["gates"]["external_four_indicator_outputs"] == "pass"
    assert facts["gates"]["external_indicator_controller_function"] == "pass"

    pads["R3", "1"] = "GPIO2"
    facts = _facts(parts, names, pads, mcu_facts)
    assert "external_four_indicator_outputs" not in facts["gates"]
    assert "external_indicator_controller_function" not in facts["gates"]
