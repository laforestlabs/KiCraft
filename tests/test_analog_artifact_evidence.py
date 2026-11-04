import pytest

from kicraft.eval.analog_artifact_evidence import (
    analog_conditioner_facts,
    temt6000_readout_facts,
)

def _fixture_supply_voltage(rail, ground):
    return 5.0 if (rail, ground) == ("AV5", "AGND") else None



def _temperature_artifact(*, broken=False):
    parts = [
        {"ref": "U1", "mpn": "MCP6002-I/SN", "symbol": "Amplifier_Operational:MCP6002-xSN", "footprint": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"},
        {"ref": "S1", "mpn": "MCP9700AT-E/TT", "symbol": "Sensor_Temperature:MCP9700Ax-ETT", "footprint": "Package_TO_SOT_SMD:SOT-23"},
        {"ref": "S2", "mpn": "MCP9700AT-E/TT", "symbol": "Sensor_Temperature:MCP9700Ax-ETT", "footprint": "Package_TO_SOT_SMD:SOT-23"},
        {"ref": "J1", "mpn": "TSW-103-07-G-S", "symbol": "Connector_Generic:Conn_01x03", "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical"},
        {"ref": "C1", "symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric", "value": "100nF"},
        {"ref": "C2", "symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric", "value": "100nF"},
        {"ref": "C3", "symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric", "value": "100nF"},
        {"ref": "R1", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "10k"},
        {"ref": "R2", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "20k"},
        {"ref": "R3", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "10k"},
        {"ref": "R4", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "20k"},
    ]
    nets = {
        ("U1", "8"): "AV5", ("U1", "4"): "AGND", ("U1", "3"): "S1", ("U1", "2"): "FB1", ("U1", "1"): "OUT1", ("U1", "5"): "S2", ("U1", "6"): "FB2", ("U1", "7"): "OUT2",
        ("S1", "1"): "AV5", ("S1", "2"): "S1", ("S1", "3"): "AGND",
        ("S2", "1"): "AV5", ("S2", "2"): "S2", ("S2", "3"): "AGND",
        ("J1", "1"): "AGND", ("J1", "2"): "OUT1", ("J1", "3"): "BAD" if broken else "OUT2",
        ("C1", "1"): "AV5", ("C1", "2"): "AGND", ("C2", "1"): "AV5", ("C2", "2"): "AGND",
        ("C3", "1"): "AV5", ("C3", "2"): "AGND",
        ("R1", "1"): "OUT1", ("R1", "2"): "FB1",
        ("R2", "1"): "FB1", ("R2", "2"): "AGND",
        ("R3", "1"): "OUT2", ("R3", "2"): "FB2",
        ("R4", "1"): "FB2", ("R4", "2"): "AGND",
    }
    return parts, nets


def _isolated_temperature_artifact(*, defect=None):
    parts, pad_nets = _temperature_artifact()
    pad_nets.update({
        ("U1", "1"): "STAGE1", ("U1", "7"): "STAGE2",
        ("R1", "1"): "STAGE1", ("R3", "1"): "STAGE2",
        ("U2", "1"): "STAGE1", ("U2", "2"): "OUT1", ("U2", "3"): "AGND",
        ("U2", "4"): "ISOLATION_RELEASE", ("U2", "5"): "HOST5",
        ("U3", "1"): "STAGE2", ("U3", "2"): "OUT2", ("U3", "3"): "AGND",
        ("U3", "4"): "ISOLATION_RELEASE", ("U3", "5"): "HOST5",
        ("C4", "1"): "HOST5", ("C4", "2"): "AGND",
        ("C5", "1"): "HOST5", ("C5", "2"): "AGND",
    })
    parts.extend([
        {"ref": "U2", "mpn": "SN74LVC1G66DBVR", "symbol": "74xGxx:74LVC1G66", "footprint": "Package_TO_SOT_SMD:SOT-23-5"},
        {"ref": "U3", "mpn": "SN74LVC1G66DBVR", "symbol": "74xGxx:74LVC1G66", "footprint": "Package_TO_SOT_SMD:SOT-23-5"},
        {"ref": "C4", "symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric", "value": "100nF"},
        {"ref": "C5", "symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric", "value": "100nF"},
    ])
    if defect == "control":
        pad_nets["U3", "4"] = "DISABLED"
    elif defect == "supply":
        pad_nets["U3", "5"] = "AV5"
    elif defect == "signal":
        pad_nets["U3", "2"] = "OPEN"
    state = {
        "architecture": {
            "requirements": [{
                "family": "analog-conditioner",
                "parameters": {"power_cycle_isolation": True},
            }],
        },
    }
    return parts, pad_nets, state


def test_isolated_conditioner_requires_enabled_reviewed_switch_paths():
    parts, pad_nets, state = _isolated_temperature_artifact()
    facts = analog_conditioner_facts(parts, {}, pad_nets, state=state, reset_observed=True, supply_voltage=_fixture_supply_voltage)

    assert facts["net_paths"]["analog_conditioner_outputs"] is True
    assert facts["analog_output_nets"] == ["OUT1", "OUT2"]
    assert "analog_output_endpoint_v" in facts


def test_isolated_conditioner_never_bridges_disabled_or_damaged_switches():
    for defect in ("control", "supply", "signal"):
        parts, pad_nets, state = _isolated_temperature_artifact(defect=defect)
        facts = analog_conditioner_facts(parts, {}, pad_nets, state=state, reset_observed=True, supply_voltage=_fixture_supply_voltage)

        assert "analog_conditioner_outputs" not in facts["net_paths"]
        assert "analog_output_nets" not in facts
        assert "analog_output_endpoint_v" not in facts

    parts, pad_nets, state = _isolated_temperature_artifact()
    facts = analog_conditioner_facts(parts, {}, pad_nets, state=state, reset_observed=False, supply_voltage=_fixture_supply_voltage)

    assert "analog_conditioner_outputs" not in facts["net_paths"]
    assert "analog_output_nets" not in facts
    assert "analog_output_endpoint_v" not in facts


@pytest.mark.parametrize(
    "pins, observed",
    [(("7", "8"), True), (("1", "2"), False), (("4", "5"), False)],
)
def test_conditioner_requires_actual_adc_capable_contacts(pins, observed):
    parts, pad_nets = _temperature_artifact()
    parts[:] = [part for part in parts if part["ref"] != "J1"]
    pad_nets = {
        endpoint: net for endpoint, net in pad_nets.items()
        if endpoint[0] != "J1"
    }
    parts.append({
        "ref": "U2", "mpn": "STM32G030F6P6",
        "symbol": "MCU_ST_STM32G0:STM32G030F6Px",
        "footprint": "Package_SO:TSSOP-20_4.4x6.5mm_P0.65mm",
    })
    pad_nets["U2", pins[0]] = "OUT1"
    pad_nets["U2", pins[1]] = "OUT2"
    state = {
        "architecture": {
            "requirements": [
                {
                    "family": "analog-conditioner",
                    "ports": {"analog_out_1": "OUT1", "analog_out_2": "OUT2"},
                },
                {
                    "family": "stm32g0",
                    "ports": {"adc_0": "OUT1", "adc_1": "OUT2"},
                },
            ],
        },
    }

    facts = analog_conditioner_facts(parts, {}, pad_nets, state=state, supply_voltage=_fixture_supply_voltage)

    assert ("analog_conditioner_outputs" in facts["net_paths"]) is observed
    if observed:
        assert "analog_output_endpoint_v" in facts
    else:
        assert "analog_output_endpoint_v" not in facts


def test_conditioner_requires_sourced_five_volts_and_distinct_bypasses():
    parts, nets = _temperature_artifact()
    for voltage in (None, 3.3):
        facts = analog_conditioner_facts(
            parts, {}, nets, supply_voltage=lambda rail, ground: voltage
        )
        assert "external_temperature_acquisition_function" not in facts["gates"]
        assert "analog_output_max_v" not in facts
    for ref in ("C1", "C2", "C3"):
        damaged = dict(nets)
        damaged.pop((ref, "1"))
        facts = analog_conditioner_facts(parts, {}, damaged, supply_voltage=_fixture_supply_voltage)
        assert "external_temperature_acquisition_function" not in facts["gates"]


@pytest.mark.parametrize(("symbol", "value"), [("Device:C", "1nF"), ("Device:R", "1k")])
@pytest.mark.parametrize("isolated", [False, True])
def test_conditioner_rejects_delivered_output_loads(symbol, value, isolated):
    if isolated:
        parts, nets, state = _isolated_temperature_artifact()
    else:
        parts, nets = _temperature_artifact()
        state = None
    parts.append({"ref": "XLOAD", "symbol": symbol, "value": value})
    nets["XLOAD", "1"] = "OUT1"
    nets["XLOAD", "2"] = "AGND"
    facts = analog_conditioner_facts(
        parts, {}, nets, state=state, reset_observed=isolated,
        supply_voltage=_fixture_supply_voltage,
    )
    assert "external_temperature_acquisition_function" not in facts["gates"]
    assert "analog_output_max_v" not in facts


def test_temperature_conditioner_facts_require_delivered_sensor_feedback_and_header_topology():
    parts, pad_nets = _temperature_artifact()
    facts = analog_conditioner_facts(parts, {}, pad_nets, supply_voltage=_fixture_supply_voltage)
    assert facts["channel_counts"]["analog_conditioner"] == 2
    assert facts["net_paths"] == {
        "analog_conditioner_power": True,
        "analog_conditioner_outputs": True,
        "analog_conditioner_feedback": True,
        "analog_conditioner_sensor_path": True,
    }
    assert facts["gates"]["external_temperature_acquisition_function"] == "pass"
    assert facts["gates"]["external_temperature_sensor_function"] == "pass"
    assert facts["analog_output_nets"] == ["OUT1", "OUT2"]
    assert facts["analog_output_load_min_ohm"] == 10_000.0


def test_conditioner_observer_never_proves_output_when_header_path_is_corrupt():
    parts, pad_nets = _temperature_artifact(broken=True)
    facts = analog_conditioner_facts(parts, {}, pad_nets, supply_voltage=_fixture_supply_voltage)
    assert "analog_conditioner_outputs" not in facts["net_paths"]
    assert "external_temperature_acquisition_function" not in facts["gates"]
    assert "analog_output_nets" not in facts
    assert "analog_output_endpoint_v" not in facts


def test_conditioner_observer_rejects_wrong_documented_feedback_value():
    parts, pad_nets = _temperature_artifact()
    next(part for part in parts if part["ref"] == "R1")["value"] = "12k"
    facts = analog_conditioner_facts(parts, {}, pad_nets, supply_voltage=_fixture_supply_voltage)
    assert "analog_conditioner_feedback" not in facts["net_paths"]
    assert "external_temperature_acquisition_function" not in facts["gates"]
    assert "analog_output_nets" not in facts
    assert "analog_output_endpoint_v" not in facts


def _conditioner_artifact(sensor_type, channels, *, trim=False):
    """Build delivered-pad fixtures without recipe or net-name evidence."""
    parts, pad_nets = [], {}
    sensor_parts = {
        "temperature": ("MCP9700AT-E/TT", "Sensor_Temperature:MCP9700Ax-ETT", "Package_TO_SOT_SMD:SOT-23"),
        "humidity": ("HIH-4010-001", "hih-4010-001:HIH-4010-001", "hih-4010-001:HIH-4010-001_P2.54mm"),
        "ambient-light": ("TEMT6000X01", "Sensor_Optical:SFH309", "LED_SMD:LED_1206_3216Metric"),
        "hall-effect": ("DRV5055A3QDBZR", "Sensor_Magnetic:DRV5055A3xDBZxQ1", "Package_TO_SOT_SMD:SOT-23"),
        "current-sense": ("WSL2512R1000FEA", "Device:R", "Resistor_SMD:R_2512_6332Metric"),
    }
    for package in range((channels + 1) // 2):
        ref = f"U{package + 1}"
        parts.extend([
            {"ref": ref, "mpn": "MCP6002-I/SN", "symbol": "Amplifier_Operational:MCP6002-xSN", "footprint": "Package_SO:SOIC-8_3.9x4.9mm_P1.27mm"},
            {"ref": f"CA{package + 1}", "symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric", "value": "100nF"},
        ])
        pad_nets[ref, "8"] = pad_nets[f"CA{package + 1}", "1"] = "AV5"
        pad_nets[ref, "4"] = pad_nets[f"CA{package + 1}", "2"] = "AGND"
    for index in range(channels):
        number = index + 1
        package = index // 2 + 1
        plus, minus, output = ("3", "2", "1") if index % 2 == 0 else ("5", "6", "7")
        amp, signal, divider, feedback, out = f"U{package}", f"SIG{number}", f"DIV{number}", f"FB{number}", f"OUT{number}"
        mpn, symbol, footprint = sensor_parts[sensor_type]
        sensor_ref = f"SH{number}" if sensor_type == "current-sense" else f"S{number}"
        parts.extend([
            {"ref": sensor_ref, "mpn": mpn, "symbol": symbol, "footprint": footprint},
            {"ref": f"J{number}", "mpn": "TSW-103-07-G-S", "symbol": "Connector_Generic:Conn_01x03", "footprint": "Connector_PinHeader_2.54mm:PinHeader_1x03_P2.54mm_Vertical"},
        ])
        pad_nets[amp, plus] = signal if sensor_type in {"temperature", "current-sense"} else divider
        pad_nets[amp, output] = out
        pad_nets[f"J{number}", "1"] = "AGND"
        pad_nets[f"J{number}", "2"] = out
        if sensor_type == "temperature":
            pad_nets.update({
                (sensor_ref, "1"): "AV5", (sensor_ref, "2"): signal, (sensor_ref, "3"): "AGND",
                (amp, minus): feedback,
            })
            parts.extend([
                {"ref": f"CS{number}", "symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric", "value": "100nF"},
                {"ref": f"RT{number}", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "10k"},
                {"ref": f"RB{number}", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "20k"},
            ])
            pad_nets.update({
                (f"CS{number}", "1"): "AV5", (f"CS{number}", "2"): "AGND",
                (f"RT{number}", "1"): out, (f"RT{number}", "2"): feedback,
                (f"RB{number}", "1"): feedback, (f"RB{number}", "2"): "AGND",
            })
        elif sensor_type == "current-sense":
            pad_nets.update({
                (sensor_ref, "1"): signal, (sensor_ref, "2"): "AGND",
                (f"J{number}", "3"): signal, (amp, minus): feedback,
            })
            parts.extend([
                {"ref": f"RT{number}", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "95.3k"},
                {"ref": f"RB{number}", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "4.02k"},
            ])
            if trim:
                trim_net = f"TRIM{number}"
                parts.append({"ref": f"RV{number}", "mpn": "3296W-1-103LF", "symbol": "trim-pot-3296w-10k:3296W-1-103LF", "footprint": "trim-pot-3296w-10k:RES-ADJ-TH_3296W"})
                pad_nets.update({
                    (f"RT{number}", "1"): out, (f"RT{number}", "2"): trim_net,
                    (f"RV{number}", "1"): trim_net, (f"RV{number}", "2"): feedback, (f"RV{number}", "3"): feedback,
                })
            else:
                pad_nets.update({(f"RT{number}", "1"): out, (f"RT{number}", "2"): feedback})
            pad_nets[f"RB{number}", "1"] = feedback
            pad_nets[f"RB{number}", "2"] = "AGND"
        else:
            pad_nets[amp, minus] = out
            if sensor_type == "humidity":
                pad_nets.update({(sensor_ref, "1"): "AGND", (sensor_ref, "2"): signal, (sensor_ref, "3"): "AV5"})
                values = (("RT", "100k", signal, divider), ("RB", "100k", divider, "AGND"))
            elif sensor_type == "ambient-light":
                pad_nets.update({(sensor_ref, "1"): "AV5", (sensor_ref, "2"): signal})
                values = (("RL", "10k", signal, "AGND"), ("RT", "15k", signal, divider), ("RB", "10k", divider, "AGND"))
            else:
                pad_nets.update({(sensor_ref, "1"): "AV5", (sensor_ref, "2"): signal, (sensor_ref, "3"): "AGND"})
                values = (("RT", "15k", signal, divider), ("RB", "10k", divider, "AGND"))
            for label, value, left, right in values:
                ref = f"{label}{number}"
                parts.append({"ref": ref, "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": value})
                pad_nets[ref, "1"], pad_nets[ref, "2"] = left, right
    return parts, pad_nets


@pytest.mark.parametrize(
    ("sensor_type", "gate"),
    [
        ("temperature", "external_temperature_acquisition_function"),
        ("humidity", "external_humidity_conditioning_function"),
        ("ambient-light", "external_ambient_light_front_end_function"),
        ("hall-effect", "external_hall_effect_sensor_function"),
    ],
)
def test_function_gates_bind_complete_delivered_sensor_identity(sensor_type, gate):
    parts, pad_nets = _conditioner_artifact(sensor_type, 2)
    facts = analog_conditioner_facts(parts, {}, pad_nets, supply_voltage=_fixture_supply_voltage)

    assert facts["gates"][gate] == "pass"
    assert "external_temperature_acquisition_function" not in facts["gates"] or sensor_type == "temperature"


@pytest.mark.parametrize(
    ("sensor_type", "channels", "gate"),
    [
        ("temperature", 4, "external_temperature_channel_count_4"),
        ("humidity", 8, "external_humidity_channel_count_8"),
        ("ambient-light", 4, "external_ambient_light_channel_count_4"),
        ("ambient-light", 8, "external_ambient_light_channel_count_8"),
    ],
)
def test_sensor_count_gates_require_the_requested_complete_channel_count(sensor_type, channels, gate):
    parts, pad_nets = _conditioner_artifact(sensor_type, channels)
    facts = analog_conditioner_facts(parts, {}, pad_nets, supply_voltage=_fixture_supply_voltage)
    assert facts["gates"][gate] == "pass"

    insufficient_parts, insufficient_nets = _conditioner_artifact(sensor_type, channels - 1)
    insufficient = analog_conditioner_facts(insufficient_parts, {}, insufficient_nets, supply_voltage=_fixture_supply_voltage)
    assert gate not in insufficient["gates"]


def test_current_sense_gates_require_kelvin_feedback_and_functional_trim():
    parts, pad_nets = _conditioner_artifact("current-sense", 2, trim=True)
    facts = analog_conditioner_facts(parts, {}, pad_nets, supply_voltage=_fixture_supply_voltage)
    expected = {
        "external_current_sense_amplifier_function",
        "external_current_sense_conditioning_function",
        "external_current_sense_front_end_function",
        "external_current_sense_sensor_function",
        "external_current_sense_channel_count_2",
        "external_trim_potentiometer_function",
    }
    assert {gate for gate, verdict in facts["gates"].items() if verdict == "pass"} == expected

    pad_nets["J1", "3"] = "NOT_KELVIN"
    no_kelvin = analog_conditioner_facts(parts, {}, pad_nets, supply_voltage=_fixture_supply_voltage)
    assert not expected & no_kelvin["gates"].keys()

    parts, pad_nets = _conditioner_artifact("current-sense", 2, trim=True)
    pad_nets["RV1", "3"] = "OPEN"
    false_trim = analog_conditioner_facts(parts, {}, pad_nets, supply_voltage=_fixture_supply_voltage)
    assert not expected & false_trim["gates"].keys()

    parts, pad_nets = _conditioner_artifact("current-sense", 2, trim=True)
    next(part for part in parts if part["ref"] == "RT1")["value"] = "100k"
    broken_feedback = analog_conditioner_facts(parts, {}, pad_nets, supply_voltage=_fixture_supply_voltage)
    assert not expected & broken_feedback["gates"].keys()


def test_ambient_light_function_requires_collector_power():
    parts, pad_nets = _conditioner_artifact("ambient-light", 4)
    pad_nets["S1", "1"] = "OPEN_COLLECTOR"
    facts = analog_conditioner_facts(parts, {}, pad_nets, supply_voltage=_fixture_supply_voltage)
    assert "external_ambient_light_front_end_function" not in facts["gates"]
    assert "external_ambient_light_channel_count_4" not in facts["gates"]


def _temt6000_adc_artifact(*, fault=None):
    from kicraft.design.part_identity import reviewed_part

    controller = reviewed_part("stm32g030f6p6")
    parts = [
        {"ref": "S1", "mpn": "TEMT6000X01", "symbol": "Sensor_Optical:SFH309", "footprint": "LED_SMD:LED_1206_3216Metric"},
        {"ref": "R1", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "10k"},
        {"ref": "C1", "symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric", "value": "1nF"},
        {"ref": "U1", "mpn": "ADS1115IDGSR", "symbol": "ads1115:ADS1115IDGSR", "footprint": "ads1115:MSOP-10_L3.0-W3.0-P0.50-LS5.0-BL"},
        {"ref": "C2", "symbol": "Device:C", "footprint": "Capacitor_SMD:C_0603_1608Metric", "value": "100nF"},
        {"ref": "U2", "mpn": controller.identity, "symbol": controller.symbol, "footprint": controller.footprint},
        {"ref": "R2", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "4.7k"},
        {"ref": "R3", "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric", "value": "4.7k"},
    ]
    pad_nets = {
        ("S1", "1"): "V3P3", ("S1", "2"): "LIGHT",
        ("R1", "1"): "LIGHT", ("R1", "2"): "AGND",
        ("C1", "1"): "LIGHT", ("C1", "2"): "AGND",
        ("U1", "8"): "V3P3", ("U1", "3"): "AGND", ("U1", "4"): "LIGHT",
        ("C2", "1"): "V3P3", ("C2", "2"): "AGND",
        ("U1", "1"): "AGND", ("U1", "9"): "SDA", ("U1", "10"): "SCL",
        ("U2", "2"): "SDA", ("U2", "1"): "SCL",
        ("R2", "1"): "V3P3", ("R2", "2"): "SDA",
        ("R3", "1"): "V3P3", ("R3", "2"): "SCL",
    }
    if fault == "no_load":
        parts.remove(next(part for part in parts if part["ref"] == "R1"))
        pad_nets.pop(("R1", "1")); pad_nets.pop(("R1", "2"))
    elif fault == "collector":
        pad_nets["S1", "1"] = "OPEN"
    elif fault == "return":
        pad_nets["R1", "2"] = pad_nets["C1", "2"] = "OPEN_RETURN"
    elif fault == "wrong_supply_domain":
        pad_nets["U1", "8"] = pad_nets["C2", "1"] = "ADC_V3P3"
    elif fault == "unpowered_adc":
        pad_nets["U1", "8"] = pad_nets["C2", "1"] = "ADC_UNPOWERED"
    elif fault == "disconnected_adc":
        pad_nets["U1", "4"] = "OTHER"
    elif fault == "counterfeit":
        next(part for part in parts if part["ref"] == "S1")["footprint"] = "LED_SMD:LED_0603_1608Metric"
    elif fault == "no_filter":
        pad_nets["C1", "1"] = "OPEN"
    elif fault == "no_adc_decoupling":
        pad_nets["C2", "1"] = "OPEN"
    elif fault == "open_adc_sda":
        pad_nets["U1", "9"] = "OPEN"
    elif fault == "open_controller_scl":
        pad_nets["U2", "1"] = "OPEN"
    elif fault == "missing_pullup":
        pad_nets["R2", "1"] = "OPEN"
    elif fault == "invalid_address":
        pad_nets["U1", "1"] = "OPEN"
    elif fault == "address_collision":
        parts.append({**next(part for part in parts if part["ref"] == "U1"), "ref": "U3"})
        pad_nets.update({("U3", pin): net for (ref, pin), net in list(pad_nets.items()) if ref == "U1"})
    def supply_voltage(rail, ground):
        return 3.3 if (rail, ground) == ("V3P3", "AGND") else None
    return parts, pad_nets, supply_voltage


def test_temt6000_readout_proves_delivered_bias_filter_and_powered_adc_acquisition():
    parts, pad_nets, supply_voltage = _temt6000_adc_artifact()
    facts = temt6000_readout_facts(
        parts, {}, pad_nets,
        mcu_facts={"complete_mcu_refs": ["U2"], "complete_mcu_supplies": {"U2": {"rail": "V3P3", "ground": "AGND"}}},
        powered_mcu_refs={"U2"},
        supply_voltage=supply_voltage,
    )

    assert facts["ambient_light_acquisition_nets"] == ["LIGHT"]
    assert facts["channel_counts"]["ambient_light_acquisition"] == 1
    assert facts["gates"]["external_ambient_light_acquisition_function"] == "pass"


@pytest.mark.parametrize(
    "fault",
    ("no_load", "no_filter", "collector", "return", "wrong_supply_domain",
     "unpowered_adc", "disconnected_adc", "counterfeit", "no_adc_decoupling",
     "open_adc_sda", "open_controller_scl", "missing_pullup", "invalid_address", "address_collision"),
)
def test_temt6000_readout_rejects_meaningful_bias_and_acquisition_faults(fault):
    parts, pad_nets, supply_voltage = _temt6000_adc_artifact(fault=fault)
    facts = temt6000_readout_facts(
        parts, {}, pad_nets,
        mcu_facts={"complete_mcu_refs": ["U2"], "complete_mcu_supplies": {"U2": {"rail": "V3P3", "ground": "AGND"}}},
        powered_mcu_refs={"U2"},
        supply_voltage=supply_voltage,
    )

    assert "ambient_light_acquisition_nets" not in facts
    assert not facts["gates"]


def test_temt6000_readout_rejects_an_adc_without_a_readable_controller_path():
    parts, pad_nets, supply_voltage = _temt6000_adc_artifact()
    facts = temt6000_readout_facts(
        parts, {}, pad_nets, mcu_facts={}, powered_mcu_refs=set(),
        supply_voltage=supply_voltage,
    )
    assert not facts["gates"]


