import pytest

from kicraft.design.part_identity import reviewed_part
from kicraft.eval.mcu_artifact_evidence import controller_function_gates, mcu_circuit_facts


def _part(ref, identity, **extra):
    record = reviewed_part(identity)
    return {
        "ref": ref,
        "mpn": record.identity,
        "symbol": record.symbol,
        "footprint": record.footprint,
        **extra,
    }


def _ch32_artifact(*, duplicate_outputs=False, missing_programming=False, missing_power=False):
    parts = [
        _part("U1", "ch32v003j4m6"),
        {"ref": "C1", "value": "100nF"},
        {"ref": "J1"}, {"ref": "J2"},
    ]
    nets = {
        ("U1", "4"): "+3V3", ("U1", "2"): "GND", ("U1", "8"): "SWIO",
        ("U1", "1"): "IO_A", ("U1", "3"): "IO_B",
        ("C1", "1"): "+3V3", ("C1", "2"): "GND",
        ("J1", "1"): "+3V3", ("J1", "2"): "GND", ("J1", "3"): "SWIO",
        ("J2", "1"): "IO_A", ("J2", "2"): "IO_B",
    }
    if duplicate_outputs:
        nets["U1", "3"] = "IO_A"
        nets["J2", "2"] = "IO_A"
    if missing_programming:
        del nets["J1", "3"]
    if missing_power:
        del nets["U1", "4"]
    return parts, nets


def test_ch32_core_requires_delivered_programming_and_distinct_gpio_paths():
    parts, nets = _ch32_artifact()
    facts = mcu_circuit_facts(parts, {}, nets)

    assert facts["complete_mcu_refs"] == ["U1"]
    assert facts["net_paths"]["ch32v003j4m6_complete_core"]
    assert facts["channel_counts"]["complete_mcu_gpio_output"] == 2
    assert not facts["gates"]

    missing_parts, missing_nets = _ch32_artifact(missing_programming=True)
    assert not mcu_circuit_facts(missing_parts, {}, missing_nets)["complete_mcu_refs"]
    unpowered_parts, unpowered_nets = _ch32_artifact(missing_power=True)
    assert not mcu_circuit_facts(unpowered_parts, {}, unpowered_nets)["complete_mcu_refs"]

    duplicate_parts, duplicate_nets = _ch32_artifact(duplicate_outputs=True)
    duplicate = mcu_circuit_facts(duplicate_parts, {}, duplicate_nets)
    assert "complete_mcu_gpio_output" not in duplicate["channel_counts"]
    assert not duplicate["gates"]


def _c3_artifact(*, missing_decoupling=False):
    parts = [
        _part("U1", "esp32-c3-mini-1-n4"),
        {"ref": "C1", "value": "10uF"}, {"ref": "C2", "value": "100nF"},
        {"ref": "C3", "value": "1uF"},
        {"ref": "R1", "value": "10k"}, {"ref": "R2", "value": "10k"},
        {"ref": "R3", "value": "10k"}, {"ref": "SW1"}, {"ref": "SW2"},
        {"ref": "J1"}, {"ref": "J2"},
    ]
    ground_pins = ("1", "2", "11", "14", *tuple(str(pin) for pin in range(36, 54)))
    nets = {
        ("U1", "3"): "+3V3", ("U1", "8"): "EN", ("U1", "23"): "BOOT",
        ("U1", "22"): "GPIO8", ("U1", "26"): "USB_DM", ("U1", "27"): "USB_DP",
        ("U1", "5"): "IO_A", ("U1", "6"): "IO_B",
        **{("U1", pin): "GND" for pin in ground_pins},
        ("C1", "1"): "+3V3", ("C1", "2"): "GND",
        ("C2", "1"): "+3V3", ("C2", "2"): "GND",
        ("C3", "1"): "EN", ("C3", "2"): "GND",
        ("R1", "1"): "+3V3", ("R1", "2"): "EN",
        ("R2", "1"): "+3V3", ("R2", "2"): "BOOT",
        ("R3", "1"): "+3V3", ("R3", "2"): "GPIO8",
        ("SW1", "1"): "EN", ("SW1", "2"): "GND",
        ("SW2", "1"): "BOOT", ("SW2", "2"): "GND",
        ("J1", "1"): "USB_DM", ("J1", "2"): "USB_DP",
        ("J2", "1"): "IO_A", ("J2", "2"): "IO_B",
    }
    if missing_decoupling:
        parts = [part for part in parts if part["ref"] != "C2"]
        del nets["C2", "1"]
        del nets["C2", "2"]
    return parts, nets


def test_esp32_c3_core_requires_full_native_usb_recovery_support():
    parts, nets = _c3_artifact()
    facts = mcu_circuit_facts(parts, {}, nets)
    assert facts["complete_mcu_refs"] == ["U1"]
    assert facts["net_paths"]["esp32_c3_complete_core"]

    defective_parts, defective_nets = _c3_artifact(missing_decoupling=True)
    defective = mcu_circuit_facts(defective_parts, {}, defective_nets)
    assert "esp32_c3_complete_core" not in defective["net_paths"]
    assert not defective["complete_mcu_refs"]


def _s3_mini_artifact(*, missing_usb=False):
    parts = [
        _part("U1", "esp32-s3-mini-1-n8"),
        {"ref": "C1", "value": "10uF"}, {"ref": "C2", "value": "100nF"},
        {"ref": "C3", "value": "1uF"},
        {"ref": "R1", "value": "10k"}, {"ref": "R2", "value": "10k"},
        {"ref": "SW1"}, {"ref": "SW2"}, {"ref": "J1"},
    ]
    ground_pins = ("1", "2", "42", "43", *tuple(str(pin) for pin in range(46, 61)), "GND")
    nets = {
        ("U1", "3"): "+3V3", ("U1", "45"): "EN", ("U1", "4"): "BOOT",
        ("U1", "23"): "USB_DM", ("U1", "24"): "USB_DP",
        **{("U1", pin): "GND" for pin in ground_pins},
        ("C1", "1"): "+3V3", ("C1", "2"): "GND",
        ("C2", "1"): "+3V3", ("C2", "2"): "GND",
        ("C3", "1"): "EN", ("C3", "2"): "GND",
        ("R1", "1"): "+3V3", ("R1", "2"): "EN",
        ("R2", "1"): "+3V3", ("R2", "2"): "BOOT",
        ("SW1", "1"): "EN", ("SW1", "2"): "GND",
        ("SW2", "1"): "BOOT", ("SW2", "2"): "GND",
        ("J1", "1"): "USB_DM", ("J1", "2"): "USB_DP",
    }
    if missing_usb:
        del nets["J1", "2"]
    return parts, nets


def test_esp32_s3_mini_core_requires_native_usb_path():
    parts, nets = _s3_mini_artifact()
    facts = mcu_circuit_facts(parts, {}, nets)
    assert facts["complete_mcu_refs"] == ["U1"]
    assert facts["net_paths"]["esp32_s3_complete_core"]

    defective_parts, defective_nets = _s3_mini_artifact(missing_usb=True)
    assert not mcu_circuit_facts(defective_parts, {}, defective_nets)["complete_mcu_refs"]


def test_s3u_requires_usb_and_undriven_psram_straps():
    parts, nets = _s3_mini_artifact()
    parts[0] = _part("U1", "esp32-s3-mini-1u-n4r2")
    del nets["U1", "GND"]
    nets.update({("U1", str(pin)): "GND" for pin in range(61, 66)})
    assert mcu_circuit_facts(parts, {}, nets)["complete_mcu_refs"] == ["U1"]
    for pin in ("26", "41"):
        broken = {**nets, ("U1", pin): "+3V3"}
        assert not mcu_circuit_facts(parts, {}, broken)["complete_mcu_refs"]
    del nets["J1", "2"]
    assert not mcu_circuit_facts(parts, {}, nets)["complete_mcu_refs"]

@pytest.mark.parametrize(("family", "tx_pin", "rx_pin", "en_pin", "boot_pin"), [
    ("c3", "31", "30", "8", "23"),
    ("s3", "39", "40", "45", "4"),
])
def test_reviewed_rom_uart_core_requires_each_header_contact(family, tx_pin, rx_pin, en_pin, boot_pin):
    if family == "c3":
        parts, nets = _c3_artifact()
        parts[0] = _part("U1", "esp32-c3-mini-1-h4x")
        usb_pins = ("26", "27")
    else:
        parts, nets = _s3_mini_artifact()
        parts[0] = _part("U1", "esp32-s3-mini-1u-n4r2")
        del nets["U1", "GND"]
        nets.update({("U1", str(pin)): "GND" for pin in range(61, 66)})
        usb_pins = ("23", "24")
    for pin in usb_pins:
        nets.pop(("U1", pin))
    parts = [part for part in parts if part["ref"] != "J1"]
    nets = {key: value for key, value in nets.items() if key[0] != "J1"}
    parts.append({"ref": "J9", "mpn": "68000-406HLF"})
    nets.update({
        ("U1", tx_pin): "TX", ("U1", rx_pin): "RX",
        ("J9", "1"): "+3V3", ("J9", "2"): "GND",
        ("J9", "3"): "TX", ("J9", "4"): "RX",
        ("J9", "5"): nets["U1", en_pin], ("J9", "6"): nets["U1", boot_pin],
    })
    assert mcu_circuit_facts(parts, {}, nets)["complete_mcu_refs"] == ["U1"]
    for pin in range(1, 7):
        broken = {key: value for key, value in nets.items() if key != ("J9", str(pin))}
        assert not mcu_circuit_facts(parts, {}, broken)["complete_mcu_refs"]
    broken = {key: value for key, value in nets.items() if key != ("U1", "3")}
    assert not mcu_circuit_facts(parts, {}, broken)["complete_mcu_refs"]



def _usb_c_series_artifact(factory):
    parts, nets = factory()
    parts = [part for part in parts if part["ref"] != "J1"]
    parts.extend([
        _part("J1", "type-c-31-m-12"),
        _part("U2", "usblc6-2sc6"),
        {"ref": "R20", "symbol": "Device:R", "value": "22R"},
        {"ref": "R21", "symbol": "Device:R", "value": "22R"},
    ])
    del nets["J1", "1"]
    del nets["J1", "2"]
    nets.update({
        ("R20", "1"): "USB_DM", ("R20", "2"): "DM_ESD",
        ("R21", "1"): "USB_DP", ("R21", "2"): "DP_ESD",
        ("U2", "1"): "DM_ESD", ("U2", "6"): "DM_CONN",
        ("U2", "3"): "DP_ESD", ("U2", "4"): "DP_CONN",
        ("J1", "A7"): "DM_CONN", ("J1", "B7"): "DM_CONN",
        ("J1", "A6"): "DP_CONN", ("J1", "B6"): "DP_CONN",
    })
    return parts, nets


@pytest.mark.parametrize(
    ("factory", "core_path"),
    [(_c3_artifact, "esp32_c3_complete_core"),
     (_s3_mini_artifact, "esp32_s3_complete_core")],
    ids=["c3", "s3"],
)
def test_esp32_core_accepts_delivered_usb_series_esd_and_direct_header(factory, core_path):
    for parts, nets in (factory(), _usb_c_series_artifact(factory)):
        facts = mcu_circuit_facts(parts, {}, nets)
        assert facts["complete_mcu_refs"] == ["U1"]
        assert facts["net_paths"][core_path]


@pytest.mark.parametrize(
    ("factory", "core_path"),
    [(_c3_artifact, "esp32_c3_complete_core"),
     (_s3_mini_artifact, "esp32_s3_complete_core")],
    ids=["c3", "s3"],
)
@pytest.mark.parametrize(("dm_pin", "dp_pin"), [("A7", "A6"), ("B7", "B6")])
@pytest.mark.parametrize("resistance", ["0.1R", "100R"])
def test_esp32_core_accepts_reviewed_usb_contact_and_resistance_boundaries(
    factory, core_path, dm_pin, dp_pin, resistance
):
    parts, nets = _usb_c_series_artifact(factory)
    for pin in ("A7", "B7", "A6", "B6"):
        if pin not in {dm_pin, dp_pin}:
            del nets["J1", pin]
    for part in parts:
        if part["ref"] in {"R20", "R21"}:
            part["value"] = resistance
    facts = mcu_circuit_facts(parts, {}, nets)
    assert facts["complete_mcu_refs"] == ["U1"]
    assert facts["net_paths"][core_path]


@pytest.mark.parametrize(
    ("factory", "core_path"),
    [(_c3_artifact, "esp32_c3_complete_core"),
     (_s3_mini_artifact, "esp32_s3_complete_core")],
    ids=["c3", "s3"],
)
@pytest.mark.parametrize(
    "defect",
    [
        "open_dm", "open_dp", "missing_dm_contact", "missing_dp_contact",
        "crossed", "shorted", "bridged_lines", "split_connectors",
        "half_crossed", "opposite_dm_contact", "opposite_dp_contact",
        "high_resistance", "unknown_resistance", "unsupported_bridge",
        "missing_resistor_pad", "missing_esd_pad", "unreviewed_esd",
        "supply_contacts", "missing_source", "same_source",
    ],
)
def test_esp32_core_rejects_incomplete_or_invalid_usb_paths(factory, core_path, defect):
    parts, nets = _usb_c_series_artifact(factory)
    if defect == "open_dm":
        nets["R20", "2"] = "OPEN_DM"
    elif defect == "open_dp":
        nets["R21", "2"] = "OPEN_DP"
    elif defect in {"missing_dm_contact", "missing_dp_contact"}:
        for pin in (("A7", "B7") if defect == "missing_dm_contact" else ("A6", "B6")):
            del nets["J1", pin]
    elif defect == "crossed":
        for pin in ("A7", "B7"):
            nets["J1", pin] = "DP_CONN"
        for pin in ("A6", "B6"):
            nets["J1", pin] = "DM_CONN"
    elif defect == "half_crossed":
        nets["J1", "B7"] = "DP_CONN"
        nets["J1", "B6"] = "DM_CONN"
    elif defect == "opposite_dm_contact":
        nets["J1", "B7"] = "DP_CONN"
    elif defect == "opposite_dp_contact":
        nets["J1", "B6"] = "DM_CONN"
    elif defect == "shorted":
        nets["U2", "4"] = "DM_CONN"
        for pin in ("A6", "B6"):
            nets["J1", pin] = "DM_CONN"
    elif defect == "bridged_lines":
        parts.append({"ref": "R22", "symbol": "Device:R", "value": "22R"})
        nets.update({("R22", "1"): "DM_ESD", ("R22", "2"): "DP_ESD"})
    elif defect == "split_connectors":
        parts.append(_part("J3", "type-c-31-m-12"))
        for pin in ("A6", "B6"):
            nets["J3", pin] = nets.pop(("J1", pin))
    elif defect in {"high_resistance", "unknown_resistance", "unsupported_bridge"}:
        resistor = next(part for part in parts if part["ref"] == "R20")
        if defect == "unsupported_bridge":
            resistor["symbol"] = "Device:L"
        else:
            resistor["value"] = "101R" if defect == "high_resistance" else "unknown"
    elif defect == "missing_resistor_pad":
        del nets["R20", "2"]
    elif defect == "missing_esd_pad":
        del nets["U2", "3"]
    elif defect == "unreviewed_esd":
        esd = next(part for part in parts if part["ref"] == "U2")
        esd.update({"mpn": "unreviewed", "symbol": "Device:D_TVS", "footprint": ""})
    elif defect == "supply_contacts":
        for pin in ("A7", "B7", "A6", "B6"):
            del nets["J1", pin]
        nets["J1", "A4B9"] = "DM_CONN"
        nets["J1", "B4A9"] = "DP_CONN"
    else:
        dm_pin, dp_pin = (("26", "27") if factory is _c3_artifact else ("23", "24"))
        if defect == "missing_source":
            del nets["U1", dm_pin]
        else:
            nets["U1", dp_pin] = nets["U1", dm_pin]

    facts = mcu_circuit_facts(parts, {}, nets)
    assert not facts["complete_mcu_refs"]
    assert core_path not in facts["net_paths"]


def test_controller_gates_require_complete_powered_core_and_bound_peripherals():
    parts, nets = _ch32_artifact()
    facts = mcu_circuit_facts(parts, {}, nets)
    connector_facts = {
        "gates": {
            "external_qwiic_i2c_connector_binding": "pass",
            "external_uart_header_binding": "pass",
            "external_two_jst_xh_connectors": "pass",
        },
    }
    analog_facts = {
        "gates": {"external_current_sense_sensor_function": "pass"},
        "analog_output_nets": ["IO_A"],
    }
    supply = lambda rail, ground: 3.3 if (rail, ground) == ("+3V3", "GND") else None
    gates = controller_function_gates(
        parts, {}, nets, state={}, mcu_facts=facts, connector_facts=connector_facts,
        analog_facts=analog_facts, outputs={}, supply_voltage=supply,
    )
    assert gates == {
        "external_development_board_function": "pass",
        "external_interface_board_function": "pass",
        "external_sensor_breakout_function": "pass",
    }

    unpowered = controller_function_gates(
        parts, {}, nets, state={}, mcu_facts=facts, connector_facts=connector_facts,
        analog_facts=analog_facts, outputs={}, supply_voltage=lambda *_: None,
    )
    assert not unpowered
    unbound_analog = controller_function_gates(
        parts, {}, nets, state={}, mcu_facts=facts, connector_facts=connector_facts,
        analog_facts=analog_facts | {"analog_output_nets": ["OTHER_NET"]},
        outputs={}, supply_voltage=supply,
    )
    assert "external_sensor_breakout_function" not in unbound_analog
