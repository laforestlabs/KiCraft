from __future__ import annotations

from copy import deepcopy

from kicraft.eval.connector_artifact_evidence import connector_circuit_facts

def _facts(
    parts, pads, requested, *, outputs=(), complete_mcu_refs=None,
    powered_mcu_refs=(), complete_mcu_gpio_nets=None, complete_mcu_supplies=None,
):
    return connector_circuit_facts(
        parts,
        {},
        pads,
        requested=requested,
        state={},
        outputs=outputs,
        complete_mcu_refs={"U1"} if complete_mcu_refs is None else complete_mcu_refs,
        powered_mcu_refs=powered_mcu_refs,
        complete_mcu_gpio_nets=complete_mcu_gpio_nets,
        complete_mcu_supplies=complete_mcu_supplies,
    )


def _header(ref: str, contacts: int, nets: list[str]) -> tuple[dict, dict[tuple[str, str], str]]:
    return (
        {
            "ref": ref,
            "symbol": f"Connector_Generic:Conn_01x{contacts:02d}",
            "footprint": f"Connector_PinHeader_2.54mm:PinHeader_1x{contacts:02d}_P2.54mm_Vertical",
        },
        {(ref, str(pin)): net for pin, net in enumerate(nets, start=1)},
    )


def _attached(pads: dict[tuple[str, str], str]) -> dict[tuple[str, str], str]:
    """Add a non-connector delivered pad on every requested connector net."""
    return {
        **pads,
        **{("U_ATTACH", str(index)): net for index, net in enumerate(sorted(set(pads.values())), start=1)},
    }


def test_connector_contact_gates_require_reviewed_complete_native_contacts():
    header, header_pads = _header("J8", 8, [f"H{pin}" for pin in range(1, 9)])
    parts = [
        header,
        {
            "ref": "T1", "mpn": "WJ126V-5.0-02P-14-00A",
            "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P",
            "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
        },
        {
            "ref": "T2", "mpn": "WJ126V-5.0-02P-14-00A",
            "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P",
            "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
        },
        {
            "ref": "X1", "mpn": "B2B-XH-A(LF)(SN)",
            "symbol": "b2b-xh-a-lf-sn:B2B-XH-A",
            "footprint": "b2b-xh-a-lf-sn:CONN-TH_B2B-XH-A-LF-SN",
        },
        {
            "ref": "X2", "mpn": "B2B-XH-A(LF)(SN)",
            "symbol": "b2b-xh-a-lf-sn:B2B-XH-A",
            "footprint": "b2b-xh-a-lf-sn:CONN-TH_B2B-XH-A-LF-SN",
        },
    ]
    pads = {
        **header_pads,
        ("T1", "1"): "T1A", ("T1", "2"): "T1B",
        ("T2", "1"): "T2A", ("T2", "2"): "T2B",
        ("X1", "1"): "X1A", ("X1", "2"): "X1B",
        ("X2", "1"): "X2A", ("X2", "2"): "X2B",
    }
    pads = _attached(pads)
    requested = {
        "external_eight_pin_0p1_header",
        "external_four_screw_terminal_contacts",
        "external_two_jst_xh_connectors",
    }
    facts = _facts(parts, pads, requested)
    assert facts["gates"] == {gate: "pass" for gate in requested}

    # Four named BOM contacts do not qualify when a terminal is electrically open.
    floating = deepcopy(pads)
    floating[("T2", "2")] = ""
    assert "external_four_screw_terminal_contacts" not in _facts(
        parts, floating, requested
    )["gates"]

    # A 2.54 mm six-position header cannot claim the requested eight-contact form.
    wrong_count_parts = [part for part in parts if part["ref"] != "J8"]
    six, six_pads = _header("J6", 6, [f"S{pin}" for pin in range(1, 7)])
    assert "external_eight_pin_0p1_header" not in _facts(
        [*wrong_count_parts, six], _attached({**pads, **six_pads}), requested
    )["gates"]


def _screw_terminal_io_fixture():
    mcu = {
        "ref": "U1", "mpn": "ESP32-C3-MINI-1-N4",
        "symbol": "esp32-c3-mini-1-n4:ESP32-C3-MINI-1-N4",
        "footprint": "esp32-c3-mini-1-n4:WIFIM-SMD_ESP32-C3-MINI-1",
    }
    terminal = {
        "mpn": "WJ126V-5.0-02P-14-00A",
        "symbol": "screw-terminal-5mm-2p:WJ126V-5.0-2P",
        "footprint": "screw-terminal-5mm-2p:CONN-TH_WJ126V-5.0-2P",
    }
    parts = [mcu, {"ref": "T1", **terminal}, {"ref": "T2", **terminal}]
    pads = {
        ("U1", "1"): "GND", ("U1", "3"): "V33",
        ("U1", "12"): "IO_A", ("U1", "13"): "IO_B",
        ("T1", "1"): "IO_A", ("T1", "2"): "GND",
        ("T2", "1"): "IO_B", ("T2", "2"): "GND",
    }
    evidence = {
        "powered_mcu_refs": {"U1"},
        "complete_mcu_gpio_nets": {
            "IO_A": ["U1.12"],
            "IO_B": ["U1.13"],
        },
        "complete_mcu_supplies": {"U1": {"rail": "V33", "ground": "GND"}},
    }
    return parts, pads, evidence


def test_four_screw_terminal_io_gate_requires_powered_same_domain_mcu_gpio():
    requested = {"external_four_screw_terminal_io_contacts"}
    parts, pads, evidence = _screw_terminal_io_fixture()
    assert _facts(parts, pads, requested, **evidence)["gates"] == {
        "external_four_screw_terminal_io_contacts": "pass"
    }

    # The second two-contact terminal must land on the powered controller too.
    disconnected = deepcopy(pads)
    disconnected["T2", "1"] = ""
    assert not _facts(parts, disconnected, requested, **evidence)["gates"]

    # Two terminals containing only same-domain supply/return do not provide I/O.
    rail_only = deepcopy(pads)
    rail_only.update({
        ("T1", "1"): "V33", ("T1", "2"): "GND",
        ("T2", "1"): "V33", ("T2", "2"): "GND",
    })
    assert not _facts(parts, rail_only, requested, **evidence)["gates"]

    # Every selected terminal group needs an actual powered GPIO, not just one.
    one_group_rail_only = deepcopy(pads)
    one_group_rail_only["T2", "1"] = "V33"
    assert not _facts(parts, one_group_rail_only, requested, **evidence)["gates"]

    # A GPIO paired with a return from a different electrical domain is not valid.
    cross_domain = deepcopy(pads)
    cross_domain["T2", "2"] = "OTHER_GND"
    cross_domain["R_FAULT", "1"] = "OTHER_GND"
    assert not _facts(parts, cross_domain, requested, **evidence)["gates"]

    # Complete-core GPIO facts do not qualify until their supply is observed.
    assert not _facts(
        parts, pads, requested, **(evidence | {"powered_mcu_refs": set()})
    )["gates"]

    wrong_identity = deepcopy(parts)
    wrong_identity[1]["footprint"] = (
        "Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical"
    )
    assert not _facts(wrong_identity, pads, requested, **evidence)["gates"]


def _interface_fixture():
    mcu = {
        "ref": "U1", "mpn": "ESP32-C3-MINI-1-N4",
        "symbol": "esp32-c3-mini-1-n4:ESP32-C3-MINI-1-N4",
        "footprint": "esp32-c3-mini-1-n4:WIFIM-SMD_ESP32-C3-MINI-1",
    }
    uart, uart_pads = _header("JU", 4, ["GND", "V33", "TX", "RX"])
    i2c, i2c_pads = _header("JI", 4, ["GND", "V33", "SDA", "SCL"])
    spi, spi_pads = _header("JS", 6, ["GND", "V33", "SCLK", "MOSI", "MISO", "CS"])
    qwiic = {
        "ref": "JQ", "mpn": "SM04B-SRSS-TB(LF)(SN)",
        "symbol": "jst-sh-4p-qwiic:SM04B-SRSS-TB",
        "footprint": "jst-sh-4p-qwiic:CONN-SMD_4P-P1.00_SM04B-SRSS-TB-LF-SN",
    }
    qwiic_pads = {
        ("JQ", "1"): "GND", ("JQ", "2"): "V33",
        ("JQ", "3"): "SDA",
        ("JQ", "4"): "SCL",
    }
    pads = {
        **uart_pads, **i2c_pads, **spi_pads, **qwiic_pads,
        ("U1", "1"): "GND", ("U1", "3"): "V33",
        ("U1", "31"): "TX", ("U1", "30"): "RX",
        ("U1", "12"): "SDA", ("U1", "13"): "SCL",
        ("U1", "16"): "SCLK", ("U1", "18"): "MOSI",
        ("U1", "19"): "MISO", ("U1", "20"): "CS",
    }
    return [mcu, uart, i2c, spi, qwiic], pads


def test_protocol_headers_and_qwiic_require_native_capable_contacts_and_power_pairing():
    parts, pads = _interface_fixture()
    requested = {
        "external_uart_header_binding",
        "external_i2c_header_binding",
        "external_spi_header_binding",
        "external_qwiic_i2c_connector_binding",
    }
    facts = _facts(
        parts, pads, requested,
        outputs=({"output_net": "V33", "ground_net": "GND", "voltage": 3.3},),
    )
    assert facts["gates"] == {gate: "pass" for gate in requested}

    # A GPIO matrix can exchange roles, but two shorted contacts cannot carry
    # independent clock and data conductors.
    shorted_pads = deepcopy(pads)
    shorted_pads[("JQ", "4")] = shorted_pads[("JQ", "3")]
    assert "external_qwiic_i2c_connector_binding" not in _facts(
        parts, shorted_pads, {"external_qwiic_i2c_connector_binding"},
        outputs=({"output_net": "V33", "ground_net": "GND", "voltage": 3.3},),
    )["gates"]

    # A four-contact JST-XH is neither the reviewed 1.00 mm Qwiic receptacle
    # nor its fixed GND/3V3/SDA/SCL contact map.
    wrong_pinout_parts = deepcopy(parts)
    wrong_qwiic = next(part for part in wrong_pinout_parts if part["ref"] == "JQ")
    wrong_qwiic.update({
        "mpn": "B4B-XH-A(LF)(SN)",
        "symbol": "Connector_Generic:Conn_01x04",
        "footprint": "Connector_JST:JST_XH_B4B-XH-A_1x04_P2.50mm_Vertical",
    })
    assert "external_qwiic_i2c_connector_binding" not in _facts(
        wrong_pinout_parts, pads, {"external_qwiic_i2c_connector_binding"},
        outputs=({"output_net": "V33", "ground_net": "GND", "voltage": 3.3},),
    )["gates"]

    # A claimed 3.3 V label is not a Qwiic supply proof without a delivered
    # reviewed output observation.
    assert "external_qwiic_i2c_connector_binding" not in _facts(
        parts, pads, {"external_qwiic_i2c_connector_binding"}
    )["gates"]


def _buttons_fixture(*, shared_control: bool = False, invalid_pull: bool = False):
    parts = [{
        "ref": "U1", "mpn": "ESP32-C3-MINI-1-N4",
        "symbol": "esp32-c3-mini-1-n4:ESP32-C3-MINI-1-N4",
        "footprint": "esp32-c3-mini-1-n4:WIFIM-SMD_ESP32-C3-MINI-1",
    }]
    pads = {("U1", "1"): "GND", ("U1", "3"): "V33"}
    controls = ["BTN1", "BTN1" if shared_control else "BTN2", "BTN1" if shared_control else "BTN3"]
    for index, (pin, control) in enumerate(zip(("12", "13", "16"), controls, strict=True), start=1):
        parts.extend((
            {
                "ref": f"SW{index}", "mpn": "TL3342F260QG",
                "symbol": "Switch:SW_Push",
                "footprint": "Button_Switch_SMD:SW_SPST_TL3342",
            },
            {
                "ref": f"R{index}", "mpn": "ERJ-3EKF1002V", "value": "10k",
                "symbol": "Device:R", "footprint": "Resistor_SMD:R_0603_1608Metric",
            },
        ))
        pads.update({
            ("U1", pin): control,
            (f"SW{index}", "1"): control, (f"SW{index}", "2"): "GND",
            (f"R{index}", "1"): control,
            (f"R{index}", "2"): "GND" if invalid_pull else "V33",
        })
    return parts, pads


def test_three_push_buttons_require_distinct_mcu_controls_returns_and_default_pulls():
    requested = {"external_three_push_button_functions"}
    parts, pads = _buttons_fixture()
    assert "external_three_push_button_functions" not in _facts(parts, pads, requested)["gates"]
    assert _facts(parts, pads, requested, powered_mcu_refs={"U1"})["gates"] == {
        "external_three_push_button_functions": "pass"
    }
    for kwargs in ({"shared_control": True}, {"invalid_pull": True}):
        parts, pads = _buttons_fixture(**kwargs)
        assert "external_three_push_button_functions" not in _facts(
            parts, pads, requested, powered_mcu_refs={"U1"}
        )["gates"]
