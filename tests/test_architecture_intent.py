"""The intent-shaped architecture slot and its derivation.

Covers `docs/plans/architecture-constructive-slot-2026-09-14.md`: the derivation writes the
bookkeeping the model used to hand-write (net names, port bindings, endpoints and directions,
board-edge connectors), and refuses only what it cannot derive from. The reference case is the
real board the diagnosis was measured on (`KC-WGJ6XE`): its intent-shaped slot commits through the
real architecture normalization with zero blocking diagnostics.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.design.architecture_intent import (
    ArchitectureIntent,
    ArchitectureIntentError,
    derive_architecture,
)
from kicraft.server.stage_contracts import _normalize_stage_response
from kicraft.server import stage_runtime as stage_driver_mod

REPO = Path(__file__).resolve().parents[1]


def _frozen_prompt_state(board: str = "825") -> dict:
    state_path = Path.home() / ".kicraft" / "projects" / "1" / board / ".kicraft" / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    return {"intent": state["intent"], "functional_spec": state["functional_spec"]}


def _hub75_intent() -> dict:
    """The reference board's intent: USB-C 5 V in, ESP32-S3, HUB75 display, LED and speaker out."""
    return {
        "topologies": {
            "POWER": "Synchronous buck converter, VBUS to 3.3V at 1 A",
            "MCU": "ESP32-S3-WROOM-1-N16R8 native-USB controller",
        },
        "comms_protocols": ["USB 2.0 FS", "HUB75"],
        "mcu_present": True,
        "power": {
            "rails": {
                "VBUS": {"voltage": 5.0, "from": "usb_power.vbus"},
                "+3V3": {"voltage": 3.3, "from": "buck.output"},
            }
        },
        "sheets": [
            {
                "name": "USB INPUT",
                "stem": "USB_INPUT",
                "role": "power_input",
                "function": "Accept USB-C power.",
            },
            {
                "name": "POWER",
                "stem": "POWER",
                "role": "regulator",
                "function": "Synchronous buck converter regulating VBUS to 3.3V at 1 A for the logic rail.",
            },
            {
                "name": "MCU",
                "stem": "MCU",
                "role": "mcu",
                "function": "Run firmware and generate the interface signals.",
            },
            {
                "name": "HUB75",
                "stem": "HUB75",
                "role": "display",
                "function": "Level-shift the HUB75 display interface.",
            },
            {
                "name": "LED",
                "stem": "LED",
                "role": "driver",
                "function": "Drive the addressable LED string data line.",
            },
        ],
        "requirements": [
            {
                "id": "usb_power",
                "sheet": "USB INPUT",
                "role": "power_input",
                "family": "usb-c-power-sink",
                "functional_blocks": ["USB_C_PD_INPUT"],
            },
            {
                "id": "buck",
                "sheet": "POWER",
                "role": "regulator",
                "family": "tlv62569-3v3",
                "supply": "VBUS",
                "functional_blocks": ["POWER_DISTRIBUTION"],
            },
            {
                "id": "esp32",
                "sheet": "MCU",
                "role": "mcu_core",
                "family": "esp32-s3-wroom-1-module",
                "exact_part": "ESP32-S3-WROOM-1-N16R8",
                "supply": "+3V3",
                "programming": "native_usb",
                "functional_blocks": ["ESP32_S3_CONTROLLER"],
            },
            {
                "id": "hub75",
                "sheet": "HUB75",
                "role": "bus_interface",
                "family": "hub75-level-shift-interface",
                "supply": "VBUS",
                "functional_blocks": ["HUB75_DISPLAY_INTERFACE"],
            },
            {
                "id": "led",
                "sheet": "LED",
                "role": "driver",
                "family": "ws2812-output",
                "supply": "VBUS",
                "functional_blocks": ["ADDRESSABLE_LED_OUTPUT"],
            },
        ],
        "signals": [
            {"name": "USB_D_P", "from": "esp32.usb_dp", "to": "edge:USB_DATA"},
            {"name": "USB_D_N", "from": "esp32.usb_dm", "to": "edge:USB_DATA"},
            {
                "name": "HUB75_R{n}",
                "from": "esp32.output_hub75_r{n}",
                "to": "hub75.r{n}",
                "start": 0,
                "end": 1,
            },
            {
                "name": "HUB75_G{n}",
                "from": "esp32.output_hub75_g{n}",
                "to": "hub75.g{n}",
                "start": 0,
                "end": 1,
            },
            {
                "name": "HUB75_B{n}",
                "from": "esp32.output_hub75_b{n}",
                "to": "hub75.b{n}",
                "start": 0,
                "end": 1,
            },
            {"name": "HUB75_ADDR", "from": "esp32.output_hub75_a", "to": "hub75.addr_a"},
            {"name": "HUB75_ADDR_B", "from": "esp32.output_hub75_b", "to": "hub75.addr_b"},
            {"name": "HUB75_ADDR_C", "from": "esp32.output_hub75_c", "to": "hub75.addr_c"},
            {"name": "HUB75_CLK", "from": "esp32.output_hub75_clk", "to": "hub75.clk"},
            {"name": "HUB75_LAT", "from": "esp32.output_hub75_lat", "to": "hub75.lat"},
            {"name": "HUB75_OE", "from": "esp32.output_hub75_oe", "to": "hub75.oe"},
            {"name": "LED_DATA", "from": "esp32.output_led", "to": "led.data_in"},
            {
                "name": "LED_OUT",
                "from": "led.data_out",
                "to": "edge:LED_STRING",
                "rails": ["VBUS"],
            },
            {"name": "SPEAKER_PWM", "from": "esp32.output_pwm0", "to": "edge:SPEAKER"},
        ],
        "assumptions": ["USB-C input is a 5 V sink using standard CC pull-downs (defaulted)"],
    }


def _half_usb_pair_intent() -> dict:
    """The reference intent with the USB_D_N line landing on a test header, not the socket.

    The USB socket therefore receives only `usb_dp`: the one defect is an incomplete data
    connector. `esp32.usb_dm` is still wired (to the header), so the draft trips exactly
    that one refusal rather than the required-port gate reporting the same missing signal
    a second time.
    """
    intent = _hub75_intent()
    intent["sheets"].append(
        {
            "name": "USB TEST",
            "stem": "USB_TEST",
            "role": "connector",
            "function": "USB data test points.",
        }
    )
    intent["requirements"].append(
        {
            "id": "usb_test",
            "sheet": "USB TEST",
            "role": "connector",
            "family": "pin-header",
            "parameters": {"rows": 1, "gender": "male"},
            "functional_blocks": ["USB_C_PD_INPUT"],
        }
    )
    intent["signals"] = [
        {**row, "to": "usb_test.pin1"} if row["name"] == "USB_D_N" else row
        for row in intent["signals"]
    ]
    return intent


def _requirement(architecture, requirement_id: str):
    return next(row for row in architecture.requirements if row.id == requirement_id)


def _net(architecture, name: str):
    return next(row for row in architecture.inter_sheet_nets if row.name == name)


def test_reference_intent_commits_with_no_blocking_diagnostics():
    """The measured failure case: the derived slot needs no correction at all."""
    if not (
        Path.home() / ".kicraft" / "projects" / "1" / "825" / ".kicraft" / "state.json"
    ).is_file():
        pytest.skip("frozen reference board KC-WGJ6XE is not on this machine")
    architecture = derive_architecture(_hub75_intent())
    normalized, _ = _normalize_stage_response(
        "architecture",
        architecture.model_dump(exclude_none=True),
        _frozen_prompt_state(),
    )
    assert [row["recipe"] for row in normalized["recipe_selections"]] == [
        "esp32-s3-wroom-1-minimal@1",
        "hub75-sn74hct245-interface@1",
        "tlv62569-3v3@1",
        "usb-c-5v-sink@1",
        "usb-c-usb2-device@1",
        "ws2812-output@1",
    ]
    # The two compiler-built headers need a BOM pass, exactly as they do on the explicit path.
    assert normalized["unresolved_requirement_ids"] == ["esp32_speaker", "led_led_string"]


def test_signal_endpoints_are_bound_and_directed_on_both_sheets():
    architecture = derive_architecture(_hub75_intent())
    hub75 = _requirement(architecture, "hub75")
    esp32 = _requirement(architecture, "esp32")
    assert esp32.ports["output_hub75_r0"] == "HUB75_R0"
    assert hub75.ports["r0"] == "HUB75_R0"
    assert [(row.sheet, row.direction) for row in _net(architecture, "HUB75_R0").endpoints] == [
        ("MCU", "output"),
        ("HUB75", "input"),
    ]
    assert [(row.sheet, row.direction) for row in _net(architecture, "USB_D_P").endpoints] == [
        ("MCU", "bidirectional"),
        ("USB DATA", "bidirectional"),
    ]
    assert {
        _requirement(architecture, requirement_id).compiler_origin
        for requirement_id in ("esp32_speaker", "led_led_string")
    } == {"edge_connector"}


def test_ranges_expand_to_one_net_per_index():
    architecture = derive_architecture(_hub75_intent())
    names = {row.name for row in architecture.inter_sheet_nets}
    assert {"HUB75_R0", "HUB75_R1", "HUB75_B0", "HUB75_B1"} <= names


def test_unused_groundable_recipe_port_is_tied_low():
    """`hub75.addr_d` is a spare address channel on a 1/16-scan panel: tied, not rejected."""
    architecture = derive_architecture(_hub75_intent())
    assert _requirement(architecture, "hub75").ports["addr_d"] == "GND"
    assert any("addr_d tied to GND" in row for row in architecture.assumptions)


def test_an_unbound_enable_port_is_tied_to_the_input_rail():
    """A `_buck` regulator with nothing wired to `enable` keeps EN on its input rail.

    The fleet's always-on regulator wires EN to VIN in the emitted circuit; publishing the control
    port must not change that default, so the compiler derives the tie the recipe declares.
    """
    architecture = derive_architecture(_hub75_intent())
    assert _requirement(architecture, "buck").ports["enable"] == "VBUS"
    assert any("enable tied to input (VBUS) (derived)" in row for row in architecture.assumptions)


def test_native_usb_pair_gets_a_real_data_connector():
    architecture = derive_architecture(_hub75_intent())
    connector = _requirement(architecture, "esp32_usb_data")
    assert connector.family == "usb-c-usb2-device"
    # The MCU recipe already expands the 22R MCU-side pair: the socket must not add a second.
    assert connector.parameters == {"series_resistors": False}
    assert connector.ports == {
        "vbus": "VBUS",
        "gnd": "GND",
        "usb_dm": "USB_D_N",
        "usb_dp": "USB_D_P",
    }


def test_off_board_signal_gets_a_connector_holding_its_signal_gnd_and_rails():
    architecture = derive_architecture(_hub75_intent())
    led_string = _requirement(architecture, "led_led_string")
    assert led_string.role == "connector"
    assert led_string.ports == {"pin1": "LED_OUT", "pin2": "GND", "pin3": "VBUS"}
    speaker = _requirement(architecture, "esp32_speaker")
    assert speaker.ports == {"pin1": "SPEAKER_PWM", "pin2": "GND"}


def test_signal_restating_a_rail_connection_joins_the_rail():
    """A model that both declares a rail and names the same wire is not refused for it."""
    intent = _hub75_intent()
    intent["signals"] = [
        *intent["signals"],
        {"name": "VBUS_IN", "from": "usb_power.vbus", "to": "buck.input"},
    ]
    architecture = derive_architecture(intent)
    assert _requirement(architecture, "buck").ports["input"] == "VBUS"
    assert not any(row.name == "VBUS_IN" for row in architecture.inter_sheet_nets)
    assert any("join rail VBUS" in row for row in architecture.assumptions)


def test_signal_restating_a_rail_at_the_edge_exposes_the_rail():
    """The same statement pointed off-board gets a connector carrying that rail, not a crash.

    The canary (`r2r-dac`, 2026-09-15) hit `KeyError: 'logic_power_power_input'` here: this
    branch opens the edge connector without binding a signal to it, and the close-out loop
    indexed `bindings[connector_id]` directly.
    """
    intent = _hub75_intent()
    intent["signals"] = [
        *intent["signals"],
        {"name": "VBUS_IN", "from": "usb_power.vbus", "to": "edge:POWER_INPUT"},
    ]
    architecture = derive_architecture(intent)
    connector = _requirement(architecture, "usb_power_power_input")
    assert connector.sheet == "POWER INPUT"
    assert sorted(connector.ports) == ["pin1", "pin2"]
    assert connector.ports["pin1"] == "GND"
    assert connector.ports["pin2"] == "VBUS"
    # The rail the connector exposes gains this sheet as an endpoint (direction input:
    # a header that carries the rail out is fed by it).
    assert ("POWER INPUT", "input") in [
        (row.sheet, row.direction) for row in _net(architecture, "VBUS").endpoints
    ]


def test_edge_label_naming_a_declared_sheet_puts_the_connector_there():
    intent = _hub75_intent()
    intent["sheets"].append(
        {
            "name": "SPEAKER",
            "stem": "SPEAKER",
            "role": "connector",
            "function": "Speaker output connector.",
        }
    )
    intent["signals"] = [
        row if row["name"] != "SPEAKER_PWM" else {**row, "to": "edge:SPEAKER"}
        for row in intent["signals"]
    ]
    architecture = derive_architecture(intent)
    speaker = next(
        row
        for row in architecture.requirements
        if row.sheet == "SPEAKER" and row.role == "connector"
    )
    assert speaker.ports == {"pin1": "SPEAKER_PWM", "pin2": "GND"}
    assert [(row.sheet, row.direction) for row in _net(architecture, "SPEAKER_PWM").endpoints] == [
        ("MCU", "output"),
        ("SPEAKER", "input"),
    ]


def test_interfaces_follow_the_ports_the_design_bound():
    """A declared interface with no bound member is not a request, and a bound member implies one."""
    intent = _hub75_intent()
    intent["requirements"] = [
        {**row, "interfaces": ["parallel_output", "pwm", "i2c_controller"]}
        if row["id"] == "esp32"
        else row
        for row in intent["requirements"]
    ]
    architecture = derive_architecture(intent)
    mcu = _requirement(architecture, "esp32")
    # The MCU bound `output_*` pins only: no bus was requested, so no bus is asked of the allocator.
    assert mcu.interfaces == []
    assert "parallel_output_count" not in mcu.parameters

    intent["requirements"] = [
        {**row, "interfaces": ["i2c_controller"]} if row["id"] == "esp32" else row
        for row in intent["requirements"]
    ]
    intent["signals"] = [
        *intent["signals"],
        {"name": "I2C_SDA", "from": "esp32.sda", "to": "esp32.scl"},
    ]
    # `sda`/`scl` bound on an allocatable recipe are an I2C request even before the bus is
    # complete; the interface follows the ports, and the resolver still reports an idle bus.
    architecture = derive_architecture(intent)
    assert _requirement(architecture, "esp32").interfaces == ["i2c_controller"]


def test_parallel_output_count_comes_from_the_bound_ports():
    intent = _hub75_intent()
    intent["sheets"].append(
        {
            "name": "LED BANK",
            "stem": "LED_BANK",
            "role": "connector",
            "function": "Three-channel LED output bank.",
        }
    )
    intent["requirements"] = [
        {**row, "interfaces": ["parallel_output"]} if row["id"] == "esp32" else row
        for row in intent["requirements"]
    ] + [
        {
            "id": "led_bank",
            "sheet": "LED BANK",
            "role": "connector",
            "family": "pin-header",
            "parameters": {"rows": 1, "gender": "male"},
            "functional_blocks": ["ADDRESSABLE_LED_OUTPUT"],
        }
    ]
    intent["signals"] = [
        {"name": f"LED_{n}", "from": f"esp32.parallel_{n}", "to": f"led_bank.pin{n + 1}"}
        for n in range(3)
    ] + intent["signals"]
    architecture = derive_architecture(intent)
    mcu = _requirement(architecture, "esp32")
    assert mcu.parameters["parallel_output_count"] == 3
    assert mcu.interfaces == ["parallel_output"]


def test_connector_supply_exposes_the_rail_on_a_pin():
    """A connector does not draw from a rail, it exposes one — the pin is derived, not refused."""
    intent = _hub75_intent()
    intent["sheets"].append(
        {
            "name": "LED STRING",
            "stem": "LED_STRING",
            "role": "connector",
            "function": "Addressable LED string output connector.",
        }
    )
    intent["requirements"].append(
        {
            "id": "led_string",
            "sheet": "LED STRING",
            "role": "connector",
            "family": "pin-header",
            "parameters": {"rows": 1, "gender": "male"},
            "supply": "VBUS",
            "functional_blocks": ["ADDRESSABLE_LED_OUTPUT"],
        }
    )
    intent["signals"] = [
        row if row["name"] != "LED_DATA" else {**row, "to": ["led.data_in", "led_string.pin1"]}
        for row in intent["signals"]
    ]
    architecture = derive_architecture(intent)
    assert _requirement(architecture, "led_string").ports == {
        "pin1": "LED_DATA",
        "pin2": "GND",
        "pin3": "VBUS",
    }


def test_abbreviated_requirement_reference_resolves_when_unambiguous():
    intent = _hub75_intent()
    intent["signals"] = [
        {**row, "to": ["hub.addr_b"]} if row["name"] == "HUB75_ADDR_B" else row
        for row in intent["signals"]
    ]
    architecture = derive_architecture(intent)
    assert _requirement(architecture, "hub75").ports["addr_b"] == "HUB75_ADDR_B"

    ambiguous = _hub75_intent()
    ambiguous["sheets"].append(
        {
            "name": "LED STRING",
            "stem": "LED_STRING",
            "role": "connector",
            "function": "Addressable LED string output connector.",
        }
    )
    ambiguous["requirements"].append(
        {
            "id": "led_string",
            "sheet": "LED STRING",
            "role": "connector",
            # A genuinely model-owned family: this test is about the abbreviated
            # reference, and a known lowerer here would add its own (real)
            # contract diagnostic, so the assertion would name two causes at once.
            "family": "led-string-output",
            "functional_blocks": ["ADDRESSABLE_LED_OUTPUT"],
        }
    )
    ambiguous["signals"] = [
        {**row, "to": ["le.pin1"]} if row["name"] == "SPEAKER_PWM" else row
        for row in ambiguous["signals"]
    ]
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(ambiguous)
    assert [row.code for row in excinfo.value.diagnostics] == ["unknown_signal_requirement"]


def test_signal_restating_a_ground_connection_joins_ground():
    """Wiring ground explicitly is redundant, not a contradiction."""
    intent = _hub75_intent()
    intent["signals"] = [
        *intent["signals"],
        {"name": "MCU_GND", "from": "esp32.gnd", "to": ["hub75.gnd", "edge:SPEAKER"]},
    ]
    architecture = derive_architecture(intent)
    assert _requirement(architecture, "hub75").ports["gnd"] == "GND"
    assert not any(row.name == "MCU_GND" for row in architecture.inter_sheet_nets)


def test_optional_continuation_output_is_exposed_only_when_a_peer_exists():
    """A chain driver's continuation output is optional, not invented.

    The WS2812 recipe marks `data_out` optional because a chain's last pixel
    legitimately leaves DOUT unconnected (the round-ring reference relies on it).
    The architecture must therefore expose it only when the intent declares a
    peer — and then bind it to that peer's net, never to a made-up connector.
    """
    base = _hub75_intent()
    with_peer = derive_architecture(base)
    led = _requirement(with_peer, "led")
    assert led.ports["data_out"] == "LED_OUT"
    assert [(row.sheet, row.direction) for row in _net(with_peer, "LED_OUT").endpoints] == [
        ("LED", "output"),
        ("LED STRING", "input"),
    ]

    without_peer = _hub75_intent()
    without_peer["signals"] = [row for row in without_peer["signals"] if row["name"] != "LED_OUT"]
    architecture = derive_architecture(without_peer)
    assert set(_requirement(architecture, "led").ports) == {"vdd", "gnd", "data_in"}
    # No connector is invented for an output the intent never declared.
    assert not any(row.id == "led_data_out" for row in architecture.requirements)


def test_half_a_usb_pair_is_refused_by_name():
    intent = _hub75_intent()
    intent["signals"] = [row for row in intent["signals"] if row["name"] != "USB_D_N"]
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)
    # One missing line, two refusals: the socket reports the half pair, and the required-port
    # gate names the MCU pin nothing wires. Both point at the same `usb_dm`.
    assert [row.code for row in excinfo.value.diagnostics] == [
        "incomplete_usb_edge",
        "unbound_required_port",
    ]
    assert all("usb_dm" in row.message for row in excinfo.value.diagnostics)


def test_port_that_does_not_exist_is_refused_with_the_valid_ones():
    intent = _hub75_intent()
    intent["signals"] = [
        *intent["signals"],
        # HUB75's '245 interface has r0/r1, not a third colour channel.
        {"name": "HUB75_R2", "from": "hub75.r2", "to": "esp32.output_pwm1"},
    ]
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)
    diagnostic = excinfo.value.diagnostics[0]
    assert diagnostic.code == "unknown_interface_port"
    assert "requested=r2" in diagnostic.evidence
    assert "r0" in diagnostic.evidence[0]


def test_supply_must_name_a_declared_rail():
    intent = _hub75_intent()
    intent["requirements"] = [
        {**row, "supply": "+3V3"} if row["id"] == "hub75" else row for row in intent["requirements"]
    ]
    # `+3V3` exists, so the failure has to come from a rail nobody declared.
    intent["power"]["rails"].pop("+3V3")
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)
    codes = {row.code for row in excinfo.value.diagnostics}
    assert "unknown_supply_rail" in codes


def test_uncurated_part_without_a_declared_interface_is_refused_once():
    """A part nothing implements is a missing feature, not an unproven one.

    §4.3 A's table lists this code as RECORD; §4.1's boundary overrides it, and the reason is
    mechanical: with no curated recipe and no declared interface the requirement has no catalog,
    so it never enters the architecture — the design would ship with the part absent from the
    netlist (measured 2026-09-20: three production drafts whose only advisory was this code died
    on the ownership check one validation later instead). The refusal names the repair, once.
    """
    intent = _hub75_intent()
    intent["sheets"].append(
        {
            "name": "SENSOR",
            "stem": "SENSOR",
            "role": "sensor",
            "function": "Read temperature and humidity.",
        }
    )
    intent["requirements"].append(
        {
            "id": "bme",
            "sheet": "SENSOR",
            "role": "sensor",
            "family": "bme280",
            "exact_part": "BME280",
            "supply": "+3V3",
            "functional_blocks": ["SENSOR"],
        }
    )
    intent["requirements"] = [
        {**row, "interfaces": ["i2c_controller"]} if row["id"] == "esp32" else row
        for row in intent["requirements"]
    ]
    intent["signals"] += [
        {"name": "I2C_SDA", "from": "esp32.sda", "to": "bme.sda"},
        {"name": "I2C_SCL", "from": "esp32.scl", "to": "bme.scl"},
    ]
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)
    assert [row.code for row in excinfo.value.diagnostics] == ["unknown_part_refused"]
    assert "bme280" in excinfo.value.diagnostics[0].message

    for row in intent["requirements"]:
        if row["id"] == "bme":
            row["declared_ports"] = [
                {"key": "vdd", "direction": "power", "function": "3.3 V supply"},
                {"key": "gnd", "direction": "power", "function": "ground"},
                {"key": "sda", "direction": "bidirectional", "function": "I2C data"},
                {"key": "scl", "direction": "input", "function": "I2C clock"},
            ]
    architecture = derive_architecture(intent)
    assert architecture.declared_interfaces == ["bme"]
    assert [row.code for row in architecture.advisories] == []
    assert _requirement(architecture, "bme").ports == {
        "gnd": "GND",
        "vdd": "+3V3",
        "sda": "I2C_SDA",
        "scl": "I2C_SCL",
    }


def test_intent_slot_commits_on_the_first_draft_through_the_real_driver(tmp_path):
    """The plan's primary endpoint, at unit level: zero corrections, no provider spend.

    Drives the real `architecture` stage (contract, prompt, decode, normalize, commit) with the
    intent-shaped slot, and requires a first-attempt commit.
    """
    from test_stage_driver_retry import _OK_INTENT, _ScriptedClient

    from kicraft.server.config import Settings
    from kicraft.server.session import run_session

    def _reply(payload: dict) -> dict:
        return {
            "text": json.dumps(payload),
            "reasoning": "",
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    client = _ScriptedClient([_reply(json.loads(_OK_INTENT)), _reply(_hub75_intent())])
    client.s = Settings(api_key="test")
    result = run_session(
        tmp_path, "a USB-C ESP32-S3 HUB75 controller", ["intent", "architecture"], client=client
    )
    architecture = next(row for row in result["results"] if row["stage"] == "architecture")
    assert architecture["commit_ok"] is True, architecture
    assert architecture["attempts"] == 1, architecture
    # The provider is asked for the intent-shaped architecture contract.
    assert (
        client.calls[-1]["response_format"]["json_schema"]["name"]
        == "kicraft_architecture_response_v2"
    )
    system_prompt = client.calls[-1]["messages"][0]["content"]
    assert "compiler writes" in system_prompt


def test_rejected_first_draft_publishes_its_defect_class(tmp_path):
    """The plan's measurement instrument: the events stream names the classes it rejected."""
    from test_stage_driver_retry import _OK_INTENT, _ScriptedClient

    from kicraft.server.config import Settings
    from kicraft.server.session import run_session

    def _reply(payload: dict) -> dict:
        return {
            "text": json.dumps(payload),
            "reasoning": "",
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    half = _half_usb_pair_intent()
    events: list[dict] = []
    client = _ScriptedClient(
        [_reply(json.loads(_OK_INTENT)), _reply(half), _reply(_hub75_intent())]
    )
    client.s = Settings(api_key="test")
    run_session(
        tmp_path,
        "a USB-C ESP32-S3 HUB75 controller",
        ["intent", "architecture"],
        client=client,
        progress=events.append,
    )
    done = next(
        event
        for event in events
        if event.get("kind") == "stage_done" and event.get("stage") == "architecture"
    )
    assert done["drafts"] == 2
    assert done["first_draft_accepted"] is False
    assert done["defect_codes"] == ["incomplete_usb_edge"]
    assert done["unknown_part_refused"] == 0
    assert done["declared_interfaces"] == 0
    # The reader refused the first draft: one contract rejection, no semantic
    # repair, and the first draft never reached the commit gates.
    assert done["contract_rejections"] == 1
    assert done["first_draft_contract_clean"] is False
    assert done["semantic_repair_rounds"] == 0


def test_semantic_repair_round_is_counted_apart_from_contract_rejections(tmp_path):
    """A reader-clean first draft repaired for a design statement (next-steps plan §3).

    The endpoint split exists for this run: the first draft was accepted by the
    contract, so it is `first_draft_contract_clean`, but a repair round followed
    and `first_draft_accepted` (zero corrections) stays false.
    """
    from test_stage_driver_retry import _ScriptedClient

    from kicraft.server.config import Settings
    from kicraft.server.session import run_session

    def _reply(payload: dict) -> dict:
        return {
            "text": json.dumps(payload),
            "reasoning": "",
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    intent = {
        "goal": "a USB-C ESP32-S3 HUB75 controller",
        "constraints": [],
        "named_parts": [],
        "inferred_expertise": "intermediate",
        "assumptions": [],
        "project_stem": "USB_HUB75",
    }
    # 0.25 A cannot be the ESP32-S3's 3.3V source; the reviewed 2 A buck can.
    weak = _hub75_intent()
    for row in weak["requirements"]:
        if row["id"] == "buck":
            row["family"] = "mcp1700-3v3"
    events: list[dict] = []
    client = _ScriptedClient([_reply(intent), _reply(weak), _reply(_hub75_intent())])
    client.s = Settings(api_key="test")
    result = run_session(
        tmp_path,
        "a USB-C ESP32-S3 HUB75 controller",
        ["intent", "architecture"],
        client=client,
        progress=events.append,
        # The non-interactive drive keeps today's repair instead of asking the
        # user, which is the path this instrument has to measure.
        instruction=stage_driver_mod.NONINTERACTIVE_DEFAULTS_INSTRUCTION,
    )
    done = next(
        event
        for event in events
        if event.get("kind") == "stage_done" and event.get("stage") == "architecture"
    )
    assert result["status"] == "ok"
    assert done["drafts"] == 2
    assert done["contract_rejections"] == 0
    assert done["first_draft_contract_clean"] is True
    assert done["first_draft_accepted"] is False
    assert done["semantic_repair_rounds"] == 1
    assert "architecture_mcu_regulator_incomplete" in done["defect_codes"]


def test_missing_external_load_current_parks_with_one_question(tmp_path):
    """The load current is the user's fact: ask once, never repair or invent (§4 B2)."""
    from test_stage_driver_retry import _ScriptedClient

    from kicraft.server.config import Settings
    from kicraft.server.stage_runtime import drive_stage

    state = _frozen_prompt_state()
    workspace = tmp_path / "ws"
    state_path = workspace / ".kicraft" / "state.json"
    state_path.parent.mkdir(parents=True)
    state_path.write_text(json.dumps(state), encoding="utf-8")

    def _reply(payload: dict) -> dict:
        return {
            "text": json.dumps(payload),
            "reasoning": "",
            "finish_reason": "stop",
            "cost_usd": 0.0,
        }

    def _drive(
        brief: str, answers: list[dict] | None = None
    ) -> tuple[dict, list[dict], _ScriptedClient]:
        events: list[dict] = []
        client = _ScriptedClient(
            [_reply(_hub75_intent()), _reply(_hub75_intent()), _reply(_hub75_intent())]
        )
        client.s = Settings(api_key="test")
        result = drive_stage(
            client,
            "architecture",
            brief,
            state_path,
            workspace,
            progress=events.append,
            answers=answers,
        )
        return result, events, client

    # The brief states no current: park with exactly one question, one provider
    # call, no repair round, and no commit.
    parked, events, client = _drive("a USB-C ESP32-S3 HUB75 controller and LED string")
    assert parked["needs_input"] is True
    assert parked["commit_ok"] is False
    assert len(client.calls) == 1
    assert len(parked["questions"]) == 1
    assert parked["questions"][0]["blocking"] is True
    assert not [event for event in events if event.get("kind") == "retry"]
    assert [event["kind"] for event in events if event.get("kind") == "question"] == ["question"]
    # Durable: a reopened project shows the question.
    open_questions = json.loads(state_path.read_text(encoding="utf-8"))["open_questions"]
    assert [row["text"] for row in open_questions] == [parked["questions"][0]["text"]]

    # The brief already carries the number: the model is told to state it
    # (today's repair), and nobody asks the user. The finding itself is
    # diagnosed either way — the brief decides only whether the user is asked.
    repaired, events, client = _drive(
        "a USB-C ESP32-S3 HUB75 controller and LED string drawing 2 A at 5 V"
    )
    assert not repaired.get("needs_input")
    assert not [event for event in events if event.get("kind") == "question"]
    assert [
        event["code"]
        for event in events
        if event.get("kind") == "stage_diagnostic"
        and event.get("code") == "architecture_external_load_current_unspecified"
    ]
    assert len(client.calls) >= 2  # a repair call, not the user's answer

    # The user answered the parked question: the same board is never asked twice
    # for a number it now holds.
    resumed, events, client = _drive(
        "a USB-C ESP32-S3 HUB75 controller and LED string",
        answers=[
            {
                "text": parked["questions"][0]["text"],
                "answer": "Up to 2 A",
            }
        ],
    )
    assert not resumed.get("needs_input")
    assert not [event for event in events if event.get("kind") == "question"]


def test_intent_slot_rejects_unknown_fields_and_partial_ranges():
    with pytest.raises(ValueError):
        ArchitectureIntent.model_validate({**_hub75_intent(), "power_nets": ["VBUS"]})
    intent = _hub75_intent()
    intent["signals"] = [
        *intent["signals"],
        {"name": "X{n}", "from": "a.b{n}", "to": "c.d", "start": 0},
    ]
    with pytest.raises(ValueError):
        ArchitectureIntent.model_validate(intent)


def test_declared_interface_supply_binds_the_pin_it_names():
    """A declared interface that names its supply pin `vcc` (not `vdd`) still takes the rail.

    The canary (2026-09-15, `rs485-terminal`, `thermocouple-amp`, `lora-node`) refused four
    briefs whose interfaces name `vcc`; the rail lookup only knew `vdd`/`vm`/`vin`/`input`.
    """
    intent = _hub75_intent()
    intent["requirements"] = [
        (
            {
                **row,
                "declared_ports": [
                    {"key": "vcc", "direction": "power", "function": "5 V input"},
                    {"key": "gnd", "direction": "power", "function": "ground"},
                ],
            }
            if row["id"] == "usb_power"
            else row
        )
        for row in intent["requirements"]
    ]
    intent["requirements"].append(
        {
            "id": "xcvr",
            "sheet": "MCU",
            "role": "bus_interface",
            "family": "transceiver",
            "supply": "VBUS",
            "declared_ports": [
                {"key": "vcc", "direction": "power", "function": "logic supply"},
                {"key": "gnd", "direction": "power", "function": "ground"},
                {"key": "a", "direction": "bidirectional", "function": "bus A"},
            ],
            "functional_blocks": ["ESP32_S3_CONTROLLER"],
        }
    )
    architecture = derive_architecture(intent)
    assert _requirement(architecture, "xcvr").ports == {"vcc": "VBUS", "gnd": "GND"}


def test_lowerer_family_supply_uses_the_keys_the_family_publishes():
    """A lowerer family publishes its own port keys; a rail binds to the supply one.

    The canary (2026-09-15, `esp32-s3-sensor` and eight more) refused requirements whose family
    is a lowerer, because the derivation treated every lowerer as an empty open vocabulary.
    """
    intent = _hub75_intent()
    intent["requirements"].append(
        {
            "id": "servo_power",
            "sheet": "HUB75",
            "role": "connector",
            "family": "connector-bank",
            "parameters": {"channels": 1},
            "supply": "VBUS",
            "functional_blocks": ["HUB75_DISPLAY_INTERFACE"],
        }
    )
    intent["signals"].append(
        {"name": "SERVO", "from": "esp32.output_servo", "to": "servo_power.signal0"}
    )
    architecture = derive_architecture(intent)
    ports = _requirement(architecture, "servo_power").ports
    assert ports["vdd"] == "VBUS"
    assert ports["gnd"] == "GND"


def test_qualified_supply_pin_is_picked_by_the_rail_name():
    """Two qualified supply pins on one interface: the rail's own name decides.

    The canary (2026-09-15, `rs485-terminal`) refused an isolator declaring `+5V_LOGIC` whose
    interface names `vdd_logic` and `vdd_field` — neither is a bare `vdd`.
    """
    intent = _hub75_intent()
    intent["requirements"].append(
        {
            "id": "logic_reg",
            "sheet": "POWER",
            "role": "regulator",
            "family": "tlv62569-3v3",
            "supply": "VBUS",
            "functional_blocks": ["POWER_DISTRIBUTION"],
        }
    )
    intent["power"]["rails"]["+5V_LOGIC"] = {"voltage": 3.3, "from": "logic_reg.output"}
    intent["requirements"].append(
        {
            "id": "isolator",
            "sheet": "MCU",
            "role": "bus_interface",
            "family": "isolator",
            "supply": "+5V_LOGIC",
            "declared_ports": [
                {"key": "vdd_logic", "direction": "power", "function": "logic-side supply"},
                {"key": "vdd_field", "direction": "power", "function": "field-side supply"},
                {"key": "logic_tx", "direction": "output", "function": "logic-side transmit"},
            ],
            "functional_blocks": ["ESP32_S3_CONTROLLER"],
        }
    )
    architecture = derive_architecture(intent)
    ports = _requirement(architecture, "isolator").ports
    assert ports["vdd_logic"] == "+5V_LOGIC"
    assert "vdd_field" not in ports


def test_declared_interface_persists_explicit_isolated_supply_and_reference_domains():
    intent = _hub75_intent()
    intent["power"]["rails"].update(
        {
            "GND_LOGIC": {"voltage": 0.0},
            "GND_FIELD": {"voltage": 0.0},
        }
    )
    intent["requirements"].append(
        {
            "id": "max31855",
            "sheet": "MCU",
            "role": "sensor",
            "family": "max31855",
            "exact_part": "MAX31855KASA+",
            "supply_bindings": {"vdd": "+3V3"},
            "reference_bindings": {"gnd_logic": "GND_LOGIC", "gnd_field": "GND_FIELD"},
            "declared_ports": [
                {
                    "key": "vdd",
                    "pin": "4",
                    "direction": "power",
                    "function": "logic supply",
                    "supply_rail": "+3V3",
                },
                {
                    "key": "gnd_logic",
                    "pin": "2",
                    "direction": "power",
                    "function": "logic reference",
                    "reference_domain": "GND_LOGIC",
                },
                {
                    "key": "gnd_field",
                    "pin": "1",
                    "direction": "power",
                    "function": "thermocouple reference",
                    "reference_domain": "GND_FIELD",
                },
                {"key": "sck", "pin": "5", "direction": "input", "function": "SPI clock"},
            ],
            "functional_blocks": ["ESP32_S3_CONTROLLER"],
        }
    )

    requirement = _requirement(derive_architecture(intent), "max31855")

    assert requirement.ports == {
        "vdd": "+3V3",
        "gnd_logic": "GND_LOGIC",
        "gnd_field": "GND_FIELD",
    }
    assert requirement.declared_interface is not None
    assert {port.key: port.pin for port in requirement.declared_interface.ports}["sck"] == "5"


def test_declared_interface_refuses_conflicting_per_port_supply_domains():
    intent = _hub75_intent()
    intent["requirements"].append(
        {
            "id": "isolator",
            "sheet": "MCU",
            "role": "bus_interface",
            "family": "isolator",
            "supply_bindings": {"vdd_logic": "+3V3"},
            "declared_ports": [
                {
                    "key": "vdd_logic",
                    "pin": "1",
                    "direction": "power",
                    "function": "logic supply",
                    "supply_rail": "VBUS",
                }
            ],
            "functional_blocks": ["ESP32_S3_CONTROLLER"],
        }
    )

    with pytest.raises(ArchitectureIntentError) as rejected:
        derive_architecture(intent)

    assert "conflicting_supply_binding" in {row.code for row in rejected.value.diagnostics}


def test_valid_status_led_uses_only_its_published_drive_and_reference_ports():
    intent = _hub75_intent()
    intent["requirements"].append(
        {
            "id": "status",
            "sheet": "MCU",
            "role": "driver",
            "family": "status-led",
            "parameters": {"rail_voltage": 3.3, "led_vf": 2.0, "target_current_ma": 2},
            "functional_blocks": ["ESP32_S3_CONTROLLER"],
        }
    )
    intent["signals"].append(
        {"name": "STATUS", "from": "esp32.output_status", "to": "status.drive"}
    )

    requirement = _requirement(derive_architecture(intent), "status")

    assert requirement.ports == {"drive": "STATUS", "gnd": "GND"}


def test_typed_original_obligation_must_be_owned_and_persists_on_requirement():
    intent = _hub75_intent()
    obligation = {
        "kind": "physical",
        "original_obligation_id": "status_led",
        "component_class": "status-led",
    }
    intent["obligations"] = [obligation]
    next(row for row in intent["requirements"] if row["id"] == "led")["obligations"] = [obligation]

    architecture = derive_architecture(intent)
    requirement = _requirement(architecture, "led")

    assert architecture.obligations[0].model_dump() == obligation
    assert requirement.obligations[0].model_dump() == obligation


def test_quantitative_obligations_reject_nonfinite_values():
    from kicraft.design.models import QuantitativeObligation

    with pytest.raises(ValueError):
        QuantitativeObligation(
            kind="quantitative",
            original_obligation_id="supply_current",
            quantity="output current",
            relation="minimum",
            value=float("nan"),
            unit="A",
        )


def test_approved_uno_template_requires_and_constructs_each_explicit_stacking_owner():
    from kicraft.form_factors import get_template

    template = get_template("arduino_uno_shield")
    assert template is not None and template.validated
    intent = _hub75_intent()
    intent["standard_form_factor"] = template.key
    intent["sheets"].append(
        {
            "name": "UNO HEADERS",
            "stem": "UNO_HEADERS",
            "role": "connector",
            "function": "Arduino Uno shield stacking interface.",
        }
    )
    for connector in template.fixed_connectors:
        intent["requirements"].append(
            {
                "id": f"uno_{connector.role}",
                "sheet": "UNO HEADERS",
                "role": "connector",
                "family": "pin-header",
                "parameters": {"rows": 1, "gender": "female"},
                "standard_stacking_role": connector.role,
                "ties": {
                    f"pin{index}": net for index, net in enumerate(connector.net_by_pin, start=1)
                },
                "functional_blocks": ["UNO_HOST_INTERFACE"],
            }
        )

    architecture = derive_architecture(intent)

    owned = {
        requirement.standard_stacking_role: requirement
        for requirement in architecture.requirements
        if requirement.standard_stacking_role
    }
    assert set(owned) == {connector.role for connector in template.fixed_connectors}
    assert owned["power"].ports["pin6"] == "GND"
    assert owned["power"].ports["pin1"] == "NC"


def test_unused_template_pin_names_become_no_connects_not_nets():
    """A template pin name is the HOST's label, not a net of this board.

    The four headers carry the template's full 32-pin map (SCL, SDA, AREF, D13..D0,
    A0..A5, IOREF, RESET), while a shield that never uses those signals realizes only the
    rails it draws from. Binding the rest would put a host label on a net with a single
    pin -- exactly the dangling label §9.15 refuses, and what stopped the proto-shield
    brief from wiring ("deterministic wiring binds a singleton signal without a declared
    architecture endpoint"). The pin stays on the connector (its physical size comes from
    the port count) and becomes a no-connect.
    """
    from kicraft.form_factors import get_template

    template = get_template("arduino_uno_shield")
    intent = _hub75_intent()
    intent["standard_form_factor"] = template.key
    intent["sheets"].append(
        {
            "name": "UNO HEADERS",
            "stem": "UNO_HEADERS",
            "role": "connector",
            "function": "Arduino Uno shield stacking interface.",
        }
    )
    for connector in template.fixed_connectors:
        intent["requirements"].append(
            {
                "id": f"uno_{connector.role}",
                "sheet": "UNO HEADERS",
                "role": "connector",
                "family": "pin-header",
                "parameters": {"rows": 1, "gender": "female"},
                "standard_stacking_role": connector.role,
                "functional_blocks": ["UNO_HOST_INTERFACE"],
            }
        )

    architecture = derive_architecture(intent)
    owned = {
        requirement.standard_stacking_role: requirement
        for requirement in architecture.requirements
        if requirement.standard_stacking_role
    }
    # Host signal labels nobody on this board drives: no-connect, never a one-pin net.
    for role, pins in (
        ("digital_high", ("pin1", "pin2", "pin3", "pin5")),
        ("digital_low", ("pin1", "pin2")),
        ("analog", ("pin1", "pin6")),
    ):
        for pin in pins:
            assert owned[role].ports[pin] == "NC", (role, pin, owned[role].ports[pin])
    # The rails the shield draws from stay bound, and ground stays ground.
    assert owned["digital_high"].ports["pin4"] == "GND"
    assert owned["power"].ports["pin4"] == "+3V3"
    assert owned["power"].ports["pin5"] == "+5V"
    assert owned["power"].ports["pin1"] == "NC"
    # The connector is still the template's own size (the ports carry every pin).
    assert len(owned["digital_high"].ports) == 10
    assert len(owned["analog"].ports) == 6


def _declared_port_misuse_intent(
    *, vdd_both: bool, sig_reference: bool, spare_both: bool = False
) -> dict:
    """The reference intent plus one uncurated part carrying declared ports."""
    intent = _hub75_intent()
    intent["sheets"].append(
        {
            "name": "SENSOR",
            "stem": "SENSOR",
            "role": "analog_block",
            "function": "Condition a sensor signal.",
        }
    )
    vdd = {"key": "vdd", "direction": "power", "function": "3.3 V supply", "supply_rail": "+3V3"}
    if vdd_both:
        vdd["reference_domain"] = "GND"
    sig = {"key": "sig", "direction": "output", "function": "conditioned output"}
    if sig_reference:
        sig["reference_domain"] = "GND"
    # A pin the design neither powers nor wires: a stated rail *and* a stated reference are
    # two claims about one net, with no signal to arbitrate them.
    spare = {"key": "spare", "direction": "input", "function": "unused input"}
    if spare_both:
        spare["supply_rail"] = "+3V3"
        spare["reference_domain"] = "GND"
    intent["requirements"].append(
        {
            "id": "sensor_af",
            "sheet": "SENSOR",
            "role": "analog_block",
            "family": "uncurated-sensor-frontend",
            "declared_ports": [
                vdd,
                {
                    "key": "gnd",
                    "direction": "power",
                    "function": "ground",
                    "reference_domain": "GND",
                },
                {
                    "key": "csb",
                    "direction": "input",
                    "function": "chip select, tied high",
                    "supply_rail": "+3V3",
                },
                sig,
                *([spare] if spare_both else []),
            ],
        }
    )
    intent["signals"].append({"name": "SENSOR_SIG", "from": "sensor_af.sig", "to": "edge:SENSOR"})
    return intent


def test_declared_port_domain_statement_on_a_signal_port_is_derived():
    """`reference_domain: GND` on a signal port is the signal's domain, not the pin's tie.

    A model commonly writes reference_domain='GND' (and supply_rail=<its own supply>) on
    every declared port meaning "ground-referenced" / "belongs to the 3V3 domain". The field
    name cannot tell that apart from the net the pin is tied to, and the design's own signals
    already state the pin's net, so the compiler takes the signal and records the statement
    instead of failing the run for it (moves 1a-1b of the yield-recovery plan).
    """
    intent = _declared_port_misuse_intent(vdd_both=True, sig_reference=True)
    architecture = derive_architecture(intent)
    requirement = next(row for row in architecture.requirements if row.id == "sensor_af")
    assert requirement.ports["sig"] == "SENSOR_SIG"
    notes = "\n".join(architecture.assumptions)
    assert "the signal owns the pin" in notes
    assert "the supply input owns the pin" in notes


def test_declared_port_tie_misuse_is_still_refused_when_unplaceable():
    """What the compiler cannot place stays a refusal, named where it is written.

    A pin the design neither powers nor wires, carrying a rail *and* a reference, is two
    claims about one net with no signal to arbitrate: that stays `declared_port_double_bound`.
    """
    intent = _declared_port_misuse_intent(vdd_both=False, sig_reference=False, spare_both=True)
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)
    codes = {row.code for row in excinfo.value.diagnostics}
    assert "declared_port_double_bound" in codes


def test_declared_port_unrelated_domain_on_a_signal_port_is_still_refused():
    """A net that is not the requirement's own supply or ground is a different statement.

    The derive covers the domain statement about the rail that feeds the requirement. A signal
    port pinned to some other domain is not that, so `declared_signal_port_tied` still names it.
    """
    intent = _declared_port_misuse_intent(vdd_both=False, sig_reference=False)
    intent["power"]["rails"]["GND_LOGIC"] = {"voltage": 0.0}
    sensor = next(row for row in intent["requirements"] if row["id"] == "sensor_af")
    sig = next(port for port in sensor["declared_ports"] if port["key"] == "sig")
    sig["reference_domain"] = "GND_LOGIC"
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)
    codes = {row.code for row in excinfo.value.diagnostics}
    assert "declared_signal_port_tied" in codes


def test_declared_port_tie_on_supply_ground_and_strap_pins_is_legal():
    """The field's real meaning: the supply pin, the ground pin, and a strapped pin.

    `vdd` carries its rail, `gnd` is tied to its reference, and `csb` is a signal pin
    held at the rail — none of these is the misuse, so the guard must not fire.
    """
    intent = _declared_port_misuse_intent(vdd_both=False, sig_reference=False)
    try:
        derive_architecture(intent)
    except ArchitectureIntentError as exc:
        codes = {row.code for row in exc.value.diagnostics}
        assert "declared_port_double_bound" not in codes
        assert "declared_signal_port_tied" not in codes


def test_obligation_ownership_refusal_names_the_fix():
    """An obligation listed only at the top level must be refused with an actionable message.

    The draft has to be repairable from the error alone: name the obligation and the
    invariant (the top-level `obligations` list is the union of the requirements' own
    rows), not just "ownership mismatch", which the model cannot act on.
    """
    from pydantic import ValidationError

    intent = _hub75_intent()
    intent["obligations"] = [
        {
            "kind": "physical",
            "original_obligation_id": "single-sensor-input",
            "component_class": "single-sensor-input",
        }
    ]
    with pytest.raises(ValidationError) as excinfo:
        derive_architecture(intent)
    message = str(excinfo.value)
    assert "single-sensor-input" in message
    assert "listed_at_top_level_only" in message
    assert "union of the requirements" in message


def test_two_rails_from_one_source_port_merge_into_one_node():
    """`VBUS` and `+5V` declared from one port are one rail, not a conflict.

    The beacon draft (frozen replay `curR-b3`) declared both from `usb.vbus`. A port carries one
    net, so the second rail's bind refused `conflicting_port_binding` and the model re-emitted the
    same draft on every attempt. One node declared twice is normalized instead.
    """
    intent = _hub75_intent()
    intent["power"]["rails"]["+5V"] = {"voltage": 5.0, "from": "usb_power.vbus"}
    next(row for row in intent["requirements"] if row["id"] == "hub75")["supply"] = "+5V"
    next(row for row in intent["requirements"] if row["id"] == "led")["supply_bindings"] = {
        "vdd": "+5V"
    }

    architecture = derive_architecture(intent)

    assert architecture.power_nets == ["GND", "+3V3", "VBUS"]
    assert _requirement(architecture, "hub75").ports["vdd_5v"] == "VBUS"
    assert _requirement(architecture, "led").ports["vdd"] == "VBUS"
    assert [row.message for row in architecture.advisories if row.code == "rail_alias_merged"] == [
        "rail '+5V' is the same node as 'VBUS' (both declared from 'usb_power.vbus'); "
        "the design carries one rail"
    ]


def test_declared_supply_rail_named_by_the_alias_is_rewritten():
    """An alias reachable through a declared port's `supply_rail` is rewritten too.

    `supply_rail` on a declared port feeds the same supply-binding table as `supply_bindings`;
    leaving the alias there would swap the merge for an `unknown_supply_rail` refusal.
    """
    intent = _declared_port_misuse_intent(vdd_both=False, sig_reference=False)
    intent["power"]["rails"]["+5V"] = {"voltage": 5.0, "from": "usb_power.vbus"}
    csb = next(
        port
        for port in intent["requirements"][-1]["declared_ports"]
        if port["key"] == "csb"
    )
    csb["supply_rail"] = "+5V"

    architecture = derive_architecture(intent)

    assert next(row for row in architecture.requirements if row.id == "sensor_af").ports[
        "csb"
    ] == "VBUS"


def test_two_rails_from_one_port_at_different_voltages_still_refuse():
    """Voltage is the merge's boundary: 5 V and 3.3 V off one port are two different nodes."""
    intent = _hub75_intent()
    intent["power"]["rails"]["+5V"] = {"voltage": 3.3, "from": "usb_power.vbus"}

    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)

    row = next(d for d in excinfo.value.diagnostics if d.code == "conflicting_port_binding")
    assert "rail '+5V'" in row.message


def test_a_signal_named_after_the_alias_keeps_the_rail_declaration():
    """A rail a signal carries the name of is never collapsed: that conflict is its own refusal."""
    intent = _hub75_intent()
    intent["power"]["rails"]["+5V"] = {"voltage": 5.0, "from": "usb_power.vbus"}
    intent["signals"] = [
        {**row, "name": "+5V"} if row["name"] == "USB_D_P" else row for row in intent["signals"]
    ]

    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)

    codes = {row.code for row in excinfo.value.diagnostics}
    assert "conflicting_port_binding" in codes  # the duplicate rail bind is untouched
    assert "signal_names_rail" in codes


def test_one_port_carries_one_net_and_the_refusal_names_the_menu():
    """Two signals on one port must name the alternatives (the top live failure).

    `switch-input` has a single `signal` port, so three microstep switches are three
    requirements — not one port bound three times. The refusal must show the menu.
    """
    intent = _hub75_intent()
    intent["sheets"].append(
        {
            "name": "SW",
            "stem": "SW",
            "role": "user_io",
            "function": "Microstep select switches.",
        }
    )
    intent["requirements"].append(
        {
            "id": "microstep",
            "sheet": "SW",
            "role": "user_io",
            "family": "switch-input",
            "parameters": {"pull_policy": "internal"},
            "functional_blocks": ["MICROSTEP_SELECT"],
        }
    )
    intent["signals"] = [
        *intent["signals"],
        {"name": "MS1", "from": "esp32.gpio4", "to": "microstep.signal"},
        {"name": "MS2", "from": "esp32.gpio5", "to": "microstep.signal"},
    ]
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)
    row = next(d for d in excinfo.value.diagnostics if d.code == "conflicting_port_binding")
    assert "one port carries one net" in row.message
    assert "gnd,signal,vdd" in row.message  # the requirement's actual menu


def test_pattern_lowerer_port_refusal_prints_its_contract_not_none():
    """A pattern lowerer publishes words, not keys; the menu must not read "(none)".

    Every port refusal embeds the requirement's port menu. For a family whose ports
    are a pattern (screw-terminal's pinN/pN), an empty menu teaches the draft nothing.
    """
    intent = _hub75_intent()
    intent["sheets"].append(
        {
            "name": "TERM",
            "stem": "TERM",
            "role": "connector",
            "function": "Sensor terminal.",
        }
    )
    intent["requirements"].append(
        {
            "id": "sensor_term",
            "sheet": "TERM",
            "role": "connector",
            "family": "screw-terminal",
            "functional_blocks": ["SENSOR_INPUT"],
        }
    )
    intent["signals"] = [
        *intent["signals"],
        {"name": "SENSOR_ADC", "from": "esp32.gpio6", "to": "sensor_term.nope"},
    ]
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)
    rows = [d for d in excinfo.value.diagnostics if d.code == "unknown_interface_port"]
    assert rows, [d.code for d in excinfo.value.diagnostics]
    assert "(none)" not in rows[0].evidence[0]
    assert "pin" in rows[0].evidence[0].lower()


def test_lowerer_supply_without_a_published_port_is_derived_not_refused():
    """A status LED draws its current from its own `drive` signal, not from a rail pin.

    The canary (2026-09-17, `esp32-s3-sensor`, `chamfered-badge`, `star-ornament`) refused
    twelve drafts with `unsupported_supply_port` because the lowerer publishes no supply
    contact. The rail is the intent and it lands on drive/gnd; nothing else is invented, and
    the derivation says so in the assumptions.
    """
    intent = _hub75_intent()
    intent["requirements"].append(
        {
            "id": "status",
            "sheet": "MCU",
            "role": "driver",
            "family": "status-led",
            "parameters": {"rail_voltage": 3.3, "led_vf": 2.0, "target_current_ma": 2},
            "supply": "+3V3",
            "functional_blocks": ["ESP32_S3_CONTROLLER"],
        }
    )
    intent["signals"].append(
        {"name": "STATUS", "from": "esp32.output_status", "to": "status.drive"}
    )

    architecture = derive_architecture(intent)

    assert _requirement(architecture, "status").ports == {"drive": "STATUS", "gnd": "GND"}
    assert any(
        "status" in note and "publishes no supply port" in note for note in architecture.assumptions
    )


def test_published_lowerer_supply_contact_takes_the_declared_rail():
    """A lowerer that publishes `vdd` gets its rail there without the model binding it.

    The port set a lowerer's own build code needs is the compiler's to complete: the rail a
    requirement declares lands on the published supply contact, so `unbound_required_port`
    and the contract check read a document the model never wrote.
    """
    intent = _hub75_intent()
    intent["requirements"].append(
        {
            "id": "pullups",
            "sheet": "MCU",
            "role": "bus_interface",
            "family": "i2c-pullups",
            "parameters": {"speed_hz": 400000, "bus_capacitance_pf": 50, "voltage": 3.3},
            "supply": "+3V3",
            "functional_blocks": ["ESP32_S3_CONTROLLER"],
        }
    )
    intent["signals"].extend(
        [
            {"name": "SDA", "from": "esp32.sda", "to": "pullups.sda"},
            {"name": "SCL", "from": "esp32.scl", "to": "pullups.scl"},
        ]
    )

    ports = _requirement(derive_architecture(intent), "pullups").ports

    assert ports["vdd"] == "+3V3"
    assert ports["sda"] == "SDA" and ports["scl"] == "SCL"


def test_unpublished_authored_supply_port_falls_back_to_the_family_port():
    """A rail named on a port the family does not publish still reaches the part.

    `supply_bindings` is an optional refinement: when the named pin is not one the family
    publishes, the rail is bound to the family's own supply port instead of refusing the
    draft for a name the compiler can derive.
    """
    intent = _hub75_intent()
    intent["requirements"].append(
        {
            "id": "pullups",
            "sheet": "MCU",
            "role": "bus_interface",
            "family": "i2c-pullups",
            "supply_bindings": {"vdd_logic": "+3V3"},
            "parameters": {"speed_hz": 400000, "bus_capacitance_pf": 50, "voltage": 3.3},
            "functional_blocks": ["ESP32_S3_CONTROLLER"],
        }
    )
    intent["signals"].extend(
        [
            {"name": "SDA", "from": "esp32.sda", "to": "pullups.sda"},
            {"name": "SCL", "from": "esp32.scl", "to": "pullups.scl"},
        ]
    )

    ports = _requirement(derive_architecture(intent), "pullups").ports

    assert ports["vdd"] == "+3V3"
    assert "vdd_logic" not in ports


def test_two_signals_out_of_one_source_port_join_the_first_net():
    """One physical pin carries one net, under the first name the design gave it.

    The canary (2026-09-17, `stm32-min` NRST, `stepper-a4988` DIR, `speaker-crossover`
    WOOFER_OUT) refused drafts whose second signal left a port the first signal already
    bound. That is the same pin, so the peers join the existing net and the join is
    reported rather than refused.
    """
    intent = _hub75_intent()
    intent["sheets"].append(
        {
            "name": "SWD",
            "stem": "SWD",
            "role": "programming",
            "function": "Programming header.",
        }
    )
    intent["requirements"].append(
        {
            "id": "swd",
            "sheet": "SWD",
            "role": "programming",
            "family": "pin-header",
            "parameters": {"rows": 1, "gender": "male"},
            "functional_blocks": ["ESP32_S3_CONTROLLER"],
        }
    )
    intent["signals"].extend(
        [
            {"name": "SWD_CLK", "from": "esp32.gpio7", "to": "swd.pin1"},
            {"name": "SWD_DIO", "from": "esp32.gpio7", "to": "swd.pin2"},
        ]
    )

    architecture = derive_architecture(intent)

    assert _requirement(architecture, "swd").ports == {"pin1": "SWD_CLK", "pin2": "SWD_CLK"}
    assert any("join net SWD_CLK" in note for note in architecture.assumptions)


def test_standard_stacking_pinmap_is_derived_from_the_template():
    """The template owns the pin/net map; the role is the design statement.

    The canary (2026-09-17, `proto-shield`, `snowman-ornament`) refused three stacking
    connectors for a map the approved template already fixes, then reported every net in
    that map a second time. The map is derived; a *different* authored map is still refused.
    """
    from kicraft.form_factors import get_template

    template = get_template("arduino_uno_shield")
    intent = _hub75_intent()
    intent["standard_form_factor"] = template.key
    intent["sheets"].append(
        {
            "name": "UNO HEADERS",
            "stem": "UNO_HEADERS",
            "role": "connector",
            "function": "Arduino Uno shield stacking interface.",
        }
    )
    for connector in template.fixed_connectors:
        intent["requirements"].append(
            {
                "id": f"uno_{connector.role}",
                "sheet": "UNO HEADERS",
                "role": "connector",
                "family": "pin-header",
                "parameters": {"rows": 1, "gender": "female"},
                "standard_stacking_role": connector.role,
                "functional_blocks": ["UNO_HOST_INTERFACE"],
            }
        )

    architecture = derive_architecture(intent)
    owned = {
        requirement.standard_stacking_role: requirement
        for requirement in architecture.requirements
        if requirement.standard_stacking_role
    }
    assert owned["power"].ports["pin6"] == "GND"
    assert owned["power"].ports["pin1"] == "NC"

    wrong = _hub75_intent()
    wrong["standard_form_factor"] = template.key
    wrong["sheets"] = [*wrong["sheets"], *intent["sheets"][-1:]]
    wrong["requirements"] = [
        *wrong["requirements"],
        *[
            {**row, "ties": {"pin1": "GND"}}
            for row in intent["requirements"]
            if row.get("standard_stacking_role") == "power"
        ],
    ]
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(wrong)
    assert "invalid_standard_stacking_pinmap" in {d.code for d in excinfo.value.diagnostics}


def test_architecture_top_level_obligations_are_written_from_the_committed_set():
    """The top-level list is the committed set; the draft states ownership only.

    The canary (2026-09-17, `r2r-dac`, `round-led-ring`, `rounded-c3-devboard`,
    `snowman-ornament`) refused six drafts per run for an obligation list the compiler can
    write. The draft's job is to attach each committed row, once and verbatim, to the
    requirement that implements it.
    """
    obligation = {
        "kind": "physical",
        "original_obligation_id": "status-led",
        "component_class": "status-led",
    }
    prompt_state = {
        "intent": {"goal": "reference board", "obligations": [obligation]},
        "functional_spec": {"obligations": [obligation]},
    }
    intent = _hub75_intent()
    next(row for row in intent["requirements"] if row["id"] == "led")["obligations"] = [
        {**obligation, "component_class": "led"}
    ]

    payload, _expanded = _normalize_stage_response("architecture", intent, prompt_state)

    assert payload["obligations"] == [obligation]
    restored = next(row for row in payload["requirements"] if row["id"] == "led")
    assert restored["obligations"] == [obligation]
    # The parsed architecture the stage commits agrees with the payload it was built from.
    from kicraft.design.models import Architecture

    assert (
        Architecture.model_validate(payload)
        .obligations[0]
        .model_dump(mode="json", exclude_none=True)
        == obligation
    )


def test_committed_obligation_with_no_implementing_requirement_is_refused():
    """An obligation nobody owns cannot be derived: the refusal names the fix.

    The direction that *is* derivable (the draft only attaches rows) is written by the
    compiler; this is the one the draft has to get right.
    """
    from kicraft.server.stage_contracts import StageSchemaError

    obligation = {
        "kind": "physical",
        "original_obligation_id": "status-led",
        "component_class": "status-led",
    }
    prompt_state = {"intent": {"goal": "reference board", "obligations": [obligation]}}

    with pytest.raises(StageSchemaError) as rejected:
        _normalize_stage_response("architecture", _hub75_intent(), prompt_state)

    assert rejected.value.diagnostic["code"] == "source_obligation_not_retained"
    assert rejected.value.diagnostic["evidence"] == [obligation]


def test_unknown_slot_field_names_itself_and_the_fix():
    """An invented key must be repairable from the refusal alone.

    The provider schema is generated from the slot models, so an `extra_forbidden` error is a
    key the draft invented; the pydantic text names the key but not the repair.
    """
    from kicraft.server.stage_contracts import StageSchemaError

    intent = _hub75_intent()
    intent["questions_asked"] = ["Which LED colour?"]

    with pytest.raises(StageSchemaError) as rejected:
        _normalize_stage_response("architecture", intent, {"intent": {}, "functional_spec": {}})

    message = str(rejected.value)
    assert "questions_asked" in message
    assert "remove each unknown key" in message


def test_quantity_obligation_may_stand_alone_at_the_top_level():
    """A count over the whole design is not an implementation claim.

    The canary (2026-09-17, `rc-lowpass-bnc`, 2 BNC jacks + a trim pot) attached the physical rows
    to their parts and left `two-bnc-connectors-count` at the top level: two jacks are two
    requirements, so no single one implements the count. Refusing it cost the whole design.
    """
    quantity = {
        "kind": "quantity",
        "original_obligation_id": "two-bnc-connectors-count",
        "subject": "bnc connectors",
        "minimum": 2,
    }
    prompt_state = {
        "intent": {"goal": "reference board", "obligations": [quantity]},
        "functional_spec": {"obligations": [quantity]},
    }

    payload, _expanded = _normalize_stage_response("architecture", _hub75_intent(), prompt_state)

    assert payload["obligations"] == [quantity]
    assert not any(row.get("obligations") for row in payload["requirements"])


def test_one_obligation_may_be_implemented_by_several_requirements():
    """Three binding posts are three requirements, each claiming the one binding-post obligation.

    The canary (2026-09-17, `speaker-crossover`) attached the same row to all three connectors and
    was refused as a duplicate owner. The rows are identical, so ownership is unambiguous; the BOM
    unit counts the groups per requirement.
    """
    obligation = {
        "kind": "physical",
        "original_obligation_id": "binding_post_terminal",
        "component_class": "binding-post-terminal",
    }
    prompt_state = {
        "intent": {"goal": "reference board", "obligations": [obligation]},
        "functional_spec": {"obligations": [obligation]},
    }
    intent = _hub75_intent()
    intent["sheets"].append(
        {
            "name": "OUT",
            "stem": "OUT",
            "role": "connector",
            "function": "Speaker output terminals.",
        }
    )
    for index in (1, 2, 3):
        intent["requirements"].append(
            {
                "id": f"post_{index}",
                "sheet": "OUT",
                "role": "connector",
                "family": "binding-post",
                "obligations": [obligation],
                "declared_ports": [
                    {
                        "key": "signal",
                        "pin": str(index),
                        "direction": "bidirectional",
                        "function": "speaker output contact",
                    }
                ],
                "functional_blocks": ["ESP32_S3_CONTROLLER"],
            }
        )

    payload, _expanded = _normalize_stage_response("architecture", intent, prompt_state)

    owning = [row["id"] for row in payload["requirements"] if row.get("obligations")]
    assert owning == ["post_1", "post_2", "post_3"]
    assert payload["obligations"] == [obligation]


# A board fabrication feature and an absent class: the two obligations that are not parts. The
# raw rows spell the feature/class the way a brief does, so the canonicalising validator is
# exercised too.
_FABRICATION_OBLIGATION = {
    "kind": "fabrication",
    "original_obligation_id": "copper_heatsink_area",
    "feature": "Copper_Area",
    "minimum": 300,
    "unit": "mm2",
}
_FABRICATION_CANONICAL = {**_FABRICATION_OBLIGATION, "feature": "copper-area"}
_NEGATIVE_OBLIGATION = {
    "kind": "negative",
    "original_obligation_id": "no_microcontroller",
    "absent_class": "Microcontroller",
}
_NEGATIVE_CANONICAL = {**_NEGATIVE_OBLIGATION, "absent_class": "microcontroller"}
# The live row from KC-CTBW6M (project 44/917) and KC-9FPA59 (project 1/919): a four-layer
# stack-up the intent stage reads out of the brief, which no requirement can implement.
_STACKUP_OBLIGATION = {
    "kind": "quantitative",
    "original_obligation_id": "pcb-layer-count",
    "quantity": "PCB copper layers",
    "relation": "equal",
    "value": 4.0,
    "unit": "layers",
}


@pytest.mark.parametrize(
    "obligation, canonical",
    [
        pytest.param(_FABRICATION_OBLIGATION, _FABRICATION_CANONICAL, id="fabrication"),
        pytest.param(_NEGATIVE_OBLIGATION, _NEGATIVE_CANONICAL, id="negative"),
        pytest.param(_STACKUP_OBLIGATION, _STACKUP_OBLIGATION, id="stackup"),
    ],
)
def test_board_fact_obligation_commits_with_no_owning_requirement(obligation, canonical):
    """A printed board feature, an absent class, and a build stack-up are board-level facts.

    The canary (2026-09-17, `led-cc-driver`, `star-ornament`, `buck-3a`, `thermocouple-amp`)
    turned "printed copper area as a heatsink" and "no microcontroller" into `physical`
    obligations with a component class no BOM line can ever be, so
    `physical-obligation-unfulfilled` refused those designs forever. The stack-up row above is the
    live KC-CTBW6M / KC-9FPA59 refusal: a four-layer board names no requirement, so requiring one
    refuses the design.
    """
    from kicraft.design.models import Architecture

    prompt_state = {
        "intent": {"goal": "reference board", "obligations": [obligation]},
        "functional_spec": {"obligations": [obligation]},
    }

    payload, _expanded = _normalize_stage_response("architecture", _hub75_intent(), prompt_state)

    assert payload["obligations"] == [canonical]
    # No requirement implements it, and the architecture stage commits it that way.
    assert not any(row.get("obligations") for row in payload["requirements"])
    assert (
        Architecture.model_validate(payload)
        .obligations[0]
        .model_dump(mode="json", exclude_none=True)
        == canonical
    )


@pytest.mark.parametrize(
    "row",
    [
        pytest.param(
            {
                "kind": "quantitative",
                "original_obligation_id": "input_voltage",
                "quantity": "input voltage",
                "relation": "equal",
                "value": 5.0,
                "unit": "V",
            },
            id="input-voltage",
        ),
        pytest.param(
            {
                "kind": "quantitative",
                "original_obligation_id": "display_layers",
                "quantity": "display layers",
                "relation": "equal",
                "value": 4.0,
                "unit": "layers",
            },
            id="part-layer-count",
        ),
    ],
)
def test_a_part_limit_quantitative_row_still_needs_an_owner(row):
    """The stack-up exemption is a board/PCB subject, not any row spelled in `layers`.

    A supply voltage is electrical and a display's layer count belongs to the display: neither is
    a property of the printed board, so the retention gate must still refuse both ownerless. This
    is the negative control that proves the new shape does not widen the exemption.
    """
    from kicraft.server.stage_contracts import StageSchemaError, validate_obligation_retention

    with pytest.raises(StageSchemaError) as refused:
        validate_obligation_retention(
            "architecture",
            {},
            {
                "intent": {"obligations": [row]},
                "functional_spec": {"obligations": [row]},
            },
        )

    assert refused.value.diagnostic["code"] == "source_obligation_not_retained"
    assert refused.value.diagnostic["evidence"] == [row]


def test_a_board_stackup_row_is_board_level_and_an_outline_row_still_is():
    """The predicate admits both board shapes and refuses a part limit spelled in the same unit."""
    from kicraft.design.models import is_board_level_quantitative_obligation

    def quantitative(quantity: str, unit: str) -> dict:
        return {
            "kind": "quantitative",
            "original_obligation_id": "row",
            "quantity": quantity,
            "relation": "equal",
            "value": 4.0,
            "unit": unit,
        }

    assert is_board_level_quantitative_obligation(quantitative("PCB copper layers", "layers"))
    assert is_board_level_quantitative_obligation(quantitative("four layer PCB stack-up", "layers"))
    # The original outline shape is unchanged.
    assert is_board_level_quantitative_obligation(quantitative("PCB copper thickness", "mm"))
    assert not is_board_level_quantitative_obligation(quantitative("header pitch", "inch"))


def test_board_fact_obligations_survive_intent_to_architecture_verbatim():
    """Retention is keyed by `(kind, original_obligation_id)`, so the new kinds flow through.

    Measured with the real helpers rather than assumed: the intent row is canonicalised once,
    `restore_source_obligations` writes the top-level list from the committed set, the
    functional_spec copy is compared row-for-row and still refuses a dropped row, and the
    architecture stage commits both rows with no owner.
    """
    from kicraft.server.stage_contracts import (
        StageSchemaError,
        restore_source_obligations,
        validate_obligation_retention,
    )

    rows = [_FABRICATION_CANONICAL, _NEGATIVE_CANONICAL]
    prompt_state = {
        "intent": {"goal": "reference board", "obligations": [*rows]},
        "functional_spec": {"obligations": [*rows]},
    }

    restored = restore_source_obligations({}, {"intent": prompt_state["intent"]})
    assert restored["obligations"] == rows

    spec = {
        "blocks": [{"name": "DRIVER", "category": "drive", "purpose": "Drive the LED string."}],
        "obligations": [*rows],
    }
    committed, _expanded = _normalize_stage_response(
        "functional_spec", spec, {"intent": prompt_state["intent"]}
    )
    assert committed["obligations"] == rows

    with pytest.raises(StageSchemaError) as dropped:
        _normalize_stage_response(
            "functional_spec",
            {**spec, "obligations": rows[:1]},
            {"intent": prompt_state["intent"]},
        )
    assert dropped.value.diagnostic["evidence"] == [_NEGATIVE_CANONICAL]
    # Neither row needs a requirement, in the retention check or in the committed architecture.
    validate_obligation_retention("architecture", {}, prompt_state)

    payload, _expanded = _normalize_stage_response("architecture", _hub75_intent(), prompt_state)

    assert payload["obligations"] == rows
    assert not any(row.get("obligations") for row in payload["requirements"])


def test_fabrication_obligation_limit_needs_a_minimum():
    """A unit with no minimum states no limit: the model refuses the meaningless row."""
    from kicraft.design.models import FabricationObligation

    with pytest.raises(ValueError):
        FabricationObligation(
            kind="fabrication",
            original_obligation_id="copper_heatsink_area",
            feature="copper-area",
            unit="mm2",
        )


def test_ownership_exemption_is_per_row_and_does_not_shield_a_physical_row():
    """A `fabrication` row beside an unowned `physical` row does not excuse the physical one."""
    from kicraft.server.stage_contracts import StageSchemaError

    physical = {
        "kind": "physical",
        "original_obligation_id": "status_led",
        "component_class": "status-led",
    }
    prompt_state = {
        "intent": {
            "goal": "reference board",
            "obligations": [physical, _FABRICATION_CANONICAL],
        },
        "functional_spec": {"obligations": [physical, _FABRICATION_CANONICAL]},
    }
    intent = _hub75_intent()
    next(row for row in intent["requirements"] if row["id"] == "led")["obligations"] = [physical]

    payload, _expanded = _normalize_stage_response("architecture", intent, prompt_state)

    assert payload["obligations"] == [physical, _FABRICATION_CANONICAL]
    assert [row["id"] for row in payload["requirements"] if row.get("obligations")] == ["led"]

    for row in intent["requirements"]:
        row.pop("obligations", None)
    with pytest.raises(StageSchemaError) as refused:
        _normalize_stage_response("architecture", intent, prompt_state)
    assert refused.value.diagnostic["evidence"] == [physical]


def test_board_outline_measurement_may_stand_alone_but_electrical_limit_may_not():
    """Only a geometric board fact is ownerless; V/A/Hz limits remain implementation claims."""
    from kicraft.server.stage_contracts import StageSchemaError

    board_diameter = {
        "kind": "quantitative",
        "original_obligation_id": "board_diameter",
        "quantity": "board diameter",
        "relation": "maximum",
        "value": 60,
        "unit": "mm",
    }
    prompt_state = {
        "intent": {"goal": "reference board", "obligations": [board_diameter]},
        "functional_spec": {"obligations": [board_diameter]},
    }
    payload, _expanded = _normalize_stage_response("architecture", _hub75_intent(), prompt_state)
    assert payload["obligations"] == [board_diameter]
    assert not any(row.get("obligations") for row in payload["requirements"])

    output_voltage = {
        "kind": "quantitative",
        "original_obligation_id": "output_voltage",
        "quantity": "output voltage",
        "relation": "equal",
        "value": 3.3,
        "unit": "V",
    }
    with pytest.raises(StageSchemaError) as rejected:
        _normalize_stage_response(
            "architecture",
            _hub75_intent(),
            {
                "intent": {"goal": "reference board", "obligations": [output_voltage]},
                "functional_spec": {"obligations": [output_voltage]},
            },
        )
    assert rejected.value.diagnostic["evidence"] == [output_voltage]

    implementing_intent = _hub75_intent()
    next(row for row in implementing_intent["requirements"] if row["id"] == "buck")[
        "obligations"
    ] = [output_voltage]
    retained, _expanded = _normalize_stage_response(
        "architecture",
        implementing_intent,
        {
            "intent": {"goal": "reference board", "obligations": [output_voltage]},
            "functional_spec": {"obligations": [output_voltage]},
        },
    )
    assert next(row for row in retained["requirements"] if row["id"] == "buck")["obligations"] == [
        output_voltage
    ]


def test_board_outline_is_normalized_from_semantic_evidence_not_shape_word():
    """A board outline becomes fabrication; a realizable mounting hole remains physical."""
    from kicraft.design.stage_semantics import complete_intent_classification
    from kicraft.server.stage_contracts import normalize_board_outline_obligations

    outline = {
        "kind": "physical",
        "original_obligation_id": "snowman_outline",
        "component_class": "snowman-shaped-board",
    }
    mounting_hole = {
        "kind": "physical",
        "original_obligation_id": "mounting_hole",
        "component_class": "mounting-hole",
    }
    normalized = complete_intent_classification(
        "A snowman-shaped board.", {"obligations": [outline, mounting_hole]}
    )
    assert normalized["obligations"] == [
        {
            "kind": "fabrication",
            "original_obligation_id": "snowman_outline",
            "feature": "snowman-shaped-board",
        },
        mounting_hole,
    ]
    # The compiler also repairs legacy source rows before source comparison.
    assert (
        normalize_board_outline_obligations({"obligations": [outline, mounting_hole]})[
            "obligations"
        ]
        == normalized["obligations"]
    )


def test_only_unique_reviewed_physical_owner_is_attached():
    """Reviewed evidence can repair omitted ownership, but never chooses between candidates."""
    from kicraft.server.stage_contracts import (
        attach_uniquely_provable_physical_obligations,
        physical_obligation_candidate_requirement_ids,
    )

    obligation = {
        "kind": "physical",
        "original_obligation_id": "warm_white_led",
        "component_class": "warm-white-led",
    }
    source = {
        "intent": {"obligations": [obligation]},
        "functional_spec": {"obligations": [obligation]},
    }
    exact_led = "E6C0805WWAY1UDA(1.1T M)"
    unique = attach_uniquely_provable_physical_obligations(
        {"requirements": [{"id": "indicator", "exact_part": exact_led}]},
        source,
    )
    assert unique["requirements"][0]["obligations"] == [obligation]
    assert "unique reviewed recipe/lowerer evidence" in unique["assumptions"][0]

    ambiguous = {
        "requirements": [
            {"id": "indicator_a", "exact_part": exact_led},
            {"id": "indicator_b", "exact_part": exact_led},
        ]
    }
    assert physical_obligation_candidate_requirement_ids(ambiguous, obligation) == [
        "indicator_a",
        "indicator_b",
    ]
    assert attach_uniquely_provable_physical_obligations(ambiguous, source) == ambiguous

    from kicraft.server.stage_contracts import StageSchemaError, validate_obligation_retention

    with pytest.raises(StageSchemaError) as rejected:
        validate_obligation_retention("architecture", ambiguous, source)
    assert rejected.value.diagnostic["candidate_requirement_ids"] == {
        "warm_white_led": ["indicator_a", "indicator_b"]
    }


def test_unique_physical_owner_is_attached_before_architecture_commit():
    """The architecture normalizer uses the same evidence-backed attachment path."""
    obligation = {
        "kind": "physical",
        "original_obligation_id": "microcontroller",
        "component_class": "microcontroller",
    }
    payload, _expanded = _normalize_stage_response(
        "architecture",
        _hub75_intent(),
        {
            "intent": {"obligations": [obligation]},
            "functional_spec": {"obligations": [obligation]},
        },
    )
    assert next(row for row in payload["requirements"] if row["id"] == "esp32")["obligations"] == [
        obligation
    ]


_PROTOTYPING_AREA_OBLIGATION = {
    "kind": "fabrication",
    "original_obligation_id": "prototyping_area",
    "feature": "prototyping-area",
}


def _prototyping_area_intent() -> dict:
    """A brief whose only stated board feature is the pad field: no part, no net, no signal."""
    return {
        "mcu_present": False,
        "sheets": [],
        "requirements": [],
        "signals": [],
        "obligations": [dict(_PROTOTYPING_AREA_OBLIGATION)],
    }


def _prototyping_area_spec(name: str = "PROTOTYPING AREA") -> dict:
    return {
        "blocks": [
            {
                "name": name,
                "category": "interface",
                "purpose": "Bare pad field the user solders through-hole parts into.",
                "count": 1,
            }
        ],
        "connections": [],
    }


def test_fabrication_obligation_derives_the_prototyping_area_sheet_and_requirement():
    """A `fabrication` obligation is the whole statement; the derivation writes the rest.

    The measured failure: the brief asks for a prototyping area, the model records it as an
    adjective, and every later stage has nothing to build -- a sheet with no requirement dies at
    BOM (empty sheet) and a part the model invents from prose dies at the architecture gates.
    Here the field is the only thing the board has: the spec declares no block for it (it must
    not -- a block is a user-visible function), so the derived requirement owns none.
    """
    from kicraft.design.models import Architecture, FunctionalSpec
    from kicraft.design.synthesis.validation import (
        check_every_block_has_sheet,
        check_fs_connections_mapped,
    )

    spec = {"blocks": [], "connections": []}
    architecture = derive_architecture(_prototyping_area_intent(), spec)

    sheet = next(row for row in architecture.sheets if row.stem == "PROTOTYPING_AREA")
    assert sheet.name == "PROTOTYPING AREA"
    assert "solders" in sheet.function

    requirement = _requirement(architecture, "prototyping_area")
    assert requirement.sheet == "PROTOTYPING AREA"
    assert requirement.role == "user_io"
    assert requirement.family == "prototyping-area"
    assert requirement.parameters == {"rows": 5, "cols": 5, "pitch_mm": 2.54}
    assert requirement.ports == {}
    assert requirement.functional_blocks == []
    # The row is a board-level fact: `fabrication` is ownership-exempt and owns no requirement.
    assert requirement.obligations == []
    assert [row.model_dump(exclude_none=True) for row in architecture.obligations] == [
        _PROTOTYPING_AREA_OBLIGATION
    ]
    assert any(
        row.startswith("prototyping_area:") and row.endswith("(derived)")
        for row in architecture.assumptions
    )

    # The gates the architecture stage commits on (`cli_app.py` R4) pass with no block owning the
    # field: the board's own `fabrication` row is the exemption from block membership, and the
    # field binds no net to cross a sheet with. The real normalization keeps the requirement.
    functional_spec = FunctionalSpec.model_validate(spec)
    assert check_every_block_has_sheet(functional_spec, architecture).ok
    assert check_fs_connections_mapped(functional_spec, architecture).ok
    payload, _expanded = _normalize_stage_response(
        "architecture",
        architecture.model_dump(exclude_none=True),
        {"intent": _prototyping_area_intent(), "functional_spec": spec},
    )
    committed = Architecture.model_validate(payload)
    assert [row.id for row in committed.requirements] == ["prototyping_area"]
    # No recipe is invented for a board feature, and the requirement itself resolves to the
    # deterministic pad-field lowerer: its sheet is BOM work with a known build rather than an
    # empty sheet the model would have to invent parts for.
    assert committed.unresolved_requirement_ids == ["prototyping_area"]
    assert committed.recipe_selections == []
    from kicraft.design.lowering import lower_requirement

    assert lower_requirement(_requirement(committed, "prototyping_area")) is not None


def test_prototyping_area_requirement_claims_the_committed_block_that_asked_for_it():
    """No block is expected by default; one is claimed only if a spec declares it anyway.

    The functional spec must not declare a block for the pad field (it is a board feature, not a
    user-visible function), so the derived requirement owns no block in the normal case. The
    matching is kept for the spec that declares one anyway: the requirement must then claim that
    block's exact name, or the block-coverage gate reports it unowned.
    """
    from kicraft.design.models import FunctionalSpec

    matched = derive_architecture(_prototyping_area_intent(), _prototyping_area_spec("PROTO BOARD"))
    assert _requirement(matched, "prototyping_area").functional_blocks == ["PROTO BOARD"]

    # The committed slot reaches the derivation as a mapping or as its own model.
    committed = FunctionalSpec.model_validate(_prototyping_area_spec("PAD_FIELD"))
    assert _requirement(
        derive_architecture(_prototyping_area_intent(), committed), "prototyping_area"
    ).functional_blocks == ["PAD_FIELD"]

    for spec in (None, _prototyping_area_spec("POWER INPUT"), {"blocks": []}):
        architecture = derive_architecture(_prototyping_area_intent(), spec)
        assert _requirement(architecture, "prototyping_area").functional_blocks == []


def test_derived_pad_field_clears_the_commit_gates_with_no_spec_block():
    """The default shape: no block for the field, and R4 still commits the board.

    A `fabrication` row is a property of the board, so the derived requirement implements no
    functional block and claims none; `check_every_block_has_sheet` exempts exactly that case
    (keyed by the row's obligation id) while every other requirement still declares its block.
    """
    from kicraft.design.models import FunctionalSpec
    from kicraft.design.synthesis.validation import (
        check_every_block_has_sheet,
        check_fs_connections_mapped,
    )

    intent = _hub75_intent()
    intent["obligations"] = [dict(_PROTOTYPING_AREA_OBLIGATION)]
    spec = {
        "blocks": [
            {"name": name, "category": "interface", "purpose": "Stated function.", "count": 1}
            for name in (
                "ESP32_S3_CONTROLLER",
                "POWER_DISTRIBUTION",
                "HUB75_DISPLAY_INTERFACE",
                "ADDRESSABLE_LED_OUTPUT",
                "USB_C_PD_INPUT",
            )
        ],
        "connections": [],
    }

    architecture = derive_architecture(intent, spec)

    assert _requirement(architecture, "prototyping_area").functional_blocks == []
    functional_spec = FunctionalSpec.model_validate(spec)
    assert check_every_block_has_sheet(functional_spec, architecture).ok
    assert check_fs_connections_mapped(functional_spec, architecture).ok


def test_a_requirement_owned_fabrication_row_still_derives_the_pad_field():
    """`fabrication` may ride a requirement (it is ownership-exempt); the fact still counts."""
    intent = _hub75_intent()
    next(row for row in intent["requirements"] if row["id"] == "led")["obligations"] = [
        dict(_PROTOTYPING_AREA_OBLIGATION)
    ]

    architecture = derive_architecture(intent, _prototyping_area_spec())

    assert _requirement(architecture, "prototyping_area").sheet == "PROTOTYPING AREA"


def test_second_derivation_pass_does_not_duplicate_the_prototyping_area():
    """The derivation is a pure function of the obligation, and a declared field is left alone."""
    spec = _prototyping_area_spec()
    first = derive_architecture(_prototyping_area_intent(), spec)
    second = derive_architecture(_prototyping_area_intent(), spec)
    assert first.model_dump() == second.model_dump()
    assert [row.stem for row in first.sheets] == ["PROTOTYPING_AREA"]

    # A model that stated the feature itself -- its own sheet prose, its own family, its own
    # ports -- keeps every one of those. The guard reads the id and the family, so no second
    # requirement and no second sheet appear at all.
    declared = {
        "sheets": [
            {
                "name": "PROTO AREA",
                "stem": "PROTOTYPING_AREA",
                "role": "user_io",
                "function": "Pad field the user solders into (model text).",
            }
        ],
        "requirements": [
            {
                "id": "prototyping_area",
                "sheet": "PROTO AREA",
                "role": "user_io",
                "family": "pin-header",
                "parameters": {"rows": 1, "gender": "male"},
                "functional_blocks": ["PROTOTYPING AREA"],
                "ties": {"pin1": "GND"},
            }
        ],
        "signals": [],
        "obligations": [dict(_PROTOTYPING_AREA_OBLIGATION)],
    }
    architecture = derive_architecture(declared, spec)

    assert [row.stem for row in architecture.sheets] == ["PROTOTYPING_AREA"]
    assert architecture.sheets[0].function == "Pad field the user solders into (model text)."
    requirement = _requirement(architecture, "prototyping_area")
    assert requirement.sheet == "PROTO AREA"
    assert requirement.family == "pin-header"
    assert requirement.parameters == {"rows": 1, "gender": "male"}
    assert requirement.ports == {"pin1": "GND"}
    assert not any("derived from the committed" in row for row in architecture.assumptions)


def test_declared_prototyping_area_sheet_is_reused_not_duplicated():
    """A model-declared sheet keeps its prose; only the missing requirement is derived onto it."""
    intent = _prototyping_area_intent()
    intent["sheets"] = [
        {
            "name": "PROTO AREA",
            "stem": "PROTOTYPING_AREA",
            "role": "user_io",
            "function": "Pad field the user solders into (model text).",
        }
    ]

    architecture = derive_architecture(intent, _prototyping_area_spec())

    assert [row.stem for row in architecture.sheets] == ["PROTOTYPING_AREA"]
    assert architecture.sheets[0].function == "Pad field the user solders into (model text)."
    assert _requirement(architecture, "prototyping_area").sheet == "PROTO AREA"


def test_intent_without_the_fabrication_obligation_derives_no_prototyping_area():
    """A design that never named the feature is untouched, even when a spec block names it."""
    with_block = derive_architecture(_hub75_intent(), _prototyping_area_spec())

    assert with_block.model_dump() == derive_architecture(_hub75_intent()).model_dump()
    assert not any(row.stem == "PROTOTYPING_AREA" for row in with_block.sheets)
    assert not any(row.id == "prototyping_area" for row in with_block.requirements)


def test_unreviewed_exact_part_for_a_covered_class_is_recorded_with_its_options():
    """§4.3 A RECORD class: the part is shipped as chosen, with the reviewed options listed.

    The proto-shield runs answered the demanded `voltage-regulator` class with the familiar but
    unreviewed `AMS1117-3.3`. The board is buildable and very likely correct, so the choice is
    recorded on the artifact — with the reviewed identities that would have proven it — instead
    of killing the run (design-yield-recovery plan §4.3 A; measured as a null while it stayed a
    refusal, because every run it rescued died one stage later).
    """
    intent = _hub75_intent()
    buck = next(row for row in intent["requirements"] if row["id"] == "buck")
    buck["exact_part"] = "AMS1117-3.3"
    buck["obligations"] = [
        {
            "kind": "physical",
            "original_obligation_id": "regulator",
            "component_class": "voltage-regulator",
        }
    ]
    architecture = derive_architecture(intent)
    recorded = [a for a in architecture.advisories if a.code == "unreviewed_exact_part"]
    assert len(recorded) == 1
    # The reviewed options travel with the note, so a reviewer can act on it in one step.
    assert "me6211c33m5g-n" in recorded[0].evidence
    assert "ap2112k-3.3trg1" in recorded[0].evidence
    assert next(r for r in architecture.requirements if r.id == "buck").exact_part == "AMS1117-3.3"

    # The reviewed identity for the same class passes untouched, with nothing to record.
    buck["exact_part"] = "ME6211C33M5G-N"
    architecture = derive_architecture(intent)
    assert (
        next(r for r in architecture.requirements if r.id == "buck").exact_part == "ME6211C33M5G-N"
    )
    assert [a.code for a in architecture.advisories] == []


def test_model_declared_pad_field_is_normalized_to_the_reviewed_default():
    """The pad field's size and its netless nature are the reviewed contract.

    A real run declared 10x15 (150 plated pads) and its board was the one the layout engine
    could not route, while the acceptance check asks for a usable field of >=25 positions.
    The model keeps its sheet, id and role; the size and the empty port set are derived.
    """
    intent = _hub75_intent()
    intent["obligations"] = [
        {
            "kind": "fabrication",
            "original_obligation_id": "prototyping_area",
            "feature": "prototyping-area",
        }
    ]
    intent["sheets"].append(
        {"name": "PROTO FIELD", "stem": "PROTO_FIELD", "role": "interface", "function": "Pads."}
    )
    intent["requirements"].append(
        {
            "id": "prototyping_area",
            "sheet": "PROTO FIELD",
            "role": "connector",
            "family": "prototyping-area",
            "parameters": {"rows": 10, "cols": 15, "pitch_mm": 2.54},
            "functional_blocks": ["POWER_DISTRIBUTION"],
        }
    )
    architecture = derive_architecture(intent)
    field = next(r for r in architecture.requirements if r.id == "prototyping_area")
    assert field.parameters == {"rows": 5, "cols": 5, "pitch_mm": 2.54}
    assert field.ports == {}
    assert field.sheet == "PROTO FIELD"  # the model's own sheet is kept


def test_stacking_owners_share_one_interface_block():
    """An extra block on one connector is dropped only when something else still owns it.

    The wiring stage may permute which owner carries which template geometry, so connectors
    that agree on one interface block are normalised to it -- but a block NO requirement
    would still implement must not be dropped, because the architecture commit refuses a
    functional block with no implementation requirement
    ("functional block 'POWER_INPUT' has no implementation requirement on a sheet").
    """
    from kicraft.form_factors import get_template

    template = get_template("arduino_uno_shield")
    intent = _hub75_intent()
    intent["standard_form_factor"] = template.key
    intent["sheets"].append(
        {
            "name": "UNO HEADERS",
            "stem": "UNO_HEADERS",
            "role": "connector",
            "function": "Arduino Uno shield stacking interface.",
        }
    )
    for connector in template.fixed_connectors:
        blocks = ["UNO_HOST_INTERFACE"]
        if connector.role == "power":
            blocks.append("POWER_DISTRIBUTION")  # the deviation
        intent["requirements"].append(
            {
                "id": f"uno_{connector.role}",
                "sheet": "UNO HEADERS",
                "role": "connector",
                "family": "pin-header",
                "parameters": {"rows": 1, "gender": "female"},
                "standard_stacking_role": connector.role,
                "functional_blocks": blocks,
            }
        )

    # `POWER_DISTRIBUTION` is also owned by the buck requirement in this fixture, so the
    # connectors normalise to the interface block they all implement.
    architecture = derive_architecture(intent)
    owned = {
        requirement.standard_stacking_role: requirement
        for requirement in architecture.requirements
        if requirement.standard_stacking_role
    }
    assert set(owned) == {c.role for c in template.fixed_connectors}
    assert {tuple(sorted(r.functional_blocks)) for r in owned.values()} == {("UNO_HOST_INTERFACE",)}

    # A block NO other requirement implements must survive, or it would be left with no
    # implementation requirement at all -- which the architecture commit refuses.
    alone = _hub75_intent()
    alone["standard_form_factor"] = template.key
    alone["sheets"].append(
        {
            "name": "UNO HEADERS",
            "stem": "UNO_HEADERS",
            "role": "connector",
            "function": "Arduino Uno shield stacking interface.",
        }
    )
    for connector in template.fixed_connectors:
        blocks = ["UNO_HOST_INTERFACE"]
        if connector.role == "power":
            blocks.append("HOST_POWER_ENTRY")
        alone["requirements"].append(
            {
                "id": f"uno_{connector.role}",
                "sheet": "UNO HEADERS",
                "role": "connector",
                "family": "pin-header",
                "parameters": {"rows": 1, "gender": "female"},
                "standard_stacking_role": connector.role,
                "functional_blocks": blocks,
            }
        )
    kept = derive_architecture(alone)
    owned = {
        requirement.standard_stacking_role: requirement
        for requirement in kept.requirements
        if requirement.standard_stacking_role
    }
    assert tuple(sorted(owned["power"].functional_blocks)) == (
        "HOST_POWER_ENTRY",
        "UNO_HOST_INTERFACE",
    )


def _one_part_intent(requirement: dict, signals: list[dict] | None = None) -> dict:
    """The smallest intent that can carry one lowerer-family part of its own."""
    return {
        "topologies": {"POWER": "5 V input header"},
        "comms_protocols": [],
        "mcu_present": False,
        "power": {"rails": {"+5V": {"voltage": 5.0, "from": "input.pin1"}}},
        "sheets": [
            {
                "name": "INPUT",
                "stem": "INPUT",
                "role": "connector",
                "function": "5 V input header.",
            },
            {
                "name": "PART",
                "stem": "PART",
                "role": "analog_block",
                "function": "The part under test.",
            },
        ],
        "requirements": [
            {
                "id": "input",
                "sheet": "INPUT",
                "role": "connector",
                "family": "pin-header",
                "parameters": {"rows": 1, "gender": "male"},
                "ties": {"pin1": "+5V", "pin2": "GND"},
                "functional_blocks": ["POWER_IN"],
            },
            {"sheet": "PART", "functional_blocks": ["THE_PART"], **requirement},
        ],
        "signals": signals or [],
        "assumptions": [],
    }


def test_published_return_contact_is_answered_with_the_design_ground():
    """A lowerer's return contact is a fact about the part, not a statement to extract.

    A coin cell's `negative` is the return of the reviewed part, so the design's own ground
    answers it: the compiler binds the published return instead of refusing the run for the
    one contact the draft did not restate (move 1a). The draft binds `positive` itself.
    """
    intent = _one_part_intent(
        {
            "id": "battery",
            "role": "power_input",
            "family": "coin-cell-holder",
            "parameters": {"cell_format": "CR2032"},
            "exact_part": "BS-07-A1BJ001",
            "ties": {},
        },
        [{"name": "VBAT", "from": "battery.positive", "to": "edge:VBAT_OUT"}],
    )
    architecture = derive_architecture(intent)
    battery = _requirement(architecture, "battery")
    assert battery.ports["positive"] == "VBAT"
    assert battery.ports["negative"] == "GND"


def test_connector_contact_naming_a_declared_rail_binds_that_rail():
    """A connector pin a signal names after a declared rail is that rail's exposure.

    The compiler already derives this shape when `power.rails[net].from_ref` names the pin;
    a draft that states it with a signal names a declared net on a pin it also named, so the
    contact carries the rail instead of the run dying on the signal and then again on the
    contact the refusal left unbound.
    """
    intent = _one_part_intent(
        {
            "id": "host",
            "role": "connector",
            "family": "pin-header",
            "parameters": {"rows": 1, "gender": "female"},
        },
        [{"name": "+5V", "from": "host.pin1", "to": "edge:RAIL_OUT"}],
    )
    intent["power"]["rails"]["+5V"]["from"] = "host.pin2"
    architecture = derive_architecture(intent)
    host = _requirement(architecture, "host")
    assert host.ports["pin1"] == "+5V"
    assert any("carries rail '+5V'" in row for row in architecture.assumptions)


def test_duplicate_power_statements_are_dropped_before_the_compiler_refuses_them():
    """A refusal at decode costs a whole ladder round; the duplicate statement is mechanical.

    Live walkthrough (2026-09-25, seed 37): one draft failed outright after five rejections,
    two of them "signal 'MOTOR_SUPPLY': port 'vm' of 'hbridge' is already bound to
    'MOTOR_VOLTAGE'" and "signal 'GND_INPUT': port 'gnd' of 'reg' is already bound to 'GND'".
    """
    import copy

    from kicraft.design.architecture_intent import (
        complete_architecture_payload,
        derive_architecture,
    )

    payload = _hub75_intent()
    # The compiler accepts this design as it stands.
    assert derive_architecture(payload, None) is not None

    mcu = next(
        row for row in payload["requirements"] if str(row.get("role")) == "mcu_core"
    )
    duplicated = copy.deepcopy(payload)
    duplicated["signals"] = [
        *duplicated["signals"],
        # Restates the supply the rail already feeds...
        {"name": "MCU_SUPPLY", "from": "input.positive", "to": f"{mcu['id']}.vdd"},
        # ...and a return pin, which is implicit on every part.
        {"name": "MCU_GND", "from": "input.positive", "to": f"{mcu['id']}.gnd"},
    ]

    completed = complete_architecture_payload(duplicated)

    names = [row["name"] for row in completed["signals"]]
    assert "MCU_SUPPLY" not in names and "MCU_GND" not in names
    assert [row for row in names if row in [s["name"] for s in payload["signals"]]] == [
        row["name"] for row in payload["signals"]
    ]
    # The writer's payload is untouched, and the design still derives.
    assert len(duplicated["signals"]) == len(payload["signals"]) + 2
    assert derive_architecture(completed, None) is not None


def test_a_tie_field_on_a_port_that_carries_a_signal_is_cleared():
    """The signal is what names that net; the tie field names a second one on the same pin."""
    from kicraft.design.architecture_intent import complete_architecture_payload

    payload = {
        "requirements": [
            {"id": "mcu", "role": "mcu_core", "family": "esp32-c3-mini-1-module"},
            {
                "id": "sense",
                "role": "sensor",
                "family": "temperature-sensor",
                "declared_ports": [
                    {
                        "key": "out",
                        "pin": "1",
                        "direction": "output",
                        "function": "reading",
                        "supply_rail": "VIN18",
                        "reference_domain": "GND",
                    }
                ],
            },
        ],
        "signals": [{"name": "SENSOR_OUT", "from": "mcu.output_1", "to": "sense.out"}],
    }

    completed = complete_architecture_payload(payload)

    entry = completed["requirements"][1]["declared_ports"][0]
    assert "supply_rail" not in entry and "reference_domain" not in entry
    assert entry["function"] == "reading"
    assert payload["requirements"][1]["declared_ports"][0]["supply_rail"] == "VIN18"


def test_a_payload_without_duplicates_is_returned_untouched():
    from kicraft.design.architecture_intent import complete_architecture_payload

    payload = {
        "requirements": [{"id": "mcu", "role": "mcu_core", "family": "esp32-c3-mini-1-module"}],
        "signals": [{"name": "LED_DRIVE", "from": "mcu.output_status", "to": "led.anode"}],
    }
    assert complete_architecture_payload(payload) is payload


def test_a_lowerer_family_is_replaced_by_the_reviewed_carriers_family():
    """The writer's generic connector family cannot implement a demanded class that has a carrier.

    Live walkthrough (2026-09-25): `lowerer pin-header@1 does not implement the exact part
    'B2B-XH-A(LF)(SN)'` on every attempt, and the BOM then exhausted its rounds on
    `missing-requirement-implementation=['motor_a']`, because the parts stage may not reopen a
    family. The carrier's own family is adopted, the lowerer-only parameters go, and the
    interface is built from the signals the draft already sends.
    """
    from kicraft.design.architecture_intent import complete_architecture_payload

    payload = {
        "power": {"rails": {"+18V": {"voltage": 18.0, "from": "power_in.pin1"}}},
        "requirements": [
            {"id": "power_in", "role": "power_input", "family": "screw-terminal",
             "parameters": {"rows": 1}},
            {"id": "motor_a", "role": "connector", "family": "pin-header",
             "parameters": {"rows": 1, "gender": "female"},
             "obligations": [{"kind": "physical", "original_obligation_id": "xh",
                              "component_class": "jst-xh-connector"}]},
        ],
        "signals": [
            {"name": "MOTOR_A1", "from": "hbridge.aout1", "to": "motor_a.pin1"},
            {"name": "MOTOR_A2", "from": "hbridge.aout2", "to": "motor_a.pin2"},
        ],
    }

    complete_architecture_payload(payload)

    connector = payload["requirements"][1]
    assert connector["family"] == "jst-xh-connector"
    assert connector["parameters"] == {}          # lowerer-only keys dropped
    assert connector["declared_ports"] == [
        {"key": "pin1", "pin": "1", "direction": "passive", "function": "carries MOTOR_A1"},
        {"key": "pin2", "pin": "2", "direction": "passive", "function": "carries MOTOR_A2"},
    ]
    # The terminal's missing return is completed in the spelling the draft used.
    assert payload["requirements"][0]["ties"] == {"pin2": "GND"}


def test_a_named_reviewed_part_on_a_lowerer_family_adopts_that_parts_family():
    from kicraft.design.architecture_intent import complete_architecture_payload

    payload = {
        "requirements": [
            {"id": "motor_a", "role": "connector", "family": "pin-header",
             "exact_part": "B2B-XH-A(LF)(SN)", "parameters": {"rows": 1, "gender": "female"}}
        ],
        "signals": [{"name": "MOTOR_A1", "from": "hbridge.aout1", "to": "motor_a.pin1"}],
    }
    complete_architecture_payload(payload)
    connector = payload["requirements"][0]
    assert connector["family"] == "jst-xh-connector"
    assert connector["parameters"] == {}
    assert connector["declared_ports"][0]["pin"] == "1"


def test_a_terminal_spelled_positive_gets_its_negative_return():
    from kicraft.design.architecture_intent import complete_architecture_payload

    payload = {
        "requirements": [{"id": "power_in", "role": "power_input", "family": "screw-terminal"}],
        "signals": [{"name": "VIN", "from": "power_in.positive", "to": "reg.input"}],
    }
    complete_architecture_payload(payload)
    assert payload["requirements"][0]["ties"] == {"negative": "GND"}


def test_a_writer_declared_usb_socket_is_dropped_only_when_the_compiler_writes_one():
    from kicraft.design.architecture_intent import complete_architecture_payload

    base = {
        "requirements": [
            {"id": "mcu", "role": "mcu_core", "family": "esp32-c3-mini-1-module"},
            {"id": "usb_socket", "role": "connector", "family": "usb-c-usb2-device"},
        ],
    }
    with_edge = {
        **base,
        "signals": [
            {"name": "USB_DM", "from": "mcu.usb_dm", "to": "edge:USB"},
            {"name": "USB_DP", "from": "mcu.usb_dp", "to": "edge:USB"},
        ],
    }
    complete_architecture_payload(with_edge)
    assert [row["id"] for row in with_edge["requirements"]] == ["mcu"]

    # No edge: the requirement is the design's own statement and stays.
    no_edge = {**base, "signals": []}
    complete_architecture_payload(no_edge)
    assert [row["id"] for row in no_edge["requirements"]] == ["mcu", "usb_socket"]

    # A signal referencing it keeps it: dropping it would leave the signal dangling.
    referenced = {
        **with_edge,
        "requirements": [*base["requirements"]],
        "signals": [
            {"name": "USB_DM", "from": "mcu.usb_dm", "to": "usb_socket.dm"},
            {"name": "USB_DP", "from": "mcu.usb_dp", "to": "edge:USB"},
        ],
    }
    complete_architecture_payload(referenced)
    assert "usb_socket" in [row["id"] for row in referenced["requirements"]]
