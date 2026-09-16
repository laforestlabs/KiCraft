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


def test_ranges_expand_to_one_net_per_index():
    architecture = derive_architecture(_hub75_intent())
    names = {row.name for row in architecture.inter_sheet_nets}
    assert {"HUB75_R0", "HUB75_R1", "HUB75_B0", "HUB75_B1"} <= names


def test_unused_groundable_recipe_port_is_tied_low():
    """`hub75.addr_d` is a spare address channel on a 1/16-scan panel: tied, not rejected."""
    architecture = derive_architecture(_hub75_intent())
    assert _requirement(architecture, "hub75").ports["addr_d"] == "GND"
    assert any("addr_d tied to GND" in row for row in architecture.assumptions)


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
    without_peer["signals"] = [
        row for row in without_peer["signals"] if row["name"] != "LED_OUT"
    ]
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
    assert _requirement(architecture, "bme").ports == {
        "gnd": "GND",
        "vdd": "+3V3",
        "sda": "I2C_SDA",
        "scl": "I2C_SCL",
    }
    assert any("pin functions are a claim" in row for row in architecture.assumptions)


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
                    f"pin{index}": net
                    for index, net in enumerate(connector.net_by_pin, start=1)
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
