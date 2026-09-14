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
    ] + [row for row in intent["signals"] if not row["name"].startswith("HUB75_R")]
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
        row if row["name"] != "LED_DATA" else {**row, "to": "led_string.pin1"}
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
            "family": "pin-header",
            "parameters": {"rows": 1, "gender": "male"},
            "functional_blocks": ["ADDRESSABLE_LED_OUTPUT"],
        }
    )
    ambiguous["signals"] = [
        {**row, "to": ["le.pin1"]} if row["name"] == "HUB75_ADDR_B" else row
        for row in ambiguous["signals"]
    ]
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(ambiguous)
    assert [row.code for row in excinfo.value.diagnostics] == ["unknown_signal_requirement"]


def test_half_a_usb_pair_is_refused_by_name():
    intent = _hub75_intent()
    intent["signals"] = [row for row in intent["signals"] if row["name"] != "USB_D_N"]
    with pytest.raises(ArchitectureIntentError) as excinfo:
        derive_architecture(intent)
    assert [row.code for row in excinfo.value.diagnostics] == ["incomplete_usb_edge"]


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
    intent-shaped slot the flag asks for, and requires a first-attempt commit.
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
    client.s = Settings(api_key="test", architecture_slot="intent")
    result = run_session(
        tmp_path, "a USB-C ESP32-S3 HUB75 controller", ["intent", "architecture"], client=client
    )
    architecture = next(row for row in result["results"] if row["stage"] == "architecture")
    assert architecture["commit_ok"] is True, architecture
    assert architecture["attempts"] == 1, architecture
    # The provider was asked for the intent slot, not the explicit one.
    assert client.calls[-1]["response_format"]["json_schema"]["name"].startswith(
        "kicraft_architecture_intent_response"
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

    half = _hub75_intent()
    half["signals"] = [row for row in half["signals"] if row["name"] != "USB_D_N"]
    events: list[dict] = []
    client = _ScriptedClient(
        [_reply(json.loads(_OK_INTENT)), _reply(half), _reply(_hub75_intent())]
    )
    client.s = Settings(api_key="test", architecture_slot="intent")
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
