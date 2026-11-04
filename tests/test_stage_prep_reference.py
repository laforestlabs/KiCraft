"""Architecture checkpoint capabilities must construct the requested circuits."""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from kicraft.design.cli_app import main
from kicraft.design.models import ConversationState, IntentSlot


def _state(tmp_path, intent: IntentSlot) -> Path:
    state_path = tmp_path / ".kicraft" / "state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = ConversationState(project_stem="REFCHECK", intent=intent)
    state_path.write_text(state.model_dump_json(indent=2))
    return state_path


def _prep_extras(state_path) -> dict:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        rc = main(["stage-prep", "architecture", str(state_path)])
    assert rc == 0, buffer.getvalue()
    return json.loads(buffer.getvalue())["extras"]


def _intent() -> IntentSlot:
    return IntentSlot(
        goal="Convert a host-board 5 V supply to 3.3 V with a status LED.",
        obligations=[
            {"kind": "physical", "original_obligation_id": "header",
             "component_class": "pin-header"},
            {"kind": "physical", "original_obligation_id": "status_led",
             "component_class": "led"},
        ],
    )


@pytest.mark.parametrize("exact_part", ["MCP6002-I/SN", "LM358DR", "TLV9062IDR"])
def test_architecture_catalog_choices_construct_four_light_channels(tmp_path, exact_part):
    from kicraft.design.lowering import lower_requirement
    from kicraft.design.architecture_intent import derive_architecture

    extras = _prep_extras(_state(tmp_path, _intent()))
    conditioner = next(
        row for row in extras["circuit_lowerers"]
        if "analog-conditioner" in row["families"]
    )
    selected = next(part for part in conditioner["reviewed_exact_parts"] if part == exact_part)
    choices = conditioner["parameter_choices"]
    channels = next(count for count in choices["channel_count"] if count == 4)
    parameters = {
        "amplifier_exact_part": next(part for part in choices["amplifier_exact_part"] if part == selected),
        "sensor_type": next(kind for kind in choices["sensor_type"] if kind == "ambient-light"),
        "channel_count": channels,
        "analog_rail_v": next(voltage for voltage in choices["analog_rail_v"] if voltage == 5.0),
        "consumer_input_max_v": 3.3,
    }
    architecture = derive_architecture({
        "sheets": [{"name": "SENSORS", "stem": "SENSORS", "role": "analog", "function": "Acquisition"}],
        "requirements": [
            {"id": "source", "sheet": "SENSORS", "role": "power_input",
             "family": "screw-terminal", "parameters": {"rows": 1}, "ties": {"pin2": "GND"}},
            {"id": "acquisition", "sheet": "SENSORS", "role": "analog_block",
             "family": "analog-conditioner", "exact_part": selected,
             "parameters": parameters, "supply": "V5", "ties": {"vminus": "GND"}},
            {"id": "output", "sheet": "SENSORS", "role": "connector",
             "family": "pin-header", "parameters": {"rows": 1, "gender": "female"},
             "ties": {"pin5": "GND", "pin6": "GND"}},
        ],
        "signals": [
            {"name": f"OUT{channel}", "from": f"acquisition.analog_out_{channel}",
             "to": f"output.pin{channel}"}
            for channel in range(1, channels + 1)
        ],
        "power": {"rails": {"V5": {"voltage": 5.0, "from": "source.pin1"}}},
        "mcu_present": False,
    })
    artifact = lower_requirement(next(row for row in architecture.requirements if row.id == "acquisition"))
    amplifier = next(group for group in artifact.groups if group.role == "amplifier")
    assert (amplifier.mpn, amplifier.quantity) == (exact_part, 2)
    assert {
        pin.net for pin in artifact.pins
        if pin.role == "amplifier" and pin.pin in {"1", "7"}
    } == {"OUT1", "OUT2", "OUT3", "OUT4"}


def test_architecture_prep_still_reports_state_and_leaves(tmp_path) -> None:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        assert main(["stage-prep", "architecture", str(_state(tmp_path, _intent()))]) == 0
    payload = json.loads(buffer.getvalue())

    assert payload["state"]["project_stem"] == "REFCHECK"
    assert "leaves_block" in payload["extras"]


def test_an_intent_that_demands_nothing_gets_no_class_block(tmp_path) -> None:
    extras = _prep_extras(_state(tmp_path, IntentSlot(goal="A bare board with a header.")))
    assert "reviewed_class_options" not in extras
    assert "standard_form_factor" not in extras


def test_prep_rejects_an_unknown_stage(tmp_path) -> None:
    with pytest.raises(SystemExit):
        main(["stage-prep", "not-a-stage", str(_state(tmp_path, _intent()))])
