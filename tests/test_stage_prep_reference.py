"""The architecture checkpoint must show what the draft is actually offered.

`kicraft stage-prep architecture` printed only the reusable-leaf list while the provider call
received the curated circuit recipes, the generic building blocks and the reviewed carrying parts
for each demanded class from the runtime. A reviewer judging the stage -- and an unattended run
deciding whether to redraft -- read a checkpoint that under-reported the reference data the stage
had. Both now read one builder, so this test pins the two together.
"""
from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from kicraft.design.cli_app import main
from kicraft.design.models import ConversationState, IntentSlot
from kicraft.design.stage_reference import architecture_reference_extras


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


def test_architecture_prep_carries_the_reference_blocks(tmp_path) -> None:
    extras = _prep_extras(_state(tmp_path, _intent()))

    assert extras["circuit_recipes"], "the curated recipes the stage chooses from"
    assert extras["circuit_lowerers"], "the generic building blocks"
    assert "led:" in extras["reviewed_class_options"], "the reviewed carrier for a demanded class"
    # The builders are the runtime's own, so the checkpoint cannot drift from the draft.
    assert extras["circuit_recipes"] == architecture_reference_extras(
        _intent().model_dump()
    )["circuit_recipes"]


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
