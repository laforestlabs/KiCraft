"""Tests for the LLM-facing scaffolding that don't make real API calls.

Anything that requires `ANTHROPIC_API_KEY` is skipped automatically.
"""
from __future__ import annotations

import os

import pytest

from kicraft.upstream.llm import LLMError, pydantic_tool
from kicraft.upstream.models import (
    Architecture,
    BOM,
    BomPart,
    ConversationState,
    FunctionalSpec,
    IntentSlot,
    InterSheetNet,
    Question,
    Sheet,
    SheetPin,
)
from kicraft.upstream.orchestrator import (
    _ASK_TOOL,
    _RESPOND_TOOL,
    _RUN_STAGE_TOOL,
    _derive_project_stem,
    _slot_attr_for,
)
from kicraft.upstream.stages._runner import _load_prompt


def test_orchestrator_tool_schemas_are_well_formed() -> None:
    for tool in (_RUN_STAGE_TOOL, _ASK_TOOL, _RESPOND_TOOL):
        assert "name" in tool
        assert "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]


def test_run_stage_enum_lists_all_stages() -> None:
    stages = _RUN_STAGE_TOOL["input_schema"]["properties"]["stage"]["enum"]
    assert {"intent", "functional_spec", "architecture", "bom", "synthesis"} == set(stages)


def test_all_prompts_exist_and_nonempty() -> None:
    for name in ("intent", "functional_spec", "architecture", "bom", "orchestrator"):
        text = _load_prompt(name)
        assert len(text) > 200, f"prompt {name} is suspiciously short"


def test_pydantic_tool_builds_valid_schema() -> None:
    tool = pydantic_tool("emit_intent", "test", IntentSlot)
    assert tool["name"] == "emit_intent"
    assert tool["input_schema"]["type"] == "object"
    # Field must surface in the schema.
    assert "goal" in tool["input_schema"]["properties"]


def test_slot_attr_mapping_is_complete() -> None:
    for stage in ("intent", "functional_spec", "architecture", "bom"):
        attr = _slot_attr_for(stage)
        # Every mapped slot must exist on ConversationState.
        assert hasattr(ConversationState(), attr)


@pytest.mark.parametrize(
    "goal,expected_prefix",
    [
        ("USB-powered 3.3V regulator demo", "USB"),
        ("AC line-powered LED driver", "LINE"),
        ("hi", "PROJECT"),
    ],
)
def test_project_stem_derivation(goal: str, expected_prefix: str) -> None:
    stem = _derive_project_stem(IntentSlot(goal=goal))
    assert stem.startswith(expected_prefix)


def test_llm_client_raises_without_api_key(monkeypatch) -> None:
    from kicraft.upstream.llm import _client

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LLMError):
        _client()


@pytest.mark.skipif(
    not os.environ.get("RUN_LIVE_LLM_TESTS"),
    reason="Live LLM tests disabled (set RUN_LIVE_LLM_TESTS=1 to enable)",
)
def test_live_intent_round_trip() -> None:
    """Smoke: real Anthropic call returns a valid IntentSlot."""
    from kicraft.upstream.models import ChatMsg
    from kicraft.upstream.stages.intent import run as run_intent

    state = ConversationState(
        history=[
            ChatMsg(
                role="user",
                content=(
                    "I want a USB-C powered board with a 3.3V LDO and a status LED. "
                    "Target fab is JLCPCB. Budget under $5 BOM."
                ),
            )
        ]
    )
    slot, questions = run_intent(state)
    assert slot.goal
    for q in questions:
        assert q.stage == "intent"


# ---------- CLI synthesize via saved state ----------


def test_cli_synthesize_from_saved_state(tmp_path) -> None:
    """CLI `--synthesize` reads a saved state JSON and writes the file set."""
    from kicraft.upstream.cli import main

    # Build a complete state and save it.
    state = ConversationState(
        project_stem="CLI_TEST",
        intent=IntentSlot(goal="cli test demo"),
        functional_spec=FunctionalSpec(
            blocks=[]
        ),
        architecture=Architecture(
            sheets=[Sheet(name="REG", stem="REG", function="ldo")],
            power_nets=["VBUS", "+3V3", "GND"],
            inter_sheet_nets=[],
        ),
        bom=BOM(
            parts=[
                BomPart(
                    ref="R1",
                    value="10k",
                    symbol="Device:R",
                    footprint="Resistor_SMD:R_0402_1005Metric",
                    sheet="REG",
                )
            ]
        ),
    )
    state_path = tmp_path / "state.json"
    state_path.write_text(state.model_dump_json())

    out_dir = tmp_path / "out"
    rc = main(["--load", str(state_path), "--synthesize", str(out_dir)])
    assert rc == 0, "CLI synthesize should succeed"
    assert (out_dir / "CLI_TEST.kicad_sch").is_file()
    assert (out_dir / "REG.kicad_sch").is_file()
    assert (out_dir / "CLI_TEST.kicad_pro").is_file()
    assert (out_dir / "CLI_TEST_autoplacer.json").is_file()
