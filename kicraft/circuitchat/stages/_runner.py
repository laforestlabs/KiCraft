"""Shared helper that runs one LLM-driven stage.

Each stage module thinly wraps this with its own slot model, prompt file
name, and required-prerequisite check. Keeping the LLM call site in one
place means prompt-caching headers and model selection only need to be
right once.
"""
from __future__ import annotations

from importlib import resources
from typing import Type, TypeVar

from pydantic import BaseModel

from ..llm import OPUS_MODEL, ToolResult, call_with_tool, pretty_state_block, pydantic_tool
from ..models import ConversationState, Question


SlotT = TypeVar("SlotT", bound=BaseModel)


def _load_prompt(name: str) -> str:
    return resources.files("kicraft.circuitchat.prompts").joinpath(f"{name}.md").read_text()


class _StageOutputBase(BaseModel):
    """Tool-call response wrapper. Concrete subclasses add the typed slot field."""

    open_questions: list[Question] = []


def run_llm_stage(
    *,
    stage_name: str,
    state: ConversationState,
    output_model: Type[BaseModel],
    slot_field: str,
    model: str = OPUS_MODEL,
) -> tuple[BaseModel, list[Question]]:
    """Drive one LLM-driven stage and return (slot_value, [Question]).

    Args:
        stage_name: stage identifier used both for the prompt file name and
            for stamping questions with `Question.stage`.
        state: full ConversationState; the LLM sees a JSON snapshot.
        output_model: a Pydantic class with two fields — `<slot_field>` and
            `open_questions: list[Question]`.
        slot_field: name of the slot attribute on output_model.
        model: which Claude to use.
    """
    system = _load_prompt(stage_name)
    history = [{"role": m.role, "content": m.content} for m in state.history]
    state_dict = state.model_dump(mode="json")
    # Drop history from the snapshot — the model already sees it as messages.
    state_dict.pop("history", None)
    tool = pydantic_tool(
        name=f"emit_{stage_name}",
        description=f"Emit the {stage_name} stage output for the current state.",
        model_cls=output_model,
    )
    result: ToolResult = call_with_tool(
        system=system,
        history=history,
        tool=tool,
        model=model,
        extra_user_blocks=[pretty_state_block(state_dict)],
    )
    parsed = output_model.model_validate(result.input)
    slot = getattr(parsed, slot_field)
    questions = list(getattr(parsed, "open_questions", []))
    # Stamp questions with the stage that emitted them so the orchestrator
    # and `ConversationState.replace_open_questions_for_stage` can route
    # them correctly.
    for q in questions:
        q.stage = stage_name
    return slot, questions
