"""Stage 1: Intent."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import ConversationState, IntentSlot, Question
from ._runner import run_llm_stage


class IntentStageOutput(BaseModel):
    """Tool schema for the intent stage."""

    intent: IntentSlot
    open_questions: list[Question] = Field(default_factory=list)


def run(state: ConversationState) -> tuple[IntentSlot, list[Question]]:
    """Capture intent from the conversation. No prerequisites."""
    slot, questions = run_llm_stage(
        stage_name="intent",
        state=state,
        output_model=IntentStageOutput,
        slot_field="intent",
    )
    return slot, questions
