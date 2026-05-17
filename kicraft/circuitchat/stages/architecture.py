"""Stage 3: Architecture (topologies + sheet hierarchy + inter-sheet nets)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import Architecture, ConversationState, Question
from ._runner import run_llm_stage
from .functional_spec import StagePrerequisiteError


class ArchitectureStageOutput(BaseModel):
    architecture: Architecture
    open_questions: list[Question] = Field(default_factory=list)


def run(state: ConversationState) -> tuple[Architecture, list[Question]]:
    if state.intent is None or state.functional_spec is None:
        raise StagePrerequisiteError(
            "architecture stage requires intent and functional_spec slots first"
        )
    slot, questions = run_llm_stage(
        stage_name="architecture",
        state=state,
        output_model=ArchitectureStageOutput,
        slot_field="architecture",
    )
    return slot, questions
