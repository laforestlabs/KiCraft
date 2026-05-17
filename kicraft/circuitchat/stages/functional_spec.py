"""Stage 2: Functional spec."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import ConversationState, FunctionalSpec, Question
from ._runner import run_llm_stage


class StagePrerequisiteError(RuntimeError):
    pass


class FunctionalSpecStageOutput(BaseModel):
    functional_spec: FunctionalSpec
    open_questions: list[Question] = Field(default_factory=list)


def run(state: ConversationState) -> tuple[FunctionalSpec, list[Question]]:
    if state.intent is None:
        raise StagePrerequisiteError(
            "functional_spec stage requires intent slot to be populated first"
        )
    slot, questions = run_llm_stage(
        stage_name="functional_spec",
        state=state,
        output_model=FunctionalSpecStageOutput,
        slot_field="functional_spec",
    )
    return slot, questions
