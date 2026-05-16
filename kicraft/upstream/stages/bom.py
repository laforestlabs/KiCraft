"""Stage 4: BOM (real parts, refs, footprints, sheet assignments)."""
from __future__ import annotations

from pydantic import BaseModel, Field

from ..models import BOM, ConversationState, Question
from ._runner import run_llm_stage
from .functional_spec import StagePrerequisiteError


class BOMStageOutput(BaseModel):
    bom: BOM
    open_questions: list[Question] = Field(default_factory=list)


def run(state: ConversationState) -> tuple[BOM, list[Question]]:
    if (
        state.intent is None
        or state.functional_spec is None
        or state.architecture is None
    ):
        raise StagePrerequisiteError(
            "bom stage requires intent, functional_spec, and architecture slots first"
        )
    slot, questions = run_llm_stage(
        stage_name="bom",
        state=state,
        output_model=BOMStageOutput,
        slot_field="bom",
    )
    return slot, questions
