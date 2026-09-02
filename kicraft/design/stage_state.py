"""Pure helpers for invalidating stale design-stage state."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Any

DESIGN_STAGES = ("intent", "functional_spec", "architecture", "bom", "wiring")


def downstream_stages(stage: str) -> tuple[str, ...]:
    """Return the canonical design stages strictly after ``stage``."""
    if stage not in DESIGN_STAGES:
        return ()
    return DESIGN_STAGES[DESIGN_STAGES.index(stage) + 1 :]


def invalidate_downstream(
    state: MutableMapping[str, Any], stage: str
) -> tuple[str, ...]:
    """Remove every state field made stale by accepting ``stage``."""
    invalidated = downstream_stages(stage)
    for downstream in invalidated:
        if downstream == "wiring":
            bom = state.get("bom")
            if isinstance(bom, MutableMapping):
                bom["connections"] = []
                bom["no_connect_pins"] = []
        else:
            state[downstream] = None

        statuses = state.get("stage_status")
        if isinstance(statuses, MutableMapping):
            statuses.pop(downstream, None)

    state["open_questions"] = [
        question
        for question in (state.get("open_questions") or [])
        if not isinstance(question, MutableMapping)
        or question.get("stage") not in invalidated
    ]
    return invalidated
