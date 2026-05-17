"""Per-turn orchestrator for the CircuitChat pipeline.

One LLM call per user turn picks exactly one of three actions:

- `run_stage` — invoke a stage to update its slot.
- `ask` — surface a batch of clarifying questions.
- `respond` — natural-language reply.

The orchestrator is implemented as a forced tool-choice between three
tools. It is intentionally stateless: each turn re-derives its decision
from the conversation history and the current state snapshot.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from importlib import resources
from typing import Any

from .llm import SONNET_MODEL, ToolResult, call_with_tool, pretty_state_block
from .models import ChatMsg, ConversationState, Question
from .stages import STAGES, intent as stage_intent
from .stages._runner import _load_prompt
from .stages.functional_spec import StagePrerequisiteError


logger = logging.getLogger(__name__)


# ---------- tool schemas ----------


_STAGE_NAMES = list(STAGES.keys()) + ["synthesis"]


_RUN_STAGE_TOOL = {
    "name": "run_stage",
    "description": "Invoke a pipeline stage to produce or refresh its state slot.",
    "input_schema": {
        "type": "object",
        "properties": {
            "stage": {
                "type": "string",
                "enum": _STAGE_NAMES,
                "description": "Which stage to run.",
            },
            "rationale": {
                "type": "string",
                "description": "One-sentence reason for running this stage now.",
            },
        },
        "required": ["stage", "rationale"],
    },
}


_ASK_TOOL = {
    "name": "ask",
    "description": (
        "Surface clarifying questions to the user. Use when you need more "
        "information before a stage can run, or when stages emitted blocking "
        "or material open_questions you have not yet surfaced."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "preface": {
                "type": "string",
                "description": "Short prefatory sentence before the question list.",
            },
            "questions": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 5,
            },
        },
        "required": ["questions"],
    },
}


_RESPOND_TOOL = {
    "name": "respond",
    "description": (
        "Reply to the user in natural prose without running a stage. Use for "
        "chit-chat, summaries, confirmations, and stage-completion proposals."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
        },
        "required": ["text"],
    },
}


_TOOL_BY_NAME = {t["name"]: t for t in (_RUN_STAGE_TOOL, _ASK_TOOL, _RESPOND_TOOL)}


# ---------- decision ----------


@dataclass
class OrchestratorDecision:
    action: str  # "run_stage" | "ask" | "respond"
    payload: dict[str, Any]


def _decide(state: ConversationState) -> OrchestratorDecision:
    """One LLM call that picks among the three orchestrator tools."""
    system = _load_prompt("orchestrator")
    history = [{"role": m.role, "content": m.content} for m in state.history]
    state_dict = state.model_dump(mode="json")
    state_dict.pop("history", None)

    # We need an "any of these tools" forced choice. Anthropic supports
    # `tool_choice={"type": "any"}` to force SOME tool. We then route on
    # the returned name.
    from .llm import _client  # local import; keeps tests cheap

    client = _client()
    resp = client.messages.create(
        model=SONNET_MODEL,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=list(_TOOL_BY_NAME.values()),
        tool_choice={"type": "any"},
        messages=history
        + [
            {
                "role": "user",
                "content": [pretty_state_block(state_dict)],
            }
        ]
        if history
        else [
            {
                "role": "user",
                "content": [pretty_state_block(state_dict)],
            }
        ],
    )
    for block in resp.content:
        if block.type == "tool_use" and block.name in _TOOL_BY_NAME:
            payload = block.input if isinstance(block.input, dict) else {}
            return OrchestratorDecision(action=block.name, payload=payload)
    # Fall back to a respond-with-empty if Anthropic returns text only.
    text = "".join(b.text for b in resp.content if b.type == "text").strip()
    return OrchestratorDecision(action="respond", payload={"text": text or "(no response)"})


# ---------- per-turn entry point ----------


def run_turn(state: ConversationState, user_message: str) -> ConversationState:
    """Process one user turn. Returns the updated state.

    Mutates `state` in place (also returns it for fluent use).
    """
    state.history.append(ChatMsg(role="user", content=user_message))
    decision = _decide(state)
    logger.info("orchestrator decision: %s", decision.action)

    if decision.action == "run_stage":
        stage_name = decision.payload.get("stage", "")
        _execute_stage(state, stage_name)
    elif decision.action == "ask":
        preface = decision.payload.get("preface") or "A few questions before I proceed:"
        questions = decision.payload.get("questions") or []
        body = preface + "\n" + "\n".join(f"- {q}" for q in questions)
        state.history.append(ChatMsg(role="assistant", content=body))
    else:  # respond
        text = decision.payload.get("text", "")
        state.history.append(ChatMsg(role="assistant", content=text))

    return state


def _execute_stage(state: ConversationState, stage_name: str) -> None:
    """Run a single stage and merge its output into `state`."""
    if stage_name == "synthesis":
        # Synthesis is mechanical and writes to disk; orchestrator surfaces
        # the result as a text message. Caller wires in the target dir.
        state.history.append(
            ChatMsg(
                role="assistant",
                content=(
                    "Synthesis is a write-to-disk action. "
                    "Click the 'Synthesize' button or run "
                    "`kicraft-new --synthesize` to emit the file set."
                ),
            )
        )
        return

    stage_mod = STAGES.get(stage_name)
    if stage_mod is None:
        state.history.append(
            ChatMsg(role="assistant", content=f"Unknown stage: {stage_name!r}")
        )
        return

    try:
        slot, questions = stage_mod.run(state)
    except StagePrerequisiteError as e:
        state.history.append(ChatMsg(role="assistant", content=str(e)))
        return

    # Merge slot into state.
    setattr(state, _slot_attr_for(stage_name), slot)
    state.replace_open_questions_for_stage(stage_name, questions)

    # If the slot defines a project_stem (Intent does, implicitly via a
    # custom rule we apply below), propagate it.
    if stage_name == "intent" and state.project_stem is None:
        state.project_stem = _derive_project_stem(slot)

    state.history.append(
        ChatMsg(
            role="assistant",
            content=f"Ran stage `{stage_name}`. State slot `{stage_name}` updated.",
        )
    )


_SLOT_ATTR = {
    "intent": "intent",
    "functional_spec": "functional_spec",
    "architecture": "architecture",
    "bom": "bom",
}


def _slot_attr_for(stage_name: str) -> str:
    return _SLOT_ATTR[stage_name]


def _derive_project_stem(intent_slot) -> str:
    """Heuristic: derive a project stem from the intent goal.

    Picks the first 3 significant words of the goal, joined and uppercased.
    The synthesis stage requires a non-None project_stem; the orchestrator
    sets one as soon as Intent runs so the user can override later if they
    care.
    """
    import re

    words = re.findall(r"[A-Za-z0-9]+", intent_slot.goal)
    significant = [w for w in words if len(w) >= 3][:3]
    stem = "_".join(w.upper() for w in significant) or "PROJECT"
    return stem[:32]
