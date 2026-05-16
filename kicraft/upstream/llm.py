"""Anthropic client wrapper with prompt caching and tool-use helpers.

One entry point per shape of LLM call this pipeline makes:

- `call_with_tool(...)` — single forced tool-use call. Returns the parsed
  tool input. Stages use this for structured output (their Pydantic slot)
  and the orchestrator uses it for its action choice.
- `call_text(...)` — plain text generation. Used for narration in expert-
  mode-off.

Both helpers add `cache_control` to the system prompt and conversation
history prefix so subsequent turns hit the prompt cache (the brief calls
this out as a requirement).

Models are pinned to the latest Claude family per the assistant's
knowledge cutoff: opus-4-7 for reasoning-heavy stages, sonnet-4-6 for
the orchestrator's per-turn decision.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Sequence

OPUS_MODEL = "claude-opus-4-7"
SONNET_MODEL = "claude-sonnet-4-6"


class LLMError(RuntimeError):
    """Raised when the LLM call fails (network, schema mismatch, etc.)."""


@dataclass
class ToolResult:
    name: str
    input: dict[str, Any]
    raw_response: Any  # the anthropic Message object, for debugging


def _client():
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMError(
            "ANTHROPIC_API_KEY not set. Export it before running the upstream pipeline."
        )
    return anthropic.Anthropic(api_key=api_key)


def _format_history(history: Sequence[dict[str, str]]) -> list[dict[str, Any]]:
    """Convert {role, content} pairs to anthropic messages."""
    return [{"role": m["role"], "content": m["content"]} for m in history]


def call_with_tool(
    system: str,
    history: Sequence[dict[str, str]],
    tool: dict[str, Any],
    *,
    model: str = OPUS_MODEL,
    max_tokens: int = 4096,
    extra_user_blocks: Sequence[dict[str, Any]] = (),
) -> ToolResult:
    """Force the model to emit one tool use of `tool`. Returns parsed input.

    Args:
        system: System prompt (cached).
        history: Sequence of {role, content} dicts (cached as a prefix).
        tool: A single tool schema dict {name, description, input_schema}.
        model: Model id; default Opus 4.7.
        max_tokens: Output cap.
        extra_user_blocks: Additional content blocks appended to the last
            user message (e.g. a state snapshot as a JSON block).

    Raises:
        LLMError: model didn't call the tool, or returned invalid JSON.
    """
    client = _client()
    messages = _format_history(history)
    if extra_user_blocks and messages and messages[-1]["role"] == "user":
        # Convert the last user content to a block list and append.
        last = messages[-1]
        if isinstance(last["content"], str):
            last["content"] = [{"type": "text", "text": last["content"]}]
        last["content"] = list(last["content"]) + list(extra_user_blocks)

    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        tools=[tool],
        tool_choice={"type": "tool", "name": tool["name"]},
        messages=messages,
    )

    for block in resp.content:
        if block.type == "tool_use" and block.name == tool["name"]:
            tool_input = block.input
            if not isinstance(tool_input, dict):
                raise LLMError(
                    f"tool {tool['name']!r} returned non-dict input: {type(tool_input).__name__}"
                )
            return ToolResult(name=tool["name"], input=tool_input, raw_response=resp)
    raise LLMError(
        f"model did not call required tool {tool['name']!r}; "
        f"response blocks: {[b.type for b in resp.content]}"
    )


def call_text(
    system: str,
    history: Sequence[dict[str, str]],
    *,
    model: str = SONNET_MODEL,
    max_tokens: int = 2048,
) -> str:
    """Plain text completion. Used for natural-prose narration."""
    client = _client()
    resp = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=_format_history(history),
    )
    parts = [b.text for b in resp.content if b.type == "text"]
    return "".join(parts).strip()


# ---------- helpers for building tool schemas from Pydantic models ----------


def pydantic_tool(name: str, description: str, model_cls) -> dict[str, Any]:
    """Build an Anthropic tool schema from a Pydantic v2 model class.

    `model_cls.model_json_schema()` produces a JSON Schema that Anthropic's
    tool-use endpoint accepts as `input_schema`. We strip the top-level
    `title` (Anthropic doesn't use it) and add the model's own docstring as
    description if no other is provided.
    """
    schema = model_cls.model_json_schema()
    schema.pop("title", None)
    return {
        "name": name,
        "description": description or (model_cls.__doc__ or "").strip(),
        "input_schema": schema,
    }


def pretty_state_block(state_dict: dict[str, Any]) -> dict[str, Any]:
    """Wrap a state snapshot for inclusion as an extra user content block."""
    return {
        "type": "text",
        "text": "Current state snapshot (JSON):\n" + json.dumps(state_dict, indent=2),
    }
