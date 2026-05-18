"""KiCraft CircuitChat: natural language to KiCad schematic file set.

The LLM-driven stages (intent / functional_spec / architecture / bom) and
the per-turn orchestrator now live in the Claude Code skill at
``.claude/skills/circuitchat/``. This package ships the schema
(``models.ConversationState``), the deterministic synthesis step
(``synthesize.run``), the leaf-library helpers (``library``), and the
``kicraft-circuitchat`` CLI (``cli_app``) that the skill shells out to
for validation and synthesis. The downstream file contract is documented
in ``docs/circuitchat_schematic_prompt.md`` at the LLUPS repo root.
"""
from __future__ import annotations
