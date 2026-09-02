"""KiCraft: natural language to KiCad schematic file set.

The LLM-authored stages (intent / functional_spec / architecture / bom / wiring)
live in the portable Agent Skill at ``.agents/skills/kicraft/``. This package
ships the schema (``models.ConversationState``), the deterministic synthesis step
(``synthesize.run``), the leaf-library helpers (``library``), and the
``kicraft`` CLI (``cli_app``) that the skill shells out to
for validation and synthesis. The downstream file contract is documented
in ``docs/kicraft_schematic_prompt.md`` at the LLUPS repo root.
"""
from __future__ import annotations
