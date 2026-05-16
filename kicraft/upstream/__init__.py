"""KiCraft upstream pipeline: natural language to KiCad schematic file set.

Five-stage pipeline (4 LLM-driven, 1 mechanical) plus an orchestrator that
turns a multi-turn chat into the file set that the downstream KiCraft layout
and routing pipeline ingests. The downstream file contract is documented in
`docs/upstream_schematic_prompt.md` at the LLUPS repo root.
"""
from __future__ import annotations
