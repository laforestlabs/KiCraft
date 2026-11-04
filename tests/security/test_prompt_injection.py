"""Prompt-injection blast-radius: the brief is untrusted user input fed to an LLM
that has BOM tools. A malicious brief telling the model to call tools maliciously
must not reach beyond the fixed, whitelisted KiCraft CLI subcommands -- it cannot
read the environment, run arbitrary shell, or fetch arbitrary parts forever.
"""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("pydantic")
from kicraft.server import stage_bom_tools, stage_runtime, stage_state_io  # noqa: E402


def _executor(workspace):
    return stage_bom_tools.build_bom_executor(
        workspace, stage_state_io.run_design_cli, stage_state_io.KICRAFT
    )


def test_bom_executor_only_dispatches_whitelisted_tools(tmp_path):
    """The executor maps a fixed set of tool NAMES to fixed CLI subcommands; an
    unknown/injected tool name returns an error string, never executes."""
    execute = _executor(tmp_path)
    assert execute("definitely_not_a_tool", {}) == "unknown tool: definitely_not_a_tool"
    # an injected attempt to smuggle a shell command as a tool name is just unknown
    assert execute("list_parts; rm -rf /", {}).startswith("unknown tool:")









