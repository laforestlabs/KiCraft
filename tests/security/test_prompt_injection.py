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


def test_executor_never_uses_shell_or_interpolates_args():
    """Tool args become argv elements (subprocess list form), never a shell string,
    so brief-controlled args cannot inject a command. Assert no shell=True and that
    every dispatch builds a KICRAFT argv list."""
    src = inspect.getsource(stage_bom_tools.build_bom_executor)
    assert "shell=True" not in src
    assert "runner(command_prefix +" in src
    run_src = inspect.getsource(stage_state_io.run_design_cli)
    assert "shell=True" not in run_src


def test_tool_loop_is_bounded_against_runaway_calls():
    """A brief that goads the model into endless tool calls is bounded: the client
    caps total tool calls and forces a final answer (client.py convergence caps)."""
    import pytest
    pytest.importorskip("requests")
    from kicraft.server import client
    assert client._MAX_TOTAL_TOOL_CALLS <= 32  # a hard ceiling exists
    assert client._MAX_REDUNDANT_TOOL_CALLS >= 1
    # BOM round budget is small and per-attempt (stage_driver), bounding spend.
    assert stage_runtime._BOM_MAX_ROUNDS <= 8


def test_add_part_tool_only_passes_lcsc_id_as_argv(tmp_path):
    """The add_part_from_lcsc dispatch puts the (untrusted) lcsc_id into an argv
    list passed to the CLI, not into a shell -- so an id like 'C1; curl evil' is an
    inert argument string, not a command."""
    src = inspect.getsource(stage_bom_tools.build_bom_executor)
    # the dangerous tool builds an explicit argv list with --from-lcsc
    assert '"add-part", "--from-lcsc"' in src or "'add-part', '--from-lcsc'" in src
