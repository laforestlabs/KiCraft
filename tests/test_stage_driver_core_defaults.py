"""Core-components registry consumption in the web pipeline.

The registry rows (curated default part per functional block, synced into
the DB from the repo catalog) are threaded web -> run_session -> drive_chain
-> drive_stage and rendered into the architecture/bom prompts as
extras.core_defaults_block; the stage specs and the BOM tool-docs tell the
model how to adopt a matching default: bundle-backed rows come straight from
the parts library (no fetch), LCSC-only rows via one add_part_from_lcsc
call. Offline: real stage-prep subprocesses, fake LLM client, no network.
"""
from __future__ import annotations

from kicraft.parts_library.core_blocks import load_core_catalog, resolve_block
from kicraft.server.session import run_session
from kicraft.server.stage_driver import (
    _format_core_defaults_block,
    build_system,
    drive_stage,
)


def _catalog_rows() -> list[dict]:
    """Catalog blocks flattened to the DB row shape the formatter receives."""
    rows = []
    for block in load_core_catalog().blocks:
        row = resolve_block(block)
        row["enabled"] = True
        rows.append(row)
    return rows


# ---- formatter ---------------------------------------------------------------

def test_block_renders_catalog_rows_compactly():
    block = _format_core_defaults_block(_catalog_rows())
    assert block is not None
    assert "ldo-3v3-500ma" in block and "C82942" in block
    assert "usb-uart-bridge" in block and "CH340C" in block
    # series rows render '-' LCSC and bundle cells, not 'None'
    assert "| UNI-ROYAL 0603WAF series | - | 0603 | - |" in block
    assert "None" not in block
    # bundle-backed rows carry the bundle name + the no-fetch adoption rule
    assert "| ams1117-3v3 |" in block
    assert "ALREADY in the parts library" in block
    assert "Do NOT call add_part_from_lcsc" in block
    # notes and snapshot data are deliberately excluded (token budget)
    assert "Runner-up" not in block and "snapshot" not in block
    assert "536893" not in block  # a stock figure
    assert "0.0506" not in block  # a price figure
    # WLP magnetometer gets the package caveat footer
    assert "Package caveats" in block and "MMC5603NJ" in block
    assert len(block) < 7500


def test_block_filters_disabled_and_empties_to_none():
    assert _format_core_defaults_block([]) is None
    assert _format_core_defaults_block(None) is None
    rows = [{"function_key": "x-block", "display_name": "X", "qualifier": None,
             "default_mpn": "M1", "default_lcsc": "C1", "package": "SOIC-8",
             "enabled": False}]
    assert _format_core_defaults_block(rows) is None
    rows[0]["enabled"] = True
    assert "x-block" in _format_core_defaults_block(rows)


# ---- system prompts carry the adoption rule ------------------------------------

def test_bom_system_prompt_mentions_core_defaults():
    low = build_system("bom").lower()
    assert "core_defaults_block" in low
    assert "add_part_from_lcsc" in low


def test_architecture_system_prompt_mentions_core_defaults():
    low = build_system("architecture").lower()
    assert "core_defaults_block" in low
    assert "per core defaults" in low  # the assumptions-naming example


# ---- drive_stage injection ------------------------------------------------------

class _RecordingClient:
    """Returns garbage (commit fails, which is fine: the assertions are on the
    prompt that was sent), recording every messages list it receives."""

    def __init__(self):
        self.calls: list[list[dict]] = []

    def chat(self, messages, max_tokens=4096, progress=None, meta_ctx=None):
        self.calls.append([dict(m) for m in messages])
        return {"text": "not json", "cost_usd": 0.0, "reasoning": "",
                "finish_reason": "stop"}

    def chat_with_tools(self, messages, tools, executor, max_tokens=4096,
                        max_rounds=6, progress=None, meta_ctx=None):
        self.calls.append([dict(m) for m in messages])
        return {"text": "not json", "cost_usd": 0.0, "rounds": 1,
                "tool_calls": 0, "finish_reason": "stop"}


def _user_prompt(client: _RecordingClient) -> str:
    assert client.calls, "the fake client was never called"
    return client.calls[0][1]["content"]


def _drive(tmp_path, stage, core_defaults):
    (tmp_path / ".kicraft").mkdir(parents=True, exist_ok=True)
    state_path = tmp_path / ".kicraft" / "state.json"
    state_path.write_text("{}", encoding="utf-8")
    client = _RecordingClient()
    r = drive_stage(client, stage, "a USB-powered LED", state_path, tmp_path,
                    max_retries=0, core_defaults=core_defaults)
    assert r["commit_ok"] is False  # garbage reply; prompt is what we test
    return client


def test_drive_stage_injects_db_rows_overriding_nothing_else(tmp_path):
    edited = [{"function_key": "usb-uart-bridge", "display_name": "USB-UART bridge",
               "qualifier": "USB 2.0 FS", "default_mpn": "FT232RL-EDITED",
               "default_lcsc": "C52717", "package": "SSOP-28", "enabled": True}]
    client = _drive(tmp_path, "architecture", edited)
    prompt = _user_prompt(client)
    assert "core_defaults_block" in prompt
    assert "FT232RL-EDITED" in prompt          # the DB-edited row is what renders
    assert "ME6211C33M5G-N" not in prompt      # not the bundled seed


def test_drive_stage_bom_gets_the_block_too(tmp_path):
    client = _drive(tmp_path, "bom", _catalog_rows())
    prompt = _user_prompt(client)
    assert "core_defaults_block" in prompt and "ldo-3v3-500ma" in prompt


def test_drive_stage_without_rows_has_no_block(tmp_path):
    for empty in (None, []):
        client = _drive(tmp_path, "architecture", empty)
        assert "core_defaults_block" not in _user_prompt(client)


def test_drive_stage_intent_never_gets_the_block(tmp_path):
    client = _drive(tmp_path, "intent", _catalog_rows())
    assert "core_defaults_block" not in _user_prompt(client)


# ---- the Settings kill switch ----------------------------------------------------

def test_settings_flag_defaults_on_and_env_disables(monkeypatch):
    from kicraft.server.config import Settings
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.delenv("KICRAFT_CORE_DEFAULTS", raising=False)
    s = Settings.from_env(dotenv=False)
    assert s.enable_core_defaults is True
    assert "enable_core_defaults" in s.redacted()
    monkeypatch.setenv("KICRAFT_CORE_DEFAULTS", "0")
    assert Settings.from_env(dotenv=False).enable_core_defaults is False


# ---- run_session forwards the rows ----------------------------------------------

def test_run_session_threads_core_defaults(tmp_path):
    client = _RecordingClient()
    client.guard = type("G", (), {"status": lambda self: {}})()
    res = run_session(tmp_path, "a USB-powered LED", ["architecture"],
                      client=client, core_defaults=_catalog_rows())
    assert res["status"] == "failed"  # garbage replies; threading is what we test
    assert "ldo-3v3-500ma" in _user_prompt(client)
