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

import pytest

from kicraft.parts_library.core_blocks import load_core_catalog, resolve_block
from kicraft.server.session import run_session
from kicraft.server.stage_driver import (
    _format_core_defaults_block,
    build_system,
    drive_stage,
)


@pytest.fixture(autouse=True)
def _no_local_jlc_catalog(tmp_path_factory, monkeypatch):
    """Point the offline JLC catalog at a missing file: the formatter now
    drops rows that are dry in the catalog, and these tests must not depend
    on whichever real dump the host happens to have installed (fail-open =
    unfiltered). The filtering itself is tested below with a fixture DB."""
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB",
                       str(tmp_path_factory.mktemp("jlc") / "absent.sqlite3"))


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


def _mk_catalog(tmp_path, monkeypatch, rows):
    import sqlite3
    db = tmp_path / "jlc.sqlite3"
    con = sqlite3.connect(db)
    con.execute("""CREATE TABLE jlc_components (
        lcsc INTEGER PRIMARY KEY, mfr TEXT, package TEXT, manufacturer TEXT,
        library_type TEXT, stock INTEGER, price TEXT, description TEXT,
        joints INTEGER)""")
    con.executemany("INSERT INTO jlc_components (lcsc, mfr, package, manufacturer, library_type, stock, price, description) VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB", str(db))


def _row(key, mpn, lcsc):
    return {"function_key": key, "display_name": key, "qualifier": None,
            "default_mpn": mpn, "default_lcsc": lcsc, "package": "SOIC-8",
            "enabled": True}


def test_block_drops_rows_dry_in_the_catalog_with_a_caveat(tmp_path, monkeypatch):
    # A core default the current dump shows draining (or pruned out entirely)
    # must not be offered to the model as a no-research adoption.
    _mk_catalog(tmp_path, monkeypatch, [
        (100, "GOOD-IC", "SOIC-8", "x", "base", 500_000, "1-:0.1", "good"),
        (200, "DRY-IC", "SOIC-8", "x", "expand", 12, "1-:0.1", "draining"),
    ])
    block = _format_core_defaults_block([
        _row("good-block", "GOOD-IC", "C100"),
        _row("dry-block", "DRY-IC", "C200"),
        _row("gone-block", "GONE-IC", "C300"),  # pruned out of the catalog
    ])
    assert "good-block" in block
    assert "| dry-block |" not in block and "| gone-block |" not in block
    assert "2 core default(s) omitted" in block
    assert "DRY-IC" in block and "GONE-IC" in block  # named in the caveat


def test_block_keeps_lcsc_less_series_rows_when_filtering(tmp_path, monkeypatch):
    # Passive-series rows carry no C-number: nothing to stock-check.
    _mk_catalog(tmp_path, monkeypatch, [
        (100, "GOOD-IC", "SOIC-8", "x", "base", 500_000, "1-:0.1", "good"),
    ])
    row = _row("passive-series", "UNI-ROYAL 0603WAF series", None)
    block = _format_core_defaults_block([_row("good-block", "GOOD-IC", "C100"),
                                         row])
    assert "passive-series" in block and "omitted" not in block


def test_block_unfiltered_when_catalog_absent(tmp_path, monkeypatch):
    # Fail open: with no dump installed the block renders as before.
    monkeypatch.setenv("KICRAFT_JLCPARTS_DB", str(tmp_path / "absent.sqlite3"))
    block = _format_core_defaults_block([_row("x-block", "M1", "C1")])
    assert "x-block" in block and "omitted" not in block


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

    def chat(self, messages, max_tokens=4096, temperature=0.2, progress=None, meta_ctx=None,
             reasoning=None):
        self.calls.append([dict(m) for m in messages])
        return {"text": "not json", "cost_usd": 0.0, "reasoning": "",
                "finish_reason": "stop"}

    def chat_with_tools(self, messages, tools, executor, max_tokens=4096,
                        temperature=0.2, max_rounds=6, progress=None, meta_ctx=None,
                        reasoning=None):
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


def test_bundle_row_dropped_when_cached_retail_dry(monkeypatch):
    # 2026-07-19 review §5.2: bundle rows were invisible to the dry filter
    # (their C# lives in the vendored manifest) while the prompt tells the
    # model NOT to re-verify them -- a retail-dry bundle default (drv8833
    # C50506: 3,299 assembly / 0 retail) was a guaranteed §9.26 bounce.
    from kicraft.server import stage_driver as sd

    monkeypatch.setattr(sd.jlcparts, "available", lambda: True)
    monkeypatch.setattr(
        sd.jlcparts, "lookup", lambda cid: {"stock": 3299}
    )
    monkeypatch.setattr(sd, "_bundle_sourcing_lcsc", lambda b: "C50506")
    monkeypatch.setattr(sd.lcsc_retail, "enabled", lambda: True)
    monkeypatch.setattr(sd.lcsc_retail, "retail_floor", lambda: 5)
    monkeypatch.setattr(
        sd.lcsc_retail,
        "cached_stock",
        lambda cid: 0 if cid == "C50506" else None,
    )
    rows = [
        {"function_key": "dc-motor-driver", "default_mpn": "DRV8833PWPR",
         "bundle": "drv8833", "enabled": True},
        {"function_key": "usb-uart-bridge", "default_mpn": "CH340C",
         "default_lcsc": "C84681", "enabled": True},
    ]
    out = sd._format_core_defaults_block(rows)
    assert out is not None
    table_rows = [ln for ln in out.splitlines() if ln.startswith("|")]
    assert not any("dc-motor-driver" in ln for ln in table_rows)
    assert any("usb-uart-bridge" in ln for ln in table_rows)
    assert "retail-dry" in out  # named in the dropped-rows caveat line


def test_bundle_row_kept_without_fresh_retail_reading(monkeypatch):
    # No fresh cached reading -> no drop (offline path must not guess).
    from kicraft.server import stage_driver as sd

    monkeypatch.setattr(sd.jlcparts, "available", lambda: True)
    monkeypatch.setattr(sd.jlcparts, "lookup", lambda cid: {"stock": 3299})
    monkeypatch.setattr(sd, "_bundle_sourcing_lcsc", lambda b: "C50506")
    monkeypatch.setattr(sd.lcsc_retail, "enabled", lambda: True)
    monkeypatch.setattr(sd.lcsc_retail, "cached_stock", lambda cid: None)
    rows = [{"function_key": "dc-motor-driver", "default_mpn": "DRV8833PWPR",
             "bundle": "drv8833", "enabled": True}]
    out = sd._format_core_defaults_block(rows)
    assert out is not None and "dc-motor-driver" in out
