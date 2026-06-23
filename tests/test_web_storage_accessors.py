"""Storage-layout accessors + reader parity (refactor roadmap Phase 4a, step 1).

`_kicraft_dir` / `_state_path` (storage.py) let every state/check/price reader take
EITHER root layout unchanged:

  - workspace  ->  <root>/.kicraft/...   (dotted; scratch under KICRAFT_WORK_DIR)
  - durable    ->  <root>/kicraft/...    (no dot; the saved project tree)
  - legacy     ->  <root>/state.json     (top-level; projects predating kicraft/)

This is the no-behavior-change groundwork for view-from-durable: today every reader
is called with a workspace root, so `_kicraft_dir` returns `.kicraft` and behavior is
identical; these tests pin that the SAME readers now also resolve the durable and
legacy roots to the same answers. Pure-data: no NiceGUI, no pcbnew, no LLM.

See docs/plans/view-from-durable-refactor-v2.md "Complete reader inventory".
"""
from __future__ import annotations

import json

from kicraft.server import web
from kicraft.server.session import read_state
from kicraft.server.storage import _kicraft_dir, _read_project_stem, _state_path

# --- fixtures: lay one project's metadata down under each of the three layouts ---

_STATE = {"project_stem": "USB_PD_TRIGGER", "intent": {"x": 1}, "bom": {"connections": [1]}}
_SYNTH_CHECK = {
    "status": "fail",
    # _erc_offenders filters on "ERC" (uppercase) being in the check name.
    "checks": [
        {"name": "§9.12 ERC", "ok": False, "offenders": ["U1.3 dangling", "R2.1 no-net"]},
        {"name": "§9.x DRC", "ok": True, "offenders": []},
    ],
}
_PRICES = {"_schema": web._PRICE_SCHEMA, "prices": {"C123": {"unit": 0.5}}}


def _write_meta(metadir, *, state=True, check=True, prices=False):
    metadir.mkdir(parents=True, exist_ok=True)
    if state:
        (metadir / "state.json").write_text(json.dumps(_STATE), encoding="utf-8")
    if check:
        (metadir / "synthesis_check.json").write_text(json.dumps(_SYNTH_CHECK), encoding="utf-8")
    if prices:
        (metadir / web._PRICE_FILE).write_text(json.dumps(_PRICES), encoding="utf-8")


def _workspace_root(tmp_path, **kw):
    root = tmp_path / "ws"
    _write_meta(root / ".kicraft", **kw)
    return root


def _durable_root(tmp_path, **kw):
    root = tmp_path / "durable"
    _write_meta(root / "kicraft", **kw)
    return root


def _legacy_root(tmp_path):
    """Pre-kicraft/ project: only a top-level state.json, no metadata dir."""
    root = tmp_path / "legacy"
    root.mkdir(parents=True, exist_ok=True)
    (root / "state.json").write_text(json.dumps(_STATE), encoding="utf-8")
    return root


# --- _kicraft_dir / _state_path resolution --------------------------------------

def test_kicraft_dir_prefers_dotted_then_bare_then_defaults(tmp_path):
    dotted = tmp_path / "a"; (dotted / ".kicraft").mkdir(parents=True)
    assert _kicraft_dir(dotted) == dotted / ".kicraft"

    bare = tmp_path / "b"; (bare / "kicraft").mkdir(parents=True)
    assert _kicraft_dir(bare) == bare / "kicraft"

    both = tmp_path / "c"
    (both / ".kicraft").mkdir(parents=True); (both / "kicraft").mkdir(parents=True)
    assert _kicraft_dir(both) == both / ".kicraft"  # dotted wins (checked first)

    neither = tmp_path / "d"; neither.mkdir()
    assert _kicraft_dir(neither) == neither / "kicraft"  # durable name for new paths


def test_state_path_resolves_all_three_layouts_then_falls_through(tmp_path):
    assert _state_path(_workspace_root(tmp_path)) == tmp_path / "ws" / ".kicraft" / "state.json"
    assert _state_path(_durable_root(tmp_path)) == tmp_path / "durable" / "kicraft" / "state.json"
    assert _state_path(_legacy_root(tmp_path)) == tmp_path / "legacy" / "state.json"

    empty = tmp_path / "empty"; empty.mkdir()
    # No state under any layout: falls through to the legacy top-level path (which
    # is also absent), so callers .read() -> miss -> {} / None, same as before.
    assert _state_path(empty) == empty / "state.json"
    assert not _state_path(empty).is_file()


# --- reader parity: identical answer regardless of which root layout it's given ---

def test_read_state_parity_across_layouts(tmp_path):
    expect = _STATE
    assert read_state(_workspace_root(tmp_path)) == expect
    assert read_state(_durable_root(tmp_path)) == expect
    assert read_state(_legacy_root(tmp_path)) == expect
    assert read_state(tmp_path / "nope") == {}  # absent -> {}


def test_web_read_state_json_parity_across_layouts(tmp_path):
    assert web._read_state_json(_workspace_root(tmp_path)) == _STATE
    assert web._read_state_json(_durable_root(tmp_path)) == _STATE
    assert web._read_state_json(_legacy_root(tmp_path)) == _STATE


def test_read_project_stem_parity_across_layouts(tmp_path):
    assert _read_project_stem(_workspace_root(tmp_path)) == "USB_PD_TRIGGER"
    assert _read_project_stem(_durable_root(tmp_path)) == "USB_PD_TRIGGER"
    assert _read_project_stem(_legacy_root(tmp_path)) == "USB_PD_TRIGGER"


def test_synth_check_readers_parity_workspace_vs_durable(tmp_path):
    ws, dur = _workspace_root(tmp_path), _durable_root(tmp_path)
    assert web._synth_check_failures(ws) == web._synth_check_failures(dur)
    assert web._synth_check_failures(ws) == ["§9.12 ERC: U1.3 dangling", "§9.12 ERC: R2.1 no-net"]
    assert web._erc_offenders(ws) == web._erc_offenders(dur) == ["U1.3 dangling", "R2.1 no-net"]
    assert web._quality_badge_from_ws(ws) == web._quality_badge_from_ws(dur) == "erc_errors"


def test_price_cache_roundtrips_through_durable_dir(tmp_path):
    """_save_price_cache writes under the resolved metadata dir; on a durable root
    that is `kicraft/` (no dot) -- exactly the write-through view-from-durable wants."""
    dur = _durable_root(tmp_path, state=False, check=False)
    key = "C999"
    with web._PRICE_LOCK:
        web._PRICE_CACHE[key] = {"unit": 1.23}
    try:
        web._save_price_cache(dur, {key})
        # Landed under the bare durable name, not a new .kicraft.
        assert (dur / "kicraft" / web._PRICE_FILE).is_file()
        assert not (dur / ".kicraft").exists()
        with web._PRICE_LOCK:
            web._PRICE_CACHE.pop(key, None)
        web._load_price_cache(dur)  # reads it back from the same dir
        with web._PRICE_LOCK:
            assert web._PRICE_CACHE.get(key) == {"unit": 1.23}
    finally:
        with web._PRICE_LOCK:
            web._PRICE_CACHE.pop(key, None)
