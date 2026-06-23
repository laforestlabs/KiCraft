"""Storage-layout accessors + the readers built on them (refactor roadmap Phase 4a).

One name, no fallback: a project's run-metadata always lives at ``<root>/.kicraft/``
(state.json, synthesis_check.json, the price cache). `_kicraft_dir` / `_state_path`
(storage.py) are the single definition of that layout; every state/check/price reader
goes through them. Pure-data: no NiceGUI, no pcbnew, no LLM.
"""
from __future__ import annotations

import json

from kicraft.server import web
from kicraft.server.session import read_state
from kicraft.server.storage import _kicraft_dir, _read_project_stem, _state_path

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


def _root(tmp_path, *, state=True, check=True, prices=False):
    """A project root with metadata under the one canonical `.kicraft/` dir."""
    root = tmp_path / "proj"
    meta = root / ".kicraft"
    meta.mkdir(parents=True, exist_ok=True)
    if state:
        (meta / "state.json").write_text(json.dumps(_STATE), encoding="utf-8")
    if check:
        (meta / "synthesis_check.json").write_text(json.dumps(_SYNTH_CHECK), encoding="utf-8")
    if prices:
        (meta / web._PRICE_FILE).write_text(json.dumps(_PRICES), encoding="utf-8")
    return root


# --- _kicraft_dir / _state_path: the one layout, no fallback --------------------

def test_kicraft_dir_and_state_path_are_dotkicraft(tmp_path):
    root = tmp_path / "x"
    assert _kicraft_dir(root) == root / ".kicraft"
    assert _state_path(root) == root / ".kicraft" / "state.json"
    # No probing/fallback: the same answer whether or not anything exists on disk.
    (root / ".kicraft").mkdir(parents=True)
    assert _kicraft_dir(root) == root / ".kicraft"


# --- the readers, against the canonical layout ----------------------------------

def test_read_state_reads_dotkicraft(tmp_path):
    assert read_state(_root(tmp_path)) == _STATE
    assert read_state(tmp_path / "nope") == {}  # absent -> {}


def test_web_read_state_json_reads_dotkicraft(tmp_path):
    assert web._read_state_json(_root(tmp_path)) == _STATE
    assert web._read_state_json(tmp_path / "nope") == {}


def test_read_project_stem_reads_dotkicraft(tmp_path):
    assert _read_project_stem(_root(tmp_path)) == "USB_PD_TRIGGER"


def test_synth_check_readers(tmp_path):
    root = _root(tmp_path)
    assert web._synth_check_failures(root) == [
        "§9.12 ERC: U1.3 dangling", "§9.12 ERC: R2.1 no-net"]
    assert web._erc_offenders(root) == ["U1.3 dangling", "R2.1 no-net"]
    assert web._quality_badge_from_ws(root) == "erc_errors"


def test_price_cache_roundtrips_through_dotkicraft(tmp_path):
    """_save_price_cache writes under <root>/.kicraft/ and _load_price_cache reads it
    back from the same place -- the build-in-place price cache, no copy."""
    root = _root(tmp_path, state=False, check=False)
    key = "C999"
    with web._PRICE_LOCK:
        web._PRICE_CACHE[key] = {"unit": 1.23}
    try:
        web._save_price_cache(root, {key})
        assert (root / ".kicraft" / web._PRICE_FILE).is_file()
        with web._PRICE_LOCK:
            web._PRICE_CACHE.pop(key, None)
        web._load_price_cache(root)
        with web._PRICE_LOCK:
            assert web._PRICE_CACHE.get(key) == {"unit": 1.23}
    finally:
        with web._PRICE_LOCK:
            web._PRICE_CACHE.pop(key, None)
