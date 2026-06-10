"""Tests for the public-browser FTS5 search index in kicraft.server.accounts.

Exercises the indexer (build_fts_document) and the catalog full-text query: a
project is found by a part it contains ("esp32") and by what it does ("plant
watering"), and private/failed projects never surface. Pure stdlib + sqlite.
"""
from __future__ import annotations

import json

import pytest

from kicraft.server.accounts import AccountStore, build_fts_document


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


def _state(goal, *, named_parts=None, bom_mpns=None, blocks=None):
    """A minimal state.json-shaped dict for the indexer."""
    return {
        "intent": {"goal": goal, "named_parts": named_parts or []},
        "bom": {"parts": [{"mpn": m, "value": "", "sourcing_note": ""}
                          for m in (bom_mpns or [])]},
        "functional_spec": {"blocks": blocks or []},
        "architecture": {"sheets": []},
    }


def _indexed(store, user_id, brief, state, *, status="ok", is_public=True, stem="BOARD"):
    """Create + finish a project, persist its state.json, and index it."""
    pid = store.create_project(user_id, brief, is_public=is_public)
    pdir = store.projects_dir / str(user_id) / str(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "state.json").write_text(json.dumps(state), encoding="utf-8")
    store.finish_project(pid, status, stem=stem, dir_path=str(pdir))
    store.reindex_search(pid)
    return pid


def test_fts_available(store):
    # The browser's part/function search needs FTS5 in the bundled SQLite. The app
    # still runs without it (LIKE fallback), but these tests can't exercise it.
    assert store._fts_enabled is True


def test_build_fts_document_flattens_state():
    doc = build_fts_document(
        "an esp32 board",
        _state("a plant watering controller",
               named_parts=["ESP32-S3"], bom_mpns=["AMS1117-3.3"],
               blocks=[{"name": "PUMP", "purpose": "drive the water pump"}]))
    assert "esp32" in doc["brief"].lower()
    assert "ESP32-S3" in doc["parts"] and "AMS1117-3.3" in doc["parts"]
    assert "watering" in doc["goal"]
    assert "pump" in doc["blocks"].lower()


def test_build_fts_document_tolerates_empty():
    assert build_fts_document(None, None) == {
        "brief": "", "goal": "", "parts": "", "blocks": ""}


def test_search_matches_by_part(store):
    u = store.create_user("a@e.st", "pw")
    esp = _indexed(store, u.id, "a microcontroller board",
                   _state("a sensor hub", bom_mpns=["ESP32-S3-MINI-1"]), stem="ESP")
    _indexed(store, u.id, "an op-amp board",
             _state("an analog board", bom_mpns=["LM358"]), stem="OPAMP")
    hits = [r["id"] for r in store.list_public_projects(query="esp32")]
    assert hits == [esp]  # matched on the BOM mpn, not the brief; LM358 excluded


def test_search_matches_by_function(store):
    u = store.create_user("a@e.st", "pw")
    plant = _indexed(store, u.id, "board one",
                     _state("plant watering controller with a pump"), stem="PLANT")
    _indexed(store, u.id, "board two", _state("a bike light"), stem="LIGHT")
    assert [r["id"] for r in store.list_public_projects(query="watering")] == [plant]
    # porter stemming: the stem "water" still finds "watering"
    assert plant in [r["id"] for r in store.list_public_projects(query="water")]


def test_search_excludes_private_and_failed(store):
    u = store.create_user("a@e.st", "pw")
    _indexed(store, u.id, "secret", _state("esp32", bom_mpns=["ESP32-S3"]),
             is_public=False, stem="SECRET")
    _indexed(store, u.id, "broken", _state("esp32", bom_mpns=["ESP32-S3"]),
             status="failed", stem="FAILED")
    assert store.list_public_projects(query="esp32") == []


def test_search_sanitizes_operator_chars(store):
    u = store.create_user("a@e.st", "pw")
    _indexed(store, u.id, "esp board", _state("esp32", bom_mpns=["ESP32-S3"]))
    store.list_public_projects(query='"*^():')   # operator-only: must not raise
    hits = store.list_public_projects(query='esp32"')  # stray quote stripped
    assert len(hits) == 1


def test_reindex_removes_on_private(store):
    u = store.create_user("a@e.st", "pw")
    pid = _indexed(store, u.id, "esp board", _state("esp32", bom_mpns=["ESP32-S3"]))
    assert len(store.list_public_projects(query="esp32")) == 1
    store.set_visibility(pid, False)
    store.reindex_search(pid)
    assert store.list_public_projects(query="esp32") == []


def test_backfill_indexes_existing(store):
    u = store.create_user("a@e.st", "pw")
    pid = store.create_project(u.id, "esp board")          # ok+public but not indexed
    pdir = store.projects_dir / str(u.id) / str(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "state.json").write_text(
        json.dumps(_state("esp32", bom_mpns=["ESP32-S3"])), encoding="utf-8")
    store.finish_project(pid, "ok", stem="ESP", dir_path=str(pdir))
    assert store.list_public_projects(query="esp32") == []  # nothing indexed yet
    assert store.backfill_search() == 1
    assert [r["id"] for r in store.list_public_projects(query="esp32")] == [pid]
