"""Core-components registry: seeding, CRUD, validation, and seed-file health.

The registry is seeded once from the bundled JSON (flag in app_settings) and
then owned by the admin page; these tests pin the one-shot semantics (restarts
never re-seed, deletions never resurrect) and the single validator shared by
the seed loader and the admin editor.
"""
import json

import pytest

from kicraft.server.accounts import (
    _FUNCTION_KEY_RE,
    CORE_COMPONENT_CATEGORIES,
    CORE_COMPONENTS_SEED_PATH,
    AccountStore,
)


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


def _reopen(tmp_path) -> AccountStore:
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


# ---- seeding ----------------------------------------------------------------

def test_seed_populates_fresh_store(store):
    rows = store.list_core_components()
    assert len(rows) >= 40
    assert {r["category"] for r in rows} == set(CORE_COMPONENT_CATEGORIES)
    keys = [r["function_key"] for r in rows]
    assert len(keys) == len(set(keys))
    passives = [r for r in rows if r["category"] == "passives"]
    assert passives and all(r["default_lcsc"] is None for r in passives)
    # Non-series rows carry a verified snapshot.
    for r in rows:
        if r["default_lcsc"] is not None:
            assert r["price_usd"] is not None and r["price_usd"] > 0
            assert r["stock"] is not None and r["stock"] > 0
            assert r["snapshot_date"]


def test_seed_idempotent_across_restarts(store, tmp_path):
    n = len(store.list_core_components())
    assert len(_reopen(tmp_path).list_core_components()) == n


def test_seed_does_not_resurrect_deleted_rows(store, tmp_path):
    rows = store.list_core_components()
    store.delete_core_component(rows[0]["id"])
    after = _reopen(tmp_path).list_core_components()
    assert len(after) == len(rows) - 1
    assert rows[0]["function_key"] not in {r["function_key"] for r in after}
    # Even a fully emptied table stays empty: the flag, not the row count,
    # gates seeding.
    for r in after:
        store.delete_core_component(r["id"])
    assert _reopen(tmp_path).list_core_components() == []


# ---- create / validation ------------------------------------------------------

def test_create_normalizes(store):
    c = store.create_core_component(
        function_key="My-Block", display_name="  X  ", category="drivers",
        default_mpn=" P1 ", default_lcsc="c123", price_usd="0.25", stock="42")
    assert c["function_key"] == "my-block"      # slug lowered
    assert c["display_name"] == "X"
    assert c["default_mpn"] == "P1"
    assert c["default_lcsc"] == "C123"
    assert c["price_usd"] == 0.25 and c["stock"] == 42
    assert c["enabled"] is True


@pytest.mark.parametrize("kwargs", [
    dict(function_key="x"),                       # too short for the slug regex
    dict(function_key="Bad Key"),                 # whitespace
    dict(category="audio"),                       # unknown category
    dict(display_name="   "),                     # blank required field
    dict(default_mpn=""),                         # blank required field
    dict(default_lcsc="LCSC-14259"),              # not a C-number
    dict(price_usd=-1),
    dict(stock=-5),
])
def test_create_validates(store, kwargs):
    base = dict(function_key="valid-key", display_name="Block",
                category="power", default_mpn="MPN1")
    with pytest.raises(ValueError):
        store.create_core_component(**{**base, **kwargs})


def test_create_duplicate_key_case_insensitive(store):
    store.create_core_component(function_key="dup-key", display_name="A",
                                category="power", default_mpn="M1")
    with pytest.raises(ValueError, match="already exists"):
        store.create_core_component(function_key="DUP-KEY", display_name="B",
                                    category="power", default_mpn="M2")


# ---- get / update / snapshot / delete ----------------------------------------

def test_get_is_case_insensitive(store):
    assert store.get_core_component("LDO-3V3-1A")["function_key"] == "ldo-3v3-1a"
    assert store.get_core_component("nope") is None
    assert store.get_core_component("") is None


def test_update_round_trip(store):
    c = store.create_core_component(function_key="upd-key", display_name="A",
                                    category="sensors", default_mpn="M1")
    u = store.update_core_component(c["id"], qualifier="  q1  ", enabled=False,
                                    default_lcsc="14259")
    assert u["qualifier"] == "q1"
    assert u["enabled"] is False
    assert u["default_lcsc"] == "C14259"
    assert u["updated_at"] >= c["updated_at"]
    with pytest.raises(ValueError):
        store.update_core_component(c["id"], nonsense_field=1)
    with pytest.raises(ValueError):
        store.update_core_component(999999, display_name="B")
    with pytest.raises(ValueError):
        store.update_core_component(c["id"])  # nothing to update


def test_update_duplicate_key_rejected(store):
    a = store.create_core_component(function_key="key-a", display_name="A",
                                    category="power", default_mpn="M1")
    store.create_core_component(function_key="key-b", display_name="B",
                                category="power", default_mpn="M2")
    with pytest.raises(ValueError, match="already exists"):
        store.update_core_component(a["id"], function_key="key-b")


def test_record_snapshot(store):
    c = store.create_core_component(function_key="snap-key", display_name="A",
                                    category="power", default_mpn="M1")
    assert c["snapshot_date"] is None
    s = store.record_core_component_snapshot(c["id"], price_usd=0.5, stock=1000)
    assert s["price_usd"] == 0.5 and s["stock"] == 1000
    assert s["snapshot_date"] and len(s["snapshot_date"]) == 10  # ISO date


def test_delete(store):
    c = store.create_core_component(function_key="del-key", display_name="A",
                                    category="power", default_mpn="M1")
    store.delete_core_component(c["id"])
    assert store.get_core_component("del-key") is None
    with pytest.raises(ValueError):
        store.delete_core_component(c["id"])


# ---- listing -------------------------------------------------------------------

def test_list_ordering_and_filters(store):
    rows = store.list_core_components()
    cats = [r["category"] for r in rows]
    assert cats == sorted(cats)  # grouped by category
    for cat in CORE_COMPONENT_CATEGORIES:
        orders = [r["sort_order"] for r in rows if r["category"] == cat]
        assert orders == sorted(orders)
    only_power = store.list_core_components(category="power")
    assert only_power and all(r["category"] == "power" for r in only_power)
    first = rows[0]
    store.update_core_component(first["id"], enabled=False)
    visible = store.list_core_components(include_disabled=False)
    assert first["function_key"] not in {r["function_key"] for r in visible}
    assert len(visible) == len(rows) - 1


# ---- bundled seed file (packaging/CI guard) -------------------------------------

def test_seed_json_is_valid():
    entries = json.loads(CORE_COMPONENTS_SEED_PATH.read_text(encoding="utf-8"))
    assert isinstance(entries, list) and len(entries) >= 40
    keys = [e["function_key"] for e in entries]
    assert len(keys) == len(set(keys))
    for e in entries:
        assert _FUNCTION_KEY_RE.fullmatch(e["function_key"]), e["function_key"]
        assert e["category"] in CORE_COMPONENT_CATEGORIES, e["function_key"]
        assert e["display_name"].strip(), e["function_key"]
        assert e["default_mpn"].strip(), e["function_key"]
        if e.get("price_usd") is not None:
            assert e["price_usd"] > 0, e["function_key"]
        if e.get("stock") is not None:
            assert e["stock"] > 0, e["function_key"]
        if e.get("default_lcsc") is not None:
            assert e["default_lcsc"].startswith("C"), e["function_key"]
            # A recorded part pick must carry its verification snapshot.
            assert e.get("snapshot_date"), e["function_key"]
