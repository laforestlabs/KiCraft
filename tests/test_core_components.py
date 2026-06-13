"""Core-components registry: catalog sync, runtime-state ownership, validation.

The repo catalog (kicraft/parts_library/core_blocks.json) is the source of
truth; AccountStore re-syncs the core_components table from it on every init.
These tests pin the cache semantics: canonical fields always mirror the
catalog (hand edits revert, hand-deleted rows resurrect, removed keys
disappear), while DB-owned runtime state (enabled, price/stock snapshots)
survives syncs. The catalog guard at the bottom is the CI tripwire for the
catalog file itself.
"""
import pytest

from kicraft.parts_library.core_blocks import (
    CORE_BLOCKS_PATH,
    CORE_COMPONENT_CATEGORIES,
    FUNCTION_KEY_RE,
    CoreBlockCatalog,
    load_core_catalog,
    resolve_block,
)
from kicraft.parts_library.loader import vendored_parts_dir
from kicraft.parts_library.manifest import load_manifest
from kicraft.server.accounts import AccountStore


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


def _reopen(tmp_path) -> AccountStore:
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


def _catalog_of(*blocks: dict) -> CoreBlockCatalog:
    return CoreBlockCatalog.model_validate(
        {"schema_version": "1", "blocks": list(blocks)})


# ---- sync: catalog -> DB ------------------------------------------------------

def test_sync_populates_fresh_store(store):
    rows = store.list_core_components()
    catalog = load_core_catalog()
    assert len(rows) == len(catalog.blocks)
    assert {r["category"] for r in rows} == set(CORE_COMPONENT_CATEGORIES)
    keys = [r["function_key"] for r in rows]
    assert len(keys) == len(set(keys))
    passives = [r for r in rows if r["category"] == "passives"]
    assert passives and all(r["default_lcsc"] is None for r in passives)
    # Snapshots are runtime state, never synced from the catalog.
    assert all(r["price_usd"] is None and r["snapshot_date"] is None
               for r in rows)


def test_sync_idempotent_across_restarts(store, tmp_path):
    before = {r["function_key"]: r for r in store.list_core_components()}
    after = {r["function_key"]: r
             for r in _reopen(tmp_path).list_core_components()}
    assert after == before  # a no-op sync does not even touch updated_at


def test_sync_preserves_runtime_state_but_reverts_canonical_fields(
        store, tmp_path):
    row = store.get_core_component("ldo-3v3-1a")
    store.update_core_component(row["id"], enabled=False)
    store.record_core_component_snapshot(row["id"], price_usd=0.5, stock=42)
    # Simulate canonical drift (the admin page no longer offers this, but the
    # store method still validates and writes).
    store.update_core_component(row["id"], display_name="Hand-edited")
    after = _reopen(tmp_path).get_core_component("ldo-3v3-1a")
    assert after["enabled"] is False
    assert after["price_usd"] == 0.5 and after["stock"] == 42
    assert after["snapshot_date"]
    assert after["display_name"] == row["display_name"]  # reverted


def test_sync_deletes_rows_whose_key_left_the_catalog(store, tmp_path):
    row = store.get_core_component("ldo-3v3-1a")
    store.update_core_component(row["id"], function_key="zzz-custom-row")
    after = _reopen(tmp_path)
    assert after.get_core_component("zzz-custom-row") is None  # not in catalog
    assert after.get_core_component("ldo-3v3-1a") is not None  # re-inserted
    assert len(after.list_core_components()) == len(load_core_catalog().blocks)


def test_hand_deleted_row_resurrects(store, tmp_path):
    row = store.get_core_component("ldo-3v3-1a")
    with store._conn() as conn:
        conn.execute("DELETE FROM core_components WHERE id=?", (row["id"],))
    assert _reopen(tmp_path).get_core_component("ldo-3v3-1a") is not None


def test_bundle_rows_derive_from_manifests(store):
    base = vendored_parts_dir()
    bundle_rows = [r for r in store.list_core_components() if r["bundle"]]
    assert bundle_rows
    for r in bundle_rows:
        manifest = load_manifest(base / r["bundle"])
        assert r["default_mpn"] == manifest.mpn
        assert r["default_lcsc"] == (manifest.sourcing or {}).get("lcsc")


def test_two_rows_may_share_one_bundle(store, tmp_path, monkeypatch):
    cat = _catalog_of(
        {"function_key": "fn-one", "display_name": "One", "category": "sensors",
         "bundle": "mpu6050"},
        {"function_key": "fn-two", "display_name": "Two", "category": "sensors",
         "bundle": "mpu6050"},
    )
    monkeypatch.setattr("kicraft.parts_library.core_blocks.load_core_catalog",
                        lambda path=None: cat)
    fresh = _reopen(tmp_path)
    rows = fresh.list_core_components()
    assert {r["function_key"] for r in rows} == {"fn-one", "fn-two"}
    assert all(r["bundle"] == "mpu6050" for r in rows)


def test_sync_survives_unreadable_catalog(store, tmp_path, monkeypatch):
    n = len(store.list_core_components())

    def boom(path=None):
        raise OSError("no catalog")

    monkeypatch.setattr(
        "kicraft.parts_library.core_blocks.load_core_catalog", boom)
    assert len(_reopen(tmp_path).list_core_components()) == n  # rows kept


def test_sync_keeps_row_when_bundle_unresolvable(store, tmp_path, monkeypatch):
    # The key stays in the catalog but its bundle manifest cannot be read:
    # the existing DB row must be kept (warn), not deleted.
    before = store.get_core_component("ldo-3v3-1a")
    real = load_core_catalog()
    blocks = []
    for b in real.blocks:
        d = b.model_dump(exclude_none=True)
        if b.function_key == "ldo-3v3-1a":
            d["bundle"] = "no-such-bundle"
        blocks.append(d)
    cat = _catalog_of(*blocks)
    monkeypatch.setattr("kicraft.parts_library.core_blocks.load_core_catalog",
                        lambda path=None: cat)
    after = _reopen(tmp_path).get_core_component("ldo-3v3-1a")
    assert after is not None
    assert after["default_mpn"] == before["default_mpn"]


# ---- runtime mutations (the surface the admin page still owns) ------------------

def test_get_is_case_insensitive(store):
    assert store.get_core_component("LDO-3V3-1A")["function_key"] == "ldo-3v3-1a"
    assert store.get_core_component("nope") is None
    assert store.get_core_component("") is None


def test_update_validates(store):
    row = store.get_core_component("ldo-3v3-1a")
    u = store.update_core_component(row["id"], qualifier="  q1  ",
                                    enabled=False, default_lcsc="14259")
    assert u["qualifier"] == "q1"
    assert u["enabled"] is False
    assert u["default_lcsc"] == "C14259"
    with pytest.raises(ValueError):
        store.update_core_component(row["id"], nonsense_field=1)
    with pytest.raises(ValueError):
        store.update_core_component(row["id"], category="audio")
    with pytest.raises(ValueError):
        store.update_core_component(row["id"], default_lcsc="LCSC-14259")
    with pytest.raises(ValueError):
        store.update_core_component(row["id"], bundle="Bad Bundle")
    with pytest.raises(ValueError):
        store.update_core_component(row["id"], price_usd=-1)
    with pytest.raises(ValueError):
        store.update_core_component(999999, display_name="B")
    with pytest.raises(ValueError):
        store.update_core_component(row["id"])  # nothing to update


def test_record_snapshot(store):
    row = store.get_core_component("ldo-3v3-1a")
    assert row["snapshot_date"] is None
    s = store.record_core_component_snapshot(row["id"], price_usd=0.5,
                                             stock=1000)
    assert s["price_usd"] == 0.5 and s["stock"] == 1000
    assert s["snapshot_date"] and len(s["snapshot_date"]) == 10  # ISO date


def test_create_and_delete_are_gone():
    assert not hasattr(AccountStore, "create_core_component")
    assert not hasattr(AccountStore, "delete_core_component")


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


# ---- catalog guard (packaging/CI tripwire) --------------------------------------

# Transitional rows (default_lcsc, not yet vendored) count down to zero as the
# vendoring batches land; bump this as each batch flips its rows to `bundle:`.
EXPECTED_TRANSITIONAL_ROWS = 2


def test_catalog_is_valid_and_bundles_resolve():
    catalog = load_core_catalog()  # pydantic-valid incl. unique keys
    assert CORE_BLOCKS_PATH.is_file()
    assert len(catalog.blocks) >= 40
    base = vendored_parts_dir()
    transitional = 0
    for block in catalog.blocks:
        assert FUNCTION_KEY_RE.fullmatch(block.function_key)
        if block.bundle is not None:
            manifest = load_manifest(base / block.bundle)  # raises if broken
            row = resolve_block(block)
            assert row["default_mpn"] == manifest.mpn
        elif block.stock is not None:
            assert block.category == "passives"
        else:
            transitional += 1
    assert transitional == EXPECTED_TRANSITIONAL_ROWS, (
        f"{transitional} transitional (LCSC-only) rows; expected "
        f"{EXPECTED_TRANSITIONAL_ROWS}. After vendoring a batch, flip its "
        f"catalog rows to bundle: and update EXPECTED_TRANSITIONAL_ROWS.")
