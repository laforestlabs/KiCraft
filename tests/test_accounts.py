"""Tests for kicraft.server.accounts: hashing, users, tiers, and quota metering.

Pure stdlib + sqlite (no pcbnew, no network), so it runs fast anywhere.
"""
from __future__ import annotations

import datetime as dt
import json
import sqlite3

import pytest

from kicraft.server.accounts import (
    AccountStore,
    hash_password,
    verify_password,
)


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


def _backdate(store, project_id, days_ago):
    """Move a project's created_at into the past to exercise the quota window."""
    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago)).isoformat()
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE projects SET created_at=? WHERE id=?", (ts, project_id))


# ---- password hashing -----------------------------------------------------

def test_password_roundtrip():
    h = hash_password("correct horse battery staple")
    assert h.startswith("scrypt$")
    assert verify_password("correct horse battery staple", h)
    assert not verify_password("wrong", h)


def test_verify_rejects_garbage():
    assert not verify_password("x", "not-a-hash")
    assert not verify_password("x", "")


# ---- users ----------------------------------------------------------------

def test_create_and_authenticate(store):
    u = store.create_user("Alice@Example.com", "pw123")
    assert u.email == "alice@example.com"  # normalized to lowercase
    assert u.tier == "free"
    got = store.authenticate("alice@example.com", "pw123")
    assert got is not None and got.id == u.id
    assert store.authenticate("alice@example.com", "nope") is None
    assert store.authenticate("ghost@example.com", "pw123") is None


def test_duplicate_email_rejected(store):
    store.create_user("bob@example.com", "pw")
    with pytest.raises(ValueError):
        store.create_user("BOB@example.com", "other")  # case-insensitive clash


def test_invalid_input_rejected(store):
    with pytest.raises(ValueError):
        store.create_user("not-an-email", "pw")
    with pytest.raises(ValueError):
        store.create_user("ok@e.st", "")  # empty password


def test_set_tier(store):
    store.create_user("carol@example.com", "pw")
    assert store.set_tier("carol@example.com", "pro").tier == "pro"
    with pytest.raises(ValueError):
        store.set_tier("carol@example.com", "platinum")  # unknown tier
    with pytest.raises(ValueError):
        store.set_tier("ghost@example.com", "pro")  # no such user


# ---- consent + data controls ----------------------------------------------

def test_consent_recorded_on_create(store):
    u = store.create_user("c@e.st", "pw",
                          accepted_terms_version="2026-06-04", allow_training=False)
    assert u.accepted_terms_version == "2026-06-04"
    assert u.accepted_terms_at is not None
    assert u.allow_training is False
    got = store.get_user(u.id)  # round-trips through the DB
    assert got.accepted_terms_version == "2026-06-04"
    assert got.allow_training is False


def test_consent_defaults_when_omitted(store):
    u = store.create_user("c2@e.st", "pw")
    assert u.accepted_terms_version is None  # so the re-consent gate fires
    assert u.accepted_terms_at is None
    assert u.allow_training is True  # training defaults on


def test_record_consent_stamps_version(store):
    u = store.create_user("r@e.st", "pw")  # signed up before this terms version
    assert store.get_user(u.id).accepted_terms_version is None
    store.record_consent(u.id, "2026-06-04")
    got = store.get_user(u.id)
    assert got.accepted_terms_version == "2026-06-04"
    assert got.accepted_terms_at is not None


def test_set_training_pref_toggles(store):
    u = store.create_user("t@e.st", "pw")
    store.set_training_pref(u.id, False)
    assert store.get_user(u.id).allow_training is False
    store.set_training_pref(u.id, True)
    assert store.get_user(u.id).allow_training is True


def test_legacy_db_upgrades_without_losing_rows(tmp_path):
    """A DB created before consent tracking gains the columns on open, keeps rows."""
    db = tmp_path / "accounts.db"
    with sqlite3.connect(db) as conn:  # pre-consent users schema
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,"
            "tier TEXT NOT NULL DEFAULT 'free', created_at TEXT NOT NULL,"
            "last_login_at TEXT)")
        conn.execute(
            "INSERT INTO users (email, password_hash, tier, created_at) "
            "VALUES ('old@e.st', 'scrypt$x', 'pro', '2026-01-01T00:00:00+00:00')")
    store = AccountStore(db, tmp_path / "projects")  # _ensure_columns migrates
    u = store.get_user_by_email("old@e.st")
    assert u is not None and u.tier == "pro"  # existing data preserved
    assert u.accepted_terms_version is None  # legacy user is re-prompted
    assert u.allow_training is True  # backfilled default


def test_export_user_dumps_metadata_without_password(store):
    u = store.create_user("exp@e.st", "pw",
                          accepted_terms_version="2026-06-04", allow_training=False)
    store.finish_project(store.create_project(u.id, "board one"), "ok", stem="ONE")
    data = store.export_user(u.id)
    assert data["user"]["email"] == "exp@e.st"
    assert data["user"]["allow_training"] is False
    assert "password_hash" not in data["user"]  # User dataclass omits it
    assert "password" not in json.dumps(data)
    assert len(data["projects"]) == 1
    assert data["projects"][0]["brief"] == "board one"


def test_delete_user_purges_rows_and_files(store):
    u = store.create_user("del@e.st", "pw")
    pid = store.create_project(u.id, "x")
    store.finish_project(pid, "ok", stem="X")
    tree = store.projects_dir / str(u.id) / str(pid)  # simulate persisted files
    tree.mkdir(parents=True, exist_ok=True)
    (tree / "brief.txt").write_text("hi", encoding="utf-8")
    purged = store.delete_user(u.id)
    assert purged is not None
    assert not (store.projects_dir / str(u.id)).exists()  # tree gone
    assert store.get_user(u.id) is None  # row gone
    assert store.list_projects(u.id) == []  # projects gone


# ---- quota metering -------------------------------------------------------

def test_quota_status_reflects_tier(store):
    u = store.create_user("d@e.st", "pw")
    q = store.quota_status(u)
    assert (q["tier"], q["limit"], q["window_days"]) == ("free", 1, 7)
    assert q["used"] == 0 and q["remaining"] == 1


def test_free_quota_one_per_week(store):
    u = store.create_user("free@e.st", "pw")
    pid = store.create_project(u.id, "a board")
    store.finish_project(pid, "ok", stem="BOARD")
    assert store.quota_status(u)["remaining"] == 0  # used the 1/week
    _backdate(store, pid, 8)  # now older than the 7-day window
    assert store.quota_status(u)["remaining"] == 1  # window cleared it


def test_running_design_reserves_a_slot(store):
    u = store.create_user("run@e.st", "pw")
    store.create_project(u.id, "in flight")  # left at status 'running'
    assert store.quota_status(u)["remaining"] == 0  # reserved before it finishes


def test_awaiting_input_holds_the_slot(store):
    u = store.create_user("park@e.st", "pw")
    pid = store.create_project(u.id, "parked on a question")
    store.update_project_status(pid, "awaiting_input")
    assert store.quota_status(u)["remaining"] == 0  # a parked run still holds the slot
    store.finish_project(pid, "failed")  # abandoning the design frees it
    assert store.quota_status(u)["remaining"] == 1


def test_failed_designs_do_not_count(store):
    store.create_user("fail@e.st", "pw")
    u = store.set_tier("fail@e.st", "pro")
    for _ in range(3):
        pid = store.create_project(u.id, "x")
        store.finish_project(pid, "failed")
    assert store.quota_status(u)["used"] == 0  # failures free the slot
    assert store.quota_status(u)["remaining"] == 5


def test_pro_and_max_limits(store):
    store.create_user("p@e.st", "pw")
    u = store.set_tier("p@e.st", "pro")
    for _ in range(5):
        store.finish_project(store.create_project(u.id, "x"), "ok")
    assert store.quota_status(u)["remaining"] == 0
    u = store.set_tier("p@e.st", "max")  # same 5 now count against 25
    assert store.quota_status(u)["remaining"] == 20


def test_window_expiry_for_month_tier(store):
    store.create_user("w@e.st", "pw")
    u = store.set_tier("w@e.st", "pro")
    old = store.create_project(u.id, "old")
    store.finish_project(old, "ok")
    _backdate(store, old, 40)  # outside the 30-day window
    store.finish_project(store.create_project(u.id, "new"), "ok")
    assert store.quota_status(u)["used"] == 1  # only the recent one counts


# ---- projects -------------------------------------------------------------

def test_projects_listed_newest_first(store):
    u = store.create_user("l@e.st", "pw")
    p1 = store.create_project(u.id, "first")
    p2 = store.create_project(u.id, "second")
    store.finish_project(p1, "ok", stem="FIRST", zip_path="/tmp/f.zip")
    projs = store.list_projects(u.id)
    assert [p.id for p in projs] == [p2, p1]
    first = store.get_project(p1)
    assert first.project_stem == "FIRST" and first.zip_path == "/tmp/f.zip"
