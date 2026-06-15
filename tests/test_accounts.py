"""Tests for kicraft.server.accounts: hashing, users, tiers, and quota metering.

Pure stdlib + sqlite (no pcbnew, no network), so it runs fast anywhere.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sqlite3

import pytest

from kicraft.server.accounts import (
    TIERS,
    AccountStore,
    grant_expiry,
    hash_password,
    is_admin,
    new_board_code,
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


def _set_reset_times(store, *, created_secs_ago=None, expires_secs_from_now=None):
    """Rewrite every password_resets row's created_at / expires_at to a relative
    time, so tests can step past the cooldown or force a token to expire without
    sleeping. A negative expires_secs_from_now puts expiry in the past."""
    now = dt.datetime.now(dt.timezone.utc)
    sets, params = [], []
    if created_secs_ago is not None:
        sets.append("created_at=?")
        params.append((now - dt.timedelta(seconds=created_secs_ago)).isoformat())
    if expires_secs_from_now is not None:
        sets.append("expires_at=?")
        params.append((now + dt.timedelta(seconds=expires_secs_from_now)).isoformat())
    with sqlite3.connect(store.path) as conn:
        conn.execute(f"UPDATE password_resets SET {', '.join(sets)}", params)


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
    assert u.session_epoch == 0  # backfilled so existing sessions stay valid
    assert u.tier_expires_at is None  # backfilled: existing tiers never lapse


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


def test_delete_project_purges_one_row_and_tree_keeps_others(store):
    u = store.create_user("p@e.st", "pw")
    keep = store.create_project(u.id, "keep")
    store.finish_project(keep, "ok", stem="KEEP")
    drop = store.create_project(u.id, "drop")
    store.finish_project(drop, "ok", stem="DROP")
    drop_tree = store.projects_dir / str(u.id) / str(drop)
    drop_tree.mkdir(parents=True, exist_ok=True)
    (drop_tree / "kicraft_project.zip").write_text("zip", encoding="utf-8")

    purged = store.delete_project(drop)
    assert purged == str(drop_tree)
    assert not drop_tree.exists()                       # only that tree is gone
    assert store.get_project(drop) is None              # only that row is gone
    assert store.get_project(keep) is not None          # sibling untouched
    assert [p.id for p in store.list_projects(u.id)] == [keep]
    assert store.get_user(u.id) is not None             # account intact


def test_delete_project_missing_is_a_noop(store):
    assert store.delete_project(999999) is None


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


# ---- password reset + session epoch ---------------------------------------

def test_reset_token_roundtrip(store):
    store.create_user("reset@e.st", "oldpw")
    token = store.create_reset_token("reset@e.st")
    assert token and len(token) > 20
    assert store.verify_reset_token(token).email == "reset@e.st"  # resolves the user
    updated = store.consume_reset_token(token, "newpw")
    assert updated is not None
    assert store.authenticate("reset@e.st", "oldpw") is None  # old password dead
    assert store.authenticate("reset@e.st", "newpw") is not None  # new one works


def test_reset_token_is_single_use(store):
    store.create_user("once@e.st", "pw")
    token = store.create_reset_token("once@e.st")
    assert store.consume_reset_token(token, "first") is not None
    assert store.consume_reset_token(token, "second") is None  # already spent
    assert store.verify_reset_token(token) is None
    assert store.authenticate("once@e.st", "first") is not None  # second never applied


def test_reset_token_expires(store):
    store.create_user("exp@e.st", "pw")
    token = store.create_reset_token("exp@e.st")
    _set_reset_times(store, expires_secs_from_now=-1)  # already expired
    assert store.verify_reset_token(token) is None
    assert store.consume_reset_token(token, "newpw") is None
    assert store.authenticate("exp@e.st", "pw") is not None  # unchanged


def test_new_reset_token_invalidates_prior(store):
    store.create_user("rotate@e.st", "pw")
    first = store.create_reset_token("rotate@e.st")
    _set_reset_times(store, created_secs_ago=120)  # step past the cooldown
    second = store.create_reset_token("rotate@e.st")
    assert second and second != first
    assert store.verify_reset_token(first) is None  # the old link is dead
    assert store.verify_reset_token(second).email == "rotate@e.st"


def test_create_reset_token_unknown_email_returns_none(store):
    assert store.create_reset_token("ghost@e.st") is None  # no enumeration signal


def test_create_reset_token_cooldown(store):
    store.create_user("cool@e.st", "pw")
    assert store.create_reset_token("cool@e.st") is not None
    assert store.create_reset_token("cool@e.st") is None  # within the cooldown window
    _set_reset_times(store, created_secs_ago=120)
    assert store.create_reset_token("cool@e.st") is not None  # cooldown elapsed


def test_reset_garbage_and_empty_token(store):
    assert store.verify_reset_token("not-a-real-token") is None
    assert store.verify_reset_token("") is None
    assert store.consume_reset_token("not-a-real-token", "pw") is None
    assert store.consume_reset_token("", "pw") is None


def test_consume_empty_password_raises_and_keeps_token(store):
    store.create_user("e@e.st", "pw")
    token = store.create_reset_token("e@e.st")
    with pytest.raises(ValueError):
        store.consume_reset_token(token, "")  # rejected before the token is spent
    assert store.verify_reset_token(token) is not None  # still usable


def test_set_password_bumps_session_epoch(store):
    u = store.create_user("epoch@e.st", "pw")
    assert u.session_epoch == 0
    store.set_password(u.id, "pw2")
    assert store.get_user(u.id).session_epoch == 1  # every change rotates the epoch
    assert store.authenticate("epoch@e.st", "pw2") is not None


def test_set_password_validates(store):
    u = store.create_user("v@e.st", "pw")
    with pytest.raises(ValueError):
        store.set_password(u.id, "")  # empty rejected
    with pytest.raises(ValueError):
        store.set_password(999999, "pw")  # no such user


def test_consume_bumps_session_epoch(store):
    u = store.create_user("ce@e.st", "pw")
    token = store.create_reset_token("ce@e.st")
    updated = store.consume_reset_token(token, "newpw")
    assert updated.session_epoch == 1  # reset evicts other sessions via the epoch


def test_session_epoch_defaults_to_zero(store):
    u = store.create_user("z@e.st", "pw")
    assert u.session_epoch == 0
    assert store.get_user(u.id).session_epoch == 0  # round-trips through the DB


# ---- roles, admin gate, and the retired admin tier ------------------------

def test_new_user_defaults_to_user_role(store):
    u = store.create_user("r@e.st", "pw")
    assert u.role == "user"
    assert is_admin(u) is False


def test_set_role_grants_and_revokes_admin(store):
    promoted = store.set_role(store.create_user("r@e.st", "pw").email, "admin")
    assert promoted.role == "admin" and is_admin(promoted)
    assert is_admin(store.get_user_by_email("r@e.st"))  # round-trips through the DB
    demoted = store.set_role("r@e.st", "user")
    assert demoted.role == "user" and not is_admin(demoted)
    by_id = store.set_role(promoted.id, "admin")  # addressable by id (dashboard path)
    assert by_id.role == "admin"


def test_set_role_validates(store):
    store.create_user("r@e.st", "pw")
    with pytest.raises(ValueError):
        store.set_role("r@e.st", "wizard")     # unknown role
    with pytest.raises(ValueError):
        store.set_role("ghost@e.st", "admin")  # no such user


def test_is_admin_handles_none(store):
    assert is_admin(None) is False


def test_count_role(store):
    store.create_user("a@e.st", "pw")
    store.create_user("b@e.st", "pw")
    assert store.count_role("admin") == 0
    store.set_role("a@e.st", "admin")
    assert store.count_role("admin") == 1
    assert store.count_role("user") == 1


def test_admin_is_no_longer_a_billing_tier(store):
    assert "admin" not in TIERS
    store.create_user("a@e.st", "pw")
    with pytest.raises(ValueError):
        store.set_tier("a@e.st", "admin")  # admin is a role now, not a tier


def test_staff_bypass_quota(store):
    u = store.create_user("op@e.st", "pw")  # free tier: 1/week
    store.create_project(u.id, "b1")        # consume the one slot
    assert store.can_design(u) is False
    assert store.quota_status(u)["unlimited"] is False
    u = store.set_role("op@e.st", "admin")
    assert store.can_design(u) is True
    q = store.quota_status(u)
    assert q["unlimited"] is True
    assert q["limit"] is None and q["remaining"] == float("inf")


def test_legacy_admin_tier_migrates_to_role(tmp_path):
    """A user on the retired 'admin' billing tier is promoted to the admin role and
    reset to free when the role column is introduced; a paid customer is untouched;
    re-opening the DB changes nothing (idempotent backfill)."""
    db = tmp_path / "accounts.db"
    with sqlite3.connect(db) as conn:  # pre-role users schema
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,"
            "tier TEXT NOT NULL DEFAULT 'free', created_at TEXT NOT NULL,"
            "last_login_at TEXT)")
        conn.execute("INSERT INTO users (email, password_hash, tier, created_at) "
                     "VALUES ('op@e.st', 'scrypt$x', 'admin', '2026-01-01T00:00:00+00:00')")
        conn.execute("INSERT INTO users (email, password_hash, tier, created_at) "
                     "VALUES ('cust@e.st', 'scrypt$x', 'max', '2026-01-02T00:00:00+00:00')")
    store = AccountStore(db, tmp_path / "projects")  # _ensure_columns migrates
    op = store.get_user_by_email("op@e.st")
    assert op.role == "admin" and op.tier == "free" and is_admin(op)  # promoted, reset
    cust = store.get_user_by_email("cust@e.st")
    assert cust.role == "user" and cust.tier == "max"  # paid customer untouched

    reopened = AccountStore(db, tmp_path / "projects")  # idempotent on re-open
    op2 = reopened.get_user_by_email("op@e.st")
    assert op2.role == "admin" and op2.tier == "free"
    assert reopened.count_role("admin") == 1


# ---- admin stats ----------------------------------------------------------

def _add_project(store, user_id, status="ok", cost=None, days_ago=0):
    """Create + finish a project, optionally back-dating it to exercise day buckets."""
    pid = store.create_project(user_id, "brief")
    store.finish_project(pid, status, "stem" if status == "ok" else None, cost)
    if days_ago:
        ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days_ago)).isoformat()
        with sqlite3.connect(store.path) as conn:
            conn.execute("UPDATE projects SET created_at=?, finished_at=? WHERE id=?",
                         (ts, ts, pid))
    return pid


def test_overview_stats(store):
    a = store.create_user("a@e.st", "pw")
    b = store.create_user("b@e.st", "pw")
    store.set_tier("b@e.st", "max")
    store.set_role("a@e.st", "admin")
    _add_project(store, a.id, "ok", 1.50)
    _add_project(store, a.id, "failed", None)
    _add_project(store, b.id, "ok", 2.00)
    s = store.overview_stats(window_days=30)
    assert s["users_total"] == 2 and s["admins"] == 1
    assert s["projects_total"] == 3
    assert s["spend_total_usd"] == pytest.approx(3.50)
    assert s["spend_avg_usd"] == pytest.approx(1.75)  # mean of the two non-null costs
    assert s["avg_latency_s"] is not None


def test_overview_stats_empty_db(store):
    s = store.overview_stats()
    assert s["users_total"] == 0 and s["projects_total"] == 0
    assert s["spend_total_usd"] == 0.0
    assert s["spend_avg_usd"] is None and s["avg_latency_s"] is None


def test_distributions(store):
    a = store.create_user("a@e.st", "pw")
    store.create_user("b@e.st", "pw")
    store.set_tier("b@e.st", "max")
    _add_project(store, a.id, "ok", 1.0)
    _add_project(store, a.id, "failed", None)
    assert dict(store.tier_distribution()) == {"free": 1, "max": 1}
    assert dict(store.status_distribution()) == {"ok": 1, "failed": 1}


def test_per_day_series_bucket_by_date(store):
    u = store.create_user("a@e.st", "pw")
    _add_project(store, u.id, "ok", 1.0, days_ago=0)
    _add_project(store, u.id, "ok", 3.0, days_ago=0)
    _add_project(store, u.id, "ok", 5.0, days_ago=10)
    today = dt.date.today().isoformat()
    ppd = dict(store.projects_per_day(60))
    assert ppd[today] == 2 and sum(ppd.values()) == 3
    spd = dict(store.spend_per_day(60))
    assert spd[today] == pytest.approx(4.0)  # 1.0 + 3.0 on the same day
    assert sum(spd.values()) == pytest.approx(9.0)
    assert sum(dict(store.projects_per_day(5)).values()) == 2  # 10d-ago is outside 5d


def test_signups_per_day(store):
    store.create_user("a@e.st", "pw")
    store.create_user("b@e.st", "pw")
    assert dict(store.signups_per_day(30))[dt.date.today().isoformat()] == 2


# ---- invite codes -----------------------------------------------------------

def test_create_and_check_invite_code(store):
    c = store.create_invite_code("FREEMAX", "max", duration_days=30, max_uses=5)
    assert (c["code"], c["tier"], c["duration_days"], c["max_uses"]) \
        == ("FREEMAX", "max", 30, 5)
    assert c["enabled"] is True and c["use_count"] == 0
    got = store.check_invite_code("FREEMAX")
    assert got is not None and got["tier"] == "max"
    assert store.check_invite_code("freemax") is not None  # case-insensitive
    assert store.check_invite_code("  FREEMAX  ") is not None  # whitespace-tolerant
    assert store.check_invite_code("NOPE") is None
    assert store.check_invite_code("") is None


def test_create_invite_code_validates(store):
    with pytest.raises(ValueError):
        store.create_invite_code("has space", "free")  # charset
    with pytest.raises(ValueError):
        store.create_invite_code("ab", "free")  # too short
    with pytest.raises(ValueError):
        store.create_invite_code("OK_CODE", "platinum")  # unknown tier
    with pytest.raises(ValueError):
        store.create_invite_code("OK_CODE", "free", duration_days=0)
    with pytest.raises(ValueError):
        store.create_invite_code("OK_CODE", "free", max_uses=0)
    store.create_invite_code("DUPE", "free")
    with pytest.raises(ValueError):
        store.create_invite_code("dupe", "pro")  # duplicate, case-insensitively


def test_disable_and_reenable_invite_code(store):
    c = store.create_invite_code("BETA", "pro")
    store.set_invite_code_enabled(c["id"], False)
    assert store.check_invite_code("BETA") is None  # disabled codes don't redeem
    listed = store.list_invite_codes()[0]
    assert listed["enabled"] is False and listed["disabled_at"] is not None
    store.set_invite_code_enabled(c["id"], True)
    assert store.check_invite_code("BETA") is not None
    assert store.list_invite_codes()[0]["disabled_at"] is None
    with pytest.raises(ValueError):
        store.set_invite_code_enabled(99999, False)  # no such code


def test_invite_code_max_uses_exhausts(store):
    c = store.create_invite_code("TWICE", "pro", max_uses=2)
    store.record_invite_use(c["id"])
    assert store.check_invite_code("TWICE") is not None  # 1 of 2 used
    store.record_invite_use(c["id"])
    assert store.check_invite_code("TWICE") is None  # used up
    listed = store.list_invite_codes()[0]
    assert listed["use_count"] == 2 and listed["last_used_at"] is not None


def test_invite_codes_listed_newest_first(store):
    store.create_invite_code("FIRST", "free")
    store.create_invite_code("SECOND", "max", duration_days=7)
    assert [c["code"] for c in store.list_invite_codes()] == ["SECOND", "FIRST"]


# ---- tier expiry ------------------------------------------------------------

def _past_iso(days=1):
    return (dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)).isoformat()


def test_grant_expiry():
    assert grant_expiry(None) is None  # forever
    assert grant_expiry(0) is None
    soon = grant_expiry(7)
    assert soon > dt.datetime.now(dt.timezone.utc).isoformat()  # in the future


def test_code_granted_tier_lapses_on_read(store):
    u = store.create_user("comp@e.st", "pw", tier="max",
                          tier_expires_at=_past_iso())
    got = store.get_user(u.id)
    assert got.tier == "free" and got.tier_expires_at is None  # lapsed + cleared
    with sqlite3.connect(store.path) as conn:  # persisted, not just in the object
        row = conn.execute("SELECT tier, tier_expires_at FROM users WHERE id=?",
                           (u.id,)).fetchone()
    assert row == ("free", None)


def test_unexpired_grant_keeps_tier(store):
    u = store.create_user("max@e.st", "pw", tier="max",
                          tier_expires_at=grant_expiry(30))
    got = store.get_user(u.id)
    assert got.tier == "max" and got.tier_expires_at is not None
    assert store.quota_status(got)["limit"] == TIERS["max"]["limit"]


def test_authenticate_downgrades_expired_grant(store):
    store.create_user("auth@e.st", "pw", tier="pro", tier_expires_at=_past_iso())
    got = store.authenticate("auth@e.st", "pw")
    assert got is not None and got.tier == "free"


def test_expire_due_tiers_sweep(store):
    store.create_user("a@e.st", "pw", tier="max", tier_expires_at=_past_iso())
    store.create_user("b@e.st", "pw", tier="max", tier_expires_at=grant_expiry(30))
    store.create_user("c@e.st", "pw")  # plain free, no expiry
    assert store.expire_due_tiers() == 1  # only the lapsed grant
    assert dict(store.tier_distribution()) == {"free": 2, "max": 1}
    tiers = {u.email: u.tier for u in store.list_users()}
    assert tiers == {"a@e.st": "free", "b@e.st": "max", "c@e.st": "free"}


def test_set_tier_clears_expiry(store):
    u = store.create_user("man@e.st", "pw", tier="pro",
                          tier_expires_at=grant_expiry(7))
    got = store.set_tier(u.email, "max")  # manual assignment = indefinite
    assert got.tier == "max" and got.tier_expires_at is None


# ---- site settings ----------------------------------------------------------

def test_signup_defaults_closed(store):
    assert store.signup_open() is False  # invite-only until flipped at launch


def test_signup_open_roundtrip(store):
    store.set_signup_open(True)
    assert store.signup_open() is True
    store.set_signup_open(False)
    assert store.signup_open() is False


def test_get_setting_default(store):
    assert store.get_setting("missing") is None
    assert store.get_setting("missing", "fallback") == "fallback"
    store.set_setting("k", "v1")
    store.set_setting("k", "v2")  # upsert overwrites
    assert store.get_setting("k") == "v2"


def test_users_with_project_counts_left_join(store):
    a = store.create_user("a@e.st", "pw")
    store.create_user("zero@e.st", "pw")  # no projects
    _add_project(store, a.id, "ok", 2.50)
    _add_project(store, a.id, "ok", 1.50)
    rows = {r["email"]: r for r in store.users_with_project_counts()}
    assert rows["a@e.st"]["project_count"] == 2
    assert rows["a@e.st"]["spend_usd"] == pytest.approx(4.0)
    assert rows["a@e.st"]["last_project_at"] is not None
    assert rows["zero@e.st"]["project_count"] == 0  # LEFT JOIN keeps zero-project user
    assert rows["zero@e.st"]["spend_usd"] == 0


# ---- public browser: visibility, metrics, likes, clone --------------------

def _ok_public(store, user_id, brief="a board", stem="BOARD"):
    """Create + finish a public, completed project; return its id."""
    pid = store.create_project(user_id, brief)
    store.finish_project(pid, "ok", stem=stem)
    return pid


def test_new_project_columns_default(store):
    u = store.create_user("np@e.st", "pw")
    p = store.get_project(store.create_project(u.id, "x"))
    assert p.is_public is True               # public by default (community rule)
    assert p.cloned_from_id is None
    assert (p.view_count, p.clone_count, p.like_count) == (0, 0, 0)
    assert p.quality is None


def test_create_project_private_override(store):
    u = store.create_user("priv@e.st", "pw")
    pid = store.create_project(u.id, "x", is_public=False)  # paid private clone path
    assert store.get_project(pid).is_public is False


def test_project_visibility_follows_tier(store):
    free = store.create_user("ff@e.st", "pw")
    store.create_user("pp@e.st", "pw")
    paid = store.set_tier("pp@e.st", "pro")
    assert store.get_project(store.create_project(free.id, "x")).is_public is True
    assert store.get_project(store.create_project(paid.id, "x")).is_public is False


def test_set_visibility_roundtrips(store):
    u = store.create_user("vis@e.st", "pw")
    pid = _ok_public(store, u.id)
    store.set_visibility(pid, False)
    assert store.get_project(pid).is_public is False
    store.set_visibility(pid, True)
    assert store.get_project(pid).is_public is True


def test_record_view_increments(store):
    u = store.create_user("rv@e.st", "pw")
    pid = _ok_public(store, u.id)
    for _ in range(3):
        store.record_view(pid)
    assert store.get_project(pid).view_count == 3


def test_toggle_like_dedup_and_count(store):
    owner = store.create_user("o@e.st", "pw")
    a = store.create_user("la@e.st", "pw")
    b = store.create_user("lb@e.st", "pw")
    pid = _ok_public(store, owner.id)
    assert store.toggle_like(a.id, pid) is True        # like
    assert store.has_liked(a.id, pid) is True
    assert store.get_project(pid).like_count == 1
    assert store.toggle_like(a.id, pid) is False       # same user unlikes (dedup)
    assert store.has_liked(a.id, pid) is False
    assert store.get_project(pid).like_count == 0
    store.toggle_like(a.id, pid)                        # two distinct likers
    store.toggle_like(b.id, pid)
    assert store.get_project(pid).like_count == 2


def test_clone_from_and_increment_clone_count(store):
    u = store.create_user("cf@e.st", "pw")
    src = _ok_public(store, u.id)
    clone = store.create_project(u.id, "cloned")
    store.set_cloned_from(clone, src)
    store.increment_clone_count(src)
    store.increment_clone_count(src)
    assert store.get_project(clone).cloned_from_id == src
    assert store.get_project(src).clone_count == 2


def test_list_public_only_ok_and_public(store):
    u = store.create_user("lp@e.st", "pw")
    ok_pub = _ok_public(store, u.id, stem="OKPUB")
    priv = _ok_public(store, u.id, stem="PRIV")
    store.set_visibility(priv, False)                  # ok but private -> excluded
    store.finish_project(store.create_project(u.id, "f"), "failed")  # failed -> excluded
    store.create_project(u.id, "r")                    # running -> excluded
    rows = store.list_public_projects()
    assert [r["id"] for r in rows] == [ok_pub]
    assert rows[0]["owner_email"] == "lp@e.st"         # join surfaces the owner
    assert store.count_public_projects() == 1


def test_popularity_ordering(store):
    u = store.create_user("pop@e.st", "pw")
    liker = store.create_user("lk@e.st", "pw")
    a = _ok_public(store, u.id, stem="A")
    b = _ok_public(store, u.id, stem="B")
    c = _ok_public(store, u.id, stem="C")
    for _ in range(5):
        store.record_view(a)            # a score = 5  (view weight 1)
    store.increment_clone_count(b)      # b score = 4  (clone weight 4)
    store.toggle_like(liker.id, c)      # c score = 3  (like weight 3)
    order = [r["id"] for r in store.list_public_projects(sort="popularity")]
    assert order == [a, b, c]


def test_sort_new_and_clones(store):
    u = store.create_user("sn@e.st", "pw")
    first = _ok_public(store, u.id, stem="FIRST")
    second = _ok_public(store, u.id, stem="SECOND")
    assert store.list_public_projects(sort="new")[0]["id"] == second  # newest finish
    store.increment_clone_count(first)
    assert store.list_public_projects(sort="clones")[0]["id"] == first


def test_badge_filter(store):
    u = store.create_user("bd@e.st", "pw")
    fab = _ok_public(store, u.id, stem="FAB")
    store.set_quality(fab, "fab_ready")
    erc = _ok_public(store, u.id, stem="ERC")
    store.set_quality(erc, "erc_errors")
    assert [r["id"] for r in store.list_public_projects(badge="fab_ready")] == [fab]
    assert store.count_public_projects(badge="fab_ready") == 1


def test_migration_adds_project_columns_to_legacy_db(tmp_path):
    """A projects table created before the browser feature gains the columns on
    open; a free user's rows stay public, a paid user's rows backfill to private
    (no retroactive exposure of paid work). Idempotent on re-open."""
    db = tmp_path / "accounts.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,"
            "tier TEXT NOT NULL DEFAULT 'free', created_at TEXT NOT NULL,"
            "last_login_at TEXT)")
        conn.execute("INSERT INTO users (email, password_hash, tier, created_at) "
                     "VALUES ('free@e.st','scrypt$x','free','2026-01-01T00:00:00+00:00')")
        conn.execute("INSERT INTO users (email, password_hash, tier, created_at) "
                     "VALUES ('paid@e.st','scrypt$x','pro','2026-01-01T00:00:00+00:00')")
        conn.execute(  # pre-browser projects schema (no new columns)
            "CREATE TABLE projects (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "user_id INTEGER NOT NULL, brief TEXT NOT NULL, project_stem TEXT,"
            "status TEXT NOT NULL DEFAULT 'running', created_at TEXT NOT NULL,"
            "finished_at TEXT, cost_usd REAL, dir_path TEXT, zip_path TEXT)")
        conn.execute("INSERT INTO projects (user_id, brief, status, created_at) "
                     "VALUES (1,'free board','ok','2026-01-02T00:00:00+00:00')")
        conn.execute("INSERT INTO projects (user_id, brief, status, created_at) "
                     "VALUES (2,'paid board','ok','2026-01-02T00:00:00+00:00')")
    store = AccountStore(db, tmp_path / "projects")  # _ensure_project_columns migrates
    free_proj = store.list_projects(1)[0]
    paid_proj = store.list_projects(2)[0]
    assert free_proj.is_public is True       # free user's project stays public
    assert paid_proj.is_public is False      # paid user's existing project privatized
    assert free_proj.view_count == 0 and free_proj.quality is None  # defaults backfilled
    reopened = AccountStore(db, tmp_path / "projects")  # idempotent
    assert reopened.list_projects(2)[0].is_public is False


def test_delete_user_purges_likes_and_keeps_others(store):
    owner = store.create_user("own@e.st", "pw")
    other = store.create_user("oth@e.st", "pw")
    pid = _ok_public(store, owner.id)
    other_pid = _ok_public(store, other.id, stem="OTHER")
    store.toggle_like(other.id, pid)        # other likes owner's project
    store.toggle_like(owner.id, other_pid)  # owner likes other's project
    store.delete_user(owner.id)
    assert store.get_project(pid) is None                 # owner's project gone
    assert store.has_liked(owner.id, other_pid) is False  # owner's like cleaned up
    assert store.get_project(other_pid) is not None       # other's project intact


# ---- board ids + support reports -------------------------------------------

_BOARD_CODE_RE = re.compile(r"KC-[2-9A-HJKMNP-Z]{6}\Z")


def test_board_code_format():
    for _ in range(50):
        assert _BOARD_CODE_RE.fullmatch(new_board_code())


def test_new_projects_get_unique_board_codes(store):
    u = store.create_user("bc@e.st", "pw")
    p1 = store.get_project(store.create_project(u.id, "one"))
    p2 = store.get_project(store.create_project(u.id, "two"))
    assert _BOARD_CODE_RE.fullmatch(p1.board_code)
    assert _BOARD_CODE_RE.fullmatch(p2.board_code)
    assert p1.board_code != p2.board_code
    assert {p.board_code for p in store.list_projects(u.id)} \
        == {p1.board_code, p2.board_code}


def test_board_code_backfilled_on_migration(tmp_path):
    """A DB from before board ids gains a unique code on every existing project."""
    db = tmp_path / "accounts.db"
    with sqlite3.connect(db) as conn:  # pre-board_code projects schema
        conn.execute(
            "CREATE TABLE projects (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "user_id INTEGER NOT NULL, brief TEXT NOT NULL, project_stem TEXT,"
            "status TEXT NOT NULL DEFAULT 'running', created_at TEXT NOT NULL,"
            "finished_at TEXT, cost_usd REAL, dir_path TEXT, zip_path TEXT)")
        for brief in ("old one", "old two"):
            conn.execute(
                "INSERT INTO projects (user_id, brief, status, created_at) "
                "VALUES (1, ?, 'failed', '2026-01-01T00:00:00+00:00')", (brief,))
    store = AccountStore(db, tmp_path / "projects")  # migrates + backfills
    codes = [p.board_code for p in store.list_projects(1)]
    assert len(codes) == 2 and len(set(codes)) == 2
    assert all(_BOARD_CODE_RE.fullmatch(c) for c in codes)
    reopened = AccountStore(db, tmp_path / "projects")  # idempotent: codes stable
    assert [p.board_code for p in reopened.list_projects(1)] == codes


def test_support_report_roundtrip(store):
    u = store.create_user("sup@e.st", "pw")
    pid = store.create_project(u.id, "blinky")
    p = store.get_project(pid)
    rid = store.create_support_report(
        user_id=u.id, project_id=pid, board_code=p.board_code,
        kind="error_auto", diagnostics={"build_log_tail": ["boom"]})
    new = store.list_support_reports(status="new")
    assert [r.id for r in new] == [rid]
    r = new[0]
    assert r.kind == "error_auto" and r.status == "new"
    assert r.board_code == p.board_code and r.project_id == pid
    assert r.message is None
    assert r.diagnostics == {"build_log_tail": ["boom"]}
    store.set_support_report_message(rid, "it broke while routing")
    assert store.list_support_reports()[0].message == "it broke while routing"
    store.set_support_report_status(rid, "reviewed")
    assert store.list_support_reports(status="new") == []
    assert store.list_support_reports(status="reviewed")[0].id == rid


def test_support_reports_exported_and_purged_with_user(store):
    u = store.create_user("priv@e.st", "pw")
    other = store.create_user("other@e.st", "pw")
    rid = store.create_support_report(user_id=u.id, kind="user", message="help")
    other_rid = store.create_support_report(user_id=other.id, kind="user")
    data = store.export_user(u.id)
    assert [r["id"] for r in data["support_reports"]] == [rid]
    store.delete_user(u.id)
    remaining = store.list_support_reports()
    assert [r.id for r in remaining] == [other_rid]  # other users' reports intact
