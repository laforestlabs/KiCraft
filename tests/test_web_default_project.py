"""The workspace must come back to the design that needs the user.

Regression tests for two reports from a first-run user:

1. Kicking off a design, clicking through to /parts, and returning to "/" landed
   on a blank composer with no way back to the in-flight run. Fix: a process-wide
   live-run registry (``web._LIVE_RUNS``) that pages re-attach through, plus a
   ``projects.viewed_at`` marker so a finished-but-unseen result is auto-opened.
   ``_pick_default_project`` encodes the priority: parked run (blocked on the
   user) > live run > newest unseen finished result > blank composer.

2. The "Open" button only existed once a run had PERSISTED artifacts (dir_path,
   written at the END of a run), so an early-stage project had no Open button
   until a reload caught it later. Fix: Open also shows while the run is live in
   the registry -- pinned here at the registry level (UI gating reads it).

Pure store + module-function tests (no NiceGUI client, no network, no build).
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import time

import pytest

from kicraft.server import web
from kicraft.server.accounts import AccountStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
    monkeypatch.setattr(web, "_STORE", s)  # _store() returns this instance
    return s


@pytest.fixture
def user_id(store):
    return store.create_user("dev@example.com", "hunter2hunter2").id


@pytest.fixture
def live_runs(monkeypatch):
    runs: dict = {}
    monkeypatch.setattr(web, "_LIVE_RUNS", runs)
    return runs


# ---- viewed_at lifecycle (accounts) ----------------------------------------


def test_new_and_finished_projects_track_viewed_at(store, user_id):
    pid = store.create_project(user_id, "usb battery bank")
    assert store.get_project(pid).viewed_at is None

    store.finish_project(pid, "ok", stem="BANK", dir_path="/tmp/x")
    assert store.get_project(pid).viewed_at is None, "finishing is not seeing"

    store.mark_viewed(pid)
    assert store.get_project(pid).viewed_at is not None

    # Re-running makes a new result: the seen-marker must reset so the
    # workspace auto-opens the eventual outcome again.
    store.update_project_status(pid, "running")
    assert store.get_project(pid).viewed_at is None


def test_viewed_at_migration_backfills_already_finished_rows(tmp_path):
    """A deployed DB upgrades in place: finished rows count as already seen
    (no surprise auto-open of months-old projects), unfinished rows stay NULL."""
    db = tmp_path / "accounts.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE projects ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,"
            "brief TEXT NOT NULL, project_stem TEXT,"
            "status TEXT NOT NULL DEFAULT 'running', created_at TEXT NOT NULL,"
            "finished_at TEXT, cost_usd REAL, dir_path TEXT, zip_path TEXT,"
            "is_public INTEGER NOT NULL DEFAULT 1, cloned_from_id INTEGER,"
            "view_count INTEGER NOT NULL DEFAULT 0,"
            "clone_count INTEGER NOT NULL DEFAULT 0,"
            "like_count INTEGER NOT NULL DEFAULT 0, quality TEXT)")
        conn.execute(
            "INSERT INTO projects (user_id, brief, status, created_at, finished_at)"
            " VALUES (1, 'old done', 'ok', '2026-01-01T00:00:00', "
            "'2026-01-01T01:00:00')")
        conn.execute(
            "INSERT INTO projects (user_id, brief, status, created_at)"
            " VALUES (1, 'old running', 'running', '2026-01-02T00:00:00')")

    store = AccountStore(db, tmp_path / "projects")  # runs the migration
    done, running = store.get_project(1), store.get_project(2)
    assert done.viewed_at == "2026-01-01T01:00:00"
    assert running.viewed_at is None


# ---- default-project selection ----------------------------------------------


def test_blank_composer_when_nothing_needs_attention(store, user_id, live_runs):
    assert web._pick_default_project(user_id) is None

    pid = store.create_project(user_id, "seen already")
    store.finish_project(pid, "ok", dir_path="/tmp/x")
    store.mark_viewed(pid)
    assert web._pick_default_project(user_id) is None


def test_live_run_wins_over_unseen_finished(store, user_id, live_runs):
    done = store.create_project(user_id, "finished, unseen")
    store.finish_project(done, "ok", dir_path="/tmp/x")

    running = store.create_project(user_id, "usb battery bank")
    live_runs[running] = {"running": True, "user_id": user_id}

    assert web._pick_default_project(user_id).id == running


def test_parked_question_outranks_live_run(store, user_id, live_runs):
    running = store.create_project(user_id, "still going")
    live_runs[running] = {"running": True, "user_id": user_id}

    parked = store.create_project(user_id, "needs an answer")
    store.finish_project(parked, "awaiting_input", dir_path="/tmp/parked")

    assert web._pick_default_project(user_id).id == parked


def test_unseen_finished_opens_then_stops_once_viewed(store, user_id, live_runs):
    pid = store.create_project(user_id, "fresh result")
    store.finish_project(pid, "failed", dir_path="/tmp/x")

    assert web._pick_default_project(user_id).id == pid
    store.mark_viewed(pid)
    assert web._pick_default_project(user_id) is None


def test_orphaned_running_row_is_skipped(store, user_id, live_runs):
    """A 'running' row whose worker died with the server must not auto-open a
    blank shell: with no live state and no artifacts there is nothing to show."""
    store.create_project(user_id, "lost to a restart")
    assert web._pick_default_project(user_id) is None


# ---- orphan reconciliation (_reconcile_orphan_projects) ----------------------


def _backdate_secs(store, project_id, secs_ago):
    ts = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=secs_ago)).isoformat()
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE projects SET created_at=? WHERE id=?", (ts, project_id))


def test_reconcile_closes_early_orphan_and_frees_quota(store, user_id, live_runs):
    """A run lost during the LLM stages (old, no build_jobs row, not live) is
    marked 'interrupted', which both ends the phantom and frees its quota slot."""
    user = store.get_user(user_id)
    before = store.quota_status(user)["remaining"]

    pid = store.create_project(user_id, "lost to a restart")
    _backdate_secs(store, pid, 300)
    assert store.quota_status(user)["remaining"] == before - 1  # slot consumed

    web._reconcile_orphan_projects()

    p = store.get_project(pid)
    assert p.status == "interrupted"
    assert p.finished_at is not None
    assert p.brief == "lost to a restart"  # preserved, so Retry can reuse it
    assert store.quota_status(user)["remaining"] == before  # slot freed


def test_reconcile_leaves_live_recent_and_build_stage_runs(store, user_id, live_runs):
    """The sweep must not touch a healthy live run, a just-started run inside the
    registration window, or a run that already reached the build queue (the
    build-job reaper owns that one and may still recover artifacts)."""
    live = store.create_project(user_id, "still running here")
    _backdate_secs(store, live, 300)
    live_runs[live] = {"running": True, "user_id": user_id}

    recent = store.create_project(user_id, "just started")  # within the age floor

    build_stage = store.create_project(user_id, "reached the build")
    _backdate_secs(store, build_stage, 300)
    store.enqueue_build(workspace="/ws", project_id=build_stage, user_id=user_id)

    web._reconcile_orphan_projects()

    assert store.get_project(live).status == "running"
    assert store.get_project(recent).status == "running"
    assert store.get_project(build_stage).status == "running"


# ---- live-run registry around _run_design ------------------------------------


def _drive(monkeypatch, state, session_result):
    """Run _run_design with the LLM session and persistence stubbed out, while
    recording whether the run was registered DURING the session (that is what
    makes the Open button exist from the first second of a run)."""
    seen = {}

    def fake_run_session(ws, brief, stages, **kw):
        seen["registered_during_run"] = web._LIVE_RUNS.get(state["project_id"]) is state
        return session_result

    monkeypatch.setattr(web, "run_session", fake_run_session)
    monkeypatch.setattr(web, "_persist_project", lambda st: None)
    web._run_design(state, ["intent"])
    return seen


def test_terminal_run_registers_then_evicts(tmp_path, live_runs, monkeypatch):
    state = web._fresh_run_state()
    state.update(project_id=7, user_id=1, ws=str(tmp_path))
    seen = _drive(monkeypatch, state, {"status": "error"})

    assert seen["registered_during_run"], "Open must work while the run is live"
    assert 7 not in web._LIVE_RUNS, "terminal runs hand over to the saved project"
    assert state["done"] and not state["running"]


def test_parked_run_stays_registered(tmp_path, live_runs, monkeypatch):
    state = web._fresh_run_state()
    state.update(project_id=8, user_id=1, ws=str(tmp_path))
    _drive(monkeypatch, state,
           {"status": "awaiting_input", "questions": [{"text": "AA or 18650?"}]})

    assert web._LIVE_RUNS.get(8) is state, \
        "a parked run must stay attachable so any page can answer it"
    assert state["awaiting_input"] and state["questions"]


def test_old_worker_cannot_evict_newer_run(tmp_path, live_runs, monkeypatch):
    """If a rerun of the same project registered a newer state dict, the old
    worker finishing must not knock the live run out of the registry."""
    old = web._fresh_run_state()
    old.update(project_id=9, user_id=1, ws=str(tmp_path))
    newer = web._fresh_run_state()

    def fake_run_session(ws, brief, stages, **kw):
        web._LIVE_RUNS[9] = newer  # a second run took over mid-flight
        return {"status": "error"}

    monkeypatch.setattr(web, "run_session", fake_run_session)
    monkeypatch.setattr(web, "_persist_project", lambda st: None)
    web._run_design(old, ["intent"])
    assert web._LIVE_RUNS.get(9) is newer


# ---- BOM price fetch actually starts (dead-thread regression) -----------------


def test_ensure_bom_prices_fetches_in_background(monkeypatch):
    """_ensure_bom_prices defined its worker but never started the thread (the
    start line sat unreachable in _price_for_lcsc), so live BOM pricing silently
    hung at '...' forever. Pin that the fetch runs and bumps prices_rev."""
    key = "kw:test-part-thread-regression"
    # _price_key now lives in kicraft.server.pricing; _ensure_bom_prices calls it
    # via web's namespace, so patch it there to pin the cache key deterministically.
    monkeypatch.setattr(web, "_price_key", lambda p: "kw:test-part-thread-regression")
    monkeypatch.setattr(web, "_safe_fetch",
                        lambda k: {"unit_price": 0.5, "lcsc": "C1", "stock": 1})
    with web._PRICE_LOCK:
        web._PRICE_CACHE.pop(key, None)
        web._PRICE_INFLIGHT.discard(key)

    state = {"prices_rev": 0}
    web._ensure_bom_prices([{"value": "anything"}], None, state)
    deadline = time.monotonic() + 5
    while state["prices_rev"] == 0 and time.monotonic() < deadline:
        time.sleep(0.01)

    assert state["prices_rev"] == 1, "price worker never ran"
    with web._PRICE_LOCK:
        assert web._PRICE_CACHE.get(key) == {"unit_price": 0.5, "lcsc": "C1",
                                             "stock": 1}
        assert key not in web._PRICE_INFLIGHT
