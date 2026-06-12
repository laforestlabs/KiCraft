"""Tests for the host build queue: flock slots, the build_jobs table, the
standalone worker, and walk-away notifications.

Pure stdlib + sqlite (no pcbnew, no network); the worker tests run tiny Python
one-liners as the "build", so they exercise the real subprocess/log/status
machinery in well under a second each.
"""
from __future__ import annotations

import datetime as dt
import os
import sqlite3
import sys
import threading
import time

import pytest

import kicraft.build_slots as build_slots
from kicraft.build_slots import ACQUIRED_MARKER, WAITING_MARKER, build_slot, slot_count
from kicraft.server.accounts import AccountStore
from kicraft.server.build_worker import BuildWorker
from kicraft.server import notify
from kicraft.server.config import Settings


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


@pytest.fixture
def slots_env(tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS_DIR", str(tmp_path / "slots"))
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "1")
    monkeypatch.setattr(build_slots, "_POLL_S", 0.05)
    return tmp_path / "slots"


# --------------------------------------------------------------------------- #
# build_slots
# --------------------------------------------------------------------------- #
def test_slot_count_env(monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "4")
    assert slot_count() == 4
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "0")
    assert slot_count() == 0
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "garbage")
    assert slot_count() == max(1, (os.cpu_count() or 1) // 6)
    monkeypatch.delenv("KICRAFT_BUILD_SLOTS")
    assert slot_count() >= 1


def test_slot_disabled_yields_none(monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "0")
    with build_slot() as idx:
        assert idx is None


def test_slot_acquire_emits_marker(slots_env):
    lines = []
    with build_slot(echo=lines.append) as idx:
        assert idx == 0
    assert any(ACQUIRED_MARKER in line for line in lines)
    assert not any(WAITING_MARKER in line for line in lines)  # uncontended


def test_slot_contention_waits_and_recovers(slots_env):
    """With 1 slot, a second acquirer emits the waiting marker and only enters
    after the holder releases."""
    order = []
    release = threading.Event()
    held = threading.Event()

    def holder():
        with build_slot():
            held.set()
            release.wait(5)
        order.append("released")

    def waiter_lines(line):
        if WAITING_MARKER in line:
            order.append("waited")

    t = threading.Thread(target=holder)
    t.start()
    assert held.wait(5)
    got = {}

    def waiter():
        with build_slot(echo=waiter_lines) as idx:
            got["idx"] = idx
            order.append("acquired")

    w = threading.Thread(target=waiter)
    w.start()
    time.sleep(0.3)  # long enough for at least one failed poll
    release.set()
    w.join(5)
    t.join(5)
    assert got.get("idx") == 0
    assert "waited" in order
    assert order.index("waited") < order.index("acquired")


# --------------------------------------------------------------------------- #
# accounts: build_jobs
# --------------------------------------------------------------------------- #
def test_enqueue_claim_fifo(store):
    a = store.enqueue_build(workspace="/ws/a")
    b = store.enqueue_build(workspace="/ws/b")
    j = store.claim_next_build("pid:1")
    assert j.id == a and j.status == "running" and j.attempts == 1
    assert store.claim_next_build("pid:1").id == b
    assert store.claim_next_build("pid:1") is None


def test_claim_is_atomic(store):
    j = store.enqueue_build(workspace="/ws")
    assert store.claim_build(j, "pid:1") is True
    assert store.claim_build(j, "pid:2") is False  # already running


def test_queue_position_and_counts(store):
    a = store.enqueue_build(workspace="/a")
    b = store.enqueue_build(workspace="/b")
    c = store.enqueue_build(workspace="/c")
    assert store.build_queue_position(c) == (2, 3, 0)
    store.claim_build(a, "pid:1")
    assert store.build_queue_position(c) == (1, 2, 1)
    assert store.count_running_builds() == 1
    store.finish_build(a, rc=0)
    assert store.build_queue_position(b) == (0, 2, 0)


def test_avg_build_seconds(store):
    assert store.avg_build_seconds() is None
    j = store.enqueue_build(workspace="/ws")
    store.claim_build(j, "pid:1")
    # Backdate started_at so the duration is a known ~60s.
    past = (dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=60)).isoformat()
    with sqlite3.connect(store.path) as conn:
        conn.execute("UPDATE build_jobs SET started_at=? WHERE id=?", (past, j))
    store.finish_build(j, rc=0)
    avg = store.avg_build_seconds()
    assert avg is not None and 55 <= avg <= 65


def test_requeue_stale_builds(store):
    dead = store.enqueue_build(workspace="/dead")
    live = store.enqueue_build(workspace="/live")
    store.claim_build(dead, "pid:999999999")  # certainly not a live pid
    store.claim_build(live, f"pid:{os.getpid()}")
    assert store.requeue_stale_builds() == 1
    assert store.get_build_job(dead).status == "queued"
    assert store.get_build_job(live).status == "running"
    # Second dead claim exhausts the attempt budget -> failed, not requeued.
    store.claim_build(dead, "pid:999999999")
    assert store.get_build_job(dead).attempts == 2
    assert store.requeue_stale_builds() == 1
    assert store.get_build_job(dead).status == "failed"


def test_list_unfinalized_builds(store):
    u = store.create_user("o@example.com", "pw12345678")
    pid = store.create_project(u.id, "a board")
    j = store.enqueue_build(workspace="/ws", project_id=pid, user_id=u.id)
    store.claim_build(j, "pid:1")
    assert store.list_unfinalized_builds() == []  # running job: leave it alone
    store.finish_build(j, rc=0)
    orphans = store.list_unfinalized_builds()
    assert [o.id for o in orphans] == [j]
    store.finish_project(pid, "ok")
    assert store.list_unfinalized_builds() == []  # project finalized


def test_worker_heartbeat(store):
    assert store.build_worker_alive() is False
    store.beat_build_worker()
    assert store.build_worker_alive() is True
    assert store.build_worker_alive(max_age_s=0.0) is False


def test_notify_email_pref_roundtrip(store):
    u = store.create_user("n@example.com", "pw12345678")
    assert u.notify_email is True
    store.set_notify_email(u.id, False)
    assert store.get_user(u.id).notify_email is False


def test_notify_email_migration(tmp_path):
    """A pre-existing users table without the column gains it (default on)."""
    db = tmp_path / "old.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE users (id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "email TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL,"
            "tier TEXT NOT NULL DEFAULT 'free', created_at TEXT NOT NULL,"
            "last_login_at TEXT)")
        conn.execute("INSERT INTO users (email, password_hash, tier, created_at) "
                     "VALUES ('old@example.com', 'x', 'free', '2026-01-01')")
    s = AccountStore(db, tmp_path / "projects")
    assert s.get_user_by_email("old@example.com").notify_email is True


# --------------------------------------------------------------------------- #
# build worker
# --------------------------------------------------------------------------- #
def _ws(tmp_path, name="ws"):
    ws = tmp_path / name
    (ws / ".kicraft").mkdir(parents=True)
    (ws / ".kicraft" / "state.json").write_text("{}", encoding="utf-8")
    return ws


def _drain(worker):
    for t in worker._threads:
        t.join(timeout=10)


def test_worker_runs_job_and_logs(store, tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "0")  # no host gating in tests
    ws = _ws(tmp_path)
    j = store.enqueue_build(workspace=str(ws))
    w = BuildWorker(store, build_cmd=[sys.executable, "-c",
                                      "print('hello from build')"],
                    poll_s=0.05, max_jobs=1)
    assert w.run_once() is True
    _drain(w)
    job = store.get_build_job(j)
    assert job.status == "done" and job.rc == 0
    assert "hello from build" in (ws / ".kicraft" / "build.log").read_text()


def test_worker_records_nonzero_rc(store, tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "0")
    ws = _ws(tmp_path)
    j = store.enqueue_build(workspace=str(ws))
    w = BuildWorker(store, build_cmd=[sys.executable, "-c", "raise SystemExit(7)"],
                    poll_s=0.05)
    w.run_once()
    _drain(w)
    job = store.get_build_job(j)
    assert job.status == "done" and job.rc == 7


def test_worker_missing_workspace_fails_job(store, tmp_path):
    j = store.enqueue_build(workspace=str(tmp_path / "nope"))
    w = BuildWorker(store, build_cmd=["true"], poll_s=0.05)
    w.run_once()
    _drain(w)
    job = store.get_build_job(j)
    assert job.status == "failed" and job.rc is None


def test_worker_timeout_kills_silent_build(store, tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "0")
    ws = _ws(tmp_path)
    j = store.enqueue_build(workspace=str(ws))
    w = BuildWorker(store, build_cmd=[sys.executable, "-c",
                                      "import time; time.sleep(60)"],
                    timeout_s=0.4, poll_s=0.05)
    t0 = time.monotonic()
    w.run_once()
    _drain(w)
    assert time.monotonic() - t0 < 10
    job = store.get_build_job(j)
    assert job.status == "done" and job.rc not in (0, None)
    assert "killed" in (ws / ".kicraft" / "build.log").read_text()


def test_worker_shutdown_requeues(store, tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "0")
    ws = _ws(tmp_path)
    j = store.enqueue_build(workspace=str(ws))
    w = BuildWorker(store, build_cmd=[sys.executable, "-c",
                                      "import time; print('up', flush=True); "
                                      "time.sleep(60)"],
                    poll_s=0.05)
    w.run_once()
    deadline = time.monotonic() + 5
    while store.get_build_job(j).status != "running" or j not in w._procs:
        assert time.monotonic() < deadline, "build never registered"
        time.sleep(0.05)
    w._shutdown()
    assert store.get_build_job(j).status == "queued"


# --------------------------------------------------------------------------- #
# notify
# --------------------------------------------------------------------------- #
def _settings():
    return Settings(api_key="test", public_url="https://kicraft.io")


@pytest.fixture
def sent(monkeypatch):
    calls = []
    monkeypatch.setattr(notify, "send_email",
                        lambda settings, to, subject, body:
                        calls.append((to, subject, body)) or True)
    # Each test starts with a clean activity map.
    monkeypatch.setattr(notify, "_last_seen", {})
    return calls


def test_notify_sends_when_user_away(store, sent):
    u = store.create_user("away@example.com", "pw12345678")
    ok = notify.notify_run_event(store, _settings(), user_id=u.id, project_id=7,
                                 status="ok", brief="a tiny 555 blinker")
    assert ok and len(sent) == 1
    to, subject, body = sent[0]
    assert to == "away@example.com"
    assert "ready" in subject and "555" in subject
    assert "https://kicraft.io/?project=7" in body


def test_notify_question_subject(store, sent):
    u = store.create_user("q@example.com", "pw12345678")
    notify.notify_run_event(store, _settings(), user_id=u.id, project_id=1,
                            status="awaiting_input", brief="cat feeder")
    assert "question" in sent[0][1]


def test_notify_suppressed_while_watching(store, sent):
    u = store.create_user("here@example.com", "pw12345678")
    notify.mark_active(u.id)
    assert not notify.notify_run_event(store, _settings(), user_id=u.id,
                                       project_id=1, status="ok")
    assert sent == []
    # The restart-recovery sweep forces past the activity check.
    assert notify.notify_run_event(store, _settings(), user_id=u.id, project_id=1,
                                   status="ok", skip_if_active=False)
    assert len(sent) == 1


def test_notify_respects_opt_out(store, sent):
    u = store.create_user("optout@example.com", "pw12345678")
    store.set_notify_email(u.id, False)
    assert not notify.notify_run_event(store, _settings(), user_id=u.id,
                                       project_id=1, status="ok")
    assert sent == []


def test_notify_ignores_other_statuses(store, sent):
    u = store.create_user("s@example.com", "pw12345678")
    assert not notify.notify_run_event(store, _settings(), user_id=u.id,
                                       project_id=1, status="running")
    assert sent == []


# --------------------------------------------------------------------------- #
# web log drain (pure helper; importing web pulls nicegui, which the web tests
# already depend on)
# --------------------------------------------------------------------------- #
def test_drain_build_log_partial_lines(tmp_path):
    from kicraft.server.web import _drain_build_log

    log = tmp_path / "build.log"
    events = []
    progress = events.append
    offset, rem = _drain_build_log(log, 0, "", progress)
    assert (offset, rem) == (0, "") and events == []  # file not created yet
    log.write_text("one\ntwo\npart")
    offset, rem = _drain_build_log(log, offset, rem, progress)
    assert [e["text"] for e in events] == ["one", "two"] and rem == "part"
    with log.open("a") as f:
        f.write("ial\n")
    offset, rem = _drain_build_log(log, offset, rem, progress)
    assert [e["text"] for e in events] == ["one", "two", "partial"] and rem == ""


# --------------------------------------------------------------------------- #
# job kinds (manual_route)
# --------------------------------------------------------------------------- #
def test_enqueue_kind_round_trips_and_defaults(store, tmp_path):
    ws = _ws(tmp_path)
    j_default = store.enqueue_build(workspace=str(ws))
    j_manual = store.enqueue_build(workspace=str(ws), kind="manual_route")
    assert store.get_build_job(j_default).kind == "build"
    assert store.get_build_job(j_manual).kind == "manual_route"


def test_worker_dispatches_command_by_kind(store, tmp_path, monkeypatch):
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "0")
    ws = _ws(tmp_path)
    j = store.enqueue_build(workspace=str(ws), kind="manual_route")
    w = BuildWorker(
        store, poll_s=0.05, max_jobs=1,
        commands={
            "build": [sys.executable, "-c", "print('wrong command')"],
            "manual_route": [sys.executable, "-c", "print('manual route ran')"],
        })
    assert w.run_once() is True
    _drain(w)
    job = store.get_build_job(j)
    assert job.status == "done" and job.rc == 0
    log = (ws / ".kicraft" / "build.log").read_text()
    assert "manual route ran" in log
    assert "wrong command" not in log


def test_worker_fails_unknown_kind_instead_of_running_build(
        store, tmp_path, monkeypatch):
    """Deploy-skew safety: an old worker that does not know a job kind
    must fail the job, never fall back to the 'build' command."""
    monkeypatch.setenv("KICRAFT_BUILD_SLOTS", "0")
    ws = _ws(tmp_path)
    j = store.enqueue_build(workspace=str(ws), kind="frobnicate")
    w = BuildWorker(store, poll_s=0.05, max_jobs=1,
                    commands={"build": [sys.executable, "-c", "print('nope')"]})
    assert w.run_once() is True
    _drain(w)
    job = store.get_build_job(j)
    assert job.status == "failed"
    assert "nope" not in (
        (ws / ".kicraft" / "build.log").read_text()
        if (ws / ".kicraft" / "build.log").is_file() else "")


def test_kind_column_migrates_legacy_db(tmp_path):
    """A deployed build_jobs table without the kind column upgrades in
    place; pre-existing rows read back as kind='build'."""
    db = tmp_path / "accounts.db"
    with sqlite3.connect(db) as conn:
        conn.execute(
            "CREATE TABLE build_jobs ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT, project_id INTEGER,"
            "user_id INTEGER, workspace TEXT NOT NULL,"
            "status TEXT NOT NULL DEFAULT 'queued', rc INTEGER,"
            "created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,"
            "attempts INTEGER NOT NULL DEFAULT 0, claimed_by TEXT,"
            "log_path TEXT)")
        conn.execute(
            "INSERT INTO build_jobs (workspace, status, created_at)"
            " VALUES ('/tmp/ws', 'queued', '2026-01-01T00:00:00')")

    store = AccountStore(db, tmp_path / "projects")  # runs the migration
    job = store.get_build_job(1)
    assert job is not None and job.kind == "build"
