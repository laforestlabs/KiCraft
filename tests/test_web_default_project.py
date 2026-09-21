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
import json
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


# ---- project presentation (web._project_presentation) ------------------------


def _mk_project(store, user_id, *, status="running", brief="usb battery bank"):
    pid = store.create_project(user_id, brief)
    if status != "running":
        store.finish_project(pid, status)
    return store.get_project(pid)


def _with_root(store, project, *, state=None, zip_bytes=b"zip"):
    """Give a project a real durable root: state.json, optionally a fab zip."""
    root = store.projects_dir / str(project.user_id) / str(project.id)
    if isinstance(state, dict):
        (root / ".kicraft").mkdir(parents=True, exist_ok=True)
        (root / ".kicraft" / "state.json").write_text(json.dumps(state), encoding="utf-8")
    store.finish_project(project.id, project.status, stem=project.project_stem,
                         dir_path=str(root),
                         zip_path=(str(root / "kicraft_project.zip")
                                   if zip_bytes else None))
    if zip_bytes:
        root.mkdir(parents=True, exist_ok=True)
        (root / "kicraft_project.zip").write_bytes(zip_bytes)
    return store.get_project(project.id)


def test_presentation_running_attempt_owns_its_activity(store, user_id):
    p = _mk_project(store, user_id)
    live = {"running": True, "activity": {
        "stage": "bom", "phase_status": "running", "started_at": "2026-01-01T00:00:00+00:00",
        "last_activity": "Using lookup_lcsc"}}
    pres = web._project_presentation(p, live=live)
    assert pres["status"] == "running"
    assert pres["stage"] == "bom"
    assert pres["headline"] == "Choosing components"
    assert pres["action"] is None  # nothing to ask of the user while it runs
    assert pres["download_ready"] is False


def test_presentation_running_row_with_no_work_is_interrupted(store, user_id):
    """Rule 4: a running row with neither a live attempt nor a build job is lost."""
    p = _mk_project(store, user_id)
    pres = web._project_presentation(p, live=None, job=None)
    assert pres["status"] == "interrupted"
    assert pres["headline"] == "Run was interrupted"


def test_presentation_queued_and_running_jobs_are_not_interrupted(store, user_id):
    """A build the worker drives outlives the web process: without _LIVE_RUNS it
    must still read as genuine work, never as an interrupted run."""
    p = _mk_project(store, user_id)
    _with_root(store, p)
    ws = str(store.projects_dir / str(user_id) / str(p.id))
    job_id = store.enqueue_build(workspace=ws, project_id=p.id, user_id=user_id)
    job = store.get_build_job(job_id)

    pres = web._project_presentation(store.get_project(p.id), job=job)
    assert pres["status"] == "queued"
    assert pres["headline"] == "Queued for board build"
    assert pres["detail"]  # position / approximate ETA
    assert pres["download_ready"] is False

    store.claim_build(job_id, "pid:1")
    pres = web._project_presentation(store.get_project(p.id), job=store.get_build_job(job_id))
    assert pres["status"] == "running"

    store.finish_build(job_id, rc=0)
    pres = web._project_presentation(store.get_project(p.id), job=store.get_build_job(job_id))
    assert pres["status"] == "finalizing"


def test_presentation_blocks_duplicate_execution_from_a_surviving_job(store, user_id):
    """`_project_run_live` must refuse a second rebuild while this project's
    newest job is queued/running, even with an empty live registry."""
    p = _mk_project(store, user_id)
    ws = str(store.projects_dir / str(user_id) / str(p.id))
    store.enqueue_build(workspace=ws, project_id=p.id, user_id=user_id)
    state = web._fresh_run_state()
    state.update(project_id=p.id, user_id=user_id)
    assert web._project_run_live(state) is True


def test_presentation_ok_row_without_a_package_is_unavailable(store, user_id):
    """Never a false Download: an `ok` row whose package is gone says so."""
    p = _mk_project(store, user_id, status="ok")
    pres = web._project_presentation(p)
    assert pres["status"] == "unavailable"
    assert pres["headline"] == "Files unavailable"
    assert pres["download_ready"] is False
    assert pres["zip_path"] is None


def test_presentation_complete_offers_only_the_current_package(store, user_id):
    p = _mk_project(store, user_id, status="ok")
    state = {"project_stem": "USB_BANK",
             "stage_status": {k: {"ok": True} for k in web.DESIGN_STAGES},
             "artifacts": {"build_warnings": ["minor courtyard clip"]}}
    p = _with_root(store, p, state=state)
    root = store.projects_dir / str(user_id) / str(p.id)
    gen = root / "generated" / "USB_BANK"
    (gen / "fab").mkdir(parents=True, exist_ok=True)
    (gen / "USB_BANK.kicad_sch").write_text("(kicad_sch)", encoding="utf-8")
    (gen / "USB_BANK.kicad_pcb").write_text("(kicad_pcb)", encoding="utf-8")

    pres = web._project_presentation(store.get_project(p.id))
    # The fab package exists but was never produced by a recorded ok attempt with
    # zip_path pointing at it -> still no download.
    assert pres["status"] in ("complete", "complete_with_warnings")
    assert pres["download_ready"] is True
    assert pres["zip_path"] == str(root / "kicraft_project.zip")
    assert pres["title"] == "USB_BANK"
    assert any(i["severity"] == "warning" for i in pres["issues"])


def test_presentation_failed_row_never_offers_an_old_package(store, user_id):
    p = _mk_project(store, user_id, status="failed")
    p = _with_root(store, p, zip_bytes=b"old package")
    pres = web._project_presentation(p)
    assert pres["status"] == "failed"
    assert pres["download_ready"] is False
    assert pres["zip_path"] is None


def test_presentation_parked_live_run_is_not_running(store, user_id):
    p = _mk_project(store, user_id, status="awaiting_input")
    live = {"awaiting_input": True, "running": False,
            "questions": [{"stage": "wiring", "text": "which connector?"}]}
    pres = web._project_presentation(p, live=live)
    assert pres["status"] == "awaiting_input"
    assert pres["stage"] == "wiring"
    assert pres["headline"] == "Waiting for your answer"
    assert pres["action"] == "answer"


def test_presentation_actions_follow_real_prerequisites(store, user_id):
    """Continue only while design stages remain; Rebuild once they are committed;
    start-over only when there is no recoverable workspace."""
    p = _mk_project(store, user_id, status="failed")
    untouched = web._project_presentation(p)
    assert untouched["action"] == "new_from_brief"

    partial = _with_root(
        store, _mk_project(store, user_id, status="failed"),
        state={"project_stem": "PARTIAL", "stage_status": {"intent": {"ok": True}}})
    assert web._project_presentation(partial)["action"] == "continue"

    committed = _with_root(
        store, _mk_project(store, user_id, status="failed"),
        state={"project_stem": "COMMITTED",
               "stage_status": {k: {"ok": True} for k in web.DESIGN_STAGES}})
    assert web._project_presentation(committed)["action"] == "rebuild"


def test_presentation_names_a_null_stem_project_from_its_brief(store, user_id):
    p = _mk_project(store, user_id, brief="USB-C temperature logger")
    pres = web._project_presentation(p)
    assert pres["title"] == "USB-C temperature logger"


def test_every_presentation_uses_the_documented_vocabulary(store, user_id):
    """Status and action are a closed vocabulary: a new value must be added to
    activity.PRESENTATION_STATUSES / ACTIONS, not invented ad hoc."""
    from kicraft.server import activity

    rows = [
        web._project_presentation(_mk_project(store, user_id)),
        web._project_presentation(_mk_project(store, user_id, status="ok")),
        web._project_presentation(_mk_project(store, user_id, status="failed")),
        web._project_presentation(
            _mk_project(store, user_id), live={"running": True, "activity": {}}),
        web._project_presentation(
            _mk_project(store, user_id), live={"awaiting_input": True, "running": False,
                                               "questions": [{"stage": "wiring"}]}),
    ]
    for pres in rows:
        assert pres["status"] in activity.PRESENTATION_STATUSES
        assert pres["action"] in (*activity.ACTIONS, None)


def test_presentation_recovers_a_workspace_a_legacy_row_forgot(store, user_id):
    """A row with no dir_path still presents its committed name when the owned
    directory survives; the directory is never created on read."""
    p = _mk_project(store, user_id, status="failed")
    root = store.projects_dir / str(user_id) / str(p.id)
    (root / ".kicraft").mkdir(parents=True, exist_ok=True)
    (root / ".kicraft" / "state.json").write_text(
        json.dumps({"project_stem": "RECOVERED"}), encoding="utf-8")
    assert web._project_presentation(store.get_project(p.id))["title"] == "RECOVERED"

    missing = store.create_project(user_id, "never built")
    row = store.get_project(missing)
    assert web._project_presentation(row)["title"] == "never built"
    assert not (store.projects_dir / str(user_id) / str(missing)).exists()


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
