"""Support flow: the run worker's failure path auto-files a support report
(board id attached), the diagnostics snapshot stays a bounded,
automated-review-ready summary, and the workspace dialog (manual Support
button + post-error auto-open) records the user's optional feedback.

The dialog tests use NiceGUI's User simulation (no real browser, no LLM, no
build), mirroring tests/test_web_index_autoopen.py's harness.
"""
from __future__ import annotations

import importlib
import sys

import pytest
from nicegui.testing.user_simulation import user_simulation

from kicraft.server.accounts import AccountStore
from kicraft.server.config import LEGAL_VERSION
import kicraft.server.web as web

EMAIL, PASSWORD = "dev@example.com", "hunter2hunter2"
WEB = "kicraft.server.web"


@pytest.fixture
def store(tmp_path):
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


@pytest.fixture
def swapped_store(store):
    old = web._STORE
    web._STORE = store
    yield store
    web._STORE = old


def _failed_state(store):
    user = store.create_user("fail@example.com", "hunter2hunter2")
    pid = store.create_project(user.id, "a board that will not build")
    p = store.get_project(pid)
    state = web._fresh_run_state()
    state.update(user_id=user.id, project_id=pid, board_code=p.board_code,
                 brief=p.brief, ok=False,
                 events=[{"kind": "stage_done", "stage": "intent"},
                         {"kind": "build_log", "text": "routing failed: exit 7"}])
    return state, p


def test_failure_files_support_report(swapped_store):
    state, p = _failed_state(swapped_store)
    web._file_failure_report(state)
    reports = swapped_store.list_support_reports(status="new")
    assert len(reports) == 1
    r = reports[0]
    assert r.kind == "error_auto"
    assert r.board_code == p.board_code and r.project_id == p.id
    assert state["support_report_id"] == r.id  # the dialog attaches feedback here
    d = r.diagnostics
    assert d["run_status"] == "failed"
    assert d["stages_done"] == ["intent"]
    assert d["build_log_tail"] == ["routing failed: exit 7"]
    assert d["board_code"] == p.board_code
    assert d["brief"] == "a board that will not build"


class _ExplodingStore:
    def create_support_report(self, **_kw):
        raise RuntimeError("db locked")


def test_failure_report_never_raises_on_store_error(store):
    """Best-effort contract: a reporting hiccup must not crash the run worker."""
    state, _ = _failed_state(store)
    old = web._STORE
    web._STORE = _ExplodingStore()
    try:
        web._file_failure_report(state)
    finally:
        web._STORE = old
    assert state.get("support_report_id") is None


def test_diagnostics_snapshot_is_bounded(swapped_store):
    state, _ = _failed_state(swapped_store)
    state["brief"] = "x" * 5000
    state["events"] = [{"kind": "build_log", "text": f"line {i}"}
                       for i in range(500)]
    d = web._collect_support_diagnostics(state)
    assert len(d["brief"]) == 2000
    assert len(d["build_log_tail"]) == 60
    assert d["build_log_tail"][-1] == "line 499"


# ---- the dialog itself (simulated browser) ----------------------------------

@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def harness(tmp_path):
    async with user_simulation() as u:
        mod = sys.modules.get(WEB)
        sim_web = importlib.reload(mod) if mod else importlib.import_module(WEB)
        sim_store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
        sim_web._STORE = sim_store
        real_fetch = sim_web._safe_fetch  # no live pricing from test threads
        sim_web._safe_fetch = lambda key: sim_web._FETCH_ERROR
        acct = sim_store.create_user(EMAIL, PASSWORD)
        sim_store.record_consent(acct.id, LEGAL_VERSION)
        try:
            yield u, sim_web, sim_store, acct
        finally:
            sim_web._safe_fetch = real_fetch
            sim_web._STORE = None
            sim_web._LIVE_RUNS.clear()


async def _login(u):
    await u.open("/login")
    u.find("Email").type(EMAIL)
    u.find("Password").type(PASSWORD).trigger("keydown.enter")
    await u.should_see("design a PCB from a sentence")


@pytest.mark.anyio
async def test_support_button_files_user_report(harness):
    u, sim_web, sim_store, acct = harness
    from nicegui import ui

    await _login(u)
    u.find("Support", kind=ui.button).click()
    await u.should_see("Contact support")
    await u.should_see("(no design open)")  # blank composer: no board id yet
    u.find("Anything you'd like to add?").type("the examples page 404s")
    u.find("Send report", kind=ui.button).click()
    await u.should_see("Your reference is report #")

    reports = sim_store.list_support_reports(status="new")
    assert len(reports) == 1
    r = reports[0]
    assert r.kind == "user" and r.user_id == acct.id
    assert r.message == "the examples page 404s"
    assert r.board_code is None and r.project_id is None


@pytest.mark.anyio
async def test_failed_run_auto_opens_dialog_and_attaches_feedback(harness):
    u, sim_web, sim_store, acct = harness
    from nicegui import ui

    pid = sim_store.create_project(acct.id, "doomed board")
    p = sim_store.get_project(pid)
    # The terminal state a failed worker leaves behind, error report included
    # (in production _run_design files it via _file_failure_report).
    live = sim_web._fresh_run_state()
    live.update(done=True, ok=False, user_id=acct.id, project_id=pid,
                brief="doomed board", board_code=p.board_code)
    sim_web._file_failure_report(live)
    rid = live["support_report_id"]
    assert rid is not None
    sim_web._LIVE_RUNS[pid] = live

    await _login(u)
    await u.open(f"/?project={pid}")  # attach to the failed live run
    await u.should_see("Something went wrong")        # dialog auto-opened once
    await u.should_see(f"Board ID: {p.board_code}")   # the workspace chip
    u.find("Anything you'd like to add?").type("I only changed the LED color")
    u.find("Send report", kind=ui.button).click()
    await u.should_see(f"Your reference is {p.board_code}")

    reports = sim_store.list_support_reports()
    assert len(reports) == 1  # feedback joined the auto-filed row, no duplicate
    assert reports[0].id == rid
    assert reports[0].kind == "error_auto"
    assert reports[0].message == "I only changed the LED color"
