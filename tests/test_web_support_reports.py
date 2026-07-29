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
        # Never let a dialog submit spawn a REAL headless `claude` run: this
        # exact path launched one phantom investigation (real Anthropic
        # spend, 30-min watchdog) per suite run from 2026-07-12 to
        # 2026-07-20. enqueue_investigation also refuses under pytest now;
        # this records the calls for assertions instead.
        real_auto = sim_web._auto_investigate_if_enabled
        auto_calls: list[int] = []
        sim_web._auto_investigate_if_enabled = auto_calls.append
        acct = sim_store.create_user(EMAIL, PASSWORD)
        sim_store.record_consent(acct.id, LEGAL_VERSION)
        try:
            yield u, sim_web, sim_store, acct
        finally:
            sim_web._safe_fetch = real_fetch
            sim_web._auto_investigate_if_enabled = real_auto
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


# ---- admin support page: investigations + highlighting ----------------------

from kicraft.server import investigate_runner as ir  # noqa: E402
from kicraft.server.accounts import SupportReport  # noqa: E402
from kicraft.server.routes_admin import _is_user_reported  # noqa: E402


def _user_report(store, *, board_code="KC-ABC234", message="it broke",
                 project_id=None):
    user = store.create_user("rep@example.com", "hunter2hunter2")
    return store.create_support_report(
        user_id=user.id, project_id=project_id, board_code=board_code,
        kind="user", message=message)


def _mk_report(kind, message):
    return SupportReport(id=1, created_at="", user_id=None, project_id=None,
                         board_code=None, kind=kind, status="new",
                         message=message, diagnostics={})


def test_is_user_reported_highlight_rule():
    assert _is_user_reported(_mk_report("user", None)) is True
    assert _is_user_reported(_mk_report("error_auto", "I changed the LED")) is True
    assert _is_user_reported(_mk_report("error_auto", "   ")) is False
    assert _is_user_reported(_mk_report("error_auto", None)) is False


def test_auto_investigate_setting_defaults_on(store):
    assert store.get_setting("support.auto_investigate", "1") == "1"
    store.set_setting("support.auto_investigate", "0")
    assert store.get_setting("support.auto_investigate", "1") == "0"


def test_investigation_lifecycle(store):
    rid = _user_report(store)
    inv = store.create_investigation(report_id=rid, board_code="KC-ABC234")
    assert store.get_investigation(inv).status == "queued"
    assert store.active_investigation_exists(rid) is True

    assert store.start_investigation(inv) is True
    assert store.start_investigation(inv) is False   # guarded: already running
    assert store.active_investigation_exists(rid) is True  # running still active

    store.finish_investigation(inv, rc=0, report_md="# report", status="done")
    got = store.latest_investigation(rid)
    assert got.status == "done" and got.rc == 0 and got.report_md == "# report"
    assert store.active_investigation_exists(rid) is False  # terminal
    assert store.latest_investigations_by_report()[rid].id == inv


def test_enqueue_dedups_and_needs_a_locatable_target(store, tmp_path):
    calls = []
    rec = lambda s, inv_id, target: calls.append((inv_id, target))  # noqa: E731

    rid = _user_report(store, board_code="KC-DEF345")
    rep = store.get_support_report(rid)
    assert ir._resolve_target(store, rep) == "KC-DEF345"

    inv_id = ir.enqueue_investigation(store, rep, log_dir=tmp_path / "logs",
                                      runner=rec)
    assert inv_id is not None
    # A row is queued now, so a second enqueue (double click / manual+auto) no-ops
    assert ir.enqueue_investigation(store, rep, runner=rec) is None

    # A report with neither a board code nor a locatable project is un-investigable
    rid2 = store.create_support_report(kind="user", message="no board")
    rep2 = store.get_support_report(rid2)
    assert ir._resolve_target(store, rep2) is None
    assert ir.enqueue_investigation(store, rep2, runner=rec) is None


def test_run_investigation_stores_report_and_status(store, monkeypatch):
    rid = _user_report(store, board_code="KC-GHI456")
    monkeypatch.setattr(ir, "_run_claude",
                        lambda s, i, t: (0, "# Investigation\nlooks fine"))
    inv = store.create_investigation(report_id=rid, board_code="KC-GHI456")
    ir.run_investigation(store, inv, "KC-GHI456")
    got = store.get_investigation(inv)
    assert got.status == "done" and got.rc == 0
    assert "Investigation" in got.report_md

    monkeypatch.setattr(ir, "_run_claude", lambda s, i, t: (2, "route failed"))
    inv2 = store.create_investigation(report_id=rid, board_code="KC-GHI456")
    ir.run_investigation(store, inv2, "KC-GHI456")
    assert store.get_investigation(inv2).status == "failed"

    def boom(*_a):
        raise RuntimeError("subprocess exploded")
    monkeypatch.setattr(ir, "_run_claude", boom)
    inv3 = store.create_investigation(report_id=rid, board_code="KC-GHI456")
    ir.run_investigation(store, inv3, "KC-GHI456")  # crash must finalize the row
    assert store.get_investigation(inv3).status == "failed"


def test_run_claude_reports_missing_binary(store, monkeypatch):
    monkeypatch.setattr(ir, "_claude_bin", lambda: None)
    rid = _user_report(store, board_code="KC-JKL567")
    inv = store.create_investigation(report_id=rid, board_code="KC-JKL567")
    rc, out = ir._run_claude(store, inv, "KC-JKL567")
    assert rc is None and "claude" in out.lower()


def test_auto_investigate_respects_toggle(swapped_store, monkeypatch):
    calls = []
    monkeypatch.setattr(ir, "enqueue_investigation",
                        lambda store, report, **kw: calls.append(report.id))
    rid = _user_report(swapped_store, board_code="KC-MNO678")

    web._auto_investigate_if_enabled(rid)          # default: on
    assert calls == [rid]

    swapped_store.set_setting("support.auto_investigate", "0")
    web._auto_investigate_if_enabled(rid)          # off: no new enqueue
    assert calls == [rid]


def test_auto_investigate_errors_off_by_default_and_capped(swapped_store, monkeypatch):
    """error_auto rows only trigger headless triage when the dedicated toggle
    is ON, and never past the daily cap (each run is real LLM spend)."""
    calls = []
    monkeypatch.setattr(ir, "enqueue_investigation",
                        lambda store, report, **kw: calls.append(report.id))
    rid = _user_report(swapped_store, board_code="KC-PQR789")

    web._auto_investigate_error_if_enabled(rid)      # default: OFF
    assert calls == []

    swapped_store.set_setting("support.auto_investigate_errors", "1")
    web._auto_investigate_error_if_enabled(rid)
    assert calls == [rid]

    # at the cap: no new enqueue
    for _ in range(web._AUTO_ERROR_INVESTIGATE_DAILY_CAP):
        swapped_store.create_investigation(report_id=rid, board_code="KC-PQR789")
    web._auto_investigate_error_if_enabled(rid)
    assert calls == [rid]

    web._auto_investigate_error_if_enabled(None)     # no report id: no crash


def test_investigations_created_since_counts(store):
    rid = _user_report(store, board_code="KC-STU890")
    assert store.investigations_created_since("2000-01-01T00:00:00") == 0
    store.create_investigation(report_id=rid, board_code="KC-STU890")
    store.create_investigation(report_id=rid, board_code="KC-STU890")
    assert store.investigations_created_since("2000-01-01T00:00:00") == 2
    assert store.investigations_created_since("2999-01-01T00:00:00") == 0


def test_run_claude_sets_headless_env_and_model(store, monkeypatch, tmp_path):
    """The headless run must advertise itself to the skill (the skill budgets
    replays off KICRAFT_INVESTIGATE_HEADLESS) and pin an explicit model."""
    seen = {}

    class FakeProc:
        stdout = iter(["report line\n"])
        pid = 0

        def poll(self):
            return 0

        def wait(self):
            return 0

    def fake_popen(cmd, **kw):
        seen["cmd"] = cmd
        seen["env"] = kw.get("env") or {}
        return FakeProc()

    monkeypatch.setattr(ir, "_claude_bin", lambda: "/usr/bin/claude")
    monkeypatch.setattr(ir.subprocess, "Popen", fake_popen)
    rid = _user_report(store, board_code="KC-VWX901")
    inv = store.create_investigation(report_id=rid, board_code="KC-VWX901")
    rc, out = ir._run_claude(store, inv, "KC-VWX901")
    assert rc == 0 and "report line" in out
    assert seen["env"].get("KICRAFT_INVESTIGATE_HEADLESS") == "1"
    assert "--model" in seen["cmd"]
