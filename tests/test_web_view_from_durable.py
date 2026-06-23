"""Reopen reads the durable project tree directly — refactor roadmap Phase 4a.

A reopened project is READ from its durable tree (projects_dir/<uid>/<pid>/) with no
per-reopen 17-29 MB scratch copy; a workspace is materialized LAZILY by
`_ensure_workspace` only on the first WRITE action, rehydrated (copytree) so
previously-committed slots survive (the §4 data-loss guard in
docs/plans/view-from-durable-refactor-v2.md). Legacy projects without a durable
dir_path still fall back to an eager rehydrate.
"""
from __future__ import annotations

import importlib
import os
import shutil
import sys
from pathlib import Path

import pytest
from nicegui.testing.user_simulation import user_simulation

from kicraft.server import web
from kicraft.server.accounts import AccountStore
from kicraft.server.config import LEGAL_VERSION

STATE_FIXTURE = Path(__file__).parent / "fixtures" / "bmp280_reader_state.json"


# --------------------------- _read_root / flag (pure) ---------------------------

def test_read_root_prefers_ws_then_view_root_then_none():
    assert web._read_root({"ws": "/w", "view_root": "/d"}) == Path("/w")
    assert web._read_root({"ws": None, "view_root": "/d"}) == Path("/d")
    assert web._read_root({"ws": None, "view_root": None}) is None
    assert web._read_root({}) is None


# ------------------------------ _ensure_workspace -------------------------------

class _Proj:
    def __init__(self, dir_path, stem):
        self.dir_path = str(dir_path)
        self.project_stem = stem
        self.id = 1


def test_ensure_workspace_noop_when_ws_exists(tmp_path):
    state = {"ws": str(tmp_path)}
    assert web._ensure_workspace(state, _Proj(tmp_path, "X")) == tmp_path
    assert state["ws"] == str(tmp_path)  # unchanged


def test_ensure_workspace_none_without_project():
    assert web._ensure_workspace({"ws": None, "project_id": None}, None) is None


def test_ensure_workspace_resolves_durable_project_dir(tmp_path, monkeypatch):
    """Build-in-place: _ensure_workspace points ws at the durable project dir itself
    (no scratch workspace, no copy) when ws is None but a project id is set, and
    re-points project_dir/token at its generated tree."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-not-used")
    store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
    acct = store.create_user("ew@example.com", "hunter2hunter2")
    pid = store.create_project(acct.id, "bmp280 reader")
    base = store.projects_dir / str(acct.id) / str(pid)  # the durable build dir
    (base / ".kicraft").mkdir(parents=True)               # the build's native layout
    shutil.copy2(STATE_FIXTURE, base / ".kicraft" / "state.json")
    gen = base / "generated" / "USB_BMP280_READER"
    gen.mkdir(parents=True)
    (gen / "USB_BMP280_READER.kicad_sch").write_text("(kicad_sch)", encoding="utf-8")

    prev = web._STORE
    web._STORE = store
    try:
        state = {"ws": None, "view_root": None, "user_id": acct.id, "project_id": pid,
                 "stem": None, "project_dir": None, "token": None}
        ws = web._ensure_workspace(state)
        assert ws == base                       # the durable dir IS the workspace
        assert state["ws"] == str(base)
        assert Path(ws).parent == store.projects_dir / str(acct.id)
        assert state["project_dir"] == str(gen)  # generated tree, in place (not copied)
        assert web._ensure_workspace(state) == base  # idempotent no-op
    finally:
        web._STORE = prev


# --------------- integration: reopen reads durable, makes NO workspace -----------

EMAIL, PASSWORD = "vfd@example.com", "hunter2hunter2"
WEB = "kicraft.server.web"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def reopen_harness(tmp_path):
    """user_simulation harness with an isolated KICRAFT_WORK_DIR so the test can assert
    that reopening a project creates zero scratch workspaces there."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    prev = {k: os.environ.get(k) for k in ("KICRAFT_WORK_DIR", "OPENROUTER_API_KEY")}
    os.environ["KICRAFT_WORK_DIR"] = str(work_dir)
    os.environ.setdefault("OPENROUTER_API_KEY", "test-not-used")
    async with user_simulation() as u:
        mod = sys.modules.get(WEB)
        web_mod = importlib.reload(mod) if mod else importlib.import_module(WEB)
        store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
        web_mod._STORE = store
        real_fetch = web_mod._safe_fetch
        web_mod._safe_fetch = lambda key: web_mod._FETCH_ERROR
        acct = store.create_user(EMAIL, PASSWORD)
        store.record_consent(acct.id, LEGAL_VERSION)
        try:
            yield u, web_mod, store, acct, work_dir
        finally:
            web_mod._safe_fetch = real_fetch
            web_mod._STORE = None
            web_mod._LIVE_RUNS.clear()
            for k, v in prev.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v


def _persist_durable(store, user_id, brief, stem):
    """A finished durable project laid down as _persist_project does, WITH a
    generated tree so view-from-durable resolves the schematic dir."""
    pid = store.create_project(user_id, brief)
    base = store.projects_dir / str(user_id) / str(pid)
    (base / "kicraft").mkdir(parents=True)
    (base / "brief.txt").write_text(brief, encoding="utf-8")
    shutil.copy2(STATE_FIXTURE, base / "state.json")
    shutil.copy2(STATE_FIXTURE, base / "kicraft" / "state.json")
    gen = base / "generated" / stem
    gen.mkdir(parents=True)
    (gen / f"{stem}.kicad_sch").write_text("(kicad_sch)", encoding="utf-8")
    store.finish_project(pid, "ok", stem=stem, dir_path=str(base))
    return pid


async def _login(u):
    await u.open("/login")
    u.find("Email").type(EMAIL)
    u.find("Password").type(PASSWORD).trigger("keydown.enter")
    await u.should_see("design a PCB from a sentence")


@pytest.mark.anyio
async def test_reopen_reads_durable_and_makes_no_workspace(reopen_harness):
    u, web_mod, store, acct, work_dir = reopen_harness
    pid = _persist_durable(store, acct.id, "bmp280 reader", "USB_BMP280_READER")

    await _login(u)
    await u.open(f"/?project={pid}")          # reopen the finished project
    await u.should_see("USB_BMP280_READER")   # header/stem rendered from durable
    await u.should_see("Done. Your KiCad project is ready.")
    await u.should_see("Parts")               # BOM inspector populated from durable state.json

    # The headline acceptance: no scratch workspace was minted to view it.
    assert list(work_dir.iterdir()) == []
