"""Simulated-browser tests for the dedicated /projects page.

The user's project list moved off the workspace ("Your projects" expander) onto
its own page reachable from the top-bar / hamburger ("My projects"). On that
page a paid (pro/max) plan can flip a completed board Public so it lists in the
community browser; a free plan sees the always-public note instead.

Uses NiceGUI's User simulation -- no real browser, no LLM, no build. Mirrors the
harness in test_web_index_autoopen.py: the simulation context resets NiceGUI's
globals on entry, so web is (re)imported inside the context to register its pages
on the fresh app.
"""
from __future__ import annotations

import importlib
import sys

import pytest
from nicegui.testing.user_simulation import user_simulation

from kicraft.server.accounts import AccountStore
from kicraft.server.config import LEGAL_VERSION

pytestmark = pytest.mark.anyio

EMAIL, PASSWORD = "dev@example.com", "hunter2hunter2"
WEB = "kicraft.server.web"


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.fixture
async def harness(tmp_path):
    async with user_simulation() as u:
        mod = sys.modules.get(WEB)
        web = importlib.reload(mod) if mod else importlib.import_module(WEB)
        store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
        web._STORE = store
        web._safe_fetch = lambda key: web._FETCH_ERROR  # no live pricing in tests
        acct = store.create_user(EMAIL, PASSWORD)
        store.record_consent(acct.id, LEGAL_VERSION)
        try:
            yield u, web, store, acct
        finally:
            web._STORE = None
            web._LIVE_RUNS.clear()


def _ok_project(store, user_id: int, brief: str, stem: str) -> int:
    """A finished ('ok') project row with a dir on disk, so the row renders its
    Open action and the visibility control (which only shows for status=='ok')."""
    pid = store.create_project(user_id, brief)
    base = store.projects_dir / str(user_id) / str(pid)
    base.mkdir(parents=True, exist_ok=True)
    store.finish_project(pid, "ok", stem=stem, dir_path=str(base))
    return pid


async def _login(u):
    await u.open("/login")
    u.find("Email").type(EMAIL)
    u.find("Password").type(PASSWORD).trigger("keydown.enter")
    await u.should_see("design a PCB from a sentence")  # the workspace header


async def test_workspace_links_to_projects_page_not_inline_list(harness):
    """The move: the workspace nav offers 'My projects' and no longer carries the
    inline 'Your projects' expander."""
    u, web, store, acct = harness
    _ok_project(store, acct.id, "usb battery bank", "USB_BANK")
    await _login(u)
    await u.should_see("My projects")          # the new nav entry
    await u.should_not_see("Your projects")    # the expander is gone from "/"


async def test_projects_page_lists_user_projects(harness):
    u, web, store, acct = harness
    _ok_project(store, acct.id, "usb battery bank", "USB_BANK")
    await _login(u)
    await u.open("/projects")
    await u.should_see("Your projects")  # the page heading
    await u.should_see("USB_BANK")
    await u.should_see("Open")


async def test_free_plan_sees_always_public_note_no_toggle(harness):
    """Free projects are always public: the page states the upgrade path and shows
    no per-project visibility switch."""
    from nicegui import ui

    u, web, store, acct = harness
    _ok_project(store, acct.id, "usb battery bank", "USB_BANK")
    await _login(u)
    await u.open("/projects")
    await u.should_see("keep projects private")  # free-plan upgrade note
    with pytest.raises(AssertionError):  # no visibility switch on a free plan
        u.find(kind=ui.switch)


async def test_paid_plan_can_publish_a_project(harness):
    """A max plan's project starts private (is_public False at create); flipping the
    Public switch on /projects makes it public and re-indexes it for the browser."""
    from nicegui import ui

    u, web, store, acct = harness
    store.set_tier(EMAIL, "max")
    pid = _ok_project(store, acct.id, "esp32 weather sensor", "WEATHER")
    assert store.get_project(pid).is_public is False  # paid default = private

    await _login(u)
    await u.open("/projects")
    await u.should_see("WEATHER")
    u.find(kind=ui.switch).click()  # the only switch on the page: flip to Public
    await u.should_see("Now public in the community")
    assert store.get_project(pid).is_public is True
