"""Simulated-browser smoke test of /admin/core-components.

Uses NiceGUI's User simulation (same harness shape as test_web_index_autoopen):
no real browser, no LLM. Covers the page rendering the seeded registry for an
admin and the _require_admin bounce for a normal user. Row mutations are not
driven here; they are thin wrappers over the store methods covered by
test_core_components.py.
"""
from __future__ import annotations

import importlib
import sys

import pytest
from nicegui.testing.user_simulation import user_simulation

from kicraft.server.accounts import AccountStore
from kicraft.server.config import LEGAL_VERSION

pytestmark = pytest.mark.anyio

ADMIN_EMAIL, USER_EMAIL, PASSWORD = (
    "admin@example.com", "user@example.com", "hunter2hunter2")
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
        # No live pricing in tests (same guard as the autoopen harness).
        real_fetch = web._safe_fetch
        web._safe_fetch = lambda key: web._FETCH_ERROR
        admin = store.create_user(ADMIN_EMAIL, PASSWORD)
        store.record_consent(admin.id, LEGAL_VERSION)
        store.set_role(ADMIN_EMAIL, "admin")
        try:
            yield u, web, store
        finally:
            web._safe_fetch = real_fetch
            web._STORE = None
            web._LIVE_RUNS.clear()


async def _login(u, email: str) -> None:
    await u.open("/login")
    u.find("Email").type(email)
    u.find("Password").type(PASSWORD).trigger("keydown.enter")
    await u.should_see("design a PCB from a sentence")  # the workspace header


async def test_admin_page_renders_seeded_registry(harness):
    u, web, store = harness
    await _login(u, ADMIN_EMAIL)
    await u.open("/admin/core-components")
    await u.should_see("Core components")
    await u.should_see("ldo-3v3-1a")      # a seeded power row
    await u.should_see("AMS1117-3.3")     # its default part
    await u.should_see("Passives")        # category sections render
    await u.should_see("series")          # series rows show 'series', no LCSC id


async def test_non_admin_is_bounced(harness):
    u, web, store = harness
    other = store.create_user(USER_EMAIL, PASSWORD)
    store.record_consent(other.id, LEGAL_VERSION)
    await _login(u, USER_EMAIL)
    await u.open("/admin/core-components")
    # _require_admin redirects non-staff to the workspace.
    await u.should_see("design a PCB from a sentence")
