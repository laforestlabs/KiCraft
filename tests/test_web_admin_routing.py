"""Simulated-browser test of /admin/routing.

Uses NiceGUI's User simulation (same harness shape as test_web_core_components):
no real browser, no LLM. Covers the admin render, the _require_admin bounce for
a normal user, and a save round-trip through the routing file that the next
Settings.from_env() picks up.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path

import pytest
from nicegui.testing.user_simulation import user_simulation

from kicraft.server.accounts import AccountStore
from kicraft.server.config import DESIGN_PROFILES, LEGAL_VERSION

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


async def test_admin_page_renders_profiles_and_effective_settings(harness):
    u, web, store = harness
    await _login(u, ADMIN_EMAIL)
    await u.open("/admin/routing")
    await u.should_see("Design-model routing")
    await u.should_see(DESIGN_PROFILES["luna"]["model"])  # the luna profile summary
    await u.should_see("deepseek-flash")           # the deepseek profile summary
    await u.should_see("Effective settings")
    await u.should_see("Max tokens per call")
    await u.should_see("Reasoning/thinking budget")
    # No secrets or hard ceilings may be rendered.
    await u.should_not_see("OPENROUTER_API_KEY")
    await u.should_not_see("KICRAFT_TOTAL_USD_CEILING")


async def test_non_admin_is_bounced(harness):
    u, web, store = harness
    other = store.create_user(USER_EMAIL, PASSWORD)
    store.record_consent(other.id, LEGAL_VERSION)
    await _login(u, USER_EMAIL)
    await u.open("/admin/routing")
    # _require_admin redirects non-staff to the workspace.
    await u.should_see("design a PCB from a sentence")


async def test_save_round_trips_through_the_routing_file(harness):
    u, web, store = harness
    await _login(u, ADMIN_EMAIL)
    await u.open("/admin/routing")

    with u.client:
        next(iter(u.find(marker="routing-profile-select").elements)).value = "deepseek"
        next(iter(u.find(marker="routing-pipeline-select").elements)).value = "legacy"
    u.find(marker="routing-tokens").clear().type("2048")
    u.find(marker="routing-budget").clear().type("0.05")
    u.find(marker="routing-reasoning").clear().type("1024")
    u.find(marker="routing-temperature").clear().type("0.25")
    u.find(marker="routing-save").click()

    path = os.environ["KICRAFT_ROUTING_CONFIG"]
    written = json.loads(Path(path).read_text())
    assert written == {
        "active_profile": "deepseek",
        "pipeline": "legacy",
        "design_reasoning_tokens": 1024,
        "design_temperature": 0.25,
        "max_tokens_per_call": 2048,
        "project_llm_budget_usd": 0.05,
    }

    # The next resolution picks the saved profile + behavior knobs up.
    from kicraft.server.config import Settings

    fresh = Settings.from_env()
    assert fresh.design_profile == "deepseek"
    assert fresh.backend == "deepseek"
    assert fresh.max_tokens_per_call == 2048
    assert fresh.project_llm_budget_usd == pytest.approx(0.05)
    assert fresh.design_reasoning_tokens == 1024
    assert fresh.design_temperature == pytest.approx(0.25)
