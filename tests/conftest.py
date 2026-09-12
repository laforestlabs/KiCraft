"""Suite-wide guards.

The live lcsc.com retail-stock check (``kicraft.parts_library.lcsc_retail``)
is consulted by the §9.26 BOM gate, ``lookup_lcsc_id`` and web pricing
whenever it is enabled — which is the default. Force it off for every test so
nothing hits the network implicitly (e.g. a stage-commit test running the
real gate against the host's real jlcparts dump). Tests that exercise retail
behavior install a fake module object (monkeypatching the importing module's
``lcsc_retail`` attribute) or re-enable via ``KICRAFT_LCSC_RETAIL=1``.

Live-store isolation (2026-07-20): ``web._STORE`` lazily builds the
PRODUCTION store from .env whenever any test touches a ``_store()`` path
without swapping it — which silently spammed the live ``accounts.db`` with
342 junk error_auto support rows and spawned 46 REAL headless agent
investigations (external provider spend not visible in the OpenRouter ledger).
The autouse guard points the store env at a per-test tmp dir
(exported vars beat .env) and resets the lazy singleton, so no test can
reach the live DB or projects tree by accident again.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_live_retail_stock(monkeypatch):
    monkeypatch.setenv("KICRAFT_LCSC_RETAIL", "0")


@pytest.fixture(autouse=True)
def _default_llm_keys(monkeypatch):
    # The default `luna` design profile runs on OpenRouter and therefore needs
    # only OPENROUTER_API_KEY (read from .env); the alternative `deepseek`
    # profile needs DEEPSEEK_API_KEY. Provide a throwaway value so either
    # profile resolves offline. Tests that assert the missing-key SystemExit
    # delete it themselves.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    # Settings.from_env(dotenv=True) reads the production .env via setdefault, so
    # a stale .env KICRAFT_MODEL/KICRAFT_PROVIDER_ORDER/KICRAFT_MAX_PRICE_* would
    # conflict with the current profile. Pre-seed them with the luna profile's
    # own values so tests never depend on what .env happens to contain. Tests
    # that exercise a different profile delete these themselves (e.g.
    # _clear_profile_env).
    from kicraft.server.config import DESIGN_PROFILES

    luna = DESIGN_PROFILES["luna"]
    monkeypatch.setenv("KICRAFT_MODEL", str(luna["model"]))
    monkeypatch.setenv(
        "KICRAFT_PROVIDER_ORDER", ",".join(str(p) for p in luna["provider_order"])
    )
    monkeypatch.setenv("KICRAFT_MAX_PRICE_PROMPT", str(luna["max_price_prompt"]))
    monkeypatch.setenv("KICRAFT_MAX_PRICE_COMPLETION", str(luna["max_price_completion"]))
    # Escalation and provider fallback are disabled in the steady state; a stale
    # .env value naming a removed profile must not crash startup.
    monkeypatch.setenv("KICRAFT_ESCALATION_PROFILE", "")
    monkeypatch.setenv("KICRAFT_PROVIDER_FALLBACK_PROFILE", "")


@pytest.fixture(autouse=True)
def _isolated_routing_config(monkeypatch, tmp_path):
    # Never let a test read (or write) the operator's real routing file; the
    # default path is ~/.kicraft/routing.json.
    monkeypatch.setenv("KICRAFT_ROUTING_CONFIG", str(tmp_path / "routing.json"))


@pytest.fixture(autouse=True)
def _no_live_store(monkeypatch, tmp_path_factory):
    root = tmp_path_factory.mktemp("kicraft-store")
    monkeypatch.setenv("KICRAFT_USERS_DB", str(root / "accounts.db"))
    monkeypatch.setenv("KICRAFT_PROJECTS_DIR", str(root / "projects"))
    try:
        from kicraft.server import web
        monkeypatch.setattr(web, "_STORE", None)
    except Exception:
        # web has heavy imports (NiceGUI); a test env without them still
        # gets the env-var guard above.
        pass
