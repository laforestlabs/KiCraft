"""End-to-end (simulated browser) test of the workspace default view.

Reproduces the reported flow: a user kicks off a design, clicks away to
/parts, and returns to "/". The index page must re-attach to the in-flight
run (live "Designing ..." status line from the first second of the run),
not land on a blank "new project" composer. Uses NiceGUI's User simulation --
no real browser, no LLM, no build: the "run" is a state dict in
web._LIVE_RUNS, exactly what a worker thread registers.

web.py registers its pages at import time, but the simulation context resets
NiceGUI's globals on entry (and drops the module from sys.modules on exit),
so the harness (re)imports web inside the context to register its pages on
the fresh app, and yields that module for the test to poke.
"""
from __future__ import annotations

import importlib
import shutil
import sys
from pathlib import Path

import pytest
from nicegui.testing.user_simulation import user_simulation

from kicraft.server.accounts import AccountStore
from kicraft.server.config import LEGAL_VERSION

pytestmark = pytest.mark.anyio

EMAIL, PASSWORD = "dev@example.com", "hunter2hunter2"
WEB = "kicraft.server.web"
STATE_FIXTURE = Path(__file__).parent / "fixtures" / "bmp280_reader_state.json"


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
        # No live pricing in tests: resolve every BOM price fetch to the error
        # sentinel instantly instead of hitting easyeda/jlcpcb from a thread.
        real_fetch = web._safe_fetch
        web._safe_fetch = lambda key: web._FETCH_ERROR
        acct = store.create_user(EMAIL, PASSWORD)
        acct = store.consume_verification_token(
            store.create_verification_token(acct.id))  # verified so clone works
        store.record_consent(acct.id, LEGAL_VERSION)
        try:
            yield u, web, store, acct
        finally:
            web._safe_fetch = real_fetch
            web._STORE = None
            web._LIVE_RUNS.clear()


def _persisted_project(store, user_id: int, brief: str, stem: str) -> int:
    """Lay a finished project on disk the way build-in-place does (brief +
    .kicraft/state.json) and record its row, so open/clone flows run against the
    real artifact layout."""
    pid = store.create_project(user_id, brief)
    base = store.projects_dir / str(user_id) / str(pid)
    (base / ".kicraft").mkdir(parents=True)
    (base / "brief.txt").write_text(brief, encoding="utf-8")
    shutil.copy2(STATE_FIXTURE, base / ".kicraft" / "state.json")
    store.finish_project(pid, "ok", stem=stem, dir_path=str(base))
    return pid


async def _login(u):
    await u.open("/login")
    u.find("Email").type(EMAIL)
    # Enter in the password field submits (the page wires keydown.enter);
    # find("Sign in").click() would hit the "Sign in to design..." LABEL first.
    u.find("Password").type(PASSWORD).trigger("keydown.enter")
    await u.should_see("design a PCB from a sentence")  # the workspace header


async def test_returning_to_index_attaches_to_live_run(harness):
    u, web, store, acct = harness
    pid = store.create_project(acct.id, "usb battery bank")
    live = web._fresh_run_state()
    live.update(running=True, user_id=acct.id, project_id=pid,
                brief="usb battery bank")
    web._LIVE_RUNS[pid] = live

    await _login(u)  # lands on "/" -- the return to the workspace
    await u.should_see("usb battery bank")  # attached to the run, not blank
    # The live status line, not a blank composer. (An earlier revision
    # asserted "Open", which only ever matched placeholder text of the
    # since-removed stage-edit panel -- the project list lives on /projects.)
    await u.should_see("live progress is in")


async def test_blank_composer_when_nothing_needs_attention(harness):
    # The project list moved to /projects, so a fresh user lands on the blank
    # composer (first-run welcome), not an inline "No projects yet" list.
    u, web, store, acct = harness
    await _login(u)
    await u.should_see("Welcome to KiCraft")


async def test_cloned_project_opens_with_bom(harness):
    """Reported: 'when i cloned a public project i couldnt see the BOM at all,
    just a blank screen.' A clone must land the user IN the cloned project
    (auto-open of the fresh unseen result) with its BOM inspector populated
    from the cloned state.json -- not on a blank composer."""
    u, web, store, acct = harness

    owner = store.create_user("owner@example.com", "ownerpw12345")
    src_id = _persisted_project(store, owner.id, "bmp280 reader", "BMP280_READER")

    pid, err = web._clone_project(store.get_project(src_id), acct,
                                  make_private=False)
    assert err is None and pid is not None

    await _login(u)  # the post-clone ui.navigate.to("/")
    await u.should_see("BMP280_READER")  # auto-opened the clone
    # The BOM tab's inspector renders the cloned parts (render-timer driven).
    await u.should_see("Summary")
    await u.should_see("Parts")


async def test_clone_deep_link_outranks_parked_default(harness):
    """The clone flow navigates to /?project=<clone id>. Without the deep link
    a plain "/" would auto-open an older parked run instead of the fresh copy
    (parked runs outrank unseen finished results in the default pick)."""
    u, web, store, acct = harness
    store.set_tier(EMAIL, "max")  # room for two designs: the parked + the clone
    parked = _persisted_project(store, acct.id, "old parked design", "PARKED_ONE")
    store.update_project_status(parked, "awaiting_input")

    owner = store.create_user("owner2@example.com", "ownerpw12345")
    src = _persisted_project(store, owner.id, "bmp280 reader", "BMP280_READER")
    pid, err = web._clone_project(store.get_project(src), store.get_user(acct.id),
                                  make_private=False)
    assert err is None

    await _login(u)  # plain "/": the parked design wins the default pick
    await u.should_see("waiting for your answer below")

    await u.open(f"/?project={pid}")  # what do_clone navigates to
    await u.should_see("Done. Your KiCad project is ready.")
