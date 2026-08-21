"""Simulated-browser coverage for the admin project browser and clone boundary."""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest
from nicegui.testing.user_simulation import user_simulation

from kicraft.server.accounts import AccountStore
from kicraft.server.config import LEGAL_VERSION

pytestmark = pytest.mark.anyio

ADMIN_EMAIL = "admin@example.com"
OWNER_EMAIL = "owner@example.com"
USER_EMAIL = "user@example.com"
PASSWORD = "hunter2hunter2"
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
        owner = store.create_user(OWNER_EMAIL, PASSWORD)
        store.record_consent(admin.id, LEGAL_VERSION)
        store.record_consent(owner.id, LEGAL_VERSION)
        store.set_role(ADMIN_EMAIL, "admin")
        try:
            yield u, web, store, store.get_user(admin.id), owner
        finally:
            web._safe_fetch = real_fetch
            web._STORE = None
            web._LIVE_RUNS.clear()


def _persisted_private_project(store: AccountStore, owner_id: int) -> tuple[int, Path]:
    brief = "private USB-C environmental sensor"
    stem = "PRIVATE_SENSOR"
    pid = store.create_project(owner_id, brief, is_public=False)
    base = store.projects_dir / str(owner_id) / str(pid)
    gen = base / "generated" / stem
    gen.mkdir(parents=True, exist_ok=True)
    (base / "brief.txt").write_text(brief, encoding="utf-8")
    (base / "events.jsonl").write_text('{"kind":"source-event"}\n', encoding="utf-8")
    state = {
        "project_stem": stem,
        "intent": {"goal": brief, "named_parts": ["ESP32-S3"]},
        "functional_spec": {"blocks": []},
        "architecture": {"sheets": []},
        "bom": {"parts": [{
            "ref": "U1", "value": "MCU", "mpn": "ESP32-S3",
            "footprint": "QFN-56", "sheet": "main",
        }]},
    }
    (base / ".kicraft").mkdir(exist_ok=True)
    (base / ".kicraft" / "state.json").write_text(
        json.dumps(state), encoding="utf-8")
    (gen / f"{stem}.kicad_sch").write_text("(kicad_sch)\n", encoding="utf-8")
    (gen / f"{stem}.kicad_pcb").write_text("(kicad_pcb)\n", encoding="utf-8")
    store.finish_project(pid, "ok", stem=stem, dir_path=str(base))
    return pid, base


async def _login(u, email: str) -> None:
    await u.open("/login")
    u.find("Email").type(email)
    u.find("Password").type(PASSWORD).trigger("keydown.enter")
    await u.should_see("design a PCB from a sentence")


async def test_admin_can_browse_and_view_private_project(harness):
    u, web, store, admin, owner = harness
    source_id, _source_dir = _persisted_private_project(store, owner.id)

    await _login(u, ADMIN_EMAIL)
    await u.open("/admin/projects")
    await u.should_see("Project browser")
    await u.should_see(OWNER_EMAIL)
    await u.should_see("PRIVATE_SENSOR")
    await u.should_see("private USB-C environmental sensor")
    await u.should_see("private")

    u.find(marker="admin-project-view").click()
    await u.should_see("Source project id")
    await u.should_see(str(source_id))
    await u.should_see(OWNER_EMAIL)
    await u.should_see("Schematic")
    await u.should_see("Board")
    await u.should_see("Bill of materials (1 parts)")
    await u.should_see("ESP32-S3")
    await u.should_see("Clone to my workspace")
    await u.should_not_see("Edit board layout")
    await u.should_not_see("Placement rules")
    await u.should_not_see("Rebuild board")
    await u.should_not_see("Continue design")


async def test_non_admin_cannot_open_admin_project_browser(harness):
    u, web, store, admin, owner = harness
    _persisted_private_project(store, owner.id)
    normal = store.create_user(USER_EMAIL, PASSWORD)
    store.record_consent(normal.id, LEGAL_VERSION)

    await _login(u, USER_EMAIL)
    await u.open("/admin/projects")
    await u.should_see("design a PCB from a sentence")
    await u.should_not_see("Project browser")
    await u.should_not_see(OWNER_EMAIL)


async def test_admin_clone_rechecks_source_and_opens_owned_copy(harness):
    u, web, store, admin, owner = harness
    source_id, source_dir = _persisted_private_project(store, owner.id)
    source_state = (source_dir / ".kicraft" / "state.json").read_bytes()
    source_sch = (source_dir / "generated" / "PRIVATE_SENSOR" /
                  "PRIVATE_SENSOR.kicad_sch").read_bytes()
    source_events = (source_dir / "events.jsonl").read_bytes()

    await _login(u, ADMIN_EMAIL)
    # The existing workspace ownership guard must reject the foreign source.
    await u.open(f"/?project={source_id}")
    await u.should_see("Welcome to KiCraft")
    await u.should_not_see("PRIVATE_SENSOR")

    await u.open("/admin/projects")
    u.find(marker="admin-project-view").click()
    await u.should_see("Clone to my workspace")
    u.find(marker="admin-project-clone").click()
    await u.should_see("PRIVATE_SENSOR")
    await u.should_see("Parts")
    await u.should_see("ESP32-S3")

    clones = [p for p in store.list_projects(admin.id) if p.cloned_from_id == source_id]
    assert len(clones) == 1
    clone = clones[0]
    assert clone.status == "ok"
    assert Path(clone.dir_path) == store.projects_dir / str(admin.id) / str(clone.id)
    clone_dir = Path(clone.dir_path)
    assert (clone_dir / ".kicraft" / "state.json").read_bytes() == source_state
    assert (clone_dir / "generated" / "PRIVATE_SENSOR" /
            "PRIVATE_SENSOR.kicad_sch").read_bytes() == source_sch
    assert not (clone_dir / "events.jsonl").exists()
    assert store.get_project(source_id).clone_count == 1
    assert source_dir.exists()
    assert (source_dir / ".kicraft" / "state.json").read_bytes() == source_state
    assert (source_dir / "generated" / "PRIVATE_SENSOR" /
            "PRIVATE_SENSOR.kicad_sch").read_bytes() == source_sch
    assert (source_dir / "events.jsonl").read_bytes() == source_events

    # The clone deep-link is accepted because it is admin-owned.
    await u.open(f"/?project={clone.id}")
    await u.should_see("Parts")
    await u.should_see("ESP32-S3")
