"""Web manual-layout editor: gating, entry points, and editor render.

Pure-helper tests plus NiceGUI User-simulation flows (same harness as
test_web_index_autoopen): a pro user opening a finished project gets an
"Edit layout" button on the place/route tab that swaps the slot to the
editor (canvas + outline/shape controls + mounting holes + save); a
free user sees the button disabled with an upgrade link; a reopened
FAILED project whose leaves routed gets the rescue CTA.

No real browser, no build, no LLM: leaf artifacts are dummy files
(leaf discovery degrades gracefully when kicad-cli can't render them).
"""
from __future__ import annotations

import importlib
import json
import shutil
import sys
import types
from pathlib import Path

import pytest

from kicraft.server.accounts import AccountStore
from kicraft.server.config import LEGAL_VERSION
from kicraft.server.layout_panel import (
    _project_render_url_for,
    leaf_artifacts_exist,
    manual_preview_name,
    user_may_edit_layout,
)

pytestmark = pytest.mark.anyio

EMAIL, PASSWORD = "dev@example.com", "hunter2hunter2"
WEB = "kicraft.server.web"
STATE_FIXTURE = Path(__file__).parent / "fixtures" / "bmp280_reader_state.json"


@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---- pure helpers ------------------------------------------------------------


def test_user_may_edit_layout_tiers():
    def u(tier=None, role=""):
        return types.SimpleNamespace(tier=tier, role=role)

    assert user_may_edit_layout(u(tier="pro"))
    assert user_may_edit_layout(u(tier="max"))
    assert user_may_edit_layout(u(tier="free", role="admin"))
    assert not user_may_edit_layout(u(tier="free"))
    assert not user_may_edit_layout(u(tier=None))
    assert not user_may_edit_layout(None)


def test_leaf_artifacts_exist(tmp_path):
    assert not leaf_artifacts_exist(tmp_path)
    leaf = tmp_path / ".experiments" / "subcircuits" / "leaf__abc"
    leaf.mkdir(parents=True)
    assert not leaf_artifacts_exist(tmp_path)  # dir without routed board
    (leaf / "leaf_routed.kicad_pcb").write_text("x", encoding="utf-8")
    assert leaf_artifacts_exist(tmp_path)


def test_project_render_url_for(tmp_path):
    png = tmp_path / ".experiments" / "subcircuits" / "x" / "renders" / "leaf_canvas.png"
    png.parent.mkdir(parents=True)
    png.write_bytes(b"png")
    url = _project_render_url_for(tmp_path, "TOK")(png)
    assert url is not None
    assert url.startswith("/project/TOK/render/.experiments/subcircuits/x/renders/leaf_canvas.png?v=")
    # Outside the project dir -> not servable.
    other = tmp_path.parent / "elsewhere.png"
    other.write_bytes(b"png")
    assert _project_render_url_for(tmp_path, "TOK")(other) is None


def test_manual_preview_name():
    assert manual_preview_name("FOO") == "FOO_manual_preview.kicad_pcb"


# ---- simulation flows --------------------------------------------------------

user_simulation = pytest.importorskip(
    "nicegui.testing.user_simulation"
).user_simulation


@pytest.fixture
async def harness(tmp_path):
    async with user_simulation() as u:
        mod = sys.modules.get(WEB)
        web = importlib.reload(mod) if mod else importlib.import_module(WEB)
        store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
        web._STORE = store
        real_fetch = web._safe_fetch
        web._safe_fetch = lambda key: web._FETCH_ERROR
        acct = store.create_user(EMAIL, PASSWORD)
        store.record_consent(acct.id, LEGAL_VERSION)
        try:
            yield u, web, store, acct
        finally:
            web._safe_fetch = real_fetch
            web._STORE = None
            web._LIVE_RUNS.clear()


def _persisted_project_with_board(
    store, user_id: int, *, status: str = "ok", with_pcb: bool = True
) -> int:
    """A finished project whose generated tree carries leaf artifacts
    (dummy files; the editor's leaf discovery falls back to metadata
    dims when rendering fails) and, optionally, the promoted board.

    The stem must match the state fixture's project_stem and the dir
    must hold a .kicad_sch: that's how _discover_generated_dir finds
    the project (and how open_project derives pcb_ready)."""
    stem = "USB_BMP280_READER"  # == STATE_FIXTURE's project_stem
    pid = store.create_project(user_id, f"{stem} brief")
    base = store.projects_dir / str(user_id) / str(pid)
    (base / ".kicraft").mkdir(parents=True)
    (base / "brief.txt").write_text("brief", encoding="utf-8")
    shutil.copy2(STATE_FIXTURE, base / ".kicraft" / "state.json")

    pd = base / "generated" / stem
    leaf = pd / ".experiments" / "subcircuits" / "leaf__abc123"
    (leaf / "renders").mkdir(parents=True)
    (pd / f"{stem}.kicad_sch").write_text("(kicad_sch)", encoding="utf-8")
    (leaf / "metadata.json").write_text(json.dumps({
        "instance_path": "/power",
        "sheet_name": "POWER",
        "local_board_outline": {"width_mm": 20.0, "height_mm": 12.0},
    }), encoding="utf-8")
    (leaf / "leaf_routed.kicad_pcb").write_text("(kicad_pcb)", encoding="utf-8")
    if with_pcb:
        (pd / f"{stem}.kicad_pcb").write_text("(kicad_pcb)", encoding="utf-8")
    store.finish_project(pid, status, stem=stem, dir_path=str(base))
    return pid


async def _login(u):
    await u.open("/login")
    u.find("Email").type(EMAIL)
    u.find("Password").type(PASSWORD).trigger("keydown.enter")
    await u.should_see("design a PCB from a sentence")


async def test_pro_user_opens_layout_editor(harness):
    u, web, store, acct = harness
    store.set_tier(EMAIL, "pro")
    _persisted_project_with_board(store, acct.id)

    await _login(u)  # auto-opens the unseen finished project
    await u.should_see("USB_BMP280_READER")
    await u.should_see("Edit layout")

    from nicegui import ui
    u.find("Edit layout", kind=ui.button).click()
    await u.should_see("Manual layout")
    await u.should_see("Save & stamp preview")
    await u.should_see("Mounting Holes")
    await u.should_see("View options")

    # Back out: the close is deferred one tick and the board view is
    # repainted by the page's 0.2 s render timer, so allow more retries
    # than the 3 x 0.1 s default.
    u.find("Back to board", kind=ui.button).click()
    await u.should_see("Edit layout", retries=30)


async def test_free_user_sees_upgrade_gate(harness):
    u, web, store, acct = harness
    _persisted_project_with_board(store, acct.id)

    await _login(u)
    await u.should_see("Edit layout")
    await u.should_see("Upgrade")  # the /pricing link next to the disabled button

    from nicegui import ui
    btn = u.find("Edit layout", kind=ui.button).elements.pop()
    assert not btn.enabled


async def test_failed_project_offers_rescue(harness):
    u, web, store, acct = harness
    store.set_tier(EMAIL, "max")
    _persisted_project_with_board(
        store, acct.id, status="failed", with_pcb=False
    )

    await _login(u)
    await u.should_see("USB_BMP280_READER")
    await u.should_see("Rescue: lay out the board manually")

    from nicegui import ui
    u.find("Rescue: lay out the board manually", kind=ui.button).click()
    await u.should_see("Manual layout")
    await u.should_see("Save & stamp preview")


async def test_pro_user_opens_placement_rules_panel(harness):
    u, web, store, acct = harness
    store.set_tier(EMAIL, "pro")
    _persisted_project_with_board(store, acct.id)

    await _login(u)
    await u.should_see("Placement rules")

    from nicegui import ui
    u.find("Placement rules", kind=ui.button).click()
    await u.should_see("Apply & re-place", retries=10)
    await u.should_see("Auto size")
    # Grouped component rows come from the schematic; the dummy sheet
    # parses to no components, so just the chrome is asserted here (the
    # rules data layer itself is covered by test_gui_per_component_overrides
    # and test_placement_section).

    u.find("Back to board", kind=ui.button).click()
    await u.should_see("Placement rules", retries=30)
