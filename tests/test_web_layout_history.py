"""Capability delivery and durable controls for failed and legacy builds."""
import json
from contextlib import asynccontextmanager

import pytest
from nicegui import ui
from starlette.responses import FileResponse

from kicraft.layout_history import start_capture
from kicraft.layout_history_export import finalize_capture
from kicraft.server import render_serving
from kicraft.server.layout_history import LayoutHistoryPanel
from kicraft.server.stagetabs import StageTabs


def _publish(project, *, rc=6, generation=2):
    root = start_capture(project, run_id="test", generation=generation)
    root.scene([], phase="empty", geometry_source="native_board")
    root.end("complete")
    manifest = finalize_capture(project, root.descriptor.capture_id, build_rc=rc)
    return json.loads(manifest.read_text())


def test_failed_build_is_viewable_and_downloadable(tmp_path):
    manifest = _publish(tmp_path)
    token = render_serving._register_project_dir(tmp_path)
    response = render_serving.serve_project_history(token, manifest["capture_id"])
    assert isinstance(response, FileResponse)
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert "connect-src 'none'" in response.headers["content-security-policy"]
    download = render_serving.download_project_history(token, manifest["capture_id"])
    assert "attachment" in download.headers["content-disposition"]


@pytest.mark.parametrize("capture", ["../outside", "f" * 32, "x" * 32, "a" * 31, "a" * 33])
def test_invalid_or_missing_captures_are_not_served(tmp_path, capture):
    token = render_serving._register_project_dir(tmp_path)
    assert render_serving.serve_project_history(token, capture).status_code == 404


def test_forged_token_and_cross_project_are_rejected(tmp_path):
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir(); other.mkdir()
    capture = _publish(project)["capture_id"]
    assert render_serving.serve_project_history("bad.token", capture).status_code == 404
    token = render_serving._register_project_dir(other)
    assert render_serving.serve_project_history(token, capture).status_code == 404


@pytest.mark.parametrize("target", ["manifest.json", "timelapse.html", "directory"])
def test_symlink_escape_is_rejected(tmp_path, target):
    project = tmp_path / "project"
    project.mkdir()
    capture = _publish(project)["capture_id"]
    directory = project / "layout-history" / capture
    token = render_serving._register_project_dir(project)
    if target == "directory":
        moved = tmp_path / "moved"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
    else:
        path = directory / target
        moved = tmp_path / target
        path.rename(moved)
        path.symlink_to(moved)
    assert render_serving.serve_project_history(token, capture).status_code == 404


def test_history_html_cannot_use_generic_file_route(tmp_path):
    (tmp_path / "timelapse.html").write_text("test")
    token = render_serving._register_project_dir(tmp_path)
    assert render_serving.serve_project_file(token, "timelapse.html").status_code == 404


def test_stale_generation_is_not_current_and_legacy_is_readable(tmp_path):
    token = render_serving._register_project_dir(tmp_path)
    with ui.column():
        legacy = LayoutHistoryPanel(tmp_path, token)
        assert "wasn't recorded" in legacy._status.text
        _publish(tmp_path, generation=1)
        stale = LayoutHistoryPanel(tmp_path, token, generation=2)
        assert stale._capture_id is None
        assert "wasn't recorded" in stale._status.text
        current = LayoutHistoryPanel(tmp_path, token, generation=1)
        assert current._capture_id is not None


def test_panel_survives_board_and_inspector_updates(tmp_path):
    _publish(tmp_path)
    token = render_serving._register_project_dir(tmp_path)
    with ui.column():
        tabs = StageTabs()
        with tabs.history_slot("place_route"):
            panel = LayoutHistoryPanel(tmp_path, token)
        panel.open()
        dialog, frame, content = panel._dialog, panel._frame, panel._frame.content
        tabs.view_slot("place_route").clear()
        tabs.set_inspector("place_route", [])
        panel.refresh()
        assert panel._dialog is dialog and panel._frame is frame
        assert frame.content == content
        assert 'sandbox="allow-scripts"' in content
        assert "allow-same-origin" not in content
        panel.dispose()


@asynccontextmanager
async def _isolated_web_user():
    import sys
    from nicegui.testing.user_simulation import user_simulation
    # NiceGUI removes route modules AND their parents on exit. Restore the
    # original package identities so subsequent pipeline tests retain imports.
    modules = {name: module for name, module in sys.modules.items()
               if name == "kicraft" or name.startswith("kicraft.")}
    try:
        async with user_simulation() as user:
            yield user
    finally:
        sys.modules.update(modules)


@pytest.mark.anyio
@pytest.mark.parametrize("status", ["ok", "failed", "running"])
async def test_real_web_place_route_mounts_history_for_current_attempt(tmp_path, monkeypatch, status):
    import importlib
    from pathlib import Path
    from kicraft.server import web
    from kicraft.server.accounts import AccountStore
    from kicraft.server.config import LEGAL_VERSION

    async with _isolated_web_user() as user:
        web = importlib.reload(web)
        store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
        monkeypatch.setattr(web, "_STORE", store)
        monkeypatch.setattr(web, "_safe_fetch", lambda _: web._FETCH_ERROR)
        monkeypatch.setattr(web, "_LIVE_RUNS", {})
        account = store.create_user("history@example.com", "history-password")
        store.consume_verification_token(store.create_verification_token(account.id))
        store.record_consent(account.id, LEGAL_VERSION)
        from account_fixtures import seed_project
        project_id = seed_project(store, account.id, "History test")
        workspace = store.projects_dir / str(account.id) / str(project_id)
        project = workspace / "generated" / "attempt-1" / "BOARD"
        project.mkdir(parents=True)
        state = json.loads((Path(__file__).parent / "fixtures" / "bmp280_reader_state.json").read_text())
        state["project_stem"] = "BOARD"
        state["artifacts"] = {
            "project_dir": str(project), "project_stem": "BOARD",
            "root_sch": str(project / "BOARD.kicad_sch"), "leaf_schs": [],
            "kicad_pro": str(project / "BOARD.kicad_pro"),
            "autoplacer_json": str(project / "BOARD_autoplacer.json"),
        }
        (workspace / ".kicraft").mkdir(exist_ok=True)
        (workspace / ".kicraft" / "state.json").write_text(json.dumps(state))
        (project / "BOARD.kicad_sch").write_text("(kicad_sch)")
        if status == "ok":
            (project / "BOARD.kicad_pcb").write_text("(kicad_pcb)")
        _publish(project, rc=0 if status == "ok" else 6, generation=1)
        if status != "running":
            store.finish_project(project_id, status, stem="BOARD", dir_path=str(workspace))
        else:
            live = web._fresh_run_state()
            live.update(project_id=project_id, user_id=account.id, ws=str(workspace),
                        project_dir=str(project), stem="BOARD", running=True,
                        attempt_generation=1, token=render_serving._register_project_dir(project))
            web._LIVE_RUNS[project_id] = live
        await user.open("/login")
        user.find("Email").type("history@example.com")
        user.find("Password").type("history-password").trigger("keydown.enter")
        await user.should_see("design a PCB from a sentence")
        await user.open(f"/?project={project_id}")
        user.find("Place/Route").click()
        await user.should_see("Watch layout history")
        await user.should_see("Download timelapse (.html)")


@pytest.fixture
def anyio_backend():
    return "asyncio"
