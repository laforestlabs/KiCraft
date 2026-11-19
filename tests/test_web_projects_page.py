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

import hashlib
import importlib
import json
import sys
import zipfile

import pytest
from nicegui.testing.user_simulation import user_simulation

from account_fixtures import seed_project

from kicraft.build_outcome import REVIEW_REQUIRED_RC, USB_REVIEW_WARNING
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
        acct = store.consume_verification_token(store.create_verification_token(acct.id))
        store.record_consent(acct.id, LEGAL_VERSION)
        try:
            yield u, web, store, acct
        finally:
            web._STORE = None
            web._LIVE_RUNS.clear()


def _ok_project(store, user_id: int, brief: str, stem: str) -> int:
    """A finished ('ok') project row with a dir on disk, so the row renders its
    Open action and the visibility control (which only shows for status=='ok')."""
    pid = seed_project(store, user_id, brief)
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


async def test_project_visibility_refreshes_after_external_change(harness):
    import asyncio
    from nicegui import ui

    u, _web, store, acct = harness
    store.set_tier(EMAIL, "max")
    pid = _ok_project(store, acct.id, "private sensor", "SENSOR")
    await _login(u)
    await u.open("/projects")
    await u.should_see("SENSOR")
    assert next(iter(u.find(kind=ui.switch).elements)).value is False
    store.set_visibility(pid, True)
    await asyncio.sleep(2.5)
    assert next(iter(u.find(kind=ui.switch).elements)).value is True


async def test_downgraded_private_project_is_not_labeled_public(harness):
    u, _web, store, acct = harness
    store.set_tier(EMAIL, "max")
    pid = _ok_project(store, acct.id, "private sensor", "SENSOR")
    store.set_tier(EMAIL, "free")
    assert store.get_project(pid).is_public is False
    await _login(u)
    await u.open("/projects")
    await u.should_see("Private (retained after downgrade)")
    await u.should_not_see("Public in the community browser")


async def test_delete_requires_confirmation_then_removes_project(harness):
    """Delete pops a confirm dialog (project survives until confirmed), and the
    confirm purges the row from the store and the page."""
    from nicegui import ui

    u, web, store, acct = harness
    pid = _ok_project(store, acct.id, "throwaway blinky", "BLINKY")
    base = store.projects_dir / str(acct.id) / str(pid)
    assert base.exists()

    await _login(u)
    await u.open("/projects")
    await u.should_see("BLINKY")

    # The row Delete opens the confirm dialog (the per-project name proves it
    # actually opened); nothing is removed yet.
    u.find(marker="row-delete").click()
    await u.should_see('"BLINKY" will be deleted')
    assert store.get_project(pid) is not None  # still there pre-confirm

    # Confirm: row gone from the store, its on-disk tree purged, page refreshed.
    u.find(marker="confirm-delete").click()
    await u.should_see('Deleted "BLINKY"')
    assert store.get_project(pid) is None
    assert not base.exists()
    await u.should_see("No projects yet")


async def test_delete_can_be_cancelled(harness):
    """Cancel on the confirm dialog leaves the project untouched."""
    u, web, store, acct = harness
    pid = _ok_project(store, acct.id, "keep me", "KEEPER")

    await _login(u)
    await u.open("/projects")
    await u.should_see("KEEPER")
    u.find(marker="row-delete").click()
    await u.should_see('"KEEPER" will be deleted')
    u.find("Cancel").click()
    assert store.get_project(pid) is not None  # untouched


# ---- identity, truthful status, and no stale downloads ----------------------


async def test_row_is_named_from_its_brief_before_any_commit(harness):
    """A project with no committed stem yet is identifiable by its brief -- never
    an empty or "(building...)" label."""
    u, web, store, acct = harness
    seed_project(store, acct.id, "USB-C temperature logger")

    await _login(u)
    await u.open("/projects")
    await u.should_see("USB-C temperature logger")


async def test_committed_stem_appears_without_a_page_reload(harness):
    """The row is polled: a stem committed after the page loaded replaces the
    brief-derived title in place (and the brief stays the fallback)."""
    u, web, store, acct = harness
    pid = seed_project(store, acct.id, "usb battery bank")

    await _login(u)
    await u.open("/projects")
    await u.should_see("usb battery bank")

    base = store.projects_dir / str(acct.id) / str(pid)
    (base / ".kicraft").mkdir(parents=True, exist_ok=True)
    (base / ".kicraft" / "state.json").write_text(
        '{"project_stem": "USB_BANK"}', encoding="utf-8")
    store.finish_project(pid, "running", stem="USB_BANK", dir_path=str(base))

    await u.should_see("USB_BANK", retries=40)  # the 2 s poll picks it up


async def test_two_rows_keep_their_own_identity_across_updates(harness):
    """Rows are updated by project id: one project changing must never rewrite or
    reorder another's row."""
    u, web, store, acct = harness
    store.set_tier(EMAIL, "pro")  # both rows must fit within the project quota
    first = seed_project(store, acct.id, "first board")
    second = seed_project(store, acct.id, "second board")
    store.finish_project(second, "failed")

    await _login(u)
    await u.open("/projects")
    await u.should_see("first board")
    await u.should_see("second board")
    await u.should_see("Failed")

    store.finish_project(first, "interrupted")
    await u.should_see("Interrupted", retries=40)
    await u.should_see("Failed")  # the other row is untouched


async def test_queued_build_without_a_live_registry_is_not_interrupted(harness):
    """A build the worker drives outlives the web process: with an empty
    _LIVE_RUNS the row must read as queued work, not as a lost run, and must not
    offer a duplicate build."""
    u, web, store, acct = harness
    pid = seed_project(store, acct.id, "queued board")
    base = store.projects_dir / str(acct.id) / str(pid)
    base.mkdir(parents=True, exist_ok=True)
    store.enqueue_build(workspace=str(base), project_id=pid, user_id=acct.id)

    await _login(u)
    await u.open("/projects")
    await u.should_see("Queued for board build")
    await u.should_not_see("Interrupted")
    await u.should_not_see("Rebuild board")


async def test_failed_row_with_an_old_package_offers_no_download(harness):
    """A leftover zip from an earlier build is evidence, not a deliverable: a
    failed row never offers it (the rebuild action is what it offers instead)."""
    u, web, store, acct = harness
    pid = seed_project(store, acct.id, "doomed board")
    base = store.projects_dir / str(acct.id) / str(pid)
    (base / ".kicraft").mkdir(parents=True, exist_ok=True)
    (base / ".kicraft" / "state.json").write_text(
        json.dumps({"project_stem": "DOOMED",
                    "stage_status": {k: {"ok": True} for k in web.DESIGN_STAGES}}),
        encoding="utf-8")
    (base / "kicraft_project.zip").write_bytes(b"stale package")
    store.finish_project(pid, "failed", stem="DOOMED", dir_path=str(base),
                         zip_path=str(base / "kicraft_project.zip"))

    await _login(u)
    await u.open("/projects")
    await u.should_see("Failed")
    await u.should_not_see("Download")
    await u.should_see("Rebuild board")


def _review_project(store, user_id, *, status="review_required"):
    pid = seed_project(store, user_id, "USB controller")
    root = store.projects_dir / str(user_id) / str(pid)
    project = root / "generated" / "attempt-0" / "USB"
    project.mkdir(parents=True)
    (root / ".kicraft").mkdir()
    board = project / "USB.kicad_pcb"
    board.write_bytes(b"(kicad_pcb)")
    schematic = project / "USB.kicad_sch"
    schematic.write_text("(kicad_sch)", encoding="utf-8")
    config = project / "USB_autoplacer.json"
    config.write_text("{}", encoding="utf-8")
    kicad_project = project / "USB.kicad_pro"
    kicad_project.write_text("{}", encoding="utf-8")
    digest = hashlib.sha256(board.read_bytes()).hexdigest()
    provenance = board.with_suffix(".provenance.json")
    provenance.write_text(json.dumps({
        "run_id": "layout-run", "run_started_at": 10.0,
        "fresh": True, "source_kind": "routed", "md5": hashlib.md5(board.read_bytes()).hexdigest(),
    }), encoding="utf-8")
    verification = board.with_suffix(".verification.receipt.json")
    verification.write_text(json.dumps({
        "ok": True, "physical_ok": True, "review_required": True,
        "fab_acceptable": False, "board_sha256": digest,
    }), encoding="utf-8")
    receipt = board.with_suffix(".engineering-review.receipt.json")
    receipt.write_text(json.dumps({
        "mode": "engineering_review", "physical_ok": True, "fabrication_approved": False,
        "board_sha256": digest, "attempt_generation": 0, "verification_receipt": verification.name,
        "run_id": "layout-run", "run_started_at": 10.0,
    }), encoding="utf-8")
    archive = project / "USB_review.zip"
    with zipfile.ZipFile(archive, "w") as out:
        out.writestr("DRAFT-NOT-FOR-FABRICATION.txt", USB_REVIEW_WARNING + "\n")
        for path in (board, schematic, config, kicad_project, receipt, verification):
            out.write(path, path.name)
    state = {
        "project_stem": "USB",
        "stage_status": {k: {"ok": True} for k in
                         ("intent", "functional_spec", "architecture", "bom", "wiring")},
        "artifacts": {"status": "review_required", "project_dir": str(project),
                      "routed_pcb": str(board), "review_zip": str(archive), "fab_zip": None},
    }
    (root / ".kicraft" / "state.json").write_text(json.dumps(state), encoding="utf-8")
    store.finish_project(pid, status, stem="USB", dir_path=str(root))
    return pid, root, board, archive, receipt, provenance


async def test_review_draft_list_and_reopen(harness):
    from nicegui import ui

    u, _, store, acct = harness
    pid, _, _, archive, _, _ = _review_project(store, acct.id)
    await _login(u)
    await u.open("/projects")
    await u.should_see("Draft ready — USB fabrication confirmation needed")
    await u.should_not_see("Failed")
    u.find("Download editable draft source (.zip)", kind=ui.button).click()
    download = await u.download.next()
    assert download.status_code == 200
    assert download.content == archive.read_bytes()
    assert "USB_review.zip" in download.headers["content-disposition"]
    await u.open(f"/?project={pid}")
    await u.should_see("Download editable draft source (.zip)")
    await u.should_not_see("Download KiCad project (.zip)")


@pytest.mark.parametrize("damage", [
    "board", "receipt", "run", "generation", "archive", "source", "fabrication_member", "old_archive",
])
async def test_review_draft_rejects_stale_evidence(harness, damage):
    u, _, store, acct = harness
    pid, root, board, archive, receipt, provenance = _review_project(store, acct.id)
    if damage == "board":
        board.write_bytes(b"changed board")
    elif damage == "receipt":
        data = json.loads(receipt.read_text(encoding="utf-8"))
        data["physical_ok"] = False
        receipt.write_text(json.dumps(data), encoding="utf-8")
    elif damage == "run":
        data = json.loads(provenance.read_text(encoding="utf-8"))
        data["run_id"] = "different-build"
        provenance.write_text(json.dumps(data), encoding="utf-8")
    elif damage == "generation":
        store.begin_project_attempt(pid)
        store.finish_project(pid, "review_required", dir_path=str(root))
    elif damage == "archive":
        archive.write_bytes(b"not a zip")
    elif damage == "fabrication_member":
        fab = board.parent / "fab" / "USB.gbr"
        fab.parent.mkdir()
        fab.write_text("older manufacturing output", encoding="utf-8")
        with zipfile.ZipFile(archive, "a") as out:
            out.write(fab, "fab/USB.gbr")
    elif damage == "old_archive":
        # Even a current board and mutually agreeing receipts cannot bless
        # an archive from an earlier run (including an identical board).
        for path in (provenance, receipt):
            data = json.loads(path.read_text(encoding="utf-8"))
            data["run_id"] = "new-build"
            path.write_text(json.dumps(data), encoding="utf-8")
    else:
        (board.parent / "USB.kicad_sch").write_text("changed schematic", encoding="utf-8")
    await _login(u)
    await u.open("/projects")
    await u.should_see("The draft or its verification no longer matches the current attempt.")
    await u.should_not_see("Download editable draft source (.zip)")
    generation = store.get_project(pid).attempt_generation
    assert (await u.http_client.get(f"/project-draft/{pid}/{generation}")).status_code == 404


async def test_review_download_rechecks_board_after_render(harness):
    from nicegui import ui

    u, _, store, acct = harness
    _, _, board, _, _, _ = _review_project(store, acct.id)
    await _login(u)
    await u.open("/projects")
    button = u.find("Download editable draft source (.zip)", kind=ui.button)
    board.write_bytes(b"changed after rendering the button")
    button.click()
    download = await u.download.next()
    assert download.status_code == 404
    assert "Draft is no longer available." in download.text


async def test_fab_button_from_previous_attempt_cannot_offer_old_package(harness):
    from nicegui import ui

    u, _, store, acct = harness
    pid, root, _, _, _, _ = _review_project(store, acct.id)
    fab = root / "kicraft_project.zip"
    fab.write_bytes(b"previous fabrication package")
    store.finish_project(pid, "ok", stem="USB", dir_path=str(root), zip_path=str(fab))
    await _login(u)
    await u.open("/projects")
    button = u.find("Download", kind=ui.button)
    store.begin_project_attempt(pid)
    store.finish_project(pid, "review_required", dir_path=str(root))
    button.click()
    await u.should_see("This fabrication package is no longer current.")


async def test_public_review_draft_is_not_a_download_or_clone(harness):
    u, _, store, _ = harness
    other = store.create_user("other@example.com", PASSWORD)
    pid, _, board, _, _, _ = _review_project(store, other.id)
    assert (await u.http_client.get(f"/project-draft/{pid}/0")).status_code == 404
    await _login(u)
    await u.open("/browse")
    await u.should_see("Draft — not approved for fabrication")
    await u.open(f"/p/{pid}")
    await u.should_see("Draft ready — USB fabrication confirmation needed")
    await u.should_see("Editable sources are available only to the project owner.")
    await u.should_not_see("Download")
    await u.should_not_see("Clone")
    assert (await u.http_client.get(f"/project-draft/{pid}/0")).status_code == 404
    board.write_bytes(b"changed since published")
    await u.open(f"/p/{pid}")
    await u.should_see("Saved draft unavailable — not approved for fabrication")
    await u.should_see("This project's files aren't available to preview.")


@pytest.mark.parametrize("has_evidence", [True, False])
async def test_restart_only_delivers_review_attempt_with_current_evidence(harness, monkeypatch, has_evidence):
    u, web, store, acct = harness
    monkeypatch.setattr(web.notify, "notify_run_event", lambda *a, **kw: None)
    pid, root, _, archive, _, _ = _review_project(store, acct.id, status="running")
    if not has_evidence:
        archive.unlink()
    job_id = store.enqueue_build(workspace=str(root), project_id=pid, user_id=acct.id,
                                 attempt_generation=0)
    assert store.claim_build(job_id, "test-worker")
    assert store.finish_build(job_id, rc=REVIEW_REQUIRED_RC)
    web._finalize_orphan(store.get_build_job(job_id))
    await _login(u)
    await u.open("/projects")
    if has_evidence:
        await u.should_see("Draft ready — USB fabrication confirmation needed")
        await u.should_see("Download editable draft source (.zip)")
    else:
        await u.should_see("Failed")
        await u.should_see("The build did not produce a current, verifiable editable draft.")
        await u.should_not_see("Download")


@pytest.mark.parametrize("kind", ["build", "manual_route"])
async def test_rebuild_review_is_terminal_without_failure(harness, monkeypatch, kind):
    u, web, store, acct = harness
    monkeypatch.setattr(web.notify, "notify_run_event", lambda *a, **kw: None)
    pid, root, _, _, _, _ = _review_project(store, acct.id, status="running")
    state = {
        "ws": str(root), "project_id": pid, "user_id": acct.id, "attempt_generation": 0,
        "brief": "USB controller", "events": [], "running": True,
        "zip": str(root / "kicraft_project.zip"),
    }
    (root / "kicraft_project.zip").write_bytes(b"previous fabrication package")
    monkeypatch.setattr(web, "_drive_build_queue", lambda *a, **kw: REVIEW_REQUIRED_RC)
    web._rerun_build_worker(state, kind)
    await _login(u)
    await u.open(f"/?project={pid}")
    await u.should_see("Download editable draft source (.zip)")
    await u.should_not_see("Download KiCad project (.zip)")
    await u.should_not_see("Run stopped")
