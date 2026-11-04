"""Tests for the public project browser web helpers (kicraft.server.web).

Exercises the pieces that need no live UI/connection context: the public detail
gate (the privacy boundary), the board thumbnail/source resolvers, the quality
badge derivation, and the clone-into-account action with its tier and quota gates.
Helpers are called directly and _store() is pointed at a tmp AccountStore,
mirroring tests/test_web_project_token.py.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from account_fixtures import seed_project

from kicraft.server import web
from kicraft.server.accounts import AccountStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    st = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
    monkeypatch.setattr(web, "_STORE", st)  # web._store() returns this
    return st

def _verify(store, user):
    """Mark a user email-verified via the real token flow, returning the refreshed
    user. Clone/quota tests isolate that gate from the email-verification gate."""
    tok = store.create_verification_token(user.id)
    return store.consume_verification_token(tok)


def _persist(store, user_id, brief, *, stem="BOARD", status="ok", is_public=True,
             with_render=False, with_pcb=True, parts=None):
    """Create + finish a project and lay down its on-disk tree, returning its id."""
    pid = seed_project(store, user_id, brief, is_public=is_public)
    base = store.projects_dir / str(user_id) / str(pid)
    gen = base / "generated" / stem
    gen.mkdir(parents=True, exist_ok=True)
    (base / "brief.txt").write_text(brief, encoding="utf-8")
    (base / "events.jsonl").write_text('{"kind":"x"}\n', encoding="utf-8")
    state = {"intent": {"goal": brief, "named_parts": []},
             "bom": {"parts": parts or []},
             "functional_spec": {"blocks": []}, "architecture": {"sheets": []}}
    (base / ".kicraft").mkdir(exist_ok=True)
    (base / ".kicraft" / "state.json").write_text(json.dumps(state), encoding="utf-8")
    (gen / f"{stem}.kicad_sch").write_text("(kicad_sch)\n", encoding="utf-8")
    if with_pcb:
        (gen / f"{stem}.kicad_pcb").write_text("(kicad_pcb)\n", encoding="utf-8")
    if with_render:
        r = gen / ".experiments" / "subcircuits" / "aa__bb" / "renders"
        r.mkdir(parents=True, exist_ok=True)
        (r / "routed_front_all.png").write_bytes(b"\x89PNG\r\n")
    store.finish_project(pid, status, stem=stem, dir_path=str(base))
    return pid


# ---- the public detail gate (privacy boundary) ----------------------------

def test_public_project_or_none_accepts_public_ok(store):
    u = _verify(store, store.create_user("a@e.st", "pw"))
    pid = _persist(store, u.id, "esp32 board")
    p = web._public_project_or_none(pid)
    assert p is not None and p.id == pid


def test_public_project_or_none_rejects_private_failed_missing(store):
    u = _verify(store, store.create_user("a@e.st", "pw", tier="pro"))
    priv = _persist(store, u.id, "secret", is_public=False)
    failed = _persist(store, u.id, "broken", status="failed")
    assert web._public_project_or_none(priv) is None         # private
    assert web._public_project_or_none(failed) is None        # not completed
    assert web._public_project_or_none(999999) is None        # missing
    assert web._public_project_or_none("not-an-int") is None  # garbage id


# ---- board thumbnail + source resolvers -----------------------------------

def test_board_thumb_url_none_without_render(store):
    u = _verify(store, store.create_user("a@e.st", "pw"))
    p = store.get_project(_persist(store, u.id, "b", with_render=False))
    assert web._board_thumb_url(p.dir_path, p.project_stem) is None


def test_board_thumb_url_present_with_render(store):
    u = _verify(store, store.create_user("a@e.st", "pw"))
    p = store.get_project(_persist(store, u.id, "b", with_render=True))
    url = web._board_thumb_url(p.dir_path, p.project_stem)
    assert url and url.startswith("/project/") and "/render/" in url
    assert "routed_front_all.png" in url


def test_board_source_prefers_stem_pcb(store):
    u = _verify(store, store.create_user("a@e.st", "pw"))
    p = store.get_project(_persist(store, u.id, "b", stem="MYBOARD"))
    gen = web._persisted_generated_dir(p.dir_path, p.project_stem)
    assert gen is not None and gen.name == "MYBOARD"
    _url, name = web._board_source(gen, "MYBOARD", "tok")
    assert name == "MYBOARD.kicad_pcb"


# ---- quality badge --------------------------------------------------------

def test_quality_badge_from_ws(tmp_path):
    def ws_with(name, payload):
        ws = tmp_path / name
        (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
        (ws / ".kicraft" / "synthesis_check.json").write_text(
            json.dumps(payload), encoding="utf-8")
        return ws

    clean = ws_with("clean", {"status": "ok", "checks": [{"name": "9.12 ERC", "ok": True}]})
    failed = ws_with("failed", {"status": "failed", "failed_checks": ["9.12 ERC"]})
    mixed = ws_with("mixed", {"status": "ok", "checks": [{"name": "X", "ok": False}]})
    assert web._quality_badge_from_ws(clean) == "fab_ready"
    assert web._quality_badge_from_ws(failed) == "erc_errors"
    assert web._quality_badge_from_ws(mixed) == "erc_errors"   # a failing check counts
    assert web._quality_badge_from_ws(tmp_path / "nope") == "unverified"  # no file


# ---- clone-into-account ---------------------------------------------------

def test_clone_creates_owned_public_copy(store):
    a = _verify(store, store.create_user("owner@e.st", "pw"))
    b = store.create_user("cloner@e.st", "pw")
    b = _verify(store, b)  # isolate clone behavior from the verification gate
    src_id = _persist(store, a.id, "esp32 board", stem="SRC",
                      parts=[{"ref": "U1", "mpn": "ESP32-S3"}])
    new_id, err = web._clone_project(store.get_project(src_id), b, make_private=False)
    assert err is None and new_id is not None
    clone = store.get_project(new_id)
    assert clone.user_id == b.id
    assert clone.cloned_from_id == src_id
    assert clone.status == "ok" and clone.is_public is True
    cdir = Path(clone.dir_path)
    assert cdir == store.projects_dir / str(b.id) / str(new_id)   # cloner's namespace
    assert (cdir / "generated" / "SRC" / "SRC.kicad_pcb").is_file()  # tree copied
    assert (cdir / ".kicraft" / "state.json").is_file()
    assert not (cdir / "events.jsonl").exists()                   # fresh history
    assert store.get_project(src_id).clone_count == 1            # source bumped


def test_clone_rebases_current_generation_artifacts(store):
    owner = _verify(store, store.create_user("source@e.st", "pw"))
    cloner = _verify(store, store.create_user("copy@e.st", "pw"))
    pid = _persist(store, owner.id, "board", stem="BOARD")
    source = store.get_project(pid)
    root = Path(source.dir_path)
    old = root / "generated" / "BOARD"
    current = root / "generated" / "attempt-7" / "BOARD"
    current.parent.mkdir(parents=True)
    old.rename(current)
    old.mkdir(parents=True)
    (old / "historical-only.kicad_sch").write_text("historical")
    (root / "kicraft_project.zip").write_bytes(b"stale package")
    state_path = root / ".kicraft" / "state.json"
    state = json.loads(state_path.read_text())
    state["project_stem"] = "BOARD"
    state["artifacts"] = {
        "project_dir": str(current),
        "root_sch": str(current / "BOARD.kicad_sch"),
        "routed_pcb": str(current / "BOARD.kicad_pcb"),
        "leaf_schs": [],
    }
    state_path.write_text(json.dumps(state))
    before = state_path.read_bytes()

    clone_id, error = web._clone_project(source, cloner, make_private=False)

    assert error is None
    clone = store.get_project(clone_id)
    clone_root = Path(clone.dir_path)
    artifact = json.loads((clone_root / ".kicraft" / "state.json").read_text())["artifacts"]
    clone_dir = clone_root / "generated" / "attempt-7" / "BOARD"
    assert Path(artifact["project_dir"]) == clone_dir
    assert Path(artifact["root_sch"]).read_text() == "(kicad_sch)\n"
    assert Path(artifact["routed_pcb"]).read_text() == "(kicad_pcb)\n"
    assert web._persisted_generated_dir(clone_root, "BOARD") == clone_dir
    assert not (clone_root / "generated" / "BOARD").exists()
    assert clone.zip_path is None
    assert not (clone_root / "kicraft_project.zip").exists()
    assert state_path.read_bytes() == before


def test_clone_rejects_symlinked_source_artifacts(store, tmp_path):
    owner = _verify(store, store.create_user("source-link@e.st", "pw"))
    cloner = _verify(store, store.create_user("copy-link@e.st", "pw"))
    pid = _persist(store, owner.id, "board", stem="BOARD")
    source = store.get_project(pid)
    outside = tmp_path / "outside.kicad_sch"
    outside.write_text("not source evidence")
    (Path(source.dir_path) / "generated" / "BOARD" / "outside.kicad_sch").symlink_to(outside)
    clone_id, error = web._clone_project(source, cloner, make_private=False)
    assert clone_id is None and error == "copy_error"
    assert store.get_project(pid).clone_count == 0


def test_clone_rejects_artifact_path_outside_source(store, tmp_path):
    owner = _verify(store, store.create_user("source-outside@e.st", "pw"))
    cloner = _verify(store, store.create_user("copy-outside@e.st", "pw"))
    pid = _persist(store, owner.id, "board", stem="BOARD")
    source = store.get_project(pid)
    state_path = Path(source.dir_path) / ".kicraft" / "state.json"
    state = json.loads(state_path.read_text())
    state["artifacts"] = {"project_dir": str(tmp_path)}
    state_path.write_text(json.dumps(state))
    clone_id, error = web._clone_project(source, cloner, make_private=False)
    assert clone_id is None and error == "copy_error"
    assert store.get_project(pid).clone_count == 0


def test_clone_refuses_source_with_active_run(store):
    owner = _verify(store, store.create_user("source-active@e.st", "pw"))
    cloner = _verify(store, store.create_user("copy-active@e.st", "pw"))
    pid = _persist(store, owner.id, "board", stem="BOARD")
    source = store.get_project(pid)
    store.begin_project_attempt(pid)
    clone_id, error = web._clone_project(source, cloner, make_private=False)
    assert clone_id is None and error == "missing"
    assert store.list_projects(cloner.id) == []


def test_clone_free_user_forced_public(store):
    a = _verify(store, store.create_user("owner@e.st", "pw"))
    free = _verify(store, store.create_user("free@e.st", "pw"))  # free tier
    source = store.get_project(_persist(store, a.id, "b"))
    new_id, err = web._clone_project(source, free, make_private=True)  # asks private
    assert err is None
    assert store.get_project(new_id).is_public is True  # tier gate forces public


def test_clone_paid_user_can_be_private(store):
    a = _verify(store, store.create_user("owner@e.st", "pw"))
    store.create_user("paid@e.st", "pw")
    paid = store.set_tier("paid@e.st", "pro")
    paid = _verify(store, paid)
    source = store.get_project(_persist(store, a.id, "b"))
    new_id, err = web._clone_project(source, paid, make_private=True)
    assert err is None
    assert store.get_project(new_id).is_public is False


def test_clone_blocked_by_quota(store):
    a = _verify(store, store.create_user("owner@e.st", "pw"))
    free = _verify(store, store.create_user("free2@e.st", "pw"))  # free: 1 design / week
    store.finish_project(store.create_project(free.id, "mine"), "ok")  # use the slot
    source = store.get_project(_persist(store, a.id, "b"))
    new_id, err = web._clone_project(source, free, make_private=False)
    assert new_id is None and err == "quota"


def test_can_make_private_tier_gate(store):
    free = store.create_user("f@e.st", "pw")
    store.create_user("p@e.st", "pw")
    paid = store.set_tier("p@e.st", "max")
    assert web._can_make_private(free) is False
    assert web._can_make_private(paid) is True
    assert web._can_make_private(None) is False
