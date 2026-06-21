"""Tests for registering self-eval runs as admin-only projects and for the
self-eval full-fidelity event sink."""
from __future__ import annotations

import json

from kicraft.server.accounts import AccountStore
from kicraft.eval.self_eval import _event_writer


def _store(tmp_path):
    return AccountStore(tmp_path / "accounts.db", tmp_path / "projects")


def _make_run(root, batch: str, name: str, *, stem="DEMO_BOARD",
              brief="A demo board", board=True):
    run = root / batch / name
    (run / ".kicraft").mkdir(parents=True)
    (run / ".kicraft" / "state.json").write_text(
        json.dumps({"project_stem": stem}), encoding="utf-8")
    (run / "brief.txt").write_text(brief, encoding="utf-8")
    if board:
        gen = run / "generated" / stem
        gen.mkdir(parents=True)
        (gen / f"{stem}.kicad_pcb").write_text("(kicad_pcb)", encoding="utf-8")
    return run


def test_sync_registers_run_as_private_admin_project(tmp_path):
    store = _store(tmp_path)
    admin = store.create_user("admin@e.st", "pw")
    store.set_role("admin@e.st", "admin")
    root = tmp_path / "self_eval"
    run = _make_run(root, "20260101T000000Z", "run_00_demo")

    res = store.sync_eval_projects(roots=[root])
    assert res == {"registered": 1, "skipped": 0, "owner_id": admin.id}

    projs = store.list_projects(admin.id)
    assert len(projs) == 1
    p = projs[0]
    assert p.dir_path == str(run)
    assert p.is_public is False
    assert p.board_code.startswith("EV-")
    assert p.status == "ok"
    assert p.brief == "A demo board"
    assert p.project_stem == "DEMO_BOARD"
    assert p.viewed_at is not None  # pre-seen, so it never hijacks the auto-open


def test_sync_is_idempotent(tmp_path):
    store = _store(tmp_path)
    admin = store.create_user("admin@e.st", "pw")
    store.set_role("admin@e.st", "admin")
    root = tmp_path / "self_eval"
    _make_run(root, "b1", "run_00_demo")

    first = store.sync_eval_projects(roots=[root])
    second = store.sync_eval_projects(roots=[root])
    assert first["registered"] == 1
    assert second == {"registered": 0, "skipped": 1, "owner_id": admin.id}
    assert len(store.list_projects(admin.id)) == 1


def test_sync_status_failed_without_a_board(tmp_path):
    store = _store(tmp_path)
    store.create_user("admin@e.st", "pw")
    store.set_role("admin@e.st", "admin")
    root = tmp_path / "self_eval"
    _make_run(root, "b1", "run_01_dead", board=False)  # no .kicad_pcb

    store.sync_eval_projects(roots=[root])
    p = store.list_projects(store.first_admin_id())[0]
    assert p.status == "failed"


def test_sync_noop_without_admin(tmp_path):
    store = _store(tmp_path)
    store.create_user("plain@e.st", "pw")  # no admin role granted
    root = tmp_path / "self_eval"
    _make_run(root, "b1", "run_00_demo")

    res = store.sync_eval_projects(roots=[root])
    assert res == {"registered": 0, "skipped": 0, "owner_id": None}


def test_sync_owns_runs_under_the_given_owner(tmp_path):
    store = _store(tmp_path)
    store.create_user("a@e.st", "pw")
    store.set_role("a@e.st", "admin")
    other = store.create_user("b@e.st", "pw")
    root = tmp_path / "self_eval"
    _make_run(root, "b1", "run_00_demo")

    store.sync_eval_projects(roots=[root], owner_id=other.id)
    assert len(store.list_projects(other.id)) == 1


def test_event_writer_lean_drops_token_deltas(tmp_path):
    p = tmp_path / "lean.jsonl"
    w = _event_writer(p, full=False)
    w({"kind": "stage_done", "ok": True})
    w({"kind": "answer_delta", "text": "x"})
    w({"kind": "reasoning_delta", "text": "y"})
    kinds = [json.loads(l)["kind"] for l in p.read_text().splitlines()]
    assert kinds == ["stage_done"]


def test_event_writer_full_keeps_everything(tmp_path):
    p = tmp_path / "full.jsonl"
    w = _event_writer(p, full=True)
    w({"kind": "stage_done", "ok": True})
    w({"kind": "answer_delta", "text": "x"})
    w({"kind": "reasoning_delta", "text": "y"})
    kinds = [json.loads(l)["kind"] for l in p.read_text().splitlines()]
    assert kinds == ["stage_done", "answer_delta", "reasoning_delta"]
