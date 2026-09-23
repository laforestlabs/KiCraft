"""What a user is told when a design stops, and what they can do about it.

Four defects the live Surprise-me walkthrough found on 2026-09-23:

1. The failure text was a machine dump. The legacy driver prints its result
   mapping on stdout (``-> {'ok': False, 'errors': [...]}``); the activity line
   adopted that line and ``run_finished`` then copied it into the failure fact,
   so the workspace, the projects row and the screen-reader announcement all read
   as unreadable Python -- while the sentences INSIDE the dump named the exact
   gate that refused the design.
2. A reopened project could not say why it stopped: a legacy run records no
   ``stage_status`` entry for a stage that never committed, so a wiring failure
   was reported as "stopped during BOM / no detailed cause was saved" even though
   its own transcript named the failing stage and the offending net.
3. A design that stopped in its LAST stage (wiring) offered "Rebuild board" --
   a button that silently did nothing, because a design that never committed has
   nothing to build. The honest action there is to finish the design stages.
4. The screen-reader alert kept the previous attempt's sentence and spoke the raw
   dump.

Pure module-function + store tests: no NiceGUI client, no LLM, no build.
"""
from __future__ import annotations

import json

import pytest

from kicraft.server import activity, web
from kicraft.server.accounts import AccountStore

DUMP = ("-> {'ok': False, 'errors': ['9.11 net coverage: wiring produced no "
        "connections — every part pin is uncovered'], 'offenders': ['U2.3']}")
LEGACY_BLOCK = ("[legacy] rc=1\n"
                "  [ok  ] intent           cost=$0.0007  attempts=1\n"
                "  [FAIL] wiring           cost=$0.0039  attempts=1\n"
                f"         {DUMP}")


@pytest.fixture
def store(tmp_path, monkeypatch):
    s = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
    monkeypatch.setattr(web, "_STORE", s)
    web._DISK_SIGNALS.clear()
    yield s
    web._DISK_SIGNALS.clear()


@pytest.fixture
def user_id(store):
    return store.create_user("dev@example.com", "hunter2hunter2").id


# ---- the failure text a person may be shown --------------------------------


def test_dump_becomes_the_sentences_it_contains():
    """The gate that refused the design is the useful part of a dump."""
    assert activity.human_failure_text(DUMP) == (
        "9.11 net coverage: wiring produced no connections — every part pin is "
        "uncovered")


def test_dump_inside_a_multi_line_log_block_still_yields_its_cause():
    """The driver's whole block lands on one activity line; the dump is not the
    first line of it."""
    assert activity.human_failure_text(LEGACY_BLOCK) == (
        "9.11 net coverage: wiring produced no connections — every part pin is "
        "uncovered")


def test_dump_with_nothing_to_say_is_refused():
    """Better to have no headline than to quote ``{'ok': False}``."""
    assert activity.human_failure_text("{'ok': False}") is None
    assert activity.human_failure_text("-> None") is None
    assert activity.human_failure_text("Traceback (most recent call last):") is None


def test_ordinary_statements_and_log_lines_pass_through():
    for text in ("Wiring failed", "Build stopped during Wiring",
                 "[3/5] verify: DRC clean", "Run stopped"):
        assert activity.human_failure_text(text) == text


def test_run_finished_keeps_the_recorded_cause_over_a_raw_dump():
    """A structured event already said WHY ("Wiring failed"); the driver's stdout
    dump that follows must not replace it."""
    act = activity.reduce_activity(None, {"kind": "run_started", "run_id": "r"})
    act = activity.reduce_activity(act, {"kind": "stage_done", "stage": "wiring",
                                         "ok": False})
    act = activity.reduce_activity(act, {"kind": "build_log", "text": DUMP})
    act = activity.reduce_activity(act, {"kind": "run_finished", "status": "failed",
                                         "stage": "wiring"})
    assert act["failure"]["message"] == "Wiring failed"
    assert act["failure"]["stage"] == "wiring"


def test_run_finished_without_a_recorded_cause_lifts_the_dumps_sentences():
    act = activity.reduce_activity(None, {"kind": "run_started", "run_id": "r"})
    act = activity.reduce_activity(act, {"kind": "build_log", "text": DUMP})
    act = activity.reduce_activity(act, {"kind": "run_finished", "status": "failed",
                                         "stage": "wiring"})
    assert act["failure"]["message"].startswith("9.11 net coverage")
    assert "'ok': False" not in act["failure"]["message"]


# ---- what a reopened project reports ---------------------------------------


def _failed_workspace(root, *, failing_stage: str, pending_question: bool):
    """A workspace shaped like a live run that died before any board: four stages
    committed, its last stage uncommitted, and its transcript naming both the
    stage and the gate that refused it."""
    (root / ".kicraft").mkdir(parents=True, exist_ok=True)
    state = {
        "project_stem": "EIGHT_CHANNEL_LIGHT_FRONTEND",
        "stage_status": {s: {"ok": True}
                         for s in ("intent", "functional_spec", "architecture", "bom")},
        "open_questions": (
            [{"text": "Add the MCU decoupling caps?", "stage": failing_stage,
              "blocking": True, "answer": None}] if pending_question else []),
    }
    (root / ".kicraft" / "state.json").write_text(json.dumps(state))
    events = [
        {"kind": "run_started", "run_id": "p1-x", "seq": 1},
        {"kind": "stage_start", "stage": failing_stage, "seq": 2},
        {"kind": "stage_done", "stage": failing_stage, "ok": False, "seq": 3},
        {"kind": "build_log", "text": LEGACY_BLOCK, "seq": 4},
        {"kind": "run_finished", "status": "failed", "stage": failing_stage, "seq": 5},
    ]
    (root / "events.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events))


def _failed_project(store, user_id, brief, *, failing_stage, pending_question=False):
    pid = store.create_project(user_id, brief)
    root = store.projects_dir / str(user_id) / str(pid)
    _failed_workspace(root, failing_stage=failing_stage,
                      pending_question=pending_question)
    store.finish_project(pid, "failed", dir_path=str(root),
                         stem="EIGHT_CHANNEL_LIGHT_FRONTEND")
    web._DISK_SIGNALS.clear()
    return store.get_project(pid)


def test_driver_prose_is_not_quoted_as_a_failure_cause():
    """A driver's informational line describes what the pipeline is doing. Only a
    self-identifying gate sentence may be reported as the reason a run stopped."""
    assert web._gate_cause(
        "native design: keeping post-wiring authorship in its own schema; "
        "current manufacturing will build an isolated snapshot") is None
    assert web._gate_cause(
        "  [ok  ] intent cost=$0.0007 attempts=1") is None
    assert web._gate_cause("-> {'ok': False, 'errors': ['9.11 net coverage: "
                           "wiring produced no connections']}").startswith(
        "9.11 net coverage")


@pytest.mark.parametrize("pending_question", [False, True])
def test_reopened_failure_names_its_stage_and_cause(store, user_id, pending_question):
    """A wiring failure reads as a wiring failure with the offending gate -- not
    as "stopped during BOM" with no cause."""
    p = _failed_project(store, user_id, "eight ambient-light channels",
                        failing_stage="wiring", pending_question=pending_question)

    pres = web._project_presentation(p)

    assert pres["status"] == "failed"
    assert pres["stage"] == "wiring"
    assert pres["headline"].startswith("9.11 net coverage")
    assert "{'ok'" not in pres["headline"]
    assert "{'ok'" not in pres["detail"]


@pytest.mark.parametrize("pending_question", [False, True])
def test_failed_last_stage_offers_finishing_the_design(store, user_id,
                                                       pending_question):
    """The action offered must be possible: the design's last stage never
    committed, so the way forward is to run it -- "Rebuild board" is a dead end
    (there is nothing to build) and must not be offered."""
    p = _failed_project(store, user_id, "eight ambient-light channels",
                        failing_stage="wiring", pending_question=pending_question)

    assert web._project_presentation(p)["action"] == "continue"


def test_failure_of_a_committed_design_still_offers_a_rebuild(store, user_id):
    """The opposite case is unchanged: when every design stage committed, a
    failed board build is a rebuild, not a design re-drive."""
    pid = store.create_project(user_id, "a routed board")
    root = store.projects_dir / str(user_id) / str(pid)
    (root / ".kicraft").mkdir(parents=True, exist_ok=True)
    (root / ".kicraft" / "state.json").write_text(json.dumps({
        "project_stem": "DONE_BOARD",
        "stage_status": {s: {"ok": True} for s in
                         ("intent", "functional_spec", "architecture", "bom", "wiring")},
        "bom": {"connections": [{"net_name": "+3V3", "endpoints": []}]},
    }))
    store.finish_project(pid, "failed", dir_path=str(root), stem="DONE_BOARD")
    web._DISK_SIGNALS.clear()

    assert web._project_presentation(store.get_project(pid))["action"] == "rebuild"
