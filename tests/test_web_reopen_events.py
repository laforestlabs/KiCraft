"""Event durability, ordering and attribution (web._record_progress_event,
web._load_events, web._load_attempts).

The web transcript used to be a single flat list: journal-only provenance events
were appended AFTER the final transcript, so an old attempt's failure could sit
behind a newer attempt's success, and a reopened page could not tell the current
attempt from history. These are pure-data tests: no NiceGUI, no pcbnew, no
provider.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from kicraft.server import activity, web


@pytest.fixture
def root(tmp_path, monkeypatch):
    """A run whose durable dir is `tmp_path`."""
    monkeypatch.setattr(web, "_project_dir", lambda _state: tmp_path)
    return tmp_path


def _state(**over):
    state = {"events": [], "run_id": "p7-100", "_event_seq": 0, "journal": None}
    state.update(over)
    return state


def _kinds(events):
    return [event.get("kind") for event in events]


# ---- reading ---------------------------------------------------------------

def test_load_events_reads_and_tolerates_corrupt_lines(tmp_path):
    (tmp_path / "events.jsonl").write_text(
        '{"kind": "build_log", "text": "a"}\n'
        "not json -- skipped\n"
        "\n"  # blank -- skipped
        '{"kind": "stage", "text": "b"}\n'
        "[1, 2, 3]\n",  # non-dict JSON -- skipped
        encoding="utf-8",
    )
    evs = web._load_events(tmp_path)
    assert [e.get("text") for e in evs] == ["a", "b"]


def test_load_events_missing_or_none_is_empty(tmp_path):
    assert web._load_events(None) == []
    assert web._load_events(tmp_path) == []  # no events.jsonl in this dir


def test_torn_final_line_is_ignored(tmp_path):
    """A snapshot interrupted mid-write leaves a torn last line, not a crash."""
    (tmp_path / "events.jsonl").write_text(
        '{"kind": "build_log", "text": "kept"}\n{"kind": "build_log", "text": "tor',
        encoding="utf-8")
    assert [e.get("text") for e in web._load_events(tmp_path)] == ["kept"]


def test_metadata_free_legacy_transcript_stays_one_older_group(tmp_path):
    """A pre-envelope transcript has no run ids or sequence numbers: it must
    survive as its own history group, with its original order, without invented
    timestamps or reassignment to a newer attempt."""
    (tmp_path / "events.jsonl").write_text(
        json.dumps({"kind": "build_log", "text": "legacy one"}) + "\n"
        + json.dumps({"kind": "stage_done", "stage": "intent"}) + "\n",
        encoding="utf-8")
    events = web._load_events(tmp_path)
    groups = activity.group_attempts(events)
    assert len(groups) == 1 and groups[0]["legacy"] is True
    assert [e.get("text") for e in groups[0]["events"]] == ["legacy one", None]


# ---- writing ---------------------------------------------------------------

def test_record_stamps_an_envelope_and_journals_structural_kinds(root):
    state = _state()
    recorded = web._record_progress_event(
        state, {"kind": "work_unit_plan", "stage": "bom", "unit_id": "bom-s000"})

    assert recorded["version"] == 1
    assert recorded["run_id"] == "p7-100"
    assert recorded["seq"] == 1
    assert recorded["ts"]
    assert recorded["event_id"] == "p7-100:1"
    # Durable while the run is in flight: a crash still leaves the record.
    journaled = [json.loads(line)
                 for line in (root / "provenance.jsonl").read_text().splitlines()]
    assert [e["event_id"] for e in journaled] == ["p7-100:1"]


def test_token_deltas_stay_memory_only_but_are_stamped(root):
    """Deltas are high-volume and low-information: never journaled, but stamped
    so the final transcript merges deterministically."""
    state = _state()
    web._record_progress_event(state, {"kind": "reasoning_delta", "text": "private draft"})

    (event,) = state["events"]
    assert event["kind"] == "reasoning_delta" and event["text"] == "private draft"
    assert event["event_id"] == "p7-100:1"
    assert not (root / "provenance.jsonl").exists()


def test_journal_write_failure_keeps_live_activity(root):
    """A storage failure must degrade to a warning -- the run keeps streaming."""
    state = _state()
    state["journal"] = web._EventJournal(Path("/dev/null/provenance.jsonl"))

    web._record_progress_event(state, {"kind": "tool", "call_id": "c-1", "name": "x"})

    assert state["journal_failed"] is True
    assert state["events"][0]["kind"] == "tool"
    assert state["activity"]["tools"][0]["name"] == "x"


def test_tool_calls_and_results_pair_by_local_id(root):
    state = _state()
    for ordinal in (1, 2):
        web._record_progress_event(state, {
            "kind": "tool", "call_id": f"tok-{ordinal}", "name": "lookup_lcsc",
            "args": {"mpn": "TPS54331"}})
    for ordinal in (1, 2):
        web._record_progress_event(state, {
            "kind": "tool_result", "call_id": f"tok-{ordinal}", "name": "lookup_lcsc",
            "output": f"result {ordinal}", "ok": True, "duration_ms": 12,
            "output_chars": 8, "output_truncated": False})

    tools = state["activity"]["tools"]
    assert [t["call_id"] for t in tools] == ["tok-1", "tok-2"]
    assert [t["output"] for t in tools] == ["result 1", "result 2"]
    assert all(t["status"] == "returned" and t["duration_ms"] == 12 for t in tools)


# ---- reopen: ordering, no duplication, no stale state ----------------------

def test_reopen_shows_the_attempt_events_once_in_order(root):
    state = _state()
    web._record_progress_event(state, {"kind": "stage_start", "stage": "bom"})
    web._record_progress_event(
        state, {"kind": "tool", "call_id": "tok-1", "name": "lookup_lcsc", "args": {}})
    web._record_progress_event(
        state, {"kind": "tool", "call_id": "tok-2", "name": "lookup_lcsc", "args": {}})
    web._record_progress_event(
        state, {"kind": "tool_result", "call_id": "tok-1", "name": "lookup_lcsc",
                "output": "one", "ok": True})
    web._record_progress_event(
        state, {"kind": "tool_result", "call_id": "tok-2", "name": "lookup_lcsc",
                "output": "two", "ok": False})
    web._record_progress_event(
        state, {"kind": "retry", "stage": "bom", "failure_kind": "commit_rejected",
                "errors": ["missing footprint"]})
    web._record_progress_event(
        state, {"kind": "stage_diagnostic", "stage": "bom", "code": "unresolved_pin",
                "message": "U1 pin 7 unconnected"})
    web._record_progress_event(
        state, {"kind": "run_error", "failure_kind": "unexpected_error",
                "exception_type": "TimeoutError", "message": "TimeoutError: slow"})
    web._record_progress_event(state, {"kind": "run_finished", "status": "failed"})

    # The snapshot the finalize writes, WITHOUT calling _persist_project: the
    # journal alone must already carry the whole structural history.
    del state["events"]
    reopened = web._load_events(root)
    assert _kinds(reopened) == [
        "stage_start", "tool", "tool", "tool_result", "tool_result", "retry",
        "stage_diagnostic", "run_error", "run_finished",
    ]
    assert [e["seq"] for e in reopened] == list(range(1, 10))
    assert len({e["event_id"] for e in reopened}) == 9


def test_snapshot_and_journal_are_not_duplicated_on_reopen(root):
    """events.jsonl and provenance.jsonl hold the same structural events; the
    merged read must yield one copy of each."""
    state = _state()
    web._record_progress_event(state, {"kind": "stage_start", "stage": "intent"})
    web._record_progress_event(state, {"kind": "stage_done", "stage": "intent", "ok": True})
    snapshot = web._serialize_events(state["events"], workspace=str(root))
    (root / "events.jsonl").write_text(snapshot, encoding="utf-8")

    reopened = web._load_events(root)
    assert _kinds(reopened) == ["stage_start", "stage_done"]
    assert len(reopened) == 2


def test_journal_only_events_sort_inside_their_own_attempt(root):
    """A crash leaves journal-only events; they belong to the attempt that wrote
    them, not appended after whatever the snapshot already held."""
    older = {
        "kind": "stage_start", "stage": "intent", "version": 1, "run_id": "p9-old",
        "seq": 1, "ts": "2026-01-01T00:00:01+00:00", "event_id": "p9-old:1",
    }
    newer = {
        "kind": "run_started", "version": 1, "run_id": "p9-new", "seq": 1,
        "ts": "2026-02-01T00:00:00+00:00", "event_id": "p9-new:1",
    }
    journal_only = {
        "kind": "stage_done", "stage": "intent", "ok": False, "version": 1,
        "run_id": "p9-old", "seq": 2, "ts": "2026-01-01T00:00:02+00:00",
        "event_id": "p9-old:2",
    }
    (root / "events.jsonl").write_text(
        "\n".join(json.dumps(e) for e in (older, newer)) + "\n", encoding="utf-8")
    (root / "provenance.jsonl").write_text(
        json.dumps(journal_only) + "\n", encoding="utf-8")

    groups = web._load_attempts(root)
    assert [[e["event_id"] for e in g["events"]] for g in groups] == [
        ["p9-old:1", "p9-old:2"], ["p9-new:1"]]


def test_newer_attempt_does_not_inherit_the_older_failure(root):
    """A continuation is history plus a fresh attempt: the older run_finished
    stays in its own group and the current one starts clean."""
    state = _state()
    web._record_progress_event(state, {"kind": "stage_start", "stage": "wiring"})
    web._record_progress_event(
        state, {"kind": "run_error", "failure_kind": "provider_error",
                "message": "provider_error: 503"})
    web._record_progress_event(state, {"kind": "run_finished", "status": "failed"})

    second = _start_attempt_for_test(state)
    web._record_progress_event(second, {"kind": "stage_start", "stage": "wiring"})
    web._record_progress_event(second, {"kind": "run_finished", "status": "ok"})

    attempts = activity.group_attempts(web._load_events(root))
    assert len(attempts) == 2
    assert attempts[0]["events"][-1]["status"] == "failed"
    current = activity.current_attempt(attempts)
    assert current["events"][0]["kind"] == "run_started"
    assert current["events"][-1]["status"] == "ok"
    assert current["events"][-1]["run_id"] != attempts[0]["events"][0]["run_id"]
    # And the reducer of the current attempt reports the CURRENT outcome.
    assert activity.reduce_activity(None, current["events"][0])["phase_status"] == "running"
    act = web._reduce_attempt_activity(current["events"])
    assert act["phase_status"] == "complete" and act["failure"] is None


def _start_attempt_for_test(state: dict) -> dict:
    """`_start_attempt` without touching the store (no project id)."""
    web._start_attempt(state, mode="continue")
    return state
