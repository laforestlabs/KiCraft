"""The reopen path loads events.jsonl back into state['events'] (web._load_events).

events.jsonl is persisted at finalize but was historically never read back, so a
reopened project showed a blank build timeline / reasoning panel. Pure-data helper:
no NiceGUI, no pcbnew.
"""
from __future__ import annotations

from kicraft.server import web


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
