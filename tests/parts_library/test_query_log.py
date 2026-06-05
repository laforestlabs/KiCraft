"""Tests for kicraft.parts_library.query_log: best-effort JSONL telemetry."""
from __future__ import annotations

from pathlib import Path

from kicraft.parts_library import query_log


def _set_log(monkeypatch, path: Path) -> None:
    monkeypatch.setenv(query_log.ENV_PATH, str(path))
    monkeypatch.delenv(query_log.ENV_CALLER, raising=False)
    monkeypatch.delenv(query_log.ENV_RUN_ID, raising=False)


def test_record_appends_and_reads(monkeypatch, tmp_path):
    log = tmp_path / "q.jsonl"
    _set_log(monkeypatch, log)
    query_log.record("lookup_lcsc_id", outcome="hit", query="AMS1117-3.3",
                     lcsc="C6186", library_name="ams1117-3v3")
    query_log.record("search_footprints", outcome="miss", query="weird", n_matches=0)
    events = list(query_log.read_events(log))
    assert len(events) == 2
    e0 = events[0]
    assert e0["tool"] == "lookup_lcsc_id" and e0["outcome"] == "hit"
    assert e0["lcsc"] == "C6186" and e0["library_name"] == "ams1117-3v3"
    assert e0["ts"].endswith("Z")
    assert e0["project"]  # cwd basename is attached


def test_none_valued_extras_dropped(monkeypatch, tmp_path):
    log = tmp_path / "q.jsonl"
    _set_log(monkeypatch, log)
    query_log.record("lookup_symbol", outcome="hit", query="Device:R", lib=None)
    e = list(query_log.read_events(log))[0]
    assert "lib" not in e  # None extras are not written


def test_read_default_path_uses_env(monkeypatch, tmp_path):
    log = tmp_path / "default.jsonl"
    _set_log(monkeypatch, log)
    assert list(query_log.read_events()) == []      # absent -> empty
    query_log.record("list_parts", outcome="listed", n_active=3)
    assert len(list(query_log.read_events())) == 1   # read_events() honors $KICRAFT_QUERY_LOG


def test_caller_and_run_id_from_env(monkeypatch, tmp_path):
    log = tmp_path / "q.jsonl"
    monkeypatch.setenv(query_log.ENV_PATH, str(log))
    monkeypatch.setenv(query_log.ENV_CALLER, "web")
    monkeypatch.setenv(query_log.ENV_RUN_ID, "p9-123")
    query_log.record("add_part_from_lcsc", outcome="fetched", lcsc="C1")
    e = list(query_log.read_events(log))[0]
    assert e["caller"] == "web" and e["run_id"] == "p9-123"


def test_record_never_raises(monkeypatch, tmp_path):
    # Parent path is a regular file, so mkdir(parents=True) fails -> must swallow.
    blocker = tmp_path / "blocker"
    blocker.write_text("not a dir")
    monkeypatch.setenv(query_log.ENV_PATH, str(blocker / "sub" / "q.jsonl"))
    query_log.record("list_parts", outcome="listed")  # no exception
    assert not (blocker / "sub").exists()


def test_read_skips_malformed(tmp_path):
    log = tmp_path / "q.jsonl"
    log.write_text(
        '{"tool":"a","outcome":"hit"}\n'
        "NOT JSON\n"
        "\n"
        '["not","a","dict"]\n'
        '{"tool":"b"}\n'
    )
    tools = [e.get("tool") for e in query_log.read_events(log)]
    assert tools == ["a", "b"]
