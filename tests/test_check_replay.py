"""The replay harness: which candidates it finds, what it recomputes, how it aggregates."""

from __future__ import annotations

import json

from kicraft.eval import check_replay
from kicraft.eval.check_replay import aggregate, iter_run_dirs, render_table, replay_run


def _write_run(root, name, *, stage="functional_spec", candidate=None, ok=True, attempts=3):
    run_dir = root / "campaign_x" / name
    (run_dir / ".kicraft").mkdir(parents=True)
    (run_dir / "brief.txt").write_text(
        "A 5 V controller board with a status indicator.", encoding="utf-8"
    )
    (run_dir / ".kicraft" / "state.json").write_text(
        json.dumps(
            {
                "intent": {"goal": "g"},
                stage: candidate
                or {
                    "blocks": [
                        {
                            "name": "POWER_DISTRIBUTION",
                            "category": "power",
                            "purpose": "Distributes the 5 V rail.",
                        }
                    ],
                    "connections": [],
                    "assumptions": [],
                },
                "stage_status": {
                    stage: {
                        "ok": ok,
                        "attempts": attempts,
                        "repair_attempted": True,
                        "rounds": None,
                        "failure_kind": None,
                    },
                    "wiring": {"ok": True, "attempts": 1, "repair_attempted": False, "rounds": None},
                },
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_iter_run_dirs_finds_state_files(tmp_path):
    _write_run(tmp_path, "run_01")
    _write_run(tmp_path, "run_02")
    (tmp_path / "not_a_run").mkdir()
    assert [path.name for path in iter_run_dirs(tmp_path)] == ["run_01", "run_02"]


def test_replay_recomputes_hits_and_keeps_the_archived_counters(tmp_path):
    run_dir = _write_run(tmp_path, "run_01")
    replay = replay_run(run_dir)

    assert replay.committed is False  # wiring ok but the other stages are absent
    codes = {hit.code for hit in replay.hits}
    assert "functional_spec_nonfunctional_block" in codes
    outcome = replay.outcome("functional_spec")
    assert outcome is not None and outcome.attempts == 3 and outcome.ok is True
    assert replay.replayed_stages == ("intent", "functional_spec")


def test_normalization_runs_once_and_never_researches(tmp_path, monkeypatch):
    """The runtime normalizes before diagnosing, and the harness must not touch the library."""
    from kicraft.design import part_research

    calls: list[int] = []

    def boom(classes, limit=2):
        calls.append(1)
        raise AssertionError("a replay must never research a part")

    monkeypatch.setattr(part_research, "research_uncovered_classes", boom)
    run_dir = _write_run(tmp_path, "run_01")
    replay_run(run_dir)  # must not raise
    assert calls == []


def test_aggregate_separates_firings_on_committed_boards(tmp_path):
    firing = _write_run(tmp_path, "run_01")  # the block makes the check fire
    clean = _write_run(tmp_path, "run_02", candidate={"blocks": [], "connections": [], "assumptions": []})
    report = aggregate([replay_run(firing), replay_run(clean)])

    assert report["runs_replayed"] == 2
    row = report["codes"]["functional_spec_nonfunctional_block"]
    assert row["fires"] == 1
    assert row["fires_on_committed"] == 1
    assert row["stages_firing"] == {"functional_spec": 1}
    assert report["kpi"]["functional_spec"]["mean_attempts"] == 3.0
    assert "functional_spec_nonfunctional_block" in render_table(report)


def test_replay_document_ignores_absent_stage_slots(tmp_path):
    run_dir = tmp_path / "campaign_y" / "run_09"
    (run_dir / ".kicraft").mkdir(parents=True)
    (run_dir / ".kicraft" / "state.json").write_text(json.dumps({"bom": None}), encoding="utf-8")
    replay = replay_run(run_dir)
    assert replay.hits == ()
    assert replay.replayed_stages == ()
    assert replay.error is None


def test_a_broken_state_file_is_reported_not_raised(tmp_path):
    run_dir = tmp_path / "campaign_z" / "run_10"
    (run_dir / ".kicraft").mkdir(parents=True)
    (run_dir / ".kicraft" / "state.json").write_text("{not json", encoding="utf-8")
    replay = replay_run(run_dir)
    assert replay.error and "JSONDecodeError" in replay.error
    assert check_replay.aggregate([replay])["runs_replayed"] == 0
