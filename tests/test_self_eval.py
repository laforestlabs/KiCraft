"""Offline tests for the batch self-eval harness (kicraft.eval.self_eval).

The intricate orchestration — park/auto-answer resume loop, the events.jsonl
whitelist, per-brief error isolation, and report compilation — is exercised here
with the real driver/eval calls monkeypatched out, so nothing hits OpenRouter,
pcbnew, or a build. (The real park->answer->resume path against the deterministic
stage CLIs is covered by tests/test_session.py.)
"""
from __future__ import annotations

import json
from pathlib import Path

from kicraft.eval import self_eval as se


# --------------------------------------------------------------------------- #
# auto-answer + event writer
# --------------------------------------------------------------------------- #
def test_auto_answers_picks_first_suggested_option():
    a = se._auto_answers([
        {"text": "Battery?", "options": ["LiPo 1S", "AA"]},
        {"text": "Color?", "options": []},
    ])
    assert a[0] == {"text": "Battery?", "answer": "LiPo 1S"}     # suggested option
    assert a[1]["text"] == "Color?"
    assert "default" in a[1]["answer"].lower()                   # fallback when no options
    assert se._auto_answers(None) == []


def test_event_writer_keeps_only_design_and_build_kinds(tmp_path):
    p = tmp_path / "events.jsonl"
    prog = se._event_writer(p)
    for ev in [
        {"kind": "stage_start", "stage": "intent"},
        {"kind": "reasoning_delta", "text": "thinking"},   # client stream -> dropped
        {"kind": "answer_delta", "text": "{"},             # dropped
        {"kind": "tool", "name": "list_parts"},            # dropped
        {"kind": "tool_result", "output": "..."},          # dropped
        {"kind": "question", "stage": "intent"},
        {"kind": "retry", "stage": "bom"},
        {"kind": "stage_done", "stage": "bom", "ok": True},
        {"kind": "build_start"},
        {"kind": "build_log", "text": "ok"},
        {"kind": "build_done", "ok": True},
        "not-a-dict",                                      # ignored, no crash
    ]:
        prog(ev)
    kinds = [json.loads(line)["kind"] for line in p.read_text().splitlines()]
    assert kinds == ["stage_start", "question", "retry", "stage_done",
                     "build_start", "build_log", "build_done"]


# --------------------------------------------------------------------------- #
# run_design: park -> auto-answer -> resume -> complete
# --------------------------------------------------------------------------- #
_FULL_STATE = {"intent": {}, "functional_spec": {}, "architecture": {},
               "bom": {"parts": [{"ref": "R1"}], "connections": [{"net_name": "VBUS"}]}}


def test_run_design_parks_then_resumes_with_suggested_option(tmp_path, monkeypatch):
    rundir = tmp_path / "run"
    (rundir / ".kicraft").mkdir(parents=True)
    seen_answers, calls = [], {"n": 0}

    def fake_run_session(ws, brief, stages, answers=None, client=None, progress=None, run_id=None):
        calls["n"] += 1
        seen_answers.append(answers)
        if progress:
            progress({"kind": "stage_start", "stage": stages[0]})
        if calls["n"] == 1:                                  # first pass: park on a question
            if progress:
                progress({"kind": "question", "stage": "intent"})
            return {"status": "awaiting_input", "last_stage": "intent",
                    "questions": [{"text": "Battery?", "options": ["LiPo 1S"], "blocking": True}],
                    "results": [{"cost_usd": 0.01, "needs_input": True}]}
        Path(ws, ".kicraft", "state.json").write_text(json.dumps(_FULL_STATE))  # resume completes
        if progress:
            progress({"kind": "stage_done", "stage": "wiring", "ok": True})
        return {"status": "ok", "last_stage": "wiring",
                "results": [{"cost_usd": 0.02, "commit_ok": True}], "questions": None}

    monkeypatch.setattr(se, "run_session", fake_run_session)
    events = rundir / "events.jsonl"
    d = se.run_design(object(), "a USB LED", rundir, se._event_writer(events))

    assert d["status"] == "ok" and d["questions"] == 1 and d["rounds"] == 2
    assert round(d["cost_usd"], 4) == 0.03                  # park attempt + resume both billed
    assert seen_answers[0] is None                          # opening pass asks nothing
    assert seen_answers[1] == [{"text": "Battery?", "answer": "LiPo 1S"}]
    kinds = [json.loads(line)["kind"] for line in events.read_text().splitlines()]
    assert "question" in kinds and "stage_done" in kinds


def test_run_design_failed_stage_stops_and_reports_error(tmp_path, monkeypatch):
    (tmp_path / ".kicraft").mkdir(parents=True)

    def fake_run_session(ws, brief, stages, **kw):
        return {"status": "failed", "last_stage": "bom",
                "results": [{"cost_usd": 0.04, "error": "stage-commit rejected"}], "questions": None}

    monkeypatch.setattr(se, "run_session", fake_run_session)
    d = se.run_design(object(), "x", tmp_path, lambda ev: None)
    assert d["status"] == "failed" and "rejected" in d["error"] and d["cost_usd"] == 0.04


# --------------------------------------------------------------------------- #
# evaluate_one: drive + build + score, error isolation
# --------------------------------------------------------------------------- #
def _fake_report(grade="A", final=92.0, verdict="SHIP", gates=(), judge_cost=0.004):
    return {
        "score": {"grade": grade, "final": final, "weighted": final, "verdict": verdict, "note": ""},
        "judge": {"ran": True, "ok": True, "cost_usd": judge_cost},
        "gates": {"triggered": [{"id": g, "cap": 45} for g in gates]},
        "dimensions": {"pipeline_completion": {"level": 4}, "electrical_soundness": {"level": 4}},
    }


def test_evaluate_one_happy_path_drives_builds_and_scores(tmp_path, monkeypatch):
    def fake_run_session(ws, brief, stages, **kw):
        Path(ws, ".kicraft", "state.json").write_text(json.dumps(_FULL_STATE))
        return {"status": "ok", "results": [{"cost_usd": 0.05}], "questions": None, "last_stage": "wiring"}

    built = {}

    def fake_run_build(rundir, progress, timeout_s=1200):
        progress({"kind": "build_start"})
        progress({"kind": "build_done", "ok": True, "rc": 0})
        built["dir"] = str(rundir)
        return 0

    seen_kw: dict = {}

    def fake_eval(rd, client, **kw):
        seen_kw.update(kw)
        return _fake_report()

    monkeypatch.setattr(se, "run_session", fake_run_session)
    monkeypatch.setattr(se, "run_build", fake_run_build)
    monkeypatch.setattr(se, "evaluate_project", fake_eval)

    rec = se.evaluate_one(object(), 1, "An ESP32-S3 plant monitor", tmp_path,
                          judge_model="judge-x", skip_judge=False)

    assert rec["design_status"] == "ok"
    assert rec["build_rc"] == 0 and rec["build_label"] == "fab-ready"
    assert rec["grade"] == "A" and rec["final"] == 92.0 and rec["verdict"] == "SHIP"
    assert rec["design_cost_usd"] == 0.05 and rec["judge_cost_usd"] == 0.004
    assert rec["gates"] == [] and "error" not in rec
    assert (tmp_path / rec["stem"] / "brief.txt").read_text().startswith("An ESP32-S3 plant")
    assert built["dir"].endswith(rec["stem"])
    # the real wall-clock window is passed so latency always scores (grade finalizes)
    assert seen_kw.get("started_at") and seen_kw.get("finished_at")
    assert seen_kw.get("judge_model") == "judge-x" and seen_kw.get("skip_judge") is False


def test_evaluate_one_skips_build_when_design_incomplete(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "run_session",
                        lambda ws, brief, stages, **kw: {"status": "failed", "last_stage": "bom",
                                                         "results": [{"cost_usd": 0.01, "error": "x"}],
                                                         "questions": None})

    def must_not_build(*a, **k):
        raise AssertionError("build must not run when the design did not complete")

    monkeypatch.setattr(se, "run_build", must_not_build)
    monkeypatch.setattr(se, "evaluate_project",
                        lambda rd, client, **kw:
                        _fake_report(grade="F", final=10.0, verdict="BROKEN", gates=("synthesis_broken",)))

    rec = se.evaluate_one(object(), 2, "a broken brief", tmp_path, judge_model=None, skip_judge=False)
    assert rec["design_status"] == "failed"
    assert rec["build_rc"] is None and rec["build_label"] is None
    assert rec["grade"] == "F" and rec["gates"] == ["synthesis_broken"]


def test_evaluate_one_isolates_exceptions(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("spend ceiling exceeded")

    monkeypatch.setattr(se, "run_session", boom)
    rec = se.evaluate_one(object(), 3, "a brief", tmp_path, judge_model=None, skip_judge=True)
    assert "spend ceiling exceeded" in rec["error"]
    assert "duration_s" in rec and rec["index"] == 3


# --------------------------------------------------------------------------- #
# selection + report compilation
# --------------------------------------------------------------------------- #
def test_select_limit_and_only():
    ps = [f"p{i}" for i in range(1, 10)]
    assert [i for i, _ in se._select(ps, 3, None)] == [1, 2, 3]
    assert [i for i, _ in se._select(ps, None, "1,5,9")] == [1, 5, 9]
    assert len(se._select(ps, None, None)) == 9


def test_compile_report_aggregates_and_writes(tmp_path):
    records = [
        {"index": 1, "prompt": "a", "stem": "run_01_A", "rundir": "/r/1", "grade": "A",
         "final": 92.0, "verdict": "SHIP", "build_rc": 0, "build_label": "fab-ready",
         "questions": 0, "gates": [], "design_cost_usd": 0.05, "judge_cost_usd": 0.004},
        {"index": 2, "prompt": "b", "stem": "run_02_B", "rundir": "/r/2", "grade": "C",
         "final": 55.0, "verdict": "REWORK", "build_rc": 5, "build_label": "ERC errors",
         "questions": 1, "gates": ["erc_errors"], "design_cost_usd": 0.06, "judge_cost_usd": 0.004},
        {"index": 3, "prompt": "c", "stem": "run_03_C", "rundir": "/r/3",
         "error": "RuntimeError: boom", "design_cost_usd": 0.0},
    ]
    meta = {"started_at": "2026-06-08T00:00:00+00:00", "out_dir": str(tmp_path),
            "design_model": "m", "judge": True, "judge_model": "j"}
    s = se.compile_report(records, tmp_path, meta)

    assert s["n"] == 3 and s["graded_n"] == 2 and s["n_errored"] == 1 and s["fab_ready"] == 1
    assert s["mean_final"] == 73.5 and s["median_final"] == 73.5
    assert s["grade_counts"] == {"A": 1, "C": 1, "ERROR": 1}
    assert s["gate_counts"] == {"erc_errors": 1}
    assert round(s["total_cost_usd"], 4) == 0.118
    assert (tmp_path / "summary.json").is_file()
    md = (tmp_path / "summary.md").read_text()
    assert "Needs attention" in md and "erc_errors" in md and "RuntimeError" in md
