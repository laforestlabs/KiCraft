"""Offline tests for the batch self-eval harness (kicraft.eval.self_eval).

The intricate orchestration — park/auto-answer resume loop, the events.jsonl
whitelist, per-brief error isolation, and report compilation — is exercised here
with the real driver/eval calls monkeypatched out, so nothing hits OpenRouter,
pcbnew, or a build. (The real park->answer->resume path against the deterministic
stage CLIs is covered by tests/test_session.py.)
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

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

    entry = {"slug": "esp32-plant", "archetype": "rf_antenna",
             "brief": "An ESP32-S3 plant monitor"}
    rec = se.evaluate_one(object(), 1, entry, tmp_path,
                          judge_model="judge-x", skip_judge=False)

    assert rec["design_status"] == "ok"
    assert rec["build_rc"] == 0 and rec["build_label"] == "fab-ready"
    assert rec["grade"] == "A" and rec["final"] == 92.0 and rec["verdict"] == "SHIP"
    assert rec["design_cost_usd"] == 0.05 and rec["judge_cost_usd"] == 0.004
    assert rec["gates"] == [] and "error" not in rec
    assert rec["slug"] == "esp32-plant" and rec["archetype"] == "rf_antenna"
    assert rec["stem"] == "run_01_esp32-plant"
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

    rec = se.evaluate_one(object(), 2,
                          {"slug": "broken", "archetype": "single_passive", "brief": "a broken brief"},
                          tmp_path, judge_model=None, skip_judge=False)
    assert rec["design_status"] == "failed"
    assert rec["build_rc"] is None and rec["build_label"] is None
    assert rec["grade"] == "F" and rec["gates"] == ["synthesis_broken"]


def test_evaluate_one_isolates_exceptions(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("spend ceiling exceeded")

    monkeypatch.setattr(se, "run_session", boom)
    rec = se.evaluate_one(object(), 3,
                          {"slug": "abrief", "archetype": "single_passive", "brief": "a brief"},
                          tmp_path, judge_model=None, skip_judge=True)
    assert "spend ceiling exceeded" in rec["error"]
    assert "duration_s" in rec and rec["index"] == 3 and rec["slug"] == "abrief"


# --------------------------------------------------------------------------- #
# parallel execution + build gate + resume
# --------------------------------------------------------------------------- #
def _patch_llm_env(monkeypatch):
    """main() imports Settings/CappedOpenRouterClient lazily; stub both so no env
    or network is touched."""
    from kicraft.server import client as client_mod
    from kicraft.server import config as config_mod
    monkeypatch.setattr(
        config_mod.Settings, "from_env",
        classmethod(lambda cls: SimpleNamespace(model="test-model", eval_judge_model=None)))
    monkeypatch.setattr(client_mod, "CappedOpenRouterClient", lambda s: object())


def _fake_rec(idx, entry, out_dir, grade="A", **extra):
    stem = se._stem_for(idx, entry)
    return {"index": idx, "slug": entry["slug"], "archetype": entry["archetype"],
            "prompt": entry["brief"], "stem": stem,
            "rundir": str(Path(out_dir) / stem), "grade": grade,
            "final": 90.0, "verdict": "SHIP", "build_rc": 0, "build_label": "fab-ready",
            "questions": 0, "gates": [], "design_cost_usd": 0.01, "judge_cost_usd": 0.0,
            "duration_s": 0.1, **extra}


def test_evaluate_one_build_gate_caps_concurrent_builds(tmp_path, monkeypatch):
    def fake_run_session(ws, brief, stages, **kw):
        Path(ws, ".kicraft", "state.json").write_text(json.dumps(_FULL_STATE))
        return {"status": "ok", "results": [{"cost_usd": 0.0}], "questions": None,
                "last_stage": "wiring"}

    state, lock = {"now": 0, "max": 0}, threading.Lock()

    def fake_run_build(rundir, progress, timeout_s=1200):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.1)
        with lock:
            state["now"] -= 1
        return 0

    monkeypatch.setattr(se, "run_session", fake_run_session)
    monkeypatch.setattr(se, "run_build", fake_run_build)
    monkeypatch.setattr(se, "evaluate_project", lambda rd, client, **kw: _fake_report())

    gate = threading.BoundedSemaphore(1)
    with ThreadPoolExecutor(max_workers=3) as ex:
        futs = [ex.submit(se.evaluate_one, object(), i,
                          {"slug": f"b{i}", "archetype": "x", "brief": f"brief number {i}"},
                          tmp_path, judge_model=None, skip_judge=True, build_gate=gate)
                for i in (1, 2, 3)]
        recs = [f.result() for f in futs]

    assert all(r["build_rc"] == 0 for r in recs)
    assert state["max"] == 1          # the gate never let two builds overlap


def test_run_build_timeout_restarts_at_slot_acquired_marker(tmp_path, monkeypatch):
    # kicraft.build_slots contract: time queued for a host-wide build slot is not
    # build time. The child "queues" 0.8s, emits ACQUIRED_MARKER, then "routes"
    # 1.0s — total wall exceeds the 1.4s timeout, so it survives only if the
    # watchdog clock restarts at the marker.
    from kicraft.build_slots import ACQUIRED_MARKER
    child = ("import time; time.sleep(0.8); "
             f"print({ACQUIRED_MARKER!r}, flush=True); time.sleep(1.0)")
    monkeypatch.setattr(se, "_BUILD_CMD", [sys.executable, "-c", child])
    events = []
    assert se.run_build(tmp_path, events.append, timeout_s=1.4) == 0
    assert any(ACQUIRED_MARKER in (e.get("text") or "") for e in events)

    # negative control: with no marker the watchdog still kills a stuck build
    monkeypatch.setattr(se, "_BUILD_CMD",
                        [sys.executable, "-c", "import time; time.sleep(30)"])
    assert se.run_build(tmp_path, events.append, timeout_s=0.4) < 0


def test_main_parallel_overlaps_briefs_and_orders_records(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "BRIEFS", [
        {"slug": "alpha", "archetype": "x", "brief": "alpha brief"},
        {"slug": "beta", "archetype": "x", "brief": "beta brief"},
        {"slug": "gamma", "archetype": "y", "brief": "gamma brief"}])
    _patch_llm_env(monkeypatch)
    state, lock = {"now": 0, "max": 0}, threading.Lock()

    def fake_evaluate_one(client, idx, entry, out_dir, **kw):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.2 if idx == 1 else 0.05)   # brief 1 finishes LAST
        with lock:
            state["now"] -= 1
        return _fake_rec(idx, entry, out_dir)

    monkeypatch.setattr(se, "evaluate_one", fake_evaluate_one)
    assert se.main(["--parallel", "3", "--no-judge", "--out", str(tmp_path)]) == 0

    summ = json.loads((tmp_path / "summary.json").read_text())
    assert [r["index"] for r in summ["runs"]] == [1, 2, 3]   # index order, not finish order
    assert state["max"] >= 2                                 # briefs genuinely overlapped
    assert summ["parallel"] == 3 and summ["build_slots"] == 2
    assert isinstance(summ["wall_s"], (int, float))


def test_main_sequential_checkpoints_summary_after_each_brief(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "BRIEFS", [
        {"slug": "alpha", "archetype": "x", "brief": "alpha brief"},
        {"slug": "beta", "archetype": "x", "brief": "beta brief"}])
    _patch_llm_env(monkeypatch)
    seen_runs_at_call = []

    def fake_evaluate_one(client, idx, entry, out_dir, **kw):
        p = Path(out_dir) / "summary.json"
        prior = json.loads(p.read_text())["runs"] if p.exists() else []
        seen_runs_at_call.append(len(prior))
        return _fake_rec(idx, entry, out_dir)

    monkeypatch.setattr(se, "evaluate_one", fake_evaluate_one)
    assert se.main(["--parallel", "1", "--no-judge", "--out", str(tmp_path)]) == 0
    assert seen_runs_at_call == [0, 1]        # brief 2 saw brief 1 already checkpointed


def test_main_defaults_to_parallel(tmp_path, monkeypatch):
    # every entry point (CLI, /self-eval, admin GUI) relies on the harness itself
    # defaulting to the parallel sweet spot — no flags required
    monkeypatch.setattr(se, "BRIEFS", [
        {"slug": "alpha", "archetype": "x", "brief": "alpha brief"},
        {"slug": "beta", "archetype": "x", "brief": "beta brief"},
        {"slug": "gamma", "archetype": "y", "brief": "gamma brief"}])
    _patch_llm_env(monkeypatch)
    monkeypatch.setattr(se, "evaluate_one",
                        lambda client, idx, entry, out_dir, **kw: _fake_rec(idx, entry, out_dir))
    assert se.main(["--no-judge", "--out", str(tmp_path)]) == 0
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert summ["parallel"] == 3 and summ["build_slots"] == 2
    assert [r["index"] for r in summ["runs"]] == [1, 2, 3]


def test_main_resolves_relative_out_dir(tmp_path, monkeypatch):
    # design stages run subprocesses with cwd=workspace, so a relative rundir would
    # nest the .kicraft tree inside itself; main() must hand out an absolute out_dir
    monkeypatch.setattr(se, "BRIEFS", [{"slug": "alpha", "archetype": "x", "brief": "alpha brief"}])
    _patch_llm_env(monkeypatch)
    monkeypatch.chdir(tmp_path)
    seen = {}

    def fake_evaluate_one(client, idx, entry, out_dir, **kw):
        seen["out_dir"] = Path(out_dir)
        return _fake_rec(idx, entry, out_dir)

    monkeypatch.setattr(se, "evaluate_one", fake_evaluate_one)
    assert se.main(["--no-judge", "--out", "rel/batch"]) == 0
    assert seen["out_dir"].is_absolute()
    assert seen["out_dir"] == (tmp_path / "rel" / "batch").resolve()


def test_main_resume_reuses_completed_and_reruns_failed(tmp_path, monkeypatch):
    entries = [{"slug": "alpha", "archetype": "x", "brief": "alpha brief"},
               {"slug": "beta", "archetype": "x", "brief": "beta brief"}]
    monkeypatch.setattr(se, "BRIEFS", entries)
    _patch_llm_env(monkeypatch)

    # Prior batch: brief 1 completed (its eval report exists), brief 2 errored and
    # left a stale workspace behind.
    good = _fake_rec(1, entries[0], tmp_path)
    report = Path(good["rundir"]) / "eval" / "report.json"
    report.parent.mkdir(parents=True)
    report.write_text("{}")
    good["report_path"] = str(report)
    stale = tmp_path / se._stem_for(2, entries[1]) / ".kicraft"
    stale.mkdir(parents=True)
    (stale / "state.json").write_text("{}")
    bad = {"index": 2, "slug": entries[1]["slug"], "archetype": entries[1]["archetype"],
           "prompt": entries[1]["brief"], "stem": se._stem_for(2, entries[1]),
           "rundir": str(stale.parent), "error": "RuntimeError: boom", "design_cost_usd": 0.0}
    se.compile_report([good, bad], tmp_path,
                      {"started_at": "x", "out_dir": str(tmp_path),
                       "design_model": "m", "judge": False})

    ran = []

    def fake_evaluate_one(client, idx, entry, out_dir, **kw):
        ran.append(idx)
        # the failed brief's stale workspace must have been wiped before the re-run
        assert not (Path(out_dir) / se._stem_for(idx, entry)).exists()
        return _fake_rec(idx, entry, out_dir, grade="B")

    monkeypatch.setattr(se, "evaluate_one", fake_evaluate_one)
    assert se.main(["--resume", str(tmp_path), "--no-judge"]) == 0

    assert ran == [2]                          # only the errored brief re-ran
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert [r["index"] for r in summ["runs"]] == [1, 2]
    assert summ["runs"][0]["grade"] == "A"     # reused untouched
    assert summ["runs"][1]["grade"] == "B"     # error replaced by the re-run
    assert summ["resumed_reused_n"] == 1 and summ["n_errored"] == 0


# --------------------------------------------------------------------------- #
# selection + report compilation
# --------------------------------------------------------------------------- #
def test_select_limit_and_only():
    es = [{"slug": f"s{i}", "archetype": "a", "brief": f"p{i}"} for i in range(1, 10)]
    assert [i for i, _ in se._select(es, 3, None)] == [1, 2, 3]
    assert [i for i, _ in se._select(es, None, "s1,s5,s9")] == [1, 5, 9]   # by slug
    assert [i for i, _ in se._select(es, None, "1,5,9")] == [1, 5, 9]      # numeric fallback
    assert [e["slug"] for _, e in se._select(es, None, "s2")] == ["s2"]
    assert len(se._select(es, None, None)) == 9


def test_compile_report_aggregates_and_writes(tmp_path):
    records = [
        {"index": 1, "slug": "aa", "archetype": "usb_c_connector", "prompt": "a",
         "stem": "run_01_aa", "rundir": "/r/1", "grade": "A",
         "final": 92.0, "verdict": "SHIP", "build_rc": 0, "build_label": "fab-ready",
         "questions": 0, "gates": [], "design_cost_usd": 0.05, "judge_cost_usd": 0.004},
        {"index": 2, "slug": "bb", "archetype": "fine_pitch", "prompt": "b",
         "stem": "run_02_bb", "rundir": "/r/2", "grade": "C",
         "final": 55.0, "verdict": "REWORK", "build_rc": 5, "build_label": "ERC errors",
         "questions": 1, "gates": ["erc_errors"], "design_cost_usd": 0.06, "judge_cost_usd": 0.004},
        {"index": 3, "slug": "cc", "archetype": "fine_pitch", "prompt": "c",
         "stem": "run_03_cc", "rundir": "/r/3",
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
    # per-archetype rollup localizes regressions to a stress dimension
    arche = s["archetype_stats"]
    assert arche["usb_c_connector"] == {
        "n": 1, "graded_n": 1, "fab_ready": 1, "grade_counts": {"A": 1}, "mean_final": 92.0}
    assert arche["fine_pitch"]["n"] == 2 and arche["fine_pitch"]["graded_n"] == 1
    assert arche["fine_pitch"]["mean_final"] == 55.0 and arche["fine_pitch"]["fab_ready"] == 0
    assert arche["fine_pitch"]["grade_counts"] == {"C": 1, "ERROR": 1}
    assert (tmp_path / "summary.json").is_file()
    md = (tmp_path / "summary.md").read_text()
    assert "Needs attention" in md and "erc_errors" in md and "RuntimeError" in md
    assert "By archetype" in md and "usb_c_connector" in md
