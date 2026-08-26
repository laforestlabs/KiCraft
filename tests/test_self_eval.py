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

from kicraft.eval import self_eval as se


# --------------------------------------------------------------------------- #
# WS6: BOM-reconcile re-drive shared by the web app and the eval driver
# --------------------------------------------------------------------------- #
_DEFICIT_PARK = {
    "status": "awaiting_input",
    "last_stage": "wiring",
    # A NON-passive ask (crystal): the deterministic passive-add pass cannot
    # provision it, so these orchestration tests exercise the LLM re-drive
    # path (and its stuck-loop detection) rather than the deterministic
    # wiring-only fast path.
    "questions": [
        {"text": "add a 40MHz crystal (X2) for U1", "reconcile_target": "bom", "blocking": True}
    ],
}


def _ws_with_bom(tmp_path, refs):
    """A workspace whose committed state.json carries a BOM with *refs*."""
    p = tmp_path / ".kicraft"
    p.mkdir(parents=True, exist_ok=True)
    (p / "state.json").write_text(
        json.dumps({"bom": {"parts": [{"ref": r} for r in refs]}}), encoding="utf-8"
    )
    return tmp_path


def test_maybe_bom_reconcile_redrives_on_deficit(monkeypatch, tmp_path):
    from kicraft.server import session

    ws = _ws_with_bom(tmp_path, ["U1", "C1"])
    calls = []

    def fake_run_session(w, brief, stages, **kw):
        calls.append((list(stages), kw.get("instruction")))
        _ws_with_bom(tmp_path, ["U1", "C1", "C2"])  # the pass adds a part
        return {
            "status": "ok",
            "results": [{"cost_usd": 0.01}],
            "questions": None,
            "last_stage": "wiring",
        }

    monkeypatch.setattr(session, "run_session", fake_run_session)
    res, passes = session.maybe_bom_reconcile(ws, "brief", dict(_DEFICIT_PARK))
    assert passes == 1
    assert res["status"] == "ok"
    assert calls and calls[0][0] == ["bom", "wiring"]
    assert "missing supporting parts" in calls[0][1]


def test_maybe_bom_reconcile_noop_without_reconcile_target(monkeypatch):
    from kicraft.server import session

    def boom(*a, **k):
        raise AssertionError("run_session must not be re-driven for a plain park")

    monkeypatch.setattr(session, "run_session", boom)
    park = {
        "status": "awaiting_input",
        "last_stage": "wiring",
        "questions": [{"text": "which LED color?", "blocking": True}],
    }
    res, passes = session.maybe_bom_reconcile("/ws", "brief", park)
    assert passes == 0
    assert res is park


def test_maybe_bom_reconcile_budget_cap_stops_redriving(monkeypatch):
    from kicraft.server import session

    def boom(*a, **k):
        raise AssertionError("budget exhausted: reconcile must not re-drive")

    monkeypatch.setattr(session, "run_session", boom)
    res, passes = session.maybe_bom_reconcile(
        "/ws", "brief", dict(_DEFICIT_PARK), reconcile_passes=session.BOM_RECONCILE_MAX_PASSES
    )
    assert passes == session.BOM_RECONCILE_MAX_PASSES
    assert res == _DEFICIT_PARK


def test_maybe_bom_reconcile_chain_counts_passes(monkeypatch, tmp_path):
    # N3: a deficit CHAIN (each pass genuinely adds parts and wiring then
    # surfaces the NEXT, different shortfall -- every real chain observed
    # names a new part each link) advances the counter one pass at a time up
    # to the budget -- the old single-shot guard made every chain >= 2
    # unwinnable by construction. A pass that changes the BOM while the SAME
    # deficit re-parks is no longer a chain link: that is run_22's
    # added-something-irrelevant pathology, covered by the pointed-retry test
    # in test_bom_reconcile_deterministic.py (2026-07-27 fix-plan P1).
    from kicraft.server import session

    ws = _ws_with_bom(tmp_path, ["U1"])
    n = [0]

    def fake_run_session(w, brief, stages, **kw):
        n[0] += 1
        _ws_with_bom(tmp_path, ["U1"] + [f"C{i}" for i in range(n[0])])
        park = dict(_DEFICIT_PARK)  # wiring parks again on the NEXT deficit
        park["questions"] = [
            {"text": f"add a 1uF cap for U{n[0] + 1}", "reconcile_target": "bom", "blocking": True}
        ]
        return park

    monkeypatch.setattr(session, "run_session", fake_run_session)
    passes = 0
    res = dict(_DEFICIT_PARK)
    for expected in (1, 2, 3):
        res, passes = session.maybe_bom_reconcile(ws, "brief", res, reconcile_passes=passes)
        assert passes == expected
    # Budget now exhausted: a 4th call must not re-drive.
    res2, passes = session.maybe_bom_reconcile(ws, "brief", res, reconcile_passes=passes)
    assert passes == session.BOM_RECONCILE_MAX_PASSES and res2 is res
    assert n[0] == 3


def test_maybe_bom_reconcile_nochange_pass_exhausts_budget(monkeypatch, tmp_path):
    # A pass that changes nothing in the committed BOM is a stuck loop, not a
    # chain: the budget is spent immediately (run_21-style single no-op stop).
    from kicraft.server import session

    ws = _ws_with_bom(tmp_path, ["U1", "C1"])

    def fake_run_session(w, brief, stages, **kw):
        return dict(_DEFICIT_PARK)  # parks again, BOM untouched

    monkeypatch.setattr(session, "run_session", fake_run_session)
    res, passes = session.maybe_bom_reconcile(ws, "brief", dict(_DEFICIT_PARK))
    assert passes == session.BOM_RECONCILE_MAX_PASSES


# --------------------------------------------------------------------------- #
# WS9: outline check must gate on build outcome (not grade the seed stub)
# --------------------------------------------------------------------------- #
def test_outline_check_none_for_rectangular_brief(tmp_path):
    assert se._outline_check({}, tmp_path, build_rc=0) is None


def test_outline_check_reports_no_built_parent_when_leaf_phase_died(tmp_path, monkeypatch):
    # rc=6 (route/infra abort) leaves only the rectangular seed stub; grading it
    # faked a "hexagon came out rectangular" failure. It must report a distinct
    # 'no built parent' with pass=None instead.
    called = {"eval": False}
    monkeypatch.setattr(
        se,
        "evaluate_outline_shape",
        lambda *a, **k: called.__setitem__("eval", True) or {"level": 0, "pass": False},
    )
    oc = se._outline_check({"outline_shape": "hexagon"}, tmp_path, build_rc=6)
    assert oc == {
        "pass": None,
        "level": None,
        "expected_shape": "hexagon",
        "reason": "no built parent (build rc=6)",
    }
    assert not called["eval"]  # never even tried to classify the stub


def test_outline_check_classifies_promoted_parent(tmp_path, monkeypatch):
    # rc=7 (routed parent present, DRC failed) DID stamp the shape -> classify it.
    board = tmp_path / "generated" / "PROJ"
    board.mkdir(parents=True)
    (board / "PROJ.kicad_pcb").write_text("(kicad_pcb)")
    monkeypatch.setattr(
        se,
        "evaluate_outline_shape",
        lambda b, exp: {"level": 4, "detected_family": exp, "expected_shape": exp},
    )
    oc = se._outline_check({"outline_shape": "hexagon"}, tmp_path, build_rc=7)
    assert oc["pass"] is True
    assert oc["level"] == 4


# --------------------------------------------------------------------------- #
# auto-answer + event writer
# --------------------------------------------------------------------------- #
def test_auto_answers_picks_first_suggested_option():
    a = se._auto_answers(
        [
            {"text": "Battery?", "options": ["LiPo 1S", "AA"]},
            {"text": "Color?", "options": []},
        ]
    )
    assert a[0] == {"text": "Battery?", "answer": "LiPo 1S"}  # suggested option
    assert a[1]["text"] == "Color?"
    assert "default" in a[1]["answer"].lower()  # fallback when no options
    assert se._auto_answers(None) == []


def test_event_writer_keeps_only_design_and_build_kinds(tmp_path):
    p = tmp_path / "events.jsonl"
    prog = se._event_writer(p)
    for ev in [
        {"kind": "stage_start", "stage": "intent"},
        {"kind": "reasoning_delta", "text": "thinking"},  # client stream -> dropped
        {"kind": "answer_delta", "text": "{"},  # dropped
        {"kind": "tool", "name": "list_parts"},  # dropped
        {"kind": "tool_result", "output": "..."},  # dropped
        {"kind": "question", "stage": "intent"},
        {"kind": "retry", "stage": "bom"},
        {"kind": "stage_done", "stage": "bom", "ok": True},
        {"kind": "build_start"},
        {"kind": "build_log", "text": "ok"},
        {"kind": "build_done", "ok": True},
        "not-a-dict",  # ignored, no crash
    ]:
        prog(ev)
    kinds = [json.loads(line)["kind"] for line in p.read_text().splitlines()]
    assert kinds == [
        "stage_start",
        "question",
        "retry",
        "stage_done",
        "build_start",
        "build_log",
        "build_done",
    ]


# --------------------------------------------------------------------------- #
# run_design: park -> auto-answer -> resume -> complete
# --------------------------------------------------------------------------- #
_FULL_STATE = {
    "intent": {},
    "functional_spec": {},
    "architecture": {},
    "bom": {"parts": [{"ref": "R1"}], "connections": [{"net_name": "VBUS"}]},
}


def test_run_design_parks_then_resumes_with_suggested_option(tmp_path, monkeypatch):
    rundir = tmp_path / "run"
    (rundir / ".kicraft").mkdir(parents=True)
    seen_answers, calls = [], {"n": 0}

    def fake_run_session(ws, brief, stages, answers=None, client=None, progress=None, run_id=None):
        calls["n"] += 1
        seen_answers.append(answers)
        if progress:
            progress({"kind": "stage_start", "stage": stages[0]})
        if calls["n"] == 1:  # first pass: park on a question
            if progress:
                progress({"kind": "question", "stage": "intent"})
            return {
                "status": "awaiting_input",
                "last_stage": "intent",
                "questions": [{"text": "Battery?", "options": ["LiPo 1S"], "blocking": True}],
                "results": [{"cost_usd": 0.01, "needs_input": True}],
            }
        Path(ws, ".kicraft", "state.json").write_text(json.dumps(_FULL_STATE))  # resume completes
        if progress:
            progress({"kind": "stage_done", "stage": "wiring", "ok": True})
        return {
            "status": "ok",
            "last_stage": "wiring",
            "results": [{"cost_usd": 0.02, "commit_ok": True}],
            "questions": None,
        }

    monkeypatch.setattr(se, "run_session", fake_run_session)
    events = rundir / "events.jsonl"
    d = se.run_design(object(), "a USB LED", rundir, se._event_writer(events))

    assert d["status"] == "ok" and d["questions"] == 1 and d["rounds"] == 2
    assert round(d["cost_usd"], 4) == 0.03  # park attempt + resume both billed
    assert seen_answers[0] is None  # opening pass asks nothing
    assert seen_answers[1] == [{"text": "Battery?", "answer": "LiPo 1S"}]
    kinds = [json.loads(line)["kind"] for line in events.read_text().splitlines()]
    assert "question" in kinds and "stage_done" in kinds


def test_run_design_failed_stage_stops_and_reports_error(tmp_path, monkeypatch):
    (tmp_path / ".kicraft").mkdir(parents=True)

    def fake_run_session(ws, brief, stages, **kw):
        return {
            "status": "failed",
            "last_stage": "bom",
            "results": [{"cost_usd": 0.04, "error": "stage-commit rejected"}],
            "questions": None,
        }

    monkeypatch.setattr(se, "run_session", fake_run_session)
    d = se.run_design(object(), "x", tmp_path, lambda ev: None)
    assert d["status"] == "failed" and "rejected" in d["error"] and d["cost_usd"] == 0.04


# --------------------------------------------------------------------------- #
# evaluate_one: drive + build + score, error isolation
# --------------------------------------------------------------------------- #
def _fake_report(grade="A", final=92.0, verdict="SHIP", gates=(), judge_cost=0.004):
    return {
        "score": {
            "grade": grade,
            "final": final,
            "weighted": final,
            "verdict": verdict,
            "note": "",
        },
        "judge": {"ran": True, "ok": True, "cost_usd": judge_cost},
        "gates": {"triggered": [{"id": g, "cap": 45} for g in gates]},
        "dimensions": {"pipeline_completion": {"level": 4}, "electrical_soundness": {"level": 4}},
    }


def test_evaluate_one_happy_path_drives_builds_and_scores(tmp_path, monkeypatch):
    def fake_run_session(ws, brief, stages, **kw):
        Path(ws, ".kicraft", "state.json").write_text(json.dumps(_FULL_STATE))
        return {
            "status": "ok",
            "results": [{"cost_usd": 0.05}],
            "questions": None,
            "last_stage": "wiring",
        }

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

    entry = {"slug": "esp32-plant", "archetype": "rf_antenna", "brief": "An ESP32-S3 plant monitor"}
    rec = se.evaluate_one(object(), 1, entry, tmp_path, judge_model="judge-x", skip_judge=False)

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
    monkeypatch.setattr(
        se,
        "run_session",
        lambda ws, brief, stages, **kw: {
            "status": "failed",
            "last_stage": "bom",
            "results": [{"cost_usd": 0.01, "error": "x"}],
            "questions": None,
        },
    )

    def must_not_build(*a, **k):
        raise AssertionError("build must not run when the design did not complete")

    monkeypatch.setattr(se, "run_build", must_not_build)
    monkeypatch.setattr(
        se,
        "evaluate_project",
        lambda rd, client, **kw: _fake_report(
            grade="F", final=10.0, verdict="BROKEN", gates=("synthesis_broken",)
        ),
    )

    rec = se.evaluate_one(
        object(),
        2,
        {"slug": "broken", "archetype": "single_passive", "brief": "a broken brief"},
        tmp_path,
        judge_model=None,
        skip_judge=False,
    )
    assert rec["design_status"] == "failed"
    assert rec["build_rc"] is None and rec["build_label"] is None
    assert rec["grade"] == "F" and rec["gates"] == ["synthesis_broken"]


def test_evaluate_one_isolates_exceptions(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("spend ceiling exceeded")

    monkeypatch.setattr(se, "run_session", boom)
    rec = se.evaluate_one(
        object(),
        3,
        {"slug": "abrief", "archetype": "single_passive", "brief": "a brief"},
        tmp_path,
        judge_model=None,
        skip_judge=True,
    )
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
        config_mod.Settings,
        "from_env",
        classmethod(lambda cls: cls(api_key="test", model="test-model")),
    )
    monkeypatch.setattr(client_mod, "CappedOpenRouterClient", lambda s: object())


def _fake_rec(idx, entry, out_dir, grade="A", **extra):
    stem = se._stem_for(idx, entry)
    return {
        "index": idx,
        "slug": entry["slug"],
        "archetype": entry["archetype"],
        "prompt": entry["brief"],
        "stem": stem,
        "rundir": str(Path(out_dir) / stem),
        "grade": grade,
        "final": 90.0,
        "verdict": "SHIP",
        "build_rc": 0,
        "build_label": "fab-ready",
        "questions": 0,
        "gates": [],
        "design_cost_usd": 0.01,
        "judge_cost_usd": 0.0,
        "duration_s": 0.1,
        **extra,
    }


def test_evaluate_one_build_gate_caps_concurrent_builds(tmp_path, monkeypatch):
    def fake_run_session(ws, brief, stages, **kw):
        Path(ws, ".kicraft", "state.json").write_text(json.dumps(_FULL_STATE))
        return {
            "status": "ok",
            "results": [{"cost_usd": 0.0}],
            "questions": None,
            "last_stage": "wiring",
        }

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
        futs = [
            ex.submit(
                se.evaluate_one,
                object(),
                i,
                {"slug": f"b{i}", "archetype": "x", "brief": f"brief number {i}"},
                tmp_path,
                judge_model=None,
                skip_judge=True,
                build_gate=gate,
            )
            for i in (1, 2, 3)
        ]
        recs = [f.result() for f in futs]

    assert all(r["build_rc"] == 0 for r in recs)
    assert state["max"] == 1  # the gate never let two builds overlap


def test_run_build_timeout_restarts_at_slot_acquired_marker(tmp_path, monkeypatch):
    # kicraft.build_slots contract: time queued for a host-wide build slot is not
    # build time. The child "queues" 0.8s, emits ACQUIRED_MARKER, then "routes"
    # 1.0s — total wall exceeds the 1.4s timeout, so it survives only if the
    # watchdog clock restarts at the marker.
    from kicraft.build_slots import ACQUIRED_MARKER

    child = f"import time; time.sleep(0.8); print({ACQUIRED_MARKER!r}, flush=True); time.sleep(1.0)"
    monkeypatch.setattr(se, "_BUILD_CMD", [sys.executable, "-c", child])
    events = []
    assert se.run_build(tmp_path, events.append, timeout_s=1.4) == 0
    assert any(ACQUIRED_MARKER in (e.get("text") or "") for e in events)

    # negative control: with no marker the watchdog still kills a stuck build
    monkeypatch.setattr(se, "_BUILD_CMD", [sys.executable, "-c", "import time; time.sleep(30)"])
    assert se.run_build(tmp_path, events.append, timeout_s=0.4) < 0


def test_main_parallel_overlaps_briefs_and_orders_records(tmp_path, monkeypatch):
    monkeypatch.setattr(
        se,
        "BRIEFS",
        [
            {"slug": "alpha", "archetype": "x", "brief": "alpha brief"},
            {"slug": "beta", "archetype": "x", "brief": "beta brief"},
            {"slug": "gamma", "archetype": "y", "brief": "gamma brief"},
        ],
    )
    _patch_llm_env(monkeypatch)
    state, lock = {"now": 0, "max": 0}, threading.Lock()

    def fake_evaluate_one(client, idx, entry, out_dir, **kw):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.2 if idx == 1 else 0.05)  # brief 1 finishes LAST
        with lock:
            state["now"] -= 1
        return _fake_rec(idx, entry, out_dir)

    monkeypatch.setattr(se, "evaluate_one", fake_evaluate_one)
    assert se.main(["--parallel", "3", "--no-judge", "--out", str(tmp_path)]) == 0

    summ = json.loads((tmp_path / "summary.json").read_text())
    assert [r["index"] for r in summ["runs"]] == [1, 2, 3]  # index order, not finish order
    assert state["max"] >= 2  # briefs genuinely overlapped
    # build_slots defaults host-aware (max(1, cores//6), capped at cores) so it
    # can never over-subscribe -- never a fixed 2 that thrashes a 2-core box.
    from kicraft.build_slots import host_cpu_count

    assert summ["parallel"] == 3 and 1 <= summ["build_slots"] <= host_cpu_count()
    assert isinstance(summ["wall_s"], (int, float))


def test_main_sequential_checkpoints_summary_after_each_brief(tmp_path, monkeypatch):
    monkeypatch.setattr(
        se,
        "BRIEFS",
        [
            {"slug": "alpha", "archetype": "x", "brief": "alpha brief"},
            {"slug": "beta", "archetype": "x", "brief": "beta brief"},
        ],
    )
    _patch_llm_env(monkeypatch)
    seen_runs_at_call = []

    def fake_evaluate_one(client, idx, entry, out_dir, **kw):
        p = Path(out_dir) / "summary.json"
        prior = json.loads(p.read_text())["runs"] if p.exists() else []
        seen_runs_at_call.append(len(prior))
        return _fake_rec(idx, entry, out_dir)

    monkeypatch.setattr(se, "evaluate_one", fake_evaluate_one)
    assert se.main(["--parallel", "1", "--no-judge", "--out", str(tmp_path)]) == 0
    assert seen_runs_at_call == [0, 1]  # brief 2 saw brief 1 already checkpointed


def test_main_defaults_to_parallel(tmp_path, monkeypatch):
    # every entry point (CLI, /self-eval, admin GUI) relies on the harness itself
    # defaulting to the parallel sweet spot — no flags required
    monkeypatch.setattr(
        se,
        "BRIEFS",
        [
            {"slug": "alpha", "archetype": "x", "brief": "alpha brief"},
            {"slug": "beta", "archetype": "x", "brief": "beta brief"},
            {"slug": "gamma", "archetype": "y", "brief": "gamma brief"},
        ],
    )
    _patch_llm_env(monkeypatch)
    monkeypatch.setattr(
        se, "evaluate_one", lambda client, idx, entry, out_dir, **kw: _fake_rec(idx, entry, out_dir)
    )
    assert se.main(["--no-judge", "--out", str(tmp_path)]) == 0
    summ = json.loads((tmp_path / "summary.json").read_text())
    from kicraft.build_slots import host_cpu_count

    assert summ["parallel"] == 3 and 1 <= summ["build_slots"] <= host_cpu_count()
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
    entries = [
        {"slug": "alpha", "archetype": "x", "brief": "alpha brief"},
        {"slug": "beta", "archetype": "x", "brief": "beta brief"},
    ]
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
    bad = {
        "index": 2,
        "slug": entries[1]["slug"],
        "archetype": entries[1]["archetype"],
        "prompt": entries[1]["brief"],
        "stem": se._stem_for(2, entries[1]),
        "rundir": str(stale.parent),
        "error": "RuntimeError: boom",
        "design_cost_usd": 0.0,
    }
    se.compile_report(
        [good, bad],
        tmp_path,
        {"started_at": "x", "out_dir": str(tmp_path), "design_model": "m", "judge": False},
    )

    ran = []

    def fake_evaluate_one(client, idx, entry, out_dir, **kw):
        ran.append(idx)
        # the failed brief's stale workspace must have been wiped before the re-run
        assert not (Path(out_dir) / se._stem_for(idx, entry)).exists()
        return _fake_rec(idx, entry, out_dir, grade="B")

    monkeypatch.setattr(se, "evaluate_one", fake_evaluate_one)
    assert se.main(["--resume", str(tmp_path), "--no-judge"]) == 0

    assert ran == [2]  # only the errored brief re-ran
    summ = json.loads((tmp_path / "summary.json").read_text())
    assert [r["index"] for r in summ["runs"]] == [1, 2]
    assert summ["runs"][0]["grade"] == "A"  # reused untouched
    assert summ["runs"][1]["grade"] == "B"  # error replaced by the re-run
    assert summ["resumed_reused_n"] == 1 and summ["n_errored"] == 0


# --------------------------------------------------------------------------- #
# selection + report compilation
# --------------------------------------------------------------------------- #
def _write_synth_check(rundir: Path, failed: list[str]) -> None:
    d = rundir / ".kicraft"
    d.mkdir(parents=True, exist_ok=True)
    (d / "synthesis_check.json").write_text(json.dumps({"failed_checks": failed}))


def test_make_judge_client_relaxes_routing_for_stronger_judge(monkeypatch):
    from kicraft.server.config import Settings

    captured = {}

    def fake_make_client(settings=None):
        captured["settings"] = settings
        return object()

    monkeypatch.setattr(se, "make_client", fake_make_client, raising=False)
    monkeypatch.setattr("kicraft.server.client.make_client", fake_make_client)

    s = Settings(api_key="k", model="deepseek/deepseek-v4-flash", review_model="minimax/minimax-m3")
    # judge != design model -> an independently capped role client is built
    jc = se._make_judge_client(s, "minimax/minimax-m3", skip_judge=False)
    assert jc is not None
    judge_settings = captured["settings"]
    assert judge_settings.provider_order == ["coreweave/fp4"]
    assert judge_settings.max_price_prompt == 0.30

    # judge == design model -> reuse the design client (None)
    assert se._make_judge_client(s, s.model, skip_judge=False) is None
    # --no-judge -> no judge client
    assert se._make_judge_client(s, "minimax/minimax-m3", skip_judge=True) is None


def test_build_label_rc5_distinguishes_failed_check(tmp_path):
    # rc=5 fires for ANY failed §9.x check, not just ERC — the label must read
    # synthesis_check.json instead of hard-coding "ERC errors".
    erc = tmp_path / "erc"
    _write_synth_check(erc, ["9.12 ERC", "9.10 pin existence"])
    assert se._build_label(5, erc) == "ERC errors"

    netlist = tmp_path / "netlist"  # #11 fpc-breakout: 0 ERC errors, §9.13 failed
    _write_synth_check(netlist, ["9.13 netlist faithfulness"])
    assert se._build_label(5, netlist) == "netlist faithfulness"

    other = tmp_path / "other"
    _write_synth_check(other, ["9.7 refdes uniqueness"])
    assert se._build_label(5, other) == "synthesis check failed"

    missing = tmp_path / "missing"  # no synthesis_check.json -> safe fallback
    missing.mkdir()
    assert se._build_label(5, missing) == "synthesis check failed"

    # Non-rc5 labels and None are unchanged.
    assert se._build_label(0, erc) == "fab-ready"
    assert se._build_label(7, erc) == "not fab-ready (DRC)"
    assert se._build_label(None, erc) is None


def test_run_key_single_vs_repeats():
    assert se._run_key("buck-3a", None) == "buck-3a"
    assert se._run_key("buck-3a", 2) == "buck-3a__r2"


def test_per_brief_stats_median_and_iqr():
    recs = [
        {"slug": "a", "archetype": "x", "final": 50.0, "build_rc": 0, "grade": "C"},
        {"slug": "a", "archetype": "x", "final": 90.0, "build_rc": 7, "grade": "A"},
        {"slug": "a", "archetype": "x", "final": 70.0, "build_rc": 0, "grade": "B"},
        {"slug": "b", "archetype": "y", "final": 80.0, "build_rc": 0, "grade": "B"},
    ]
    pb = se._per_brief_stats(recs)
    assert pb["a"]["n"] == 3 and pb["a"]["median_final"] == 70.0
    assert pb["a"]["min_final"] == 50.0 and pb["a"]["max_final"] == 90.0
    assert pb["a"]["iqr"] > 0  # spread across the 3 repeats
    assert pb["a"]["fab_ready"] == 2  # two of three rc==0
    assert pb["b"]["median_final"] == 80.0 and pb["b"]["iqr"] == 0.0  # single sample


def test_compile_report_repeats_aggregates_brief_medians(tmp_path):
    # Two briefs, 2 repeats each; brief medians de-noise the headline.
    records = []
    for slug, finals in (("aa", [60.0, 80.0]), ("bb", [40.0, 90.0])):
        for rep, f in enumerate(finals, start=1):
            records.append(
                {
                    "index": 1,
                    "slug": slug,
                    "repeat": rep,
                    "archetype": "z",
                    "prompt": "p",
                    "stem": f"run_01_{slug}__r{rep}",
                    "rundir": "/r",
                    "grade": "B",
                    "final": f,
                    "verdict": "OK",
                    "build_rc": 0,
                }
            )
    meta = {"started_at": "t", "out_dir": str(tmp_path), "repeats": 2, "judge": False}
    summary = se.compile_report(records, tmp_path, meta)
    assert summary["n"] == 4 and summary["n_briefs"] == 2
    # brief medians: aa -> 70, bb -> 65; mean of those = 67.5
    assert summary["brief_median_mean"] == 67.5
    assert "per_brief" in summary and set(summary["per_brief"]) == {"aa", "bb"}
    md = (tmp_path / "summary.md").read_text()
    assert "median over repeats" in md and "per-brief median" in md


def test_select_limit_and_only():
    es = [{"slug": f"s{i}", "archetype": "a", "brief": f"p{i}"} for i in range(1, 10)]
    assert [i for i, _ in se._select(es, 3, None)] == [1, 2, 3]
    assert [i for i, _ in se._select(es, None, "s1,s5,s9")] == [1, 5, 9]  # by slug
    assert [i for i, _ in se._select(es, None, "1,5,9")] == [1, 5, 9]  # numeric fallback
    assert [e["slug"] for _, e in se._select(es, None, "s2")] == ["s2"]
    assert len(se._select(es, None, None)) == 9


def test_compile_report_aggregates_and_writes(tmp_path):
    records = [
        {
            "index": 1,
            "slug": "aa",
            "archetype": "usb_c_connector",
            "prompt": "a",
            "stem": "run_01_aa",
            "rundir": "/r/1",
            "grade": "A",
            "final": 92.0,
            "verdict": "SHIP",
            "build_rc": 0,
            "build_label": "fab-ready",
            "questions": 0,
            "gates": [],
            "design_cost_usd": 0.05,
            "judge_cost_usd": 0.004,
        },
        {
            "index": 2,
            "slug": "bb",
            "archetype": "fine_pitch",
            "prompt": "b",
            "stem": "run_02_bb",
            "rundir": "/r/2",
            "grade": "C",
            "final": 55.0,
            "verdict": "REWORK",
            "build_rc": 5,
            "build_label": "ERC errors",
            "questions": 1,
            "gates": ["erc_errors"],
            "design_cost_usd": 0.06,
            "judge_cost_usd": 0.004,
        },
        {
            "index": 3,
            "slug": "cc",
            "archetype": "fine_pitch",
            "prompt": "c",
            "stem": "run_03_cc",
            "rundir": "/r/3",
            "error": "RuntimeError: boom",
            "design_cost_usd": 0.0,
        },
    ]
    meta = {
        "started_at": "2026-06-08T00:00:00+00:00",
        "out_dir": str(tmp_path),
        "design_model": "m",
        "judge": True,
        "judge_model": "j",
    }
    s = se.compile_report(records, tmp_path, meta)

    assert s["n"] == 3 and s["graded_n"] == 2 and s["n_errored"] == 1 and s["fab_ready"] == 1
    assert s["mean_final"] == 73.5 and s["median_final"] == 73.5
    assert s["grade_counts"] == {"A": 1, "C": 1, "ERROR": 1}
    assert s["gate_counts"] == {"erc_errors": 1}
    assert round(s["total_cost_usd"], 4) == 0.118
    # per-archetype rollup localizes regressions to a stress dimension
    arche = s["archetype_stats"]
    assert arche["usb_c_connector"] == {
        "n": 1,
        "graded_n": 1,
        "fab_ready": 1,
        "grade_counts": {"A": 1},
        "mean_final": 92.0,
    }
    assert arche["fine_pitch"]["n"] == 2 and arche["fine_pitch"]["graded_n"] == 1
    assert arche["fine_pitch"]["mean_final"] == 55.0 and arche["fine_pitch"]["fab_ready"] == 0
    assert arche["fine_pitch"]["grade_counts"] == {"C": 1, "ERROR": 1}
    assert (tmp_path / "summary.json").is_file()
    md = (tmp_path / "summary.md").read_text()
    assert "Needs attention" in md and "erc_errors" in md and "RuntimeError" in md
    assert "By archetype" in md and "usb_c_connector" in md
