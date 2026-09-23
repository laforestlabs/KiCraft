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
import pytest


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
        {
            "kind": "serialization_recovery",
            "stage": "bom",
            "failure_kind": "truncated_json",
            "resolution_ledger_entries": 2,
        },
        {
            "kind": "candidate_decoded",
            "stage": "bom",
            "attempt": 2,
            "mode": "full",
        },
        {
            "kind": "work_unit_attempt",
            "stage": "bom",
            "unit_id": "bom-s001",
            "outcome": "invalid_work_unit",
        },
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
        "serialization_recovery",
        "candidate_decoded",
        "work_unit_attempt",
        "stage_done",
        "build_start",
        "build_log",
        "build_done",
    ]


def test_stage_failure_attribution_keeps_only_terminal_diagnostic(tmp_path):
    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"kind": "stage_start", "stage": "bom"},
                {
                    "kind": "retry",
                    "stage": "bom",
                    "failure_kind": "commit_rejected",
                    "commit_gate_codes": ["stale_rejected_attempt"],
                },
                {
                    "kind": "stage_done",
                    "stage": "bom",
                    "ok": False,
                    "failure_kind": "commit_process_failed",
                    "defect_codes": ["terminal_process_failure"],
                    "diagnostic": {"code": "form_factor_stack_unowned"},
                    "work_unit_ids": ["bom-s001"],
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    attribution = se._stage_failure_attribution(
        {"stage_status": {"bom": {"ok": False, "failure_kind": "commit_process_failed"}}},
        events,
    )
    assert attribution["failed_stage"] == "bom"
    assert attribution["failure_kind"] == "commit_process_failed"
    assert attribution["failure_codes"] == ["terminal_process_failure"]
    assert attribution["terminal_diagnostic"] == {"code": "form_factor_stack_unowned"}
    assert attribution["work_unit_ids"] == ["bom-s001"]
    assert attribution["stage_subprocess_crash"] is True


def test_stage_failure_report_reads_completed_campaign_without_provider_calls(tmp_path):
    from kicraft.eval.stage_failure_report import analyze_campaign

    run_dir = tmp_path / "run_01_case"
    (run_dir / ".kicraft").mkdir(parents=True)
    (run_dir / ".kicraft" / "state.json").write_text(
        json.dumps(
            {
                "stage_status": {
                    "intent": {"ok": True},
                    "functional_spec": {
                        "ok": False,
                        "failure_kind": "invalid_schema",
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "events.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "slug": "case",
                        "rundir": str(run_dir),
                        "design_committed": False,
                        "design_failure_kind": "invalid_schema",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    report = analyze_campaign(tmp_path)

    assert report["stage_counts"] == {"functional_spec": 1}
    assert report["failure_matrix"][0]["failure_kind"] == "invalid_schema"


# --------------------------------------------------------------------------- #
# run_design: park -> auto-answer -> resume -> complete
# --------------------------------------------------------------------------- #
_FULL_STATE = {
    # A minimal state that satisfies the current stage contracts: the nested
    # models require their own top-level fields, and `design_committed` is read
    # from stage_status, so a fixture without these drives no build at all.
    "project_stem": "B1",
    "intent": {"goal": "a USB LED"},
    "functional_spec": {"blocks": []},
    "architecture": {"sheets": [], "power_nets": [], "inter_sheet_nets": [], "requirements": []},
    "bom": {
        "parts": [
            {
                "ref": "R1",
                "value": "1k",
                "symbol": "Device:R",
                "footprint": "Resistor_SMD:R_0603_1608Metric",
                "sheet": "MAIN",
            }
        ],
        "connections": [
            {"net_name": "VBUS", "sheet": "MAIN", "endpoints": [{"ref": "R1", "pin": "1"}]}
        ],
    },
    "stage_status": {
        stage: {"ok": True}
        for stage in ("intent", "functional_spec", "architecture", "bom", "wiring")
    },
}


def test_run_design_parks_then_resumes_with_suggested_option(tmp_path, monkeypatch):
    rundir = tmp_path / "run"
    (rundir / ".kicraft").mkdir(parents=True)
    seen_answers, seen_instructions, calls = [], [], {"n": 0}

    def fake_run_session(
        ws,
        brief,
        stages,
        answers=None,
        instruction=None,
        client=None,
        progress=None,
        run_id=None,
    ):
        calls["n"] += 1
        seen_answers.append(answers)
        seen_instructions.append(instruction)
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
    assert seen_instructions == [se.NONINTERACTIVE_DEFAULTS_INSTRUCTION] * 2
    kinds = [json.loads(line)["kind"] for line in events.read_text().splitlines()]
    assert "question" in kinds and "stage_done" in kinds


def test_run_design_failed_stage_stops_and_reports_error(tmp_path, monkeypatch):
    (tmp_path / ".kicraft").mkdir(parents=True)

    def fake_run_session(ws, brief, stages, **kw):
        return {
            "status": "failed",
            "last_stage": "bom",
            "failure_kind": "commit_rejected",
            "results": [{"cost_usd": 0.04, "error": "stage-commit rejected"}],
            "questions": None,
        }

    monkeypatch.setattr(se, "run_session", fake_run_session)
    d = se.run_design(object(), "x", tmp_path, lambda ev: None)
    assert d["status"] == "failed" and "rejected" in d["error"] and d["cost_usd"] == 0.04
    assert d["failure_kind"] == "commit_rejected"


def test_run_design_retries_provider_busy_at_top_level(tmp_path, monkeypatch):
    (tmp_path / ".kicraft").mkdir(parents=True)
    calls = {"count": 0}
    delays = []
    events = []

    def fake_run_session(ws, brief, stages, **kw):
        calls["count"] += 1
        if calls["count"] == 1:
            return {
                "status": "failed",
                "last_stage": "intent",
                "failure_kind": "provider_rate_limited",
                "results": [
                    {
                        "cost_usd": 0.0,
                        "error": "provider temporarily rate limited the request",
                    }
                ],
            }
        Path(ws, ".kicraft", "state.json").write_text(json.dumps(_FULL_STATE))
        return {"status": "ok", "last_stage": "wiring", "results": []}

    monkeypatch.setattr(se, "run_session", fake_run_session)
    monkeypatch.setattr(se.time, "sleep", delays.append)
    result = se.run_design(object(), "x", tmp_path, events.append)
    assert result["status"] == "ok"
    assert calls["count"] == 2 and delays == [60.0]
    assert events == [
        {
            "kind": "top_level_retry",
            "failure_kind": "provider_rate_limited",
            "retry_action": "retry_stage",
            "retry_attempt": 1,
            "delay_s": 60.0,
        }
    ]


def test_rate_limited_result_is_rerun_on_resume(tmp_path):
    report = tmp_path / "report.json"
    report.write_text("{}")
    assert not se._reusable(
        {
            "report_path": str(report),
            "design_failure_kind": "provider_rate_limited",
        }
    )


def test_resume_corpus_comes_from_manifest_not_partial_checkpoint(tmp_path):
    (tmp_path / "campaign_manifest.json").write_text(
        json.dumps(
            {
                "immutable": {
                    "corpus": [
                        {"slug": "alpha"},
                        {"slug": "beta"},
                        {"slug": "gamma"},
                    ]
                }
            }
        )
    )
    assert se._resume_corpus_slugs(tmp_path, {"alpha": {"slug": "alpha"}}) == {
        "alpha",
        "beta",
        "gamma",
    }


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

    monkeypatch.setattr(
        se,
        "run_post_wiring_lifecycle",
        lambda *args, **kwargs: {
            "execution_mode": kwargs["execution_mode"],
            "post_wiring_review": {"status": "completed", "cost_usd": 0.002},
            "silkscreen": {"status": "completed", "cost_usd": 0.003},
        },
    )

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
    assert rec["run_id"] == seen_kw["run_id"]

    assert rec["execution_mode"] == "full"
    assert rec["lifecycle"]["post_wiring_review"]["cost_usd"] == 0.002
    assert rec["lifecycle"]["silkscreen"]["cost_usd"] == 0.003


def test_legacy_evaluation_preserves_state_for_legacy_build(tmp_path, monkeypatch):
    state_bytes = json.dumps(_FULL_STATE).encode()

    def design(ws, *args, **kwargs):
        se.pipeline_dispatch.write_marker(ws, "legacy")
        (ws / ".kicraft" / "state.json").write_bytes(state_bytes)
        return {"status": "ok", "results": [], "last_stage": "wiring"}

    def current_tail(state_path, *args, **kwargs):
        Path(state_path).write_text('{"corrupted_by_current_schema": true}')
        return {}

    def build(ws, *args, **kwargs):
        return 0 if (ws / ".kicraft" / "state.json").read_bytes() == state_bytes else 2

    monkeypatch.setattr(se, "run_session", design)
    monkeypatch.setattr(se, "run_post_wiring_lifecycle", current_tail)
    monkeypatch.setattr(se, "run_build", build)
    monkeypatch.setattr(se, "evaluate_project", lambda *args, **kwargs: _fake_report())
    rec = se.evaluate_one(
        object(),
        1,
        {"slug": "legacy-isolation", "archetype": "test", "brief": "RC filter"},
        tmp_path,
        judge_model=None,
        skip_judge=True,
    )
    assert rec["build_rc"] == 0
    assert (Path(rec["rundir"]) / ".kicraft" / "state.json").read_bytes() == state_bytes


def test_evaluate_one_design_only_skips_build_and_scoring(tmp_path, monkeypatch):
    def fake_run_session(ws, brief, stages, **kw):
        state = {
            **_FULL_STATE,
            "stage_status": {
                stage: {"ok": True}
                for stage in ("intent", "functional_spec", "architecture", "bom", "wiring")
            },
        }
        Path(ws, ".kicraft", "state.json").write_text(json.dumps(state))
        return {
            "status": "ok",
            "results": [{"cost_usd": 0.03}],
            "questions": None,
            "last_stage": "wiring",
        }

    monkeypatch.setattr(se, "run_session", fake_run_session)
    monkeypatch.setattr(
        se,
        "run_build",
        lambda *args, **kwargs: pytest.fail("design-only must not build"),
    )
    monkeypatch.setattr(
        se,
        "evaluate_project",
        lambda *args, **kwargs: pytest.fail("design-only must not score"),
    )

    rec = se.evaluate_one(
        object(),
        1,
        {"slug": "design-only", "archetype": "test", "brief": "A design-only brief"},
        tmp_path,
        judge_model=None,
        skip_judge=True,
        design_only=True,
    )

    assert rec["design_status"] == "ok"
    assert rec["design_committed"] is True
    assert rec["build_rc"] is None
    assert rec["design_cost_usd"] == 0.03
    assert "grade" not in rec


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


def test_same_brief_campaigns_isolate_budget_and_reported_spend(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kicraft.server.config import Settings
    from kicraft.server.spend_guard import SpendGuard

    guard = SpendGuard(
        Settings(
            api_key="test",
            ledger_path=tmp_path / "ledger.db",
            kill_switch=False,
            project_llm_budget_usd=0.10,
            daily_usd_ceiling=10.0,
            total_usd_ceiling=10.0,
        )
    )

    def paid_session(ws, brief, stages, *, run_id, **kwargs):
        guard.preflight(call_ceiling_usd=0.06, run_id=run_id)
        guard.record("fixture", 10, 2, 0.06, {"run_id": run_id, "stage": "intent"})
        return {
            "status": "failed",
            "last_stage": "intent",
            "results": [{"cost_usd": 0.06, "error": "controlled paid failure"}],
        }

    monkeypatch.setattr(se.time, "time", lambda: 1700000000)
    monkeypatch.setattr(se, "run_session", paid_session)
    entry = {"slug": "same-brief", "archetype": "fixture", "brief": "An isolated financial probe"}
    records = [
        se.evaluate_one(
            SimpleNamespace(guard=guard),
            9,
            entry,
            tmp_path / campaign,
            judge_model=None,
            skip_judge=True,
        )
        for campaign in ("campaign-a", "campaign-b")
    ]

    assert records[0]["run_id"] != records[1]["run_id"]
    for record in records:
        assert "error" not in record
        assert "budget_refusal" not in record
        assert record["design_error"] == "controlled paid failure"
        assert record["design_cost_usd"] == pytest.approx(0.06)
        assert record["stage_cost_usd"] == {"intent": pytest.approx(0.06)}
        assert guard.spent_for_run(record["run_id"]) == pytest.approx(0.06)
        assert Path(record["rundir"]).name == "run_09_same-brief"
        report = json.loads(Path(record["report_path"]).read_text())
        assert report["run_id"] == record["run_id"]
        assert report["metrics"]["token_usage"] == {
            "input_tokens": 10,
            "output_tokens": 2,
            "total_tokens": 12,
            "turns": 1,
            "estimated_cost_usd": 0.06,
            "cost_known": True,
            "by_model": {"fixture": 1},
        }
    assert guard.spent_total() == pytest.approx(0.12)


def test_needs_attention_lists_every_incomplete_or_nonzero_build_run(tmp_path):
    """A pre-build failure or a non-rc0 build is reported whatever its grade."""
    records = [
        # A high-graded design that never committed (the frozen #21 shape).
        {
            "index": 21,
            "slug": "proto-shield",
            "stem": "PROTO_SHIELD",
            "final": 76.0,
            "grade": "B",
            "design_committed": False,
            "failed_stage": "wiring",
            "build_rc": None,
        },
        # A committed design whose build returned a routing failure code.
        {
            "index": 10,
            "slug": "rp2040-min",
            "stem": "RP2040_MIN",
            "final": 83.5,
            "grade": "B",
            "design_committed": True,
            "build_rc": 6,
        },
        # A clean committed build: NOT a needs-attention entry.
        {
            "index": 1,
            "slug": "rc-lowpass-bnc",
            "stem": "RC_FILTER_BREAKOUT",
            "final": 79.5,
            "grade": "B",
            "design_committed": True,
            "build_rc": 0,
        },
    ]
    summary = se.compile_report(records, tmp_path, {})
    attention = (tmp_path / "summary.md").read_text().split("## Needs attention")[-1]
    assert "PROTO_SHIELD" in attention or "proto-shield" in attention
    assert "RP2040_MIN" in attention or "rp2040-min" in attention
    assert "RC_FILTER_BREAKOUT" not in attention
    assert summary["n"] == 3


def test_budget_exception_preserves_paid_failed_stage_and_campaign_cost(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from kicraft.server.config import Settings
    from kicraft.server.spend_guard import SpendGuard

    guard = SpendGuard(
        Settings(
            api_key="test",
            ledger_path=tmp_path / "ledger.db",
            kill_switch=False,
            daily_usd_ceiling=10.0,
            total_usd_ceiling=10.0,
        )
    )

    def paid_failure(ws, brief, stages, *, run_id, progress, **kwargs):
        guard.record("m", 1, 1, 0.0048, {"run_id": run_id, "stage": "intent"})
        progress({"kind": "stage_start", "stage": "intent"})
        progress({"kind": "stage_done", "stage": "intent", "ok": True, "cost": 0.0048})
        progress({"kind": "stage_start", "stage": "bom"})
        guard.record("m", 1, 1, 0.06, {"run_id": run_id, "stage": "bom"})
        progress(
            {
                "kind": "work_unit_attempt",
                "stage": "bom",
                "unit_id": "bom-s002",
                "outcome": "invalid_work_unit",
                "cost_usd": 0.06,
            }
        )
        guard.preflight(call_ceiling_usd=0.0384, run_id=run_id)
        pytest.fail("over-budget call was admitted")

    monkeypatch.setattr(se, "run_session", paid_failure)
    rec = se.evaluate_one(
        SimpleNamespace(guard=guard),
        1,
        {"slug": "failed", "archetype": "x", "brief": "a board"},
        tmp_path,
        judge_model=None,
        skip_judge=True,
        design_only=True,
    )
    assert rec["design_committed"] is False
    assert rec["failed_stage"] == "bom"
    assert rec["failure_kind"] == "budget_refused"
    assert rec["design_cost_usd"] == pytest.approx(0.0648)
    assert rec["stage_cost_usd"] == {"intent": pytest.approx(0.0048), "bom": pytest.approx(0.06)}
    assert rec["budget_refusal"]["call_ceiling_usd"] == pytest.approx(0.0384)
    success = {
        "index": 2,
        "slug": "passed",
        "design_committed": True,
        "design_cost_usd": 0.01,
        "judge_cost_usd": 0.002,
    }
    report = se.compile_report([rec, success], tmp_path, {})
    assert report["stage_outcomes"] == {"bom": 1, "passed": 1}
    assert report["failed_run_cost_usd"] == pytest.approx(0.0648)
    assert report["cost_per_committed_design_usd"] == pytest.approx(0.0768)
    assert se._campaign_costs([rec])["cost_per_committed_design_usd"] is None


def test_failure_matrix_separates_repair_exhaustion_from_subprocess_crash(tmp_path):
    """A stage-commit crash, a bounded-repair exhaustion and a build-only failure
    must be distinguishable: collapsing them hides whether the pipeline or the
    design failed.  (The frozen #21 shape was a traceback reported as a design
    failure and graded B.)"""
    from kicraft.eval.stage_failure_report import analyze_campaign

    crashed = tmp_path / "crashed"
    crashed.mkdir()
    (crashed / "events.jsonl").write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"kind": "stage_start", "stage": "wiring"},
                {
                    "kind": "stage_crash",
                    "stage": "wiring",
                    "returncode": 1,
                    "traceback": "Traceback (most recent call last): ...",
                },
            ]
        )
    )
    repaired = tmp_path / "repaired"
    repaired.mkdir()
    (repaired / "events.jsonl").write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"kind": "stage_start", "stage": "bom"},
                {
                    "kind": "work_unit_attempt",
                    "stage": "bom",
                    "outcome": "invalid_work_unit",
                    "cost_usd": 0.01,
                },
                {"kind": "work_unit_repair_exhausted", "stage": "bom", "unit_id": "bom-s002"},
            ]
        )
    )
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {
                        "slug": "crashed",
                        "rundir": str(crashed),
                        "design_committed": False,
                        "failure_kind": "commit_process_failed",
                    },
                    {
                        "slug": "repaired",
                        "rundir": str(repaired),
                        "design_committed": False,
                        "failure_kind": "unit_repair_exhausted",
                    },
                    # The design committed; only the build (router) failed.  It is not a design failure.
                    {
                        "slug": "route-failed",
                        "design_committed": True,
                        "build_rc": 6,
                        "design_cost_usd": 0.02,
                    },
                ]
            }
        )
    )
    report = analyze_campaign(tmp_path)
    matrix = {
        (row["stage"], row["failure_kind"]): row["briefs"] for row in report["failure_matrix"]
    }
    assert matrix[("wiring", "commit_process_failed")] == ["crashed"]
    assert matrix[("bom", "unit_repair_exhausted")] == ["repaired"]
    assert matrix[("passed", "passed")] == ["route-failed"]


def test_legacy_failure_report_recovers_cost_without_double_counting(tmp_path):
    from kicraft.eval.stage_failure_report import analyze_campaign

    failed = tmp_path / "failed"
    failed.mkdir()
    events = [
        {"kind": "stage_start", "stage": "bom"},
        {"kind": "work_unit_attempt", "stage": "bom", "cost_usd": 0.01},
        {"kind": "stage_done", "stage": "bom", "ok": True, "cost": 0.01},
        {"kind": "stage_start", "stage": "wiring"},
        {"kind": "work_unit_attempt", "stage": "wiring", "cost_usd": 0.0548},
    ]
    (failed / "events.jsonl").write_text("\n".join(json.dumps(event) for event in events))
    (tmp_path / "summary.json").write_text(
        json.dumps(
            {
                "runs": [
                    {"slug": "failed", "rundir": str(failed), "error": "BudgetExceeded: refused"},
                    {"slug": "passed", "design_committed": True, "design_cost_usd": 0.00872},
                ],
            }
        )
    )
    report = analyze_campaign(tmp_path)
    assert report["stage_counts"] == {"wiring": 1, "passed": 1}
    assert report["failure_matrix"][1]["failure_kind"] == "budget_refused"
    assert report["total_cost_usd"] == pytest.approx(0.07352)
    assert report["failed_run_cost_usd"] == pytest.approx(0.0648)
    assert report["cost_per_committed_design_usd"] == pytest.approx(0.07352)


def test_source_fingerprint_changes_for_dirty_and_untracked_source(tmp_path, monkeypatch):
    import subprocess

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=tmp_path)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    source = tmp_path / "kicraft" / "eval" / "self_eval.py"
    source.parent.mkdir(parents=True)
    source.write_text("original\n")
    git("add", ".")
    git("commit", "-qm", "initial")
    monkeypatch.setattr(se, "__file__", str(source))
    initial = se._source_fingerprint()
    source.write_text("changed\n")
    assert se._source_fingerprint() != initial
    source.write_text("original\n")
    assert se._source_fingerprint() == initial
    (source.parent / "new_runtime.py").write_text("new source\n")
    assert se._source_fingerprint() != initial


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
    monkeypatch.setattr(se, "run_post_wiring_lifecycle", lambda *args, **kwargs: {})
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


def test_full_report_attends_every_frozen_non_rc0_run(tmp_path):
    frozen = json.loads(
        (Path(__file__).parents[1] / "logs/self_eval/20260915T132650Z/summary.json").read_text()
    )
    records = frozen["runs"]
    non_rc0 = [record for record in records if record.get("build_rc") != 0]
    assert len(non_rc0) == 23

    summary = se.compile_report(
        records,
        tmp_path,
        {key: value for key, value in frozen.items() if key != "runs"},
    )
    attention = (tmp_path / "summary.md").read_text().split("## Needs attention", 1)[1]
    for record in non_rc0:
        assert f"**#{record['index']}** {record['stem']}" in attention
    assert "**#10** run_10_rp2040-min" in attention
    assert "**#21** run_21_proto-shield" in attention
    assert summary["execution_mode"] == "full"
    assert {
        phase: summary["lifecycle"][phase]["status"]
        for phase in ("post_wiring_review", "silkscreen", "judge")
    } == {
        "post_wiring_review": "not-recorded",
        "silkscreen": "not-recorded",
        "judge": "not-recorded",
    }


def test_design_only_success_is_not_reported_as_a_build_failure(tmp_path):
    summary = se.compile_report(
        [
            {
                "index": 1,
                "slug": "canary",
                "archetype": "fixture",
                "prompt": "p",
                "stem": "run_01_canary",
                "rundir": "/r",
                "execution_mode": "design-only",
                "design_committed": True,
                "build_rc": None,
            }
        ],
        tmp_path,
        {"design_only": True, "execution_mode": "design-only"},
    )
    assert "Needs attention" not in (tmp_path / "summary.md").read_text()
    assert summary["fab_ready"] == 0
