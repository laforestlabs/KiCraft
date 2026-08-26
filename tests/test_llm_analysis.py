from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from kicraft.eval import llm_analysis as analysis
from kicraft.eval.llm_canary import COHORT, ENVELOPE_USD
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS

SENTINEL = "FORBIDDEN-CUSTOMER-MODEL-CONTENT"
DESIGNER = {
    "profile": "flash",
    "model": "deepseek/deepseek-v4-flash-0731",
    "provider_order": ["deepinfra/fp8"],
    "max_price_prompt": 0.11,
    "max_price_completion": 0.24,
}
JUDGE = {
    "model": "minimax/minimax-m3",
    "provider_order": ["coreweave/fp4"],
    "max_price_prompt": 0.30,
    "max_price_completion": 1.25,
}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ledger(path: Path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE spend (id INTEGER PRIMARY KEY, ts TEXT, model TEXT, "
        "input_tokens INTEGER, output_tokens INTEGER, cost_usd REAL, meta TEXT)"
    )
    conn.execute(
        "CREATE TABLE stage_runs (id INTEGER PRIMARY KEY, ts TEXT, run_id TEXT, "
        "stage TEXT, ok INTEGER, attempts INTEGER, rounds INTEGER, tool_calls INTEGER, "
        "wall_s REAL, cpu_s REAL, cost_usd REAL, failure_kind TEXT, "
        "emitted_collection_count INTEGER, expanded_component_count INTEGER)"
    )
    return conn


def _campaign(tmp_path: Path, *, mode="pass", with_spend=False, candidate_unknown=False):
    batch = tmp_path / "batch"
    batch.mkdir(parents=True)
    ledger = tmp_path / "ledger.db"
    conn = _ledger(ledger)
    by_slug = {entry["slug"]: entry for entry in BENCHMARK_PROMPTS}
    cohort = [
        {
            "slug": slug,
            "archetype": by_slug[slug]["archetype"],
            "brief_sha256": hashlib.sha256(by_slug[slug]["brief"].encode()).hexdigest(),
        }
        for slug in COHORT
    ]
    corpus = [
        {
            "index": index,
            "slug": row["slug"],
            "brief_hash": analysis._stable_hash(by_slug[row["slug"]]["brief"]),
        }
        for index, row in enumerate(cohort, 1)
    ]
    preflights = {}
    for role, settings in (("designer", DESIGNER), ("judge", JUDGE)):
        path = batch / f"preflight-{role}.json"
        _write(
            path,
            {
                "ok": True,
                "role": role,
                "model": settings["model"],
                "provider_order": settings["provider_order"],
            },
        )
        preflights[role] = {
            "path": path.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    immutable = {
        "campaign_id": "campaign-test",
        "envelope_usd": ENVELOPE_USD,
        "checkout": {"commit": "abc123", "dirty_paths": [], "dirty_diff_sha256": "0" * 64},
        "cohort": cohort,
        "designer": DESIGNER,
        "judge": JUDGE,
    }
    _write(
        batch / "canary_manifest.json",
        {
            "schema_version": 1,
            "run_status": "batch_complete",
            "immutable": immutable,
            "preflights": preflights,
        },
    )
    _write(
        batch / "campaign_manifest.json",
        {
            "schema_version": 1,
            "immutable": {
                "code_revision": "abc123",
                "design_profile": DESIGNER["profile"],
                "design_model": DESIGNER["model"],
                "design_provider_order": DESIGNER["provider_order"],
                "judge_model": JUDGE["model"],
                "judge_provider_order": JUDGE["provider_order"],
                "repeats": 1,
                "response_policies": {
                    "design": "kicraft_<stage>_response_v1",
                    "judge": "kicraft_eval_judge_v1",
                    "wiring_patch": "kicraft_wiring_patch_v1",
                },
                "corpus": corpus,
            },
        },
    )
    records = []
    stages = ("intent", "functional_spec", "architecture", "bom", "wiring")
    for index, slug in enumerate(COHORT, 1):
        run_id = f"prun_{index:02d}_{slug}-1"
        rundir = batch / f"run_{index:02d}_{slug}"
        (rundir / ".kicraft").mkdir(parents=True)
        (rundir / "eval").mkdir()
        (rundir / "brief.txt").write_text(SENTINEL + "\n")
        failed = mode == "fail" and index == 1
        events = []
        statuses = {}
        for stage in stages:
            ok = not failed or stage != "intent"
            if failed and stage != "intent":
                break
            events.extend(
                [
                    {"kind": "stage_start", "stage": stage, "reasoning": SENTINEL},
                    {"kind": "stage_done", "stage": stage, "ok": ok, "text": SENTINEL},
                ]
            )
            statuses[stage] = (
                {"ok": True}
                if ok
                else {"ok": False, "failure_kind": "transport_error", "error": SENTINEL}
            )
        if not failed:
            if mode == "findings" and index == 1:
                events.append({"kind": "retry", "stage": "intent", "errors": []})
            if candidate_unknown and index == 1:
                events.append(
                    {
                        "kind": "candidate_decoded",
                        "stage": "bom",
                        "attempt": 1,
                        "serialization_recovery": False,
                        "clean_slate": False,
                        "expanded_component_count": 0,
                        "unknown_sheet_references": [{"ref": "R9", "sheet": "TYPO"}],
                        "candidate": SENTINEL,
                    }
                )
            events.extend([{"kind": "build_start"}, {"kind": "build_done", "ok": False, "rc": 6}])
        events.extend(
            [
                {"kind": "reasoning_delta", "text": SENTINEL},
                {"kind": "answer_delta", "text": SENTINEL},
                {"kind": "tool_result", "name": "lookup", "output": SENTINEL},
            ]
        )
        (rundir / "events.jsonl").write_text("".join(json.dumps(event) + "\n" for event in events))
        state = {
            "stage_status": statuses,
            "architecture": {
                "sheets": [{"name": "MAIN", "stem": "MAIN", "function": "all"}],
                "power_nets": [],
                "inter_sheet_nets": [],
            },
            "bom": {"parts": [], "connections": [], "no_connect_pins": []},
            "history": [{"answer": SENTINEL}],
        }
        _write(rundir / ".kicraft" / "state.json", state)
        _write(rundir / "eval" / "report.json", {"raw": SENTINEL})
        attempts = 2 if mode == "findings" and index == 1 else 1
        failure_kind = "transport_error" if failed else None
        conn.execute(
            "INSERT INTO stage_runs (ts,run_id,stage,ok,attempts,rounds,tool_calls,"
            "wall_s,cpu_s,cost_usd,failure_kind,emitted_collection_count,"
            "expanded_component_count) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                "2026-08-26T00:00:00+00:00",
                run_id,
                "intent",
                0 if failed else 1,
                attempts,
                1,
                0,
                1.0,
                0.1,
                0.001 if with_spend else 0.0,
                failure_kind,
                7,
                3,
            ),
        )
        cost = 0.0
        if with_spend:
            meta = {
                "run_id": run_id,
                "stage": "intent",
                "phase": "stream",
                "profile": DESIGNER["profile"],
                "provider": DESIGNER["provider_order"][0],
                "finish_reason": "stop",
                "response_policy_name": "kicraft_intent_response_v1",
                "reasoning_policy_name": "design",
                "cached_tokens": 2,
            }
            conn.execute(
                "INSERT INTO spend (ts,model,input_tokens,output_tokens,cost_usd,meta) VALUES (?,?,?,?,?,?)",
                (
                    "2026-08-26T00:00:00+00:00",
                    DESIGNER["model"],
                    10,
                    5,
                    0.001,
                    json.dumps(meta),
                ),
            )
            cost = 0.001
        records.append(
            {
                "slug": slug,
                "archetype": by_slug[slug]["archetype"],
                "run_id": run_id,
                "stem": rundir.name,
                "rundir": str(rundir),
                "design_status": "failed" if failed else "ok",
                "design_error": "transport error" if failed else None,
                "design_cost_usd": cost,
                "judge_cost_usd": 0.0,
                "questions": 0,
                "build_rc": None if failed else 6,
                "duration_s": float(index),
            }
        )
    conn.commit()
    conn.close()
    _write(
        batch / "summary.json",
        {
            "runs": records,
            "repeats": 1,
            "parallel": 1,
            "build_slots": 1,
            "full_events": True,
            "judge": True,
            "design_model": DESIGNER["model"],
            "design_profile": DESIGNER["profile"],
            "design_provider_order": DESIGNER["provider_order"],
            "judge_model": JUDGE["model"],
        },
    )
    return batch, ledger


def _analyze(tmp_path, **kwargs):
    batch, ledger = _campaign(tmp_path, **kwargs)
    rc = analysis.analyze_batch(
        batch,
        baseline=Path("/home/kicraft/.kicraft/self_eval/20260825T033602Z"),
        ledger=ledger,
        projects_dir=tmp_path / "no-production",
    )
    return batch, rc, json.loads((batch / "llm_analysis.json").read_text())


def test_writes_reports_for_synthetic_nine_run_fixture_and_redacts_forbidden_content(tmp_path):
    batch, rc, report = _analyze(tmp_path, with_spend=True)
    assert rc == 0 and report["verdict"] == "PASS"
    assert report["integrity"]["valid"] and len(report["runs"]) == 9
    assert report["aggregates"]["design_complete"] == {"numerator": 9, "denominator": 9}
    assert report["aggregates"]["total_cost_usd"] == pytest.approx(0.009)
    for output in (batch / "llm_analysis.json", batch / "llm_analysis.md"):
        assert SENTINEL not in output.read_text()
    markdown = (batch / "llm_analysis.md").read_text()
    for heading in (
        "Verdict",
        "Identity",
        "Canary table",
        "Recent failure-class deltas",
        "Production comparison",
        "Build outcomes",
        "Recommendation",
        "Reproduction",
    ):
        assert f"# {heading}" in markdown


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (
            lambda batch, ledger: (batch / "preflight-judge.json").unlink(),
            "missing preflight-judge.json",
        ),
        (
            lambda batch, ledger: _write(
                batch / "canary_manifest.json",
                {**json.loads((batch / "canary_manifest.json").read_text()), "run_status": "ready"},
            ),
            "run_status",
        ),
        (
            lambda batch, ledger: _write(
                batch / "canary_manifest.json",
                {
                    **json.loads((batch / "canary_manifest.json").read_text()),
                    "immutable": {
                        **json.loads((batch / "canary_manifest.json").read_text())["immutable"],
                        "checkout": {
                            "commit": "abc123",
                            "dirty_paths": ["kicraft/server/config.py"],
                            "dirty_diff_sha256": "1" * 64,
                        },
                    },
                },
            ),
            "uncommitted runtime/config changes",
        ),
        (
            lambda batch, ledger: _write(
                batch / "summary.json",
                {**json.loads((batch / "summary.json").read_text()), "parallel": 2},
            ),
            "parallel",
        ),
        (
            lambda batch, ledger: (batch / f"run_01_{COHORT[0]}" / "events.jsonl").unlink(),
            "missing run_01",
        ),
        (
            lambda batch, ledger: (
                sqlite3.connect(ledger).execute("DELETE FROM stage_runs").connection.commit()
            ),
            "attribution",
        ),
    ],
)
def test_integrity_failures_are_invalid_campaign(tmp_path, mutation, expected):
    batch, ledger = _campaign(tmp_path)
    mutation(batch, ledger)
    rc = analysis.analyze_batch(
        batch,
        baseline=batch,
        ledger=ledger,
        projects_dir=tmp_path / "none",
    )
    report = json.loads((batch / "llm_analysis.json").read_text())
    assert rc == 2 and report["verdict"] == "INVALID_CAMPAIGN"
    assert expected in " ".join(report["integrity"]["errors"])


def test_all_four_verdicts_and_candidate_sheet_stop_gate(tmp_path):
    batch, rc, report = _analyze(tmp_path / "pass")
    assert (rc, report["verdict"]) == (0, "PASS")
    batch, rc, report = _analyze(tmp_path / "findings", mode="findings")
    assert (rc, report["verdict"]) == (0, "PASS_WITH_LLM_FINDINGS")
    batch, rc, report = _analyze(tmp_path / "fail", mode="fail")
    assert (rc, report["verdict"]) == (1, "FAIL_LLM")
    assert report["runs"][0]["classification"] == "operational"
    batch, rc, report = _analyze(tmp_path / "unknown", candidate_unknown=True)
    assert (rc, report["verdict"]) == (1, "FAIL_LLM")
    assert report["runs"][0]["bom"]["candidate_unknown_sheet_references"] == [
        {"ref": "R9", "sheet": "TYPO"}
    ]
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    assert (
        analysis.analyze_batch(
            invalid, baseline=invalid, ledger=invalid / "none.db", projects_dir=invalid
        )
        == 2
    )


def test_classification_precedence():
    state = {"stage_status": {"bom": {"ok": False, "failure_kind": "provider_error"}}}
    record = {"design_error": "invalid_json commit rejected"}
    assert analysis._classify(record, state, [])[0] == "operational"
    state["stage_status"]["bom"]["failure_kind"] = "reasoning_loop"
    assert analysis._classify({}, state, [])[0] == "reasoning"
    state["stage_status"]["bom"]["failure_kind"] = "truncated_json"
    assert analysis._classify({}, state, [])[0] == "serialization"
    state["stage_status"]["bom"]["failure_kind"] = "invalid_schema"
    assert analysis._classify({}, state, [])[0] == "schema_contract"
    state["stage_status"]["bom"]["failure_kind"] = "commit_rejected"
    assert analysis._classify({}, state, [])[0] == "commit_contract"


def test_tool_signatures_mpn_normalization_and_wiring_progress():
    first = analysis.normalize_tool_signature(
        {"name": "lookup", "args": {"mpn": "ABC-123", "nested": [SENTINEL], "limit": 3}}
    )
    second = analysis.normalize_tool_signature(
        {"name": "lookup", "args": {"limit": 3, "mpn": "abc-123"}}
    )
    assert first == second and SENTINEL not in first
    events = [
        {
            "kind": "retry",
            "stage": "wiring",
            "errors": ["§9.19 short"],
            "offenders": ["U1.1", "R1.1"],
        },
        {"kind": "retry", "stage": "wiring", "errors": ["§9.15 dangling"], "offenders": ["R1.1"]},
        {
            "kind": "candidate_decoded",
            "stage": "wiring",
            "clean_slate": True,
            "expanded_component_count": 0,
        },
        {
            "kind": "retry",
            "stage": "wiring",
            "errors": ["§9.19 short"],
            "offenders": ["U1.1", "R1.1"],
        },
    ]
    result = analysis._wiring_analysis(events, {"wiring": {"cost_usd": 0.2}})
    assert result["families"] == {"9.15_dangling_net": 1, "9.19_multi_net_pin": 2}
    assert result["ordered_rejection_signatures"][1]["progress"] == "progress"
    assert result["ordered_rejection_signatures"][2]["progress"] == "no_progress"
    assert result["no_progress"] is True
    assert result["full_calls"] == 1
    assert result["correction_calls"] == 0


def test_stage_modes_tokens_cost_and_not_observable_fields(tmp_path):
    batch, _, report = _analyze(tmp_path, with_spend=True)
    stage = report["runs"][0]["stages"]["intent"]
    assert stage["billed_provider_calls"] == 1
    assert stage["call_modes"] == {"normal": 1}
    assert stage["tokens"]["input"] == 10 and stage["tokens"]["cache"] == 2
    assert stage["tokens"]["reasoning"]["status"] == "not_observable"
    assert stage["cost_usd"] == pytest.approx(0.001)
    assert analysis.check_batch(batch, ledger=tmp_path / "ledger.db") == 0


def test_production_last_25_plus_witness_selection_is_redacted(tmp_path):
    root = tmp_path / "projects"
    for index in range(30):
        path = root / "2" / str(800 + index) / ".kicraft" / "state.json"
        _write(path, {"stage_status": {"intent": {"ok": True}}, "brief": SENTINEL})
    for witness in analysis._WITNESSES:
        user, project = witness.split("/")
        _write(
            root / user / project / ".kicraft" / "state.json",
            {
                "stage_status": {"bom": {"ok": False, "failure_kind": "commit_rejected"}},
                "architecture": {"sheets": [{"name": "MAIN"}]},
                "bom": {"parts": [{"ref": "R1", "sheet": "TYPO"}]},
                "brief": SENTINEL,
            },
        )
    result = analysis._production(root)
    assert result["window_count"] == 25
    ids = {row["project_id"] for row in result["projects"]}
    assert set(analysis._WITNESSES) <= ids
    assert SENTINEL not in json.dumps(result)
    assert result["unknown_sheet_references"] >= 4
