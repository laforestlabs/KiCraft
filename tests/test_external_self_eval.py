"""External campaigns preserve the first observation and its full denominator."""

from __future__ import annotations

import json

import pytest

from kicraft.eval import self_eval as se
from kicraft.server.config import Settings


def _inputs(tmp_path):
    manifest = {
        "schema_version": 1,
        "corpus_id": "fresh-development",
        "sampling": {"kind": "synthetic", "description": "Synthetic development examples."},
        "capability_envelope": {"max_input_v": 24},
        "policy": {
            "max_cost_usd": 0.6,
            "max_duration_s": 600,
            "max_park_rounds": 12,
            "build_timeout_s": 180,
        },
        "briefs": [
            {
                "slug": slug,
                "brief": f"Make a {slug} LED indicator.",
                "archetype": "indicator",
                "families": ["led"],
                "eligible": True,
                "provenance": {"kind": "synthetic"},
            }
            for slug in ("alpha", "beta")
        ],
    }
    obligations = {
        "schema_version": 1,
        "corpus_id": manifest["corpus_id"],
        "briefs": {
            slug: [
                {
                    "id": f"{slug}.led",
                    "check": {"kind": "part_class_count", "part_class": "led", "minimum": 1},
                }
            ]
            for slug in ("alpha", "beta")
        },
    }
    paths = (tmp_path / "briefs.json", tmp_path / "obligations.json")
    for path, value in zip(paths, (manifest, obligations)):
        path.write_text(json.dumps(value))
    return paths


def _settings(tmp_path, monkeypatch):
    settings = Settings(
        api_key="",
        ledger_path=tmp_path / "ledger.db",
        total_usd_ceiling=10,
        daily_usd_ceiling=10,
        project_llm_budget_usd=0.6,
    )
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setenv("KICRAFT_TOTAL_USD_CEILING", "10")
    monkeypatch.setenv("KICRAFT_PROJECT_LLM_BUDGET_USD", "0.6")
    monkeypatch.setenv("KICRAFT_EVAL_STRICT_BUDGET", "0")
    monkeypatch.setattr(se, "_source_fingerprint", lambda: "unchanged")
    monkeypatch.setattr(se, "_make_judge_client", lambda *args: None)
    from kicraft.server import client

    monkeypatch.setattr(client, "make_client", lambda settings: object())


def test_resume_keeps_failed_attempts_and_recovers_deleted_external_inputs(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    _settings(tmp_path, monkeypatch)
    calls = []

    def fail(client, index, entry, out_dir, **kwargs):
        calls.append(entry["slug"])
        return {
            "index": index,
            "slug": entry["slug"],
            "archetype": entry["archetype"],
            "prompt": entry["brief"],
            "repeat": None,
            "product_success": False,
            "product_failure_kind": "provider_failure",
            "attempt_completed": True,
            "error": "provider failed",
            "design_committed": False,
        }

    monkeypatch.setattr(se, "evaluate_one", fail)
    out = tmp_path / "campaign"
    args = [
        "--out",
        str(out),
        "--brief-manifest",
        str(paths[0]),
        "--obligations",
        str(paths[1]),
        "--campaign-budget-usd",
        "5",
        "--parallel",
        "1",
        "--build-slots",
        "1",
        "--no-judge",
    ]
    assert se.main(args) == 0
    assert calls == ["alpha", "beta"]
    for path in paths:
        path.unlink()
    assert (
        se.main(["--resume", str(out), "--parallel", "1", "--build-slots", "1", "--no-judge"]) == 0
    )
    assert calls == ["alpha", "beta"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["product_denominator"] == 2
    assert summary["product_success"] == 0
    assert summary["product_failure_families"] == {"provider_failure": 2}


def test_external_interrupted_attempt_is_preserved_not_retried(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    _settings(tmp_path, monkeypatch)
    out = tmp_path / "campaign"

    def fail(client, index, entry, out_dir, **kwargs):
        return {
            "index": index,
            "slug": entry["slug"],
            "archetype": "indicator",
            "repeat": None,
            "product_success": False,
            "product_failure_kind": "design_not_committed",
        }

    monkeypatch.setattr(se, "evaluate_one", fail)
    assert (
        se.main(
            [
                "--out",
                str(out),
                "--brief-manifest",
                str(paths[0]),
                "--obligations",
                str(paths[1]),
                "--campaign-budget-usd",
                "5",
                "--parallel",
                "1",
                "--build-slots",
                "1",
                "--no-judge",
            ]
        )
        == 0
    )
    summary_path = out / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["runs"] = summary["runs"][:1]
    summary_path.write_text(json.dumps(summary))
    attempt = out / "run_02_beta" / "attempt.json"
    attempt.unlink()
    sentinel = attempt.parent / "expensive-artifact.txt"
    sentinel.write_text("preserve")
    monkeypatch.setattr(
        se, "evaluate_one", lambda *a, **kw: pytest.fail("must not retry an interrupted attempt")
    )
    assert (
        se.main(["--resume", str(out), "--parallel", "1", "--build-slots", "1", "--no-judge"]) == 2
    )
    result = json.loads(summary_path.read_text())
    assert result["product_failure_families"]["interrupted_attempt"] == 1
    assert result["campaign_complete"] is False
    assert sentinel.read_text() == "preserve"


def test_general_brief_does_not_auto_answer_a_blocking_question(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "read_state", lambda _: {})
    monkeypatch.setattr(se, "remaining_stages", lambda _: ["intent"])
    calls = []

    def park(*args, **kwargs):
        calls.append(kwargs.get("answers"))
        return {
            "status": "awaiting_input",
            "questions": [{"text": "May I omit the sensor?", "options": ["Yes"]}],
        }

    monkeypatch.setattr(se, "run_session", park)
    result = se.run_design(
        object(), "Include a sensor", tmp_path, lambda _: None, auto_answer_questions=False
    )
    assert result["failure_kind"] == "clarification_required"
    assert calls == [None]


def test_external_budget_refusal_preserves_denominator_without_dispatch(tmp_path, monkeypatch):
    paths = _inputs(tmp_path)
    _settings(tmp_path, monkeypatch)
    monkeypatch.setattr(
        se, "evaluate_one", lambda *a, **kw: pytest.fail("no budget reservation available")
    )
    out = tmp_path / "campaign"
    assert (
        se.main(
            [
                "--out",
                str(out),
                "--brief-manifest",
                str(paths[0]),
                "--obligations",
                str(paths[1]),
                "--campaign-budget-usd",
                "0.5",
                "--parallel",
                "1",
                "--build-slots",
                "1",
                "--no-judge",
            ]
        )
        == 2
    )
    summary = json.loads((out / "summary.json").read_text())
    assert summary["product_denominator"] == 2
    assert summary["product_failure_families"] == {"campaign_budget_exhausted": 2}
    assert summary["campaign_complete"] is False
