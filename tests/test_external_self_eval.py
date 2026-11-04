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


@pytest.mark.parametrize(
    ("free", "remaining", "initial", "allowed"),
    [
        (8 * 1024**3, 2 * 1024**3, True, True),
        (8 * 1024**3 - 1, 2 * 1024**3, True, False),
        (6 * 1024**3, 1024**3, False, True),
        (6 * 1024**3 - 1, 1024**3, False, False),
        (7 * 1024**3, 0, True, False),
    ],
)
def test_storage_admission_protects_reserved_writes(
    tmp_path, monkeypatch, free, remaining, initial, allowed
):
    from types import SimpleNamespace
    import errno

    monkeypatch.delenv("KICRAFT_EVAL_INITIAL_STORAGE_RECEIPT", raising=False)
    monkeypatch.setenv("KICRAFT_EVAL_RESERVED_WRITE_BYTES", str(remaining))
    monkeypatch.setattr(se.shutil, "disk_usage", lambda _: SimpleNamespace(free=free))
    if allowed:
        se._require_storage(tmp_path, initial=initial)
    else:
        with pytest.raises(OSError) as error:
            se._require_storage(tmp_path, initial=initial)
        assert error.value.errno == errno.ENOSPC

def test_storage_receipt_allows_continuation_but_preserves_floor(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import errno

    receipt = tmp_path / "initial.json"
    receipt.write_text(json.dumps({
        "initial_numeric_gate_met": True,
        "free_after_bytes": 8 * 1024**3,
        "initial_minimum_bytes": 8 * 1024**3,
        "production_floor_bytes": 4 * 1024**3,
    }))
    original = receipt.read_bytes()
    monkeypatch.setenv("KICRAFT_EVAL_INITIAL_STORAGE_RECEIPT", str(receipt))
    monkeypatch.setenv("KICRAFT_EVAL_RESERVED_WRITE_BYTES", str(1024**3))
    monkeypatch.setattr(se.shutil, "disk_usage", lambda _: SimpleNamespace(free=6 * 1024**3))
    se._require_storage(tmp_path, initial=True)
    monkeypatch.setattr(se.shutil, "disk_usage", lambda _: SimpleNamespace(free=6 * 1024**3 - 1))
    with pytest.raises(OSError) as error:
        se._require_storage(tmp_path, initial=True)
    assert error.value.errno == errno.ENOSPC
    monkeypatch.setenv("KICRAFT_EVAL_RESERVED_WRITE_BYTES", "0")
    monkeypatch.setattr(se.shutil, "disk_usage", lambda _: SimpleNamespace(free=4 * 1024**3 - 1))
    with pytest.raises(OSError) as error:
        se._require_storage(tmp_path, initial=True)
    assert error.value.errno == errno.ENOSPC
    assert receipt.read_bytes() == original


@pytest.mark.parametrize("contents", [
    None, "not json", "[]",
    json.dumps({"initial_numeric_gate_met": False, "free_after_bytes": 8 * 1024**3,
                "initial_minimum_bytes": 8 * 1024**3, "production_floor_bytes": 4 * 1024**3}),
    json.dumps({"initial_numeric_gate_met": True, "free_after_bytes": 8 * 1024**3 - 1,
                "initial_minimum_bytes": 8 * 1024**3, "production_floor_bytes": 4 * 1024**3}),
    json.dumps({"initial_numeric_gate_met": True, "free_after_bytes": 8 * 1024**3,
                "initial_minimum_bytes": 8 * 1024**3 - 1, "production_floor_bytes": 4 * 1024**3}),
    json.dumps({"initial_numeric_gate_met": True, "free_after_bytes": 8 * 1024**3,
                "initial_minimum_bytes": 8 * 1024**3, "production_floor_bytes": 4 * 1024**3 - 1}),
    json.dumps({"initial_numeric_gate_met": True, "free_after_bytes": True,
                "initial_minimum_bytes": 8 * 1024**3, "production_floor_bytes": 4 * 1024**3}),
])
def test_storage_invalid_initial_receipt_refused(tmp_path, monkeypatch, contents):
    receipt = tmp_path / "initial.json"
    if contents is not None:
        receipt.write_text(contents)
    monkeypatch.setenv("KICRAFT_EVAL_INITIAL_STORAGE_RECEIPT", str(receipt))
    monkeypatch.setenv("KICRAFT_EVAL_RESERVED_WRITE_BYTES", "0")
    with pytest.raises(ValueError, match="initial storage receipt"):
        se._require_storage(tmp_path, initial=True)

def test_storage_receipt_on_another_filesystem_refused(tmp_path, monkeypatch):
    from types import SimpleNamespace

    receipt = tmp_path / "initial.json"
    receipt.write_text(json.dumps({
        "initial_numeric_gate_met": True,
        "free_after_bytes": 8 * 1024**3,
        "initial_minimum_bytes": 8 * 1024**3,
        "production_floor_bytes": 4 * 1024**3,
    }))
    monkeypatch.setenv("KICRAFT_EVAL_INITIAL_STORAGE_RECEIPT", str(receipt))
    monkeypatch.setenv("KICRAFT_EVAL_RESERVED_WRITE_BYTES", "0")
    original_stat = type(tmp_path).stat

    def stat(path, *args, **kwargs):
        if path == tmp_path:
            return SimpleNamespace(st_dev=receipt.stat().st_dev + 1)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(type(tmp_path), "stat", stat)
    with pytest.raises(ValueError, match="different filesystems"):
        se._require_storage(tmp_path, initial=True)



def test_external_storage_refusal_preserves_denominator_without_dispatch(tmp_path, monkeypatch):
    from types import SimpleNamespace

    paths = _inputs(tmp_path)
    _settings(tmp_path, monkeypatch)
    monkeypatch.setenv("KICRAFT_EVAL_RESERVED_WRITE_BYTES", str(1024**3))
    readings = iter([8 * 1024**3, 6 * 1024**3 - 1, 6 * 1024**3 - 1])
    monkeypatch.setattr(
        se.shutil, "disk_usage", lambda _: SimpleNamespace(free=next(readings))
    )
    monkeypatch.setattr(
        se, "evaluate_one", lambda *a, **kw: pytest.fail("storage reservation unavailable")
    )
    out = tmp_path / "campaign"
    assert se.main(
        [
            "--out", str(out),
            "--brief-manifest", str(paths[0]),
            "--obligations", str(paths[1]),
            "--campaign-budget-usd", "5",
            "--parallel", "1",
            "--build-slots", "1",
            "--no-judge",
        ]
    ) == 2
    summary = json.loads((out / "summary.json").read_text())
    assert summary["product_denominator"] == 2
    assert summary["product_failure_families"] == {"campaign_storage_exhausted": 2}
    assert summary["campaign_complete"] is False
    assert not (out / "run_01_alpha").exists()
    assert not (out / "run_02_beta").exists()
