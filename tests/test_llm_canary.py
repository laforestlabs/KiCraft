from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from kicraft.eval import llm_canary as canary
from kicraft.server.config import DESIGN_PROFILES, Settings
from kicraft.tuning.benchmark import BENCHMARK_PROMPTS


def _settings() -> Settings:
    profile = DESIGN_PROFILES["flash"]
    return Settings(
        api_key="secret",
        design_profile="flash",
        model=str(profile["model"]),
        provider_order=list(profile["provider_order"]),
        max_price_prompt=float(profile["max_price_prompt"]),
        max_price_completion=float(profile["max_price_completion"]),
    )


def _status(remaining=10.0):
    return {
        "daily_remaining_usd": remaining,
        "total_remaining_usd": remaining,
        "kill_switch": False,
    }


def test_fixed_cohort_order_archetypes_and_hashes():
    rows = canary._cohort()
    assert [row["slug"] for row in rows] == list(canary.COHORT)
    assert len({row["archetype"] for row in rows}) == 9
    by_slug = {entry["slug"]: entry for entry in BENCHMARK_PROMPTS}
    assert [row["brief_sha256"] for row in rows] == [
        hashlib.sha256(by_slug[slug]["brief"].encode()).hexdigest()
        for slug in canary.COHORT
    ]


def test_checkout_identity_rejects_every_dirty_path(monkeypatch):
    def fake_git(*args, text=True):
        if args == ("rev-parse", "HEAD"):
            return "abc123\n"
        if args == ("status", "--porcelain=v1", "-z"):
            return b" M kicraft/server/config.py\0"
        raise AssertionError(args)

    monkeypatch.setattr(canary, "_git", fake_git)
    with pytest.raises(RuntimeError, match="uncommitted runtime/config changes"):
        canary._checkout_identity()


def test_custom_or_mutated_designer_is_rejected():
    with pytest.raises(RuntimeError, match="named DESIGN_PROFILES"):
        canary._resolved_roles(Settings(api_key="x", design_profile="custom"))
    settings = _settings()
    settings.model = "mutable/latest"
    with pytest.raises(RuntimeError, match="do not match"):
        canary._resolved_roles(settings)


@pytest.mark.parametrize(
    "status, message",
    [
        ({**_status(), "kill_switch": True}, "kill switch"),
        (_status(0.69), "daily"),
        ({**_status(), "total_remaining_usd": 0.69}, "total"),
    ],
)
def test_campaign_headroom_refusal(status, message):
    with pytest.raises(RuntimeError, match=message):
        canary._require_headroom(status, canary.ENVELOPE_USD)


def test_occupied_slot_refuses_before_paid_preflight(tmp_path, monkeypatch):
    paid = []
    monkeypatch.setattr(canary.Settings, "from_env", classmethod(lambda cls: _settings()))
    monkeypatch.setattr(canary, "_manifest_identity", lambda settings, campaign_id: {"campaign_id": campaign_id})
    monkeypatch.setattr(canary, "probe_build_slots", lambda: [1])
    monkeypatch.setattr(canary, "preflight_role", lambda *a, **k: paid.append(True))
    assert canary._run_new(tmp_path / "batch") == 2
    assert paid == []
    manifest = json.loads((tmp_path / "batch" / "canary_manifest.json").read_text())
    assert manifest["run_status"] == "preflight_failed"


def test_sanitized_manifest_preflights_and_campaign_metadata(tmp_path, monkeypatch):
    settings = _settings()
    identity = {"campaign_id": "llm-canary-fixed", "envelope_usd": canary.ENVELOPE_USD}

    class Guard:
        def __init__(self, _settings):
            pass

        def status(self):
            return _status()

    def fake_preflight(_settings, *, role, model, meta_ctx):
        assert meta_ctx == {"campaign_id": "llm-canary-fixed"}
        return {
            "ok": True,
            "role": role,
            "model": model,
            "provider_order": list(_settings.provider_order),
            "reply_head": "forbidden",
            "api_key": "forbidden",
            "smoke": {"ok": True, "cost_usd": 0.01, "raw": "forbidden"},
        }

    monkeypatch.setattr(canary.Settings, "from_env", classmethod(lambda cls: settings))
    monkeypatch.setattr(canary.uuid, "uuid4", lambda: "fixed")
    monkeypatch.setattr(canary, "_manifest_identity", lambda *_: identity)
    monkeypatch.setattr(canary, "probe_build_slots", lambda: [])
    monkeypatch.setattr(canary, "SpendGuard", Guard)
    monkeypatch.setattr(canary, "preflight_role", fake_preflight)
    monkeypatch.setattr(canary, "_run_batch", lambda *a, **k: 0)
    batch = tmp_path / "batch"
    assert canary._run_new(batch) == 0
    manifest = json.loads((batch / "canary_manifest.json").read_text())
    assert manifest["run_status"] == "ready"
    assert manifest["immutable"] == identity
    for name in ("preflight-designer.json", "preflight-judge.json"):
        text = (batch / name).read_text()
        assert "forbidden" not in text and "api_key" not in text and "reply_head" not in text


def test_exact_batch_and_resume_argv(tmp_path):
    batch = tmp_path.resolve()
    common = [
        "--only",
        ",".join(canary.COHORT),
        "--out",
        str(batch),
        "--repeats",
        "1",
        "--parallel",
        "1",
        "--build-slots",
        "1",
        "--build-timeout",
        "2400",
    ]
    assert canary._batch_argv(batch, resume=False) == [
        sys.executable,
        "-m",
        "kicraft.eval.self_eval",
        *common,
    ]
    assert canary._batch_argv(batch, resume=True) == [
        sys.executable,
        "-m",
        "kicraft.eval.self_eval",
        "--resume",
        str(batch),
        *common,
    ]


def test_tee_preserves_terminal_and_log(tmp_path, capsys):
    log = tmp_path / "canary.log"
    rc = canary._tee_subprocess([sys.executable, "-c", "print('tee-canary')"], log)
    assert rc == 0
    assert "tee-canary" in capsys.readouterr().out
    assert log.read_text() == "tee-canary\n"


def test_resume_rejects_identity_and_cohort_drift(tmp_path, monkeypatch):
    settings = _settings()
    expected = {
        "campaign_id": "c1",
        "checkout": {"commit": "abc"},
        "designer": {"model": settings.model},
        "judge": {"model": settings.eval_judge_model},
        "envelope_usd": canary.ENVELOPE_USD,
    }
    (tmp_path / "canary_manifest.json").write_text(
        json.dumps({"immutable": {**expected, "checkout": {"commit": "drift"}}})
    )
    (tmp_path / "campaign_manifest.json").write_text(json.dumps({"immutable": {}}))
    monkeypatch.setattr(canary, "_manifest_identity", lambda *_: expected)
    with pytest.raises(RuntimeError, match="frozen identity"):
        canary._validate_resume(tmp_path, settings)


def test_partial_harness_failure_updates_status_without_replacing_identity(tmp_path, monkeypatch):
    identity = {"campaign_id": "c1", "frozen": True}
    manifest = tmp_path / "canary_manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1, "immutable": identity, "run_status": "ready"}))
    settings = SimpleNamespace(api_key="secret", ledger_path=tmp_path / "ledger.db")
    monkeypatch.setattr(canary, "probe_build_slots", lambda: [])
    monkeypatch.setattr(canary, "_tee_subprocess", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    assert canary._run_batch(tmp_path, manifest, resume=False, settings=settings) == 2
    saved = json.loads(manifest.read_text())
    assert saved["immutable"] == identity
    assert saved["run_status"] == "batch_failed"
    assert saved["operational_error"] == {"kind": "harness", "message": "boom"}
