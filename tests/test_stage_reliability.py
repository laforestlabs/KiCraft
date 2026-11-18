from pathlib import Path
import json

from kicraft.eval.stage_reliability import (
    exact_zero_failure_lower_bound,
    load_corpus,
    statistical_release_gate,
)


def test_committed_corpus_has_full_stage_denominators():
    corpus = load_corpus(Path("kicraft/eval/stage_reliability_corpus.json"))
    counts = {stage: 0 for stage in ("intent", "functional_spec", "architecture", "bom", "wiring")}
    for case in corpus["cases"]:
        counts[case["stage"]] += 1
    assert set(counts.values()) == {306}


def test_exact_zero_failure_bound_and_release_gate():
    assert exact_zero_failure_lower_bound(299) > 0.99
    rows = [
        {
            "commit_ok": True,
            "semantic_clean": True,
            "cost_usd": 1.0,
            "wall_s": 1.0,
            "valid": True,
        }
        for _ in range(299)
    ]
    result = statistical_release_gate(rows, baseline_p95_cost=1.0, baseline_p95_wall_s=1.0)
    assert result["passed"]
    rows[0]["semantic_clean"] = False
    assert not statistical_release_gate(rows)["gates"]["semantic_reliability"]


def test_recovery_baseline_fixture_covers_each_failure_mechanism_without_reasoning():
    fixture = json.loads(
        Path("tests/fixtures/stage_reliability/recovery_baseline_20260910.json").read_text(
            encoding="utf-8"
        )
    )
    assert fixture["privacy"] == {
        "contains_reasoning": False,
        "contains_credentials": False,
        "contains_binary_artifacts": False,
    }
    assert {case["mechanism"] for case in fixture["cases"]} == {
        "functional_spec_collection_overflow",
        "architecture_invalid_schema",
        "architecture_collection_overflow",
        "unresolved_footprint",
        "unresolved_symbol",
        "symbol_pad_mismatch",
        "absent_mpn_sentinel",
        "named_part_omission",
        "recipe_duplicate",
        "protected_identity_collision",
        "bom_partial_wiring_coverage",
        "wiring_pin_inventory",
        "aggregate_dangling_signal",
    }
    serialized = json.dumps(fixture).lower()
    assert "openrouter_api_key" not in serialized
    assert "reasoning_delta" not in serialized
    assert "answer_delta" not in serialized


def test_surprise_freeze_is_iid_and_records_repeated_draws(tmp_path, monkeypatch):
    import random
    from types import SimpleNamespace
    from kicraft.eval import surprise_campaign as campaign

    settings = SimpleNamespace(project_llm_budget_usd=.6, daily_usd_ceiling=20, total_usd_ceiling=250)
    monkeypatch.setattr(campaign, "_identity", lambda _: {"settings": {"project_llm_budget_usd": .6}})
    monkeypatch.setattr(campaign, "freeze_smoke_cohort", lambda: [])
    monkeypatch.setattr(campaign, "BRIEF_SEED_MODULO", 2)
    first = campaign.freeze_campaign(tmp_path / "one", count=10, settings=settings)
    second = campaign.freeze_campaign(tmp_path / "two", count=10, settings=settings)
    rng = random.Random(20261118)
    assert [t["seed"] for t in first["trials"]] == [rng.randrange(2) for _ in range(10)]
    assert first["trials"] == second["trials"]
    assert len({t["id"] for t in first["trials"]}) == 10
    assert len({t["seed"] for t in first["trials"]}) < 10


def test_surprise_budget_reserves_preparation_and_inflight(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kicraft.eval import surprise_campaign as campaign

    settings = SimpleNamespace(project_llm_budget_usd=.6, daily_usd_ceiling=20, total_usd_ceiling=250)
    monkeypatch.setattr(campaign, "_identity", lambda _: {"settings": {"project_llm_budget_usd": .6}})
    monkeypatch.setattr(campaign, "freeze_smoke_cohort", lambda: [])
    campaign.freeze_campaign(tmp_path, count=3, budget_usd=1.0, settings=settings)
    assert campaign.reserve_spend(tmp_path, "preparation-1", .6)
    assert not campaign.reserve_spend(tmp_path, "trial-000", .6, kind="trial")
    campaign.settle_spend(tmp_path, "preparation-1", .2)
    assert campaign.reserve_spend(tmp_path, "trial-000", .6, kind="trial")
    assert not campaign.reserve_spend(tmp_path, "trial-001", .6, kind="trial")


def test_surprise_report_keeps_missing_and_failed_denominator():
    from kicraft.eval.surprise_campaign import campaign_report

    trials = [{"id": str(i), "family": "one"} for i in range(300)]
    rows = {t["id"]: {"status": "settled", "kind": "trial", "amount_usd": 0,
                      "result": {"id": t["id"], "family": "one", "complete_path": True, "semantic_clean": True}}
            for t in trials}
    assert campaign_report({"trials": trials}, {"entries": rows})["release_passed"]
    rows.pop("0")
    rows["1"]["result"]["complete_path"] = False
    report = campaign_report({"trials": trials}, {"entries": rows})
    assert report["count"] == 300
    assert report["complete_path_passes"] == 298
    assert not report["release_passed"]


def test_surprise_resume_rejects_identity_drift(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from kicraft.eval import surprise_campaign as campaign

    settings = SimpleNamespace(project_llm_budget_usd=.6, daily_usd_ceiling=20, total_usd_ceiling=250)
    monkeypatch.setattr(campaign, "_identity", lambda _: {"settings": {"project_llm_budget_usd": .6}})
    monkeypatch.setattr(campaign, "freeze_smoke_cohort", lambda: [])
    campaign.freeze_campaign(tmp_path, count=1, settings=settings)
    monkeypatch.setattr(campaign, "_identity", lambda _: {"changed": True})
    with pytest.raises(ValueError, match="identity drift"):
        campaign.resume_campaign(tmp_path, settings=settings)


def test_smoke_cohort_uses_generator_families():
    from kicraft.eval.surprise_campaign import freeze_smoke_cohort, _family

    cohort = freeze_smoke_cohort(["original 82", "original 83", "original 84"])
    assert [row["seed"] for row in cohort[:3]] == [82, 83, 84]
    assert all(row["family"] == _family(row["seed"]) for row in cohort)
    assert len({row["family"] for row in cohort}) == 5


def test_surprise_resume_keeps_interrupted_trials_and_runs_only_unstarted(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kicraft.eval import surprise_campaign as campaign

    settings = SimpleNamespace(project_llm_budget_usd=.6, daily_usd_ceiling=20,
                               total_usd_ceiling=250, ledger_path=tmp_path / "spend.db")
    identity = {"settings": {"project_llm_budget_usd": .6}, "core_defaults": None}
    monkeypatch.setattr(campaign, "_identity", lambda _: identity)
    monkeypatch.setattr(campaign, "freeze_smoke_cohort", lambda: [])
    campaign.freeze_campaign(tmp_path, count=2, budget_usd=2, settings=settings)
    campaign.reserve_spend(tmp_path, "trial-000", .6, kind="trial")
    called = []

    def run(out, trial, defaults, ledger_path, *, settings=None):
        called.append(trial["id"])
        return {"id": trial["id"], "family": trial["family"], "status": "finished",
                "cost_usd": .1, "cost_receipted": True, "complete_path": True, "semantic_clean": True}

    monkeypatch.setattr(campaign, "_run_trial", run)
    report = campaign.resume_campaign(tmp_path, settings=settings)
    assert called == ["trial-001"]
    assert report["count"] == 2
    assert report["complete_path_passes"] == 1
    assert report["failures"]["interrupted"] == 1
    campaign.resume_campaign(tmp_path, settings=settings)
    assert called == ["trial-001"]


def test_surprise_carryover_seals_and_preserves_exposure(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from kicraft.eval import surprise_campaign as campaign

    settings = SimpleNamespace(project_llm_budget_usd=.6, daily_usd_ceiling=20, total_usd_ceiling=250)
    monkeypatch.setattr(campaign, "_identity", lambda _: {"settings": {"project_llm_budget_usd": .6}})
    monkeypatch.setattr(campaign, "freeze_smoke_cohort", lambda: [])
    old, new = tmp_path / "old", tmp_path / "new"
    campaign.freeze_campaign(old, count=1, budget_usd=1, settings=settings)
    assert campaign.reserve_spend(old, "prep", .6)
    campaign.settle_spend(old, "prep", .2)
    manifest = campaign.freeze_campaign(new, count=1, budget_usd=1, settings=settings, carryover=old)
    ledger = json.loads((new / "ledger.json").read_text())
    assert manifest["carryover"]
    assert campaign.campaign_report(manifest, ledger)["campaign_exposure_usd"] == .2
    carried = next(iter(ledger["entries"].values()))
    assert carried["source_entry_id"] == "prep"
    assert carried["source_entry"]["amount_usd"] == .2
    assert (new / "trials" / "trial-000" / "brief.txt").read_text() == manifest["trials"][0]["brief"]
    with pytest.raises(ValueError):
        campaign.reserve_spend(old, "later", .6)
    assert campaign.reserve_spend(new, "trial-000", .6, kind="trial")
    assert not campaign.reserve_spend(new, "prep2", .6)


def test_surprise_resume_rejects_endpoint_drift_without_publishing_credentials(tmp_path, monkeypatch):
    from dataclasses import replace
    import pytest
    from kicraft.eval import surprise_campaign as campaign
    from kicraft.server.config import Settings

    settings = Settings(
        api_key="private-design-key-sentinel", deepseek_api_key="private-second-key-sentinel",
        enable_core_defaults=False, daily_usd_ceiling=20, total_usd_ceiling=250,
        project_llm_budget_usd=.6,
    )
    monkeypatch.setattr(campaign, "freeze_smoke_cohort", lambda: [])
    campaign.freeze_campaign(tmp_path, count=1, settings=settings)
    saved = (tmp_path / "manifest.json").read_text()
    assert settings.api_key not in saved and settings.deepseek_api_key not in saved
    with pytest.raises(ValueError, match="identity drift"):
        campaign.resume_campaign(
            tmp_path, settings=replace(settings, base_url="https://different.invalid/api/v1"),
        )


def test_surprise_missing_post_cannot_be_hidden_by_duplicate_outcomes():
    from kicraft.eval.surprise_campaign import _attempts_cover_receipts

    first = {"stage": "architecture", "attempt": 1, "call_mode": "normal", "model": "designer"}
    second = {**first, "attempt": 2, "call_mode": "draft_audit", "model": "auditor"}
    calls = [{"model": row["model"], "meta": row} for row in (first, second)]
    assert not _attempts_cover_receipts(calls, [second, second])
    assert not _attempts_cover_receipts(calls, [first, {**second, "model": "designer"}])
    assert _attempts_cover_receipts(calls, [first, second])
    calls.append({"model": "auditor", "meta": {"phase": "decision_not_dispatched"}})
    assert _attempts_cover_receipts(calls, [first, second])


def test_surprise_release_requires_clean_semantics_even_after_commit():
    from kicraft.eval.surprise_campaign import campaign_report

    trials = [{"id": str(i), "family": "one"} for i in range(300)]
    entries = {
        trial["id"]: {
            "status": "settled", "kind": "trial", "amount_usd": 0,
            "result": {"id": trial["id"], "family": "one", "complete_path": True,
                       "semantic_clean": True, "architecture_attempts": 1},
        }
        for trial in trials
    }
    for semantic_clean, reason in (
        (False, "semantic_rejected"), (None, "missing_semantic_evidence"),
    ):
        entries["0"]["result"]["semantic_clean"] = semantic_clean
        report = campaign_report({"trials": trials}, {"entries": entries})
        assert report["count"] == 300
        assert report["complete_path_passes"] == report["first_call_success"] == 299
        assert report["failures"] == {reason: 1}
        assert not report["release_passed"]
        assert report["zero_failure_lower_bound"] is None


def test_surprise_trial_bounds_known_receipts_and_unsettled_posts(tmp_path, monkeypatch):
    from kicraft.eval import surprise_campaign as campaign
    from kicraft.server.config import Settings
    from kicraft.server.spend_guard import SpendGuard

    settings = Settings(api_key="test", ledger_path=tmp_path / "spend.db", project_llm_budget_usd=.6)
    guard = SpendGuard(settings)
    monkeypatch.setattr(campaign, "make_client", lambda _: None)

    def run(workspace, brief, stages, *, run_id, **kwargs):
        reservation = guard.reserve(.2, run_id)
        guard.settle_reservation(reservation, "model", 1, 1, .07, meta={"run_id": run_id})
        uncertain = guard.reserve(.12, run_id)
        guard.mark_uncertain(uncertain)
        return {"status": "failed"}

    monkeypatch.setattr(campaign, "run_session", run)
    brief = "unchanged brief"
    import hashlib
    trial = {"id": "trial-000", "family": "one", "brief": brief,
             "brief_sha256": hashlib.sha256(brief.encode()).hexdigest()}
    path = tmp_path / "trials" / trial["id"]
    path.mkdir(parents=True)
    (path / "brief.txt").write_text(brief)
    row = campaign._run_trial(tmp_path, trial, None, settings.ledger_path, settings=settings)
    assert row["cost_usd"] == .07
    assert abs(row["outstanding_bound_usd"] - .19) < 1e-9
    assert not row["cost_receipted"]
    assert not row["complete_path"]


def test_surprise_ended_bounds_release_headroom_but_missing_bounds_do_not(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kicraft.eval import surprise_campaign as campaign

    settings = SimpleNamespace(project_llm_budget_usd=.6, daily_usd_ceiling=20,
                               total_usd_ceiling=250, ledger_path=tmp_path / "spend.db")
    identity = {"settings": {"project_llm_budget_usd": .6}, "core_defaults": None}
    monkeypatch.setattr(campaign, "_identity", lambda _: identity)
    monkeypatch.setattr(campaign, "freeze_smoke_cohort", lambda: [])
    campaign.freeze_campaign(tmp_path, count=3, parallel=1, budget_usd=.85, settings=settings)
    seen = []

    def run(out, trial, defaults, ledger_path, *, settings=None):
        seen.append(trial["id"])
        return {"id": trial["id"], "family": trial["family"], "status": "finished",
                "cost_usd": .07, "cost_receipted": False, "complete_path": True,
                "semantic_clean": True,
                "outstanding_bound_usd": .19 if len(seen) == 1 else None}

    monkeypatch.setattr(campaign, "_run_trial", run)
    report = campaign.resume_campaign(tmp_path, settings=settings)
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert seen == ["trial-000", "trial-001"]
    assert ledger["entries"]["trial-000"]["amount_usd"] == .19
    assert ledger["entries"]["trial-001"]["amount_usd"] == .6
    assert all(e["status"] == "reserved" for e in ledger["entries"].values())
    assert report["campaign_exposure_usd"] <= .85
    assert report["complete_path_passes"] == 0
    assert report["p50_cost_usd"] is None


def test_surprise_campaign_limit_and_active_holds(tmp_path, monkeypatch):
    from types import SimpleNamespace
    import pytest
    from kicraft.eval import surprise_campaign as campaign

    settings = SimpleNamespace(project_llm_budget_usd=.6, daily_usd_ceiling=20, total_usd_ceiling=250)
    monkeypatch.setattr(campaign, "_identity", lambda _: {"settings": {"project_llm_budget_usd": .6}})
    monkeypatch.setattr(campaign, "freeze_smoke_cohort", lambda: [])
    manifest = campaign.freeze_campaign(tmp_path / "new", count=1, budget_usd=35, settings=settings)
    assert manifest["budget_usd"] == 35
    assert campaign.reserve_spend(tmp_path / "new", "active", .6, kind="trial")
    ledger = json.loads((tmp_path / "new" / "ledger.json").read_text())
    assert ledger["entries"]["active"]["amount_usd"] == .6
    with pytest.raises(ValueError):
        campaign.freeze_campaign(tmp_path / "too-large", count=1, budget_usd=35.01, settings=settings)
