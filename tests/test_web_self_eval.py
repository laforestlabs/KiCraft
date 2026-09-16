"""Offline tests for the admin web self-evaluation (kicraft.eval).

Everything here runs without a network or an API key: the Class-J judge is driven
by a fake client returning scripted text. Covers the web Class-C collector, the
judge's parse/repair/fail-closed behavior, the end-to-end evaluate_project driver,
and the admin tier gate.
"""
from __future__ import annotations

import json
from pathlib import Path

from kicraft.eval import load_rubric
from kicraft.eval.judge import grade_class_j
from kicraft.eval.metrics_web import collect_web_metrics
from kicraft.eval.run_web import build_run_digest, evaluate_project
from kicraft.eval.scoring import eval_script_gates, score_class_c_dims


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _make_project(root, *, retries=0, questions=0, erc_errors=0, erc_warnings=0,
                  synth_status="ok", synthesized=True, with_build=True,
                  with_brief=True):
    """Build a finished-looking web project dir under `root` (named '123' so the
    run_id prefix is well-formed) and return its path."""
    base = root / "123"
    gen = base / "generated" / "DEMO"
    gen.mkdir(parents=True)

    if with_brief:
        (base / "brief.txt").write_text("A 3V3 USB temperature logger with a BMP280 over I2C.")

    state = {
        "intent": {"summary": "USB temp logger", "constraints": ["3V3", "USB-C", "SMT"]},
        "functional_spec": {"blocks": ["usb", "ldo", "mcu", "sensor"]},
        "architecture": {"topology": "USB-C -> LDO 3V3 -> MCU + BMP280 (I2C)"},
        "bom": {"parts": [{"ref": "U1", "mpn": "AP2112K-3.3"}, {"ref": "U2", "mpn": "BMP280"}],
                "connections": [{"a": "U1.VOUT", "b": "U2.VDD"}], "no_connect_pins": []},
        "assumptions": ["I2C address 0x76 (defaulted)"],
        "open_questions": [],
        "history": [{"stage": "intent", "timestamp": "2026-06-05T10:00:00+00:00"}],
        "project_stem": "DEMO",
    }
    (base / "state.json").write_text(json.dumps(state))

    if synthesized:
        (gen / "DEMO.kicad_sch").write_text("(kicad_sch)")
        (gen / "DEMO.kicad_pcb").write_text("(kicad_pcb)")
    (gen / "synthesis_check.json").write_text(json.dumps(
        {"status": synth_status, "checks": [], "checked_at": "2026-06-05T10:06:00+00:00"}))
    violations = ([{"severity": "error"}] * erc_errors) + ([{"severity": "warning"}] * erc_warnings)
    (gen / "DEMO_erc.rpt").write_text(json.dumps({"sheets": [{"violations": violations}]}))

    events = [{"kind": "stage_done", "stage": s, "ok": True, "attempts": 1}
              for s in ("intent", "functional_spec", "architecture", "bom", "wiring")]
    events += [{"kind": "retry", "stage": "bom", "errors": ["e"]} for _ in range(retries)]
    events += [{"kind": "question", "stage": "intent", "questions": [{"text": "?"}]}
               for _ in range(questions)]
    if with_build:
        events += [{"kind": "build_start"},
                   {"kind": "build_log", "text": "ok"},
                   {"kind": "build_done", "ok": synthesized}]
    with (base / "events.jsonl").open("w") as f:
        for e in events:
            f.write(json.dumps(e) + "\n")
    return base


def _digest_metrics():
    return {
        "synth": {"status": "ok", "failed_checks": []},
        "erc": {"errors": 0, "warnings": 0},
        "transcript": {"failed_commits": 0, "ask_questions": 0, "crashes": 0},
        "generated": {"synthesized": True, "pcb": 1, "sch": 1},
    }


def test_frozen_r2r_dac_digest_retains_c1_supply_nets():
    run_dir = (
        Path(__file__).resolve().parents[1]
        / "logs/self_eval/20260915T132650Z/run_02_r2r-dac"
    )
    digest = build_run_digest(run_dir, _digest_metrics())
    assert "C1: value='100nF'" in digest
    assert "+5V [sheet='BUFFER']: U1.5, C1.1" in digest
    assert "GND [sheet='BUFFER']: U1.2, C1.2" in digest


def test_run_digest_no_bom_is_incomplete_not_an_mcu_claim(tmp_path):
    project = tmp_path / "no-bom"
    project.mkdir()
    (project / "brief.txt").write_text("Controller board")
    (project / "state.json").write_text(json.dumps({"intent": {"goal": "Controller board"}}))
    digest = build_run_digest(project, _digest_metrics())
    assert "BOM / PINS / NETS — INCOMPLETE" in digest
    assert "do not claim a missing or unprogrammable delivered MCU" in digest


class FakeClient:
    """Returns scripted replies; records the meta_ctx of the last call."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.calls = 0
        self.last_meta = None

    def chat(self, messages, **kw):
        text = self.replies[min(self.calls, len(self.replies) - 1)]
        self.calls += 1
        self.last_meta = kw.get("meta_ctx")
        return {"text": text, "cost_usd": 0.001}


def _verdict(level=3, gates=None):
    jids = [d["id"] for d in load_rubric()["dimensions"] if d["class"] == "J"]
    return json.dumps({"dimensions": {i: {"level": level, "evidence": "ev"} for i in jids},
                       "triggered_gates": gates or []})


# --------------------------------------------------------------------------- #
# Class-C web collector
# --------------------------------------------------------------------------- #
def test_collect_web_metrics_clean_run(tmp_path):
    base = _make_project(tmp_path)
    m = collect_web_metrics(base)
    rub = load_rubric()
    dims = score_class_c_dims(m, rub)

    assert m["transcript"]["present"] is True
    assert m["transcript"]["failed_commits"] == 0
    assert dims["pipeline_completion"]["level"] == 4
    assert dims["computing_error_cleanliness"]["level"] == 4
    # clean convergence is authoritative (not partial) on the web event stream
    assert dims["convergence_efficiency"]["level"] == 4
    assert dims["convergence_efficiency"]["partial"] is False
    assert eval_script_gates(m, rub) == []


def test_collect_web_metrics_distinguishes_stage_subprocess_crash(tmp_path):
    from kicraft.eval.metrics_web import analyze_events

    events = tmp_path / "events.jsonl"
    events.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"kind": "stage_start", "stage": "wiring"},
                {
                    "kind": "stage_done",
                    "stage": "wiring",
                    "ok": False,
                    "failure_kind": "commit_process_failed",
                },
                {"kind": "build_log", "text": "Traceback (most recent call last)"},
            ]
        )
    )
    transcript = analyze_events(events)
    assert transcript["stage_subprocess_crashes"] == 1
    assert transcript["build_crashes"] == 1
    assert transcript["crashes"] == 2


def test_token_metrics_distinguish_exact_run_from_project_aggregation(tmp_path):
    from kicraft.server.config import Settings
    from kicraft.server.spend_guard import SpendGuard

    base = _make_project(tmp_path)
    guard = SpendGuard(Settings(api_key="test", ledger_path=tmp_path / "ledger.db"))
    guard.record("design", 10, 2, 0.01, {"run_id": "p123-current", "stage": "intent"})
    guard.record("design", 20, 3, 0.02, {"run_id": "p123-current", "stage": "bom"})
    guard.record("sibling", 100, 20, 0.10, {"run_id": "p123-current-other"})
    guard.record("history", 200, 30, 0.20, {"run_id": "p123-previous"})
    guard.record("unrelated", 1000, 200, 1.0, {"run_id": "p1234-current"})

    report = evaluate_project(
        base, None, ledger_path=guard.path, run_id="p123-current", skip_judge=True,
    )
    assert report["run_id"] == "p123-current"
    assert report["metrics"]["token_usage"] == {
        "input_tokens": 30, "output_tokens": 5, "total_tokens": 35,
        "turns": 2, "estimated_cost_usd": 0.03,
        "cost_known": True, "by_model": {"design": 2},
    }
    # Explicit exact identity wins even if the caller also supplies a broad prefix.
    exact = collect_web_metrics(
        base, ledger_path=guard.path, run_id="p123-current", run_id_prefix="p123",
    )
    assert exact["token_usage"] == report["metrics"]["token_usage"]
    # A missing exact run must not fall back to the project's historical spend.
    missing = collect_web_metrics(base, ledger_path=guard.path, run_id="p123-absent")
    assert missing["token_usage"] is None

    project = evaluate_project(base, None, ledger_path=guard.path, skip_judge=True)
    assert project["metrics"]["token_usage"] == {
        "input_tokens": 330, "output_tokens": 55, "total_tokens": 385,
        "turns": 4, "estimated_cost_usd": 0.33,
        "cost_known": True, "by_model": {"design": 2, "sibling": 1, "history": 1},
    }
    custom = collect_web_metrics(base, ledger_path=guard.path, run_id_prefix="p1234-")
    assert custom["token_usage"]["total_tokens"] == 1200
    assert custom["token_usage"]["estimated_cost_usd"] == 1.0


def test_collect_web_metrics_convergence_penalty(tmp_path):
    base = _make_project(tmp_path, retries=3)
    m = collect_web_metrics(base)
    dims = score_class_c_dims(m, load_rubric())
    # 3 error-driven re-commits -> convergence level 1, authoritatively
    assert m["transcript"]["failed_commits"] == 3
    assert dims["convergence_efficiency"]["level"] == 1
    assert dims["convergence_efficiency"]["partial"] is False


def test_collect_web_metrics_erc_gate(tmp_path):
    base = _make_project(tmp_path, erc_errors=2)
    m = collect_web_metrics(base)
    rub = load_rubric()
    dims = score_class_c_dims(m, rub)
    gates = eval_script_gates(m, rub)
    assert dims["computing_error_cleanliness"]["level"] == 1  # 1-10 ERC errors
    assert any(g["id"] == "erc_errors" and g["cap"] == 45 for g in gates)


def test_collect_web_metrics_synthesis_broken_gate(tmp_path):
    # build attempted but no KiCad files produced -> synthesis_broken gate
    base = _make_project(tmp_path, synthesized=False)
    m = collect_web_metrics(base)
    rub = load_rubric()
    gates = eval_script_gates(m, rub)
    assert m["transcript"]["synth_attempts"] == 1
    assert any(g["id"] == "synthesis_broken" for g in gates)


def test_friction_scores_against_the_web_band(tmp_path):
    # 2026-07-19 review §8.1: with band=None the axis was a hardcoded 3 for
    # every run. Web runs now carry a deterministic (0, 2) band (curated
    # self-eval briefs are answerable without clarification), so the axis is
    # a real signal: quiet run -> 4, interrogating a complete brief -> down.
    base = _make_project(tmp_path, questions=1)
    m = collect_web_metrics(base)
    dims = score_class_c_dims(m, load_rubric())
    assert m["expected_question_band"] == (0, 2)
    assert dims["interaction_friction"]["partial"] is False
    assert dims["interaction_friction"]["level"] == 4  # in band, zero excess
    assert m["perm"]["excess"] == 0


def test_friction_penalizes_interrogating_a_complete_brief(tmp_path):
    base = _make_project(tmp_path, questions=7)
    m = collect_web_metrics(base)
    dims = score_class_c_dims(m, load_rubric())
    assert dims["interaction_friction"]["level"] <= 1  # far out of band


# --------------------------------------------------------------------------- #
# Class-J judge
# --------------------------------------------------------------------------- #
def test_judge_valid_verdict_and_gate(tmp_path):
    rub = load_rubric()
    client = FakeClient(_verdict(level=2, gates=[{"id": "unprogrammable_mcu", "evidence": "no SWD/USB"}]))
    out = grade_class_j(client, "DIGEST", rub, model="x", meta_ctx={"run_id": "p1-7"})
    assert out["ok"] and client.calls == 1
    assert all(v["level"] == 2 for v in out["dimensions"].values())
    assert out["gates"] == [{"id": "unprogrammable_mcu", "cap": 50, "by": "observer", "why": "no SWD/USB"}]
    # judge call is tagged for the ledger
    assert client.last_meta["phase"] == "eval_judge"
    assert client.last_meta["run_id"] == "p1-7"
    assert client.last_meta["stage"] == "judge"
    assert client.last_meta["attempt"] == 0


def test_judge_repairs_after_malformed(tmp_path):
    rub = load_rubric()
    client = FakeClient("here are my thoughts, no json", _verdict(level=3))
    out = grade_class_j(client, "DIGEST", rub, model="x")
    assert out["ok"] and client.calls == 2


def test_judge_fails_closed(tmp_path):
    rub = load_rubric()
    client = FakeClient("nope", "still nope")
    out = grade_class_j(client, "DIGEST", rub, model="x")
    assert out["ok"] is False
    assert all(v["level"] is None for v in out["dimensions"].values())
    assert out["error"]


def test_judge_rejects_out_of_range_level(tmp_path):
    rub = load_rubric()
    jids = [d["id"] for d in rub["dimensions"] if d["class"] == "J"]
    bad = json.dumps({"dimensions": {i: {"level": 9, "evidence": "x"} for i in jids}})
    out = grade_class_j(FakeClient(bad, bad), "DIGEST", rub, model="x")
    assert out["ok"] is False


# --------------------------------------------------------------------------- #
# evaluate_project end to end
# --------------------------------------------------------------------------- #
def _assert_report_shape(report):
    for key in ("scenario", "run_id", "rubric_version", "rubric_sha256",
                "metrics", "dimensions", "gates", "score"):
        assert key in report, key
    assert len(report["dimensions"]) == 11
    for v in report["dimensions"].values():
        assert v["level"] is None or (isinstance(v["level"], int) and 0 <= v["level"] <= 4)


def test_evaluate_project_full(tmp_path):
    base = _make_project(tmp_path)
    client = FakeClient(_verdict(level=4))
    report = evaluate_project(base, client, judge_model="judge-x", run_id="p123-9")
    _assert_report_shape(report)
    assert report["judge"]["ran"] and report["judge"]["ok"]
    assert client.last_meta["run_id"] == "p123-9"
    # all 11 dims graded -> finalized with a numeric grade
    assert all(v["level"] is not None for v in report["dimensions"].values())
    assert report["score"]["grade"] in ("A", "B", "C", "D", "F")
    assert report["score"]["final"] is not None
    # persisted + round-trips
    saved = json.loads((base / "eval" / "report.json").read_text())
    assert saved["score"]["final"] == report["score"]["final"]


def test_evaluate_project_skip_judge_is_class_c_only(tmp_path):
    base = _make_project(tmp_path)
    report = evaluate_project(base, None, skip_judge=True)
    _assert_report_shape(report)
    assert report["judge"]["ran"] is False
    assert report["score"]["final"] is None
    assert report["dimensions"]["electrical_soundness"]["level"] is None
    assert "judge skipped" in report["score"]["note"]


def test_evaluate_project_judge_failure_withholds_grade(tmp_path):
    base = _make_project(tmp_path)
    report = evaluate_project(base, FakeClient("garbage", "garbage"), judge_model="judge-x")
    assert report["judge"]["ran"] and report["judge"]["ok"] is False
    assert report["score"]["final"] is None  # not finalized on a failed judge
    # Class-C dims still scored
    assert report["dimensions"]["pipeline_completion"]["level"] is not None
    assert "judge failed" in report["score"]["note"]


def test_evaluate_project_erc_gate_caps_grade(tmp_path):
    base = _make_project(tmp_path, erc_errors=3)
    report = evaluate_project(base, FakeClient(_verdict(level=4)), judge_model="judge-x")
    caps = [g["cap"] for g in report["gates"]["triggered"]]
    assert 45 in caps  # erc_errors gate
    assert report["score"]["final"] <= 45


# --------------------------------------------------------------------------- #
# admin role gate
# --------------------------------------------------------------------------- #
def test_admin_role_and_is_admin(tmp_path):
    from kicraft.server.accounts import TIERS, AccountStore, is_admin

    assert "admin" not in TIERS  # admin is a role now, orthogonal to the billing tier
    store = AccountStore(tmp_path / "accounts.db", tmp_path / "projects")
    u = store.create_user("a@example.com", "pw")
    assert not is_admin(u)
    promoted = store.set_role("a@example.com", "admin")  # grant the role, not a tier
    assert promoted.role == "admin" and is_admin(promoted)

    other = store.create_user("b@example.com", "pw")
    assert is_admin(other) is False
    assert is_admin(None) is False


def test_rubric_hash_is_stable_and_verifies():
    # load_rubric(verify=True) raises on a stale stored hash; reaching here means
    # the moved rubric still matches its stamp.
    rub = load_rubric()
    assert rub["meta"]["sha256"] == rub["_computed_sha256"]
    assert len(rub["dimensions"]) == 11


# --------------------------------------------------------------------------- #
# Layer-4 review outcome parsed from build_log events (admin dashboard)
# --------------------------------------------------------------------------- #
def _bl(text):
    return json.dumps({"kind": "build_log", "text": text})


def test_build_review_outcome_blocked():
    from kicraft.server.routes_admin import _build_review_outcome
    lines = [
        _bl("[build] 4/5 verify: shorts=0 unconnected=0"),
        _bl("[build]     review BLOCKER: [clock] oscillator pins shorted to GND"),
        _bl("[build]     kept board X.kicad_pcb for inspection (no fab package; "
            "electrical review found a blocker)"),
    ]
    r = _build_review_outcome(lines)
    assert r["status"] == "blocked" and r["n"] == 1
    assert "oscillator" in r["blockers"][0]


def test_build_review_outcome_passed():
    from kicraft.server.routes_admin import _build_review_outcome
    r = _build_review_outcome([_bl("[build] 4/5 electrical review: 2 non-blocking finding(s), cost $0.0010")])
    assert r["status"] == "passed" and r["n"] == 2 and r["blockers"] == []


def test_build_review_outcome_none_when_gate_not_reached():
    # A board that failed DRC/route never reaches the review gate -> no markers.
    from kicraft.server.routes_admin import _build_review_outcome
    assert _build_review_outcome([_bl("[build] 4/5 verify: shorts=3 unconnected=5")]) is None
    assert _build_review_outcome([]) is None


def test_build_review_outcome_skipped():
    from kicraft.server.routes_admin import _build_review_outcome
    r = _build_review_outcome([_bl("[build] electrical review skipped (SystemExit: no key)")])
    assert r["status"] == "skipped"
