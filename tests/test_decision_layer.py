"""Jev decisions through OpenRouter: the request shape, the parsing, and the confidence rule."""

from __future__ import annotations

import json

import pytest

from kicraft.server.decision_layer import (
    Answer,
    DecisionUnavailable,
    Question,
    by_key,
    decide,
    parse_answers,
)
from kicraft.server.config import Settings
from kicraft.server.spend_guard import SpendGuard

_CHOICE_PAYLOAD = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "department": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
            "confidence": 0.81,
        }
    },
    "usage": {"input_tokens": 318, "output_tokens": 34, "cost": 1.3e-05},
}

_NOUL_PAYLOAD = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {"urgent": {"type": "noul", "noul": 0.95}},
}

_SCORE_PAYLOAD = {
    "model": "typesafe/jev-1.13-20260917",
    "answers": {
        "severity": {
            "type": "score",
            "score": 1.43,
            "legend": {"0": "cosmetic", "1": "workaround", "2": "blocking"},
            "probabilities": {"0": 0.0, "1": 0.57, "2": 0.43},
            "confidence": 0.35,
        }
    },
}


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, response):
        self.response = response
        self.calls: list[dict] = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return self.response


def _questions():
    return [
        Question(key="urgent", prompt="Does this convey urgency?", kind="noul"),
        Question(
            key="department",
            prompt="Which team should handle this?",
            kind="choice",
            options=("billing", "technical", "sales"),
            descriptions={"billing": "Payments, invoicing, refunds"},
        ),
        Question(
            key="severity",
            prompt="How severe is it?",
            kind="score",
            options=("cosmetic", "workaround", "blocking"),
        ),
    ]


def test_question_payloads_are_the_documented_shapes():
    _urgent, department, severity = _questions()
    assert _urgent.payload() == {
        "type": "noul",
        "instructions": "Does this convey urgency?",
    }
    assert department.payload() == {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {"billing": "Payments, invoicing, refunds", "technical": None, "sales": None},
    }
    assert severity.payload() == {
        "type": "score",
        "instructions": "How severe is it?",
        "criteria": ["cosmetic", "workaround", "blocking"],
    }


def test_question_kinds_are_validated():
    with pytest.raises(ValueError):
        Question(key="x", prompt="?", kind="choice")
    with pytest.raises(ValueError):
        Question(key="x", prompt="?", kind="score", options=("only-one",))
    with pytest.raises(ValueError):
        Question(key="x", prompt="?", kind="score", options=tuple(str(i) for i in range(11)))
    with pytest.raises(ValueError):
        Question(key="x", prompt="?", kind="noul", options=("yes",))


def test_answers_carry_their_distribution_and_confidence():
    urgent = by_key(parse_answers(_NOUL_PAYLOAD, _questions()))["urgent"]
    # A noul carries only the probability of yes; the confidence is the same rule Jev uses for
    # a two-option choice: how far the peak stands above a coin toss.
    assert urgent.value is True
    assert urgent.probability == 0.95
    assert urgent.confidence == pytest.approx(0.9)
    assert urgent.is_confident(0.85)

    department = by_key(parse_answers(_CHOICE_PAYLOAD, _questions()))["department"]
    assert department.value == "billing"
    assert department.probabilities["billing"] == 0.88
    assert department.confidence == 0.81

    severity = by_key(parse_answers(_SCORE_PAYLOAD, _questions()))["severity"]
    assert severity.value == 1.43
    assert severity.confidence == 0.35


def test_a_malformed_or_off_set_answer_is_dropped_never_guessed():
    payload = {
        "answers": {
            "department": {  # option the caller never offered
                "type": "choice",
                "choice": "legal",
                "probabilities": {"legal": 1.0},
                "confidence": 1.0,
            },
            "severity": {  # wrong type for the question
                "type": "noul",
                "noul": 1.0,
            },
            "unasked": {"type": "noul", "noul": 1.0},
        }
    }
    assert parse_answers(payload, _questions()) == []
    # A noul out of range is clamped; a choice without a listed option is not repaired.
    clamped = parse_answers({"answers": {"urgent": {"type": "noul", "noul": 3.0}}}, _questions())
    assert clamped[0].probability == 1.0
    assert clamped[0].value is True


@pytest.mark.parametrize(
    ("key", "row"),
    [
        ("urgent", {"type": "noul", "noul": "nan"}),
        ("urgent", {"type": "noul", "noul": float("inf")}),
        ("department", {"type": "choice", "choice": "billing", "confidence": "nan"}),
        ("department", {"type": "choice", "choice": "billing", "probabilities": []}),
        ("department", {"type": "choice", "choice": "billing", "probabilities": {"billing": float("nan")}}),
        ("severity", {"type": "score", "score": "nan", "confidence": 1}),
    ],
)
def test_malformed_numeric_decisions_are_never_applied(key, row):
    assert parse_answers({"answers": {key: row}}, _questions()) == []


@pytest.mark.parametrize("headers,expected", [
    ({}, "gen-decision"),
    ({"x-request-id": "req-decision"}, "req-decision"),
])
def test_decision_response_id_is_recorded(monkeypatch, tmp_path, headers, expected):
    from kicraft.cli.web_cost_report import load_stage_attempts

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    response = _FakeResponse({**_NOUL_PAYLOAD, "id": "gen-decision"})
    response.headers = headers
    settings = Settings(api_key="test", ledger_path=tmp_path / "ledger.db")
    decide({}, _questions()[:1], session=_FakeSession(response), guard=SpendGuard(settings),
           run_id="decision-run", stage="architecture")
    attempts = load_stage_attempts(settings.ledger_path)
    assert attempts[0]["request_id"] == expected


def test_decide_posts_the_documented_body_once(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("KICRAFT_JEV_BASE_URL", "https://openrouter.test")
    payload = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"urgent": {"type": "noul", "noul": 0.95}},
        "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 1e-06},
    }
    session = _FakeSession(_FakeResponse(payload))
    guard = SpendGuard(Settings(api_key="test", ledger_path=tmp_path / "ledger.db"))
    billed: list[dict] = []

    answers = decide(
        {"draft": "state"},
        _questions()[:1],
        session=session,
        guard=guard,
        run_id="p7-run",
        stage="architecture",
        recorder=billed.append,
    )

    assert len(session.calls) == 1
    call = session.calls[0]
    assert call["url"] == "https://openrouter.test/api/alpha/decisions"
    assert call["headers"]["Authorization"] == "Bearer test-key"
    assert call["json"] == {
        "model": "typesafe/jev-1.13",
        "state": {"draft": "state"},
        "questions": {"urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
    }
    assert by_key(answers)["urgent"].probability == 0.95
    assert billed == [payload["usage"]]
    assert guard.spent_for_run("p7-run") == pytest.approx(1e-6)
    assert guard.status()["reserved_exposure_usd"] == 0
    from kicraft.cli.web_cost_report import load_rows, load_stage_attempts
    spend = load_rows(tmp_path / "ledger.db")
    attempts = load_stage_attempts(tmp_path / "ledger.db")
    receipt = next(row for row in spend if row["meta"].get("phase") == "decision")
    row, = attempts
    assert (row["stage"], row["attempt"], row["call_mode"], row["model"]) == (
        receipt["stage"], receipt["meta"]["attempt"], receipt["meta"]["call_mode"], receipt["model"]
    )
    assert row["cost_usd"] == pytest.approx(receipt["cost_usd"])
    assert row["http_status"] == 200
    assert guard.status()["uncertain_exposure_usd"] == 0


def test_decide_reports_every_way_it_cannot_take_a_decision(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("kicraft.server.decision_layer._api_key", lambda: "")
    monkeypatch.setenv("KICRAFT_JEV_BASE_URL", "https://openrouter.test")
    with pytest.raises(DecisionUnavailable, match="no OpenRouter API key"):
        decide({"s": 1}, _questions()[:1])

    monkeypatch.setattr("kicraft.server.decision_layer._api_key", lambda: "k")
    guard = SpendGuard(Settings(api_key="test", ledger_path=tmp_path / "ledger.db"))
    with pytest.raises(DecisionUnavailable):
        decide(
            {"s": 1},
            _questions()[:1],
            guard=guard,
            session=_FakeSession(_FakeResponse({"error": {"message": "bad state"}}, status_code=400)),
        )
    assert decide({"s": 1}, [], session=_FakeSession(_FakeResponse({}))) == []


@pytest.mark.parametrize("failure", ["no_receipt", "cancel"])
def test_decision_missing_receipt_or_cancellation_retains_exposure(
    monkeypatch, tmp_path, failure
):
    monkeypatch.setattr("kicraft.server.decision_layer._api_key", lambda: "k")
    guard = SpendGuard(Settings(api_key="k", ledger_path=tmp_path / "ledger.db"))
    if failure == "cancel":
        class CancelSession:
            def post(self, *args, **kwargs):
                raise RuntimeError("cancelled")

        session = CancelSession()
        with pytest.raises(RuntimeError, match="cancelled"):
            decide({}, _questions()[:1], guard=guard, run_id="p7-run", session=session)
    else:
        session = _FakeSession(_FakeResponse({"answers": {}}))
        assert decide({}, _questions()[:1], guard=guard, run_id="p7-run", session=session) == []
    assert guard.spent_for_run("p7-run") == 0
    assert guard.status()["uncertain_exposure_usd"] > 0
    assert guard.status()["reserved_exposure_usd"] == 0


def test_decision_budget_refusal_precedes_provider_call(monkeypatch, tmp_path):
    from kicraft.server.spend_guard import BudgetExceeded

    monkeypatch.setattr("kicraft.server.decision_layer._api_key", lambda: "k")
    guard = SpendGuard(Settings(
        api_key="k", ledger_path=tmp_path / "ledger.db", project_llm_budget_usd=1e-12,
    ))
    session = _FakeSession(_FakeResponse({"answers": {}}))
    with pytest.raises(BudgetExceeded):
        decide({}, _questions()[:1], guard=guard, run_id="p7-run", session=session)
    assert not session.calls


def test_unpriced_decision_model_is_refused_before_dispatch(monkeypatch, tmp_path):
    monkeypatch.setattr("kicraft.server.decision_layer._api_key", lambda: "k")
    guard = SpendGuard(Settings(api_key="k", ledger_path=tmp_path / "ledger.db"))
    session = _FakeSession(_FakeResponse({"answers": {}}))
    with pytest.raises(DecisionUnavailable, match="no bounded decision pricing"):
        decide({}, _questions()[:1], model="unknown/model", guard=guard, session=session)
    assert not session.calls
    assert guard.status()["reserved_exposure_usd"] == 0


def test_answer_confident_threshold():
    answer = Answer(key="k", kind="noul", value=True, confidence=0.5, probability=0.75)
    assert not answer.is_confident(0.75)
    assert answer.is_confident(0.5)


@pytest.mark.parametrize("stage", ["intent", "functional_spec"])
def test_reconciliation_dispatch_records_each_real_decision(monkeypatch, tmp_path, stage):
    from types import SimpleNamespace
    from kicraft.cli.web_cost_report import load_rows, load_stage_attempts
    from kicraft.server import decision_layer, stage_runtime

    monkeypatch.setattr(decision_layer, "_api_key", lambda: "k")
    response = _FakeResponse({
        "answers": {"urgent": {"type": "noul", "noul": 0.95}},
        "usage": {"input_tokens": 10, "output_tokens": 0, "cost": 1e-6},
    })
    response.headers = {"x-request-id": "req-decision"}
    session = _FakeSession(response)
    monkeypatch.setattr(decision_layer.requests, "post", session.post)
    guard = SpendGuard(Settings(api_key="k", ledger_path=tmp_path / "ledger.db"))
    client = SimpleNamespace(
        guard=guard,
        s=SimpleNamespace(enable_name_reconciliation=True, draft_audit_model=decision_layer.DEFAULT_MODEL),
    )
    count = iter((1, 2))
    decider = stage_runtime._name_reconciliation_decider(
        client, stage=stage, run_id="run", reserve_call=lambda mode: next(count),
    )
    assert decider({"draft": 1}, _questions()[:1])["urgent"].value is True
    assert decider({"draft": 2}, _questions()[:1])["urgent"].value is True
    rows = load_stage_attempts(tmp_path / "ledger.db")
    receipts = [r for r in load_rows(tmp_path / "ledger.db") if r["meta"].get("phase") == "decision"]
    assert len(session.calls) == len(rows) == len(receipts) == 2
    assert [(r["attempt"], r["call_mode"], r["model"], r["request_id"]) for r in rows] == [
        (1, "name_reconciliation", decision_layer.DEFAULT_MODEL, "req-decision"),
        (2, "name_reconciliation", decision_layer.DEFAULT_MODEL, "req-decision"),
    ]
    assert [(r["stage"], r["meta"]["attempt"], r["meta"]["call_mode"], r["model"]) for r in receipts] == [
        (r["stage"], r["attempt"], r["call_mode"], r["model"]) for r in rows
    ]


def test_rejected_or_unavailable_decisions_never_store_body_or_phantom_attempt(monkeypatch, tmp_path):
    from kicraft.cli.web_cost_report import load_stage_attempts
    monkeypatch.setattr("kicraft.server.decision_layer._api_key", lambda: "k")
    guard = SpendGuard(Settings(api_key="k", ledger_path=tmp_path / "ledger.db"))
    session = _FakeSession(_FakeResponse({"error": {"message": "secret request body"}}, 400))
    with pytest.raises(DecisionUnavailable) as error:
        decide({}, _questions()[:1], guard=guard, stage="intent", run_id="run",
               reserve_call=lambda: 4, call_mode="name_reconciliation", session=session)
    assert "secret" not in str(error.value)
    rows = load_stage_attempts(tmp_path / "ledger.db")
    assert len(rows) == 1
    assert (rows[0]["attempt"], rows[0]["http_status"], rows[0]["outcome"]) == (
        4, 400, "provider_request_rejected"
    )
    assert "secret" not in json.dumps(rows)
    with pytest.raises(DecisionUnavailable):
        decide({}, _questions()[:1], guard=guard, stage="intent", run_id="run",
               reserve_call=lambda: False, session=session)
    assert len(session.calls) == 1
    assert len(load_stage_attempts(tmp_path / "ledger.db")) == 1


def test_architecture_audit_records_actual_decision_model_and_budget_slot(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from kicraft.cli.web_cost_report import load_rows, load_stage_attempts
    from kicraft.server import decision_layer, draft_audit, stage_runtime

    monkeypatch.setattr(decision_layer, "_api_key", lambda: "k")
    monkeypatch.setattr(draft_audit, "architecture_questions", lambda candidate: _questions()[:1])
    response = _FakeResponse({"answers": {}, "usage": {"input_tokens": 5, "cost": 1e-6}})
    session = _FakeSession(response)
    monkeypatch.setattr(decision_layer.requests, "post", session.post)
    guard = SpendGuard(Settings(api_key="k", ledger_path=tmp_path / "ledger.db"))
    client = SimpleNamespace(guard=guard, s=SimpleNamespace(
        enable_draft_audit=True, draft_audit_model=decision_layer.DEFAULT_MODEL,
        draft_audit_confidence=0.7,
    ))
    slots = []
    def reserve(mode):
        slots.append(mode)
        return 4
    stage_runtime._audit_architecture_draft(client, {}, run_id="run", reserve_call=reserve)
    row, = load_stage_attempts(tmp_path / "ledger.db")
    receipt = next(r for r in load_rows(tmp_path / "ledger.db") if r["meta"].get("phase") == "decision")
    assert slots == ["draft_audit"]
    assert (row["attempt"], row["call_mode"], row["model"]) == (
        4, "draft_audit", decision_layer.DEFAULT_MODEL
    )
    assert (receipt["meta"]["attempt"], receipt["meta"]["call_mode"], receipt["model"]) == (
        row["attempt"], row["call_mode"], row["model"]
    )
