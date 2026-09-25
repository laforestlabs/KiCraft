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


def test_decide_posts_the_documented_body_once(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setenv("KICRAFT_JEV_BASE_URL", "https://openrouter.test")
    payload = {
        "model": "typesafe/jev-1.13-20260917",
        "answers": {"urgent": {"type": "noul", "noul": 0.95}},
        "usage": {"input_tokens": 10, "output_tokens": 2, "cost": 1e-06},
    }
    session = _FakeSession(_FakeResponse(payload))
    billed: list[dict] = []

    answers = decide(
        {"draft": "state"},
        _questions()[:1],
        session=session,
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


def test_decide_reports_every_way_it_cannot_take_a_decision(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr("kicraft.server.decision_layer._api_key", lambda: "")
    with pytest.raises(DecisionUnavailable):
        decide({"s": 1}, _questions()[:1])

    monkeypatch.setattr("kicraft.server.decision_layer._api_key", lambda: "k")
    with pytest.raises(DecisionUnavailable):
        decide(
            {"s": 1},
            _questions()[:1],
            session=_FakeSession(_FakeResponse({"error": {"message": "bad state"}}, status_code=400)),
        )
    assert decide({"s": 1}, [], session=_FakeSession(_FakeResponse({}))) == []


def test_answer_confident_threshold():
    answer = Answer(key="k", kind="noul", value=True, confidence=0.5, probability=0.75)
    assert not answer.is_confident(0.75)
    assert answer.is_confident(0.5)
