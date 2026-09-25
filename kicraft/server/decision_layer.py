"""Typed decisions from Jev (TypeSafe System One) through OpenRouter.

A language model asked to *write* a design produces prose that has to be parsed and repaired; a
decision model asked to *decide* among stated alternatives produces a value with a calibrated
probability, and a value can be checked. This module is the client for that second shape.

The transport is OpenRouter's decisions endpoint, because that is where the account already
authenticates, bills and rate-limits:

* ``POST {base}/api/alpha/decisions`` with ``{"model": "typesafe/jev-1.13", "state": ...,
  "questions": {<id>: <question>}}``;
* Jev refuses the ordinary ``/chat/completions`` route (it returns no free-form text at all), so
  a decision can never be smuggled in as a chat call.

Three primitives, and nothing else — ``noul`` (yes/no, answered as the probability of yes),
``choice`` (one of a closed set, answered with the full distribution) and ``score`` (an ordered
rubric, answered with a probability-weighted value). Every answer carries the probability
distribution it came from, and the confidence derived from it, so a caller can act on a decision
only when the model is sure and ask a human otherwise.

The call goes through the same spend guard and ledger as every other model call in KiCraft, and
the price is input-only ($0.042 / 1M tokens for Jev 1.13), so a draft audit over a 10 KB state
costs a fraction of a cent.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal, Mapping, Sequence

import requests

QuestionKind = Literal["noul", "choice", "score"]

#: Jev 1.13 through OpenRouter. Not listed in the public model catalogue (the account resolves it
#: directly), and rejected on /chat/completions, so the id is pinned here deliberately.
DEFAULT_MODEL = "typesafe/jev-1.13"
DEFAULT_TIMEOUT_S = 60.0
DECISIONS_PATH = "/api/alpha/decisions"


class DecisionUnavailable(RuntimeError):
    """No decision could be taken: no key, no endpoint, or an unusable response."""


@dataclass(frozen=True)
class Question:
    """One typed question about the state.

    ``kind`` maps to the Jev primitive: ``noul`` (yes/no), ``choice`` (closed set),
    ``score`` (ordered rubric). ``options`` is the closed set for a choice and the ordered levels
    for a score; each may carry a short description, which Jev uses as the rubric.
    """

    key: str
    prompt: str
    kind: QuestionKind
    options: tuple[str, ...] = ()
    descriptions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind == "choice" and not self.options:
            raise ValueError(f"choice question {self.key!r} needs options")
        if self.kind == "score" and len(self.options) < 2:
            raise ValueError(f"score question {self.key!r} needs at least two levels")
        if self.kind == "score" and len(self.options) > 10:
            raise ValueError(f"score question {self.key!r} accepts at most ten levels")
        if self.kind == "noul" and self.options:
            raise ValueError(f"noul question {self.key!r} takes no options")

    def payload(self) -> dict:
        """The question as Jev states it: type, instructions, criteria."""
        if self.kind == "choice":
            criteria = {
                option: self.descriptions.get(option) for option in self.options
            }
        elif self.kind == "score":
            criteria = list(self.options)
        else:
            criteria = None
        body: dict[str, Any] = {"type": self.kind, "instructions": self.prompt}
        if criteria is not None:
            body["criteria"] = criteria
        return body


@dataclass(frozen=True)
class Answer:
    """One typed answer, with the distribution it came from and its confidence (0.0-1.0)."""

    key: str
    kind: QuestionKind
    value: Any
    confidence: float
    probabilities: Mapping[str, float] = field(default_factory=dict)
    probability: float | None = None

    def is_confident(self, threshold: float) -> bool:
        return self.confidence >= threshold


def _normalized_confidence(probabilities: Mapping[str, float]) -> float:
    """TypeSafe's confidence rule: how far the peak stands above a uniform distribution.

    ``(n * peak - 1) / (n - 1)``, clamped to 0..1. A uniform distribution scores 0 (the answer
    is a coin toss among the options) and a single certain option scores 1. Jev reports
    ``confidence`` itself for choice/score answers; this reproduces it for a noul, whose answer
    carries only the probability of yes.
    """
    values = [float(value) for value in probabilities.values()]
    count = len(values)
    if count < 2:
        return 1.0 if count == 1 else 0.0
    peak = max(values)
    return max(0.0, min(1.0, (count * peak - 1) / (count - 1)))


def _base_url() -> str:
    explicit = os.environ.get("KICRAFT_JEV_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    try:
        from .config import Settings

        base = Settings.from_env().base_url or "https://openrouter.ai/api/v1"
    except Exception:  # pragma: no cover - configuration failure is reported below
        base = "https://openrouter.ai/api/v1"
    return base.rstrip("/")[: -len("/api/v1")].rstrip("/") if base.endswith("/api/v1") else base


def _api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or ""
    if key:
        return key
    try:
        from .config import Settings

        return Settings.from_env().api_key or ""
    except Exception:  # pragma: no cover
        return ""


def endpoint() -> str:
    return f"{_base_url()}{DECISIONS_PATH}"


def parse_answers(payload: Mapping[str, Any], questions: Sequence[Question]) -> list[Answer]:
    """Typed answers from a decisions response; a malformed answer is dropped, never guessed.

    Dropping is deliberate: an unanswered question must read as "no decision", because callers
    auto-apply confident answers.
    """
    by_key = {question.key: question for question in questions}
    rows = payload.get("answers")
    if not isinstance(rows, Mapping):
        return []
    answers: list[Answer] = []
    for key, row in rows.items():
        question = by_key.get(str(key))
        if question is None or not isinstance(row, Mapping):
            continue
        kind = str(row.get("type") or "")
        if kind != question.kind:
            continue
        if question.kind == "noul":
            try:
                probability = float(row.get("noul"))
            except (TypeError, ValueError):
                continue
            probability = max(0.0, min(1.0, probability))
            distribution = {"yes": probability, "no": 1.0 - probability}
            answers.append(
                Answer(
                    key=str(key),
                    kind="noul",
                    value=probability >= 0.5,
                    confidence=_normalized_confidence(distribution),
                    probabilities=distribution,
                    probability=probability,
                )
            )
            continue
        raw_probabilities = row.get("probabilities")
        probabilities = {
            str(option): float(value)
            for option, value in (raw_probabilities or {}).items()
            if isinstance(value, (int, float))
        }
        try:
            confidence = float(row.get("confidence"))
        except (TypeError, ValueError):
            confidence = _normalized_confidence(probabilities)
        confidence = max(0.0, min(1.0, confidence))
        if question.kind == "choice":
            choice = row.get("choice")
            if not isinstance(choice, str) or choice not in question.options:
                continue
            answers.append(
                Answer(
                    key=str(key),
                    kind="choice",
                    value=choice,
                    confidence=confidence,
                    probabilities=probabilities,
                )
            )
            continue
        try:
            score = float(row.get("score"))
        except (TypeError, ValueError):
            continue
        answers.append(
            Answer(
                key=str(key),
                kind="score",
                value=max(0.0, min(float(len(question.options) - 1), score)),
                confidence=confidence,
                probabilities=probabilities,
            )
        )
    return answers


def decide(
    state: Any,
    questions: Sequence[Question],
    *,
    model: str = DEFAULT_MODEL,
    timeout: float = DEFAULT_TIMEOUT_S,
    recorder=None,
    session: requests.Session | None = None,
) -> list[Answer]:
    """Take one decision over ``state``: every question answered in a single pass.

    ``recorder`` is called with the response ``usage`` (tokens and cost) so the caller can bill
    the call through the ledger exactly like any other model call.
    """
    questions = list(questions)
    if not questions:
        return []
    key = _api_key()
    if not key:
        raise DecisionUnavailable("no OpenRouter API key is configured")
    body = {
        "model": model,
        "state": state if isinstance(state, (str, list, dict)) else str(state),
        "questions": {question.key: question.payload() for question in questions},
    }
    http = session or requests
    try:
        response = http.post(
            endpoint(),
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=timeout,
        )
    except requests.RequestException as exc:
        raise DecisionUnavailable(f"the decisions endpoint could not be reached: {exc}") from exc
    if response.status_code >= 400:
        detail = ""
        try:
            detail = str((response.json() or {}).get("error", {}).get("message") or "")
        except Exception:
            detail = (response.text or "")[:300]
        raise DecisionUnavailable(
            f"the decisions endpoint refused the request ({response.status_code}): {detail[:400]}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise DecisionUnavailable(f"the decisions endpoint returned no JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise DecisionUnavailable("the decisions payload is not an object")
    if recorder is not None:
        usage = payload.get("usage")
        if isinstance(usage, Mapping):
            recorder(usage)
    return parse_answers(payload, questions)


def by_key(answers: Iterable[Answer]) -> dict[str, Answer]:
    return {answer.key: answer for answer in answers}
