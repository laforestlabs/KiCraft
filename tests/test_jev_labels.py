"""The Jev labelling tool: typed questions, a content-addressed cache, and agreement."""

from __future__ import annotations

import json


from kicraft.eval import jev_labels
from kicraft.eval.jev_labels import Hit, agreement, label_hits, question_for
from kicraft.server.decision_layer import Answer


def _hit(run, stage="functional_spec", code="functional_spec_partial_ground_flow"):
    return Hit(run=str(run), stage=stage, code=code, message="m", brief="a brief", evidence=("x",))


def _run_with_candidate(tmp_path, name="run_01", stage="functional_spec"):
    run_dir = tmp_path / name
    (run_dir / ".kicraft").mkdir(parents=True)
    (run_dir / ".kicraft" / "state.json").write_text(
        json.dumps(
            {
                stage: {
                    "blocks": [{"name": "A", "category": "process", "purpose": "p"}],
                    "connections": [],
                    "assumptions": [],
                }
            }
        ),
        encoding="utf-8",
    )
    return run_dir


def test_questions_are_only_for_the_checks_under_study():
    assert question_for(_hit("/tmp/x")) is not None
    assert question_for(_hit("/tmp/x", code="functional_spec_self_loop")) is None
    question = question_for(_hit("/tmp/x", code="intent_quantity_subject_unbound"))
    assert question is not None
    # The spelling answer must be offered: without it the model cannot say "the class exists",
    # and the earlier two-option version labelled the audio-jack case as a missing class.
    assert jev_labels._COUNT_CLASS_MISSPELLED in question.options


def test_answers_are_cached_content_addressed(tmp_path, monkeypatch):
    run_dir = _run_with_candidate(tmp_path)
    calls: list[int] = []

    def fake_decide(state, questions, *, model=None, **kwargs):
        calls.append(1)
        return [
            Answer(
                key=question.key,
                kind="choice",
                value=question.options[0],
                confidence=0.9,
                probabilities={option: 0.9 for option in question.options},
            )
            for question in questions
        ]

    monkeypatch.setattr(jev_labels, "decide", fake_decide)
    cache = tmp_path / "cache"
    first = label_hits([_hit(run_dir)], cache_dir=cache)
    second = label_hits([_hit(run_dir)], cache_dir=cache)

    assert len(calls) == 1, "the second pass must read the cache, not the provider"
    assert first[0].value == second[0].value
    assert first[0].cached is False and second[0].cached is True
    assert len(list(cache.glob("*.json"))) == 1
    assert all(label.confident for label in first)


def test_below_floor_answers_are_kept_but_flag_not_confident(tmp_path, monkeypatch):
    run_dir = _run_with_candidate(tmp_path)
    monkeypatch.setattr(
        jev_labels,
        "decide",
        lambda state, questions, **kwargs: [
            Answer(
                key=question.key,
                kind="choice",
                value=question.options[1],
                confidence=0.4,
                probabilities={option: 0.4 for option in question.options},
            )
            for question in questions
        ],
    )
    labels = label_hits([_hit(run_dir)], cache_dir=tmp_path / "cache", confidence=0.7)
    assert labels[0].value == labels[0].value  # the reading is kept as data
    assert labels[0].confident is False
    assert "below confidence floor" in labels[0].notes[0]


def test_agreement_compares_hand_labels_only():
    label = jev_labels.Label(
        key="ground_return",
        run="r",
        stage="functional_spec",
        code="functional_spec_partial_ground_flow",
        question="q",
        value=jev_labels._GROUND_RETURN_PROVIDED,
        confidence=0.9,
        cached=False,
        confident=True,
    )
    hand = {("r", "functional_spec", "functional_spec_partial_ground_flow"): {
        "label": jev_labels._GROUND_RETURN_PROVIDED, "rationale": "shipped board"
    }}
    report = agreement([label], hand)
    assert (report["checked"], report["agreed"], report["rate"]) == (1, 1, 1.0)
    assert report["agreed_confident"] == 1

    disagreeing = {("r", "functional_spec", "functional_spec_partial_ground_flow"): {
        "label": jev_labels._GROUND_RETURN_MISSING, "rationale": "real gap"
    }}
    assert agreement([label], disagreeing)["agreed"] == 0
    assert agreement([label], {})["checked"] == 0
