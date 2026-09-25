"""Name reconciliation: code shortlists, the decision model picks, only confident picks apply."""

from __future__ import annotations

from types import SimpleNamespace

from kicraft.server.decision_layer import Answer
from kicraft.server.reconciliation import (
    KEEP,
    PROPERTY,
    reconcile_bom_classes,
    reconcile_declared_port_pins,
    resolve_demanded_classes,
    resolve_quantity_subjects,
    shortlist_reviewed_classes,
    shortlist_slot_classes,
)


def _decider(script):
    """A decider that answers from ``script`` (question id -> (value, confidence))."""
    calls = []

    def decide_batch(state, questions):
        calls.append({"state": state, "questions": list(questions)})
        return {
            question.key: Answer(
                key=question.key,
                kind=question.kind,
                value=script[question.key][0],
                confidence=script[question.key][1],
            )
            for question in questions
            if question.key in script
        }

    decide_batch.calls = calls
    return decide_batch


def test_shortlists_are_closed_and_deterministic():
    # A qualifier-laden class shortlists the reviewed class that shares its tokens.
    assert "motor-driver" in shortlist_reviewed_classes("gps-motor-driver-module")
    # The demanded class's own spelling ranks first; its token-sharing neighbours follow, so the
    # model can be offered a bounded list rather than the whole vocabulary.
    assert shortlist_reviewed_classes("motor-driver")[0] == "motor-driver"
    assert 1 < len(shortlist_reviewed_classes("motor-driver")) <= 12
    assert shortlist_slot_classes("relay channels", ["through-hole-relay", "screw-terminal"]) == (
        "through-hole-relay",
    )
    assert shortlist_slot_classes("temperature channels", ["through-hole-relay"]) == ()


def test_a_confident_reading_rewrites_the_demanded_class():
    candidate = {
        "obligations": [
            {"kind": "physical", "original_obligation_id": "x", "component_class": "actuator-driver"}
        ],
        "assumptions": [],
    }
    listing = shortlist_reviewed_classes("actuator-driver")
    assert listing, "an uncovered class with reviewed neighbours still gets a closed list"
    expected = listing[0]
    decider = _decider({"class_0": (expected, 0.9)})

    completed, notes = resolve_demanded_classes(candidate, decider=decider)

    assert completed["obligations"][0]["component_class"] == expected
    assert notes and f"read as reviewed class {expected!r}" in completed["assumptions"][-1]


def test_a_low_confidence_or_kept_class_is_left_alone():
    candidate = {
        "obligations": [
            {"kind": "physical", "original_obligation_id": "x", "component_class": "actuator-driver"}
        ],
        "assumptions": [],
    }
    candidate_option = shortlist_reviewed_classes("actuator-driver")[0]
    for script in ({"class_0": (candidate_option, 0.4)}, {"class_0": (KEEP, 0.99)}):
        completed, notes = resolve_demanded_classes(candidate, decider=_decider(script))
        assert completed == candidate and notes == []

    # A class with a deterministic spelling relation, or with no reviewed neighbour, is never asked.
    settled = {
        "obligations": [
            # No reviewed class shares a token with this one, so there is nothing to offer.
            {"kind": "physical", "original_obligation_id": "a", "component_class": "zzz-widget"},
            {"kind": "physical", "original_obligation_id": "b", "component_class": "smt-voltage-regulator"},
        ]
    }
    decider = _decider({})
    resolve_demanded_classes(settled, decider=decider)
    assert decider.calls == []


def test_a_count_is_bound_to_its_class_or_declared_a_property():
    candidate = {
        "obligations": [
            {"kind": "physical", "original_obligation_id": "r", "component_class": "through-hole-relay"},
            {"kind": "quantity", "original_obligation_id": "rc", "subject": "relay channels", "minimum": 4},
        ],
        "assumptions": [],
    }
    decider = _decider({"count_1": ("through-hole-relay", 0.85)})

    completed, notes = resolve_quantity_subjects(candidate, decider=decider)

    assert completed["obligations"][1]["subject"] == "through-hole-relay"
    assert notes and "(defaulted)" in completed["assumptions"][-1]

    # "a property of one part" leaves the row (and the stage's own refusal) alone.
    property_decider = _decider({"count_1": (PROPERTY, 0.95)})
    untouched, notes = resolve_quantity_subjects(candidate, decider=property_decider)
    assert untouched == candidate and notes == []


def test_a_group_that_implements_the_class_is_named_when_it_is_the_only_one():
    groups = [
        SimpleNamespace(id="jst_a", value="XH-2", symbol="Connector:Conn_01x02", footprint="F", mpn="B2B-XH-A", quantity=2),
        SimpleNamespace(id="led", value="green", symbol="Device:LED", footprint="F", mpn=None, quantity=1),
    ]
    decider = _decider({"implements_0": (True, 0.88), "implements_1": (False, 0.95)})

    notes = reconcile_bom_classes(["actuator-driver"], groups, decider=decider)

    assert len(notes) == 1
    assert "jst_a" in notes[0] and "bind the group" in notes[0]
    # The pair question is one call for the whole batch.
    assert len(decider.calls) == 1


def test_an_ambiguous_or_unsure_group_recommends_nothing():
    groups = [
        SimpleNamespace(id="a", value="x", symbol="S1", footprint="F", mpn=None, quantity=1),
        SimpleNamespace(id="b", value="y", symbol="S2", footprint="F", mpn=None, quantity=1),
    ]
    both = _decider({"implements_0": (True, 0.9), "implements_1": (True, 0.9)})
    assert reconcile_bom_classes(["actuator-driver"], groups, decider=both) == []

    unsure = _decider({"implements_0": (True, 0.3), "implements_1": (False, 0.9)})
    assert reconcile_bom_classes(["actuator-driver"], groups, decider=unsure) == []


def test_a_free_text_contact_is_read_as_a_published_pin():
    claims = [
        {
            "requirement_id": "input",
            "key": "positive",
            "function": "18 V DC positive input",
            "symbol": "WJ126V-5.0-02P-14-00A",
            "claimed": "positive",
            "pins": ["1", "2"],
        }
    ]
    decider = _decider({"pin_0": ("1", 0.86)})

    notes = reconcile_declared_port_pins(claims, decider=decider)

    assert len(notes) == 1
    assert "reads as contact '1'" in notes[0]

    keep = _decider({"pin_0": (KEEP, 0.9)})
    assert reconcile_declared_port_pins(claims, decider=keep) == []


def test_an_unreachable_decider_leaves_the_candidate_alone(monkeypatch):
    """No answers is the safe failure: the stage's own refusal stays, nothing is invented."""
    from kicraft.server import reconciliation as R

    def boom(*_args, **_kwargs):
        raise R.DecisionUnavailable("provider unreachable")

    monkeypatch.setattr(R, "decide", boom)
    decider = R.jev_decider("typesafe/jev-1.13")
    assert decider({"anything": 1}, [R.Question(key="class_0", prompt="p", kind="noul")]) == {}

    candidate = {
        "obligations": [
            {"kind": "physical", "original_obligation_id": "x", "component_class": "actuator-driver"}
        ],
        "assumptions": [],
    }
    completed, notes = R.resolve_demanded_classes(candidate, decider=decider)
    assert completed == candidate and notes == []
