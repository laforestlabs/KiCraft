"""Direct unit tests for the judge's reply parsing (2026-07-19 review §8.4).

_extract_json / _coerce_level / _validate stand between an LLM's free-text
reply and the Class-J scores that drive self-eval fix priorities; they were
previously only exercised indirectly through FakeClient fixtures.
"""
from __future__ import annotations

from kicraft.eval.judge import _coerce_level, _extract_json, _validate, grade_class_j

JDIMS = [{"id": "spec_compliance"}, {"id": "intent_fidelity"}]
OGATES = [{"id": "wrong_board", "cap": 39}]


def _verdict(levels=(3, 4), gates=None):
    v = {
        "dimensions": {
            "spec_compliance": {"level": levels[0], "evidence": "a"},
            "intent_fidelity": {"level": levels[1], "evidence": "b"},
        }
    }
    if gates is not None:
        v["triggered_gates"] = gates
    return v

def _rubric():
    dims = [
        {
            "id": item["id"],
            "class": "J",
            "weight": 1,
            "anchors": {"0": "bad", "1": "weak", "2": "ok", "3": "good", "4": "great"},
        }
        for item in JDIMS
    ]
    return {"dimensions": dims, "gates": []}


# --- _extract_json ---------------------------------------------------------

def test_extract_plain_json():
    assert _extract_json('{"a": 1}') == {"a": 1}


def test_extract_json_code_fence():
    assert _extract_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_extract_bare_fence():
    assert _extract_json('```\n{"a": 1}\n```') == {"a": 1}


def test_extract_json_with_surrounding_prose():
    assert _extract_json('Here is my verdict:\n{"a": {"b": 2}}\nDone.') == {
        "a": {"b": 2}
    }


def test_extract_json_truncated_returns_none():
    assert _extract_json('{"a": {"b": 2}') is None


def test_extract_json_empty_and_proseless():
    assert _extract_json("") is None
    assert _extract_json("no json here") is None


def test_extract_json_braces_in_prose_before_object():
    text = 'score {level} placeholder... {"a": 1} trailing'
    # First "{" starts at the placeholder; balanced-scan on it fails json
    # parsing and the helper gives up (documented current behavior: one
    # balanced-candidate attempt, no rescan).
    assert _extract_json(text) is None


# --- _coerce_level ---------------------------------------------------------

def test_coerce_valid_ints_and_floats():
    assert _coerce_level(3) == 3
    assert _coerce_level(0) == 0
    assert _coerce_level(4.0) == 4


def test_coerce_rejects_bools_range_and_junk():
    assert _coerce_level(True) is None
    assert _coerce_level(False) is None
    assert _coerce_level(5) is None
    assert _coerce_level(-1) is None
    assert _coerce_level(3.5) is None
    assert _coerce_level("3") is None
    assert _coerce_level(None) is None


# --- _validate -------------------------------------------------------------

def test_validate_happy_path():
    ok, dims, gates, rejected, err = _validate(_verdict(), JDIMS, OGATES)
    assert ok and err is None
    assert dims["spec_compliance"] == {"level": 3, "evidence": "a"}
    assert gates == [] and rejected == []


def test_validate_rejects_non_dict_and_missing_dimensions():
    ok, _, _, _, err = _validate(None, JDIMS, OGATES)
    assert not ok and "no JSON object" in err
    ok, _, _, _, err = _validate({"foo": 1}, JDIMS, OGATES)
    assert not ok and "dimensions" in err


def test_validate_rejects_missing_dim_and_bad_level():
    v = _verdict()
    del v["dimensions"]["intent_fidelity"]
    ok, _, _, _, err = _validate(v, JDIMS, OGATES)
    assert not ok and "intent_fidelity" in err

    v = _verdict(levels=(True, 4))
    ok, _, _, _, err = _validate(v, JDIMS, OGATES)
    assert not ok and "spec_compliance" in err


def test_validate_keeps_only_known_gates():
    v = _verdict(gates=[
        {"id": "wrong_board", "evidence": "board mismatched"},
        {"id": "invented_gate", "evidence": "x"},
        "not-a-dict",
    ])
    ok, _, gates, rejected, _ = _validate(v, JDIMS, OGATES)
    assert ok
    assert [g["id"] for g in gates] == ["wrong_board"]
    assert gates[0]["cap"] == 39
    assert gates[0]["why"] == "board mismatched"
    assert rejected == []


def test_validate_evidence_none_becomes_empty_string():
    v = _verdict()
    v["dimensions"]["spec_compliance"]["evidence"] = None
    ok, dims, _, _, _ = _validate(v, JDIMS, OGATES)
    assert ok
    assert dims["spec_compliance"]["evidence"] == ""


# --- gate polarity (2026-07-27 batch fix-plan P0) --------------------------
#
# The judge must never apply a gate whose own evidence refutes it. The two
# payloads below are VERBATIM from batch 20260727T045000Z (run_17 and run_34),
# each of which was wrongly capped B->D / C->D.

SUB_GATES = [{"id": "silent_substitution", "cap": 55},
             {"id": "unprogrammable_mcu", "cap": 50}]


def test_validate_gate_triggered_false_is_rejected_not_applied():
    v = _verdict(gates=[
        {"id": "silent_substitution", "triggered": False,
         "evidence": "spec'd part shipped as spec'd"},
    ])
    ok, _, gates, rejected, _ = _validate(v, JDIMS, SUB_GATES)
    assert ok and gates == []
    assert [r["id"] for r in rejected] == ["silent_substitution"]
    assert rejected[0]["rejected_because"] == "triggered: false"


def test_validate_gate_triggered_true_is_applied():
    v = _verdict(gates=[
        {"id": "unprogrammable_mcu", "triggered": True,
         "evidence": "no programming header, gap not surfaced"},
    ])
    ok, _, gates, rejected, _ = _validate(v, JDIMS, SUB_GATES)
    assert ok and rejected == []
    assert [g["id"] for g in gates] == ["unprogrammable_mcu"]
    assert gates[0]["cap"] == 50


def test_validate_legacy_self_negating_substitution_evidence_screened():
    # run_17 led-cc-driver, batch 20260727T045000Z: capped 80.5 -> 55 by this.
    v = _verdict(gates=[{
        "id": "silent_substitution",
        "evidence": ("No named parts were specified by the user, so there is "
                     "nothing to silently substitute against. PT4115 and "
                     "JNJ-LTJW0115W120 were defaulted as assumptions, "
                     "recorded openly."),
    }])
    ok, _, gates, rejected, _ = _validate(v, JDIMS, SUB_GATES)
    assert ok and gates == []
    assert [r["id"] for r in rejected] == ["silent_substitution"]
    assert rejected[0]["rejected_because"] == "self-negating evidence"


def test_validate_legacy_gate_does_not_trigger_evidence_screened():
    # run_34 snowman-ornament, batch 20260727T045000Z: capped 73 -> 50 by this.
    v = _verdict(gates=[{
        "id": "unprogrammable_mcu",
        "evidence": ("UPDI header J1 is included (1x03 with VCC, GND, UPDI) so "
                     "a first-flash programming path exists. Gate does not "
                     "trigger."),
    }])
    ok, _, gates, rejected, _ = _validate(v, JDIMS, SUB_GATES)
    assert ok and gates == []
    assert rejected and rejected[0]["rejected_because"] == "self-negating evidence"


def test_validate_legacy_affirmative_evidence_with_not_surfaced_still_applies():
    # Affirmative evidence routinely SAYS "the gap is not surfaced" -- that
    # phrasing must never be screened (run_10's legit-shaped evidence).
    v = _verdict(gates=[{
        "id": "unprogrammable_mcu",
        "evidence": ("SWCLK is only routed to HEADER; no SWD header enumerated. "
                     "The gap is not surfaced in open_questions."),
    }])
    ok, _, gates, rejected, _ = _validate(v, JDIMS, SUB_GATES)
    assert ok and rejected == []
    assert [g["id"] for g in gates] == ["unprogrammable_mcu"]


def test_validate_explicit_true_with_negative_sounding_evidence_applies():
    # An explicit triggered: true wins over the negation screen (the screen is
    # only for legacy entries lacking the boolean).
    v = _verdict(gates=[{
        "id": "silent_substitution", "triggered": True,
        "evidence": "substitution does not trigger any assumption note, so it held silently",
    }])
    ok, _, gates, rejected, _ = _validate(v, JDIMS, SUB_GATES)
    assert ok and rejected == []
    assert [g["id"] for g in gates] == ["silent_substitution"]


def test_grade_reports_client_reasoning_abort_as_distinct_retry_defect():
    class AbortClient:
        def __init__(self):
            self.calls = 0

        def chat(self, messages, **kwargs):
            self.calls += 1
            return {
                "text": "",
                "finish_reason": "reasoning_loop",
                "loop_abort_reason": "hard_ceiling",
                "cost_usd": 0.01,
            }

    client = AbortClient()
    result = grade_class_j(client, "digest", _rubric(), max_attempts=2)
    assert result["ok"] is False
    assert result["error"] == "client aborted judge reasoning (hard_ceiling)"
    assert client.calls == 2
