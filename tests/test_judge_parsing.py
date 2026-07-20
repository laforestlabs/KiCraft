"""Direct unit tests for the judge's reply parsing (2026-07-19 review §8.4).

_extract_json / _coerce_level / _validate stand between an LLM's free-text
reply and the Class-J scores that drive self-eval fix priorities; they were
previously only exercised indirectly through FakeClient fixtures.
"""
from __future__ import annotations

from kicraft.eval.judge import _coerce_level, _extract_json, _validate

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
    ok, dims, gates, err = _validate(_verdict(), JDIMS, OGATES)
    assert ok and err is None
    assert dims["spec_compliance"] == {"level": 3, "evidence": "a"}
    assert gates == []


def test_validate_rejects_non_dict_and_missing_dimensions():
    ok, _, _, err = _validate(None, JDIMS, OGATES)
    assert not ok and "no JSON object" in err
    ok, _, _, err = _validate({"foo": 1}, JDIMS, OGATES)
    assert not ok and "dimensions" in err


def test_validate_rejects_missing_dim_and_bad_level():
    v = _verdict()
    del v["dimensions"]["intent_fidelity"]
    ok, _, _, err = _validate(v, JDIMS, OGATES)
    assert not ok and "intent_fidelity" in err

    v = _verdict(levels=(True, 4))
    ok, _, _, err = _validate(v, JDIMS, OGATES)
    assert not ok and "spec_compliance" in err


def test_validate_keeps_only_known_gates():
    v = _verdict(gates=[
        {"id": "wrong_board", "evidence": "board mismatched"},
        {"id": "invented_gate", "evidence": "x"},
        "not-a-dict",
    ])
    ok, _, gates, _ = _validate(v, JDIMS, OGATES)
    assert ok
    assert [g["id"] for g in gates] == ["wrong_board"]
    assert gates[0]["cap"] == 39
    assert gates[0]["why"] == "board mismatched"


def test_validate_evidence_none_becomes_empty_string():
    v = _verdict()
    v["dimensions"]["spec_compliance"]["evidence"] = None
    ok, dims, _, _ = _validate(v, JDIMS, OGATES)
    assert ok
    assert dims["spec_compliance"]["evidence"] == ""
