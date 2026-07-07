"""Rubric v2: the board_self_description dimension (Phase 4c).

Locks the invariants the hash tripwire can't express in a unit run: weights
still total 100, the new Class-J dimension exists, and the judge contract
(derived from the rubric) includes it so a run can't finalize without it."""
from kicraft.eval.rubric import load_rubric


def test_weights_still_sum_to_100():
    dims = load_rubric()["dimensions"]
    assert sum(d["weight"] for d in dims) == 100


def test_board_self_description_present_as_class_j():
    dims = {d["id"]: d for d in load_rubric()["dimensions"]}
    d = dims["board_self_description"]
    assert d["class"] == "J"
    assert d["weight"] == 4
    # anchored 0-4 like every other dimension
    assert set(d["anchors"]) == {0, 1, 2, 3, 4}


def test_judge_contract_requires_the_new_dimension():
    # The judge renders + requires exactly the Class-J dims from the rubric, so a
    # new J dimension is graded automatically (and finalize_report raises if the
    # observer omits it). Assert it flows into that derived contract.
    from kicraft.eval import judge

    rubric = load_rubric()
    jdims = judge._class_j_dims(rubric)
    jids = {d["id"] for d in jdims}
    assert "board_self_description" in jids
    assert '"board_self_description"' in judge._output_contract(jdims)
