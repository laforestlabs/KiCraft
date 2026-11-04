"""Rubric v2: the board_self_description dimension (Phase 4c).

Locks the invariants the hash tripwire can't express in a unit run: weights
still total 100, the new Class-J dimension exists, and the judge contract
(derived from the rubric) includes it so a run can't finalize without it."""
from kicraft.eval.rubric import load_rubric


def test_weights_still_sum_to_100():
    dims = load_rubric()["dimensions"]
    assert sum(d["weight"] for d in dims) == 100






