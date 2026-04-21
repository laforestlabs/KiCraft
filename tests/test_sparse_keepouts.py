from kicraft.autoplacer.brain.subcircuit_composer import (
    LeafBlockerSet,
    can_overlap_sparse,
)
from kicraft.autoplacer.brain.types import Point


def _blockers(
    *,
    front=(),
    back=(),
    tht=(),
    outline=((0.0, 0.0), (10.0, 10.0)),
) -> LeafBlockerSet:
    def _rect(raw):
        return (Point(raw[0][0], raw[0][1]), Point(raw[1][0], raw[1][1]))

    return LeafBlockerSet(
        front_pads=tuple(_rect(rect) for rect in front),
        back_pads=tuple(_rect(rect) for rect in back),
        tht_drills=tuple(_rect(rect) for rect in tht),
        leaf_outline=_rect(outline),
    )


def test_two_front_only_leaves_cannot_overlap_sparse():
    a = _blockers(front=(((0.0, 0.0), (5.0, 5.0)),))
    b = _blockers(front=(((2.0, 2.0), (7.0, 7.0)),))
    assert can_overlap_sparse(a, Point(0.0, 0.0), 0.0, b, Point(0.0, 0.0), 0.0) is False


def test_two_back_only_leaves_cannot_overlap_sparse():
    a = _blockers(back=(((0.0, 0.0), (5.0, 5.0)),))
    b = _blockers(back=(((2.0, 2.0), (7.0, 7.0)),))
    assert can_overlap_sparse(a, Point(0.0, 0.0), 0.0, b, Point(0.0, 0.0), 0.0) is False


def test_front_only_can_overlap_back_only_sparse():
    a = _blockers(front=(((0.0, 0.0), (5.0, 5.0)),))
    b = _blockers(back=(((0.0, 0.0), (5.0, 5.0)),))
    assert can_overlap_sparse(a, Point(0.0, 0.0), 0.0, b, Point(0.0, 0.0), 0.0) is True


def test_mixed_leaf_can_overlap_when_same_side_pads_clear():
    mixed = _blockers(
        front=(((0.0, 0.0), (2.0, 2.0)),),
        back=(((6.0, 6.0), (8.0, 8.0)),),
    )
    back_only = _blockers(back=(((0.0, 0.0), (2.0, 2.0)),))
    assert can_overlap_sparse(mixed, Point(0.0, 0.0), 0.0, back_only, Point(0.0, 0.0), 0.0) is True


def test_mixed_leaf_cannot_overlap_when_same_side_back_pads_intersect():
    mixed = _blockers(back=(((0.0, 0.0), (3.0, 3.0)),))
    back_only = _blockers(back=(((2.0, 2.0), (4.0, 4.0)),))
    assert can_overlap_sparse(mixed, Point(0.0, 0.0), 0.0, back_only, Point(0.0, 0.0), 0.0) is False


def test_tht_drill_over_front_pad_is_forbidden():
    tht = _blockers(tht=(((1.0, 1.0), (3.0, 3.0)),))
    front = _blockers(front=(((2.0, 2.0), (4.0, 4.0)),))
    assert can_overlap_sparse(tht, Point(0.0, 0.0), 0.0, front, Point(0.0, 0.0), 0.0) is False


def test_outline_overlap_without_pad_overlap_is_allowed_opposite_side():
    a = _blockers(
        front=(((0.0, 0.0), (1.0, 1.0)),),
        outline=((0.0, 0.0), (10.0, 10.0)),
    )
    b = _blockers(
        back=(((9.0, 9.0), (10.0, 10.0)),),
        outline=((0.0, 0.0), (10.0, 10.0)),
    )
    assert can_overlap_sparse(a, Point(0.0, 0.0), 0.0, b, Point(0.0, 0.0), 0.0) is True


def test_same_side_outline_overlap_is_forbidden_even_when_pads_do_not_touch():
    a = _blockers(
        front=(((0.0, 0.0), (1.0, 1.0)),),
        outline=((0.0, 0.0), (10.0, 10.0)),
    )
    b = _blockers(
        front=(((12.0, 12.0), (13.0, 13.0)),),
        outline=((5.0, 5.0), (15.0, 15.0)),
    )
    assert can_overlap_sparse(a, Point(0.0, 0.0), 0.0, b, Point(0.0, 0.0), 0.0) is False
