from kicraft.autoplacer.brain.subcircuit_composer import (
    LeafBlockerSet,
    can_overlap_sparse,
)
from kicraft.autoplacer.brain.types import Component, Layer, Point
from kicraft.cli.compose_subcircuits import (
    _keep_in_conflict,
    _find_non_overlapping_origin,
    _preview_parent_local_keep_in_rects,
)
from kicraft.autoplacer.brain.subcircuit_composer import AttachmentConstraint, PlacementConstraintEntry
from kicraft.autoplacer.brain.subcircuit_composer import PlacementModel
from kicraft.autoplacer.brain.subcircuit_instances import TransformedSubcircuit
from kicraft.autoplacer.brain.types import SubCircuitInstance, SubCircuitLayout, SubCircuitId


def _model(blockers: LeafBlockerSet, width: float = 10.0, height: float = 10.0) -> PlacementModel:
    layout_id = SubCircuitId(sheet_name="LEAF", sheet_file="leaf.kicad_sch", instance_path="/leaf")
    transformed = TransformedSubcircuit(
        instance=SubCircuitInstance(
            layout_id=layout_id,
            origin=Point(0.0, 0.0),
            rotation=0.0,
            transformed_bbox=(width, height),
        ),
        layout=SubCircuitLayout(
            subcircuit_id=layout_id,
            components={},
            traces=[],
            vias=[],
            bounding_box=(width, height),
        ),
        bounding_box=(Point(0.0, 0.0), Point(width, height)),
    )
    return PlacementModel(
        rotation=0.0,
        transformed=transformed,
        layer_envelopes=([], [], []),
        blocker_set=blockers,
        constraint_entries=[],
    )


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


def test_constrained_children_bypass_keep_in_conflict():
    blockers = _blockers(
        front=(((0.0, 0.0), (10.0, 4.0)),),
        outline=((0.0, 0.0), (10.0, 10.0)),
    )
    assert _keep_in_conflict(
        blockers,
        Point(0.0, 0.0),
        0.0,
        [(Point(0.0, 6.0), Point(4.0, 10.0))],
    ) is True


def test_constrained_child_model_can_ignore_keep_in_filter_when_requested():
    blockers = _blockers(
        front=(((0.0, 0.0), (10.0, 4.0)),),
        outline=((0.0, 0.0), (10.0, 10.0)),
    )
    constraint = AttachmentConstraint(
        ref="EDGE",
        target="edge",
        value="left",
        inward_keep_in_mm=0.0,
        outward_overhang_mm=0.0,
        source="child_artifact",
        child_index=0,
        strict=True,
    )
    model = PlacementModel(
        rotation=0.0,
        transformed=_model(blockers).transformed,
        layer_envelopes=([], [], []),
        blocker_set=blockers,
        constraint_entries=[PlacementConstraintEntry(constraint, Point(0.0, 5.0))],
    )
    origin = _find_non_overlapping_origin(
        proposed=Point(0.0, 0.0),
        frame_min=Point(0.0, 0.0),
        frame_max=Point(40.0, 40.0),
        model=model,
        placed_bboxes=[],
        placed_envelopes=[],
        spacing_mm=2.0,
        parent_local_keep_in_rects=[],
    )
    assert origin == Point(0.0, 0.0)


def test_parent_local_keep_ins_can_be_previewed_around_constrained_rects():
    class _Constraint:
        ref = "MH1"
        inward_keep_in_mm = 2.5
        outward_overhang_mm = 0.0
        target = "corner"
        value = "top-left"

    hole = Component(
        ref="MH1",
        value="",
        pos=Point(2.0, 2.0),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=4.0,
        height_mm=4.0,
        pads=[],
        body_center=Point(2.0, 2.0),
    )

    keep_ins = _preview_parent_local_keep_in_rects(
        {"MH1": hole},
        [_Constraint()],
        (Point(0.0, 0.0), Point(40.0, 40.0)),
        occupied_rects=[(Point(0.0, 0.0), Point(12.0, 12.0))],
    )
    assert keep_ins
    keep_in = keep_ins[0]
    assert keep_in[0].x >= 12.0 or keep_in[0].y >= 12.0
    assert keep_in[0] != Point(-2.5, -2.5)


def test_unconstrained_keep_in_still_shifts_outline_clear():
    blockers = _blockers(
        front=(((0.0, 0.0), (2.0, 2.0)),),
        outline=((0.0, 0.0), (10.0, 10.0)),
    )
    model = _model(blockers)
    origin = _find_non_overlapping_origin(
        proposed=Point(0.0, 0.0),
        frame_min=Point(0.0, 0.0),
        frame_max=Point(40.0, 40.0),
        model=model,
        placed_bboxes=[],
        placed_envelopes=[],
        spacing_mm=2.0,
        parent_local_keep_in_rects=[(Point(0.0, 0.0), Point(9.0, 9.0))],
    )
    assert origin.x >= 9.0 or origin.y >= 9.0
