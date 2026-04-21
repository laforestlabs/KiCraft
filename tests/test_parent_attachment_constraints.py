import pytest

from kicraft.autoplacer.brain.subcircuit_composer import (
    AttachmentConstraint,
    PlacementConstraintEntry,
    PlacementModel,
    constraint_aware_outline,
    place_constrained_child,
)
from kicraft.autoplacer.brain.subcircuit_instances import TransformedSubcircuit
from kicraft.autoplacer.brain.types import Point, SubCircuitId, SubCircuitInstance, SubCircuitLayout


def _make_model(
    bbox_min: Point,
    bbox_max: Point,
    entries: list[PlacementConstraintEntry],
) -> PlacementModel:
    layout = SubCircuitLayout(
        subcircuit_id=SubCircuitId(
            sheet_name="CHILD",
            sheet_file="child.kicad_sch",
            instance_path="/child",
        ),
        bounding_box=(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y),
    )
    transformed = TransformedSubcircuit(
        instance=SubCircuitInstance(
            layout_id=layout.subcircuit_id,
            origin=Point(0.0, 0.0),
            rotation=0.0,
            transformed_bbox=(bbox_max.x - bbox_min.x, bbox_max.y - bbox_min.y),
        ),
        layout=layout,
        bounding_box=(bbox_min, bbox_max),
    )
    return PlacementModel(
        rotation=0.0,
        transformed=transformed,
        layer_envelopes=([], [], []),
        constraint_entries=entries,
    )


def _edge_constraint(ref: str, value: str, child_index: int = 0) -> AttachmentConstraint:
    return AttachmentConstraint(
        ref=ref,
        target="edge",
        value=value,
        inward_keep_in_mm=0.0,
        outward_overhang_mm=0.0,
        source="child_artifact",
        child_index=child_index,
        strict=True,
    )


def _zone_constraint(ref: str, value: str, child_index: int = 0) -> AttachmentConstraint:
    return AttachmentConstraint(
        ref=ref,
        target="zone",
        value=value,
        inward_keep_in_mm=2.5,
        outward_overhang_mm=0.0,
        source="child_artifact",
        child_index=child_index,
        strict=False,
    )


def test_multi_ref_same_side_exact_constraints_share_target():
    model = _make_model(
        Point(-3.0, -2.0),
        Point(5.0, 4.0),
        [
            PlacementConstraintEntry(_edge_constraint("J2", "right"), Point(10.0, 1.0)),
            PlacementConstraintEntry(_edge_constraint("J3", "right"), Point(10.0, 3.0)),
        ],
    )

    origin, _ = place_constrained_child(
        model,
        parent_outline_min=Point(0.0, 0.0),
        parent_outline_max=Point(100.0, 60.0),
    )

    assert origin.x == pytest.approx(90.0)


def test_multi_ref_zone_constraints_use_band_membership():
    model = _make_model(
        Point(-20.0, -5.0),
        Point(20.0, 5.0),
        [
            PlacementConstraintEntry(_zone_constraint("BT1", "bottom"), Point(0.0, 10.0)),
            PlacementConstraintEntry(_zone_constraint("BT2", "bottom"), Point(0.0, 10.0)),
        ],
    )

    origin, _ = place_constrained_child(
        model,
        parent_outline_min=Point(0.0, 0.0),
        parent_outline_max=Point(120.0, 80.0),
    )

    assert origin.y == pytest.approx(67.5)


def test_conflicting_same_side_constraints_raise_error():
    model = _make_model(
        Point(0.0, 0.0),
        Point(20.0, 10.0),
        [
            PlacementConstraintEntry(_edge_constraint("J2", "right"), Point(8.0, 5.0)),
            PlacementConstraintEntry(_edge_constraint("J3", "right"), Point(12.0, 5.0)),
        ],
    )

    with pytest.raises(ValueError, match="J2=|J3="):
        _ = place_constrained_child(
            model,
            parent_outline_min=Point(0.0, 0.0),
            parent_outline_max=Point(100.0, 50.0),
        )


def test_constraint_aware_outline_never_shrinks_below_geometry_union():
    constraint = AttachmentConstraint(
        ref="J1",
        target="edge",
        value="left",
        inward_keep_in_mm=1.0,
        outward_overhang_mm=0.0,
        source="child_artifact",
        child_index=0,
        strict=True,
    )

    min_pt, max_pt = constraint_aware_outline(
        placed_bboxes=[
            (Point(-10.0, -5.0), Point(20.0, 15.0)),
            (Point(25.0, 0.0), Point(45.0, 10.0)),
        ],
        attachment_constraints=[constraint],
        constrained_ref_world_anchors={"J1": Point(-5.0, 4.0)},
        margin_mm=1.5,
    )

    assert min_pt.x == pytest.approx(-10.0)
    assert max_pt.x == pytest.approx(46.5)


def test_constraint_aware_outline_uses_parent_local_anchor_when_provided():
    constraint = AttachmentConstraint(
        ref="H1",
        target="corner",
        value="bottom-right",
        inward_keep_in_mm=2.5,
        outward_overhang_mm=0.0,
        source="parent_local",
        child_index=None,
        strict=True,
    )

    min_pt, max_pt = constraint_aware_outline(
        placed_bboxes=[(Point(0.0, 0.0), Point(40.0, 20.0))],
        attachment_constraints=[constraint],
        constrained_ref_world_anchors={"H1": Point(77.5, 57.5)},
        margin_mm=2.0,
    )

    assert max_pt.x == pytest.approx(80.0)
    assert max_pt.y == pytest.approx(60.0)
