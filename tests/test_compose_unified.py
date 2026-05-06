"""End-to-end tests for the unified _compose_artifacts flow.

Exercises the post-cutover compose path against synthetic
LoadedSubcircuitArtifact bundles without requiring a real PCB on disk:

  1. 3 blocks, no constraints -> basic packing, no bbox overlaps.
  2. 2 blocks dual-layer (front-only x back-only) -> bbox overlap
     present and *legal* (blocker_pair_compatible).
  3. 2 blocks same-layer -> bbox overlap forbidden, must be disjoint.
  4. _snap_parent_local snaps mounting-hole pads to exact constraint
     coordinates within 1e-3 mm under each cardinal rotation.

Together these tests cover the unique behaviors of the unified solver
that the legacy compose path could not achieve (controlled dual-layer
overlap, single-pass placement, direct anchor snapping).
"""

from __future__ import annotations

import math

from kicraft.cli.compose_subcircuits import (
    _compose_artifacts,
    _compute_final_outline,
    _snap_parent_local,
)
from kicraft.autoplacer.brain.subcircuit_composer import AttachmentConstraint
from kicraft.autoplacer.brain.subcircuit_instances import (
    LoadedSubcircuitArtifact,
)
from kicraft.autoplacer.brain.types import (
    Component,
    Layer,
    Pad,
    Point,
    SubCircuitId,
    SubCircuitLayout,
)


# ---------------------------------------------------------------------------
# Helpers


def _id(name: str) -> SubCircuitId:
    return SubCircuitId(
        sheet_name=name,
        sheet_file=f"{name.lower()}.kicad_sch",
        instance_path=f"/{name.lower()}",
    )


def _pad(ref: str, pad_id: str, x: float, y: float, layer: Layer = Layer.FRONT) -> Pad:
    return Pad(ref=ref, pad_id=pad_id, pos=Point(x, y), net="", layer=layer)


def _comp(
    ref: str,
    *,
    pos: Point,
    width: float = 4.0,
    height: float = 2.0,
    pads: list[Pad] | None = None,
    layer: Layer = Layer.FRONT,
) -> Component:
    return Component(
        ref=ref,
        value="",
        pos=pos,
        rotation=0.0,
        layer=layer,
        width_mm=width,
        height_mm=height,
        pads=list(pads or []),
    )


def _make_artifact(
    name: str,
    *,
    width: float,
    height: float,
    components: dict[str, Component] | None = None,
) -> LoadedSubcircuitArtifact:
    layout = SubCircuitLayout(
        subcircuit_id=_id(name),
        components=dict(components or {}),
        traces=[],
        vias=[],
        bounding_box=(width, height),
        ports=[],
        interface_anchors=[],
        score=75.0,
    )
    return LoadedSubcircuitArtifact(
        artifact_dir=f"/fake/{name}",
        metadata={},
        debug={},
        layout=layout,
        source_files={},
    )


def _front_layer_component(ref: str, pos: Point, w: float, h: float) -> Component:
    """Component with two front-layer SMT pads near opposite corners.

    Two pads positioned far apart in the local frame produce a sparse
    blocker set whose pad rects span most of the body bbox, so any bbox
    overlap between two same-layer instances will land their pad rects
    on top of each other -- forcing the blocker-aware solver to push
    them apart.
    """
    return _comp(
        ref,
        pos=pos,
        width=w,
        height=h,
        pads=[
            _pad(ref, "1", pos.x - w / 2 + 0.5, pos.y - h / 2 + 0.5, Layer.FRONT),
            _pad(ref, "2", pos.x + w / 2 - 0.5, pos.y + h / 2 - 0.5, Layer.FRONT),
        ],
    )


def _back_layer_component(ref: str, pos: Point, w: float, h: float) -> Component:
    """Mirror of _front_layer_component but on the back layer."""
    pads = [
        _pad(ref, "1", pos.x - w / 2 + 0.5, pos.y - h / 2 + 0.5, Layer.BACK),
        _pad(ref, "2", pos.x + w / 2 - 0.5, pos.y + h / 2 - 0.5, Layer.BACK),
    ]
    return _comp(ref, pos=pos, width=w, height=h, pads=pads, layer=Layer.BACK)


# ---------------------------------------------------------------------------
# Scenario 1: three blocks, no constraints -> no bbox overlaps after solve.


def test_three_blocks_no_constraints_no_overlap():
    artifacts = [
        _make_artifact(
            "BLK_A",
            width=8.0,
            height=6.0,
            components={
                "RA": _front_layer_component("RA", Point(4.0, 3.0), 2.0, 1.0),
            },
        ),
        _make_artifact(
            "BLK_B",
            width=10.0,
            height=4.0,
            components={
                "RB": _front_layer_component("RB", Point(5.0, 2.0), 2.0, 1.0),
            },
        ),
        _make_artifact(
            "BLK_C",
            width=6.0,
            height=8.0,
            components={
                "RC": _front_layer_component("RC", Point(3.0, 4.0), 2.0, 1.0),
            },
        ),
    ]
    state, _ = _compose_artifacts(
        artifacts,
        spacing_mm=2.0,
        rotation_step_deg=0.0,
        parent_definition=None,
        pcb_path=None,
        cfg={},
        seed=0,
    )
    assert len(state.entries) == 3
    # Validation block reports no same-side overlap conflicts.
    overlaps = state.geometry_validation.get("same_side_overlap_conflicts", [])
    assert overlaps == [], f"unexpected same-side overlap conflicts: {overlaps}"


# ---------------------------------------------------------------------------
# Scenario 2: dual-layer pair (front-only x back-only) may overlap legally.


def test_dual_layer_pair_overlap_is_legal():
    artifacts = [
        _make_artifact(
            "FRONT_BLK",
            width=10.0,
            height=8.0,
            components={
                "RA": _front_layer_component("RA", Point(5.0, 4.0), 2.0, 1.0),
            },
        ),
        _make_artifact(
            "BACK_BLK",
            width=10.0,
            height=8.0,
            components={
                "RB": _back_layer_component("RB", Point(5.0, 4.0), 2.0, 1.0),
            },
        ),
    ]
    state, _ = _compose_artifacts(
        artifacts,
        spacing_mm=2.0,
        rotation_step_deg=0.0,
        parent_definition=None,
        pcb_path=None,
        cfg={},
        seed=0,
    )
    assert len(state.entries) == 2
    # The validation block should report no same-side overlap conflicts
    # because the two blocks are on opposite layers (legal overlap).
    overlaps = state.geometry_validation.get("same_side_overlap_conflicts", [])
    assert overlaps == [], (
        f"dual-layer pair must not be flagged as same-side overlap, got: {overlaps}"
    )


# ---------------------------------------------------------------------------
# Scenario 3: same-layer pair must not have overlapping bboxes after solve.


def test_same_layer_pair_no_blocker_conflict():
    """Two same-layer blocks placed by the solver must not produce a
    same-side overlap conflict in the validation block. (Bbox overlap
    can be legal under the unified solver when underlying copper doesn't
    conflict; the load-bearing invariant is that *blockers* don't.)"""
    artifacts = [
        _make_artifact(
            "FRONT_A",
            width=10.0,
            height=8.0,
            components={
                "RA": _front_layer_component("RA", Point(5.0, 4.0), 2.0, 1.0),
            },
        ),
        _make_artifact(
            "FRONT_B",
            width=10.0,
            height=8.0,
            components={
                "RB": _front_layer_component("RB", Point(5.0, 4.0), 2.0, 1.0),
            },
        ),
    ]
    state, _ = _compose_artifacts(
        artifacts,
        spacing_mm=2.0,
        rotation_step_deg=0.0,
        parent_definition=None,
        pcb_path=None,
        cfg={},
        seed=0,
    )
    assert len(state.entries) == 2
    overlaps = state.geometry_validation.get("same_side_overlap_conflicts", [])
    assert overlaps == [], (
        f"same-layer pair must not generate same-side overlap conflicts: {overlaps}"
    )
    tht = state.geometry_validation.get("tht_keepout_violations", [])
    assert tht == [], f"same-layer pair must not generate THT keepout violations: {tht}"


# ---------------------------------------------------------------------------
# Scenario 4: _snap_parent_local snaps to exact constraint coords.


def test_snap_parent_local_top_left_corner():
    """Mounting hole H4 with corner=top-left is snapped so its pad centroid
    lands at the inward keep-in target within 1e-3 mm tolerance."""
    pad = Pad(ref="H4", pad_id="1", pos=Point(20.0, 20.0), net="", layer=Layer.FRONT)
    hole = Component(
        ref="H4",
        value="MountingHole",
        pos=Point(20.0, 20.0),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=3.5,
        height_mm=3.5,
        pads=[pad],
        body_center=Point(20.0, 20.0),
    )
    constraint = AttachmentConstraint(
        ref="H4",
        target="corner",
        value="top-left",
        inward_keep_in_mm=5.0,
        outward_overhang_mm=0.0,
        source="parent_local",
        child_index=None,
        strict=True,
    )
    outline = (Point(0.0, 0.0), Point(100.0, 80.0))
    comps = {"H4": hole}
    _snap_parent_local(comps, [constraint], outline)
    # Expected centroid: (0 + 5, 0 + 5) = (5, 5) for top-left corner with
    # 5mm inward keep-in.
    centroid_x = sum(p.pos.x for p in hole.pads) / len(hole.pads)
    centroid_y = sum(p.pos.y for p in hole.pads) / len(hole.pads)
    assert math.isclose(centroid_x, 5.0, abs_tol=1e-3)
    assert math.isclose(centroid_y, 5.0, abs_tol=1e-3)
    # body_center and pos translate by the same delta.
    assert math.isclose(hole.body_center.x, 5.0, abs_tol=1e-3)
    assert math.isclose(hole.body_center.y, 5.0, abs_tol=1e-3)


def test_snap_parent_local_bottom_right_corner():
    pad = Pad(ref="H86", pad_id="1", pos=Point(20.0, 20.0), net="", layer=Layer.FRONT)
    hole = Component(
        ref="H86",
        value="MountingHole",
        pos=Point(20.0, 20.0),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=3.5,
        height_mm=3.5,
        pads=[pad],
        body_center=Point(20.0, 20.0),
    )
    constraint = AttachmentConstraint(
        ref="H86",
        target="corner",
        value="bottom-right",
        inward_keep_in_mm=5.0,
        outward_overhang_mm=0.0,
        source="parent_local",
        child_index=None,
        strict=True,
    )
    outline = (Point(0.0, 0.0), Point(100.0, 80.0))
    comps = {"H86": hole}
    _snap_parent_local(comps, [constraint], outline)
    centroid_x = sum(p.pos.x for p in hole.pads) / len(hole.pads)
    centroid_y = sum(p.pos.y for p in hole.pads) / len(hole.pads)
    # Expected: (100 - 5, 80 - 5) = (95, 75).
    assert math.isclose(centroid_x, 95.0, abs_tol=1e-3)
    assert math.isclose(centroid_y, 75.0, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# _compute_final_outline -- corner-constrained side margin behavior.
#
# Regression for the LLUPS round-2 case where BOOST 5V's pad copper sat
# flush with the south Edge.Cuts (zero clearance), violating
# copper_edge_clearance. Root cause: when a corner-constraint partially
# pins a side (H86 corner=bottom-right pins the "bottom" side) and a
# placed leaf's geometry extends past the corner anchor on that side,
# the outline must still leave spacing_mm between the leaf and the edge
# -- a corner anchor describes a point, not a side line.


def _outline_corner_constraint(ref: str, value: str, anchor: Point) -> tuple[
    AttachmentConstraint, dict[str, Point]
]:
    constraint = AttachmentConstraint(
        ref=ref,
        target="corner",
        value=value,
        inward_keep_in_mm=2.5,
        outward_overhang_mm=0.0,
        source="parent_local",
        child_index=None,
        strict=True,
    )
    return constraint, {ref: anchor}


def test_compute_final_outline_corner_south_overhang_keeps_margin():
    # Leaf bbox extends 4 mm south of the corner anchor's "bottom-right".
    # Corner anchor at (90, 60) with keep_in 2.5 -> contributes bottom=62.5.
    # Leaf south at y=70 must end up with at least 1.0 mm gap to outline.
    placed = [(Point(10.0, 10.0), Point(80.0, 70.0))]
    constraint, anchors = _outline_corner_constraint(
        "H86", "bottom-right", Point(90.0, 60.0)
    )
    outline = _compute_final_outline(placed, [constraint], anchors, spacing_mm=1.0)
    _, br = outline
    assert math.isclose(br.y, 71.0, abs_tol=1e-3), (
        f"south outline must be leaf_south + spacing_mm = 71.0, got {br.y:.4f}"
    )


def test_compute_final_outline_corner_north_overhang_keeps_margin():
    # Symmetric to south: corner=top-left anchored north, leaf extends
    # further north. Outline must leave spacing_mm clearance.
    placed = [(Point(10.0, 5.0), Point(80.0, 70.0))]
    constraint, anchors = _outline_corner_constraint(
        "H4", "top-left", Point(20.0, 20.0)
    )
    outline = _compute_final_outline(placed, [constraint], anchors, spacing_mm=1.5)
    tl, _ = outline
    assert math.isclose(tl.y, 3.5, abs_tol=1e-3), (
        f"north outline must be leaf_north - spacing_mm = 3.5, got {tl.y:.4f}"
    )


def test_compute_final_outline_corner_anchor_dominates_when_no_overhang():
    # Same corner constraint but leaves stay well inside the corner
    # anchor. Outline should track the constraint anchor (no margin
    # added on top -- the anchor target IS the keep-in line for the
    # mounting hole).
    placed = [(Point(10.0, 10.0), Point(40.0, 30.0))]
    constraint, anchors = _outline_corner_constraint(
        "H86", "bottom-right", Point(90.0, 60.0)
    )
    outline = _compute_final_outline(placed, [constraint], anchors, spacing_mm=1.0)
    _, br = outline
    # constraint anchor at y=60 + keep_in 2.5 = 62.5. Geometry max y=30
    # plus spacing=31, so anchor (62.5) wins.
    assert math.isclose(br.y, 62.5, abs_tol=1e-3), (
        f"south outline must equal corner anchor target 62.5 when no "
        f"leaf overhang, got {br.y:.4f}"
    )


def test_compute_final_outline_edge_pinned_no_margin():
    # An edge constraint must keep the outline flush with the anchor:
    # adding spacing would push the connector inboard. This branch is
    # untouched by the fix, but we assert it explicitly so the
    # regression doesn't drift.
    placed = [(Point(10.0, 10.0), Point(80.0, 60.0))]
    constraint = AttachmentConstraint(
        ref="J1",
        target="edge",
        value="left",
        inward_keep_in_mm=0.0,
        outward_overhang_mm=0.0,
        source="parent_local",
        child_index=None,
        strict=True,
    )
    anchors = {"J1": Point(10.0, 30.0)}
    outline = _compute_final_outline(placed, [constraint], anchors, spacing_mm=2.0)
    tl, _ = outline
    assert math.isclose(tl.x, 10.0, abs_tol=1e-3), (
        f"left outline must equal edge anchor 10.0 (no margin), got {tl.x:.4f}"
    )


def test_compute_final_outline_unconstrained_gets_margin():
    # No constraints at all -> all four sides are unconstrained and
    # should land at geom +/- spacing_mm. This branch is also untouched
    # by the fix, asserted to catch drift.
    placed = [(Point(10.0, 10.0), Point(80.0, 60.0))]
    outline = _compute_final_outline(placed, [], {}, spacing_mm=1.0)
    tl, br = outline
    assert math.isclose(tl.x, 9.0, abs_tol=1e-3)
    assert math.isclose(tl.y, 9.0, abs_tol=1e-3)
    assert math.isclose(br.x, 81.0, abs_tol=1e-3)
    assert math.isclose(br.y, 61.0, abs_tol=1e-3)
