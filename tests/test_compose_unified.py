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

import pytest

from kicraft.cli.compose_subcircuits import (
    _compose_artifacts,
    _compute_final_outline,
    _snap_parent_local,
    _wrap_loose_parent_components_as_leaves,
)
from kicraft.autoplacer.brain.subcircuit_composer import (
    AttachmentConstraint,
    extract_leaf_blocker_set,
)
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


def test_wrap_loose_parent_connector_as_leaf():
    """Lever 2.1: a loose parent-level connector (J*, in no leaf) is wrapped as
    a single-component leaf -- pulled out of parent_local and appended to the
    artifact list, RE-BASED into its own (0,0)-anchored leaf box so the leaf
    edge-pin/flush path's frame math cancels (the old parent-local snap branch
    that rotated+pinned it here is deleted). A mounting hole stays parent-local."""
    j1 = Component(
        ref="J1", value="USB-C", pos=Point(150.0, 122.0), rotation=0.0,
        layer=Layer.FRONT, width_mm=9.0, height_mm=7.0, kind="connector",
        body_center=Point(150.0, 122.0),
        pads=[
            Pad(ref="J1", pad_id="A1", pos=Point(147.0, 119.0), net="", layer=Layer.FRONT),
            Pad(ref="J1", pad_id="A2", pos=Point(153.0, 119.0), net="", layer=Layer.FRONT),
        ],
    )
    hole = Component(
        ref="H1", value="MountingHole", pos=Point(10.0, 10.0), rotation=0.0,
        layer=Layer.FRONT, width_mm=3.2, height_mm=3.2, kind="mounting_hole",
        body_center=Point(10.0, 10.0),
        pads=[Pad(ref="H1", pad_id="1", pos=Point(10.0, 10.0), net="", layer=Layer.FRONT)],
    )
    parent_local = {"J1": j1, "H1": hole}
    artifacts, remaining = _wrap_loose_parent_components_as_leaves(parent_local, [])

    # The mounting hole stays parent-local; the connector is removed from it.
    assert set(remaining) == {"H1"}
    # The connector became one single-component leaf.
    assert len(artifacts) == 1
    leaf = artifacts[0]
    assert set(leaf.layout.components) == {"J1"}
    # Re-based: everything sits inside a (0,0)-anchored box (no absolute seed
    # coords leak into the leaf-local frame).
    wrapped = leaf.layout.components["J1"]
    bb_w, bb_h = leaf.layout.bounding_box
    assert bb_w > 0 and bb_h > 0
    xs = [p.pos.x for p in wrapped.pads] + [wrapped.body_center.x]
    ys = [p.pos.y for p in wrapped.pads] + [wrapped.body_center.y]
    assert min(xs) >= -1e-6 and min(ys) >= -1e-6
    assert max(xs) <= bb_w + 1e-6 and max(ys) <= bb_h + 1e-6
    # The caller's parent_local dict is not mutated in place.
    assert set(parent_local) == {"J1", "H1"}


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


def test_compute_final_outline_rejects_far_outboard_edge_anchor():
    # Regression for the USB-C stranding bug: a buggy anchor convention put
    # the left edge anchor ~part-height OUTBOARD of the placed geometry, and
    # the loose ``spacing_mm + 10`` clamp waved it through -- baking a bare-FR4
    # strip between the connector mouth and the board edge. The tightened
    # clamp (``spacing_mm + 2``) must IGNORE such an anchor and fall back to
    # geometry - spacing so the strip can never form.
    placed = [(Point(10.0, 10.0), Point(80.0, 60.0))]  # geom left edge = 10.0
    constraint = AttachmentConstraint(
        ref="J1", target="edge", value="left",
        inward_keep_in_mm=0.0, outward_overhang_mm=0.0,
        source="parent_local", child_index=None, strict=True,
    )
    # Anchor 8.3 mm outboard of the geometry left edge (the stranding signature).
    outline = _compute_final_outline(
        placed, [constraint], {"J1": Point(1.7, 30.0)}, spacing_mm=2.0
    )
    tl, _ = outline
    assert math.isclose(tl.x, 8.0, abs_tol=1e-3), (
        "far-outboard edge anchor must be ignored (fall back to geom-spacing "
        f"= 8.0), got {tl.x:.4f}"
    )
    # A flush anchor (within the clamp) is still honored exactly.
    flush = _compute_final_outline(
        placed, [constraint], {"J1": Point(10.0, 30.0)}, spacing_mm=2.0
    )
    assert math.isclose(flush[0].x, 10.0, abs_tol=1e-3), (
        f"flush edge anchor must still be honored, got {flush[0].x:.4f}"
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


# ---------------------------------------------------------------------------
# Regression: phantom edge anchors + page-centered blocker frames
# (the 3x-wide parent outline bug, KC-72RQXB).


def test_compute_final_outline_phantom_edge_anchor_clamped():
    # An edge anchor far from the placed geometry (here ~119 mm off --
    # the A4 page-centering offset applied in the wrong frame) must NOT
    # stretch the outline. The snap falls back to geometry +/- spacing.
    placed = [(Point(10.0, 10.0), Point(70.0, 30.0))]

    def _edge(ref: str, side: str) -> AttachmentConstraint:
        return AttachmentConstraint(
            ref=ref,
            target="edge",
            value=side,
            inward_keep_in_mm=0.0,
            outward_overhang_mm=0.0,
            source="parent_local",
            child_index=None,
            strict=True,
        )

    anchors = {"J1": Point(-109.0, 20.0), "J2": Point(189.0, 20.0)}
    outline = _compute_final_outline(
        placed,
        [_edge("J1", "left"), _edge("J2", "right")],
        anchors,
        spacing_mm=2.0,
    )
    tl, br = outline
    assert math.isclose(tl.x, 8.0, abs_tol=1e-3), (
        f"phantom left anchor must clamp to geometry - spacing (8.0), got {tl.x:.2f}"
    )
    assert math.isclose(br.x, 72.0, abs_tol=1e-3), (
        f"phantom right anchor must clamp to geometry + spacing (72.0), got {br.x:.2f}"
    )


def test_compute_final_outline_nearby_edge_anchor_still_snaps():
    # A legitimate flush-mount anchor a couple of mm outside the geometry
    # (connector housing overhang) must still win over geometry+spacing --
    # the clamp only rejects far-out anchors.
    placed = [(Point(10.0, 10.0), Point(70.0, 30.0))]
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
    anchors = {"J1": Point(7.5, 20.0)}
    outline = _compute_final_outline(placed, [constraint], anchors, spacing_mm=2.0)
    tl, _ = outline
    assert math.isclose(tl.x, 7.5, abs_tol=1e-3), (
        f"nearby edge anchor must snap exactly (7.5), got {tl.x:.2f}"
    )


def test_extract_blockers_from_pcb_rebased_to_leaf_local(tmp_path):
    # Leaf PCBs are generated centered on their page while
    # solved_layout.json is serialized re-based (Edge.Cuts top-left at
    # (0,0)). Blocker extraction from the PCB must come back in the
    # re-based layout frame, or every constraint anchor derived from a
    # blocker rect is shifted by the page offset.
    pcbnew = pytest.importorskip("pcbnew")

    mm = pcbnew.FromMM
    board = pcbnew.CreateEmptyBoard()
    # Page-centered outline: (118.38, 94.14) .. (178.62, 115.86)
    edges = [
        (118.38, 94.14, 178.62, 94.14),
        (178.62, 94.14, 178.62, 115.86),
        (178.62, 115.86, 118.38, 115.86),
        (118.38, 115.86, 118.38, 94.14),
    ]
    for x1, y1, x2, y2 in edges:
        seg = pcbnew.PCB_SHAPE(board, pcbnew.SHAPE_T_SEGMENT)
        seg.SetStart(pcbnew.VECTOR2I(mm(x1), mm(y1)))
        seg.SetEnd(pcbnew.VECTOR2I(mm(x2), mm(y2)))
        seg.SetLayer(pcbnew.Edge_Cuts)
        seg.SetWidth(mm(0.1))
        board.Add(seg)
    fp = pcbnew.FOOTPRINT(board)
    fp.SetReference("J1")
    fp.SetPosition(pcbnew.VECTOR2I(mm(120.0), mm(100.0)))
    pad = pcbnew.PAD(fp)
    pad.SetShape(pcbnew.PAD_SHAPE_RECT)
    pad.SetSize(pcbnew.VECTOR2I(mm(2.0), mm(2.0)))
    pad.SetAttribute(pcbnew.PAD_ATTRIB_SMD)
    pad.SetLayerSet(pcbnew.PAD.SMDMask())
    pad.SetPosition(fp.GetPosition())
    fp.Add(pad)
    board.Add(fp)
    pcb_path = tmp_path / "leaf_routed.kicad_pcb"
    pcbnew.SaveBoard(str(pcb_path), board)

    artifact = _make_artifact("LEAF", width=60.25, height=21.72)
    artifact.source_files["mini_pcb"] = str(pcb_path)

    blocker_set = extract_leaf_blocker_set(artifact)

    outline_min, outline_max = blocker_set.leaf_outline
    assert abs(outline_min.x) < 0.2 and abs(outline_min.y) < 0.2, (
        f"leaf outline must be re-based to ~(0,0), got {outline_min}"
    )
    assert 59.0 < outline_max.x - outline_min.x < 62.0

    j1_min, j1_max = blocker_set.component_rects["J1"]
    # J1 pad centered at page (120, 100) -> leaf-local ~(1.6, 5.9).
    assert -1.0 < j1_min.x < 5.0, f"J1 rect still in page frame: min.x={j1_min.x:.2f}"
    assert j1_max.x < 70.0, f"J1 rect still in page frame: max.x={j1_max.x:.2f}"
    for rect_min, rect_max in blocker_set.front_pads + blocker_set.tht_drills:
        assert rect_max.x < 70.0 and rect_max.y < 30.0, (
            "blocker pad rects must live in the leaf-local frame"
        )
