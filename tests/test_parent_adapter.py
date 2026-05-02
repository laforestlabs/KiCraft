"""Unit tests for parent_adapter.py: parent-side <-> solver-side bridge.

The parent_adapter module is dormant scaffolding in PR4 -- no callers
are wired up. These tests verify the math:

  - rotation-aware artifact origin recovery
  - anchor_offset_mm computation in unrotated local frame
  - synthetic-ref rewriting in pre-placement net inference
"""

from __future__ import annotations

import math

import pytest

from kicraft.autoplacer.brain.parent_adapter import (
    artifact_to_component,
    attachment_constraints_to_zones,
    infer_interconnect_nets_pre_placement,
    placements_from_solved_state,
    synthetic_block_ref,
)
from kicraft.autoplacer.brain.subcircuit_composer import (
    AttachmentConstraint,
    DerivedAttachmentConstraints,
    PlacementSpec,
    derive_attachment_constraints,
)
from kicraft.autoplacer.brain.subcircuit_instances import (
    LoadedSubcircuitArtifact,
)
from kicraft.autoplacer.brain.types import (
    Component,
    InterfaceAnchor,
    InterfaceDirection,
    InterfacePort,
    InterfaceRole,
    InterfaceSide,
    Layer,
    Pad,
    Point,
    SubCircuitDefinition,
    SubCircuitId,
    SubCircuitLayout,
    SubcircuitAccessPolicy,
)


# ---------------------------------------------------------------------------
# Synthetic artifact builders


def _id(sheet: str) -> SubCircuitId:
    return SubCircuitId(
        sheet_name=sheet,
        sheet_file=f"{sheet.lower()}.kicad_sch",
        instance_path=f"/{sheet.lower()}",
    )


def _pad(ref: str, pad_id: str, x: float, y: float, net: str = "") -> Pad:
    return Pad(ref=ref, pad_id=pad_id, pos=Point(x, y), net=net, layer=Layer.FRONT)


def _comp(
    ref: str,
    *,
    pos: Point,
    width: float = 4.0,
    height: float = 2.0,
    pads: list[Pad] | None = None,
    is_through_hole: bool = False,
    kind: str = "",
) -> Component:
    return Component(
        ref=ref,
        value="",
        pos=pos,
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=width,
        height_mm=height,
        pads=list(pads or []),
        kind=kind,
        is_through_hole=is_through_hole,
    )


def _layout(
    sheet: str,
    *,
    width: float,
    height: float,
    components: dict[str, Component] | None = None,
    ports: list[InterfacePort] | None = None,
    anchors: list[InterfaceAnchor] | None = None,
) -> SubCircuitLayout:
    return SubCircuitLayout(
        subcircuit_id=_id(sheet),
        components=dict(components or {}),
        traces=[],
        vias=[],
        bounding_box=(width, height),
        ports=list(ports or []),
        interface_anchors=list(anchors or []),
        score=75.0,
    )


def _artifact(layout: SubCircuitLayout) -> LoadedSubcircuitArtifact:
    return LoadedSubcircuitArtifact(
        artifact_dir="/fake",
        metadata={},
        debug={},
        layout=layout,
        source_files={},
    )


def _port(name: str, net_name: str) -> InterfacePort:
    return InterfacePort(
        name=name,
        net_name=net_name,
        role=InterfaceRole.UNKNOWN,
        direction=InterfaceDirection.PASSIVE,
        preferred_side=InterfaceSide.ANY,
        access_policy=SubcircuitAccessPolicy.INTERFACE_ONLY,
        cardinality=1,
        bus_index=None,
        required=True,
        description="",
    )


# ---------------------------------------------------------------------------
# synthetic_block_ref


def test_synthetic_block_ref_strips_unsafe_chars():
    assert synthetic_block_ref(0, "USB Port") == "BLOCK_0_USB_Port"
    assert synthetic_block_ref(2, "Power/3.3V") == "BLOCK_2_Power_3_3V"


def test_synthetic_block_ref_handles_empty_name():
    assert synthetic_block_ref(7, "") == "BLOCK_7_child"


# ---------------------------------------------------------------------------
# artifact_to_component


def test_artifact_to_component_populates_block_metadata():
    """A single component fills the layout's body extent. The synthetic
    block's width/height/body_center should track that content extent --
    not the leaf-board outline (which can be larger when leaves have
    padding around their components)."""
    layout = _layout(
        "USB",
        width=10.0,
        height=8.0,
        components={
            "J1": _comp(
                "J1",
                pos=Point(5.0, 4.0),
                width=10.0,
                height=8.0,
                pads=[_pad("J1", "1", 1.0, 4.0, "VBUS"), _pad("J1", "2", 9.0, 4.0, "GND")],
            )
        },
    )
    comp = artifact_to_component(_artifact(layout), ref="BLOCK_0_USB", rotation=0.0)

    assert comp.ref == "BLOCK_0_USB"
    assert comp.kind == "subcircuit"
    # Content extent (J1 spans the full 10 x 8) plus a 2 mm safety margin
    # per side so the placer's bbox-overlap resolver leaves a routing
    # channel between adjacent leaves.
    assert comp.width_mm == 14.0
    assert comp.height_mm == 12.0
    assert comp.block_blocker_set is not None
    assert comp.block_artifact_origin_offset == Point(5.0, 4.0)
    assert comp.block_side in {"front", "back", "dual", "none"}
    assert comp.allowed_rotations is None  # populated separately


# ---------------------------------------------------------------------------
# placements_from_solved_state: round-trip rotation math


@pytest.mark.parametrize("rotation", [0.0, 90.0, 180.0, 270.0])
def test_placements_round_trip_each_rotation(rotation):
    """When the solver places the body center at world position P, rotated
    by r, the recovered artifact origin must satisfy
    P = origin + rotated(content_center, r). This test exercises the round
    trip for each cardinal rotation. The synthetic layout has a single
    component spanning the full 10 x 8 area so content bbox equals the
    leaf bbox (centered at (5, 4))."""
    layout = _layout(
        "LEAF",
        width=10.0,
        height=8.0,
        components={
            "U1": _comp(
                "U1",
                pos=Point(5.0, 4.0),
                width=10.0,
                height=8.0,
                pads=[_pad("U1", "1", 5.0, 4.0)],
            )
        },
    )
    artifact = _artifact(layout)
    refs = {0: "BLOCK_0_LEAF"}

    block_pos = Point(50.0, 30.0)
    comp = artifact_to_component(artifact, ref="BLOCK_0_LEAF", rotation=rotation)
    comp.pos = Point(block_pos.x, block_pos.y)

    placements = placements_from_solved_state(
        {"BLOCK_0_LEAF": comp}, [artifact], refs
    )
    assert "/leaf" in placements
    placement = placements["/leaf"]
    assert placement.rotation == rotation

    # Verify forward direction: body_center = origin + rotated(content_center, rot)
    rad = math.radians(rotation)
    cos_r, sin_r = math.cos(rad), math.sin(rad)
    expected_body_x = placement.origin.x + (5.0 * cos_r - 4.0 * sin_r)
    expected_body_y = placement.origin.y + (5.0 * sin_r + 4.0 * cos_r)
    assert math.isclose(expected_body_x, block_pos.x, abs_tol=1e-9)
    assert math.isclose(expected_body_y, block_pos.y, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# attachment_constraints_to_zones: anchor_offset_mm in unrotated frame


def _build_derived_for_left_edge(layout: SubCircuitLayout) -> DerivedAttachmentConstraints:
    artifact = _artifact(layout)
    cfg = {"connector_edge_inset_mm": 1.0}
    return derive_attachment_constraints(
        [artifact],
        parent_local_components={},
        component_zones={"J1": "edge:left"},
        cfg=cfg,
        rotation_step_deg=0.0,
    )


def test_attachment_constraints_to_zones_emits_left_edge_anchor():
    """Left-edge constraint on J1 produces a zone with edge=left and an
    anchor_offset_mm that points from the body center to the local J1 pad
    cluster (which lives at x ~ 1.0, the left side)."""
    layout = _layout(
        "USB",
        width=10.0,
        height=8.0,
        components={
            "J1": _comp(
                "J1",
                pos=Point(2.0, 4.0),
                width=4.0,
                height=2.0,
                pads=[_pad("J1", "1", 1.0, 4.0, "VBUS")],
                kind="connector",
            )
        },
    )
    derived = _build_derived_for_left_edge(layout)
    artifacts = [_artifact(layout)]
    refs = {0: "BLOCK_0_USB"}

    zones, allowed = attachment_constraints_to_zones(derived, refs, artifacts)
    assert "BLOCK_0_USB" in zones
    entry = zones["BLOCK_0_USB"]
    assert entry["edge"] == "left"
    assert entry["rotation"] == 0.0
    # body_center = (5, 4); local anchor is on the left side of J1's pads.
    # anchor_offset_mm.x must be negative (anchor is left of body center).
    assert entry["anchor_offset_mm"].x < 0.0
    # allowed_rotations covers the four cardinal angles.
    assert sorted(allowed["BLOCK_0_USB"]) == [0.0, 90.0, 180.0, 270.0]


def test_attachment_constraints_to_zones_skips_unmapped_children():
    """Children without a synthetic ref entry in the map are silently
    skipped -- the adapter is permissive, so partial maps are valid."""
    layout = _layout(
        "USB",
        width=10.0,
        height=8.0,
        components={
            "J1": _comp(
                "J1",
                pos=Point(2.0, 4.0),
                width=4.0,
                height=2.0,
                pads=[_pad("J1", "1", 1.0, 4.0)],
                kind="connector",
            )
        },
    )
    derived = _build_derived_for_left_edge(layout)
    artifacts = [_artifact(layout)]

    zones, allowed = attachment_constraints_to_zones(derived, {}, artifacts)
    assert zones == {}
    assert allowed == {}


def test_attachment_constraints_to_zones_anchor_is_unrotated():
    """anchor_offset_mm must be in unrotated local frame regardless of the
    chosen rotation candidate -- the solver applies the rotation per
    comp.rotation at placement time. Rotating the layout's chosen base
    rotation should NOT pre-rotate the offset."""
    layout = _layout(
        "USB",
        width=10.0,
        height=8.0,
        components={
            "J1": _comp(
                "J1",
                pos=Point(2.0, 4.0),
                width=4.0,
                height=2.0,
                pads=[_pad("J1", "1", 1.0, 4.0)],
                kind="connector",
            )
        },
    )

    artifact = _artifact(layout)
    refs = {0: "BLOCK_0_USB"}

    # Build derived with rotation_step_deg=0 -> base rotation 0.
    derived_at_0 = derive_attachment_constraints(
        [artifact],
        parent_local_components={},
        component_zones={"J1": "edge:left"},
        cfg={"connector_edge_inset_mm": 1.0},
        rotation_step_deg=0.0,
    )

    # Build derived with rotation_step_deg=90 -> base rotation 90.
    derived_at_90 = derive_attachment_constraints(
        [artifact],
        parent_local_components={},
        component_zones={"J1": "edge:left"},
        cfg={"connector_edge_inset_mm": 1.0},
        rotation_step_deg=90.0,
    )

    zones_0, _ = attachment_constraints_to_zones(derived_at_0, refs, [artifact])
    zones_90, _ = attachment_constraints_to_zones(derived_at_90, refs, [artifact])

    # The offset ITSELF (unrotated) must match across rotation_step_deg
    # choices: it is purely a function of the layout's local geometry,
    # not of the rotation candidate enumeration. The solver applies the
    # rotation per comp.rotation at placement time.
    offset_0 = zones_0["BLOCK_0_USB"]["anchor_offset_mm"]
    offset_90 = zones_90["BLOCK_0_USB"]["anchor_offset_mm"]
    assert math.isclose(offset_0.x, offset_90.x, abs_tol=1e-9)
    assert math.isclose(offset_0.y, offset_90.y, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# infer_interconnect_nets_pre_placement: synthetic-ref rewriting


def test_pre_placement_nets_rewrite_pad_refs_to_synthetic():
    """A net with leaf pad refs (J1, R5) on different children must come
    out keyed off the corresponding synthetic block refs."""
    layout_a = _layout(
        "USB",
        width=10.0,
        height=8.0,
        components={
            "J1": _comp(
                "J1",
                pos=Point(2.0, 4.0),
                width=4.0,
                height=2.0,
                pads=[_pad("J1", "1", 1.0, 4.0, net="VBUS")],
            )
        },
        ports=[_port("vbus_out", "VBUS")],
        anchors=[InterfaceAnchor(port_name="vbus_out", pos=Point(1.0, 4.0), pad_ref=("J1", "1"))],
    )
    layout_b = _layout(
        "REG",
        width=8.0,
        height=6.0,
        components={
            "U1": _comp(
                "U1",
                pos=Point(4.0, 3.0),
                width=4.0,
                height=2.0,
                pads=[_pad("U1", "1", 3.0, 3.0, net="VBUS")],
            )
        },
        ports=[_port("vbus_in", "VBUS")],
        anchors=[InterfaceAnchor(port_name="vbus_in", pos=Point(3.0, 3.0), pad_ref=("U1", "1"))],
    )
    parent_def = SubCircuitDefinition(
        id=_id("PARENT"),
        schematic_path="",
        component_refs=[],
        ports=[],
        child_ids=[layout_a.subcircuit_id, layout_b.subcircuit_id],
        parent_id=None,
        is_leaf=False,
    )
    refs = {0: "BLOCK_0_USB", 1: "BLOCK_1_REG"}
    nets = infer_interconnect_nets_pre_placement(
        parent_def, [_artifact(layout_a), _artifact(layout_b)], refs
    )
    assert "VBUS" in nets
    pad_owners = {ref for ref, _ in nets["VBUS"].pad_refs}
    assert pad_owners == {"BLOCK_0_USB", "BLOCK_1_REG"}


def test_pre_placement_nets_drops_singleton_after_rewrite():
    """If two leaf refs would collapse into one block ref (both pads on
    the same block), the resulting net has only one block-level
    contributor and is dropped per the >=2-ref filter."""
    layout = _layout(
        "USB",
        width=10.0,
        height=8.0,
        components={
            "J1": _comp(
                "J1",
                pos=Point(2.0, 4.0),
                width=4.0,
                height=2.0,
                pads=[
                    _pad("J1", "1", 1.0, 4.0, net="GND"),
                    _pad("J1", "2", 3.0, 4.0, net="GND"),
                ],
            )
        },
        ports=[_port("gnd", "GND")],
        anchors=[InterfaceAnchor(port_name="gnd", pos=Point(1.0, 4.0), pad_ref=("J1", "1"))],
    )
    parent_def = SubCircuitDefinition(
        id=_id("PARENT"),
        schematic_path="",
        component_refs=[],
        ports=[],
        child_ids=[layout.subcircuit_id],
        parent_id=None,
        is_leaf=False,
    )
    nets = infer_interconnect_nets_pre_placement(
        parent_def, [_artifact(layout)], {0: "BLOCK_0_USB"}
    )
    assert "GND" not in nets
