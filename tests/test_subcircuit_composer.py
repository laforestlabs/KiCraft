"""Tests for kicraft.autoplacer.brain.subcircuit_composer -- parent composition.

All tests use synthetic/mock data only; no pcbnew dependency.
"""

from __future__ import annotations

import pytest

from kicraft.cli.compose_subcircuits import _resolve_constraint_anchor_positions
from kicraft.autoplacer.brain import geometry
from kicraft.autoplacer.brain.subcircuit_composer import (
    AttachmentConstraint,
    ChildArtifactPlacement,
    ChildPlacement,
    LeafBlockerSet,
    PlacementConstraintEntry,
    PlacementModel,
    _compute_local_anchor_offset,
    _transform_local_point,
    build_parent_composition,
    derive_attachment_constraints,
    child_layer_envelopes,
    composition_summary,
    estimate_layer_aware_parent_board_size,
    estimate_parent_board_size,
    packed_extents_outline,
    _derive_board_outline,
)
from kicraft.autoplacer.brain.subcircuit_instances import (
    LoadedSubcircuitArtifact,
    TransformedSubcircuit,
    transform_loaded_artifact,
)
from kicraft.autoplacer.brain.types import (
    AntennaEdgeIntent,
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
    SubCircuitInstance,
    SubCircuitLayout,
    SubcircuitAccessPolicy,
    TraceSegment,
    Via,
)
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_subcircuit_id(name: str) -> SubCircuitId:
    return SubCircuitId(
        sheet_name=name,
        sheet_file=f"{name.lower()}.kicad_sch",
        instance_path=f"/{name.lower()}",
    )


def _make_port(
    name: str,
    net_name: str,
    role: InterfaceRole = InterfaceRole.UNKNOWN,
) -> InterfacePort:
    return InterfacePort(
        name=name,
        net_name=net_name,
        role=role,
        direction=InterfaceDirection.PASSIVE,
        preferred_side=InterfaceSide.ANY,
        access_policy=SubcircuitAccessPolicy.INTERFACE_ONLY,
        cardinality=1,
        bus_index=None,
        required=True,
        description="",
    )


def _make_pad(ref: str, pad_id: str, x: float, y: float, net: str) -> Pad:
    return Pad(ref=ref, pad_id=pad_id, pos=Point(x, y), net=net, layer=Layer.FRONT)


def _make_component(
    ref: str,
    x: float,
    y: float,
    w: float = 2.0,
    h: float = 1.0,
    pads: list[Pad] | None = None,
) -> Component:
    return Component(
        ref=ref,
        value="10k",
        pos=Point(x, y),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=w,
        height_mm=h,
        pads=list(pads or []),
    )


def _make_blocker_set(
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


def _make_layout(
    name: str,
    components: dict[str, Component],
    traces: list[TraceSegment] | None = None,
    vias: list[Via] | None = None,
    anchors: list[InterfaceAnchor] | None = None,
    ports: list[InterfacePort] | None = None,
    bbox: tuple[float, float] | None = None,
) -> SubCircuitLayout:
    """Build a SubCircuitLayout from synthetic data."""
    sc_id = _make_subcircuit_id(name)

    # Auto-derive bbox from components if not given
    if bbox is None:
        if components:
            max_x = max(c.pos.x + c.width_mm / 2 for c in components.values())
            max_y = max(c.pos.y + c.height_mm / 2 for c in components.values())
            bbox = (max_x, max_y)
        else:
            bbox = (0.0, 0.0)

    return SubCircuitLayout(
        subcircuit_id=sc_id,
        components=dict(components),
        traces=list(traces or []),
        vias=list(vias or []),
        bounding_box=bbox,
        ports=list(ports or []),
        interface_anchors=list(anchors or []),
        score=75.0,
    )


def _make_parent_def(
    name: str = "PARENT",
    child_names: list[str] | None = None,
) -> SubCircuitDefinition:
    child_ids = [_make_subcircuit_id(n) for n in (child_names or [])]
    return SubCircuitDefinition(
        id=_make_subcircuit_id(name),
        schematic_path=f"/fake/{name.lower()}.kicad_sch",
        component_refs=[],
        ports=[],
        child_ids=child_ids,
        parent_id=None,
        is_leaf=False,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestComposeTwoChildren:
    """test_compose_two_children -- compose 2 child layouts, verify merged."""

    def test_compose_two_children(self):
        pad_a1 = _make_pad("R1", "1", 1.0, 1.0, "NET_A")
        pad_a2 = _make_pad("R1", "2", 3.0, 1.0, "SHARED")
        comp_a = _make_component("R1", 2.0, 1.0, pads=[pad_a1, pad_a2])
        layout_a = _make_layout("CHILD_A", {"R1": comp_a})

        pad_b1 = _make_pad("C1", "1", 1.0, 1.0, "NET_B")
        pad_b2 = _make_pad("C1", "2", 3.0, 1.0, "SHARED")
        comp_b = _make_component("C1", 2.0, 1.0, pads=[pad_b1, pad_b2])
        layout_b = _make_layout("CHILD_B", {"C1": comp_b})

        parent = _make_parent_def("PARENT", ["CHILD_A", "CHILD_B"])

        result = build_parent_composition(
            parent,
            child_placements=[
                ChildPlacement(layout=layout_a, origin=Point(0.0, 0.0)),
                ChildPlacement(layout=layout_b, origin=Point(10.0, 0.0)),
            ],
        )

        assert result.child_count == 2
        # Both child components appear in merged state
        assert "R1" in result.board_state.components
        assert "C1" in result.board_state.components
        assert result.component_count == 2


class TestChildCopperPreserved:
    """test_child_copper_preserved -- child traces/vias appear in merged state."""

    def test_child_copper_preserved(self):
        pad1 = _make_pad("R1", "1", 1.0, 1.0, "N")
        pad2 = _make_pad("R1", "2", 5.0, 1.0, "N")
        comp = _make_component("R1", 3.0, 1.0, w=4.0, pads=[pad1, pad2])

        trace = TraceSegment(
            start=Point(1.0, 1.0), end=Point(5.0, 1.0),
            layer=Layer.FRONT, net="N", width_mm=0.2,
        )
        via = Via(pos=Point(3.0, 1.0), net="N")

        layout = _make_layout("CHILD", {"R1": comp}, traces=[trace], vias=[via])
        parent = _make_parent_def("P", ["CHILD"])

        result = build_parent_composition(
            parent,
            child_placements=[
                ChildPlacement(layout=layout, origin=Point(0.0, 0.0)),
            ],
        )

        # The child trace should be preserved (possibly transformed)
        assert result.trace_count >= 1
        assert result.via_count >= 1


class TestNoRefCollision:
    """test_no_ref_collision -- overlapping refs raises ValueError."""

    def test_no_ref_collision(self):
        comp1 = _make_component("R1", 2.0, 1.0)
        comp2 = _make_component("R1", 12.0, 1.0)  # Same ref!

        layout_a = _make_layout("CHILD_A", {"R1": comp1})
        layout_b = _make_layout("CHILD_B", {"R1": comp2})

        parent = _make_parent_def("P", ["CHILD_A", "CHILD_B"])

        with pytest.raises(ValueError, match="[Cc]ollision|[Dd]uplicate|R1"):
            build_parent_composition(
                parent,
                child_placements=[
                    ChildPlacement(layout=layout_a, origin=Point(0.0, 0.0)),
                    ChildPlacement(layout=layout_b, origin=Point(20.0, 0.0)),
                ],
            )


class TestDeriveBoardOutline:
    """test_derive_board_outline -- outline encompasses all children with margin."""

    def test_derive_board_outline(self):
        comp_a = _make_component("R1", 5.0, 5.0, w=4.0, h=2.0)
        comp_b = _make_component("C1", 20.0, 15.0, w=3.0, h=2.0)

        outline = _derive_board_outline(
            {"R1": comp_a, "C1": comp_b},
            traces=[],
            vias=[],
            child_anchor_maps={},
            margin_mm=2.0,
        )

        tl, br = outline
        # R1 bbox: (3, 4) to (7, 6)
        # C1 bbox: (18.5, 14) to (21.5, 16)
        # With margin=2: tl ~= (1, 2), br ~= (23.5, 18)
        assert tl.x == pytest.approx(1.0)
        assert tl.y == pytest.approx(2.0)
        assert br.x == pytest.approx(23.5)
        assert br.y == pytest.approx(18.0)




class TestInferInterconnectNets:
    """test_infer_interconnect_nets -- matching ports create interconnects."""

    def test_infer_interconnect_nets(self):
        port_a = _make_port("VOUT", "VOUT")
        port_b = _make_port("VOUT", "VOUT")

        pad_a = _make_pad("R1", "2", 3.0, 1.0, "VOUT")
        comp_a = _make_component("R1", 2.0, 1.0, pads=[pad_a])
        anchor_a = InterfaceAnchor(
            port_name="VOUT", pos=Point(3.0, 1.0),
            layer=Layer.FRONT, pad_ref=("R1", "2"),
        )
        layout_a = _make_layout(
            "CHILD_A", {"R1": comp_a},
            ports=[port_a], anchors=[anchor_a],
        )

        pad_b = _make_pad("C1", "1", 1.0, 1.0, "VOUT")
        comp_b = _make_component("C1", 2.0, 1.0, pads=[pad_b])
        anchor_b = InterfaceAnchor(
            port_name="VOUT", pos=Point(1.0, 1.0),
            layer=Layer.FRONT, pad_ref=("C1", "1"),
        )
        layout_b = _make_layout(
            "CHILD_B", {"C1": comp_b},
            ports=[port_b], anchors=[anchor_b],
        )

        parent = _make_parent_def("P", ["CHILD_A", "CHILD_B"])

        # Compose with origins spaced apart
        result = build_parent_composition(
            parent,
            child_placements=[
                ChildPlacement(layout=layout_a, origin=Point(0.0, 0.0)),
                ChildPlacement(layout=layout_b, origin=Point(15.0, 0.0)),
            ],
        )

        # The composer should infer at least the VOUT net as an interconnect
        # (since both children expose a VOUT port)
        interconnect_names_upper = {
            name.upper() for name in result.inferred_interconnect_nets
        }
        assert "VOUT" in interconnect_names_upper




class TestEstimateParentBoardSize:
    def test_empty_children_returns_minimum(self):
        w, h = estimate_parent_board_size([])
        assert w == 10.0
        assert h == 10.0

    def test_single_child(self):
        w, h = estimate_parent_board_size([(20.0, 10.0)])
        assert w > 20.0
        assert h > 10.0

    def test_multiple_children_larger_than_single(self):
        single_w, single_h = estimate_parent_board_size([(20.0, 10.0)])
        multi_w, multi_h = estimate_parent_board_size(
            [(20.0, 10.0), (15.0, 8.0), (10.0, 5.0)]
        )
        assert multi_w * multi_h > single_w * single_h

    def test_routing_overhead_increases_size(self):
        w_low, h_low = estimate_parent_board_size(
            [(20.0, 10.0)], routing_overhead_factor=1.0
        )
        w_high, h_high = estimate_parent_board_size(
            [(20.0, 10.0)], routing_overhead_factor=2.0
        )
        assert w_high * h_high > w_low * h_low

    def test_margin_adds_clearance(self):
        w_no, h_no = estimate_parent_board_size(
            [(20.0, 10.0)], margin_mm=0.0
        )
        w_yes, h_yes = estimate_parent_board_size(
            [(20.0, 10.0)], margin_mm=3.0
        )
        assert w_yes > w_no
        assert h_yes > h_no

    def test_zero_size_children_handled(self):
        w, h = estimate_parent_board_size([(0.0, 0.0), (5.0, 3.0)])
        assert w > 0.0
        assert h > 0.0


class TestEstimateLayerAwareParentBoardSize:
    def test_front_and_back_only_children_share_area(self):
        layer_agnostic = estimate_parent_board_size([(20.0, 10.0), (20.0, 10.0)])
        layer_aware = estimate_layer_aware_parent_board_size(
            [
                ([(Point(0.0, 0.0), Point(20.0, 10.0))], [], []),
                ([], [(Point(0.0, 0.0), Point(20.0, 10.0))], []),
            ]
        )
        assert layer_aware[0] * layer_aware[1] < layer_agnostic[0] * layer_agnostic[1] * 0.7

    def test_tht_child_matches_layer_agnostic_area(self):
        layer_agnostic = estimate_parent_board_size([(20.0, 10.0)])
        layer_aware = estimate_layer_aware_parent_board_size(
            [
                (
                    [(Point(0.0, 0.0), Point(20.0, 10.0))],
                    [],
                    [(Point(0.0, 0.0), Point(20.0, 10.0))],
                )
            ]
        )
        assert layer_aware == layer_agnostic


class TestChildLayerEnvelopes:
    def test_tht_keepout_uses_pad_span_not_body_bbox(self):
        battery = Component(
            ref="BT1",
            value="18650",
            pos=Point(40.0, 50.0),
            rotation=0.0,
            layer=Layer.BACK,
            width_mm=70.0,
            height_mm=20.0,
            body_center=Point(40.0, 50.0),
            is_through_hole=True,
            pads=[
                Pad(ref="BT1", pad_id="1", pos=Point(10.0, 50.0), net="VBAT", layer=Layer.FRONT),
                Pad(ref="BT1", pad_id="2", pos=Point(70.0, 50.0), net="GND", layer=Layer.FRONT),
            ],
        )
        transformed = TransformedSubcircuit(
            instance=SubCircuitInstance(
                layout_id=_make_subcircuit_id("BAT"),
                origin=Point(0.0, 0.0),
                rotation=0.0,
                transformed_bbox=(70.0, 20.0),
            ),
            layout=_make_layout("BAT", {"BT1": battery}),
            transformed_components={"BT1": battery},
            bounding_box=(Point(5.0, 40.0), Point(75.0, 60.0)),
        )

        front_surface, back_surface, tht_keepout = child_layer_envelopes(transformed)

        assert front_surface == []
        assert back_surface == [(Point(5.0, 40.0), Point(75.0, 60.0))]
        assert tht_keepout == [
            (Point(9.4, 49.4), Point(10.6, 50.6)),
            (Point(69.4, 49.4), Point(70.6, 50.6)),
        ]


class TestPackedExtentsOutline:
    def test_empty_returns_default(self):
        tl, br = packed_extents_outline([])
        assert tl.x == 0.0
        assert br.x == 10.0

    def test_single_child(self):
        tl, br = packed_extents_outline(
            [(Point(0.0, 0.0), Point(20.0, 10.0))], margin_mm=1.0
        )
        assert tl.x == pytest.approx(-1.0)
        assert tl.y == pytest.approx(-1.0)
        assert br.x == pytest.approx(21.0)
        assert br.y == pytest.approx(11.0)

    def test_two_children_side_by_side(self):
        tl, br = packed_extents_outline(
            [
                (Point(0.0, 0.0), Point(20.0, 10.0)),
                (Point(25.0, 0.0), Point(40.0, 8.0)),
            ],
            margin_mm=1.5,
        )
        assert br.x == pytest.approx(25.0 + 15.0 + 1.5)
        assert br.y == pytest.approx(10.0 + 1.5)

    def test_zero_margin(self):
        tl, br = packed_extents_outline(
            [(Point(5.0, 3.0), Point(15.0, 11.0))], margin_mm=0.0
        )
        assert tl.x == pytest.approx(5.0)
        assert tl.y == pytest.approx(3.0)
        assert br.x == pytest.approx(15.0)
        assert br.y == pytest.approx(11.0)






# ---------------------------------------------------------------------------
# Rotation-convention guard for the compose anchor/overlap transform.
#
# Regression for the USB-C edge connector stranded ~8.8mm inboard
# (docs/plans/usb-c-edge-connector-stranding-three-bugs.md). The leaf is
# PLACED with the KiCad-CW convention (geometry.transform_point); the edge
# anchor was computed with the OPPOSITE handedness (math-CCW), so for a
# 90/270-rotated connector leaf the anchor reflected across the part body and
# landed ~part-height outboard of the real mouth -> empty-FR4 strip. The fix
# routes _transform_local_point through the single tested convention.


def test_transform_local_point_uses_kicad_cw_placement_convention():
    point = Point(10.5, 4.245)
    origin = Point(-7.19, 41.48)
    for rotation in (0.0, 90.0, 180.0, 270.0, 45.0, 30.0):
        got = _transform_local_point(point, origin, rotation)
        want = geometry.transform_point(point, origin, rotation)
        assert got.x == pytest.approx(want.x), (rotation, got, want)
        assert got.y == pytest.approx(want.y), (rotation, got, want)


def test_transform_local_point_90deg_is_not_math_ccw():
    # Pin the actual handedness so a future refactor can't silently regress to
    # math-CCW. KiCad-CW 90deg maps local (1,0) -> (0,-1); math-CCW -> (0,+1).
    got = _transform_local_point(Point(1.0, 0.0), Point(0.0, 0.0), 90.0)
    assert got.x == pytest.approx(0.0)
    assert got.y == pytest.approx(-1.0), f"expected KiCad-CW (0,-1), got {got}"


def test_antenna_intent_drives_parent_rotation_anchor_and_propagation():
    component = _make_component("U1", 10.0, 16.4, w=18.0, h=25.0)
    intent = AntennaEdgeIntent(
        owner_ref="U1",
        source="footprint_rule_area",
        source_id="antenna_keepout",
        local_direction="top",
        local_anchor_mm=-16.4,
        local_anchor_midpoint=Point(0.0, -16.4),
        local_polygon=(
            Point(-9.0, -16.4),
            Point(9.0, -16.4),
            Point(9.0, -10.0),
            Point(-9.0, -10.0),
        ),
        target_edge="top",
    )
    layout = _make_layout("ANTENNA", {"U1": component})
    layout.antenna_edge_intents = [intent]
    artifact = LoadedSubcircuitArtifact(
        artifact_dir="/tmp/antenna",
        metadata={},
        debug={},
        layout=layout,
    )

    derived = derive_attachment_constraints([artifact], {}, {}, {})

    assert len(derived.constraints) == 1
    constraint = derived.constraints[0]
    assert constraint.anchor_kind == "antenna"
    spec = derived.child_specs[0]
    assert spec.rotation_candidates == [0.0]
    entry = spec.models[0.0].constraint_entries[0]
    assert entry.local_anchor_offset.y == pytest.approx(0.0)

    parent = _make_parent_def("PARENT", ["ANTENNA"])
    composition = build_parent_composition(
        parent,
        child_artifact_placements=[
            ChildArtifactPlacement(
                artifact=artifact,
                origin=Point(0.0, 0.0),
                rotation=0.0,
            )
        ],
    )
    assert composition.board_state.antenna_edge_intents == [intent]


def test_parent_anchor_resolver_does_not_add_child_origin_twice():
    component = _make_component("U1", 10.0, 16.4, w=18.0, h=25.0)
    intent = AntennaEdgeIntent(
        owner_ref="U1",
        source="footprint_rule_area",
        source_id="antenna_keepout",
        local_direction="top",
        local_anchor_mm=-16.4,
        local_anchor_midpoint=Point(0.0, -16.4),
        local_polygon=(),
        target_edge="top",
    )
    layout = _make_layout("ANTENNA", {"U1": component})
    layout.antenna_edge_intents = [intent]
    artifact = LoadedSubcircuitArtifact(
        artifact_dir="/tmp/antenna",
        metadata={},
        debug={},
        layout=layout,
    )
    derived = derive_attachment_constraints([artifact], {}, {}, {})
    placement = ChildArtifactPlacement(
        artifact=artifact,
        origin=Point(5.0, 7.0),
        rotation=0.0,
    )
    transformed = transform_loaded_artifact(
        artifact,
        origin=placement.origin,
        rotation=placement.rotation,
    )

    anchors = _resolve_constraint_anchor_positions(
        derived,
        {artifact.instance_path: placement},
        [artifact],
        {0: transformed},
        {},
    )

    assert anchors["U1"] == Point(15.0, 7.0)
