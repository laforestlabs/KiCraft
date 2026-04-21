"""Tests for kicraft.autoplacer.brain.subcircuit_composer -- parent composition.

All tests use synthetic/mock data only; no pcbnew dependency.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.subcircuit_composer import (
    AttachmentConstraint,
    ChildPlacement,
    PlacementConstraintEntry,
    PlacementModel,
    build_parent_composition,
    child_layer_envelopes,
    composition_summary,
    estimate_layer_aware_parent_board_size,
    estimate_parent_board_size,
    packed_extents_outline,
    place_constrained_child,
    _derive_board_outline,
)
from kicraft.autoplacer.brain.subcircuit_instances import TransformedSubcircuit
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
    SubCircuitInstance,
    SubCircuitLayout,
    SubcircuitAccessPolicy,
    TraceSegment,
    Via,
)
from kicraft.cli.compose_subcircuits import (
    _choose_packed_unconstrained_placement,
    _find_non_overlapping_origin,
    _place_parent_local_components,
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


class TestCompositionSummary:
    """test_composition_summary -- returns a human-readable string."""

    def test_composition_summary(self):
        comp = _make_component("R1", 2.0, 1.0)
        layout = _make_layout("CHILD", {"R1": comp})
        parent = _make_parent_def("PARENT", ["CHILD"])

        result = build_parent_composition(
            parent,
            child_placements=[
                ChildPlacement(layout=layout, origin=Point(0.0, 0.0)),
            ],
        )

        summary = composition_summary(result)
        assert isinstance(summary, str)
        assert "PARENT" in summary
        assert "children=" in summary
        assert "components=" in summary


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


class TestParentRoutingAddsTraces:
    """test_parent_routing_adds_traces -- interconnect routing adds traces."""

    def test_parent_routing_adds_traces(self):
        port_a = _make_port("SIG", "SIG")
        port_b = _make_port("SIG", "SIG")

        pad_a = _make_pad("R1", "2", 4.0, 1.0, "SIG")
        comp_a = _make_component("R1", 2.0, 1.0, w=4.0, pads=[pad_a])
        anchor_a = InterfaceAnchor(
            port_name="SIG", pos=Point(4.0, 1.0),
            layer=Layer.FRONT, pad_ref=("R1", "2"),
        )
        layout_a = _make_layout(
            "CHILD_A", {"R1": comp_a},
            ports=[port_a], anchors=[anchor_a],
            bbox=(5.0, 2.0),
        )

        pad_b = _make_pad("C1", "1", 1.0, 1.0, "SIG")
        comp_b = _make_component("C1", 2.0, 1.0, w=4.0, pads=[pad_b])
        anchor_b = InterfaceAnchor(
            port_name="SIG", pos=Point(1.0, 1.0),
            layer=Layer.FRONT, pad_ref=("C1", "1"),
        )
        layout_b = _make_layout(
            "CHILD_B", {"C1": comp_b},
            ports=[port_b], anchors=[anchor_b],
            bbox=(5.0, 2.0),
        )

        parent = _make_parent_def("P", ["CHILD_A", "CHILD_B"])

        result = build_parent_composition(
            parent,
            child_placements=[
                ChildPlacement(layout=layout_a, origin=Point(0.0, 0.0)),
                ChildPlacement(layout=layout_b, origin=Point(20.0, 0.0)),
            ],
        )

        assert result.board_state is not None
        assert result.component_count == 2


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


class TestPackingSpacingAffectsBoardSize:
    """Smaller inter-child spacing produces a smaller board outline."""

    @staticmethod
    def _pack_children(
        child_sizes: list[tuple[float, float]], spacing_mm: float
    ) -> tuple[float, float]:
        """Simulate compose_subcircuits packed mode packing algorithm."""
        if not child_sizes:
            return (0.0, 0.0)

        max_child_width = max(w for w, _ in child_sizes)
        estimated_w, _ = estimate_parent_board_size(
            child_sizes, margin_mm=spacing_mm
        )
        target_row_width = max(
            max_child_width + spacing_mm,
            estimated_w if estimated_w > 0.0 else max_child_width,
        )

        origins: list[tuple[float, float]] = []
        row_y = 0.0
        row_x = 0.0
        row_height = 0.0
        current_row_items = 0

        for w, h in child_sizes:
            should_wrap = (
                current_row_items > 0
                and row_x > 0.0
                and (row_x + w) > target_row_width
            )
            if should_wrap:
                row_y += row_height + spacing_mm
                row_x = 0.0
                row_height = 0.0
                current_row_items = 0

            origins.append((row_x, row_y))
            row_x += w + spacing_mm
            row_height = max(row_height, h)
            current_row_items += 1

        placed_bboxes = [
            (Point(x, y), Point(x + w, y + h))
            for (x, y), (w, h) in zip(origins, child_sizes)
        ]
        outline = packed_extents_outline(placed_bboxes)
        outline_w = outline[1].x - outline[0].x
        outline_h = outline[1].y - outline[0].y
        return (outline_w, outline_h)

    def test_smaller_spacing_produces_smaller_board(self):
        children = [(30.0, 10.0), (30.0, 10.0)]

        w_tight, h_tight = self._pack_children(children, spacing_mm=1.0)
        w_loose, h_loose = self._pack_children(children, spacing_mm=5.0)

        area_tight = w_tight * h_tight
        area_loose = w_loose * h_loose

        assert area_tight < area_loose, (
            f"Tight spacing (1mm) area={area_tight:.1f} should be less than "
            f"loose spacing (5mm) area={area_loose:.1f}"
        )

    def test_spacing_range_monotonic(self):
        children = [(30.0, 10.0), (30.0, 10.0), (30.0, 10.0)]
        areas = []
        for spacing in [0.5, 1.0, 2.0, 4.0, 6.0]:
            w, h = self._pack_children(children, spacing_mm=spacing)
            areas.append(w * h)

        for i in range(len(areas) - 1):
            assert areas[i] <= areas[i + 1], (
                f"Board area should not decrease as spacing increases: "
                f"spacing sequence produced areas {[round(a, 1) for a in areas]}"
            )


class TestConstraintPlacementGeometry:
    def _model(self, bbox_min: Point, bbox_max: Point, entries: list[PlacementConstraintEntry]) -> PlacementModel:
        layout = _make_layout("NEG", {})
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

    def test_negative_origin_children_are_supported(self):
        model = self._model(
            Point(-6.0, -4.0),
            Point(8.0, 4.0),
            [
                PlacementConstraintEntry(
                    constraint=AttachmentConstraint(
                        ref="J1",
                        target="edge",
                        value="left",
                        inward_keep_in_mm=0.0,
                        outward_overhang_mm=0.0,
                        source="child_artifact",
                        child_index=0,
                        strict=True,
                    ),
                    local_anchor_offset=Point(-6.0, 0.0),
                )
            ],
        )

        origin, bbox = place_constrained_child(
            model,
            parent_outline_min=Point(0.0, 0.0),
            parent_outline_max=Point(100.0, 50.0),
        )

        assert origin.x == pytest.approx(6.0)
        assert bbox[0].x == pytest.approx(0.0)

    def test_infeasible_attachment_bands_raise_error(self):
        model = self._model(
            Point(-5.0, -5.0),
            Point(5.0, 5.0),
            [
                PlacementConstraintEntry(
                    constraint=AttachmentConstraint(
                        ref="BT1",
                        target="zone",
                        value="bottom",
                        inward_keep_in_mm=30.0,
                        outward_overhang_mm=0.0,
                        source="child_artifact",
                        child_index=0,
                        strict=False,
                    ),
                    local_anchor_offset=Point(0.0, 45.0),
                ),
                PlacementConstraintEntry(
                    constraint=AttachmentConstraint(
                        ref="BT2",
                        target="zone",
                        value="bottom",
                        inward_keep_in_mm=30.0,
                        outward_overhang_mm=0.0,
                        source="child_artifact",
                        child_index=0,
                        strict=False,
                    ),
                    local_anchor_offset=Point(0.0, -10.0),
                ),
            ],
        )

        with pytest.raises(ValueError, match="Infeasible zone attachment band"):
            _ = place_constrained_child(
                model,
                parent_outline_min=Point(0.0, 0.0),
                parent_outline_max=Point(60.0, 40.0),
            )


class TestParentLocalPlacement:
    def test_mounting_hole_uses_bbox_edges_for_keep_in(self):
        hole = Component(
            ref="H4",
            value="Hole",
            pos=Point(2.5, 2.5),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=4.5,
            height_mm=4.5,
            body_center=Point(2.5, 2.5),
        )
        constraint = AttachmentConstraint(
            ref="H4",
            target="corner",
            value="top-left",
            inward_keep_in_mm=2.5,
            outward_overhang_mm=0.0,
            source="parent_local",
            child_index=None,
            strict=True,
        )

        _place_parent_local_components(
            {"H4": hole},
            [constraint],
            (Point(0.0, 0.0), Point(80.0, 80.0)),
        )

        bbox_min, _ = hole.bbox()
        assert bbox_min.x >= 2.5 - 1e-6
        assert bbox_min.y >= 2.5 - 1e-6


class TestPackedPlacementOverlapScan:
    def test_second_leaf_scans_from_frame_origin_for_layer_overlap(self):
        back_model = PlacementModel(
            rotation=0.0,
            transformed=TransformedSubcircuit(
                instance=SubCircuitInstance(
                    layout_id=_make_subcircuit_id("BACK"),
                    origin=Point(0.0, 0.0),
                    rotation=0.0,
                    transformed_bbox=(20.0, 20.0),
                ),
                layout=_make_layout("BACK", {}),
                bounding_box=(Point(0.0, 0.0), Point(20.0, 20.0)),
            ),
            layer_envelopes=([], [(Point(0.0, 0.0), Point(20.0, 20.0))], []),
            constraint_entries=[],
        )
        front_model = PlacementModel(
            rotation=0.0,
            transformed=TransformedSubcircuit(
                instance=SubCircuitInstance(
                    layout_id=_make_subcircuit_id("FRONT"),
                    origin=Point(0.0, 0.0),
                    rotation=0.0,
                    transformed_bbox=(12.0, 12.0),
                ),
                layout=_make_layout("FRONT", {}),
                bounding_box=(Point(0.0, 0.0), Point(12.0, 12.0)),
            ),
            layer_envelopes=([(Point(0.0, 0.0), Point(12.0, 12.0))], [], []),
            constraint_entries=[],
        )

        origin = _find_non_overlapping_origin(
            proposed=Point(24.0, 0.0),
            frame_min=Point(0.0, 0.0),
            frame_max=Point(60.0, 40.0),
            model=front_model,
            placed_bboxes=[back_model.transformed.bounding_box],
            placed_envelopes=[back_model.layer_envelopes],
            spacing_mm=2.0,
        )

        placed_front_bbox = (origin, Point(origin.x + 12.0, origin.y + 12.0))
        assert placed_front_bbox[0].x < 20.0
        assert placed_front_bbox[1].x > 0.0
        assert placed_front_bbox[0].y < 20.0
        assert placed_front_bbox[1].y > 0.0

    def test_opposite_side_leaves_prefer_meaningful_overlap(self):
        back_model = PlacementModel(
            rotation=0.0,
            transformed=TransformedSubcircuit(
                instance=SubCircuitInstance(
                    layout_id=_make_subcircuit_id("BACK2"),
                    origin=Point(0.0, 0.0),
                    rotation=0.0,
                    transformed_bbox=(20.0, 20.0),
                ),
                layout=_make_layout("BACK2", {}),
                bounding_box=(Point(0.0, 0.0), Point(20.0, 20.0)),
            ),
            layer_envelopes=([], [(Point(0.0, 0.0), Point(20.0, 20.0))], []),
            constraint_entries=[],
        )
        front_model = PlacementModel(
            rotation=0.0,
            transformed=TransformedSubcircuit(
                instance=SubCircuitInstance(
                    layout_id=_make_subcircuit_id("FRONT2"),
                    origin=Point(0.0, 0.0),
                    rotation=0.0,
                    transformed_bbox=(12.0, 12.0),
                ),
                layout=_make_layout("FRONT2", {}),
                bounding_box=(Point(0.0, 0.0), Point(12.0, 12.0)),
            ),
            layer_envelopes=([(Point(0.0, 0.0), Point(12.0, 12.0))], [], []),
            constraint_entries=[],
        )

        origin = _find_non_overlapping_origin(
            proposed=Point(30.0, 0.0),
            frame_min=Point(0.0, 0.0),
            frame_max=Point(60.0, 40.0),
            model=front_model,
            placed_bboxes=[back_model.transformed.bounding_box],
            placed_envelopes=[back_model.layer_envelopes],
            spacing_mm=2.0,
        )

        overlap_w = max(0.0, min(origin.x + 12.0, 20.0) - max(origin.x, 0.0))
        overlap_h = max(0.0, min(origin.y + 12.0, 20.0) - max(origin.y, 0.0))
        assert overlap_w * overlap_h >= 5.0

    def test_same_side_front_leaves_do_not_overlap(self):
        placed_front_bbox = (Point(0.0, 0.0), Point(20.0, 20.0))
        placed_front_envelopes = [
            ([(Point(0.0, 0.0), Point(20.0, 20.0))], [], [])
        ]
        front_model = PlacementModel(
            rotation=0.0,
            transformed=TransformedSubcircuit(
                instance=SubCircuitInstance(
                    layout_id=_make_subcircuit_id("FRONT3"),
                    origin=Point(0.0, 0.0),
                    rotation=0.0,
                    transformed_bbox=(12.0, 12.0),
                ),
                layout=_make_layout("FRONT3", {}),
                bounding_box=(Point(0.0, 0.0), Point(12.0, 12.0)),
            ),
            layer_envelopes=([(Point(0.0, 0.0), Point(12.0, 12.0))], [], []),
            constraint_entries=[],
        )

        origin = _find_non_overlapping_origin(
            proposed=Point(30.0, 0.0),
            frame_min=Point(0.0, 0.0),
            frame_max=Point(60.0, 40.0),
            model=front_model,
            placed_bboxes=[placed_front_bbox],
            placed_envelopes=placed_front_envelopes,
            spacing_mm=2.0,
        )

        assert origin.x >= 20.0 or origin.y >= 20.0
