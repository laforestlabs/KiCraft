"""Tests for kicraft.autoplacer.brain.types — core data structures."""

from __future__ import annotations


import pytest

from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Net,
    Pad,
    PlacementScore,
    Point,
    SubCircuitId,
    SubCircuitLayout,
    TraceSegment,
    Via,
)


# ---- Point ----------------------------------------------------------------

class TestPoint:
    def test_creation(self):
        p = Point(x=3.0, y=4.0)
        assert p.x == 3.0
        assert p.y == 4.0

    def test_dist(self):
        a = Point(0.0, 0.0)
        b = Point(3.0, 4.0)
        assert a.dist(b) == pytest.approx(5.0)

    def test_dist_same_point(self):
        p = Point(1.0, 2.0)
        assert p.dist(p) == pytest.approx(0.0)

    def test_angle_to(self):
        origin = Point(0.0, 0.0)
        right = Point(1.0, 0.0)
        assert origin.angle_to(right) == pytest.approx(0.0)

    def test_add(self):
        result = Point(1.0, 2.0) + Point(3.0, 4.0)
        assert result.x == pytest.approx(4.0)
        assert result.y == pytest.approx(6.0)

    def test_sub(self):
        result = Point(5.0, 7.0) - Point(2.0, 3.0)
        assert result.x == pytest.approx(3.0)
        assert result.y == pytest.approx(4.0)

    def test_mul(self):
        result = Point(2.0, 3.0) * 2.5
        assert result.x == pytest.approx(5.0)
        assert result.y == pytest.approx(7.5)

    def test_hash_equal_points(self):
        """Points with same coords (within rounding) hash the same."""
        a = Point(1.0, 2.0)
        b = Point(1.0, 2.0)
        assert hash(a) == hash(b)

    def test_hash_can_be_set_member(self):
        """Points can be added to a set."""
        s = {Point(1.0, 2.0), Point(1.0, 2.0), Point(3.0, 4.0)}
        assert len(s) == 2


# ---- Pad ------------------------------------------------------------------

class TestPad:
    def test_creation(self):
        pad = Pad(
            ref="U1",
            pad_id="1",
            pos=Point(10.0, 20.0),
            net="VCC",
            layer=Layer.FRONT,
        )
        assert pad.ref == "U1"
        assert pad.pad_id == "1"
        assert pad.pos.x == 10.0
        assert pad.net == "VCC"
        assert pad.layer == Layer.FRONT

    def test_layer_back(self):
        pad = Pad(ref="R1", pad_id="2", pos=Point(0, 0), net="GND", layer=Layer.BACK)
        assert pad.layer == Layer.BACK


# ---- Component -----------------------------------------------------------

class TestComponent:
    def test_minimal_creation(self):
        c = Component(
            ref="R1",
            value="10k",
            pos=Point(0.0, 0.0),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=1.6,
            height_mm=0.8,
        )
        assert c.ref == "R1"
        assert c.value == "10k"
        assert c.layer == Layer.FRONT
        assert c.locked is False
        assert c.kind == ""
        assert c.pads == []

    def test_area(self):
        c = Component(
            ref="U1", value="IC", pos=Point(0, 0),
            rotation=0, layer=Layer.FRONT,
            width_mm=10.0, height_mm=5.0,
        )
        assert c.area == pytest.approx(50.0)

    def test_bbox_without_clearance(self):
        c = Component(
            ref="C1", value="100nF", pos=Point(10.0, 20.0),
            rotation=0, layer=Layer.FRONT,
            width_mm=2.0, height_mm=1.0,
        )
        tl, br = c.bbox()
        assert tl.x == pytest.approx(9.0)
        assert tl.y == pytest.approx(19.5)
        assert br.x == pytest.approx(11.0)
        assert br.y == pytest.approx(20.5)

    def test_bbox_with_clearance(self):
        c = Component(
            ref="C1", value="100nF", pos=Point(10.0, 20.0),
            rotation=0, layer=Layer.FRONT,
            width_mm=2.0, height_mm=1.0,
        )
        tl, br = c.bbox(clearance=0.5)
        assert tl.x == pytest.approx(8.5)
        assert tl.y == pytest.approx(19.0)
        assert br.x == pytest.approx(11.5)
        assert br.y == pytest.approx(21.0)

    def test_with_pads(self):
        pad1 = Pad(ref="R1", pad_id="1", pos=Point(0, 0), net="A", layer=Layer.FRONT)
        pad2 = Pad(ref="R1", pad_id="2", pos=Point(1, 0), net="B", layer=Layer.FRONT)
        c = Component(
            ref="R1", value="10k", pos=Point(0.5, 0),
            rotation=0, layer=Layer.FRONT,
            width_mm=1.6, height_mm=0.8,
            pads=[pad1, pad2],
        )
        assert len(c.pads) == 2

    def test_optional_fields(self):
        c = Component(
            ref="J1", value="USB", pos=Point(0, 0),
            rotation=90, layer=Layer.FRONT,
            width_mm=8.0, height_mm=6.0,
            kind="connector",
            is_through_hole=True,
            locked=True,
        )
        assert c.kind == "connector"
        assert c.is_through_hole is True
        assert c.locked is True
        assert c.rotation == 90


# ---- Net ------------------------------------------------------------------

class TestNet:
    def test_creation(self):
        n = Net(name="VCC", pad_refs=[("U1", "3"), ("C1", "1")])
        assert n.name == "VCC"
        assert len(n.pad_refs) == 2

    def test_component_refs_property(self):
        n = Net(name="SDA", pad_refs=[("U1", "5"), ("U2", "7"), ("U1", "12")])
        refs = n.component_refs
        assert refs == {"U1", "U2"}

    def test_defaults(self):
        n = Net(name="GND")
        assert n.priority == 0
        assert n.width_mm == pytest.approx(0.127)
        assert n.is_power is False
        assert n.pad_refs == []


# ---- TraceSegment ---------------------------------------------------------

class TestTraceSegment:
    def test_creation(self):
        t = TraceSegment(
            start=Point(0, 0),
            end=Point(3, 4),
            layer=Layer.FRONT,
            net="CLK",
            width_mm=0.127,
        )
        assert t.net == "CLK"
        assert t.layer == Layer.FRONT

    def test_length(self):
        t = TraceSegment(
            start=Point(0, 0), end=Point(3, 4),
            layer=Layer.FRONT, net="CLK", width_mm=0.2,
        )
        assert t.length == pytest.approx(5.0)

    def test_zero_length(self):
        t = TraceSegment(
            start=Point(5, 5), end=Point(5, 5),
            layer=Layer.BACK, net="GND", width_mm=0.5,
        )
        assert t.length == pytest.approx(0.0)


# ---- Via ------------------------------------------------------------------

class TestVia:
    def test_creation(self):
        v = Via(pos=Point(10, 20), net="VCC")
        assert v.pos.x == 10
        assert v.net == "VCC"

    def test_defaults(self):
        v = Via(pos=Point(0, 0), net="GND")
        assert v.drill_mm == pytest.approx(0.3)
        assert v.size_mm == pytest.approx(0.6)

    def test_custom_sizes(self):
        v = Via(pos=Point(0, 0), net="PWR", drill_mm=0.4, size_mm=0.8)
        assert v.drill_mm == pytest.approx(0.4)
        assert v.size_mm == pytest.approx(0.8)


# ---- PlacementScore -------------------------------------------------------

class TestPlacementScore:
    def test_default_creation(self):
        ps = PlacementScore()
        assert ps.total == pytest.approx(0.0)
        assert ps.net_distance == pytest.approx(0.0)
        assert ps.crossover_count == 0

    def test_fields_are_numeric(self):
        ps = PlacementScore(
            net_distance=80.0,
            crossover_score=70.0,
            compactness=60.0,
            edge_compliance=90.0,
            rotation_score=50.0,
            board_containment=95.0,
            courtyard_overlap=100.0,
        )
        for attr in [
            "net_distance", "crossover_score", "compactness",
            "edge_compliance", "rotation_score", "board_containment",
            "courtyard_overlap",
        ]:
            val = getattr(ps, attr)
            assert isinstance(val, (int, float)), f"{attr} should be numeric"

    def test_compute_total(self):
        ps = PlacementScore(
            net_distance=100.0,
            crossover_score=100.0,
            compactness=100.0,
            edge_compliance=100.0,
            rotation_score=100.0,
            board_containment=100.0,
            courtyard_overlap=100.0,
            smt_opposite_tht=100.0,
            group_coherence=100.0,
            aspect_ratio=100.0,
            topology_structure=100.0,
        )
        total = ps.compute_total()
        # All scores at 100 with default weights summing to 1.0 => total = 100
        assert total == pytest.approx(100.0)
        assert ps.total == pytest.approx(100.0)

    def test_compute_total_partial(self):
        ps = PlacementScore(
            net_distance=50.0,
            crossover_score=0.0,
            compactness=0.0,
            edge_compliance=0.0,
            rotation_score=0.0,
            board_containment=0.0,
            courtyard_overlap=0.0,
            smt_opposite_tht=0.0,
            group_coherence=0.0,
            aspect_ratio=0.0,
            topology_structure=0.0,
        )
        total = ps.compute_total()
        # Only net_distance contributes: 50 * 0.20 = 10
        assert total == pytest.approx(10.0)

    def test_compute_total_custom_weights(self):
        ps = PlacementScore(net_distance=80.0, compactness=60.0)
        total = ps.compute_total(weights={"net_distance": 0.5, "compactness": 0.5})
        assert total == pytest.approx(70.0)


# ---- BoardState -----------------------------------------------------------

class TestBoardState:
    def test_empty_creation(self):
        bs = BoardState()
        assert bs.components == {}
        assert bs.nets == {}
        assert bs.traces == []
        assert bs.vias == []

    def test_board_dimensions(self):
        bs = BoardState()
        assert bs.board_width == pytest.approx(90.0)
        assert bs.board_height == pytest.approx(58.0)

    def test_board_center(self):
        bs = BoardState()
        center = bs.board_center
        assert center.x == pytest.approx(45.0)
        assert center.y == pytest.approx(29.0)

    def test_custom_outline(self):
        bs = BoardState(board_outline=(Point(10, 10), Point(60, 40)))
        assert bs.board_width == pytest.approx(50.0)
        assert bs.board_height == pytest.approx(30.0)


# ---- SubCircuitLayout -----------------------------------------------------

class TestSubCircuitLayout:
    def test_creation_empty(self):
        scid = SubCircuitId(
            sheet_name="CHARGER",
            sheet_file="charger.kicad_sch",
            instance_path="/root/charger",
        )
        scl = SubCircuitLayout(subcircuit_id=scid)
        assert scl.subcircuit_id.sheet_name == "CHARGER"
        assert scl.components == {}
        assert scl.traces == []
        assert scl.vias == []
        assert scl.frozen is True

    def test_bounding_box_defaults(self):
        scid = SubCircuitId(
            sheet_name="X", sheet_file="x.kicad_sch", instance_path="/x",
        )
        scl = SubCircuitLayout(subcircuit_id=scid)
        assert scl.width == pytest.approx(0.0)
        assert scl.height == pytest.approx(0.0)
        assert scl.area == pytest.approx(0.0)

    def test_bounding_box_set(self):
        scid = SubCircuitId(
            sheet_name="Y", sheet_file="y.kicad_sch", instance_path="/y",
        )
        scl = SubCircuitLayout(subcircuit_id=scid, bounding_box=(20.0, 15.0))
        assert scl.width == pytest.approx(20.0)
        assert scl.height == pytest.approx(15.0)
        assert scl.area == pytest.approx(300.0)

    def test_with_components(self):
        scid = SubCircuitId(
            sheet_name="Z", sheet_file="z.kicad_sch", instance_path="/z",
        )
        comp = Component(
            ref="R1", value="10k", pos=Point(0, 0),
            rotation=0, layer=Layer.FRONT,
            width_mm=1.6, height_mm=0.8,
        )
        scl = SubCircuitLayout(
            subcircuit_id=scid,
            components={"R1": comp},
        )
        assert len(scl.components) == 1
        assert "R1" in scl.components


# ---- Layer enum -----------------------------------------------------------

class TestLayer:
    def test_front(self):
        assert Layer.FRONT == 0

    def test_back(self):
        assert Layer.BACK == 1

    def test_int_comparison(self):
        assert Layer.FRONT < Layer.BACK
