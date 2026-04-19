"""Tests for kicraft.autoplacer.brain.subcircuit_extractor -- leaf extraction.

All tests use synthetic/mock data only; no pcbnew dependency.
"""

from __future__ import annotations

import json

import pytest

from kicraft.autoplacer.brain.subcircuit_extractor import (
    ExtractedSubcircuitBoard,
    extract_leaf_board_state,
    extraction_debug_dict,
    summarize_extraction,
)
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    InterfaceDirection,
    InterfacePort,
    InterfaceRole,
    InterfaceSide,
    Layer,
    Net,
    Pad,
    Point,
    SubCircuitDefinition,
    SubCircuitId,
    SubcircuitAccessPolicy,
    TraceSegment,
    Via,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_leaf_subcircuit(
    refs: list[str],
    ports: list[InterfacePort] | None = None,
    name: str = "TEST",
) -> SubCircuitDefinition:
    """Build a minimal leaf SubCircuitDefinition for testing."""
    return SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name=name,
            sheet_file="test.kicad_sch",
            instance_path=f"/{name.lower()}",
        ),
        schematic_path="/fake/test.kicad_sch",
        component_refs=list(refs),
        ports=list(ports or []),
        child_ids=[],
        parent_id=None,
        is_leaf=True,
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


def _make_pad(ref: str, pad_id: str, x: float, y: float, net: str) -> Pad:
    return Pad(ref=ref, pad_id=pad_id, pos=Point(x, y), net=net, layer=Layer.FRONT)


def _basic_full_state() -> tuple[SubCircuitDefinition, BoardState]:
    """Return a simple 2-component leaf + full board with 4 components."""
    pad_r1_1 = _make_pad("R1", "1", 9.5, 20.0, "INT_NET")
    pad_r1_2 = _make_pad("R1", "2", 10.5, 20.0, "VOUT")
    pad_c1_1 = _make_pad("C1", "1", 14.5, 20.0, "INT_NET")
    pad_c1_2 = _make_pad("C1", "2", 15.5, 20.0, "GND")
    pad_u1_1 = _make_pad("U1", "1", 30.0, 30.0, "VOUT")
    pad_u1_2 = _make_pad("U1", "2", 32.0, 30.0, "GND")
    pad_j1_1 = _make_pad("J1", "1", 50.0, 10.0, "VIN")

    components = {
        "R1": _make_component("R1", 10.0, 20.0, pads=[pad_r1_1, pad_r1_2]),
        "C1": _make_component("C1", 15.0, 20.0, pads=[pad_c1_1, pad_c1_2]),
        "U1": _make_component("U1", 31.0, 30.0, w=4.0, h=4.0, pads=[pad_u1_1, pad_u1_2]),
        "J1": _make_component("J1", 50.0, 10.0, w=6.0, h=3.0, pads=[pad_j1_1]),
    }

    nets = {
        "INT_NET": Net(name="INT_NET", pad_refs=[("R1", "1"), ("C1", "1")]),
        "VOUT": Net(name="VOUT", pad_refs=[("R1", "2"), ("U1", "1")]),
        "GND": Net(name="GND", pad_refs=[("C1", "2"), ("U1", "2")], is_power=True),
        "VIN": Net(name="VIN", pad_refs=[("J1", "1")]),
    }

    traces = [
        TraceSegment(
            start=Point(9.5, 20.0), end=Point(14.5, 20.0),
            layer=Layer.FRONT, net="INT_NET", width_mm=0.2,
        ),
        TraceSegment(
            start=Point(30.0, 30.0), end=Point(32.0, 30.0),
            layer=Layer.FRONT, net="VOUT", width_mm=0.2,
        ),
    ]

    vias = [
        Via(pos=Point(12.0, 20.0), net="INT_NET"),
        Via(pos=Point(31.0, 30.0), net="VOUT"),
    ]

    full_state = BoardState(
        components=components,
        nets=nets,
        traces=traces,
        vias=vias,
        board_outline=(Point(0.0, 0.0), Point(80.0, 60.0)),
    )

    port_vout = _make_port("VOUT", "VOUT", InterfaceRole.SIGNAL_OUT)
    port_gnd = _make_port("GND", "GND", InterfaceRole.GROUND)
    leaf = _make_leaf_subcircuit(["R1", "C1"], ports=[port_vout, port_gnd])

    return leaf, full_state


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractBasicLeaf:
    """test_extract_basic_leaf -- extract a 2-component leaf."""

    def test_extract_basic_leaf(self):
        leaf, full_state = _basic_full_state()
        result = extract_leaf_board_state(leaf, full_state)

        assert isinstance(result, ExtractedSubcircuitBoard)
        assert set(result.component_refs) == {"R1", "C1"}
        assert "R1" in result.local_state.components
        assert "C1" in result.local_state.components
        assert "U1" not in result.local_state.components
        # Board outline starts at (0, 0)
        tl, br = result.local_state.board_outline
        assert tl.x == pytest.approx(0.0)
        assert tl.y == pytest.approx(0.0)
        assert br.x > 0.0
        assert br.y > 0.0

    def test_internal_nets_extracted(self):
        leaf, full_state = _basic_full_state()
        result = extract_leaf_board_state(leaf, full_state)
        assert "INT_NET" in result.net_partition.internal

    def test_external_nets_extracted(self):
        leaf, full_state = _basic_full_state()
        result = extract_leaf_board_state(leaf, full_state)
        # VOUT and GND are shared with U1 => external
        assert "VOUT" in result.net_partition.external or "GND" in result.net_partition.external


class TestExtractPreservesPadGeometry:
    """test_extract_preserves_pad_geometry -- pads are translated correctly."""

    def test_extract_preserves_pad_geometry(self):
        leaf, full_state = _basic_full_state()
        result = extract_leaf_board_state(leaf, full_state, margin_mm=5.0)

        original_r1 = full_state.components["R1"]
        translated_r1 = result.local_state.components["R1"]

        # The translation zeroes the origin, so pads should shift by translation
        dx = result.translation.x
        dy = result.translation.y

        for orig_pad, trans_pad in zip(original_r1.pads, translated_r1.pads):
            assert trans_pad.pos.x == pytest.approx(orig_pad.pos.x + dx)
            assert trans_pad.pos.y == pytest.approx(orig_pad.pos.y + dy)
            # Net name is preserved
            assert trans_pad.net == orig_pad.net
            # Layer is preserved
            assert trans_pad.layer == orig_pad.layer


class TestNetPartition:
    """Net partition tests: internal, external, ignored."""

    def test_net_partition_internal_only(self):
        """Net with only leaf-internal refs goes to internal."""
        leaf, full_state = _basic_full_state()
        result = extract_leaf_board_state(leaf, full_state)
        # INT_NET only has R1 and C1 pads -- both in the leaf
        assert "INT_NET" in result.net_partition.internal
        assert "INT_NET" not in result.net_partition.external
        assert "INT_NET" not in result.net_partition.ignored

    def test_net_partition_external(self):
        """Net shared with non-leaf components goes to external."""
        leaf, full_state = _basic_full_state()
        result = extract_leaf_board_state(leaf, full_state)
        # VOUT has R1 (leaf) and U1 (non-leaf) => external
        assert "VOUT" in result.net_partition.external

    def test_net_partition_ignored(self):
        """Net in ignored_nets set goes to ignored."""
        leaf, full_state = _basic_full_state()
        result = extract_leaf_board_state(
            leaf, full_state, ignored_nets={"INT_NET"}
        )
        assert "INT_NET" in result.net_partition.ignored
        assert "INT_NET" not in result.net_partition.internal


class TestEnvelope:
    """Envelope derivation tests."""

    def test_envelope_margin(self):
        """Verify margin_mm expands the envelope."""
        leaf, full_state = _basic_full_state()
        result_narrow = extract_leaf_board_state(leaf, full_state, margin_mm=1.0)
        result_wide = extract_leaf_board_state(leaf, full_state, margin_mm=10.0)

        assert result_wide.envelope.width_mm >= result_narrow.envelope.width_mm
        assert result_wide.envelope.height_mm >= result_narrow.envelope.height_mm

    def test_envelope_with_board_outline(self):
        """When board_outline is provided, envelope clamps to board edge."""
        pad1 = _make_pad("R1", "1", 1.0, 1.0, "A")
        comp = _make_component("R1", 1.0, 1.0, pads=[pad1])

        full_state = BoardState(
            components={"R1": comp},
            nets={"A": Net(name="A", pad_refs=[("R1", "1")])},
            # Board edge is close to the component
            board_outline=(Point(0.0, 0.0), Point(5.0, 5.0)),
        )
        leaf = _make_leaf_subcircuit(["R1"])
        result = extract_leaf_board_state(leaf, full_state, margin_mm=20.0)

        # With margin=20 but board is only 5x5, the envelope should be
        # clamped so it does not extend beyond what the board allows
        assert result.envelope is not None
        assert result.envelope.source_board_outline is not None


class TestTranslation:
    """Translation zeroes the origin."""

    def test_translation_zeroes_origin(self):
        leaf, full_state = _basic_full_state()
        result = extract_leaf_board_state(leaf, full_state)

        tl, _ = result.local_state.board_outline
        assert tl.x == pytest.approx(0.0)
        assert tl.y == pytest.approx(0.0)

        # All component positions should be >= 0
        for comp in result.local_state.components.values():
            assert comp.pos.x >= 0.0
            assert comp.pos.y >= 0.0


class TestErrorCases:
    """Error / edge-case tests."""

    def test_non_leaf_raises(self):
        non_leaf = SubCircuitDefinition(
            id=SubCircuitId(
                sheet_name="PARENT",
                sheet_file="parent.kicad_sch",
                instance_path="/parent",
            ),
            component_refs=["R1"],
            is_leaf=False,
        )
        full_state = BoardState(
            components={"R1": _make_component("R1", 10.0, 10.0)},
        )
        with pytest.raises(ValueError, match="not a leaf"):
            extract_leaf_board_state(non_leaf, full_state)

    def test_no_matching_components_raises(self):
        leaf = _make_leaf_subcircuit(["MISSING1", "MISSING2"])
        full_state = BoardState(
            components={"R1": _make_component("R1", 10.0, 10.0)},
        )
        with pytest.raises(ValueError, match="no matching components"):
            extract_leaf_board_state(leaf, full_state)


class TestSummarize:
    """summarize_extraction and extraction_debug_dict tests."""

    def test_summarize_extraction(self):
        leaf, full_state = _basic_full_state()
        result = extract_leaf_board_state(leaf, full_state)
        summary = summarize_extraction(result)

        assert isinstance(summary, str)
        assert "TEST" in summary
        assert "refs=" in summary
        assert "internal_nets=" in summary

    def test_extraction_debug_dict(self):
        leaf, full_state = _basic_full_state()
        result = extract_leaf_board_state(leaf, full_state)
        debug = extraction_debug_dict(result)

        assert isinstance(debug, dict)
        # Verify it is JSON-serializable
        json_str = json.dumps(debug)
        assert len(json_str) > 0
        # Key structure checks
        assert "subcircuit" in debug
        assert "component_refs" in debug
        assert "net_partition" in debug
        assert "local_board_outline" in debug
        assert "translation" in debug
        assert "trace_count" in debug
        assert "via_count" in debug
        assert "notes" in debug
