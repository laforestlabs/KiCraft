"""Tests for kicraft.autoplacer.brain.leaf_geometry -- connector pad margin.

All tests use synthetic/mock data only; no pcbnew dependency.
"""

from __future__ import annotations

import pytest

from kicraft.autoplacer.brain.leaf_geometry import tight_leaf_geometry_bounds
from kicraft.autoplacer.brain.subcircuit_extractor import (
    ExtractedSubcircuitBoard,
    NetPartition,
)
from kicraft.autoplacer.brain.types import (
    BoardState,
    Component,
    Layer,
    Pad,
    Point,
    SubCircuitDefinition,
    SubCircuitId,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pad(ref: str, pad_id: str, x: float, y: float, net: str) -> Pad:
    return Pad(ref=ref, pad_id=pad_id, pos=Point(x, y), net=net, layer=Layer.FRONT)


def _make_connector(
    ref: str = "J1",
    pos_x: float = 5.0,
    pos_y: float = 5.0,
) -> Component:
    """Build a minimal connector component with two pads."""
    return Component(
        ref=ref,
        value="USB-C",
        pos=Point(pos_x, pos_y),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=4.0,
        height_mm=6.0,
        pads=[
            _make_pad(ref, "1", 3.5, 4.0, "VBUS"),
            _make_pad(ref, "2", 3.5, 6.0, "GND"),
        ],
        locked=True,
        kind="connector",
    )


def _make_passive(
    ref: str = "R1",
    pos_x: float = 10.0,
    pos_y: float = 5.0,
) -> Component:
    """Build a minimal passive component with two pads."""
    return Component(
        ref=ref,
        value="10k",
        pos=Point(pos_x, pos_y),
        rotation=0.0,
        layer=Layer.FRONT,
        width_mm=1.6,
        height_mm=0.8,
        pads=[
            _make_pad(ref, "1", pos_x - 0.5, pos_y, "NET1"),
            _make_pad(ref, "2", pos_x + 0.5, pos_y, "NET2"),
        ],
        locked=False,
        kind="passive",
    )


def _make_leaf_definition(refs: list[str]) -> SubCircuitDefinition:
    """Build a minimal leaf SubCircuitDefinition."""
    return SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name="LEAF_TEST",
            sheet_file="test.kicad_sch",
            instance_path="/leaf_test",
        ),
        component_refs=list(refs),
        is_leaf=True,
    )


def _make_extraction(components: dict[str, Component]) -> ExtractedSubcircuitBoard:
    """Build a minimal ExtractedSubcircuitBoard from a component dict."""
    refs = list(components.keys())
    board_state = BoardState(
        components=components,
        nets={},
        traces=[],
        vias=[],
        board_outline=(Point(0.0, 0.0), Point(50.0, 50.0)),
    )
    return ExtractedSubcircuitBoard(
        subcircuit=_make_leaf_definition(refs),
        full_state=board_state,
        local_state=board_state,
        component_refs=refs,
        interface_ports=[],
        net_partition=NetPartition(),
        translation=Point(0.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestTightBoundsConnectorPadMargin:
    """Tests for the connector_pad_margin_mm parameter."""

    def test_tight_bounds_without_connector_margin(self):
        """Default margin=0: bounds come from component bbox and pad centers."""
        connector = _make_connector()
        components = {"J1": connector}
        extraction = _make_extraction(components)

        bounds = tight_leaf_geometry_bounds(extraction, components, {})

        # Component bbox: center at (5,5), w=4, h=6 => TL(3,2), BR(7,8)
        # Pad centers: (3.5, 4.0) and (3.5, 6.0) -- inside the bbox
        # So the bounds should match the component bbox
        assert bounds["min_x"] == pytest.approx(3.0)
        assert bounds["min_y"] == pytest.approx(2.0)
        assert bounds["max_x"] == pytest.approx(7.0)
        assert bounds["max_y"] == pytest.approx(8.0)

    def test_tight_bounds_with_connector_margin(self):
        """connector_pad_margin_mm=1.0 expands bounds by pad margin."""
        connector = _make_connector()
        components = {"J1": connector}
        extraction = _make_extraction(components)

        bounds = tight_leaf_geometry_bounds(
            extraction, components, {},
            connector_pad_margin_mm=1.0,
        )

        # Component bbox: TL(3,2), BR(7,8)
        # Pad centers: (3.5, 4.0) and (3.5, 6.0)
        # With margin=1.0 on connector pads:
        #   pad1 contributes x range [2.5, 4.5], y range [3.0, 5.0]
        #   pad2 contributes x range [2.5, 4.5], y range [5.0, 7.0]
        # Combined with component bbox:
        #   min_x = min(3.0, 2.5) = 2.5
        #   min_y = min(2.0, 3.0) = 2.0 (bbox still dominates)
        #   max_x = max(7.0, 4.5) = 7.0 (bbox still dominates)
        #   max_y = max(8.0, 7.0) = 8.0 (bbox still dominates)
        assert bounds["min_x"] == pytest.approx(2.5)
        assert bounds["min_y"] == pytest.approx(2.0)
        assert bounds["max_x"] == pytest.approx(7.0)
        assert bounds["max_y"] == pytest.approx(8.0)
        # Width should be wider than without margin
        assert bounds["width_mm"] == pytest.approx(7.0 - 2.5)

    def test_tight_bounds_connector_margin_only_affects_connectors(self):
        """Margin expands connector pads but NOT passive pads."""
        connector = _make_connector()
        passive = _make_passive()
        components = {"J1": connector, "R1": passive}
        extraction = _make_extraction(components)

        # Get bounds without margin for baseline
        bounds_no_margin = tight_leaf_geometry_bounds(
            extraction, components, {},
        )

        # Get bounds with margin
        bounds_with_margin = tight_leaf_geometry_bounds(
            extraction, components, {},
            connector_pad_margin_mm=1.0,
        )

        # The connector pad at x=3.5 with margin=1.0 pushes min_x to 2.5
        # (from 3.0 which is the connector bbox left edge)
        assert bounds_with_margin["min_x"] < bounds_no_margin["min_x"]

        # The passive at x=10, pads at 9.5 and 10.5, bbox TL.x=9.2, BR.x=10.8
        # Without margin the right edge is from the passive bbox at 10.8
        # With margin, the passive should NOT be expanded, so max_x stays
        assert bounds_with_margin["max_x"] == pytest.approx(
            bounds_no_margin["max_x"]
        )

    def test_tight_bounds_connector_margin_zero_is_noop(self):
        """Explicit connector_pad_margin_mm=0.0 gives same result as default."""
        connector = _make_connector()
        passive = _make_passive()
        components = {"J1": connector, "R1": passive}
        extraction = _make_extraction(components)

        bounds_default = tight_leaf_geometry_bounds(
            extraction, components, {},
        )
        bounds_explicit_zero = tight_leaf_geometry_bounds(
            extraction, components, {},
            connector_pad_margin_mm=0.0,
        )

        assert bounds_explicit_zero["min_x"] == pytest.approx(bounds_default["min_x"])
        assert bounds_explicit_zero["min_y"] == pytest.approx(bounds_default["min_y"])
        assert bounds_explicit_zero["max_x"] == pytest.approx(bounds_default["max_x"])
        assert bounds_explicit_zero["max_y"] == pytest.approx(bounds_default["max_y"])
        assert bounds_explicit_zero["width_mm"] == pytest.approx(
            bounds_default["width_mm"]
        )
        assert bounds_explicit_zero["height_mm"] == pytest.approx(
            bounds_default["height_mm"]
        )
