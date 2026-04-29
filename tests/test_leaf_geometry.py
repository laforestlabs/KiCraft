"""Tests for kicraft.autoplacer.brain.leaf_geometry tight bounds.

Verifies that ``tight_leaf_geometry_bounds`` reports the union of every
component's ``physical_bbox()`` -- courtyard plus actual pad copper
extents -- so a connector whose pads stick out past the courtyard
correctly grows the leaf outline.

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


def _make_pad(
    ref: str,
    pad_id: str,
    x: float,
    y: float,
    net: str,
    *,
    size_mm: Point | None = None,
) -> Pad:
    return Pad(
        ref=ref,
        pad_id=pad_id,
        pos=Point(x, y),
        net=net,
        layer=Layer.FRONT,
        size_mm=size_mm,
    )


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


class TestTightBoundsPhysicalExtent:
    """Tight bounds reflect each component's physical extent (courtyard
    plus pad copper bboxes), not just courtyard plus pad centers."""

    def test_pad_centers_inside_courtyard_match_bbox(self):
        """Pads with no recorded size and centers inside courtyard => bounds = courtyard."""
        connector = _make_connector()
        components = {"J1": connector}
        extraction = _make_extraction(components)

        bounds = tight_leaf_geometry_bounds(extraction, components, {})

        # Courtyard: center (5,5), w=4, h=6 => TL(3,2), BR(7,8).
        # Pad centers (3.5, 4.0)/(3.5, 6.0) are inside; no size_mm => contribute centers only.
        assert bounds["min_x"] == pytest.approx(3.0)
        assert bounds["min_y"] == pytest.approx(2.0)
        assert bounds["max_x"] == pytest.approx(7.0)
        assert bounds["max_y"] == pytest.approx(8.0)

    def test_pad_size_extends_bounds_outboard_of_courtyard(self):
        """A pad whose copper extends past the courtyard pushes bounds out."""
        connector = Component(
            ref="J1",
            value="USB-C",
            pos=Point(5.0, 5.0),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=4.0,
            height_mm=6.0,
            pads=[
                # Pad copper centered at (3.5, 4.0) with width 2mm reaches
                # x=2.5 -- outside the courtyard's left edge at x=3.0.
                _make_pad("J1", "1", 3.5, 4.0, "VBUS",
                          size_mm=Point(2.0, 1.5)),
                _make_pad("J1", "2", 3.5, 6.0, "GND",
                          size_mm=Point(2.0, 1.5)),
            ],
            locked=True,
            kind="connector",
        )
        components = {"J1": connector}
        extraction = _make_extraction(components)
        bounds = tight_leaf_geometry_bounds(extraction, components, {})

        # Courtyard left edge x=3.0; pad bbox extends to x=2.5 -> bounds at 2.5
        assert bounds["min_x"] == pytest.approx(2.5)
        # Courtyard dominates on the other three sides.
        assert bounds["min_y"] == pytest.approx(2.0)
        assert bounds["max_x"] == pytest.approx(7.0)
        assert bounds["max_y"] == pytest.approx(8.0)

    def test_works_for_any_component_kind_not_only_connectors(self):
        """A passive with oversized pads grows the bounds too -- the previous
        connector-only band-aid no longer applies."""
        passive = Component(
            ref="R1",
            value="10k",
            pos=Point(10.0, 5.0),
            rotation=0.0,
            layer=Layer.FRONT,
            width_mm=1.6,
            height_mm=0.8,
            pads=[
                # Pad copper at x=10.5 with width 2mm reaches x=11.5 --
                # outside the courtyard's right edge at x=10.8.
                _make_pad("R1", "1", 9.5, 5.0, "NET1",
                          size_mm=Point(2.0, 1.0)),
                _make_pad("R1", "2", 10.5, 5.0, "NET2",
                          size_mm=Point(2.0, 1.0)),
            ],
            locked=False,
            kind="passive",
        )
        components = {"R1": passive}
        extraction = _make_extraction(components)
        bounds = tight_leaf_geometry_bounds(extraction, components, {})

        # Pad copper extends past courtyard.right (10.8) to 11.5
        assert bounds["max_x"] == pytest.approx(11.5)
        # Pad copper extends past courtyard.left (9.2) to 8.5
        assert bounds["min_x"] == pytest.approx(8.5)
