"""Leaf placement utilities for extracted subcircuits.

This module provides interface anchor inference and silkscreen generation
for solved leaf subcircuits.

Current scope:
- infer physical interface anchors from solved component pad positions
- generate silkscreen outline and label elements for leaf boards

Routing is handled exclusively by FreeRouting (see leaf_routing.py).
Placement solving is handled by PlacementSolver (see placement_solver.py).

Design notes:
- The extracted local board state already lives in a translated local
  coordinate system with a synthetic local board outline.
- The resulting `SubCircuitLayout` is treated as a frozen rigid artifact
  candidate for later parent-level composition.
"""

from __future__ import annotations

import math

from .subcircuit_extractor import ExtractedSubcircuitBoard
from .types import (
    Component,
    InterfaceAnchor,
    InterfacePort,
    Point,
    SilkscreenElement,
)


def infer_interface_anchors(
    ports: list[InterfacePort],
    components: dict[str, Component],
) -> list[InterfaceAnchor]:
    """Infer physical interface anchors from solved component pads.

    Current heuristic:
    - for each interface port, find all pads on the matching net
    - choose the pad closest to the outer edge of the solved local bbox
    - use that pad position as the interface anchor

    Net matching is normalized so schematic-facing names like `VBUS` can match
    PCB pad nets like `/VBUS`.

    This is intentionally simple for the first milestone. Later versions can
    incorporate:
    - preferred-side-aware anchor selection
    - connector pin ordering
    - explicit interface annotations
    - synthetic anchor points distinct from actual pads
    """
    if not ports or not components:
        return []

    bbox = _compute_component_bbox(components)
    anchors: list[InterfaceAnchor] = []

    for port in ports:
        candidate_pads = []
        for comp in components.values():
            for pad in comp.pads:
                if _nets_match(pad.net, port.net_name):
                    candidate_pads.append(pad)

        if not candidate_pads:
            continue

        best_pad = min(
            candidate_pads,
            key=lambda pad: _edge_distance_score(
                pad.pos,
                bbox["min_x"],
                bbox["min_y"],
                bbox["max_x"],
                bbox["max_y"],
            ),
        )

        anchors.append(
            InterfaceAnchor(
                port_name=port.name,
                pos=Point(best_pad.pos.x, best_pad.pos.y),
                layer=best_pad.layer,
                pad_ref=(best_pad.ref, best_pad.pad_id),
            )
        )

    return anchors


def _build_leaf_silkscreen(
    solved_components: dict[str, Component],
    bbox: dict[str, float],
    extraction: ExtractedSubcircuitBoard,
    config: dict,
) -> list[SilkscreenElement]:
    """Build silkscreen elements (rounded-rect outline + text label) for a leaf.

    The label text is looked up from the project config `group_labels` mapping
    (IC ref -> label text).  If no matching label is found, an empty list is
    returned so unlabeled leaves pass through silently.
    """
    group_labels: dict[str, str] = config.get("group_labels", {})
    if not group_labels:
        return []

    label_text = ""
    for ref, label in group_labels.items():
        if ref in solved_components:
            label_text = label
            break

    if not label_text:
        return []

    margin = float(config.get("silkscreen_margin_mm", 0.5))
    radius = float(config.get("silkscreen_corner_radius_mm", 1.0))
    stroke_width = float(config.get("silkscreen_stroke_width_mm", 0.15))
    font_height = float(config.get("silkscreen_font_height_mm", 0.8))
    font_width = float(config.get("silkscreen_font_width_mm", 0.8))
    font_thickness = float(config.get("silkscreen_font_thickness_mm", 0.15))
    layer = "F.SilkS"

    x0 = bbox["min_x"] - margin
    y0 = bbox["min_y"] - margin
    x1 = bbox["max_x"] + margin
    y1 = bbox["max_y"] + margin

    r = min(radius, (x1 - x0) / 2, (y1 - y0) / 2)
    points: list[Point] = []
    corners = [
        (x0 + r, y0 + r, math.pi, math.pi / 2),       # top-left
        (x1 - r, y0 + r, math.pi / 2, 0),              # top-right
        (x1 - r, y1 - r, 0, -math.pi / 2),             # bottom-right
        (x0 + r, y1 - r, -math.pi / 2, -math.pi),      # bottom-left
    ]
    n_arc = 8
    for cx, cy, a_start, a_end in corners:
        for i in range(n_arc):
            t = a_start + (a_end - a_start) * i / (n_arc - 1)
            px = cx + r * math.cos(t)
            py = cy - r * math.sin(t)  # KiCad Y-down
            points.append(Point(px, py))

    poly_element = SilkscreenElement(
        kind="poly",
        layer=layer,
        points=points,
        stroke_width=stroke_width,
    )

    text_x = x0 + r + 0.3
    text_y = y0 + font_height / 2 + 0.5
    text_element = SilkscreenElement(
        kind="text",
        layer=layer,
        text=label_text,
        pos=Point(text_x, text_y),
        font_height=font_height,
        font_width=font_width,
        font_thickness=font_thickness,
    )

    return [poly_element, text_element]


def _compute_component_bbox(components: dict[str, Component]) -> dict[str, float]:
    """Compute a tight bbox around solved components and pads."""
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for comp in components.values():
        tl, br = comp.bbox()
        min_x = min(min_x, tl.x)
        min_y = min(min_y, tl.y)
        max_x = max(max_x, br.x)
        max_y = max(max_y, br.y)

        for pad in comp.pads:
            min_x = min(min_x, pad.pos.x)
            min_y = min(min_y, pad.pos.y)
            max_x = max(max_x, pad.pos.x)
            max_y = max(max_y, pad.pos.y)

    if min_x == float("inf"):
        min_x = min_y = max_x = max_y = 0.0

    return {
        "min_x": min_x,
        "min_y": min_y,
        "max_x": max_x,
        "max_y": max_y,
        "width_mm": max(0.0, max_x - min_x),
        "height_mm": max(0.0, max_y - min_y),
    }


def _edge_distance_score(
    pos: Point,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> float:
    """Lower score means closer to an outer edge of the local bbox."""
    return min(
        abs(pos.x - min_x),
        abs(pos.x - max_x),
        abs(pos.y - min_y),
        abs(pos.y - max_y),
    )


def _normalize_net_name(net_name: str) -> str:
    """Normalize schematic/PCB net names for interface matching."""
    return str(net_name or "").strip().lstrip("/").upper()


def _nets_match(left: str, right: str) -> bool:
    """Return True when two net names refer to the same logical net."""
    return _normalize_net_name(left) == _normalize_net_name(right)


__all__ = [
    "infer_interface_anchors",
]
