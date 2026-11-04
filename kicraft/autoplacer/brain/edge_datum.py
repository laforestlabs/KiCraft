"""Shared geometry rules for exact board-edge interface datums."""

from __future__ import annotations

from typing import Any

from kicraft.autoplacer.brain.types import Component


_EDGE_SIDES = frozenset({"left", "right", "top", "bottom"})


def exact_edge_interfaces(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Return valid, explicitly datum-controlled edge-interface declarations."""
    interfaces: list[dict[str, Any]] = []
    for raw in cfg.get("edge_interfaces", []) or []:
        if not isinstance(raw, dict) or not raw.get("exact_edge_datum", False):
            continue
        refs = [str(ref) for ref in raw.get("refs", []) or []]
        side = str(raw.get("side", ""))
        try:
            pitch = float(raw.get("pitch_mm", 0.0))
        except (TypeError, ValueError):
            pitch = 0.0
        if refs and side in _EDGE_SIDES and pitch > 0.0:
            interfaces.append({**raw, "refs": refs, "side": side, "pitch_mm": pitch})
    return interfaces


def exact_edge_interface_refs(cfg: dict[str, Any]) -> set[str]:
    """References whose position is a function of a candidate outline."""
    return {ref for interface in exact_edge_interfaces(cfg) for ref in interface["refs"]}


def edge_datum_requirements(
    components: dict[str, Component],
    cfg: dict[str, Any],
    *,
    edge_margin_mm: float,
) -> dict[str, Any]:
    """Measure exact edge interfaces independently of their current outline.

    The returned depths are measured from the datum pad to the in-board physical
    extent.  Parallel-axis floors include the declared pitch span and the same
    edge margin used by the placement routines.  This deliberately leaves
    undecodable and overlapping declarations to normal legality checks.
    """
    margin = max(0.0, float(edge_margin_mm))
    depths = {side: 0.0 for side in _EDGE_SIDES}
    width_floor = 0.0
    height_floor = 0.0
    groups: list[dict[str, Any]] = []
    for interface in exact_edge_interfaces(cfg):
        refs = interface["refs"]
        members = [components.get(ref) for ref in refs]
        if any(member is None or not member.pads for member in members):
            continue
        side = interface["side"]
        span = (len(refs) - 1) * interface["pitch_mm"]
        inboard_depth = 0.0
        parallel_extent = 0.0
        for member in members:
            assert member is not None
            datum = member.pads[0]
            tl, br = member.physical_bbox()
            if side == "left":
                inboard_depth = max(inboard_depth, br.x - datum.pos.x)
                parallel_extent = max(parallel_extent, datum.pos.y - tl.y, br.y - datum.pos.y)
            elif side == "right":
                inboard_depth = max(inboard_depth, datum.pos.x - tl.x)
                parallel_extent = max(parallel_extent, datum.pos.y - tl.y, br.y - datum.pos.y)
            elif side == "top":
                inboard_depth = max(inboard_depth, br.y - datum.pos.y)
                parallel_extent = max(parallel_extent, datum.pos.x - tl.x, br.x - datum.pos.x)
            else:
                inboard_depth = max(inboard_depth, datum.pos.y - tl.y)
                parallel_extent = max(parallel_extent, datum.pos.x - tl.x, br.x - datum.pos.x)
        depths[side] = max(depths[side], max(0.0, inboard_depth))
        parallel_floor = span + 2.0 * max(margin, parallel_extent)
        if side in {"left", "right"}:
            height_floor = max(height_floor, parallel_floor)
        else:
            width_floor = max(width_floor, parallel_floor)
        groups.append(
            {
                "refs": list(refs),
                "side": side,
                "pitch_mm": interface["pitch_mm"],
                "span_mm": span,
                "inboard_depth_mm": max(0.0, inboard_depth),
                "parallel_floor_mm": parallel_floor,
            }
        )
    return {
        "groups": groups,
        "refs": {ref for group in groups for ref in group["refs"]},
        "depths_mm": depths,
        "parallel_width_floor_mm": width_floor,
        "parallel_height_floor_mm": height_floor,
    }


def edge_datum_outline_floors(
    requirements: dict[str, Any],
    *,
    content_width_mm: float,
    content_height_mm: float,
    content_margin_mm: float,
) -> tuple[float, float]:
    """Return candidate-outline floors for content bracketed by edge datums."""
    depths = requirements["depths_mm"]
    margin = max(0.0, float(content_margin_mm))
    width = max(
        float(requirements["parallel_width_floor_mm"]),
        max(0.0, content_width_mm) + depths["left"] + depths["right"] + 2.0 * margin,
    )
    height = max(
        float(requirements["parallel_height_floor_mm"]),
        max(0.0, content_height_mm) + depths["top"] + depths["bottom"] + 2.0 * margin,
    )
    return width, height
