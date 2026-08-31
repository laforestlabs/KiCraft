"""Final geometry verification for persisted antenna edge contracts."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable

import pcbnew

from . import geometry
from .types import AntennaEdgeIntent, Point

_SIDES = {"left", "right", "top", "bottom"}
_VECTORS = {
    "left": Point(-1.0, 0.0),
    "right": Point(1.0, 0.0),
    "top": Point(0.0, -1.0),
    "bottom": Point(0.0, 1.0),
}


@dataclass(frozen=True, slots=True)
class AntennaEdgeVerdict:
    ref: str
    edge: str
    gap_mm: float
    direction_ok: bool
    pads_inboard: bool


def intent_from_dict(row: dict[str, Any]) -> AntennaEdgeIntent:
    direction = str(row.get("local_direction", ""))
    edge = str(row.get("target_edge", ""))
    if direction not in _SIDES or edge not in _SIDES:
        raise ValueError(f"invalid antenna intent direction/edge: {direction}/{edge}")
    midpoint = row.get("local_anchor_midpoint")
    if not isinstance(midpoint, dict):
        raise ValueError("antenna intent missing local_anchor_midpoint")
    polygon = row.get("local_polygon") or []
    return AntennaEdgeIntent(
        owner_ref=str(row.get("owner_ref", "")),
        source=str(row.get("source", "")),
        source_id=str(row.get("source_id", "")),
        local_direction=direction,
        local_anchor_mm=float(row.get("local_anchor_mm", 0.0)),
        local_anchor_midpoint=Point(float(midpoint["x"]), float(midpoint["y"])),
        local_polygon=tuple(
            Point(float(point["x"]), float(point["y"]))
            for point in polygon
            if isinstance(point, dict) and "x" in point and "y" in point
        ),
        target_edge=edge,
        inset_mm=float(row.get("inset_mm", 0.0)),
        explicit_edge=bool(row.get("explicit_edge", False)),
        explicit_rotation=bool(row.get("explicit_rotation", False)),
    )


def verify_antenna_edges(
    board_path: str,
    intents: Iterable[AntennaEdgeIntent | dict[str, Any]],
    *,
    tolerance_mm: float = 0.10,
    pad_inset_margin_mm: float = 0.3,
) -> tuple[list[AntennaEdgeVerdict], list[str]]:
    """Measure final anchor, direction, and pad containment from persisted intent."""
    board = pcbnew.LoadBoard(board_path)
    edge_box = board.GetBoardEdgesBoundingBox()
    edges = {
        "left": pcbnew.ToMM(edge_box.GetLeft()),
        "right": pcbnew.ToMM(edge_box.GetRight()),
        "top": pcbnew.ToMM(edge_box.GetTop()),
        "bottom": pcbnew.ToMM(edge_box.GetBottom()),
    }
    footprints = {fp.GetReferenceAsString(): fp for fp in board.Footprints()}
    verdicts: list[AntennaEdgeVerdict] = []
    violations: list[str] = []

    for raw_intent in intents:
        intent = intent_from_dict(raw_intent) if isinstance(raw_intent, dict) else raw_intent
        fp = footprints.get(intent.owner_ref)
        if fp is None:
            violations.append(f"antenna_constraint_owner_missing:{intent.owner_ref}")
            continue
        flipped = fp.GetLayer() == pcbnew.B_Cu
        midpoint = intent.local_anchor_midpoint
        direction = _VECTORS[intent.local_direction]
        if flipped:
            midpoint = Point(-midpoint.x, midpoint.y)
            direction = Point(-direction.x, direction.y)
        pos = fp.GetPosition()
        origin = Point(pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))
        anchor = geometry.transform_point(
            midpoint, origin, float(fp.GetOrientationDegrees())
        )
        actual_direction = geometry.rotate_vector(
            direction, float(fp.GetOrientationDegrees())
        )
        expected_direction = _VECTORS[intent.target_edge]
        direction_ok = (
            abs(actual_direction.x - expected_direction.x) <= 1e-6
            and abs(actual_direction.y - expected_direction.y) <= 1e-6
        )
        if intent.target_edge == "left":
            gap = anchor.x - edges["left"]
        elif intent.target_edge == "right":
            gap = edges["right"] - anchor.x
        elif intent.target_edge == "top":
            gap = anchor.y - edges["top"]
        else:
            gap = edges["bottom"] - anchor.y

        pads_inboard = True
        for pad in fp.Pads():
            box = pad.GetBoundingBox()
            if (
                pcbnew.ToMM(box.GetLeft()) < edges["left"] + pad_inset_margin_mm
                or pcbnew.ToMM(box.GetRight()) > edges["right"] - pad_inset_margin_mm
                or pcbnew.ToMM(box.GetTop()) < edges["top"] + pad_inset_margin_mm
                or pcbnew.ToMM(box.GetBottom()) > edges["bottom"] - pad_inset_margin_mm
            ):
                pads_inboard = False
                break

        verdicts.append(
            AntennaEdgeVerdict(
                ref=intent.owner_ref,
                edge=intent.target_edge,
                gap_mm=gap,
                direction_ok=direction_ok,
                pads_inboard=pads_inboard,
            )
        )
        if abs(gap - intent.inset_mm) > tolerance_mm:
            violations.append(
                f"antenna_stranded:{intent.owner_ref}@{gap:.2f}mm({intent.target_edge})"
            )
        if not direction_ok:
            actual_angle = math.degrees(math.atan2(actual_direction.y, actual_direction.x)) % 360.0
            expected_angle = math.degrees(math.atan2(expected_direction.y, expected_direction.x)) % 360.0
            violations.append(
                f"antenna_misoriented:{intent.owner_ref}({actual_angle:.0f}->{expected_angle:.0f})"
            )
        if not pads_inboard:
            violations.append(f"antenna_edge_pad_conflict:{intent.owner_ref}")

    return verdicts, violations
