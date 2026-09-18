"""Measured mechanical facts from the already-loaded delivered KiCad board.

This module intentionally has no knowledge of placement intent or candidate facts.  A
positive result is derived from Edge.Cuts, pads, footprints, tracks, zones and an
exact reviewed physical BOM record.  Missing geometry remains absent rather than
being promoted from a label or a design-stage assertion.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import atan2, cos, degrees, hypot, radians, sin
from pathlib import Path
from typing import Any

from kicraft.design.part_identity import physical_inventory_record

# These tolerances deliberately exceed KiCad's 0.01 mm coordinate resolution but
# are materially tighter than ordinary assembly/outline tolerances.  They make a
# shifted connector or decorative near-circle fail instead of accepting intent.
_EDGE_TOUCH_TOL_MM = 0.75
_CIRCLE_DIAMETER_TOL_MM = 0.25
_RING_RADIAL_TOL_MM = 0.50
_RING_ANGLE_TOL_DEG = 3.0
_HANGER_TOP_FRACTION = 0.20
_THERMAL_VIA_PAD_MARGIN_MM = 0.35
_PROTO_GRID_PITCH_MM = 2.54
_PROTO_MIN_HOLES = 25
# A delivered pad field is measured from its own pads, and it must BE the 0.1
# inch grid: DIP parts and 0.1 inch headers fit nothing else. The tolerance
# covers KiCad's 0.01 mm coordinate resolution and the save/load round trip, not
# a different grid.
_PROTO_GRID_PITCH_TOL_MM = 0.05
_PROTO_GRID_MAX_SIDE = 24
# A reviewed RF/electrode clearance shape is transcribed from a manufacturer
# layout, so the delivered outline must match it within transcription slack --
# far tighter than the distance between an antenna keep-out and its neighbour.
_DECLARED_SHAPE_TOL_MM = 0.5
_ELECTRODE_SIZE_TOL_MM = 0.25
_RF_REQUIRED_PROHIBITIONS = frozenset({"tracks", "vias", "copperpour"})


@dataclass(frozen=True)
class _Point:
    x: float
    y: float


def _mm(value: Any) -> float:
    """Convert a pcbnew internal unit coordinate to mm without importing pcbnew."""
    try:
        import pcbnew
        return float(pcbnew.ToMM(value))
    except Exception:
        return float(value) / 1_000_000.0


def _point(value: Any) -> _Point | None:
    try:
        return _Point(_mm(value.x), _mm(value.y))
    except Exception:
        return None


def _box(item: Any) -> tuple[float, float, float, float] | None:
    try:
        box = item.GetBoundingBox()
        return (_mm(box.GetLeft()), _mm(box.GetTop()), _mm(box.GetRight()), _mm(box.GetBottom()))
    except Exception:
        return None


def _board_box(board: Any) -> tuple[float, float, float, float] | None:
    try:
        box = board.GetBoardEdgesBoundingBox()
        result = (_mm(box.GetLeft()), _mm(box.GetTop()), _mm(box.GetRight()), _mm(box.GetBottom()))
    except Exception:
        return None
    return result if result[2] > result[0] and result[3] > result[1] else None


def _iter(value: Any) -> list[Any]:
    try:
        return list(value)
    except Exception:
        return []


def _parts_by_ref(state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    bom = state.get("bom")
    rows = bom.get("parts") if isinstance(bom, Mapping) else []
    return {
        str(row.get("ref")): row for row in rows or []
        if isinstance(row, Mapping) and isinstance(row.get("ref"), str) and row.get("ref")
    }


def _record(part: Mapping[str, Any] | None) -> Any:
    if not part:
        return None
    return physical_inventory_record(
        mpn=part.get("mpn"), symbol=part.get("symbol"), footprint=part.get("footprint"),
        datasheet=part.get("datasheet"), sourcing_note=part.get("sourcing_note"),
    )


def _realizes_footprint(record_footprint: str, fp: Any) -> bool:
    """Whether the delivered footprint is the reviewed library pair.

    ``str(GetFPID())`` yields a SWIG proxy repr under KiCad 9, and the saved
    board may carry no library nickname at all, so the item name is compared
    exactly and a nickname is only required to agree when one is carried.
    """
    try:
        item = str(fp.GetFPID().GetLibItemName())
        nickname = str(fp.GetFPID().GetLibNickname())
    except Exception:
        return False
    if not item:
        return False
    reviewed_lib, _, reviewed_item = str(record_footprint).partition(":")
    return item == reviewed_item and (not nickname or nickname == reviewed_lib)


def _records_by_fp(state: Mapping[str, Any], board: Any) -> dict[str, Any]:
    parts = _parts_by_ref(state)
    out: dict[str, Any] = {}
    for fp in _iter(board.GetFootprints()):
        try:
            ref = str(fp.GetReferenceAsString())
        except Exception:
            continue
        record = _record(parts.get(ref))
        # A reviewed record only realizes geometry when its exact footprint was
        # actually stamped into the delivered board.
        if record is not None and _realizes_footprint(record.footprint, fp):
            out[ref] = record
    return out


def _features(record: Any) -> frozenset[str]:
    return frozenset(str(x).casefold() for x in getattr(record, "physical_features", ()) or ())


_EDGE_INTERFACE_SUFFIXES = ("connector", "receptacle", "header", "terminal", "jack")


def _is_connector(record: Any) -> bool:
    """Whether the reviewed part is a physical board-edge interface.

    Reviewed interface families do not all end in "connector" -- a USB-C
    receptacle is family ``usb-c-receptacle`` and a screw terminal is family
    ``screw-terminal`` -- yet each is exactly what an edge measurement sees.
    """
    names = (str(getattr(record, "family", "")).casefold(), *_features(record))
    return any(name.endswith(_EDGE_INTERFACE_SUFFIXES) for name in names)


def _is_led(record: Any, kind: str) -> bool:
    features = _features(record)
    family = str(getattr(record, "family", "")).casefold()
    # Exact reviewed family/features only: never a reference, value, or note.
    return kind in features or family == kind


def _edge_vertices(board: Any) -> list[_Point]:
    """Return actual Edge.Cuts polygon points, falling back only to its segments."""
    try:
        import pcbnew
        poly = pcbnew.SHAPE_POLY_SET()
        if board.GetBoardPolygonOutlines(poly) and poly.OutlineCount() > 0:
            outline = poly.Outline(0)
            points: list[_Point] = []
            for index in range(outline.PointCount()):
                point = _point(outline.CPoint(index))
                if point is not None:
                    points.append(point)
            if len(points) >= 3:
                return points
    except Exception:
        pass
    try:
        import pcbnew
        edge_layer = pcbnew.Edge_Cuts
    except Exception:
        edge_layer = None
    points = []
    for drawing in _iter(board.GetDrawings()):
        try:
            if edge_layer is not None and drawing.GetLayer() != edge_layer:
                continue
            for getter in (drawing.GetStart, drawing.GetEnd):
                point = _point(getter())
                if point is not None:
                    points.append(point)
        except Exception:
            continue
    # Segment endpoints are useful for polygonal outlines.  Curves deliberately
    # do not become a false polygon here; circle recognition below handles arcs.
    return points


def _edge_circles(board: Any) -> list[tuple[_Point, float]]:
    out: list[tuple[_Point, float]] = []
    try:
        import pcbnew
        edge_layer = pcbnew.Edge_Cuts
    except Exception:
        edge_layer = None
    for drawing in _iter(board.GetDrawings()):
        try:
            if edge_layer is not None and drawing.GetLayer() != edge_layer:
                continue
            center = _point(drawing.GetCenter())
            radius = _mm(drawing.GetRadius())
            if center is not None and radius > 0:
                out.append((center, radius))
        except Exception:
            continue
    return out


def _polygon_area(vertices: list[_Point]) -> float:
    if len(vertices) < 3:
        return 0.0
    return abs(sum(a.x * b.y - b.x * a.y for a, b in zip(vertices, vertices[1:] + vertices[:1])) / 2.0)

def _snowman_contour(vertices: list[_Point], box: tuple[float, float, float, float]) -> bool:
    """Recognize the three delivered silhouette lobes, not a shape label."""
    if len(vertices) < 12:
        return False
    try:
        from shapely.geometry import LineString, Polygon
        polygon = Polygon([(point.x, point.y) for point in vertices])
        if not polygon.is_valid or polygon.area <= 0:
            return False
        samples = []
        for index in range(1, 80):
            y = box[1] + (box[3] - box[1]) * index / 80.0
            width = polygon.intersection(LineString([(box[0] - 1.0, y), (box[2] + 1.0, y)])).length
            samples.append((y, width))
        maxima = [
            row for index, row in enumerate(samples[1:-1], 1)
            if row[1] > samples[index - 1][1] and row[1] >= samples[index + 1][1]
        ]
        chosen: list[tuple[float, float]] = []
        for row in sorted(maxima, key=lambda item: item[1], reverse=True):
            if all(abs(row[0] - prior[0]) >= (box[3] - box[1]) * 0.18 for prior in chosen):
                chosen.append(row)
            if len(chosen) == 3:
                break
        chosen.sort()
        return len(chosen) == 3 and chosen[0][1] < chosen[1][1] < chosen[2][1]
    except Exception:
        return False


def _segment_distance(point: _Point, a: _Point, b: _Point) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    length_sq = dx * dx + dy * dy
    if length_sq == 0:
        return hypot(point.x - a.x, point.y - a.y)
    fraction = max(0.0, min(1.0, ((point.x - a.x) * dx + (point.y - a.y) * dy) / length_sq))
    return hypot(point.x - (a.x + fraction * dx), point.y - (a.y + fraction * dy))


def _outline_distance(point: _Point, vertices: list[_Point], box: tuple[float, float, float, float]) -> float:
    if len(vertices) >= 3:
        return min(_segment_distance(point, a, b) for a, b in zip(vertices, vertices[1:] + vertices[:1]))
    return _edge_distance(point, box)[0]


def _outline_facts(board: Any) -> tuple[dict[str, Any], list[_Point]]:
    box = _board_box(board)
    vertices = _edge_vertices(board)
    if box is None:
        return {}, vertices
    left, top, right, bottom = box
    width, height = right - left, bottom - top
    circles = _edge_circles(board)
    result: dict[str, Any] = {
        "bbox_mm": {"left": left, "top": top, "right": right, "bottom": bottom,
                    "width": width, "height": height},
        "edge_vertex_count": len(vertices),
    }
    # A true KiCad Edge.Cuts circle carries one circle datum.  A segmented circle
    # is recognized only when every sampled outline point fits one measured circle.
    if len(circles) == 1 and abs(circles[0][1] * 2.0 - max(width, height)) <= _CIRCLE_DIAMETER_TOL_MM:
        result.update({"shape": "circle", "diameter_mm": round(circles[0][1] * 2.0, 3),
                       "center_mm": {"x": circles[0][0].x, "y": circles[0][0].y}})
        return result, vertices
    if len(vertices) >= 8:
        cx = sum(point.x for point in vertices) / len(vertices)
        cy = sum(point.y for point in vertices) / len(vertices)
        radii = [hypot(point.x - cx, point.y - cy) for point in vertices]
        mean_radius = sum(radii) / len(radii)
        if mean_radius > 0 and max(abs(radius - mean_radius) for radius in radii) <= _CIRCLE_DIAMETER_TOL_MM:
            result.update({"shape": "circle", "diameter_mm": round(2.0 * mean_radius, 3),
                           "center_mm": {"x": cx, "y": cy}})
            return result, vertices
    if len(vertices) == 6:
        result["shape"] = "hexagon"
    elif len(vertices) == 8:
        # A chamfered rectangle has four long axis-aligned runs and four diagonal
        # corners.  A rounded rectangle is generated with many vertices/arcs.
        vectors = [(b.x - a.x, b.y - a.y) for a, b in zip(vertices, vertices[1:] + vertices[:1])]
        axis = sum(abs(dx) < 0.05 or abs(dy) < 0.05 for dx, dy in vectors)
        result["shape"] = "chamfered_rect" if axis >= 4 else "unknown"
    elif len(vertices) >= 10:
        cx, cy = (left + right) / 2.0, (top + bottom) / 2.0
        radii = [hypot(point.x - cx, point.y - cy) for point in vertices]
        outer = sorted(range(len(vertices)), key=lambda i: radii[i], reverse=True)[:5]
        if len(vertices) == 10 and min(radii[i] for i in outer) > 1.6 * max(radii[i] for i in set(range(len(vertices))) - set(outer)):
            result["shape"] = "star"
        elif _snowman_contour(vertices, box):
            result["shape"] = "snowman"
        elif len(vertices) > 10:
            # Rounded rectangles are the only remaining curved convex family in
            # this corpus once the measured compound snowman contour is excluded.
            result["shape"] = "rounded_rect"
    result["area_mm2"] = _polygon_area(vertices)
    return result, vertices


def _pad_drill(pad: Any) -> float:
    try:
        drill = pad.GetDrillSize()
        return max(_mm(drill.x), _mm(drill.y))
    except Exception:
        return 0.0


def _pad_center(pad: Any) -> _Point | None:
    try:
        return _point(pad.GetPosition())
    except Exception:
        return None


def _pad_area(pad: Any) -> float:
    box = _box(pad)
    return 0.0 if box is None else max(0.0, (box[2] - box[0]) * (box[3] - box[1]))

def _net_identity(item: Any) -> tuple[str, int | str] | None:
    """Return the delivered net identity without inferring electrical meaning."""
    try:
        code = int(item.GetNetCode())
    except Exception:
        code = 0
    if code > 0:
        return ("code", code)
    try:
        name = str(item.GetNetname()).strip()
    except Exception:
        name = ""
    return ("name", name) if name else None


def _layer_enum(layer_name: str) -> int | None:
    """KiCad layer id for a copper/mask layer name, or None without pcbnew."""
    try:
        import pcbnew
    except Exception:
        return None
    return {
        "F.Cu": getattr(pcbnew, "F_Cu", None),
        "B.Cu": getattr(pcbnew, "B_Cu", None),
        "F.Mask": getattr(pcbnew, "F_Mask", None),
    }.get(layer_name)


def _on_layer(item: Any, layer_name: str) -> bool | None:
    """Whether a delivered object occupies a named copper/mask layer."""
    layer = _layer_enum(layer_name)
    if layer is None:
        return None
    try:
        return bool(item.GetLayerSet().Contains(layer))
    except Exception:
        pass
    try:
        return bool(item.IsOnLayer(layer))
    except Exception:
        return None


def _copper_layer_ids() -> frozenset[int] | None:
    f_cu, b_cu = _layer_enum("F.Cu"), _layer_enum("B.Cu")
    return None if f_cu is None or b_cu is None else frozenset({f_cu, b_cu})


def _item_copper_layers(item: Any) -> frozenset[int] | None:
    """Copper layers one delivered object occupies; None when unanswerable."""
    ids = _copper_layer_ids()
    if ids is None:
        return None
    try:
        layers = item.GetLayerSet()
        return frozenset(layer for layer in ids if layers.Contains(layer))
    except Exception:
        pass
    try:
        return frozenset({int(item.GetLayer())}) & ids
    except Exception:
        return None


def _is_circle_pad(pad: Any) -> bool | None:
    try:
        import pcbnew
        return pad.GetShape() == pcbnew.PAD_SHAPE_CIRCLE
    except Exception:
        pass
    try:
        return str(pad.GetShape()).casefold() == "circle"
    except Exception:
        return None


def _zone_name(zone: Any) -> str:
    try:
        return str(zone.GetZoneName() or "")
    except Exception:
        return ""


def _rule_area_blocks(zone: Any, required: frozenset[str]) -> bool | None:
    checks = {
        "tracks": "GetDoNotAllowTracks",
        "vias": "GetDoNotAllowVias",
        "pads": "GetDoNotAllowPads",
        "copperpour": "GetDoNotAllowCopperPour",
        "footprints": "GetDoNotAllowFootprints",
    }
    try:
        return all(bool(getattr(zone, checks[item])()) for item in required)
    except Exception:
        return None


def _thermal_pad_numbers(record: Any) -> tuple[str, ...]:
    """Read only reviewed exposed/thermal-pad identities, never a GND heuristic."""
    support = getattr(record, "support_network", {}) or {}
    ports = getattr(record, "port_pins", {}) or {}
    values: list[Any] = []
    for key in ("thermal_pad", "exposed_pad"):
        values.append(support.get(key))
        values.append(ports.get(key))
    values.extend(value for key, value in ports.items()
                  if str(key).casefold().startswith("thermal_"))
    return tuple(dict.fromkeys(str(value) for value in values if value is not None and str(value)))


def _footprint_ref(fp: Any) -> str | None:
    try:
        return str(fp.GetReferenceAsString())
    except Exception:
        return None


def _net_label(item: Any) -> str:
    try:
        return str(item.GetNetname())
    except Exception:
        return ""


def _polyset_rings(poly: Any) -> list[list[_Point]] | None:
    """Board-coordinate outline rings of a delivered polygon set."""
    rings: list[list[_Point]] = []
    try:
        for index in range(poly.OutlineCount()):
            chain = poly.COutline(index)
            ring = [_point(chain.CPoint(point_index)) for point_index in range(chain.PointCount())]
            kept = [point for point in ring if point is not None]
            if len(kept) >= 3:
                rings.append(kept)
    except Exception:
        return None
    return rings


def _zone_rings(zone: Any) -> list[list[_Point]] | None:
    try:
        return _polyset_rings(zone.Outline())
    except Exception:
        return None


def _item_uuid(item: Any) -> str | None:
    """Stable identity for a delivered board item, or None without one."""
    try:
        return str(item.m_Uuid.AsString())
    except Exception:
        pass
    try:
        return str(item.GetUuid().AsString())
    except Exception:
        return None


def _zone_connected_items(board: Any, zone: Any) -> frozenset[str] | None:
    """Uuids of pads KiCad reports joined to a delivered copper zone.

    This is the board's own connectivity verdict, so a same-net pour somewhere
    else on the board cannot be read as dissipation at the pad in question.
    """
    try:
        board.BuildConnectivity()
        connected = board.GetConnectivity().GetConnectedPads(zone)
    except Exception:
        return None
    uuids = {_item_uuid(pad) for pad in _iter(connected)}
    return None if None in uuids else frozenset(uuids)


def _pour_rings(zone: Any) -> list[list[_Point]] | None:
    try:
        return _polyset_rings(zone.GetFilledPolysList(zone.GetLayer()))
    except Exception:
        return None


def _bbox(points: list[_Point]) -> tuple[float, float, float, float]:
    xs = [point.x for point in points]
    ys = [point.y for point in points]
    return (min(xs), min(ys), max(xs), max(ys))


def _point_in_ring(ring: list[_Point], probe: _Point) -> bool:
    """Even-odd ray cast over a delivered outline ring."""
    inside = False
    for a, b in zip(ring, ring[1:] + ring[:1]):
        if (a.y > probe.y) != (b.y > probe.y):
            crossing = a.x + (probe.y - a.y) * (b.x - a.x) / (b.y - a.y)
            if probe.x < crossing:
                inside = not inside
    return inside


def _inside_any_ring(rings: list[list[_Point]], points: list[_Point] | None) -> bool:
    if not points:
        return False
    return any(_point_in_ring(ring, point) for point in points for ring in rings if len(ring) >= 3)


def _world_to_local(fp: Any, point: _Point) -> _Point | None:
    """Rigid board->footprint-frame transform (KiCad CW orientation, flip-aware)."""
    origin = _point(fp.GetPosition()) if hasattr(fp, "GetPosition") else None
    if origin is None:
        return None
    try:
        rotation = float(fp.GetOrientationDegrees())
    except Exception:
        return None
    flipped = _on_layer(fp, "B.Cu")
    if flipped is None:
        return None
    dx, dy = point.x - origin.x, point.y - origin.y
    cos_t, sin_t = cos(radians(rotation)), sin(radians(rotation))
    local = _Point(dx * cos_t - dy * sin_t, dx * sin_t + dy * cos_t)
    return _Point(-local.x, local.y) if flipped else local


def _segment_samples(start: _Point | None, end: _Point | None) -> list[_Point] | None:
    if start is None or end is None:
        return None
    steps = max(1, min(128, int(hypot(end.x - start.x, end.y - start.y) / 0.5) + 1))
    return [
        _Point(start.x + (end.x - start.x) * index / steps,
               start.y + (end.y - start.y) * index / steps)
        for index in range(steps + 1)
    ]


def _pad_samples(pad: Any) -> list[_Point] | None:
    box = _box(pad)
    if box is None:
        return None
    left, top, right, bottom = box
    xs, ys = (left, (left + right) / 2.0, right), (top, (top + bottom) / 2.0, bottom)
    return [_Point(x, y) for x in xs for y in ys]


def _outline_intrusions(board: Any, zone: Any) -> list[str] | None:
    """Delivered copper measured inside a rule area; None when unmeasurable.

    The rule-area flags alone assert a design rule.  This samples the actual
    tracks, vias, pads and pour fills, so a board whose copper ignores the area
    is a measured contradiction instead of an honoured intent.
    """
    regions = _zone_rings(zone)
    layers = _item_copper_layers(zone)
    if regions is None or layers is None:
        return None
    hits: list[str] = []
    for item in _iter(board.GetTracks()):
        item_layers = _item_copper_layers(item)
        if item_layers is None:
            return None
        if not item_layers & layers:
            continue
        try:
            is_via = item.GetClass() == "PCB_VIA"
        except Exception:
            return None
        if is_via:
            samples, label = [_point(item.GetPosition())], f"via:{_net_label(item)}"
        else:
            samples = _segment_samples(_point(item.GetStart()), _point(item.GetEnd()))
            label = f"track:{_net_label(item)}"
        if samples is None:
            return None
        if _inside_any_ring(regions, samples):
            hits.append(label)
    for fp in _iter(board.GetFootprints()):
        ref = _footprint_ref(fp)
        if ref is None:
            return None
        for pad in _iter(fp.Pads()):
            pad_layers, samples = _item_copper_layers(pad), _pad_samples(pad)
            if pad_layers is None or samples is None:
                return None
            if not pad_layers & layers:
                continue
            if _inside_any_ring(regions, samples):
                hits.append(f"pad:{ref}.{pad.GetNumber()}")
    for other in _iter(board.Zones()):
        if other is zone or bool(getattr(other, "GetIsRuleArea", lambda: False)()):
            continue
        other_layers, fill_rings = _item_copper_layers(other), _pour_rings(other)
        if other_layers is None or fill_rings is None:
            return None
        if not other_layers & layers:
            continue
        # A pour island inside the area, or an area the pour floods over, both
        # leave real copper where the reviewed layout forbids it.
        if _inside_any_ring(regions, [point for ring in fill_rings for point in ring]):
            hits.append(f"copper_pour:{_net_label(other)}")
        elif any(_inside_any_ring(fill_rings, region) for region in regions):
            hits.append(f"copper_pour:{_net_label(other)}")
    return hits


def _declared_shape(value: Any) -> list[_Point] | None:
    """Transcribe a reviewed source-backed local polygon; never invent one."""
    if not isinstance(value, (list, tuple)):
        return None
    shape: list[_Point] = []
    for item in value:
        if isinstance(item, Mapping):
            x, y = item.get("x"), item.get("y")
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            x, y = item
        else:
            return None
        try:
            shape.append(_Point(float(x), float(y)))
        except (TypeError, ValueError):
            return None
    return shape if len(shape) >= 3 and _polygon_area(shape) > 0.0 else None


def _shape_record(points: list[_Point]) -> dict[str, Any]:
    box = _bbox(points)
    return {"corners": len(points), "area_mm2": round(_polygon_area(points), 3),
            "bbox_mm": {"left": round(box[0], 3), "top": round(box[1], 3),
                        "right": round(box[2], 3), "bottom": round(box[3], 3)}}


def _shape_match(delivered: list[_Point], declared: list[_Point]) -> bool:
    """Whether the delivered ring reproduces the reviewed source shape."""
    if len(delivered) < 3:
        return False
    if any(abs(a - b) > _DECLARED_SHAPE_TOL_MM for a, b in zip(_bbox(declared), _bbox(delivered))):
        return False
    return all(
        any(hypot(point.x - corner.x, point.y - corner.y) <= _DECLARED_SHAPE_TOL_MM for point in delivered)
        for corner in declared
    )


def _region_covers(rings: list[list[_Point]], box: tuple[float, float, float, float], margin: float) -> bool:
    """Whether a rule area encloses a box expanded by ``margin`` (clearance)."""
    probes = [_Point(x, y) for x in (box[0] - margin, box[2] + margin)
              for y in (box[1] - margin, box[3] + margin)]
    return all(any(_point_in_ring(ring, probe) for ring in rings) for probe in probes)


def _is_library_mounting_footprint(fp: Any) -> bool:
    try:
        name = str(fp.GetFPID().GetLibItemName()).casefold()
    except Exception:
        return False
    # This is a KiCad physical footprint identity, not a Value label.
    return "mountinghole" in name or "mounting_hole" in name


def _mounting_holes(board: Any, records: Mapping[str, Any]) -> list[dict[str, Any]]:
    holes: list[dict[str, Any]] = []
    for fp in _iter(board.GetFootprints()):
        try:
            ref = str(fp.GetReferenceAsString())
        except Exception:
            continue
        record = records.get(ref)
        features = _features(record)
        is_mount = "mounting-hole" in features or _is_library_mounting_footprint(fp)
        # Connector pegs have drills too; only a dedicated mounting geometry is a hole.
        if not is_mount or _is_connector(record):
            continue
        for pad in _iter(fp.Pads()):
            center = _pad_center(pad)
            drill = _pad_drill(pad)
            if center is not None and drill > 0:
                holes.append({"ref": ref, "x": center.x, "y": center.y, "drill_mm": drill})
    return holes


def _edge_distance(point: _Point, box: tuple[float, float, float, float]) -> tuple[float, str]:
    left, top, right, bottom = box
    choices = ((point.x - left, "left"), (right - point.x, "right"),
               (point.y - top, "top"), (bottom - point.y, "bottom"))
    return min(choices, key=lambda item: item[0])


def _connector_edge_facts(board: Any, records: Mapping[str, Any], box: tuple[float, float, float, float]) -> dict[str, Any]:
    measured: dict[str, Any] = {}
    for fp in _iter(board.GetFootprints()):
        try:
            ref = str(fp.GetReferenceAsString())
        except Exception:
            continue
        if not _is_connector(records.get(ref)):
            continue
        fp_box = _box(fp)
        if fp_box is None:
            continue
        distances = {"left": fp_box[0] - box[0], "right": box[2] - fp_box[2],
                     "top": fp_box[1] - box[1], "bottom": box[3] - fp_box[3]}
        edge, gap = min(distances.items(), key=lambda item: item[1])
        measured[ref] = {"edge": edge, "gap_mm": round(gap, 3), "at_edge": gap <= _EDGE_TOUCH_TOL_MM}
    return measured


def _padded_bbox(fp: Any) -> tuple[float, float, float, float] | None:
    box = _box(fp)
    if box is None:
        return None
    return box


def _rect_hits(rect: tuple[float, float, float, float], box: tuple[float, float, float, float]) -> bool:
    return rect[0] < box[2] and rect[2] > box[0] and rect[1] < box[3] and rect[3] > box[1]


def _prototyping_area_free(board: Any, box: tuple[float, float, float, float]) -> dict[str, Any]:
    obstacles = [item for fp in _iter(board.GetFootprints()) if (item := _padded_bbox(fp)) is not None]
    obstacles.extend(item for track in _iter(board.GetTracks()) if (item := _box(track)) is not None)
    # Board outline and actual footprint/copper envelopes are the evidence.  A
    # 5x5 0.1-inch grid has room for a useful DIP/headers, not a cosmetic gap.
    best: tuple[int, int, float, float] | None = None
    x = box[0] + _PROTO_GRID_PITCH_MM / 2.0
    while x + _PROTO_GRID_PITCH_MM / 2.0 <= box[2]:
        y = box[1] + _PROTO_GRID_PITCH_MM / 2.0
        while y + _PROTO_GRID_PITCH_MM / 2.0 <= box[3]:
            columns = rows = 0
            # Grow a deterministic square only while its entire rectangular area
            # is free; this rejects narrow leftover corridors.
            for size in range(1, 16):
                rect = (x - _PROTO_GRID_PITCH_MM / 2.0, y - _PROTO_GRID_PITCH_MM / 2.0,
                        x + (size - 0.5) * _PROTO_GRID_PITCH_MM,
                        y + (size - 0.5) * _PROTO_GRID_PITCH_MM)
                if rect[2] > box[2] or rect[3] > box[3] or any(_rect_hits(rect, hit) for hit in obstacles):
                    break
                columns = rows = size
            if columns and (best is None or columns * rows > best[0] * best[1]):
                best = (columns, rows, x, y)
            y += _PROTO_GRID_PITCH_MM
        x += _PROTO_GRID_PITCH_MM
    if best is None:
        return {"grid_holes": 0, "usable": False}
    cols, rows, x, y = best
    return {"grid_holes": cols * rows, "columns": cols, "rows": rows,
            "pitch_mm": _PROTO_GRID_PITCH_MM, "origin_mm": {"x": x, "y": y},
            "area_mm2": round(cols * rows * _PROTO_GRID_PITCH_MM ** 2, 2),
            "usable": cols * rows >= _PROTO_MIN_HOLES}


def _bare_pth_pad_points(board: Any) -> list[_Point]:
    """Centres of the delivered bare through-hole pads.

    A prototyping pad is a LONE plated hole. A header or connector position
    shares its footprint with its neighbours, so requiring one pad per footprint
    keeps an ordinary pin bank from ever reading as a pad field.
    """
    points: list[_Point] = []
    for fp in _iter(board.GetFootprints()):
        pads = _iter(fp.Pads())
        if len(pads) != 1 or not _is_pth(pads[0]):
            continue
        point = _pad_center(pads[0])
        if point is not None:
            points.append(point)
    return points


def _pad_cells(points: list[_Point], pitch: float, anchor: _Point) -> dict[tuple[int, int], _Point]:
    """The delivered pads that sit on the ``pitch`` lattice through ``anchor``.

    A cell is filled only by a pad within ``_PROTO_GRID_PITCH_TOL_MM`` of it, so
    a field at any other spacing never fills the square below.
    """
    cells: dict[tuple[int, int], _Point] = {}
    for point in points:
        column = round((point.x - anchor.x) / pitch)
        row = round((point.y - anchor.y) / pitch)
        if column < 0 or row < 0:
            continue
        if abs(point.x - anchor.x - column * pitch) > _PROTO_GRID_PITCH_TOL_MM:
            continue
        if abs(point.y - anchor.y - row * pitch) > _PROTO_GRID_PITCH_TOL_MM:
            continue
        cells.setdefault((column, row), point)
    return cells


def _pad_square_side(cells: dict[tuple[int, int], _Point]) -> int:
    """Largest fully-populated square grown from the lattice origin.

    Every cell of the square must hold a delivered pad, so a scatter of
    unrelated holes cannot accumulate into a field.
    """
    side = 0
    while side < _PROTO_GRID_MAX_SIDE and all(
        (column, row) in cells for column in range(side + 1) for row in range(side + 1)
    ):
        side += 1
    return side


def _delivered_pitch(cells: dict[tuple[int, int], _Point], side: int) -> float:
    """Mean cell spacing of a matched square, measured from its own pads."""
    if side < 2:
        return 0.0
    row = sorted(cells[(column, 0)].x for column in range(side))
    return sum(row[index + 1] - row[index] for index in range(side - 1)) / (side - 1)


def _proto_pad_grid(board: Any) -> dict[str, Any]:
    """The delivered pad field, measured from its own pads.

    The field *is* the deliverable, so it is proved by copper rather than by a
    label: a complete square of lone through-hole pads on ONE 0.1 inch grid, at
    the 0.1 inch pitch within ``_PROTO_GRID_PITCH_TOL_MM``. Anything else is not
    this feature -- a wider field fits no DIP part and no 0.1 inch header.
    """
    points = _bare_pth_pad_points(board)
    if len(points) < _PROTO_MIN_HOLES:
        return {"holes": 0, "pads": len(points), "usable": False}
    best: tuple[int, int, _Point, dict[tuple[int, int], _Point]] | None = None
    for anchor in points:
        cells = _pad_cells(points, _PROTO_GRID_PITCH_MM, anchor)
        side = _pad_square_side(cells)
        if best is None or side * side > best[0]:
            best = (side * side, side, anchor, cells)
    if best is None or best[0] < _PROTO_MIN_HOLES:
        return {"holes": 0, "pads": len(points), "usable": False}
    holes, side, anchor, cells = best
    pitch = _delivered_pitch(cells, side)
    return {
        "holes": holes,
        "pads": len(points),
        "columns": side,
        "rows": side,
        "pitch_mm": round(pitch, 3),
        "origin_mm": {"x": round(anchor.x, 3), "y": round(anchor.y, 3)},
        "usable": abs(pitch - _PROTO_GRID_PITCH_MM) <= _PROTO_GRID_PITCH_TOL_MM,
    }


def _prototyping_area(board: Any, box: tuple[float, float, float, float]) -> dict[str, Any]:
    """A usable prototyping area is free board space OR a delivered pad field.

    Free area and a real 2.54 mm pad bank are different evidence for the same
    feature: the field's own pads are obstacles to the free-area search, so a
    board that DELIVERS the field can have little free area left and must still
    pass. Both measurements are reported; the existing keys keep the free-area
    numbers they always carried.
    """
    free_area = _prototyping_area_free(board, box)
    pad_grid = _proto_pad_grid(board)
    return {
        **free_area,
        "pad_grid": pad_grid,
        "free_area": free_area,
        "usable": bool(free_area["usable"] or pad_grid["usable"]),
    }


def _is_pth(pad: Any) -> bool:
    return _pad_drill(pad) > 0


def _header_pitch(fp: Any) -> float | None:
    points = [point for pad in _iter(fp.Pads()) if _is_pth(pad) and (point := _pad_center(pad)) is not None]
    distances = sorted(hypot(a.x - b.x, a.y - b.y) for index, a in enumerate(points) for b in points[index + 1:])
    return distances[0] if distances else None


def _uno_geometry(board: Any, records: Mapping[str, Any], holes: list[dict[str, Any]], box: tuple[float, float, float, float]) -> dict[str, Any]:
    width, height = box[2] - box[0], box[3] - box[1]
    size_ok = abs(width - 68.58) <= 2.0 and abs(height - 53.34) <= 2.0
    hole_ok = len(holes) >= 4
    headers: list[dict[str, Any]] = []
    for fp in _iter(board.GetFootprints()):
        try:
            ref = str(fp.GetReferenceAsString())
        except Exception:
            continue
        if not _is_connector(records.get(ref)):
            continue
        pth_count = sum(_is_pth(pad) for pad in _iter(fp.Pads()))
        pitch = _header_pitch(fp)
        if pth_count >= 6 and pitch is not None and abs(pitch - 2.54) <= 0.08:
            headers.append({"ref": ref, "pins": pth_count, "pitch_mm": round(pitch, 3), "owned": True})
    counts = sorted(item["pins"] for item in headers)
    canonical_headers = all(counts.count(required) >= quantity for required, quantity in ((6, 1), (8, 2), (10, 1)))
    return {"dimensions_mm": {"width": round(width, 3), "height": round(height, 3)},
            "mounting_holes": len(holes), "headers": headers,
            "canonical": bool(size_ok and hole_ok and canonical_headers),
            "ownership_verified": bool(headers and all(item["owned"] for item in headers))}


def _header_grid(fp: Any) -> dict[str, Any]:
    """Prove a physical 2x10 2.54-mm through-hole header, orientation agnostic."""
    points = [point for pad in _iter(fp.Pads()) if _is_pth(pad) and (point := _pad_center(pad)) is not None]
    if len(points) != 20:
        return {"pth_pins": len(points), "pass": False}

    def clusters(values: list[float]) -> list[float]:
        out: list[float] = []
        for value in sorted(values):
            if not out or abs(value - out[-1]) > 0.08:
                out.append(value)
            else:
                out[-1] = (out[-1] + value) / 2
        return out

    xs, ys = clusters([point.x for point in points]), clusters([point.y for point in points])
    dimensions = (len(xs), len(ys))
    if sorted(dimensions) != [2, 10]:
        return {"pth_pins": len(points), "columns": len(xs), "rows": len(ys), "pass": False}
    long_axis = xs if len(xs) == 10 else ys
    short_axis = ys if len(ys) == 2 else xs
    pitch_ok = (
        all(abs(b - a - _PROTO_GRID_PITCH_MM) <= 0.08 for a, b in zip(long_axis, long_axis[1:]))
        and abs(short_axis[1] - short_axis[0] - _PROTO_GRID_PITCH_MM) <= 0.08
    )
    return {
        "pth_pins": len(points), "columns": len(xs), "rows": len(ys),
        "pitch_mm": _PROTO_GRID_PITCH_MM, "pass": pitch_ok,
    }


def _thermal_geometry(board: Any, records: Mapping[str, Any], *, require_vias: bool) -> dict[str, Any]:
    """Measure only reviewed exposed-pad copper and same-net thermal support.

    A packet name proves nothing: the exposed/thermal pad identity comes from
    the reviewed device record, and the support copper must be on the delivered
    pad's own net -- a GND via beside a drain-connected pad is not dissipation.
    """
    vias: list[tuple[_Point, tuple[str, int | str]]] = []
    tracks: list[tuple[tuple[float, float, float, float], tuple[str, int | str]]] = []
    for item in _iter(board.GetTracks()):
        net = _net_identity(item)
        if net is None:
            continue
        try:
            is_via = item.GetClass() == "PCB_VIA"
        except Exception:
            is_via = False
        if is_via:
            point = _point(item.GetPosition())
            if point is not None:
                vias.append((point, net))
        elif (item_box := _box(item)) is not None:
            tracks.append((item_box, net))

    reviewed_pad_count = 0
    delivered_pad_count = 0
    evidence: list[dict[str, Any]] = []
    for fp in _iter(board.GetFootprints()):
        ref = _footprint_ref(fp)
        if ref is None:
            continue
        thermal_pads = _thermal_pad_numbers(records.get(ref))
        if not thermal_pads:
            continue
        reviewed_pad_count += len(thermal_pads)
        for pad in _iter(fp.Pads()):
            if str(pad.GetNumber()) not in thermal_pads:
                continue
            pad_box, net = _box(pad), _net_identity(pad)
            if pad_box is None or net is None:
                continue
            delivered_pad_count += 1
            expanded = (
                pad_box[0] - _THERMAL_VIA_PAD_MARGIN_MM, pad_box[1] - _THERMAL_VIA_PAD_MARGIN_MM,
                pad_box[2] + _THERMAL_VIA_PAD_MARGIN_MM, pad_box[3] + _THERMAL_VIA_PAD_MARGIN_MM,
            )
            matching_vias = [point for point, via_net in vias
                             if via_net == net and expanded[0] <= point.x <= expanded[2]
                             and expanded[1] <= point.y <= expanded[3]]
            matching_tracks = [track_box for track_box, track_net in tracks
                               if track_net == net and _rect_hits(expanded, track_box)]
            pad_layers = _item_copper_layers(pad)
            pad_uuid = _item_uuid(pad)
            if pad_layers is None or pad_uuid is None:
                return {"status": "unverified", "reason": "delivered_thermal_copper_unmeasurable", "ref": ref}
            pours: list[str] = []
            for other in _iter(board.Zones()):
                if bool(getattr(other, "GetIsRuleArea", lambda: False)()) or _net_identity(other) != net:
                    continue
                other_layers = _item_copper_layers(other)
                if other_layers is None:
                    return {"status": "unverified", "reason": "delivered_thermal_copper_unmeasurable", "ref": ref}
                if not other_layers & pad_layers:
                    continue
                # A zone elsewhere on the same net is not dissipation at this pad,
                # so the pad-to-zone link comes from KiCad's own connectivity.
                joined = _zone_connected_items(board, other)
                if joined is None:
                    return {"status": "unverified", "reason": "delivered_thermal_copper_unmeasurable", "ref": ref}
                if pad_uuid in joined:
                    pours.append(_net_label(other))
            support = [name for name, present in (
                ("same_net_via", matching_vias), ("same_net_track", matching_tracks), ("same_net_pour", pours),
            ) if present]
            evidence.append({
                "ref": ref, "pad": str(pad.GetNumber()), "pad_area_mm2": round(_pad_area(pad), 3),
                "net": _net_label(pad), "supported": bool(support), "support": support,
                "same_net_via_count": len(matching_vias), "same_net_track_count": len(matching_tracks),
                "same_net_pours": pours,
            })
    if not reviewed_pad_count:
        return {"status": "unverified", "reason": "reviewed_thermal_pad_identity_missing"}
    if not delivered_pad_count:
        return {"status": "fail", "reason": "reviewed_thermal_pad_absent_from_board",
                "reviewed_thermal_pad_count": reviewed_pad_count}
    unsupported = [row["ref"] for row in evidence if not row["supported"]]
    if unsupported:
        return {"status": "fail", "reason": "no_same_net_thermal_copper", "refs": unsupported,
                "power_device_thermal_pads": evidence}
    without_via = [row["ref"] for row in evidence if not row["same_net_via_count"]]
    if require_vias and without_via:
        return {"status": "fail", "reason": "no_same_net_thermal_via", "refs": without_via,
                "power_device_thermal_pads": evidence}
    return {
        "status": "pass", "requires_same_net_via": require_vias,
        "reviewed_thermal_pad_count": reviewed_pad_count,
        "power_device_thermal_pads": evidence,
        "thermal_via_count": sum(row["same_net_via_count"] for row in evidence),
    }


def _rf_antenna_geometry(board: Any, records: Mapping[str, Any], box: tuple[float, float, float, float]) -> dict[str, Any]:
    """Measure a reviewed chip antenna's source-backed clearance on the board.

    A reviewed chip antenna only proves geometry when the record carries the
    manufacturer's own clearance outline and the delivered rule area reproduces
    it while keeping real copper out.  A named area with prohibition flags and
    an edge position alone is intent, so it stays unverified.
    """
    declared: list[tuple[str, Any, Mapping[str, Any], list[_Point]]] = []
    for fp in _iter(board.GetFootprints()):
        ref = _footprint_ref(fp)
        if ref is None:
            continue
        record = records.get(ref)
        if "chip-antenna" not in _features(record):
            continue
        metadata = (getattr(record, "support_network", {}) or {}).get("rf_antenna_geometry")
        if not isinstance(metadata, Mapping):
            return {"status": "unverified", "reason": "reviewed_rf_antenna_geometry_missing", "ref": ref}
        shape = _declared_shape(metadata.get("local_shape"))
        if shape is None:
            return {"status": "unverified", "reason": "reviewed_rf_antenna_shape_missing", "ref": ref}
        declared.append((ref, fp, metadata, shape))
    if not declared:
        return {"status": "unverified", "reason": "reviewed_chip_antenna_missing"}

    evidence: list[dict[str, Any]] = []
    for ref, fp, metadata, shape in declared:
        zone_name = str(metadata.get("rule_area_name") or "")
        source = str(metadata.get("source") or "")
        prohibited = frozenset(str(item) for item in metadata.get("prohibited") or ())
        if not zone_name or not source or not _RF_REQUIRED_PROHIBITIONS <= prohibited:
            return {"status": "unverified", "reason": "reviewed_rf_antenna_constraint_incomplete", "ref": ref}
        zones = [
            zone for zone in _iter(fp.Zones())
            if _zone_name(zone) == zone_name
            and bool(getattr(zone, "GetIsRuleArea", lambda: False)())
        ]
        if not zones:
            return {"status": "fail", "reason": "delivered_antenna_clearance_missing",
                    "ref": ref, "rule_area": zone_name}
        zone = zones[0]
        if _rule_area_blocks(zone, prohibited) is not True:
            return {"status": "fail", "reason": "delivered_antenna_clearance_permits_copper", "ref": ref}
        rings = _zone_rings(zone)
        if not rings:
            return {"status": "unverified", "reason": "delivered_antenna_clearance_unmeasurable", "ref": ref}
        delivered = [local for ring in rings for point in ring if (local := _world_to_local(fp, point)) is not None]
        if len(delivered) != sum(len(ring) for ring in rings):
            return {"status": "unverified", "reason": "delivered_antenna_clearance_unmeasurable", "ref": ref}
        if not _shape_match(delivered, shape):
            return {"status": "fail", "reason": "delivered_antenna_clearance_mismatch", "ref": ref,
                    "declared_shape": _shape_record(shape), "delivered_shape": _shape_record(delivered)}
        intrusions = _outline_intrusions(board, zone)
        if intrusions is None:
            return {"status": "unverified", "reason": "delivered_antenna_clearance_unmeasurable", "ref": ref}
        if intrusions:
            return {"status": "fail", "reason": "delivered_copper_inside_antenna_clearance", "ref": ref,
                    "intrusions": intrusions}
        edge_gap = None
        if metadata.get("edge_required") is True:
            fp_box = _box(fp)
            edge_gap = None if fp_box is None else min(
                fp_box[0] - box[0], box[2] - fp_box[2], fp_box[1] - box[1], box[3] - fp_box[3],
            )
            if edge_gap is None or edge_gap > _EDGE_TOUCH_TOL_MM:
                return {"status": "fail", "reason": "delivered_antenna_not_at_board_edge", "ref": ref,
                        "edge_gap_mm": None if edge_gap is None else round(edge_gap, 3)}
        evidence.append({
            "ref": ref, "rule_area": zone_name, "source": source,
            "clearance_area_mm2": round(_polygon_area(shape), 3),
            "edge_gap_mm": None if edge_gap is None else round(edge_gap, 3),
        })
    return {"status": "pass", "antennas": evidence}


def _touch_electrode_geometry(board: Any, records: Mapping[str, Any]) -> dict[str, Any]:
    """Measure a reviewed board-fabricated electrode and its underlay keepout.

    The proof is delivered copper: a round front-side electrode of the reviewed
    diameter under solder mask, plus a back-side rule area that encloses the
    reviewed clearance and actually contains no copper.
    """
    found = False
    evidence: list[dict[str, Any]] = []
    for fp in _iter(board.GetFootprints()):
        ref = _footprint_ref(fp)
        if ref is None:
            continue
        record = records.get(ref)
        if "capacitive-touch-pad" not in _features(record):
            continue
        found = True
        metadata = (getattr(record, "support_network", {}) or {}).get("touch_electrode")
        if not isinstance(metadata, Mapping):
            return {"status": "unverified", "reason": "reviewed_touch_electrode_geometry_missing", "ref": ref}
        zone_name = str(metadata.get("rule_area_name") or "")
        source = str(metadata.get("source") or "")
        prohibited = frozenset(str(item) for item in metadata.get("prohibited") or ())
        try:
            diameter = float(metadata["electrode_diameter_mm"])
            clearance = float(metadata["underlay_clearance_mm"])
        except (KeyError, TypeError, ValueError):
            return {"status": "unverified", "reason": "reviewed_touch_electrode_constraint_incomplete", "ref": ref}
        if not zone_name or not source or not _RF_REQUIRED_PROHIBITIONS <= prohibited:
            return {"status": "unverified", "reason": "reviewed_touch_electrode_constraint_incomplete", "ref": ref}
        pads = [pad for pad in _iter(fp.Pads()) if str(pad.GetNumber()) == "1"]
        if len(pads) != 1:
            return {"status": "fail", "reason": "delivered_electrode_pad_missing", "ref": ref}
        pad, pad_box = pads[0], _box(pads[0])
        front, back, mask, circle = (
            _on_layer(pad, "F.Cu"), _on_layer(pad, "B.Cu"), _on_layer(pad, "F.Mask"), _is_circle_pad(pad),
        )
        if pad_box is None or None in {front, back, mask, circle}:
            return {"status": "unverified", "reason": "delivered_electrode_unmeasurable", "ref": ref}
        width, height = pad_box[2] - pad_box[0], pad_box[3] - pad_box[1]
        if not (front is True and back is False and circle is True
                and abs(width - diameter) <= _ELECTRODE_SIZE_TOL_MM
                and abs(height - diameter) <= _ELECTRODE_SIZE_TOL_MM):
            return {"status": "fail", "reason": "delivered_electrode_geometry_mismatch", "ref": ref,
                    "electrode_mm": {"width": round(width, 3), "height": round(height, 3)},
                    "reviewed_diameter_mm": diameter}
        if mask is True:
            return {"status": "fail", "reason": "delivered_electrode_bare_copper", "ref": ref}
        zones = [
            zone for zone in _iter(fp.Zones())
            if _zone_name(zone) == zone_name
            and bool(getattr(zone, "GetIsRuleArea", lambda: False)())
        ]
        if not zones:
            return {"status": "fail", "reason": "delivered_electrode_underlay_clearance_missing", "ref": ref}
        zones = [zone for zone in zones if _item_copper_layers(zone)]
        if not zones:
            return {"status": "unverified", "reason": "delivered_electrode_underlay_unmeasurable", "ref": ref}
        zone = zones[0]
        if _rule_area_blocks(zone, prohibited) is not True:
            return {"status": "fail", "reason": "delivered_electrode_underlay_permits_copper", "ref": ref}
        rings = _zone_rings(zone)
        if not rings:
            return {"status": "unverified", "reason": "delivered_electrode_underlay_unmeasurable", "ref": ref}
        if not _region_covers(rings, pad_box, clearance - _ELECTRODE_SIZE_TOL_MM):
            return {"status": "fail", "reason": "delivered_electrode_underlay_too_small", "ref": ref,
                    "reviewed_clearance_mm": clearance}
        intrusions = _outline_intrusions(board, zone)
        if intrusions is None:
            return {"status": "unverified", "reason": "delivered_electrode_underlay_unmeasurable", "ref": ref}
        if intrusions:
            return {"status": "fail", "reason": "delivered_copper_under_electrode", "ref": ref,
                    "intrusions": intrusions}
        evidence.append({"ref": ref, "diameter_mm": diameter, "covered_by_mask": True,
                         "underlay_rule_area": zone_name, "clearance_mm": clearance, "source": source})
    if not found:
        return {"status": "unverified", "reason": "reviewed_touch_electrode_missing"}
    return {"status": "pass", "electrodes": evidence}

def _qualified_leds(board: Any, records: Mapping[str, Any], kind: str) -> list[tuple[str, _Point]]:
    out = []
    for fp in _iter(board.GetFootprints()):
        try:
            ref = str(fp.GetReferenceAsString())
            point = _point(fp.GetPosition())
        except Exception:
            continue
        if point is not None and _is_led(records.get(ref), kind):
            out.append((ref, point))
    return out


def _ring_leds(leds: list[tuple[str, _Point]]) -> dict[str, Any]:
    if len(leds) < 12:
        return {"count": len(leds), "pass": False}
    selected = leds[:12]
    cx = sum(point.x for _, point in selected) / len(selected)
    cy = sum(point.y for _, point in selected) / len(selected)
    radii = [hypot(point.x - cx, point.y - cy) for _, point in selected]
    mean = sum(radii) / len(radii)
    angles = sorted((degrees(atan2(point.y - cy, point.x - cx)) + 360.0) % 360.0 for _, point in selected)
    gaps = [angles[index + 1] - angles[index] for index in range(len(angles) - 1)] + [angles[0] + 360.0 - angles[-1]]
    ideal = 360.0 / len(selected)
    radial = max(abs(radius - mean) for radius in radii)
    angular = max(abs(gap - ideal) for gap in gaps)
    return {"count": len(leds), "center_mm": {"x": round(cx, 3), "y": round(cy, 3)}, "radius_mm": round(mean, 3),
            "max_radial_deviation_mm": round(radial, 3), "max_angular_deviation_deg": round(angular, 3),
            "tolerance": {"radial_mm": _RING_RADIAL_TOL_MM, "angular_deg": _RING_ANGLE_TOL_DEG},
            "pass": radial <= _RING_RADIAL_TOL_MM and angular <= _RING_ANGLE_TOL_DEG}


def _star_leds(vertices: list[_Point], leds: list[tuple[str, _Point]], box: tuple[float, float, float, float]) -> dict[str, Any]:
    if len(vertices) < 10 or len(leds) < 5:
        return {"count": len(leds), "pass": False}
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    tips = sorted(vertices, key=lambda point: hypot(point.x - cx, point.y - cy), reverse=True)[:5]
    tolerance = min(5.0, 0.15 * max(box[2] - box[0], box[3] - box[1]))
    owners: dict[int, str] = {}
    for ref, led in leds:
        nearest = min(range(5), key=lambda index: hypot(led.x - tips[index].x, led.y - tips[index].y))
        distance = hypot(led.x - tips[nearest].x, led.y - tips[nearest].y)
        if distance <= tolerance and nearest not in owners:
            owners[nearest] = ref
    return {"count": len(leds), "tip_leds": owners, "tip_tolerance_mm": round(tolerance, 3), "pass": len(owners) == 5}


def _snowman_leds(vertices: list[_Point], leds: list[tuple[str, _Point]], box: tuple[float, float, float, float]) -> dict[str, Any]:
    # The compound outline is assessed from delivered contour widths.  Three
    # LEDs merely arranged vertically on a rectangle never create these sections.
    if len(vertices) < 12 or len(leds) < 3:
        return {"count": len(leds), "sections": [], "pass": False}
    try:
        from shapely.geometry import LineString, Polygon
        polygon = Polygon([(point.x, point.y) for point in vertices])
        if not polygon.is_valid or polygon.area <= 0:
            return {"count": len(leds), "sections": [], "pass": False}
        top, bottom = box[1], box[3]
        samples = []
        for index in range(1, 60):
            y = top + (bottom - top) * index / 60.0
            width = polygon.intersection(LineString([(box[0] - 1.0, y), (box[2] + 1.0, y)])).length
            samples.append((y, width))
        # Three broad local maxima in order head/body/base, with real necks.
        maxima = [row for index, row in enumerate(samples[1:-1], 1) if row[1] >= samples[index - 1][1] and row[1] >= samples[index + 1][1]]
        maxima = sorted(maxima, key=lambda row: row[1], reverse=True)[:3]
        maxima.sort()
        if len(maxima) != 3 or not (maxima[0][1] < maxima[1][1] < maxima[2][1]):
            return {"count": len(leds), "sections": [], "pass": False}
        cuts = ((maxima[0][0] + maxima[1][0]) / 2.0, (maxima[1][0] + maxima[2][0]) / 2.0)
        section_refs = {"head": [], "body": [], "base": []}
        for ref, led in leds:
            section = "head" if led.y < cuts[0] else "body" if led.y < cuts[1] else "base"
            section_refs[section].append(ref)
        passed = all(section_refs[name] for name in ("head", "body", "base"))
        return {"count": len(leds), "section_centers_y_mm": [round(row[0], 3) for row in maxima],
                "sections": [name for name in ("base", "body", "head") if section_refs[name]],
                "led_refs": section_refs, "pass": passed}
    except Exception:
        return {"count": len(leds), "sections": [], "pass": False}


def extract_geometry_facts(rundir: Path, state: dict, board: object, contract: dict) -> dict:
    """Return only fact deltas proved by the supplied, already-loaded board."""
    del rundir
    if board is None or not isinstance(state, Mapping) or not isinstance(contract, Mapping):
        return {}
    slug = str(contract.get("slug") or "")
    records = _records_by_fp(state, board)
    outline, vertices = _outline_facts(board)
    box = _board_box(board)
    if box is None:
        return {"geometry_diagnostics": {"reason": "no_edge_cuts_bounding_box"}}
    holes = _mounting_holes(board, records)
    connector_edges = _connector_edge_facts(board, records, box)
    has_closed_outline = len(vertices) >= 3 or bool(_edge_circles(board))
    facts: dict[str, Any] = {
        "geometry_counts": {"mounting_hole": len(holes)},
        "gates": {"geometry": "pass" if has_closed_outline else "fail"},
        "geometry_diagnostics": {"outline": outline, "mounting_holes": holes,
                                 "connector_edges": connector_edges},
    }
    if outline.get("shape"):
        facts["outline"] = {key: value for key, value in outline.items() if key in {"shape", "diameter_mm"}}
    if holes:
        top_limit = box[1] + (box[3] - box[1]) * _HANGER_TOP_FRACTION
        top_holes = [hole for hole in holes if hole["y"] <= top_limit]
        if top_holes:
            facts["geometry_counts"]["hang_hole"] = len(top_holes)
    if slug == "rp2040-min":
        # A castellated pad's copper/drill centre must land on Edge.Cuts, not just
        # be adjacent to an arbitrary board side.  The conservative 0.25 mm band
        # tolerates manufacturing datum error while rejecting interior GPIO rows.
        candidates = []
        for fp in _iter(board.GetFootprints()):
            try:
                ref = str(fp.GetReferenceAsString())
            except Exception:
                continue
            record = records.get(ref)
            if not ({"castellated-gpio", "gpio-castellations"} & _features(record)):
                continue
            for pad in _iter(fp.Pads()):
                point = _pad_center(pad)
                if point is not None and _outline_distance(point, vertices, box) <= 0.25:
                    candidates.append(f"{ref}.{pad.GetNumber()}")
        facts["geometry_diagnostics"]["castellations"] = {"pads": candidates, "edge_band_mm": 0.25}
        facts["gates"]["castellations"] = "pass" if candidates else "fail"
    if slug in {"buck-3a", "highside-switch-10a", "led-cc-driver"}:
        # Only the buck contract asks for thermal-via copper; the other two ask
        # for the declared thermal feature, so a same-net track or pour suffices.
        thermal = _thermal_geometry(board, records, require_vias=slug == "buck-3a")
        facts["geometry_diagnostics"]["thermal"] = thermal
        if thermal["status"] != "unverified":
            facts.setdefault("gates", {})["thermal_geometry"] = thermal["status"]
    if slug == "proto-shield":
        uno = _uno_geometry(board, records, holes, box)
        proto = _prototyping_area(board, box)
        facts["geometry_diagnostics"].update({"uno": uno, "prototyping_area": proto})
        facts.setdefault("gates", {}).update({
            "uno_shield_geometry": "pass" if uno["canonical"] and uno["ownership_verified"] else "fail",
            "prototyping_area": "pass" if proto["usable"] else "fail",
        })
    if slug in {"servo-driver-16", "audio-jack-buffer"}:
        feature = "servo_headers_edge" if slug == "servo-driver-16" else "audio_jacks_edge"
        qualified: list[dict[str, Any]] = []
        for fp in _iter(board.GetFootprints()):
            try:
                ref = str(fp.GetReferenceAsString())
            except Exception:
                continue
            edge = connector_edges.get(ref)
            record = records.get(ref)
            if edge is None or not edge["at_edge"]:
                continue
            if slug == "servo-driver-16":
                if str(getattr(record, "family", "")).casefold() == "pin-header" and sum(_is_pth(pad) for pad in _iter(fp.Pads())) == 3:
                    qualified.append(edge)
            elif "audio-jack-3-5mm" in _features(record):
                qualified.append(edge)
        minimum = 16 if slug == "servo-driver-16" else 4
        facts["gates"][feature] = "pass" if len(qualified) >= minimum else "fail"
    if slug == "nrf52-beacon":
        rf = _rf_antenna_geometry(board, records, box)
        facts["geometry_diagnostics"]["rf_antenna"] = rf
        # A missing reviewed constraint stays unverified: emitting a gate value
        # here would turn absent evidence into a fabricated verdict.
        if rf["status"] != "unverified":
            facts["gates"]["rf_antenna_geometry"] = rf["status"]
    if slug == "chamfered-badge":
        touch = _touch_electrode_geometry(board, records)
        facts["geometry_diagnostics"]["touch_electrodes"] = touch
        if touch["status"] != "unverified":
            facts["gates"]["touch_no_underlay"] = touch["status"]
    if slug == "rounded-c3-devboard":
        usb = [value for ref, value in connector_edges.items()
               if "usb-c-receptacle" in _features(records.get(ref)) and value["at_edge"]]
        gpio: list[dict[str, Any]] = []
        for ref, value in connector_edges.items():
            record = records.get(ref)
            if "header" not in _features(record) or not value["at_edge"]:
                continue
            fp = next((item for item in _iter(board.GetFootprints())
                       if _footprint_ref(item) == ref), None)
            grid = _header_grid(fp) if fp is not None else {"pass": False, "pth_pins": 0}
            if grid["pass"]:
                gpio.append({**value, "ref": ref, "pins": grid["pth_pins"]})
        opposite = {"left": "right", "right": "left", "top": "bottom", "bottom": "top"}
        usb_edge = bool(usb)
        gpio_opposite = bool(usb and any(item["edge"] == opposite[usb[0]["edge"]] for item in gpio))
        facts["geometry_diagnostics"]["gpio_header_grid"] = gpio
        facts["net_paths"] = {"usb_c_edge": usb_edge, "gpio_header_opposite_edge": gpio_opposite}
        facts["gates"]["connector_edges"] = "pass" if usb_edge and gpio_opposite else "fail"
    if slug == "round-led-ring":
        ring = _ring_leds(_qualified_leds(board, records, "ws2812b"))
        facts["geometry_diagnostics"]["ws2812b_ring"] = ring
        facts["gates"].update({
            "ws2812b_even_circle": "pass" if ring["pass"] else "fail",
            "led_placement": "pass" if ring["pass"] else "fail",
        })
    if slug == "star-ornament":
        star = _star_leds(vertices, _qualified_leds(board, records, "warm-white-led"), box)
        facts["geometry_diagnostics"]["star_leds"] = star
        facts["gates"].update({
            "star_point_led_placement": "pass" if star["pass"] else "fail",
            "led_placement": "pass" if star["pass"] else "fail",
        })
    if slug == "snowman-ornament":
        snowman = _snowman_leds(vertices, _qualified_leds(board, records, "warm-white-led"), box)
        facts["geometry_diagnostics"]["snowman_leds"] = snowman
        if snowman["pass"]:
            facts["snowman_led_sections"] = snowman["sections"]
    return facts
