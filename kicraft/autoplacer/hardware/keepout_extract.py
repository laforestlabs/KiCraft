"""Extract antenna / RF keep-clear rects from a loaded pcbnew board.

Two sources, both owner-tagged so the placer can exempt the part the keep-out
belongs to:

* **preserve** -- footprint-internal rule-area zones that keep footprints/pads
  out (stock KiCad RF footprints, and KiCraft library footprints carrying an
  on-module antenna strip after Fix 0). pcbnew reports a placed footprint
  zone's outline already in board coordinates, so it is taken as-is.

* **inject** -- a config-driven per-footprint-family near-field rect
  (``cfg["antenna_keepouts"]``, keyed by footprint-name glob). The vendored
  easyeda imports dropped the stock keep-clear, and Fix 0 bakes only a modest
  on-module strip, so the larger near-field clearance is injected here. The
  spec is a rect in the footprint's LOCAL frame; it is transformed to board
  coordinates by the footprint's placed position and orientation.

Both sources are emitted (unioned by the solver's per-rect push), so a footprint
that has both an internal strip and a family-spec match is protected by both.
"""
from __future__ import annotations

from fnmatch import fnmatch
from dataclasses import dataclass
from typing import Any

import pcbnew

from ..brain import geometry
from ..brain.types import AntennaEdgeIntent, Component, KeepoutRect, Point

@dataclass(frozen=True)
class TrackViaRuleArea:
    """A placed KiCad rule area with item-specific copper prohibitions."""

    zone: Any
    blocks_tracks: bool
    blocks_vias: bool


@dataclass(frozen=True)
class AntennaExtraction:
    """Bounded semantic antenna extraction result."""

    intents: tuple[AntennaEdgeIntent, ...]
    diagnostics: tuple[str, ...]


_SIDES = frozenset({"left", "right", "top", "bottom"})


def _zone_name(zone: Any) -> str:
    try:
        return str(zone.GetZoneName() or "")
    except Exception:
        return ""


def _polygon_points_mm(poly: Any) -> list[Point]:
    points: list[Point] = []
    for outline_index in range(poly.OutlineCount()):
        outline = poly.COutline(outline_index)
        for point_index in range(outline.PointCount()):
            point = outline.CPoint(point_index)
            points.append(Point(pcbnew.ToMM(point.x), pcbnew.ToMM(point.y)))
    return points


def _world_to_footprint_local(
    point: Point, origin: Point, rotation_deg: float, flipped: bool = False
) -> Point:
    local = geometry.rotate_vector(point - origin, -rotation_deg)
    return Point(-local.x, local.y) if flipped else local


def _direction_and_anchor(
    polygon: list[Point],
    envelope_center: Point,
    *,
    min_offset_mm: float,
    dominance_ratio: float,
) -> tuple[str, float, Point] | None:
    if not polygon:
        return None
    centroid = Point(
        sum(point.x for point in polygon) / len(polygon),
        sum(point.y for point in polygon) / len(polygon),
    )
    dx = centroid.x - envelope_center.x
    dy = centroid.y - envelope_center.y
    ax, ay = abs(dx), abs(dy)
    dominant = max(ax, ay)
    secondary = min(ax, ay)
    if dominant < min_offset_mm or dominant < secondary * dominance_ratio:
        return None
    if ax > ay:
        direction = "right" if dx > 0 else "left"
        anchor = max(point.x for point in polygon) if dx > 0 else min(point.x for point in polygon)
        support = [point for point in polygon if abs(point.x - anchor) <= 1e-6]
        midpoint = Point(anchor, sum(point.y for point in support) / len(support))
    else:
        direction = "bottom" if dy > 0 else "top"
        anchor = max(point.y for point in polygon) if dy > 0 else min(point.y for point in polygon)
        support = [point for point in polygon if abs(point.y - anchor) <= 1e-6]
        midpoint = Point(sum(point.x for point in support) / len(support), anchor)
    return direction, anchor, midpoint


def _component_local_center(
    fp: Any,
    component: Component | None,
    origin: Point,
    rotation_deg: float,
) -> Point:
    flipped = fp.GetLayer() == pcbnew.B_Cu
    if component is not None and component.body_center is not None:
        return _world_to_footprint_local(
            component.body_center, origin, rotation_deg, flipped
        )
    pads = list(fp.Pads())
    if pads:
        local = [
            _world_to_footprint_local(
                Point(pcbnew.ToMM(pad.GetPosition().x), pcbnew.ToMM(pad.GetPosition().y)),
                origin,
                rotation_deg,
                flipped,
            )
            for pad in pads
        ]
        return Point(
            (min(point.x for point in local) + max(point.x for point in local)) / 2.0,
            (min(point.y for point in local) + max(point.y for point in local)) / 2.0,
        )
    return Point(0.0, 0.0)


def extract_antenna_edge_intents(
    board: Any,
    cfg: dict,
    components: dict[str, Component] | None = None,
) -> AntennaExtraction:
    """Extract semantic, footprint-local antenna edge contracts.

    Named footprint rule areas win over family rectangles. Generic unnamed
    rule areas are deliberately ignored. Ambiguous geometry is diagnosed once
    per owner and never guessed.
    """
    patterns = tuple(
        str(pattern).lower()
        for pattern in (cfg.get("antenna_rule_area_name_patterns") or ())
        if str(pattern)
    )
    families: dict[str, dict] = cfg.get("antenna_keepouts", {}) or {}
    antenna_components: dict[str, dict] = cfg.get("antenna_components", {}) or {}
    component_zones: dict[str, dict] = cfg.get("component_zones", {}) or {}
    enabled = bool(cfg.get("antenna_edge_pin_enabled", True))
    default_edge = str(cfg.get("antenna_default_edge", "top"))
    inset = float(cfg.get("antenna_edge_inset_mm", 0.0))
    min_offset = float(cfg.get("antenna_direction_min_offset_mm", 0.5))
    dominance = max(1.0, float(cfg.get("antenna_direction_dominance_ratio", 1.25)))
    intents: list[AntennaEdgeIntent] = []
    diagnostics: list[str] = []

    for fp in board.Footprints():
        ref = fp.GetReferenceAsString()
        explicit = antenna_components.get(ref)
        explicit = explicit if isinstance(explicit, dict) else {}
        zone_cfg = component_zones.get(ref)
        zone_cfg = zone_cfg if isinstance(zone_cfg, dict) else {}
        explicit_component_edge = zone_cfg.get("edge") in _SIDES
        explicit_antenna_edge = explicit.get("edge") in _SIDES
        if not enabled and not explicit and not explicit_component_edge:
            continue
        if not explicit_component_edge and not explicit_antenna_edge and (
            "corner" in zone_cfg or "zone" in zone_cfg
        ):
            continue
        component = (components or {}).get(ref)
        if (
            component is not None
            and component.locked
            and not explicit
            and not explicit_component_edge
        ):
            continue

        pos = fp.GetPosition()
        origin = Point(pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y))
        rotation = float(fp.GetOrientationDegrees())
        flipped = fp.GetLayer() == pcbnew.B_Cu
        center = _component_local_center(
            fp, (components or {}).get(ref), origin, rotation
        )

        named_points: list[Point] = []
        named_ids: list[str] = []
        for zone in fp.Zones():
            name = _zone_name(zone)
            if not zone.GetIsRuleArea() or not name:
                continue
            if not any(fnmatch(name.lower(), pattern) for pattern in patterns):
                continue
            world_points = _polygon_points_mm(zone.Outline())
            named_points.extend(
                _world_to_footprint_local(point, origin, rotation, flipped)
                for point in world_points
            )
            named_ids.append(name)

        family_points: list[Point] = []
        family_id = ""
        candidates = _footprint_name_candidates(fp)
        for pattern, spec in families.items():
            if _matches_family(candidates, pattern):
                family_id = str(pattern)
                family_points = [
                    Point(float(spec["x_min"]), float(spec["y_min"])),
                    Point(float(spec["x_max"]), float(spec["y_min"])),
                    Point(float(spec["x_max"]), float(spec["y_max"])),
                    Point(float(spec["x_min"]), float(spec["y_max"])),
                ]
                break

        source = "explicit" if explicit else (
            "footprint_rule_area" if named_points else "family_config"
        )
        source_id = ref if explicit else (
            ",".join(sorted(set(named_ids))) if named_points else family_id
        )
        polygon = named_points or family_points
        direction = explicit.get("local_direction")
        anchor_value = explicit.get("anchor_mm")
        inferred = _direction_and_anchor(
            polygon,
            center,
            min_offset_mm=min_offset,
            dominance_ratio=dominance,
        )
        if direction not in _SIDES:
            if inferred is None:
                if polygon or explicit:
                    diagnostics.append(f"antenna_direction_ambiguous:{ref}")
                continue
            direction, anchor, midpoint = inferred
        else:
            if anchor_value is not None:
                anchor = float(anchor_value)
                if direction in ("left", "right"):
                    midpoint = Point(anchor, float(explicit.get("anchor_midpoint_mm", 0.0)))
                else:
                    midpoint = Point(float(explicit.get("anchor_midpoint_mm", 0.0)), anchor)
            elif inferred is not None:
                _, anchor, midpoint = inferred
            else:
                diagnostics.append(f"antenna_anchor_missing:{ref}")
                continue

        target_edge = (
            str(zone_cfg["edge"])
            if explicit_component_edge
            else str(explicit["edge"])
            if explicit_antenna_edge
            else default_edge
        )
        if target_edge not in _SIDES:
            diagnostics.append(f"antenna_target_edge_invalid:{ref}")
            continue
        intents.append(
            AntennaEdgeIntent(
                owner_ref=ref,
                source=source,
                source_id=source_id,
                local_direction=str(direction),
                local_anchor_mm=float(anchor),
                local_anchor_midpoint=midpoint,
                local_polygon=tuple(polygon),
                target_edge=target_edge,
                inset_mm=float(explicit.get("inset_mm", inset)),
                explicit_edge=explicit_component_edge or explicit_antenna_edge,
                explicit_rotation=("rotation" in zone_cfg or "rotation" in explicit),
            )
        )

    return AntennaExtraction(tuple(intents), tuple(diagnostics))

def collect_track_via_rule_areas(board: Any) -> list[TrackViaRuleArea]:
    """Collect board- and footprint-local rule areas in board coordinates."""
    areas: list[TrackViaRuleArea] = []
    zones = list(board.Zones())
    for fp in board.GetFootprints():
        zones.extend(fp.Zones())
    for zone in zones:
        if not zone.GetIsRuleArea():
            continue
        blocks_tracks = bool(zone.GetDoNotAllowTracks())
        blocks_vias = bool(zone.GetDoNotAllowVias())
        if blocks_tracks or blocks_vias:
            areas.append(
                TrackViaRuleArea(
                    zone=zone,
                    blocks_tracks=blocks_tracks,
                    blocks_vias=blocks_vias,
                )
            )
    return areas


def _vector_mm(point: tuple[float, float]) -> pcbnew.VECTOR2I:
    return pcbnew.VECTOR2I(pcbnew.FromMM(point[0]), pcbnew.FromMM(point[1]))


def track_intersects_rule_area(
    a: tuple[float, float],
    b: tuple[float, float],
    half_width_mm: float,
    area: TrackViaRuleArea,
) -> bool:
    """Whether a physical track intersects a track-blocking rule area."""
    if not area.blocks_tracks:
        return False
    segment = pcbnew.SEG(_vector_mm(a), _vector_mm(b))
    clearance = max(0, int(pcbnew.FromMM(half_width_mm)))
    return bool(area.zone.Outline().Collide(segment, clearance))


def via_intersects_rule_area(
    center: tuple[float, float],
    radius_mm: float,
    area: TrackViaRuleArea,
) -> bool:
    """Whether a via barrel intersects a via-blocking rule area."""
    if not area.blocks_vias:
        return False
    clearance = max(0, int(pcbnew.FromMM(radius_mm)))
    return bool(area.zone.Outline().Collide(_vector_mm(center), clearance))


def _footprint_name_candidates(fp) -> list[str]:
    """Strings to match a footprint against an antenna_keepouts glob."""
    names: list[str] = []
    try:
        fpid = fp.GetFPID()
        item = fpid.GetLibItemName()
        names.append(str(item))
        names.append(fp.GetFPIDAsString())
    except Exception:
        pass
    try:
        names.append(fp.GetValue())
    except Exception:
        pass
    return [n for n in names if n]


def _matches_family(name_candidates: list[str], pattern: str) -> bool:
    pat = pattern.lower()
    return any(fnmatch(n.lower(), pat) for n in name_candidates)


def _transform_local_rect(
    spec: dict, origin_x: float, origin_y: float, rotation_deg: float
) -> tuple[Point, Point]:
    """Transform a local-frame rect to a board-coord AABB.

    Uses KiCad's footprint orientation convention via
    :func:`geometry.transform_point` (each local corner -> board coords). For
    90/180/270 the AABB is exact; for arbitrary angles it is the conservative
    bounding box of the rotated rect (over-approximating the keep-out, safe).
    """
    origin = Point(origin_x, origin_y)
    corners = [
        (spec["x_min"], spec["y_min"]),
        (spec["x_max"], spec["y_min"]),
        (spec["x_max"], spec["y_max"]),
        (spec["x_min"], spec["y_max"]),
    ]
    pts = [
        geometry.transform_point(Point(lx, ly), origin, rotation_deg)
        for lx, ly in corners
    ]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return Point(min(xs), min(ys)), Point(max(xs), max(ys))


def extract_keepout_rects(board, cfg: dict) -> list[KeepoutRect]:
    """Return owner-tagged board-coord keep-out rects from ``board``."""
    families: dict[str, dict] = cfg.get("antenna_keepouts", {}) or {}
    rects: list[KeepoutRect] = []

    for fp in board.Footprints():
        ref = fp.GetReferenceAsString()

        # --- preserve: footprint-internal rule-area keep-outs ---
        for zone in list(fp.Zones()):  # ZONES is iterable; no GetCount()
            if not zone.GetIsRuleArea():
                continue
            # Only zones that keep components out matter to the placer.
            if not (zone.GetDoNotAllowFootprints() or zone.GetDoNotAllowPads()):
                continue
            bb = zone.Outline().BBox()  # board coords for a placed footprint
            opos = fp.GetPosition()
            rects.append(
                KeepoutRect(
                    tl=Point(pcbnew.ToMM(bb.GetLeft()), pcbnew.ToMM(bb.GetTop())),
                    br=Point(pcbnew.ToMM(bb.GetRight()), pcbnew.ToMM(bb.GetBottom())),
                    owner_ref=ref,
                    source="preserve",
                    owner_origin=Point(pcbnew.ToMM(opos.x), pcbnew.ToMM(opos.y)),
                )
            )

        # --- inject: config family-spec near-field rect ---
        if not families:
            continue
        candidates = _footprint_name_candidates(fp)
        if not candidates:
            continue
        pos = fp.GetPosition()
        ox, oy = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
        rot = fp.GetOrientationDegrees()
        for pattern, spec in families.items():
            if not _matches_family(candidates, pattern):
                continue
            tl, br = _transform_local_rect(spec, ox, oy, rot)
            rects.append(
                KeepoutRect(
                    tl=tl,
                    br=br,
                    owner_ref=ref,
                    source="inject",
                    owner_origin=Point(ox, oy),
                )
            )

    return rects
