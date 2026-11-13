"""Conservative two-face free-space candidates for signal repair.

This module only plans.  The caller owns stamping, rollback, and native DRC.
Coordinates passed to native geometry are KiCad internal units.
"""
from __future__ import annotations

from collections import deque
import math

import pcbnew
import shapely
from shapely.geometry import LineString, Point, box

from kicraft.autoplacer.brain.breakout_stubs import (
    BreakoutSpec, STAMP_CLEARANCE_GUARD_MM, _own_clearance_mm, _pad_hole_overhang_mm,
)
from kicraft.autoplacer.brain.gnd_pour import (
    _board_edge_geometry, _parent_ground_reservations, _smd_via_clearance_iu,
)
from kicraft.autoplacer.fab_profile import fab_floors, fanout_via
from kicraft.autoplacer.hardware.ground_reservations import _mesh, _mesh_centerline, _native_polygon
from kicraft.autoplacer.hardware.keepout_extract import collect_track_via_rule_areas


_LAYERS = (("B.Cu", pcbnew.B_Cu), ("F.Cu", pcbnew.F_Cu))
_SCALE = 1_000_000
# Polygonal buffers must enclose, rather than inscribe, the requested distance.
_BUFFER_FACTOR = 1 / math.cos(math.pi / 128)


def _buffer(geometry, distance):
    return geometry.buffer(distance * _BUFFER_FACTOR + 4, quad_segs=32)


def _polygons(geometry):
    if geometry.is_empty:
        return
    if geometry.geom_type == "Polygon":
        yield geometry
    elif geometry.geom_type in ("MultiPolygon", "GeometryCollection"):
        for part in geometry.geoms:
            yield from _polygons(part)


def _native_region(geometry):
    """Round only after an inward allowance, so conversion cannot gain space."""
    region = pcbnew.SHAPE_POLY_SET()
    for part in _polygons(geometry.buffer(-4, quad_segs=32)):
        for hole, ring in enumerate((part.exterior, *part.interiors)):
            if hole:
                region.NewHole()
            else:
                region.NewOutline()
            for x, y in list(ring.coords)[:-1]:
                region.Append(round(x), round(y), -1, hole - 1)
    return region


def _outline_polygons(shape):
    return [_native_polygon(shape, i) for i in range(shape.OutlineCount())]


def _bbox(item):
    bounds = item.GetBoundingBox()
    return box(bounds.GetLeft(), bounds.GetTop(), bounds.GetRight(), bounds.GetBottom())


def _mesh_repair_specs(board, net: str, sources, targets, cfg) -> list[BreakoutSpec]:
    """Plan a shortest-transition path between native-proven same-net anchors.

    Anchors are ``(xy_mm, frozenset(layer_names))``.  The result is a batch of
    free-coordinate stubs; no board or configuration state is changed.
    Invalid supplied ground reservation plans raise rather than omit protection.
    """
    owner, masks = (None, [])
    if "_parent_ground_reservation_plan" in cfg:
        owner, masks = _parent_ground_reservations(cfg["_parent_ground_reservation_plan"])
        if any(zone.Outline().OutlineCount() == 0 for zone in masks):
            raise ValueError("parent ground reservation plan has empty geometry")
    edge = _board_edge_geometry(board)
    if edge is None or not sources or not targets or not net:
        return []
    pads = [pad for fp in board.GetFootprints() for pad in fp.Pads()]
    own_pads = [pad for pad in pads if pad.GetNetname() == net and pad.GetNetCode()]
    if not own_pads:
        return []
    floor = fab_floors(cfg)
    width = int(pcbnew.FromMM(floor["track_mm"]))
    diameter, drill = fanout_via(cfg)
    radius = pcbnew.FromMM(diameter) / 2
    drill_radius = pcbnew.FromMM(drill) / 2
    hole_clearance_mm = float(cfg.get("hole_clearance_min_mm", 0.25))
    hole_clearance = pcbnew.FromMM(hole_clearance_mm)
    hole_min = pcbnew.ToMM(board.GetDesignSettings().m_HoleToHoleMin)
    if hole_min <= 0:
        hole_min = float(cfg.get("hole_to_hole_min_mm", 0.25))
    hole_min = pcbnew.FromMM(hole_min)
    # Any leg may use any same-net pad as its clearance stand-in.
    source_max = max(_own_clearance_mm(p, layer, floor["clearance_mm"])
                     for p in own_pads for _, layer in _LAYERS)
    src_clearance = {layer: source_max for _, layer in _LAYERS}
    outlines = _outline_polygons(edge["polygon"])
    surface = shapely.union_all(outlines).buffer(-(edge["clearance_iu"] + 4), quad_segs=32)
    areas = [(area.zone, area.blocks_tracks, area.blocks_vias)
             for area in collect_track_via_rule_areas(board)]
    areas.extend((zone, True, True) for zone in masks)
    layer_components = {}
    free_by_layer = {}
    for name, layer in _LAYERS:
        blockers = []
        for pad in pads:
            foreign = not pad.GetNetCode() or pad.GetNetname() != net
            size = pad.GetDrillSize()
            has_hole = size.x > 0 or size.y > 0
            if foreign and (pad.IsOnLayer(layer) or has_hole):
                required = STAMP_CLEARANCE_GUARD_MM + max(
                    floor["clearance_mm"], src_clearance[layer],
                    *(_own_clearance_mm(pad, face, floor["clearance_mm"])
                      for _, face in _LAYERS))
                overhang = _pad_hole_overhang_mm(pad)
                if overhang is not None:
                    required = max(required, hole_clearance_mm + overhang)
                blockers.append(_buffer(_bbox(pad), pcbnew.FromMM(required)))
            if has_hole and (foreign or pad.GetAttribute() == pcbnew.PAD_ATTRIB_NPTH):
                pos = pad.GetPosition()
                blockers.append(_buffer(Point(pos.x, pos.y), max(size.x, size.y) / 2 + hole_clearance))
        for track in board.GetTracks():
            if track.GetNetname() == net:
                continue
            via = track.GetClass() == "PCB_VIA"
            if not via and track.GetLayer() != layer:
                continue
            if via:
                p = track.GetPosition()
                shape = Point(p.x, p.y)
                try:
                    track_width = track.GetWidth(layer)
                except TypeError:
                    track_width = track.GetWidth()
            else:
                if track.GetClass() == "PCB_ARC":
                    shape = _bbox(track)
                else:
                    a, b = track.GetStart(), track.GetEnd()
                    shape = LineString(((a.x, a.y), (b.x, b.y)))
                track_width = track.GetWidth()
            pair = max(floor["clearance_mm"], src_clearance[layer],
                       _own_clearance_mm(track, layer, floor["clearance_mm"]))
            blockers.append(_buffer(shape, track_width / 2 + pcbnew.FromMM(pair + STAMP_CLEARANCE_GUARD_MM)))
            if via:
                blockers.append(_buffer(shape, track.GetDrillValue() / 2 + hole_clearance))
        drawings = list(board.GetDrawings())
        for footprint in board.GetFootprints():
            drawings.extend(footprint.GraphicalItems())
        for drawing in drawings:
            if drawing.GetLayer() != layer:
                continue
            if getattr(drawing, "GetNetname", lambda: "")() == net:
                continue
            pair = max(floor["clearance_mm"], src_clearance[layer],
                       _own_clearance_mm(drawing, layer, floor["clearance_mm"]))
            blockers.append(_buffer(_bbox(drawing), pcbnew.FromMM(pair + STAMP_CLEARANCE_GUARD_MM)))
        for zone, tracks, _vias in areas:
            if tracks and zone.IsOnLayer(layer):
                separation = max(floor["clearance_mm"], src_clearance[layer])
                blockers.extend(_buffer(poly, pcbnew.FromMM(separation + STAMP_CLEARANCE_GUARD_MM))
                                for poly in _outline_polygons(zone.Outline()))
        # Filled pours yield to new copper on refill. Immutable ground
        # witnesses above, not the old fill cache, reserve their routing space.
        free = surface.difference(shapely.union_all(blockers))
        free_by_layer[name] = free
        layer_components[name] = _mesh(_native_region(free), width)

    # Via barrel must fit both faces.  Additionally block all solder lands,
    # via-only rule areas and every existing drilled hole, including same-net.
    via_free = free_by_layer["B.Cu"].buffer(-(radius * _BUFFER_FACTOR + 4), quad_segs=32)
    via_free = via_free.intersection(
        free_by_layer["F.Cu"].buffer(-(radius * _BUFFER_FACTOR + 4), quad_segs=32))
    via_blockers = []
    for pad in pads:
        if pad.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN):
            via_blockers.append(_buffer(
                _bbox(pad),
                radius + _smd_via_clearance_iu(pad) + pcbnew.FromMM(STAMP_CLEARANCE_GUARD_MM),
            ))
        size = pad.GetDrillSize()
        if size.x > 0 or size.y > 0:
            pos = pad.GetPosition()
            via_blockers.append(_buffer(Point(pos.x, pos.y), max(size.x, size.y) / 2 + drill_radius + hole_min))
    for track in board.GetTracks():
        if track.GetClass() == "PCB_VIA":
            pos = track.GetPosition()
            via_blockers.append(_buffer(Point(pos.x, pos.y), track.GetDrillValue() / 2 + drill_radius + hole_min))
    for zone, _tracks, vias in areas:
        if vias and any(zone.IsOnLayer(layer) for _, layer in _LAYERS):
            via_blockers.extend(_buffer(poly, radius + pcbnew.FromMM(STAMP_CLEARANCE_GUARD_MM))
                                for poly in _outline_polygons(zone.Outline()))
    via_free = via_free.difference(shapely.union_all(via_blockers))

    graph = {}
    front = layer_components["F.Cu"]
    front_tree = shapely.STRtree([component[1] for component in front])
    for bi, back in enumerate(layer_components["B.Cu"]):
        available = via_free.intersection(back[1])
        if available.is_empty:
            continue
        for fi in sorted(int(i) for i in front_tree.query(available, predicate="intersects")):
            windows = available.intersection(front[fi][1])
            for window in _polygons(windows):
                point = window.representative_point()
                xy = (round(point.x), round(point.y))
                # Rounding must not move a transition out of the safe window.
                if not window.contains(Point(xy)):
                    continue
                bnode, fnode = ("B.Cu", bi), ("F.Cu", fi)
                graph.setdefault(bnode, []).append((fnode, xy))
                graph.setdefault(fnode, []).append((bnode, xy))
    endpoint_nodes = []
    for anchors in (sources, targets):
        found = {}
        for xy, layers in anchors:
            point = Point(*(round(v * _SCALE) for v in xy))
            for name, _layer in _LAYERS:
                if name in layers:
                    for index, component in enumerate(layer_components[name]):
                        if component[1].covers(point):
                            found.setdefault((name, index), (round(point.x), round(point.y)))
        endpoint_nodes.append(found)
    queue = deque(endpoint_nodes[0])
    previous = {node: None for node in endpoint_nodes[0]}
    target = None
    while queue:
        node = queue.popleft()
        if node in endpoint_nodes[1]:
            target = node
            break
        for other, xy in graph.get(node, ()):
            if other not in previous:
                previous[other] = (node, xy)
                queue.append(other)
    if target is None:
        return []
    transitions = []
    node = target
    while previous[node] is not None:
        parent, xy = previous[node]
        transitions.append((parent, node, xy))
        node = parent
    transitions.reverse()
    for index, (_old, _new, xy) in enumerate(transitions):
        if any(math.dist(xy, prior[2]) < 2 * drill_radius + hole_min + 8
               for prior in transitions[:index]):
            return []
    initial = node
    start = endpoint_nodes[0][initial]
    legs = [(old, xy, True) for old, _new, xy in transitions]
    legs.append((target, endpoint_nodes[1][target], False))
    specs = []
    for node, end, change_layer in legs:
        points = _mesh_centerline(layer_components[node[0]][node[1]], start, end)
        if points is None or start == end:
            return []
        xy = tuple(v / _SCALE for v in start)
        specs.append(BreakoutSpec(
            ref="track", pad="native-connected-anchor", start_xy=xy, near_xy=xy,
            net=net, waypoints=[(x / _SCALE, y / _SCALE) for x, y in points[1:]],
            layer=node[0], via_at_end=change_layer, via_size_mm=diameter,
            via_drill_mm=drill, strict_same_footprint=True, via_outside_smd_pads=True,
        ))
        start = end
    return specs
