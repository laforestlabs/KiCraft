"""Routing-only protection for native local ground-return paths.

These rule areas are not copper or qualification: the final native ground,
thermal and DRC audits remain authoritative.
"""
from __future__ import annotations

from collections import deque
import math

import pcbnew
import shapely
from shapely.geometry import LineString, Point, Polygon
from shapely.ops import nearest_points
from shapely.strtree import STRtree

from .local_contract import _copper_graph, _filled_straight_bond, _ground_pad_vias, _ground_regions


def _ground_pads(board, ref: str, pin: str, net: str) -> list:
    footprint = board.FindFootprintByReference(ref)
    pads = [] if footprint is None else [
        pad for pad in footprint.Pads()
        if pad.GetNumber() == str(pin) and pad.GetNetname() == net
    ]
    if not pads:
        raise ValueError(f"local ground reservation pad/net mismatch: {ref}.{pin}")
    return pads


def _point(position):
    return (position.x, position.y)


def _polygon(points):
    shape = pcbnew.SHAPE_POLY_SET()
    shape.NewOutline()
    for x, y in points:
        shape.Append(int(x), int(y))
    return shape


def _square(x, y, radius):
    return _polygon(((x - radius, y - radius), (x + radius, y - radius),
                     (x + radius, y + radius), (x - radius, y + radius)))


def _direct_regions(pad, regions, layer):
    if not pad.IsOnLayer(layer):
        return set()
    shape = pcbnew.SHAPE_POLY_SET()
    pad.TransformShapeToPolygon(shape, layer, 0, pcbnew.FromMM(0.001), pcbnew.ERROR_INSIDE)
    result = set()
    for index, (region_layer, region) in enumerate(regions):
        if region_layer != layer:
            continue
        overlap = pcbnew.SHAPE_POLY_SET(region)
        overlap.BooleanIntersection(shape)
        if overlap.OutlineCount() and abs(overlap.Area()) > 0:
            result.add(index)
    return result


def _contains(shape, point):
    return shape.Contains(pcbnew.VECTOR2I(*point))


def _native_polygon(shape, index):
    """Convert one native outline and its holes without changing its geometry."""
    def ring(chain):
        return [_point(chain.CPoint(i)) for i in range(chain.PointCount())]

    polygon = Polygon(ring(shape.Outline(index)),
                      [ring(shape.Hole(index, i)) for i in range(shape.HoleCount(index))])
    native_area = abs(shape.UnitSet(index).Area())
    tolerance = max(4.0, native_area * 1e-12)
    if not polygon.is_valid and not polygon.is_empty:
        # KiCad can encode touching filled contours as one self-touching ring.
        # GEOS requires separate polygons. Accept that decomposition only when
        # every vertex remains on the native grid and native XOR proves that
        # neither copper nor holes changed.
        candidate = shapely.make_valid(polygon)
        pending, parts = [candidate], []
        while pending:
            part = pending.pop()
            if part.geom_type == "Polygon":
                parts.append(part)
            elif part.geom_type in ("MultiPolygon", "GeometryCollection"):
                pending.extend(part.geoms)
        # A repaired contour can include zero-area lines. Keep only copper
        # surfaces, and still require native XOR to prove exact equivalence.
        candidate = shapely.union_all(parts)
        rebuilt = pcbnew.SHAPE_POLY_SET()
        for part in parts:
            for hole, chain in enumerate((part.exterior, *part.interiors)):
                if hole:
                    rebuilt.NewHole()
                else:
                    rebuilt.NewOutline()
                for x, y in list(chain.coords)[:-1]:
                    if x != round(x) or y != round(y):
                        raise ValueError("local ground reservation native polygon conversion is invalid")
                    rebuilt.Append(int(x), int(y), -1, hole - 1)
        if parts and candidate.is_valid and abs(candidate.area - native_area) <= tolerance:
            difference = pcbnew.SHAPE_POLY_SET(shape.UnitSet(index))
            difference.BooleanXor(rebuilt)
            if difference.OutlineCount() == 0:
                polygon = candidate
    if not polygon.is_valid or polygon.is_empty or abs(polygon.area - native_area) > tolerance:
        raise ValueError("local ground reservation native polygon conversion is invalid")
    return polygon


def _mesh(region, width):
    """Build one constrained triangle mesh per native-eroded component."""
    eroded = pcbnew.SHAPE_POLY_SET(region)
    eroded.Deflate(max(1, (width + 1) // 2) + 2,
                   pcbnew.CORNER_STRATEGY_ROUND_ALL_CORNERS, max(1, pcbnew.FromMM(0.001)))
    components = []
    for index in range(eroded.OutlineCount()):
        shape = eroded.UnitSet(index)
        polygon = _native_polygon(shape, 0)
        triangles = list(shapely.constrained_delaunay_triangles(polygon).geoms)
        if not triangles:
            raise ValueError("local ground reservation triangulation is empty")
        union = shapely.union_all(triangles)
        tolerance = max(4.0, polygon.area * 1e-12)
        if (abs(sum(triangle.area for triangle in triangles) - polygon.area) > tolerance
                or union.symmetric_difference(polygon).area > tolerance):
            raise ValueError("local ground reservation triangulation changes native fill")
        adjacency = [[] for _ in triangles]
        edges = {}
        for number, triangle in enumerate(triangles):
            vertices = list(triangle.exterior.coords)[:-1]
            if len(vertices) != 3 or triangle.area <= 0:
                raise ValueError("local ground reservation has invalid mesh triangle")
            for a, b in zip(vertices, vertices[1:] + vertices[:1]):
                edge = tuple(sorted((a, b)))
                if edge[0] == edge[1]:
                    continue
                if edge in edges:
                    other = edges.pop(edge)
                    portal = (round((a[0] + b[0]) / 2), round((a[1] + b[1]) / 2))
                    adjacency[number].append((other, portal))
                    adjacency[other].append((number, portal))
                else:
                    edges[edge] = number
        components.append((shape, polygon, triangles, STRtree(triangles), adjacency))
    return components


def _anchor(region, component, anchor, pad, layer, *, pad_cache=None, region_cache=None):
    # Every padless bond below requires its start inside the original fill.
    # Hole centres cannot qualify, regardless of how many triangles we try.
    if pad is None and not _contains(region, anchor):
        return None
    shape, polygon, triangles, tree, _ = component
    allowed = shape
    contact = None
    if pad is not None:
        key = (id(region), id(pad), layer)
        cached = None if pad_cache is None else pad_cache.get(key)
        if cached is None:
            pad_shape = pcbnew.SHAPE_POLY_SET()
            pad.TransformShapeToPolygon(pad_shape, layer, 0,
                                        pcbnew.FromMM(0.001), pcbnew.ERROR_INSIDE)
            contact = pcbnew.SHAPE_POLY_SET(region)
            contact.BooleanIntersection(pad_shape)
            cached = (pad_shape, contact, {})
            if pad_cache is not None:
                pad_cache[key] = cached
        pad_shape, contact, contact_parts = cached
        allowed = pcbnew.SHAPE_POLY_SET(shape)
        allowed.BooleanIntersection(pad_shape)
    if _contains(allowed, anchor):
        return anchor, anchor
    key = id(region)
    cached_region = None if region_cache is None else region_cache.get(key)
    if cached_region is None:
        material = shapely.union_all([
            _native_polygon(region, index) for index in range(region.OutlineCount())
        ])
        shapely.prepare(material)
        region_parts = shapely.get_parts(material)
        cached_region = (material, region_parts, shapely.STRtree(region_parts))
        if region_cache is not None:
            region_cache[key] = cached_region
    material, region_parts, region_tree = cached_region
    if pad is None:
        containing = region_tree.query(Point(anchor), predicate="intersects")
        if not any(region_parts[int(index)].intersects(polygon) for index in containing):
            # No finite-width bond can cross disconnected original copper.
            return None

    def bond(start, end):
        # A native finite-width bond cannot leave the original filled copper.
        # Reject blocked centre lines cheaply; native containment still decides.
        return (material.covers(LineString((start, end)))
                and _filled_straight_bond(region, start, end))

    def candidate(part, material, target):
        nearest_x, nearest_y = nearest_points(part, Point(target))[0].coords[0]
        inside_x, inside_y = part.representative_point().coords[0]
        dx, dy = inside_x - nearest_x, inside_y - nearest_y
        previous = None
        # Native Contains excludes boundaries. Move towards a known interior
        # point, checking each candidate against the actual native material.
        for fraction in (0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4,
                         1e-3, 1e-2, 0.1, 0.5, 1.0):
            point = (round(nearest_x + fraction * dx),
                     round(nearest_y + fraction * dy))
            if point == previous:
                continue
            previous = point
            if part.covers(Point(point)) and _contains(material, point):
                return point
        return None

    parts = ([polygon] if pad is None else
             [_native_polygon(allowed, index) for index in range(allowed.OutlineCount())])
    candidates = [candidate(part, allowed, anchor) for part in parts]
    for point in sorted((p for p in candidates if p is not None),
                        key=lambda p: (math.dist(anchor, p), p)):
        if polygon.covers(Point(point)) and (pad is not None or bond(anchor, point)):
            return point, point if pad is not None else anchor
    if pad is None:
        # The nearest part may be across a hole; try other triangle interiors,
        # still requiring a native copper bond to the actual via centre.
        point_geometry = Point(anchor)
        nearest = sorted(int(i) for i in tree.query_nearest(point_geometry))
        for triangle_index in nearest:
            point = candidate(triangles[triangle_index], shape, anchor)
            if point is not None and bond(anchor, point):
                return point, anchor
        nearest_set = set(nearest)
        remaining = sorted((i for i in range(len(triangles)) if i not in nearest_set),
                           key=lambda i: triangles[i].distance(point_geometry))
        for triangle_index in remaining:
            point = candidate(triangles[triangle_index], shape, anchor)
            if point is not None and bond(anchor, point):
                return point, anchor
    else:
        # Erosion can remove all material inside a small or edge-connected pad.
        # Start at actual pad/fill contact, then prove a native copper bond to
        # the eroded component rather than assuming the pad centre is filled.
        for index in range(contact.OutlineCount()):
            if index not in contact_parts:
                contact_parts[index] = _native_polygon(contact, index)
            part = contact_parts[index]
            containing = region_tree.query(part, predicate="intersects")
            if not any(region_parts[int(i)].intersects(polygon) for i in containing):
                continue
            # The closest triangle is normally sufficient. If it cannot
            # establish a native bond, retain the complete distance-ordered
            # search, including every other triangle.
            nearest = sorted(int(i) for i in tree.query_nearest(part))
            for triangle_index in nearest:
                triangle = triangles[triangle_index]
                near_contact, near_plane = nearest_points(part, triangle)
                start = candidate(part, contact, (near_plane.x, near_plane.y))
                end = candidate(triangle, shape, (near_contact.x, near_contact.y))
                if start is not None and end is not None and bond(start, end):
                    return end, start
            nearest_set = set(nearest)
            remaining = sorted((i for i in range(len(triangles)) if i not in nearest_set),
                               key=lambda i: triangles[i].distance(part))
            for triangle_index in remaining:
                triangle = triangles[triangle_index]
                near_contact, near_plane = nearest_points(part, triangle)
                start = candidate(part, contact, (near_plane.x, near_plane.y))
                end = candidate(triangle, shape, (near_contact.x, near_contact.y))
                if start is not None and end is not None and bond(start, end):
                    return end, start
    return None


def _mesh_centerline(component, start, end):
    """Find and safely shorten a centerline within one native-eroded mesh."""
    shape, polygon, triangles, tree, adjacency = component
    starts = sorted(int(i) for i in tree.query(Point(start), predicate="intersects"))
    ends = set(int(i) for i in tree.query(Point(end), predicate="intersects"))
    if not starts or not ends:
        return None
    queue = deque(starts)
    previous = {i: None for i in starts}
    target = None
    while queue:
        node = queue.popleft()
        if node in ends:
            target = node
            break
        for other, portal in adjacency[node]:
            if other not in previous:
                previous[other] = (node, portal)
                queue.append(other)
    if target is None:
        return None
    portals = []
    while previous[target] is not None:
        target, portal = previous[target]
        portals.append(portal)
    points = [start, *reversed(portals), end]
    # Native bond proof is retained even for polygon-covered shortcuts.
    shapely.prepare(polygon)
    shortcut = [points[0]]
    current = 0
    while current < len(points) - 1:
        next_index = len(points) - 1
        while next_index > current + 1:
            if (polygon.covers(LineString((points[current], points[next_index])))
                    and _filled_straight_bond(shape, points[current], points[next_index])):
                break
            next_index -= 1
        shortcut.append(points[next_index])
        current = next_index
    return shortcut


def _path(region, width, first, second, components, layer, *, anchors=None):
    """Route across positive-length triangle portals in native-eroded fill."""
    for component in components:
        shape, polygon, triangles, tree, adjacency = component
        if anchors is None:
            first_anchor = _anchor(region, component, *first, layer)
            second_anchor = _anchor(region, component, *second, layer)
        else:
            first_anchor, second_anchor = anchors
        if first_anchor is None or second_anchor is None:
            continue
        start, first_contact = first_anchor
        end, second_contact = second_anchor
        if (not tree.query(Point(start), predicate="intersects").size
                or not tree.query(Point(end), predicate="intersects").size):
            raise ValueError("local ground reservation anchor is outside triangulation")
        shortcut = _mesh_centerline(component, start, end)
        if shortcut is None:
            continue
        endpoints = [first_contact, second_contact]
        points = [endpoints[0], *shortcut, endpoints[1]]
        half = max(1, (width + 1) // 2)
        tube = pcbnew.SHAPE_POLY_SET()
        for x, y in points:
            tube.BooleanAdd(_square(x, y, half))
        for a, b in zip(points, points[1:]):
            dx, dy = b[0] - a[0], b[1] - a[1]
            length = math.hypot(dx, dy)
            if not length:
                continue
            ox, oy = -dy * half / length, dx * half / length
            tube.BooleanAdd(_polygon(((round(a[0] + ox), round(a[1] + oy)),
                                      (round(b[0] + ox), round(b[1] + oy)),
                                      (round(b[0] - ox), round(b[1] - oy)),
                                      (round(a[0] - ox), round(a[1] - oy)))))
        tube.BooleanIntersection(region)
        for index in range(tube.OutlineCount()):
            part = tube.UnitSet(index)
            if (all(_contains(part, p) for p in endpoints) and abs(part.Area()) > 0):
                return part, endpoints
    return None


def _stamp(board, shape, zone_name, clearance_compensation, layer):
    if clearance_compensation:
        shape = pcbnew.SHAPE_POLY_SET(shape)
        error = max(1, pcbnew.FromMM(0.001))
        # Round-corner polygonization may lie inside the ideal offset.
        shape.Inflate(clearance_compensation + error,
                      pcbnew.CORNER_STRATEGY_ROUND_ALL_CORNERS, error)
    count, area = 0, 0.0
    for index in range(shape.OutlineCount()):
        part = shape.UnitSet(index)
        if not abs(part.Area()):
            continue
        zone = pcbnew.ZONE(board)
        zone.SetLayer(layer)
        zone.SetIsRuleArea(True)
        zone.SetDoNotAllowTracks(True)
        zone.SetDoNotAllowVias(True)
        zone.SetDoNotAllowPads(False)
        zone.SetDoNotAllowCopperPour(False)
        zone.SetZoneName(zone_name)
        outline = zone.Outline()
        outline.AddOutline(part.Outline(0))
        for hole in range(part.HoleCount(0)):
            outline.AddHole(part.Hole(0, hole), 0)
        board.Add(zone)
        count += 1
        area += abs(part.Area()) / float(pcbnew.FromMM(1)) ** 2
    return count, area


def reserve_local_ground_planes(
    board, specs: list[dict], zone_name: str, *,
    router_clearance_mm: float,
) -> dict:
    """Reserve thin connected native witnesses for local ground returns."""
    if not math.isfinite(router_clearance_mm) or router_clearance_mm < 0:
        raise ValueError("invalid router clearance for local ground reservation")
    router_clearance = pcbnew.FromMM(router_clearance_mm)
    groups = {}
    for spec in specs:
        ground = spec.get("ground_return")
        if not ground:
            continue
        limit = float(ground["max_via_distance_mm"])
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError("invalid local ground reservation distance")
        net, owner, pin = ground["net"], ground["owner_ref"], str(ground["owner_pin"])
        member, member_pin = spec["bypass_ref"], str(ground["member_pin"])
        group = groups.setdefault((net, owner, pin), [])
        for a in _ground_pads(board, owner, pin, net):
            for b in _ground_pads(board, member, member_pin, net):
                group.append(((a, (owner, pin)), (b, (member, member_pin)), limit))

    # Mirror _ground_regions' order so each island retains its source rules.
    regions_cache, widths_cache, clearance_cache, graph_cache, mesh_cache = {}, {}, {}, {}, {}
    for net, _, _ in groups:
        if net in regions_cache:
            continue
        regions_cache[net] = _ground_regions(board, net)
        widths, clearances = [], []
        for zone in board.Zones():
            if zone.GetNetname() != net or zone.GetIsRuleArea():
                continue
            for layer in range(pcbnew.B_Cu + 1):
                if zone.IsOnLayer(layer):
                    count = zone.GetFilledPolysList(layer).OutlineCount()
                    widths.extend([zone.GetMinThickness()] * count)
                    clearances.extend([max(0, zone.GetOwnClearance(layer) - router_clearance)] * count)
        widths_cache[net] = widths
        clearance_cache[net] = clearances

    landing_cache, stamped_landings = {}, set()
    rows = []
    for (net, owner, pin), group in sorted(groups.items()):
        regions, widths, clearances = regions_cache[net], widths_cache[net], clearance_cache[net]
        count, area, anchors = 0, 0.0, []
        for first, second, limit in group:
            choices = []
            for pad, pin_id in (first, second):
                bonds = {}
                for layer in (pcbnew.B_Cu, pcbnew.F_Cu):
                    for index in sorted(_direct_regions(pad, regions, layer)):
                        bonds.setdefault(index, []).append((_point(pad.GetPosition()), pad, pin_id, None))
                for via, indices, landing_regions in _ground_pad_vias(
                    board, net, pad, pin_id, limit, regions, graph_cache
                ):
                    via_point = _point(via.GetPosition())
                    landing = None
                    if landing_regions:
                        landing_key = (net, pad.m_Uuid.AsString(), via_point, tuple(sorted(landing_regions)))
                        if landing_key not in landing_cache:
                            landing_cache[landing_key] = None
                            pad_point = _point(pad.GetPosition())
                            for landing_index in sorted(landing_regions):
                                if widths[landing_index] <= 0:
                                    raise ValueError(f"local ground reservation has invalid zone width: {owner}.{pin}")
                                mesh_key = (net, landing_index, widths[landing_index])
                                if mesh_key not in mesh_cache:
                                    mesh_cache[mesh_key] = _mesh(regions[landing_index][1], widths[landing_index])
                                route = _path(
                                    regions[landing_index][1], widths[landing_index],
                                    (pad_point, pad), (via_point, None),
                                    mesh_cache[mesh_key], regions[landing_index][0],
                                )
                                if route is not None and _filled_straight_bond(route[0], pad_point, via_point):
                                    landing_cache[landing_key] = (landing_key, landing_index, route[0])
                                    break
                        landing = landing_cache[landing_key]
                        if landing is None:
                            continue
                    for index in sorted(indices):
                        bonds.setdefault(index, []).append((via_point, None, pin_id, landing))
                choices.append(bonds)
            witness = None
            for index in sorted(choices[0].keys() & choices[1].keys(),
                                key=lambda i: (regions[i][0] != pcbnew.B_Cu, i)):
                if widths[index] <= 0:
                    raise ValueError(f"local ground reservation has invalid zone width: {owner}.{pin}")
                key = (net, index, widths[index])
                if key not in mesh_cache:
                    mesh_cache[key] = _mesh(regions[index][1], widths[index])
                for a in choices[0][index]:
                    for b in choices[1][index]:
                        witness = _path(regions[index][1], widths[index], a[:2], b[:2],
                                        mesh_cache[key], regions[index][0])
                        if witness is not None:
                            anchors.append({"region": index, "points": [list(p) for p in witness[1]],
                                            "pins": [list(a[2]), list(b[2])],
                                            "kinds": ["pad" if a[1] is not None else "via",
                                                      "pad" if b[1] is not None else "via"]})
                            break
                    if witness is not None:
                        break
                if witness is not None:
                    break
            if witness is None:
                raise ValueError(f"local ground reservation has no connected F.Cu or B.Cu witness: {owner}.{pin}")
            n, stamped_area = _stamp(board, witness[0], zone_name, clearances[index], regions[index][0])
            count, area = count + n, area + stamped_area
            for endpoint in (a, b):
                if endpoint[3] is None:
                    continue
                landing_key, landing_index, landing_shape = endpoint[3]
                if landing_key in stamped_landings:
                    continue
                n, stamped_area = _stamp(
                    board, landing_shape, zone_name, clearances[landing_index], regions[landing_index][0],
                )
                count, area = count + n, area + stamped_area
                stamped_landings.add(landing_key)
        rows.append({"net": net, "owner_ref": owner, "owner_pin": pin,
                     "zones": count, "reservation_area_mm2": area, "anchors": anchors})
    return {"groups": rows, "zones": sum(row["zones"] for row in rows)}


def reserve_connected_ground(
    board, net: str, zone_name: str, *, width_mm: float, router_clearance_mm: float,
) -> dict:
    """Protect a thin filled-copper forest joining every physical ground pad.

    The fixed copper graph supplies zero-cost connections; its pad nodes identify
    physical pad instances, including duplicate reference/pin numbers.  A
    candidate plane edge is considered only when both contacts anchor in the
    same native-eroded component. Failed native proofs retain all alternate
    contacts; neither a search cap nor a guessed polygon shortcut qualifies.
    """
    if not math.isfinite(width_mm) or width_mm <= 0:
        raise ValueError("invalid global ground witness width")
    width = pcbnew.FromMM(width_mm)
    router_clearance = pcbnew.FromMM(router_clearance_mm)
    regions = _ground_regions(board, net)
    widths = []
    clearances = []
    for zone in board.Zones():
        if zone.GetNetname() != net or zone.GetIsRuleArea():
            continue
        for layer in range(pcbnew.B_Cu + 1):
            if zone.IsOnLayer(layer):
                count = zone.GetFilledPolysList(layer).OutlineCount()
                clearances.extend([max(0, zone.GetOwnClearance(layer) - router_clearance)] * count)
                widths.extend([max(width, zone.GetMinThickness())] * count)
    graph, _, _ = _copper_graph(board, net)
    parents = {}

    def find(node):
        parents.setdefault(node, node)
        root = node
        while parents[root] != root:
            root = parents[root]
        while parents[node] != node:
            next_node = parents[node]
            parents[node] = root
            node = next_node
        return root

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parents[b] = a

    for node, edges in graph.items():
        for other, _, edge_width in edges:
            if edge_width != 0:
                union(node, other)

    pads = [(fp, pad) for fp in board.GetFootprints() for pad in fp.Pads()
            if pad.GetNetname() == net]
    terminals = []
    pad_contacts = []
    for index, (fp, pad) in enumerate(pads):
        plated = pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH
        layers = [layer for layer in (pcbnew.F_Cu, pcbnew.B_Cu) if pad.IsOnLayer(layer)]
        if not layers:
            raise ValueError(f"ground pad has no outer copper: {fp.GetReferenceAsString()}.{pad.GetNumber()}")
        nodes = [("pad", index)] if plated else [("pad", index, layer) for layer in layers]
        terminals.extend(nodes)
        for layer in layers:
            node = nodes[0] if plated else ("pad", index, layer)
            pad_contacts.append((node, _point(pad.GetPosition()), pad, layer))
    if len(terminals) <= 1:
        return {"net": net, "pads": len(pads), "zones": 0, "reservation_area_mm2": 0.0}

    # A fixed track/via graph node already inside an eroded component is a
    # valid contact. Pads may anchor through any actual pad/fill overlap.
    fixed = []
    for node in graph:
        if (isinstance(node, tuple) and len(node) == 2
                and isinstance(node[0], tuple) and len(node[0]) == 2):
            fixed.append((node, node[0], None, node[1]))
    # Equal-distance forest edges must not depend on serialized copper UUID
    # order. Node identity still tracks every distinct physical pad instance.
    contacts = sorted(pad_contacts + fixed, key=lambda contact: (
        contact[2] is None, contact[3], contact[1],
        "" if contact[2] is None else contact[2].GetParentFootprint().GetReferenceAsString(),
        "" if contact[2] is None else contact[2].GetNumber(),
    ))
    buckets = []
    pad_cache = {}
    region_cache = {}
    for index, (layer, region) in enumerate(regions):
        components = _mesh(region, widths[index])
        for component in components:
            shape = component[0]
            candidates = []
            deferred = []
            for node, point, pad, contact_layer in contacts:
                if contact_layer != layer:
                    continue
                if pad is None:
                    if not _contains(shape, point):
                        deferred.append((node, point, pad))
                        continue
                else:
                    key = (id(region), id(pad), layer)
                    cached = pad_cache.get(key)
                    if cached is None:
                        pad_shape = pcbnew.SHAPE_POLY_SET()
                        pad.TransformShapeToPolygon(pad_shape, layer, 0,
                                                    pcbnew.FromMM(0.001), pcbnew.ERROR_INSIDE)
                        contact = pcbnew.SHAPE_POLY_SET(region)
                        contact.BooleanIntersection(pad_shape)
                        cached = (pad_shape, contact, {})
                        pad_cache[key] = cached
                    pad_shape = pcbnew.SHAPE_POLY_SET(cached[0])
                    pad_shape.BooleanIntersection(shape)
                    if not pad_shape.OutlineCount() or not abs(pad_shape.Area()):
                        deferred.append((node, point, pad))
                        continue
                anchor = _anchor(region, component, point, pad, layer,
                                 pad_cache=pad_cache, region_cache=region_cache)
                if anchor is not None:
                    candidates.append((node, point, pad, anchor))
            buckets.append((index, component, candidates, deferred))

    rejected = set()
    zones, area = 0, 0.0
    layers = {}
    # Expand back-face deferred contacts first; every front-only alternative
    # remains available if the currently proved contacts cannot join all pads.
    deferred_contacts = iter(
        (index, component, candidates, node, point, pad)
        for index, component, candidates, deferred in sorted(
            buckets, key=lambda bucket: regions[bucket[0]][0] != pcbnew.B_Cu
        )
        for node, point, pad in deferred
    )
    while len({find(node) for node in terminals}) > 1:
        best = None
        best_terminal = None
        terminal_roots = {find(node) for node in terminals}
        for bucket_id, (region_index, _, candidates, _) in enumerate(buckets):
            for a in range(len(candidates)):
                first = candidates[a]
                for b in range(a + 1, len(candidates)):
                    second = candidates[b]
                    if find(first[0]) == find(second[0]) or (bucket_id, a, b) in rejected:
                        continue
                    distance = math.dist(first[3][0], second[3][0])
                    key = (distance, regions[region_index][0] != pcbnew.B_Cu, bucket_id, a, b)
                    if best is None or key < best:
                        best = key
                    if ((find(first[0]) in terminal_roots or find(second[0]) in terminal_roots)
                            and (best_terminal is None or key < best_terminal)):
                        best_terminal = key
        # Grow terminal-bearing trees first, retaining padless-to-padless edges
        # as a complete Steiner fallback when no such edge is available.
        best = best_terminal if best_terminal is not None else best
        if best is None:
            # Expand lazily: every deferred contact remains available if the
            # current forest cannot connect all terminals. Once connected,
            # padless components need not be joined merely for their own sake.
            for index, component, candidates, node, point, pad in deferred_contacts:
                anchor = _anchor(regions[index][1], component, point, pad,
                                 regions[index][0], pad_cache=pad_cache,
                                 region_cache=region_cache)
                if anchor is not None:
                    candidates.append((node, point, pad, anchor))
                    break
            else:
                raise ValueError(f"connected {net} ground pads have no complete native-filled witness forest")
            continue
        _, _, bucket_id, a, b = best
        index, component, candidates, _ = buckets[bucket_id]
        first, second = candidates[a], candidates[b]
        witness = _path(regions[index][1], widths[index], first[1:3], second[1:3],
                        [component], regions[index][0], anchors=(first[3], second[3]))
        if witness is None:
            rejected.add((bucket_id, a, b))
            continue
        count, stamped_area = _stamp(
            board, witness[0], zone_name, clearances[index], regions[index][0]
        )
        if not count:
            rejected.add((bucket_id, a, b))
            continue
        zones += count
        area += stamped_area
        layer_name = board.GetLayerName(regions[index][0])
        layer_stats = layers.setdefault(layer_name, {"zones": 0, "reservation_area_mm2": 0.0})
        layer_stats["zones"] += count
        layer_stats["reservation_area_mm2"] += stamped_area
        union(first[0], second[0])
    return {"net": net, "pads": len(pads), "zones": zones,
            "reservation_area_mm2": area, "layers": layers}
