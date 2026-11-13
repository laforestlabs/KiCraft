"""Native copper-path audit for reviewed local power contracts.

The graph uses track centre lines, native effective pad shapes and copper layers.
Unsupported geometry is rejected rather than approximated.
"""
from __future__ import annotations

import heapq
import math
from itertools import count

import pcbnew


def _point(x, y):
    return pcbnew.VECTOR2I(round(x), round(y))


def _position(point):
    return (int(point.x), int(point.y))


def _along(a, b, p):
    """Projection parameter for a point on a segment, within 2 internal units."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    d2 = dx * dx + dy * dy
    if not d2:
        return 0.0 if math.hypot(p[0] - a[0], p[1] - a[1]) <= 2 else None
    t = ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / d2
    if t < -1e-9 or t > 1 + 1e-9:
        return None
    if math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]) > 2:
        return None
    return min(1.0, max(0.0, t))


def _crossings(a, b, c, d):
    """Centreline intersections, including collinear overlap endpoints."""
    x, y = b[0] - a[0], b[1] - a[1]
    u, v = d[0] - c[0], d[1] - c[1]
    det = x * v - y * u
    if det:
        dx, dy = c[0] - a[0], c[1] - a[1]
        t = (dx * v - dy * u) / det
        s = (dx * y - dy * x) / det
        if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= s <= 1 + 1e-9:
            return [(round(a[0] + t * x), round(a[1] + t * y))]
        return []
    return list({p for p in (a, b, c, d) if _along(a, b, p) is not None
                 and _along(c, d, p) is not None})

def _project(a, b, p):
    dx, dy = b[0] - a[0], b[1] - a[1]
    d2 = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / d2)) if d2 else 0
    return (round(a[0] + t * dx), round(a[1] + t * dy))


def _closest_segment_points(a, b, c, d):
    candidates = [(p, _project(c, d, p)) for p in (a, b)]
    candidates.extend((_project(a, b, p), p) for p in (c, d))
    return min(candidates, key=lambda pair: math.dist(*pair))


def _via_width(via, layer):
    try:
        return via.GetWidth(layer)
    except TypeError:
        return via.GetWidth()

def _contact_width(first_width, second_width, distance):
    """Conservative common chord of two overlapping round track endcaps."""
    r1, r2 = first_width / 2, second_width / 2
    if distance >= r1 + r2:
        return 0
    if distance <= abs(r1 - r2):
        return min(first_width, second_width)
    x = (distance * distance + r1 * r1 - r2 * r2) / (2 * distance)
    return math.floor(2 * math.sqrt(max(0.0, r1 * r1 - x * x)))


def _track_via_annular_contact(
    a: tuple[int, int], b: tuple[int, int], track_width: int,
    center: tuple[int, int], via_width: int, drill_width: int,
) -> tuple[tuple[int, int], int] | None:
    """Prove a full-track-width transverse section inside actual via copper."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy)
    if (length <= 4 or track_width <= 0 or drill_width <= 0
            or via_width <= drill_width):
        return None
    ux, uy = dx / length, dy / length
    projection = (center[0] - a[0]) * ux + (center[1] - a[1]) * uy
    offset = abs((a[0] - center[0]) * -uy + (a[1] - center[1]) * ux)
    low, high = 2 - projection, length - 2 - projection
    outer_radius, inner_radius = via_width / 2, drill_width / 2
    candidates = dict.fromkeys(
        max(low, min(high, s))
        for s in (low, high, 0, -(inner_radius + 2), inner_radius + 2)
    )
    best = None
    best_distance = math.inf
    for s in candidates:
        if abs(s) >= outer_radius:
            continue
        outer = math.sqrt(outer_radius * outer_radius - s * s) - offset - 2
        half_width = min(track_width / 2, outer)
        if abs(s) <= inner_radius:
            inner = offset - math.sqrt(inner_radius * inner_radius - s * s) - 2
            half_width = min(half_width, inner)
        if math.floor(2 * max(0.0, half_width)) < track_width:
            continue
        # Strict radial margins and the inset segment establish an open
        # longitudinal neighbourhood, not merely endpoint/tangent contact.
        distance = math.hypot(s, offset)
        if distance < best_distance:
            best_distance = distance
            best = (
                (round(a[0] + (projection + s) * ux),
                 round(a[1] + (projection + s) * uy)),
                track_width,
            )
    return best


def _shape_boundaries(shape, a, b):
    """Intersect a centreline with exact native effective-shape primitives.

    Effective pad shapes are unions of rectangles, polygons, circles and
    stroked segments.  A stroked segment is a capsule, not its polygonal
    rendering.  Return all primitive boundary intersections; redundant cuts
    inside a union are harmless.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    d2 = dx * dx + dy * dy
    if not d2:
        return []
    points = []

    def circle(center, radius):
        ox, oy = a[0] - center[0], a[1] - center[1]
        dot = ox * dx + oy * dy
        disc = dot * dot - d2 * (ox * ox + oy * oy - radius * radius)
        if disc < 0:
            return
        root = math.sqrt(disc)
        for t in ((-dot - root) / d2, (-dot + root) / d2):
            if 0 <= t <= 1:
                points.append((round(a[0] + t * dx), round(a[1] + t * dy)))

    def edges(vertices):
        for c, d in zip(vertices, vertices[1:] + vertices[:1]):
            points.extend(_crossings(a, b, c, d))

    kind = shape.TypeName()
    if kind == "SH_COMPOUND":
        for child in pcbnew.Cast_to_SHAPE_COMPOUND(shape).GetSubshapes():
            points.extend(_shape_boundaries(child, a, b))
    elif kind == "SH_RECT":
        rect = pcbnew.Cast_to_SHAPE_RECT(shape)
        x, y = _position(rect.GetPosition())
        w, h = _position(rect.GetSize())
        edges([(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    elif kind == "SH_SIMPLE":
        poly = pcbnew.Cast_to_SHAPE_SIMPLE(shape)
        edges([_position(poly.CPoint(i)) for i in range(poly.PointCount())])
    elif kind == "SH_CIRCLE":
        disk = pcbnew.Cast_to_SHAPE_CIRCLE(shape)
        circle(_position(disk.GetCenter()), disk.GetRadius())
    elif kind == "SH_SEGMENT":
        capsule = pcbnew.Cast_to_SHAPE_SEGMENT(shape)
        seg = capsule.GetSeg()
        c, d = _position(seg.A), _position(seg.B)
        radius = capsule.GetWidth() / 2
        circle(c, radius)
        circle(d, radius)
        sx, sy = d[0] - c[0], d[1] - c[1]
        length = math.hypot(sx, sy)
        if length:
            # Intersect the two straight sides without rounding their offsets.
            nx, ny = -sy * radius / length, sx * radius / length
            for sign in (-1, 1):
                px, py = c[0] + sign * nx, c[1] + sign * ny
                det = dx * sy - dy * sx
                if det:
                    t = ((px - a[0]) * sy - (py - a[1]) * sx) / det
                    u = ((px - a[0]) * dy - (py - a[1]) * dx) / det
                    if 0 <= t <= 1 and 0 <= u <= 1:
                        points.append((round(a[0] + t * dx), round(a[1] + t * dy)))
    else:
        raise ValueError(f"unsupported native pad shape {kind}")
    return points


def _pad_shapes(pad, layer):
    # Custom polygons can contain disjoint islands and holes.  Do not infer
    # electrical continuity between them from a single virtual pad node.
    if pad.GetShape() == pcbnew.PAD_SHAPE_CUSTOM:
        raise ValueError("unsupported custom pad copper geometry")
    shape = pad.GetEffectiveShape(layer)
    hole = pad.GetEffectiveHoleShape() if pad.HasHole() else None
    return shape, hole


def _pad_contains(shape, hole, point, margin=0):
    p = _point(*point)
    # A drill boundary belongs to the surrounding plated copper; allow the
    # two-IU tolerance used by centreline intersection rounding, but never
    # admit a point in the drilled interior.
    hole_margin = margin if margin else -2
    return shape.Collide(p, margin) and not (hole and hole.Collide(p, hole_margin))


def _copper_graph(board, net):
    """Return (adjacency, pads-by-ref/pin, pad nodes) for one physical net."""
    tracks, vias, pads = [], [], []
    for item in board.GetTracks():
        if item.GetNetname() != net:
            continue
        if isinstance(item, pcbnew.PCB_VIA):
            vias.append(item)
        elif item.GetClass() == "PCB_TRACK":
            a, b = _position(item.GetStart()), _position(item.GetEnd())
            tracks.append((item, a, b, item.GetLayer()))
        else:
            raise ValueError(f"{net}: unsupported non-linear copper {item.GetClass()}")
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad.GetNetname() == net:
                pads.append((fp.GetReferenceAsString(), pad.GetNumber(), pad))
    cuts = [{a, b} for _, a, b, _ in tracks]
    contacts = []
    pad_contacts = {}
    pad_shapes = {}
    for i, (track, a, b, layer) in enumerate(tracks):
        for j in range(i):
            other, c, d, other_layer = tracks[j]
            if layer != other_layer:
                continue
            crossing = _crossings(a, b, c, d)
            for point in crossing:
                cuts[i].add(point)
                cuts[j].add(point)
            if not crossing:
                first, second = _closest_segment_points(a, b, c, d)
                distance = math.dist(first, second)
                width = _contact_width(track.GetWidth(), other.GetWidth(), distance)
                if width > 0:
                    cuts[i].add(first)
                    cuts[j].add(second)
                    contacts.append(((first, layer), (second, layer), distance, width))
        for via in vias:
            if not via.IsOnLayer(layer):
                continue
            point = _position(via.GetPosition())
            closest = _project(a, b, point)
            distance = math.dist(point, closest)
            if distance < (track.GetWidth() + _via_width(via, layer)) / 2:
                cuts[i].add(closest)
                # A via's drill removes copper from the nominal disk.  Keep
                # offset contacts for connectivity, but do not claim a
                # current-carrying width without proving the annular overlap.
                contacts.append(((closest, layer), (point, layer), distance, 0))
                if distance:
                    proof = _track_via_annular_contact(
                        a, b, track.GetWidth(), point,
                        _via_width(via, layer), via.GetDrillValue(),
                    )
                    if proof is not None:
                        contact, width = proof
                        cuts[i].add(contact)
                        contacts.append((
                            (contact, layer), (point, layer),
                            math.dist(contact, point), width,
                        ))
        for index, (_, _, pad) in enumerate(pads):
            if not pad.IsOnLayer(layer):
                continue
            key = (index, layer)
            if key not in pad_shapes:
                pad_shapes[key] = _pad_shapes(pad, layer)
            shape, hole = pad_shapes[key]
            boundaries = _shape_boundaries(shape, a, b)
            if hole:
                boundaries.extend(_shape_boundaries(hole, a, b))
            cuts[i].update(boundaries)
            for point in boundaries:
                if (shape.Collide(_point(*point), 2)
                        and not (hole and hole.Collide(_point(*point), -2))):
                    pad_contacts.setdefault(key, set()).add(point)
            # Preserve finite-width side contacts without considering the
            # intervening centreline to be covered by pad copper.
            closest = _project(a, b, _position(pad.GetPosition()))
            for point in (a, b, closest):
                if _pad_contains(shape, hole, point, track.GetWidth() // 2):
                    cuts[i].add(point)
                    pad_contacts.setdefault(key, set()).add(point)
    for index, (_, _, pad) in enumerate(pads):
        for via in vias:
            point = _position(via.GetPosition())
            for layer in (pcbnew.F_Cu, pcbnew.B_Cu):
                if pad.IsOnLayer(layer) and via.IsOnLayer(layer):
                    key = (index, layer)
                    if key not in pad_shapes:
                        pad_shapes[key] = _pad_shapes(pad, layer)
                    shape, hole = pad_shapes[key]
                    if _pad_contains(shape, hole, point, _via_width(via, layer) // 2):
                        pad_contacts.setdefault(key, set()).add(point)
    graph = {}
    def connect(a, b, length, width):
        graph.setdefault(a, []).append((b, length, width))
        graph.setdefault(b, []).append((a, length, width))
    for (track, a, b, layer), points in zip(tracks, cuts):
        ordered = sorted(points, key=lambda p: _along(a, b, p))
        for left, right in zip(ordered, ordered[1:]):
            midpoint = ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
            p = _point(*midpoint)
            if any(hole and hole.Collide(p, -2)
                   for (index, pad_layer), (_, hole) in pad_shapes.items()
                   if pad_layer == layer):
                continue
            covered = any(shape.Collide(p, 0) and not (hole and hole.Collide(p, 0))
                          for (index, pad_layer), (shape, hole) in pad_shapes.items()
                          if pad_layer == layer)
            length = pcbnew.ToMM(round(math.hypot(right[0] - left[0], right[1] - left[1])))
            # -2 preserves physical length without adding exposed neck length.
            width = -2.0 if covered else pcbnew.ToMM(track.GetWidth())
            connect((left, layer), (right, layer), length, width)
    for first, second, distance, width in contacts:
        if first != second:
            connect(first, second, pcbnew.ToMM(round(distance)), pcbnew.ToMM(width))
    nodes = list(graph)
    for via in vias:
        pos = _position(via.GetPosition())
        layers = [layer for layer in (pcbnew.F_Cu, pcbnew.B_Cu) if via.IsOnLayer(layer)]
        for layer in layers[1:]:
            # A plated via changes layers but does not terminate a neckdown.
            connect((pos, layers[0]), (pos, layer), 0.0, -1.0)
    nodes = list(graph)
    by_pin = {}
    pad_nodes = []
    for index, (ref, number, pad) in enumerate(pads):
        virtual = ("pad", index)
        by_pin.setdefault((ref, number), []).append(virtual)
        pad_nodes.append(virtual)
        # The virtual node joins the actual copper layers only for PTH pads.
        # SMD layers remain separate even if an unusual layer set lists two.
        layer_nodes = {}
        for n in nodes:
            point, layer = n
            if not pad.IsOnLayer(layer):
                continue
            key = (index, layer)
            if key not in pad_shapes:
                pad_shapes[key] = _pad_shapes(pad, layer)
            shape, hole = pad_shapes[key]
            if (_pad_contains(shape, hole, point)
                    or point in pad_contacts.get(key, ())):
                layer_nodes.setdefault(layer, []).append(n)
        center = _position(pad.GetPosition())
        def connect_pad(node, physical):
            point, layer = physical
            shape, hole = pad_shapes[(index, layer)]
            length = pcbnew.ToMM(round(math.dist(center, point)))
            if (shape.Collide(_point(*point), 0)
                    and not (hole and hole.Collide(_point(*point), 0))):
                connect(node, physical, length, -2.0)
            else:
                # Split finite side contacts at the actual pad boundary.  The
                # outside portion has no proven current width; pad copper is
                # still counted in physical length but not exposed neck length.
                boundaries = _shape_boundaries(shape, center, point)
                if boundaries and not shape.Collide(_point(*point), 0):
                    boundary = max(boundaries, key=lambda p: math.dist(center, p))
                    covered = pcbnew.ToMM(round(math.dist(center, boundary)))
                    outside = pcbnew.ToMM(round(math.dist(boundary, point)))
                    contact = ("pad-contact", index, layer, point)
                    connect(node, contact, covered, -2.0)
                    connect(contact, physical, outside, 0.0)
                else:
                    connect(node, physical, length, 0.0)
        if pad.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
            for matching in layer_nodes.values():
                for n in matching:
                    connect_pad(virtual, n)
        else:
            for layer, matching in layer_nodes.items():
                layer_virtual = ("pad", index, layer)
                by_pin[(ref, number)].append(layer_virtual)
                pad_nodes.append(layer_virtual)
                for n in matching:
                    connect_pad(layer_virtual, n)
    return graph, by_pin, pad_nodes


def _narrow_pad_distances(graph, pad_nodes, min_width):
    """Shortest distances from pads using narrow copper and zero-length links."""
    serial = count()
    distances = {p: 0.0 for p in pad_nodes}
    queue = [(0.0, next(serial), p) for p in pad_nodes]
    heapq.heapify(queue)
    while queue:
        distance, _, node = heapq.heappop(queue)
        if distance != distances[node]:
            continue
        for other, length, width in graph.get(node, ()):
            if width >= min_width and length > 0:
                continue
            candidate = distance + (0.0 if width == -2.0 else length)
            if candidate < distances.get(other, math.inf):
                distances[other] = candidate
                heapq.heappush(queue, (candidate, next(serial), other))
    return distances


def _shortest_qualified(graph, sources, targets, min_width, pad_nodes):
    """Find shortest path while bounding every continuous narrow pad escape."""
    serial = count()
    queue = [(0.0, next(serial), source, 0.0) for source in sources]
    heapq.heapify(queue)
    labels = {}
    targets = set(targets)
    pads = set(pad_nodes)
    pad_distances = _narrow_pad_distances(graph, pad_nodes, min_width)
    while queue:
        length, _, node, narrow = heapq.heappop(queue)
        if node in pads:
            narrow = 0.0
        prior = labels.setdefault(node, [])
        if any(old_length <= length + 1e-9 and old_narrow <= narrow + 1e-9
               for old_length, old_narrow in prior):
            continue
        prior[:] = [(l, n) for l, n in prior if not (length <= l + 1e-9 and narrow <= n + 1e-9)]
        prior.append((length, narrow))
        if node in targets:
            return length
        for other, distance, width in graph.get(node, ()):
            if (distance > 0 and width >= 0 and width + 1e-6 < min_width and
                    (pad_distances.get(node, math.inf) > 1.5 + 1e-6
                     or pad_distances.get(other, math.inf) > 1.5 + 1e-6)):
                continue
            run = (narrow if width < 0 else
                   narrow + distance if width + 1e-6 < min_width else 0.0)
            if run <= 1.5 + 1e-6:
                heapq.heappush(queue, (length + distance, next(serial), other, run))
    return None


def _unqualified_narrow_run(graph, pad_nodes, min_width):
    """Reject unanchored or cumulatively overlength continuous neckdowns."""
    distances = _narrow_pad_distances(graph, pad_nodes, min_width)
    narrow_edges = [(node, other, length) for node, edges in graph.items()
                    for other, length, width in edges
                    if length > 0 and 0 <= width + 1e-6 < min_width]
    if any(distances.get(a, math.inf) > 1.5 + 1e-6
           or distances.get(b, math.inf) > 1.5 + 1e-6 for a, b, _ in narrow_edges):
        return True
    pads = set(pad_nodes)
    serial = count()
    for source in pads:
        best = {source: 0.0}
        queue = [(0.0, next(serial), source)]
        while queue:
            distance, _, node = heapq.heappop(queue)
            if distance != best[node]:
                continue
            if distance > 1.5 + 1e-6:
                return True
            # Reaching another actual pad ends this neckdown.
            if node != source and node in pads:
                continue
            for other, length, width in graph.get(node, ()):
                if width >= min_width and length > 0:
                    continue
                candidate = distance + (0.0 if width == -2.0 else length)
                if candidate < best.get(other, math.inf):
                    best[other] = candidate
                    heapq.heappush(queue, (candidate, next(serial), other))
    return False


def _ground_regions(board, net):
    """Native filled islands, kept separate even when they share a net name."""
    regions = []
    for zone in board.Zones():
        if zone.GetNetname() != net or zone.GetIsRuleArea():
            continue
        for layer in range(pcbnew.B_Cu + 1):
            if not zone.IsOnLayer(layer):
                continue
            fill = zone.GetFilledPolysList(layer)
            for index in range(fill.OutlineCount()):
                regions.append((layer, fill.UnitSet(index)))
    return regions


def _filled_straight_bond(region, start, end):
    """Prove a narrow straight copper corridor, including any polygon holes."""
    if not region.Contains(_point(*start)) or not region.Contains(_point(*end)):
        return False
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = math.hypot(dx, dy)
    if not length:
        return True
    half_width = pcbnew.FromMM(0.001)
    ox, oy = -dy * half_width / length, dx * half_width / length
    corridor = pcbnew.SHAPE_POLY_SET()
    corridor.NewOutline()
    for x, y in ((start[0] + ox, start[1] + oy), (end[0] + ox, end[1] + oy),
                 (end[0] - ox, end[1] - oy), (start[0] - ox, start[1] - oy)):
        corridor.Append(round(x), round(y))
    corridor.BooleanSubtract(region)
    return not corridor.OutlineCount() or abs(corridor.Area()) == 0


def _ground_pad_vias(board, net, pad, pin, max_via, regions, graph_cache):
    """Return (via, opposite regions, filled landing regions needing protection)."""
    layers = [layer for layer in range(pcbnew.B_Cu + 1) if pad.IsOnLayer(layer)]
    origin = _position(pad.GetPosition())
    result = []
    for via in board.GetTracks():
        if not isinstance(via, pcbnew.PCB_VIA) or via.GetNetname() != net:
            continue
        point = _position(via.GetPosition())
        if pcbnew.ToMM(round(math.dist(origin, point))) > max_via + 1e-6:
            continue
        landing_regions = {
            index for index, (layer, region) in enumerate(regions)
            if layer in layers and via.IsOnLayer(layer)
            and _filled_straight_bond(region, origin, point)
        }
        if net not in graph_cache:
            graph_cache[net] = _copper_graph(board, net)
        graph, pins, pad_nodes = graph_cache[net]
        targets = [(point, layer) for layer in layers if via.IsOnLayer(layer)]
        length = _shortest_qualified(graph, pins.get(pin, ()), targets, 0.0, pad_nodes)
        fixed_bond = length is not None and length <= max_via + 1e-6
        if not fixed_bond and not landing_regions:
            continue
        if fixed_bond:
            landing_regions = set()
        bonds = {
            index for index, (layer, region) in enumerate(regions)
            if layer not in layers and via.IsOnLayer(layer) and region.Contains(_point(*point))
        }
        if bonds:
            result.append((via, bonds, landing_regions))
    # Native serialization can reorder identical copper by newly generated UUID.
    # Choose physical access by geometry, never by that incidental item order.
    result.sort(key=lambda row: (
        _position(row[0].GetPosition()), row[0].GetDrillValue(),
        tuple((layer, _via_width(row[0], layer))
              for layer in (pcbnew.F_Cu, pcbnew.B_Cu) if row[0].IsOnLayer(layer)),
    ))
    return result


def _ground_pad_regions(board, net, pad, pin, max_via, regions, graph_cache, *, vias):
    """Regions bonded directly, or through a short actual pad-to-via path."""
    result = set()
    layers = [layer for layer in range(pcbnew.B_Cu + 1) if pad.IsOnLayer(layer)]
    for layer in layers:
        shape = pcbnew.SHAPE_POLY_SET()
        pad.TransformShapeToPolygon(shape, layer, 0, pcbnew.FromMM(0.001), pcbnew.ERROR_INSIDE)
        for index, (region_layer, region) in enumerate(regions):
            if region_layer != layer:
                continue
            overlap = pcbnew.SHAPE_POLY_SET(region)
            overlap.BooleanIntersection(shape)
            if overlap.OutlineCount() and abs(overlap.Area()) > 0:
                result.add(index)
    if vias and any(layer not in layers for layer, _ in regions):
        for _, bonds, _ in _ground_pad_vias(board, net, pad, pin, max_via, regions, graph_cache):
            result.update(bonds)
    return result


def _audit_ground_return(board, spec, get_pad, regions_cache, graph_cache):
    contract = spec["ground_return"]
    if not isinstance(contract, dict):
        raise ValueError("invalid ground return contract")
    net = contract["net"]
    max_via = float(contract["max_via_distance_mm"])
    if not isinstance(net, str) or not net or not math.isfinite(max_via) or max_via <= 0:
        raise ValueError("invalid ground return net or via distance")
    if any(not isinstance(contract.get(key), str) or not contract[key].strip()
           for key in ("owner_ref", "owner_pin", "member_pin")):
        raise ValueError("ground return requires explicit owner ref and pins")
    owner = (contract["owner_ref"], contract["owner_pin"])
    member = (spec["bypass_ref"], str(contract["member_pin"]))
    pads = [get_pad(*owner), get_pad(*member)]
    native_net = board.GetNetInfo().GetNetItem(net)
    if (not native_net or native_net.GetNetCode() == 0
            or any(p.GetNetCode() != native_net.GetNetCode() for p in pads)):
        raise ValueError(f"{owner}/{member}: ground return net mismatch")
    if net not in regions_cache:
        regions_cache[net] = _ground_regions(board, net)
    regions = regions_cache[net]
    bonds = [_ground_pad_regions(board, net, p, pin, max_via, regions, graph_cache, vias=False)
             for p, pin in zip(pads, (owner, member))]
    if bonds[0].intersection(bonds[1]):
        return
    bonds = [_ground_pad_regions(board, net, p, pin, max_via, regions, graph_cache, vias=True)
             for p, pin in zip(pads, (owner, member))]
    if not bonds[0].intersection(bonds[1]):
        raise ValueError(f"{owner}/{member}: no shared filled ground region with local copper bonds")


def audit_local_copper(board, specs):
    """Audit real routed geometry; return human-readable fabrication violations."""
    cache, regions_cache, errors = {}, {}, []
    def pad(ref, number):
        fp = board.FindFootprintByReference(ref)
        matches = [] if fp is None else [p for p in fp.Pads() if p.GetNumber() == str(number)]
        if len(matches) != 1:
            raise ValueError(f"local contract {ref}.{number}: missing or ambiguous native pad")
        return matches[0]
    for spec in specs:
        if "ground_return" in spec:
            try:
                _audit_ground_return(board, spec, pad, regions_cache, cache)
            except (KeyError, TypeError, ValueError, RuntimeError, AttributeError) as exc:
                errors.append(f"local ground return: {exc}")
        if "max_route_length_mm" in spec:
            pairs = spec.get("pin_pairs")
            if pairs is None:
                pairs = [spec["power_pins"]]
            for owner_pin, dest_pin in pairs:
                ref = spec.get("route_anchor_ref", spec["owner_ref"])
                pin = spec.get("route_anchor_pin", owner_pin)
                dest_ref = spec["bypass_ref"]
                source, target = pad(ref, pin), pad(dest_ref, dest_pin)
                net = source.GetNetname()
                if not net or target.GetNetname() != net:
                    errors.append(f"local contract {ref}.{pin}/{dest_ref}.{dest_pin}: current-path net mismatch")
                    continue
                if net not in cache:
                    cache[net] = _copper_graph(board, net)
                graph, pins, pads = cache[net]
                length = _shortest_qualified(graph, pins.get((ref, str(pin)), ()),
                                             pins.get((dest_ref, str(dest_pin)), ()),
                                             float(spec["min_route_width_mm"]), pads)
                if length is None:
                    errors.append(f"local contract {ref}.{pin}/{dest_ref}.{dest_pin}: no qualified current-carrying copper path")
                elif length > float(spec["max_route_length_mm"]) + 1e-6:
                    errors.append(f"local contract {ref}.{pin}/{dest_ref}.{dest_pin}: copper path {length:.3f} mm exceeds "
                                  f"{float(spec['max_route_length_mm']):.3f} mm")
        net = spec.get("current_carrying_net")
        if net:
            if net not in cache:
                cache[net] = _copper_graph(board, net)
            graph, _, pads = cache[net]
            if _unqualified_narrow_run(graph, pads, 0.5):
                errors.append(f"current net {net}: narrow copper exceeds 1.5 mm pad escape")
    return errors
