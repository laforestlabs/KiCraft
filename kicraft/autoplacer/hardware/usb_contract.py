"""Fail-closed native audit of promoted USB pair copper.

Only a fabricator-approved geometry tied to the *actual* board stackup can
establish impedance.  This module deliberately does not treat routing settings
or a nominal dielectric constant as such an approval.  Geometries outside the
exact straight, parallel primitive proof below are reported as unsupported, not
accepted on the strength of a sampled centreline.
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
from pathlib import Path

import pcbnew

from kicraft.autoplacer.hardware.local_contract import (
    _closest_segment_points, _copper_graph, _crossings, _ground_regions,
    _position, _via_width,
)
from kicraft.autoplacer.kicad_routing_tools import (
    _usb_stackup, calculate_usb_review_geometry, preflight_kicad_routing_tools,
)


def _as_dict(value):
    return value.model_dump() if hasattr(value, "model_dump") else value




def _shortest(graph, sources, targets):
    targets = set(targets)
    heap, distance, previous = [], {}, {}
    serial = 0
    for source in sources:
        distance[source] = 0.0
        heapq.heappush(heap, (0.0, serial, source))
        serial += 1
    while heap:
        length, _, node = heapq.heappop(heap)
        if length != distance[node]:
            continue
        if node in targets:
            path = []
            while node in previous:
                parent, edge_length, width = previous[node]
                path.append((parent, node, edge_length, width))
                node = parent
            return length, list(reversed(path))
        for neighbor, edge_length, width in graph.get(node, ()):
            candidate = length + edge_length
            if candidate < distance.get(neighbor, math.inf):
                distance[neighbor] = candidate
                previous[neighbor] = (node, edge_length, width)
                heapq.heappush(heap, (candidate, serial, neighbor))
                serial += 1
    return None


def _physical(node):
    return (isinstance(node, tuple) and len(node) == 2
            and isinstance(node[0], tuple) and len(node[0]) == 2
            and isinstance(node[1], int))


def _path_geometry(path):
    tracks, transitions, errors = [], [], []
    last_layer = None
    last_point = None
    for a, b, length, width in path:
        for node in (a, b):
            if not _physical(node):
                continue
            point, layer = node
            if last_layer is not None and layer != last_layer:
                transitions.append((last_point, point, last_layer, layer))
            last_layer, last_point = layer, point
        if _physical(a) and _physical(b) and a[1] == b[1] and length:
            if width == -2.0:
                continue
            if width <= 0:
                errors.append("route includes an unproved finite copper contact")
                continue
            tracks.append((a[0], b[0], a[1], pcbnew.FromMM(width)))
    return tracks, transitions, errors


def _endpoint_nodes(board, pins, endpoint, net):
    endpoint = _as_dict(endpoint)
    ref, pin = endpoint["ref"], str(endpoint["pin"])
    fp = board.FindFootprintByReference(ref)
    matches = [p for p in fp.Pads() if str(p.GetNumber()) == pin] if fp else []
    if len(matches) != 1 or matches[0].GetNetname() != net:
        raise ValueError(f"{ref}.{pin}: missing, ambiguous, or wrong polarity/net (expected {net})")
    nodes = pins.get((ref, pin), [])
    if not nodes:
        raise ValueError(f"{ref}.{pin}: no copper graph node")
    return nodes


def _segment_path(board, segment, polarity, cache):
    net = segment[f"{polarity}_net"]
    if net not in cache:
        cache[net] = _copper_graph(board, net)
    graph, pins, _ = cache[net]
    starts = segment[f"{polarity}_from"]
    ends = segment[f"{polarity}_to"]
    if not starts or not ends:
        raise ValueError(f"{net}: empty endpoint group")
    # Every declared terminal must be connected; path metrics use the worst
    # endpoint pair, never an arbitrary convenient connector contact.
    rows = []
    for start in starts:
        sources = _endpoint_nodes(board, pins, start, net)
        for end in ends:
            targets = _endpoint_nodes(board, pins, end, net)
            result = _shortest(graph, sources, targets)
            if result is None:
                raise ValueError(f"{net}: expected terminals are not copper-connected")
            rows.append(result)
    return max(rows, key=lambda row: row[0])


def _orientation_lengths(board, segments, cache, boundaries):
    """Measure corresponding ordered connector orientations, never mixed aliases."""
    groups = [len(_as_dict(segment)[f"{polarity}_{side}"])
              for segment in segments for polarity in ("positive", "negative")
              for side in ("from", "to")]
    count = max(groups, default=1)
    if any(size not in (1, count) for size in groups):
        raise ValueError("USB orientation endpoint groups have incompatible sizes")
    results = []
    for orientation in range(count):
        lengths = {polarity: sum(
            boundary[polarity]["pad_center_distance_mm"] for boundary in boundaries)
            for polarity in ("positive", "negative")}
        for raw_segment in segments:
            segment = _as_dict(raw_segment)
            for polarity in ("positive", "negative"):
                net = segment[f"{polarity}_net"]
                graph, pins, _ = cache[net]
                start = segment[f"{polarity}_from"]
                end = segment[f"{polarity}_to"]
                start = start[orientation if len(start) > 1 else 0]
                end = end[orientation if len(end) > 1 else 0]
                path = _shortest(graph, _endpoint_nodes(board, pins, start, net),
                                 _endpoint_nodes(board, pins, end, net))
                if path is None:
                    raise ValueError(f"{net}: orientation {orientation} is not copper-connected")
                lengths[polarity] += path[0]
        results.append(lengths)
    return results


def _component_boundary_length(board, first, second):
    """Pad-centre contribution across a producer-reviewed series/ESD part."""
    if len(first) != 1 or len(second) != 1:
        raise ValueError("component chain boundary requires exactly one terminal per side")
    a, b = _as_dict(first[0]), _as_dict(second[0])
    if a["ref"] != b["ref"] or a["pin"] == b["pin"]:
        raise ValueError("component chain boundary does not cross distinct pins of one reviewed component")
    fp = board.FindFootprintByReference(a["ref"])
    if fp is None:
        raise ValueError(f"component chain boundary footprint {a['ref']} is missing")
    pads = []
    for endpoint in (a, b):
        matches = [pad for pad in fp.Pads() if str(pad.GetNumber()) == str(endpoint["pin"])]
        if len(matches) != 1:
            raise ValueError(f"component chain boundary {a['ref']}.{endpoint['pin']} is missing or ambiguous")
        pads.append(_position(matches[0].GetPosition()))
    return pcbnew.ToMM(round(math.dist(*pads)))


def _corridor(region, corners):
    polygon = pcbnew.SHAPE_POLY_SET()
    polygon.NewOutline()
    for x, y in corners:
        polygon.Append(round(x), round(y))
    polygon.BooleanSubtract(region)
    return not polygon.OutlineCount() or abs(polygon.Area()) == 0


def _paired_geometry(positive, negative, regions, approval):
    """Cover each native straight by parallel counterpart intervals.

    Different segment splits and straight/45-degree bends are supported.
    Short end pieces are limited to two approved pitches, not an arbitrary
    route-length percentage. Reference copper covers both tracks and their
    intervening corridor, including bend/escape neighborhoods.
    """
    if not positive or not negative:
        raise ValueError("pair corridor has missing straight spans")
    geometries = approval["geometry_by_layer"]
    coverage = [[[] for _ in positive], [[] for _ in negative]]
    measurements = []
    common_regions = {}
    for group in (positive, negative):
        for a, b, layer, actual_width in group:
            geometry = geometries["F.Cu" if layer == pcbnew.F_Cu else "B.Cu"]
            width = pcbnew.FromMM(float(geometry["width_mm"]))
            gap = pcbnew.FromMM(float(geometry["gap_mm"]))
            if actual_width != width:
                raise ValueError("actual track width differs from approved impedance geometry")
            vx, vy = b[0]-a[0], b[1]-a[1]
            length = math.hypot(vx, vy)
            if not length:
                raise ValueError("zero-length pair track")
            # Cover the full pair and a conservative bend neighborhood.
            half = 1.5 * width + gap
            nx, ny = -vy * half / length, vx * half / length
            corners = [(a[0]-nx,a[1]-ny), (b[0]-nx,b[1]-ny),
                       (b[0]+nx,b[1]+ny), (a[0]+nx,a[1]+ny)]
            opposite = pcbnew.B_Cu if layer == pcbnew.F_Cu else pcbnew.F_Cu
            matching = {i for i, (rl, region) in enumerate(regions)
                        if rl == opposite and _corridor(region, corners)}
            if not matching:
                raise ValueError("reference plane has a gap, void, hole, or missing filled island under pair")
            common_regions[opposite] = common_regions.get(opposite, matching) & matching
            if not common_regions[opposite]:
                raise ValueError("pair spans do not share one continuous reference island")
    for i, (a, b, layer, width) in enumerate(positive):
        vx, vy = b[0]-a[0], b[1]-a[1]
        length = math.hypot(vx, vy)
        gap = pcbnew.FromMM(float(geometries["F.Cu" if layer == pcbnew.F_Cu else "B.Cu"]["gap_mm"]))
        for j, (c, d, nl, nw) in enumerate(negative):
            if layer != nl:
                continue
            # Check actual nearest copper even at bends/escapes; parallel
            # coverage alone cannot detect a close approach or crossover.
            if (max(a[0], b[0]) + width + gap >= min(c[0], d[0]) and
                max(c[0], d[0]) + width + gap >= min(a[0], b[0]) and
                max(a[1], b[1]) + width + gap >= min(c[1], d[1]) and
                max(c[1], d[1]) + width + gap >= min(a[1], b[1])):
                left_point, right_point = _closest_segment_points(a, b, c, d)
                if _crossings(a, b, c, d) or math.dist(left_point, right_point) + 2 < (width + nw) / 2 + gap:
                    raise ValueError("pair copper violates approved gap at bend or escape")
            wx, wy = d[0]-c[0], d[1]-c[1]
            other_length = math.hypot(wx, wy)
            if vx * wy != vy * wx or vx * wx + vy * wy <= 0:
                continue
            offset = abs(vx * (c[1]-a[1]) - vy * (c[0]-a[0])) / length
            if abs(offset - (width + gap)) > 1.5:
                continue
            start = ((c[0]-a[0])*vx + (c[1]-a[1])*vy) / length
            end = ((d[0]-a[0])*vx + (d[1]-a[1])*vy) / length
            left, right = max(0.0, start), min(length, end)
            if right <= left:
                continue
            coverage[0][i].append((left, right))
            coverage[1][j].append(((left-start)*other_length/(end-start),
                                    (right-start)*other_length/(end-start)))
            measurements.append({"layer": layer, "width_mm": pcbnew.ToMM(width),
                                 "gap_mm": pcbnew.ToMM(round(offset-width)),
                                 "length_mm": pcbnew.ToMM(round(right-left))})
    if not measurements:
        raise ValueError("no parallel pair corridor at approved gap")
    for index, group in enumerate((positive, negative)):
        for piece, ((a, b, layer, width), intervals) in enumerate(zip(group, coverage[index])):
            length = math.dist(a, b)
            gap = pcbnew.FromMM(float(geometries["F.Cu" if layer == pcbnew.F_Cu else "B.Cu"]["gap_mm"]))
            bound = 2 * (width + gap)
            intervals.sort()
            merged = []
            for left, right in intervals:
                if merged and left <= merged[-1][1] + 2:
                    merged[-1] = (merged[-1][0], max(merged[-1][1], right))
                else:
                    merged.append((left, right))
            if not merged:
                if (length > bound or
                    (piece > 0 and not coverage[index][piece-1]) or
                    (piece+1 < len(group) and not coverage[index][piece+1])):
                    raise ValueError("unpaired span exceeds two approved pair pitches")
            elif (merged[0][0] > bound or length-merged[-1][1] > bound
                  or any(right[0]-left[1] > 2 for left, right in zip(merged, merged[1:]))):
                raise ValueError("unpaired span or internal corridor gap exceeds bounded escape")
    return measurements, common_regions


def _return_vias(board, transitions, regions, reference_net, approval, common_reference):
    limit = float(approval.get("max_return_via_distance_mm", 0))
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("grounded return transition requires an approved maximum via distance")
    grounded = [item for item in board.GetTracks()
                if isinstance(item, pcbnew.PCB_VIA) and item.GetNetname() == reference_net]
    results = []
    for before, after, first_layer, second_layer in transitions:
        if before != after:
            raise ValueError("signal layer transition is not a proved colocated via")
        found = None
        for via in grounded:
            point = _position(via.GetPosition())
            distance = pcbnew.ToMM(round(math.dist(before, point)))
            if distance > limit or not via.IsOnLayer(first_layer) or not via.IsOnLayer(second_layer):
                continue
            def bonded(wanted):
                radius = _via_width(via, wanted) / 2
                x, y = point
                square = [(x-radius, y-radius), (x+radius, y-radius),
                          (x+radius, y+radius), (x-radius, y+radius)]
                return any(i in common_reference.get(wanted, set()) and
                           layer == wanted and _corridor(region, square)
                           for i, (layer, region) in enumerate(regions))
            if all(bonded(wanted) for wanted in (first_layer, second_layer)):
                found = {"signal_position_mm": [pcbnew.ToMM(x) for x in before],
                         "return_via_position_mm": [pcbnew.ToMM(x) for x in point],
                         "distance_mm": distance}
                break
        if found is None:
            raise ValueError("no nearby grounded via bonded to filled reference on both transition layers")
        results.append(found)
    return results


def _bounded_distances(graph, sources, width):
    """Distances from a declared pad without traversing main-profile copper."""
    heap, distances = [], {}
    serial = 0
    for node in sources:
        distances[node] = 0.0
        heapq.heappush(heap, (0.0, serial, node))
        serial += 1
    while heap:
        distance, _, node = heapq.heappop(heap)
        if distance != distances[node]:
            continue
        for other, length, edge_width in graph.get(node, ()):
            if edge_width != -2.0 and not math.isclose(edge_width, width, abs_tol=1e-6):
                continue
            candidate = distance + length
            if candidate < distances.get(other, math.inf):
                distances[other] = candidate
                heapq.heappush(heap, (candidate, serial, other))
                serial += 1
    return distances

def _contact_join_geometry(board, segment, polarity, side, gap, width):
    """Independently derive the envelope of declared native contact copper."""
    group = [_as_dict(item) for item in segment[f"{polarity}_{side}"]]
    if len(group) < 2:
        return None
    ref = group[0]["ref"]
    if any(item["ref"] != ref for item in group):
        raise ValueError("terminal aliases must belong to the same component")
    fp = board.FindFootprintByReference(ref)
    pads = []
    for endpoint in group:
        matches = [pad for pad in fp.Pads()
                   if str(pad.GetNumber()) == str(endpoint["pin"])] if fp else []
        if len(matches) != 1 or matches[0].GetNetname() != segment[f"{polarity}_net"]:
            raise ValueError(f"{ref}.{endpoint['pin']}: missing, ambiguous, or wrong polarity/net")
        pad = matches[0]
        if pad.GetShape() == pcbnew.PAD_SHAPE_CUSTOM:
            raise ValueError("custom alias contact pad copper is unsupported")
        offset = pad.GetOffset()
        if offset.x or offset.y:
            raise ValueError("offset alias contact pad copper is unsupported")
        degrees = float(pad.GetOrientationDegrees())
        nearest = round(degrees / 90) * 90
        if 1e-9 < abs(degrees - nearest) < 1:
            raise ValueError("near-orthogonal alias contact pad orientation is unsupported")
        center = tuple(pcbnew.ToMM(v) for v in _position(pad.GetPosition()))
        size = pad.GetSize()
        hx, hy = pcbnew.ToMM(size.x) / 2, pcbnew.ToMM(size.y) / 2
        angle = math.radians(float(pad.GetOrientationDegrees()))
        cosine, sine = math.cos(angle), math.sin(angle)
        offsets = [(x * cosine + y * sine, -x * sine + y * cosine)
                   for x in (-hx, hx) for y in (-hy, hy)]
        pads.append((center, offsets))
    corners = [(center[0] + x, center[1] + y)
               for center, offsets in pads for x, y in offsets]
    margin = gap + width
    envelope = [min(p[0] for p in corners) - margin, min(p[1] for p in corners) - margin,
                max(p[0] for p in corners) + margin, max(p[1] for p in corners) + margin]
    bound = 0.0
    for i, (a, _) in enumerate(pads):
        for b, _ in pads[i + 1:]:
            dx, dy = b[0] - a[0], b[1] - a[1]
            distance = math.hypot(dx, dy)
            if not distance:
                raise ValueError("coincident alias contact centers are unsupported")
            nx, ny = -dy / distance, dx / distance
            half = max(abs(x * nx + y * ny)
                       for _, offsets in pads for x, y in offsets)
            bound = max(bound, distance + 2 * (half + margin))
    return {"segment_id": segment.get("id"), "polarity": polarity, "side": side,
            "ref": ref, "pins": [str(item["pin"]) for item in group],
            "max_length_mm": bound, "envelope_mm": envelope}


def _inside_contact_envelope(point, envelope):
    x, y = (pcbnew.ToMM(v) for v in point)
    return (envelope[0] - 1e-6 <= x <= envelope[2] + 1e-6
            and envelope[1] - 1e-6 <= y <= envelope[3] + 1e-6)


def _unclassified_review_spans(board, segment, geometry):
    """Retain physical evidence even when a subsequent proof fails."""
    spans = []
    for polarity in ("positive", "negative"):
        for track in board.GetTracks():
            if (track.GetNetname() != segment[f"{polarity}_net"]
                    or isinstance(track, pcbnew.PCB_VIA)
                    or track.GetClass() != "PCB_TRACK"):
                continue
            layer = track.GetLayer()
            name = "F.Cu" if layer == pcbnew.F_Cu else "B.Cu" if layer == pcbnew.B_Cu else None
            if name is None:
                continue
            if track.GetWidth() == pcbnew.FromMM(float(geometry["geometry_by_layer"][name]["width_mm"])):
                continue
            a, b = _position(track.GetStart()), _position(track.GetEnd())
            spans.append({"kind": "unclassified_uncontrolled", "net": track.GetNetname(),
                          "polarity": polarity, "endpoints": [[pcbnew.ToMM(v) for v in a],
                                                               [pcbnew.ToMM(v) for v in b]],
                          "length_mm": pcbnew.ToMM(round(math.dist(a, b))),
                          "width_mm": pcbnew.ToMM(track.GetWidth()), "layer": name,
                          "impedance_qualified": False})
    return spans


def _bounded_review_segment(board, segment, cache, regions, geometry):
    """Classify every native track, not only the shortest terminal paths."""
    policy = geometry["launch_policy"]
    if policy.get("mode") != "bounded_uncontrolled" or policy.get("provenance") != "uncontrolled_launch":
        raise ValueError("invalid bounded launch policy provenance")
    width, gap, bound = (float(policy[key]) for key in ("width_mm", "gap_mm", "max_length_mm"))
    if not all(math.isfinite(value) and value > 0 for value in (width, gap, bound)):
        raise ValueError("invalid bounded launch dimensions")
    for dimensions in geometry["geometry_by_layer"].values():
        if bound > 2 * (float(dimensions["width_mm"]) + float(dimensions["gap_mm"])) + 1e-6:
            raise ValueError("launch bound exceeds two main pair pitches")
        if gap + 1e-6 < float(dimensions["gap_mm"]):
            raise ValueError("launch gap below required pair clearance")
    main, narrow, uncontrolled, joins = {}, {}, [], []
    matching = {}
    for polarity in ("positive", "negative"):
        net = segment[f"{polarity}_net"]
        graph, pins, _ = cache[net]
        endpoints = []
        start_nodes = []
        declared = set()
        for side in ("from", "to"):
            group = [_as_dict(item) for item in segment[f"{polarity}_{side}"]]
            if len({item["ref"] for item in group}) != 1:
                raise ValueError("terminal aliases must belong to the same component")
            for endpoint in group:
                declared.add((endpoint["ref"], str(endpoint["pin"])))
                nodes = _endpoint_nodes(board, pins, endpoint, net)
                if side == "from":
                    start_nodes.extend(nodes)
                endpoints.append((side, endpoint, _bounded_distances(graph, nodes, width)))
            contact = _contact_join_geometry(board, segment, polarity, side, gap, width)
            if contact:
                claimed = policy.get("contact_join_bounds")
                if claimed is not None:
                    matching_claims = [item for item in claimed
                                       if all(item.get(key) == contact[key] for key in
                                              ("segment_id", "polarity", "side", "ref", "pins"))]
                    if len(matching_claims) != 1:
                        raise ValueError("contact join profile does not match declared aliases")
                    supplied = matching_claims[0]
                    supplied_envelope = supplied["envelope_mm"]
                    if (len(supplied_envelope) != 4
                            or not math.isclose(float(supplied["max_length_mm"]),
                                                contact["max_length_mm"], rel_tol=0, abs_tol=1e-6)
                            or any(not math.isclose(float(a), b, rel_tol=0, abs_tol=1e-6)
                                   for a, b in zip(supplied_envelope, contact["envelope_mm"]))):
                        raise ValueError("contact join profile disagrees with native pad copper")
                members = [(endpoint, distances) for endpoint_side, endpoint, distances in endpoints
                           if endpoint_side == side]
                restricted = {}
                for node, edges in graph.items():
                    restricted[node] = [
                        (other, length, edge_width) for other, length, edge_width in edges
                        if (edge_width == -2.0 or math.isclose(edge_width, width, abs_tol=1e-6))
                        and (not (_physical(node) and _physical(other) and edge_width > 0)
                             or (_inside_contact_envelope(node[0], contact["envelope_mm"])
                                 and _inside_contact_envelope(other[0], contact["envelope_mm"])))]
                lengths = []
                for index, (first, _) in enumerate(members):
                    for second, _ in members[index + 1:]:
                        path = _shortest(restricted, _endpoint_nodes(board, pins, first, net),
                                         _endpoint_nodes(board, pins, second, net))
                        if path is None or path[0] > contact["max_length_mm"] + 1e-6:
                            raise ValueError(f"{net}: alias contact join exceeds pad-envelope length or envelope bound")
                        lengths.append({"pins": [str(first["pin"]), str(second["pin"])],
                                        "length_mm": path[0]})
                contact["paths"] = lengths
                joins.append(contact)
        if set(pins) != declared:
            raise ValueError(f"{net}: copper contacts undeclared terminals")
        main[polarity], narrow[polarity] = [], []
        for track in board.GetTracks():
            if track.GetNetname() != net or isinstance(track, pcbnew.PCB_VIA):
                continue
            if track.GetClass() != "PCB_TRACK":
                raise ValueError(f"{net}: unsupported non-linear copper")
            a, b = _position(track.GetStart()), _position(track.GetEnd())
            layer, actual = track.GetLayer(), track.GetWidth()
            layer_name = "F.Cu" if layer == pcbnew.F_Cu else "B.Cu" if layer == pcbnew.B_Cu else None
            if layer_name is None:
                raise ValueError("USB track on unsupported layer")
            distance_a = _shortest(graph, start_nodes, [(a, layer)])
            distance_b = _shortest(graph, start_nodes, [(b, layer)])
            if distance_a is None or distance_b is None:
                raise ValueError(f"{net}: disconnected USB copper")
            if distance_a[0] > distance_b[0]:
                a, b = b, a
            profile = geometry["geometry_by_layer"][layer_name]
            span = (a, b, layer, actual)
            if actual == pcbnew.FromMM(float(profile["width_mm"])):
                main[polarity].append(span)
                continue
            if actual != pcbnew.FromMM(width):
                raise ValueError("actual track width differs from main or launch geometry")
            length = pcbnew.ToMM(round(math.dist(a, b)))
            if not length:
                raise ValueError("zero-length launch track")
            candidates = []
            for side, endpoint, distances in endpoints:
                distance = min(distances.get((a, layer), math.inf),
                               distances.get((b, layer), math.inf)) + length
                if distance <= bound + 1e-6:
                    candidates.append((side, endpoint, distance))
            aliases = []
            for contact in joins:
                if contact["polarity"] != polarity:
                    continue
                members = [(endpoint, distances) for side, endpoint, distances in endpoints
                           if side == contact["side"]]
                for index, (_, first_distances) in enumerate(members):
                    for _, second_distances in members[index + 1:]:
                        da = min(first_distances.get((a, layer), math.inf),
                                 first_distances.get((b, layer), math.inf))
                        db = min(second_distances.get((a, layer), math.inf),
                                 second_distances.get((b, layer), math.inf))
                        if (da + db + length <= contact["max_length_mm"] + 1e-6
                                and _inside_contact_envelope(a, contact["envelope_mm"])
                                and _inside_contact_envelope(b, contact["envelope_mm"])):
                            aliases.append(contact)
            if not candidates and not aliases:
                raise ValueError(f"{net}: narrow copper is not a bounded contiguous declared-pad launch or contact join")
            kind = "contact_join" if aliases else "endpoint_launch"
            narrow[polarity].append(span)
            uncontrolled.append({"kind": kind, "net": net, "polarity": polarity,
                                 "endpoints": [[pcbnew.ToMM(v) for v in a],
                                               [pcbnew.ToMM(v) for v in b]],
                                 "length_mm": length, "width_mm": pcbnew.ToMM(actual),
                                 "layer": layer_name, "impedance_qualified": False})
            vx, vy = b[0]-a[0], b[1]-a[1]
            norm = math.hypot(vx, vy)
            half = 1.5 * actual + pcbnew.FromMM(gap)
            nx, ny = -vy * half / norm, vx * half / norm
            corners = [(a[0]-nx, a[1]-ny), (b[0]-nx, b[1]-ny),
                       (b[0]+nx, b[1]+ny), (a[0]+nx, a[1]+ny)]
            opposite = pcbnew.B_Cu if layer == pcbnew.F_Cu else pcbnew.F_Cu
            found = {i for i, (rl, region) in enumerate(regions)
                     if rl == opposite and _corridor(region, corners)}
            matching[opposite] = matching.get(opposite, found) & found
            if not matching[opposite]:
                raise ValueError("reference plane missing or discontinuous under launch")
    # Check all actual copper, including alias branches, for pair clearance.
    for a, b, layer, aw in main["positive"] + narrow["positive"]:
        for c, d, other_layer, bw in main["negative"] + narrow["negative"]:
            if layer != other_layer:
                continue
            left, right = _closest_segment_points(a, b, c, d)
            if _crossings(a, b, c, d) or math.dist(left, right) + 2 < (aw + bw) / 2 + pcbnew.FromMM(gap):
                raise ValueError("pair copper violates gap at launch or contact join")
    paired = []
    if main["positive"] or main["negative"]:
        paired, main_matching = _paired_geometry(main["positive"], main["negative"], regions, geometry)
        for layer, found in main_matching.items():
            matching[layer] = matching.get(layer, found) & found
            if not matching[layer]:
                raise ValueError("main and launch do not share a continuous reference island")
    return paired, matching, uncontrolled, joins


def audit_usb_routing(board, contracts, config=None):
    """Audit promoted native copper; missing fabrication evidence is a failure."""
    config = config or {}
    review = config.get("usb_geometry_mode") == "engineering_review"
    requested_launches = config.get("usb_review_allow_bounded_launches", False)
    valid_launch_config = (isinstance(requested_launches, bool)
                           and (not requested_launches or review))
    bounded = valid_launch_config and requested_launches
    receipt = {"ok": True, "contracts": [], "errors": [],
               "geometry_ok": True, "geometry_errors": [],
               "fabrication_approved": True}
    cache = {}
    for raw in contracts:
        contract = _as_dict(raw)
        name = contract.get("id", "<unnamed>")
        row = {"id": name, "segments": [], "failures": [],
               "geometry_errors": [], "fabrication_approved": False}
        receipt["contracts"].append(row)
        def fail(message, *, geometry=True):
            row["failures"].append(message)
            receipt["errors"].append(f"{name}: {message}")
            if geometry:
                row["geometry_errors"].append(message)
                receipt["geometry_errors"].append(f"{name}: {message}")
        if not valid_launch_config:
            fail("bounded USB launches require a strict boolean and engineering_review mode")
        approval = config.get("usb_impedance_fabrication_approval")
        approval_ok = False
        try:
            if not isinstance(approval, dict) or not isinstance(approval.get("source"), str) or not approval["source"].strip():
                raise ValueError("fabricator-approved impedance geometry is unavailable; two-ounce controlled impedance is unproved")
            if board.GetCopperLayerCount() != 2:
                raise ValueError("only two-layer stackup proof is supported")
            geometry = approval["geometry_by_layer"]
            front, back = geometry["F.Cu"], geometry["B.Cu"]
            width, gap = float(front["width_mm"]), float(front["gap_mm"])
            if not all(math.isfinite(v) and v > 0 for v in (
                width, gap, float(back["width_mm"]), float(back["gap_mm"])
            )):
                raise ValueError("invalid approved impedance geometry")
            if not math.isclose(gap, float(back["gap_mm"]), abs_tol=1e-6):
                raise ValueError("different layer gaps are unsupported by the pinned impedance solver")
            nets = [net for raw_segment in contract["segments"]
                    for net in (_as_dict(raw_segment)["positive_net"], _as_dict(raw_segment)["negative_net"])]
            runtime = preflight_kicad_routing_tools(config)
            result = _usb_stackup(Path(board.GetFileName()), runtime,
                                   {"track_width": width, "clearance": gap},
                                   contract, nets, float(config.get("kicad_routing_tools_timeout_s", 300)))
            actual_hash = hashlib.sha256(json.dumps(
                result["layers"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            if approval.get("stackup_sha256") != actual_hash:
                raise ValueError("approved stackup does not match promoted board stackup")
            for layer, planned in result["planned_geometry"].items():
                if not math.isclose(float(geometry[layer]["width_mm"]), planned["width_mm"], abs_tol=1e-6):
                    raise ValueError(f"fabricator-approved geometry disagrees with impedance solver on {layer}")
            if not review:
                row["stackup_sha256"] = actual_hash
                row["impedance_approval"] = dict(approval)
                row["impedance_solver"] = result
            approval_ok = True
        except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
            fail(str(exc), geometry=not review)
        if review:
            fail("fabrication release blocked: engineering-review geometry is calculated and unconfirmed",
                 geometry=False)
        row["fabrication_approved"] = approval_ok and not review and valid_launch_config
        receipt["fabrication_approved"] = (
            receipt["fabrication_approved"] and row["fabrication_approved"])
        geometry = approval if approval_ok and not review else None
        if review:
            try:
                nets = [net for raw_segment in contract["segments"]
                        for net in (_as_dict(raw_segment)["positive_net"],
                                    _as_dict(raw_segment)["negative_net"])]
                geometry = calculate_usb_review_geometry(Path(board.GetFileName()), config, contract, nets)
                if (geometry.get("provenance") != "calculated_unconfirmed"
                        or geometry.get("fabrication_approved") is not False):
                    raise ValueError("invalid engineering-review geometry provenance")
                row["review_geometry"] = geometry
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
                geometry = None
                fail(f"engineering-review geometry unavailable: {exc}")
        geometry_available = geometry is not None
        lengths = {"positive": 0.0, "negative": 0.0}
        counts = {"positive": 0, "negative": 0}
        length_complete = {"positive": True, "negative": True}
        transitions_complete = {"positive": True, "negative": True}
        segments = contract.get("segments", [])
        common_reference = {}
        pending_transitions = []
        coupled_main = False
        if not segments:
            length_complete = {"positive": False, "negative": False}
            transitions_complete = {"positive": False, "negative": False}
            fail("contract contains no terminal segments")
        regions = _ground_regions(board, contract.get("reference_net", ""))
        row["component_boundaries"] = []
        for left, right in zip(segments, segments[1:]):
            left, right = _as_dict(left), _as_dict(right)
            boundary = {}
            for polarity in ("positive", "negative"):
                try:
                    distance = _component_boundary_length(
                        board, left[f"{polarity}_to"], right[f"{polarity}_from"])
                    lengths[polarity] += distance
                    boundary[polarity] = {"pad_center_distance_mm": distance,
                        "component": _as_dict(left[f"{polarity}_to"][0])["ref"]}
                except (KeyError, TypeError, ValueError) as exc:
                    length_complete[polarity] = False
                    fail(f"component boundary {left.get('id')} to {right.get('id')}/{polarity}: {exc}")
            row["component_boundaries"].append(boundary)
        for raw_segment in segments:
            segment = _as_dict(raw_segment)
            sr = {"id": segment.get("id"), "conductors": {}}
            row["segments"].append(sr)
            if bounded and geometry_available:
                try:
                    sr["uncontrolled_spans"] = _unclassified_review_spans(board, segment, geometry)
                except (KeyError, TypeError, ValueError) as exc:
                    fail(f"{segment.get('id')}: cannot enumerate uncontrolled copper: {exc}")
            tracks, transitions = {}, {}
            for polarity in ("positive", "negative"):
                try:
                    if segment["positive_net"] == segment["negative_net"]:
                        raise ValueError("positive and negative terminals share a net")
                    length, path = _segment_path(board, segment, polarity, cache)
                    lengths[polarity] += length
                    sr["conductors"][polarity] = {"net": segment[f"{polarity}_net"],
                                                   "length_mm": length}
                    path_tracks, via_changes, geometry_errors = _path_geometry(path)
                    counts[polarity] += len(via_changes)
                    transitions[polarity] = via_changes
                    sr["conductors"][polarity]["transitions"] = len(via_changes)
                    if bounded:
                        used_vias = {point for first, second, _, _ in via_changes
                                     for point in (first, second)}
                        if any(_position(via.GetPosition()) not in used_vias
                               for via in board.GetTracks()
                               if isinstance(via, pcbnew.PCB_VIA)
                               and via.GetNetname() == segment[f"{polarity}_net"]):
                            raise ValueError("unmeasured USB branch or dangling via")
                    if geometry_errors:
                        for error in dict.fromkeys(geometry_errors):
                            fail(f"{segment.get('id')}/{polarity}: {error}")
                    else:
                        tracks[polarity] = path_tracks
                except (KeyError, TypeError, ValueError) as exc:
                    if polarity not in sr["conductors"]:
                        length_complete[polarity] = False
                    transitions_complete[polarity] = False
                    fail(f"{segment.get('id')}/{polarity}: {exc}")
            if len(tracks) == 2 and geometry_available:
                try:
                    if bounded:
                        spans, matching, uncontrolled, joins = _bounded_review_segment(
                            board, segment, cache, regions, geometry)
                        sr["uncontrolled_spans"] = uncontrolled
                        sr["contact_join_bounds"] = joins
                        path_main = {
                            polarity: [track for track in tracks[polarity]
                                       if track[3] == pcbnew.FromMM(float(
                                           geometry["geometry_by_layer"][
                                               "F.Cu" if track[2] == pcbnew.F_Cu else "B.Cu"]["width_mm"]))]
                            for polarity in ("positive", "negative")}
                        if spans:
                            _paired_geometry(path_main["positive"], path_main["negative"], regions, geometry)
                        else:
                            local_geometry = dict(geometry)
                            local_geometry["geometry_by_layer"] = {
                                layer: {"width_mm": geometry["launch_policy"]["width_mm"],
                                        "gap_mm": geometry["launch_policy"]["gap_mm"]}
                                for layer in ("F.Cu", "B.Cu")}
                            _, local_matching = _paired_geometry(
                                tracks["positive"], tracks["negative"], regions, local_geometry)
                            for layer, indices in local_matching.items():
                                matching[layer] = matching.get(layer, indices) & indices
                                if not matching[layer]:
                                    raise ValueError("local pair lacks continuous reference island")
                        coupled_main = coupled_main or bool(spans)
                    else:
                        spans, matching = _paired_geometry(tracks["positive"], tracks["negative"], regions, geometry)
                    sr["paired_spans"] = spans
                    for layer, indices in matching.items():
                        common_reference[layer] = common_reference.get(layer, indices) & indices
                        if not common_reference[layer]:
                            raise ValueError("USB segments do not share a continuous reference island")
                except (KeyError, TypeError, ValueError) as exc:
                    fail(f"{segment.get('id')}: {exc}")
            if any(transitions.values()):
                pending_transitions.append((sr, transitions))
        if bounded and not coupled_main:
            fail("contract has no coupled main-profile copper")
        for sr, transitions in pending_transitions:
            try:
                if not geometry_available:
                    raise ValueError("grounded return transition proof lacks impedance geometry")
                sr["return_vias"] = {polarity: _return_vias(
                    board, changes, regions, contract["reference_net"], geometry, common_reference)
                    for polarity, changes in transitions.items()}
            except (KeyError, TypeError, ValueError) as exc:
                fail(f"{sr['id']}: {exc}")
        row["length_mm"] = {polarity: lengths[polarity] if length_complete[polarity] else None
                            for polarity in ("positive", "negative")}
        row["transitions"] = {polarity: counts[polarity] if transitions_complete[polarity] else None
                              for polarity in ("positive", "negative")}
        row["skew_mm"] = (abs(lengths["positive"] - lengths["negative"])
                          if all(length_complete.values()) else None)
        if all(length_complete.values()) and any(
                len(_as_dict(segment)[f"{polarity}_{side}"]) > 1
                for segment in segments for polarity in ("positive", "negative")
                for side in ("from", "to")):
            try:
                orientations = _orientation_lengths(board, segments, cache, row["component_boundaries"])
                row["orientation_lengths_mm"] = orientations
                row["skew_mm"] = max(abs(item["positive"] - item["negative"])
                                     for item in orientations)
            except (KeyError, TypeError, ValueError) as exc:
                row["skew_mm"] = None
                fail(f"orientation path proof: {exc}")
        skew_limit = float(contract.get("max_skew_mm", 0.5))
        if review:
            skew_limit = min(skew_limit, 0.5)
        if row["skew_mm"] is not None and row["skew_mm"] > skew_limit + 1e-9:
            fail(f"end-to-end skew {row['skew_mm']:.6f} mm exceeds limit")
        transition_limit = int(contract.get("max_layer_transitions", 2))
        if review:
            transition_limit = min(transition_limit, 2)
        if any(counts[p] > transition_limit
               for p in ("positive", "negative") if transitions_complete[p]):
            fail("signal layer transition count exceeds limit")
        row["geometry_ok"] = geometry_available and not row["geometry_errors"]
        receipt["geometry_ok"] = receipt["geometry_ok"] and row["geometry_ok"]
        row["ok"] = not row["failures"]
    receipt["ok"] = not receipt["errors"] and receipt["fabrication_approved"] and receipt["geometry_ok"]
    return receipt
