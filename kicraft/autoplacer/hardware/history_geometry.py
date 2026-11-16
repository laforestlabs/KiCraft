"""Best-effort snapshots of already-loaded native KiCad boards.

This module deliberately does not import pcbnew until extraction is requested.
Each thread owns its writer; pcbnew subprocesses therefore never share a spool.
"""
from __future__ import annotations

import atexit
import math
import threading
import time
import warnings

from kicraft.layout_history import HistoryDescriptor

_local = threading.local()
_writers = []
_lock = threading.Lock()


def _writer():
    descriptor = HistoryDescriptor.from_env()
    previous = getattr(_local, "descriptor", None)
    writer = getattr(_local, "writer", None)
    if descriptor != previous and writer is not None:
        writer.end("complete")
        with _lock:
            if writer in _writers:
                _writers.remove(writer)
        writer = None
    _local.descriptor = descriptor
    if descriptor is not None and (writer is None or writer.closed):
        try:
            writer = descriptor.writer("finishing", "Native board mutations")
            with _lock:
                _writers.append(writer)
        except Exception as error:
            warnings.warn(f"[layout-history] cannot open native scope: {error}", RuntimeWarning)
    _local.writer = writer
    return writer


def close_native_history(outcome="complete"):
    """Close the current thread's native scope after its final mutation."""
    writer = getattr(_local, "writer", None)
    if writer is not None:
        writer.end(outcome)
        with _lock:
            if writer in _writers:
                _writers.remove(writer)
        _local.writer = None


def close_capture_history(capture_id):
    """Close this process's finished native scopes for one capture only.

    Call after the capture's worker threads have stopped writing and before
    publication; other concurrently active captures remain untouched.
    """
    with _lock:
        selected = [writer for writer in _writers
                    if writer.descriptor.capture_id == capture_id]
        _writers[:] = [writer for writer in _writers
                       if writer.descriptor.capture_id != capture_id and not writer.closed]
    for writer in selected:
        writer.end("complete")


def _close_all():
    with _lock:
        selected = _writers[:]
        _writers.clear()
    for writer in selected:
        writer.end("complete")


atexit.register(_close_all)


def _uuid(item):
    try:
        return str(item.GetUuid().AsString())
    except AttributeError:
        return str(item.m_Uuid.AsString())


def _mm(value):
    return round(float(value) / 1_000_000, 4)


def _xy(point):
    return [_mm(point.x), _mm(point.y)]


def _rings(poly, transform=lambda point: _xy(point)):
    def ring(chain):
        return [transform(chain.CPoint(i)) for i in range(chain.PointCount())]
    return [{"points": ring(poly.Outline(i)),
             "holes": [ring(poly.Hole(i, j)) for j in range(poly.HoleCount(i))]}
            for i in range(poly.OutlineCount())]


def _arc_points(start, mid, end):
    """Tessellate a three-point circular arc with at most 0.001 mm sagitta."""
    ax, ay = start
    bx, by = mid
    cx, cy = end
    d = 2 * (ax * (by-cy) + bx * (cy-ay) + cx * (ay-by))
    if abs(d) < 1e-12:
        return [start, end]
    aa, bb, cc = ax*ax+ay*ay, bx*bx+by*by, cx*cx+cy*cy
    ox = (aa*(by-cy) + bb*(cy-ay) + cc*(ay-by))/d
    oy = (aa*(cx-bx) + bb*(ax-cx) + cc*(bx-ax))/d
    radius = math.hypot(ax-ox, ay-oy)
    a0 = math.atan2(ay-oy, ax-ox)
    am = (math.atan2(by-oy, bx-ox) - a0) % (2*math.pi)
    ae = (math.atan2(cy-oy, cx-ox) - a0) % (2*math.pi)
    sweep = ae if am <= ae else ae - 2*math.pi
    max_step = 2 * math.acos(max(-1.0, 1 - min(0.001/radius, 1.0)))
    count = max(1, math.ceil(abs(sweep) / max_step))
    points = [[round(ox + radius*math.cos(a0+sweep*i/count), 4),
               round(oy + radius*math.sin(a0+sweep*i/count), 4)]
              for i in range(count+1)]
    points[0], points[-1] = start, end
    return points


def _bezier_points(a, b, c, d):
    points = [a]
    def midpoint(p, q):
        return [(p[0]+q[0])/2, (p[1]+q[1])/2]
    def distance_to_segment(p, x, y):
        dx, dy = y[0]-x[0], y[1]-x[1]
        length = dx*dx + dy*dy
        t = max(0, min(1, ((p[0]-x[0])*dx+(p[1]-x[1])*dy)/length)) if length else 0
        return math.hypot(p[0]-x[0]-t*dx, p[1]-x[1]-t*dy)
    def subdivide(p, q, r, s):
        if max(distance_to_segment(q, p, s), distance_to_segment(r, p, s)) <= 0.001:
            points.append([round(s[0], 4), round(s[1], 4)])
            return
        pq, qr, rs = midpoint(p, q), midpoint(q, r), midpoint(r, s)
        left, right = midpoint(pq, qr), midpoint(qr, rs)
        center = midpoint(left, right)
        subdivide(p, pq, left, center)
        subdivide(center, right, rs, s)
    subdivide(a, b, c, d)
    return points


def _edge_rings(board, pcbnew):
    """Stitch native Edge.Cuts primitives and classify enclosed cutouts."""
    paths = []
    closed = []
    drawings = list(board.GetDrawings())
    for footprint in board.GetFootprints():
        drawings.extend(footprint.GraphicalItems())
    for item in drawings:
        if item.GetLayer() != pcbnew.Edge_Cuts or not isinstance(item, pcbnew.PCB_SHAPE):
            continue
        kind = item.GetShape()
        if kind == pcbnew.S_SEGMENT:
            paths.append([_xy(item.GetStart()), _xy(item.GetEnd())])
        elif kind == pcbnew.S_ARC:
            paths.append(_arc_points(_xy(item.GetStart()), _xy(item.GetArcMid()),
                                     _xy(item.GetEnd())))
        elif kind == pcbnew.S_CURVE:
            paths.append(_bezier_points(_xy(item.GetStart()), _xy(item.GetBezierC1()),
                                        _xy(item.GetBezierC2()), _xy(item.GetEnd())))
        elif kind == pcbnew.S_CIRCLE:
            center = _xy(item.GetCenter())
            radius = _mm(item.GetRadius())
            if radius:
                step = 2 * math.acos(max(-1.0, 1-min(0.001/radius, 1.0)))
                count = max(8, math.ceil(2*math.pi/step))
                closed.append([[round(center[0]+radius*math.cos(2*math.pi*i/count), 4),
                                round(center[1]+radius*math.sin(2*math.pi*i/count), 4)]
                               for i in range(count)])
        elif kind == pcbnew.S_RECT:
            a, b = _xy(item.GetStart()), _xy(item.GetEnd())
            closed.append([a, [b[0], a[1]], b, [a[0], b[1]]])
        elif kind == pcbnew.S_POLYGON:
            for ring in _rings(item.GetPolyShape()):
                closed.append(ring["points"])
                closed.extend(ring["holes"])
        else:
            raise ValueError(f"unsupported Edge.Cuts shape {kind}")
    def close(a, b):
        return math.dist(a, b) <= 0.01
    while paths:
        chain = paths.pop()
        while not close(chain[0], chain[-1]):
            match = next((i for i, part in enumerate(paths)
                          if close(chain[-1], part[0]) or close(chain[-1], part[-1])), None)
            if match is None:
                raise ValueError("unclosed Edge.Cuts contour")
            part = paths.pop(match)
            if close(chain[-1], part[-1]):
                part.reverse()
            chain.extend(part[1:])
        closed.append(chain[:-1])
    def contains(ring, p):
        inside = False
        for a, b in zip(ring, ring[1:] + ring[:1]):
            if (a[1] > p[1]) != (b[1] > p[1]) and p[0] < (
                    (b[0]-a[0])*(p[1]-a[1])/(b[1]-a[1])+a[0]):
                inside = not inside
        return inside
    parents = []
    for i, ring in enumerate(closed):
        enclosing = [j for j, outer in enumerate(closed)
                     if j != i and contains(outer, ring[0])]
        parents.append(enclosing)
    result = []
    for i, ring in enumerate(closed):
        if len(parents[i]) % 2:
            continue
        holes = [child for j, child in enumerate(closed)
                 if i in parents[j] and len(parents[j]) == len(parents[i]) + 1]
        result.append({"points": ring, "holes": holes})
    return result


def extract_native_scene(board, history, *, phase):
    """Extract native identity, copper, footprints, zones and board outline."""
    import pcbnew

    objects = []
    layers = list(board.GetEnabledLayers().CuStack())
    layer_name = board.GetLayerName
    for fp in board.GetFootprints():
        origin = fp.GetPosition()
        angle = fp.GetOrientationDegrees()
        rad = math.radians(angle)
        co, si = math.cos(rad), math.sin(rad)
        back = fp.GetLayer() == pcbnew.B_Cu
        def local(point):
            x, y = _mm(point.x - origin.x), _mm(point.y - origin.y)
            local_x = co*x + si*y
            return [round(-local_x if back else local_x, 4), round(-si*x + co*y, 4)]
        pads = []
        for pad in fp.Pads():
            pad_layers = [layer for layer in layers if pad.IsOnLayer(layer)]
            copper_shapes = {}
            for layer in pad_layers:
                shape = pcbnew.SHAPE_POLY_SET()
                pad.TransformShapeToPolygon(shape, layer, 0, pcbnew.FromMM(0.001),
                                            pcbnew.ERROR_INSIDE)
                copper_shapes[layer_name(layer)] = _rings(shape, local)
            size, drill = pad.GetSize(), pad.GetDrillSize()
            pads.append({"id": _uuid(pad), "number": pad.GetNumber(),
                         "net": pad.GetNetname(), "pos": local(pad.GetPosition()),
                         "size": _xy(size), "drill": _xy(drill),
                         "width_mm": _mm(size.x), "height_mm": _mm(size.y),
                         "copper_shapes": copper_shapes,
                         "shape": int(pad.GetShape()),
                         "rotation": round(pad.GetOrientationDegrees() - angle, 4),
                         "layers": [layer_name(layer) for layer in pad_layers]})
        courtyard = []
        for graphic in fp.GraphicalItems():
            layer = graphic.GetLayer()
            if layer not in (pcbnew.F_CrtYd, pcbnew.B_CrtYd):
                continue
            shape = pcbnew.SHAPE_POLY_SET()
            graphic.TransformShapeToPolygon(shape, layer, 0, pcbnew.FromMM(0.001),
                                            pcbnew.ERROR_INSIDE)
            courtyard.extend(_rings(shape, local))
        template = history.define({"geometry_source": "native_board", "pads": pads,
                                   "courtyards": courtyard}, phase=phase)
        objects.append({"id": _uuid(fp), "type": "footprint", "ref": fp.GetReference(),
                        "value": fp.GetValue(), "template_id": template,
                        "pose": {"x": _mm(origin.x), "y": _mm(origin.y),
                                 "rotation": round(angle, 4),
                                 "side": "back" if back else "front"}})
    for item in board.GetTracks():
        base = {"id": _uuid(item), "net": item.GetNetname()}
        if isinstance(item, pcbnew.PCB_VIA):
            width = item.GetWidth(item.TopLayer())
            base.update(type="via", position=_xy(item.GetPosition()),
                        diameter=_mm(width), drill=_mm(item.GetDrillValue()),
                        layers=[layer_name(layer) for layer in layers if item.IsOnLayer(layer)])
        else:
            base.update(type="arc" if isinstance(item, pcbnew.PCB_ARC) else "track",
                        start=_xy(item.GetStart()), end=_xy(item.GetEnd()),
                        width=_mm(item.GetWidth()), layer=layer_name(item.GetLayer()))
            if base["type"] == "arc":
                base["mid"] = _xy(item.GetMid())
        objects.append(base)
    for zone in board.Zones():
        keepout = bool(zone.GetIsRuleArea())
        for layer in layers:
            if not zone.IsOnLayer(layer):
                continue
            if keepout:
                poly = pcbnew.SHAPE_POLY_SET()
                zone.TransformShapeToPolygon(poly, layer, 0, pcbnew.FromMM(0.001),
                                             pcbnew.ERROR_INSIDE)
            else:
                poly = zone.GetFilledPolysList(layer)
            geometry = {"contours": _rings(poly)}
            template = history.define(geometry, phase=phase)
            objects.append({"id": _uuid(zone) + ":" + layer_name(layer),
                            "type": "keepout" if keepout else "zone", "template_id": template,
                            "layer": layer_name(layer), "net": zone.GetNetname()})
    contours = _edge_rings(board, pcbnew)
    if contours:
        template = history.define({"contours": contours}, phase=phase)
        objects.append({"id": "board:outline", "type": "outline", "template_id": template})
    return objects


def capture_board(board, *, phase, history=None):
    """Capture a loaded board without changing its geometry or build result."""
    writer = history if history is not None else _writer()
    if writer is None or writer.disabled or writer.closed:
        return
    started = time.process_time_ns()
    try:
        objects = extract_native_scene(board, writer, phase=phase)
        writer.update_scene(objects, phase=phase, geometry_source="native_board")
        return True
    except Exception as error:
        writer.decision("skip", phase=phase, gap="native_extraction_failed", reason=str(error))
        warnings.warn(f"[layout-history] native extraction failed: {error}", RuntimeWarning)
        return False
    finally:
        writer.cpu_ns += time.process_time_ns() - started


def capture_restored_file(path, *, phase):
    """Measure a byte-restored board only when a capture is active."""
    writer = _writer()
    if writer is None:
        return
    try:
        import pcbnew
        capture_board(pcbnew.LoadBoard(str(path)), phase=phase, history=writer)
    except Exception as error:
        writer.decision("skip", phase=phase, gap="restore_extraction_failed", reason=str(error))
        warnings.warn(f"[layout-history] restore extraction failed: {error}", RuntimeWarning)


def capture_restored_external(path, *, phase):
    """Capture a restored board when the caller has no pcbnew-loaded object."""
    writer = _writer()
    if writer is None:
        return
    try:
        import json
        import subprocess
        import sys
        from kicraft.autoplacer.routing_board import _kicad_subprocess_env
        env = writer.child_env(_kicad_subprocess_env())
        result = subprocess.run(
            [sys.executable, "-m", __name__, "--board", str(path), "--phase", phase],
            env=env, capture_output=True, text=True, timeout=60,
        )
        if result.returncode:
            raise RuntimeError(result.stderr[-1000:])
        scope_id = json.loads(result.stdout.strip().splitlines()[-1])["scope_id"]
        writer.decision("restore", phase=phase, reason="restored board measured in child scope",
                        source_scope_id=scope_id)
    except Exception as error:
        writer.decision("skip", phase=phase, gap="restore_extraction_failed", reason=str(error))
        warnings.warn(f"[layout-history] restore extraction failed: {error}", RuntimeWarning)


def main():
    import argparse
    import json
    import pcbnew

    parser = argparse.ArgumentParser(description="Capture a selected native board")
    parser.add_argument("--board", required=True)
    parser.add_argument("--phase", default="delivered")
    args = parser.parse_args()
    writer = _writer()
    if writer is None:
        print(json.dumps({"scope_id": None}))
        return
    try:
        board = pcbnew.LoadBoard(args.board)
        if board is None:
            raise ValueError("KiCad could not load selected board")
        if not capture_board(board, phase=args.phase, history=writer):
            raise RuntimeError("native extraction failed")
        if args.phase == "delivered":
            writer.decision("select", phase=args.phase, reason="selected delivered board")
        outline = pcbnew.SHAPE_POLY_SET()
        origin = None
        if board.GetBoardPolygonOutlines(outline) and outline.OutlineCount():
            bbox = outline.BBox()
            origin = [_mm(bbox.GetLeft()), _mm(bbox.GetTop())]
        footprints = {fp.GetReference(): [*_xy(fp.GetPosition()),
                                            round(fp.GetOrientationDegrees(), 4)]
                      for fp in board.GetFootprints()}
        writer.end("complete")
        print(json.dumps({"scope_id": writer.scope_id, "outline_origin": origin,
                          "footprints": footprints}))
    except Exception as error:
        writer.decision("skip", phase=args.phase, gap="native_extraction_failed", reason=str(error))
        writer.end("error")
        raise


if __name__ == "__main__":
    main()
