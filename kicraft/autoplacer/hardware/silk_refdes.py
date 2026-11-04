"""Place readable reference designators outside component bodies.

Courtyards are placement envelopes, not empty space: centering a reference in
one can print it on a component, its polarity marks, or exposed pads. Search
outside all courtyards and preserve an unplaceable reference on Fab instead.
"""
from __future__ import annotations

import pcbnew

from kicraft.autoplacer.hardware.silk_geometry import bbox_inside_poly, boxes_overlap

# Readability floor matches kicad_pro.DEFAULT_RULES["min_text_height"].
_TEXT_HEIGHTS_MM = (1.0, 0.9, 0.8)
_THICKNESS_RATIO = 0.15  # KiCad's nominal stroke:height ratio


def _courtyard_bbox(fp):
    layer = pcbnew.F_CrtYd if fp.GetLayer() == pcbnew.F_Cu else pcbnew.B_CrtYd
    try:
        bb = fp.GetCourtyard(layer).BBox()
    except Exception:
        return None
    if bb.GetWidth() <= 0 or bb.GetHeight() <= 0:
        return None
    return bb



def _move_to_fab(fp, item) -> None:
    item.SetLayer(pcbnew.B_Fab if fp.GetLayer() == pcbnew.B_Cu else pcbnew.F_Fab)
    item.SetVisible(True)  # keep for assembly / pick-and-place


def _board_polygon(board):
    shape = pcbnew.SHAPE_POLY_SET()
    if not board.GetBoardPolygonOutlines(shape) or shape.OutlineCount() != 1:
        return None
    def points(chain):
        return [(pcbnew.ToMM(chain.CPoint(i).x), pcbnew.ToMM(chain.CPoint(i).y))
                for i in range(chain.PointCount())]
    outer = points(shape.COutline(0))
    holes = [points(shape.CHole(0, i)) for i in range(shape.HoleCount(0))]
    return (outer, holes) if len(outer) >= 3 else None


def _box_mm(bb):
    return tuple(pcbnew.ToMM(v) for v in
                 (bb.GetLeft(), bb.GetTop(), bb.GetRight(), bb.GetBottom()))


def _inside_outline(box, polygon, clearance, holes=()):
    # The probes alone do not prove containment for a concave outline. Also
    # reject any box intersecting the bounding box of an outline segment.
    if not bbox_inside_poly(box, polygon, margin=clearance):
        return False
    expanded = (box[0] - clearance, box[1] - clearance,
                box[2] + clearance, box[3] + clearance)
    for hole in holes:
        if len(hole) < 3:
            return False
        bounds = (min(p[0] for p in hole), min(p[1] for p in hole),
                  max(p[0] for p in hole), max(p[1] for p in hole))
        if boxes_overlap(expanded, bounds):
            return False
    for a, b in zip(polygon, polygon[1:] + polygon[:1]):
        edge = (min(a[0], b[0]), min(a[1], b[1]),
                max(a[0], b[0]), max(a[1], b[1]))
        if (expanded[0] <= edge[2] and edge[0] <= expanded[2] and
                expanded[1] <= edge[3] and edge[1] <= expanded[3]):
            return False
    return True


def legalize_refdes(board, cfg: dict | None = None) -> dict[str, list[str]]:
    """Search outside component courtyards; keep unplaceable refs on Fab."""
    cfg = cfg or {}
    clearance = max(0.2, float(cfg.get("refdes_courtyard_margin_mm", 0.1)))
    polygon = _board_polygon(board)
    footprints = list(board.Footprints())
    obstacles = {"F": [], "B": []}
    envelopes = {}
    for fp in footprints:
        side = "B" if fp.GetLayer() == pcbnew.B_Cu else "F"
        start = len(obstacles[side])
        court = _courtyard_bbox(fp)
        if court is not None:
            obstacles[side].append(_box_mm(court))
        for pad in fp.Pads():
            box = _box_mm(pad.GetBoundingBox())
            obstacles[side].append(box)
            if pad.HasHole():
                obstacles["F" if side == "B" else "B"].append(box)
        for item in fp.GraphicalItems():
            layer = item.GetLayer()
            if layer == pcbnew.F_SilkS:
                obstacles["F"].append(_box_mm(item.GetBoundingBox()))
            elif layer == pcbnew.B_SilkS:
                obstacles["B"].append(_box_mm(item.GetBoundingBox()))
        if court is not None:
            own = obstacles[side][start:]
            envelopes[fp.GetReferenceAsString()] = (
                min(box[0] for box in own), min(box[1] for box in own),
                max(box[2] for box in own), max(box[3] for box in own),
            )
    for item in board.GetDrawings():
        if item.GetLayer() == pcbnew.F_SilkS:
            obstacles["F"].append(_box_mm(item.GetBoundingBox()))
        elif item.GetLayer() == pcbnew.B_SilkS:
            obstacles["B"].append(_box_mm(item.GetBoundingBox()))

    kept, moved = [], []
    for fp in footprints:
        item = fp.Reference()
        if not item.IsVisible():
            continue
        side = "B" if fp.GetLayer() == pcbnew.B_Cu else "F"
        if item.GetLayer() not in (pcbnew.F_SilkS, pcbnew.B_SilkS):
            continue
        ref = fp.GetReferenceAsString()
        envelope = envelopes.get(ref)
        original_pos, original_size, original_thick = item.GetPosition(), item.GetTextSize(), item.GetTextThickness()
        placed = False
        if envelope is not None and polygon is not None:
            left, top, right, bottom = envelope
            for height in _TEXT_HEIGHTS_MM:
                h = pcbnew.FromMM(height)
                item.SetTextSize(pcbnew.VECTOR2I(h, h))
                item.SetTextThickness(pcbnew.FromMM(height * _THICKNESS_RATIO))
                bb = _box_mm(item.GetBoundingBox())
                half_w, half_h = (bb[2] - bb[0]) / 2, (bb[3] - bb[1]) / 2
                # Strict clearance comparisons and integer coordinates must not
                # turn a boundary-equal placement into a rejected collision.
                gap = clearance + 0.001
                candidates = [(pcbnew.ToMM(original_pos.x), pcbnew.ToMM(original_pos.y)),
                              ((left + right) / 2, top - gap - half_h),
                              ((left + right) / 2, bottom + gap + half_h),
                              (left - gap - half_w, (top + bottom) / 2),
                              (right + gap + half_w, (top + bottom) / 2)]
                for x, y in candidates:
                    item.SetPosition(pcbnew.VECTOR2I(pcbnew.FromMM(x), pcbnew.FromMM(y)))
                    box = _box_mm(item.GetBoundingBox())
                    if (_inside_outline(box, polygon[0], clearance, polygon[1]) and
                            not any(boxes_overlap(box, obstacle, clearance) for obstacle in obstacles[side])):
                        obstacles[side].append(box)
                        kept.append(ref)
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            item.SetPosition(original_pos)
            item.SetTextSize(original_size)
            item.SetTextThickness(original_thick)
            _move_to_fab(fp, item)
            moved.append(ref)
    return {"kept": kept, "moved_to_fab": moved}
