"""Single Edge.Cuts AABB parser. Used by the renderer, the manual
layout runner, and the verification tool so all three see the same
physical board extent for a given leaf PCB.

Also exposes :func:`classify_edge_cuts_shape` / :func:`classify_ring`: a
deterministic outline-shape classifier (segment count + circularity + corner
and concavity counts) used by the self-eval to check that a built board's
outline matches the requested form factor, without an LLM."""

from __future__ import annotations

import math
import re
from pathlib import Path

# Matches a top-level ``(gr_<kind> ...)`` block and stops at the next
# top-level section token or end-of-file. KiCad writes tracks
# (``segment``/``arc``/``via``), zones, and groups AFTER graphics, so the
# last gr_ block used to swallow all of them and fold their start/end/xy
# points into the "Edge.Cuts" extent whenever no gr_/footprint followed
# (verified on 88/400 real generated boards). The lookahead lists every
# section token that can follow graphics EXCEPT ``(arc`` -- a gr_poly's
# ``pts`` may legitimately embed ``(arc (start..) (mid..) (end..))``
# points (which _POINT_RE's ``mid`` case parses), and KiCraft boards'
# tracks are KiCad Routing Tools segments/vias, never leading track arcs.
_BLOCK_RE = re.compile(
    r'\(gr_(line|arc|rect|poly|circle)\s+(.*?)\)\s*'
    r'(?=\(gr_|\(footprint|\(segment|\(via|\(zone|\(group|\(dimension|\(image|\Z)',
    re.S,
)
_POINT_RE = re.compile(
    r'(?:\((?:start|end|center|mid)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\))'
    r'|(?:\(xy\s+([-\d.eE+]+)\s+([-\d.eE+]+)\))'
)


def parse_edge_cuts_aabb(
    pcb_path: Path,
) -> tuple[float, float, float, float] | None:
    """AABB of every ``gr_line`` / ``gr_arc`` / ``gr_rect`` / ``gr_poly``
    / ``gr_circle`` on the Edge.Cuts layer of ``pcb_path``. Returns
    ``(min_x, min_y, max_x, max_y)`` in the PCB's coordinate system
    (mm) or ``None`` when the file is missing, unreadable, or has no
    Edge.Cuts geometry.

    The result is the PHYSICAL extent of the board outline -- what
    gets stamped on the parent for a leaf, what defines collision
    on the manual layout canvas, and what the rendered PNG should
    contain by construction.
    """
    try:
        text = pcb_path.read_text(encoding="utf-8")
    except OSError:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for m in _BLOCK_RE.finditer(text):
        blk = m.group(0)
        if 'Edge.Cuts' not in blk:
            continue
        for pm in _POINT_RE.finditer(blk):
            if pm.group(1) is not None:
                xs.append(float(pm.group(1)))
                ys.append(float(pm.group(2)))
            else:
                xs.append(float(pm.group(3)))
                ys.append(float(pm.group(4)))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


# --------------------------------------------------------------------------- #
# Shape classification (deterministic; for the self-eval outline check)
# --------------------------------------------------------------------------- #

# Requested-shape name -> the coarse family the classifier reports. The check
# is family-level on purpose: distinguishing circle from rounded_rect (both
# round) or hexagon from octagon (both polygon) is not what the rubric needs.
SHAPE_FAMILY: dict[str, str] = {
    "rect": "rectangular",
    "circle": "round",
    "rounded_rect": "round",
    "chamfered_rect": "polygon",
    "triangle": "polygon",
    "pentagon": "polygon",
    "hexagon": "polygon",
    "octagon": "polygon",
    "star": "star",
    "heart": "compound",
    "snowman": "compound",
}


def family_for_shape(name: str) -> str:
    """Coarse outline family expected for a requested shape name."""
    return SHAPE_FAMILY.get((name or "").strip().lower(), "polygon")


def _close(a, b, tol=1e-3) -> bool:
    return abs(a[0] - b[0]) <= tol and abs(a[1] - b[1]) <= tol


def _order_ring(segments, tol=1e-3):
    """Chain ``[(p0, p1), ...]`` edge segments into an ordered vertex ring."""
    if not segments:
        return []
    ring = [segments[0][0], segments[0][1]]
    used = [False] * len(segments)
    used[0] = True
    for _ in range(len(segments) - 1):
        last = ring[-1]
        for i, (a, b) in enumerate(segments):
            if used[i]:
                continue
            if _close(a, last, tol):
                ring.append(b)
                used[i] = True
                break
            if _close(b, last, tol):
                ring.append(a)
                used[i] = True
                break
        else:
            break
    if len(ring) > 1 and _close(ring[0], ring[-1], tol):
        ring.pop()
    return ring


def classify_ring(points) -> dict:
    """Classify a closed vertex ring into an outline ``label`` + ``family``.

    Features: vertex count, circularity ``4*pi*area / perimeter**2`` (1.0 for a
    circle, pi/4 for a square), the number of *sharp* corners (turn > 25 deg)
    and *concave* (reflex) corners. Families: ``rectangular`` (4 right-angle
    corners, low circularity), ``round`` (circle / rounded-rect), ``polygon``
    (3/5/6/7/8 convex corners, incl. chamfered rect), ``star`` (many alternating
    reflex corners), ``compound`` (a few reflex corners on an otherwise smooth
    boundary -- snowman / heart). ``unknown`` for degenerate input.
    """
    ring = [(float(x), float(y)) for x, y in points]
    # Drop a duplicate closing point if present.
    if len(ring) > 1 and _close(ring[0], ring[-1]):
        ring.pop()
    n = len(ring)
    base = {"label": "unknown", "family": "unknown", "n_vertices": n,
            "n_corners": 0, "n_concave": 0, "circularity": 0.0}
    if n < 3:
        return base

    area2 = 0.0
    perimeter = 0.0
    for i in range(n):
        x0, y0 = ring[i]
        x1, y1 = ring[(i + 1) % n]
        area2 += x0 * y1 - x1 * y0
        perimeter += math.hypot(x1 - x0, y1 - y0)
    area = abs(area2) / 2.0
    circ = (4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0 else 0.0

    turns = []
    for i in range(n):
        p = ring[i - 1]
        c = ring[i]
        q = ring[(i + 1) % n]
        v1 = (c[0] - p[0], c[1] - p[1])
        v2 = (q[0] - c[0], q[1] - c[1])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        turns.append(math.atan2(cross, dot))
    ccw = sum(turns) > 0
    sharp = math.radians(25)
    reflex = math.radians(8)
    n_corners = sum(1 for t in turns if abs(t) > sharp)
    # Reflex turns oppose the overall winding.
    n_concave = sum(1 for t in turns if (t < -reflex if ccw else t > reflex))

    out = {**base, "n_corners": n_corners, "n_concave": n_concave,
           "circularity": round(circ, 4)}

    if n_concave >= 2:
        # Many alternating reflex corners -> star; a few on an otherwise smooth
        # boundary -> compound (snowman / heart).
        if n_corners >= 8 and n_concave >= 4:
            out.update(label="star", family="star")
        else:
            out.update(label="compound", family="compound")
        return out
    if n_corners <= 2:
        # No sharp corners -> a smooth/round boundary (circle or rounded-rect).
        # A true rectangle keeps its four ~90 deg corners and is caught below.
        out.update(label=("round" if circ > 0.9 else "rounded"), family="round")
        return out
    if n_corners == 4:
        out.update(label="rect", family="rectangular")
        return out
    label = {3: "triangle", 5: "pentagon", 6: "hexagon",
             7: "heptagon", 8: "octagon"}.get(n_corners, "polygon")
    out.update(label=label, family="polygon")
    return out


def _edge_segments(text: str):
    segs = []
    for m in _BLOCK_RE.finditer(text):
        blk = m.group(0)
        if "Edge.Cuts" not in blk:
            continue
        kind = m.group(1)
        pts = []
        for pm in _POINT_RE.finditer(blk):
            if pm.group(1) is not None:
                pts.append((float(pm.group(1)), float(pm.group(2))))
            else:
                pts.append((float(pm.group(3)), float(pm.group(4))))
        if kind == "line" and len(pts) >= 2:
            segs.append((pts[0], pts[1]))
        elif kind == "rect" and len(pts) >= 2:
            (x0, y0), (x1, y1) = pts[0], pts[1]
            corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            for i in range(4):
                segs.append((corners[i], corners[(i + 1) % 4]))
        elif kind == "poly" and len(pts) >= 3:
            for i in range(len(pts)):
                segs.append((pts[i], pts[(i + 1) % len(pts)]))
        elif kind == "circle":
            return "circle"
    return segs


def classify_edge_cuts_shape(pcb_path: Path) -> dict | None:
    """Classify the Edge.Cuts outline of ``pcb_path`` (see :func:`classify_ring`).
    Returns ``None`` when the file is missing/unreadable or has no Edge.Cuts."""
    try:
        text = Path(pcb_path).read_text(encoding="utf-8")
    except OSError:
        return None
    segs = _edge_segments(text)
    if segs == "circle":
        return {"label": "round", "family": "round", "n_vertices": 0,
                "n_corners": 0, "n_concave": 0, "circularity": 1.0}
    if not segs:
        return None
    ring = _order_ring(segs)
    if len(ring) < 3:
        return None
    return classify_ring(ring)
