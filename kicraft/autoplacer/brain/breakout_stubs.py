"""Deliberate breakout stubs / same-net tie copper for footprint pads.

This module is the reusable primitive for stamping locked same-net ties and
escapes as copper. A :class:`BreakoutSpec` describes the copper to lay for one
pad as an explicit polyline (``waypoints``) and/or a *radial* escape (straight
out from the footprint centre through the pad). :func:`add_breakout_stubs` lays
the segments as **locked** tracks (optionally dropping a via at the end) so a
subsequent KiCad Routing Tools pass run with
``routing_preserve_existing_copper=True`` keeps them and routes the rest from
the accessible endpoints.

Consumers: power-pour fragmentation ties (:func:`auto_power_tie_specs`,
:func:`perimeter_tie_specs`), PTH shield grounding (:func:`shield_tie_specs`),
array-routing daisy-chain/ring ties, and GND plane bonding (``gnd_pour.py``).
This is not an escape-planning engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from kicraft.autoplacer.fab_profile import fab_floors
from kicraft.autoplacer.hardware.keepout_extract import (
    collect_track_via_rule_areas,
    track_intersects_rule_area,
    via_intersects_rule_area,
)

import pcbnew

_LAYERS = {"F.Cu": pcbnew.F_Cu, "B.Cu": pcbnew.B_Cu}

# Guard held ABOVE the resolved clearance rule by every margin this module
# computes. The clearance here is enforced by sampled HitTest(margin) checks
# whose geometry model can land a hair under KiCad DRC's exact measurement on
# rotated pads (run_13 nRF52 aQFN: stubs at 0.1520 vs the 0.1530 rule, 1 um
# short). The board still verifies at the true rule, so the guard can never
# mask a violation -- but anything PLANNING copper for this module to stamp
# must plan at rule + guard, or its output gets dropped here as too close.
STAMP_CLEARANCE_GUARD_MM = 0.01


@dataclass(slots=True)
class BreakoutSpec:
    """How to escape one pad.

    ref / pad:
        Footprint reference (e.g. ``"J1"``) and pad number (e.g. ``"B5"``).
    waypoints:
        Absolute board points (mm) the stub passes through after leaving the
        pad centre. Use for curated, obstacle-aware escapes (the reliable mode
        for nets the autorouter abandons). Empty -> use a radial escape.
    length_mm:
        Radial escape length when *waypoints* is empty: a single segment from
        the pad centre straight out from the footprint centre.
    width_mm:
        Track width; ``None`` falls back to the fabrication track floor.
    layer:
        Copper layer the stub is drawn on.
    via_at_end:
        Drop a layer-changing via at the final point (to escape onto the other
        layer when this one is congested).
    via_size_mm / via_drill_mm:
        Override the via geometry for THIS stub; ``None`` uses the board's
        netclass via (``cfg['via_size_mm']``/``via_drill_mm``, 0.6/0.3). The
        escape planner sets the smaller fanout class here: a 0.6 mm via has no
        legal position beside a 0.5 mm-pitch inner ring at any clearance, so a
        dog-bone out of a fine-pitch package is only expressible with it.
    near_xy:
        Disambiguates the pad when the footprint carries SEVERAL pads with the
        same number (ESP32-class modules number every ground pad "GND"): the
        matching pad nearest this board point (mm) is used. Without it the
        first match wins -- which for a strand repair tied the wrong, already-
        connected pad and silently left the stranded one (run_01/run_03).
    start_xy / net:
        Free-coordinate anchor (C1 v2 track-endpoint anchors): start the tie
        at this board point (mm) on ``net`` instead of at a pad -- the mode
        for an unconnected edge whose endpoint is a dangling track/via stub
        (``ref``/``pad`` then serve only as the skip-label). Requires
        ``waypoints`` (there is no footprint to escape radially from); every
        clearance/copper/via guard still applies, using the net's nearest
        pad as the netclass stand-in.
    """

    ref: str
    pad: str
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    length_mm: float = 1.5
    width_mm: float | None = None
    layer: str = "F.Cu"
    via_at_end: bool = False
    near_xy: tuple[float, float] | None = None
    start_xy: tuple[float, float] | None = None
    net: str | None = None
    via_size_mm: float | None = None
    via_drill_mm: float | None = None


def _find_pad(
    board: "pcbnew.BOARD", ref: str, pad_number: str, near_xy: tuple[float, float] | None = None
):
    """The footprint's pad with *pad_number*; nearest to *near_xy* when several
    pads share that number (module footprints number every ground pad "GND")."""
    for fp in board.GetFootprints():
        if fp.GetReferenceAsString() == ref:
            matches = [p for p in fp.Pads() if p.GetNumber() == pad_number]
            if not matches:
                return None, None
            if near_xy is not None and len(matches) > 1:
                matches.sort(
                    key=lambda p: (
                        (pcbnew.ToMM(p.GetPosition().x) - near_xy[0]) ** 2
                        + (pcbnew.ToMM(p.GetPosition().y) - near_xy[1]) ** 2
                    )
                )
            return fp, matches[0]
    return None, None


def _nearest_same_net_pad(board: "pcbnew.BOARD", net_name: str, near_xy: tuple[float, float]):
    """Any pad on *net_name*, nearest *near_xy*: the netclass/margin stand-in
    for a tie anchored at bare track copper (the guard machinery is
    pad-shaped, but the net's clearance rules are pad-independent)."""
    best_fp, best_pad, best_d = None, None, None
    for fp in board.GetFootprints():
        for p in fp.Pads():
            if p.GetNetname() != net_name:
                continue
            d = (pcbnew.ToMM(p.GetPosition().x) - near_xy[0]) ** 2 + (
                pcbnew.ToMM(p.GetPosition().y) - near_xy[1]
            ) ** 2
            if best_d is None or d < best_d:
                best_fp, best_pad, best_d = fp, p, d
    return best_fp, best_pad


def _foreign_pads(board: "pcbnew.BOARD", net_code: int, *, exclude=None) -> list:
    """Every pad that copper on *net_code* must stay clear of.

    A same-net pad is a valid landing (a stub may touch it), so it is not an
    obstacle. A no-net pad (code 0) is foreign to everything -- it is an NC or
    mechanical pad a trace must never cross. *exclude* drops one pad (the escape
    source) for the degenerate case of a no-net source pad.
    """
    out = []
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad is exclude:
                continue
            if net_code and pad.GetNetCode() == net_code:
                continue
            out.append(pad)
    return out


def _segment_clears_pads(
    pads: list,
    a_mm: tuple[float, float],
    b_mm: tuple[float, float],
    clearance_mm: float,
    *,
    step_mm: float = 0.1,
) -> bool:
    """True when the segment *a*->*b* keeps >= *clearance_mm* from every pad.

    Samples the segment at ``step_mm`` and rejects it if any sample falls within
    a pad grown by the clearance (``HitTest`` margin). Conservative by design --
    a near-miss counts as a hit -- so a stub is dropped rather than shorted.
    """
    margin = int(pcbnew.FromMM(clearance_mm))
    ax, ay = a_mm
    bx, by = b_mm
    steps = max(1, int(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / step_mm))
    for k in range(steps + 1):
        t = k / steps
        pt = pcbnew.VECTOR2I(
            int(pcbnew.FromMM(ax + (bx - ax) * t)),
            int(pcbnew.FromMM(ay + (by - ay) * t)),
        )
        if any(pad.HitTest(pt, margin) for pad in pads):
            return False
    return True


def _own_clearance_mm(pad, layer_id, fallback_mm: float) -> float:
    """Resolved clearance constraint of *pad* in mm (netclass-aware).

    Falls back to *fallback_mm* on boards without project netclasses (tests,
    bare .kicad_pcb files) or older pcbnew APIs.
    """
    try:
        v = pcbnew.ToMM(pad.GetOwnClearance(layer_id))
    except Exception:
        return fallback_mm
    return v if v > 0 else fallback_mm


def _pad_ref(pad) -> str:
    try:
        fp = pad.GetParentFootprint()
        return fp.GetReferenceAsString() if fp else ""
    except Exception:
        return ""


def _pad_hole_overhang_mm(pad) -> float | None:
    """Hole radius beyond the pad's narrow copper radius, or no hole."""
    try:
        if pad.GetAttribute() not in (
            pcbnew.PAD_ATTRIB_PTH,
            pcbnew.PAD_ATTRIB_NPTH,
        ):
            return None
        drill = pad.GetDrillSize()
        size = pad.GetSize()
        drill_r = max(pcbnew.ToMM(drill.x), pcbnew.ToMM(drill.y)) / 2.0
        copper_r = min(pcbnew.ToMM(size.x), pcbnew.ToMM(size.y)) / 2.0
    except Exception:
        return None
    return max(0.0, drill_r - copper_r)


def _foreign_pad_margins(
    board: "pcbnew.BOARD",
    src_pad,
    *,
    floor_mm: float,
    half_width_mm: float,
    layer_id,
    strict_same_fp: bool = False,
    hole_clearance_mm: float = 0.25,
) -> tuple[list, list]:
    """Per-pad guard margins for copper on *src_pad*'s net: ``(path, tip)``.

    KiCad and KiCad Routing Tools both resolve a pair clearance as the LARGER of the
    two items' constraints, so a stub held only to the flat config floor can
    end inside a Power-netclass pad's keep-out: legal to this module, illegal
    to the router, which then abandons the net exactly as if the stub were
    absent (the rc7 CC2 signature).

    *path* margins guard the stamped copper itself: full pair clearance vs
    pads of OTHER footprints (a violation there is a real, unwaived DRC
    error). For the source footprint's OWN pads the margin depends on
    *strict_same_fp*: True holds the full pair clearance there too -- the
    final verify DRC does NOT waive footprint-internal pad-track violations
    (a stub grazing a same-footprint GND pad at 0.05 mm is a hard error, the
    KC-UXASHQ U1.5-vs-U1.6 signature) -- while False keeps the historical
    collision-only margin for pads genuinely hemmed in by their own row.
    Hole-bearing pads always enforce hole-to-copper clearance, even when their
    annulus is narrower than the drill. *tip* margins use the same rule.
    """
    src_cl = _own_clearance_mm(src_pad, layer_id, floor_mm)
    src_ref = _pad_ref(src_pad)
    collide_mm = half_width_mm + 0.05
    path: list = []
    tip: list = []
    for pad in _foreign_pads(board, src_pad.GetNetCode(), exclude=src_pad):
        pair = STAMP_CLEARANCE_GUARD_MM + max(
            floor_mm, src_cl, _own_clearance_mm(pad, layer_id, floor_mm)
        )
        overhang = _pad_hole_overhang_mm(pad)
        required = (
            pair
            if overhang is None
            else max(
                pair,
                hole_clearance_mm + overhang,
            )
        )
        same_fp = src_ref and _pad_ref(pad) == src_ref
        # Margins bound the track centerline, so include half its width.
        # Same-footprint leniency remains for dense SMD rows, but never relaxes
        # the hole-to-copper rule of a PTH/NPTH pad.
        path_mm = (
            (
                collide_mm
                if overhang is None
                else max(collide_mm, hole_clearance_mm + overhang + half_width_mm)
            )
            if (same_fp and not strict_same_fp)
            else required + half_width_mm
        )
        path.append((pad, int(pcbnew.FromMM(path_mm))))
        tip.append((pad, int(pcbnew.FromMM(required + half_width_mm))))
    return path, tip


def _point_clears_obstacles(obstacles: list, x_mm: float, y_mm: float) -> bool:
    pt = pcbnew.VECTOR2I(int(pcbnew.FromMM(x_mm)), int(pcbnew.FromMM(y_mm)))
    return not any(pad.HitTest(pt, margin) for pad, margin in obstacles)


def _segment_clears_obstacles(
    obstacles: list,
    a_mm: tuple[float, float],
    b_mm: tuple[float, float],
    *,
    step_mm: float = 0.1,
) -> bool:
    """Like :func:`_segment_clears_pads` but with a per-pad margin.

    *obstacles* is ``[(pad, margin_int)]`` from :func:`_foreign_pad_margins`.
    """
    ax, ay = a_mm
    bx, by = b_mm
    steps = max(1, int(((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5 / step_mm))
    for k in range(steps + 1):
        t = k / steps
        if not _point_clears_obstacles(obstacles, ax + (bx - ax) * t, ay + (by - ay) * t):
            return False
    return True


def _seg_seg_dist_mm(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> float:
    """Minimum distance between segments *a*->*b* and *c*->*d* (0 if crossing)."""

    def pt_seg(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
        qx, qy = x1 + t * dx, y1 + t * dy
        return ((px - qx) ** 2 + (py - qy) ** 2) ** 0.5

    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(*a, *b, *c)
    o2 = orient(*a, *b, *d)
    o3 = orient(*c, *d, *a)
    o4 = orient(*c, *d, *b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)):
        return 0.0
    return min(
        pt_seg(*c, *a, *b),
        pt_seg(*d, *a, *b),
        pt_seg(*a, *c, *d),
        pt_seg(*b, *c, *d),
    )


def _radial_escape_end(
    path_obstacles: list,
    tip_obstacles: list,
    start_mm: tuple[float, float],
    dir_unit: tuple[float, float],
    requested_mm: float,
    *,
    min_useful_mm: float,
    max_extra_mm: float = 2.5,
    step_mm: float = 0.05,
    inner_box: tuple[float, float, float, float] | None = None,
) -> tuple[float, float] | None:
    """End point of a radial escape whose tip KiCad Routing Tools can legally attach to.

    Marches outward sample by sample. The march stops at the first *path*
    collision (crossing a foreign pad is a short) or at the board's inner box.
    Among the sampled points whose *tip* margins all clear, prefer the first
    one at/past *requested_mm* -- extending up to *max_extra_mm* beyond it when
    the tip is still boxed in at the requested length -- else the farthest
    legal tip short of it. ``None`` when the direction never yields a legal
    tip at least *min_useful_mm* out.
    """
    sx, sy = start_mm
    ux, uy = dir_unit
    best_short = None
    d = step_mm
    while d <= requested_mm + max_extra_mm + 1e-9:
        x, y = sx + ux * d, sy + uy * d
        if not _points_within_box_mm([(x, y)], inner_box):
            break
        if not _point_clears_obstacles(path_obstacles, x, y):
            break
        if d >= min_useful_mm and _point_clears_obstacles(tip_obstacles, x, y):
            if d >= requested_mm - 1e-9:
                return (x, y)
            best_short = (x, y)
        d += step_mm
    return best_short


def _nearest_on_rect(
    px: float, py: float, x1: float, y1: float, x2: float, y2: float
) -> tuple[float, float]:
    """Nearest point on the rectangle border to an interior point."""
    dl, dr, dt, db = px - x1, x2 - px, py - y1, y2 - py
    m = min(dl, dr, dt, db)
    if m == dl:
        return (x1, py)
    if m == dr:
        return (x2, py)
    if m == dt:
        return (px, y1)
    return (px, y2)


def _rect_perimeter_path(
    b1: tuple[float, float],
    b2: tuple[float, float],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> list[tuple[float, float]]:
    """Waypoints walking the rectangle border from b1 to b2 the short way.

    Both endpoints lie on the border; the returned list is the corner points
    strictly between them (exclusive of b1/b2), in travel order. Staying on the
    border keeps the path outside everything the rectangle encloses.
    """
    w, h = x2 - x1, y2 - y1
    per = 2 * (w + h)
    corners = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def pos(p):
        x, y = p
        if abs(y - y1) <= abs(y - y2) and abs(y - y1) <= min(abs(x - x1), abs(x - x2)):
            return x - x1  # top edge
        if abs(x - x2) <= abs(x - x1) and abs(x - x2) <= min(abs(y - y1), abs(y - y2)):
            return w + (y - y1)  # right edge
        if abs(y - y2) <= abs(y - y1):
            return w + h + (x2 - x)  # bottom edge
        return 2 * w + h + (y2 - y)  # left edge

    corner_pos = [0.0, w, w + h, 2 * w + h]
    p1, p2 = pos(b1), pos(b2)
    cw = (p2 - p1) % per
    ccw = per - cw
    out = []
    if cw <= ccw:
        for cpos, c in zip(corner_pos, corners):
            if (cpos - p1) % per < cw and (cpos - p1) % per > 1e-9:
                out.append((c, (cpos - p1) % per))
    else:
        for cpos, c in zip(corner_pos, corners):
            if (p1 - cpos) % per < ccw and (p1 - cpos) % per > 1e-9:
                out.append((c, (p1 - cpos) % per))
    out.sort(key=lambda t: t[1])
    return [c for c, _ in out]


def _pads_bbox_mm(pads: list, margin_mm: float = 0.0) -> tuple[float, float, float, float]:
    """``(x1, y1, x2, y2)`` in mm enclosing every pad's box, grown by *margin_mm*.

    This is the footprint's *pad field* -- deliberately not ``fp.GetBoundingBox()``,
    which silkscreen, courtyard and the reference designator inflate well beyond
    the copper and make asymmetric.
    """
    boxes = [p.GetBoundingBox() for p in pads]
    return (
        min(pcbnew.ToMM(b.GetLeft()) for b in boxes) - margin_mm,
        min(pcbnew.ToMM(b.GetTop()) for b in boxes) - margin_mm,
        max(pcbnew.ToMM(b.GetRight()) for b in boxes) + margin_mm,
        max(pcbnew.ToMM(b.GetBottom()) for b in boxes) + margin_mm,
    )


def _board_inner_box_mm(
    board: "pcbnew.BOARD",
) -> tuple[float, float, float, float] | None:
    """``(x1, y1, x2, y2)`` in mm that stamped copper must stay inside.

    The Edge.Cuts bounding box shrunk by the board's copper-to-edge clearance.
    Locked copper outside the outline is fatal: KiCad Routing Tools 1.9.0 reads the
    corner as "wire corner outside board" and hangs without producing a routed session
    (the brief-2 VOLTAGE SELECT leaf burned its whole build budget this way).
    Exact for the rectangular outlines KiCraft generates; for a non-rectangular
    outline the bbox is larger than the board, so this check can only
    under-reject, never drop a valid stub. ``None`` when the board has no
    outline yet (nothing to violate).
    """
    box = board.GetBoardEdgesBoundingBox()
    if box.GetWidth() <= 0 or box.GetHeight() <= 0:
        return None
    try:
        inset = max(pcbnew.ToMM(board.GetDesignSettings().m_CopperEdgeClearance), 0.05)
    except AttributeError:
        inset = 0.05
    x1 = pcbnew.ToMM(box.GetLeft()) + inset
    y1 = pcbnew.ToMM(box.GetTop()) + inset
    x2 = pcbnew.ToMM(box.GetRight()) - inset
    y2 = pcbnew.ToMM(box.GetBottom()) - inset
    if x1 >= x2 or y1 >= y2:
        return None
    return (x1, y1, x2, y2)


def _points_within_box_mm(
    points: list[tuple[float, float]],
    box: tuple[float, float, float, float] | None,
) -> bool:
    """True when every point lies inside *box* (or there is no box to violate).

    Checking only the endpoints is sufficient: the box is convex, so a straight
    segment between two contained points is contained too.
    """
    if box is None:
        return True
    x1, y1, x2, y2 = box
    return all(x1 <= x <= x2 and y1 <= y <= y2 for x, y in points)


def perimeter_tie_specs(
    board: "pcbnew.BOARD",
    ref: str,
    net_names: list[str] | None = None,
    *,
    margin_mm: float = 1.0,
    layer: str = "F.Cu",
    min_pads: int = 2,
    clearance_mm: float = 0.153,
) -> list[BreakoutSpec]:
    """Tie a footprint's same-net pads with a path routed around its bbox.

    For each net that has >= *min_pads* pads on *ref* (restrict with
    *net_names*), connect the two farthest-apart pads: directly when the
    straight segment crosses no foreign pad, otherwise with a waypoint path
    that leaves each pad, hops just outside the footprint's bounding box, and
    walks the box perimeter between them -- clamped to the board outline so no
    locked copper is ever stamped off the board. That keeps a power pour from
    fragmenting: the connector's spread power pads (e.g. USB-C VBUS) become
    one net island.
    """
    specs: list[BreakoutSpec] = []
    fp = next((f for f in board.GetFootprints() if f.GetReferenceAsString() == ref), None)
    if fp is None:
        return specs
    pads_all = list(fp.Pads())
    if not pads_all:
        return specs
    # Walk the *pad field*, not fp.GetBoundingBox(): the latter is grown and made
    # asymmetric by silkscreen, courtyard and the reference designator, which can
    # put the nearest border on the far side of the part and send a tie's lead-in
    # leg straight across its other pads (a short). The two farthest same-net pads
    # are always hull-extremal, so their perpendicular lead-ins to a box that hugs
    # the copper -- and the border walk itself -- stay clear of every pad.
    x1, y1, x2, y2 = _pads_bbox_mm(pads_all, margin_mm)

    by_net: dict[str, list] = {}
    for pad in fp.Pads():
        n = pad.GetNetname()
        if not n or (net_names is not None and n not in net_names):
            continue
        by_net.setdefault(n, []).append(pad)

    inner_box = _board_inner_box_mm(board)

    for net, pads in by_net.items():
        if len(pads) < min_pads:
            continue
        pts = [(pcbnew.ToMM(p.GetPosition().x), pcbnew.ToMM(p.GetPosition().y)) for p in pads]
        # Farthest-apart pair.
        i, j, best = 0, 1, -1.0
        for a in range(len(pts)):
            for b in range(a + 1, len(pts)):
                d = (pts[a][0] - pts[b][0]) ** 2 + (pts[a][1] - pts[b][1]) ** 2
                if d > best:
                    best, i, j = d, a, b
        p1, p2 = pts[i], pts[j]
        # A same-net pad is a valid landing, not an obstacle -- so when the
        # straight farthest-pad segment clears every FOREIGN pad (e.g. a DIP
        # switch's adjacent commons, where the only pad in between is on the
        # same net), tie directly. Shortest copper, and immune to the off-board
        # walk below. Both endpoints are placed pads, so the segment is always
        # on the board.
        path_obs, _tip_obs = _foreign_pad_margins(
            board,
            pads[i],
            floor_mm=clearance_mm,
            half_width_mm=0.0765,
            layer_id=_LAYERS.get(layer, pcbnew.F_Cu),
        )
        if _segment_clears_obstacles(path_obs, p1, p2):
            specs.append(
                BreakoutSpec(ref=ref, pad=pads[i].GetNumber(), waypoints=[p2], layer=layer)
            )
            continue
        # A pad field closer to the board edge than its margin puts part of the
        # walk rectangle off the board -- and stamping locked off-board copper
        # hangs KiCad Routing Tools. Clamp the rectangle to the inner box; valid only
        # while the clamped border still clears every pad it must walk around
        # (the raw pad bbox plus clearance). When even that fails, skip the tie
        # loudly: KiCad Routing Tools still routes the net, the pour just may fragment.
        rx1, ry1, rx2, ry2 = _pads_bbox_mm(pads_all, 0.0)
        wx1, wy1, wx2, wy2 = x1, y1, x2, y2
        if inner_box is not None:
            wx1, wy1 = max(wx1, inner_box[0]), max(wy1, inner_box[1])
            wx2, wy2 = min(wx2, inner_box[2]), min(wy2, inner_box[3])
        if not (
            wx1 <= rx1 - clearance_mm
            and wy1 <= ry1 - clearance_mm
            and wx2 >= rx2 + clearance_mm
            and wy2 >= ry2 + clearance_mm
        ):
            print(
                f"  WARNING: perimeter tie {ref} net {net} skipped: "
                "pad field too close to the board edge for an on-board walk"
            )
            continue
        b1 = _nearest_on_rect(p1[0], p1[1], wx1, wy1, wx2, wy2)
        b2 = _nearest_on_rect(p2[0], p2[1], wx1, wy1, wx2, wy2)
        corners = _rect_perimeter_path(b1, b2, wx1, wy1, wx2, wy2)
        waypoints = [b1, *corners, b2, p2]
        specs.append(
            BreakoutSpec(ref=ref, pad=pads[i].GetNumber(), waypoints=waypoints, layer=layer)
        )
    return specs


def auto_power_tie_specs(
    board: "pcbnew.BOARD",
    cfg: dict[str, Any] | None = None,
) -> list[BreakoutSpec]:
    """Perimeter-tie specs for every footprint with spread power-net pads.

    Auto-detects the case a power pour can't close on a 2-layer board: a
    footprint (typically a connector) with >= 2 pads on a power rail that signal
    traces fragment the pour around. Skips GND (handled by the GND plane +
    thermal vias) and single-power-pad parts. Footprint refs in
    ``power_tie_exclude_refs`` are skipped.
    """
    cfg = cfg or {}
    gnd_name = cfg.get("gnd_zone_net", "GND")
    exclude = set(cfg.get("power_tie_exclude_refs", []) or [])
    margin = float(cfg.get("power_tie_margin_mm", 1.0))
    from kicraft.design.models import is_power_or_ground_name

    specs: list[BreakoutSpec] = []
    for fp in board.GetFootprints():
        ref = fp.GetReferenceAsString()
        if ref in exclude:
            continue
        power_nets = set()
        counts: dict[str, int] = {}
        for pad in fp.Pads():
            n = pad.GetNetname()
            if not n or n == gnd_name:
                continue
            if is_power_or_ground_name(n):
                counts[n] = counts.get(n, 0) + 1
        power_nets = {n for n, c in counts.items() if c >= 2}
        if not power_nets:
            continue
        specs.extend(
            perimeter_tie_specs(
                board,
                ref,
                net_names=sorted(power_nets),
                margin_mm=margin,
                clearance_mm=fab_floors(cfg)["clearance_mm"],
            )
        )
    return specs


def shield_tie_specs(
    board: "pcbnew.BOARD",
    cfg: dict[str, Any] | None = None,
) -> list[BreakoutSpec]:
    """Tie each netted through-hole pad to its nearest same-net pad.

    A connector's through-hole shield/shell legs (USB-C TYPE-C-31 pads 1-4 on
    GND) sit where neither GND plane can reach them: the F.Cu fill is walled
    out of the fine-pitch pad row by the Power-netclass clearance, and the
    B.Cu fill around the slot holes loses its thermal spokes -- so the legs
    facing the pad row survive as unconnected ratlines on an otherwise-routed
    board. A short locked same-net track to the nearest pad (preferring an
    SMD pad, which the pour and router do reach) closes each leg
    deterministically. The stamp-time guards in :func:`add_breakout_stubs`
    drop any tie whose straight path would cross foreign copper.
    """
    cfg = cfg or {}
    if not cfg.get("shield_tie_enabled", True):
        return []
    exclude = set(cfg.get("shield_tie_exclude_refs", []) or [])
    max_mm = float(cfg.get("shield_tie_max_mm", 4.0))

    specs: list[BreakoutSpec] = []
    for fp in board.GetFootprints():
        ref = fp.GetReferenceAsString()
        if ref in exclude:
            continue
        pads = list(fp.Pads())
        for pad in pads:
            if pad.GetAttribute() != pcbnew.PAD_ATTRIB_PTH:
                continue
            net_code = pad.GetNetCode()
            if net_code == 0:
                continue
            mates = [q for q in pads if q is not pad and q.GetNetCode() == net_code]
            if not mates:
                continue
            smd = [q for q in mates if q.GetAttribute() == pcbnew.PAD_ATTRIB_SMD]
            pool = smd or mates
            p = pad.GetPosition()
            mate = min(
                pool,
                key=lambda m: (m.GetPosition().x - p.x) ** 2 + (m.GetPosition().y - p.y) ** 2,
            )
            mp = mate.GetPosition()
            d_mm = ((pcbnew.ToMM(mp.x - p.x)) ** 2 + (pcbnew.ToMM(mp.y - p.y)) ** 2) ** 0.5
            # Touching pads are already connected; a far mate means this is not
            # the shield-leg shape and a straight tie would cross the part.
            if d_mm < 0.1 or d_mm > max_mm:
                continue
            # The tie joins the legs to the SMD pad, but on the PARENT that
            # whole cluster can still be an island: the F.Cu pour is
            # clearance-walled out of the connector area and the B.Cu plane
            # loses its spokes to the slot holes -- the round then burns on
            # 1-3 GND ratlines until the build clock dies. A via at the SMD
            # end bonds the island straight down to the B.Cu plane. (Never on
            # a PTH mate: that would drill into the mate's own hole.)
            specs.append(
                BreakoutSpec(
                    ref=ref,
                    pad=pad.GetNumber(),
                    waypoints=[(pcbnew.ToMM(mp.x), pcbnew.ToMM(mp.y))],
                    via_at_end=bool(smd) and bool(cfg.get("shield_tie_via", True)),
                )
            )
    return specs


def add_breakout_stubs(
    pcb_path: str,
    specs: list[BreakoutSpec],
    *,
    cfg: dict[str, Any] | None = None,
    lock: bool = True,
) -> dict[str, Any]:
    """Lay each spec as locked track(s) (+ optional via) on the board, save.

    Returns ``{stubs, segments, vias, skipped}``. A spec whose pad cannot be
    found, or whose net is unset, is skipped (counted, never raises) so a
    finishing step never fails the board.
    """
    cfg = cfg or {}
    default_w = fab_floors(cfg)["track_mm"]
    summary: dict[str, Any] = {
        "stubs": 0,
        "segments": 0,
        "vias": 0,
        "skipped": [],
    }
    if not specs:
        return summary

    board = pcbnew.LoadBoard(pcb_path)
    rule_areas = collect_track_via_rule_areas(board)
    inner_box = _board_inner_box_mm(board)
    floor_mm = fab_floors(cfg)["clearance_mm"]
    hole_clearance_mm = float(cfg.get("hole_clearance_min_mm", 0.25))
    # Copper stamped by THIS call, for mutual clearance between specs: a tie
    # and a later escape stub are stamped blind to each other, and two locked
    # tracks 0.05 mm apart are a violation no router pass can repair (and the
    # gate cannot waive: a track-track DRC block names no footprint).
    # (net_code, a_mm, b_mm, half_width_mm, own_clearance_mm, layer) per segment.
    stamped: list[tuple[int, tuple, tuple, float, float, Any]] = []
    # Copper already ON the board before this call: empty for a fresh leaf,
    # but the parent re-tie pass stamps into a board carrying every leaf's
    # routed traces -- crossing one of those is a short the foreign-PAD guard
    # cannot see. A via is a zero-length segment on every copper layer.
    for t in board.GetTracks():
        is_via = t.GetClass() == "PCB_VIA"
        if is_via:
            p = (pcbnew.ToMM(t.GetPosition().x), pcbnew.ToMM(t.GetPosition().y))
            a_mm, b_mm, t_layer = p, p, None
        else:
            a_mm = (pcbnew.ToMM(t.GetStart().x), pcbnew.ToMM(t.GetStart().y))
            b_mm = (pcbnew.ToMM(t.GetEnd().x), pcbnew.ToMM(t.GetEnd().y))
            t_layer = t.GetLayer()
        try:
            t_half_w = pcbnew.ToMM(t.GetWidth()) / 2.0
        except TypeError:
            t_half_w = 0.3
        stamped.append((t.GetNetCode(), a_mm, b_mm, t_half_w, floor_mm, t_layer))

    # Existing drilled holes (vias + PTH pads): a tip via's hole wall must
    # keep the board's hole-to-hole minimum from every one of them. Vias
    # stamped by THIS call join the list as they land. via_pts additionally
    # tracks via NETS: a second tie ending on a pad that already carries a
    # same-net via needs its track but not a duplicate via.
    _h2h = pcbnew.ToMM(board.GetDesignSettings().m_HoleToHoleMin)
    hole_min_mm = _h2h if _h2h > 0 else float(cfg.get("hole_to_hole_min_mm", 0.25))
    holes: list[tuple[float, float, float]] = []
    via_pts: list[tuple[int, float, float]] = []
    for t in board.GetTracks():
        if t.GetClass() == "PCB_VIA":
            p = t.GetPosition()
            x_mm, y_mm = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
            holes.append((x_mm, y_mm, pcbnew.ToMM(t.GetDrillValue()) / 2.0))
            via_pts.append((t.GetNetCode(), x_mm, y_mm))
    for _fp in board.GetFootprints():
        for _p in _fp.Pads():
            ds = _p.GetDrillSize()
            if ds.x > 0 or ds.y > 0:
                pp = _p.GetPosition()
                holes.append(
                    (
                        pcbnew.ToMM(pp.x),
                        pcbnew.ToMM(pp.y),
                        max(pcbnew.ToMM(ds.x), pcbnew.ToMM(ds.y)) / 2.0,
                    )
                )

    def _pt(xy: tuple[float, float]):
        return pcbnew.VECTOR2I(pcbnew.FromMM(xy[0]), pcbnew.FromMM(xy[1]))

    for spec in specs:
        if spec.start_xy is not None:
            # Free-coordinate anchor: the net's nearest pad stands in for
            # netclass clearance / foreign-pad margins; geometry starts at
            # start_xy. Waypoints are mandatory (no footprint to escape).
            if not spec.waypoints:
                summary["skipped"].append(f"{spec.ref}.{spec.pad}:free_anchor_needs_waypoints")
                continue
            fp, pad = _nearest_same_net_pad(board, spec.net or "", spec.start_xy)
            if pad is None:
                summary["skipped"].append(f"{spec.ref}.{spec.pad}:net_not_found")
                continue
        else:
            fp, pad = _find_pad(board, spec.ref, spec.pad, spec.near_xy)
            if pad is None:
                summary["skipped"].append(f"{spec.ref}.{spec.pad}:pad_not_found")
                continue
        net_code = pad.GetNetCode()
        if net_code == 0:
            summary["skipped"].append(f"{spec.ref}.{spec.pad}:no_net")
            continue
        layer = _LAYERS.get(spec.layer)
        if layer is None:
            summary["skipped"].append(f"{spec.ref}.{spec.pad}:bad_layer:{spec.layer}")
            continue
        width = pcbnew.FromMM(spec.width_mm if spec.width_mm else default_w)
        half_width_mm = pcbnew.ToMM(width) / 2.0
        src_cl_mm = _own_clearance_mm(pad, layer, floor_mm)
        # A free-coordinate anchor does not start inside the stand-in pad's
        # own row, so the same-footprint leniency is unjustified there: a
        # USB_DN tie anchored at a track end grazed the stand-in footprint's
        # USB_DP pad at 0.05 mm under the relaxed margins (run_10).
        path_obs, tip_obs = _foreign_pad_margins(
            board,
            pad,
            floor_mm=floor_mm,
            half_width_mm=half_width_mm,
            layer_id=layer,
            strict_same_fp=spec.start_xy is not None,
            hole_clearance_mm=hole_clearance_mm,
        )

        def _track_hits_keepout(points: list[tuple[float, float]]) -> bool:
            return any(
                track_intersects_rule_area(a, b, half_width_mm, area)
                for a, b in zip(points, points[1:])
                for area in rule_areas
            )

        def _via_hits_keepout(xy: tuple[float, float]) -> bool:
            return any(via_intersects_rule_area(xy, via_r_mm, area) for area in rule_areas)

        def _conflicts_with_copper(points: list[tuple[float, float]]) -> bool:
            """True when the path runs too close to other-net stamped/board copper."""
            for a, b in zip(points, points[1:]):
                for o_net, o_a, o_b, o_hw, o_cl, o_layer in stamped:
                    if o_net == net_code:
                        continue
                    if o_layer is not None and o_layer != layer:
                        continue
                    need = max(floor_mm, src_cl_mm, o_cl) + half_width_mm + o_hw
                    if _seg_seg_dist_mm(a, b, o_a, o_b) < need:
                        return True
            return False

        # A tip via is wider than the track (0.6 mm barrel vs ~0.15 mm trace)
        # and drills a hole: it must clear EVERY foreign pad by pair clearance
        # + via radius (tip margins only cover the track half-width), stamped/
        # board copper on all layers, and every existing drilled hole by the
        # board's hole-to-hole minimum. A spec whose via cannot land is
        # dropped whole BEFORE its segments stamp -- a stub with no plane via
        # is dead copper (the pour island it would join is removed).
        via_size_mm = float(spec.via_size_mm or cfg.get("via_size_mm", 0.6))
        via_drill_size_mm = float(spec.via_drill_mm or cfg.get("via_drill_mm", 0.3))
        via_r_mm = via_size_mm / 2.0
        via_drill_r_mm = via_drill_size_mm / 2.0
        via_obs = (
            [(p, m + int(pcbnew.FromMM(via_r_mm - half_width_mm))) for p, m in tip_obs]
            if spec.via_at_end
            else []
        )

        def _via_redundant(xy: tuple[float, float]) -> bool:
            """A same-net via already sits under the stub end: the plane bond
            exists, so stamp the track but not a duplicate via (two shield
            legs tying to the same SMD pad is the common case)."""
            return any(
                n == net_code and ((xy[0] - vx) ** 2 + (xy[1] - vy) ** 2) ** 0.5 <= via_r_mm
                for n, vx, vy in via_pts
            )

        def _via_fits(xy: tuple[float, float]) -> bool:
            if not _point_clears_obstacles(via_obs, xy[0], xy[1]):
                return False
            for o_net, o_a, o_b, o_hw, o_cl, _o_layer in stamped:
                if o_net == net_code:
                    continue  # a via spans all layers: check every foreign seg
                need = max(floor_mm, src_cl_mm, o_cl) + via_r_mm + o_hw
                if _seg_seg_dist_mm(xy, xy, o_a, o_b) < need:
                    return False
            return not any(
                ((xy[0] - hx) ** 2 + (xy[1] - hy) ** 2) ** 0.5 < hr + via_drill_r_mm + hole_min_mm
                for hx, hy, hr in holes
            )

        if spec.start_xy is not None:
            start_mm = (float(spec.start_xy[0]), float(spec.start_xy[1]))
        else:
            pad_pos = pad.GetPosition()
            start_mm = (pcbnew.ToMM(pad_pos.x), pcbnew.ToMM(pad_pos.y))
        if spec.waypoints:
            points = [start_mm, *spec.waypoints]
            # Hard invariant: never stamp a path that crosses a pad of another
            # net (or a no-net pad) -- that is a short. The tie geometry routes
            # clear of the source footprint's own pads, but a curated path or a
            # neighbour the geometry could not see may still intrude. A partial
            # tie is useless, so drop the whole spec rather than clip it.
            if not all(
                _segment_clears_obstacles(path_obs, a, b) for a, b in zip(points, points[1:])
            ):
                summary["skipped"].append(f"{spec.ref}.{spec.pad}:waypoint_crosses_pad")
                continue
            if _track_hits_keepout(points):
                summary["skipped"].append(f"{spec.ref}.{spec.pad}:track_keepout")
                continue
            if _conflicts_with_copper(points):
                summary["skipped"].append(f"{spec.ref}.{spec.pad}:conflicts_with_stamped_stub")
                continue
            if spec.via_at_end and not _via_redundant(points[-1]) and _via_hits_keepout(points[-1]):
                summary["skipped"].append(f"{spec.ref}.{spec.pad}:via_keepout")
                continue
            if spec.via_at_end and not _via_redundant(points[-1]) and not _via_fits(points[-1]):
                summary["skipped"].append(f"{spec.ref}.{spec.pad}:via_blocked")
                continue
        else:
            fc = fp.GetPosition()
            cx, cy = pcbnew.ToMM(fc.x), pcbnew.ToMM(fc.y)
            dx, dy = start_mm[0] - cx, start_mm[1] - cy
            norm = (dx * dx + dy * dy) ** 0.5
            dir_unit = (1.0, 0.0) if norm < 1e-6 else (dx / norm, dy / norm)
            # March out radially: never cross a neighbouring pad (a short), and
            # only stamp a stub whose TIP clears every foreign pad by the
            # netclass pair clearance -- a tip the router cannot legally attach
            # to leaves the net exactly as unrouted as no stub at all.
            # The radial direction is right for a part whose pads ring its
            # centre (QFN) but for a connector ROW it can run diagonally ALONG
            # the row, colliding forever (the USB-C CC2 signature) -- so fall
            # back to the four axis directions. A direction whose tip is legal
            # but whose run lands beside already-stamped copper (e.g. the VBUS
            # perimeter tie one pad-row out) is rejected HERE so the next
            # direction still gets its chance.
            # Two margin rounds: STRICT same-footprint margins first -- the
            # verify DRC does not waive a stub grazing a sibling pad (the
            # diagonal-stub-past-the-GND-pad signature) -- then the historical
            # collision-only margins, so a pad genuinely walled in by its own
            # row still escapes.
            strict_path_obs, _ = _foreign_pad_margins(
                board,
                pad,
                floor_mm=floor_mm,
                half_width_mm=half_width_mm,
                layer_id=layer,
                strict_same_fp=True,
                hole_clearance_mm=hole_clearance_mm,
            )
            points = None
            track_keepout_seen = False
            via_keepout_seen = False
            for path_set in (strict_path_obs, path_obs):
                for du in (dir_unit, (1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)):
                    end = _radial_escape_end(
                        path_set,
                        tip_obs,
                        start_mm,
                        du,
                        spec.length_mm,
                        min_useful_mm=max(floor_mm, pcbnew.ToMM(width)),
                        inner_box=inner_box,
                    )
                    if end is None:
                        continue
                    cand = [start_mm, end]
                    if _track_hits_keepout(cand):
                        track_keepout_seen = True
                        continue
                    if _conflicts_with_copper(cand):
                        continue
                    if spec.via_at_end and not _via_redundant(end) and _via_hits_keepout(end):
                        via_keepout_seen = True
                        continue
                    if spec.via_at_end and not _via_redundant(end) and not _via_fits(end):
                        continue
                    points = cand
                    break
                if points is not None:
                    break
            if points is None:
                reason = (
                    "track_keepout"
                    if track_keepout_seen
                    else "via_keepout"
                    if via_keepout_seen
                    else "no_safe_radial_escape"
                )
                summary["skipped"].append(f"{spec.ref}.{spec.pad}:{reason}")
                continue

        # Hard invariant: never stamp locked copper outside the board outline.
        # KiCad Routing Tools 1.9.0 hangs (no routed session, no error) on a locked wire corner
        # off the board, burning the leaf's whole routing budget. The start is
        # exempt -- it is a placed pad's centre, the placement gate's job. The
        # box is convex, so in-box endpoints mean in-box segments.
        if not _points_within_box_mm(points[1:], inner_box):
            summary["skipped"].append(f"{spec.ref}.{spec.pad}:off_board")
            continue

        for a, b in zip(points, points[1:]):
            track = pcbnew.PCB_TRACK(board)
            track.SetStart(_pt(a))
            track.SetEnd(_pt(b))
            track.SetWidth(width)
            track.SetLayer(layer)
            track.SetNetCode(net_code)
            if lock:
                track.SetLocked(True)
            board.Add(track)
            summary["segments"] += 1
            stamped.append((net_code, a, b, half_width_mm, src_cl_mm, layer))

        if spec.via_at_end and not _via_redundant(points[-1]):
            via = pcbnew.PCB_VIA(board)
            via.SetPosition(_pt(points[-1]))
            via.SetDrill(pcbnew.FromMM(via_drill_size_mm))
            try:
                via.SetWidth(pcbnew.FromMM(via_size_mm))
            except TypeError:
                via.SetWidth(layer, pcbnew.FromMM(via_size_mm))
            via.SetNetCode(net_code)
            if lock:
                via.SetLocked(True)
            board.Add(via)
            summary["vias"] += 1
            # Later specs must respect this via: its hole (hole-to-hole), its
            # barrel (a zero-length all-layer segment for copper checks), and
            # its net (so a tie ending here skips its now-redundant via).
            holes.append((points[-1][0], points[-1][1], via_drill_r_mm))
            stamped.append((net_code, points[-1], points[-1], via_r_mm, src_cl_mm, None))
            via_pts.append((net_code, points[-1][0], points[-1][1]))

        summary["stubs"] += 1

    board.Save(pcb_path)
    return summary
