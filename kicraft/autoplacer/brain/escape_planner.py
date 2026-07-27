"""Escape planning for trapped fine-pitch pads -- pure geometry, no pcbnew.

A fine-pitch package with an inner pad ring (nRF52840 aQFN-73, ESP32, RP2040,
STM32) has netted pads whose only exits are narrow *designed* lanes. KiCraft's
historical escape stamping (``breakout_stubs._radial_escape_end``) marches in
ONE fixed direction -- radially out from the footprint centre -- and, when that
direction is blocked, stamps the farthest legal point it reached. On a ring
package that is a 0.2 mm nub inside a dead channel: it connects nothing, becomes
a foreign-copper obstacle, and turns the DRC edge into ``Track [net] <-> Pad``.
The pads were *geometrically unreachable before FreeRouting ever ran*, and
nothing said so -- the failure surfaced as router exhaustion at leaf round 9.

This module answers the question the stamper never asked: **for this pad field
and this rule set, how does each netted pad get out?** It is deliberately pure
geometry over plain dataclasses so it can be unit-tested against a checked-in
pad table (``tests/data/aqfn73_pads.json``) with no board, no KiCad and no
router in the loop -- and so the 2 um / 15 um margins that decide the aQFN-73
verdicts are pinned by tests instead of discovered by a self-eval batch.

Per netted pad the planner returns exactly one :class:`Escape`:

``open``
    The pad sits on the pad field's outer row, so open copper is directly in
    front of it and there is nothing to thread. Nothing to stamp -- the router
    escapes these unaided, and always has.
``tie``
    An inner pad with a legal short segment to a **same-net** pad that is itself
    already out (an outer-row pad, or the exposed pad the GND pour stitches):
    the classic inner-GND-pad-to-exposed-pad case. Not an escape at all -- a
    direct connection, and the shortest correct answer whenever it exists.
``via``
    The classic dog-bone fanout: a legal via centre beside the pad, from which
    the net continues on the other layer. This is the *uniform* strategy a ring
    package is designed for and needs no lane coordination -- the disc search
    honours clearance to the exposed pad, the ring neighbours, every foreign
    pad, and every fanout via already assigned on this footprint.
``lane``
    On-layer fallback: a straight or one-bend path through a **lane** (a gap in
    the outer pad row wide enough to carry a track), rationed by capacity so two
    pads never fight over one lane.
``infeasible``
    No exit exists at this rule set. The caller must stamp **nothing** (a nub is
    an obstacle plus a false "partially routed" signal) and report it honestly.

Everything is decided by exact clearance arithmetic against the pad rectangles;
the lane model only *proposes and rations* candidate exits -- it never decides
legality on its own.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

__all__ = [
    "Pad",
    "Rules",
    "Lane",
    "Escape",
    "EscapePlan",
    "plan_escapes",
    "find_lanes",
    "via_fanout_center",
    "pads_from_dicts",
    "planning_rules",
    "capability_rules_from_config",
]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Pad:
    """One pad of the footprint under analysis.

    Positions/sizes are in mm in ONE consistent frame (the caller's choice --
    board frame or footprint frame). ``w``/``h`` are the pad's **axis-aligned
    extent**, so a rotated pad is passed as its bounding box: conservative by
    construction, which is the right bias for a clearance decision.

    ``layers`` is the set of copper layers the pad occupies (an SMD pad is on
    one, a PTH pad on both). Tracks are checked only against pads sharing their
    layer; a via spans every layer and is checked against all of them.
    ``drill`` (> 0) makes the pad a drilled hole for hole-to-hole checks.
    """

    number: str
    net: str
    x: float
    y: float
    w: float
    h: float
    layers: frozenset[str] = frozenset({"F.Cu"})
    drill: float = 0.0


@dataclass(frozen=True, slots=True)
class Rules:
    """The fab/routing rule set an escape is evaluated against.

    ``track_mm``/``clearance_mm`` size the on-layer corridor (a lone trace needs
    ``track + 2*clearance`` of gap); ``via_diameter_mm``/``via_drill_mm`` size
    the fanout via. ``hole_to_hole_mm`` is the board's drill-to-drill minimum and
    ``hole_clearance_mm`` its hole-to-COPPER minimum -- the rule that actually
    binds a small fanout via: at a 0.2 mm drill, shrinking the via diameter stops
    buying room because the drill's 0.25 mm copper keep-out takes over.
    """

    track_mm: float = 0.153
    clearance_mm: float = 0.153
    via_diameter_mm: float = 0.35
    via_drill_mm: float = 0.2
    hole_to_hole_mm: float = 0.25
    hole_clearance_mm: float = 0.25

    @property
    def track_keepout_mm(self) -> float:
        """Distance a track CENTERLINE must hold from a foreign pad edge."""
        return self.clearance_mm + self.track_mm / 2.0

    @property
    def via_keepout_mm(self) -> float:
        """Distance a via CENTRE must hold from a foreign pad edge.

        The larger of the copper rule (annulus + clearance) and the drill rule
        (hole radius + hole-to-copper). Both are real KiCad DRC checks and both
        bite on a fine-pitch fanout.
        """
        return max(
            self.clearance_mm + self.via_diameter_mm / 2.0,
            self.hole_clearance_mm + self.via_drill_mm / 2.0,
        )

    def lane_capacity(self, gap_mm: float) -> int:
        """How many tracks fit side by side in a ``gap_mm`` opening.

        ``floor((gap - c) / (track + c))`` -- one clearance on each flank plus
        one clearance between neighbours. Capacity >= 1 is exactly
        ``gap >= track + 2*clearance``, so this single formula also decides
        whether a lane is usable at all.
        """
        step = self.track_mm + self.clearance_mm
        if step <= 0:
            return 0
        return max(0, int(math.floor((gap_mm - self.clearance_mm) / step + 1e-9)))


# --------------------------------------------------------------------------- #
# Outputs
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Lane:
    """A gap in the outer pad row that a track can leave through.

    ``side`` is one of ``x-``/``x+``/``y-``/``y+`` (the pad-field edge the lane
    crosses); ``lo``/``hi`` bound the gap along that edge and ``gap_mm`` is the
    true edge-to-edge opening. ``capacity`` is how many tracks the rule set
    admits -- the number that makes a 0.75 mm depopulated lane carry one track
    at 0.153 mm and two at 0.127 mm.
    """

    side: str
    lo: float
    hi: float
    gap_mm: float
    capacity: int

    def key(self) -> tuple[str, float, float]:
        return (self.side, round(self.lo, 4), round(self.hi, 4))


@dataclass(frozen=True, slots=True)
class Escape:
    """The planned exit for one netted pad. ``kind`` is documented module-top."""

    pad: str
    net: str
    kind: str
    polyline: tuple[tuple[float, float], ...] = ()
    via_center: tuple[float, float] | None = None
    lane: Lane | None = None
    margin_mm: float = 0.0
    reason: str = ""
    # Other kinds that were ALSO legal for this pad. Recorded (not acted on) so
    # a rule-set comparison can say "this pad gains an on-layer exit at 0.127"
    # without re-planning, which is the whole point of pinning the margins.
    alternatives: tuple[str, ...] = ()

    @property
    def feasible(self) -> bool:
        return self.kind != "infeasible"

    @property
    def needs_stamp(self) -> bool:
        """``open`` pads are the router's job; everything feasible else is ours."""
        return self.kind in ("tie", "via", "lane")

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {"pad": self.pad, "net": self.net, "kind": self.kind}
        if self.polyline:
            d["polyline"] = [[round(x, 4), round(y, 4)] for x, y in self.polyline]
        if self.via_center is not None:
            d["via_center"] = [round(self.via_center[0], 4), round(self.via_center[1], 4)]
        if self.lane is not None:
            d["lane"] = {
                "side": self.lane.side,
                "gap_mm": round(self.lane.gap_mm, 4),
                "capacity": self.lane.capacity,
            }
        if self.margin_mm:
            d["margin_mm"] = round(self.margin_mm, 4)
        if self.reason:
            d["reason"] = self.reason
        return d


@dataclass(slots=True)
class EscapePlan:
    """Every netted pad's verdict for one footprint at one rule set.

    Keyed by a per-pad *uid*, which is the pad number when that number is unique
    (the normal case) and ``number#i`` when it is not -- module footprints
    routinely number every ground pad "GND", and keying by number alone would
    silently keep one verdict and drop the rest.
    """

    escapes: dict[str, Escape] = field(default_factory=dict)
    lanes: list[Lane] = field(default_factory=list)
    field_box: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @property
    def trapped(self) -> list[str]:
        """Pads that cannot simply be routed away by the router as-is."""
        return sorted(
            e.pad for e in self.escapes.values() if e.kind != "open"
        )

    @property
    def infeasible(self) -> list[str]:
        return sorted(e.pad for e in self.escapes.values() if e.kind == "infeasible")

    @property
    def stampable(self) -> list[Escape]:
        return sorted(
            (e for e in self.escapes.values() if e.needs_stamp), key=lambda e: e.pad
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "escapes": [e.to_dict() for e in sorted(self.escapes.values(), key=lambda e: e.pad)],
            "lanes": [
                {"side": ln.side, "gap_mm": round(ln.gap_mm, 4), "capacity": ln.capacity}
                for ln in self.lanes
            ],
            "trapped": self.trapped,
            "infeasible": self.infeasible,
        }


# --------------------------------------------------------------------------- #
# Geometry primitives
# --------------------------------------------------------------------------- #

_Rect = tuple[float, float, float, float]


def _rect(p: Pad) -> _Rect:
    return (p.x - p.w / 2.0, p.y - p.h / 2.0, p.x + p.w / 2.0, p.y + p.h / 2.0)


def _dist_point_rect(px: float, py: float, r: _Rect) -> float:
    dx = max(r[0] - px, 0.0, px - r[2])
    dy = max(r[1] - py, 0.0, py - r[3])
    return math.hypot(dx, dy)


def _seg_seg_dist(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> float:
    """Minimum distance between segments a->b and c->d (0 when they cross)."""

    def pt_seg(px, py, x1, y1, x2, y2):
        dx, dy = x2 - x1, y2 - y1
        l2 = dx * dx + dy * dy
        t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / l2))
        return math.hypot(px - (x1 + t * dx), py - (y1 + t * dy))

    def orient(px, py, qx, qy, rx, ry):
        return (qx - px) * (ry - py) - (qy - py) * (rx - px)

    o1 = orient(*a, *b, *c)
    o2 = orient(*a, *b, *d)
    o3 = orient(*c, *d, *a)
    o4 = orient(*c, *d, *b)
    if ((o1 > 0) != (o2 > 0)) and ((o3 > 0) != (o4 > 0)):
        return 0.0
    return min(
        pt_seg(*c, *a, *b), pt_seg(*d, *a, *b), pt_seg(*a, *c, *d), pt_seg(*b, *c, *d)
    )


def _dist_seg_rect(a: tuple[float, float], b: tuple[float, float], r: _Rect) -> float:
    """Minimum distance between a segment and an axis-aligned rectangle.

    Exact: an endpoint inside the rect gives 0 through ``_dist_point_rect``, a
    crossing segment gives 0 through ``_seg_seg_dist`` on the crossed edge, and
    otherwise the minimum lies on a rect edge or at an endpoint.
    """
    x1, y1, x2, y2 = r
    best = min(_dist_point_rect(*a, r), _dist_point_rect(*b, r))
    if best <= 0.0:
        return 0.0
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    for i in range(4):
        best = min(best, _seg_seg_dist(a, b, corners[i], corners[(i + 1) % 4]))
        if best <= 0.0:
            return 0.0
    return best


def _pads_box(pads: Sequence[Pad]) -> _Rect:
    rects = [_rect(p) for p in pads]
    return (
        min(r[0] for r in rects),
        min(r[1] for r in rects),
        max(r[2] for r in rects),
        max(r[3] for r in rects),
    )


# --------------------------------------------------------------------------- #
# Obstacle sets
# --------------------------------------------------------------------------- #


def _track_obstacles(
    pads: Sequence[Pad], src: Pad
) -> tuple[list[tuple[Pad, _Rect]], list[tuple[Pad, _Rect]]]:
    """``(foreign, same_net)`` pads a track leaving *src* interacts with.

    A same-net pad is never an obstacle -- landing on it is a legal connection --
    so it is returned separately: the ray march uses it as a **terminator**
    (reaching it is a tie, not an escape), which is what keeps an inner GND pad
    beside a same-net exposed pad from being mis-read as "already free".
    Only pads sharing a copper layer with *src* can obstruct a track on it.
    """
    foreign: list[tuple[Pad, _Rect]] = []
    same: list[tuple[Pad, _Rect]] = []
    for p in pads:
        if p is src or (p.number == src.number and p.x == src.x and p.y == src.y):
            continue
        if not (p.layers & src.layers):
            continue
        if src.net and p.net == src.net:
            same.append((p, _rect(p)))
        else:
            foreign.append((p, _rect(p)))
    return foreign, same


def _via_obstacles(pads: Sequence[Pad], src: Pad) -> list[tuple[Pad, _Rect]]:
    """Foreign pads a VIA must clear -- on every layer (a via spans the stack)."""
    out: list[tuple[Pad, _Rect]] = []
    for p in pads:
        if p is src or (p.number == src.number and p.x == src.x and p.y == src.y):
            continue
        if src.net and p.net == src.net:
            continue
        out.append((p, _rect(p)))
    return out


def _near(
    obstacles: list[tuple[Pad, _Rect]], x: float, y: float, reach: float
) -> list[_Rect]:
    """Prune to obstacles whose rect could matter within *reach* of (x, y)."""
    return [r for _p, r in obstacles if _dist_point_rect(x, y, r) <= reach]


def _segment_clears(
    rects: Iterable[_Rect],
    a: tuple[float, float],
    b: tuple[float, float],
    keepout: float,
) -> tuple[bool, float]:
    """``(ok, worst_margin)`` for a segment held *keepout* off every rect."""
    worst = math.inf
    for r in rects:
        d = _dist_seg_rect(a, b, r)
        if d < worst:
            worst = d
        if d < keepout:
            return False, d
    return True, (0.0 if worst is math.inf else worst)


# --------------------------------------------------------------------------- #
# Lanes -- gaps in the outer pad row
# --------------------------------------------------------------------------- #


def find_lanes(
    pads: Sequence[Pad], rules: Rules, *, row_tol_mm: float = 0.05
) -> list[Lane]:
    """Gaps in the OUTERMOST pad row of each side, with per-lane capacity.

    The outer row is what a track must cross to leave the pad field, so its
    gaps -- the package's *designed* escape lanes plus every depopulated ring
    position -- are the on-layer exits. Each gap is measured edge to edge and
    rationed by :meth:`Rules.lane_capacity`, so a 0.75 mm depopulated lane
    carries one 0.153 mm track and two 0.127 mm ones instead of silently
    accepting both and shipping a clearance violation.
    """
    if not pads:
        return []
    box = _pads_box(pads)
    lanes: list[Lane] = []
    for side in ("x-", "x+", "y-", "y+"):
        axis = 0 if side[0] == "x" else 1
        edge = box[axis] if side[1] == "-" else box[axis + 2]
        # The outer row: pads whose outer edge sits on the field boundary.
        row: list[_Rect] = []
        for p in pads:
            r = _rect(p)
            outer = r[axis] if side[1] == "-" else r[axis + 2]
            if abs(outer - edge) <= row_tol_mm:
                row.append(r)
        if not row:
            continue
        # Gaps along the perpendicular axis, bounded by the field box so the
        # row's two ends count as lanes too (the corner exits).
        oth = 1 - axis
        spans = sorted((r[oth], r[oth + 2]) for r in row)
        merged: list[list[float]] = []
        for lo, hi in spans:
            if merged and lo <= merged[-1][1] + 1e-9:
                merged[-1][1] = max(merged[-1][1], hi)
            else:
                merged.append([lo, hi])
        # Interleave the field-box ends with the occupied spans, then take every
        # OTHER interval: [box_lo, span0_lo], [span0_hi, span1_lo], ...,
        # [spanN_hi, box_hi]. The row's two ends are lanes too -- those are the
        # corner exits a ring package leaves when its rows stop short.
        edges = [box[oth]] + [v for m in merged for v in m] + [box[oth + 2]]
        for i in range(0, len(edges) - 1, 2):
            lo, hi = edges[i], edges[i + 1]
            gap = hi - lo
            if gap <= 0:
                continue
            cap = rules.lane_capacity(gap)
            if cap > 0:
                lanes.append(Lane(side=side, lo=lo, hi=hi, gap_mm=gap, capacity=cap))
    return lanes


def _outer_row_pads(
    pads: Sequence[Pad],
    box: _Rect,
    uid_of: dict[int, str] | None = None,
    row_tol_mm: float = 0.05,
) -> set[str]:
    """Pad numbers sitting on the pad field's boundary on at least one side.

    These are the pads with nothing of their own footprint between them and
    open copper: whatever else is true, they do not have to thread a lane, so
    the router escapes them without help. Everything netted that is NOT on the
    boundary is an *inner* pad and must be planned -- that single distinction is
    what separates "the router can do this" from the aQFN inner ring, whose only
    exits are the package's designed lanes.
    """
    out: set[str] = set()
    for p in pads:
        r = _rect(p)
        if (
            abs(r[0] - box[0]) <= row_tol_mm
            or abs(r[1] - box[1]) <= row_tol_mm
            or abs(r[2] - box[2]) <= row_tol_mm
            or abs(r[3] - box[3]) <= row_tol_mm
        ):
            out.add(uid_of[id(p)] if uid_of else p.number)
    return out


# --------------------------------------------------------------------------- #
# Exit search
# --------------------------------------------------------------------------- #


def via_fanout_center(
    src: Pad,
    pads: Sequence[Pad],
    rules: Rules,
    *,
    taken_vias: Sequence[tuple[str, float, float]] = (),
    taken_paths: Sequence[tuple[str, tuple[float, float], tuple[float, float]]] = (),
    search_radius_mm: float = 0.75,
    step_mm: float = 0.005,
    angle_step_deg: float = 2.0,
    prefer_outward_from: tuple[float, float] | None = None,
) -> tuple[tuple[float, float], float] | None:
    """Nearest legal dog-bone via centre beside *src*, or ``None``.

    Sweeps a disc around the pad and returns the closest centre at which the via
    annulus clears every foreign pad and every already-assigned fanout via by
    the rule clearance, its drill clears every drilled hole by the hole-to-hole
    minimum, and the pad-to-via stub itself is legal. Preference order is
    (distance, then outwardness) so the via lands in the ring channel rather
    than on top of the exposed pad -- but legality, never preference, decides.

    ``taken_vias`` carries the fanout vias already placed for THIS footprint:
    two 0.4 mm vias at 0.5 mm ring pitch violate via-via clearance, and the disc
    search resolves that by itself -- the second pad's via simply shifts along
    the ring into an empty neighbouring position. No special-casing, just honest
    clearance checks in the search.
    """
    obstacles = _via_obstacles(pads, src)
    reach = search_radius_mm + rules.via_keepout_mm + 2.0
    rects = _near(obstacles, src.x, src.y, reach)
    track_foreign, _same = _track_obstacles(pads, src)
    stub_rects = _near(track_foreign, src.x, src.y, reach)
    holes = [
        (p.x, p.y, p.drill / 2.0)
        for p in pads
        if p.drill > 0.0 and not (p.number == src.number and p.x == src.x and p.y == src.y)
    ]
    via_r = rules.via_diameter_mm / 2.0
    drill_r = rules.via_drill_mm / 2.0
    keep = rules.via_keepout_mm

    cx, cy = prefer_outward_from if prefer_outward_from is not None else (src.x, src.y)
    out_x, out_y = src.x - cx, src.y - cy
    out_n = math.hypot(out_x, out_y)
    if out_n > 1e-9:
        out_x, out_y = out_x / out_n, out_y / out_n
    else:
        out_x, out_y = 0.0, 0.0

    best: tuple[tuple[float, float, float], tuple[float, float], float] | None = None
    n_ang = max(1, int(round(360.0 / angle_step_deg)))
    for i in range(n_ang):
        ang = math.radians(i * angle_step_deg)
        ux, uy = math.cos(ang), math.sin(ang)
        outward = -(ux * out_x + uy * out_y)  # smaller = more outward
        off = step_mm
        while off <= search_radius_mm + 1e-9:
            vx, vy = src.x + ux * off, src.y + uy * off
            worst = math.inf
            ok = True
            for r in rects:
                d = _dist_point_rect(vx, vy, r)
                if d < keep:
                    ok = False
                    break
                worst = min(worst, d)
            # `keep` already folds in the hole-to-copper rule via
            # Rules.via_keepout_mm, so a single distance test covers both.
            if ok:
                for o_net, tx, ty in taken_vias:
                    if src.net and o_net == src.net:
                        continue
                    d = math.hypot(vx - tx, vy - ty)
                    if d < 2.0 * via_r + rules.clearance_mm or d < (
                        2.0 * drill_r + rules.hole_to_hole_mm
                    ):
                        ok = False
                        break
                    worst = min(worst, d - via_r)
            if ok:
                for hx, hy, hr in holes:
                    if math.hypot(vx - hx, vy - hy) < hr + drill_r + rules.hole_to_hole_mm:
                        ok = False
                        break
            if ok:
                need = via_r + rules.clearance_mm + rules.track_mm / 2.0
                for o_net, pa, pb in taken_paths:
                    if src.net and o_net == src.net:
                        continue
                    if _seg_seg_dist((vx, vy), (vx, vy), pa, pb) < need:
                        ok = False
                        break
            if ok:
                stub_ok, _m = _segment_clears(
                    stub_rects, (src.x, src.y), (vx, vy), rules.track_keepout_mm
                )
                ok = stub_ok
            if ok:
                margin = (0.0 if worst is math.inf else worst) - via_r
                rank = (round(off, 4), round(outward, 4), round(vx, 4))
                if best is None or rank < best[0]:
                    best = (rank, (vx, vy), margin)
                break  # nearest legal offset on this ray; try the next angle
            off += step_mm
    if best is None:
        return None
    return best[1], best[2]


def _lane_exit_points(
    lane: Lane, box: _Rect, rules: Rules, margin: float, src: Pad
) -> list[tuple[float, float]]:
    """Candidate exit points just outside the field box within *lane*.

    ``side`` names the axis the lane CROSSES, so for an ``x`` side the exit's x
    is fixed at the box edge and the lane's lo/hi bound its y (and vice versa).
    The legal centerline interval is sampled at half a track pitch and returned
    nearest-to-the-pad first, so two escapes sharing a wide lane naturally land
    at different positions instead of both aiming for the middle.
    """
    axis = 0 if lane.side[0] == "x" else 1
    perp = (box[axis] - margin) if lane.side[1] == "-" else (box[axis + 2] + margin)
    half = rules.track_keepout_mm
    lo, hi = lane.lo + half, lane.hi - half
    if lo > hi:
        return []
    step = max(1e-3, (rules.track_mm + rules.clearance_mm) / 2.0)
    n = min(24, int((hi - lo) / step) + 1)
    alongs = {round(lo, 4), round(hi, 4)}
    for i in range(n + 1):
        alongs.add(round(lo + (hi - lo) * (i / max(1, n)), 4))
    src_along = src.y if axis == 0 else src.x
    ordered = sorted(alongs, key=lambda a: (round(abs(a - src_along), 4), a))
    return [((perp, a) if axis == 0 else (a, perp)) for a in ordered]


def _nearest_landing(src: Pad, target: Pad, inset_mm: float = 0.05) -> tuple[float, float]:
    """Where a tie from *src* should land on *target*: its nearest point, nudged
    *inset_mm* inward so the copper genuinely overlaps instead of grazing.

    Landing on the pad CENTRE would run a 3 mm track across a 4.85 mm exposed
    pad to say what 0.3 mm of copper already says.
    """
    r = _rect(target)
    lx = min(max(src.x, r[0]), r[2])
    ly = min(max(src.y, r[1]), r[3])
    dx, dy = target.x - lx, target.y - ly
    n = math.hypot(dx, dy)
    if n <= 1e-9:
        return (lx, ly)
    step = min(inset_mm, n)
    return (lx + dx / n * step, ly + dy / n * step)


def _path_clears_committed(
    pts: tuple[tuple[float, float], ...],
    committed: Sequence[tuple[str, tuple[float, float], tuple[float, float]]],
    committed_vias: Sequence[tuple[str, float, float]],
    rules: Rules,
    net: str = "",
) -> bool:
    """Keep a candidate escape off the escapes already committed for other nets.

    Lane *capacity* rations how many tracks an opening can carry; this rations
    WHERE they go. Without it two escapes legally assigned to one 1.5 mm lane
    can still be stamped 0.26 mm apart -- a violation the board-level stamp
    guard then resolves by silently dropping one of them. Same-net copper is
    skipped: it is a valid landing, not an obstacle.
    """
    need_seg = rules.track_mm + rules.clearance_mm
    need_via = rules.track_mm / 2.0 + rules.clearance_mm + rules.via_diameter_mm / 2.0
    for a, b in zip(pts, pts[1:]):
        for o_net, c, d in committed:
            if net and o_net == net:
                continue
            if _seg_seg_dist(a, b, c, d) < need_seg:
                return False
        for o_net, vx, vy in committed_vias:
            if net and o_net == net:
                continue
            if _seg_seg_dist(a, b, (vx, vy), (vx, vy)) < need_via:
                return False
    return True


def _lane_escape(
    src: Pad,
    lane: Lane,
    box: _Rect,
    rules: Rules,
    rects: list[_Rect],
    margin: float,
    *,
    max_len_mm: float = 2.5,
    committed: Sequence[tuple[str, tuple[float, float], tuple[float, float]]] = (),
    committed_vias: Sequence[tuple[str, float, float]] = (),
) -> tuple[tuple[tuple[float, float], ...], float] | None:
    """Shortest straight or one-bend path from *src* out through *lane*.

    The bend candidates are the two L-shapes through the exit point's along/perp
    coordinates -- enough to express "slide sideways in the ring channel, then
    straight out of the lane", which is the move a fixed radial ray cannot make
    and the reason XL1 had no exit while XL2 (whose radial ray happened to line
    up with the same opening) did.

    ``max_len_mm`` keeps an *escape* an escape: past a couple of millimetres the
    path has stopped leaving the pad field and started routing the net, which is
    FreeRouting's job -- and, for a poured net whose same-net copper is nearby,
    an unbounded search will happily propose a 6 mm track straight across the
    exposed pad because same-net copper is not an obstacle.
    """
    keep = rules.track_keepout_mm
    start = (src.x, src.y)
    best: tuple[tuple[float, int], tuple[tuple[float, float], ...], float] | None = None
    for exit_pt in _lane_exit_points(lane, box, rules, margin, src):
        cands: list[tuple[tuple[float, float], ...]] = [(start, exit_pt)]
        for bend in ((exit_pt[0], src.y), (src.x, exit_pt[1])):
            if (
                math.hypot(bend[0] - start[0], bend[1] - start[1]) > 1e-6
                and math.hypot(bend[0] - exit_pt[0], bend[1] - exit_pt[1]) > 1e-6
            ):
                cands.append((start, bend, exit_pt))
        for pts in cands:
            length = sum(
                math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:])
            )
            if length > max_len_mm:
                continue
            rank = (round(length, 4), len(pts))
            if best is not None and rank >= best[0]:
                continue
            ok = True
            worst = math.inf
            for a, b in zip(pts, pts[1:]):
                seg_ok, m = _segment_clears(rects, a, b, keep)
                if not seg_ok:
                    ok = False
                    break
                worst = min(worst, m)
            if not ok or not _path_clears_committed(
                pts, committed, committed_vias, rules, src.net
            ):
                continue
            best = (rank, pts, 0.0 if worst is math.inf else worst)
    if best is None:
        return None
    return best[1], best[2]


# --------------------------------------------------------------------------- #
# The planner
# --------------------------------------------------------------------------- #


def plan_escapes(
    pads: Sequence[Pad],
    rules: Rules,
    *,
    exit_margin_mm: float = 0.15,
    via_search_radius_mm: float = 0.75,
    max_escape_len_mm: float = 2.5,
    allow_via: bool = True,
    self_via_pad_mm: float = 0.6,
    thermal_pad_area_mm2: float = 4.0,
    only_pads: Sequence[str] | None = None,
) -> EscapePlan:
    """Plan one exit per netted pad of *pads* at *rules*.

    Deterministic: pads are processed in sorted order, and every search returns
    its first legal candidate under a total ordering, so the same pad field and
    rule set always yield the same plan (fanout-via assignment included, since
    later pads see earlier pads' vias).
    """
    plan = EscapePlan()
    if not pads:
        return plan
    box = _pads_box(pads)
    plan.field_box = box
    lanes = find_lanes(pads, rules)
    plan.lanes = lanes
    remaining = {ln.key(): ln.capacity for ln in lanes}
    # Pad-field centre: the "outward" reference the via search prefers. The box
    # centre, not the footprint origin -- an asymmetric pad field would bias the
    # other way.
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    # A pad that can host a via in its OWN copper is not an escape problem: the
    # exposed/thermal pad gnd_pour stitches with a via array, and any pad wide
    # enough to take an in-pad via. Same test gnd_pour.gnd_escape_specs uses, so
    # the two agree on which pads need help. Everything finer is on its own.
    def _self_via_capable(p: Pad) -> bool:
        return min(p.w, p.h) >= self_via_pad_mm or p.w * p.h >= thermal_pad_area_mm2

    # uid per pad: the pad number when unique, else number#i. Module footprints
    # routinely number every ground pad "GND", and keying verdicts by number
    # alone would keep one and silently drop the rest. Keyed by id() rather than
    # index because every lookup site already holds the Pad; `pads` is the
    # caller's live sequence for the whole call, so no id can be recycled.
    seen: dict[str, int] = {}
    uid_of: dict[int, str] = {}
    counts: dict[str, int] = {}
    for p in pads:
        counts[p.number] = counts.get(p.number, 0) + 1
    for p in pads:
        if counts[p.number] == 1:
            uid_of[id(p)] = p.number
        else:
            i = seen.get(p.number, 0)
            seen[p.number] = i + 1
            uid_of[id(p)] = f"{p.number}#{i}"
    outer = _outer_row_pads(pads, box, uid_of)

    wanted = set(only_pads) if only_pads is not None else None
    targets = sorted(
        (
            p
            for p in pads
            if p.net
            and (wanted is None or p.number in wanted or uid_of[id(p)] in wanted)
            and not _self_via_capable(p)
        ),
        key=lambda p: uid_of[id(p)],
    )

    # Pass 1: outer-row pads have open copper in front of them -- the router
    # escapes those unaided. Everything else is an inner pad that must thread a
    # lane or drop a via, and that is what this module exists to decide.
    pending: list[Pad] = []
    for p in targets:
        if uid_of[id(p)] in outer:
            plan.escapes[uid_of[id(p)]] = Escape(
                pad=p.number,
                net=p.net,
                kind="open",
                reason="on the pad field's outer row -- open copper in front of it",
            )
        else:
            pending.append(p)
    if not pending:
        return plan

    diag = math.hypot(box[2] - box[0], box[3] - box[1])
    reach = diag / 2.0 + exit_margin_mm + 1.0

    # Pass 2a: what on-layer lanes can each inner pad actually reach? This is
    # also the scarcity signal for the assignment below -- a pad with one
    # candidate lane must pick before a pad with three, or it gets starved.
    lane_fit: dict[str, list[tuple[float, Lane]]] = {}
    obstacles: dict[str, list[_Rect]] = {}
    for p in pending:
        foreign_pads, _same = _track_obstacles(pads, p)
        rects = _near(foreign_pads, p.x, p.y, reach)
        obstacles[uid_of[id(p)]] = rects
        fits: list[tuple[float, Lane]] = []
        for ln in lanes:
            got = _lane_escape(
                p, ln, box, rules, rects, exit_margin_mm, max_len_mm=max_escape_len_mm
            )
            if got is not None:
                pts, _m = got
                fits.append(
                    (
                        sum(
                            math.hypot(b[0] - a[0], b[1] - a[1])
                            for a, b in zip(pts, pts[1:])
                        ),
                        ln,
                    )
                )
        fits.sort(key=lambda t: (round(t[0], 4), t[1].key()))
        lane_fit[uid_of[id(p)]] = fits

    # Pass 2b: assign. Scarcest-first over lanes so a pad with one candidate
    # lane is not starved by a pad that had three -- two pads never fight over
    # one lane, and the loser falls through to a via rather than to nothing.
    # Each committed escape becomes an obstacle for the ones after it, so the
    # plan is internally legal before a single track is stamped.
    taken_vias: list[tuple[str, float, float]] = []
    taken_paths: list[tuple[str, tuple[float, float], tuple[float, float]]] = []

    def _commit(net: str, pts: tuple[tuple[float, float], ...]) -> None:
        taken_paths.extend((net, a, b) for a, b in zip(pts, pts[1:]))

    # A same-net landing only helps if the pad it lands ON is itself out of
    # trouble: two trapped pads tying to each other just makes a bigger trapped
    # island. "Resolved" = an outer-row pad (open copper in front of it) or a
    # pad wide enough to host its own via, which is the exposed pad -- the one
    # gnd_pour stitches to the plane with a thermal-via array.
    resolved_targets = outer | {
        uid_of[id(p)] for p in pads if _self_via_capable(p)
    }

    order = sorted(
        pending, key=lambda p: (len(lane_fit[uid_of[id(p)]]), uid_of[id(p)])
    )
    for p in order:
        uid = uid_of[id(p)]
        _foreign, same_pads = _track_obstacles(pads, p)
        rects = obstacles[uid]
        alts: list[str] = []

        # (a) Same-net landing. Not an escape at all -- a direct connection to
        #     this footprint's own copper on the same net (an inner GND pad
        #     beside the exposed pad). Always the shortest correct answer when
        #     it exists, costs no hole, and cannot be starved, so it goes first.
        tie: tuple[Pad, tuple[tuple[float, float], ...], float] | None = None
        for tp, trect in sorted(same_pads, key=lambda t: _dist_point_rect(p.x, p.y, t[1])):
            if _dist_point_rect(p.x, p.y, trect) > max_escape_len_mm:
                break
            if uid_of[id(tp)] not in resolved_targets:
                continue
            pts = ((p.x, p.y), _nearest_landing(p, tp))
            ok, m = _segment_clears(rects, pts[0], pts[1], rules.track_keepout_mm)
            if ok and _path_clears_committed(pts, taken_paths, taken_vias, rules, p.net):
                tie = (tp, pts, m)
                break
        if tie is not None:
            tp, pts, m = tie
            _commit(p.net, pts)
            plan.escapes[uid] = Escape(
                pad=p.number,
                net=p.net,
                kind="tie",
                polyline=pts,
                margin_mm=m,
                reason=f"same-net landing on pad {tp.number}",
            )
            continue

        # Strategy order for a real escape: on-layer lane, then dog-bone via.
        #
        # The plan this implements specified via-FIRST, arguing the dog-bone is
        # uniform and needs no lane coordination. With the lane assignment above
        # actually built, that argument no longer pays: on the witness aQFN-73,
        # via-first drops all 12 inner netted pads (VBAT, ANT_RF and the crystal
        # included) onto B.Cu -- 12 punctures through the GND plane, each one
        # forcing its net to continue on the plane layer of a 2-layer board.
        # A lane escape costs no hole, keeps the signal on the signal layer, and
        # is the move the package's depopulated lanes exist for. So: lane where
        # one is legally free, via where the geometry genuinely leaves no
        # on-layer exit (AC13's fully-populated wall -- the case the 0.4/0.2 via
        # class was added for, and which nothing else can solve).
        chosen: tuple[Lane, tuple[tuple[float, float], ...], float] | None = None
        for _length, ln in lane_fit[uid]:
            if remaining.get(ln.key(), 0) <= 0:
                continue
            got = _lane_escape(
                p,
                ln,
                box,
                rules,
                rects,
                exit_margin_mm,
                max_len_mm=max_escape_len_mm,
                committed=taken_paths,
                committed_vias=taken_vias,
            )
            if got is not None:
                chosen = (ln, got[0], got[1])
                break
        via = (
            via_fanout_center(
                p,
                pads,
                rules,
                taken_vias=taken_vias,
                taken_paths=taken_paths,
                search_radius_mm=via_search_radius_mm,
                prefer_outward_from=(cx, cy),
            )
            if allow_via
            else None
        )
        if via is not None:
            alts.append("via")
        # (b) On-layer lane, capacity and position both honoured.
        if chosen is not None:
            ln, pts, m = chosen
            remaining[ln.key()] -= 1
            _commit(p.net, pts)
            plan.escapes[uid] = Escape(
                pad=p.number,
                net=p.net,
                kind="lane",
                polyline=pts,
                lane=ln,
                margin_mm=m,
                alternatives=tuple(alts),
                reason=f"{ln.side} lane {ln.gap_mm:.3f} mm (capacity {ln.capacity})",
            )
            continue
        # (c) Dog-bone fanout via: placement-independent, cannot be starved by a
        #     neighbour, and the only exit that exists for a wall-locked pad.
        if via is not None:
            center, m = via
            taken_vias.append((p.net, center[0], center[1]))
            _commit(p.net, ((p.x, p.y), center))
            plan.escapes[uid] = Escape(
                pad=p.number,
                net=p.net,
                kind="via",
                polyline=((p.x, p.y), center),
                via_center=center,
                margin_mm=m,
                alternatives=("lane",) if lane_fit[uid] else (),
                reason=f"dog-bone fanout via {rules.via_diameter_mm:g}/{rules.via_drill_mm:g}",
            )
            continue
        # (d) Nothing legal exists at this rule set. Say so; NEVER stamp a nub --
        #     it connects nothing, obstructs the router, and reads as progress.
        why = "no legal via centre, no free lane, no same-net landing"
        if lane_fit[uid]:
            why = "every reachable lane is at capacity and no via centre is legal"
        plan.escapes[uid] = Escape(
            pad=p.number, net=p.net, kind="infeasible", reason=why
        )
    return plan


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


def pads_from_dicts(rows: Iterable[dict[str, Any]]) -> list[Pad]:
    """Build pads from plain dicts (the checked-in fixture format)."""
    out: list[Pad] = []
    for d in rows:
        layers = d.get("layers")
        out.append(
            Pad(
                number=str(d["number"]),
                net=str(d.get("net", "") or ""),
                x=float(d["x"]),
                y=float(d["y"]),
                w=float(d["w"]),
                h=float(d["h"]),
                layers=frozenset(layers) if layers else frozenset({"F.Cu"}),
                drill=float(d.get("drill", 0.0) or 0.0),
            )
        )
    return out


def planning_rules(
    cfg: dict[str, Any], *, netclass_clearance_mm: float | None = None
) -> Rules:
    """The rule set an escape must actually satisfy to survive being stamped.

    NOT the config floors. KiCad resolves a clearance pair as the LARGER of the
    two items' netclass constraints, and ``breakout_stubs`` holds a further
    +10 um geometry guard above that -- so a plan drawn at the bare floor is a
    plan whose escapes get silently dropped at stamp time. Callers with a board
    in hand pass the measured netclass clearance; callers without one (the
    placement path) get the profile's value.
    """
    from kicraft.autoplacer.brain.breakout_stubs import STAMP_CLEARANCE_GUARD_MM
    from kicraft.autoplacer.fab_profile import (
        NETCLASS_CLEARANCE_MM,
        fab_floors,
        fanout_via,
    )

    floors = fab_floors(cfg)
    via_d, via_dr = fanout_via(cfg)
    netclass = (
        NETCLASS_CLEARANCE_MM if netclass_clearance_mm is None else netclass_clearance_mm
    )
    return Rules(
        track_mm=float(cfg.get("freerouting_fine_pitch_track_mm", floors["track_mm"])),
        clearance_mm=STAMP_CLEARANCE_GUARD_MM
        + max(
            netclass,
            float(cfg.get("freerouting_min_clearance_mm", floors["clearance_mm"])),
        ),
        via_diameter_mm=via_d,
        via_drill_mm=via_dr,
        hole_to_hole_mm=float(cfg.get("hole_to_hole_min_mm", 0.25)),
        hole_clearance_mm=float(cfg.get("hole_clearance_min_mm", 0.25)),
    )


def capability_rules_from_config(cfg: dict[str, Any]) -> Rules:
    """The rule set the FAB can do -- the "feasible only at capability" probe."""
    from kicraft.autoplacer.fab_profile import fanout_via, fab_floors

    floors = fab_floors(cfg)
    via_d, via_dr = fanout_via(cfg)
    return Rules(
        track_mm=floors["track_mm"],
        clearance_mm=floors["clearance_mm"],
        via_diameter_mm=via_d,
        via_drill_mm=via_dr,
        hole_to_hole_mm=float(cfg.get("hole_to_hole_min_mm", 0.25)),
        hole_clearance_mm=float(cfg.get("hole_clearance_min_mm", 0.25)),
    )
