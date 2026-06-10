"""Deliberate breakout stubs / escape routes for footprint pads.

A gridless autorouter struggles to escape a dense, fine-pitch connector (USB-C,
board-to-board, QFN). Its inner pins are boxed in by neighbours, and on a
2-layer board with a ground plane there is only one signal layer to escape onto,
so a pin can be left unrouted even when a clear path exists.

The fix PCB designers use by hand is a **deliberate fanout**: pre-route a short
stub from each pad out of the pad field to a breakout point in open copper, then
let the autorouter finish from there. This module is the reusable primitive for
that. A :class:`BreakoutSpec` describes the escape for one pad as an explicit
polyline (``waypoints``) and/or a *radial* escape (straight out from the
footprint centre through the pad). :func:`add_breakout_stubs` lays the segments
as **locked** tracks (optionally dropping a via at the end) so a subsequent
FreeRouting pass run with ``freerouting_preserve_existing_copper=True`` keeps
them and routes the rest from the accessible breakout endpoints.

Specs are footprint-relative-friendly: a curated connector entry stores the
escape once (see the curated leaf library) and it is reused on every board that
places that part.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pcbnew

_LAYERS = {"F.Cu": pcbnew.F_Cu, "B.Cu": pcbnew.B_Cu}


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
        Track width; ``None`` falls back to ``cfg['freerouting_fine_pitch_track_mm']``.
    layer:
        Copper layer the stub is drawn on.
    via_at_end:
        Drop a layer-changing via at the final point (to escape onto the other
        layer when this one is congested).
    """

    ref: str
    pad: str
    waypoints: list[tuple[float, float]] = field(default_factory=list)
    length_mm: float = 1.5
    width_mm: float | None = None
    layer: str = "F.Cu"
    via_at_end: bool = False


def _find_pad(board: "pcbnew.BOARD", ref: str, pad_number: str):
    for fp in board.GetFootprints():
        if fp.GetReferenceAsString() == ref:
            for pad in fp.Pads():
                if pad.GetNumber() == pad_number:
                    return fp, pad
    return None, None


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


def _safe_radial_length(
    board: "pcbnew.BOARD",
    src_pad,
    start_mm: tuple[float, float],
    dir_unit: tuple[float, float],
    requested_mm: float,
    clearance_mm: float,
) -> float:
    """Largest escape length along *dir_unit* that stays clear of other pads.

    A radial escape on a dense connector can run straight across a neighbouring
    pad (a different net, or an NC pad) -- which the autorouter then reports as a
    short. March along the ray and stop just before the first point that falls
    within *clearance_mm* of any foreign pad. Returns 0.0 when even the first
    step collides (no safe escape in this direction).
    """
    others = _foreign_pads(board, src_pad.GetNetCode(), exclude=src_pad)
    if not others:
        return requested_mm

    step = 0.1
    sx, sy = start_mm
    ux, uy = dir_unit
    safe = 0.0
    d = step
    margin = int(pcbnew.FromMM(clearance_mm))
    while d <= requested_mm + 1e-9:
        pt = pcbnew.VECTOR2I(
            int(pcbnew.FromMM(sx + ux * d)), int(pcbnew.FromMM(sy + uy * d))
        )
        if any(pad.HitTest(pt, margin) for pad in others):
            break
        safe = d
        d += step
    return safe


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


def _pads_bbox_mm(
    pads: list, margin_mm: float = 0.0
) -> tuple[float, float, float, float]:
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
    Locked copper outside the outline is fatal: FreeRouting 1.9.0 reads the
    corner as "wire corner outside board" and hangs without producing a SES
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
    fp = next(
        (f for f in board.GetFootprints() if f.GetReferenceAsString() == ref), None
    )
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
        obstacles = _foreign_pads(board, pads[i].GetNetCode())
        if _segment_clears_pads(obstacles, p1, p2, clearance_mm):
            specs.append(
                BreakoutSpec(
                    ref=ref, pad=pads[i].GetNumber(), waypoints=[p2], layer=layer
                )
            )
            continue
        # A pad field closer to the board edge than its margin puts part of the
        # walk rectangle off the board -- and stamping locked off-board copper
        # hangs FreeRouting. Clamp the rectangle to the inner box; valid only
        # while the clamped border still clears every pad it must walk around
        # (the raw pad bbox plus clearance). When even that fails, skip the tie
        # loudly: FreeRouting still routes the net, the pour just may fragment.
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
            BreakoutSpec(
                ref=ref, pad=pads[i].GetNumber(), waypoints=waypoints, layer=layer
            )
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
                clearance_mm=float(cfg.get("freerouting_min_clearance_mm", 0.153)),
            )
        )
    return specs


def auto_signal_escape_specs(
    board: "pcbnew.BOARD",
    cfg: dict[str, Any] | None = None,
) -> list[BreakoutSpec]:
    """Radial escape stubs for the *signal* pads of dense connectors.

    Companion to :func:`auto_power_tie_specs`. A fine-pitch connector (USB-C,
    board-to-board) boxes its inner signal pins in among its power/ground pads,
    so a gridless autorouter abandons a net like a USB-C CC pin -> its pulldown
    resistor even though a short escape would free it. For every footprint dense
    enough to need help -- the same signal ``auto_power_tie_specs`` keys on: a
    connector with >= 2 pads on a non-GND power rail (spread VBUS) -- this emits a
    short radial escape out of each *multi-pad signal* net's pad on that
    footprint, so FreeRouting finishes from open copper. Excluded: power/ground
    pads (left to the pour + perimeter tie), single-pad signal nets (an interface
    net with nothing to route to in-leaf -- it closes at compose), and refs in
    ``signal_escape_exclude_refs``. The collision guard in
    :func:`add_breakout_stubs` clips or drops any escape that would cross a
    neighbour pad, so this never introduces a short.
    """
    cfg = cfg or {}
    if not cfg.get("auto_signal_escape", True):
        return []
    gnd_name = cfg.get("gnd_zone_net", "GND")
    exclude = set(cfg.get("signal_escape_exclude_refs", []) or [])
    length = float(cfg.get("signal_escape_length_mm", 1.5))
    layer = cfg.get("signal_escape_layer", "F.Cu")
    from kicraft.design.models import is_power_or_ground_name

    # net -> pad count across the whole leaf. A single-pad net has nothing to
    # route to on the leaf (it's an interface net that closes at compose), so an
    # escape on it would be wasted.
    net_pads: dict[str, int] = {}
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            n = pad.GetNetname()
            if n:
                net_pads[n] = net_pads.get(n, 0) + 1

    specs: list[BreakoutSpec] = []
    for fp in board.GetFootprints():
        ref = fp.GetReferenceAsString()
        if ref in exclude:
            continue
        power_counts: dict[str, int] = {}
        signal_pads: list[str] = []
        for pad in fp.Pads():
            n = pad.GetNetname()
            if not n:
                continue
            if is_power_or_ground_name(n):
                if n != gnd_name:
                    power_counts[n] = power_counts.get(n, 0) + 1
            elif net_pads.get(n, 0) >= 2:
                signal_pads.append(pad.GetNumber())
        # Only a spread-power connector (>= 2 pads on one non-GND power net -- the
        # USB-C VBUS signature) is treated as dense enough to need escapes; plain
        # 2-pin connectors and ICs route fine without them.
        if not any(c >= 2 for c in power_counts.values()):
            continue
        for num in signal_pads:
            specs.append(BreakoutSpec(ref=ref, pad=num, length_mm=length, layer=layer))
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
    default_w = float(cfg.get("freerouting_fine_pitch_track_mm", 0.153))
    summary: dict[str, Any] = {
        "stubs": 0,
        "segments": 0,
        "vias": 0,
        "skipped": [],
    }
    if not specs:
        return summary

    board = pcbnew.LoadBoard(pcb_path)
    inner_box = _board_inner_box_mm(board)

    def _pt(xy: tuple[float, float]):
        return pcbnew.VECTOR2I(pcbnew.FromMM(xy[0]), pcbnew.FromMM(xy[1]))

    for spec in specs:
        fp, pad = _find_pad(board, spec.ref, spec.pad)
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

        pad_pos = pad.GetPosition()
        start_mm = (pcbnew.ToMM(pad_pos.x), pcbnew.ToMM(pad_pos.y))
        if spec.waypoints:
            points = [start_mm, *spec.waypoints]
            # Hard invariant: never stamp a path that crosses a pad of another
            # net (or a no-net pad) -- that is a short. The tie geometry routes
            # clear of the source footprint's own pads, but a curated path or a
            # neighbour the geometry could not see may still intrude. A partial
            # tie is useless, so drop the whole spec rather than clip it.
            clearance = float(cfg.get("freerouting_min_clearance_mm", 0.153))
            obstacles = _foreign_pads(board, net_code)
            if not all(
                _segment_clears_pads(obstacles, a, b, clearance)
                for a, b in zip(points, points[1:])
            ):
                summary["skipped"].append(f"{spec.ref}.{spec.pad}:waypoint_crosses_pad")
                continue
        else:
            fc = fp.GetPosition()
            cx, cy = pcbnew.ToMM(fc.x), pcbnew.ToMM(fc.y)
            dx, dy = start_mm[0] - cx, start_mm[1] - cy
            norm = (dx * dx + dy * dy) ** 0.5
            dir_unit = (1.0, 0.0) if norm < 1e-6 else (dx / norm, dy / norm)
            # Clip the radial escape so it never crosses a neighbouring pad
            # (which would short). Skip the stub when no safe escape exists in
            # this direction rather than emit a shorting trace.
            clearance = float(cfg.get("freerouting_min_clearance_mm", 0.153))
            safe_len = _safe_radial_length(
                board, pad, start_mm, dir_unit, spec.length_mm, clearance
            )
            min_useful = max(clearance, pcbnew.ToMM(width))
            if safe_len < min_useful:
                summary["skipped"].append(
                    f"{spec.ref}.{spec.pad}:no_safe_radial_escape"
                )
                continue
            end = (
                start_mm[0] + dir_unit[0] * safe_len,
                start_mm[1] + dir_unit[1] * safe_len,
            )
            points = [start_mm, end]

        # Hard invariant: never stamp locked copper outside the board outline.
        # FreeRouting 1.9.0 hangs (no SES, no error) on a locked wire corner
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

        if spec.via_at_end:
            via = pcbnew.PCB_VIA(board)
            via.SetPosition(_pt(points[-1]))
            via.SetDrill(pcbnew.FromMM(float(cfg.get("via_drill_mm", 0.3))))
            try:
                via.SetWidth(pcbnew.FromMM(float(cfg.get("via_size_mm", 0.6))))
            except TypeError:
                via.SetWidth(layer, pcbnew.FromMM(float(cfg.get("via_size_mm", 0.6))))
            via.SetNetCode(net_code)
            if lock:
                via.SetLocked(True)
            board.Add(via)
            summary["vias"] += 1

        summary["stubs"] += 1

    board.Save(pcb_path)
    return summary
