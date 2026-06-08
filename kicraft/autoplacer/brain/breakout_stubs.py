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
    within *clearance_mm* of any pad other than the source pad. Returns 0.0 when
    even the first step collides (no safe escape in this direction).
    """
    others = []
    src_net = src_pad.GetNetCode()
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            if pad is src_pad:
                continue
            # A same-net pad is a valid landing, not an obstacle.
            if pad.GetNetCode() == src_net and src_net != 0:
                continue
            others.append(pad)
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
        # HitTest with an accuracy margin ~ clearance treats each pad as slightly
        # grown, so we stop before violating clearance to it.
        if any(pad.HitTest(pt, margin) for pad in others):
            break
        safe = d
        d += step
    return safe


def radial_escape_point(
    fp_center_mm: tuple[float, float],
    pad_mm: tuple[float, float],
    length_mm: float,
) -> tuple[float, float]:
    """Point ``length_mm`` beyond the pad along the centre->pad ray.

    That direction points out of the pad field for an edge/connector pad, so a
    segment to this point clears the footprint without crossing its other pads.
    Degenerate (pad at centre) falls back to a straight +x escape.
    """
    cx, cy = fp_center_mm
    px, py = pad_mm
    dx, dy = px - cx, py - cy
    norm = (dx * dx + dy * dy) ** 0.5
    if norm < 1e-6:
        return (px + length_mm, py)
    ux, uy = dx / norm, dy / norm
    return (px + ux * length_mm, py + uy * length_mm)


def radial_breakout_specs(
    board: "pcbnew.BOARD",
    ref: str,
    pad_numbers: list[str] | None = None,
    *,
    length_mm: float = 1.5,
    layer: str = "F.Cu",
    via_at_end: bool = False,
    nets_only: list[str] | None = None,
) -> list[BreakoutSpec]:
    """Build radial-escape specs for a footprint's pads.

    By default every netted pad on *ref* gets a radial escape. Restrict with
    *pad_numbers* (specific pads) or *nets_only* (pads on these nets).
    """
    specs: list[BreakoutSpec] = []
    for fp in board.GetFootprints():
        if fp.GetReferenceAsString() != ref:
            continue
        for pad in fp.Pads():
            num = pad.GetNumber()
            net = pad.GetNetname()
            if pad_numbers is not None and num not in pad_numbers:
                continue
            if nets_only is not None and net not in nets_only:
                continue
            if not net:
                continue
            specs.append(
                BreakoutSpec(
                    ref=ref,
                    pad=num,
                    length_mm=length_mm,
                    layer=layer,
                    via_at_end=via_at_end,
                )
            )
    return specs


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
        out.sort(key=lambda t: t[1])
    else:
        for cpos, c in zip(corner_pos, corners):
            if (p1 - cpos) % per < ccw and (p1 - cpos) % per > 1e-9:
                out.append((c, (p1 - cpos) % per))
        out.sort(key=lambda t: t[1])
    return [c for c, _ in out]


def perimeter_tie_specs(
    board: "pcbnew.BOARD",
    ref: str,
    net_names: list[str] | None = None,
    *,
    margin_mm: float = 1.0,
    layer: str = "F.Cu",
    min_pads: int = 2,
) -> list[BreakoutSpec]:
    """Tie a footprint's same-net pads with a path routed around its bbox.

    For each net that has >= *min_pads* pads on *ref* (restrict with
    *net_names*), connect the two farthest-apart pads with a waypoint path that
    leaves each pad, hops just outside the footprint's bounding box, and walks
    the box perimeter between them. That keeps a power pour from fragmenting:
    the connector's spread power pads (e.g. USB-C VBUS) become one net island.
    """
    specs: list[BreakoutSpec] = []
    fp = next(
        (f for f in board.GetFootprints() if f.GetReferenceAsString() == ref), None
    )
    if fp is None:
        return specs
    bb = fp.GetBoundingBox()
    m = margin_mm
    x1 = pcbnew.ToMM(bb.GetX()) - m
    y1 = pcbnew.ToMM(bb.GetY()) - m
    x2 = pcbnew.ToMM(bb.GetX() + bb.GetWidth()) + m
    y2 = pcbnew.ToMM(bb.GetY() + bb.GetHeight()) + m

    by_net: dict[str, list] = {}
    for pad in fp.Pads():
        n = pad.GetNetname()
        if not n or (net_names is not None and n not in net_names):
            continue
        by_net.setdefault(n, []).append(pad)

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
        b1 = _nearest_on_rect(p1[0], p1[1], x1, y1, x2, y2)
        b2 = _nearest_on_rect(p2[0], p2[1], x1, y1, x2, y2)
        corners = _rect_perimeter_path(b1, b2, x1, y1, x2, y2)
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
                board, ref, net_names=sorted(power_nets), margin_mm=margin
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
            # Explicit (curated) path -- the caller owns collision-avoidance.
            points = [start_mm, *spec.waypoints]
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
