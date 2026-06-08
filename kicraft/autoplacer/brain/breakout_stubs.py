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
