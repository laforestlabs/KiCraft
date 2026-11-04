"""Pre-route escape reservations for dense fine-pitch signal pads.

KiCad Routing Tools cannot create an initial route when a fine-pitch pad is
already boxed in by locked plane/tie copper.  This module emits only guarded
``BreakoutSpec`` dog-bones; the shared stamper remains the authority for actual
pad, copper, hole and keepout clearance before any copper is written.
"""
from __future__ import annotations

import math
from typing import Any

import pcbnew

from kicraft.autoplacer.brain.breakout_stubs import (
    BreakoutSpec,
    STAMP_CLEARANCE_GUARD_MM,
    _own_clearance_mm,
)
from kicraft.autoplacer.fab_profile import fab_floors, fanout_via
from kicraft.design.models import is_power_or_ground_name

_MAX_PITCH_MM = 0.6
_MAX_NARROW_PAD_MM = 0.3
_MIN_PAD_FIELD_SIZE = 8
_ESCAPE_LENGTH_MM = 1.0


def _escape_axis(footprint: Any, pad: Any) -> tuple[float, float]:
    """Leave a rectangular pad along its outward long axis, not across its row."""
    centre = footprint.GetPosition()
    pos = pad.GetPosition()
    x_mm, y_mm = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
    dx = x_mm - pcbnew.ToMM(centre.x)
    dy = y_mm - pcbnew.ToMM(centre.y)
    size = pad.GetSize()
    if size.x != size.y:
        angle = math.radians(float(pad.GetOrientationDegrees()))
        ux, uy = (
            (math.cos(angle), -math.sin(angle))
            if size.x > size.y else (math.sin(angle), math.cos(angle))
        )
        if dx * ux + dy * uy < 0:
            ux, uy = -ux, -uy
    else:
        norm = math.hypot(dx, dy)
        ux, uy = (1.0, 0.0) if norm < 1e-6 else (dx / norm, dy / norm)
    return ux, uy


def _staggered_fanout_paths(
    footprint: Any,
    pads: list[Any],
    *,
    via_size_mm: float,
    width_mm: float,
    clearance_mm: float,
) -> list[tuple[Any, list[tuple[float, float]]]]:
    """Fan tracks before staggering vias; never discard an alternate signal.

    At larger netclass clearances a far-row track cannot pass a near-row via
    on the original pad pitch. Spread the tracks only after they leave the
    pad row, using the same resolved clearance as the shared stamper.
    """
    guard = STAMP_CLEARANCE_GUARD_MM
    lane_mm = via_size_mm / 2 + width_mm / 2 + clearance_mm + guard
    tip_spacing_mm = 2 * lane_mm
    rows: dict[tuple[float, float], list[tuple[Any, float, float, float]]] = {}
    for pad in pads:
        ux, uy = _escape_axis(footprint, pad)
        position, size = pad.GetPosition(), pad.GetSize()
        x, y = pcbnew.ToMM(position.x), pcbnew.ToMM(position.y)
        key = (round(ux, 9), round(uy, 9))
        rows.setdefault(key, []).append((
            pad, x * ux + y * uy, -x * uy + y * ux,
            max(pcbnew.ToMM(size.x), pcbnew.ToMM(size.y)) / 2,
        ))
    plans: dict[tuple, list[tuple[float, float]]] = {}
    selected_tips: list[tuple[float, float]] = []
    for (ux, uy), row in sorted(rows.items()):
        row.sort(key=lambda item: item[2])
        original = [item[2] for item in row]
        expanded = [original[0]]
        for tangent in original[1:]:
            expanded.append(max(tangent, expanded[-1] + lane_mm))
        shift = (sum(expanded) - sum(original)) / len(row)
        expanded = [tangent - shift for tangent in expanded]
        max_shift = max(abs(a - b) for a, b in zip(original, expanded, strict=True))
        lead_plane = max(normal + half_pad for _, normal, _, half_pad in row)
        lead_plane += clearance_mm + width_mm / 2 + guard
        # Limit the diagonal slope so its perpendicular track spacing stays
        # legal, not just the separation measured along the pad row.
        pitches = [b - a for a, b in zip(original, original[1:])]
        track_lane = width_mm + clearance_mm + guard
        slope = min(1.0, math.sqrt(max(0.0, (min(pitches) / track_lane) ** 2 - 1))) if pitches else 1.0
        if max_shift > 1e-9 and slope <= 1e-9:
            # An already illegal row cannot be repaired by fanning tracks.
            # Leave the original pitch for the stamper to reject explicitly.
            expanded, max_shift = original, 0.0
        fan_plane = max(
            max(normal for _, normal, _, _ in row) + _ESCAPE_LENGTH_MM,
            lead_plane + (max_shift / slope if max_shift else 0.0) + guard,
        )
        for (pad, _normal, tangent, _half_pad), target in zip(row, expanded, strict=True):
            def point(normal, lateral):
                return ux * normal - uy * lateral, uy * normal + ux * lateral

            lead, fan = point(lead_plane, tangent), point(fan_plane, target)
            # Keep near-row vias beyond the diagonal fanout segments, whose
            # perpendicular distance can be smaller than the final lane pitch.
            end_plane = fan_plane + lane_mm
            while True:
                tip = point(end_plane, target)
                if all(math.dist(tip, other) >= tip_spacing_mm for other in selected_tips):
                    break
                end_plane += tip_spacing_mm
            waypoints = [lead, fan]
            if tip != fan:
                waypoints.append(tip)
            plans[_pad_identity(pad)] = waypoints
            selected_tips.append(tip)
    return [(pad, plans[_pad_identity(pad)]) for pad in sorted(pads, key=_pad_number_key)]


def _pad_identity(pad: Any) -> tuple:
    """Stable physical identity across separate SWIG ``Pads()`` wrappers."""
    try:
        uuid = pad.m_Uuid.AsString()
        if uuid and uuid != "00000000-0000-0000-0000-000000000000":
            return ("uuid", uuid)
    except (AttributeError, TypeError):
        pass
    pos = pad.GetPosition()
    return ("physical", pad.GetNumber(), pos.x, pos.y)


def _pad_number_key(pad: Any) -> tuple[int, int | str]:
    """Natural pad-number ordering, with deterministic connector-label fallback."""
    number = pad.GetNumber()
    return (0, int(number)) if number.isdigit() else (1, number)


def _is_fine_pitch_signal_pad(footprint: Any, pad: Any) -> bool:
    """Whether *pad* needs a pre-router dog-bone reservation.

    This deliberately describes copper geometry rather than a package or
    reference designator: a multipad SMD footprint with a narrow pad beside
    another SMD pad is the class that loses its exit corridor to earlier locked
    copper.  Actual corridor legality is checked by ``add_breakout_stubs``.
    """
    if pad.GetAttribute() not in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN):
        return False
    if not pad.GetNetname() or is_power_or_ground_name(pad.GetNetname()):
        return False

    pads = [
        other
        for other in footprint.Pads()
        if other.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN)
    ]
    if len(pads) < _MIN_PAD_FIELD_SIZE:
        return False
    size = pad.GetSize()
    if min(pcbnew.ToMM(size.x), pcbnew.ToMM(size.y)) > _MAX_NARROW_PAD_MM:
        return False

    pos = pad.GetPosition()
    identity = _pad_identity(pad)
    return any(
        _pad_identity(other) != identity
        and (pcbnew.ToMM(other.GetPosition().x - pos.x) ** 2
             + pcbnew.ToMM(other.GetPosition().y - pos.y) ** 2) ** 0.5 <= _MAX_PITCH_MM
        for other in pads
    )


def fine_pitch_signal_escape_specs(
    board: Any,
    cfg: dict[str, Any] | None = None,
) -> list[BreakoutSpec]:
    """Return guarded, passably spaced dog-bones for dense SMD pad fields.

    The specs use the fabrication-profile fanout via, not the router's normal
    netclass via.  The latter is intentionally 0.6/0.3 mm and does not fit the
    fine-pitch geometry this reservation addresses.  No spec is an exemption:
    the shared breakout stamper drops one unless its complete path, tip via,
    drilled-hole spacing, rule areas and already-reserved copper all clear.
    Each stub uses the source pad's copper face, including bottom-side pads;
    a pad with no outer copper layer has no supported escape.
    """
    cfg = cfg or {}
    via_size_mm, via_drill_mm = fanout_via(cfg)
    floors = fab_floors(cfg)
    width_mm = floors["track_mm"]
    specs: list[BreakoutSpec] = []
    for footprint in sorted(board.GetFootprints(), key=lambda fp: fp.GetReferenceAsString()):
        candidates = []
        pad_layers = {}
        for pad in footprint.Pads():
            if not _is_fine_pitch_signal_pad(footprint, pad):
                continue
            if pad.IsOnLayer(pcbnew.F_Cu):
                layer, layer_name = pcbnew.F_Cu, "F.Cu"
            elif pad.IsOnLayer(pcbnew.B_Cu):
                layer, layer_name = pcbnew.B_Cu, "B.Cu"
            else:
                continue
            candidates.append(pad)
            pad_layers[_pad_identity(pad)] = (layer, layer_name)
        clearance_mm = max(
            floors["clearance_mm"],
            *(_own_clearance_mm(pad, pad_layers[_pad_identity(pad)][0], floors["clearance_mm"])
              for pad in candidates),
        ) if candidates else floors["clearance_mm"]
        for pad, waypoints in _staggered_fanout_paths(
            footprint, candidates, via_size_mm=via_size_mm,
            width_mm=width_mm, clearance_mm=clearance_mm,
        ):
            pos = pad.GetPosition()
            specs.append(
                BreakoutSpec(
                    ref=footprint.GetReferenceAsString(),
                    pad=pad.GetNumber(),
                    near_xy=(pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)),
                    waypoints=waypoints,
                    width_mm=width_mm,
                    layer=pad_layers[_pad_identity(pad)][1],
                    via_at_end=True,
                    strict_same_footprint=True,
                    via_size_mm=via_size_mm,
                    via_drill_mm=via_drill_mm,
                )
            )
    return specs
