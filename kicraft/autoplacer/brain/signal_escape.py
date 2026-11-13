"""Pre-route escape reservations for dense and cross-leaf signal pads.

The shared stamper remains the authority for pad, copper, hole, mask, edge
and keepout clearance before any of these dog-bones is written.
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


def signal_escape_specs(
    board: Any,
    cfg: dict[str, Any] | None = None,
    external_net_names: set[str] | None = None,
) -> list[BreakoutSpec]:
    """Return guarded dog-bones for SMD signal fields and cross-leaf signals.

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
    power_nets = set(cfg.get("power_nets", ()))
    specs: list[BreakoutSpec] = []
    for footprint in sorted(board.GetFootprints(), key=lambda fp: fp.GetReferenceAsString()):
        # Wider local output pads can be caged by the same earlier power
        # copper as fine-pitch inputs. Reserve the whole signal pad field,
        # not just external nets or an arbitrary pitch/width subset.
        is_pad_field = sum(
            pad.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN)
            for pad in footprint.Pads()
        ) >= _MIN_PAD_FIELD_SIZE
        candidates = []
        pad_layers = {}
        for pad in footprint.Pads():
            if pad.GetNetname() in power_nets:
                continue
            is_signal = (
                pad.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN)
                and bool(pad.GetNetname())
                and not is_power_or_ground_name(pad.GetNetname())
            )
            if not is_signal or (
                not is_pad_field and pad.GetNetname() not in (external_net_names or ())
            ):
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
                    short_fanout_fallback=True,
                )
            )
    return specs


def unlanded_interface_components(board: Any, net_names: set[str]) -> list[dict[str, Any]]:
    """Find physically separate interface-pad components without a via or PTH.

    KiCad's native connectivity accounts for finite copper shapes, including
    contacts away from pad centres. A via elsewhere on the same net is not a
    landing for a disconnected component.
    """
    nets = {name for name in net_names if name and not is_power_or_ground_name(name)}
    pads = [pad for fp in board.GetFootprints() for pad in fp.Pads()
            if pad.GetNetname() in nets]
    board.BuildConnectivity()
    connectivity = board.GetConnectivity()
    visited: set[str] = set()
    missing = []
    for pad in pads:
        uid = pad.m_Uuid.AsString()
        if uid in visited:
            continue
        pending = [pad]
        members = []
        landed = False
        while pending:
            item = pending.pop()
            item_id = item.m_Uuid.AsString()
            if item_id in visited or item.GetNetname() != pad.GetNetname():
                continue
            visited.add(item_id)
            if item.Type() == pcbnew.PCB_VIA_T:
                landed = True
            if isinstance(item, pcbnew.PAD):
                members.append(f"{item.GetParentFootprint().GetReferenceAsString()}.{item.GetNumber()}")
                if item.GetAttribute() == pcbnew.PAD_ATTRIB_PTH:
                    landed = True
            pending.extend(connectivity.GetConnectedTracks(item))
            pending.extend(connectivity.GetConnectedPads(item))
        if not landed:
            missing.append({"net": pad.GetNetname(), "pads": sorted(members)})
    return missing


