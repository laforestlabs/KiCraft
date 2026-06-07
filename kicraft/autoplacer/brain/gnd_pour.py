"""Ground-plane finishing for routed leaf and parent boards.

After FreeRouting finishes *signal* routing, the GND/thermal pad of an IC or
module -- e.g. the exposed-pad (EP) array in the center of an ESP32-WROOM -- is
boxed in by the surrounding pin ring with no surface escape on its own layer.
The standard PCB fix is a dense **thermal-via array** dropping the pad to a
ground pour on the opposite layer (thermal transfer + electrical escape), plus a
**full B.Cu GND pour** that ties the vias and remaining GND copper together.

This runs as a post-routing step on each leaf (and the composed parent). The
default ``PCB_VIA`` is a through via (F.Cu<->B.Cu), so a via on an F.Cu thermal
pad connects it to the B.Cu pour automatically.

Thermal-pad detection: a pad is stitched when it is GND (or shares a pad number
with a GND pad -- EP sub-pads often share one number but only one carries the
net) AND is either large (>= ``thermal_pad_area_mm2``) or *interior* to a
multi-pin footprint (well inside the pad bounding box). That captures the whole
EP array while leaving perimeter GND pins -- which route normally -- alone.
"""
from __future__ import annotations

from typing import Any

import pcbnew


def _grid_positions(center_nm: int, half_nm: int, pitch_nm: int) -> list[int]:
    """Evenly spaced coordinates spanning [center-half, center+half] (nm)."""
    if half_nm <= 0:
        return [center_nm]
    intervals = int((2 * half_nm) // pitch_nm)
    if intervals < 1:
        return [center_nm]
    step = (2 * half_nm) // intervals
    return [center_nm - half_nm + i * step for i in range(intervals + 1)]


def pour_gnd_planes(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
    layers: tuple[str, ...] = ("B.Cu",),
) -> dict[str, Any]:
    """Create/reuse a GND zone on each given layer and fill them (no vias).

    Used after routing to tie every GND pad to ground via **thermal relief on
    its own layer**: an F.Cu SMD GND pad can't reach a B.Cu-only plane, so we
    pour F.Cu as well. Layer-stitching vias (F.Cu<->B.Cu) come from
    :func:`add_gnd_pour_and_thermal_vias`; this only manages the zones + fill,
    so it is safe to call repeatedly (idempotent w.r.t. zones, adds no vias).
    """
    cfg = cfg or {}
    gnd_name = cfg.get("gnd_zone_net", "GND")
    if not gnd_name:
        return {"zones": 0}
    board = pcbnew.LoadBoard(pcb_path)
    gnd_net = board.GetNetInfo().GetNetItem(gnd_name)
    if not gnd_net or gnd_net.GetNetCode() == 0:
        return {"zones": 0, "error": f"net {gnd_name!r} not found"}

    margin = pcbnew.FromMM(float(cfg.get("gnd_zone_margin_mm", 0.5)))
    rect = board.GetBoardEdgesBoundingBox()
    x1, y1 = rect.GetX() + margin, rect.GetY() + margin
    x2 = rect.GetX() + rect.GetWidth() - margin
    y2 = rect.GetY() + rect.GetHeight() - margin
    layer_map = {"B.Cu": pcbnew.B_Cu, "F.Cu": pcbnew.F_Cu}

    zones = 0
    for lname in layers:
        target_layer = layer_map.get(lname)
        if target_layer is None:
            continue
        zone = None
        for z in board.Zones():
            if (
                z.GetLayer() == target_layer
                and z.GetNetname() == gnd_name
                and not z.GetIsRuleArea()
            ):
                zone = z
                break
        if zone is None:
            zone = pcbnew.ZONE(board)
            zone.SetNet(gnd_net)
            zone.SetLayer(target_layer)
            zone.SetIsRuleArea(False)
            zone.SetLocalClearance(
                pcbnew.FromMM(float(cfg.get("zone_clearance_mm", 0.3)))
            )
            zone.SetMinThickness(
                pcbnew.FromMM(float(cfg.get("zone_min_thickness_mm", 0.25)))
            )
            zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
            zone.SetThermalReliefGap(
                pcbnew.FromMM(float(cfg.get("zone_thermal_gap_mm", 0.5)))
            )
            zone.SetThermalReliefSpokeWidth(
                pcbnew.FromMM(float(cfg.get("zone_thermal_spoke_mm", 0.5)))
            )
            zone.SetAssignedPriority(0)
            board.Add(zone)
            try:
                zone.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
            except Exception:
                try:
                    zone.SetIslandRemovalMode(0)
                except Exception:
                    pass
        outline = zone.Outline()
        outline.RemoveAllContours()
        outline.NewOutline()
        for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
            outline.Append(int(px), int(py))
        zones += 1

    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(pcb_path)
    return {"zones": zones}


def add_gnd_pour_and_thermal_vias(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stitch GND/thermal pads to a B.Cu GND pour with thermal vias, fill, save.

    Returns {gnd_pads_stitched, thermal_vias_added, zone_filled}. Idempotent
    w.r.t. the pour (reuses an existing B.Cu GND zone); call once per board.
    """
    cfg = cfg or {}
    gnd_name = cfg.get("gnd_zone_net", "GND")
    summary: dict[str, Any] = {
        "gnd_pads_stitched": 0,
        "thermal_vias_added": 0,
        "zone_filled": False,
        "net": gnd_name,
    }
    if not gnd_name:
        return summary

    board = pcbnew.LoadBoard(pcb_path)
    gnd_net = board.GetNetInfo().GetNetItem(gnd_name)
    if not gnd_net or gnd_net.GetNetCode() == 0:
        summary["error"] = f"net {gnd_name!r} not found"
        return summary
    gnd_code = gnd_net.GetNetCode()

    via_drill = pcbnew.FromMM(float(cfg.get("via_drill_mm", 0.3)))
    via_size = pcbnew.FromMM(float(cfg.get("via_size_mm", 0.6)))
    pitch = pcbnew.FromMM(float(cfg.get("thermal_via_pitch_mm", 1.2)))
    inset = pcbnew.FromMM(float(cfg.get("thermal_via_inset_mm", 0.5)))
    area_threshold = float(cfg.get("thermal_pad_area_mm2", 4.0))

    def _add_via(x: int, y: int) -> None:
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(int(x), int(y)))
        via.SetDrill(via_drill)
        try:
            via.SetWidth(via_size)
        except TypeError:
            via.SetWidth(pcbnew.F_Cu, via_size)
        via.SetNetCode(gnd_code)
        board.Add(via)

    # --- 1. Thermal-via arrays under GND thermal / exposed pads ---
    for fp in board.GetFootprints():
        pads = list(fp.Pads())
        smd = [
            p for p in pads
            if p.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN)
        ]
        if not smd:
            continue
        multipad = len(pads) >= 6  # ICs/modules; passives connect via their traces
        # Pad numbers that carry GND -- EP sub-pads share a number but often only
        # one is netted, so treat the whole number-group as GND.
        gnd_numbers = {p.GetNumber() for p in pads if p.GetNetCode() == gnd_code}

        for pad in smd:
            is_gnd = pad.GetNetCode() == gnd_code or pad.GetNumber() in gnd_numbers
            if not is_gnd:
                continue
            size = pad.GetSize()
            min_dim_mm = min(pcbnew.ToMM(size.x), pcbnew.ToMM(size.y))
            large = pcbnew.ToMM(size.x) * pcbnew.ToMM(size.y) >= area_threshold
            fits_via = min_dim_mm >= pcbnew.ToMM(via_size)
            # Stitch every GND pad on a multi-pin footprint that can host a via:
            # this ties BOTH the perimeter GND pins (so the F.Cu GND network drops
            # to the plane) AND the interior EP down to the B.Cu pour, unifying GND
            # into one net. Also stitch any large thermal pad. Tiny passive GND
            # pads are left to their traces (they reach GND through the IC vias).
            if not ((multipad and fits_via) or large):
                continue
            pos = pad.GetPosition()
            # Net any stray (number-shared) EP sub-pad to GND so it joins the plane.
            if pad.GetNetCode() != gnd_code:
                pad.SetNetCode(gnd_code)
            if large:
                vx = _grid_positions(pos.x, size.x // 2 - inset, pitch)
                vy = _grid_positions(pos.y, size.y // 2 - inset, pitch)
            else:
                vx, vy = [pos.x], [pos.y]
            placed = 0
            for x in vx:
                for y in vy:
                    _add_via(x, y)
                    placed += 1
            if placed:
                summary["gnd_pads_stitched"] += 1
                summary["thermal_vias_added"] += placed

    # --- 2. Full B.Cu GND pour (rule-area keepouts, e.g. the antenna, are
    #        respected automatically by the filler) ---
    layer_name = cfg.get("gnd_zone_layer", "B.Cu")
    target_layer = pcbnew.B_Cu if layer_name == "B.Cu" else pcbnew.F_Cu
    margin = pcbnew.FromMM(float(cfg.get("gnd_zone_margin_mm", 0.5)))
    rect = board.GetBoardEdgesBoundingBox()
    x1, y1 = rect.GetX() + margin, rect.GetY() + margin
    x2 = rect.GetX() + rect.GetWidth() - margin
    y2 = rect.GetY() + rect.GetHeight() - margin

    zone = None
    for z in board.Zones():
        if (
            z.GetLayer() == target_layer
            and z.GetNetname() == gnd_name
            and not z.GetIsRuleArea()
        ):
            zone = z
            break
    if zone is None:
        zone = pcbnew.ZONE(board)
        zone.SetNet(gnd_net)
        zone.SetLayer(target_layer)
        zone.SetIsRuleArea(False)
        zone.SetLocalClearance(pcbnew.FromMM(float(cfg.get("zone_clearance_mm", 0.3))))
        zone.SetMinThickness(
            pcbnew.FromMM(float(cfg.get("zone_min_thickness_mm", 0.25)))
        )
        zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
        zone.SetThermalReliefGap(
            pcbnew.FromMM(float(cfg.get("zone_thermal_gap_mm", 0.5)))
        )
        zone.SetThermalReliefSpokeWidth(
            pcbnew.FromMM(float(cfg.get("zone_thermal_spoke_mm", 0.5)))
        )
        zone.SetAssignedPriority(0)
        board.Add(zone)
    # Drop pour islands that aren't tied to GND (avoids isolated-copper DRC).
    try:
        zone.SetIslandRemovalMode(pcbnew.ISLAND_REMOVAL_MODE_ALWAYS)
    except Exception:
        try:
            zone.SetIslandRemovalMode(0)
        except Exception:
            pass
    outline = zone.Outline()
    outline.RemoveAllContours()
    outline.NewOutline()
    for px, py in ((x1, y1), (x2, y1), (x2, y2), (x1, y2)):
        outline.Append(int(px), int(py))

    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    summary["zone_filled"] = True

    board.Save(pcb_path)
    return summary
