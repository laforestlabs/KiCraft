"""Ground-plane finishing for routed leaf and parent boards.

Ordinary SMD ground pads use guarded off-pad dogbones. In-pad thermal vias
are reserved for explicitly reviewed manufacturer land patterns; pad size and
net name alone do not establish permission.
"""

from __future__ import annotations

import math
import re
from typing import Any

from kicraft.autoplacer.fab_profile import fab_floors

import pcbnew
from kicraft.autoplacer.hardware.keepout_extract import (
    collect_track_via_rule_areas,
    via_intersects_rule_area,
)

_REVIEWED_THERMAL_LANDS = {
    ("ESP32-S3-MINI-1U-N4R2", "BULETM-SMD_ESPRESSIF_ESP32-S3-MINI-1U-N8"): ("61", 1.2),
    ("ESP32-S3-MINI-1U-N8", "BULETM-SMD_ESPRESSIF_ESP32-S3-MINI-1U-N8"): ("61", 1.2),
    ("ESP32-C3-MINI-1-H4X", "WIFIM-SMD_ESP32-C3-MINI-1"): ("49", 1.45),
    ("ESP32-C3-MINI-1-N4", "WIFIM-SMD_ESP32-C3-MINI-1"): ("49", 1.45),
}


def _authorized_thermal_pad(fp: Any, pad: Any) -> bool:
    """Reviewed module/footprint/land combinations, not a GND-size heuristic.

    Espressif S3 MINI-1U datasheet v1.7 fig. 11-2, pad 61:
    https://documentation.espressif.com/esp32-s3-mini-1_mini-1u_datasheet_en.pdf
    Espressif C3 MINI-1 datasheet v2.2 fig. 11-1, pad 49:
    https://documentation.espressif.com/esp32-c3_mini_1_datasheet_en.pdf
    Both drawings explicitly identify vias for the thermal pad.
    """
    land = _REVIEWED_THERMAL_LANDS.get((fp.GetValue(), fp.GetFPIDAsString().split(":")[-1]))
    if land is None:
        return False
    number, size_mm = land
    return (
        pad.GetNumber() == number
        and abs(pcbnew.ToMM(pad.GetSize().x) - size_mm) < 0.001
        and abs(pcbnew.ToMM(pad.GetSize().y) - size_mm) < 0.001
    )

def _smd_copper_layer(pad: Any) -> str | None:
    """Choose a copper face the SMD land actually occupies."""
    if pad.IsOnLayer(pcbnew.F_Cu):
        return "F.Cu"
    if pad.IsOnLayer(pcbnew.B_Cu):
        return "B.Cu"
    return None



def _via_hits_smd_pad(
    board: Any, point: Any, radius_iu: int, *, authorized_net: int | None = None
) -> bool:
    """Keep via copper clear of SMD mask openings on either board face."""
    return any(
        p.HitTest(
            point,
            radius_iu + max(
                pcbnew.FromMM(0.1),
                p.GetSolderMaskExpansion(pcbnew.F_Mask),
                p.GetSolderMaskExpansion(pcbnew.B_Mask),
            ),
        )
        for fp in board.GetFootprints()
        for p in fp.Pads()
        if p.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN)
        and not (
            authorized_net is not None
            and p.GetNetCode() == authorized_net
            and _authorized_thermal_pad(fp, p)
        )
    )



def _board_edge_geometry(board: Any) -> dict[str, Any] | None:
    """Tessellate every Edge.Cuts contour for exact via-edge clearance checks.

    A bounding box is only a cheap early rejection: concave outlines and
    internal cutouts need distances to every tessellated Edge.Cuts segment.
    Missing or malformed polygon geometry is deliberately not treated as a
    legal placement surface.
    """
    poly = pcbnew.SHAPE_POLY_SET()
    try:
        if not board.GetBoardPolygonOutlines(poly) or poly.OutlineCount() < 1:
            return None
        bbox = poly.BBox()
        box = (int(bbox.GetLeft()), int(bbox.GetTop()), int(bbox.GetRight()), int(bbox.GetBottom()))
        if box[0] >= box[2] or box[1] >= box[3]:
            return None
        edge_clearance_iu = int(
            pcbnew.FromMM(
                max(pcbnew.ToMM(board.GetDesignSettings().m_CopperEdgeClearance), 0.05)
            )
        )
        # A tessellated cutout can understate the true curved boundary. Reserve
        # KiCad's polygon approximation error as well as the physical clearance.
        edge_clearance_iu += max(0, int(board.GetDesignSettings().m_MaxError))
        segments: list[tuple[int, int, int, int]] = []
        for outline_index in range(poly.OutlineCount()):
            contours = [poly.Outline(outline_index)]
            for hole_index in range(poly.HoleCount(outline_index)):
                contours.append(poly.Hole(outline_index, hole_index))
            for contour in contours:
                point_count = contour.PointCount()
                if point_count < 3:
                    return None
                points = [contour.CPoint(index) for index in range(point_count)]
                segments.extend(
                    (int(a.x), int(a.y), int(b.x), int(b.y))
                    for a, b in zip(points, [*points[1:], points[0]])
                )
        return {
            "box": box,
            "clearance_iu": edge_clearance_iu,
            "polygon": poly,
            "segments": tuple(segments),
        }
    except Exception:
        return None


def _via_clears_board_edge(
    x_iu: int,
    y_iu: int,
    *,
    edge_geometry: dict[str, Any] | None,
    via_radius_iu: int,
) -> bool:
    """Whether the full via barrel clears every Edge.Cuts contour and cutout."""
    if edge_geometry is None:
        return False
    x1, y1, x2, y2 = edge_geometry["box"]
    inset = int(edge_geometry["clearance_iu"]) + via_radius_iu
    if x_iu < x1 + inset or x_iu > x2 - inset or y_iu < y1 + inset or y_iu > y2 - inset:
        return False
    point = pcbnew.VECTOR2I(x_iu, y_iu)
    if not edge_geometry["polygon"].Contains(point):
        return False

    min_distance_sq = None
    for ax, ay, bx, by in edge_geometry["segments"]:
        dx, dy = bx - ax, by - ay
        length_sq = dx * dx + dy * dy
        if length_sq == 0:
            distance_sq = (x_iu - ax) ** 2 + (y_iu - ay) ** 2
        else:
            t = max(0.0, min(1.0, ((x_iu - ax) * dx + (y_iu - ay) * dy) / length_sq))
            px, py = ax + t * dx, ay + t * dy
            distance_sq = (x_iu - px) ** 2 + (y_iu - py) ** 2
        min_distance_sq = (
            distance_sq if min_distance_sq is None else min(min_distance_sq, distance_sq)
        )
    return min_distance_sq is not None and min_distance_sq >= inset * inset



def _grid_positions(center_nm: int, half_nm: int, pitch_nm: int) -> list[int]:
    """Evenly spaced coordinates spanning [center-half, center+half] (nm)."""
    if half_nm <= 0:
        return [center_nm]
    intervals = int((2 * half_nm) // pitch_nm)
    if intervals < 1:
        return [center_nm]
    step = (2 * half_nm) // intervals
    return [center_nm - half_nm + i * step for i in range(intervals + 1)]


def _apply_gnd_pad_connection(zone: Any, cfg: dict[str, Any]) -> None:
    """Set the GND pour's pad-connection mode.

    Solid (full) by default -- the same call :func:`pour_power_planes` makes for
    power planes, for the same reason: on a dense pin field (an ESP32 module's
    GND ring, a fine-pitch connector) thermal-relief spokes need a gap wider than
    the pad pitch, so fewer than the DRC-required two spokes resolve and KiCad
    flags ``starved_thermal``. A solid tie has no spokes to starve and is the
    low-impedance connection a ground plane wants anyway; it only adds same-net
    copper, so it can never create a short or narrow another net's clearance. Set
    ``gnd_plane_pad_connection: "thermal"`` to restore relief (e.g. a
    hand-assembled board whose small passives would heat-sink into the plane).
    """
    if str(cfg.get("gnd_plane_pad_connection", "full")).lower() == "thermal":
        zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
        zone.SetThermalReliefGap(pcbnew.FromMM(float(cfg.get("zone_thermal_gap_mm", 0.5))))
        zone.SetThermalReliefSpokeWidth(pcbnew.FromMM(float(cfg.get("zone_thermal_spoke_mm", 0.5))))
    else:
        zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)


def _copper_layer_ids(board: Any) -> dict[str, int]:
    """Every enabled copper layer of this board, name -> layer id.

    Read from the board rather than a F.Cu/B.Cu constant, so a four-layer design's inner
    layers resolve instead of being skipped as unknown names.
    """
    return {board.GetLayerName(layer): layer for layer in board.GetEnabledLayers().CuStack()}


def _plane_layers(layer_ids: dict[str, int], layers: tuple[str, ...]) -> tuple[str, ...]:
    """The requested plane layers plus every inner copper layer the board declares.

    A stack with inner layers keeps its planes inside and its signals on the outer pair, so
    the inner layers are poured as well. A conventional two-layer board is unchanged.
    """
    inner = [name for name in layer_ids if re.fullmatch(r"In\d+\.Cu", name)]
    return tuple(dict.fromkeys([*layers, *inner]))


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

    A board with inner copper layers also gets the GND plane on those layers.
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
    layer_ids = _copper_layer_ids(board)
    layers = _plane_layers(layer_ids, tuple(layers))

    zones = 0
    for lname in layers:
        target_layer = layer_ids.get(lname)
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
            zone.SetLocalClearance(pcbnew.FromMM(float(cfg.get("zone_clearance_mm", 0.3))))
            zone.SetMinThickness(pcbnew.FromMM(float(cfg.get("zone_min_thickness_mm", 0.25))))
            _apply_gnd_pad_connection(zone, cfg)
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


def _detect_power_nets(board: Any, cfg: dict[str, Any]) -> list[str]:
    """Power rails to pour, in priority order.

    Honours an explicit ``power_plane_nets`` list; otherwise auto-detects nets
    that classify as power (but not the GND pour net), ranked by pad count so
    the dominant rail (e.g. VBUS on a USB board) is poured first. Limited to
    ``power_plane_max_nets`` to avoid fragmenting one layer between rivals.
    """
    explicit = cfg.get("power_plane_nets")
    gnd_name = cfg.get("gnd_zone_net", "GND")
    if explicit:
        return [n for n in explicit if n and n != gnd_name]

    from kicraft.design.models import is_power_or_ground_name

    pad_counts: dict[str, int] = {}
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            name = pad.GetNetname()
            if not name or name == gnd_name:
                continue
            if is_power_or_ground_name(name):
                pad_counts[name] = pad_counts.get(name, 0) + 1
    ranked = sorted(pad_counts, key=lambda n: (-pad_counts[n], n))
    max_nets = int(cfg.get("power_plane_max_nets", 1))
    return ranked[:max_nets]


def pour_power_planes(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
    layers: tuple[str, ...] = ("F.Cu",),
) -> dict[str, Any]:
    """Pour the primary power rail(s) as a plane on ``layers`` and fill.

    Power pads on a dense connector (paired USB-C VBUS pads, a regulator input,
    bulk caps) are tedious for the autorouter to tie together pad-to-pad. A
    power plane connects them through copper instead -- the same trick the GND
    pour uses for ground. Poured at a higher priority than the GND plane so the
    two coexist on a shared layer (power wins its region, GND fills the rest).

    Idempotent w.r.t. zones (reuses an existing same-net/layer zone); adds no
    vias. Returns ``{nets, zones}``.
    """
    cfg = cfg or {}
    summary: dict[str, Any] = {"nets": [], "zones": 0}
    if not cfg.get("power_plane_enabled", True):
        return summary

    board = pcbnew.LoadBoard(pcb_path)
    power_nets = _detect_power_nets(board, cfg)
    if not power_nets:
        return summary

    margin = pcbnew.FromMM(float(cfg.get("gnd_zone_margin_mm", 0.5)))
    rect = board.GetBoardEdgesBoundingBox()
    x1, y1 = rect.GetX() + margin, rect.GetY() + margin
    x2 = rect.GetX() + rect.GetWidth() - margin
    y2 = rect.GetY() + rect.GetHeight() - margin
    layer_map = _copper_layer_ids(board)
    priority = int(cfg.get("power_plane_priority", 1))

    for net_name in power_nets:
        net = board.GetNetInfo().GetNetItem(net_name)
        if not net or net.GetNetCode() == 0:
            continue
        for lname in layers:
            target_layer = layer_map.get(lname)
            if target_layer is None:
                continue
            zone = None
            for z in board.Zones():
                if (
                    z.GetLayer() == target_layer
                    and z.GetNetname() == net_name
                    and not z.GetIsRuleArea()
                ):
                    zone = z
                    break
            if zone is None:
                zone = pcbnew.ZONE(board)
                zone.SetNet(net)
                zone.SetLayer(target_layer)
                zone.SetIsRuleArea(False)
                zone.SetLocalClearance(pcbnew.FromMM(float(cfg.get("zone_clearance_mm", 0.3))))
                zone.SetMinThickness(pcbnew.FromMM(float(cfg.get("zone_min_thickness_mm", 0.25))))
                # Solid pad connection (not thermal): thermal-relief spokes need
                # a gap wider than a dense connector's pad pitch, so they never
                # form and the power pad stays isolated from the plane. A power
                # plane wants the low-impedance solid tie anyway.
                if str(cfg.get("power_plane_pad_connection", "full")).lower() == "thermal":
                    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_THERMAL)
                    zone.SetThermalReliefGap(
                        pcbnew.FromMM(float(cfg.get("zone_thermal_gap_mm", 0.5)))
                    )
                    zone.SetThermalReliefSpokeWidth(
                        pcbnew.FromMM(float(cfg.get("zone_thermal_spoke_mm", 0.5)))
                    )
                else:
                    zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)
                board.Add(zone)
            # Higher priority than GND (0) so power wins its region on a shared
            # layer; GND fills around it.
            zone.SetAssignedPriority(priority)
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
            summary["zones"] += 1
        summary["nets"].append(net_name)

    # Fill every zone together so priorities resolve power vs. GND overlap.
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(pcb_path)
    return summary


def _collect_net_clusters(
    board: "pcbnew.BOARD", net_name: str
) -> tuple[dict[int, list[int]], list[dict]]:
    """Geometric union-find over one net's pads/vias/tracks/fill islands.

    Returns ``(clusters, all_nodes)``: *clusters* maps a root index to the
    member indices of one electrically-contiguous group, *all_nodes* holds the
    node dicts those indices point into. Empty when the net has no items.
    """
    net = board.GetNetInfo().GetNetItem(net_name)
    if not net or net.GetNetCode() == 0:
        return {}, []
    net_code = net.GetNetCode()
    F, B = pcbnew.F_Cu, pcbnew.B_Cu

    def _pts_around(x_mm: float, y_mm: float, r_mm: float) -> list[tuple[float, float]]:
        if r_mm <= 0:
            return [(x_mm, y_mm)]
        out = [(x_mm, y_mm)]
        for k in range(8):
            a = k * 0.785398
            out.append((x_mm + r_mm * math.cos(a), y_mm + r_mm * math.sin(a)))
        return out

    nodes: list[dict] = []
    for fp in board.GetFootprints():
        for p in fp.Pads():
            if p.GetNetCode() != net_code:
                continue
            pos = p.GetPosition()
            x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            try:
                sz = p.GetSize()
            except TypeError:
                sz = p.GetSize(F)
            r = min(pcbnew.ToMM(sz.x), pcbnew.ToMM(sz.y)) / 2.0
            is_pth = p.GetAttribute() == pcbnew.PAD_ATTRIB_PTH
            nodes.append(
                {
                    "kind": "pad",
                    "ref": fp.GetReferenceAsString(),
                    "num": p.GetNumber(),
                    "layers": {F, B} if is_pth else {F if p.IsOnLayer(F) else B},
                    "pts": _pts_around(x, y, r),
                    "xy": (x, y),
                    "r": r,
                }
            )
    for t in board.GetTracks():
        if t.GetNetCode() != net_code:
            continue
        if t.GetClass() == "PCB_VIA":
            pos = t.GetPosition()
            x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            r = pcbnew.ToMM(t.GetWidth()) / 2.0
            nodes.append(
                {"kind": "via", "layers": {F, B}, "pts": _pts_around(x, y, r), "xy": (x, y), "r": r}
            )
        else:
            a, b2 = t.GetStart(), t.GetEnd()
            ax, ay = pcbnew.ToMM(a.x), pcbnew.ToMM(a.y)
            bx, by = pcbnew.ToMM(b2.x), pcbnew.ToMM(b2.y)
            # Sample ALONG the segment, not just its endpoints: a fill island
            # abutting a track mid-run (an edge-spine rail beside the pour)
            # unions through an interior point that endpoint-only probing
            # misses, splitting one electrical cluster into phantoms.
            length = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
            n_samples = max(1, min(24, int(length / 1.0)))
            pts = [
                (ax + (bx - ax) * k / n_samples, ay + (by - ay) * k / n_samples)
                for k in range(n_samples + 1)
            ]
            nodes.append(
                {
                    "kind": "trk",
                    "layers": {t.GetLayer()},
                    "pts": pts,
                    "xy": ((ax + bx) / 2, (ay + by) / 2),
                }
            )
    islands: list[dict] = []
    for z in board.Zones():
        if z.GetNetname() != net_name or z.GetIsRuleArea():
            continue
        layer = z.GetLayer()
        fill = z.GetFilledPolysList(layer)
        for i in range(fill.OutlineCount()):
            bb = fill.Outline(i).BBox()
            islands.append(
                {
                    "kind": "island",
                    "layers": {layer},
                    "fill": fill,
                    "idx": i,
                    "xy": (pcbnew.ToMM(bb.Centre().x), pcbnew.ToMM(bb.Centre().y)),
                }
            )

    all_nodes = nodes + islands
    parent = list(range(len(all_nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    # pads/vias/tracks <-> islands: any probe point inside the island's fill.
    for ii, isl in enumerate(islands):
        gi = len(nodes) + ii
        for ni, n in enumerate(nodes):
            if not (n["layers"] & isl["layers"]):
                continue
            for px, py in n["pts"]:
                if isl["fill"].Contains(
                    pcbnew.VECTOR2I(pcbnew.FromMM(px), pcbnew.FromMM(py)), isl["idx"]
                ):
                    union(ni, gi)
                    break
    # pads/vias/tracks <-> each other: shared layer + a probe point within
    # 0.05 mm of another's probe point (track ends land on pad/via centres).
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if not (nodes[i]["layers"] & nodes[j]["layers"]):
                continue
            done = False
            for ax, ay in nodes[i]["pts"]:
                for bx, by in nodes[j]["pts"]:
                    if (ax - bx) ** 2 + (ay - by) ** 2 < 0.0025:
                        union(i, j)
                        done = True
                        break
                if done:
                    break

    clusters: dict[int, list[int]] = {}
    for i in range(len(all_nodes)):
        clusters.setdefault(find(i), []).append(i)
    return clusters, all_nodes


def repair_stranded_net(
    pcb_path: str,
    net_name: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tie a poured net's stranded clusters back to its main one with guarded tracks.

    A poured net (the GND plane, or a power rail plane) is supposed to reach
    every one of its pads through copper. In a crowded region the fill
    fragments around foreign tracks/pads, and a pad can end up on a tiny fill
    island with no path to the main cluster: no via to drop through, no
    same-net mate for a shield tie, no routed track (GND is never given to
    KiCad Routing Tools; a fine-pitch part's supply pad may be unreachable for it --
    KC-Z57JEZ U1 +3V3). This post-pour pass finds every cluster of
    ``net_name`` isolated from the main one (geometric union-find over
    pads/vias/tracks/fill islands) and stamps a direct same-net track from a
    stranded pad to the nearest main-cluster pad/via via
    :func:`add_breakout_stubs` -- inheriting its foreign-pad,
    existing-copper, netclass and outline guards -- then refills the zones so
    the pour closes around the new tie. A tie whose straight path is blocked
    is skipped (the board is no worse than before).
    """
    cfg = cfg or {}
    summary: dict[str, Any] = {
        "net": net_name,
        "clusters": 0,
        "stranded": 0,
        "tied": 0,
        "skipped": [],
        "unresolved": 0,
    }
    if not net_name:
        return summary

    board = pcbnew.LoadBoard(pcb_path)
    max_tie_mm = float(cfg.get("gnd_strand_repair_max_mm", 30.0))

    clusters, all_nodes = _collect_net_clusters(board, net_name)
    summary["clusters"] = len(clusters)
    if len(clusters) <= 1:
        return summary
    main_root = max(clusters, key=lambda r: len(clusters[r]))
    main_targets = [
        all_nodes[i] for i in clusters[main_root] if all_nodes[i]["kind"] in ("pad", "via")
    ]
    if not main_targets:
        return summary

    from kicraft.autoplacer.brain.breakout_stubs import (
        BreakoutSpec,
        add_breakout_stubs,
    )

    F, B = pcbnew.F_Cu, pcbnew.B_Cu
    via_nodes = [n for n in all_nodes if n["kind"] == "via"]
    max_targets = int(cfg.get("gnd_strand_repair_max_targets", 5))
    for root, members in clusters.items():
        if root == main_root:
            continue
        summary["stranded"] += 1
        src = next((all_nodes[i] for i in members if all_nodes[i]["kind"] == "pad"), None)
        if src is None:
            summary["skipped"].append("cluster_without_pad")
            continue
        sx, sy = src["xy"]
        # Layers the tie can START on: the pad's own copper -- or, when a
        # same-net via barrel overlaps the pad centre (the escape-stitch
        # stub+via shape), the far layer through that via. A tie drawn on a
        # layer the source pad does not reach is dead copper that REPORTS
        # success (the run_01 ESP32 signature, together with the wrong-pad
        # lookup near_xy now prevents).
        via_bridge = any(
            ((n["xy"][0] - sx) ** 2 + (n["xy"][1] - sy) ** 2) ** 0.5 <= n["r"] for n in via_nodes
        )
        start_layers = [
            (lname, lid)
            for lname, lid in (("F.Cu", F), ("B.Cu", B))
            if lid in src["layers"] or via_bridge
        ]
        start_layers.sort(key=lambda t: t[1] not in src["layers"])
        ranked = sorted(
            main_targets,
            key=lambda t: (t["xy"][0] - sx) ** 2 + (t["xy"][1] - sy) ** 2,
        )
        # The straight line to the NEAREST target often runs through exactly
        # the copper wall that stranded this cluster in the first place -- so
        # walk outward through the nearest few targets (different directions)
        # on every feasible layer until one tie lands.
        tied = False
        for tgt in ranked[:max_targets]:
            d = ((tgt["xy"][0] - sx) ** 2 + (tgt["xy"][1] - sy) ** 2) ** 0.5
            if d > max_tie_mm:
                break  # ranked by distance: everything after is farther
            for layer_name, layer_id in start_layers:
                # The tie END must land on copper too: an SMD target pad
                # bonds only on its own layer (vias and PTH pads span both).
                if layer_id not in tgt["layers"]:
                    continue
                res = add_breakout_stubs(
                    pcb_path,
                    [
                        BreakoutSpec(
                            ref=src["ref"],
                            pad=src["num"],
                            waypoints=[tgt["xy"]],
                            layer=layer_name,
                            near_xy=src["xy"],
                        )
                    ],
                    cfg=cfg,
                )
                if res.get("stubs", 0):
                    summary["tied"] += 1
                    tied = True
                    break
            if tied:
                break
        if not tied:
            summary["skipped"].append(f"{src['ref']}.{src['num']}:no_clear_path")

    if summary["tied"]:
        board = pcbnew.LoadBoard(pcb_path)
        pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        board.Save(pcb_path)
        # Verify on the refilled board: a tie that stamped but did not merge
        # its cluster (wrong pad, wrong layer, refill split) must be LOUD --
        # the silent version shipped boards whose ratsnest still showed the
        # strand while the build log said "tied".
        clusters, _ = _collect_net_clusters(board, net_name)
    summary["unresolved"] = max(0, len(clusters) - 1)
    if summary["unresolved"]:
        print(
            f"  WARNING: {net_name} strand repair left {summary['unresolved']} "
            f"cluster(s) disconnected (tied={summary['tied']}, "
            f"skipped={summary['skipped']})"
        )
    return summary


def repair_stranded_gnd(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GND strand repair: :func:`repair_stranded_net` on the GND pour net."""
    cfg = cfg or {}
    gnd_name = cfg.get("gnd_zone_net", "GND")
    if not cfg.get("gnd_strand_repair_enabled", True):
        return {"net": gnd_name, "clusters": 0, "stranded": 0, "tied": 0, "skipped": []}
    return repair_stranded_net(pcb_path, gnd_name, cfg)


def repair_stranded_power(
    pcb_path: str,
    nets: list[str] | None = None,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Strand repair for the poured power rail(s) -- the GND repair's twin.

    The power plane fragments exactly like the GND plane does (KC-Z57JEZ: the
    +3V3 F.Cu pour split into two islands around a fine-pitch LGA, stranding
    the part's supply pads), but the repair pass historically ran for GND
    only. Runs :func:`repair_stranded_net` for each poured rail. ``nets``
    defaults to the same detection :func:`pour_power_planes` uses, so callers
    that ignored its return value still repair the right rails.
    """
    cfg = cfg or {}
    out: dict[str, Any] = {"nets": [], "stranded": 0, "tied": 0, "skipped": [], "unresolved": 0}
    if not cfg.get("power_strand_repair_enabled", True):
        return out
    if nets is None:
        if not cfg.get("power_plane_enabled", True):
            return out
        board = pcbnew.LoadBoard(pcb_path)
        nets = _detect_power_nets(board, cfg)
        del board
    for net_name in nets or []:
        s = repair_stranded_net(pcb_path, net_name, cfg)
        out["nets"].append(net_name)
        out["stranded"] += s["stranded"]
        out["tied"] += s["tied"]
        out["unresolved"] += s.get("unresolved", 0)
        out["skipped"].extend(f"{net_name}:{item}" for item in s["skipped"])
    return out


def gnd_escape_specs(
    board: Any,
    cfg: dict[str, Any] | None = None,
) -> list:
    """Pre-route escape specs for fine-pitch GND pads that can't host a via.

    The post-route escape pass (in :func:`add_gnd_pour_and_thermal_vias`)
    runs LAST, after the signal breakout stubs and KiCad Routing Tools have consumed
    every exit around a dense pad row -- so exactly the pads that most need a
    plane bond find no legal path (KC-UXASHQ U1.6: hemmed in by a signal stub
    and routed tracks, left unconnected). Signal dog-bones reserve their
    fine-pitch exits first; these GND escapes then use the remaining legal
    corridors, and the router routes around all locked copper. Each spec carries
    ``via_at_end`` so the stub bonds its pad to the future B.Cu plane; the
    post-route pass sees the via and skips the pad (no double escape).
    """
    cfg = cfg or {}
    gnd_name = cfg.get("gnd_zone_net", "GND")
    if not gnd_name or not cfg.get("gnd_pre_escape", True):
        return []
    gnd_net = board.GetNetInfo().GetNetItem(gnd_name)
    if not gnd_net or gnd_net.GetNetCode() == 0:
        return []
    gnd_code = gnd_net.GetNetCode()

    from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec

    length = float(cfg.get("gnd_escape_length_mm", 1.5))
    specs: list = []
    for fp in board.GetFootprints():
        pads = list(fp.Pads())
        for pad in pads:
            if pad.GetAttribute() not in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN):
                continue
            if pad.GetNetCode() != gnd_code:
                continue
            if _authorized_thermal_pad(fp, pad):
                continue
            layer = _smd_copper_layer(pad)
            if layer is None:
                continue
            specs.append(
                BreakoutSpec(
                    ref=fp.GetReferenceAsString(),
                    pad=pad.GetNumber(),
                    length_mm=length,
                    layer=layer,
                    via_at_end=True,
                    via_outside_smd_pads=True,
                    near_xy=(pcbnew.ToMM(pad.GetPosition().x), pcbnew.ToMM(pad.GetPosition().y)),
                    strict_same_footprint=True,
                )
            )
    return specs


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

    # Track/via keep-outs (the footprint-embedded antenna near-field, board-level
    # rule areas) the finisher must keep its vias and escape stubs out of.
    keepout_areas = collect_track_via_rule_areas(board)

    def _in_keepout(pt: Any, clearance_iu: int = 0) -> bool:
        center = (pcbnew.ToMM(pt.x), pcbnew.ToMM(pt.y))
        radius_mm = pcbnew.ToMM(clearance_iu)
        return any(via_intersects_rule_area(center, radius_mm, area) for area in keepout_areas)

    via_drill = pcbnew.FromMM(float(cfg.get("via_drill_mm", 0.3)))
    via_size = pcbnew.FromMM(float(cfg.get("via_size_mm", 0.6)))
    pitch = pcbnew.FromMM(float(cfg.get("thermal_via_pitch_mm", 1.2)))
    area_threshold = float(cfg.get("thermal_pad_area_mm2", 4.0))
    inset = pcbnew.FromMM(float(cfg.get("thermal_via_inset_mm", 0.5)))
    floor_mm = fab_floors(cfg)["clearance_mm"]
    summary["thermal_vias_blocked"] = 0
    summary["escape_stitched"] = 0

    from kicraft.autoplacer.brain.breakout_stubs import _own_clearance_mm

    # GND-side clearance: KiCad resolves a pair as the LARGER of the two
    # items' netclass clearances, and GND rides the Power class (0.30 mm) on
    # generated boards -- a via held only to the 0.153 routing floor can
    # pass this guard yet land 0.26 mm from a Default-class track, a hard
    # Power-netclass DRC error (the KC-UXASHQ escape-via signature).
    # Take the true max across ALL footprints' GND pads: the old early-break
    # sampled only the first footprint with any GND pad, silently ignoring a
    # stricter netclass override elsewhere on the board -- reproducing the
    # exact KC-UXASHQ class this guard exists to prevent (2026-07-19 §3.5).
    gnd_cl_mm = floor_mm
    for _fp in board.GetFootprints():
        gp = next((p for p in _fp.Pads() if p.GetNetCode() == gnd_code), None)
        if gp is not None:
            gnd_cl_mm = max(gnd_cl_mm, _own_clearance_mm(gp, pcbnew.B_Cu, floor_mm))

    via_r_mm = pcbnew.ToMM(via_size) / 2.0
    via_radius_iu = int(via_size) // 2
    via_drill_r_mm = pcbnew.ToMM(via_drill) / 2.0
    via_edge_geometry = _board_edge_geometry(board)
    copper_obstacles: list[tuple[Any, int]] = []  # (item, HitTest margin)
    for t in board.GetTracks():
        if t.GetNetCode() == gnd_code:
            continue
        t_layer = pcbnew.B_Cu if t.GetClass() == "PCB_VIA" else t.GetLayer()
        item_cl = _own_clearance_mm(t, t_layer, floor_mm)
        copper_obstacles.append((t, int(pcbnew.FromMM(via_r_mm + max(gnd_cl_mm, item_cl)))))
    for ofp in board.GetFootprints():
        for op in ofp.Pads():
            if op.GetNetCode() == gnd_code:
                continue
            item_cl = _own_clearance_mm(op, pcbnew.B_Cu, floor_mm)
            copper_obstacles.append((op, int(pcbnew.FromMM(via_r_mm + max(gnd_cl_mm, item_cl)))))

    # Drilled holes: a new via's hole wall must keep the board's hole-to-hole
    # minimum from EVERY existing hole (vias and PTH pads) -- nothing checked
    # this before (the 0.036 mm hole pair on KC-UXASHQ). Successfully stamped
    # vias join the list so this pass spaces its own vias too.
    h2h = pcbnew.ToMM(board.GetDesignSettings().m_HoleToHoleMin)
    hole_min_mm = h2h if h2h > 0 else float(cfg.get("hole_to_hole_min_mm", 0.25))
    holes: list[tuple[float, float, float]] = []  # (x_mm, y_mm, hole_radius_mm)
    for t in board.GetTracks():
        if t.GetClass() == "PCB_VIA":
            p = t.GetPosition()
            holes.append((pcbnew.ToMM(p.x), pcbnew.ToMM(p.y), pcbnew.ToMM(t.GetDrillValue()) / 2.0))
    for ofp in board.GetFootprints():
        for op in ofp.Pads():
            ds = op.GetDrillSize()
            if ds.x > 0 or ds.y > 0:
                p = op.GetPosition()
                holes.append(
                    (
                        pcbnew.ToMM(p.x),
                        pcbnew.ToMM(p.y),
                        max(pcbnew.ToMM(ds.x), pcbnew.ToMM(ds.y)) / 2.0,
                    )
                )

    def _via_blocked(x: int, y: int) -> bool:
        """True when a GND via at (x, y) would violate another net's copper
        (netclass pair clearance) or any drilled hole (hole-to-hole minimum).

        This runs on a ROUTED board (leaf) or a composed parent full of leaf
        traces; a via stamped blind through a B.Cu track of another net is a
        hard short the router can never repair (the IP2368-bank incident:
        seven shorts from exactly this). Same-net copper is a valid landing.
        """
        if not _via_clears_board_edge(
            int(x), int(y), edge_geometry=via_edge_geometry, via_radius_iu=via_radius_iu
        ):
            return True
        pt = pcbnew.VECTOR2I(int(x), int(y))
        # A via whose barrel overlaps a track/via keep-out (the antenna
        # near-field) is an items_not_allowed DRC error the router can't repair.
        if _in_keepout(pt, int(pcbnew.FromMM(via_r_mm))):
            return True
        if _via_hits_smd_pad(board, pt, via_radius_iu, authorized_net=gnd_code):
            return True
        if any(item.HitTest(pt, margin) for item, margin in copper_obstacles):
            return True
        xm, ym = pcbnew.ToMM(int(x)), pcbnew.ToMM(int(y))
        return any(
            ((xm - hx) ** 2 + (ym - hy) ** 2) ** 0.5 < hr + via_drill_r_mm + hole_min_mm
            for hx, hy, hr in holes
        )

    def _add_via(x: int, y: int) -> bool:
        if _via_blocked(x, y):
            summary["thermal_vias_blocked"] += 1
            return False
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(int(x), int(y)))
        via.SetDrill(via_drill)
        try:
            via.SetWidth(via_size)
        except TypeError:
            via.SetWidth(pcbnew.F_Cu, via_size)
        via.SetNetCode(gnd_code)
        board.Add(via)
        holes.append((pcbnew.ToMM(int(x)), pcbnew.ToMM(int(y)), via_drill_r_mm))
        return True

    # GND pads that need stitching but cannot host an in-pad via: escape them
    # with a short guarded stub + end via instead (see below).
    escape_pads: list[tuple[str, str, tuple[float, float], str]] = []

    # GND vias already on the board (e.g. a pre-route escape stub's tip via
    # stamped by gnd_escape_specs): a pad one of those already bonds to the
    # plane must not be escaped AGAIN post-route.
    gnd_via_pts = {
        (t.GetPosition().x, t.GetPosition().y)
        for t in board.GetTracks()
        if t.GetClass() == "PCB_VIA" and t.GetNetCode() == gnd_code
    }

    def _already_bonded(pad) -> bool:
        for track in board.GetTracks():
            if (
                track.GetClass() == "PCB_VIA"
                or track.GetNetCode() != gnd_code
                or not pad.IsOnLayer(track.GetLayer())
            ):
                continue
            a, b = track.GetStart(), track.GetEnd()
            if (pad.HitTest(a) and (b.x, b.y) in gnd_via_pts) or (
                pad.HitTest(b) and (a.x, a.y) in gnd_via_pts
            ):
                return True
        return False

    # --- 1. Thermal-via arrays under GND thermal / exposed pads ---
    for fp in board.GetFootprints():
        pads = list(fp.Pads())
        smd = [
            p for p in pads if p.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN)
        ]
        if not smd:
            continue
        # Pad numbers that carry GND -- EP sub-pads share a number but often only
        # one is netted, so treat the whole number-group as GND.
        gnd_numbers = {p.GetNumber() for p in pads if p.GetNetCode() == gnd_code}

        for pad in smd:
            authorized = _authorized_thermal_pad(fp, pad)
            if pad.GetNetCode() != gnd_code and not (
                authorized and pad.GetNumber() in gnd_numbers and pad.GetNetCode() == 0
            ):
                continue
            size = pad.GetSize()
            large = pcbnew.ToMM(size.x) * pcbnew.ToMM(size.y) >= area_threshold
            pos = pad.GetPosition()
            # Net any stray (number-shared) EP sub-pad to GND so it joins the plane.
            if pad.GetNetCode() != gnd_code:
                pad.SetNetCode(gnd_code)
            if not authorized:
                layer = _smd_copper_layer(pad)
                if layer is not None and not _already_bonded(pad):
                    escape_pads.append((
                        fp.GetReferenceAsString(), pad.GetNumber(),
                        (pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)),
                        layer,
                    ))
                continue
            if large:
                vx = _grid_positions(pos.x, size.x // 2 - inset, pitch)
                vy = _grid_positions(pos.y, size.y // 2 - inset, pitch)
            else:
                vx, vy = [pos.x], [pos.y]
            placed = 0
            for x in vx:
                for y in vy:
                    if _add_via(x, y):
                        placed += 1
            if placed:
                summary["gnd_pads_stitched"] += 1
                summary["thermal_vias_added"] += placed

    # --- 1b. Escape-stitch the GND pads that cannot host an in-pad via: a
    #         short locked stub out of the pad with a via at its tip bonds the
    #         pad (and its F.Cu pour cluster) to the B.Cu plane. Reuses the
    #         breakout-stub machinery, which already guards against foreign
    #         pads, existing tracks/vias, netclass pair clearance, and the
    #         board outline -- exactly the guards a routed board demands.
    if escape_pads:
        board.Save(pcb_path)
        try:
            from kicraft.autoplacer.brain.breakout_stubs import (
                BreakoutSpec,
                add_breakout_stubs,
            )

            specs = [
                BreakoutSpec(
                    ref=ref,
                    pad=num,
                    length_mm=float(cfg.get("gnd_escape_length_mm", 1.5)),
                    layer=layer,
                    via_at_end=True,
                    via_outside_smd_pads=True,
                    strict_same_footprint=True,
                    near_xy=xy,
                )
                for ref, num, xy, layer in escape_pads
            ]
            res = add_breakout_stubs(pcb_path, specs, cfg=cfg)
            summary["escape_stitched"] = res.get("vias", 0)
            summary["escape_skipped"] = res.get("skipped", [])
        except Exception as exc:  # finishing helper must never fail the board
            summary["escape_error"] = str(exc)
        board = pcbnew.LoadBoard(pcb_path)
        gnd_net = board.GetNetInfo().GetNetItem(gnd_name)

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
        if z.GetLayer() == target_layer and z.GetNetname() == gnd_name and not z.GetIsRuleArea():
            zone = z
            break
    if zone is None:
        zone = pcbnew.ZONE(board)
        zone.SetNet(gnd_net)
        zone.SetLayer(target_layer)
        zone.SetIsRuleArea(False)
        zone.SetLocalClearance(pcbnew.FromMM(float(cfg.get("zone_clearance_mm", 0.3))))
        zone.SetMinThickness(pcbnew.FromMM(float(cfg.get("zone_min_thickness_mm", 0.25))))
        _apply_gnd_pad_connection(zone, cfg)
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

    # A stack with inner copper layers carries its planes there (signals keep the outer
    # pair), so the GND plane grows to the inner layers. The single-zone block above only
    # knows the one configured outer layer; pour_gnd_planes owns the inner-layer set.
    inner = [name for name in _copper_layer_ids(board) if re.fullmatch(r"In\d+\.Cu", name)]
    if inner:
        poured = pour_gnd_planes(pcb_path, cfg, layers=tuple(inner))
        summary["inner_plane_zones"] = int(poured.get("zones", 0))
    return summary


def stamp_gnd_edge_spine(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tie the GND pads of connectors banked along one board edge with a
    locked same-net bus track in the pads-to-edge corridor.

    A row of edge connectors is the canonical GND-strand shape (KC-YXQ4EC:
    sixteen 1x3 servo headers -> 13 disconnected B.Cu pour islands): every
    header's signal/power fan-out fences the pour between neighbours, so each
    GND pad sits on its own strip and no straight tie back to the main plane
    clears the fences. The one corridor guaranteed free of that fencing is
    OUTBOARD of the pads -- nothing routes between a connector row and the
    board edge it faces. This pass draws a spine there: one jog per GND pad
    out to the corridor, chained pad-to-pad along the edge. Links are
    stamped through :func:`add_breakout_stubs`, inheriting its foreign-copper,
    outline and hole guards -- a blocked link is skipped and the chain
    resumes past it as separate sub-chains (the board is never made worse).

    Only edges with >= 2 zoned connectors carrying PTH GND pads get a spine,
    and the caller only invokes this when the GND net is actually fragmented.
    Returns ``{stubs, edges: {edge: {...}}}``.
    """
    cfg = cfg or {}
    summary: dict[str, Any] = {"stubs": 0, "edges": {}}
    zones = cfg.get("component_zones") or {}
    gnd_name = cfg.get("gnd_zone_net", "GND")
    if not zones or not gnd_name:
        return summary

    board = pcbnew.LoadBoard(pcb_path)
    gnd_net = board.GetNetInfo().GetNetItem(gnd_name)
    if not gnd_net or gnd_net.GetNetCode() == 0:
        return summary
    gnd_code = gnd_net.GetNetCode()

    box = board.GetBoardEdgesBoundingBox()
    if box.GetWidth() <= 0 or box.GetHeight() <= 0:
        return summary
    edge_line = {
        "left": pcbnew.ToMM(box.GetLeft()),
        "right": pcbnew.ToMM(box.GetRight()),
        "top": pcbnew.ToMM(box.GetTop()),
        "bottom": pcbnew.ToMM(box.GetBottom()),
    }
    outward = {"left": -1.0, "right": 1.0, "top": -1.0, "bottom": 1.0}
    try:
        ecl = max(pcbnew.ToMM(board.GetDesignSettings().m_CopperEdgeClearance), 0.05)
    except AttributeError:
        ecl = 0.3
    floor_mm = fab_floors(cfg)["clearance_mm"]
    max_inset = float(cfg.get("gnd_edge_spine_max_inset_mm", 8.0))

    from kicraft.autoplacer.brain.breakout_stubs import (
        BreakoutSpec,
        add_breakout_stubs,
        _own_clearance_mm,
    )

    fps = {fp.GetReference(): fp for fp in board.GetFootprints()}
    # edge -> [(ref, pad_number, x_mm, y_mm, pad_outer_radius_mm)]
    groups: dict[str, list[tuple[str, str, float, float, float]]] = {}
    # edge -> the largest pair clearance the stamping guard will hold against
    # any pad of the banked connectors (netclass-aware; the guard adds 10 um).
    group_cl: dict[str, float] = {}
    for ref, zone in zones.items():
        edge = (zone or {}).get("edge")
        if edge not in edge_line:
            continue
        fp = fps.get(ref)
        if fp is None:
            continue
        for p in fp.Pads():
            cl = _own_clearance_mm(p, pcbnew.F_Cu, floor_mm)
            group_cl[edge] = max(group_cl.get(edge, floor_mm), cl)
            if p.GetNetCode() != gnd_code:
                continue
            if p.GetAttribute() != pcbnew.PAD_ATTRIB_PTH:
                continue  # the spine relies on a pad reachable on either layer
            pos = p.GetPosition()
            try:
                sz = p.GetSize()
            except TypeError:
                sz = p.GetSize(pcbnew.F_Cu)
            groups.setdefault(edge, []).append(
                (
                    ref,
                    p.GetNumber(),
                    pcbnew.ToMM(pos.x),
                    pcbnew.ToMM(pos.y),
                    max(pcbnew.ToMM(sz.x), pcbnew.ToMM(sz.y)) / 2.0,
                )
            )

    del board

    for edge, pads in groups.items():
        if len({ref for ref, *_ in pads}) < 2:
            continue  # a lone connector has nothing to chain to
        vertical = edge in ("left", "right")
        sign = outward[edge]
        line = edge_line[edge]
        # Drop pads implausibly far inboard (shaped outline / stale zone).
        pads = [p for p in pads if abs(line - (p[2] if vertical else p[3])) <= max_inset]
        if len({ref for ref, *_ in pads}) < 2:
            summary["edges"][edge] = {"skipped": "pads_not_near_edge"}
            continue
        pads.sort(key=(lambda p: p[3]) if vertical else (lambda p: p[2]))
        # Corridor: the rail must hold the stamping guard's pair clearance
        # (largest netclass clearance among the bank's pads, +10 um guard)
        # from the outermost pad copper, AND keep the copper edge inside the
        # board's copper-to-edge clearance. Centre the rail in that band --
        # hugging either bound leaves the sampled HitTest guard a hair away
        # from rejecting every chained segment.
        outermost = max((p[2] if vertical else p[3]) * sign + p[4] for p in pads)
        pair = 0.01 + max(floor_mm, group_cl.get(edge, floor_mm))
        width = float(cfg.get("gnd_edge_spine_width_mm", 0.25))
        spine_c = None
        for w in (width, floor_mm):
            band_min = outermost + pair + w / 2.0
            band_max = line * sign - ecl - w / 2.0
            if band_min <= band_max:
                spine_c = ((band_min + band_max) / 2.0) * sign
                width = w
                break
        if spine_c is None:
            summary["edges"][edge] = {"skipped": "no_corridor"}
            continue
        # Chain PAD-TO-PAD, one spec per link: pad_k -> its jog -> along the
        # corridor -> previous pad's jog -> previous pad CENTRE. Each link
        # lives on ONE copper layer but consecutive links may differ -- the
        # PTH barrels bridge them. That weave is what makes the corridor
        # usable at all: KiCad Routing Tools typically runs a power daisy-chain
        # through the same corridor, alternating layers stretch by stretch
        # (this board's 5V rail), so each GND link simply takes whichever
        # layer its own stretch leaves free. A link that fits neither layer
        # is skipped; the next link ties back across the gap to the last
        # landed pad and the guards get the longer rail to judge.
        stubs = 0
        skipped: list[str] = []
        last: tuple[str, str, float, float] | None = None
        for ref, num, px, py, _r in pads:
            if last is None:
                last = (ref, num, px, py)  # chain origin: nothing to tie yet
                continue
            _lref, _lnum, lx, ly = last
            jog = (spine_c, py) if vertical else (px, spine_c)
            last_jog = (spine_c, ly) if vertical else (lx, spine_c)
            landed = False
            reasons: list[str] = []
            for layer in ("F.Cu", "B.Cu"):
                res = add_breakout_stubs(
                    pcb_path,
                    [
                        BreakoutSpec(
                            ref=ref,
                            pad=num,
                            waypoints=[jog, last_jog, (lx, ly)],
                            layer=layer,
                            width_mm=width,
                            near_xy=(px, py),
                        )
                    ],
                    cfg=cfg,
                )
                if res.get("stubs", 0):
                    stubs += 1
                    landed = True
                    break
                reasons = res.get("skipped") or []
            # Advance the anchor even on failure: a link that crosses the
            # power rail's F<->B transition is blocked on BOTH layers, and
            # keeping the anchor there would re-attempt that same blocked
            # stretch for every later pad. Skipping forward splits the spine
            # into sub-chains around the transition; each still merges its
            # own islands and the via/track mop-up joins the few seams.
            last = (ref, num, px, py)
            if not landed:
                skipped.extend(reasons or [f"{ref}.{num}"])
        summary["stubs"] += stubs
        summary["edges"][edge] = {
            "pads": len(pads),
            "stubs": stubs,
            "skipped": skipped,
        }

    if summary["stubs"]:
        board = pcbnew.LoadBoard(pcb_path)
        pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        board.Save(pcb_path)
        del board
    return summary


def repair_parent_gnd_islands(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convergence-loop GND strand/plane repair for the parent board.

    The single-pass ``repair_stranded_gnd`` ties stranded GND fill islands
    back with same-net tracks, but the parent re-pour after power-plane
    addition (``pour_power_planes`` refills ALL zones, including GND, at
    higher priority) can re-fragment the GND plane -- leaving islands that
    a second repair pass would tie. Furthermore, the single-pass track-based
    repair cannot connect a B.Cu-only island that has no overlapping node
    from the main cluster on the same layer: a via between the B.Cu island
    and an overlapping F.Cu island from another cluster merges both into
    one.

    Strategy (loop until convergence, cap at ``max_iter``):
    1. Collect GND clusters via ``_collect_net_clusters``.
    2. If ≤1 cluster, done.
    3. **Via stitching** -- for each non-main cluster, find overlapping fill
       islands with the main cluster (or another non-main cluster) on the
       opposite layer; stamp a via at the overlap to merge them.
    4. **Track stitching** -- run the existing ``repair_stranded_net``.
    5. Refill all zones and re-check. If any islands remain, go to 1.

    Returns ``{net, clusters, stranded, tied_pads, vias, unresolved}``.
    Logs a warning if the loop exhausts without full convergence.
    """
    cfg = cfg or {}
    gnd_name = cfg.get("gnd_zone_net", "GND")
    max_iter = int(cfg.get("gnd_parent_repair_max_iter", 5))
    summary: dict[str, Any] = {
        "net": gnd_name,
        "iterations": 0,
        "clusters": 0,
        "stranded": 0,
        "tied_pads": 0,
        "vias": 0,
        "unresolved": 0,
        "converged": False,
    }
    if not gnd_name:
        return summary

    # Edge-spine pre-pass: connectors banked along one edge strand exactly
    # there (each header's GND pad on its own pour strip between the
    # neighbours' signal fans -- KC-YXQ4EC: 13 islands along a 16-header
    # edge), and neither via-stitch nor straight ties can cross the fans.
    # The spine merges the whole bank through the outboard corridor first;
    # the loop below then only has the leftovers to close. Fragmentation-
    # gated so an already-whole plane never grows copper.
    if cfg.get("gnd_edge_spine_enabled", True) and cfg.get("component_zones"):
        pre_board = pcbnew.LoadBoard(pcb_path)
        pre_clusters, _ = _collect_net_clusters(pre_board, gnd_name)
        del pre_board
        if len(pre_clusters) > 1:
            summary["edge_spine"] = stamp_gnd_edge_spine(pcb_path, cfg)

    # Reuse the netclass-aware copper collision guard from breakout stamping.
    from kicraft.autoplacer.brain.breakout_stubs import _own_clearance_mm

    F, B = pcbnew.F_Cu, pcbnew.B_Cu
    # --- Collect obstacles and holes once (they don't change between iters) ---
    board0 = pcbnew.LoadBoard(pcb_path)
    via_keepouts = collect_track_via_rule_areas(board0)
    gnd_net = board0.GetNetInfo().GetNetItem(gnd_name)
    if not gnd_net or gnd_net.GetNetCode() == 0:
        summary["error"] = f"net {gnd_name!r} not found"
        return summary
    gnd_code = gnd_net.GetNetCode()
    floor_mm = fab_floors(cfg)["clearance_mm"]
    via_drill = pcbnew.FromMM(float(cfg.get("via_drill_mm", 0.3)))
    via_size = pcbnew.FromMM(float(cfg.get("via_size_mm", 0.6)))
    via_r_mm = pcbnew.ToMM(via_size) / 2.0
    via_radius_iu = int(via_size) // 2
    via_drill_r_mm = pcbnew.ToMM(via_drill) / 2.0
    via_edge_geometry = _board_edge_geometry(board0)

    # Copper obstacles (non-GND tracks/pads) for via collision check.
    copper_obstacles: list[tuple[Any, int]] = []
    for t in board0.GetTracks():
        if t.GetNetCode() == gnd_code:
            continue
        t_layer = pcbnew.B_Cu if t.GetClass() == "PCB_VIA" else t.GetLayer()
        item_cl = _own_clearance_mm(t, t_layer, floor_mm)
        copper_obstacles.append((t, int(pcbnew.FromMM(via_r_mm + max(floor_mm, item_cl)))))
    for ofp in board0.GetFootprints():
        for op in ofp.Pads():
            if op.GetNetCode() == gnd_code:
                continue
            item_cl = _own_clearance_mm(op, pcbnew.B_Cu, floor_mm)
            copper_obstacles.append((op, int(pcbnew.FromMM(via_r_mm + max(floor_mm, item_cl)))))

    # Drilled holes.
    h2h = pcbnew.ToMM(board0.GetDesignSettings().m_HoleToHoleMin)
    hole_min_mm = h2h if h2h > 0 else float(cfg.get("hole_to_hole_min_mm", 0.25))
    holes: list[tuple[float, float, float]] = []
    for t in board0.GetTracks():
        if t.GetClass() == "PCB_VIA":
            p = t.GetPosition()
            holes.append((pcbnew.ToMM(p.x), pcbnew.ToMM(p.y), pcbnew.ToMM(t.GetDrillValue()) / 2.0))
    for ofp in board0.GetFootprints():
        for op in ofp.Pads():
            ds = op.GetDrillSize()
            if ds.x > 0 or ds.y > 0:
                p = op.GetPosition()
                holes.append(
                    (
                        pcbnew.ToMM(p.x),
                        pcbnew.ToMM(p.y),
                        max(pcbnew.ToMM(ds.x), pcbnew.ToMM(ds.y)) / 2.0,
                    )
                )
    del board0

    def _via_blocked(active_board, x_iu: int, y_iu: int) -> bool:
        if not _via_clears_board_edge(
            x_iu, y_iu, edge_geometry=via_edge_geometry, via_radius_iu=via_radius_iu
        ):
            return True
        pt = pcbnew.VECTOR2I(x_iu, y_iu)
        xy = (pcbnew.ToMM(x_iu), pcbnew.ToMM(y_iu))
        if any(via_intersects_rule_area(xy, via_r_mm, area) for area in via_keepouts):
            return True
        if _via_hits_smd_pad(active_board, pt, via_radius_iu):
            return True
        if any(item.HitTest(pt, margin) for item, margin in copper_obstacles):
            return True
        xm, ym = pcbnew.ToMM(x_iu), pcbnew.ToMM(y_iu)
        return any(
            ((xm - hx) ** 2 + (ym - hy) ** 2) ** 0.5 < hr + via_drill_r_mm + hole_min_mm
            for hx, hy, hr in holes
        )

    def _add_via(active_board, x_iu: int, y_iu: int) -> bool:
        if _via_blocked(active_board, x_iu, y_iu):
            return False
        via = pcbnew.PCB_VIA(active_board)
        via.SetPosition(pcbnew.VECTOR2I(x_iu, y_iu))
        via.SetDrill(via_drill)
        try:
            via.SetWidth(via_size)
        except TypeError:
            via.SetWidth(pcbnew.F_Cu, via_size)
        via.SetNetCode(gnd_code)
        active_board.Add(via)
        holes.append((pcbnew.ToMM(x_iu), pcbnew.ToMM(y_iu), via_drill_r_mm))
        return True

    for iteration in range(1, max_iter + 1):
        board = pcbnew.LoadBoard(pcb_path)
        clusters, all_nodes = _collect_net_clusters(board, gnd_name)
        n_clusters = len(clusters)
        summary["iterations"] = iteration
        summary["clusters"] = n_clusters
        if iteration == 1:
            summary["stranded"] = max(0, n_clusters - 1)

        if n_clusters <= 1:
            summary["converged"] = True
            summary["unresolved"] = 0
            del board
            break

        # --- Via stitching: connect overlap islands on opposite layers ---
        main_root = max(clusters, key=lambda r: len(clusters[r]))
        main_cluster = set(clusters[main_root])
        main_islands = [all_nodes[i] for i in main_cluster if all_nodes[i]["kind"] == "island"]

        for root, members in clusters.items():
            if root == main_root:
                continue
            # Find islands in this stranded cluster
            stranded_islands = [all_nodes[i] for i in members if all_nodes[i]["kind"] == "island"]
            for si in stranded_islands:
                si_layer = next(iter(si["layers"]))
                si_pts = si["fill"].Outline(si["idx"])
                # Check against main-cluster islands on the OPPOSITE layer
                for mi in main_islands:
                    mi_layer = next(iter(mi["layers"]))
                    if mi_layer == si_layer:
                        continue  # same-layer overlap irrelevant for vias
                    # Test whether any point from si's outline falls inside mi's fill
                    for k in range(si_pts.PointCount()):
                        p = si_pts.CPoint(k)
                        px_iu, py_iu = p.x, p.y
                        if mi["fill"].Contains(pcbnew.VECTOR2I(px_iu, py_iu), mi["idx"]):
                            if _add_via(board, px_iu, py_iu):
                                summary["vias"] += 1
                            break
                    else:
                        # Also test the island centre
                        bb = si_pts.BBox()
                        cx, cy = bb.GetCenter().x, bb.GetCenter().y
                        if mi["fill"].Contains(pcbnew.VECTOR2I(cx, cy), mi["idx"]):
                            if _add_via(board, cx, cy):
                                summary["vias"] += 1

        # Save vias, then run the track-based repair
        board.Save(pcb_path)
        del board

        if summary["vias"] > 0:
            # Refill so the new vias merge clusters before track repair
            board = pcbnew.LoadBoard(pcb_path)
            pcbnew.ZONE_FILLER(board).Fill(board.Zones())
            board.Save(pcb_path)
            del board

        # Run track-based repair
        track_res = repair_stranded_net(pcb_path, gnd_name, cfg)
        summary["tied_pads"] += track_res.get("tied", 0)

        # Check convergence after track repair + refill
        board = pcbnew.LoadBoard(pcb_path)
        clusters, _ = _collect_net_clusters(board, gnd_name)
        n_remaining = len(clusters)
        summary["clusters"] = n_remaining
        if n_remaining <= 1:
            summary["converged"] = True
            summary["unresolved"] = 0
            del board
            break
        del board

    # --- Final refill ---
    board = pcbnew.LoadBoard(pcb_path)
    pcbnew.ZONE_FILLER(board).Fill(board.Zones())
    board.Save(pcb_path)
    del board

    if not summary.get("converged"):
        print(
            f"  WARNING: parent GND island repair did not converge "
            f"after {max_iter} iterations; "
            f"{summary.get('clusters', '?')} clusters remain."
        )

    return summary
