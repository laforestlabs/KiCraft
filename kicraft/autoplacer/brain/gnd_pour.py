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

import math
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
        zone.SetThermalReliefGap(
            pcbnew.FromMM(float(cfg.get("zone_thermal_gap_mm", 0.5)))
        )
        zone.SetThermalReliefSpokeWidth(
            pcbnew.FromMM(float(cfg.get("zone_thermal_spoke_mm", 0.5)))
        )
    else:
        zone.SetPadConnection(pcbnew.ZONE_CONNECTION_FULL)


def _collect_keepout_zones(board: Any) -> list[Any]:
    """Rule-area keep-out zones that forbid tracks or vias -- both board-level
    and the ones embedded inside footprints (the ESP32-S3-MINI/WROOM antenna
    near-field ``antenna_keepout`` ships inside the module's .kicad_mod).

    FreeRouting's DSN export already steers *signal* traces clear of these, but
    the post-route GND finisher (in-pad/array thermal vias + small-pad escape
    stubs) had no such guard, so it stamped GND vias and 1 mm stubs straight into
    U1's antenna keep-out (the KC-S8PC37 signature: 30 items_not_allowed).
    """
    zones: list[Any] = []
    for z in board.Zones():
        if z.GetIsRuleArea() and (z.GetDoNotAllowVias() or z.GetDoNotAllowTracks()):
            zones.append(z)
    for fp in board.GetFootprints():
        for z in fp.Zones():
            if z.GetIsRuleArea() and (
                z.GetDoNotAllowVias() or z.GetDoNotAllowTracks()
            ):
                zones.append(z)
    return zones


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
    layer_map = {"B.Cu": pcbnew.B_Cu, "F.Cu": pcbnew.F_Cu}
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
                zone.SetLocalClearance(
                    pcbnew.FromMM(float(cfg.get("zone_clearance_mm", 0.3)))
                )
                zone.SetMinThickness(
                    pcbnew.FromMM(float(cfg.get("zone_min_thickness_mm", 0.25)))
                )
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
            nodes.append({
                "kind": "pad", "ref": fp.GetReferenceAsString(), "num": p.GetNumber(),
                "layers": {F, B} if is_pth else {F if p.IsOnLayer(F) else B},
                "pts": _pts_around(x, y, r), "xy": (x, y), "r": r,
            })
    for t in board.GetTracks():
        if t.GetNetCode() != net_code:
            continue
        if t.GetClass() == "PCB_VIA":
            pos = t.GetPosition()
            x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            r = pcbnew.ToMM(t.GetWidth()) / 2.0
            nodes.append({"kind": "via", "layers": {F, B},
                          "pts": _pts_around(x, y, r), "xy": (x, y), "r": r})
        else:
            a, b2 = t.GetStart(), t.GetEnd()
            nodes.append({"kind": "trk", "layers": {t.GetLayer()},
                          "pts": [(pcbnew.ToMM(a.x), pcbnew.ToMM(a.y)),
                                  (pcbnew.ToMM(b2.x), pcbnew.ToMM(b2.y))],
                          "xy": (pcbnew.ToMM((a.x + b2.x) / 2), pcbnew.ToMM((a.y + b2.y) / 2))})
    islands: list[dict] = []
    for z in board.Zones():
        if z.GetNetname() != net_name or z.GetIsRuleArea():
            continue
        layer = z.GetLayer()
        fill = z.GetFilledPolysList(layer)
        for i in range(fill.OutlineCount()):
            bb = fill.Outline(i).BBox()
            islands.append({"kind": "island", "layers": {layer}, "fill": fill,
                            "idx": i,
                            "xy": (pcbnew.ToMM(bb.Centre().x), pcbnew.ToMM(bb.Centre().y))})

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
            for (px, py) in n["pts"]:
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
            for (ax, ay) in nodes[i]["pts"]:
                for (bx, by) in nodes[j]["pts"]:
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
    FreeRouting; a fine-pitch part's supply pad may be unreachable for it --
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
    summary: dict[str, Any] = {"net": net_name, "clusters": 0, "stranded": 0,
                               "tied": 0, "skipped": [], "unresolved": 0}
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
        all_nodes[i] for i in clusters[main_root]
        if all_nodes[i]["kind"] in ("pad", "via")
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
            ((n["xy"][0] - sx) ** 2 + (n["xy"][1] - sy) ** 2) ** 0.5 <= n["r"]
            for n in via_nodes
        )
        start_layers = [
            (lname, lid) for lname, lid in (("F.Cu", F), ("B.Cu", B))
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
                    [BreakoutSpec(ref=src["ref"], pad=src["num"],
                                  waypoints=[tgt["xy"]], layer=layer_name,
                                  near_xy=src["xy"])],
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
        print(f"  WARNING: {net_name} strand repair left {summary['unresolved']} "
              f"cluster(s) disconnected (tied={summary['tied']}, "
              f"skipped={summary['skipped']})")
    return summary


def repair_stranded_gnd(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """GND strand repair: :func:`repair_stranded_net` on the GND pour net."""
    cfg = cfg or {}
    gnd_name = cfg.get("gnd_zone_net", "GND")
    if not cfg.get("gnd_strand_repair_enabled", True):
        return {"net": gnd_name, "clusters": 0, "stranded": 0, "tied": 0,
                "skipped": []}
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
    out: dict[str, Any] = {"nets": [], "stranded": 0, "tied": 0, "skipped": [],
                           "unresolved": 0}
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
    runs LAST, after the signal breakout stubs and FreeRouting have consumed
    every exit around a dense pad row -- so exactly the pads that most need a
    plane bond find no legal path (KC-UXASHQ U1.6: hemmed in by a signal stub
    and routed tracks, left unconnected). GND is never routed, so its escapes
    can claim space FIRST: stamp these before the signal stubs and the
    router, which both route around locked copper. Each spec carries
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

    via_size_mm = float(cfg.get("via_size_mm", 0.6))
    area_threshold = float(cfg.get("thermal_pad_area_mm2", 4.0))
    length = float(cfg.get("gnd_escape_length_mm", 1.0))
    specs: list = []
    for fp in board.GetFootprints():
        pads = list(fp.Pads())
        if len(pads) < 3:  # escapes stay multipad-only, matching post-route
            continue
        for pad in pads:
            if pad.GetAttribute() not in (pcbnew.PAD_ATTRIB_SMD,
                                          pcbnew.PAD_ATTRIB_CONN):
                continue
            if pad.GetNetCode() != gnd_code:
                continue
            size = pad.GetSize()
            w, h = pcbnew.ToMM(size.x), pcbnew.ToMM(size.y)
            if min(w, h) >= via_size_mm or w * h >= area_threshold:
                continue  # the post-route in-pad via handles it
            specs.append(BreakoutSpec(ref=fp.GetReferenceAsString(),
                                      pad=pad.GetNumber(),
                                      length_mm=length, via_at_end=True))
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
    keepout_zones = _collect_keepout_zones(board)

    def _in_keepout(pt: Any, clearance_iu: int = 0) -> bool:
        """True when ``pt`` lies in (or within ``clearance_iu`` of) a track/via
        keep-out, so a via of that radius -- or an escape stub reaching that far
        -- would intrude on the protected area."""
        for z in keepout_zones:
            bb = z.GetBoundingBox()
            if clearance_iu:
                bb.Inflate(int(clearance_iu))
            if bb.Contains(pt):
                return True
        return False

    via_drill = pcbnew.FromMM(float(cfg.get("via_drill_mm", 0.3)))
    via_size = pcbnew.FromMM(float(cfg.get("via_size_mm", 0.6)))
    pitch = pcbnew.FromMM(float(cfg.get("thermal_via_pitch_mm", 1.2)))
    inset = pcbnew.FromMM(float(cfg.get("thermal_via_inset_mm", 0.5)))
    area_threshold = float(cfg.get("thermal_pad_area_mm2", 4.0))
    floor_mm = float(cfg.get("freerouting_min_clearance_mm", 0.153))
    summary["thermal_vias_blocked"] = 0
    summary["escape_stitched"] = 0

    from kicraft.autoplacer.brain.breakout_stubs import _own_clearance_mm

    # GND-side clearance: KiCad resolves a pair as the LARGER of the two
    # items' netclass clearances, and GND rides the Power class (0.30 mm) on
    # generated boards -- a via held only to the 0.153 freerouting floor can
    # pass this guard yet land 0.26 mm from a Default-class track, a hard
    # Power-netclass DRC error (the KC-UXASHQ escape-via signature).
    gnd_cl_mm = floor_mm
    for _fp in board.GetFootprints():
        gp = next((p for p in _fp.Pads() if p.GetNetCode() == gnd_code), None)
        if gp is not None:
            gnd_cl_mm = max(gnd_cl_mm, _own_clearance_mm(gp, pcbnew.B_Cu, floor_mm))
            break

    via_r_mm = pcbnew.ToMM(via_size) / 2.0
    via_drill_r_mm = pcbnew.ToMM(via_drill) / 2.0
    copper_obstacles: list[tuple[Any, int]] = []  # (item, HitTest margin)
    for t in board.GetTracks():
        if t.GetNetCode() == gnd_code:
            continue
        t_layer = pcbnew.B_Cu if t.GetClass() == "PCB_VIA" else t.GetLayer()
        item_cl = _own_clearance_mm(t, t_layer, floor_mm)
        copper_obstacles.append(
            (t, int(pcbnew.FromMM(via_r_mm + max(gnd_cl_mm, item_cl))))
        )
    for ofp in board.GetFootprints():
        for op in ofp.Pads():
            if op.GetNetCode() == gnd_code:
                continue
            item_cl = _own_clearance_mm(op, pcbnew.B_Cu, floor_mm)
            copper_obstacles.append(
                (op, int(pcbnew.FromMM(via_r_mm + max(gnd_cl_mm, item_cl))))
            )

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
            holes.append((pcbnew.ToMM(p.x), pcbnew.ToMM(p.y),
                          pcbnew.ToMM(t.GetDrillValue()) / 2.0))
    for ofp in board.GetFootprints():
        for op in ofp.Pads():
            ds = op.GetDrillSize()
            if ds.x > 0 or ds.y > 0:
                p = op.GetPosition()
                holes.append((pcbnew.ToMM(p.x), pcbnew.ToMM(p.y),
                              max(pcbnew.ToMM(ds.x), pcbnew.ToMM(ds.y)) / 2.0))

    def _via_blocked(x: int, y: int) -> bool:
        """True when a GND via at (x, y) would violate another net's copper
        (netclass pair clearance) or any drilled hole (hole-to-hole minimum).

        This runs on a ROUTED board (leaf) or a composed parent full of leaf
        traces; a via stamped blind through a B.Cu track of another net is a
        hard short the router can never repair (the IP2368-bank incident:
        seven shorts from exactly this). Same-net copper is a valid landing.
        """
        pt = pcbnew.VECTOR2I(int(x), int(y))
        # A via whose barrel overlaps a track/via keep-out (the antenna
        # near-field) is an items_not_allowed DRC error the router can't repair.
        if _in_keepout(pt, int(pcbnew.FromMM(via_r_mm))):
            return True
        if any(item.HitTest(pt, margin) for item, margin in copper_obstacles):
            return True
        xm, ym = pcbnew.ToMM(int(x)), pcbnew.ToMM(int(y))
        return any(
            ((xm - hx) ** 2 + (ym - hy) ** 2) ** 0.5
            < hr + via_drill_r_mm + hole_min_mm
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
    escape_pads: list[tuple[str, str]] = []

    # GND vias already on the board (e.g. a pre-route escape stub's tip via
    # stamped by gnd_escape_specs): a pad one of those already bonds to the
    # plane must not be escaped AGAIN post-route.
    escape_len_mm = float(cfg.get("gnd_escape_length_mm", 1.0))
    prebonded_reach = escape_len_mm + pcbnew.ToMM(via_size)
    gnd_via_pts = [
        (pcbnew.ToMM(t.GetPosition().x), pcbnew.ToMM(t.GetPosition().y))
        for t in board.GetTracks()
        if t.GetClass() == "PCB_VIA" and t.GetNetCode() == gnd_code
    ]

    def _already_bonded(pad) -> bool:
        p = pad.GetPosition()
        px, py = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
        return any(((px - vx) ** 2 + (py - vy) ** 2) ** 0.5 <= prebonded_reach
                   for vx, vy in gnd_via_pts)

    # --- 1. Thermal-via arrays under GND thermal / exposed pads ---
    for fp in board.GetFootprints():
        pads = list(fp.Pads())
        smd = [
            p for p in pads
            if p.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN)
        ]
        if not smd:
            continue
        # Escape stubs (extra copper near the pad row) stay reserved for
        # multi-pin parts: >= 3 includes the SOT-23-class regulators whose
        # lone GND pad otherwise floats as an F.Cu pour island (run_03 U1.5 /
        # run_05 U2.2). In-pad vias below have no such gate: a 2-pad
        # decoupling cap's GND pad strands just as hard when no F.Cu GND
        # copper reaches it and the B.Cu pour is its only path (KC-UXASHQ
        # C2.2) -- _via_blocked alone decides whether the via is safe.
        multipad = len(pads) >= 3
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
            pos = pad.GetPosition()
            # Net any stray (number-shared) EP sub-pad to GND so it joins the plane.
            if pad.GetNetCode() != gnd_code:
                pad.SetNetCode(gnd_code)
            # Stitch every GND pad on a multi-pin footprint: this ties BOTH the
            # perimeter GND pins (so the F.Cu GND network drops to the plane)
            # AND the interior EP down to the B.Cu pour, unifying GND into one
            # net. A pad too small for an in-pad via is escaped with a short
            # guarded stub + end via instead (stamped after this pass).
            if not (fits_via or large):
                # An escape stub reaches gnd_escape_length + an end via out of
                # the pad: if that lands in a keep-out, skip it (the pad still
                # bonds through its footprint's other GND pins / the pour).
                escape_reach = pcbnew.FromMM(
                    float(cfg.get("gnd_escape_length_mm", 1.0))
                    + pcbnew.ToMM(via_size)
                )
                if (
                    multipad
                    and not _already_bonded(pad)
                    and not _in_keepout(pos, int(escape_reach))
                ):
                    escape_pads.append((fp.GetReferenceAsString(), pad.GetNumber()))
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
                    length_mm=float(cfg.get("gnd_escape_length_mm", 1.0)),
                    via_at_end=True,
                )
                for ref, num in escape_pads
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

    # Reuse the via collision guard from add_gnd_pour_and_thermal_vias.
    from kicraft.autoplacer.brain.breakout_stubs import _own_clearance_mm

    F, B = pcbnew.F_Cu, pcbnew.B_Cu
    # --- Collect obstacles and holes once (they don't change between iters) ---
    board0 = pcbnew.LoadBoard(pcb_path)
    gnd_net = board0.GetNetInfo().GetNetItem(gnd_name)
    if not gnd_net or gnd_net.GetNetCode() == 0:
        summary["error"] = f"net {gnd_name!r} not found"
        return summary
    gnd_code = gnd_net.GetNetCode()
    floor_mm = float(cfg.get("freerouting_min_clearance_mm", 0.153))
    via_drill = pcbnew.FromMM(float(cfg.get("via_drill_mm", 0.3)))
    via_size = pcbnew.FromMM(float(cfg.get("via_size_mm", 0.6)))
    via_r_mm = pcbnew.ToMM(via_size) / 2.0
    via_drill_r_mm = pcbnew.ToMM(via_drill) / 2.0

    # Copper obstacles (non-GND tracks/pads) for via collision check.
    copper_obstacles: list[tuple[Any, int]] = []
    for t in board0.GetTracks():
        if t.GetNetCode() == gnd_code:
            continue
        t_layer = pcbnew.B_Cu if t.GetClass() == "PCB_VIA" else t.GetLayer()
        item_cl = _own_clearance_mm(t, t_layer, floor_mm)
        copper_obstacles.append(
            (t, int(pcbnew.FromMM(via_r_mm + max(floor_mm, item_cl))))
        )
    for ofp in board0.GetFootprints():
        for op in ofp.Pads():
            if op.GetNetCode() == gnd_code:
                continue
            item_cl = _own_clearance_mm(op, pcbnew.B_Cu, floor_mm)
            copper_obstacles.append(
                (op, int(pcbnew.FromMM(via_r_mm + max(floor_mm, item_cl))))
            )

    # Drilled holes.
    h2h = pcbnew.ToMM(board0.GetDesignSettings().m_HoleToHoleMin)
    hole_min_mm = h2h if h2h > 0 else float(cfg.get("hole_to_hole_min_mm", 0.25))
    holes: list[tuple[float, float, float]] = []
    for t in board0.GetTracks():
        if t.GetClass() == "PCB_VIA":
            p = t.GetPosition()
            holes.append((pcbnew.ToMM(p.x), pcbnew.ToMM(p.y),
                          pcbnew.ToMM(t.GetDrillValue()) / 2.0))
    for ofp in board0.GetFootprints():
        for op in ofp.Pads():
            ds = op.GetDrillSize()
            if ds.x > 0 or ds.y > 0:
                p = op.GetPosition()
                holes.append((pcbnew.ToMM(p.x), pcbnew.ToMM(p.y),
                              max(pcbnew.ToMM(ds.x), pcbnew.ToMM(ds.y)) / 2.0))
    del board0

    def _via_blocked(x_iu: int, y_iu: int) -> bool:
        pt = pcbnew.VECTOR2I(x_iu, y_iu)
        if any(item.HitTest(pt, margin) for item, margin in copper_obstacles):
            return True
        xm, ym = pcbnew.ToMM(x_iu), pcbnew.ToMM(y_iu)
        return any(
            ((xm - hx) ** 2 + (ym - hy) ** 2) ** 0.5
            < hr + via_drill_r_mm + hole_min_mm
            for hx, hy, hr in holes
        )

    def _add_via(x_iu: int, y_iu: int) -> bool:
        if _via_blocked(x_iu, y_iu):
            return False
        via = pcbnew.PCB_VIA(board)
        via.SetPosition(pcbnew.VECTOR2I(x_iu, y_iu))
        via.SetDrill(via_drill)
        try:
            via.SetWidth(via_size)
        except TypeError:
            via.SetWidth(pcbnew.F_Cu, via_size)
        via.SetNetCode(gnd_code)
        board.Add(via)
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
        main_islands = [
            all_nodes[i] for i in main_cluster
            if all_nodes[i]["kind"] == "island"
        ]

        for root, members in clusters.items():
            if root == main_root:
                continue
            # Find islands in this stranded cluster
            stranded_islands = [
                all_nodes[i] for i in members
                if all_nodes[i]["kind"] == "island"
            ]
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
                        if mi["fill"].Contains(
                            pcbnew.VECTOR2I(px_iu, py_iu), mi["idx"]
                        ):
                            if _add_via(px_iu, py_iu):
                                summary["vias"] += 1
                            break
                    else:
                        # Also test the island centre
                        bb = si_pts.BBox()
                        cx, cy = bb.GetCenter().x, bb.GetCenter().y
                        if mi["fill"].Contains(
                            pcbnew.VECTOR2I(cx, cy), mi["idx"]
                        ):
                            if _add_via(cx, cy):
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
        print(f"  WARNING: parent GND island repair did not converge "
              f"after {max_iter} iterations; "
              f"{summary.get('clusters', '?')} clusters remain.")

    return summary
