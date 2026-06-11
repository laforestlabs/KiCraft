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


def repair_stranded_gnd(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Tie GND clusters stranded from the main plane back with guarded tracks.

    GND is never routed by FreeRouting -- the plane is supposed to reach every
    GND pad. In a crowded region the B.Cu plane fragments around signal
    tracks, and a THT connector GND pin (run_03 J7.2: a 2-pin LED-channel
    header) can end up on a tiny fill island with no path to the main plane:
    no via to drop through, no same-net mate for a shield tie, no GND track.
    This post-pour pass finds every GND cluster isolated from the main one
    (geometric union-find over pads/vias/tracks/fill islands) and stamps a
    direct same-net track from a stranded pad to the nearest main-cluster
    pad/via via :func:`add_breakout_stubs` -- inheriting its foreign-pad,
    existing-copper, netclass and outline guards -- then refills the zones so
    the pour closes around the new tie. A tie whose straight path is blocked
    is skipped (the board is no worse than before).
    """
    cfg = cfg or {}
    summary: dict[str, Any] = {"clusters": 0, "stranded": 0, "tied": 0, "skipped": []}
    if not cfg.get("gnd_strand_repair_enabled", True):
        return summary
    gnd_name = cfg.get("gnd_zone_net", "GND")
    if not gnd_name:
        return summary

    board = pcbnew.LoadBoard(pcb_path)
    gnd_net = board.GetNetInfo().GetNetItem(gnd_name)
    if not gnd_net or gnd_net.GetNetCode() == 0:
        return summary
    gnd_code = gnd_net.GetNetCode()
    max_tie_mm = float(cfg.get("gnd_strand_repair_max_mm", 30.0))

    # --- collect GND nodes -------------------------------------------------
    # Each node: (kind, payload, layers, probe_points_mm). A PTH pad or via
    # spans both layers; an SMD pad only its own; a fill island only its zone's.
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
            if p.GetNetCode() != gnd_code:
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
                "pts": _pts_around(x, y, r), "xy": (x, y),
            })
    for t in board.GetTracks():
        if t.GetNetCode() != gnd_code:
            continue
        if t.GetClass() == "PCB_VIA":
            pos = t.GetPosition()
            x, y = pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)
            nodes.append({"kind": "via", "layers": {F, B},
                          "pts": _pts_around(x, y, pcbnew.ToMM(t.GetWidth()) / 2.0),
                          "xy": (x, y)})
        else:
            a, b2 = t.GetStart(), t.GetEnd()
            nodes.append({"kind": "trk", "layers": {t.GetLayer()},
                          "pts": [(pcbnew.ToMM(a.x), pcbnew.ToMM(a.y)),
                                  (pcbnew.ToMM(b2.x), pcbnew.ToMM(b2.y))],
                          "xy": (pcbnew.ToMM((a.x + b2.x) / 2), pcbnew.ToMM((a.y + b2.y) / 2))})
    islands: list[dict] = []
    for z in board.Zones():
        if z.GetNetname() != gnd_name or z.GetIsRuleArea():
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

    specs = []
    for root, members in clusters.items():
        if root == main_root:
            continue
        summary["stranded"] += 1
        src = next((all_nodes[i] for i in members if all_nodes[i]["kind"] == "pad"), None)
        if src is None:
            summary["skipped"].append("cluster_without_pad")
            continue
        sx, sy = src["xy"]
        tgt = min(main_targets,
                  key=lambda t: (t["xy"][0] - sx) ** 2 + (t["xy"][1] - sy) ** 2)
        d = ((tgt["xy"][0] - sx) ** 2 + (tgt["xy"][1] - sy) ** 2) ** 0.5
        if d > max_tie_mm:
            summary["skipped"].append(f"{src['ref']}.{src['num']}:too_far:{d:.1f}mm")
            continue
        specs.append((src["ref"], src["num"], tgt["xy"]))

    if not specs:
        return summary
    from kicraft.autoplacer.brain.breakout_stubs import (
        BreakoutSpec,
        add_breakout_stubs,
    )

    # Try F.Cu first, retry the leftovers on B.Cu (the strand is usually in a
    # region where one layer is crowded and the other open).
    remaining = specs
    for layer_name in ("F.Cu", "B.Cu"):
        if not remaining:
            break
        batch = [BreakoutSpec(ref=r, pad=n, waypoints=[xy], layer=layer_name)
                 for r, n, xy in remaining]
        res = add_breakout_stubs(pcb_path, batch, cfg=cfg)
        summary["tied"] += res.get("stubs", 0)
        failed_keys = {s.split(":")[0] for s in res.get("skipped", [])}
        remaining = [(r, n, xy) for r, n, xy in remaining
                     if f"{r}.{n}" in failed_keys]
    summary["skipped"].extend(f"{r}.{n}:no_clear_path" for r, n, _ in remaining)

    if summary["tied"]:
        board = pcbnew.LoadBoard(pcb_path)
        pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        board.Save(pcb_path)
    return summary


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
    via_clearance = pcbnew.FromMM(
        float(cfg.get("freerouting_min_clearance_mm", 0.153))
    )
    summary["thermal_vias_blocked"] = 0
    summary["escape_stitched"] = 0

    def _via_blocked(x: int, y: int) -> bool:
        """True when a GND via at (x, y) would land on another net's copper.

        This runs on a ROUTED board (leaf) or a composed parent full of leaf
        traces; a via stamped blind through a B.Cu track of another net is a
        hard short the router can never repair (the IP2368-bank incident:
        seven shorts from exactly this). Same-net copper is a valid landing.
        """
        pt = pcbnew.VECTOR2I(int(x), int(y))
        margin = int(via_size // 2 + via_clearance)
        for t in board.GetTracks():
            if t.GetNetCode() == gnd_code:
                continue
            if t.HitTest(pt, margin):
                return True
        for ofp in board.GetFootprints():
            for op in ofp.Pads():
                if op.GetNetCode() == gnd_code:
                    continue
                if op.HitTest(pt, margin):
                    return True
        return False

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
        return True

    # GND pads that need stitching but cannot host an in-pad via: escape them
    # with a short guarded stub + end via instead (see below).
    escape_pads: list[tuple[str, str]] = []

    # --- 1. Thermal-via arrays under GND thermal / exposed pads ---
    for fp in board.GetFootprints():
        pads = list(fp.Pads())
        smd = [
            p for p in pads
            if p.GetAttribute() in (pcbnew.PAD_ATTRIB_SMD, pcbnew.PAD_ATTRIB_CONN)
        ]
        if not smd:
            continue
        # ICs/modules need their GND dropped to the plane; 2-pad passives reach
        # GND through whatever they connect to. >= 3 includes the SOT-23-class
        # regulators whose lone GND pad otherwise floats as an F.Cu pour island
        # (run_03 U1.5 / run_05 U2.2 -- the post-connector-fix rc7 signature).
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
            if not multipad and not large:
                continue
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
