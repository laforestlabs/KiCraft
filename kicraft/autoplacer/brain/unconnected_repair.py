"""Constrained bend/via repair for walled-off unconnected SIGNAL nets (C1).

After routing, a few signal nets are sometimes left unrouted because the
router walled itself off (no_clear_path — memory
`kicraft-unconnected-1-cluster-walled-off-signal-power`). This pass closes
each unconnected DRC edge of such a net with guarded copper, at the same
abstraction level as the accepted pour repairs
(`gnd_pour.repair_stranded_net`), but with the richer path family the
walled-off family needs: radial escapes, L-bends and offset doglegs first,
then a native clearance-mesh candidate that may change copper faces more
than once. All candidates use :func:`breakout_stubs.add_breakout_stubs`,
which inherits every clearance / foreign-pad / outline / hole guard.
Multi-leg candidates are byte-restored unless every leg stamps.

Detection is DRC-grounded: the edges come from kicad-cli's own
``unconnected_items`` report — the exact list the fab gate fails on — not a
re-derived connectivity model (a geometric union-find over-detects on long
pads and "repairs" nets that were never broken).

The module only stamps and reports. The accept-or-revert verdict (full
re-DRC: unconnected must drop, shorts must not rise, else byte-restore) is
owned by the caller (`cli/_compose_route.py`) — see
docs/plans/unconnected-signal-repair-c1-design.md.
"""
from __future__ import annotations

import math
from pathlib import Path

from kicraft.autoplacer.fab_profile import fab_floors
import re
from dataclasses import dataclass
from typing import Any

import pcbnew

from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec, add_breakout_stubs
from kicraft.autoplacer.hardware.keepout_extract import (
    collect_track_via_rule_areas, track_intersects_rule_area,
)

_DEFAULT_MAX_TIE_MM = 60.0
_DEFAULT_ESCAPE_MM = (0.0, 1.0, 2.0, 4.0)
# add_breakout_stubs re-loads the board per call (~0.3-1 s on a parent), so
# the candidate walk is budgeted: escape dirs × lengths × paths explodes past
# any value the marginal candidate adds.
_DEFAULT_MAX_ATTEMPTS = 120

# One endpoint line of an [unconnected_items] block. Real shapes seen:
#     @(123.5788 mm, 114.2167 mm): Pad A6 [USB_D+] of J1 on F.Cu
#     @(147.5492 mm, 106.3944 mm): Track [BUTTON_GPIO] on F.Cu, length 1.1857 mm
#     @(147.6699 mm, 132.1643 mm): PTH pad 5 [nRESET] of J1
# The kind prefix ("Pad X" / "PTH pad X" / "Track" / "Via") and the layer
# suffix both vary, so capture loosely and post-process.
_ITEM_RE = re.compile(
    r"@\(([-\d.]+)\s*mm,\s*([-\d.]+)\s*mm\):\s*(.*?)\[([^\]]+)\]"
    r"(?:\s+of\s+(\S+))?(.*)$"
)
_PAD_PREFIX_RE = re.compile(r"(?i)^(?:\w+\s+)?pad\s+(\S+)\s*$")


@dataclass(slots=True)
class _End:
    xy: tuple[float, float]
    net: str
    ref: str | None      # None for track/via endpoints
    pad: str | None
    layers: set[str]     # {"F.Cu"}, {"B.Cu"} or both
    kind: str = "item"   # DRC item kind; zone coordinates are not copper anchors


def _parse_unconnected_edges(report_text: str) -> list[tuple[_End, _End]]:
    """The ``unconnected_items`` edges of a kicad-cli DRC text report."""
    edges: list[tuple[_End, _End]] = []
    block: list[_End] = []
    in_block = False
    for line in report_text.splitlines():
        header = re.match(r"^\[(\w+)\]:", line)
        if header:
            if in_block and len(block) == 2:
                edges.append((block[0], block[1]))
            in_block = header.group(1) == "unconnected_items"
            block = []
            continue
        if not in_block:
            continue
        m = _ITEM_RE.search(line)
        if not m:
            continue
        x, y, kind, net, ref, tail = m.groups()
        pad_m = _PAD_PREFIX_RE.match(kind.strip())
        pad = pad_m.group(1) if pad_m else None
        layers = {t for t in ("F.Cu", "B.Cu") if t in tail}
        if not layers:  # PTH pads carry no layer suffix: they span both
            layers = {"F.Cu", "B.Cu"}
        block.append(_End(
            xy=(float(x), float(y)), net=net,
            ref=ref if pad is not None else None, pad=pad, layers=layers,
            kind=("zone" if re.match(r"(?i)^zone\b", kind.strip())
                  else "pad" if pad_m else kind.strip().lower()),
        ))
    if in_block and len(block) == 2:
        edges.append((block[0], block[1]))
    return edges


def _connected_anchors(board: Any, connectivity: Any, end: _End) -> list[_End]:
    """Add real via/PTH access in the reported item's copper component."""
    point = pcbnew.VECTOR2I(*(pcbnew.FromMM(v) for v in end.xy))
    tolerance = pcbnew.FromMM(0.0001)  # Text DRC coordinates have four decimals.
    if end.ref is not None:
        items = [pad for fp in board.GetFootprints()
                 if fp.GetReferenceAsString() == end.ref
                 for pad in fp.Pads() if pad.GetNumber() == end.pad]
    else:
        items = board.GetTracks()
    seeds = [
        item for item in items
        if item.GetNetname() == end.net
        and any(name in end.layers and item.GetLayerSet().Contains(layer)
                for name, layer in (("F.Cu", pcbnew.F_Cu), ("B.Cu", pcbnew.B_Cu)))
        and item.HitTest(point, tolerance)
    ]
    anchors = [end]
    visited: set[str] = set()
    for seed in seeds:
        if seed.m_Uuid.AsString() in visited:
            continue
        if visited:
            # Rounded coordinates can be ambiguous. Never borrow access from
            # a different disconnected component merely sharing a net name.
            return [end]
        pending = [seed]
        while pending:
            item = pending.pop()
            uid = item.m_Uuid.AsString()
            if uid in visited or item.GetNetname() != end.net:
                continue
            visited.add(uid)
            is_pad = isinstance(item, pcbnew.PAD)
            if item.Type() == pcbnew.PCB_VIA_T or (
                is_pad and item.GetAttribute() == pcbnew.PAD_ATTRIB_PTH
            ):
                pos = item.GetPosition()
                layers = {
                    name for name, layer in
                    (("F.Cu", pcbnew.F_Cu), ("B.Cu", pcbnew.B_Cu))
                    if item.GetLayerSet().Contains(layer)
                }
                if layers:
                    anchors.append(_End(
                        xy=(pcbnew.ToMM(pos.x), pcbnew.ToMM(pos.y)),
                        net=end.net,
                        ref=item.GetParentFootprint().GetReferenceAsString() if is_pad else None,
                        pad=item.GetNumber() if is_pad else None,
                        layers=layers,
                    ))
            pending.extend(connectivity.GetConnectedTracks(item))
            pending.extend(connectivity.GetConnectedPads(item))
    return list({
        (anchor.xy, anchor.ref, anchor.pad, tuple(sorted(anchor.layers))): anchor
        for anchor in anchors
    }.values())


def _zone_component_pairs(
    board: Any, net: str, a: _End, b: _End, max_gap: float,
) -> list[tuple[list[_End], list[_End]]]:
    """Resolve zone reports to real pad/via access on filled components.

    A zone report's coordinate identifies the zone object, not a point on the
    reported filled island.  Never feed it to the router.  Cluster membership
    comes from the existing filled-copper analysis; only native pads and vias
    supply the actual routing anchors.
    """
    from kicraft.autoplacer.brain.gnd_pour import _collect_net_clusters

    clusters, nodes = _collect_net_clusters(board, net)
    groups: list[tuple[bool, list[_End]]] = []
    for indices in clusters.values():
        members = [nodes[i] for i in indices]
        anchors = []
        for node in members:
            if node["kind"] not in ("pad", "via"):
                continue
            layers = {
                name for name, layer in (("F.Cu", pcbnew.F_Cu), ("B.Cu", pcbnew.B_Cu))
                if layer in node["layers"]
            }
            if layers:
                anchors.append(_End(
                    xy=node["xy"], net=net, ref=node.get("ref"),
                    pad=node.get("num"), layers=layers, kind=node["kind"],
                ))
        if anchors:
            anchors.sort(key=lambda e: (e.xy, e.ref or "", e.pad or ""))
            groups.append((any(n["kind"] == "island" for n in members), anchors))

    # For a mixed edge, the non-zone item must really belong to one side.
    # Track endpoints can be used to resolve membership, but are never
    # substituted for the pad/via anchors used to route a zone edge.
    fixed = b if a.kind == "zone" and b.kind != "zone" else (
        a if b.kind == "zone" and a.kind != "zone" else None
    )
    fixed_groups: set[int] | None = None
    if fixed is not None:
        board.BuildConnectivity()
        connected = _connected_anchors(board, board.GetConnectivity(), fixed)
        # DRC text rounds coordinates to four decimal places. Keep native
        # item identity while allowing that same endpoint lookup tolerance.
        fixed_groups = {
            i for i, (_, anchors) in enumerate(groups)
            if any(
                anchor.ref == end.ref and anchor.pad == end.pad
                and math.dist(anchor.xy, end.xy) <= 0.0001
                for anchor in anchors for end in connected
            )
        }
        if len(fixed_groups) != 1:
            return []

    candidates = []
    for i, (island_a, anchors_a) in enumerate(groups):
        for j in range(i + 1, len(groups)):
            island_b, anchors_b = groups[j]
            if fixed_groups is None:
                if not island_a or not island_b:
                    continue
            elif not ((i in fixed_groups and island_b)
                      or (j in fixed_groups and island_a)):
                continue
            distance = min(math.dist(x.xy, y.xy) for x in anchors_a for y in anchors_b)
            if distance <= max_gap:
                candidates.append((distance, anchors_a[0].xy, anchors_b[0].xy,
                                   anchors_a, anchors_b))
    candidates.sort(key=lambda item: item[:3])
    return [(x, y) for _, _, _, x, y in candidates]


def _escape_dirs(board: "pcbnew.BOARD", end: _End) -> list[tuple[float, float]]:
    """Unit exit directions for a pad, most-promising first.

    All 8 compass directions, ordered radially-away-from-the-footprint-centre
    first — the stub guards prune illegal ones, and the wall shape decides
    which exit works (a header-row end pin must exit PERPENDICULAR to the
    row, which the pure radial heuristic got exactly wrong)."""
    if end.ref is None:
        return []
    compass = [
        (math.cos(k * math.pi / 4.0), math.sin(k * math.pi / 4.0))
        for k in range(8)
    ]
    for fp in board.GetFootprints():
        if fp.GetReferenceAsString() != end.ref:
            continue
        c = fp.GetPosition()
        dx = end.xy[0] - pcbnew.ToMM(c.x)
        dy = end.xy[1] - pcbnew.ToMM(c.y)
        norm = math.hypot(dx, dy)
        if norm >= 0.05:
            ux, uy = dx / norm, dy / norm
            compass.sort(key=lambda d: -(d[0] * ux + d[1] * uy))
        return compass
    return compass


def _candidate_paths(
    src: tuple[float, float],
    tgt: tuple[float, float],
    escape_pt: tuple[float, float] | None,
) -> list[list[tuple[float, float]]]:
    """Waypoint lists from *src* to *tgt*, cheapest shape first.

    Straight, the two L-bends, then 3-segment doglegs whose midpoints are
    offset perpendicular to the straight line — the off-grid bend that
    threads the corridor routing's grid-habit copper left open. Offsets
    scale with the gap so a 20 mm abandoned route explores real detours, not
    just millimetre wiggles. When *escape_pt* is given every path leaves the
    pad through it first.
    """
    head = [escape_pt] if escape_pt else []
    base = escape_pt or src
    sx, sy = base
    tx, ty = tgt
    paths: list[list[tuple[float, float]]] = [head + [tgt]]
    if abs(sx - tx) > 0.01 and abs(sy - ty) > 0.01:
        paths.append(head + [(sx, ty), tgt])
        paths.append(head + [(tx, sy), tgt])
    dx, dy = tx - sx, ty - sy
    length = math.hypot(dx, dy)
    if length > 0.5:
        px, py = -dy / length, dx / length  # unit perpendicular
        offsets = sorted({1.0, 2.0, 4.0, length / 4.0, length / 2.0})
        for off in offsets:
            for sign in (1.0, -1.0):
                ox, oy = px * off * sign, py * off * sign
                m1 = (sx + dx / 3.0 + ox, sy + dy / 3.0 + oy)
                m2 = (sx + 2.0 * dx / 3.0 + ox, sy + 2.0 * dy / 3.0 + oy)
                paths.append(head + [m1, m2, tgt])
    return paths


def _seg_seg_dist_mm(
    a1: tuple[float, float], a2: tuple[float, float],
    b1: tuple[float, float], b2: tuple[float, float],
) -> float:
    """Minimum distance between two 2D segments."""

    def _seg_pt(p1, p2, q):
        vx, vy = p2[0] - p1[0], p2[1] - p1[1]
        wx, wy = q[0] - p1[0], q[1] - p1[1]
        vv = vx * vx + vy * vy
        t = 0.0 if vv <= 1e-12 else max(0.0, min(1.0, (wx * vx + wy * vy) / vv))
        return math.hypot(q[0] - (p1[0] + t * vx), q[1] - (p1[1] + t * vy))

    d1x, d1y = a2[0] - a1[0], a2[1] - a1[1]
    d2x, d2y = b2[0] - b1[0], b2[1] - b1[1]
    denom = d1x * d2y - d1y * d2x
    if abs(denom) > 1e-12:
        t = ((b1[0] - a1[0]) * d2y - (b1[1] - a1[1]) * d2x) / denom
        u = ((b1[0] - a1[0]) * d1y - (b1[1] - a1[1]) * d1x) / denom
        if 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0:
            return 0.0  # proper crossing
    return min(
        _seg_pt(a1, a2, b1), _seg_pt(a1, a2, b2),
        _seg_pt(b1, b2, a1), _seg_pt(b1, b2, a2),
    )


# An obstacle is (layer, net, x1, y1, x2, y2, halfwidth_mm): tracks/vias as
# fat segments (via = zero-length), pads as fat segments across their bbox's
# long axis (coarser than the true shape -- deliberately; the authoritative
# guards inside add_breakout_stubs re-check every survivor exactly).
_Obstacle = tuple[str, str, float, float, float, float, float]


def _copper_obstacles(board: "pcbnew.BOARD") -> list[_Obstacle]:
    out: list[_Obstacle] = []
    for t in board.GetTracks():
        net = t.GetNetname()
        if isinstance(t, pcbnew.PCB_VIA):
            p = t.GetPosition()
            x, y = pcbnew.ToMM(p.x), pcbnew.ToMM(p.y)
            try:
                width = t.GetWidth(pcbnew.F_Cu)
            except TypeError:
                width = t.GetWidth()
            r = pcbnew.ToMM(width) / 2.0
            out.append(("*", net, x, y, x, y, r))  # vias span layers
            continue
        s, e = t.GetStart(), t.GetEnd()
        out.append((
            t.GetLayerName(), net,
            pcbnew.ToMM(s.x), pcbnew.ToMM(s.y),
            pcbnew.ToMM(e.x), pcbnew.ToMM(e.y),
            pcbnew.ToMM(t.GetWidth()) / 2.0,
        ))
    for fp in board.GetFootprints():
        for pad in fp.Pads():
            net = pad.GetNetname()
            bb = pad.GetBoundingBox()
            x1, y1 = pcbnew.ToMM(bb.GetLeft()), pcbnew.ToMM(bb.GetTop())
            x2, y2 = pcbnew.ToMM(bb.GetRight()), pcbnew.ToMM(bb.GetBottom())
            w, h = x2 - x1, y2 - y1
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            # Fat segment along the bbox's long axis, halfwidth = short/2.
            if w >= h:
                seg = (x1 + h / 2.0, cy, x2 - h / 2.0, cy, h / 2.0)
            else:
                seg = (cx, y1 + w / 2.0, cx, y2 - w / 2.0, w / 2.0)
            on_f = pad.IsOnLayer(pcbnew.F_Cu)
            on_b = pad.IsOnLayer(pcbnew.B_Cu)
            layer = "*" if (on_f and on_b) else ("F.Cu" if on_f else "B.Cu")
            out.append((layer, net, *seg))
    return out


# Obstacle hits within this radius of either edge endpoint are ignored: the
# stub must APPROACH copper at both ends (its own pad's neighbors, the
# target's surroundings), and breakout_stubs owns the exact rules there.
_ENDPOINT_CARVEOUT_MM = 2.0


def _path_clear(
    obstacles: list[_Obstacle],
    chain: list[tuple[float, float]],
    *,
    layer: str,
    net: str,
    margin_mm: float,
    src: tuple[float, float],
    tgt: tuple[float, float],
) -> bool:
    """Coarse in-memory screen: does the candidate chain obviously collide
    with foreign copper? False = certainly-colliding, skip the expensive
    stamping attempt; True = plausible (the stamping guards re-check exactly).
    """
    for a1, a2 in zip(chain, chain[1:]):
        for olayer, onet, x1, y1, x2, y2, half in obstacles:
            if onet == net or (olayer != "*" and olayer != layer):
                continue
            d = _seg_seg_dist_mm(a1, a2, (x1, y1), (x2, y2))
            if d >= half + margin_mm:
                continue
            # Hit -- forgive it if it sits inside an endpoint carve-out.
            ocx, ocy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            if (math.hypot(ocx - src[0], ocy - src[1]) <= _ENDPOINT_CARVEOUT_MM
                    or math.hypot(ocx - tgt[0], ocy - tgt[1])
                    <= _ENDPOINT_CARVEOUT_MM):
                continue
            return False
    return True


def _pour_nets(board: "pcbnew.BOARD", cfg: dict[str, Any]) -> set[str]:
    """Nets not owned by signal repair.

    Explicit extra nets may re-admit a zone-owned net, but never ground or a
    canonical power net: this pass stamps signal-width copper.
    """
    skip = {str(cfg.get("gnd_zone_net", "GND"))}
    power = {str(n) for n in cfg.get("power_nets") or []}
    skip.update(power)
    for n in cfg.get("power_plane_nets") or []:
        skip.add(str(n))
    for zone in board.Zones():
        name = zone.GetNetname()
        if name:
            skip.add(name)
    gnd = str(cfg.get("gnd_zone_net", "GND"))
    for n in cfg.get("signal_repair_extra_nets") or []:
        if str(n) != gnd and str(n) not in power:
            skip.discard(str(n))
    return skip


def _gap_cap_mm(board: "pcbnew.BOARD", cfg: dict[str, Any]) -> float:
    """Longest endpoint gap the repair will attempt.

    Default scales to the board diagonal -- a fixed 60 mm read every
    castellated-header fan-out edge (62-86 mm on run_10's RP2040 board) as
    gap_over_cap without trying a single candidate. An explicit
    ``signal_repair_max_mm`` still wins."""
    cap_cfg = cfg.get("signal_repair_max_mm")
    if cap_cfg is not None:
        return float(cap_cfg)
    bb = board.GetBoardEdgesBoundingBox()
    return max(_DEFAULT_MAX_TIE_MM, math.hypot(
        pcbnew.ToMM(bb.GetWidth()), pcbnew.ToMM(bb.GetHeight())
    ))


def repair_unconnected_signals(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Close every unconnected signal DRC edge that a guarded tie can reach.

    Returns ``{edges, tied, skipped, pruned}``. Mutates the board file only by adding
    guarded copper (and refilling zones so the pours close around it); an
    edge none of the candidate paths can close legally is reported in
    ``skipped`` and left exactly as found. The caller owns accept-or-revert.
    """
    cfg = cfg or {}
    max_attempts = int(
        cfg.get("signal_repair_max_attempts", _DEFAULT_MAX_ATTEMPTS)
    )
    escapes = tuple(
        float(e) for e in cfg.get("signal_repair_escape_mm", _DEFAULT_ESCAPE_MM)
    )
    summary: dict[str, Any] = {"edges": 0, "tied": 0, "skipped": [], "pruned": 0}

    from kicraft.autoplacer.routing_board import run_kicad_cli_drc

    drc = run_kicad_cli_drc(pcb_path)
    edges = _parse_unconnected_edges(drc.get("report_text") or "")
    if not edges:
        return summary

    board = pcbnew.LoadBoard(pcb_path)
    max_tie_mm = _gap_cap_mm(board, cfg)
    pour = _pour_nets(board, cfg)
    # In-memory copper map for the candidate pre-filter: without it the
    # attempt budget dies inside the first escape direction's path family
    # (each stamping attempt reloads the board), which the 20260713 batch
    # measured as attempt_budget on 10 of 12 skipped edges while NONE were
    # actually unroutable. Screening costs microseconds per candidate, so
    # the budget is spent only on paths that don't obviously cross copper.
    obstacles = _copper_obstacles(board)
    board.BuildConnectivity()
    connectivity = board.GetConnectivity()
    floors = fab_floors(cfg)
    pre_margin_mm = floors["clearance_mm"] + floors["track_mm"] / 2.0
    rule_areas = collect_track_via_rule_areas(board)
    track_rule_areas = {
        name: [area for area in rule_areas
               if area.blocks_tracks and area.zone.IsOnLayer(layer)]
        for name, layer in (("F.Cu", pcbnew.F_Cu), ("B.Cu", pcbnew.B_Cu))
    }
    edges = [e for e in edges if e[0].net not in pour]
    # Smallest gap first: cheap wins land before congested ones consume
    # stamped-copper budget (each stamped tie is an obstacle for the next).
    edges.sort(key=lambda e: math.hypot(
        e[0].xy[0] - e[1].xy[0], e[0].xy[1] - e[1].xy[1]
    ))
    summary["edges"] = len(edges)

    for a, b in edges:
        gap = math.hypot(a.xy[0] - b.xy[0], a.xy[1] - b.xy[1])
        label = (f"{a.net}:{(a.ref or 'track')}.{a.pad or ''}"
                 f"->{(b.ref or 'track')}.{b.pad or ''}")
        if a.kind == "zone" or b.kind == "zone":
            summary["skipped"].append(f"{label}:no_native_anchor")
            continue
        if gap > max_tie_mm:
            summary["skipped"].append(f"{label}:gap_{gap:.1f}mm_over_cap")
            continue
        tied = False
        attempts = 0
        pruned = 0
        # A DRC track end is not the only access to its connected component.
        # Prefer existing common back-layer access, without inventing a via at
        # a front-only coordinate. Original endpoints remain candidates.
        a_anchors = _connected_anchors(board, connectivity, a)
        b_anchors = _connected_anchors(board, connectivity, b)
        pairs = [
            pair for src in a_anchors for tgt in b_anchors
            for pair in ((src, tgt), (tgt, src))
            if math.dist(src.xy, tgt.xy) <= max_tie_mm
        ]
        pairs.sort(key=lambda pair: (
            "B.Cu" not in (pair[0].layers & pair[1].layers),
            not bool(pair[0].layers & pair[1].layers),
            math.dist(pair[0].xy, pair[1].xy),
            pair[0].xy, pair[1].xy,
        ))
        for src, tgt in pairs:
            src_is_pad = src.ref is not None and src.pad is not None
            if src_is_pad:
                dirs = _escape_dirs(board, src)
                src_escapes = escapes
            else:
                # No footprint to escape radially from: bare-coordinate
                # anchors go straight into the path family.
                dirs = []
                src_escapes = (0.0,)
            for layer in sorted(src.layers):  # B.Cu first: pour-side is open
                needs_via = layer not in tgt.layers
                for esc_len in src_escapes:
                    esc_pts = ([None] if esc_len == 0.0 or not dirs else [
                        (src.xy[0] + ux * esc_len, src.xy[1] + uy * esc_len)
                        for ux, uy in dirs
                    ])
                    for esc in esc_pts:
                        for path in _candidate_paths(src.xy, tgt.xy, esc):
                            if attempts >= max_attempts:
                                break
                            points = [src.xy] + path
                            if not _path_clear(
                                obstacles, points,
                                layer=layer, net=src.net,
                                margin_mm=pre_margin_mm,
                                src=src.xy, tgt=tgt.xy,
                            ) or any(
                                track_intersects_rule_area(
                                    start, end, floors["track_mm"] / 2.0, area,
                                )
                                for start, end in zip(points, path)
                                for area in track_rule_areas[layer]
                            ):
                                pruned += 1
                                continue
                            attempts += 1
                            res = add_breakout_stubs(
                                pcb_path,
                                [BreakoutSpec(
                                    ref=src.ref or "track",
                                    pad=(src.pad if src.pad is not None else
                                         f"{src.xy[0]:.2f},{src.xy[1]:.2f}"),
                                    waypoints=path, layer=layer,
                                    via_at_end=needs_via, near_xy=src.xy,
                                    start_xy=(None if src_is_pad else src.xy),
                                    net=(None if src_is_pad else src.net),
                                )],
                                cfg=cfg,
                            )
                            if res.get("stubs", 0):
                                summary["tied"] += 1
                                tied = True
                                break
                        if tied or attempts >= max_attempts:
                            break
                    if tied or attempts >= max_attempts:
                        break
                if tied or attempts >= max_attempts:
                    break
            if tied or attempts >= max_attempts:
                break
        if not tied and attempts < max_attempts:
            from kicraft.autoplacer.brain.signal_mesh import _mesh_repair_specs

            # Earlier ties may have added foreign copper since the initial
            # screen was built. Mesh the current board, not that stale copy.
            mesh_board = pcbnew.LoadBoard(pcb_path)
            specs = _mesh_repair_specs(
                mesh_board, a.net,
                [(end.xy, frozenset(end.layers)) for end in a_anchors],
                [(end.xy, frozenset(end.layers)) for end in b_anchors],
                cfg,
            )
            if specs and math.dist(
                specs[0].start_xy, specs[-1].waypoints[-1],
            ) <= max_tie_mm:
                attempts += 1
                original = Path(pcb_path).read_bytes()
                accepted = False
                try:
                    result = add_breakout_stubs(pcb_path, specs, cfg=cfg)
                    accepted = result.get("stubs", 0) == len(specs)
                finally:
                    if not accepted:
                        Path(pcb_path).write_bytes(original)
                if accepted:
                    summary["tied"] += 1
                    tied = True
        summary["pruned"] += pruned
        if not tied:
            # no_pad_anchor: nothing to even try -- vestigial now that a
            # track/via endpoint anchors as a free coordinate (every edge
            # generates candidates), kept for the degenerate zero-candidate
            # case. attempt_budget: ran out of STAMPING budget on plausible
            # paths. no_clear_path: the generated candidates failed screening
            # or stamping; this is not a proof that every lawful route is absent.
            reason = ("no_pad_anchor" if not attempts and not pruned
                      else "attempt_budget" if attempts >= max_attempts
                      else "no_clear_path")
            summary["skipped"].append(f"{label}:{reason}")

    if summary["tied"]:
        # New tracks must cut clearance through the pours.
        board = pcbnew.LoadBoard(pcb_path)
        pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        board.Save(pcb_path)
    return summary


def repair_unconnected_power(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Complete DRC-reported canonical power gaps with full-class copper.

    Mesh planning uses the same native connectivity anchors, ground masks and
    guarded batch stamper as signal mesh repair. The caller owns the final
    native accept/revert verdict; an incomplete batch is always restored.
    """
    from kicraft.autoplacer.brain.signal_mesh import _mesh_repair_specs
    from kicraft.autoplacer.kicad_routing_tools import net_class_geometry
    from kicraft.autoplacer.routing_board import run_kicad_cli_drc

    cfg = cfg or {}
    summary: dict[str, Any] = {"edges": 0, "tied": 0, "skipped": [], "pruned": 0}
    grounds = {str(cfg.get("gnd_zone_net", "GND")), "GND"}
    for contract in cfg.get("local_bypasses") or []:
        ground_return = contract.get("ground_return") if isinstance(contract, dict) else None
        if isinstance(ground_return, dict) and ground_return.get("net"):
            grounds.add(str(ground_return["net"]))
    power = {str(name) for name in cfg.get("power_nets") or []} - grounds - {""}
    if not power:
        return summary
    drc = run_kicad_cli_drc(pcb_path)
    edges = [
        (a, b) for a, b in _parse_unconnected_edges(drc.get("report_text") or "")
        if a.net in power and b.net == a.net
    ]
    edges.sort(key=lambda edge: math.dist(edge[0].xy, edge[1].xy))
    summary["edges"] = len(edges)
    if not edges:
        return summary

    geometry = net_class_geometry(pcb_path, cfg, {a.net for a, _ in edges})
    max_attempts = int(cfg.get("signal_repair_max_attempts", _DEFAULT_MAX_ATTEMPTS))
    path = Path(pcb_path)
    original = path.read_bytes()
    sidecars = [(p, p.read_bytes() if p.is_file() else None)
                for p in (path.with_suffix(".kicad_pro"), path.with_suffix(".kicad_dru"))]
    attempted_zone_pairs: set[tuple[str, frozenset, frozenset]] = set()
    try:
        for a, b in edges:
            board = pcbnew.LoadBoard(pcb_path)
            max_tie_mm = _gap_cap_mm(board, cfg)
            is_zone_edge = a.kind == "zone" or b.kind == "zone"
            gap = math.dist(a.xy, b.xy)
            label = (f"{a.net}:{(a.ref or ('zone' if a.kind == 'zone' else 'track'))}.{a.pad or ''}"
                     f"->{(b.ref or ('zone' if b.kind == 'zone' else 'track'))}.{b.pad or ''}")
            if not is_zone_edge and gap > max_tie_mm:
                summary["skipped"].append(f"{label}:gap_{gap:.1f}mm_over_cap")
                continue
            if max_attempts <= 0:
                summary["skipped"].append(f"{label}:attempt_budget")
                continue
            dimensions = geometry[a.net]
            per_net = dict(cfg)
            capability = dict(cfg.get("fab_capability") or {})
            capability["min_track_mm"] = max(
                fab_floors(cfg)["track_mm"], dimensions["track_width"]
            )
            per_net["fab_capability"] = capability
            per_net["escape_via_size_mm"] = dimensions["via_diameter"]
            per_net["escape_via_drill_mm"] = dimensions["via_drill"]
            board.BuildConnectivity()
            connectivity = board.GetConnectivity()
            def has_native_endpoint(end: _End) -> bool:
                point = pcbnew.VECTOR2I(*(pcbnew.FromMM(v) for v in end.xy))
                if end.ref is not None:
                    items = [pad for fp in board.GetFootprints()
                             if fp.GetReferenceAsString() == end.ref
                             for pad in fp.Pads() if pad.GetNumber() == end.pad]
                else:
                    items = board.GetTracks()
                return any(
                    item.GetNetname() == end.net
                    and any(name in end.layers and item.GetLayerSet().Contains(layer)
                            for name, layer in (("F.Cu", pcbnew.F_Cu), ("B.Cu", pcbnew.B_Cu)))
                    and item.HitTest(point, pcbnew.FromMM(0.0001))
                    for item in items
                )

            if is_zone_edge:
                non_zones = [end for end in (a, b) if end.kind != "zone"]
                if any(not has_native_endpoint(end) for end in non_zones):
                    summary["skipped"].append(f"{label}:no_native_anchor")
                    continue
                pairs = _zone_component_pairs(board, a.net, a, b, max_tie_mm)
            else:
                if not has_native_endpoint(a) or not has_native_endpoint(b):
                    summary["skipped"].append(f"{label}:no_native_anchor")
                    continue
                pairs = [(_connected_anchors(board, connectivity, a),
                          _connected_anchors(board, connectivity, b))]
            if not pairs:
                summary["skipped"].append(f"{label}:no_native_anchor")
                continue
            accepted = False
            attempts = 0
            for sources, targets in pairs:
                if is_zone_edge:
                    def anchor_keys(anchors: list[_End]) -> frozenset:
                        return frozenset((e.xy, e.ref, e.pad) for e in anchors)
                    left, right = anchor_keys(sources), anchor_keys(targets)
                    if any(net == a.net and (
                        (left & old_left and right & old_right)
                        or (left & old_right and right & old_left)
                    ) for net, old_left, old_right in attempted_zone_pairs):
                        continue
                    attempted_zone_pairs.add((a.net, left, right))
                if attempts >= max_attempts:
                    break
                attempts += 1
                specs = _mesh_repair_specs(
                    board, a.net,
                    [(end.xy, frozenset(end.layers)) for end in sources],
                    [(end.xy, frozenset(end.layers)) for end in targets],
                    per_net,
                )
                if not specs or math.dist(specs[0].start_xy, specs[-1].waypoints[-1]) > max_tie_mm:
                    continue
                before_batch = path.read_bytes()
                try:
                    result = add_breakout_stubs(pcb_path, specs, cfg=per_net)
                    accepted = result.get("stubs", 0) == len(specs)
                finally:
                    if not accepted:
                        path.write_bytes(before_batch)
                if accepted:
                    summary["tied"] += 1
                    if is_zone_edge:
                        # Rebuild filled components before processing another
                        # DRC edge; report coordinates cannot distinguish islands.
                        board = pcbnew.LoadBoard(pcb_path)
                        pcbnew.ZONE_FILLER(board).Fill(board.Zones())
                        board.Save(pcb_path)
                    break
            if not accepted:
                summary["skipped"].append(f"{label}:no_clear_path")
        if summary["tied"]:
            board = pcbnew.LoadBoard(pcb_path)
            pcbnew.ZONE_FILLER(board).Fill(board.Zones())
            board.Save(pcb_path)
    except BaseException:
        path.write_bytes(original)
        raise
    finally:
        for sidecar, content in sidecars:
            if content is None:
                sidecar.unlink(missing_ok=True)
            else:
                sidecar.write_bytes(content)
    return summary
