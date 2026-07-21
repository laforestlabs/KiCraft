"""Constrained bend/via repair for walled-off unconnected SIGNAL nets (C1).

After freerouting, a few signal nets are sometimes left unrouted because the
router walled itself off (no_clear_path — memory
`kicraft-unconnected-1-cluster-walled-off-signal-power`). This pass closes
each unconnected DRC edge of such a net with guarded copper, at the same
abstraction level as the accepted pour repairs
(`gnd_pour.repair_stranded_net`), but with the richer path family the
walled-off family needs: a radial escape off the pad first (fine-pitch pads
have exactly one legal exit direction), L-bends, perpendicular-offset
doglegs scaled to the gap, and at most one layer-changing via — all stamped
through :func:`breakout_stubs.add_breakout_stubs`, which inherits every
clearance / foreign-pad / outline / hole guard and *skips* any candidate
that cannot land legally.

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
import re
from dataclasses import dataclass
from typing import Any

import pcbnew

from kicraft.autoplacer.brain.breakout_stubs import BreakoutSpec, add_breakout_stubs

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
        ))
    if in_block and len(block) == 2:
        edges.append((block[0], block[1]))
    return edges


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
    threads the corridor freerouting's grid-habit copper left open. Offsets
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
            r = pcbnew.ToMM(t.GetWidth()) / 2.0
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
    """Nets owned by the pour repairs (GND + power planes), never touched here."""
    skip = {str(cfg.get("gnd_zone_net", "GND"))}
    for n in cfg.get("power_plane_nets") or []:
        skip.add(str(n))
    for zone in board.Zones():
        name = zone.GetNetname()
        if name:
            skip.add(name)
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

    from kicraft.autoplacer.freerouting_runner import _run_kicad_cli_drc

    drc = _run_kicad_cli_drc(pcb_path)
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
    pre_margin_mm = (
        float(cfg.get("freerouting_min_clearance_mm", 0.153))
        + float(cfg.get("freerouting_fine_pitch_track_mm", 0.153)) / 2.0
    )
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
        if gap > max_tie_mm:
            summary["skipped"].append(f"{label}:gap_{gap:.1f}mm_over_cap")
            continue
        tied = False
        attempts = 0
        pruned = 0
        # Try both directions -- pad anchors first ((a,b) then (b,a) keeps
        # the pad side leading when only one side has a pad; a track/via
        # endpoint anchors as a free coordinate (C1 v2: the pure
        # track->track edges -- run_10's USB_DN/XIN -- were unreachable
        # under the pads-only rule and always skipped no_pad_anchor).
        for src, tgt in ((a, b), (b, a)):
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
                            if not _path_clear(
                                obstacles, [src.xy] + path,
                                layer=layer, net=src.net,
                                margin_mm=pre_margin_mm,
                                src=src.xy, tgt=tgt.xy,
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
            if tied:
                break
        summary["pruned"] += pruned
        if not tied:
            # no_pad_anchor: nothing to even try -- vestigial now that a
            # track/via endpoint anchors as a free coordinate (every edge
            # generates candidates), kept for the degenerate zero-candidate
            # case. attempt_budget: ran out of STAMPING budget on plausible
            # paths. no_clear_path: every candidate was screened out or
            # rejected by the stamping guards -- with the pre-filter this
            # again means "geometry says blocked", not "gave up early".
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
