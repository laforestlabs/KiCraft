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


def repair_unconnected_signals(
    pcb_path: str,
    cfg: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Close every unconnected signal DRC edge that a guarded tie can reach.

    Returns ``{edges, tied, skipped}``. Mutates the board file only by adding
    guarded copper (and refilling zones so the pours close around it); an
    edge none of the candidate paths can close legally is reported in
    ``skipped`` and left exactly as found. The caller owns accept-or-revert.
    """
    cfg = cfg or {}
    max_tie_mm = float(cfg.get("signal_repair_max_mm", _DEFAULT_MAX_TIE_MM))
    max_attempts = int(
        cfg.get("signal_repair_max_attempts", _DEFAULT_MAX_ATTEMPTS)
    )
    escapes = tuple(
        float(e) for e in cfg.get("signal_repair_escape_mm", _DEFAULT_ESCAPE_MM)
    )
    summary: dict[str, Any] = {"edges": 0, "tied": 0, "skipped": []}

    from kicraft.autoplacer.freerouting_runner import _run_kicad_cli_drc

    drc = _run_kicad_cli_drc(pcb_path)
    edges = _parse_unconnected_edges(drc.get("report_text") or "")
    if not edges:
        return summary

    board = pcbnew.LoadBoard(pcb_path)
    pour = _pour_nets(board, cfg)
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
        # Try both directions: the anchor must be a pad, and the more open
        # side often stamps where the walled side cannot.
        for src, tgt in ((a, b), (b, a)):
            if src.ref is None or src.pad is None:
                continue
            dirs = _escape_dirs(board, src)
            for layer in sorted(src.layers):  # B.Cu first: pour-side is open
                needs_via = layer not in tgt.layers
                for esc_len in escapes:
                    esc_pts = ([None] if esc_len == 0.0 else [
                        (src.xy[0] + ux * esc_len, src.xy[1] + uy * esc_len)
                        for ux, uy in dirs
                    ])
                    for esc in esc_pts:
                        for path in _candidate_paths(src.xy, tgt.xy, esc):
                            if attempts >= max_attempts:
                                break
                            attempts += 1
                            res = add_breakout_stubs(
                                pcb_path,
                                [BreakoutSpec(
                                    ref=src.ref, pad=src.pad,
                                    waypoints=path, layer=layer,
                                    via_at_end=needs_via, near_xy=src.xy,
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
        if not tied:
            reason = ("attempt_budget" if attempts >= max_attempts
                      else "no_pad_anchor" if not attempts
                      else "no_clear_path")
            summary["skipped"].append(f"{label}:{reason}")

    if summary["tied"]:
        # New tracks must cut clearance through the pours.
        board = pcbnew.LoadBoard(pcb_path)
        pcbnew.ZONE_FILLER(board).Fill(board.Zones())
        board.Save(pcb_path)
    return summary
