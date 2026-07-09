"""Diagnostic SVG renderer for a placed leaf — the *visual* half of the
placement-streamline proof (the numeric half is ``leaf_tidiness``).

Renders one leaf as an annotated diagram so tidiness and routability problems
are legible to the eye, not just to a metric:

* **courtyards + pads**, passives tinted by their functional group (same
  ``assign_passive_groups`` the metric uses, so picture and numbers agree);
* each group's **ideal axis** (dashed) with a **residual tick** per member — you
  see how far each part sits off a straight row;
* **misoriented** passives (against their group's dominant axis) outlined in the
  warning color — the "random rotation" look, flagged;
* **courtyard overlaps** and **off-board** parts in the critical color;
* a **ratsnest** overlay (per-net MST) so routing load / congestion is visible —
  this is what exposes a tidy row strangling a routing channel;
* a **metrics panel** (orientation consensus, residual, fill) and a legend.

Pure geometry + string building — no pcbnew, no external deps. Colors are the
data-viz skill's validated palette (categorical for groups, reserved status
hues for flags). An explicit light surface is drawn so the file reads correctly
regardless of the viewer's theme.
"""

from __future__ import annotations

import math
from typing import Optional

from .leaf_tidiness import (
    LeafTidiness,
    PassiveGroup,
    assign_passive_groups,
    leaf_tidiness,
    orientation_axis,
    parts_from_components,
)
from .types import Component, Point

# --- validated palette (data-viz skill, light surface) --------------------- #
_SURFACE = "#fcfcfb"
_INK = "#0b0b0b"
_INK2 = "#52514e"
_MUTED = "#898781"
_GRID = "#e1e0d9"
_OUTLINE = "#52514e"
_WARN = "#fab219"   # misoriented
_CRIT = "#d03b3b"   # overlap / off-board
_GOOD = "#0ca30c"
# categorical slots (red/status hues omitted so flags never impersonate a group)
_GROUP_HUES = [
    "#2a78d6",  # blue
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e87ba4",  # magenta
    "#eb6834",  # orange
]
_ANCHOR_FILL = "#d8d7d2"
_PAD = "#b08040"          # copper-ish, neutral
_RATS_SIGNAL = "#256abf"  # signal ratsnest
_RATS_POWER = "#c9c8c2"   # power/GND ratsnest (faint — it's a plane)
_POWER_FANOUT = 6


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _mst_edges(pts: list[Point]) -> list[tuple[Point, Point]]:
    """Prim's MST over pad centers — the ratsnest for one net."""
    if len(pts) < 2:
        return []
    used = {0}
    edges: list[tuple[Point, Point]] = []
    while len(used) < len(pts):
        best = None
        for i in used:
            for j in range(len(pts)):
                if j in used:
                    continue
                d = pts[i].dist(pts[j])
                if best is None or d < best[0]:
                    best = (d, i, j)
        _, i, j = best
        used.add(j)
        edges.append((pts[i], pts[j]))
    return edges


def render_leaf_svg(
    components: dict[str, Component],
    board_outline: tuple[Point, Point],
    *,
    groups: Optional[list[PassiveGroup]] = None,
    metrics: Optional[LeafTidiness] = None,
    title: str = "",
    scale_px: int = 640,
) -> str:
    parts = parts_from_components(components)
    if groups is None:
        groups = assign_passive_groups(parts)
    if metrics is None:
        metrics = leaf_tidiness(parts, label=title)

    ref_group: dict[str, int] = {}
    for gi, g in enumerate(groups):
        for r in g.passive_refs:
            ref_group[r] = gi

    # --- content bbox (parts + board), transform ---
    xs, ys = [], []
    for c in components.values():
        tl, br = c.physical_bbox()
        xs += [tl.x, br.x]
        ys += [tl.y, br.y]
    (b_tl, b_br) = board_outline
    xs += [b_tl.x, b_br.x]
    ys += [b_tl.y, b_br.y]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    w = max(1e-3, max_x - min_x)
    h = max(1e-3, max_y - min_y)
    scale = scale_px / max(w, h)
    pad_l, pad_top, legend_h = 26.0, 74.0, 108.0

    def X(x: float) -> float:
        return pad_l + (x - min_x) * scale

    def Y(y: float) -> float:
        return pad_top + (y - min_y) * scale

    svg_w = pad_l * 2 + w * scale
    svg_h = pad_top + h * scale + legend_h

    out: list[str] = []
    out.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w:.0f}" '
        f'height="{svg_h:.0f}" viewBox="0 0 {svg_w:.0f} {svg_h:.0f}" '
        f'font-family="system-ui,-apple-system,\'Segoe UI\',sans-serif">'
    )
    out.append(f'<rect width="{svg_w:.0f}" height="{svg_h:.0f}" fill="{_SURFACE}"/>')

    # --- metrics panel ---
    def _fmt(v, suffix=""):
        return "—" if v is None else f"{v:.0f}{suffix}" if suffix == "%" else (
            f"{v:.2f}{suffix}" if isinstance(v, float) else f"{v}{suffix}")

    out.append(
        f'<text x="{pad_l:.0f}" y="24" fill="{_INK}" font-size="15" '
        f'font-weight="600">{_esc(title)}</text>'
    )
    kpis = [
        ("orient (grp)", _fmt(metrics.orientation_consensus_grouped_pct, "%")),
        ("orient (leaf)", _fmt(metrics.orientation_consensus_leaf_pct, "%")),
        ("residual", _fmt(metrics.alignment_residual_mm, "mm")),
        ("fill", _fmt(metrics.packing_fill_pct, "%")),
        ("groups", str(metrics.n_groups)),
    ]
    kx = pad_l
    for label, val in kpis:
        out.append(
            f'<text x="{kx:.0f}" y="46" fill="{_MUTED}" font-size="10">{label}</text>'
            f'<text x="{kx:.0f}" y="62" fill="{_INK}" font-size="14" '
            f'font-weight="600">{val}</text>'
        )
        kx += 96

    # --- board outline ---
    out.append(
        f'<rect x="{X(b_tl.x):.1f}" y="{Y(b_tl.y):.1f}" '
        f'width="{(b_br.x - b_tl.x) * scale:.1f}" height="{(b_br.y - b_tl.y) * scale:.1f}" '
        f'fill="none" stroke="{_GRID}" stroke-width="1.5"/>'
    )

    # --- ratsnest (under parts) ---
    net_pts: dict[str, list[Point]] = {}
    for c in components.values():
        for p in c.pads:
            if p.net:
                net_pts.setdefault(p.net, []).append(p.pos)
    for net, pts in sorted(net_pts.items()):
        power = len(pts) > _POWER_FANOUT
        color = _RATS_POWER if power else _RATS_SIGNAL
        wdt = 0.6 if power else 0.9
        op = 0.35 if power else 0.7
        for a, b in _mst_edges(pts):
            out.append(
                f'<line x1="{X(a.x):.1f}" y1="{Y(a.y):.1f}" x2="{X(b.x):.1f}" '
                f'y2="{Y(b.y):.1f}" stroke="{color}" stroke-width="{wdt}" '
                f'opacity="{op}"/>'
            )

    # --- components ---
    overlaps = _overlap_pairs(components)
    overlapped = {r for pair in overlaps for r in pair}
    for ref, c in sorted(components.items()):
        tl, br = c.bbox()
        gi = ref_group.get(ref)
        is_passive = c.kind == "passive"
        fill = (
            _GROUP_HUES[gi % len(_GROUP_HUES)] if (is_passive and gi is not None)
            else _ANCHOR_FILL
        )
        misorient = (
            is_passive and gi is not None
            and orientation_axis(c.rotation) != _group_dominant(components, groups[gi])
        )
        stroke = _CRIT if ref in overlapped else (_WARN if misorient else _OUTLINE)
        sw = 2.0 if (ref in overlapped or misorient) else 0.8
        out.append(
            f'<rect x="{X(tl.x):.1f}" y="{Y(tl.y):.1f}" '
            f'width="{(br.x - tl.x) * scale:.1f}" height="{(br.y - tl.y) * scale:.1f}" '
            f'fill="{fill}" fill-opacity="0.30" stroke="{stroke}" stroke-width="{sw}"/>'
        )
        # pads
        for p in c.pads:
            p_tl, p_br = p.bbox()
            out.append(
                f'<rect x="{X(p_tl.x):.1f}" y="{Y(p_tl.y):.1f}" '
                f'width="{max(1.0, (p_br.x - p_tl.x) * scale):.1f}" '
                f'height="{max(1.0, (p_br.y - p_tl.y) * scale):.1f}" '
                f'fill="{_PAD}" opacity="0.8"/>'
            )
        cx, cy = X((tl.x + br.x) / 2), Y((tl.y + br.y) / 2)
        out.append(
            f'<text x="{cx:.1f}" y="{cy + 3:.1f}" fill="{_INK}" font-size="9" '
            f'text-anchor="middle">{_esc(ref)}</text>'
        )

    # --- group ideal axes + residual ticks ---
    for gi, g in enumerate(groups):
        members = [components[r] for r in g.passive_refs if r in components]
        if len(members) < 2:
            continue
        color = _GROUP_HUES[gi % len(_GROUP_HUES)]
        cs = [(m.body_center or m.pos) for m in members]
        gxs = [p.x for p in cs]
        gys = [p.y for p in cs]
        horizontal = (max(gxs) - min(gxs)) >= (max(gys) - min(gys))
        if horizontal:
            perp = sum(gys) / len(gys)
            out.append(
                f'<line x1="{X(min(gxs)):.1f}" y1="{Y(perp):.1f}" '
                f'x2="{X(max(gxs)):.1f}" y2="{Y(perp):.1f}" stroke="{color}" '
                f'stroke-width="1" stroke-dasharray="4 3" opacity="0.9"/>'
            )
            for p in cs:
                out.append(
                    f'<line x1="{X(p.x):.1f}" y1="{Y(p.y):.1f}" x2="{X(p.x):.1f}" '
                    f'y2="{Y(perp):.1f}" stroke="{color}" stroke-width="1.4"/>'
                )
        else:
            perp = sum(gxs) / len(gxs)
            out.append(
                f'<line x1="{X(perp):.1f}" y1="{Y(min(gys)):.1f}" '
                f'x2="{X(perp):.1f}" y2="{Y(max(gys)):.1f}" stroke="{color}" '
                f'stroke-width="1" stroke-dasharray="4 3" opacity="0.9"/>'
            )
            for p in cs:
                out.append(
                    f'<line x1="{X(p.x):.1f}" y1="{Y(p.y):.1f}" x2="{X(perp):.1f}" '
                    f'y2="{Y(p.y):.1f}" stroke="{color}" stroke-width="1.4"/>'
                )

    # --- legend ---
    ly = pad_top + h * scale + 18
    out.append(
        f'<text x="{pad_l:.0f}" y="{ly:.0f}" fill="{_INK2}" font-size="10" '
        f'font-weight="600">groups (passives by anchor):</text>'
    )
    lx = pad_l
    ly2 = ly + 16
    for gi, g in enumerate(groups):
        color = _GROUP_HUES[gi % len(_GROUP_HUES)]
        out.append(
            f'<rect x="{lx:.0f}" y="{ly2 - 9:.0f}" width="10" height="10" '
            f'fill="{color}" fill-opacity="0.5" stroke="{color}"/>'
            f'<text x="{lx + 14:.0f}" y="{ly2:.0f}" fill="{_INK2}" '
            f'font-size="10">{_esc(g.anchor_ref)}</text>'
        )
        lx += 62
        if lx > svg_w - 80:
            lx = pad_l
            ly2 += 16
    # flag legend
    fy = ly2 + 22
    for i, (sw_color, label) in enumerate([
        (_WARN, "misoriented"), (_CRIT, "courtyard overlap"),
        (_RATS_SIGNAL, "signal net"), (_RATS_POWER, "power/GND net"),
    ]):
        fx = pad_l + i * 150
        if i < 2:
            out.append(
                f'<rect x="{fx:.0f}" y="{fy - 9:.0f}" width="10" height="10" '
                f'fill="none" stroke="{sw_color}" stroke-width="2"/>'
            )
        else:
            out.append(
                f'<line x1="{fx:.0f}" y1="{fy - 4:.0f}" x2="{fx + 10:.0f}" '
                f'y2="{fy - 4:.0f}" stroke="{sw_color}" stroke-width="2"/>'
            )
        out.append(
            f'<text x="{fx + 14:.0f}" y="{fy:.0f}" fill="{_INK2}" '
            f'font-size="10">{_esc(label)}</text>'
        )

    out.append("</svg>")
    return "".join(out)


def _group_dominant(components: dict[str, Component], g: PassiveGroup) -> str:
    rots = [components[r].rotation for r in g.passive_refs if r in components]
    hh = sum(1 for r in rots if orientation_axis(r) == "H")
    return "H" if hh >= (len(rots) - hh) else "V"


def _overlap_pairs(components: dict[str, Component]) -> list[tuple[str, str]]:
    items = list(components.items())
    pairs = []
    for i in range(len(items)):
        ra, a = items[i]
        a_tl, a_br = a.bbox()
        for j in range(i + 1, len(items)):
            rb, b = items[j]
            if a.layer != b.layer:
                continue
            b_tl, b_br = b.bbox()
            ox = min(a_br.x, b_br.x) - max(a_tl.x, b_tl.x)
            oy = min(a_br.y, b_br.y) - max(a_tl.y, b_tl.y)
            if ox > 0.05 and oy > 0.05:
                pairs.append((ra, rb))
    return pairs


__all__ = ["render_leaf_svg"]
