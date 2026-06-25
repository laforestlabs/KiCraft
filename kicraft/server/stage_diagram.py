"""Pure ECharts option-dict builders for the conceptual stage diagrams.

The functional_spec and architecture stages are fundamentally connectivity
graphs, but the inspector used to render them as plain tables. These builders
turn an already-committed slot into an ECharts ``graph`` series option -- a
node-link diagram -- so the user *sees* the circuit's structure at a glance.

Plain data in, option dict out -- no NiceGUI/connection context -- mirroring
``kicraft/loadtest/charts.py``. ``web.py`` feeds the returned dicts straight to
``ui.echart``. Both adapters are pure and deterministic: the same slot always
yields the identical option (coordinates included), so refreshes never shuffle
the graph.

Design (see docs/plans/stage-concept-diagrams.md):
  * layout: "none" + Python-computed coordinates (deterministic, no jitter).
  * functional_spec: directed block diagram; nodes = blocks colored by
    category, edges = connections styled by signal_type.
  * architecture: sheet connectivity; nodes = sheets (replication-collapsed),
    nets with >=3 endpoints or any power/ground net render as a net-hub node
    with spokes, 2-endpoint signal nets render as one labeled edge.
"""
from __future__ import annotations

import math

from ..design.models import is_power_or_ground_name

# -- colour / style tables (single source of truth for builders + legend) ----

# functional_spec block categories -> fixed colours. Order is the left-to-right
# signal-flow column order (power left -> sense/process middle -> drive/
# interface right).
BLOCK_CATEGORY_COLORS: dict[str, str] = {
    "power": "#f59e0b",      # amber
    "sense": "#38bdf8",      # sky
    "process": "#a78bfa",    # violet
    "drive": "#34d399",      # emerald
    "interface": "#fb7185",  # rose
}

# category -> layout column (deterministic left-to-right flow).
_CATEGORY_COLUMN: dict[str, int] = {
    "power": 0,
    "sense": 1,
    "process": 2,
    "drive": 3,
    "interface": 4,
}

# connection signal_type -> edge line style.
SIGNAL_LINE_STYLE: dict[str, dict] = {
    "power":   {"color": "#f59e0b", "width": 2.5},
    "ground":  {"color": "#94a3b8", "width": 2.5},
    "digital": {"color": "#38bdf8", "width": 1.5},
    "analog":  {"color": "#f472b6", "width": 1.5},
    "clock":   {"color": "#facc15", "width": 1.5},
    "bus":     {"color": "#a78bfa", "width": 2.0, "type": "dashed"},
    "rf":      {"color": "#fb7185", "width": 2.0},
    "other":   {"color": "#cbd5e1", "width": 1.5},
}

# architecture: node colours.
SHEET_COLOR = "#60a5fa"       # blue
POWER_HUB_COLOR = "#f59e0b"   # amber -- rails read differently from signals
SIGNAL_HUB_COLOR = "#a78bfa"  # violet

_LABEL = {"color": "#e2e8f0", "fontSize": 11}
_TITLE = {"color": "#e2e8f0", "fontSize": 13}

_COL_DX = 240   # horizontal spacing between layout columns
_ROW_DY = 130   # vertical spacing between rows in a column
_CIRCLE_R = 320  # architecture: sheet-node ring radius


def build_graph_option(nodes, edges, categories, *, title, directed=True):
    """Assemble an ECharts ``graph`` series option from prebuilt nodes/edges.

    ``nodes``  -- list of dicts: {name, x, y, category (name str), value,
                  symbol?, symbolSize?, itemStyle?}.
    ``edges``  -- list of dicts: {source, target, lineStyle?, label?, value?,
                  edgeSymbol?}.
    ``categories`` -- list of {name, itemStyle: {color}} driving the legend
                  and node colour fallback.
    ``directed`` -- series-level edgeSymbol default (arrow on target).
    """
    edge_symbol = ["none", "arrow"] if directed else ["none", "none"]
    return {
        "backgroundColor": "transparent",
        "title": {"text": title, "textStyle": _TITLE},
        "tooltip": {"trigger": "item", "formatter": "{b}<br/>{c}"},
        "legend": [{
            "top": 24,
            "textStyle": {"color": "#94a3b8", "fontSize": 10},
            "data": [c["name"] for c in categories],
        }],
        "series": [{
            "type": "graph",
            "layout": "none",
            "roam": True,
            "label": {"show": True, "color": "#e2e8f0", "fontSize": 11},
            "labelLayout": {"hideOverlap": True},
            "edgeSymbol": edge_symbol,
            "edgeSymbolSize": [0, 9],
            "categories": categories,
            "data": nodes,
            "links": edges,
            "lineStyle": {"color": "#64748b", "width": 1.5, "curveness": 0.12},
            "emphasis": {"focus": "adjacency",
                         "label": {"show": True}},
        }],
    }


# -- functional_spec ---------------------------------------------------------

def functional_spec_diagram(slot: dict) -> dict | None:
    """Build a directed block diagram from a functional_spec slot.

    Nodes = blocks[] (coloured by category, ``×N`` when count > 1). Edges =
    connections[] (directed, styled by signal_type). Returns None when there
    are no blocks (nothing to draw).
    """
    blocks = slot.get("blocks") or []
    if not blocks:
        return None
    connections = slot.get("connections") or []

    # Only include categories that are actually present, in canonical order.
    present_cats = []
    seen = set()
    for b in blocks:
        cat = b.get("category", "other")
        if cat not in seen:
            seen.add(cat)
            present_cats.append(cat)
    present_cats.sort(key=lambda c: _CATEGORY_COLUMN.get(c, 99))
    categories = [
        {"name": c, "itemStyle": {"color": BLOCK_CATEGORY_COLORS.get(c, "#94a3b8")}}
        for c in present_cats
    ]
    cat_index = {c: i for i, c in enumerate(present_cats)}

    # Deterministic coords: x by category column, y by input order within col.
    col_counts: dict[int, int] = {}
    nodes = []
    name_to_node = {}
    for b in blocks:
        name = b.get("name", "")
        cat = b.get("category", "other")
        col = _CATEGORY_COLUMN.get(cat, len(_CATEGORY_COLUMN))
        idx = col_counts.get(col, 0)
        col_counts[col] = idx + 1
        x = col * _COL_DX
        y = idx * _ROW_DY
        count = b.get("count", 1) or 1
        label = name if count <= 1 else f"{name} ×{count}"
        node = {
            "name": label,
            "x": x,
            "y": y,
            "category": cat_index.get(cat, 0),
            "value": b.get("purpose", ""),
            "symbolSize": 46,
        }
        nodes.append(node)
        name_to_node[name] = node

    edges = []
    for c in connections:
        src = c.get("from_block", "")
        tgt = c.get("to_block", "")
        sn = name_to_node.get(src)
        tn = name_to_node.get(tgt)
        if sn is None or tn is None:
            continue  # connection to an unknown block -- skip defensively
        stype = c.get("signal_type", "other")
        style = dict(SIGNAL_LINE_STYLE.get(stype, SIGNAL_LINE_STYLE["other"]))
        edges.append({
            "source": sn["name"],
            "target": tn["name"],
            "edgeSymbol": ["none", "arrow"],
            "lineStyle": style,
            "value": c.get("description", "") or stype,
        })

    return build_graph_option(
        nodes, edges, categories,
        title="Block diagram",
        directed=True,
    )


# -- architecture ------------------------------------------------------------

def _sheet_rep_map(sheets: list[dict]):
    """Map every sheet name to (representative_name, count).

    Sheets sharing a ``replication_group`` collapse to the instance-1
    representative (the geometry donor) with a ``×N`` count; unique sheets map
    to themselves with count 1.
    """
    groups: dict[str, list[dict]] = {}
    for s in sheets:
        g = s.get("replication_group")
        if g:
            groups.setdefault(g, []).append(s)
    rep_of_group: dict[str, dict] = {}
    count_of_group: dict[str, int] = {}
    for g, members in groups.items():
        rep = next(
            (m for m in members if m.get("replication_instance") == 1),
            members[0],
        )
        rep_of_group[g] = rep
        count_of_group[g] = len(members)
    out: dict[str, tuple[str, int]] = {}
    for s in sheets:
        g = s.get("replication_group")
        if g and g in rep_of_group:
            out[s.get("name", "")] = (rep_of_group[g].get("name", ""),
                                      count_of_group[g])
        else:
            out[s.get("name", "")] = (s.get("name", ""), 1)
    return out


def architecture_diagram(slot: dict) -> dict | None:
    """Build a sheet connectivity diagram from an architecture slot.

    Nodes = sheets[] (replication-collapsed). Each ``inter_sheet_net`` becomes
    either a single labelled edge (2 endpoints, non-power) or a net-hub node
    with spokes (>=3 endpoints, or any power/ground net). Returns None when
    there are no sheets.
    """
    sheets = slot.get("sheets") or []
    if not sheets:
        return None
    nets = slot.get("inter_sheet_nets") or []
    power_nets = set(slot.get("power_nets") or [])

    rep_map = _sheet_rep_map(sheets)

    # Representative sheet nodes, laid out on a circle (deterministic).
    reps: list[tuple[str, dict, int]] = []  # (rep_name, original_sheet, count)
    seen_rep: set[str] = set()
    for s in sheets:
        rep_name, count = rep_map.get(s.get("name", ""), (s.get("name", ""), 1))
        if rep_name in seen_rep:
            continue
        seen_rep.add(rep_name)
        reps.append((rep_name, s, count))

    n = len(reps)
    nodes = []
    pos: dict[str, tuple[float, float]] = {}
    for i, (rep_name, s, count) in enumerate(reps):
        angle = 2 * math.pi * i / n - math.pi / 2  # start at top
        x = _CIRCLE_R * math.cos(angle)
        y = _CIRCLE_R * math.sin(angle)
        label = rep_name if count <= 1 else f"{rep_name} ×{count}"
        pos[rep_name] = (x, y)
        nodes.append({
            "name": label,
            "x": round(x, 1),
            "y": round(y, 1),
            "category": 0,  # "Sheet"
            "value": s.get("function", ""),
            "symbol": "circle",
            "symbolSize": 50,
        })

    categories = [
        {"name": "Sheet", "itemStyle": {"color": SHEET_COLOR}},
        {"name": "Power net", "itemStyle": {"color": POWER_HUB_COLOR}},
        {"name": "Signal net", "itemStyle": {"color": SIGNAL_HUB_COLOR}},
    ]

    edges = []
    hub_index = 0
    n_hubs = sum(
        1 for net in nets
        if _is_power_net(net.get("name", ""), power_nets)
        or len(_dedup_endpoints(net, rep_map)) >= 3
    )
    for net in nets:
        net_name = net.get("name", "")
        endpoints = net.get("endpoints") or []
        # Map endpoint sheets to representatives and dedupe.
        mapped: list[tuple[str, str]] = []
        seen_ep: set[str] = set()
        for ep in endpoints:
            sheet_name = ep.get("sheet", "")
            rep_name, _ = rep_map.get(sheet_name, (sheet_name, 1))
            if rep_name in seen_ep:
                continue
            seen_ep.add(rep_name)
            mapped.append((rep_name, ep.get("direction", "bidirectional")))
        if len(mapped) < 2:
            continue  # net collapsed away or malformed -- nothing to draw

        is_power = _is_power_net(net_name, power_nets)
        if len(mapped) >= 3 or is_power:
            # Net-hub node + undirected spokes.
            hub_label = f"net:{net_name}"
            hub_cat = 1 if is_power else 2
            hub_color = POWER_HUB_COLOR if is_power else SIGNAL_HUB_COLOR
            # Place the hub at the centroid of its endpoint sheets, pulled
            # toward the ring centre, with a small deterministic per-hub offset
            # so multiple hubs don't stack on top of each other.
            cx = sum(pos[r][0] for r, _ in mapped) / len(mapped)
            cy = sum(pos[r][1] for r, _ in mapped) / len(mapped)
            cx *= 0.55
            cy *= 0.55
            offset = (hub_index - (n_hubs - 1) / 2) * 36 if n_hubs else 0.0
            hub_index += 1
            hx = round(cx + offset, 1)
            hy = round(cy, 1)
            tip = net_name + "  ·  " + ", ".join(
                f"{r}({d})" for r, d in mapped
            )
            nodes.append({
                "name": hub_label,
                "x": hx,
                "y": hy,
                "category": hub_cat,
                "value": tip,
                "symbol": "diamond",
                "symbolSize": 22,
                "itemStyle": {"color": hub_color},
                "label": {"show": True, "formatter": net_name,
                          "color": "#e2e8f0", "fontSize": 10},
            })
            for r, _ in mapped:
                if r not in pos:
                    continue
                edges.append({
                    "source": hub_label,
                    "target": r,
                    "edgeSymbol": ["none", "none"],
                    "lineStyle": {"color": hub_color, "width": 1.5},
                    "value": net_name,
                })
        else:
            # 2-endpoint signal net -> a single labelled edge.
            (r1, d1), (r2, d2) = mapped
            src, tgt, arrow = _directed_pair(r1, d1, r2, d2)
            if src is None or tgt is None or src not in pos or tgt not in pos:
                continue
            edges.append({
                "source": src,
                "target": tgt,
                "edgeSymbol": arrow,
                "lineStyle": {"color": SIGNAL_HUB_COLOR, "width": 1.8},
                "label": {"show": True, "formatter": net_name,
                          "color": "#cbd5e1", "fontSize": 10},
                "value": net_name,
            })

    return build_graph_option(
        nodes, edges, categories,
        title="Sheet connectivity",
        directed=False,
    )


def _is_power_net(name: str, power_nets: set[str]) -> bool:
    """A net is power/ground if declared in ``power_nets`` or matched by the
    project's name heuristics (covers rails like +3V3, GND, VBUS)."""
    return name in power_nets or is_power_or_ground_name(name)


def _dedup_endpoints(net: dict, rep_map: dict[str, tuple[str, int]]) -> list[str]:
    """Representative sheet names a net touches (deduped, order-preserving)."""
    out: list[str] = []
    seen: set[str] = set()
    for ep in net.get("endpoints") or []:
        sheet_name = ep.get("sheet", "")
        rep_name, _ = rep_map.get(sheet_name, (sheet_name, 1))
        if rep_name not in seen:
            seen.add(rep_name)
            out.append(rep_name)
    return out


def _directed_pair(r1, d1, r2, d2):
    """Resolve a 2-endpoint net into (source, target, edgeSymbol).

    output -> input yields an arrow; anything else (bidirectional/passive) is
    undirected with a canonical source so the link is stable.
    """
    if d1 == "output" and d2 == "input":
        return r1, r2, ["none", "arrow"]
    if d1 == "input" and d2 == "output":
        return r2, r1, ["none", "arrow"]
    # bidirectional / passive / unknown -> undirected, stable order.
    if r1 <= r2:
        return r1, r2, ["none", "none"]
    return r2, r1, ["none", "none"]
