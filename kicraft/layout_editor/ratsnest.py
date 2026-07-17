"""Cross-leaf ratsnest for the manual layout canvas.

Joins each leaf's routed interface anchors (``solved_layout.json``
``interface_anchors``: the pad position where an external net leaves
the leaf's copper) across leaves by net name, so the canvas can draw
live connection lines while the user drags. This is what turns manual
placement from "arrange colored rectangles" into "place a circuit":
routability is decided by which blocks talk to each other, and the
anchors are the exact points FreeRouting will later have to join.

Coordinate frames: ``interface_anchors`` are in the leaf solver's
re-based local frame (board top-left at 0,0), while the canvas works
in the ``leaf_routed.kicad_pcb`` page frame (the frame of the leaf's
Edge.Cuts AABB). ``metadata.json``'s ``local_board_outline`` top-left
is that re-base offset, so canvas = anchor + top_left. Falls back to
the leaf's Edge.Cuts minimum when the metadata outline is absent.

Net names: an anchor's ``port_name`` is the net as THIS leaf knows it.
For representative leaves the metadata ``interface_ports`` name ->
net_name map applies any port/net aliasing; for replicated siblings
the metadata ports are the representative's (stale), but the anchors
themselves are already net-remapped, so the port_name fallback is the
correct sibling net.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kicraft.layout_editor.leaves import LeafInfo

# Nets never drawn: plane-connected on the parent (a GND pour joins
# them without point-to-point routing), so ratsnest lines would be
# noise implying routing work that doesn't exist.
_PLANE_NETS = {"GND"}


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _leaf_anchor_offset(meta: dict[str, Any], leaf: LeafInfo) -> tuple[float, float]:
    outline = meta.get("local_board_outline") or {}
    try:
        return (float(outline["top_left_x"]), float(outline["top_left_y"]))
    except (KeyError, TypeError, ValueError):
        return (leaf.edge_min_x, leaf.edge_min_y)


def build_ratsnest(leaves: list[LeafInfo]) -> list[dict[str, Any]]:
    """Return ``[{net, anchors: [{instance_path, x, y}]}]``.

    ``x``/``y`` are in the owning leaf's canvas-local (``leaf_routed``
    page) frame -- the same frame as the leaf's Edge.Cuts AABB -- so the
    canvas applies exactly its leaf placement transform to position
    them. Only nets that anchor in two or more distinct leaves are
    returned (a single-leaf net has nothing to pull toward).
    """
    by_net: dict[str, list[dict[str, Any]]] = {}
    for leaf in leaves:
        sl = _read_json(leaf.artifact_dir / "solved_layout.json")
        if sl is None:
            continue
        meta = _read_json(leaf.artifact_dir / "metadata.json") or {}
        ports_map = {
            str(p.get("name", "")): str(p.get("net_name") or p.get("name", ""))
            for p in (meta.get("interface_ports") or [])
            if isinstance(p, dict)
        }
        off_x, off_y = _leaf_anchor_offset(meta, leaf)
        for anchor in sl.get("interface_anchors") or []:
            if not isinstance(anchor, dict):
                continue
            port = str(anchor.get("port_name", ""))
            net = ports_map.get(port, port)
            if not net or net.upper() in _PLANE_NETS:
                continue
            pos = anchor.get("pos") or {}
            try:
                x = float(pos["x"]) + off_x
                y = float(pos["y"]) + off_y
            except (KeyError, TypeError, ValueError):
                continue
            by_net.setdefault(net, []).append(
                {
                    "instance_path": leaf.instance_path,
                    "x": round(x, 3),
                    "y": round(y, 3),
                }
            )

    nets: list[dict[str, Any]] = []
    for net in sorted(by_net):
        anchors = by_net[net]
        if len({a["instance_path"] for a in anchors}) < 2:
            continue
        nets.append({"net": net, "anchors": anchors})
    return nets
