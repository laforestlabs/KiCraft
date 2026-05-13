"""Single Edge.Cuts AABB parser. Used by the renderer, the manual
layout runner, and the verification tool so all three see the same
physical board extent for a given leaf PCB."""

from __future__ import annotations

import re
from pathlib import Path

# Matches a top-level ``(gr_<kind> ...)`` block and stops at the next
# ``(gr_`` / ``(footprint`` / end-of-file. Non-greedy ``.*?`` plus the
# lookahead handles arbitrarily-nested geometry tokens inside the
# block (``(start ...)`` / ``(xy ...)`` etc.) because s-expression
# children don't start with ``(gr_`` or ``(footprint``.
_BLOCK_RE = re.compile(
    r'\(gr_(line|arc|rect|poly|circle)\s+(.*?)\)\s*(?=\(gr_|\(footprint|\Z)',
    re.S,
)
_POINT_RE = re.compile(
    r'(?:\((?:start|end|center|mid)\s+([-\d.eE+]+)\s+([-\d.eE+]+)\))'
    r'|(?:\(xy\s+([-\d.eE+]+)\s+([-\d.eE+]+)\))'
)


def parse_edge_cuts_aabb(
    pcb_path: Path,
) -> tuple[float, float, float, float] | None:
    """AABB of every ``gr_line`` / ``gr_arc`` / ``gr_rect`` / ``gr_poly``
    / ``gr_circle`` on the Edge.Cuts layer of ``pcb_path``. Returns
    ``(min_x, min_y, max_x, max_y)`` in the PCB's coordinate system
    (mm) or ``None`` when the file is missing, unreadable, or has no
    Edge.Cuts geometry.

    The result is the PHYSICAL extent of the board outline -- what
    gets stamped on the parent for a leaf, what defines collision
    on the manual layout canvas, and what the rendered PNG should
    contain by construction.
    """
    try:
        text = pcb_path.read_text(encoding="utf-8")
    except OSError:
        return None

    xs: list[float] = []
    ys: list[float] = []
    for m in _BLOCK_RE.finditer(text):
        blk = m.group(0)
        if 'Edge.Cuts' not in blk:
            continue
        for pm in _POINT_RE.finditer(blk):
            if pm.group(1) is not None:
                xs.append(float(pm.group(1)))
                ys.append(float(pm.group(2)))
            else:
                xs.append(float(pm.group(3)))
                ys.append(float(pm.group(4)))
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))
