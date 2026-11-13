"""The copper stack a ``.kicad_pcb`` declares, read without importing pcbnew.

Two consumers need this fact outside a pcbnew process: the fab gerber layer list (a
``kicad-cli`` argument) and the router's ``--layers`` (the router runs as a subprocess). Both
read the board file so a design that asked for more than two copper layers is plotted and
routed on the stack it declared, instead of on a constant.
"""

from __future__ import annotations

import re
from pathlib import Path

#: Quoted copper-layer names anywhere in a board's head -- the ``(layers ...)`` block writes
#: them first, and a zone's own layer list repeats names already seen.
_COPPER_LAYER_RE = re.compile(r'"(F\.Cu|B\.Cu|In\d+\.Cu)"')

#: Bytes read when scanning a board's head. A routed board's track list can run to megabytes
#: and the layer block is written before it.
_HEAD_BYTES = 16384

#: The layers signals keep on any stack: a board that carries planes inside routes on the
#: outer pair, which is what KiCraft's front/back copper bookkeeping can represent.
OUTER_COPPER_LAYERS = ("F.Cu", "B.Cu")


def declared_copper_layers(pcb_path: str | Path) -> list[str]:
    """Copper layers a board file declares, in stack order (F.Cu, In1..InN, B.Cu).

    De-duplicated while keeping first-seen order. Empty when the file cannot be read, which
    callers must treat as "unknown" rather than "two layers".
    """
    try:
        with open(pcb_path, encoding="utf-8", errors="replace") as handle:
            head = handle.read(_HEAD_BYTES)
    except OSError:
        return []
    return list(dict.fromkeys(_COPPER_LAYER_RE.findall(head)))


def _sexpr_end(text: str, start: int) -> int:
    """Return the end of a balanced KiCad expression, respecting quoted strings."""
    depth = 0
    quoted = escaped = False
    for i in range(start, len(text)):
        c = text[i]
        if quoted:
            if escaped:
                escaped = False
            elif c == "\\":
                escaped = True
            elif c == '"':
                quoted = False
        elif c == '"':
            quoted = True
        elif c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                return i + 1
    raise ValueError("unbalanced KiCad board expression")


def set_two_ounce_outer_stackup(pcb_path: str | Path) -> None:
    """Declare a 1.6 mm two-layer board with 0.070 mm finished outer copper.

    KiCad's Python bindings do not expose stackup item setters. Change only the
    board setup expression; subsequent native loads preserve this stackup and
    KiCad's Gerber job exporter reads it.
    """
    path = Path(pcb_path)
    if declared_copper_layers(path) != ["F.Cu", "B.Cu"]:
        raise ValueError("2 oz thermal stackup currently requires a two-layer board")
    text = path.read_text(encoding="utf-8")
    setup = re.search(r"\(setup\s", text)
    if setup is None:
        raise ValueError("PCB has no setup block")
    end = _sexpr_end(text, setup.start())
    block = text[setup.start():end]
    existing = re.search(r"\(stackup\s", block)
    stack = (
        '(stackup\n'
        '  (layer "F.SilkS" (type "Top Silk Screen"))\n'
        '  (layer "F.Paste" (type "Top Solder Paste"))\n'
        '  (layer "F.Mask" (type "Top Solder Mask") (thickness 0.01))\n'
        '  (layer "F.Cu" (type "copper") (thickness 0.07))\n'
        '  (layer "dielectric 1" (type "core") (thickness 1.44) '
        '(material "FR4") (epsilon_r 4.5) (loss_tangent 0.02))\n'
        '  (layer "B.Cu" (type "copper") (thickness 0.07))\n'
        '  (layer "B.Mask" (type "Bottom Solder Mask") (thickness 0.01))\n'
        '  (layer "B.Paste" (type "Bottom Solder Paste"))\n'
        '  (layer "B.SilkS" (type "Bottom Silk Screen"))\n'
        '  (copper_finish "None") (dielectric_constraints no)\n'
        ')'
    )
    if existing:
        block = block[:existing.start()] + stack + block[_sexpr_end(block, existing.start()):]
    else:
        block = block[:-1] + "\n  " + stack + "\n)"
    path.write_text(text[:setup.start()] + block + text[end:], encoding="utf-8")


def has_two_ounce_outer_stackup(pcb_path: str | Path) -> bool:
    text = Path(pcb_path).read_text(encoding="utf-8")
    match = re.search(r"\(stackup\s", text)
    if not match:
        return False
    block = text[match.start():_sexpr_end(text, match.start())]
    return all(re.search(
        rf'\(layer\s+"{re.escape(layer)}"\s+\(type\s+"copper"\)\s+'
        r'\(thickness\s+0\.0*7(?:0*)\s*\)', block
    ) for layer in OUTER_COPPER_LAYERS)
