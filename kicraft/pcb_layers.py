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
