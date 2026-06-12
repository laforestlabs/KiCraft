"""Compatibility shim: the manual layout schema moved to
``kicraft.layout_editor.model`` (PR: layout-editor extraction).

Import from the new location; this re-export keeps any out-of-tree
callers and saved tooling working until the offline GUI is retired.
"""

from kicraft.layout_editor.model import (
    MOUNTING_HOLE_CORNERS,
    SCHEMA_VERSION,
    ManualLayout,
    ManualLeafPlacement,
    ManualMountingHole,
    ManualParentLocalPlacement,
    load_manual_layout,
    save_manual_layout,
)

__all__ = [
    "MOUNTING_HOLE_CORNERS",
    "SCHEMA_VERSION",
    "ManualLayout",
    "ManualLeafPlacement",
    "ManualMountingHole",
    "ManualParentLocalPlacement",
    "load_manual_layout",
    "save_manual_layout",
]
