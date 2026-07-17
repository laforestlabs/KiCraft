"""GUI-agnostic manual-layout editor core.

Owns the manual layout data model, leaf discovery + canvas PNG
rendering, compose/stamp/route orchestration, the per-component
placement-rules data layer, and the shared canvas JS asset. The
web app (``kicraft.server``) is a thin host over this package.
"""

from kicraft.layout_editor.canvas import (
    build_canvas_html,
    build_canvas_init_script,
    default_asset_url,
    static_js_path,
)
from kicraft.layout_editor.leaves import (
    LeafInfo,
    discover_leaves,
    experiments_mount_url_for,
    prerender_leaf_canvases,
)
from kicraft.layout_editor.model import (
    MOUNTING_HOLE_CORNERS,
    ManualLayout,
    ManualLeafPlacement,
    ManualMountingHole,
    ManualParentLocalPlacement,
    load_manual_layout,
    save_manual_layout,
)
from kicraft.layout_editor.render import render_leaf_canvas
from kicraft.layout_editor.runner import (
    load_initial_layout,
    run_manual_compose,
    save_manual_layout_json,
)

__all__ = [
    "MOUNTING_HOLE_CORNERS",
    "LeafInfo",
    "ManualLayout",
    "ManualLeafPlacement",
    "ManualMountingHole",
    "ManualParentLocalPlacement",
    "build_canvas_html",
    "build_canvas_init_script",
    "default_asset_url",
    "discover_leaves",
    "experiments_mount_url_for",
    "load_initial_layout",
    "load_manual_layout",
    "prerender_leaf_canvases",
    "render_leaf_canvas",
    "run_manual_compose",
    "save_manual_layout",
    "save_manual_layout_json",
    "static_js_path",
]
