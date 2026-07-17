"""HTML + JS bootstrap for the manual layout canvas.

The interactive controller itself lives in
``kicraft/layout_editor/static/layout_canvas.js`` (a real, lintable
JS file served as a static asset). This module builds the host-side
markup (CSS + SVG container) and the small bootstrap script that
loads the asset once per page and calls
``window.kicraftInitLayoutCanvas(cfg)`` for each canvas init.

Hosts must serve the package's ``static/`` directory and pass the
matching ``asset_url`` (offline GUI mounts it at ``/layout-static``,
which is also the default here).
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


# Default sizing fallback for the canvas host; the host is actually
# styled with width: 100% and a viewport-relative height so it grows
# with the available browser space. These values exist only as the
# initial render-time fallback before clientWidth / clientHeight are
# known (and as the lower bound for layouts that miss those metrics).
CANVAS_WIDTH_PX = 1200
CANVAS_HEIGHT_PX = 800

STATIC_DIR = Path(__file__).parent / "static"
DEFAULT_ASSET_MOUNT = "/layout-static"


def static_js_path() -> Path:
    return STATIC_DIR / "layout_canvas.js"


@lru_cache(maxsize=1)
def _asset_version() -> int:
    """Cache-buster for the controller asset; process-stable mtime so a
    deploy (new file) busts browser caches while repeated page renders
    reuse the same URL."""
    try:
        return int(static_js_path().stat().st_mtime)
    except OSError:
        return 0


def default_asset_url(mount: str = DEFAULT_ASSET_MOUNT) -> str:
    return f"{mount}/layout_canvas.js?v={_asset_version()}"


def _build_canvas_config(
    leaves: list,
    initial: dict[str, Any],
    canvas_id: str,
    asset_url: str,
    ratsnest: list | None = None,
) -> dict[str, Any]:
    leaf_payload = [
        {
            "instance_path": lf.instance_path,
            "sheet_name": lf.sheet_name,
            "width_mm": lf.width_mm,
            "height_mm": lf.height_mm,
            "color": lf.color,
            "render_url": lf.render_url,
            # Cosmetic only -- leaf solver's silk-poly bbox.
            "silk_min_x": lf.silk_min_x,
            "silk_min_y": lf.silk_min_y,
            "silk_max_x": lf.silk_max_x,
            "silk_max_y": lf.silk_max_y,
            # Where the <image> element draws (SVG viewBox). Includes
            # footprint silk-text labels that hang past the board edge,
            # so this is BIGGER than the leaf's physical board. Used
            # only for <image> x/y/width/height; not for snap/overflow.
            "image_x_mm": lf.image_x_mm,
            "image_y_mm": lf.image_y_mm,
            "image_width_mm": lf.image_width_mm,
            "image_height_mm": lf.image_height_mm,
            # Edge.Cuts AABB -- single source of truth for physical
            # extent. Hit, drag/snap, overflow against the parent
            # outline, and inter-leaf overlap all run against this.
            # What gets stamped to the parent occupies THIS rectangle,
            # not the image rectangle.
            "edge_min_x": lf.edge_min_x,
            "edge_min_y": lf.edge_min_y,
            "edge_max_x": lf.edge_max_x,
            "edge_max_y": lf.edge_max_y,
        }
        for lf in leaves
    ]
    return {
        "leaves": leaf_payload,
        "initial": initial,
        "canvas_id": canvas_id,
        "canvas_w_px": CANVAS_WIDTH_PX,
        "canvas_h_px": CANVAS_HEIGHT_PX,
        "asset_url": asset_url,
        # Cross-leaf net links (kicraft.layout_editor.ratsnest); anchors
        # in each leaf's canvas-local frame. The controller transforms
        # them by the live placements and draws connection lines.
        "ratsnest": ratsnest or [],
    }


def build_canvas_html(
    leaves: list,  # list[LeafInfo] -- typed in caller; avoid import cycle
    initial: dict[str, Any],
    canvas_id: str,
) -> str:
    """Return the HTML markup for the canvas (CSS + SVG container).

    Use ``build_canvas_init_script(...)`` for the matching JS bootstrap;
    NiceGUI 3.x rejects ``<script>`` tags inside ``ui.html()``.
    """
    return _CANVAS_HTML_TEMPLATE.replace("__CANVAS_ID__", canvas_id)


def build_canvas_init_script(
    leaves: list,
    initial: dict[str, Any],
    canvas_id: str,
    *,
    asset_url: str | None = None,
    ratsnest: list | None = None,
) -> str:
    """Return JS source that bootstraps the canvas controller.

    Run this via ``ui.run_javascript(...)`` after the markup from
    ``build_canvas_html`` is in the DOM. Loads the shared controller
    asset on first use (subsequent inits reuse it; the controller's
    own version sentinel supersedes stale instances).
    """
    if asset_url is None:
        asset_url = default_asset_url()
    config = _build_canvas_config(leaves, initial, canvas_id, asset_url, ratsnest)
    config_json = json.dumps(config)
    return _BOOTSTRAP_TEMPLATE.format(config_json=config_json)


_BOOTSTRAP_TEMPLATE = """
(function() {{
  const cfg = {config_json};
  function boot() {{ window.kicraftInitLayoutCanvas(cfg); }}
  if (window.kicraftInitLayoutCanvas) {{ boot(); return; }}
  let tag = document.querySelector('script[data-kicraft-layout-canvas]');
  if (!tag) {{
    tag = document.createElement('script');
    tag.src = cfg.asset_url;
    tag.setAttribute('data-kicraft-layout-canvas', '1');
    document.head.appendChild(tag);
  }}
  tag.addEventListener('load', boot);
}})();
"""


_CANVAS_HTML_TEMPLATE = """
<style>
  .ml-canvas-host {
    position: relative;
    width: 100%;
    /* 180 px reserves room for: header (40), tab strip (50),
       outline inputs (50), action buttons (50), gap padding (~20).
       Anything left over goes to the canvas. min-height keeps the
       canvas usable on tall narrow viewports. */
    height: calc(100vh - 180px);
    min-height: 600px;
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 6px;
    user-select: none;
    overflow: hidden;
  }
  .ml-canvas-host svg { width: 100%; height: 100%; display: block; }
  .ml-leaf { cursor: grab; }
  .ml-leaf.dragging { cursor: grabbing; }
  .ml-leaf .ml-leaf-hit {
    fill: transparent;
    stroke: none;
  }
  .ml-leaf-img { pointer-events: none; }
  /* Sharp-cornered amber outline tracing the leaf solver's silk-poly
     bbox -- what gets stamped on the parent's F.Silkscreen layer. The
     rounded silk poly inside the PNG should sit exactly inside these
     straight edges; if not, the placement (or the solver's poly) is
     off. This is visual only; drag / snap / overflow run against the
     full content extent (image_*_mm). */
  .ml-leaf-silk-bbox {
    /* Hidden by default. The yellow rounded silk poly baked into the
       PNG is already the visible leaf boundary; an additional amber
       overlay just duplicated it (with different corners) and the user
       reported the resulting double-line as visual noise. Flip display
       to "block" for debugging if you want to verify that the silk-poly
       AABB matches the drawn poly. */
    display: none;
    fill: none;
    stroke: #fbbf24;
    stroke-width: 0.12;
    stroke-opacity: 0.55;
    pointer-events: none;
  }
  .ml-leaf-silk-bbox.snap-active {
    /* Kept for code/debug parity; snap feedback now uses per-edge
       highlights via .ml-snap-edge instead of the whole bbox. */
    stroke: #22d3ee;
    stroke-opacity: 1;
  }
  /* Highlight segments drawn over the specific constrained edges of
     the dragged leaf and the leaf (or outline) it's snapping against.
     Drawn on top of everything during a snap so the constraint is
     obvious without the whole leaf lighting up. */
  .ml-snap-edge {
    stroke: #22d3ee;
    stroke-width: 0.45;
    stroke-linecap: round;
    pointer-events: none;
  }
  .ml-rot-handle {
    fill: #facc15;
    fill-opacity: 0;
    stroke: #facc15;
    stroke-width: 0.25;
    stroke-opacity: 0;
    pointer-events: none;
    cursor: crosshair;
    transition: fill-opacity 0.12s ease, stroke-opacity 0.12s ease;
  }
  .ml-leaf.selected .ml-rot-handle {
    fill-opacity: 0.95;
    stroke-opacity: 1;
    pointer-events: auto;
  }
  .ml-outline {
    /* Pcbnew-style black fill inside the parent's Edge.Cuts so the PCB
       area is visibly distinct from the navy canvas. Leaves render on
       top with transparent backgrounds, so their PCB content composites
       over this black fill (and over any overlapping leaf content) the
       same way it would in pcbnew. */
    fill: #000000;
    stroke: #67e8f9;
    stroke-width: 0.6;
  }
  .ml-edge { fill: #67e8f9; opacity: 0.55; cursor: ew-resize; }
  .ml-edge.horizontal { cursor: ns-resize; }
  .ml-edge:hover { opacity: 0.95; }
  .ml-grid line { stroke: #1e293b; stroke-width: 0.15; }
  .ml-grid line.major { stroke: #334155; stroke-width: 0.25; }
  .ml-mhole {
    fill: none;
    stroke: #f87171;
    stroke-width: 0.25;
    pointer-events: none;
  }
  .ml-mhole-drill {
    fill: #f87171;
    pointer-events: none;
  }
  .ml-mhole-label {
    fill: #fca5a5;
    font: 600 1.6px sans-serif;
    pointer-events: none;
    text-anchor: middle;
    dominant-baseline: middle;
  }
  .ml-mhole-keepin {
    fill: #f87171;
    fill-opacity: 0.10;
    stroke: #f87171;
    stroke-width: 0.15;
    stroke-dasharray: 0.5 0.3;
    pointer-events: none;
  }
  /* Overflow indicator: turn the rotation handle red so the user can
     spot a leaf placed outside the outline without overlaying any
     extra outline on top of the leaf's baked silkscreen. */
  .ml-leaf.overflow .ml-rot-handle {
    stroke: #ef4444;
  }
  /* Overlap indicator: a red Edge.Cuts AABB outline draws on top of
     any leaf whose physical board (the rectangle stamped on the
     parent) intersects another leaf's. Save is still allowed (per
     workflow choice) but the user sees exactly which leaves are
     colliding before they hit save. The .ml-leaf-overlap rect is
     rendered AFTER the silk-bbox and hit rects so it sits on top. */
  .ml-leaf-overlap {
    fill: rgba(239, 68, 68, 0.10);
    stroke: #ef4444;
    stroke-width: 0.35;
    stroke-opacity: 0.95;
    pointer-events: none;
  }
  /* Ratsnest: dashed cross-leaf net lines (MST per net), updated live
     during drag. Nets touching the selected leaf highlight so "what
     does this block talk to" is one click away. */
  .ml-ratsnest-line {
    stroke: #94a3b8;
    stroke-width: 0.18;
    stroke-opacity: 0.55;
    stroke-dasharray: 0.9 0.5;
    pointer-events: none;
  }
  .ml-ratsnest-line.hot {
    stroke: #38bdf8;
    stroke-width: 0.32;
    stroke-opacity: 0.95;
  }
</style>
<div id="__CANVAS_ID__-host" class="ml-canvas-host">
  <svg id="__CANVAS_ID__" xmlns="http://www.w3.org/2000/svg"></svg>
</div>
"""
