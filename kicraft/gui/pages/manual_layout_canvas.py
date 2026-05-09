"""HTML/SVG/JS for the manual layout canvas.

Generates the markup + a self-contained vanilla-JS controller exposing
``window.manualLayoutCanvases[id]`` with ``getState()`` and ``reset()``.
"""

from __future__ import annotations

import json
from typing import Any


# Default sizing fallback for the canvas host; the host is actually
# styled with width: 100% and a viewport-relative height so it grows
# with the available browser space. These values exist only as the
# initial render-time fallback before clientWidth / clientHeight are
# known (and as the lower bound for layouts that miss those metrics).
CANVAS_WIDTH_PX = 1200
CANVAS_HEIGHT_PX = 800


def _build_canvas_config(
    leaves: list,
    initial: dict[str, Any],
    canvas_id: str,
) -> dict[str, Any]:
    leaf_payload = [
        {
            "instance_path": lf.instance_path,
            "sheet_name": lf.sheet_name,
            "width_mm": lf.width_mm,
            "height_mm": lf.height_mm,
            "color": lf.color,
            "render_url": lf.render_url,
            # Silk bbox in leaf-local coords. The canvas uses these
            # for the visible leaf preview so its size + rounded
            # corners match what F.Silkscreen actually shows on the
            # stamped board.
            "silk_min_x": lf.silk_min_x,
            "silk_min_y": lf.silk_min_y,
            "silk_max_x": lf.silk_max_x,
            "silk_max_y": lf.silk_max_y,
            "silk_corner_radius_mm": lf.silk_corner_radius_mm,
        }
        for lf in leaves
    ]
    return {
        "leaves": leaf_payload,
        "initial": initial,
        "canvas_id": canvas_id,
        "canvas_w_px": CANVAS_WIDTH_PX,
        "canvas_h_px": CANVAS_HEIGHT_PX,
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
    return f"""
<style>
  .ml-canvas-host {{
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
  }}
  .ml-canvas-host svg {{ width: 100%; height: 100%; display: block; }}
  .ml-leaf {{ cursor: grab; }}
  .ml-leaf.dragging {{ cursor: grabbing; }}
  .ml-leaf .ml-leaf-hit {{
    fill: transparent;
    stroke: none;
  }}
  .ml-leaf-img {{ pointer-events: none; }}
  .ml-rot-handle {{
    fill: #facc15;
    fill-opacity: 0;
    stroke: #facc15;
    stroke-width: 0.25;
    cursor: crosshair;
    transition: fill-opacity 0.12s ease;
  }}
  .ml-leaf:hover .ml-rot-handle,
  .ml-leaf.selected .ml-rot-handle {{ fill-opacity: 0.95; }}
  .ml-outline {{
    fill: none;
    stroke: #67e8f9;
    stroke-width: 0.6;
  }}
  .ml-edge {{ fill: #67e8f9; opacity: 0.55; cursor: ew-resize; }}
  .ml-edge.horizontal {{ cursor: ns-resize; }}
  .ml-edge:hover {{ opacity: 0.95; }}
  .ml-grid line {{ stroke: #1e293b; stroke-width: 0.15; }}
  .ml-grid line.major {{ stroke: #334155; stroke-width: 0.25; }}
  .ml-mhole {{
    fill: none;
    stroke: #f87171;
    stroke-width: 0.25;
    pointer-events: none;
  }}
  .ml-mhole-drill {{
    fill: #f87171;
    pointer-events: none;
  }}
  .ml-mhole-label {{
    fill: #fca5a5;
    font: 600 1.6px sans-serif;
    pointer-events: none;
    text-anchor: middle;
    dominant-baseline: middle;
  }}
  .ml-mhole-keepin {{
    fill: #f87171;
    fill-opacity: 0.10;
    stroke: #f87171;
    stroke-width: 0.15;
    stroke-dasharray: 0.5 0.3;
    pointer-events: none;
  }}
  .ml-leaf.overflow .ml-leaf-hit,
  .ml-leaf.overflow .ml-rot-handle {{
    stroke: #ef4444;
  }}
  .ml-leaf.overflow .ml-leaf-hit {{
    stroke-width: 0.5;
    stroke-dasharray: 0.6 0.4;
  }}
</style>
<div id="{canvas_id}-host" class="ml-canvas-host">
  <svg id="{canvas_id}" xmlns="http://www.w3.org/2000/svg"></svg>
</div>
"""


def build_canvas_init_script(
    leaves: list,
    initial: dict[str, Any],
    canvas_id: str,
) -> str:
    """Return JS source that bootstraps the canvas controller.

    Run this via ``ui.run_javascript(...)`` after the markup from
    ``build_canvas_html`` is in the DOM.
    """
    config = _build_canvas_config(leaves, initial, canvas_id)
    config_json = json.dumps(config)
    return _CANVAS_JS_TEMPLATE.format(config_json=config_json)


_CANVAS_JS_TEMPLATE = """
(function() {{
  const cfg = {config_json};
  const HOST_ID = cfg.canvas_id + '-host';

  // Each call to this IIFE bumps the canvas's version. The
  // refreshable body re-runs build_canvas_init_script on every
  // _manual_layout_body.refresh() so we end up with a stack of
  // IIFEs whose only conflict is the document-level keydown
  // listener (SVG-level listeners auto-clean when the SVG is
  // replaced). Listeners check this sentinel and bail if a newer
  // IIFE has registered, so only the latest version ever responds.
  window.__mlc_version = window.__mlc_version || {{}};
  window.__mlc_version[cfg.canvas_id] = (window.__mlc_version[cfg.canvas_id] || 0) + 1;
  const myVersion = window.__mlc_version[cfg.canvas_id];
  function isCurrent() {{
    return window.__mlc_version[cfg.canvas_id] === myVersion;
  }}
  const SVG_ID = cfg.canvas_id;
  const SELECTED_ID = cfg.canvas_id + '-selected';
  const COORDS_ID = cfg.canvas_id + '-coords';
  const OUTLINE_ID = cfg.canvas_id + '-outline';

  const HANDLE_THICK_MM = 1.4;
  const HANDLE_GRIP_MM = 0.8;
  const ROT_HANDLE_OFFSET_MM = 1.8;
  const ROT_HANDLE_R_MM = 0.9;
  const PADDING_X_MM = 4.0;
  // Top/bottom edge handles need more breathing room: at PADDING_MM=4
  // they end up flush against the canvas viewport, so a 1px overshoot
  // jumps the outline by several mm.
  const PADDING_Y_MM = 12.0;
  const SNAP_DEG = 90;

  function deepCopy(obj) {{ return JSON.parse(JSON.stringify(obj)); }}

  function makeState() {{
    return {{
      placements: deepCopy(cfg.initial.placements),
      board_outline: deepCopy(cfg.initial.board_outline),
      mounting_holes: deepCopy(cfg.initial.mounting_holes || []),
      selected: null,
    }};
  }}

  // Mounting holes are pinned to outline corners with a per-hole
  // inset; recompute their world positions whenever the outline
  // changes so dragging an edge handle keeps the holes glued to
  // their corners. Holes with corner=null keep whatever pos they
  // had (they're declared but not pinned).
  function recomputeMountingHoles() {{
    const out = state.board_outline;
    for (const h of state.mounting_holes) {{
      if (!h.corner) continue;
      const inset = Number(h.inset_mm) || 0;
      switch (h.corner) {{
        case 'top-left':
          h.pos = {{ x: out.min.x + inset, y: out.min.y + inset }}; break;
        case 'top-right':
          h.pos = {{ x: out.max.x - inset, y: out.min.y + inset }}; break;
        case 'bottom-left':
          h.pos = {{ x: out.min.x + inset, y: out.max.y - inset }}; break;
        case 'bottom-right':
          h.pos = {{ x: out.max.x - inset, y: out.max.y - inset }}; break;
      }}
    }}
  }}

  const initial = makeState();
  let state = makeState();

  const leafByPath = Object.fromEntries(cfg.leaves.map(l => [l.instance_path, l]));

  function viewBox() {{
    const out = state.board_outline;
    const w = out.max.x - out.min.x;
    const h = out.max.y - out.min.y;
    const vbW = Math.max(w + 2 * PADDING_X_MM, 30);
    const vbH = Math.max(h + 2 * PADDING_Y_MM, 30);
    const vbX = out.min.x - PADDING_X_MM;
    const vbY = out.min.y - PADDING_Y_MM;
    return {{ vbX, vbY, vbW, vbH }};
  }}

  function setSelected(ip) {{
    state.selected = ip;
    const sel = document.getElementById(SELECTED_ID);
    if (sel) {{
      if (ip) {{
        const lf = leafByPath[ip];
        sel.textContent = lf ? lf.sheet_name : ip;
      }} else {{
        sel.textContent = 'none';
      }}
    }}
    updateCoordsLabel();
    render();
  }}

  function updateCoordsLabel() {{
    const coords = document.getElementById(COORDS_ID);
    const outlineEl = document.getElementById(OUTLINE_ID);
    if (outlineEl) {{
      const w = (state.board_outline.max.x - state.board_outline.min.x);
      const h = (state.board_outline.max.y - state.board_outline.min.y);
      outlineEl.textContent = w.toFixed(1) + ' × ' + h.toFixed(1) + ' mm';
    }}
    if (!coords) return;
    if (!state.selected) {{ coords.textContent = '--'; return; }}
    const p = state.placements.find(p => p.instance_path === state.selected);
    if (!p) {{ coords.textContent = '--'; return; }}
    coords.textContent = 'x=' + p.origin.x.toFixed(2) + ', y=' + p.origin.y.toFixed(2)
      + ', rot=' + (p.rotation || 0).toFixed(0) + '°';
  }}

  function snapAngle(deg, shiftHeld) {{
    if (shiftHeld) return deg;
    const m = ((deg % 360) + 360) % 360;
    return Math.round(m / SNAP_DEG) * SNAP_DEG;
  }}

  // Compose-equivalent transform. KiCad / pcbnew rotates CLOCKWISE
  // (matching SetOrientationDegrees), so transform_loaded_artifact's
  // _transform_point uses:
  //   x' = x*cos + y*sin
  //   y' = -x*sin + y*cos
  // The canvas MUST use the same convention or the visual layout
  // diverges from the stamped output as soon as any leaf has a
  // non-zero rotation. Empirically: BATT at origin (88.4, -11.8)
  // rotation 90 with BT1 at leaf-local (46.8, 52.0) lands at
  // (140.4, -58.6) in the stamped board (CW); the pre-fix canvas
  // showed it at (36.4, 35.0) (CCW), explaining the "leaves wandered
  // way outside the outline in KiCad" report.
  //
  // Center-rotation pivots around the SILK BBOX center (which is what
  // the user sees), not the Edge.Cuts (0, 0)..(w, h) bbox. The two
  // can differ by several mm when the silk hugs the components more
  // tightly than the board outline.
  function silkCenterLocal(leaf) {{
    return {{
      x: (leaf.silk_min_x + leaf.silk_max_x) * 0.5,
      y: (leaf.silk_min_y + leaf.silk_max_y) * 0.5,
    }};
  }}

  function leafCenter(p, leaf) {{
    const r = (p.rotation || 0) * Math.PI / 180;
    const c = Math.cos(r), s = Math.sin(r);
    const sc = silkCenterLocal(leaf);
    return {{
      x: p.origin.x + c * sc.x + s * sc.y,
      y: p.origin.y - s * sc.x + c * sc.y,
    }};
  }}

  // Inverse for the same CW rotation: solve
  //   center = origin + R_CW(theta) * silk_center
  // for origin so the visual center stays put as the user rotates.
  function setRotationKeepCenter(p, leaf, newRotDeg) {{
    const center = leafCenter(p, leaf);
    const r = newRotDeg * Math.PI / 180;
    const c = Math.cos(r), s = Math.sin(r);
    const sc = silkCenterLocal(leaf);
    p.origin.x = center.x - (c * sc.x + s * sc.y);
    p.origin.y = center.y - (-s * sc.x + c * sc.y);
    p.rotation = newRotDeg;
  }}

  function render() {{
    const svg = document.getElementById(SVG_ID);
    if (!svg) return;
    const vb = viewBox();
    svg.setAttribute('viewBox',
      vb.vbX + ' ' + vb.vbY + ' ' + vb.vbW + ' ' + vb.vbH);
    svg.innerHTML = '';

    // Grid (every 5mm minor, 10mm major). Cover the FULL visible
    // area of the SVG element, not just the viewBox: with
    // preserveAspectRatio="xMidYMid meet" the SVG letterboxes the
    // viewBox, so the visible coordinate range is wider than the
    // viewBox in the longer axis. Without this, leaves placed outside
    // the outline (which still render correctly via the letterbox)
    // float over a gridless dark background.
    const grid = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    grid.setAttribute('class', 'ml-grid');
    const svgW = svg.clientWidth || cfg.canvas_w_px;
    const svgH = svg.clientHeight || cfg.canvas_h_px;
    const scale = Math.min(svgW / vb.vbW, svgH / vb.vbH);
    const visW = svgW / scale;
    const visH = svgH / scale;
    const visX = vb.vbX - (visW - vb.vbW) / 2;
    const visY = vb.vbY - (visH - vb.vbH) / 2;
    const x0 = Math.floor(visX / 5) * 5;
    const x1 = visX + visW;
    const y0 = Math.floor(visY / 5) * 5;
    const y1 = visY + visH;
    for (let x = x0; x <= x1; x += 5) {{
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', x); ln.setAttribute('x2', x);
      ln.setAttribute('y1', visY); ln.setAttribute('y2', y1);
      if (x % 10 === 0) ln.setAttribute('class', 'major');
      grid.appendChild(ln);
    }}
    for (let y = y0; y <= y1; y += 5) {{
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', visX); ln.setAttribute('x2', x1);
      ln.setAttribute('y1', y); ln.setAttribute('y2', y);
      if (y % 10 === 0) ln.setAttribute('class', 'major');
      grid.appendChild(ln);
    }}
    svg.appendChild(grid);

    // Outline rect
    const outline = state.board_outline;
    const outRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    outRect.setAttribute('class', 'ml-outline');
    outRect.setAttribute('x', outline.min.x);
    outRect.setAttribute('y', outline.min.y);
    outRect.setAttribute('width', outline.max.x - outline.min.x);
    outRect.setAttribute('height', outline.max.y - outline.min.y);
    svg.appendChild(outRect);

    // Edge handles
    const w = outline.max.x - outline.min.x;
    const h = outline.max.y - outline.min.y;
    addEdge(svg, 'left',   outline.min.x - HANDLE_GRIP_MM/2, outline.min.y, HANDLE_GRIP_MM, h);
    addEdge(svg, 'right',  outline.max.x - HANDLE_GRIP_MM/2, outline.min.y, HANDLE_GRIP_MM, h);
    addEdge(svg, 'top',    outline.min.x, outline.min.y - HANDLE_GRIP_MM/2, w, HANDLE_GRIP_MM, true);
    addEdge(svg, 'bottom', outline.min.x, outline.max.y - HANDLE_GRIP_MM/2, w, HANDLE_GRIP_MM, true);

    // Mounting holes (M3 default: 3.2 mm clearance, 6.4 mm pad / drill OD).
    // KEEPIN_RADIUS_MM matches the typical no-route zone the auto
    // pipeline draws around each mounting hole; keeps the user from
    // packing leaf silk / pads into a region that won't actually be
    // free of copper.
    const KEEPIN_RADIUS_MM = 3.0;
    recomputeMountingHoles();
    for (const hole of state.mounting_holes) {{
      const keepin = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      keepin.setAttribute('class', 'ml-mhole-keepin');
      keepin.setAttribute('x', hole.pos.x - KEEPIN_RADIUS_MM);
      keepin.setAttribute('y', hole.pos.y - KEEPIN_RADIUS_MM);
      keepin.setAttribute('width', KEEPIN_RADIUS_MM * 2);
      keepin.setAttribute('height', KEEPIN_RADIUS_MM * 2);
      svg.appendChild(keepin);
      const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      ring.setAttribute('class', 'ml-mhole');
      ring.setAttribute('cx', hole.pos.x);
      ring.setAttribute('cy', hole.pos.y);
      ring.setAttribute('r', 3.2);
      svg.appendChild(ring);
      const drill = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      drill.setAttribute('class', 'ml-mhole-drill');
      drill.setAttribute('cx', hole.pos.x);
      drill.setAttribute('cy', hole.pos.y);
      drill.setAttribute('r', 0.5);
      svg.appendChild(drill);
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      label.setAttribute('class', 'ml-mhole-label');
      label.setAttribute('x', hole.pos.x);
      label.setAttribute('y', hole.pos.y - 4.0);
      label.textContent = 'H' + (hole.index + 1);
      svg.appendChild(label);
    }}

    // Leaves
    const out = state.board_outline;
    for (const p of state.placements) {{
      const leaf = leafByPath[p.instance_path];
      if (!leaf) continue;
      // Visible leaf rect uses the SILK bbox so the canvas preview
      // matches what the stamped board renders (rounded corners,
      // tighter hug to the components). Edge.Cuts (width_mm/height_mm)
      // is still useful for the leaf-local origin -- placement origin
      // continues to map to leaf-local (0, 0) per composer convention.
      const sx0 = leaf.silk_min_x, sy0 = leaf.silk_min_y;
      const sx1 = leaf.silk_max_x, sy1 = leaf.silk_max_y;
      const sw = Math.max(0, sx1 - sx0), sh = Math.max(0, sy1 - sy0);
      // Rotated-bbox overflow check uses the silk corners (the
      // visible leaf shape), so the red overflow flag fires exactly
      // when what the user sees crosses the outline.
      const r = (p.rotation || 0) * Math.PI / 180;
      const rc = Math.cos(r), rs = Math.sin(r);
      function corner(lx, ly) {{
        return {{
          x: p.origin.x + rc * lx + rs * ly,
          y: p.origin.y - rs * lx + rc * ly,
        }};
      }}
      const corners = [
        corner(sx0, sy0),
        corner(sx1, sy0),
        corner(sx1, sy1),
        corner(sx0, sy1),
      ];
      const overflow = corners.some(c =>
        c.x < out.min.x - 0.01 || c.x > out.max.x + 0.01 ||
        c.y < out.min.y - 0.01 || c.y > out.max.y + 0.01
      );
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute(
        'class',
        'ml-leaf'
        + (state.selected === p.instance_path ? ' selected' : '')
        + (overflow ? ' overflow' : ''),
      );
      g.setAttribute('data-instance-path', p.instance_path);
      // SVG rotate() is CCW; KiCad rotation is CW. Negate so the
      // canvas visual matches the stamped output.
      g.setAttribute('transform',
        'translate(' + p.origin.x + ',' + p.origin.y + ') rotate(' + (-(p.rotation || 0)) + ')');

      // Routed-leaf PNG positioned and sized to the silk bbox.
      // preserveAspectRatio="xMidYMid meet" preserves the native
      // render aspect -- distorted PNGs (the previous "none" mode)
      // looked like the leaf had been squashed onto a different
      // shape. Letterboxing inside the silk rect is fine because the
      // silk poly itself defines the visible boundary; the slight
      // gap between bbox edge and image edge reads as silk margin.
      if (leaf.render_url) {{
        const img = document.createElementNS('http://www.w3.org/2000/svg', 'image');
        img.setAttribute('class', 'ml-leaf-img');
        img.setAttribute('href', leaf.render_url);
        img.setAttribute('x', sx0);
        img.setAttribute('y', sy0);
        img.setAttribute('width', sw);
        img.setAttribute('height', sh);
        img.setAttribute('preserveAspectRatio', 'xMidYMid meet');
        g.appendChild(img);
      }}

      // Hit / selection target = silk bbox, rounded corners matching
      // the leaf solver's _silkscreen_for_label output.
      const hit = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      hit.setAttribute('class', 'ml-leaf-hit');
      hit.setAttribute('x', sx0);
      hit.setAttribute('y', sy0);
      hit.setAttribute('width', sw);
      hit.setAttribute('height', sh);
      hit.setAttribute('rx', leaf.silk_corner_radius_mm || 1.0);
      hit.setAttribute('ry', leaf.silk_corner_radius_mm || 1.0);
      hit.setAttribute('fill', 'transparent');
      hit.setAttribute('stroke', 'none');
      g.appendChild(hit);

      // Rotation handle floats just outside the silk bbox top-right
      // (instead of the Edge.Cuts top-right) so it tracks the
      // visible leaf shape under rotation.
      const rot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      rot.setAttribute('class', 'ml-rot-handle');
      rot.setAttribute('cx', sx1 + ROT_HANDLE_OFFSET_MM);
      rot.setAttribute('cy', sy0 - ROT_HANDLE_OFFSET_MM);
      rot.setAttribute('r', ROT_HANDLE_R_MM);
      rot.setAttribute('data-role', 'rotate');
      g.appendChild(rot);

      svg.appendChild(g);
    }}

    bindLeafEvents(svg);
    bindEdgeEvents(svg);
    updateCoordsLabel();
  }}

  function addEdge(svg, side, x, y, w, h, horizontal=false) {{
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    r.setAttribute('class', 'ml-edge' + (horizontal ? ' horizontal' : ''));
    r.setAttribute('data-edge', side);
    r.setAttribute('x', x); r.setAttribute('y', y);
    r.setAttribute('width', w); r.setAttribute('height', h);
    svg.appendChild(r);
  }}

  function svgToWorld(svg, evt) {{
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return {{ x: 0, y: 0 }};
    const inv = ctm.inverse();
    const w = pt.matrixTransform(inv);
    return {{ x: w.x, y: w.y }};
  }}

  function bindLeafEvents(svg) {{
    svg.querySelectorAll('.ml-leaf').forEach(g => {{
      const ip = g.getAttribute('data-instance-path');
      g.addEventListener('mousedown', (e) => {{
        if (e.target && e.target.getAttribute('data-role') === 'rotate') {{
          startRotate(svg, ip, e);
        }} else {{
          startDrag(svg, ip, e);
        }}
        e.preventDefault();
      }});
    }});
    svg.addEventListener('contextmenu', (e) => {{
      const target = e.target.closest('.ml-leaf');
      if (target) {{
        e.preventDefault();
        const ip = target.getAttribute('data-instance-path');
        const p = state.placements.find(x => x.instance_path === ip);
        if (p) {{
          const leaf = leafByPath[ip];
          setRotationKeepCenter(p, leaf, snapAngle((p.rotation || 0) + SNAP_DEG, false));
          setSelected(ip);
        }}
      }}
    }});
    document.addEventListener('keydown', (e) => {{
      if (!isCurrent()) return;  // stale IIFE from a prior refresh
      if (!state.selected) return;
      if (e.key === 'r' || e.key === 'R') {{
        const p = state.placements.find(x => x.instance_path === state.selected);
        if (p) {{
          const leaf = leafByPath[state.selected];
          setRotationKeepCenter(p, leaf, snapAngle((p.rotation || 0) + SNAP_DEG, false));
          render();
        }}
      }}
    }});
  }}

  function startDrag(svg, ip, evt) {{
    setSelected(ip);
    const p = state.placements.find(x => x.instance_path === ip);
    if (!p) return;
    const start = svgToWorld(svg, evt);
    const orig = {{ x: p.origin.x, y: p.origin.y }};
    const move = (e) => {{
      const cur = svgToWorld(svg, e);
      p.origin.x = orig.x + (cur.x - start.x);
      p.origin.y = orig.y + (cur.y - start.y);
      render();
    }};
    const up = () => {{
      document.removeEventListener('mousemove', move);
      document.removeEventListener('mouseup', up);
    }};
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  }}

  function startRotate(svg, ip, evt) {{
    setSelected(ip);
    const p = state.placements.find(x => x.instance_path === ip);
    if (!p) return;
    const leaf = leafByPath[ip];
    // Drag the rotation handle relative to the leaf VISUAL CENTER so
    // the leaf appears to spin in place. setRotationKeepCenter then
    // updates origin to compensate, keeping JSON output composer-
    // compatible (rotation pivots around leaf-local 0,0 there).
    const center = leafCenter(p, leaf);
    const startW = svgToWorld(svg, evt);
    const startAngle = Math.atan2(startW.y - center.y, startW.x - center.x) * 180 / Math.PI;
    const baseRot = p.rotation || 0;
    const move = (e) => {{
      const cur = svgToWorld(svg, e);
      const ang = Math.atan2(cur.y - center.y, cur.x - center.x) * 180 / Math.PI;
      setRotationKeepCenter(p, leaf, baseRot + (ang - startAngle));
      render();
    }};
    const up = (e) => {{
      setRotationKeepCenter(p, leaf, snapAngle(p.rotation, e.shiftKey));
      render();
      document.removeEventListener('mousemove', move);
      document.removeEventListener('mouseup', up);
    }};
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  }}

  function bindEdgeEvents(svg) {{
    svg.querySelectorAll('.ml-edge').forEach(el => {{
      const side = el.getAttribute('data-edge');
      el.addEventListener('mousedown', (e) => {{
        // Capture the screen-to-viewBox scale ONCE at mousedown.
        // Edge drags grow/shrink the outline, which changes the
        // viewBox, which shifts the live CTM. If we reused
        // svgToWorld() during the move, the delta would be measured
        // against a moving reference and the drag would feel jumpy
        // (especially on the Y axis, which is the canvas's scale-
        // limiting dimension at typical aspect ratios).
        const ctm = svg.getScreenCTM();
        if (!ctm) return;
        const ctmInv = ctm.inverse();
        const mmPerPxX = ctmInv.a;
        const mmPerPxY = ctmInv.d;
        const startClientX = e.clientX;
        const startClientY = e.clientY;
        const orig = deepCopy(state.board_outline);
        const minSize = 10;
        const move = (ev) => {{
          const dx = (ev.clientX - startClientX) * mmPerPxX;
          const dy = (ev.clientY - startClientY) * mmPerPxY;
          const out = state.board_outline;
          if (side === 'left') {{
            out.min.x = Math.min(orig.min.x + dx, orig.max.x - minSize);
          }} else if (side === 'right') {{
            out.max.x = Math.max(orig.max.x + dx, orig.min.x + minSize);
          }} else if (side === 'top') {{
            out.min.y = Math.min(orig.min.y + dy, orig.max.y - minSize);
          }} else if (side === 'bottom') {{
            out.max.y = Math.max(orig.max.y + dy, orig.min.y + minSize);
          }}
          render();
        }};
        const up = () => {{
          document.removeEventListener('mousemove', move);
          document.removeEventListener('mouseup', up);
        }};
        document.addEventListener('mousemove', move);
        document.addEventListener('mouseup', up);
        e.preventDefault();
      }});
    }});
  }}

  // --- Public API exposed to Python ---
  window.manualLayoutCanvases = window.manualLayoutCanvases || {{}};
  window.manualLayoutCanvases[cfg.canvas_id] = {{
    getState: function() {{
      // Quantize to 0.001 mm and 0.1° to keep JSON tidy
      const placements = state.placements.map(p => ({{
        instance_path: p.instance_path,
        origin: {{
          x: Math.round(p.origin.x * 1000) / 1000,
          y: Math.round(p.origin.y * 1000) / 1000,
        }},
        rotation: Math.round(((p.rotation || 0) * 10)) / 10,
      }}));
      const out = state.board_outline;
      recomputeMountingHoles();
      const mounting_holes = state.mounting_holes.map(h => ({{
        index: h.index,
        corner: h.corner,
        inset_mm: Math.round(h.inset_mm * 100) / 100,
        pos: {{
          x: Math.round(h.pos.x * 1000) / 1000,
          y: Math.round(h.pos.y * 1000) / 1000,
        }},
      }}));
      return {{
        placements: placements,
        board_outline: {{
          min: {{
            x: Math.round(out.min.x * 1000) / 1000,
            y: Math.round(out.min.y * 1000) / 1000,
          }},
          max: {{
            x: Math.round(out.max.x * 1000) / 1000,
            y: Math.round(out.max.y * 1000) / 1000,
          }},
        }},
        mounting_holes: mounting_holes,
      }};
    }},
    reset: function() {{
      state = makeState();
      render();
    }},
    getOutlineSize: function() {{
      const out = state.board_outline;
      return {{
        width: Math.round((out.max.x - out.min.x) * 1000) / 1000,
        height: Math.round((out.max.y - out.min.y) * 1000) / 1000,
      }};
    }},
    setOutlineSize: function(width, height) {{
      const out = state.board_outline;
      const w = Math.max(10, Number(width) || 0);
      const h = Math.max(10, Number(height) || 0);
      // Anchor at the existing min corner so leaves don't get
      // shoved when the user only adjusted width or only height.
      out.max.x = out.min.x + w;
      out.max.y = out.min.y + h;
      render();
    }},
    setMountingHoles: function(holes) {{
      // Replace the entire mounting-holes list. Caller (Python side)
      // owns count + per-hole corner/inset choices; the canvas just
      // visualises and re-pegs positions to corners on every render.
      state.mounting_holes = (holes || []).map((h, i) => ({{
        index: typeof h.index === 'number' ? h.index : i,
        corner: h.corner || null,
        inset_mm: Number(h.inset_mm) || 5.0,
        pos: h.pos ? {{ x: Number(h.pos.x) || 0, y: Number(h.pos.y) || 0 }}
                   : {{ x: state.board_outline.min.x, y: state.board_outline.min.y }},
      }}));
      render();
    }},
    getMountingHoles: function() {{
      recomputeMountingHoles();
      return state.mounting_holes.map(h => ({{
        index: h.index,
        corner: h.corner,
        inset_mm: Math.round(h.inset_mm * 100) / 100,
        pos: {{
          x: Math.round(h.pos.x * 1000) / 1000,
          y: Math.round(h.pos.y * 1000) / 1000,
        }},
      }}));
    }},
  }};

  // The Manual Layout tab is mounted lazily by Quasar -- on initial
  // page load the SVG element is not yet in the DOM. Poll until it
  // appears, then render once. After that the tab is kept alive and
  // subsequent re-activations reuse the rendered DOM.
  function tryInit(remainingTries) {{
    if (!isCurrent()) return;  // a newer IIFE took over while we waited
    const svg = document.getElementById(SVG_ID);
    if (svg) {{
      render();
      return;
    }}
    if (remainingTries <= 0) {{
      console.warn('manual layout canvas: SVG #' + SVG_ID + ' never mounted');
      return;
    }}
    setTimeout(function() {{ tryInit(remainingTries - 1); }}, 200);
  }}
  tryInit(150);  // 30 seconds total: long enough for slow first-tab-click.
}})();
"""
