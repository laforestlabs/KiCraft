"""HTML/SVG/JS for the manual layout canvas.

Generates the markup + a self-contained vanilla-JS controller exposing
``window.manualLayoutCanvases[id]`` with ``getState()`` and ``reset()``.
"""

from __future__ import annotations

import json
from typing import Any


CANVAS_WIDTH_PX = 900
CANVAS_HEIGHT_PX = 640


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
    width: {CANVAS_WIDTH_PX}px;
    max-width: 100%;
    height: {CANVAS_HEIGHT_PX}px;
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
  .ml-leaf.selected .ml-leaf-hit {{
    stroke: #facc15;
    stroke-width: 0.4;
    stroke-dasharray: 0.6 0.4;
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
  const SVG_ID = cfg.canvas_id;
  const SELECTED_ID = cfg.canvas_id + '-selected';
  const COORDS_ID = cfg.canvas_id + '-coords';
  const OUTLINE_ID = cfg.canvas_id + '-outline';

  const HANDLE_THICK_MM = 1.4;
  const HANDLE_GRIP_MM = 0.8;
  const ROT_HANDLE_OFFSET_MM = 1.8;
  const ROT_HANDLE_R_MM = 0.9;
  const PADDING_MM = 4.0;
  const SNAP_DEG = 90;

  function deepCopy(obj) {{ return JSON.parse(JSON.stringify(obj)); }}

  function makeState() {{
    return {{
      placements: deepCopy(cfg.initial.placements),
      board_outline: deepCopy(cfg.initial.board_outline),
      selected: null,
    }};
  }}

  const initial = makeState();
  let state = makeState();

  const leafByPath = Object.fromEntries(cfg.leaves.map(l => [l.instance_path, l]));

  function viewBox() {{
    const out = state.board_outline;
    const w = out.max.x - out.min.x;
    const h = out.max.y - out.min.y;
    const vbW = Math.max(w + 2 * PADDING_MM, 30);
    const vbH = Math.max(h + 2 * PADDING_MM, 30);
    const vbX = out.min.x - PADDING_MM;
    const vbY = out.min.y - PADDING_MM;
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

  function render() {{
    const svg = document.getElementById(SVG_ID);
    if (!svg) return;
    const vb = viewBox();
    svg.setAttribute('viewBox',
      vb.vbX + ' ' + vb.vbY + ' ' + vb.vbW + ' ' + vb.vbH);
    svg.innerHTML = '';

    // Grid (every 5mm minor, 10mm major)
    const grid = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    grid.setAttribute('class', 'ml-grid');
    const x0 = Math.floor(vb.vbX / 5) * 5;
    const y0 = Math.floor(vb.vbY / 5) * 5;
    for (let x = x0; x <= vb.vbX + vb.vbW; x += 5) {{
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', x); ln.setAttribute('x2', x);
      ln.setAttribute('y1', vb.vbY); ln.setAttribute('y2', vb.vbY + vb.vbH);
      if (x % 10 === 0) ln.setAttribute('class', 'major');
      grid.appendChild(ln);
    }}
    for (let y = y0; y <= vb.vbY + vb.vbH; y += 5) {{
      const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      ln.setAttribute('x1', vb.vbX); ln.setAttribute('x2', vb.vbX + vb.vbW);
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

    // Leaves
    for (const p of state.placements) {{
      const leaf = leafByPath[p.instance_path];
      if (!leaf) continue;
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('class', 'ml-leaf' + (state.selected === p.instance_path ? ' selected' : ''));
      g.setAttribute('data-instance-path', p.instance_path);
      g.setAttribute('transform',
        'translate(' + p.origin.x + ',' + p.origin.y + ') rotate(' + (p.rotation || 0) + ')');

      // Routed-leaf PNG (pads + traces + silkscreen including the
      // leaf outline). Stretched to leaf bbox so adjacent leaves
      // touch visually; the silkscreen IS the visible outline so no
      // additional rectangle or overlay label is drawn.
      if (leaf.render_url) {{
        const img = document.createElementNS('http://www.w3.org/2000/svg', 'image');
        img.setAttribute('class', 'ml-leaf-img');
        img.setAttribute('href', leaf.render_url);
        img.setAttribute('x', 0);
        img.setAttribute('y', 0);
        img.setAttribute('width', leaf.width_mm);
        img.setAttribute('height', leaf.height_mm);
        img.setAttribute('preserveAspectRatio', 'none');
        g.appendChild(img);
      }}

      // Invisible hit target so clicks anywhere over the leaf bbox
      // start a drag, even where the PNG is dark / mostly empty. The
      // .selected state turns this into a thin amber dashed outline.
      const hit = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      hit.setAttribute('class', 'ml-leaf-hit');
      hit.setAttribute('x', 0);
      hit.setAttribute('y', 0);
      hit.setAttribute('width', leaf.width_mm);
      hit.setAttribute('height', leaf.height_mm);
      g.appendChild(hit);

      // Rotation handle: a small disc at top-right, offset outside the rect
      const rot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      rot.setAttribute('class', 'ml-rot-handle');
      rot.setAttribute('cx', leaf.width_mm + ROT_HANDLE_OFFSET_MM);
      rot.setAttribute('cy', -ROT_HANDLE_OFFSET_MM);
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
          p.rotation = snapAngle((p.rotation || 0) + SNAP_DEG, false);
          setSelected(ip);
        }}
      }}
    }});
    document.addEventListener('keydown', (e) => {{
      if (!state.selected) return;
      if (e.key === 'r' || e.key === 'R') {{
        const p = state.placements.find(x => x.instance_path === state.selected);
        if (p) {{
          p.rotation = snapAngle((p.rotation || 0) + SNAP_DEG, false);
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
    const cx = p.origin.x;
    const cy = p.origin.y;
    const startW = svgToWorld(svg, evt);
    const startAngle = Math.atan2(startW.y - cy, startW.x - cx) * 180 / Math.PI;
    const baseRot = p.rotation || 0;
    const move = (e) => {{
      const cur = svgToWorld(svg, e);
      const ang = Math.atan2(cur.y - cy, cur.x - cx) * 180 / Math.PI;
      const delta = ang - startAngle;
      p.rotation = baseRot + delta;
      render();
    }};
    const up = (e) => {{
      p.rotation = snapAngle(p.rotation, e.shiftKey);
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
        const start = svgToWorld(svg, e);
        const orig = deepCopy(state.board_outline);
        const minSize = 10;
        const move = (ev) => {{
          const cur = svgToWorld(svg, ev);
          const dx = cur.x - start.x;
          const dy = cur.y - start.y;
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
      }};
    }},
    reset: function() {{
      state = makeState();
      render();
    }},
  }};

  // The Manual Layout tab is mounted lazily by Quasar -- on initial
  // page load the SVG element is not yet in the DOM. Poll until it
  // appears, then render once. After that the tab is kept alive and
  // subsequent re-activations reuse the rendered DOM.
  function tryInit(remainingTries) {{
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
