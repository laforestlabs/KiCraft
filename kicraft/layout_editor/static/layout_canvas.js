// Manual-layout canvas controller (shared static asset).
// Extracted verbatim from the former Python f-string template in
// kicraft/gui/pages/manual_layout_canvas.py; the host bootstrap
// (kicraft/layout_editor/canvas.py) loads this file once and calls
// window.kicraftInitLayoutCanvas(cfg) per canvas init.

// --- Pure outline-shape geometry -------------------------------------------
// MUST mirror kicraft/layout_editor/outline.py exactly: same shapes, same
// polyline sampling, same containment and mounting-hole math. Exported on
// window.kicraftLayoutGeometry so the cross-language agreement test can
// drive it from node against the Python implementation.
window.kicraftLayoutGeometry = (function() {
  const MAX_SAGITTA_MM = 0.02;
  const CIRCLE_MIN_SEGMENTS = 32;
  const CIRCLE_MAX_SEGMENTS = 128;
  const ROUNDED_RECT_N_ARC = 8;
  const SQRT2 = Math.sqrt(2.0);

  // shapeSpec: { shape, corner_radius_mm, chamfer_mm }; min/max: { x, y }.
  function clampedParam(spec, min, max) {
    const half = Math.min(max.x - min.x, max.y - min.y) / 2.0;
    if (spec.shape === 'rounded_rect') return Math.min(spec.corner_radius_mm || 0, half);
    if (spec.shape === 'chamfered_rect') return Math.min(spec.chamfer_mm || 0, half);
    if (spec.shape === 'circle') return half;
    return 0.0;
  }

  function circleSegmentCount(r) {
    if (r <= MAX_SAGITTA_MM) return CIRCLE_MIN_SEGMENTS;
    const n = Math.ceil(Math.PI / Math.acos(1.0 - MAX_SAGITTA_MM / r));
    return Math.max(CIRCLE_MIN_SEGMENTS, Math.min(CIRCLE_MAX_SEGMENTS, n));
  }

  function outlinePolyline(spec, min, max) {
    const x0 = min.x, y0 = min.y, x1 = max.x, y1 = max.y;
    if (!spec || spec.shape === 'rect') {
      return [{x: x0, y: y0}, {x: x1, y: y0}, {x: x1, y: y1}, {x: x0, y: y1}];
    }
    if (spec.shape === 'rounded_rect') {
      const r = clampedParam(spec, min, max);
      const points = [];
      const corners = [
        [x0 + r, y0 + r, Math.PI, Math.PI / 2],
        [x1 - r, y0 + r, Math.PI / 2, 0],
        [x1 - r, y1 - r, 0, -Math.PI / 2],
        [x0 + r, y1 - r, -Math.PI / 2, -Math.PI],
      ];
      for (const [cx, cy, aStart, aEnd] of corners) {
        for (let i = 0; i < ROUNDED_RECT_N_ARC; i++) {
          const t = aStart + (aEnd - aStart) * i / (ROUNDED_RECT_N_ARC - 1);
          points.push({x: cx + r * Math.cos(t), y: cy - r * Math.sin(t)});
        }
      }
      return points;
    }
    if (spec.shape === 'chamfered_rect') {
      const c = clampedParam(spec, min, max);
      return [
        {x: x0, y: y0 + c}, {x: x0 + c, y: y0},
        {x: x1 - c, y: y0}, {x: x1, y: y0 + c},
        {x: x1, y: y1 - c}, {x: x1 - c, y: y1},
        {x: x0 + c, y: y1}, {x: x0, y: y1 - c},
      ];
    }
    if (spec.shape === 'circle') {
      const r = clampedParam(spec, min, max);
      const cx = (x0 + x1) / 2.0, cy = (y0 + y1) / 2.0;
      const n = circleSegmentCount(r);
      const pts = [];
      for (let k = 0; k < n; k++) {
        const t = Math.PI - 2.0 * Math.PI * k / n;
        pts.push({x: cx + r * Math.cos(t), y: cy - r * Math.sin(t)});
      }
      return pts;
    }
    return [{x: x0, y: y0}, {x: x1, y: y0}, {x: x1, y: y1}, {x: x0, y: y1}];
  }

  function outlineContainsPoint(spec, min, max, x, y, tol) {
    tol = tol || 0.0;
    const x0 = min.x, y0 = min.y, x1 = max.x, y1 = max.y;
    if (x < x0 - tol || x > x1 + tol || y < y0 - tol || y > y1 + tol) return false;
    if (!spec || spec.shape === 'rect') return true;
    if (spec.shape === 'circle') {
      const r = clampedParam(spec, min, max);
      const cx = (x0 + x1) / 2.0, cy = (y0 + y1) / 2.0;
      return Math.hypot(x - cx, y - cy) <= r + tol;
    }
    if (spec.shape === 'chamfered_rect') {
      const c = clampedParam(spec, min, max);
      const t = tol * SQRT2;
      return (
        (x - x0) + (y - y0) >= c - t
        && (x1 - x) + (y - y0) >= c - t
        && (x1 - x) + (y1 - y) >= c - t
        && (x - x0) + (y1 - y) >= c - t
      );
    }
    if (spec.shape === 'rounded_rect') {
      const r = clampedParam(spec, min, max);
      const ncx = Math.min(Math.max(x, x0 + r), x1 - r);
      const ncy = Math.min(Math.max(y, y0 + r), y1 - r);
      return Math.hypot(x - ncx, y - ncy) <= r + tol;
    }
    return true;
  }

  function mountingHolePosition(spec, min, max, corner, insetMm) {
    const signs = {
      'top-left': [1.0, 1.0],
      'top-right': [-1.0, 1.0],
      'bottom-left': [1.0, -1.0],
      'bottom-right': [-1.0, -1.0],
    }[corner];
    if (!signs) return null;
    const [sx, sy] = signs;
    const cx = sx > 0 ? min.x : max.x;
    const cy = sy > 0 ? min.y : max.y;
    const p = clampedParam(spec, min, max);
    let entry = 0.0;
    const shape = spec ? spec.shape : 'rect';
    if (shape === 'rounded_rect' || shape === 'circle') entry = p * (SQRT2 - 1.0);
    else if (shape === 'chamfered_rect') entry = p / SQRT2;
    const perAxis = entry / SQRT2 + insetMm;
    return {x: cx + sx * perAxis, y: cy + sy * perAxis};
  }

  return {
    outlinePolyline: outlinePolyline,
    outlineContainsPoint: outlineContainsPoint,
    mountingHolePosition: mountingHolePosition,
  };
})();

window.kicraftInitLayoutCanvas = function(cfg) {
  const HOST_ID = cfg.canvas_id + '-host';

  // Each call to this IIFE bumps the canvas's version. The Python
  // side calls build_canvas_init_script + ui.run_javascript whenever
  // a new leaf lands on disk, so we end up with a stack of IIFEs
  // whose only conflict is the document-level keydown listener
  // (SVG-level listeners auto-clean when render() replaces the SVG
  // contents). Listeners check this sentinel and bail if a newer
  // IIFE has registered, so only the latest version ever responds.
  window.__mlc_version = window.__mlc_version || {};
  window.__mlc_version[cfg.canvas_id] = (window.__mlc_version[cfg.canvas_id] || 0) + 1;
  const myVersion = window.__mlc_version[cfg.canvas_id];
  function isCurrent() {
    return window.__mlc_version[cfg.canvas_id] === myVersion;
  }
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
  // Edge-snap during drag: any silk edge of another leaf (or the board
  // outline) within this many mm of the dragged leaf's edge will pull
  // the drag to exact alignment. 0.5 mm catches the sub-mm sloppiness
  // that produced visible stamped-board offsets without hijacking
  // intentional gaps.
  const SNAP_THRESHOLD_MM = 0.5;

  function deepCopy(obj) { return JSON.parse(JSON.stringify(obj)); }

  function makeState() {
    return {
      placements: deepCopy(cfg.initial.placements),
      board_outline: deepCopy(cfg.initial.board_outline),
      // Shape tag + parameters; the AABB above stays the single
      // source of truth for size, so every pre-shape code path
      // (viewBox, edge handles, snapping) is untouched by shapes.
      outline_shape: deepCopy(
        cfg.initial.outline_shape
        || { shape: 'rect', corner_radius_mm: 0.0, chamfer_mm: 0.0 }
      ),
      mounting_holes: deepCopy(cfg.initial.mounting_holes || []),
      // Last stamp's positioned DRC violations (board-frame mm), pushed
      // by the host via setDrcMarkers() after each Save & stamp. They
      // describe the LAST stamped arrangement, so any drag clears them
      // rather than leaving markers floating over a changed layout.
      drc_markers: [],
      // Opaque passthrough: per-component parent-local overrides (edge
      // connectors pinned by hand or by the removed offline GUI). The
      // canvas neither renders nor edits them, but getState() must echo
      // them back or every web-panel save wipes manual_layout.json's
      // parent_local to [] while the composer still honors the key.
      parent_local: deepCopy(cfg.initial.parent_local || []),
      selected: null,
      snap_active: null,
      // Per-axis edge pair that the latest snap pinned, populated by
      // startDrag's move handler. Each entry (when non-null) has shape
      // { my_edge, other_path, other_edge }. Read by render() to
      // draw highlighted edge segments on the dragged leaf and on the
      // constraining neighbor / outline.
      snap_constraints: { x: null, y: null },
      // View options -- mutated by setViewOptions() from the Python
      // panel. Defaults match the historical canvas behavior (grid on,
      // edge-snap on, 0 mm gap between snapped leaves); the ratsnest
      // (cross-leaf net lines) defaults on because it is the main
      // placement-quality signal the canvas offers.
      view_options: {
        show_grid: true,
        snap_enabled: true,
        snap_spacing_mm: 0.0,
        show_ratsnest: true,
      },
    };
  }

  // Mounting holes are pinned to outline corners with a per-hole
  // inset; recompute their world positions whenever the outline
  // changes so dragging an edge handle keeps the holes glued to
  // their corners. Holes with corner=null keep whatever pos they
  // had (they're declared but not pinned).
  function recomputeMountingHoles() {
    const out = state.board_outline;
    for (const h of state.mounting_holes) {
      if (!h.corner) continue;
      const inset = Number(h.inset_mm) || 0;
      // Shape-aware corner peg: on rounded/chamfered/circular boards
      // the AABB corner is off-board, so the peg point walks inward
      // along the corner diagonal from where the diagonal enters the
      // shape (plain rect reduces to corner + (inset, inset)).
      const pos = window.kicraftLayoutGeometry.mountingHolePosition(
        state.outline_shape, out.min, out.max, h.corner, inset
      );
      if (pos) h.pos = pos;
    }
  }

  // Circle boards need a square AABB (the circle is its inscribed
  // circle). Called after any outline mutation; `draggedSide` picks
  // which dimension wins so an edge drag feels direct.
  function enforceShapeConstraints(draggedSide) {
    if (!state.outline_shape || state.outline_shape.shape !== 'circle') return;
    const out = state.board_outline;
    const w = out.max.x - out.min.x;
    const h = out.max.y - out.min.y;
    if (draggedSide === 'top' || draggedSide === 'bottom') {
      out.max.x = out.min.x + h;
    } else {
      out.max.y = out.min.y + w;
    }
  }

  const initial = makeState();
  let state = makeState();

  const leafByPath = Object.fromEntries(cfg.leaves.map(l => [l.instance_path, l]));

  // Camera: zoom factor + explicit center. cx/cy null = auto-fit (track
  // the outline, the historical behavior). NOT part of the undo history
  // -- moving the camera is not a document edit.
  const view = { zoom: 1.0, cx: null, cy: null };
  const ZOOM_MIN = 0.4;
  const ZOOM_MAX = 16.0;

  function viewBox() {
    const out = state.board_outline;
    const w = out.max.x - out.min.x;
    const h = out.max.y - out.min.y;
    const fitW = Math.max(w + 2 * PADDING_X_MM, 30);
    const fitH = Math.max(h + 2 * PADDING_Y_MM, 30);
    const fitCx = out.min.x - PADDING_X_MM + fitW / 2;
    const fitCy = out.min.y - PADDING_Y_MM + fitH / 2;
    const cx = view.cx === null ? fitCx : view.cx;
    const cy = view.cy === null ? fitCy : view.cy;
    const vbW = fitW / view.zoom;
    const vbH = fitH / view.zoom;
    return { vbX: cx - vbW / 2, vbY: cy - vbH / 2, vbW, vbH };
  }

  function fitView() {
    view.zoom = 1.0;
    view.cx = null;
    view.cy = null;
    render();
  }

  // --- Undo / redo ---------------------------------------------------------
  // Snapshots cover the DOCUMENT (placements, outline, shape, holes) --
  // not the camera, not view options, not selection. Rapid same-kind
  // mutations (arrow-key nudges) coalesce into one step via the tag.
  const history = [];
  const future = [];
  const HISTORY_MAX = 100;
  let lastPushTag = null;
  let lastPushAt = 0;

  function snapshotData() {
    return {
      placements: deepCopy(state.placements),
      board_outline: deepCopy(state.board_outline),
      outline_shape: deepCopy(state.outline_shape),
      mounting_holes: deepCopy(state.mounting_holes),
    };
  }

  function restoreData(snap) {
    state.placements = deepCopy(snap.placements);
    state.board_outline = deepCopy(snap.board_outline);
    state.outline_shape = deepCopy(snap.outline_shape);
    state.mounting_holes = deepCopy(snap.mounting_holes);
    state.drc_markers = [];
    if (state.selected
        && !state.placements.some(p => p.instance_path === state.selected)) {
      state.selected = null;
    }
    emitOutlineChanged();
    render();
    emitSelectionChanged();
  }

  function pushHistory(tag) {
    const now = Date.now();
    if (tag && tag === lastPushTag && now - lastPushAt < 800) {
      lastPushAt = now;
      return;
    }
    lastPushTag = tag || null;
    lastPushAt = now;
    history.push(snapshotData());
    if (history.length > HISTORY_MAX) history.shift();
    future.length = 0;
  }

  function undo() {
    if (!history.length) return;
    future.push(snapshotData());
    restoreData(history.pop());
    lastPushTag = null;
  }

  function redo() {
    if (!future.length) return;
    history.push(snapshotData());
    restoreData(future.pop());
    lastPushTag = null;
  }

  // Run a canvas-API mutation and record it as ONE undo step -- but only
  // if it actually changed the document (the panels re-push identical
  // state at mount time; that must not pollute the history).
  function applyWithHistory(mutator) {
    const before = snapshotData();
    mutator();
    if (JSON.stringify(before) !== JSON.stringify(snapshotData())) {
      history.push(before);
      if (history.length > HISTORY_MAX) history.shift();
      future.length = 0;
      lastPushTag = null;
    }
    render();
  }

  function setSelected(ip) {
    state.selected = ip;
    const sel = document.getElementById(SELECTED_ID);
    if (sel) {
      if (ip) {
        const lf = leafByPath[ip];
        sel.textContent = lf ? lf.sheet_name : ip;
      } else {
        sel.textContent = 'none';
      }
    }
    updateCoordsLabel();
    render();
    emitSelectionChanged();
  }

  function updateCoordsLabel() {
    const coords = document.getElementById(COORDS_ID);
    const outlineEl = document.getElementById(OUTLINE_ID);
    if (outlineEl) {
      const w = (state.board_outline.max.x - state.board_outline.min.x);
      const h = (state.board_outline.max.y - state.board_outline.min.y);
      outlineEl.textContent = w.toFixed(1) + ' × ' + h.toFixed(1) + ' mm';
    }
    if (!coords) return;
    if (!state.selected) { coords.textContent = '--'; return; }
    const p = state.placements.find(p => p.instance_path === state.selected);
    if (!p) { coords.textContent = '--'; return; }
    coords.textContent = 'x=' + p.origin.x.toFixed(2) + ', y=' + p.origin.y.toFixed(2)
      + ', rot=' + (p.rotation || 0).toFixed(0) + '°';
  }

  // Mirror outline size back to the host's W/H inputs (NiceGUI global
  // custom event; the Python side listens via ui.on). Fired on user
  // gestures that change the outline OUTSIDE the inputs themselves --
  // edge-handle drags and shape constraints (circle squaring) -- so the
  // numbers on screen never go stale. No-op when the host page doesn't
  // define emitEvent (e.g. a bare test harness).
  function emitOutlineChanged() {
    if (typeof window.emitEvent !== 'function') return;
    const out = state.board_outline;
    window.emitEvent('kicraft-ml-outline', {
      canvas_id: cfg.canvas_id,
      width: Math.round((out.max.x - out.min.x) * 100) / 100,
      height: Math.round((out.max.y - out.min.y) * 100) / 100,
    });
  }

  // Selection sync to the host (numeric x/y/rot editing panel): the
  // VISUAL CENTER + rotation of the selected leaf, or instance_path
  // null on deselect. Center-based (not origin) because "where is this
  // block" is what the user thinks in.
  function emitSelectionChanged() {
    if (typeof window.emitEvent !== 'function') return;
    let payload = { canvas_id: cfg.canvas_id, instance_path: null };
    if (state.selected) {
      const p = state.placements.find(x => x.instance_path === state.selected);
      const leaf = leafByPath[state.selected];
      if (p && leaf) {
        const c = leafCenter(p, leaf);
        payload = {
          canvas_id: cfg.canvas_id,
          instance_path: state.selected,
          sheet_name: leaf.sheet_name,
          cx: Math.round(c.x * 100) / 100,
          cy: Math.round(c.y * 100) / 100,
          rotation: Math.round((p.rotation || 0) * 10) / 10,
        };
      }
    }
    window.emitEvent('kicraft-ml-selected', payload);
  }

  function snapAngle(deg, shiftHeld) {
    if (shiftHeld) return deg;
    const m = ((deg % 360) + 360) % 360;
    return Math.round(m / SNAP_DEG) * SNAP_DEG;
  }

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
  // Center-rotation pivots around the leaf's physical center (Edge.Cuts
  // AABB center). This is what gets stamped on the parent, so rotating
  // around it keeps the user's mental model -- "I'm spinning the board"
  // -- consistent with the stamped result.
  function leafCenterLocal(leaf) {
    return {
      x: (leaf.edge_min_x + leaf.edge_max_x) * 0.5,
      y: (leaf.edge_min_y + leaf.edge_max_y) * 0.5,
    };
  }

  function leafCenter(p, leaf) {
    const r = (p.rotation || 0) * Math.PI / 180;
    const c = Math.cos(r), s = Math.sin(r);
    const sc = leafCenterLocal(leaf);
    return {
      x: p.origin.x + c * sc.x + s * sc.y,
      y: p.origin.y - s * sc.x + c * sc.y,
    };
  }

  // Inverse for the same CW rotation: solve
  //   center = origin + R_CW(theta) * leaf_center
  // for origin so the visual center stays put as the user rotates.
  function setRotationKeepCenter(p, leaf, newRotDeg) {
    const center = leafCenter(p, leaf);
    const r = newRotDeg * Math.PI / 180;
    const c = Math.cos(r), s = Math.sin(r);
    const sc = leafCenterLocal(leaf);
    p.origin.x = center.x - (c * sc.x + s * sc.y);
    p.origin.y = center.y - (-s * sc.x + c * sc.y);
    p.rotation = newRotDeg;
  }

  // Axis-aligned bbox of the leaf's PHYSICAL board (Edge.Cuts) in
  // PARENT (canvas-world) coords, accounting for rotation. This is the
  // single source of truth for snap + overflow + inter-leaf overlap:
  // it's the rectangle the parent stamper actually places on the
  // board, so anything aligned here matches what gets stamped.
  function leafBboxParent(p, leaf) {
    const r = (p.rotation || 0) * Math.PI / 180;
    const c = Math.cos(r), s = Math.sin(r);
    const x0 = leaf.edge_min_x, y0 = leaf.edge_min_y;
    const x1 = leaf.edge_max_x, y1 = leaf.edge_max_y;
    const tx = (x, y) => x * c + y * s + p.origin.x;
    const ty = (x, y) => -x * s + y * c + p.origin.y;
    const xs = [tx(x0, y0), tx(x1, y0), tx(x1, y1), tx(x0, y1)];
    const ys = [ty(x0, y0), ty(x1, y0), ty(x1, y1), ty(x0, y1)];
    return {
      min_x: Math.min.apply(null, xs),
      max_x: Math.max.apply(null, xs),
      min_y: Math.min.apply(null, ys),
      max_y: Math.max.apply(null, ys),
    };
  }

  // Compute the set of leaves that collide with at least one other leaf
  // in parent space. Used to red-flag overlapping placements. The
  // collision is on Edge.Cuts AABBs -- if two leaves' physical boards
  // overlap by more than EPS_MM, both go in the set.
  const OVERLAP_EPS_MM = 0.01;
  function computeOverlaps() {
    const bboxes = state.placements.map(p => {
      const leaf = leafByPath[p.instance_path];
      if (!leaf) return null;
      return { ip: p.instance_path, b: leafBboxParent(p, leaf) };
    }).filter(Boolean);
    const overlapping = new Set();
    for (let i = 0; i < bboxes.length; i++) {
      for (let j = i + 1; j < bboxes.length; j++) {
        const a = bboxes[i].b, b = bboxes[j].b;
        const ox = Math.min(a.max_x, b.max_x) - Math.max(a.min_x, b.min_x);
        const oy = Math.min(a.max_y, b.max_y) - Math.max(a.min_y, b.min_y);
        if (ox > OVERLAP_EPS_MM && oy > OVERLAP_EPS_MM) {
          overlapping.add(bboxes[i].ip);
          overlapping.add(bboxes[j].ip);
        }
      }
    }
    return overlapping;
  }

  // Minimum spanning tree over a net's anchor points (Prim; nets have
  // a handful of anchors, so O(n^2) is fine). Returns index pairs.
  // Classic ratsnest topology: every anchor connected, no cycles, and
  // total line length minimal -- the shortest "work remaining" picture.
  function mstEdges(pts) {
    const n = pts.length;
    if (n < 2) return [];
    const d2 = (a, b) => (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y);
    const inTree = new Array(n).fill(false);
    const dist = new Array(n).fill(Infinity);
    const from = new Array(n).fill(0);
    const edges = [];
    inTree[0] = true;
    for (let i = 1; i < n; i++) dist[i] = d2(pts[0], pts[i]);
    for (let k = 1; k < n; k++) {
      let best = -1;
      for (let i = 0; i < n; i++) {
        if (!inTree[i] && (best < 0 || dist[i] < dist[best])) best = i;
      }
      if (best < 0) break;
      inTree[best] = true;
      edges.push([from[best], best]);
      for (let i = 0; i < n; i++) {
        if (inTree[i]) continue;
        const d = d2(pts[best], pts[i]);
        if (d < dist[i]) { dist[i] = d; from[i] = best; }
      }
    }
    return edges;
  }

  // Ratsnest overlay: for each cross-leaf net, transform its anchors by
  // the owning leaf's live placement (same CW convention as
  // leafBboxParent) and draw the net's MST as dashed lines. Nets that
  // touch the selected leaf draw highlighted so "what does this block
  // talk to" is one click away.
  function renderRatsnest(svg) {
    if (!state.view_options.show_ratsnest) return;
    const nets = cfg.ratsnest || [];
    if (!nets.length) return;
    const placementByPath = Object.fromEntries(
      state.placements.map(p => [p.instance_path, p])
    );
    const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'ml-ratsnest');
    for (const net of nets) {
      const pts = [];
      for (const a of net.anchors || []) {
        const p = placementByPath[a.instance_path];
        if (!p) continue;
        const r = (p.rotation || 0) * Math.PI / 180;
        const c = Math.cos(r), s = Math.sin(r);
        pts.push({
          x: p.origin.x + c * a.x + s * a.y,
          y: p.origin.y - s * a.x + c * a.y,
          ip: a.instance_path,
        });
      }
      if (pts.length < 2) continue;
      const hot = state.selected && pts.some(pt => pt.ip === state.selected);
      for (const [i, j] of mstEdges(pts)) {
        const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        ln.setAttribute('class', 'ml-ratsnest-line' + (hot ? ' hot' : ''));
        ln.setAttribute('x1', pts[i].x); ln.setAttribute('y1', pts[i].y);
        ln.setAttribute('x2', pts[j].x); ln.setAttribute('y2', pts[j].y);
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = net.net;
        ln.appendChild(title);
        g.appendChild(ln);
      }
    }
    svg.appendChild(g);
  }

  // Walk every other leaf's silk bbox + the board outline edges and
  // collect the smallest x/y offset that pulls the dragged leaf's
  // edges into exact alignment, when that offset is within
  // SNAP_THRESHOLD_MM. Returns null when nothing is close enough.
  function computeDragSnap(ip, myBbox) {
    if (!state.view_options.snap_enabled) return null;
    const gap = Math.max(0, Number(state.view_options.snap_spacing_mm) || 0);
    const candidates = [];
    for (const other of state.placements) {
      if (other.instance_path === ip) continue;
      const otherLeaf = leafByPath[other.instance_path];
      if (!otherLeaf) continue;
      const ob = leafBboxParent(other, otherLeaf);
      candidates.push({ b: ob, edge_gap: gap, path: other.instance_path });
    }
    // Board outline counts too -- snap to the INSIDE of the parent
    // edges, with no gap (the gap setting is for leaf-to-leaf spacing,
    // not leaf-to-outline padding). The sentinel path '__outline__'
    // lets the renderer find the right bbox without aliasing a leaf.
    const o = state.board_outline;
    candidates.push({
      b: { min_x: o.min.x, max_x: o.max.x, min_y: o.min.y, max_y: o.max.y },
      edge_gap: 0,
      path: '__outline__',
    });

    let bestDx = 0, bestDxDist = Infinity, bestXC = null;
    let bestDy = 0, bestDyDist = Infinity, bestYC = null;
    for (const c of candidates) {
      const ob = c.b;
      const g = c.edge_gap;
      // Pair format: [my_edge, other_edge, current_offset, target_offset].
      // Snap activates when |offset - target| < SNAP_THRESHOLD_MM, and
      // moves the leaf by (target - offset). Edge-to-edge pairs use
      // ±gap; axis-alignment pairs keep target=0. The edge names are
      // carried through so the renderer can highlight exactly which
      // edges are pinned.
      const xPairs = [
        ['right', 'left',  myBbox.max_x - ob.min_x, -g],  // my right -> other left
        ['left',  'right', myBbox.min_x - ob.max_x,  g],  // my left  -> other right
        ['left',  'left',  myBbox.min_x - ob.min_x,  0],  // left-aligned
        ['right', 'right', myBbox.max_x - ob.max_x,  0],  // right-aligned
      ];
      const yPairs = [
        ['bottom', 'top',    myBbox.max_y - ob.min_y, -g],  // my bottom -> other top
        ['top',    'bottom', myBbox.min_y - ob.max_y,  g],  // my top    -> other bottom
        ['top',    'top',    myBbox.min_y - ob.min_y,  0],  // top-aligned
        ['bottom', 'bottom', myBbox.max_y - ob.max_y,  0],  // bottom-aligned
      ];
      for (const [myE, otherE, d, target] of xPairs) {
        const a = Math.abs(d - target);
        if (a < SNAP_THRESHOLD_MM && a < bestDxDist) {
          bestDx = target - d;
          bestDxDist = a;
          bestXC = { my_edge: myE, other_path: c.path, other_edge: otherE };
        }
      }
      for (const [myE, otherE, d, target] of yPairs) {
        const a = Math.abs(d - target);
        if (a < SNAP_THRESHOLD_MM && a < bestDyDist) {
          bestDy = target - d;
          bestDyDist = a;
          bestYC = { my_edge: myE, other_path: c.path, other_edge: otherE };
        }
      }
    }
    if (bestDxDist === Infinity && bestDyDist === Infinity) return null;
    return { dx: bestDx, dy: bestDy, x_constraint: bestXC, y_constraint: bestYC };
  }

  function render() {
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
    // float over a gridless dark background. Skipped entirely when
    // the View options panel toggles show_grid off.
    if (state.view_options.show_grid) {
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
      for (let x = x0; x <= x1; x += 5) {
        const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        ln.setAttribute('x1', x); ln.setAttribute('x2', x);
        ln.setAttribute('y1', visY); ln.setAttribute('y2', y1);
        if (x % 10 === 0) ln.setAttribute('class', 'major');
        grid.appendChild(ln);
      }
      for (let y = y0; y <= y1; y += 5) {
        const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
        ln.setAttribute('x1', visX); ln.setAttribute('x2', x1);
        ln.setAttribute('y1', y); ln.setAttribute('y2', y);
        if (y % 10 === 0) ln.setAttribute('class', 'major');
        grid.appendChild(ln);
      }
      svg.appendChild(grid);
    }

    // Outline: plain rect element for rect shape; closed <path> traced
    // from the shared polyline generator otherwise (the exact loop the
    // parent stamper writes to Edge.Cuts).
    const outline = state.board_outline;
    if (!state.outline_shape || state.outline_shape.shape === 'rect') {
      const outRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      outRect.setAttribute('class', 'ml-outline');
      outRect.setAttribute('x', outline.min.x);
      outRect.setAttribute('y', outline.min.y);
      outRect.setAttribute('width', outline.max.x - outline.min.x);
      outRect.setAttribute('height', outline.max.y - outline.min.y);
      svg.appendChild(outRect);
    } else {
      const pts = window.kicraftLayoutGeometry.outlinePolyline(
        state.outline_shape, outline.min, outline.max
      );
      const d = pts.map(
        (p, i) => (i === 0 ? 'M' : 'L') + p.x.toFixed(4) + ' ' + p.y.toFixed(4)
      ).join(' ') + ' Z';
      const outPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      outPath.setAttribute('class', 'ml-outline');
      outPath.setAttribute('d', d);
      svg.appendChild(outPath);
    }

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
    // Ring radius per screw size = courtyard/2 of the stock KiCad
    // MountingHole footprint the composer synthesizes (drill diameter).
    const SCREW_RING_R_MM = { 'M2': 2.2, 'M2.5': 2.7, 'M3': 3.2, 'M4': 4.3 };
    recomputeMountingHoles();
    for (const hole of state.mounting_holes) {
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
      ring.setAttribute('r', SCREW_RING_R_MM[hole.screw] || 3.2);
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
    }

    // Leaves
    const out = state.board_outline;
    const overlapping = computeOverlaps();
    for (const p of state.placements) {
      const leaf = leafByPath[p.instance_path];
      if (!leaf) continue;
      // Three leaf-local rects per placement:
      //   silk_min/max  -- leaf solver's rounded silk poly bbox.
      //                    Cosmetic; not drawn by default.
      //   image_*_mm    -- SVG viewBox of leaf_canvas.png. Used only
      //                    to position the <image> element so the
      //                    rendered PNG lands at its natural extent
      //                    (silk text labels visible). NOT used for
      //                    snap/overflow.
      //   edge_*_mm     -- Edge.Cuts AABB. The leaf's PHYSICAL extent
      //                    -- the rectangle that gets stamped on the
      //                    parent. Hit, drag/snap, overflow against
      //                    the parent outline, and inter-leaf overlap
      //                    all key off this rectangle.
      const sx0 = leaf.silk_min_x, sy0 = leaf.silk_min_y;
      const sx1 = leaf.silk_max_x, sy1 = leaf.silk_max_y;
      const sw = Math.max(0, sx1 - sx0), sh = Math.max(0, sy1 - sy0);
      const ix0 = leaf.image_x_mm, iy0 = leaf.image_y_mm;
      const ix1 = leaf.image_x_mm + leaf.image_width_mm;
      const iy1 = leaf.image_y_mm + leaf.image_height_mm;
      const ex0 = leaf.edge_min_x, ey0 = leaf.edge_min_y;
      const ex1 = leaf.edge_max_x, ey1 = leaf.edge_max_y;
      // Overflow check uses Edge.Cuts corners: the red flag fires when
      // the physical board crosses the parent outline.
      const r = (p.rotation || 0) * Math.PI / 180;
      const rc = Math.cos(r), rs = Math.sin(r);
      function corner(lx, ly) {
        return {
          x: p.origin.x + rc * lx + rs * ly,
          y: p.origin.y - rs * lx + rc * ly,
        };
      }
      const corners = [
        corner(ex0, ey0),
        corner(ex1, ey0),
        corner(ex1, ey1),
        corner(ex0, ey1),
      ];
      const overflow = corners.some(c =>
        !window.kicraftLayoutGeometry.outlineContainsPoint(
          state.outline_shape, out.min, out.max, c.x, c.y, 0.01
        )
      );
      const collides = overlapping.has(p.instance_path);
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute(
        'class',
        'ml-leaf'
        + (state.selected === p.instance_path ? ' selected' : '')
        + (overflow ? ' overflow' : '')
        + (collides ? ' overlap' : ''),
      );
      g.setAttribute('data-instance-path', p.instance_path);
      // SVG rotate() is CCW; KiCad rotation is CW. Negate so the
      // canvas visual matches the stamped output.
      g.setAttribute('transform',
        'translate(' + p.origin.x + ',' + p.origin.y + ') rotate(' + (-(p.rotation || 0)) + ')');
      // Native hover tooltip naming the block (the baked-in silk name
      // can be small or rotated off-view at low zoom).
      const leafTitle = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      leafTitle.textContent = leaf.sheet_name;
      g.appendChild(leafTitle);

      // PNG is rasterized from kicad-cli's SVG export with no trim or
      // chrome, so its pixel aspect already matches its mm aspect
      // exactly. Drawing it at the recorded SVG-viewBox rect places
      // every pixel 1:1 with the post-route board file in leaf-local
      // coords -- the silk poly INSIDE the PNG lands on the silk_min/
      // max bbox by construction, and stacked leaves composite cleanly
      // through the PNG's transparent background.
      if (leaf.render_url) {
        const img = document.createElementNS('http://www.w3.org/2000/svg', 'image');
        img.setAttribute('class', 'ml-leaf-img');
        img.setAttribute('href', leaf.render_url);
        img.setAttribute('x', leaf.image_x_mm);
        img.setAttribute('y', leaf.image_y_mm);
        img.setAttribute('width', leaf.image_width_mm);
        img.setAttribute('height', leaf.image_height_mm);
        img.setAttribute('preserveAspectRatio', 'none');
        g.appendChild(img);
      }

      // Sharp-cornered amber rectangle traced over the leaf solver's
      // silk-poly bbox. Lets the user see what gets stamped to the
      // parent's F.Silkscreen layer with a crisper edge than the
      // rounded poly inside the PNG; the rounded poly should sit
      // exactly inside these straight edges, and any corner curves
      // peeking out (~1 mm radius) make it visually obvious when a
      // placement has the silk poly somewhere unexpected.
      const silkBbox = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      const isSnapActive = state.snap_active === p.instance_path;
      silkBbox.setAttribute(
        'class', 'ml-leaf-silk-bbox' + (isSnapActive ? ' snap-active' : ''),
      );
      silkBbox.setAttribute('x', sx0);
      silkBbox.setAttribute('y', sy0);
      silkBbox.setAttribute('width', sw);
      silkBbox.setAttribute('height', sh);
      g.appendChild(silkBbox);

      // Hit / selection target = Edge.Cuts AABB. The hit area is the
      // physical board; clicking outside Edge.Cuts (e.g. on a silk
      // text label that hangs past the board) does NOT grab the leaf,
      // because that area isn't really part of this leaf's placement.
      const hit = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
      hit.setAttribute('class', 'ml-leaf-hit');
      hit.setAttribute('x', ex0);
      hit.setAttribute('y', ey0);
      hit.setAttribute('width', Math.max(0, ex1 - ex0));
      hit.setAttribute('height', Math.max(0, ey1 - ey0));
      hit.setAttribute('fill', 'transparent');
      hit.setAttribute('stroke', 'none');
      g.appendChild(hit);

      // Red Edge.Cuts overlay when this leaf collides with another.
      // Drawn AFTER the hit rect so it sits visibly on top of the leaf
      // image. Mostly transparent fill so the leaf content underneath
      // still reads.
      if (collides) {
        const ov = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        ov.setAttribute('class', 'ml-leaf-overlap');
        ov.setAttribute('x', ex0);
        ov.setAttribute('y', ey0);
        ov.setAttribute('width', Math.max(0, ex1 - ex0));
        ov.setAttribute('height', Math.max(0, ey1 - ey0));
        g.appendChild(ov);
      }

      // Rotation handle sits just outside the Edge.Cuts top-right
      // corner so it tracks the physical board under rotation.
      const rot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      rot.setAttribute('class', 'ml-rot-handle');
      rot.setAttribute('cx', ex1 + ROT_HANDLE_OFFSET_MM);
      rot.setAttribute('cy', ey0 - ROT_HANDLE_OFFSET_MM);
      rot.setAttribute('r', ROT_HANDLE_R_MM);
      rot.setAttribute('data-role', 'rotate');
      g.appendChild(rot);

      svg.appendChild(g);
    }

    // Ratsnest under the snap highlights but over the leaves, so the
    // connection lines read against the boards without hiding the
    // active snap constraint.
    renderRatsnest(svg);

    // DRC markers from the last stamp: a ring + dot at each violation's
    // board position, with the violation text as a hover tooltip. Drawn
    // on top of everything so a violation is never hidden under a leaf.
    for (const m of state.drc_markers) {
      const g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      g.setAttribute('class', 'ml-drc-marker');
      const ring = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      ring.setAttribute('cx', m.x_mm); ring.setAttribute('cy', m.y_mm);
      ring.setAttribute('r', 1.3);
      ring.setAttribute('class', 'ml-drc-ring');
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('cx', m.x_mm); dot.setAttribute('cy', m.y_mm);
      dot.setAttribute('r', 0.35);
      dot.setAttribute('class', 'ml-drc-dot');
      const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
      title.textContent = (m.type || 'violation') + ': ' + (m.description || '');
      g.appendChild(ring); g.appendChild(dot); g.appendChild(title);
      svg.appendChild(g);
    }

    // Snap-edge highlights: draw the constrained edges of the dragged
    // leaf AND the leaf (or outline) it's snapping against, drawn on
    // top of every leaf so the constraint is obvious without lighting
    // up the whole bbox. Only fires while a drag-snap is active.
    if (state.snap_active) {
      const me = state.placements.find(x => x.instance_path === state.snap_active);
      const meLeaf = me ? leafByPath[state.snap_active] : null;
      if (me && meLeaf) {
        const meBbox = leafBboxParent(me, meLeaf);
        const constraints = [state.snap_constraints.x, state.snap_constraints.y];
        for (const c of constraints) {
          if (!c) continue;
          appendSnapEdge(svg, edgeLine(meBbox, c.my_edge));
          let otherBbox = null;
          if (c.other_path === '__outline__') {
            const o = state.board_outline;
            otherBbox = { min_x: o.min.x, max_x: o.max.x, min_y: o.min.y, max_y: o.max.y };
          } else {
            const other = state.placements.find(x => x.instance_path === c.other_path);
            const otherLeaf = other ? leafByPath[c.other_path] : null;
            if (other && otherLeaf) otherBbox = leafBboxParent(other, otherLeaf);
          }
          if (otherBbox) appendSnapEdge(svg, edgeLine(otherBbox, c.other_edge));
        }
      }
    }

    bindLeafEvents(svg);
    bindEdgeEvents(svg);
    updateCoordsLabel();
  }

  function edgeLine(bbox, side) {
    // Axis-aligned line segment along the named edge of an AABB.
    // The render layer rotates leaves at the group level, but
    // leafBboxParent already returns the rotated AABB in parent
    // coords, so these segments live on the visible parent-space
    // edge regardless of the leaf's rotation.
    switch (side) {
      case 'left':   return { x1: bbox.min_x, y1: bbox.min_y, x2: bbox.min_x, y2: bbox.max_y };
      case 'right':  return { x1: bbox.max_x, y1: bbox.min_y, x2: bbox.max_x, y2: bbox.max_y };
      case 'top':    return { x1: bbox.min_x, y1: bbox.min_y, x2: bbox.max_x, y2: bbox.min_y };
      case 'bottom': return { x1: bbox.min_x, y1: bbox.max_y, x2: bbox.max_x, y2: bbox.max_y };
      default: return { x1: 0, y1: 0, x2: 0, y2: 0 };
    }
  }

  function appendSnapEdge(svg, line) {
    const ln = document.createElementNS('http://www.w3.org/2000/svg', 'line');
    ln.setAttribute('class', 'ml-snap-edge');
    ln.setAttribute('x1', line.x1);
    ln.setAttribute('y1', line.y1);
    ln.setAttribute('x2', line.x2);
    ln.setAttribute('y2', line.y2);
    svg.appendChild(ln);
  }

  function addEdge(svg, side, x, y, w, h, horizontal=false) {
    const r = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
    r.setAttribute('class', 'ml-edge' + (horizontal ? ' horizontal' : ''));
    r.setAttribute('data-edge', side);
    r.setAttribute('x', x); r.setAttribute('y', y);
    r.setAttribute('width', w); r.setAttribute('height', h);
    svg.appendChild(r);
  }

  function svgToWorld(svg, evt) {
    const pt = svg.createSVGPoint();
    pt.x = evt.clientX; pt.y = evt.clientY;
    const ctm = svg.getScreenCTM();
    if (!ctm) return { x: 0, y: 0 };
    const inv = ctm.inverse();
    const w = pt.matrixTransform(inv);
    return { x: w.x, y: w.y };
  }

  function bindLeafEvents(svg) {
    svg.querySelectorAll('.ml-leaf').forEach(g => {
      const ip = g.getAttribute('data-instance-path');
      g.addEventListener('mousedown', (e) => {
        if (e.target && e.target.getAttribute('data-role') === 'rotate') {
          startRotate(svg, ip, e);
        } else {
          startDrag(svg, ip, e);
        }
        e.preventDefault();
      });
    });
  }

  // Registered ONCE per IIFE (from fireRender), NOT from render(): render()
  // runs on every repaint (every mousemove during a drag), and the svg
  // element + document persist across repaints (render only replaces svg
  // CHILDREN), so registering here used to stack one duplicate rotate
  // handler per repaint -- a single 'r' press after a short drag applied
  // ~200 stacked -90deg rotations and ~200 re-renders.
  function bindGlobalEvents(svg) {
    svg.addEventListener('contextmenu', (e) => {
      if (!isCurrent()) return;  // stale IIFE from a prior refresh
      const target = e.target.closest('.ml-leaf');
      if (target) {
        e.preventDefault();
        const ip = target.getAttribute('data-instance-path');
        const p = state.placements.find(x => x.instance_path === ip);
        if (p) {
          const leaf = leafByPath[ip];
          pushHistory();
          state.drc_markers = [];
          // Negate to invert input mapping: a CW step (intuitive) is
          // stored as a negative rotation, which the SVG transform's
          // own rotate(-(p.rotation)) re-flips back into a CW visual,
          // so the canvas now spins the same way the user expects
          // while preserving the stamped-board parity the negation
          // exists for in the first place.
          setRotationKeepCenter(p, leaf, snapAngle((p.rotation || 0) - SNAP_DEG, false));
          setSelected(ip);
        }
      }
    });
    document.addEventListener('keydown', (e) => {
      if (!isCurrent()) return;  // stale IIFE from a prior refresh
      // Never hijack typing: the editor page carries numeric inputs
      // (outline W/H, selected-block x/y/rot, hole insets).
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA'
                || t.isContentEditable)) {
        return;
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'z' || e.key === 'Z')) {
        e.preventDefault();
        if (e.shiftKey) redo(); else undo();
        return;
      }
      if ((e.ctrlKey || e.metaKey) && (e.key === 'y' || e.key === 'Y')) {
        e.preventDefault();
        redo();
        return;
      }
      if (e.key === 'f' || e.key === 'F') {
        fitView();
        return;
      }
      if (!state.selected) return;
      if (e.key === 'r' || e.key === 'R') {
        const p = state.placements.find(x => x.instance_path === state.selected);
        if (p) {
          const leaf = leafByPath[state.selected];
          pushHistory();
          setRotationKeepCenter(p, leaf, snapAngle((p.rotation || 0) - SNAP_DEG, false));
          state.drc_markers = [];
          render();
          emitSelectionChanged();
        }
        return;
      }
      const NUDGE = { ArrowLeft: [-1, 0], ArrowRight: [1, 0],
                      ArrowUp: [0, -1], ArrowDown: [0, 1] };
      if (e.key in NUDGE) {
        const p = state.placements.find(x => x.instance_path === state.selected);
        if (p) {
          e.preventDefault();  // the page must not scroll under a nudge
          const step = e.shiftKey ? 1.0 : 0.1;
          pushHistory('nudge');
          p.origin.x += NUDGE[e.key][0] * step;
          p.origin.y += NUDGE[e.key][1] * step;
          state.drc_markers = [];
          render();
          emitSelectionChanged();
        }
      }
    });

    // Wheel zoom around the cursor. passive:false so preventDefault can
    // stop the page from scrolling while zooming the canvas.
    svg.addEventListener('wheel', (e) => {
      if (!isCurrent()) return;
      e.preventDefault();
      const before = svgToWorld(svg, e);
      const factor = e.deltaY < 0 ? 1.2 : 1 / 1.2;
      const nz = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, view.zoom * factor));
      if (nz === view.zoom) return;
      const vbNow = viewBox();
      const cxNow = vbNow.vbX + vbNow.vbW / 2;
      const cyNow = vbNow.vbY + vbNow.vbH / 2;
      // Keep the world point under the cursor fixed while the halfspan
      // scales by zoom_old/zoom_new.
      const scale = view.zoom / nz;
      view.cx = before.x - (before.x - cxNow) * scale;
      view.cy = before.y - (before.y - cyNow) * scale;
      view.zoom = nz;
      render();
    }, { passive: false });

    // Drag empty canvas (or middle-button anywhere) to pan; a no-move
    // left click on empty space deselects; double-click empty space
    // re-fits. Leaf and edge-handle gestures keep their own handlers --
    // they are skipped here via the closest() guard.
    svg.addEventListener('mousedown', (e) => {
      if (!isCurrent()) return;
      if (e.button !== 0 && e.button !== 1) return;  // right button = context menu
      const onItem = e.target.closest
        && (e.target.closest('.ml-leaf') || e.target.closest('.ml-edge'));
      if (onItem && e.button !== 1) return;
      e.preventDefault();
      const ctm = svg.getScreenCTM();
      if (!ctm) return;
      const inv = ctm.inverse();
      const mmPerPxX = inv.a;
      const mmPerPxY = inv.d;
      const startX = e.clientX;
      const startY = e.clientY;
      const vb0 = viewBox();
      const c0 = { x: vb0.vbX + vb0.vbW / 2, y: vb0.vbY + vb0.vbH / 2 };
      let moved = false;
      const move = (ev) => {
        if (Math.abs(ev.clientX - startX) + Math.abs(ev.clientY - startY) > 3) {
          moved = true;
        }
        view.cx = c0.x - (ev.clientX - startX) * mmPerPxX;
        view.cy = c0.y - (ev.clientY - startY) * mmPerPxY;
        render();
      };
      const up = () => {
        document.removeEventListener('mousemove', move);
        document.removeEventListener('mouseup', up);
        if (!moved && e.button === 0) setSelected(null);
      };
      document.addEventListener('mousemove', move);
      document.addEventListener('mouseup', up);
    });
    svg.addEventListener('dblclick', (e) => {
      if (!isCurrent()) return;
      const onItem = e.target.closest
        && (e.target.closest('.ml-leaf') || e.target.closest('.ml-edge'));
      if (!onItem) fitView();
    });
  }

  function startDrag(svg, ip, evt) {
    setSelected(ip);
    const p = state.placements.find(x => x.instance_path === ip);
    if (!p) return;
    const leaf = leafByPath[ip];
    const start = svgToWorld(svg, evt);
    const orig = { x: p.origin.x, y: p.origin.y };
    let pushed = false;  // one undo step per drag, taken on first movement
    const move = (e) => {
      if (!pushed) { pushHistory(); pushed = true; }
      // Markers describe the LAST stamped arrangement; moving anything
      // invalidates them.
      state.drc_markers = [];
      const cur = svgToWorld(svg, e);
      p.origin.x = orig.x + (cur.x - start.x);
      p.origin.y = orig.y + (cur.y - start.y);

      // Edge-snap: pull the dragged leaf's silk bbox into exact
      // alignment with any other leaf's silk edge (or the board
      // outline) that's within SNAP_THRESHOLD_MM. Shift modifier
      // disables snap so the user can drop a leaf 0.2 mm off an
      // edge when they really want to.
      const myBbox = leafBboxParent(p, leaf);
      const snap = e.shiftKey ? null : computeDragSnap(ip, myBbox);
      if (snap) {
        p.origin.x += snap.dx;
        p.origin.y += snap.dy;
        state.snap_active = ip;
        state.snap_constraints.x = snap.x_constraint;
        state.snap_constraints.y = snap.y_constraint;
      } else {
        state.snap_active = null;
        state.snap_constraints.x = null;
        state.snap_constraints.y = null;
      }
      render();
    };
    const up = () => {
      state.snap_active = null;
      state.snap_constraints.x = null;
      state.snap_constraints.y = null;
      render();
      emitSelectionChanged();
      document.removeEventListener('mousemove', move);
      document.removeEventListener('mouseup', up);
    };
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  }

  function startRotate(svg, ip, evt) {
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
    let pushed = false;  // one undo step per rotate gesture
    const move = (e) => {
      if (!pushed) { pushHistory(); pushed = true; }
      state.drc_markers = [];
      const cur = svgToWorld(svg, e);
      const ang = Math.atan2(cur.y - center.y, cur.x - center.x) * 180 / Math.PI;
      // Negate the drag delta so a visual-CW mouse motion produces a
      // visual-CW leaf rotation. The render-side rotate(-(p.rotation))
      // exists to preserve canvas/stamp parity (KiCad's CW convention
      // y-flips relative to SVG's mathematical CCW), so we invert the
      // input mapping here instead of touching the transform.
      setRotationKeepCenter(p, leaf, baseRot - (ang - startAngle));
      render();
    };
    const up = (e) => {
      setRotationKeepCenter(p, leaf, snapAngle(p.rotation, e.shiftKey));
      render();
      emitSelectionChanged();
      document.removeEventListener('mousemove', move);
      document.removeEventListener('mouseup', up);
    };
    document.addEventListener('mousemove', move);
    document.addEventListener('mouseup', up);
  }

  function bindEdgeEvents(svg) {
    svg.querySelectorAll('.ml-edge').forEach(el => {
      const side = el.getAttribute('data-edge');
      el.addEventListener('mousedown', (e) => {
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
        let pushed = false;  // one undo step per edge drag
        const move = (ev) => {
          if (!pushed) { pushHistory(); pushed = true; }
          state.drc_markers = [];
          const dx = (ev.clientX - startClientX) * mmPerPxX;
          const dy = (ev.clientY - startClientY) * mmPerPxY;
          const out = state.board_outline;
          if (side === 'left') {
            out.min.x = Math.min(orig.min.x + dx, orig.max.x - minSize);
          } else if (side === 'right') {
            out.max.x = Math.max(orig.max.x + dx, orig.min.x + minSize);
          } else if (side === 'top') {
            out.min.y = Math.min(orig.min.y + dy, orig.max.y - minSize);
          } else if (side === 'bottom') {
            out.max.y = Math.max(orig.max.y + dy, orig.min.y + minSize);
          }
          enforceShapeConstraints(side);
          render();
        };
        const up = () => {
          emitOutlineChanged();
          document.removeEventListener('mousemove', move);
          document.removeEventListener('mouseup', up);
        };
        document.addEventListener('mousemove', move);
        document.addEventListener('mouseup', up);
        e.preventDefault();
      });
    });
  }

  // --- Public API exposed to Python ---
  window.manualLayoutCanvases = window.manualLayoutCanvases || {};
  window.manualLayoutCanvases[cfg.canvas_id] = {
    getState: function() {
      // Quantize to 0.001 mm and 0.1° to keep JSON tidy
      const placements = state.placements.map(p => ({
        instance_path: p.instance_path,
        origin: {
          x: Math.round(p.origin.x * 1000) / 1000,
          y: Math.round(p.origin.y * 1000) / 1000,
        },
        rotation: Math.round(((p.rotation || 0) * 10)) / 10,
      }));
      const out = state.board_outline;
      recomputeMountingHoles();
      const mounting_holes = state.mounting_holes.map(h => ({
        index: h.index,
        corner: h.corner,
        inset_mm: Math.round(h.inset_mm * 100) / 100,
        pos: {
          x: Math.round(h.pos.x * 1000) / 1000,
          y: Math.round(h.pos.y * 1000) / 1000,
        },
        screw: h.screw || 'M3',
      }));
      const outline_min = {
        x: Math.round(out.min.x * 1000) / 1000,
        y: Math.round(out.min.y * 1000) / 1000,
      };
      const outline_max = {
        x: Math.round(out.max.x * 1000) / 1000,
        y: Math.round(out.max.y * 1000) / 1000,
      };
      const shape = state.outline_shape || { shape: 'rect' };
      return {
        placements: placements,
        // Legacy AABB key kept so an older server can still read a
        // newer canvas's payload during deploy skew.
        board_outline: { min: outline_min, max: outline_max },
        outline: {
          shape: shape.shape || 'rect',
          min: outline_min,
          max: outline_max,
          corner_radius_mm: Math.round((shape.corner_radius_mm || 0) * 100) / 100,
          chamfer_mm: Math.round((shape.chamfer_mm || 0) * 100) / 100,
        },
        mounting_holes: mounting_holes,
        parent_local: deepCopy(state.parent_local || []),
      };
    },
    reset: function() {
      applyWithHistory(() => {
        const fresh = makeState();
        state.placements = fresh.placements;
        state.board_outline = fresh.board_outline;
        state.outline_shape = fresh.outline_shape;
        state.mounting_holes = fresh.mounting_holes;
        state.drc_markers = [];
        state.selected = null;
      });
      emitOutlineChanged();
      emitSelectionChanged();
    },
    fitView: fitView,
    undo: undo,
    redo: redo,
    setPlacementCenter: function(ip, cxMm, cyMm, rotDeg) {
      // Numeric-entry placement: position the leaf's VISUAL CENTER at
      // (cxMm, cyMm) with rotation rotDeg -- the inverse of leafCenter,
      // so the readout and the input speak the same coordinates.
      const p = state.placements.find(x => x.instance_path === ip);
      const leaf = leafByPath[ip];
      if (!p || !leaf) return;
      applyWithHistory(() => {
        const rot = typeof rotDeg === 'number' ? rotDeg : (p.rotation || 0);
        const r = rot * Math.PI / 180;
        const c = Math.cos(r), s = Math.sin(r);
        const sc = leafCenterLocal(leaf);
        p.origin.x = cxMm - (c * sc.x + s * sc.y);
        p.origin.y = cyMm - (-s * sc.x + c * sc.y);
        p.rotation = rot;
        state.drc_markers = [];
      });
      emitSelectionChanged();
    },
    getOutlineSize: function() {
      const out = state.board_outline;
      return {
        width: Math.round((out.max.x - out.min.x) * 1000) / 1000,
        height: Math.round((out.max.y - out.min.y) * 1000) / 1000,
      };
    },
    setOutlineSize: function(width, height) {
      applyWithHistory(() => {
        const out = state.board_outline;
        const w = Math.max(10, Number(width) || 0);
        const h = Math.max(10, Number(height) || 0);
        // Anchor at the existing min corner so leaves don't get
        // shoved when the user only adjusted width or only height.
        out.max.x = out.min.x + w;
        out.max.y = out.min.y + h;
        enforceShapeConstraints(null);
      });
      // The constraint may have overridden the requested size (circle
      // squares the AABB); reflect the authoritative result back so
      // the inputs match what the canvas actually holds. The Python
      // handler only writes values that differ, so this cannot loop.
      emitOutlineChanged();
    },
    setOutlineShape: function(spec) {
      // Merge {shape, corner_radius_mm, chamfer_mm}. Switching to
      // circle squares the AABB on the current width.
      if (!spec || typeof spec !== 'object') return;
      applyWithHistory(() => {
        const cur = state.outline_shape;
        if (typeof spec.shape === 'string') cur.shape = spec.shape;
        if (typeof spec.corner_radius_mm === 'number') {
          cur.corner_radius_mm = Math.max(0, spec.corner_radius_mm);
        }
        if (typeof spec.chamfer_mm === 'number') {
          cur.chamfer_mm = Math.max(0, spec.chamfer_mm);
        }
        enforceShapeConstraints(null);
      });
      emitOutlineChanged();
    },
    getOutlineShape: function() {
      return deepCopy(state.outline_shape);
    },
    setMountingHoles: function(holes) {
      // Replace the entire mounting-holes list. Caller (Python side)
      // owns count + per-hole corner/inset choices; the canvas owns
      // positions: a pushed hole without pos keeps the live pos of the
      // hole it replaces (matched by index). Defaulting a pos-less hole
      // to the outline min corner used to teleport every unpinned hole
      // to the board's top-left, which Save then persisted.
      applyWithHistory(() => {
        const prevByIndex = {};
        for (const h of state.mounting_holes) prevByIndex[h.index] = h;
        const out = state.board_outline;
        state.mounting_holes = (holes || []).map((h, i) => {
          const idx = typeof h.index === 'number' ? h.index : i;
          const prev = prevByIndex[idx];
          return {
            index: idx,
            corner: h.corner || null,
            inset_mm: Number(h.inset_mm) || 5.0,
            screw: typeof h.screw === 'string' && h.screw ? h.screw : 'M3',
            pos: h.pos ? { x: Number(h.pos.x) || 0, y: Number(h.pos.y) || 0 }
                 : (prev ? prev.pos
                         : { x: (out.min.x + out.max.x) / 2,
                             y: (out.min.y + out.max.y) / 2 }),
          };
        });
      });
    },
    getMountingHoles: function() {
      recomputeMountingHoles();
      return state.mounting_holes.map(h => ({
        index: h.index,
        corner: h.corner,
        inset_mm: Math.round(h.inset_mm * 100) / 100,
        screw: h.screw || 'M3',
        pos: {
          x: Math.round(h.pos.x * 1000) / 1000,
          y: Math.round(h.pos.y * 1000) / 1000,
        },
      }));
    },
    setDrcMarkers: function(markers) {
      // Replace the marker set (board-frame mm). Pushed by the host
      // after each stamp; entries without a position are dropped here
      // so render() can trust x_mm/y_mm.
      state.drc_markers = (markers || []).filter(
        m => m && typeof m.x_mm === 'number' && typeof m.y_mm === 'number'
      );
      render();
    },
    setViewOptions: function(opts) {
      // Merge into state.view_options so callers only need to pass
      // changed fields. Keys: show_grid (bool), snap_enabled (bool),
      // snap_spacing_mm (number). Re-render so toggles take effect
      // immediately; the snap_spacing_mm value is read live during
      // each drag so no render-time work is needed for it alone.
      if (!opts || typeof opts !== 'object') return;
      if (typeof opts.show_grid === 'boolean') {
        state.view_options.show_grid = opts.show_grid;
      }
      if (typeof opts.snap_enabled === 'boolean') {
        state.view_options.snap_enabled = opts.snap_enabled;
      }
      if (typeof opts.snap_spacing_mm === 'number') {
        state.view_options.snap_spacing_mm = Math.max(0, opts.snap_spacing_mm);
      }
      if (typeof opts.show_ratsnest === 'boolean') {
        state.view_options.show_ratsnest = opts.show_ratsnest;
      }
      render();
    },
  };

  // The Manual Layout tab is mounted lazily by Quasar -- on initial
  // page load the SVG element is not yet in the DOM. Wait for it via
  // MutationObserver so the bootstrap survives an arbitrary delay
  // between page load and the first tab click (an earlier 30 s
  // timeout silently dropped renders when the user lingered in
  // Setup / Monitor before opening Manual Layout). RAF lets layout
  // settle before the first paint.
  function fireRender() {
    if (!isCurrent()) return;
    const svg = document.getElementById(SVG_ID);
    if (!svg) return;
    bindGlobalEvents(svg);  // once per IIFE; render() must never re-register
    requestAnimationFrame(function() {
      if (!isCurrent()) return;
      render();
    });
  }
  if (document.getElementById(SVG_ID)) {
    fireRender();
  } else {
    const observer = new MutationObserver(function(_muts, obs) {
      if (!isCurrent()) { obs.disconnect(); return; }
      if (document.getElementById(SVG_ID)) {
        obs.disconnect();
        fireRender();
      }
    });
    observer.observe(document.body, { childList: true, subtree: true });
  }
};
