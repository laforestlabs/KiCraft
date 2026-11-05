// Each viewer frame owns this companion and its event/interval lifetime.
(function () {
  "use strict";
  const SHEET = ":DrawingSheet";
  const SKIP = new Set([SHEET, ":Grid", ":Overlay"]);
  const hooked = new WeakSet();

  function contentBBox(viewer) {
    const layers = viewer.layers;
    // Boards: the Edge.Cuts outline IS the board. zoom_to_board() existing is
    // what distinguishes a BoardViewer from a SchematicViewer.
    if (typeof viewer.zoom_to_board === "function") {
      const edge = layers.by_name("Edge.Cuts");
      if (edge && edge.bbox && edge.bbox.w > 0 && edge.bbox.h > 0) {
        return edge.bbox;
      }
    }
    const boxes = [];
    for (const layer of layers.in_order()) {
      if (SKIP.has(layer.name)) continue;
      const b = layer.bbox;
      if (b && b.valid) boxes.push(b);
    }
    if (!boxes.length) return null;
    const combined = boxes[0].constructor.combine(boxes);
    return combined.w > 0 && combined.h > 0 ? combined : null;
  }

  function fit(viewer) {
    try {
      if (!viewer.layers || !viewer.viewport) return;
      const sheet = viewer.layers.by_name(SHEET);
      if (sheet) sheet.visible = false;
      const bbox = contentBBox(viewer);
      if (bbox) {
        viewer.viewport.camera.bbox = bbox.grow(Math.max(bbox.w, bbox.h) * 0.05);
        // Widen the pan bounds too: they were set to the page bbox, and if any
        // content sits outside the frame the pan controller would clamp the
        // camera right back onto the empty page.
        const page = viewer.drawing_sheet && viewer.drawing_sheet.page_bbox;
        if (page) {
          viewer.viewport.bounds = bbox.constructor.combine([page, bbox]).grow(50);
        }
      }
      viewer.draw();
    } catch (e) {
      // A failed fit must never take the viewer down; page zoom is the fallback.
    }
  }

  function hook(viewerEl) {
    const viewer = viewerEl.viewer;
    if (!viewer || hooked.has(viewerEl)) return;
    hooked.add(viewerEl);
    viewer.addEventListener("kicanvas:load", () => fit(viewer));
    // The first load may already have happened before this poll tick saw the
    // element; `loaded` is KiCanvas's deferred and stays resolved once open.
    if (viewer.loaded && viewer.loaded.resolved) fit(viewer);
  }

  function scan(root) {
    for (const el of root.querySelectorAll("kc-board-viewer, kc-schematic-viewer")) {
      hook(el);
    }
    for (const el of root.querySelectorAll("*")) {
      if (el.shadowRoot) scan(el.shadowRoot);
    }
  }

  setInterval(() => {
    for (const embed of document.querySelectorAll("kicanvas-embed")) {
      if (embed.shadowRoot) scan(embed.shadowRoot);
    }
  }, 400);
})();