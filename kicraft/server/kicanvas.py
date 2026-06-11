"""Reusable KiCanvas embed for the NiceGUI web app.

KiCanvas (https://kicanvas.org, theacodes/kicanvas, MIT) is a read-only,
browser-side WebGL viewer for KiCad files. We self-host the alpha bundle
(`static/kicanvas.js`, fetched from https://kicanvas.org/kicanvas/kicanvas.js)
so the product has no third-party runtime dependency or CDN drift.

The viewer renders one `<kicanvas-embed>` wrapping one or more
`<kicanvas-source>` children. For a hierarchical project, KiCanvas links the
root schematic, its leaf sheets, and the board *by filename*, so each source
carries its real KiCad filename in the `name` attribute (the URL it is fetched
from is independent and may carry a cache-busting query).

Two load-bearing NiceGUI facts:
  * `ui.html(..., sanitize=True)` (the DEFAULT) runs DOMPurify client-side and
    strips unknown custom elements, so `<kicanvas-embed>` silently vanishes. We
    MUST pass `sanitize=False`.
  * Reassigning `.content` then calling `.update()` re-sets the element's
    innerHTML, which re-parses the subtree so KiCanvas re-connects and re-fetches.
    KiCanvas (alpha) exposes no imperative reload, so a full element recreation
    plus a new `?v=` is the only reliable refresh.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from nicegui import app, ui

KICANVAS_ASSET: Path = Path(__file__).parent / "static" / "kicanvas.js"
KICANVAS_SCRIPT_URL = "/static/kicanvas.js"

# KiCanvas (alpha) always opens a document zoomed to the full drawing sheet and
# exposes no attribute to change that (its `zoom` attribute is registered but
# never read), so every preview showed the A4 frame + title block with the
# actual circuit tiny in the middle. This companion script reaches through the
# bundle's open shadow roots to each viewer element (which exposes its viewer
# object as a public `.viewer` property), hides the ":DrawingSheet" render
# layer (frame + title block — its painters target only that layer), and
# refits the camera to the painted content: the Edge.Cuts outline for boards,
# the union of painted item layers for schematics (every layer's bbox is
# populated at paint time). It re-runs on each "kicanvas:load" the viewer
# object dispatches, which covers switching sheets inside a multi-source
# embed (each load repaints the layers and re-zooms to the page). Polling is
# the only discovery mechanism: the apps/viewers are created asynchronously
# inside shadow DOM after the sources download, and the load event does not
# bubble out of the embed.
KICANVAS_CONTENT_FIT_JS = """
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
"""


@dataclass(frozen=True)
class KiCanvasSource:
    """One document inside an embed.

    url:      where the browser fetches the bytes (may carry ?v= cache-bust).
    filename: the real KiCad filename, used as <kicanvas-source name=...> so
              hierarchical sheets + board link correctly.
    """

    url: str
    filename: str


def _attr(value: str) -> str:
    """Escape a string for use inside a double-quoted HTML attribute.

    `&` is escaped first so an already-present `?v=`/`&v=` query survives; the
    browser decodes `&amp;` back to `&` when it fetches, so the real request URL
    is unchanged.
    """
    return value.replace("&", "&amp;").replace('"', "&quot;")


def kicanvas_head(script_url: str = KICANVAS_SCRIPT_URL) -> None:
    """Inject the KiCanvas ES module <script> (plus the content-fit companion
    script, see KICANVAS_CONTENT_FIT_JS) once per page.

    Idempotent within a client connection via app.storage.client, so calling it
    from several components on one page adds a single <script>. Must be called
    inside a @ui.page handler (where the client context exists).
    """
    try:
        flag = app.storage.client
    except Exception:
        flag = None
    if flag is not None:
        if flag.get("_kicanvas_head"):
            return
        flag["_kicanvas_head"] = True
    ui.add_head_html(f'<script type="module" src="{_attr(script_url)}"></script>')
    ui.add_head_html(f"<script>{KICANVAS_CONTENT_FIT_JS}</script>")


def _embed_html(sources: list[KiCanvasSource], *, controls: str = "full") -> str:
    """Build the `<kicanvas-embed>...</kicanvas-embed>` markup for `sources`."""
    children = "".join(
        f'<kicanvas-source src="{_attr(s.url)}" name="{_attr(s.filename)}"></kicanvas-source>'
        for s in sources
    )
    # `background` overrides the bundle's `:host{background-color:aqua}` (an inline
    # style beats a shadow `:host` rule). Without it, any pre-paint failure (a source
    # 404, an unsupported construct) leaves the embed a bright-cyan "teal blob"; with
    # it, a failed render is a neutral dark panel that blends with the app.
    return (
        f'<kicanvas-embed controls="{_attr(controls)}" '
        f'style="display:block;width:100%;height:100%;background:#0b1220;">'
        f'{children}</kicanvas-embed>'
    )


class KiCanvasView:
    """A self-refreshing KiCanvas embed bound to one `ui.html(sanitize=False)`.

    Create inside a layout context (e.g. `with holder:`). Call `refresh()` to
    re-fetch the same sources (new `?v=`) after an artifact is rewritten, or
    `set_sources()` to point at different files.
    """

    def __init__(
        self,
        sources: list[KiCanvasSource],
        *,
        controls: str = "full",
        height: str = "h-[520px]",
        style: str = "",
    ) -> None:
        self._sources = list(sources)
        self._controls = controls
        self._gen = 0
        # sanitize=False is REQUIRED (see module docstring) or the embed is stripped.
        # `height` is a Tailwind class (fixed pixel heights only); pass `style` (e.g.
        # "height:calc(100vh - 460px)") for a viewport-relative height instead, since
        # an inline style is reliable where a Tailwind arbitrary calc() may not be.
        self._html = ui.html(self._render(), sanitize=False).classes(
            f"w-full {height} border border-slate-700 rounded overflow-hidden bg-black"
        )
        if style:
            self._html.style(style)

    def _render(self) -> str:
        self._gen += 1
        busted = [
            KiCanvasSource(
                f"{s.url}{'&' if '?' in s.url else '?'}v={self._gen}", s.filename
            )
            for s in self._sources
        ]
        return _embed_html(busted, controls=self._controls)

    def set_sources(self, sources: list[KiCanvasSource]) -> None:
        self._sources = list(sources)
        self._html.content = self._render()
        self._html.update()

    def refresh(self) -> None:
        self._html.content = self._render()
        self._html.update()

    @property
    def element(self) -> ui.html:
        return self._html
