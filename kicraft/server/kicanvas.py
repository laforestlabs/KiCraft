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
    """Inject the KiCanvas ES module <script> once per page.

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


def _embed_html(sources: list[KiCanvasSource], *, controls: str = "full") -> str:
    """Build the `<kicanvas-embed>...</kicanvas-embed>` markup for `sources`."""
    children = "".join(
        f'<kicanvas-source src="{_attr(s.url)}" name="{_attr(s.filename)}"></kicanvas-source>'
        for s in sources
    )
    return (
        f'<kicanvas-embed controls="{_attr(controls)}" '
        f'style="display:block;width:100%;height:100%;">{children}</kicanvas-embed>'
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
    ) -> None:
        self._sources = list(sources)
        self._controls = controls
        self._gen = 0
        # sanitize=False is REQUIRED (see module docstring) or the embed is stripped.
        self._html = ui.html(self._render(), sanitize=False).classes(
            f"w-full {height} border border-slate-700 rounded overflow-hidden bg-black"
        )

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
