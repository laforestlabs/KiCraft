"""Reusable KiCanvas embed for the NiceGUI web app.

KiCanvas (https://kicanvas.org, theacodes/kicanvas, MIT) is a read-only,
browser-side WebGL viewer for KiCad files. We self-host the alpha JavaScript
bundle (`static/kicanvas.js`, fetched from https://kicanvas.org/kicanvas/kicanvas.js)
to avoid JavaScript CDN drift. The upstream bundle still requests Google Fonts
for its viewer typography and Material Symbols icons.

The viewer renders one `<kicanvas-embed>` wrapping one or more
`<kicanvas-source>` children. For a hierarchical project, KiCanvas links the
root schematic, its leaf sheets, and the board *by filename*, so each source
carries its real KiCad filename in the `name` attribute (the URL it is fetched
from is independent and may carry a cache-busting query).

The alpha embed cannot reconnect an existing element after NiceGUI reparents
its DOM: its connected callback attaches a second shadow root. Each view owns
an iframe document instead. Refreshing replaces that document, so the embed,
its WebGL resources and the content-fit companion have one shared lifetime;
parent-page rerenders never reconnect the vendor's custom element.
"""
from __future__ import annotations

import json
from hashlib import sha256

from dataclasses import dataclass
from html import escape
from pathlib import Path
from urllib.parse import quote

from nicegui import ui

KICANVAS_ASSET: Path = Path(__file__).parent / "static" / "kicanvas.js"
_FRAME_VERSION = sha256((KICANVAS_ASSET.parent / "kicanvas-frame.html").read_bytes()).hexdigest()[:12]



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
    return escape(value, quote=True)






def _frame_html(sources: list[KiCanvasSource], *, controls: str = "full") -> str:
    """Isolate the vendor lifecycle in a real same-origin HTTP document."""
    config = json.dumps(
        {"sources": [{"url": source.url, "filename": source.filename} for source in sources],
         "controls": controls},
        separators=(",", ":"),
    )
    url = f"/static/kicanvas-frame.html?v={_FRAME_VERSION}#" + quote(config, safe="")
    return (
        '<iframe title="KiCad document viewer" '
        'style="display:block;width:100%;height:100%;border:0" '
        f'src="{_attr(url)}"></iframe>'
    )


class KiCanvasView:
    """Own an isolated KiCanvas document.

    Refresh to reload current sources; set_sources replaces the document set.
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
        # Preserve the iframe markup rather than sanitizing it away.
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
        return _frame_html(busted, controls=self._controls)

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
