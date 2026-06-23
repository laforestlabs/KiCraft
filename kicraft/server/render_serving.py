"""Token-gated raw-file serving for the browser: KiCanvas KiCad files, render-gallery
PNGs, and part-library SVG previews.

Extracted from web.py (refactor roadmap Phase 3). A capability token encodes the
project dir, HMAC-signed with the server secret, so it gates access without
``app.storage.user`` (whose getter can assert outside the page/connection flow).
The token is STATELESS -- it carries its own (signed) project path -- so serving
survives a server restart and needs no in-memory map. (An in-memory map used to
back this; a ``kicraft-web`` restart wiped it, so every still-open tab 404'd on its
schematic fetches and KiCanvas painted its aqua fallback -- the "teal blob".)

Importing this module REGISTERS the routes: the ``@app.get`` decorators run at
import time.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
from pathlib import Path

from nicegui import app
from starlette.responses import FileResponse, PlainTextResponse

from ..parts_library import PART_NAME_RE
from .parts_catalog import footprint_svg, get_part, symbol_svgs

# Raw KiCad file suffixes servable by token (basename only; see serve_project_file).
_ALLOWED_SUFFIXES = (".kicad_sch", ".kicad_pcb", ".kicad_pro")


def _project_secret() -> bytes:
    """The HMAC key for project-file tokens: the same stable storage secret used to
    sign the session cookie (env in the box .env, default for local dev). Stable
    across restarts, so a token minted before a deploy still verifies afterwards."""
    return os.environ.get("KICRAFT_STORAGE_SECRET", "kicraft-dev-secret").encode("utf-8")


def _b64e(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _b64d(txt: str) -> bytes:
    return base64.urlsafe_b64decode(txt + "=" * (-len(txt) % 4))


def _register_project_dir(project_dir: Path) -> str:
    """Mint a stateless, signed token that the browser uses to fetch the project's
    raw KiCad files. The token carries the (absolute) project path plus an HMAC over
    it, so any process holding the secret can verify it -- no in-memory map, nothing
    to evict, and it survives a restart. Forgery needs the secret (HMAC)."""
    payload = _b64e(str(project_dir.resolve()).encode("utf-8"))
    sig = _b64e(hmac.new(_project_secret(), payload.encode("ascii"), hashlib.sha256).digest())
    return f"{payload}.{sig}"


def _resolve_project_token(token: str) -> Path | None:
    """The project dir a token authorizes, or None if it is malformed or its HMAC
    does not verify. Path containment/suffix checks stay with the serve handlers."""
    try:
        payload, sig = token.split(".", 1)
        expected = hmac.new(
            _project_secret(), payload.encode("ascii"), hashlib.sha256).digest()
        if not hmac.compare_digest(_b64d(sig), expected):
            return None
        return Path(_b64d(payload).decode("utf-8"))
    except Exception:  # malformed token / bad base64 / bad utf-8 -> unauthorized
        return None


@app.get("/project/{token}/{filename}")
def serve_project_file(token: str, filename: str):
    """Serve one KiCad file from a tokened project dir to the browser (KiCanvas).

    Defends three ways against traversal: basename-only (any slash rejected), a
    suffix whitelist, and a check that the resolved target sits directly in the
    project dir. `no-store` so a rewritten board is always re-fetched.
    """
    base = _resolve_project_token(token)
    name = Path(filename).name
    if base is None or name != filename or not name.endswith(_ALLOWED_SUFFIXES):
        return PlainTextResponse("not found", status_code=404)
    target = (base / name).resolve()
    if target.parent != base.resolve() or not target.is_file():
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(
        str(target),
        media_type="text/plain; charset=utf-8",
        headers={"Cache-Control": "no-store"},
    )


@app.get("/project/{token}/render/{subpath:path}")
def serve_project_render(token: str, subpath: str):
    """Serve a render PNG from a tokened project dir's `.experiments` tree.

    KiCanvas only renders KiCad files, so the place/route progress gallery shows
    the layout engine's per-leaf preview PNGs via plain <img>. These live in deep
    subpaths (`.experiments/subcircuits/<uuid>/renders/*.png`) that
    `serve_project_file` rejects, so this endpoint allows a relative subpath but
    keeps the same defense: the resolved target must stay inside the project dir,
    be a `.png`, and exist. `no-store` so an overwritten render is re-fetched."""
    base = _resolve_project_token(token)
    if base is None:
        return PlainTextResponse("not found", status_code=404)
    target = (base / subpath).resolve()
    if (not target.is_relative_to(base.resolve())
            or target.suffix != ".png" or not target.is_file()):
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(
        str(target),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


# Part-library previews: KiCanvas can't render a bare .kicad_sym/.kicad_mod, so the
# /parts catalog shows symbols and footprints as SVGs produced by kicad-cli and cached
# by content-hash (see parts_catalog). These are library reference assets (no per-user
# data), so like the /samples static files they need no auth; the /parts *page* is gated.
@app.get("/part-preview/{name}/{asset}")
def serve_part_preview(name: str, asset: str):
    """Serve a cached symbol/footprint SVG for a library part, generating on demand.

    ``asset`` is ``symbol-<n>.svg`` (1-based unit) or ``footprint.svg``. The name is
    validated and must resolve to a real bundle, so junk or a traversal name 404s.
    """
    if not PART_NAME_RE.match(name):
        return PlainTextResponse("not found", status_code=404)
    part = get_part(name)
    if part is None:
        return PlainTextResponse("not found", status_code=404)

    target: Path | None = None
    if asset == "footprint.svg":
        target = footprint_svg(part)
    else:
        m = re.fullmatch(r"symbol-(\d+)\.svg", asset)
        if m:
            svgs = symbol_svgs(part)
            idx = int(m.group(1)) - 1
            if 0 <= idx < len(svgs):
                target = svgs[idx]
    if target is None or not target.is_file():
        return PlainTextResponse("not found", status_code=404)
    return FileResponse(
        str(target),
        media_type="image/svg+xml",
        headers={"Cache-Control": "no-store"},
    )
