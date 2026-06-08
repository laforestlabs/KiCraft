"""kicraft.io web app: a gated, capped, live front end over the agent loop.

A single page: enter a brief, watch each design stage's reasoning stream in as it
commits, follow the whole pipeline on an all-phases stepper, see the schematic and
the routed board rendered natively in the browser (KiCanvas), and download the
finished KiCad project. Every model call still flows through the capped gateway
(SpendGuard), so the global spend ceilings and kill switch apply to the whole site.
Access is gated by a shared password (KICRAFT_ACCESS_PASSWORD) so only invited
users spend the balance.

Run locally:   KICRAFT_ACCESS_PASSWORD=secret python -m kicraft.server.web
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import random
import re
import shutil
import subprocess
import tempfile
import threading
import time
import types
import typing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from nicegui import app, ui
from starlette.responses import FileResponse, PlainTextResponse, RedirectResponse

from .accounts import _RESET_TTL_SECONDS, AccountStore, is_admin
from .config import LEGAL_VERSION, Settings, default_legal_dir
from .examples import CHIP_PROMPTS, EXAMPLE_PROMPTS
from .kicanvas import KICANVAS_ASSET, KiCanvasSource, KiCanvasView, kicanvas_head
from .mailer import send_reset_email
from ..parts_library import PART_NAME_RE, Tier
from .parts_catalog import (
    catalog,
    footprint_svg,
    get_part,
    kicad_cli_available,
    lcsc_url,
    symbol_svgs,
    tier_label,
    usage_markdown,
)
from .samples import SAMPLES_DIR, available_samples, featured_sample
from .session import (
    commit_slot,
    downstream_stages,
    null_downstream,
    read_state,
    record_answers,
    remaining_stages,
    run_session,
)
from .spend_guard import SpendGuard
from .stage_driver import DESIGN_STAGES, KICRAFT, SLOT_MODEL
from .stagetabs import StageTabs, demo_events

# Self-host the KiCanvas ES module bundle so the browser fetches it same-origin.
app.add_static_files("/static", str(KICANVAS_ASSET.parent))

# Prebuilt sample projects (preview renders + raw KiCad files) for the public
# landing showcase and the logged-in explorer. Public on purpose: these are
# curated, finished demos, so serving them costs nothing and needs no auth.
app.add_static_files("/samples", str(SAMPLES_DIR))

# Raw-file serving: a capability token encodes the project dir, HMAC-signed with
# the server secret, so it gates access without app.storage.user (whose getter can
# assert outside the page/connection flow). The token is STATELESS -- it carries
# its own (signed) project path -- so serving survives a server restart and needs
# no in-memory map. An in-memory map used to back this; a `kicraft-web` restart
# (a deploy) wiped it, so every still-open tab 404'd on its schematic fetches and
# KiCanvas painted its aqua fallback (the "teal blob"). See _resolve_project_token.
_ALLOWED_SUFFIXES = (".kicad_sch", ".kicad_pcb", ".kicad_pro")

# Shown in the reset email; derived from the token TTL so the two never drift.
_RESET_TTL_MINUTES = _RESET_TTL_SECONDS // 60


_STORE: AccountStore | None = None


def _store() -> AccountStore:
    """The shared accounts store, built once per process from settings."""
    global _STORE
    if _STORE is None:
        s = Settings.from_env()
        _STORE = AccountStore(s.users_db_path, s.projects_dir)
    return _STORE


def _project_spend_usd(project_id) -> float | None:
    """This project's true spend: the sum of its own ledger-recorded model calls
    (each tagged run_id='p<id>-...'), across however many runs it took. Replaces the
    old `spent_total_usd`, which was the site-wide running total -- so every project
    used to record the whole site's cumulative spend, not its own cost. Returns None
    if the ledger can't be read."""
    if project_id is None:
        return None
    try:
        return SpendGuard(Settings.from_env()).spent_for_project(project_id)
    except Exception:  # never let a billing read-back crash the run worker
        return None


def _signup_code() -> str:
    """The invite code required to register, read live (env loads in main(), so
    reading it at import time would capture an empty string). Falls back to
    KICRAFT_ACCESS_PASSWORD so an already-deployed box keeps working until its
    env is updated to the new KICRAFT_SIGNUP_CODE."""
    return (os.environ.get("KICRAFT_SIGNUP_CODE")
            or os.environ.get("KICRAFT_ACCESS_PASSWORD", "")).strip()


def _current_user():
    """The logged-in User for this page session, or None. Must be called inside a
    page/connection context (it reads app.storage.user)."""
    uid = app.storage.user.get("user_id")
    if not uid:
        return None
    user = _store().get_user(int(uid))
    if user is None:
        return None
    # Session invalidation: a password reset bumps the user's session_epoch, so a
    # session that logged in before the reset (an attacker's included) no longer
    # matches and is treated as logged out. Sessions predating this feature have no
    # stored epoch; the default 0 matches a never-reset account, so they persist.
    if app.storage.user.get("session_epoch", 0) != user.session_epoch:
        for k in ("user_id", "email", "session_epoch"):
            app.storage.user.pop(k, None)
        return None
    return user


def _require_admin():
    """Gate an admin-only page. Returns (user, None) for staff, or (None, response)
    for the page to return immediately. An authed non-admin is bounced to / rather
    than shown a 403/404 -- the /admin routes are only ever advertised to staff (the
    header link is is_admin-gated), so we don't confirm the route exists to anyone
    else. Mutating handlers re-check is_admin() on their own (defense in depth)."""
    user = _current_user()
    if user is None:
        return None, RedirectResponse("/login")
    if user.accepted_terms_version != LEGAL_VERSION:
        return None, RedirectResponse("/consent")
    if not is_admin(user):
        return None, RedirectResponse("/")
    return user, None


_LEGAL_DOCS = {"terms-of-service": "Terms of Service", "privacy-policy": "Privacy Policy"}


def _legal_markdown(name: str) -> str:
    """Load a legal document's markdown from the configured docs dir (public)."""
    if name not in _LEGAL_DOCS:
        return "# Not found"
    try:
        return (default_legal_dir() / f"{name}.md").read_text(encoding="utf-8")
    except OSError:
        return (f"# {_LEGAL_DOCS[name]} unavailable\n\nThis document could not be "
                "loaded. Please contact support.")


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


def _zip_generated(ws: Path) -> str | None:
    gen = ws / "generated"
    if not gen.is_dir():
        return None
    base = str(ws / "kicraft_project")
    return shutil.make_archive(base, "zip", root_dir=str(gen))


def _erc_offenders(ws: Path) -> list[str]:
    """The §9.12 ERC error descriptions from the build's synthesis_check.json, or
    [] if ERC was not the failing check (so recovery only fires for real ERC
    errors, not other synth failures). check_erc stores up to 20 offenders."""
    try:
        sc = json.loads((ws / ".kicraft" / "synthesis_check.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    for c in sc.get("checks") or []:
        if "ERC" in str(c.get("name", "")) and not c.get("ok"):
            return [str(o) for o in (c.get("offenders") or [])]
    return []


def _read_project_stem(ws: Path) -> str | None:
    """The project_stem committed by the intent stage (UPPER_SNAKE_CASE)."""
    try:
        data = json.loads((ws / ".kicraft" / "state.json").read_text(encoding="utf-8"))
        stem = data.get("project_stem")
        if stem:
            return str(stem)
    except (OSError, json.JSONDecodeError):
        pass
    for pro in (ws / "generated").glob("*/*.kicad_pro"):  # fallback once synth ran
        return pro.stem
    return None


def _discover_generated_dir(ws: Path | None) -> Path | None:
    """The synthesized project dir (``generated/<STEM>/``) in a workspace, found by
    inspection so the schematic stays viewable even when a run FAILS and no
    project_stem was recorded. Prefers the committed stem, then any subdir that
    actually holds schematic sheets. None until synthesis has written a sheet."""
    if ws is None:
        return None
    gen = ws / "generated"
    if not gen.is_dir():
        return None
    stem = _read_project_stem(ws)
    if stem and any((gen / stem).glob("*.kicad_sch")):
        return gen / stem
    for d in sorted(gen.iterdir()):
        if d.is_dir() and any(d.glob("*.kicad_sch")):
            return d
    return None


def _synth_check_failures(ws: Path | None) -> list[str]:
    """Failing synthesis-check lines (check name + each offender) from the build's
    synthesis_check.json, so a FAILED run shows WHAT broke -- e.g. the 9.12 ERC
    dangling-wire list -- right next to the schematic. [] if all passed or absent."""
    if ws is None:
        return []
    try:
        sc = json.loads((ws / ".kicraft" / "synthesis_check.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    out: list[str] = []
    for c in sc.get("checks") or []:
        if c.get("ok"):
            continue
        name = str(c.get("name", "check"))
        offenders = [str(o) for o in (c.get("offenders") or [])]
        if offenders:
            out.extend(f"{name}: {o}" for o in offenders)
        else:
            out.append(f"{name}: {c.get('message', 'failed')}")
    return out


def _schematic_sources(project_dir: Path, stem: str, token: str) -> list[tuple[str, str]]:
    """(url, filename) for the root schematic + every leaf sheet, root sheet first."""
    schs = list(project_dir.glob("*.kicad_sch"))
    root = f"{stem}.kicad_sch"
    schs.sort(key=lambda p: (p.name != root, p.name))
    return [(f"/project/{token}/{p.name}", p.name) for p in schs]


def _read_run_status(project_dir: Path) -> dict:
    """Place/route progress written by autoexperiment; {} if absent or mid-write."""
    try:
        data = json.loads(
            (project_dir / ".experiments" / "run_status.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _latest_render(renders: Path, kind: str) -> Path | None:
    """Newest `<kind>.png` or `round_*_<kind>.png` in a leaf's renders dir.

    The layout engine writes a stable `<kind>.png` plus a per-round
    `round_NNNN_<kind>.png`; the newest by mtime is the current preview."""
    cands = list(renders.glob(f"round_*_{kind}.png"))
    direct = renders / f"{kind}.png"
    if direct.is_file():
        cands.append(direct)
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def _leaf_meta(leaf_dir: Path) -> tuple[str | None, int | None, int | None]:
    """(sheet_name, trace_count, via_count) from a leaf's metadata.json, or Nones
    if it is absent or mid-write (metadata is finalized only when the leaf solves)."""
    try:
        m = json.loads((leaf_dir / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None, None, None
    return (m.get("sheet_name"),
            m.get("internal_trace_count"),
            m.get("internal_via_count"))


def _leaf_layout_progress(project_dir: Path, token: str) -> list[dict]:
    """Per-leaf placement/route progress for the Place/Route gallery.

    The layout engine solves each leaf (placement, then routing) and writes
    preview PNGs under `.experiments/subcircuits/<uuid>/renders/`. We surface the
    *placement* preview (produced before routing) so the build shows progress
    early, upgrading to the routed preview once it exists. Best-effort: dirs
    without a render yet are skipped. URLs carry `?v=<mtime>` so an overwritten
    render is re-fetched."""
    sub = project_dir / ".experiments" / "subcircuits"
    if not sub.is_dir():
        return []
    out: list[dict] = []
    for leaf_dir in sorted(sub.iterdir()):
        renders = leaf_dir / "renders"
        if not renders.is_dir():
            continue
        placement = _latest_render(renders, "pre_route_front_all")
        routed = _latest_render(renders, "routed_front_all")
        img = routed or placement
        if img is None:
            continue
        sheet_name, traces, vias = _leaf_meta(leaf_dir)
        rel = img.relative_to(project_dir).as_posix()
        out.append({
            "sheet_name": sheet_name or leaf_dir.name.split("__")[0][:8],
            "status": "Routed" if routed else "Placed",
            "url": f"/project/{token}/render/{rel}?v={int(img.stat().st_mtime)}",
            "traces": traces,
            "vias": vias,
        })
    out.sort(key=lambda d: d["sheet_name"])
    return out


def _render_synth_view(srcs: list[tuple[str, str]], stem: str) -> KiCanvasView:
    """Sheet selector + KiCanvas for the Synthesize tab. `srcs` is (url, filename),
    root-first. One button per sheet (root='Overview', leaves by name) swaps the
    embed to that single sheet via `set_sources`, so KiCanvas renders it directly.
    Defaults to the first leaf so a readable schematic (not the block-diagram root)
    is what the user sees first."""
    root_file = f"{stem}.kicad_sch"

    def _label(fname: str) -> str:
        if fname == root_file:
            return "Overview"
        return fname[:-len(".kicad_sch")] if fname.endswith(".kicad_sch") else fname

    default_idx = 1 if len(srcs) > 1 else 0  # first leaf when present, else root
    ui.label("Schematic").classes("text-xs font-medium").style("color:#94a3b8")
    selector = ui.row().classes("w-full flex-wrap gap-1")
    # Fill (nearly) the whole inspector column so the schematic is large enough to
    # read; the 460px offset leaves room for the labels + sheet selector above it,
    # and the floor keeps it usable (and never smaller than the old fixed height)
    # on short screens. The structured sheet/synthesis data scrolls below.
    view = KiCanvasView(
        [KiCanvasSource(srcs[default_idx][0], srcs[default_idx][1])],
        height="", style="height:calc(100vh - 460px);min-height:360px")
    with selector:
        for url, fname in srcs:
            ui.button(_label(fname),
                      on_click=lambda u=url, f=fname:
                      view.set_sources([KiCanvasSource(u, f)])) \
                .props("flat dense no-caps").classes("text-xs")
    return view


def _render_leaf_gallery(prog: list[dict], run_status: dict) -> None:
    """Per-leaf placement/route thumbnails for the Place/Route tab, so the build
    communicates progress (placement renders appear before routing). Built inside
    the place_route view_slot; the caller clears the slot before each rebuild."""
    h = run_status.get("hierarchy") or {}
    lw = h.get("leaf_workers") or {}
    total = h.get("leaf_total")
    action = h.get("current_action") or run_status.get("current_action") or ""
    head = "Leaf layout progress"
    if total:
        head += f"  ({lw.get('completed') or 0}/{total} leaves)"
    ui.label(head).classes("text-xs font-medium").style("color:#94a3b8")
    if action:
        ui.label(str(action)).classes("text-xs").style("color:#64748b")
    with ui.row().classes("w-full flex-wrap gap-2"):
        for d in prog:
            chip = "#34d399" if d["status"] == "Routed" else "#fbbf24"
            with ui.column().classes("gap-0 items-center").style("width:118px"):
                ui.image(d["url"]).classes("w-full rounded border border-slate-700") \
                    .style("background:#0b1120")
                ui.label(d["sheet_name"]).classes("text-xs truncate w-full text-center") \
                    .style("color:#cbd5e1")
                ui.label(d["status"]).classes("text-xs").style(f"color:{chip}")


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_state_json(ws: Path) -> dict:
    """The progressively-built ConversationState (each stage commits a slot); {}
    if absent or mid-write."""
    try:
        data = json.loads((ws / ".kicraft" / "state.json").read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _endpoints_str(eps) -> str:
    return ", ".join(f"{p.get('ref')}.{p.get('pin')}" for p in (eps or []))


def _build_lines_for(stage: str, lines: list[str]) -> list[str]:
    """Build-log lines that belong to a given build sub-phase, by marker."""
    def sub(t: str) -> str | None:
        if "1/5" in t or "synthesized " in t:
            return "synthesize"
        if "2/5" in t or "3/5" in t or "4/5" in t:
            return "place_route"
        if "5/5" in t:
            return "fab"
        return None
    out, cur = [], None
    for ln in lines:
        s = sub(ln)
        if s:
            cur = s
        if cur == stage:
            out.append(ln)
    return out


# LCSC part id baked into a vendored symbol/footprint name (e.g.
# "USBLC6-2SC6_C2687116"); the negative lookbehind keeps it off footprint tokens
# like "C_0805" where the C is a package-class prefix, not a catalogue id.
_LCSC_ID_RE = re.compile(r"(?<![A-Za-z0-9])C\d{4,}")
# Imperial package size in a footprint leaf, e.g. the 0805 in "C_0805_2012Metric".
_FP_SIZE_RE = re.compile(r"_(\d{3,4})(?:_|$)")


def _resolve_part(p: dict) -> tuple[str, str] | None:
    """How to find this part at a vendor, as ``(kind, query)``: an LCSC id baked
    into the symbol/footprint name ("id", vendored easyeda parts); else the
    manufacturer part number ("mpn"); else a keyword from value + package size
    ("kw", generic passives). None when there is nothing to go on. Shared by the
    vendor link and the price lookup so both point at the same part."""
    sym = p.get("symbol") or ""
    fp = p.get("footprint") or ""
    m = _LCSC_ID_RE.search(sym) or _LCSC_ID_RE.search(fp)
    if m:
        return ("id", m.group(0))
    mpn = (p.get("mpn") or "").strip()
    if mpn:
        return ("mpn", mpn)
    val = (p.get("value") or "").strip()
    size = _FP_SIZE_RE.search(fp.split(":", 1)[-1])
    terms = " ".join(t for t in (val, size.group(1) if size else "") if t)
    return ("kw", terms) if terms else None


def _vendor_cell(p: dict, prices: dict | None = None) -> dict | str:
    """A clickable LCSC link for one BOM part. When the part has been priced, link
    to the exact product we priced (its LCSC id) so the link and the cost column
    always agree and the price is verifiable; otherwise an LCSC id -> the product
    page, an MPN or generic passive -> an LCSC search. "" when nothing resolves."""
    r = _resolve_part(p)
    if not r:
        return ""
    kind, q = r
    if prices is not None:
        res = prices.get(f"{kind}:{q}")
        if isinstance(res, dict) and res.get("lcsc"):
            cid = res["lcsc"]
            return {"text": cid, "href": f"https://www.lcsc.com/product-detail/{cid}.html"}
    if kind == "id":
        return {"text": q, "href": f"https://www.lcsc.com/product-detail/{q}.html"}
    return {"text": q if kind == "mpn" else "search",
            "href": "https://www.lcsc.com/search?q=" + quote(q)}


# ---- BOM part pricing (live JLCPCB/LCSC lookups, cached) ---------------------
# Resolved unit prices keyed by a part's lookup key ("id:C123" / "mpn:.." /
# "kw:.."). Shared process-wide (a key like "kw:5.1k 0402" is project-independent)
# and persisted per project to .kicraft/bom_prices.json so a reopen is instant. A
# value is a dict (priced) or None (looked up, no match); a missing key means "not
# fetched yet" -> shown as "..." while a background fetch runs.
_PRICE_CACHE: dict[str, dict | None] = {}
_PRICE_INFLIGHT: set[str] = set()
_PRICE_LOCK = threading.Lock()
_PRICE_FILE = "bom_prices.json"
_FETCH_ERROR = object()  # sentinel: fetch raised; don't cache, allow a later retry
# Bump when _pick_price changes so persisted prices from the old logic are dropped
# and re-fetched (v2: cheapest-in-stock for MPN, not first-in-stock).
_PRICE_SCHEMA = 2


def _price_key(p: dict) -> str | None:
    r = _resolve_part(p)
    return f"{r[0]}:{r[1]}" if r else None


def _pick_price(kind: str, query: str, results: list[dict]) -> dict | None:
    """Choose one JLCPCB search result and pull its unit price. For an LCSC id the
    exact id wins (it names a specific part); otherwise the cheapest in-stock match.
    Cheapest (not first) matters because a vague MPN/keyword pulls in false
    positives: e.g. "USB1046" returns both $4+ TI TUSB1046 muxes and the $0.84 GCT
    USB connector, and the connector is the one we want. Returns ``{"unit_price",
    "lcsc","stock"}`` or None when nothing usable came back. Pure: no network."""
    def price_of(r):
        try:
            return float(r.get("price"))
        except (TypeError, ValueError):
            return None
    priced = [r for r in results if (price_of(r) or 0) > 0]
    if not priced:
        return None
    pool = [x for x in priced if (x.get("stock") or 0) > 0] or priced
    if kind == "id":
        r = next((x for x in pool if str(x.get("lcsc", "")).upper() == query.upper()),
                 min(pool, key=price_of))
    else:
        r = min(pool, key=price_of)  # cheapest in-stock for both MPN and keyword
    return {"unit_price": price_of(r), "lcsc": r.get("lcsc"), "stock": r.get("stock")}


def _search_jlcpcb(query: str) -> list[dict]:
    """JLCPCB/LCSC keyword search via easyeda2kicad. Network; may raise."""
    from easyeda2kicad.easyeda.easyeda_api import EasyedaApi
    res = EasyedaApi().search_jlcpcb_components(keyword=query, page_size=10) or {}
    return res.get("results") or []


def _fetch_price(key: str) -> dict | None:
    kind, _, query = key.partition(":")
    return _pick_price(kind, query, _search_jlcpcb(query))


def _safe_fetch(key: str):
    try:
        return _fetch_price(key)
    except Exception:
        return _FETCH_ERROR


def _fmt_price(x: float) -> str:
    return f"${x:.4f}"


def _fmt_total(x: float) -> str:
    return f"${x:,.2f}" if x >= 0.10 else f"${x:.4f}"


def _load_price_cache(ws: Path) -> None:
    """Merge a project's persisted prices into the process cache (best-effort).
    Files written by an older pricing schema (or the pre-schema flat format) are
    ignored so a _pick_price change re-fetches instead of serving stale prices."""
    try:
        data = json.loads((ws / ".kicraft" / _PRICE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict) or data.get("_schema") != _PRICE_SCHEMA:
        return
    with _PRICE_LOCK:
        for k, v in (data.get("prices") or {}).items():
            if k not in _PRICE_CACHE:
                _PRICE_CACHE[k] = v if isinstance(v, dict) else None


def _save_price_cache(ws: Path, keys: set[str]) -> None:
    """Persist this project's resolved keys (tagged with the pricing schema) so a
    reopen/restart is instant."""
    with _PRICE_LOCK:
        snap = {k: _PRICE_CACHE[k] for k in keys if k in _PRICE_CACHE}
    try:
        d = ws / ".kicraft"
        d.mkdir(parents=True, exist_ok=True)
        (d / _PRICE_FILE).write_text(
            json.dumps({"_schema": _PRICE_SCHEMA, "prices": snap}, indent=2),
            encoding="utf-8")
    except OSError:
        pass


def _ensure_bom_prices(parts: list[dict], ws: str | None, state: dict) -> None:
    """Kick a background fetch for any not-yet-priced parts, then bump
    ``state['prices_rev']`` so the render loop re-renders the BOM with the prices.
    No-op when everything is already cached or in flight."""
    keys = {k for p in parts if (k := _price_key(p))}
    if not keys:
        return
    with _PRICE_LOCK:
        todo = [k for k in keys if k not in _PRICE_CACHE and k not in _PRICE_INFLIGHT]
        _PRICE_INFLIGHT.update(todo)
    if not todo:
        return

    def work():
        try:
            with ThreadPoolExecutor(max_workers=6) as ex:
                for k, r in zip(todo, ex.map(_safe_fetch, todo)):
                    if r is not _FETCH_ERROR:
                        with _PRICE_LOCK:
                            _PRICE_CACHE[k] = r
        finally:
            with _PRICE_LOCK:
                _PRICE_INFLIGHT.difference_update(todo)
            if ws:
                _save_price_cache(Path(ws), keys)
            state["prices_rev"] = state.get("prices_rev", 0) + 1

    threading.Thread(target=work, daemon=True).start()


def _inspector_spec(stage: str, sj: dict, run_status: dict, project_dir: Path | None,
                    build_lines: list[str], *, prices: dict | None = None) -> list[dict]:
    """Build the structured project-state spec for a stage's inspector window.

    Pure-data stages read their committed slot from `sj` (state.json); the build
    stages read filesystem signals (sheets, run_status, build log). Returns the
    section list consumed by StagePanel.set_inspector; [] means "nothing yet".

    `prices` is the part-price lookup the BOM cost column reads (defaults to the
    process-wide `_PRICE_CACHE`; the demo passes a canned map so it needs no
    network).
    """
    prices = _PRICE_CACHE if prices is None else prices
    if stage == "intent":
        sl = sj.get("intent") or {}
        if not sl:
            return []
        secs = [{"type": "kv", "title": "Intent", "rows": [
            ("goal", sl.get("goal", "")),
            ("expertise", sl.get("inferred_expertise", "")),
            ("project_stem", sj.get("project_stem", ""))]}]
        if sl.get("constraints"):
            secs.append({"type": "list", "title": "Constraints", "items": sl["constraints"]})
        if sl.get("named_parts"):
            secs.append({"type": "list", "title": "Named parts", "items": sl["named_parts"]})
        if sl.get("assumptions"):
            secs.append({"type": "list", "title": "Assumptions", "items": sl["assumptions"]})
        return secs

    if stage == "functional_spec":
        sl = sj.get("functional_spec") or {}
        if not sl:
            return []
        secs = [{"type": "table", "title": "Functional blocks",
                 "columns": ["name", "category", "purpose"],
                 "rows": [[b.get("name"), b.get("category"), b.get("purpose")]
                          for b in sl.get("blocks", [])]}]
        if sl.get("connections"):
            secs.append({"type": "table", "title": "Block connections",
                         "columns": ["from", "to", "signal"],
                         "rows": [[c.get("from_block"), c.get("to_block"), c.get("signal_type")]
                                  for c in sl["connections"]]})
        if sl.get("assumptions"):
            secs.append({"type": "list", "title": "Assumptions", "items": sl["assumptions"]})
        return secs

    if stage == "architecture":
        sl = sj.get("architecture") or {}
        if not sl:
            return []
        secs = [{"type": "table", "title": "Sheets", "columns": ["name", "stem", "function"],
                 "rows": [[s.get("name"), s.get("stem"), s.get("function")]
                          for s in sl.get("sheets", [])]}]
        if sl.get("power_nets"):
            secs.append({"type": "list", "title": "Power nets", "items": sl["power_nets"]})
        if sl.get("rail_voltages"):
            secs.append({"type": "kv", "title": "Rail voltages",
                         "rows": [(k, f"{v} V") for k, v in sl["rail_voltages"].items()]})
        if sl.get("topologies"):
            secs.append({"type": "kv", "title": "Topologies",
                         "rows": list(sl["topologies"].items())})
        secs.append({"type": "kv", "title": "Misc", "rows": [
            ("mcu_present", sl.get("mcu_present", False)),
            ("comms", ", ".join(sl.get("comms_protocols", [])) or "(none)"),
            ("inter-sheet nets", len(sl.get("inter_sheet_nets", [])))]})
        return secs

    if stage == "bom":
        sl = sj.get("bom") or {}
        parts = sl.get("parts") or []
        if not parts:
            return []
        rows, total, priced, pending = [], 0.0, 0, False
        for p in parts:
            key = _price_key(p)
            if key is None:
                cost = "n/a"
            elif key in prices:
                res = prices[key]
                if isinstance(res, dict):
                    total += res["unit_price"]
                    priced += 1
                    cost = _fmt_price(res["unit_price"])
                else:
                    cost = "n/a"  # looked up, no match
            else:
                cost = "..."  # fetch in flight
                pending = True
            rows.append([p.get("ref"), p.get("value"), cost, _vendor_cell(p, prices),
                         p.get("footprint"), p.get("sheet"), p.get("symbol")])
        if pending and priced == 0:
            total_txt, note = "pricing...", "fetching live JLCPCB prices..."
        else:
            total_txt = _fmt_total(total)
            note = f"est. = cheapest in-stock JLCPCB match ({priced}/{len(parts)} priced)"
            if pending:
                note = f"fetching live JLCPCB prices... ({priced}/{len(parts)} so far)"
        secs = [{"type": "kv", "title": "Summary", "rows": [("parts", len(parts))]},
                {"type": "table", "title": "Parts",
                 "columns": ["ref", "value", "cost", "vendor", "footprint", "sheet", "symbol"],
                 "rows": rows,
                 "foot": [["", "TOTAL (est.)", total_txt, "", "", "", ""]],
                 "note": note}]
        return secs

    if stage == "wiring":
        sl = sj.get("bom") or {}
        conns = sl.get("connections") or []
        ncs = sl.get("no_connect_pins") or []
        if not conns and not ncs:
            return []
        secs = [{"type": "kv", "title": "Summary",
                 "rows": [("nets", len(conns)), ("no-connect pins", len(ncs))]}]
        if conns:
            secs.append({"type": "table", "title": "Connections",
                         "columns": ["net", "sheet", "endpoints"],
                         "rows": [[c.get("net_name"), c.get("sheet"),
                                   _endpoints_str(c.get("endpoints"))] for c in conns]})
        if ncs:
            secs.append({"type": "table", "title": "No-connect pins",
                         "columns": ["ref", "pin"],
                         "rows": [[p.get("ref"), p.get("pin")] for p in ncs]})
        return secs

    if stage == "synthesize":
        secs = []
        if project_dir is not None:
            sheets = sorted(p.name for p in project_dir.glob("*.kicad_sch"))
            if sheets:
                secs.append({"type": "list", "title": "Schematic sheets", "items": sheets})
            fails = _synth_check_failures(project_dir.parent.parent)
            if fails:  # WHY a failed run is failed -- shown even after reopen (no log)
                secs.append({"type": "list", "title": "Checks failed", "items": fails})
        log = _build_lines_for("synthesize", build_lines)
        if log:
            secs.append({"type": "list", "title": "Synthesis", "items": log})
        return secs

    if stage == "place_route":
        secs = []
        scalars = [(k, v) for k, v in (run_status or {}).items()
                   if isinstance(v, (str, int, float, bool))]
        if scalars:
            secs.append({"type": "kv", "title": "Run status", "rows": scalars})
        log = _build_lines_for("place_route", build_lines)
        if log:
            secs.append({"type": "list", "title": "Place / route", "items": log})
        return secs

    if stage == "fab":
        secs = []
        arts = sj.get("artifacts") or {}
        if arts:
            secs.append({"type": "kv", "title": "Artifacts", "rows": [
                ("status", arts.get("status", "")),
                ("fab package", "ready" if arts.get("fab_zip") else "pending")]})
        log = _build_lines_for("fab", build_lines)
        if log:
            secs.append({"type": "list", "title": "Fab export", "items": log})
        return secs

    return []


def _persist_project(ws: Path | None, state: dict) -> None:
    """Copy the run's durable artifacts out of the tempdir and finalize the row.

    Best-effort: a persistence failure must never crash the worker. Captures the
    brief, the full event stream (events.jsonl, 100% of input), the committed
    state.json, and the generated KiCad tree + zip under
    projects_dir/<user_id>/<project_id>/, then records the projects-table row.
    """
    pid = state.get("project_id")
    uid = state.get("user_id")
    if not pid or not uid:
        return
    store = _store()
    # An explicit status (e.g. "awaiting_input" when a run parks on a question)
    # wins; otherwise derive ok/failed from the run outcome.
    status = state.get("status") or ("ok" if state.get("ok") else "failed")
    stem = state.get("stem")
    dir_path = None
    zip_path = None
    try:
        base = store.projects_dir / str(uid) / str(pid)
        base.mkdir(parents=True, exist_ok=True)
        dir_path = str(base)
        (base / "brief.txt").write_text(state.get("brief", "") or "", encoding="utf-8")
        with (base / "events.jsonl").open("w", encoding="utf-8") as f:
            for ev in state.get("events", []):
                f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        if ws is not None:
            # Save the WHOLE .kicraft/ (state.json + parts/ + check files) so a
            # later resume or edit-and-rebuild has the fetched LCSC bundles, not
            # just the state. Keep a top-level state.json copy too, for readers
            # that expect it and for legacy projects predating the kicraft/ tree.
            kdir = ws / ".kicraft"
            if kdir.is_dir():
                kdst = base / "kicraft"
                shutil.rmtree(kdst, ignore_errors=True)
                shutil.copytree(kdir, kdst)
            sj = ws / ".kicraft" / "state.json"
            if sj.is_file():
                shutil.copy2(sj, base / "state.json")
            gen = ws / "generated"
            if gen.is_dir():
                dst = base / "generated"
                shutil.rmtree(dst, ignore_errors=True)
                shutil.copytree(gen, dst)
            src_zip = state.get("zip")
            if src_zip and Path(src_zip).is_file():
                zip_path = str(base / "kicraft_project.zip")
                shutil.copy2(src_zip, zip_path)
                state["zip"] = zip_path  # serve downloads from the durable copy
    except Exception as e:  # never crash the worker on persistence
        state.setdefault("events", []).append(
            {"kind": "build_log", "text": f"persist error: {e}"})
    finally:
        try:
            store.finish_project(pid, status, stem=stem, cost_usd=state.get("spend"),
                                 dir_path=dir_path, zip_path=zip_path)
        except Exception:
            pass


def _rehydrate_workspace(project) -> Path:
    """Recreate a working tempdir from a saved project's durable .kicraft/ (state +
    fetched parts) and generated tree, so the session can resume, edit, or rebuild
    against it. Falls back to the top-level state.json for legacy projects that
    predate the saved kicraft/ tree."""
    ws = Path(tempfile.mkdtemp(prefix="kicraft_resume_"))
    base = Path(project.dir_path) if project.dir_path else None
    if base and (base / "kicraft").is_dir():
        shutil.copytree(base / "kicraft", ws / ".kicraft")
    elif base and (base / "state.json").is_file():
        (ws / ".kicraft").mkdir(parents=True, exist_ok=True)
        shutil.copy2(base / "state.json", ws / ".kicraft" / "state.json")
    if base and (base / "generated").is_dir():
        shutil.copytree(base / "generated", ws / "generated")
    return ws


def _anno_kind(anno) -> tuple[str, list | None]:
    """Classify a Pydantic field annotation for the structured slot editor: one of
    'str' / 'bool' / 'list_str' / 'enum' / 'json' (the catch-all for nested types)."""
    origin = typing.get_origin(anno)
    args = typing.get_args(anno)
    if origin in (typing.Union, types.UnionType):  # unwrap Optional[X]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return _anno_kind(non_none[0])
    if anno is str:
        return ("str", None)
    if anno is bool:
        return ("bool", None)
    if origin is list and args and args[0] is str:
        return ("list_str", None)
    if origin is typing.Literal:
        return ("enum", [str(a) for a in args])
    return ("json", None)


def _render_slot_form(model, slot: dict):
    """Render one editable widget per model field; return getter() -> slot dict.
    Simple fields (str / bool / list[str] / enum) get native widgets; nested
    fields (lists of objects, dicts) fall back to a JSON box. The getter may raise
    json.JSONDecodeError if a JSON field is left malformed."""
    getters: dict = {}
    for name, field in model.model_fields.items():
        kind, choices = _anno_kind(field.annotation)
        cur = slot.get(name)
        with ui.row().classes("w-full items-start gap-2"):
            ui.label(name).classes("text-xs w-40 pt-2").style("color:#94a3b8")
            if kind == "str":
                w = ui.input(value=("" if cur is None else str(cur))).classes("flex-grow")
                getters[name] = lambda w=w: (w.value or "")
            elif kind == "bool":
                w = ui.switch(value=bool(cur))
                getters[name] = lambda w=w: bool(w.value)
            elif kind == "list_str":
                ui.label("(one per line)").classes("text-xs pt-2").style("color:#64748b")
                w = ui.textarea(value="\n".join(cur or [])).props("rows=3").classes("flex-grow")
                getters[name] = lambda w=w: [s.strip() for s in (w.value or "").splitlines()
                                             if s.strip()]
            elif kind == "enum":
                val = cur if cur in (choices or []) else (choices[0] if choices else None)
                w = ui.select(choices or [], value=val).classes("flex-grow")
                getters[name] = lambda w=w: w.value
            else:  # nested model/list/dict -> JSON box
                w = ui.textarea(value=(json.dumps(cur, indent=2) if cur is not None else "")) \
                    .props("rows=4").classes("flex-grow text-xs")
                getters[name] = lambda w=w: (json.loads(w.value) if (w.value or "").strip()
                                             else None)
    return lambda: {n: g() for n, g in getters.items()}


def _run_design(state: dict, stages, answers=None, instruction=None) -> None:
    """Drive `stages` for this page's session, streaming progress into `state`,
    then (on success) run the deterministic build. Shared by the initial design,
    resume/continue, edit-and-rerun, and answering a parked question.

    The thread only mutates `state` (appends progress events, sets flags); every
    NiceGUI element update happens in the page render timer (elements must not be
    touched off the UI context). `answers` / `instruction` apply to the first of
    `stages` (the stage being resumed or edited).
    """
    ws = (Path(state["ws"]) if state.get("ws")
          else Path(tempfile.mkdtemp(prefix="kicraft_web_")))
    state["ws"] = str(ws)
    state["status"] = None  # reset; set to awaiting_input only if we park
    # Stamp every model call of this run with a stable id so the spend ledger can
    # attribute cost per run/stage (see kicraft.cli.web_cost_report).
    run_id = f"p{state.get('project_id')}-{int(time.time())}"

    def progress(ev):
        state["events"].append(ev)

    try:
        res = run_session(ws, state.get("brief", ""), stages, answers=answers,
                          instruction=instruction, progress=progress, run_id=run_id)
        if res.get("guard"):
            state["spend"] = _project_spend_usd(state.get("project_id"))

        if res["status"] == "awaiting_input":
            # Park: the run is saved as awaiting_input and the question surfaces in
            # the UI; the user can answer now or reopen the project later.
            state["status"] = "awaiting_input"
            state["awaiting_input"] = True
            state["questions"] = res.get("questions") or []
            state["ok"] = None
            return

        state["awaiting_input"] = False
        state["questions"] = []
        if res["status"] != "ok":
            state["ok"] = False
            return

        # Deterministic (zero-LLM) build: synthesize -> place -> route -> verify ->
        # fab. `build` re-runs synthesize first, so the schematic appears as soon
        # as that step writes the sheets.
        stem = _read_project_stem(ws)
        if stem:
            project_dir = ws / "generated" / stem
            state["stem"] = stem
            state["project_dir"] = str(project_dir)
            state["token"] = _register_project_dir(project_dir)

        def _run_build() -> int:
            progress({"kind": "build_start"})
            proc = subprocess.Popen(
                KICRAFT + ["build", ".kicraft/state.json", "generated", "--no-archive"],
                cwd=str(ws), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1)
            started = time.monotonic()
            for line in proc.stdout or []:
                progress({"kind": "build_log", "text": line.rstrip()[:500]})
                if time.monotonic() - started > 1800:  # hard wall-clock bound
                    proc.kill()
                    progress({"kind": "build_log", "text": "[build exceeded 30m, killed]"})
                    break
            rc = proc.wait()
            progress({"kind": "build_done", "ok": rc == 0})
            return rc

        rc = _run_build()
        # Bounded ERC recovery: build fails (exit 5) at the §9.12 ERC gate when the
        # wiring slot leaves a real electrical error. Feed the concrete ERC errors
        # back into ONE wiring re-drive, then rebuild once. Capped at a single pass
        # (a flag, not a loop) so recovery can never run away on cost.
        if rc != 0 and not state.get("erc_recovered"):
            offenders = _erc_offenders(ws)
            if offenders:
                state["erc_recovered"] = True
                progress({"kind": "build_log",
                          "text": f"[erc-recover] {len(offenders)} ERC error(s); "
                                  "re-driving wiring once to fix them"})
                instr = ("The synthesized board failed KiCad ERC with the errors below. "
                         "Adjust connections / no_connect_pins to resolve them, keeping "
                         "every other net consistent:\n- " + "\n- ".join(offenders[:20]))
                rr = run_session(ws, state.get("brief", ""), ["wiring"],
                                 instruction=instr, progress=progress, run_id=run_id)
                if rr.get("guard"):
                    state["spend"] = _project_spend_usd(state.get("project_id"))
                if rr.get("status") == "ok":
                    rc = _run_build()
        if rc != 0:
            state["ok"] = False
            return

        state["pcb_ready"] = True
        state["zip"] = _zip_generated(ws)
        state["ok"] = bool(state["zip"])
    except Exception as e:  # surface, never crash the UI thread
        progress({"kind": "build_log", "text": f"error: {e}"})
        state["ok"] = False
    finally:
        _persist_project(ws, state)
        state["done"] = True
        state["running"] = False


def _design_worker(brief: str, state: dict) -> None:
    """Initial design from a brief: a fresh workspace, all schematic stages, build."""
    state["brief"] = brief
    state["ws"] = None  # force a fresh tempdir
    _run_design(state, list(DESIGN_STAGES))


# --------------------------------------------------------------------------- #
# Admin-only self-evaluation (Class-C deterministic + Class-J LLM judge)
# --------------------------------------------------------------------------- #
_GRADE_COLORS = {"A": "#16a34a", "B": "#65a30d", "C": "#ca8a04",
                 "D": "#ea580c", "F": "#dc2626"}


def _run_eval(project_dir: Path) -> dict:
    """Blocking: build the capped client from settings and self-evaluate one
    persisted project dir (Class-C from artifacts + Class-J via the LLM judge)."""
    from kicraft.eval.run_web import _project_times, evaluate_project

    from .client import CappedOpenRouterClient
    s = Settings.from_env()
    client = CappedOpenRouterClient(s)
    judge_model = getattr(s, "eval_judge_model", None) or s.model
    started, finished = _project_times(project_dir, s.users_db_path)
    return evaluate_project(project_dir, client, judge_model=judge_model,
                            ledger_path=s.ledger_path, started_at=started,
                            finished_at=finished)


def _render_scorecard(container, report: dict) -> None:
    """Render a self-eval report.json into `container` (called in a UI context)."""
    container.clear()
    s = report.get("score") or {}
    j = report.get("judge") or {}
    m = report.get("metrics") or {}
    tu = m.get("token_usage") or {}
    with container:
        with ui.row().classes("items-center gap-3"):
            grade = s.get("grade")
            if grade:
                ui.label(grade).classes("text-3xl font-bold") \
                    .style(f"color:{_GRADE_COLORS.get(grade, '#94a3b8')}")
                ui.label(f"{s.get('final')} / 100   {s.get('verdict') or ''}") \
                    .classes("text-lg").style("color:#e2e8f0")
            else:
                ui.label("Class-C only").classes("text-lg font-bold") \
                    .style("color:#94a3b8")
            ui.label(f"rubric v{report.get('rubric_version')}") \
                .classes("text-xs").style("color:#64748b")
        if s.get("note"):
            ui.label(s["note"]).classes("text-xs").style("color:#94a3b8")
        ui.label(
            f"synth={m.get('synthesis_status')}  "
            f"ERC={m.get('erc_errors')}e/{m.get('erc_warnings')}w  "
            f"latency={m.get('latency_min')}min  "
            f"tokens={tu.get('total_tokens', '-')}  "
            f"est=${tu.get('estimated_cost_usd', '-')}  "
            f"judge={j.get('model') or '-'} "
            f"({'ok' if j.get('ok') else (j.get('error') or 'n/a')})"
        ).classes("text-xs font-mono").style("color:#94a3b8")

        with ui.row().classes("w-full items-center gap-2 text-xs font-bold mt-1") \
                .style("color:#64748b"):
            ui.label("dimension").style("width:220px")
            ui.label("cls").style("width:26px")
            ui.label("wt").style("width:26px")
            ui.label("lvl").style("width:26px")
            ui.label("pts").style("width:38px")
            ui.label("rationale").classes("flex-1")
        for did, v in (report.get("dimensions") or {}).items():
            lvl = v.get("level")
            pts = f"{v['weight'] * lvl / 4:.1f}" if lvl is not None else "-"
            with ui.row().classes("w-full items-start gap-2 text-xs"):
                ui.label(did).style("width:220px;color:#e2e8f0")
                ui.label(v.get("class")).style(
                    f"width:26px;color:{'#60a5fa' if v.get('class') == 'C' else '#f0abfc'}")
                ui.label(str(v.get("weight"))).style("width:26px;color:#cbd5e1")
                ui.label("-" if lvl is None else str(lvl)).style("width:26px;color:#cbd5e1")
                ui.label(pts).style("width:38px;color:#cbd5e1")
                ui.label(v.get("rationale") or "").classes("flex-1").style("color:#94a3b8")

        gates = (report.get("gates") or {}).get("triggered") or []
        if gates:
            ui.label("Gates capped: "
                     + ", ".join(f"{g['id']}≤{g['cap']}" for g in gates)) \
                .classes("text-xs").style("color:#f87171")


def open_eval_dialog(project_dir, title: str) -> None:
    """Admin action: pop a dialog, run the self-eval in a worker thread, and render
    the scorecard when it lands (mirrors the design worker + render-timer idiom)."""
    if not is_admin(_current_user()):  # defense in depth; never trust UI-gating alone
        ui.notify("Admin access required.", color="warning")
        return
    project_dir = Path(project_dir)
    holder = {"done": False, "rendered": False, "report": None, "error": None}

    def worker():
        try:
            holder["report"] = _run_eval(project_dir)
        except Exception as e:  # surface in the dialog, never crash the page
            holder["error"] = str(e)
        finally:
            holder["done"] = True

    with ui.dialog() as dlg, ui.card().classes("w-[780px] max-w-[95vw]") \
            .style("background:#0f172a;border:1px solid #1e293b"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label(f"Self-evaluation: {title}").classes("text-base font-bold") \
                .style("color:#e2e8f0")
            ui.button(icon="close", on_click=dlg.close).props("flat dense round")
        status = ui.row().classes("items-center gap-2")
        with status:
            ui.spinner(size="sm")
            ui.label("Scoring Class-C and grading Class-J (LLM judge)...") \
                .classes("text-sm").style("color:#94a3b8")
        body = ui.column().classes("w-full gap-1")

    def tick():
        if not holder["done"] or holder["rendered"]:
            return
        holder["rendered"] = True
        tmr.active = False
        status.clear()
        if holder["error"]:
            with body:
                ui.label(f"Evaluation failed: {holder['error']}") \
                    .classes("text-sm").style("color:#f87171")
        else:
            _render_scorecard(body, holder["report"])

    tmr = ui.timer(0.3, tick)
    threading.Thread(target=worker, daemon=True).start()
    dlg.open()


def _legal_footer() -> None:
    """Public links to the Terms and Privacy Policy (shown on auth cards)."""
    with ui.row().classes("items-center gap-3 w-full justify-center"):
        ui.link("Terms of Service", "/terms", new_tab=True) \
            .classes("text-xs").style("color:#64748b")
        ui.link("Privacy Policy", "/privacy", new_tab=True) \
            .classes("text-xs").style("color:#64748b")


def _laforest_footer() -> None:
    """Subtle parent-company branding: a quiet link out to LaForest Labs,
    KiCraft's parent. Used on the app-shell footer and the auth cards."""
    with ui.row().classes("items-center justify-center gap-1 w-full"):
        ui.label("A").classes("text-xs").style("color:#475569")
        ui.link("LaForest Labs", "https://laforestlabs.com", new_tab=True) \
            .classes("text-xs").style("color:#64748b")
        ui.label("product").classes("text-xs").style("color:#475569")


def _legal_page(title: str, name: str) -> None:
    """Render a legal document. Public: no login required, so prospective users
    can read the Terms and Privacy Policy before signing up."""
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    with ui.column().classes("w-full max-w-3xl mx-auto p-6 gap-3"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(f"KiCraft {title}").classes("text-2xl font-bold text-white")
            ui.button("Back", icon="arrow_back",
                      on_click=lambda: ui.navigate.to("/login")) \
                .props("flat dense color=white")
        with ui.card().classes("w-full").style("background:#0f172a;border:1px solid #1e293b"):
            ui.markdown(_legal_markdown(name)).classes("w-full").style("color:#cbd5e1")


@ui.page("/terms")
def terms_page():
    _legal_page("Terms of Service", "terms-of-service")


@ui.page("/privacy")
def privacy_page():
    _legal_page("Privacy Policy", "privacy-policy")


@ui.page("/login")
def login_page(prompt: str = ""):
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    with ui.card().classes("absolute-center w-96") \
            .style("background:#0f172a;border:1px solid #1e293b"):
        ui.label("KiCraft").classes("text-2xl font-bold text-white")
        ui.label("Sign in to design a board.").classes("text-sm").style("color:#94a3b8")
        email = ui.input("Email").classes("w-full")
        pw = ui.input("Password", password=True, password_toggle_button=True).classes("w-full")

        def submit():
            user = _store().authenticate(email.value or "", pw.value or "")
            if user:
                app.storage.user["user_id"] = user.id
                app.storage.user["email"] = user.email
                app.storage.user["session_epoch"] = user.session_epoch
                if prompt:  # carry a sample's brief through into the workspace
                    app.storage.user["pending_prompt"] = prompt
                ui.navigate.to("/")
            else:
                ui.notify("Wrong email or password.", color="negative")

        pw.on("keydown.enter", submit)
        ui.button("Sign in", on_click=submit).classes("w-full")
        with ui.row().classes("w-full justify-end -mt-1"):
            ui.button("Forgot password?",
                      on_click=lambda: ui.navigate.to("/forgot")) \
                .props("flat dense no-caps").classes("text-xs")
        ui.separator().style("background:#1e293b")
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("New to KiCraft?").classes("text-xs").style("color:#94a3b8")
            ui.button("Create an account",
                      on_click=lambda: ui.navigate.to(
                          f"/signup?prompt={quote(prompt)}" if prompt else "/signup")) \
                .props("flat dense")
        _legal_footer()


@ui.page("/signup")
def signup_page(prompt: str = ""):
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    with ui.card().classes("absolute-center w-96") \
            .style("background:#0f172a;border:1px solid #1e293b"):
        ui.label("Create your account").classes("text-2xl font-bold text-white")
        ui.label("Free tier: one design per week. No credit card.") \
            .classes("text-sm").style("color:#94a3b8")
        if prompt:  # arrived from a sample card: show what they'll build first
            ui.label(f'You will start with: "{prompt}"') \
                .classes("text-xs") \
                .style("color:#cbd5e1;border-left:3px solid #60a5fa;padding-left:8px")
        email = ui.input("Email").classes("w-full")
        pw = ui.input("Password", password=True, password_toggle_button=True).classes("w-full")
        code = ui.input("Invite code", password=True).classes("w-full")

        agree = ui.checkbox("I agree to the Terms of Service and Privacy Policy") \
            .classes("text-sm")
        with ui.row().classes("items-center gap-3 -mt-2"):
            ui.link("Terms of Service", "/terms", new_tab=True) \
                .classes("text-xs").style("color:#60a5fa")
            ui.link("Privacy Policy", "/privacy", new_tab=True) \
                .classes("text-xs").style("color:#60a5fa")
        allow_training = ui.checkbox(
            "Allow KiCraft to use my designs to improve its AI models", value=True) \
            .classes("text-sm")
        ui.label("Optional, and changeable later in your profile.") \
            .classes("text-xs -mt-2").style("color:#64748b")

        def submit():
            want = _signup_code()
            if not want:
                ui.notify("Signup is not configured (set KICRAFT_SIGNUP_CODE).",
                          color="negative")
                return
            if not hmac.compare_digest((code.value or "").strip(), want):
                ui.notify("Invalid invite code.", color="negative")
                return
            if not agree.value:
                ui.notify("Please accept the Terms of Service and Privacy Policy "
                          "to create an account.", color="warning")
                return
            try:
                user = _store().create_user(
                    email.value or "", pw.value or "",
                    accepted_terms_version=LEGAL_VERSION,
                    allow_training=bool(allow_training.value))
            except ValueError as e:
                ui.notify(str(e), color="negative")
                return
            app.storage.user["user_id"] = user.id
            app.storage.user["email"] = user.email
            app.storage.user["session_epoch"] = user.session_epoch
            if prompt:  # carry a sample's brief through into the workspace
                app.storage.user["pending_prompt"] = prompt
            ui.navigate.to("/")

        pw.on("keydown.enter", submit)
        code.on("keydown.enter", submit)
        ui.button("Create account", on_click=submit).classes("w-full")
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("Already registered?").classes("text-xs").style("color:#94a3b8")
            ui.button("Sign in",
                      on_click=lambda: ui.navigate.to(
                          f"/login?prompt={quote(prompt)}" if prompt else "/login")) \
                .props("flat dense")
        ui.separator().style("background:#1e293b")
        _laforest_footer()


@ui.page("/forgot")
def forgot_page():
    """Public: request a password-reset link. Always shows the same neutral
    confirmation, so it never reveals whether an email is registered."""
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    with ui.card().classes("absolute-center w-96") \
            .style("background:#0f172a;border:1px solid #1e293b"):
        ui.label("Reset your password").classes("text-2xl font-bold text-white")
        ui.label("Enter your account email and we'll send a link to set a new "
                 "password.").classes("text-sm").style("color:#94a3b8")
        email = ui.input("Email").classes("w-full")

        def submit():
            addr = (email.value or "").strip()
            # Mint + send only for a real account, but never surface success or
            # failure differently: the message below is identical either way, so an
            # attacker can't probe which emails exist.
            if addr:
                try:
                    token = _store().create_reset_token(addr)
                    if token:
                        s = Settings.from_env()
                        url = f"{s.public_url}/reset?token={token}"
                        send_reset_email(s, addr, url, _RESET_TTL_MINUTES)
                except Exception:
                    pass
            ui.notify("If that email has an account, a reset link is on its way. "
                      "The link expires in an hour.", color="positive")

        email.on("keydown.enter", submit)
        ui.button("Send reset link", on_click=submit).classes("w-full")
        ui.separator().style("background:#1e293b")
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("Remembered it?").classes("text-xs").style("color:#94a3b8")
            ui.button("Back to sign in",
                      on_click=lambda: ui.navigate.to("/login")).props("flat dense")
        _legal_footer()


@ui.page("/reset")
def reset_page(token: str = ""):
    """Public: consume a reset token and set a new password. On success the user is
    auto-signed-in; the new session carries the bumped epoch, so every other
    session (an attacker's included) is evicted."""
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    user = _store().verify_reset_token(token)
    with ui.card().classes("absolute-center w-96") \
            .style("background:#0f172a;border:1px solid #1e293b"):
        if user is None:
            ui.label("Reset link invalid or expired") \
                .classes("text-2xl font-bold text-white")
            ui.label("Reset links are single-use and expire after an hour. Request "
                     "a fresh one to continue.").classes("text-sm").style("color:#94a3b8")
            ui.button("Request a new link",
                      on_click=lambda: ui.navigate.to("/forgot")).classes("w-full")
            _legal_footer()
            return
        ui.label("Choose a new password").classes("text-2xl font-bold text-white")
        ui.label(f"for {user.email}").classes("text-sm").style("color:#94a3b8")
        pw = ui.input("New password", password=True,
                      password_toggle_button=True).classes("w-full")
        pw2 = ui.input("Confirm new password", password=True).classes("w-full")

        def submit():
            if not (pw.value or ""):
                ui.notify("Please enter a new password.", color="warning")
                return
            if pw.value != pw2.value:
                ui.notify("Those passwords don't match.", color="warning")
                return
            updated = _store().consume_reset_token(token, pw.value)
            if updated is None:
                ui.notify("That reset link just expired. Please request a new one.",
                          color="negative")
                return
            app.storage.user["user_id"] = updated.id
            app.storage.user["email"] = updated.email
            app.storage.user["session_epoch"] = updated.session_epoch
            ui.notify("Password updated. You're signed in, and other devices have "
                      "been signed out.", color="positive")
            ui.navigate.to("/")

        pw.on("keydown.enter", submit)
        pw2.on("keydown.enter", submit)
        ui.button("Set new password and sign in", on_click=submit).classes("w-full")
        _legal_footer()


@ui.page("/consent")
def consent_page():
    """Re-consent gate: shown when a logged-in user's accepted Terms version is
    missing or older than the current LEGAL_VERSION (a fresh box account, or a
    terms bump). Index redirects here until they accept."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    if user.accepted_terms_version == LEGAL_VERSION:
        return RedirectResponse("/")
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    with ui.card().classes("absolute-center w-96") \
            .style("background:#0f172a;border:1px solid #1e293b"):
        ui.label("Please review our terms").classes("text-2xl font-bold text-white")
        ui.label("We've updated our Terms of Service and Privacy Policy. Please "
                 "accept to continue.").classes("text-sm").style("color:#94a3b8")
        with ui.row().classes("items-center gap-3"):
            ui.link("Terms of Service", "/terms", new_tab=True) \
                .classes("text-xs").style("color:#60a5fa")
            ui.link("Privacy Policy", "/privacy", new_tab=True) \
                .classes("text-xs").style("color:#60a5fa")
        agree = ui.checkbox("I agree to the Terms of Service and Privacy Policy") \
            .classes("text-sm")

        def accept():
            if not agree.value:
                ui.notify("Please accept to continue.", color="warning")
                return
            _store().record_consent(user.id, LEGAL_VERSION)
            ui.navigate.to("/")

        ui.button("Accept and continue", on_click=accept).classes("w-full")

        def logout():
            for k in ("user_id", "email"):
                app.storage.user.pop(k, None)
            ui.navigate.to("/login")

        ui.button("Log out", on_click=logout).props("flat dense color=white") \
            .classes("text-xs")


@ui.page("/profile")
def profile_page():
    """The user's profile: an account summary plus the privacy and data controls
    that used to sit in an expander on the main workspace. Reached by clicking
    your email in the header, which keeps the work GUI uncluttered."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    if user.accepted_terms_version != LEGAL_VERSION:
        return RedirectResponse("/consent")
    q = _store().quota_status(user)

    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")

    def logout():
        for k in ("user_id", "email"):
            app.storage.user.pop(k, None)
        ui.navigate.to("/login")

    with ui.header().classes("items-center justify-between") \
            .style("background:#0f172a;border-bottom:1px solid #1e293b"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("your profile").classes("text-sm").style("color:#94a3b8")
        with ui.row().classes("items-center gap-2"):
            if is_admin(user):
                ui.button("Admin", icon="admin_panel_settings",
                          on_click=lambda: ui.navigate.to("/admin")) \
                    .props("flat dense no-caps color=white").classes("text-xs") \
                    .tooltip("Admin dashboard")
            ui.button("Back to workspace", icon="arrow_back",
                      on_click=lambda: ui.navigate.to("/")) \
                .props("flat dense no-caps color=white").classes("text-xs")

    with ui.column().classes("w-full max-w-2xl mx-auto p-6 gap-4"):
        ui.label("Profile").classes("text-2xl font-bold text-white")

        with ui.card().classes("w-full gap-2") \
                .style("background:#0f172a;border:1px solid #1e293b"):
            ui.label("Account").classes("text-base font-semibold text-white")
            with ui.row().classes("items-center gap-2"):
                ui.icon("mail").style("color:#94a3b8")
                ui.label(user.email).classes("text-sm").style("color:#e2e8f0")
            period = "week" if q["window_days"] <= 7 else "month"
            with ui.row().classes("items-center gap-2"):
                ui.badge(q["label"], color="primary")
                if q.get("unlimited"):
                    ui.label("Unlimited designs (staff).") \
                        .classes("text-sm").style("color:#94a3b8")
                else:
                    ui.label(f"{q['remaining']} of {q['limit']} designs left this "
                             f"{period}.").classes("text-sm").style("color:#94a3b8")
            ui.label(f"Member since {user.created_at[:10]}.") \
                .classes("text-xs").style("color:#64748b")

        with ui.card().classes("w-full gap-2") \
                .style("background:#0f172a;border:1px solid #1e293b"):
            ui.label("Privacy & data").classes("text-base font-semibold text-white")
            train_sw = ui.switch(
                "Allow KiCraft to use my designs to improve its AI models",
                value=user.allow_training)
            train_sw.on_value_change(
                lambda e: _store().set_training_pref(user.id, bool(e.value)))
            with ui.row().classes("items-center gap-3"):
                ui.link("Terms of Service", "/terms", new_tab=True) \
                    .classes("text-xs").style("color:#60a5fa")
                ui.link("Privacy Policy", "/privacy", new_tab=True) \
                    .classes("text-xs").style("color:#60a5fa")
            ui.label("To export or delete all your data, contact "
                     "[CONTACT EMAIL].").classes("text-xs").style("color:#64748b")

        with ui.row().classes("w-full justify-end"):
            ui.button("Log out", icon="logout", on_click=logout) \
                .props("flat dense no-caps color=white").classes("text-xs")


# --------------------------------------------------------------------------- #
# Admin dashboard (stats/trends + user management). Gated by _require_admin();
# charts use the ECharts primitive bundled with NiceGUI (the web server ships
# under the `server` extra, which has no plotly -- that is a `gui`-extra dep).
# --------------------------------------------------------------------------- #
_CHART_AXIS = "#94a3b8"
_CHART_GRID = "#1e293b"


def _echart_bar(labels, values, *, title: str, color: str = "#60a5fa") -> dict:
    """ECharts bar-chart option dict. Pure (plain lists in, dict out) so it is
    unit-testable without a UI/connection context."""
    return {
        "backgroundColor": "transparent",
        "title": {"text": title, "textStyle": {"color": "#e2e8f0", "fontSize": 13}},
        "tooltip": {"trigger": "axis"},
        "grid": {"left": 50, "right": 16, "top": 44, "bottom": 56},
        "xAxis": {"type": "category", "data": list(labels),
                  "axisLabel": {"color": _CHART_AXIS, "fontSize": 10, "rotate": 45}},
        "yAxis": {"type": "value", "axisLabel": {"color": _CHART_AXIS},
                  "splitLine": {"lineStyle": {"color": _CHART_GRID}}},
        "series": [{"type": "bar", "data": list(values),
                    "itemStyle": {"color": color}}],
    }


def _echart_line(labels, values, *, title: str, color: str = "#34d399") -> dict:
    """ECharts line/area-chart option dict (pure; see _echart_bar)."""
    opt = _echart_bar(labels, values, title=title, color=color)
    opt["series"] = [{"type": "line", "data": list(values), "smooth": True,
                      "showSymbol": False, "itemStyle": {"color": color},
                      "areaStyle": {"color": color, "opacity": 0.15}}]
    return opt


def _echart_pie(pairs, *, title: str) -> dict:
    """ECharts donut option dict from (name, value) pairs (pure; see _echart_bar)."""
    return {
        "backgroundColor": "transparent",
        "title": {"text": title, "textStyle": {"color": "#e2e8f0", "fontSize": 13}},
        "tooltip": {"trigger": "item"},
        "legend": {"bottom": 0, "textStyle": {"color": _CHART_AXIS}},
        "series": [{"type": "pie", "radius": ["38%", "66%"], "center": ["50%", "46%"],
                    "data": [{"name": str(n), "value": v} for n, v in pairs],
                    "label": {"color": "#cbd5e1"}}],
    }


def _admin_header(active: str) -> None:
    """Shared header for the admin pages; `active` names the current sub-page."""
    with ui.header().classes("items-center justify-between") \
            .style("background:#0f172a;border-bottom:1px solid #1e293b"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label(f"admin · {active}").classes("text-sm").style("color:#94a3b8")
        with ui.row().classes("items-center gap-2"):
            ui.button("Overview", icon="insights",
                      on_click=lambda: ui.navigate.to("/admin")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Users", icon="group",
                      on_click=lambda: ui.navigate.to("/admin/users")) \
                .props("flat dense no-caps color=white").classes("text-xs")
            ui.button("Back to workspace", icon="arrow_back",
                      on_click=lambda: ui.navigate.to("/")) \
                .props("flat dense no-caps color=white").classes("text-xs")


def _admin_card_style() -> str:
    return "background:#0f172a;border:1px solid #1e293b;min-width:380px"


@ui.page("/admin")
def admin_overview_page():
    """Admin overview: headline stat cards + trend/distribution charts + top users.
    Read-only snapshot per load; the header's Overview button re-navigates to refresh."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect

    store = _store()
    stats = store.overview_stats()
    # Headline spend (Total spend card + Spend/day chart) comes from the SpendGuard
    # ledger, so it matches the OpenRouter dashboard exactly -- it counts every model
    # call, including non-project ones (eval/judge/smoketest). The per-user / per-
    # project / avg figures below stay project-attributed. Fall back to the project
    # numbers if the ledger can't be read.
    try:
        _guard = SpendGuard(Settings.from_env())
        ledger_total = _guard.spent_total()
        ledger_by_day = _guard.spent_by_day(30)
    except Exception:
        ledger_total = None
        ledger_by_day = store.spend_per_day(30)
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    _admin_header("overview")

    def money(x):
        return "—" if x is None else f"${x:,.2f}"

    def latency(x):
        if x is None:
            return "—"
        return f"{x / 60:.1f} min" if x >= 60 else f"{x:.0f} s"

    with ui.column().classes("w-full mx-auto p-4 gap-4").style("max-width:1400px"):
        ui.label("Admin dashboard").classes("text-2xl font-bold text-white")

        def card(label: str, value: str, hint: str = "") -> None:
            with ui.card().classes("gap-0 items-start") \
                    .style("background:#0f172a;border:1px solid #1e293b;min-width:150px"):
                ui.label(value).classes("text-2xl font-bold").style("color:#e2e8f0")
                ui.label(label).classes("text-xs").style("color:#94a3b8")
                if hint:
                    ui.label(hint).classes("text-xs").style("color:#64748b")

        w = stats["window_days"]
        with ui.row().classes("w-full flex-wrap gap-3"):
            card("Total users", str(stats["users_total"]), f"+{stats['users_new']} in {w}d")
            card("Admins", str(stats["admins"]))
            card("Total projects", str(stats["projects_total"]),
                 f"+{stats['projects_new']} in {w}d")
            spend_total = ledger_total if ledger_total is not None \
                else stats["spend_total_usd"]
            card("Total spend", money(spend_total),
                 f"${stats['spend_total_usd']:,.2f} on user projects")
            card("Avg / design", money(stats["spend_avg_usd"]))
            card("Avg latency", latency(stats["avg_latency_s"]))

        pp = store.projects_per_day(30)
        su = store.signups_per_day(30)
        sp = ledger_by_day  # ledger (all calls) -> matches the OpenRouter daily chart
        with ui.row().classes("w-full flex-wrap gap-4"):
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_line([d for d, _ in pp], [v for _, v in pp],
                                       title="Projects / day (30d)")) \
                    .classes("w-full").style("height:260px")
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_line([d for d, _ in su], [v for _, v in su],
                                       title="Signups / day (30d)", color="#60a5fa")) \
                    .classes("w-full").style("height:260px")
        with ui.row().classes("w-full flex-wrap gap-4"):
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_bar([d for d, _ in sp], [round(v, 2) for _, v in sp],
                                      title="Spend / day (30d)", color="#fbbf24")) \
                    .classes("w-full").style("height:260px")
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_pie(store.status_distribution(),
                                      title="Project status")) \
                    .classes("w-full").style("height:260px")
            with ui.card().classes("flex-1").style(_admin_card_style()):
                ui.echart(_echart_pie(store.tier_distribution(), title="User tiers")) \
                    .classes("w-full").style("height:260px")

        ui.label("Top users by projects") \
            .classes("text-base font-semibold text-white mt-2")
        top = sorted(store.users_with_project_counts(),
                     key=lambda r: r["project_count"], reverse=True)[:10]
        with ui.column().classes("w-full gap-1"):
            for r in top:
                with ui.row().classes("w-full items-center gap-3 text-xs") \
                        .style("border-top:1px solid #1e293b;padding:3px 0"):
                    ui.label(r["email"]).style("width:260px;color:#e2e8f0")
                    ui.badge(r["tier"], color="primary")
                    if r["role"] == "admin":
                        ui.badge("admin", color="purple")
                    ui.label(f"{r['project_count']} projects").style("color:#94a3b8")
                    ui.label(f"${r['spend_usd']:.2f}").style("color:#64748b")


@ui.page("/admin/users")
def admin_users_page():
    """User management: one row per user (tier, role, project_count, spend) with
    actions -- change tier, grant/revoke admin, issue a reset link, export JSON,
    delete. Every mutating handler re-checks is_admin() (defense in depth), and the
    self-demotion / last-admin guards keep the system from losing all its admins."""
    user, redirect = _require_admin()
    if redirect is not None:
        return redirect

    store = _store()
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    _admin_header("users")

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1400px"):
        ui.label("User management").classes("text-2xl font-bold text-white")
        search = ui.input(placeholder="Filter by email…").props("dense clearable") \
            .classes("w-72").style("color:#e2e8f0")
        container = ui.column().classes("w-full gap-0")

        def guard() -> bool:
            """Defense in depth: never trust the page-load gate for a mutation."""
            if not is_admin(_current_user()):
                ui.notify("Admin access required.", color="warning")
                return False
            return True

        def do_set_tier(email: str, value: str) -> None:
            if not guard():
                return
            try:
                store.set_tier(email, value)
                ui.notify(f"{email}: tier set to {value}.", color="positive")
            except ValueError as e:
                ui.notify(str(e), color="negative")
            build_users()

        def do_toggle_admin(row: dict) -> None:
            if not guard():
                return
            me = _current_user()
            making = row["role"] != "admin"
            if not making:
                if me is not None and row["id"] == me.id:
                    ui.notify("You can't revoke your own admin access.", color="warning")
                    return
                if store.count_role("admin") <= 1:
                    ui.notify("Refusing to remove the last admin.", color="warning")
                    return
            store.set_role(row["id"], "admin" if making else "user")
            ui.notify(f"{row['email']} is now {'an admin' if making else 'a user'}.",
                      color="positive")
            build_users()

        def do_reset_link(email: str) -> None:
            if not guard():
                return
            token = store.create_reset_token(email)
            if token is None:
                ui.notify("A reset link was issued moments ago; wait a minute and retry.",
                          color="warning")
                return
            url = f"{Settings.from_env().public_url}/reset?token={token}"
            with ui.dialog() as dlg, ui.card() \
                    .style("background:#0f172a;border:1px solid #1e293b;min-width:520px"):
                ui.label(f"Password-reset link for {email}") \
                    .classes("text-sm font-bold").style("color:#e2e8f0")
                ui.label(f"Valid ~{_RESET_TTL_SECONDS // 60} min, single use. "
                         "Relay it to the user out-of-band.") \
                    .classes("text-xs").style("color:#94a3b8")
                ui.input(value=url).props("readonly outlined dense") \
                    .classes("w-full").style("color:#e2e8f0")
                with ui.row().classes("w-full justify-end"):
                    ui.button("Close", on_click=dlg.close).props("flat dense")
            dlg.open()

        def do_export(uid: int) -> None:
            if not guard():
                return
            data = store.export_user(uid)
            if data is None:
                ui.notify("No such user.", color="negative")
                return
            payload = json.dumps(data, indent=2, default=str).encode("utf-8")
            ui.download(payload, f"kicraft_export_{uid}.json", "application/json")

        def do_delete(row: dict) -> None:
            if not guard():
                return
            me = _current_user()
            if me is not None and row["id"] == me.id:
                ui.notify("You can't delete your own account here.", color="warning")
                return
            if row["role"] == "admin" and store.count_role("admin") <= 1:
                ui.notify("Refusing to delete the last admin.", color="warning")
                return

            def confirm() -> None:
                if not guard():
                    dlg.close()
                    return
                store.delete_user(row["id"])
                dlg.close()
                ui.notify(f"Deleted {row['email']}.", color="positive")
                build_users()

            with ui.dialog() as dlg, ui.card() \
                    .style("background:#0f172a;border:1px solid #1e293b;min-width:420px"):
                ui.label(f"Delete {row['email']}?") \
                    .classes("text-base font-bold").style("color:#e2e8f0")
                ui.label("Removes their account, project rows, and stored files. "
                         "This is irreversible.").classes("text-xs").style("color:#f87171")
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Cancel", on_click=dlg.close).props("flat dense")
                    ui.button("Delete", color="negative", on_click=confirm).props("dense")
            dlg.open()

        def build_users() -> None:
            container.clear()
            me = _current_user()
            flt = (search.value or "").strip().lower()
            rows = store.users_with_project_counts()
            if flt:
                rows = [r for r in rows if flt in r["email"].lower()]
            with container:
                with ui.row().classes("w-full items-center gap-2 text-xs font-bold") \
                        .style("color:#64748b;padding:2px 0"):
                    ui.label("email").style("width:230px")
                    ui.label("tier").style("width:96px")
                    ui.label("role").style("width:70px")
                    ui.label("proj").style("width:48px")
                    ui.label("spend").style("width:64px")
                    ui.label("joined").style("width:84px")
                    ui.label("actions").classes("flex-1")
                if not rows:
                    ui.label("No users match.").classes("text-sm").style("color:#94a3b8")
                for r in rows:
                    is_admin_row = r["role"] == "admin"
                    is_self = me is not None and r["id"] == me.id
                    with ui.row().classes("w-full items-center gap-2 text-xs") \
                            .style("border-top:1px solid #1e293b;padding:4px 0"):
                        ui.label(r["email"] + ("  (you)" if is_self else "")) \
                            .style("width:230px;color:#e2e8f0")
                        ui.select({"free": "Free", "pro": "Pro", "max": "Max"},
                                  value=r["tier"],
                                  on_change=lambda e, em=r["email"]: do_set_tier(em, e.value)) \
                            .props("dense options-dense").style("width:96px")
                        ui.label(r["role"]).style(
                            f"width:70px;color:{'#a78bfa' if is_admin_row else '#64748b'}")
                        ui.label(str(r["project_count"])).style("width:48px;color:#cbd5e1")
                        ui.label(f"${r['spend_usd']:.2f}").style("width:64px;color:#cbd5e1")
                        ui.label((r["created_at"] or "")[:10]) \
                            .style("width:84px;color:#64748b")
                        with ui.row().classes("flex-1 gap-1 items-center"):
                            ui.button("Revoke" if is_admin_row else "Make admin",
                                      icon="remove_moderator" if is_admin_row
                                      else "admin_panel_settings",
                                      on_click=lambda row=r: do_toggle_admin(row)) \
                                .props("flat dense no-caps").classes("text-xs")
                            ui.button("Reset link", icon="link",
                                      on_click=lambda em=r["email"]: do_reset_link(em)) \
                                .props("flat dense no-caps").classes("text-xs")
                            ui.button("Export", icon="download",
                                      on_click=lambda uid=r["id"]: do_export(uid)) \
                                .props("flat dense no-caps").classes("text-xs") \
                                .tooltip("Account + project metadata as JSON "
                                         "(on-disk files via the CLI)")
                            ui.button("Delete", icon="delete", color="negative",
                                      on_click=lambda row=r: do_delete(row)) \
                                .props("flat dense no-caps").classes("text-xs")

        search.on_value_change(lambda: build_users())
        build_users()


@ui.page("/samples")
def samples_page():
    """Logged-in explorer for the showcase boards: open any sample's real schematic
    and routed PCB in KiCanvas, or send its brief into the workspace as a starting
    point. Same login + consent gating as the rest of the app. Reuses the app's own
    KiCanvas helpers (_render_synth_view / KiCanvasView), and the files are served
    from the public /samples static mount."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    if user.accepted_terms_version != LEGAL_VERSION:
        return RedirectResponse("/consent")

    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    kicanvas_head()

    with ui.header().classes("items-center justify-between") \
            .style("background:#0f172a;border-bottom:1px solid #1e293b"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("example boards").classes("text-sm").style("color:#94a3b8")
        ui.button("Back to workspace", icon="arrow_back",
                  on_click=lambda: ui.navigate.to("/")) \
            .props("flat dense no-caps color=white").classes("text-xs")

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1200px"):
        ui.label("Boards KiCraft designed").classes("text-2xl font-bold text-white")
        ui.label("Open one to explore its real schematic and routed board, or use "
                 "its brief as a starting point for your own design.") \
            .classes("text-sm").style("color:#94a3b8")

        samples = available_samples()
        if not samples:
            ui.label("No sample boards are available right now.") \
                .classes("text-sm").style("color:#64748b")
            return

        grid = ui.row().classes("w-full flex-wrap gap-4")
        viewer = ui.column().classes("w-full kc-viewer gap-2")

        def open_sample(s):
            viewer.clear()
            with viewer:
                with ui.row().classes("items-center justify-between w-full mt-2"):
                    ui.label(s.title).classes("text-lg font-bold text-white")
                    ui.button("Use as a starting point", icon="bolt",
                              on_click=lambda ss=s: ui.navigate.to(
                                  f"/?prompt={quote(ss.prompt)}")) \
                        .props("color=primary unelevated")
                ui.label(s.blurb).classes("text-sm").style("color:#94a3b8")
                # Schematic and board are both rendered visible (not in tabs/dialogs):
                # a KiCanvas WebGL canvas built inside a hidden container can size to
                # zero and never repaint, so keeping both on-screen avoids that.
                with ui.card().classes("w-full") \
                        .style("background:#0f172a;border:1px solid #1e293b"):
                    _render_synth_view(s.schematic_sources(), s.stem)
                with ui.card().classes("w-full") \
                        .style("background:#0f172a;border:1px solid #1e293b"):
                    ui.label("Board").classes("text-xs font-medium").style("color:#94a3b8")
                    url, name = s.board_source()
                    KiCanvasView([KiCanvasSource(url, name)], height="h-[520px]")
            ui.run_javascript(
                "document.querySelector('.kc-viewer')?."
                "scrollIntoView({behavior:'smooth',block:'start'})")

        with grid:
            for s in samples:
                card = ui.card().classes("w-72 gap-1 cursor-pointer") \
                    .style("background:#0f172a;border:1px solid #1e293b")
                with card:
                    ui.image(s.board_png_url).props("fit=contain") \
                        .style("height:150px;background:#0a0f1e").classes("w-full rounded")
                    ui.label(s.title).classes("text-base font-semibold text-white")
                    ui.label(f"{s.sheets} sheets / {s.parts} parts / routed") \
                        .classes("text-xs").style("color:#64748b")
                    ui.label(s.blurb).classes("text-xs").style("color:#94a3b8")
                card.on("click", lambda ss=s: open_sample(ss))


def _parts_header(subtitle_btn_label: str, subtitle_btn_target: str) -> None:
    """The shared dark header for the /parts pages: brand + a single back button."""
    with ui.header().classes("items-center justify-between") \
            .style("background:#0f172a;border-bottom:1px solid #1e293b"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("part library").classes("text-sm").style("color:#94a3b8")
        ui.button(subtitle_btn_label, icon="arrow_back",
                  on_click=lambda: ui.navigate.to(subtitle_btn_target)) \
            .props("flat dense no-caps color=white").classes("text-xs")


def _tier_badge(tier) -> None:
    """A small badge marking a part as Standard (vendored) vs the user's own."""
    ui.badge(tier_label(tier),
             color="primary" if tier == Tier.VENDORED else "teal")


@ui.page("/parts")
def parts_page():
    """Logged-in browser for the part library: every standard (vendored) part plus
    anything the user added, listed by part number. Clicking a row opens its detail
    page. Same login + consent gating as the rest of the app; no model is ever called,
    so this is pure reference content."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    if user.accepted_terms_version != LEGAL_VERSION:
        return RedirectResponse("/consent")

    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    _parts_header("Back to workspace", "/")

    parts = catalog()

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1100px"):
        ui.label("Part library").classes("text-2xl font-bold text-white")
        ui.label("Browse the standard library and parts you've added. Click a part to "
                 "see its symbol, footprint, how to use it, and where to buy it.") \
            .classes("text-sm").style("color:#94a3b8")

        if not parts:
            ui.label("No parts are available right now.") \
                .classes("text-sm").style("color:#64748b")
            return

        with ui.row().classes("w-full items-center gap-3"):
            search = ui.input(placeholder="Search by part number, name, tag...") \
                .props("dense outlined clearable dark").classes("flex-grow") \
                .style("min-width:240px")
            tier_filter = ui.toggle(["All", "Standard", "Yours"], value="All") \
                .props("dense no-caps")
        count_label = ui.label().classes("text-xs").style("color:#64748b")
        results = ui.column().classes("w-full gap-2")

        def matches(p) -> bool:
            is_std = p.tier == Tier.VENDORED
            if tier_filter.value == "Standard" and not is_std:
                return False
            if tier_filter.value == "Yours" and is_std:
                return False
            q = (search.value or "").strip().lower()
            if not q:
                return True
            m = p.manifest
            hay = " ".join([m.mpn, m.name, m.description, " ".join(m.tags)]).lower()
            return all(term in hay for term in q.split())

        def render() -> None:
            results.clear()
            shown = [p for p in parts if matches(p)]
            count_label.text = f"{len(shown)} of {len(parts)} parts"
            with results:
                for p in shown:
                    m = p.manifest
                    row = ui.row().classes(
                        "w-full items-center gap-3 cursor-pointer p-3 rounded "
                        "hover:bg-slate-800").style(
                        "background:#0f172a;border:1px solid #1e293b")
                    with row:
                        with ui.column().classes("gap-0").style("min-width:170px"):
                            ui.label(m.mpn).classes("text-sm font-bold text-white")
                            ui.label(m.name).classes("text-xs").style("color:#64748b")
                        ui.label(m.description).classes("text-xs flex-grow").style(
                            "color:#94a3b8;display:-webkit-box;-webkit-line-clamp:2;"
                            "-webkit-box-orient:vertical;overflow:hidden")
                        _tier_badge(p.tier)
                        ui.badge(m.maturity, color="grey-7")
                    row.on("click",
                           lambda pp=p: ui.navigate.to(f"/parts/{pp.manifest.name}"))

        search.on_value_change(lambda: render())
        tier_filter.on_value_change(lambda: render())
        render()


@ui.page("/parts/{name}")
def part_detail_page(name: str):
    """Detail view for one library part: its symbol and footprint (rendered to SVG by
    kicad-cli, shown on light cards), a how-to-use doc built from the manifest, and
    datasheet + LCSC links."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    if user.accepted_terms_version != LEGAL_VERSION:
        return RedirectResponse("/consent")

    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    _parts_header("Back to library", "/parts")

    part = get_part(name)
    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1100px"):
        if part is None:
            ui.label("Part not found").classes("text-2xl font-bold text-white")
            ui.label(f"No library part named '{name}'.").classes("text-sm") \
                .style("color:#94a3b8")
            ui.button("Back to library", on_click=lambda: ui.navigate.to("/parts")) \
                .props("color=primary unelevated")
            return

        m = part.manifest
        with ui.row().classes("items-center gap-3 flex-wrap"):
            ui.label(m.mpn).classes("text-2xl font-bold text-white")
            _tier_badge(part.tier)
            ui.badge(m.maturity, color="grey-7")
        ui.label(m.name).classes("text-sm").style("color:#64748b")

        # Symbol + footprint on light cards (the white-canvas look users know from KiCad).
        img_style = "height:260px;background:#ffffff;padding:10px"
        with ui.row().classes("w-full gap-4 flex-wrap"):
            with ui.card().classes("flex-grow").style(
                    "background:#0f172a;border:1px solid #1e293b;min-width:300px"):
                ui.label("Symbol").classes("text-xs font-medium").style("color:#94a3b8")
                syms = symbol_svgs(part) if kicad_cli_available() else []
                if syms:
                    for i in range(len(syms)):
                        ui.image(f"/part-preview/{m.name}/symbol-{i + 1}.svg") \
                            .props("fit=contain").classes("w-full rounded").style(img_style)
                else:
                    ui.label("Preview unavailable").classes("text-xs") \
                        .style("color:#64748b;padding:16px")
            with ui.card().classes("flex-grow").style(
                    "background:#0f172a;border:1px solid #1e293b;min-width:300px"):
                ui.label("Footprint").classes("text-xs font-medium") \
                    .style("color:#94a3b8")
                fp = footprint_svg(part) if kicad_cli_available() else None
                if fp:
                    ui.image(f"/part-preview/{m.name}/footprint.svg") \
                        .props("fit=contain").classes("w-full rounded").style(img_style)
                else:
                    ui.label("Preview unavailable").classes("text-xs") \
                        .style("color:#64748b;padding:16px")

        with ui.row().classes("w-full gap-3 flex-wrap"):
            if m.datasheet_url:
                ui.button("Datasheet", icon="description",
                          on_click=lambda u=m.datasheet_url:
                          ui.navigate.to(u, new_tab=True)) \
                    .props("outline no-caps color=white")
            url = lcsc_url(m)
            if url:
                ui.button("View on LCSC", icon="shopping_cart",
                          on_click=lambda u=url: ui.navigate.to(u, new_tab=True)) \
                    .props("outline no-caps color=white")

        with ui.card().classes("w-full") \
                .style("background:#0f172a;border:1px solid #1e293b"):
            ui.markdown(usage_markdown(part)).classes("w-full").style("color:#cbd5e1")


# ---------------------------------------------------------------------------
# Public landing page (shown at / to logged-out visitors).
# ---------------------------------------------------------------------------

# User-facing pipeline shown as a stepper. The first four mirror DESIGN_STAGES
# (intent/functional_spec/architecture/bom); the last three are the deterministic
# build phases. `True` marks the build half (accented).
_LANDING_PIPELINE = [
    ("01", "Intent", "What you're building", False),
    ("02", "Functional spec", "Blocks, rails, interfaces", False),
    ("03", "Architecture", "Topologies &amp; sheet plan", False),
    ("04", "Real BOM", "Actual orderable parts", False),
    ("05", "Synthesize", "Hierarchical schematic", True),
    ("06", "Place &amp; route", "Placed, routed, DRC-clean", True),
    ("07", "Fab files", "Gerbers + KiCad project", True),
]

# Inline SVG icons (Heroicons-style) so the page needs no icon font or CDN.
_SVG_CPU = ('<path stroke-linecap="round" stroke-linejoin="round" d="M8.25 3v1.5M4.5 '
            '8.25H3m18 0h-1.5M4.5 12H3m18 0h-1.5m-15 3.75H3m18 0h-1.5M8.25 19.5V21M12 '
            '3v1.5m0 15V21m3.75-18v1.5m0 15V21m-9-1.5h10.5a2.25 2.25 0 0 0 '
            '2.25-2.25V6.75a2.25 2.25 0 0 0-2.25-2.25H6.75A2.25 2.25 0 0 0 4.5 '
            '6.75v10.5a2.25 2.25 0 0 0 2.25 2.25Zm.75-12h9v9h-9v-9Z"/>')
_SVG_STACK = ('<path stroke-linecap="round" stroke-linejoin="round" d="M16.5 8.25V6a2.25 '
              '2.25 0 0 0-2.25-2.25H6A2.25 2.25 0 0 0 3.75 6v8.25A2.25 2.25 0 0 0 6 '
              '16.5h2.25m8.25-8.25H18a2.25 2.25 0 0 1 2.25 2.25V18A2.25 2.25 0 0 1 18 '
              '20.25h-7.5A2.25 2.25 0 0 1 8.25 18v-1.5m8.25-8.25h-6a2.25 2.25 0 0 '
              '0-2.25 2.25v6"/>')
_SVG_DOWNLOAD = ('<path stroke-linecap="round" stroke-linejoin="round" d="M3 16.5v2.25A2.25 '
                 '2.25 0 0 0 5.25 21h13.5A2.25 2.25 0 0 0 21 18.75V16.5M16.5 12 12 '
                 '16.5m0 0L7.5 12m4.5 4.5V3"/>')
_SVG_TUNE = ('<path stroke-linecap="round" stroke-linejoin="round" d="M10.5 6h9.75M10.5 '
             '6a1.5 1.5 0 1 1-3 0m3 0a1.5 1.5 0 1 0-3 0M3.75 6H7.5m3 12h9.75m-9.75 0a1.5 '
             '1.5 0 0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-3.75 0H7.5m9-6h3.75m-3.75 0a1.5 1.5 0 '
             '0 1-3 0m3 0a1.5 1.5 0 0 0-3 0m-9.75 0h9.75"/>')
_SVG_ARROW = ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
              'stroke-width="2" width="16" height="16" aria-hidden="true">'
              '<path stroke-linecap="round" stroke-linejoin="round" d="M13.5 4.5 21 '
              '12m0 0-7.5 7.5M21 12H3"/></svg>')

_LANDING_FEATURES = [
    (_SVG_CPU, "Real parts, not placeholders",
     "Every line of the BOM is a real, orderable component with a real footprint, "
     "resolved to LCSC / JLCPCB part numbers."),
    (_SVG_STACK, "Schematic and layout",
     "A hierarchical, ERC-checked schematic, then a board that is actually placed, "
     "routed, and DRC-clean. Not just a netlist."),
    (_SVG_DOWNLOAD, "Native KiCad, no lock-in",
     "Download a real .kicad_pro project plus Gerbers. Open it in KiCad, change "
     "anything, send it to any fab."),
    (_SVG_TUNE, "You stay in control",
     "Edit any stage and re-run only what changed. KiCraft asks you when a design "
     "choice is yours to make."),
]

_LANDING_HOW = [
    ("1", "Describe your board",
     "One sentence or a detailed brief. Big or small: be bold."),
    ("2", "Watch it design",
     "Follow the agent's thinking live as the schematic and the board take shape."),
    ("3", "Download &amp; build",
     "Grab the KiCad files and Gerbers, then order the PCB from any fab."),
]


def _svg_icon(path: str) -> str:
    return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            'stroke-width="1.6" width="22" height="22" aria-hidden="true">'
            f'{path}</svg>')


def _landing_sample_card(s) -> str:
    badge = '<span class="kc-badge">Featured</span>' if s.featured else ""
    href = f"/signup?prompt={quote(s.prompt)}"
    return (
        f'<a class="kc-sample kc-reveal" href="{href}">'
        f'<div class="kc-sample-art">{badge}'
        f'<img src="{s.board_png_url}" alt="{s.title} board, designed by KiCraft" '
        f'loading="lazy"></div>'
        f'<div class="kc-sample-body">'
        f'<h3>{s.title}</h3>'
        f'<div class="kc-sample-stats">{s.sheets} sheets &middot; {s.parts} parts '
        f'&middot; routed</div>'
        f'<p>{s.blurb}</p>'
        f'<div class="kc-sample-prompt">&ldquo;{s.prompt}&rdquo;</div>'
        f'<span class="kc-sample-cta">Explore this board {_SVG_ARROW}</span>'
        f'</div></a>'
    )


def _render_landing() -> None:
    """The public marketing page at / for logged-out visitors: a hero, the pipeline,
    feature cards, a gallery of real boards KiCraft designed, and CTAs that route to
    signup. Pure static content: the showcase boards are prebuilt assets, so nothing
    here calls a model (no token spend before a valid email signup)."""
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    ui.add_head_html('<link rel="stylesheet" href="/static/kc_landing.css">')
    ui.add_head_html('<style>html{scroll-behavior:smooth}</style>')
    ui.add_head_html(
        f"<script>window.KICRAFT_PROMPTS={json.dumps(EXAMPLE_PROMPTS)};</script>")
    ui.add_head_html('<script src="/static/kc_landing.js" defer></script>')

    samples = available_samples()
    hero = featured_sample()

    pipeline = "".join(
        f'<div class="kc-step{" kc-step-build" if b else ""}">'
        f'<div class="kc-step-n">{n}</div>'
        f'<div class="kc-step-name">{name}</div>'
        f'<div class="kc-step-d">{desc}</div></div>'
        for n, name, desc, b in _LANDING_PIPELINE)

    features = "".join(
        f'<div class="kc-card kc-reveal"><div class="kc-ic">{_svg_icon(svg)}</div>'
        f'<h3>{title}</h3><p>{desc}</p></div>'
        for svg, title, desc in _LANDING_FEATURES)

    how = "".join(
        f'<div class="kc-how kc-reveal"><div class="kc-num">{n}</div>'
        f'<h3>{title}</h3><p>{desc}</p></div>'
        for n, title, desc in _LANDING_HOW)

    # Seed the console with a real brief so it is never blank (the typewriter takes
    # over once the JS and the client-injected markup are both ready).
    seed = EXAMPLE_PROMPTS[0] if EXAMPLE_PROMPTS else "Describe your board. Be bold."
    seed_html = seed.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    gallery = "".join(_landing_sample_card(s) for s in samples)
    hero_art = (f'<div class="kc-hero-art"><img class="kc-board" '
                f'src="{hero.board_png_url}" alt="A PCB designed by KiCraft"></div>'
                if hero else "")
    gallery_block = (
        f'<section class="kc-section" id="samples" style="scroll-margin-top:80px">'
        f'<div class="kc-wrap">'
        f'<div class="kc-kicker">Real output</div>'
        f'<h2 class="kc-h2">Boards KiCraft designed</h2>'
        f'<p class="kc-lead">Each of these started as a single sentence. Open one to '
        f'see the schematic and the routed board, or build your own version.</p>'
        f'<div class="kc-gallery">{gallery}</div></div></section>'
    ) if samples else ""

    html = f"""
<div class="kc-landing">
  <div class="kc-nav"><div class="kc-wrap kc-nav-inner">
    <div class="kc-brand"><span class="kc-logo kc-grad">KiCraft</span>
      <span class="kc-tag">design a PCB from a sentence</span></div>
    <div class="kc-nav-actions">
      <a class="kc-nav-signin" href="/login">Sign in</a>
      <a class="kc-btn kc-btn-primary" href="/signup">Start building</a>
    </div>
  </div></div>

  <section class="kc-hero">
    <div class="kc-glow kc-glow-a"></div>
    <div class="kc-glow kc-glow-b"></div>
    <div class="kc-wrap kc-hero-grid">
      <div class="kc-hero-copy">
        <span class="kc-eyebrow">AI PCB design &middot; powered by KiCad</span>
        <h1 class="kc-h1">Describe a board.<br><span class="kc-grad">Get real KiCad files.</span></h1>
        <p class="kc-sub">KiCraft turns one sentence into a finished design: a
          hierarchical schematic, real orderable parts, and a board that is placed,
          routed, and fab-ready.</p>
        <div class="kc-console">
          <div class="kc-console-bar"><i></i><i></i><i></i></div>
          <div class="kc-console-line"><span class="kc-prompt">&gt;</span><span
            class="kc-type">{seed_html}</span><span class="kc-caret">&nbsp;</span></div>
        </div>
        <div class="kc-cta-row">
          <a class="kc-btn kc-btn-primary kc-btn-lg" href="/signup">Start building free</a>
          <a class="kc-btn kc-btn-ghost kc-btn-lg" href="#samples">See the boards</a>
        </div>
        <p class="kc-trust"><b>Free tier:</b> one full design a week. No credit card.</p>
      </div>
      {hero_art}
    </div>
  </section>

  <section class="kc-section">
    <div class="kc-wrap">
      <div class="kc-kicker">The pipeline</div>
      <h2 class="kc-h2">From a sentence to a fabricable board</h2>
      <p class="kc-lead">KiCraft runs the whole flow an engineer would, one
        reviewable stage at a time.</p>
      <div class="kc-pipe">{pipeline}</div>
    </div>
  </section>

  <section class="kc-section">
    <div class="kc-wrap">
      <div class="kc-kicker">Why it's different</div>
      <h2 class="kc-h2">Designs you can actually build</h2>
      <div class="kc-grid-4">{features}</div>
    </div>
  </section>

  {gallery_block}

  <section class="kc-section">
    <div class="kc-wrap">
      <div class="kc-kicker">How it works</div>
      <h2 class="kc-h2">Three steps to a PCB</h2>
      <div class="kc-steps3">{how}</div>
    </div>
  </section>

  <section class="kc-section">
    <div class="kc-wrap">
      <div class="kc-cta-band kc-reveal">
        <h2>Ready to build your board?</h2>
        <p>Describe it in a sentence. KiCraft does the rest.</p>
        <a class="kc-btn kc-btn-primary kc-btn-lg" href="/signup">Start building free</a>
      </div>
    </div>
  </section>

  <footer class="kc-foot"><div class="kc-wrap kc-foot-inner">
    <span>&copy; KiCraft &middot; A <a href="https://laforestlabs.com" target="_blank"
      rel="noopener">LaForest Labs</a> product</span>
    <div class="kc-foot-links">
      <a href="/terms" target="_blank" rel="noopener">Terms of Service</a>
      <a href="/privacy" target="_blank" rel="noopener">Privacy Policy</a>
      <a href="/login">Sign in</a>
    </div>
  </div></footer>
</div>
"""
    ui.html(html, sanitize=False)


@ui.page("/")
def index(prompt: str = ""):
    user = _current_user()
    if user is None:
        _render_landing()
        return
    if user.accepted_terms_version != LEGAL_VERSION:
        return RedirectResponse("/consent")
    q0 = _store().quota_status(user)

    kicanvas_head()
    ui.add_head_html('<link rel="stylesheet" href="/static/kc_onboarding.css">')
    ui.add_head_html(
        f"<script>window.KICRAFT_PROMPTS={json.dumps(EXAMPLE_PROMPTS)};"
        f"window.KICRAFT_PLACEHOLDER_FALLBACK="
        f"{json.dumps('Describe your board, big or small. Be bold.')};</script>")
    ui.add_head_html('<script src="/static/kc_onboarding.js" defer></script>')
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    first_run = not _store().list_projects(user.id)
    welcome_card = None
    arrow_hint = None
    state: dict = {
        "events": [], "rendered": 0, "running": False, "done": False, "ok": None,
        "spend": None, "zip": None, "fab_done": False, "ws": None,
        "token": None, "project_dir": None, "stem": None, "pcb_ready": False,
        "sch_view": None, "pcb_view": None, "pcb_mtime": None, "leaf_progress_sig": None,
        "state_mtime": None, "run_mtime": None, "build_lines": [],
        "user_id": None, "project_id": None, "brief": "", "account_refreshed": False,
        "status": None, "awaiting_input": False, "questions": [], "questions_rendered": None,
        "prices_rev": 0, "prices_rev_seen": 0, "prices_loaded_ws": None,
    }

    def logout():
        for k in ("user_id", "email"):
            app.storage.user.pop(k, None)
        ui.navigate.to("/login")

    with ui.header().classes("items-center justify-between") \
            .style("background:#0f172a;border-bottom:1px solid #1e293b"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("design a PCB from a sentence").classes("text-sm").style("color:#94a3b8")
        with ui.row().classes("items-center gap-3"):
            ui.button("Examples", icon="dashboard",
                      on_click=lambda: ui.navigate.to("/samples")) \
                .props("flat dense no-caps color=white").classes("text-xs") \
                .tooltip("Explore boards KiCraft designed")
            ui.button("Part library", icon="memory",
                      on_click=lambda: ui.navigate.to("/parts")) \
                .props("flat dense no-caps color=white").classes("text-xs") \
                .tooltip("Browse the standard library and parts you've added")
            if is_admin(user):
                ui.button("Admin", icon="admin_panel_settings",
                          on_click=lambda: ui.navigate.to("/admin")) \
                    .props("flat dense no-caps color=white").classes("text-xs") \
                    .tooltip("Admin dashboard")
            ui.button(user.email, icon="account_circle",
                      on_click=lambda: ui.navigate.to("/profile")) \
                .props("flat dense no-caps color=white").classes("text-xs") \
                .tooltip("Profile & account settings")
            tier_badge = ui.badge(q0["label"], color="primary")
            ui.button("Log out", on_click=logout).props("flat dense color=white").classes("text-xs")

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1600px"):
        try:
            budget = SpendGuard(Settings.from_env()).status()
            ui.label(f"Daily budget remaining: ${budget['daily_remaining_usd']:.2f} "
                     f"of ${budget['daily_ceiling_usd']:.0f}").classes("text-xs").style("color:#64748b")
        except Exception:
            ui.label("").classes("hidden")

        quota_label = ui.label().classes("text-xs").style("color:#94a3b8")

        if first_run:
            with ui.row().classes("w-full items-start justify-between kc-welcome") \
                    .style("background:#0f172a;border:1px solid #1e293b;"
                           "border-radius:8px;padding:12px 14px") as welcome_card:
                with ui.column().classes("gap-1"):
                    ui.label("Welcome to KiCraft") \
                        .classes("text-base font-semibold text-white")
                    ui.label("Describe a board in a sentence and KiCraft turns it into "
                             "real KiCad files: schematic, real parts, placed and routed. "
                             "Be bold: the bigger the ask, the better the demo.") \
                        .classes("text-sm").style("color:#94a3b8")
                    ui.button("Explore example boards", icon="dashboard",
                              on_click=lambda: ui.navigate.to("/samples")) \
                        .props("flat dense no-caps color=primary").classes("text-xs mt-1")

                def dismiss_welcome():
                    welcome_card.set_visibility(False)
                    ui.run_javascript("localStorage.setItem('kc_welcome_dismissed','1')")

                ui.button(icon="close", on_click=dismiss_welcome) \
                    .props("flat dense round color=white")

        brief = ui.textarea(
            "Describe your board",
            placeholder="Describe your board, big or small. Be bold.") \
            .props("rows=4 stack-label").classes("w-full kc-brief")

        # One-click inspiration: chips drop a full brief into the box (the cycling
        # placeholder in kc_onboarding.js supplies passive ideas). Created here for
        # position, populated below once the Design button exists so use_prompt can
        # nudge it.
        chips_row = ui.row().classes("items-center gap-2 kc-chips")

        with ui.row().classes("items-center gap-2"):
            design_btn = ui.button("Design").props("color=primary unelevated") \
                .classes("kc-design")
            continue_btn = ui.button("Continue design", icon="play_arrow") \
                .props("color=primary outline")
            continue_btn.set_visibility(False)
            if first_run:
                design_btn.classes(add="kc-pulse")
                with ui.row().classes("items-center kc-arrow") as arrow_hint:
                    ui.icon("arrow_back").classes("kc-arrow-icon")
                    ui.label("click to start")

        def use_prompt(text: str):
            brief.value = text
            brief.run_method("focus")
            design_btn.classes(add="kc-pulse")  # draw the eye to the next click

        with chips_row:
            for _chip in CHIP_PROMPTS:
                ui.button(_chip["label"],
                          on_click=lambda p=_chip["prompt"]: use_prompt(p)) \
                    .props("outline rounded dense no-caps").classes("kc-chip")
            ui.button("Surprise me", icon="casino",
                      on_click=lambda: use_prompt(random.choice(EXAMPLE_PROMPTS))) \
                .props("flat rounded dense no-caps").classes("kc-chip")

        # Prefill from a sample the visitor chose before signing up (carried via the
        # ?prompt= query or stashed across the signup hop). No run starts: the user
        # still clicks Design themselves, so no model is called without a signup.
        prefill = (prompt or app.storage.user.pop("pending_prompt", "") or "").strip()
        if prefill:
            use_prompt(prefill)

        status = ui.label("").classes("text-sm").style("color:#e2e8f0")
        spend = ui.label("").classes("text-sm").style("color:#64748b")
        question_box = ui.column().classes("w-full")

        # Per-stage tabs: each phase gets its own tab with a project-state inspector
        # (left) over the LLM thinking + activity/log windows (right). The native
        # KiCad schematic/board (KiCanvas) and the download land in the build tabs.
        tabs = StageTabs()

        # A KiCanvas view built while its tab is hidden sizes its WebGL canvas to zero
        # and never repaints; re-fit it the first time the user reveals that tab. The
        # flag is reset when each view is (re)created (see the render loop below).
        def _reveal_view(view_key: str, seen_flag: str) -> None:
            v = state.get(view_key)
            if v is not None and not state.get(seen_flag):
                state[seen_flag] = True
                v.refresh()
        tabs.on_show("synthesize", lambda: _reveal_view("sch_view", "sch_revealed"))
        tabs.on_show("place_route", lambda: _reveal_view("pcb_view", "pcb_revealed"))

        with ui.expansion("Your projects").classes("w-full mt-2") \
                .style("background:#0f172a;border:1px solid #1e293b"):
            proj_container = ui.column().classes("w-full gap-1 p-2")

        with ui.expansion("Edit a stage & re-run").classes("w-full") \
                .style("background:#0f172a;border:1px solid #1e293b"):
            edit_box = ui.column().classes("w-full gap-2 p-2")
        edit_ctx: dict = {"getter": None, "raw": None, "instr": None}

        def build_projects():
            proj_container.clear()
            with proj_container:
                projs = _store().list_projects(user.id)
                if not projs:
                    ui.label("No projects yet. Describe a board above to begin.") \
                        .classes("text-xs").style("color:#64748b")
                for p in projs:
                    with ui.row().classes("items-center gap-3 w-full"):
                        ui.label(p.project_stem or "(building...)") \
                            .classes("text-sm").style("color:#e2e8f0")
                        ui.label(p.status).classes("text-xs").style("color:#94a3b8")
                        ui.label(p.created_at[:19].replace("T", " ")) \
                            .classes("text-xs").style("color:#64748b")
                        if p.dir_path:
                            ui.button("Open", icon="folder_open",
                                      on_click=lambda pp=p: open_project(pp)).props("flat dense")
                        if p.zip_path and Path(p.zip_path).is_file():
                            ui.button("Download", icon="download",
                                      on_click=lambda zp=p.zip_path: ui.download(zp)) \
                                .props("flat dense")
                        if p.dir_path and is_admin(user):
                            ui.button("Evaluate", icon="fact_check",
                                      on_click=lambda pp=p: open_eval_dialog(
                                          pp.dir_path,
                                          pp.project_stem or f"project {pp.id}")) \
                                .props("flat dense").style("color:#a78bfa")

        def build_question_panel():
            """(Re)build the clarifying-question panel for a parked run. Always
            offers a freeform text answer; suggested options just fill it in."""
            question_box.clear()
            state["questions_rendered"] = state.get("questions")
            qs = state.get("questions") or []
            if not (state.get("awaiting_input") and qs):
                return
            stage = qs[0].get("stage", "")
            with question_box:
                with ui.card().classes("w-full") \
                        .style("background:#1f1300;border:1px solid #92400e"):
                    ui.label("The agent needs your input").classes("text-base font-bold") \
                        .style("color:#fbbf24")
                    if stage:
                        ui.label(f"Stage: {stage}").classes("text-xs").style("color:#d4d4d8")
                    widgets = []
                    for q in qs:
                        ui.label(q.get("text", "")).classes("text-sm mt-2").style("color:#fde68a")
                        ans = ui.input(placeholder="Type your answer (or pick a suggestion)") \
                            .classes("w-full")
                        for opt in (q.get("options") or []):
                            ui.button(opt, on_click=lambda o=opt, a=ans: a.set_value(o)) \
                                .props("flat dense").classes("text-xs")
                        widgets.append((q, ans))

                    def submit_answers():
                        answers = [{"text": q.get("text", ""),
                                    "answer": (a.value or "").strip()} for q, a in widgets]
                        if not any(x["answer"] for x in answers):
                            ui.notify("Type or pick at least one answer.", color="warning")
                            return
                        _answer_and_resume(stage, answers)

                    ui.button("Submit answer & continue", icon="send", color="primary",
                              on_click=submit_answers).classes("mt-2")

        def _answer_and_resume(stage, answers):
            if state["running"]:
                return
            ws = state["ws"]
            if not ws:
                ui.notify("No open design.", color="warning")
                return
            record_answers(ws, stage, answers)
            runs = [stage] + downstream_stages(stage)
            if state["project_id"]:  # same project, no new quota slot
                _store().update_project_status(state["project_id"], "running")
            state.update(running=True, done=False, ok=None, fab_done=False,
                         account_refreshed=False, awaiting_input=False, questions=[],
                         questions_rendered=None)
            question_box.clear()
            continue_btn.set_visibility(False)
            design_btn.disable()
            status.text = f"Got it. Continuing from {stage} with your answer..."
            threading.Thread(target=_run_design, args=(state, runs),
                             kwargs={"answers": answers}, daemon=True).start()

        def build_edit_panel():
            """(Re)build the stage editor for the currently open design."""
            edit_box.clear()
            with edit_box:
                if not state["ws"]:
                    ui.label("Open or run a design first, then edit a stage here.") \
                        .classes("text-xs").style("color:#64748b")
                    return
                sj = read_state(state["ws"])
                editable = [s for s in ("intent", "functional_spec", "architecture", "bom")
                            if sj.get(s)]
                if not editable:
                    ui.label("No committed stages to edit yet.") \
                        .classes("text-xs").style("color:#64748b")
                    return
                ui.label("Editing a stage re-runs the stages after it (spends tokens).") \
                    .classes("text-xs").style("color:#94a3b8")
                stage_sel = ui.select(editable, value=editable[0], label="Stage").classes("w-64")
                form_holder = ui.column().classes("w-full gap-2")

                def render_form():
                    form_holder.clear()
                    with form_holder:
                        stg = stage_sel.value
                        edit_ctx["getter"] = _render_slot_form(SLOT_MODEL[stg], sj.get(stg) or {})
                        with ui.expansion("Advanced: raw slot JSON").classes("w-full"):
                            edit_ctx["raw"] = ui.textarea(value=json.dumps(sj.get(stg), indent=2)) \
                                .props("rows=8").classes("w-full text-xs")
                        edit_ctx["instr"] = ui.textarea(
                            "Or tell the agent what to change",
                            placeholder="e.g. use a USB-C connector, not micro-USB") \
                            .props("rows=2").classes("w-full")
                        with ui.row().classes("gap-2 flex-wrap"):
                            ui.button("Save form & re-run", icon="save",
                                      on_click=lambda s=stg: _confirm_edit(s, "form")) \
                                .props("color=primary")
                            ui.button("Save JSON & re-run", icon="data_object",
                                      on_click=lambda s=stg: _confirm_edit(s, "json")) \
                                .props("outline color=white")
                            ui.button("Ask agent & re-run", icon="auto_fix_high",
                                      on_click=lambda s=stg: _confirm_edit(s, "instruction")) \
                                .props("outline color=white")

                stage_sel.on_value_change(render_form)
                render_form()

        def _confirm_edit(stage, mode):
            slot_dict = None
            instruction = None
            try:
                if mode == "form":
                    slot_dict = edit_ctx["getter"]() if edit_ctx["getter"] else None
                elif mode == "json":
                    slot_dict = json.loads((edit_ctx["raw"].value if edit_ctx["raw"] else "") or "{}")
                else:
                    instruction = ((edit_ctx["instr"].value if edit_ctx["instr"] else "") or "").strip()
                    if not instruction:
                        ui.notify("Type an instruction first.", color="warning")
                        return
            except json.JSONDecodeError as e:
                ui.notify(f"Invalid JSON: {e}", color="negative")
                return
            down = downstream_stages(stage)
            runs = ([stage] + down) if instruction else down
            verb = f"Re-draft {stage} and re-run " if instruction else "Re-run "
            tail = ", ".join(down) if down else "nothing downstream"
            with ui.dialog() as dlg, ui.card() \
                    .style("background:#0f172a;border:1px solid #1e293b"):
                ui.label("Re-run stages?").classes("text-lg font-bold text-white")
                ui.label(verb + tail + ". LLM stages spend tokens.") \
                    .classes("text-sm").style("color:#94a3b8")
                with ui.row().classes("gap-2 justify-end w-full"):
                    ui.button("Cancel", on_click=dlg.close).props("flat color=white")
                    ui.button("Confirm & run", color="primary",
                              on_click=lambda: (dlg.close(),
                                                _do_rerun(stage, slot_dict, instruction, runs)))
            dlg.open()

        def _do_rerun(stage, slot_dict, instruction, runs):
            if state["running"]:
                ui.notify("A run is already in progress.", color="warning")
                return
            ws = state["ws"]
            if not ws:
                ui.notify("No open design.", color="warning")
                return
            if slot_dict is not None:  # structured / raw-JSON edit: commit it first
                sj = read_state(ws)
                ok, out = commit_slot(ws, stage, slot_dict, brief=state.get("brief", ""),
                                      project_stem=sj.get("project_stem"))
                if not ok:
                    ui.notify(f"Edit rejected: {out.get('errors')}", color="negative")
                    return
            null_downstream(ws, stage)
            if state["project_id"]:
                _store().update_project_status(state["project_id"], "running")
            state.update(running=True, done=False, ok=None, fab_done=False,
                         account_refreshed=False, pcb_ready=False, sch_view=None,
                         pcb_view=None, pcb_mtime=None, state_mtime=None, run_mtime=None,
                         awaiting_input=False, questions=[])
            continue_btn.set_visibility(False)
            design_btn.disable()
            status.text = "Re-running: " + " -> ".join(runs) + " ..."
            threading.Thread(target=_run_design, args=(state, runs),
                             kwargs={"instruction": instruction}, daemon=True).start()

        def _continue():
            """Run the stages still missing from the current (reopened) design."""
            if state["running"]:
                return
            sj = read_state(state["ws"]) if state["ws"] else {}
            rem = remaining_stages(sj)
            if not rem:
                ui.notify("Nothing left to run for this design.", color="info")
                return
            if state["project_id"]:  # same project, no new quota slot
                _store().update_project_status(state["project_id"], "running")
            state.update(running=True, done=False, ok=None, fab_done=False,
                         account_refreshed=False)
            continue_btn.set_visibility(False)
            design_btn.disable()
            status.text = "Continuing: " + " -> ".join(rem) + " ..."
            threading.Thread(target=_run_design, args=(state, rem), daemon=True).start()

        continue_btn.on_click(_continue)

        def open_project(p):
            """Reopen a saved project: rehydrate its workspace and render its current
            slots, so the user can continue, edit, answer a parked question, or
            re-download. Reuses the same project_id (no new quota slot)."""
            if state["running"]:
                ui.notify("Wait for the current run to finish before opening another.",
                          color="warning")
                return
            ws = _rehydrate_workspace(p)
            sj = read_state(ws)
            zip_ok = bool(p.zip_path and Path(p.zip_path).is_file())
            completed = p.status == "ok"
            state.update(events=[], rendered=0, running=False, done=completed,
                         ok=(True if completed else None), spend=p.cost_usd,
                         zip=(p.zip_path if zip_ok else None), fab_done=False,
                         ws=str(ws), token=None, project_dir=None, stem=p.project_stem,
                         pcb_ready=False, sch_view=None, pcb_view=None, pcb_mtime=None,
                         state_mtime=None, run_mtime=None, build_lines=[],
                         account_refreshed=True,
                         status=("awaiting_input" if p.status == "awaiting_input" else None),
                         awaiting_input=(p.status == "awaiting_input"), questions_rendered=None,
                         questions=[q for q in (sj.get("open_questions") or [])
                                    if not q.get("answer")])
            state["user_id"] = user.id
            state["project_id"] = p.id
            state["brief"] = p.brief or ""
            tabs.reset()
            project_dir = _discover_generated_dir(ws)  # restored artifacts -> schematic /
            if project_dir is not None:                # PCB render, even if the run FAILED
                state["stem"] = project_dir.name
                state["project_dir"] = str(project_dir)
                state["token"] = _register_project_dir(project_dir)
                state["pcb_ready"] = (project_dir / f"{project_dir.name}.kicad_pcb").is_file()
            rem = remaining_stages(sj)
            continue_btn.set_visibility(bool(rem) and not state["awaiting_input"])
            if state["awaiting_input"]:
                status.text = "Reopened. This design is waiting for your answer below."
            elif rem:
                status.text = ("Reopened. Remaining: " + " -> ".join(rem)
                               + ". Click Continue design when ready.")
            else:
                status.text = ("Reopened. Design complete: download below, "
                               "or edit a stage to revise.")
            refresh_account_ui()
            ui.notify(f"Opened {p.project_stem or 'project'}.", color="positive")

        def refresh_account_ui():
            u = _current_user()
            if u is None:
                return
            q = _store().quota_status(u)
            period = "week" if q["window_days"] <= 7 else "month"
            if q.get("unlimited"):
                quota_label.text = f"{q['label']} tier: unlimited designs (staff)."
            else:
                quota_label.text = (f"{q['label']} tier: {q['remaining']} of {q['limit']} "
                                    f"designs left this {period}.")
            tier_badge.text = q["label"]
            if q["remaining"] <= 0:
                design_btn.disable()
                quota_label.style("color:#f59e0b")
            else:
                if not state["running"]:
                    design_btn.enable()
                quota_label.style("color:#94a3b8")
            build_projects()
            build_edit_panel()

        def start():
            if state["running"]:
                return
            u = _current_user()
            if u is None:
                ui.navigate.to("/login")
                return
            if not (brief.value or "").strip():
                ui.notify("Enter a brief first.", color="warning")
                return
            q = _store().quota_status(u)
            if q["remaining"] <= 0:
                period = "week" if q["window_days"] <= 7 else "month"
                ui.notify(f"You've used your {q['limit']} design(s) this {period}. "
                          "Upgrade for more.", color="warning")
                return
            pid = _store().create_project(u.id, brief.value)
            state.update(events=[], rendered=0, running=True, done=False, ok=None,
                         spend=None, zip=None, fab_done=False, ws=None, token=None,
                         project_dir=None, stem=None, pcb_ready=False, sch_view=None,
                         pcb_view=None, pcb_mtime=None, state_mtime=None, run_mtime=None,
                         build_lines=[], account_refreshed=False, status=None,
                         awaiting_input=False, questions=[], questions_rendered=None,
                         prices_rev=0, prices_rev_seen=0, prices_loaded_ws=None)
            continue_btn.set_visibility(False)
            state["user_id"] = u.id
            state["project_id"] = pid
            state["brief"] = brief.value
            tabs.reset()
            status.text = ("Designing... (intent -> functional_spec -> architecture -> bom -> "
                           "wiring -> synthesize -> place/route -> fab)")
            design_btn.disable()
            design_btn.classes(remove="kc-pulse")
            for _chrome in (welcome_card, chips_row, arrow_hint):
                if _chrome is not None:
                    _chrome.set_visibility(False)
            threading.Thread(target=_design_worker, args=(brief.value, state), daemon=True).start()

        design_btn.on_click(start)

        def render():
            evs = state["events"]
            changed = False
            while state["rendered"] < len(evs):
                e = evs[state["rendered"]]
                if e.get("kind") == "build_log":
                    state["build_lines"].append(e.get("text", ""))
                tabs.push(e)
                state["rendered"] += 1
                changed = True
            if changed:
                tabs.flush()
            if state["spend"] is not None:
                spend.text = f"Spent this design: ${state['spend']:.4f}"

            # Clarifying-question panel: (re)build only when the question set changes
            # (a worker parks the run from its thread; this picks it up next tick).
            if state.get("questions") != state.get("questions_rendered"):
                build_question_panel()

            # Design-stage inspectors: rebuild from state.json whenever it changes.
            if state["ws"]:
                # Seed the price cache from this project's persisted prices once
                # (so a reopen shows costs immediately, before any new fetch).
                if state.get("prices_loaded_ws") != state["ws"]:
                    state["prices_loaded_ws"] = state["ws"]
                    _load_price_cache(Path(state["ws"]))
                mt = _mtime(Path(state["ws"]) / ".kicraft" / "state.json")
                if mt and mt != state["state_mtime"]:
                    state["state_mtime"] = mt
                    sj = _read_state_json(Path(state["ws"]))
                    for stg in ("intent", "functional_spec", "architecture", "bom", "wiring"):
                        spec = _inspector_spec(stg, sj, {}, None, state["build_lines"])
                        if spec:
                            tabs.set_inspector(stg, spec)
                    # Live-price any BOM parts in the background (fills in the cost
                    # column + total once the fetch lands; cached parts are instant).
                    bom_parts = (sj.get("bom") or {}).get("parts") or []
                    if bom_parts:
                        _ensure_bom_prices(bom_parts, state["ws"], state)

            # Prices arrive on a background thread; re-render the BOM when they do.
            if state["ws"] and state.get("prices_rev") != state.get("prices_rev_seen"):
                state["prices_rev_seen"] = state.get("prices_rev")
                spec = _inspector_spec(
                    "bom", _read_state_json(Path(state["ws"])), {}, None,
                    state["build_lines"])
                if spec:
                    tabs.set_inspector("bom", spec)

            # Even when the build later FAILS, show the schematic as soon as synthesis
            # writes the sheets: discover the generated dir from the workspace so the
            # viewer never depends on a project_stem being recorded or on the pre-build
            # wiring having found one. Self-heals on both live and reopened runs.
            if state["project_dir"] is None and state["ws"]:
                pd = _discover_generated_dir(Path(state["ws"]))
                if pd is not None:
                    state["stem"] = pd.name
                    state["project_dir"] = str(pd)
                    state["token"] = _register_project_dir(pd)

            project_dir = Path(state["project_dir"]) if state["project_dir"] else None

            # Schematic appears in the Synthesize tab once synth writes the sheets.
            if project_dir is not None and state["sch_view"] is None:
                srcs = _schematic_sources(project_dir, state["stem"], state["token"])
                if srcs:
                    sj = _read_state_json(Path(state["ws"])) if state["ws"] else {}
                    tabs.set_inspector("synthesize", _inspector_spec(
                        "synthesize", sj, {}, project_dir, state["build_lines"]))
                    with tabs.view_slot("synthesize"):
                        state["sch_view"] = _render_synth_view(srcs, state["stem"])
                    # Painted already if synthesize is the visible tab now; otherwise
                    # mark it for a re-fit when the user first reveals it.
                    state["sch_revealed"] = tabs.active() == "synthesize"

            # Place/route: a per-leaf placement gallery streams progress while the
            # board builds; the routed parent replaces it once the build succeeds.
            if project_dir is not None:
                rmt = _mtime(project_dir / ".experiments" / "run_status.json")
                if rmt and rmt != state["run_mtime"]:
                    state["run_mtime"] = rmt
                    rs = _read_run_status(project_dir)
                    tabs.set_inspector("place_route", _inspector_spec(
                        "place_route", {}, rs, project_dir, state["build_lines"]))
                    if not state["pcb_ready"] and state["token"]:
                        prog = _leaf_layout_progress(project_dir, state["token"])
                        sig = tuple((d["sheet_name"], d["status"]) for d in prog)
                        if prog and sig != state["leaf_progress_sig"]:
                            state["leaf_progress_sig"] = sig
                            slot = tabs.view_slot("place_route")
                            slot.clear()
                            with slot:
                                _render_leaf_gallery(prog, rs)
                if state["pcb_ready"]:
                    pcb_name = f"{state['stem']}.kicad_pcb"
                    pcb_path = project_dir / pcb_name
                    pcb_url = f"/project/{state['token']}/{pcb_name}"
                    if state["pcb_view"] is None:
                        state["pcb_mtime"] = _mtime(pcb_path)
                        slot = tabs.view_slot("place_route")
                        slot.clear()  # drop the progress gallery; show the final board
                        with slot:
                            ui.label("PCB").classes("text-xs font-medium").style("color:#94a3b8")
                            state["pcb_view"] = KiCanvasView(
                                [KiCanvasSource(pcb_url, pcb_name)],
                                height="", style="height:calc(100vh - 460px);min-height:360px")
                            state["pcb_revealed"] = tabs.active() == "place_route"
                    else:
                        mt = _mtime(pcb_path)
                        if mt != state["pcb_mtime"]:
                            state["pcb_mtime"] = mt
                            state["pcb_view"].refresh()

            if state["done"]:
                design_btn.enable()
                if not state["account_refreshed"]:
                    state["account_refreshed"] = True
                    refresh_account_ui()
                if state["ok"]:
                    status.text = "Done. Your KiCad project is ready."
                    if not state["fab_done"]:
                        state["fab_done"] = True
                        sj = _read_state_json(Path(state["ws"])) if state["ws"] else {}
                        rs = _read_run_status(project_dir) if project_dir else {}
                        for stg in ("synthesize", "place_route", "fab"):  # finalize build logs
                            tabs.set_inspector(stg, _inspector_spec(
                                stg, sj, rs, project_dir, state["build_lines"]))
                        if state["zip"]:
                            with tabs.view_slot("fab"):
                                ui.button("Download KiCad project (.zip)", icon="download",
                                          on_click=lambda: ui.download(state["zip"])) \
                                    .props("color=positive")
                elif state["ok"] is False:
                    status.text = ("Build failed. The synthesized schematic is shown in "
                                   "the Synthesize tab (red) for review.")

        refresh_account_ui()
        ui.timer(0.2, render)

    with ui.footer().classes("justify-center py-1") \
            .style("background:#0b1120;border-top:1px solid #1e293b"):
        _laforest_footer()


if os.environ.get("KICRAFT_WEB_DEMO"):

    # A canned committed-state for the flashlight brief, so the demo populates the
    # per-stage inspector windows (not just the streaming) for screenshots. Shaped
    # like ConversationState so it feeds the real _inspector_spec builders.
    _DEMO_STATE = {
        "project_stem": "FLASHLIGHT",
        "intent": {"goal": "USB-C rechargeable 18650 flashlight, no microcontroller",
                   "inferred_expertise": "intermediate",
                   "constraints": ["single-sided assembly ok", "through-hole 18650 holder"],
                   "named_parts": ["TP4056", "18650 cell"],
                   "assumptions": ["USB-C 5V input (defaulted)"]},
        "functional_spec": {"blocks": [
            {"name": "USB_C_INPUT", "category": "power", "purpose": "5V from USB-C"},
            {"name": "CHARGER", "category": "power", "purpose": "TP4056 Li-ion charger"},
            {"name": "LED_DRIVER", "category": "power", "purpose": "constant-current boost"}],
            "connections": [{"from_block": "USB_C_INPUT", "to_block": "CHARGER",
                             "signal_type": "power"}],
            "assumptions": ["1A charge current (defaulted)"]},
        "architecture": {"sheets": [{"name": "MAIN", "stem": "FLASHLIGHT", "function": "all"}],
                         "power_nets": ["VBUS", "VBAT", "GND"],
                         "rail_voltages": {"VBUS": 5.0, "VBAT": 4.2},
                         "topologies": {"LED_DRIVER": "boost constant-current"},
                         "mcu_present": False, "comms_protocols": [], "inter_sheet_nets": []},
        "bom": {"parts": [
            {"ref": "U1", "value": "TP4056", "symbol": "tp4056:TP4056",
             "footprint": "tp4056:SOP-8", "sheet": "MAIN"},
            {"ref": "J1", "value": "USB-C", "symbol": "usb-c-16p:TYPE-C-31-M-12",
             "footprint": "usb-c-16p:TYPE-C", "sheet": "MAIN"},
            {"ref": "D1", "value": "white LED", "symbol": "Device:LED",
             "footprint": "LED_SMD:LED_0603_1608Metric", "sheet": "MAIN"}],
            "connections": [
                {"net_name": "VBUS", "sheet": "MAIN",
                 "endpoints": [{"ref": "J1", "pin": "A4"}, {"ref": "U1", "pin": "4"}]},
                {"net_name": "GND", "sheet": "MAIN",
                 "endpoints": [{"ref": "J1", "pin": "A1"}, {"ref": "U1", "pin": "3"}]}],
            "no_connect_pins": [{"ref": "J1", "pin": "A8"}, {"ref": "J1", "pin": "B8"}]},
        "artifacts": {"status": "ok", "fab_zip": "FLASHLIGHT_fab.zip"},
    }

    # Canned prices for the demo BOM (keyed by _price_key) so the cost column +
    # total render with no network (the demo never calls JLCPCB).
    _DEMO_PRICES = {
        "kw:TP4056": {"unit_price": 0.18, "lcsc": "C16581", "stock": 9999},
        "kw:USB-C": {"unit_price": 0.0667, "lcsc": "C165948", "stock": 9999},
        "kw:white LED 0603": {"unit_price": 0.014, "lcsc": "C72043", "stock": 9999},
    }

    @ui.page("/demo")
    def demo_page():
        """Dev-only: replay a canned design through the per-stage tabs so the layout
        and styling can be previewed and screenshotted with no spend or network.
        Registered only when KICRAFT_WEB_DEMO is set (off in production)."""
        ui.dark_mode().enable()
        ui.query("body").style("background:#0b1120")
        with ui.header().classes("items-center justify-between") \
                .style("background:#0f172a;border-bottom:1px solid #1e293b"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("design preview (demo)").classes("text-sm").style("color:#94a3b8")
        with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1600px"):
            ui.label("Replaying a canned design to preview the per-stage tabs.") \
                .classes("text-sm").style("color:#94a3b8")
            tabs = StageTabs()
            d = {"events": demo_events(), "i": 0, "build_lines": []}

            def step():
                evs = d["events"]
                pushed = 0
                while d["i"] < len(evs) and pushed < 2:  # ~2 events/tick = streaming feel
                    e = evs[d["i"]]
                    if e.get("kind") == "build_log":
                        d["build_lines"].append(e.get("text", ""))
                    tabs.push(e)
                    if e.get("kind") == "stage_done":
                        stg = e.get("stage")
                        tabs.set_inspector(stg, _inspector_spec(
                            stg, _DEMO_STATE, {}, None, [], prices=_DEMO_PRICES))
                    elif e.get("kind") == "build_done":
                        rs = {"phase": "done", "progress_percent": 100}
                        for stg in ("synthesize", "place_route", "fab"):
                            tabs.set_inspector(stg, _inspector_spec(
                                stg, _DEMO_STATE, rs, None, d["build_lines"]))
                    d["i"] += 1
                    pushed += 1
                tabs.flush()
                if d["i"] >= len(evs):
                    timer.cancel()  # replay finished; stop ticking

            timer = ui.timer(0.25, step)


def main() -> None:
    Settings.from_env()  # fail fast if OPENROUTER_API_KEY is missing; also loads .env
    if not _signup_code():
        print("WARNING: neither KICRAFT_SIGNUP_CODE nor KICRAFT_ACCESS_PASSWORD is set; "
              "no one can register. Set KICRAFT_SIGNUP_CODE before exposing kicraft.io.")
    ui.run(
        host=os.environ.get("KICRAFT_WEB_HOST", "0.0.0.0"),
        port=int(os.environ.get("KICRAFT_WEB_PORT", "8080")),
        title="KiCraft",
        storage_secret=os.environ.get("KICRAFT_STORAGE_SECRET", "kicraft-dev-secret"),
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
