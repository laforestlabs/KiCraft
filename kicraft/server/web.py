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

import asyncio
import datetime as dt
import hmac
import importlib
import json
import os
import re
import shutil
import zipfile
import ssl
import subprocess
import threading
import time
import types
import typing
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import quote

from nicegui import app, ui
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse

from .accounts import (
    _RESET_TTL_SECONDS,
    _VERIFY_TTL_SECONDS,
    DEFAULT_TIER,
    TIERS,
    AccountStore,
    grant_expiry,
    is_admin,
)
from .config import LEGAL_VERSION, Settings, default_legal_dir
from .examples import EXAMPLE_PROMPTS
from .kicanvas import KICANVAS_ASSET, KiCanvasSource, KiCanvasView, kicanvas_head
from .layout_panel import (
    LayoutEditorPanel,
    leaf_artifacts_exist,
    user_may_edit_layout,
)
from .rules_panel import PlacementRulesPanel
from .mailer import send_reset_email, send_verification_email
from ..parts_library import Tier
from ..parts_library import jlcparts, lcsc_retail
from ..tuning.benchmark import briefs as _selfeval_briefs
from .parts_catalog import (
    _content_hash_key,
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
    bom_reconcile_deficits,
    commit_slot,
    derive_stage_statuses,
    downstream_stages,
    maybe_bom_reconcile,
    null_downstream,
    read_state,
    record_answers,
    remaining_stages,
    run_session,
)
from .spend_guard import SpendGuard
from kicraft import __version__ as KICRAFT_VERSION
from kicraft.build_slots import ACQUIRED_MARKER, slot_count

from . import billing, notify
from .build_worker import JOB_KIND_COMMANDS, _kill_build
from .stage_driver import DESIGN_STAGES, SLOT_MODEL
from .stagetabs import StageTabs, _build_substage, demo_events
from . import stage_diagram
from .storage import (
    _discover_generated_dir,
    _gc_workspaces,
    _kicraft_dir,
    _new_workspace,
    _persisted_generated_dir,
    _read_project_stem,
    _state_path,
)
from .pricing import (  # pure BOM-pricing helpers; fetch/cache stay below
    _LCSC_CODE_RE,
    _fmt_price,
    _fmt_stock,
    _fmt_total,
    _pick_price,
    _price_key,
    _resolve_part,  # noqa: F401  re-exported for tests / back-compat
    _vendor_cell,
)
from . import render_serving  # importing registers the /project/<token>/... + /part-preview routes
from .render_serving import (
    _project_secret,  # noqa: F401  re-exported for tests
    _register_project_dir,
    _resolve_project_token,  # noqa: F401  re-exported for tests (security/test_capability_token)
    serve_project_file,  # noqa: F401  called directly by tests
    serve_project_render,  # noqa: F401  called directly by tests
)

# Self-host the KiCanvas ES module bundle so the browser fetches it same-origin.
app.add_static_files("/static", str(KICANVAS_ASSET.parent))

# Shared manual-layout canvas controller (kicraft.layout_editor); the editor
# bootstrap loads it from this mount on first use.
from kicraft.layout_editor.canvas import (  # noqa: E402
    DEFAULT_ASSET_MOUNT as _LAYOUT_ASSET_MOUNT,
    STATIC_DIR as _LAYOUT_STATIC_DIR,
)

app.add_static_files(_LAYOUT_ASSET_MOUNT, str(_LAYOUT_STATIC_DIR))

# Prebuilt sample projects (preview renders + raw KiCad files) for the public
# landing showcase and the logged-in explorer. Public on purpose: these are
# curated, finished demos, so serving them costs nothing and needs no auth.
app.add_static_files("/samples", str(SAMPLES_DIR))

# Global visual theme: palette + fonts + depth, injected once into every page's
# head (shared). The single source of truth for the app's look lives in
# kicraft.server.theme; import color constants from there instead of hex.
from . import theme as _theme  # noqa: E402

_theme.install()

# Sizing for the in-tab KiCanvas on the build stages (the schematic on Synthesize
# and the routed board on Place/Route). Both should nearly fill the (widened)
# inspector column so the artifact is large enough to visually inspect; the offset
# leaves room for the label + sheet selector above, and the floor keeps it usable
# on short screens. One constant so the two views never drift apart.
_BUILD_VIEW_STYLE = "height:calc(100vh - 380px);min-height:460px"

# Shown in the reset email; derived from the token TTL so the two never drift.
_RESET_TTL_MINUTES = _RESET_TTL_SECONDS // 60
# Shown in the verification email; derived from the token TTL so the two never drift.
_VERIFY_TTL_HOURS = _VERIFY_TTL_SECONDS // 3600


_STORE: AccountStore | None = None


def _store() -> AccountStore:
    """The shared accounts store, built once per process from settings."""
    global _STORE
    if _STORE is None:
        s = Settings.from_env()
        _STORE = AccountStore(s.users_db_path, s.projects_dir)
    return _STORE


# Build jobs currently driven by a live thread of THIS process; the orphan
# reaper leaves these alone. set add/discard are atomic under the GIL.
_ACTIVE_JOBS: set[int] = set()


def _drain_build_log(path: Path, offset: int, remainder: str, progress) -> tuple[int, str]:
    """Incrementally stream a worker-written build log into the event feed.
    Returns the new (byte offset, partial-line remainder)."""
    try:
        with path.open("rb") as f:
            f.seek(offset)
            chunk = f.read()
    except OSError:  # not created yet (job still queued) or transiently unreadable
        return offset, remainder
    if not chunk:
        return offset, remainder
    lines = (remainder + chunk.decode("utf-8", "replace")).split("\n")
    for line in lines[:-1]:
        progress({"kind": "build_log", "text": line.rstrip()[:500]})
    return offset + len(chunk), lines[-1]


def _iso_age_s(iso: str) -> float:
    try:
        return (dt.datetime.now(dt.timezone.utc)
                - dt.datetime.fromisoformat(iso)).total_seconds()
    except (ValueError, TypeError):
        return 0.0


def _finalize_orphan(job) -> None:
    """Finalize a project whose driving web thread died (a restart): persist
    whatever the build produced, close the project row, and email the owner.
    The LLM-stage transcript is gone with the old process; the artifacts and the
    outcome are what the user actually needs back."""
    store = _store()
    p = store.get_project(job.project_id)
    if p is None or p.status != "running":
        return
    ws = Path(job.workspace)
    st: dict = {"project_id": job.project_id, "user_id": job.user_id,
                "brief": p.brief or "", "events": [], "status": None,
                "ok": (job.status == "done" and job.rc == 0),
                "spend": _project_spend_usd(job.project_id), "notify_force": True}
    if st["ok"] and ws.is_dir():
        st["stem"] = _read_project_stem(ws)
        st["zip"] = _zip_generated(ws)
        st["ok"] = bool(st["zip"])
    _persist_project(st)


def _reconcile_orphan_projects() -> None:
    """Close runs lost to a web restart BEFORE they reached the build queue.

    A run is an in-process thread tracked in _LIVE_RUNS; the project row's
    terminal outcome is written only by that thread. If the process dies during
    the LLM schematic stages the row is stranded at 'running' forever -- the
    build-job reaper above never sees it (no build_jobs row was ever enqueued),
    it renders as a phantom 'interrupted' with no way back, and it keeps burning
    a quota slot. Mark such rows 'interrupted' (durable, frees the slot),
    preserving any spend already incurred on the lost stages. The _LIVE_RUNS
    guard protects a healthy run live in THIS process; the query's NOT IN
    build_jobs filter leaves every build-stage orphan to _finalize_orphan, which
    can still recover artifacts. No email: an infra restart is not a design
    failure, and the row + its Retry button are enough."""
    store = _store()
    for p in store.list_orphaned_running_projects():
        if p.id in _LIVE_RUNS:  # live in this process -> not an orphan
            continue
        store.finish_project(p.id, "interrupted",
                             cost_usd=_project_spend_usd(p.id))


def _orphan_reaper() -> None:
    """Background janitor for the build queue (started once in main()):
    - requeue jobs whose claimant process died mid-build;
    - finalize finished jobs nobody is tailing (their web thread predates the
      last restart), so users get their board + email instead of a project stuck
      'running' forever;
    - fail queued jobs that have neither a driving thread nor a live worker
      (nothing will ever pick them up);
    - close runs lost to a restart before any build was enqueued (no build_jobs
      row to key on), so an LLM-stage death stops masquerading as 'running';
    - hourly, garbage-collect aged-out run workspaces (restarts are no longer
      the only reclaim point precisely because builds now survive them)."""
    ticks = 0
    while True:
        try:
            ticks += 1
            if ticks % 120 == 0:  # ~hourly at the 30s cadence
                _gc_workspaces()
            store = _store()
            store.requeue_stale_builds()
            worker_up = store.build_worker_alive()
            for job in store.list_unfinalized_builds():
                if job.id in _ACTIVE_JOBS or job.project_id is None:
                    continue
                if job.status in ("done", "failed"):
                    _finalize_orphan(job)
                elif (job.status == "queued" and not worker_up
                        and _iso_age_s(job.created_at) > 120):
                    # expect='queued': if a worker claimed it in the meantime,
                    # this no-ops instead of failing a now-running build.
                    if store.finish_build(job.id, rc=None, status="failed",
                                          expect="queued"):
                        _finalize_orphan(store.get_build_job(job.id))
            # Runs lost before they ever enqueued a build (LLM-stage death) have
            # no build_jobs row above to trigger finalization -- close them here.
            _reconcile_orphan_projects()
        except Exception:  # the janitor must survive any single bad row
            pass
        time.sleep(30)


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
    """The LEGACY env-var invite code (KICRAFT_SIGNUP_CODE, older boxes used
    KICRAFT_ACCESS_PASSWORD), read live (env loads in main(), so reading it at
    import time would capture an empty string). Still honored at signup as a
    plain free-tier code so links already handed out keep working, but new
    codes -- including ones that grant a paid tier for N days -- are minted in
    the DB from /admin/invites."""
    return (os.environ.get("KICRAFT_SIGNUP_CODE")
            or os.environ.get("KICRAFT_ACCESS_PASSWORD", "")).strip()


def _client_ip(request) -> str:
    """Best-effort client IP for the per-IP signup throttle. Honors
    X-Forwarded-For (the box runs behind a proxy) and falls back to the direct
    peer address. Empty string when no request/peer is available."""
    if request is None:
        return ""
    fwd = request.headers.get("x-forwarded-for") if hasattr(request, "headers") else None
    if fwd:
        # leftmost is the originating client; subsequent hops are proxies
        return fwd.split(",")[0].strip()
    peer = getattr(request, "client", None)
    return peer.host if peer else ""


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
    notify.mark_active(user.id)  # liveness for walk-away email suppression
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


@app.post("/billing/webhook")
async def stripe_webhook(request: Request):
    """Stripe webhook: the authoritative sync path for paid tiers.

    Raw Starlette endpoint (signature verification needs the exact body bytes).
    Flow: verify signature -> claim the event id (duplicate deliveries ack with
    200 and do nothing) -> sync from re-fetched subscription state. A handler
    failure releases the claim and 500s so Stripe retries the event later.
    """
    settings = Settings.from_env()
    if not settings.billing_enabled:
        return PlainTextResponse("billing not configured", status_code=503)
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    try:
        event = billing.verify_event(settings, payload, sig)
    except ValueError:
        return PlainTextResponse("bad signature", status_code=400)

    store = _store()
    event_id = str(event.get("id") or "")
    event_type = str(event.get("type") or "")
    if not event_id:
        return PlainTextResponse("no event id", status_code=400)
    if not store.record_billing_event(event_id, event_type):
        return PlainTextResponse("duplicate", status_code=200)
    try:
        outcome = await asyncio.to_thread(
            billing.handle_event, store, settings, event,
            billing.gateway(settings))
    except Exception as e:
        store.forget_billing_event(event_id)
        print(f"[billing] {event_type} {event_id} failed: {e}", flush=True)
        return PlainTextResponse("handler error", status_code=500)
    print(f"[billing] {event_type} {event_id}: {outcome}", flush=True)
    return PlainTextResponse("ok", status_code=200)


def _zip_generated(ws: Path) -> str | None:
    """Zip the user-facing KiCad project. Skips the internal .experiments/ tree
    (per-round search state, renders): it dwarfs the actual project files and
    is useless to the person opening the download in KiCad."""
    gen = ws / "generated"
    if not gen.is_dir():
        return None
    out = ws / "kicraft_project.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(gen.rglob("*")):
            rel = p.relative_to(gen)
            if ".experiments" in rel.parts or p.is_dir():
                continue
            zf.write(p, str(rel))
    return str(out)


def _erc_offenders(ws: Path) -> list[str]:
    """The §9.12 ERC error descriptions from the build's synthesis_check.json, or
    [] if ERC was not the failing check (so recovery only fires for real ERC
    errors, not other synth failures). check_erc stores up to 20 offenders."""
    try:
        sc = json.loads((_kicraft_dir(ws) / "synthesis_check.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    for c in sc.get("checks") or []:
        if "ERC" in str(c.get("name", "")) and not c.get("ok"):
            return [str(o) for o in (c.get("offenders") or [])]
    return []


# Single source of truth lives in kicraft.server.session (shared with the
# self-eval driver, WS6); kept re-exported here for back-compat callers/tests.
from .session import bom_reconcile_instruction as _bom_reconcile_instruction  # noqa: E402


def _synth_check_failures(ws: Path | None) -> list[str]:
    """Failing synthesis-check lines (check name + each offender) from the build's
    synthesis_check.json, so a FAILED run shows WHAT broke -- e.g. the 9.12 ERC
    dangling-wire list -- right next to the schematic. [] if all passed or absent."""
    if ws is None:
        return []
    try:
        sc = json.loads((_kicraft_dir(ws) / "synthesis_check.json").read_text(encoding="utf-8"))
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


def _live_board_urls(run_status: dict, project_dir: Path, token: str) -> dict:
    """KiCanvas source URLs for the current leaf and parent boards on disk.

    Reads ``preview_paths`` from run_status.json (written by the layout engine)
    and returns a dict with ``leaf`` and ``parent`` keys, each a (url, filename)
    tuple or None. Prefers routed over stamped boards.
    """
    previews = run_status.get("preview_paths") or {}
    hier = run_status.get("hierarchy") or {}
    if not previews:
        previews = hier.get("preview_paths") or {}
    result = {"leaf": None, "parent": None}

    def _relative(p: str) -> str | None:
        try:
            return Path(p).resolve().relative_to(project_dir.resolve()).as_posix()
        except (ValueError, OSError):
            return None

    for key in ("leaf_round_routed_board", "leaf_round_pre_route_board"):
        p = previews.get(key)
        if p and Path(p).is_file():
            rel = _relative(p)
            if rel:
                result["leaf"] = (f"/project/{token}/board/{rel}", Path(p).name)
                break

    for key in ("parent_routed_board", "parent_stamped_board"):
        p = previews.get(key)
        if p and Path(p).is_file():
            rel = _relative(p)
            if rel:
                result["parent"] = (f"/project/{token}/board/{rel}", Path(p).name)
                break

    return result


def _nonpower_symbol_count(sch_path: Path) -> int:
    """Number of placed (non-power) symbol instances in a ``.kicad_sch``.

    Each placed instance carries exactly one ``(lib_id "...")``; the lib_symbols
    definition block at the top of the file does not, so counting ``lib_id``
    occurrences counts instances. Power symbols (``power:GND`` / ``power:VCC`` /
    flags) are excluded so a rail sheet thick with power flags doesn't outrank
    the real circuit. Best-effort: an unreadable sheet counts 0."""
    try:
        text = sch_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return 0
    return sum(1 for m in re.finditer(r'\(lib_id "([^"]*)"', text)
               if not m.group(1).startswith("power:"))


def _primary_sheet_index(
    srcs: list[tuple[str, str]], project_dir: Path | None
) -> int:
    """Index into `srcs` of the sheet to show first in the Synthesize tab.

    `srcs` is root-first (see `_schematic_sources`). The "primary" sheet is the
    leaf with the most non-power symbols -- the design's centerpiece (e.g. the
    LED array, not the power header) -- so the user lands on a readable,
    content-rich schematic rather than the block-diagram root. Ties keep the
    earlier (alphabetical) leaf. Falls back to the first leaf (or the root for a
    single-sheet design) when there's no project_dir or nothing to count."""
    if len(srcs) <= 1:
        return 0
    first_leaf = 1  # srcs[0] is always the root container; leaves follow
    if project_dir is None:
        return first_leaf
    best_idx, best_count = first_leaf, 0
    for i in range(1, len(srcs)):
        count = _nonpower_symbol_count(project_dir / srcs[i][1])
        if count > best_count:
            best_idx, best_count = i, count
    return best_idx


def _render_synth_view(
    srcs: list[tuple[str, str]], stem: str, project_dir: Path | None = None
) -> KiCanvasView:
    """Sheet selector + KiCanvas for the Synthesize tab. `srcs` is (url, filename),
    root-first. One button per sheet (root='Overview', leaves by name) swaps the
    embed to that single sheet via `set_sources`, so KiCanvas renders it directly.
    Defaults to the design's centerpiece sheet (most non-power symbols; see
    `_primary_sheet_index`) so a readable schematic -- not the block-diagram root
    or an incidental connector sheet -- is what the user sees first."""
    root_file = f"{stem}.kicad_sch"

    def _label(fname: str) -> str:
        if fname == root_file:
            return "Overview"
        return fname[:-len(".kicad_sch")] if fname.endswith(".kicad_sch") else fname

    default_idx = _primary_sheet_index(srcs, project_dir)
    ui.label("Schematic").classes("text-xs font-medium").style("color:#94a3b8")
    selector = ui.row().classes("w-full flex-wrap gap-1")
    # Fill (nearly) the whole inspector column so the schematic is large enough to
    # read; the offset leaves room for the labels + sheet selector above it, and
    # the floor keeps it usable on short screens. The structured sheet/synthesis
    # data scrolls below. Shared with the Place/Route board view (_BUILD_VIEW_STYLE).
    view = KiCanvasView(
        [KiCanvasSource(srcs[default_idx][0], srcs[default_idx][1])],
        height="", style=_BUILD_VIEW_STYLE)
    with selector:
        for url, fname in srcs:
            ui.button(_label(fname),
                      on_click=lambda u=url, f=fname:
                      view.set_sources([KiCanvasSource(u, f)])) \
                .props("flat dense no-caps").classes("text-xs")
    return view


def _fmt_duration(seconds: float) -> str:
    """Human-readable duration: ``2m 34s`` or ``48s``."""
    if seconds < 1:
        return "0s"
    m, s = divmod(int(seconds), 60)
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def _fmt_eta(seconds: float) -> str:
    """Human-readable ETA: ``~2m`` or ``<1m``."""
    if seconds < 1:
        return "<1m"
    m = round(seconds / 60)
    if m < 1:
        return "<1m"
    return f"~{m}m"


def _build_place_route_progress(run_status: dict, leaf_prog: list[dict]) -> dict:
    """Build a 'progress' inspector section from run_status.json fields."""
    h = run_status.get("hierarchy") or {}
    phase_raw = run_status.get("phase", "")
    pct = run_status.get("progress_percent", 0)
    elapsed = run_status.get("elapsed_s", 0)
    eta = run_status.get("eta_s", 0)
    action = h.get("current_action") or run_status.get("current_action") or ""

    if phase_raw == "done" or h.get("current_stage") == "complete":
        phase = "Layout complete"
        bar_color = "#34d399"
        action = f"{h.get('leaf_total','?')} leaves, parent routed"
    elif h.get("current_stage") == "leaves" or phase_raw == "leaf":
        phase = "Leaf phase"
        bar_color = "#60a5fa"
        completed = h.get("leaf_workers", {}).get("completed", 0)
        total = h.get("leaf_total", 0)
        if total:
            action = f"solving leaf circuits ({completed}/{total})"
    elif h.get("current_stage") == "parent" or phase_raw in ("parent", "compose"):
        phase = "Parent phase"
        bar_color = "#a78bfa"
        round_num = run_status.get("round", 0)
        total_rounds = run_status.get("total_rounds", 0)
        if total_rounds:
            action = f"compose + route round {round_num}/{total_rounds}"
    else:
        phase = "Place & route"
        bar_color = "#60a5fa"

    elapsed_str = _fmt_duration(elapsed) if elapsed else ""
    eta_str = _fmt_eta(eta) if eta and phase_raw != "done" else ""

    chips = []
    for d in (leaf_prog or []):
        chips.append({
            "label": d.get("sheet_name", "?"),
            "done": d.get("status") == "Routed",
            "active": d.get("status") == "Placed",
        })

    return {
        "type": "progress",
        "title": "Layout progress",
        "phase": phase,
        "action": action,
        "percent": pct,
        "elapsed": elapsed_str,
        "eta": eta_str,
        "bar_color": bar_color,
        "items": chips,
    }


def _render_leaf_gallery(prog: list[dict], run_status: dict) -> None:
    """Render live KiCanvas views of the current leaf and parent boards, plus
    completed leaf thumbnails below. Built inside the place_route view_slot;
    the caller clears the slot before each rebuild."""
    leaf_src = run_status.get("_live_leaf_source")
    parent_src = run_status.get("_live_parent_source")

    if leaf_src or parent_src:
        with ui.row().classes("w-full gap-2").style(
                "height:calc(100vh - 500px);min-height:320px"):
            with ui.column().classes("flex-1 min-w-0 h-full gap-1"):
                ui.label("Current leaf").classes("text-xs font-medium").style("color:#60a5fa")
                if leaf_src:
                    KiCanvasView([KiCanvasSource(leaf_src[0], leaf_src[1])],
                                 height="", style="flex:1;min-height:0")
                else:
                    ui.label("(waiting for leaf board…)").classes("text-xs italic") \
                        .style("color:#64748b")
            with ui.column().classes("flex-1 min-w-0 h-full gap-1"):
                ui.label("Parent board").classes("text-xs font-medium").style("color:#a78bfa")
                if parent_src:
                    KiCanvasView([KiCanvasSource(parent_src[0], parent_src[1])],
                                 height="", style="flex:1;min-height:0")
                else:
                    ui.label("(waiting for parent board…)").classes("text-xs italic") \
                        .style("color:#64748b")

    if prog:
        ui.label("Leaves").classes("text-xs font-medium mt-2").style("color:#94a3b8")
        with ui.row().classes("w-full flex-wrap gap-2"):
            for d in prog:
                chip = "#34d399" if d.get("status") == "Routed" else "#fbbf24"
                with ui.column().classes("gap-0 items-center").style("width:118px"):
                    ui.image(d.get("url", "")).classes("w-full rounded border border-slate-700") \
                        .style("background:var(--kc-bg)")
                    ui.label(d.get("sheet_name", "?")).classes(
                        "text-xs truncate w-full text-center").style("color:#cbd5e1")
                    ui.label(d.get("status", "?")).classes("text-xs").style(f"color:{chip}")


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except OSError:
        return None


def _read_state_json(ws: Path) -> dict:
    """The progressively-built ConversationState (each stage commits a slot); {}
    if absent or mid-write."""
    try:
        data = json.loads(_state_path(ws).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def _endpoints_str(eps) -> str:
    return ", ".join(f"{p.get('ref')}.{p.get('pin')}" for p in (eps or []))


def _build_lines_for(stage: str, lines: list[str]) -> list[str]:
    """Build-log lines that belong to a given build sub-phase, by marker.
    Shares the live tabs' classifier so reopen and live agree on the split."""
    out, cur = [], None
    for ln in lines:
        s = _build_substage(ln)
        if s:
            cur = s
        if cur == stage:
            out.append(ln)
    return out

_REVIEW_FINDING_RE = __import__("re").compile(
    r"review (BLOCKER|WARNING|NOTE):\s*\[([^\]]+)\]\s*(.+)")

def _parse_review_findings(lines: list[str]) -> list[dict]:
    """Parse structured electrical-review findings from build_log lines.

    Each finding line has the form:
        review SEVERITY: [area] issue text
    Returns a list of {severity, area, issue} dicts.
    """
    findings = []
    for ln in lines:
        m = _REVIEW_FINDING_RE.search(ln)
        if m:
            findings.append({
                "severity": m.group(1).lower(),
                "area": m.group(2).strip(),
                "issue": m.group(3).strip(),
            })
    return findings


# ---- BOM part pricing (live LCSC lookups, cached) ---------------------------
# Resolved unit prices keyed by a part's lookup key ("id:C123" / "mpn:.." /
# "kw:.."). Shared process-wide (a key like "kw:5.1k 0402" is project-independent)
# and persisted per project to .kicraft/bom_prices.json so a reopen is instant. A
# cached value is a dict (priced), None (looked up, genuinely no price), or
# _UNAVAILABLE (every pricing source was unreachable -- e.g. the JLCPCB keyword API
# is WAF-blocked); a missing key means "not fetched yet" -> shown as "..." while a
# background fetch runs.
_PRICE_CACHE: dict[str, dict | None | object] = {}
_PRICE_INFLIGHT: set[str] = set()
_PRICE_LOCK = threading.Lock()
_PRICE_FILE = "bom_prices.json"
_FETCH_ERROR = object()  # sentinel: fetch raised unexpectedly; don't cache, retry later
# sentinel: a source was reachable-but-blocked / nothing could price this part.
# Cached (so we don't hammer a dead endpoint every render) but NOT persisted, so a
# reopen re-tries and it self-heals when the blocked source comes back.
_UNAVAILABLE = object()


class _SourceUnavailable(Exception):
    """A pricing source could not be reached (transport/HTTP error, or the only
    source for this part is currently blocked) -> cache as _UNAVAILABLE."""


# Bump when the pricing logic changes so persisted prices from the old logic are
# dropped and re-fetched. v3: price LCSC ids via the easyeda.com product endpoint
# (the JLCPCB keyword API is WAF-blocked); this also drops the frozen $0.00 caches
# written while every lookup was returning "no match". v4: the offline jlcparts
# catalog adds 10/100-pc break prices and keyword pricing; drop the single-price
# v3 caches so the breaks backfill. v5: keyword/MPN picks prefer Basic parts +
# a stock floor over bare-cheapest (KC-V8YWN8: the cheapest-Extended 1k pick
# 404'd on live LCSC); drop v4 caches so old churn-prone picks re-resolve.
# v6: live lcsc.com retail stock rides every pick (retail_stock/retail_min_buy;
# KC-4AZ7PE: JLC dump said millions, storefront had 0); OOS rows can no longer
# win kw/mpn picks and id keys never price a substitute part — drop v5 caches
# so old picks re-resolve and pick up the retail reading.
_PRICE_SCHEMA = 6

# easyeda.com product endpoint: serves the same data KiCraft fetches symbols and
# footprints from, and -- unlike jlcpcb.com's keyword-search API -- is NOT behind
# the Akamai WAF, so it is the one LCSC price source that still resolves. It
# carries a single unit price (no quantity ladder).
_SSL_CTX = ssl.create_default_context()
_EASYEDA_PRODUCT_URL = "https://easyeda.com/api/products/{cid}/components"
_EASYEDA_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Referer": "https://easyeda.com/",
}


def _jlcparts_price(cid: str) -> dict | None:
    """Price + stock + 10/100-pc breaks for an LCSC id from the offline JLC
    catalog (no network). None when the catalog is absent or carries no
    usable price for the part — callers fall through to the network source."""
    part = jlcparts.lookup(cid)
    if not part:
        return None
    ladder = part.get("ladder") or []
    unit = jlcparts.price_at(ladder, 1)
    if not unit or unit <= 0:
        return None
    return {"unit_price": unit, "lcsc": part["lcsc"],
            "stock": part.get("stock") or 0,
            "price_10": jlcparts.price_at(ladder, 10),
            "price_100": jlcparts.price_at(ladder, 100)}


def _easyeda_lcsc_price(cid: str) -> dict | None:
    """Unit price + stock for an LCSC ``C####`` via the easyeda.com product API.

    Prefers the in-stock LCSC tier. Returns ``{"unit_price","lcsc","stock"}`` or
    None when the part carries no price. Raises ``_SourceUnavailable`` on any
    transport/HTTP error so a transient block is retried, not frozen as 'no price'.
    The endpoint exposes a single unit price (no quantity ladder)."""
    req = urllib.request.Request(_EASYEDA_PRODUCT_URL.format(cid=cid),
                                 headers=_EASYEDA_HEADERS)
    try:
        with urllib.request.urlopen(req, timeout=15, context=_SSL_CTX) as resp:
            data = json.loads(resp.read())
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        raise _SourceUnavailable(f"easyeda {cid}: {e}") from e
    result = (data or {}).get("result") or {}
    best = None
    # NB: neither tier tracks the lcsc.com retail storefront — "lcsc" follows
    # JLC-side inventory (verified: parts the storefront shows sold out report
    # millions here) and "szlcsc" the China catalogue. Retail stock comes from
    # lcsc_retail.stock(), not from this endpoint.
    for tier in ("lcsc", "szlcsc"):
        d = result.get(tier) or {}
        try:
            price = float(d.get("price"))
        except (TypeError, ValueError):
            continue
        if price <= 0:
            continue
        cand = {"unit_price": price, "lcsc": str(d.get("number") or cid).upper(),
                "stock": int(d.get("stock") or 0)}
        if best is None or (cand["stock"] > 0 and best["stock"] == 0):
            best = cand
    return best


def _attach_retail(pick: dict) -> dict:
    """Best-effort live lcsc.com retail reading for the picked part.
    ``retail_stock`` None = unverified (endpoint disabled or unreachable):
    the cost UI shows it as such and ``_load_price_cache`` refuses to merge
    it, so a reopen re-checks instead of freezing an unknown. One shared
    wrap: lcsc_retail.attach_stock."""
    return lcsc_retail.attach_stock(pick, pick.get("lcsc"), nullable=True)


def _fetch_price(key: str) -> dict:
    """Resolve one price key to ``{"unit_price","lcsc","stock",
    "retail_stock","retail_min_buy"}`` (plus ``price_10``/``price_100``
    breaks when the offline catalog has them), or raise ``_SourceUnavailable``
    when nothing can price it right now.

    ``id:`` keys (curated-library + easyeda-vendored parts, which dominate BOM
    cost) price via the offline JLC catalog first (qty ladder + live stock, no
    network), then the easyeda.com endpoint. ``mpn:``/``kw:`` keys (un-vendored
    MPNs, generic passives) keyword-search the offline catalog; without it
    installed they have no source (jlcpcb.com's API is WAF-blocked). Every
    pick carries a live lcsc.com retail reading (see ``_attach_retail``)."""
    kind, _, query = key.partition(":")
    if kind == "id":
        pick = _jlcparts_price(query) or _easyeda_lcsc_price(query)
        if pick is not None:
            return _attach_retail(pick)
        raise _SourceUnavailable(f"no price source for {query}")
    pick = _pick_price(kind, query, jlcparts.search(query))
    if pick is None:
        raise _SourceUnavailable(f"keyword pricing unavailable for {query!r}")
    return _attach_retail(pick)


def _safe_fetch(key: str):
    try:
        return _fetch_price(key)
    except _SourceUnavailable:
        return _UNAVAILABLE
    except Exception:
        return _FETCH_ERROR


def _load_price_cache(ws: Path) -> None:
    """Merge a project's persisted prices into the process cache (best-effort).
    Files written by an older pricing schema (or the pre-schema flat format) are
    ignored so a _pick_price change re-fetches instead of serving stale prices."""
    try:
        data = json.loads((_kicraft_dir(ws) / _PRICE_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return
    if not isinstance(data, dict) or data.get("_schema") != _PRICE_SCHEMA:
        return
    with _PRICE_LOCK:
        for k, v in (data.get("prices") or {}).items():
            if isinstance(v, dict) and v.get("retail_stock") is None:
                # Retail was unverified when this was saved — leave the key
                # unmerged so the reopen re-fetches and self-heals.
                continue
            if k not in _PRICE_CACHE:
                _PRICE_CACHE[k] = v if isinstance(v, dict) else None


def _save_price_cache(ws: Path, keys: set[str]) -> None:
    """Persist this project's resolved keys (tagged with the pricing schema) so a
    reopen/restart is instant. Only stable results (a price dict or a genuine
    'no price' None) are persisted; an _UNAVAILABLE (source blocked) is skipped so
    the reopen re-tries it and pricing self-heals when the source comes back."""
    with _PRICE_LOCK:
        snap = {k: _PRICE_CACHE[k] for k in keys
                if k in _PRICE_CACHE and _PRICE_CACHE[k] is not _UNAVAILABLE}
    try:
        d = _kicraft_dir(ws)
        d.mkdir(parents=True, exist_ok=True)
        age = jlcparts.dump_age_days()
        (d / _PRICE_FILE).write_text(
            json.dumps({"_schema": _PRICE_SCHEMA,
                        # Staleness provenance: every REAL/stock verdict an
                        # audit derives from this file is as-of the dump, not
                        # live LCSC (KC-V8YWN8: a 3-week-old dump said 2M
                        # stock for a part live LCSC 404'd).
                        "catalog_age_days": round(age, 1) if age is not None else None,
                        "prices": snap}, indent=2),
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


def _price_for_lcsc(cid: str):
    """Cached price for one LCSC ``C####`` (the part-library detail view).

    Returns a price dict, ``_UNAVAILABLE``, or None ("not fetched yet" -> a
    background fetch is running; poll again). Reuses the shared BOM price cache, so
    a part priced here is already priced when it appears in a BOM and vice versa."""
    key = f"id:{cid}"
    with _PRICE_LOCK:
        if key in _PRICE_CACHE:
            return _PRICE_CACHE[key]
        if key in _PRICE_INFLIGHT:
            return None
        _PRICE_INFLIGHT.add(key)

    def work():
        r = _safe_fetch(key)
        with _PRICE_LOCK:
            if r is not _FETCH_ERROR:
                _PRICE_CACHE[key] = r
            _PRICE_INFLIGHT.discard(key)

    threading.Thread(target=work, daemon=True).start()
    return None


def _silk_sections(sj: dict, arts: dict) -> list[dict]:
    """Read-only surface of the authored silkscreen plan and what the build-tail
    placer actually stamped. Content lives in ``state.silk_plan`` (authored
    pre-build, like ``review_findings``); placement truth in
    ``state.artifacts.silk_placed`` / ``silk_dropped``. Silk is cosmetic — it
    never gates a build — so this is purely informational. Returns [] when no
    plan was committed (older projects, or authoring disabled/failed)."""
    silk = sj.get("silk_plan") or {}
    if not silk:
        return []
    labels = silk.get("labels") or []
    placed = [str(x) for x in (arts.get("silk_placed") or [])]
    place_dropped = [str(x) for x in (arts.get("silk_dropped") or [])]
    lint_dropped = [str(x) for x in (silk.get("dropped_at_lint") or [])]
    placed_ids = set(placed)
    # Legend lines carry synthetic ids "legend:N"; separate them from content labels.
    n_legend = sum(1 for p in placed if p.startswith("legend:"))
    n_labels_placed = len(placed) - n_legend

    secs: list[dict] = [{"type": "kv", "title": "Silkscreen", "rows": [
        ("title", silk.get("title") or "(board stem)"),
        ("board code", silk.get("board_code") or "(pending)"),
        ("rev", silk.get("rev", "")),
        ("legend lines", n_legend),
        ("labels placed", f"{n_labels_placed} / {len(labels)}"),
        ("dropped (no space)", len(place_dropped))]}]

    if labels:
        rows = []
        for lb in labels:
            lid = str(lb.get("id", ""))
            text = (lb.get("text") or "").replace("\n", " / ")
            anchor = (lb.get("anchor") or {}).get("ref") or ""
            rows.append([lb.get("kind", "note"), text, anchor,
                         lb.get("priority", 2),
                         "yes" if lid in placed_ids else "no"])
        note = None
        if any(lb.get("kind") == "table" for lb in labels):
            note = ("Tables (e.g. DIP-switch settings) are LLM-authored from the "
                    "netlist — verify against the physical part before assembly.")
        secs.append({"type": "table", "title": "Board labels",
                     "columns": ["type", "text", "near", "priority", "placed"],
                     "rows": rows, "note": note})
    if place_dropped:
        secs.append({"type": "list", "title": "Dropped — no clear silk space",
                     "items": place_dropped})
    if lint_dropped:
        secs.append({"type": "list", "title": "Rejected by content lint",
                     "items": lint_dropped})
    return secs


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
        secs: list[dict] = []
        opt = stage_diagram.functional_spec_diagram(sl)
        if opt is not None:
            secs.append({"type": "graph", "title": "Concept diagram", "option": opt})
        secs.append({"type": "table", "title": "Functional blocks",
                     "columns": ["name", "category", "purpose"],
                     "rows": [[b.get("name"), b.get("category"), b.get("purpose")]
                              for b in sl.get("blocks", [])]})
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
        secs: list[dict] = []
        opt = stage_diagram.architecture_diagram(sl)
        if opt is not None:
            secs.append({"type": "graph", "title": "Concept diagram", "option": opt})
        secs.append({"type": "table", "title": "Sheets", "columns": ["name", "stem", "function"],
                 "rows": [[s.get("name"), s.get("stem"), s.get("function")]
                          for s in sl.get("sheets", [])]})
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
        rows, total, priced, pending, blocked = [], 0.0, 0, False, 0
        for p in parts:
            key = _price_key(p)
            stock_cell = ""
            if key is None:
                cost = "n/a"
            elif key in prices:
                res = prices[key]
                if isinstance(res, dict):
                    total += res["unit_price"]
                    priced += 1
                    cost = _fmt_price(res["unit_price"])
                    stock_cell = (f"{_fmt_stock(res.get('stock'))} / "
                                  f"{_fmt_stock(res.get('retail_stock'))}")
                elif res is _UNAVAILABLE:
                    cost = "—"  # priced source unreachable -> not free, just unknown
                    blocked += 1
                else:
                    cost = "n/a"  # looked up, genuinely no price
            else:
                cost = "..."  # fetch in flight
                pending = True
            rows.append([p.get("ref"), p.get("value"), cost, stock_cell,
                         _vendor_cell(p, prices),
                         p.get("footprint"), p.get("sheet"), p.get("symbol")])
        if pending and priced == 0:
            total_txt, note = "pricing...", f"fetching live LCSC prices... (0/{len(parts)} so far)"
        else:
            total_txt = _fmt_total(total)
            note = (f"est. unit price, cheapest in-stock LCSC match "
                    f"({priced}/{len(parts)} priced); stock = JLCPCB assembly "
                    f"/ lcsc.com retail ('—' = retail unverified)")
            if pending:
                note = f"fetching live LCSC prices... ({priced}/{len(parts)} so far)"
            elif blocked:
                note += f"; {blocked} unavailable (live qty-break vendor API blocked)"
        secs = [{"type": "kv", "title": "Summary", "rows": [("parts", len(parts))]},
                {"type": "table", "title": "Parts",
                 "columns": ["ref", "value", "cost", "stock (JLC/retail)",
                             "vendor", "footprint", "sheet", "symbol"],
                 "rows": rows,
                 "foot": [["", "TOTAL (est.)", total_txt, "", "", "", "", ""]],
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
        if run_status:
            leaf_prog = run_status.get("_leaf_progress") or []
            secs.append(_build_place_route_progress(run_status, leaf_prog))
            h = run_status.get("hierarchy") or {}
            copper = h.get("copper_accounting") or {}
            kv_rows = []
            for k in ("best_score", "kept_count"):
                v = run_status.get(k)
                if v is not None:
                    kv_rows.append((k, v))
            if copper:
                kv_rows.append(("traces", copper.get("routed_total_trace_count", "?")))
                kv_rows.append(("vias", copper.get("routed_total_via_count", "?")))
            bm = h.get("board_metrics") or {}
            if bm:
                kv_rows.append((
                    "board size",
                    f"{bm.get('board_width_mm', 0):.1f} x "
                    f"{bm.get('board_height_mm', 0):.1f} mm",
                ))
                kv_rows.append((
                    "area utilization",
                    f"{float(bm.get('area_utilization', 0.0)) * 100:.1f}%",
                ))
                kv_rows.append(
                    ("aspect ratio", f"{float(bm.get('aspect_ratio', 0.0)):.2f}")
                )
            if kv_rows:
                secs.append({"type": "kv", "title": "Stats", "rows": kv_rows})
        else:
            log = _build_lines_for("place_route", build_lines)
            if log:
                secs.append({"type": "list", "title": "Place / route", "items": log})
        return secs

    if stage == "electrical_review":
        secs = []
        arts = sj.get("artifacts") or {}
        # Prefer structured findings: top-level state.review_findings (written
        # by the post-wiring review), then artifacts.review_findings (legacy
        # build-tail location); fall back to parsing build_log lines.
        raw = sj.get("review_findings") or arts.get("review_findings") or []
        if raw:
            findings = [{"severity": f.get("severity", "note"),
                         "area": f.get("area", ""),
                         "issue": f.get("issue", ""),
                         "suggestion": f.get("suggestion", "")}
                        for f in raw]
        else:
            findings = _parse_review_findings(build_lines)
        if findings:
            blocked = any(f["severity"] == "blocker" for f in findings)
            n_blockers = sum(1 for f in findings if f["severity"] == "blocker")
            n_warnings = sum(1 for f in findings if f["severity"] == "warning")
            n_notes = sum(1 for f in findings if f["severity"] == "note")
            status = "BLOCKED" if blocked else ("reviewed" if findings else "pending")
            secs.append({"type": "kv", "title": "Review outcome",
                         "rows": [("status", status),
                                  ("blockers", n_blockers),
                                  ("warnings", n_warnings),
                                  ("notes", n_notes)]})
            secs.append({"type": "findings", "title": "Findings",
                         "items": findings})
        else:
            log = _build_lines_for("electrical_review", build_lines)
            if log:
                secs.append({"type": "list", "title": "Electrical review", "items": log})
        return secs

    if stage == "fab":
        secs = []
        arts = sj.get("artifacts") or {}
        if arts:
            secs.append({"type": "kv", "title": "Artifacts", "rows": [
                ("status", arts.get("status", "")),
                ("fab package", "ready" if arts.get("fab_zip") else "pending"),
                ("STEP model", "ready" if arts.get("step_file") else "pending"),
                ("3D render", "ready" if arts.get("board_3d_png") else "pending")]})
            secs.extend(_silk_sections(sj, arts))
        log = _build_lines_for("fab", build_lines)
        if log:
            secs.append({"type": "list", "title": "Fab export", "items": log})
        return secs

    return []


def _collect_support_diagnostics(state: dict) -> dict:
    """Snapshot a run's troubleshooting context for a support report: what was
    asked, how far the run got, and the concrete failure evidence (build-log
    tail, failed synthesis checks, ERC offenders). Deliberately a bounded
    summary, not the full event stream (which _persist_project already saves
    per project): this payload is what automated review reads first."""
    events = state.get("events") or []
    # Read root: the scratch workspace, or (on a reopen) the durable project
    # root -- so a reopened FAILED project still yields its synth-check / ERC
    # evidence in the support report instead of an empty one.
    read_root = Path(state["ws"]) if state.get("ws") else None
    build_tail = [e.get("text", "") for e in events
                  if e.get("kind") == "build_log"][-60:]
    # Durable per-stage outcomes first (they cover stages run before a resume);
    # the in-memory event scan remains as the legacy fallback.
    ss = (read_state(read_root) if read_root else {}).get("stage_status") or {}
    stages_done = ([s for s in DESIGN_STAGES
                    if isinstance(ss.get(s), dict) and ss[s].get("ok")]
                   or [e.get("stage") for e in events if e.get("kind") == "stage_done"])
    ws = read_root
    if state.get("status"):
        run_status = state["status"]
    elif state.get("ok") is None:
        run_status = "running"
    else:
        run_status = "ok" if state.get("ok") else "failed"
    return {
        "captured_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "app_version": KICRAFT_VERSION,
        "board_code": state.get("board_code"),
        "project_id": state.get("project_id"),
        "user_id": state.get("user_id"),
        "brief": (state.get("brief") or "")[:2000],
        "stem": state.get("stem"),
        "run_status": run_status,
        "stages_done": stages_done,
        "awaiting_input": bool(state.get("awaiting_input")),
        "pcb_ready": bool(state.get("pcb_ready")),
        "spend_usd": state.get("spend"),
        "build_log_tail": build_tail,
        "failed_checks": _synth_check_failures(ws),
        "erc_errors": (_erc_offenders(ws) if ws else []),
    }


def _file_failure_report(state: dict) -> None:
    """Auto-file a support report for a failed run, so every failure is queued
    for automated review even when nobody is watching (a walk-away user, a
    queued build). The row id lands in state["support_report_id"] so the
    error dialog can attach the user's feedback to THIS report instead of
    filing a duplicate. Best-effort: reporting must never crash the worker."""
    try:
        state["support_report_id"] = _store().create_support_report(
            user_id=state.get("user_id"), project_id=state.get("project_id"),
            board_code=state.get("board_code"), kind="error_auto",
            diagnostics=_collect_support_diagnostics(state))
    except Exception:
        pass


def _investigation_log_dir() -> Path:
    """Where headless /kicraft-investigate runs tee their stdout: a sibling of
    the projects dir (not per-project, since a report may have no project)."""
    return _store().projects_dir.parent / "support_investigations"


def _auto_investigate_if_enabled(report_id: int) -> None:
    """Kick off a headless /kicraft-investigate run for a report a *user* filed
    (the manual Support button or feedback attached to a failed run), gated by
    the admin toggle. Not called for the silent per-failure auto-file, which
    would investigate every failed build. Best-effort: triage must never break
    the support dialog."""
    try:
        store = _store()
        if store.get_setting("support.auto_investigate", "1") != "1":
            return
        report = store.get_support_report(report_id)
        if report is None:
            return
        from . import investigate_runner
        investigate_runner.enqueue_investigation(
            store, report, log_dir=_investigation_log_dir())
    except Exception:
        pass


def _project_dir(state: dict) -> Path | None:
    """The durable project directory (projects_dir/<uid>/<pid>/) -- the ONE place a
    project lives AND builds (build-in-place: no scratch workspace, no copytree).
    Created on demand. None for a run with no user/project id yet (e.g. an admin
    self-eval scratch run, which still uses a throwaway tempdir)."""
    uid, pid = state.get("user_id"), state.get("project_id")
    if not uid or not pid:
        return None
    d = _store().projects_dir / str(uid) / str(pid)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _persist_project(state: dict) -> None:
    """Finalize a build-in-place run: write brief.txt + events.jsonl into the project
    dir (the build already wrote .kicraft/, generated/ and the zip there), point the
    row at the zip, and record the projects-table row. Best-effort: a persistence
    failure must never crash the worker."""
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
        # Never truncate an existing transcript with an empty snapshot: a
        # restart-recovery finalize (_finalize_orphan) has no in-memory events,
        # and the file on disk is the project's only persisted design timeline.
        events = state.get("events", [])
        if events or not (base / "events.jsonl").exists():
            with (base / "events.jsonl").open("w", encoding="utf-8") as f:
                for ev in events:
                    f.write(json.dumps(ev, ensure_ascii=False, default=str) + "\n")
        # Build-in-place: this project's .kicraft/, generated/ and kicraft_project.zip
        # are already under `base` (the build dir IS the durable dir) -- nothing to
        # copy. Point the row at the zip if the build produced one.
        z = base / "kicraft_project.zip"
        if z.is_file():
            zip_path = str(z)
            state["zip"] = zip_path
    except Exception as e:  # never crash the worker on persistence
        state.setdefault("events", []).append(
            {"kind": "build_log", "text": f"persist error: {e}"})
    finally:
        try:
            store.finish_project(pid, status, stem=stem, cost_usd=state.get("spend"),
                                 dir_path=dir_path, zip_path=zip_path)
        except Exception as e:
            # This write flips the durable row to its terminal status; losing it
            # silently leaves a phantom 'running' project with no diagnostics.
            print(f"[persist] finish_project({pid}, {status}) failed: {e}",
                  flush=True)
        # Catalog: stamp the quality badge and (re)index for the community browser.
        # reindex_search indexes only public, completed projects and removes anything
        # else, so a failed/awaiting/private run is correctly kept out. Best-effort:
        # a catalog hiccup must never crash the worker.
        try:
            if status == "ok" and dir_path:
                store.set_quality(pid, _quality_badge_from_ws(Path(dir_path)))
            store.reindex_search(pid)
        except Exception:
            pass
        # Walk-away notification: the run just reached a state worth an email
        # (done, failed, or parked on a question). Suppressed inside notify when
        # the user is actively watching, unless a restart-recovery finalize
        # forces it (nobody was watching that run's stream by definition).
        try:
            notify.notify_run_event(
                store, Settings.from_env(), user_id=uid, project_id=pid,
                status=status, brief=state.get("brief", ""),
                skip_if_active=not state.get("notify_force"))
        except Exception:
            pass


def _derived_statuses(ws: Path | None, sj: dict, project_status: str | None,
                      zip_ok: bool) -> dict[str, str]:
    """Every tab's durable status for a (re)opened project (the pure mapping is
    session.derive_stage_statuses; this reads the filesystem signals it needs
    from the workspace: generated sheets, synth-check failures, the routed
    board). Feeds StageTabs.set_statuses so a reopened project's stage icons
    reflect the stages that actually completed."""
    sheets = pcb = False
    checks_failed = False
    if ws is not None:
        pd = _discover_generated_dir(ws)
        if pd is not None:
            sheets = any(pd.glob("*.kicad_sch"))
            pcb = (pd / f"{pd.name}.kicad_pcb").is_file()
        checks_failed = bool(_synth_check_failures(ws))
    return derive_stage_statuses(sj, project_status=project_status,
                                 sheets_exist=sheets,
                                 synth_checks_failed=checks_failed,
                                 pcb_ready=pcb, zip_ok=zip_ok)


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


# Live design runs, keyed by project id. A run's worker registers its state dict
# here so a later page load (a reload, navigating back from /parts, a second tab)
# can re-attach to the in-flight run and stream its progress, instead of landing
# on a blank composer. Parked (awaiting_input) runs stay registered so answering
# from any page resumes the ONE live workspace; terminal runs are evicted once
# their artifacts are persisted (the saved project row takes over from there).
# Single-process registry: ui.run() serves from one process (reload=False), and
# dict get/set/pop are atomic under the GIL, so no lock is needed.
_LIVE_RUNS: dict[int, dict] = {}


def _project_run_live(state: dict) -> bool:
    """True when a run for this page's project is already live anywhere in this
    process. The rebuild/resume guards must check _LIVE_RUNS as well as the
    page-local dict: two tabs opened on the same FINISHED project each hold an
    independent state dict (open_project's non-live branch), so guarding only
    state['running'] lets both start a build in the same build-in-place
    project dir -- two cli_app processes racing on one state.json."""
    if state.get("running"):
        return True
    pid = state.get("project_id")
    if not pid:
        return False
    live = _LIVE_RUNS.get(pid)
    return live is not None and live is not state and bool(live.get("running"))


def _fresh_run_state() -> dict:
    """A blank per-design run state. This dict is the unit of a design run: the
    page that starts a run shares it with the worker thread (and, via
    _LIVE_RUNS, with any other page attached to the same run), so pages must
    never recycle one for a different design -- they re-bind to a fresh dict
    (see open_project / start in index). Page-local render bookkeeping (event
    cursor, view handles, mtime caches) lives in the page's own `view` dict."""
    return {
        "events": [], "running": False, "done": False, "ok": None,
        "spend": None, "zip": None, "ws": None, "token": None,
        "project_dir": None, "stem": None, "pcb_ready": False,
        # True only for a REOPENED project whose persisted status is
        # "failed" (a live failure sets ok=False instead); the rescue
        # manual-layout CTA keys on either signal.
        "failed": False,
        "user_id": None, "project_id": None, "brief": "",
        "status": None, "awaiting_input": False, "questions": [],
        "prices_rev": 0,
        # Support: the project's human-quotable id, and the auto-filed error
        # report's row id once a failure has been logged (see _file_failure_report).
        "board_code": None, "support_report_id": None,
    }


def _load_events(dir_path) -> list[dict]:
    """Read back the persisted event stream (events.jsonl) for a reopened project.
    Inverse of the write in _persist_project. The build timeline + LLM-reasoning
    panel renders from state['events']; events.jsonl is written at finalize but was
    never read back, so a reopened project showed a blank timeline. Best-effort and
    per-line tolerant so one corrupt line never blanks the whole history."""
    if not dir_path:
        return []
    f = Path(dir_path) / "events.jsonl"
    try:
        lines = f.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(ev, dict):
            out.append(ev)
    return out


def _pick_default_project(user_id: int):
    """The project the workspace should open by default: the newest parked run
    (it is blocked on the user), else the newest live run, else the newest
    finished design whose result the user has not seen yet. None = nothing
    needs attention -> show the blank composer."""
    projs = _store().list_projects(user_id)  # newest first
    for p in projs:
        if p.status == "awaiting_input" and (p.id in _LIVE_RUNS or p.dir_path):
            return p
    for p in projs:
        if p.status == "running" and p.id in _LIVE_RUNS:
            return p
    for p in projs:
        if p.status in ("ok", "failed") and p.dir_path and not p.viewed_at:
            return p
    return None


def _row_status_display(status, live) -> tuple[str, bool]:
    """The status text + live-colour flag for one 'My Projects' row.

    A row paints from two independent signals -- the durable DB `status` (text)
    and whether a live run dict exists in this process (`live`, the colour) --
    which must never contradict. Rules:
      * a 'running' row with no live worker reads as 'interrupted' (the run was
        lost to a restart/crash), and
      * the green "this is an active design" colour is granted only to a status
        that is genuinely live ('running'/'awaiting_input'); a stale
        'interrupted'/'failed'/'ok' row that briefly coexists with a live dict
        (e.g. during a rebuild, before its status flips back to 'running') must
        stay grey rather than masquerade as live.
    Returns (shown_text, is_live)."""
    shown = status
    if status == "running" and live is None:
        shown = "interrupted"
    is_live = live is not None and shown in ("running", "awaiting_input")
    return shown, is_live


def _execute_claimed_job_local(ws: Path, state: dict, job_id: int, progress,
                               *, kind: str = "build") -> int:
    """Execute our own (self-claimed) job in-process: the pre-queue behavior,
    kept as the fallback for deploys without the worker unit. The 30m wall
    clock restarts at the slot-acquired marker so time spent queued for a host
    build slot is not billed against the job."""
    timeout_s = 1800.0
    cmd = list(JOB_KIND_COMMANDS[kind])
    if kind == "build":
        quality = _store().build_quality_for_user(state.get("user_id"))
        if quality:  # tier override (free tier -> draft); None = default
            cmd += ["--quality", quality]
    # The build self-limits at 90% of the watchdog so the layout search
    # finalizes a best-so-far board instead of being SIGKILLed mid-round with
    # zero artifacts (mirrors the standalone worker); setdefault so an
    # operator env override wins.
    env = {**os.environ}
    env.setdefault("KICRAFT_BUILD_MAX_WALL_S", f"{timeout_s * 0.9:.0f}")
    proc = subprocess.Popen(
        cmd, cwd=str(ws), stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, errors="replace", bufsize=1, start_new_session=True,
        env=env)
    # Watchdog thread, mirroring the standalone worker: the deadline must fire
    # even when the build goes silent (a hung FreeRouting prints nothing, so a
    # per-line check blocks in readline forever and the job stays 'running'
    # until the web process restarts).
    wd = {"deadline": time.monotonic() + timeout_s, "killed": False}

    def _watchdog() -> None:
        while proc.poll() is None:
            if time.monotonic() > wd["deadline"]:
                wd["killed"] = True
                _kill_build(proc)
                return
            time.sleep(5.0)

    threading.Thread(target=_watchdog, daemon=True).start()
    for line in proc.stdout or []:
        text = line.rstrip()
        progress({"kind": "build_log", "text": text[:500]})
        if ACQUIRED_MARKER in text:
            wd["deadline"] = time.monotonic() + timeout_s
    rc = proc.wait()
    if wd["killed"]:
        progress({"kind": "build_log", "text": "[build exceeded 30m, killed]"})
    _store().finish_build(job_id, rc=rc)
    progress({"kind": "build_done", "ok": rc == 0})
    return rc


def _drive_build_queue(ws: Path, state: dict, progress, *,
                       kind: str = "build") -> int:
    """Run one deterministic job through the host build queue.

    Enqueues a build_jobs row, then drives one polling state machine:
    while 'queued' it emits queue events (position/ETA) and, when no
    standalone worker is heartbeating, self-claims the row once it
    reaches the queue head and a slot is free (FIFO across this process'
    run threads via the row order); while 'running' under the worker it
    tails the job's log file into the live event stream. A worker death
    mid-build surfaces here as the row going back to 'queued' (the
    reaper requeues it), which this loop simply handles again."""
    progress({"kind": "build_start"})
    store = _store()
    log_path = ws / ".kicraft" / "build.log"
    # The worker APPENDS to build.log, so start tailing at the current end
    # (measured BEFORE enqueue, so no new line can be missed): from byte 0 a
    # rebuild would replay the entire previous build's log into this run's
    # event stream before the new build writes anything.
    try:
        offset = log_path.stat().st_size
    except OSError:
        offset = 0
    job_id = store.enqueue_build(
        workspace=str(ws), project_id=state.get("project_id"),
        user_id=state.get("user_id"), log_path=str(log_path), kind=kind)
    _ACTIVE_JOBS.add(job_id)
    try:
        last_pos = None
        tail_buf = ""
        while True:
            job = store.get_build_job(job_id)
            if job is None:
                progress({"kind": "build_done", "ok": False})
                return 1
            offset, tail_buf = _drain_build_log(
                log_path, offset, tail_buf, progress)
            if job.status in ("done", "failed"):
                rc = job.rc if job.rc is not None else 1
                progress({"kind": "build_done", "ok": rc == 0})
                return rc
            if job.status == "queued":
                ahead, depth, running = store.build_queue_position(job_id)
                if (not store.build_worker_alive() and ahead == 0
                        and running < max(1, slot_count())):
                    if store.claim_build(job_id, f"pid:{os.getpid()}"):
                        return _execute_claimed_job_local(
                            ws, state, job_id, progress, kind=kind)
                    continue  # lost the claim race; re-read the row
                if ahead != last_pos:
                    last_pos = ahead
                    avg = store.avg_build_seconds()
                    progress({"kind": "queue", "position": ahead,
                              "depth": depth,
                              "eta_s": (avg * (ahead + 1)) if avg else None})
            else:
                last_pos = None  # re-announce position if requeued later
            time.sleep(1.0)
    finally:
        _ACTIVE_JOBS.discard(job_id)


def _rerun_build_worker(state: dict, kind: str) -> None:
    """Re-run one deterministic job (kind='manual_route': route + promote a
    saved manual layout; kind='build': full LLM-free rebuild, e.g. after a
    placement-rules edit or the Rebuild button) through the build queue,
    then refresh the persisted project.

    The outcome persists either way -- success exactly like a build, and
    failure flips the durable project to failed with the failed candidate
    board on display (no keep-the-last-good-state fallback: the user asked
    failures to be loudly inspectable in the UI, and the fab tab marks any
    earlier package stale). The build log streams to the tab and the
    walk-away email goes out via _persist_project."""
    ws = Path(state["ws"])
    pid = state.get("project_id")
    if pid:
        # Reset the durable row to 'running' BEFORE going live: a rebuild can
        # target a project whose durable status is still 'interrupted'/'failed'/
        # 'ok' (from a prior reap or build), and once _LIVE_RUNS holds this run
        # the projects page would otherwise paint that stale status with the
        # live-green colour. Mirrors the answer/continue resume paths.
        _store().update_project_status(pid, "running")
        _LIVE_RUNS[pid] = state

    def progress(ev):
        state["events"].append(ev)

    try:
        rc = _drive_build_queue(ws, state, progress, kind=kind)
        # Surface whatever board the build left behind -- on a failed verify
        # the promote tail keeps the failed candidate, and inspecting it is
        # the whole point of showing failures.
        pd = _discover_generated_dir(ws)
        if pd is not None:
            state["pcb_ready"] = (pd / f"{pd.name}.kicad_pcb").is_file()
        if rc != 0:
            state["ok"] = False
            return
        state["failed"] = False  # a rescued project is failed no longer
        state["zip"] = _zip_generated(ws)
        state["ok"] = bool(state["zip"])
    except Exception as e:  # surface, never crash the UI thread
        progress({"kind": "build_log", "text": f"error: {e}"})
        state["ok"] = False
    finally:
        if not state.get("ok"):
            # A failed (re)build has no valid package: drop any stale zip so
            # the persisted row offers no download that mismatches the board.
            state["zip"] = None
            state["failed"] = True
            _file_failure_report(state)
        _persist_project(state)
        state["done"] = True
        state["running"] = False
        if (pid and state.get("status") != "awaiting_input"
                and _LIVE_RUNS.get(pid) is state):
            _LIVE_RUNS.pop(pid, None)


def _ensure_workspace(state: dict) -> Path | None:
    """Ensure state["ws"] points at the project's build directory for a WRITE action
    (continue, edit, answer, manual layout, rebuild). Build-in-place: that directory
    IS the durable project dir, so this just resolves `_project_dir` -- no copy, no
    scratch workspace. Idempotent; a no-op once ws is set (live runs, and reopened
    projects whose ws was set on open). Returns None only for an id-less scratch run.
    """
    if state.get("ws"):
        return Path(state["ws"])
    pd = _project_dir(state)
    if pd is None:
        return None
    state["ws"] = str(pd)
    gen = _discover_generated_dir(pd)
    if gen is not None:
        state["stem"] = gen.name
        state["project_dir"] = str(gen)
        state["token"] = _register_project_dir(gen)
    return pd


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
          else _new_workspace("kicraft_web_"))
    state["ws"] = str(ws)
    state["status"] = None  # reset; set to awaiting_input only if we park
    pid = state.get("project_id")
    if pid:
        _LIVE_RUNS[pid] = state
    # Stamp every model call of this run with a stable id so the spend ledger can
    # attribute cost per run/stage (see kicraft.cli.web_cost_report).
    run_id = f"p{state.get('project_id')}-{int(time.time())}"

    # Core-components registry rows for the architecture/bom prompts, fetched
    # fresh each run so admin edits apply to reruns. Registry trouble must never
    # block a design run; it degrades to "no defaults block".
    core_defaults = None
    if Settings.from_env().enable_core_defaults:
        try:
            core_defaults = _store().list_core_components(include_disabled=False)
        except Exception:
            core_defaults = None

    def progress(ev):
        state["events"].append(ev)

    try:
        res = run_session(ws, state.get("brief", ""), stages, answers=answers,
                          instruction=instruction, progress=progress, run_id=run_id,
                          core_defaults=core_defaults)
        if res.get("guard"):
            state["spend"] = _project_spend_usd(state.get("project_id"))

        # BOM self-repair: wiring parked because the BOM lacks supporting parts
        # an IC needs (e.g. too few decoupling caps). That is KiCraft's own
        # problem to solve, not a question for the user — the wiring stage tags
        # such a park with reconcile_target="bom". Re-drive bom+wiring with the
        # concrete shortfall so the parts get added and wiring re-checks, then
        # adopt that outcome. Budgeted (BOM_RECONCILE_MAX_PASSES, with a
        # no-change cutoff) so it can never run away on cost while real deficit
        # CHAINS still resolve (fix-plan N3); if it still can't resolve, the
        # user is asked as a last resort. Shared with the self-eval driver
        # (kicraft.server.session).
        _bom_passes = int(state.get("bom_reconcile_passes") or 0)
        while (res.get("status") == "awaiting_input"
               and bom_reconcile_deficits(res)):
            _prev = _bom_passes
            res, _bom_passes = maybe_bom_reconcile(
                ws, state.get("brief", ""), res, progress=progress,
                run_id=run_id, core_defaults=core_defaults,
                reconcile_passes=_bom_passes,
            )
            if _bom_passes == _prev:
                break  # budget exhausted -> the park surfaces to the user
            state["bom_reconcile_passes"] = _bom_passes
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
        # R3: LLM electrical review post-wiring, BEFORE the build. The review
        # needs only intent+bom+netlist (all present at wiring commit). It
        # deliberately ignores routed geometry. A corroborated blocker gets
        # ONE wiring re-drive (mirroring the ERC-recovery pattern below),
        # then proceeds; a second blocker surfaces in the persisted findings.
        # run_post_wiring_review owns the lifecycle: stage events + build_log
        # lines for the GUI tab, durable persistence for reopen.
        try:
            from kicraft.design.cli_app import run_post_wiring_review

            def _rewire(instr: str) -> None:
                rr = run_session(ws, state.get("brief", ""), ["wiring"],
                                 instruction=instr, progress=progress,
                                 run_id=run_id)
                if rr.get("guard"):
                    state["spend"] = _project_spend_usd(state.get("project_id"))

            run_post_wiring_review(ws / ".kicraft" / "state.json", ws,
                                   progress, _rewire)
        except Exception:  # noqa: BLE001
            pass  # fail-soft: review must never block a sound build

        # Author the silkscreen content plan (LLM authors WHAT the board
        # says; the build tail decides WHERE — or drops it honestly). Runs
        # here because the build worker is a no-LLM process: the plan is
        # committed to state.silk_plan and consumed deterministically.
        try:
            from kicraft.design.cli_app import run_silk_plan_authoring

            board_code = None
            if pid:
                try:
                    proj = _store().get_project(pid)
                    board_code = getattr(proj, "board_code", None)
                except Exception:  # noqa: BLE001
                    board_code = None
            run_silk_plan_authoring(ws / ".kicraft" / "state.json", ws,
                                    progress, board_code=board_code)
        except Exception:  # noqa: BLE001
            pass  # fail-soft: silk is cosmetic, never blocks a build

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
            return _drive_build_queue(ws, state, progress)

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
        # Surface whatever board the build left behind: on a failed verify the
        # promote tail keeps the failed candidate so it can be inspected.
        pd = _discover_generated_dir(ws)
        if pd is not None:
            state["pcb_ready"] = (pd / f"{pd.name}.kicad_pcb").is_file()
        if rc != 0:
            state["ok"] = False
            return

        state["zip"] = _zip_generated(ws)
        state["ok"] = bool(state["zip"])
    except Exception as e:  # surface, never crash the UI thread
        progress({"kind": "build_log", "text": f"error: {e}"})
        state["ok"] = False
    finally:
        _persist_project(state)
        if state.get("ok") is False:  # terminal failure (parked runs have ok=None)
            _file_failure_report(state)
        state["done"] = True
        state["running"] = False
        # Terminal runs leave the live registry (their persisted project row,
        # written above, takes over). A parked run stays registered, so a reload
        # re-attaches to the live workspace and an answer resumes it in place.
        # Identity-guarded: never evict a newer run of the same project.
        if (pid and state.get("status") != "awaiting_input"
                and _LIVE_RUNS.get(pid) is state):
            _LIVE_RUNS.pop(pid, None)


def _design_worker(brief: str, state: dict) -> None:
    """Initial design from a brief: build IN the durable project dir (build-in-place,
    no scratch workspace), running all schematic stages + the deterministic build."""
    state["brief"] = brief
    pd = _project_dir(state)  # the durable dir IS the workspace
    state["ws"] = str(pd) if pd else None  # None only for an id-less scratch run
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

    from .client import CappedOpenRouterClient, make_client
    s = Settings.from_env()
    client = CappedOpenRouterClient(s)
    # Judge defaults to a stronger, steadier model than the design model, with a
    # routing-relaxed client when it differs from the design model.
    judge_model = (getattr(s, "eval_judge_model", None)
                   or getattr(s, "review_model", None) or s.model)
    judge_client = make_client(s.for_judge()) if judge_model != s.model else None
    started, finished = _project_times(project_dir, s.users_db_path)
    return evaluate_project(project_dir, client, judge_model=judge_model,
                            judge_client=judge_client,
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
            .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
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
    ui.query("body").style("background:var(--kc-bg)")
    with ui.column().classes("w-full max-w-3xl mx-auto p-6 gap-3"):
        with ui.row().classes("items-center justify-between w-full"):
            ui.label(f"KiCraft {title}").classes("text-2xl font-bold text-white")
            ui.button("Back", icon="arrow_back",
                      on_click=lambda: ui.navigate.to("/login")) \
                .props("flat dense color=white")
        with ui.card().classes("w-full").style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
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
    ui.query("body").style("background:var(--kc-bg)")
    with ui.card().classes("absolute-center w-96") \
            .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
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
        ui.separator().style("background:var(--kc-border)")
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
    ui.query("body").style("background:var(--kc-bg)")
    # Capture the client IP once at page load (it's the HTTP request peer, not
    # available inside the websocket submit callback) for the per-IP throttle.
    try:
        signup_ip = _client_ip(app.get_request())
    except Exception:
        signup_ip = ""
    with ui.card().classes("absolute-center w-96") \
            .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
        ui.label("Create your account").classes("text-2xl font-bold text-white")
        ui.label("Free tier: one design per week. No credit card.") \
            .classes("text-sm").style("color:#94a3b8")
        if prompt:  # arrived from a sample card: show what they'll build first
            ui.label(f'You will start with: "{prompt}"') \
                .classes("text-xs") \
                .style("color:#cbd5e1;border-left:3px solid #60a5fa;padding-left:8px")
        email = ui.input("Email").classes("w-full")
        pw = ui.input("Password", password=True, password_toggle_button=True).classes("w-full")
        # When the operator opens public signup (/admin/invites), the code becomes
        # optional: blank = free tier, while a code still applies its tier grant.
        open_signup = _store().signup_open()
        code = ui.input("Invite code (optional)" if open_signup else "Invite code",
                        password=True).classes("w-full")

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
            store = _store()
            code_str = (code.value or "").strip()
            grant = None  # an invite_codes row when a DB code applies its tier
            if code_str:
                grant = store.check_invite_code(code_str)
                legacy = _signup_code()
                if grant is None and not (
                        legacy and hmac.compare_digest(code_str, legacy)):
                    ui.notify("Invalid or disabled invite code.", color="negative")
                    return
            elif not store.signup_open():
                ui.notify("An invite code is required while KiCraft is in "
                          "private beta.", color="negative")
                return
            if not agree.value:
                ui.notify("Please accept the Terms of Service and Privacy Policy "
                          "to create an account.", color="warning")
                return
            # Per-IP signup throttle: blunts automated throwaway-account
            # creation across the multi-worker deployment (DB-backed counter).
            if store.count_recent_signups_by_ip(signup_ip, 3600) >= 5:
                ui.notify("Too many signups from this network — try again later.",
                          color="negative")
                return
            store.record_signup_attempt(signup_ip)
            try:
                user = store.create_user(
                    email.value or "", pw.value or "",
                    tier=grant["tier"] if grant else DEFAULT_TIER,
                    tier_expires_at=grant_expiry(grant["duration_days"]) if grant
                    else None,
                    accepted_terms_version=LEGAL_VERSION,
                    allow_training=bool(allow_training.value))
            except ValueError as e:
                ui.notify(str(e), color="negative")
                return
            if grant:  # only a real signup consumes one of the code's uses
                store.record_invite_use(grant["id"])
            # Mint + send the verification link. The user is auto-logged-in but
            # unverified, so the Design button stays disabled until they click.
            token = store.create_verification_token(user.id)
            if token:
                try:
                    s = Settings.from_env()
                    send_verification_email(
                        s, user.email,
                        f"{s.public_url}/verify?token={token}", _VERIFY_TTL_HOURS)
                except Exception:
                    pass  # mail trouble must never block a successful signup
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
        ui.separator().style("background:var(--kc-border)")
        _laforest_footer()


@ui.page("/forgot")
def forgot_page():
    """Public: request a password-reset link. Always shows the same neutral
    confirmation, so it never reveals whether an email is registered."""
    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    with ui.card().classes("absolute-center w-96") \
            .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
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
        ui.separator().style("background:var(--kc-border)")
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
    ui.query("body").style("background:var(--kc-bg)")
    user = _store().verify_reset_token(token)
    with ui.card().classes("absolute-center w-96") \
            .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
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


@ui.page("/verify")
def verify_page(token: str = ""):
    """Consume an email-verification token on GET. Single-use and short-lived, so
    consuming on GET is safe. On success the user can start designing on their
    next page load (no session change). The 'already verified -> success'
    fallback makes email-scanner prefetch double-hits harmless: the scanner
    consumes the token first, and the user's real click still sees success."""
    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    updated = _store().consume_verification_token(token) if token else None
    with ui.card().classes("absolute-center w-96") \
            .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
        if updated is not None:
            ui.label("Email confirmed").classes("text-2xl font-bold text-white")
            ui.label("You can start designing now. Head back to the workspace.") \
                .classes("text-sm").style("color:#94a3b8")
            ui.button("Go to workspace", on_click=lambda: ui.navigate.to("/")) \
                .classes("w-full")
            _legal_footer()
            return
        # Token invalid/expired/already-used. If the logged-in user is already
        # verified, treat it as success (scanner prefetch consumed it first).
        current = _current_user()
        if current is not None and current.email_verified:
            ui.label("Email confirmed").classes("text-2xl font-bold text-white")
            ui.label("Your email is already verified.") \
                .classes("text-sm").style("color:#94a3b8")
            ui.button("Go to workspace", on_click=lambda: ui.navigate.to("/")) \
                .classes("w-full")
            _legal_footer()
            return
        ui.label("Verification link invalid or expired") \
            .classes("text-2xl font-bold text-white")
        ui.label("Verification links are single-use and expire after 24 hours. "
                 "Request a fresh one to continue.") \
            .classes("text-sm").style("color:#94a3b8")

        def resend():
            u = _current_user()
            if u is None:
                ui.navigate.to("/login")
                return
            new_token = _store().create_verification_token(u.id)
            if new_token is None:
                ui.notify("A verification link was sent recently — check your inbox "
                          "in a minute.", color="warning")
                return
            try:
                s = Settings.from_env()
                send_verification_email(
                    s, u.email, f"{s.public_url}/verify?token={new_token}",
                    _VERIFY_TTL_HOURS)
            except Exception:
                pass
            ui.notify("Verification link sent. Check your email.", color="positive")

        if current is not None:
            ui.button("Resend verification link", on_click=resend).classes("w-full")
        else:
            ui.button("Sign in to resend",
                      on_click=lambda: ui.navigate.to("/login")).classes("w-full")
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
    ui.query("body").style("background:var(--kc-bg)")
    with ui.card().classes("absolute-center w-96") \
            .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
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
    ui.query("body").style("background:var(--kc-bg)")
    _mobile_head()

    def logout():
        for k in ("user_id", "email"):
            app.storage.user.pop(k, None)
        ui.navigate.to("/login")

    with ui.header().classes("items-center justify-between") \
            .style("background:var(--kc-surface);border-bottom:1px solid var(--kc-border)"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("your profile").classes("text-sm kc-tagline").style("color:#94a3b8")
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
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
            ui.label("Account").classes("text-base font-semibold text-white")
            with ui.row().classes("items-center gap-2"):
                ui.icon("mail").style("color:#94a3b8")
                ui.label(user.email).classes("text-sm").style("color:#e2e8f0")
            period = "week" if q["window_days"] <= 7 else "month"
            with ui.row().classes("items-center gap-2"):
                ui.badge(q["label"], color="primary")
                if user.subscription_status in ("active", "trialing"):
                    # A subscriber's tier_expires_at is period end + grace, so
                    # the raw date would read ~3 days late; say what matters.
                    ui.label("renews monthly") \
                        .classes("text-xs").style("color:#94a3b8")
                elif user.tier_expires_at:  # an invite-code grant with an end date
                    ui.label(f"until {user.tier_expires_at[:10]}") \
                        .classes("text-xs").style("color:#94a3b8")
                if q.get("unlimited"):
                    ui.label("Unlimited designs (staff).") \
                        .classes("text-sm").style("color:#94a3b8")
                else:
                    ui.label(f"{q['remaining']} of {q['limit']} designs left this "
                             f"{period}.").classes("text-sm").style("color:#94a3b8")
            ui.label(f"Member since {user.created_at[:10]}.") \
                .classes("text-xs").style("color:#64748b")

        with ui.card().classes("w-full gap-2") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
            ui.label("Plan & billing").classes("text-base font-semibold text-white")
            billing_on = Settings.from_env().billing_enabled
            if user.subscription_status:
                healthy = user.subscription_status in ("active", "trialing")
                ui.label(f"Subscription: {user.subscription_status}") \
                    .classes("text-xs") \
                    .style(f"color:{'#34d399' if healthy else '#f59e0b'}")
                if user.subscription_status == "past_due":
                    ui.label("Your last payment failed; Stripe is retrying. "
                             "Update your card in the billing portal to keep "
                             "your plan.").classes("text-xs").style("color:#f59e0b")
            elif user.tier == "free":
                ui.label("You're on the free plan.") \
                    .classes("text-xs").style("color:#94a3b8")
            else:
                ui.label(f"Your {q['label']} plan was granted by an invite or "
                         "by staff; no card on file.") \
                    .classes("text-xs").style("color:#94a3b8")
            with ui.row().classes("items-center gap-2"):
                ui.button("See plans", icon="workspace_premium",
                          on_click=lambda: ui.navigate.to("/pricing")) \
                    .props("flat dense no-caps color=primary").classes("text-xs")
                if billing_on and user.stripe_customer_id:
                    ui.button("Manage billing", icon="credit_card",
                              on_click=lambda: ui.navigate.to("/billing/portal")) \
                        .props("flat dense no-caps color=primary").classes("text-xs") \
                        .tooltip("Stripe portal: update card, switch plan, "
                                 "cancel, download invoices")

        with ui.card().classes("w-full gap-2") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
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
            def _file_data_request(action: str):
                rid = _store().create_support_report(
                    user_id=user.id, kind="data_request",
                    message=f"User requests {action} of all their data "
                            f"(account {user.email}).")
                ui.notify(f"Request filed (ref #{rid}). We'll follow up at "
                          f"{user.email}.", color="positive")

            ui.label("To export or delete all your data, file a request "
                     "below; we follow up at your account email.") \
                .classes("text-xs").style("color:#64748b")
            with ui.row().classes("gap-2"):
                ui.button("Request data export",
                          on_click=lambda: _file_data_request("an export")) \
                    .props("outline no-caps size=sm color=white")
                ui.button("Request account deletion",
                          on_click=lambda: _file_data_request("deletion")) \
                    .props("outline no-caps size=sm color=negative")

        with ui.card().classes("w-full gap-2") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
            ui.label("Community visibility").classes("text-base font-semibold text-white")
            if _can_make_private(user):
                ui.label("Choose which of your completed projects appear in the "
                         "community browser from your projects page.") \
                    .classes("text-xs").style("color:#94a3b8")
            else:
                ui.label("Your completed projects are public and appear in the "
                         "community browser. Upgrade to Pro to keep projects "
                         "private.").classes("text-xs").style("color:#94a3b8")
            with ui.row().classes("items-center gap-2 mt-1"):
                ui.button("Manage your projects", icon="folder",
                          on_click=lambda: ui.navigate.to("/projects")) \
                    .props("flat dense no-caps color=primary").classes("text-xs")
                ui.button("Open community browser", icon="travel_explore",
                          on_click=lambda: ui.navigate.to("/browse")) \
                    .props("flat dense no-caps color=primary").classes("text-xs")

        with ui.row().classes("w-full justify-end"):
            ui.button("Log out", icon="logout", on_click=logout) \
                .props("flat dense no-caps color=white").classes("text-xs")


@ui.page("/projects")
def projects_page():
    """The user's own designs on their own page: every project they've started,
    newest first, with Open / Download actions and -- for paid (pro/max) plans --
    a per-project Public toggle that lists a completed board in the community
    browser. This is the home the 'Your projects' workspace expander used to be;
    living on its own page keeps the composer uncluttered and gives visibility its
    own room. Login + consent gated like the rest of the app."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    if user.accepted_terms_version != LEGAL_VERSION:
        return RedirectResponse("/consent")

    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    _mobile_head()

    # Whether this plan may keep a project private. Re-checked server-side on every
    # toggle below, never trusting this initial UI gate.
    can_private = _can_make_private(user)

    with ui.header().classes("items-center justify-between") \
            .style("background:var(--kc-surface);border-bottom:1px solid var(--kc-border)"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("your projects").classes("text-sm kc-tagline").style("color:#94a3b8")
        with ui.row().classes("items-center gap-2"):
            ui.button("Browse community", icon="travel_explore",
                      on_click=lambda: ui.navigate.to("/browse")) \
                .props("flat dense no-caps color=white").classes("text-xs") \
                .tooltip("Browse and clone community projects")
            ui.button("Back to workspace", icon="arrow_back",
                      on_click=lambda: ui.navigate.to("/")) \
                .props("flat dense no-caps color=white").classes("text-xs")

    with ui.column().classes("w-full max-w-4xl mx-auto p-6 gap-3"):
        ui.label("Your projects").classes("text-2xl font-bold text-white")
        if can_private:
            ui.label("Every board you've designed. Flip a completed project to "
                     "Public to list it in the community browser; flip it back "
                     "to keep it private.").classes("text-sm").style("color:#94a3b8")
        else:
            ui.label("Every board you've designed. On the free plan, completed "
                     "projects are public and appear in the community browser -- "
                     "upgrade to Pro to keep projects private.") \
                .classes("text-sm").style("color:#94a3b8")

        rows_box = ui.column().classes("w-full gap-2")

        # One reusable confirm dialog, built once OUTSIDE the rebuildable rows so a
        # render_rows() refresh never deletes it mid-handler. A row's Delete button
        # parks its target here, then opens it.
        del_target = {"pid": None, "stem": None}
        with ui.dialog() as del_dialog, ui.card().classes("gap-2") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
            ui.label("Delete project?").classes("text-base font-semibold text-white")
            del_msg = ui.label().classes("text-sm").style("color:#e2e8f0")
            ui.label("This permanently removes the design, its files, and any "
                     "community listing. It can't be undone.") \
                .classes("text-xs").style("color:#f59e0b")
            with ui.row().classes("w-full justify-end gap-2 mt-1"):
                ui.button("Cancel", on_click=del_dialog.close) \
                    .props("flat no-caps color=white")
                ui.button("Delete permanently", icon="delete", color="negative",
                          on_click=lambda: _do_delete()).props("no-caps") \
                    .mark("confirm-delete")

        def _ask_delete(p):
            del_target.update(pid=p.id, stem=p.project_stem or f"project {p.id}")
            del_msg.text = f'"{del_target["stem"]}" will be deleted.'
            del_dialog.open()

        def _do_delete():
            pid = del_target["pid"]
            if pid is None:
                return
            # Re-check ownership + liveness server-side; never trust the UI gate.
            p = _store().get_project(pid)
            u = _current_user()
            if p is None or u is None or p.user_id != u.id:
                ui.notify("That project is not available.", color="warning")
                del_dialog.close()
                return
            if _LIVE_RUNS.get(pid) is not None:
                ui.notify("This design is still running -- open it and finish (or "
                          "start a new one) before deleting.", color="warning")
                del_dialog.close()
                return
            try:
                _store().delete_project(pid)
            except ValueError:
                # A build that survived a web restart is not in _LIVE_RUNS but
                # still owns the project dir (build-in-place).
                ui.notify("A build is still running for this design -- try "
                          "again once it finishes.", color="warning")
                del_dialog.close()
                return
            # Notify + refresh BEFORE closing the dialog: closing first drops the
            # slot ui.notify resolves through, so the toast is lost (and the list
            # never repaints). The clicked button lives in the dialog, not in
            # rows_box, so render_rows() does not delete it.
            ui.notify(f'Deleted "{del_target["stem"]}".', color="positive")
            render_rows()
            del_dialog.close()

        def render_rows():
            rows_box.clear()
            with rows_box:
                # Self-eval runs are NOT user projects -- they live on their own
                # page (/admin/self-eval), browsable to full depth there, and never
                # in a board list. Defensively drop any stray EV- row so a leftover
                # from before the decoupling never resurfaces here.
                shown = [p for p in _store().list_projects(user.id)
                         if not (p.board_code or "").startswith("EV-")]
                if not shown:
                    ui.label("No projects yet. Describe a board in the workspace "
                             "to begin.").classes("text-sm").style("color:#64748b")
                    return
                for p in shown:
                    _render_row(p)

        def _render_row(p):
            live = _LIVE_RUNS.get(p.id)
            with ui.card().classes("w-full gap-2") \
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                with ui.row().classes("w-full items-center gap-3"):
                    ui.label(p.project_stem or "(building…)") \
                        .classes("text-sm font-semibold").style("color:#e2e8f0")
                    if p.board_code:
                        ui.label(p.board_code).classes("text-xs font-mono") \
                            .style("color:#64748b") \
                            .tooltip("Board ID. Quote it when reporting an issue.")
                    # A 'running' row with no live worker is a run the server lost
                    # (restart/crash mid-run): say so instead of a phantom run.
                    shown, is_live = _row_status_display(p.status, live)
                    ui.label(shown).classes("text-xs").style(
                        "color:#4ade80" if is_live else "color:#94a3b8")
                    ui.label(p.created_at[:19].replace("T", " ")) \
                        .classes("text-xs").style("color:#64748b")
                    ui.space()
                    # Open deep-links into the workspace, which attaches to the
                    # live run when there is one (see index()'s ?project= branch).
                    if p.dir_path or live is not None:
                        ui.button("Open", icon="folder_open",
                                  on_click=lambda pp=p: ui.navigate.to(
                                      f"/?project={pp.id}")) \
                            .props("flat dense no-caps")
                    # A lost ('interrupted', incl. the dynamic running+no-live
                    # window) or failed run can be restarted in one click from its
                    # saved brief: deep-link it into the composer (no run starts
                    # until the user clicks Design, so no quota slot is spent here).
                    if live is None and shown in ("interrupted", "failed") and p.brief:
                        ui.button("Retry", icon="replay",
                                  on_click=lambda pp=p: ui.navigate.to(
                                      f"/?prompt={quote(pp.brief or '')}")) \
                            .props("flat dense no-caps")
                    if p.zip_path and Path(p.zip_path).is_file():
                        ui.button("Download", icon="download",
                                  on_click=lambda zp=p.zip_path: ui.download(zp)) \
                            .props("flat dense no-caps")
                    if p.dir_path and is_admin(user):
                        ui.button("Evaluate", icon="fact_check",
                                  on_click=lambda pp=p: open_eval_dialog(
                                      pp.dir_path,
                                      pp.project_stem or f"project {pp.id}")) \
                            .props("flat dense no-caps").style("color:#a78bfa")
                    # Delete is offered only for a project with no live worker, so
                    # an in-flight run can't be purged from under itself; the
                    # handler re-checks liveness + ownership before deleting.
                    if live is None:
                        ui.button("Delete", icon="delete",
                                  on_click=lambda pp=p: _ask_delete(pp)) \
                            .props("flat dense no-caps").style("color:#f87171") \
                            .mark("row-delete")
                # Visibility is only meaningful for a completed board -- the
                # community browser lists status=='ok' only. Paid plans get a real
                # toggle; free plans see the always-public note.
                if p.status == "ok":
                    with ui.row().classes("w-full items-center gap-2"):
                        if can_private:
                            sw = ui.switch("Public in the community",
                                           value=p.is_public)

                            def _flip(e, pid=p.id):
                                # Re-check tier server-side: a downgraded/forged
                                # session must not move a project in the catalog.
                                if not _can_make_private(_current_user()):
                                    ui.notify("Only paid plans can change "
                                              "visibility.", color="warning")
                                    return
                                _store().set_visibility(pid, bool(e.value))
                                _store().reindex_search(pid)
                                ui.notify(
                                    "Now public in the community."
                                    if e.value else "Now private.",
                                    color="positive")

                            sw.on_value_change(_flip)
                        else:
                            ui.icon("public").classes("text-sm") \
                                .style("color:#34d399")
                            ui.label("Public in the community browser") \
                                .classes("text-xs").style("color:#94a3b8")

        render_rows()


# --------------------------------------------------------------------------- #
# Pricing + billing (Stripe Checkout / Customer Portal). The /pricing page is
# public marketing (no model calls, like the landing page); the /billing/*
# pages are thin authed redirects into Stripe-hosted flows, so no card form
# ever renders here. Tier sync happens in the /billing/webhook endpoint above.
# --------------------------------------------------------------------------- #

def _pricing_bullets(tier_key: str, info: dict) -> list[str]:
    """Feature bullets per tier, with the quota numbers taken from TIERS so the
    page can never disagree with what quota_status actually enforces."""
    period = "week" if info["window_days"] <= 7 else "month"
    designs = f"{info['limit']} full design{'s' if info['limit'] != 1 else ''} a {period}"
    if tier_key == "free":
        return [designs,
                "The whole pipeline: schematic, real parts, placed + routed",
                "Projects are public and cloneable in the community",
                "No credit card required"]
    if tier_key == "pro":
        return [designs,
                "Everything in Free, with real headroom",
                "Keep projects private",
                "Cancel anytime in the billing portal"]
    return [designs,
            "Keep projects private",
            "Room to iterate on real products",
            "Cancel anytime in the billing portal"]


def _pricing_cta(user, tier_key: str, billing_on: bool) -> tuple[str, str | None]:
    """(label, href) for a tier card's button; href None renders a disabled
    chip. Logged-out paid CTAs route to signup first (hard rule: no checkout,
    nothing chargeable, before an account exists)."""
    if user is None:
        return ("Start free", "/signup") if tier_key == "free" \
            else (f"Get {TIERS[tier_key]['label']}", "/signup")
    if user.tier == tier_key:
        return ("Current plan", None)
    if tier_key == "free":
        # Paid users downgrade by cancelling in the portal, not by "buying" free.
        return ("Your plan if you cancel", None)
    if not billing_on:
        return ("Coming soon", None)
    verb = "Switch to" if user.tier in ("pro", "max") else "Upgrade to"
    return (f"{verb} {TIERS[tier_key]['label']}", f"/billing/checkout?tier={tier_key}")


def _render_pricing(user, error: str = "") -> None:
    """The public pricing page: three tier cards driven from TIERS, plus a small
    FAQ. Static marketing chrome shared with the landing page (kc_landing.css);
    deliberately no kc-reveal/JS dependency, so it renders fully without
    kc_landing.js."""
    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    ui.add_head_html('<link rel="stylesheet" href="/static/kc_landing.css">')

    billing_on = Settings.from_env().billing_enabled
    cards = []
    for key, info in TIERS.items():
        label, href = _pricing_cta(user, key, billing_on)
        featured = key == "pro"
        price = (f'<span class="kc-price-n">${info["price_usd"]}</span>'
                 '<span class="kc-price-per">/month</span>') if info["price_usd"] \
            else '<span class="kc-price-n">$0</span>'
        bullets = "".join(f"<li>{b}</li>" for b in _pricing_bullets(key, info))
        cta = (f'<a class="kc-btn {"kc-btn-primary" if featured else "kc-btn-ghost"} '
               f'kc-price-cta" href="{href}">{label}</a>') if href \
            else f'<span class="kc-price-chip">{label}</span>'
        badge = '<span class="kc-badge kc-price-pop">Most popular</span>' if featured else ""
        cards.append(
            f'<div class="kc-price-card{" kc-price-featured" if featured else ""}">'
            f'{badge}<h3>{info["label"]}</h3>'
            f'<div class="kc-price">{price}</div>'
            f'<ul class="kc-price-feats">{bullets}</ul>{cta}</div>')

    nav_actions = (
        '<a class="kc-nav-signin" href="/">Workspace</a>' if user else
        '<a class="kc-nav-signin" href="/login">Sign in</a>'
        '<a class="kc-btn kc-btn-primary" href="/signup">Start building</a>')
    error_banner = ('<p class="kc-price-error">We could not start checkout. '
                    'Nothing was charged; please try again.</p>') if error else ""

    faq = "".join(
        f'<div class="kc-faq-item"><h3>{q}</h3><p>{a}</p></div>' for q, a in (
            ("Can I cancel anytime?",
             "Yes. Manage billing on your profile opens the Stripe portal; "
             "cancelling stops renewal and your plan runs to the end of the "
             "period you already paid for."),
            ("Do I need a credit card for Free?",
             "No. The free tier only needs an email address."),
            ("How does payment work?",
             "Monthly card subscription handled by Stripe. KiCraft never sees "
             "or stores your card number."),
            ("What happens to my projects if I downgrade?",
             "They stay yours. Designs you made private stay private; new "
             "projects follow your current plan's rules.")))

    html = f"""
<div class="kc-landing">
  <div class="kc-nav"><div class="kc-wrap kc-nav-inner">
    <div class="kc-brand"><a class="kc-logo kc-grad" href="/"
      style="text-decoration:none">KiCraft</a>
      <span class="kc-tag">design a PCB from a sentence</span></div>
    <div class="kc-nav-actions">{nav_actions}</div>
  </div></div>

  <section class="kc-section">
    <div class="kc-wrap">
      <div class="kc-kicker">Pricing</div>
      <h2 class="kc-h2">Simple plans, real boards</h2>
      <p class="kc-lead">Every plan runs the full pipeline: hierarchical
        schematic, real orderable parts, placement, routing, and fab-ready
        output. Paid plans add headroom and private projects.</p>
      {error_banner}
      <div class="kc-pricing">{"".join(cards)}</div>
      <p class="kc-price-fine">Prices in USD. Subscriptions renew monthly and
        can be cancelled anytime; payment is processed by Stripe.</p>
    </div>
  </section>

  <section class="kc-section">
    <div class="kc-wrap">
      <div class="kc-kicker">Questions</div>
      <h2 class="kc-h2">Pricing FAQ</h2>
      <div class="kc-faq">{faq}</div>
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


@ui.page("/pricing")
def pricing_page(error: str = ""):
    _render_pricing(_current_user(), error=error)


@ui.page("/billing/checkout")
async def billing_checkout_page(tier: str = ""):
    """Authed redirect into Stripe Checkout for `tier` (or into the Customer
    Portal when the user already has a live subscription; switching plans there
    avoids a second subscription). The Stripe call runs off the event loop."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/signup")
    settings = Settings.from_env()
    if tier not in ("pro", "max") or not settings.billing_enabled:
        return RedirectResponse("/pricing")
    try:
        url = await asyncio.to_thread(
            billing.checkout_or_portal_url, _store(), settings, user, tier,
            billing.gateway(settings))
    except Exception as e:
        print(f"[billing] checkout failed for user {user.id}: {e}", flush=True)
        return RedirectResponse("/pricing?error=checkout")
    return RedirectResponse(url)


@ui.page("/billing/portal")
async def billing_portal_page():
    """Authed redirect into the Stripe Customer Portal (update card, switch
    plan, cancel, download invoices)."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    settings = Settings.from_env()
    if not (settings.billing_enabled and user.stripe_customer_id):
        return RedirectResponse("/profile")
    try:
        url = await asyncio.to_thread(
            billing.portal_url, _store(), settings, user, billing.gateway(settings))
    except Exception as e:
        print(f"[billing] portal failed for user {user.id}: {e}", flush=True)
        return RedirectResponse("/profile")
    return RedirectResponse(url)


@ui.page("/billing/success")
async def billing_success_page(session_id: str = ""):
    """Where Stripe Checkout returns on success. Syncs the subscription
    immediately (after confirming the session belongs to this user), so the
    upgrade shows without waiting on the webhook; the webhook remains the
    authoritative path for everything afterwards."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    settings = Settings.from_env()
    if settings.billing_enabled and session_id:
        try:
            outcome = await asyncio.to_thread(
                billing.sync_from_checkout_session, _store(), settings, user,
                session_id, billing.gateway(settings))
            print(f"[billing] success-page sync user {user.id}: {outcome}",
                  flush=True)
        except Exception as e:
            # The webhook will still land the upgrade; never fail the page.
            print(f"[billing] success-page sync failed for user {user.id}: {e}",
                  flush=True)
    fresh = _store().get_user(user.id) or user
    q = _store().quota_status(fresh)

    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    with ui.card().classes("absolute-center w-96 items-center gap-2") \
            .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
        ui.icon("check_circle").classes("text-4xl").style("color:#34d399")
        ui.label(f"You're on {q['label']}").classes("text-xl font-bold text-white")
        period = "week" if q["window_days"] <= 7 else "month"
        if q.get("unlimited"):
            ui.label("Unlimited designs (staff).") \
                .classes("text-sm").style("color:#94a3b8")
        else:
            ui.label(f"{q['remaining']} of {q['limit']} designs available this "
                     f"{period}.").classes("text-sm").style("color:#94a3b8")
        ui.label("A receipt is on its way from Stripe. Manage or cancel the "
                 "subscription anytime from your profile.") \
            .classes("text-xs text-center").style("color:#64748b")
        ui.button("Start designing", icon="rocket_launch",
                  on_click=lambda: ui.navigate.to("/")).classes("w-full")
        ui.button("Go to profile", on_click=lambda: ui.navigate.to("/profile")) \
            .props("flat dense no-caps").classes("text-xs")




# --------------------------------------------------------------------------- #
# Public project browser: a searchable, cross-user catalog of public, completed
# designs. Free users' projects are public; paid users' are private by default
# (toggle on /profile). Anyone can clone a public project into their own account.
# Reuses the samples card grid, the parts search idiom, the KiCanvas helpers, and
# the capability-token file serving -- the privacy boundary is that a file token
# is only ever minted for a project that passes _public_project_or_none.
# --------------------------------------------------------------------------- #
_QUALITY_CHIP = {
    "fab_ready": ("Fab-ready", "#34d399"),
    "erc_errors": ("Has ERC issues", "#fbbf24"),
    "unverified": ("Unverified", "#64748b"),
}


def _quality_chip(quality) -> None:
    """A small colored badge for a project's build quality."""
    label, color = _QUALITY_CHIP.get(quality or "unverified", _QUALITY_CHIP["unverified"])
    ui.label(label).classes("text-xs rounded").style(
        f"background:var(--kc-bg);border:1px solid {color};color:{color};padding:1px 8px")


def _stat_icon(icon: str, n) -> None:
    """An icon + count pair (views / clones / likes) for a card or detail header."""
    with ui.row().classes("items-center gap-1").style("color:#64748b"):
        ui.icon(icon).style("font-size:15px")
        ui.label(str(n or 0)).classes("text-xs")


def _can_make_private(user) -> bool:
    """Whether a user's plan may keep a project private. Free projects are always
    public (the community rule); only paid (pro/max) plans can opt out. Re-checked
    server-side on every visibility/clone mutation, not just in the UI."""
    return bool(user is not None and getattr(user, "tier", None) in ("pro", "max"))


def _quality_badge_from_ws(ws: Path | None) -> str:
    """Derive the catalog quality badge from a finished run's synthesis check (in
    the workspace): 'fab_ready' = passed clean, 'erc_errors' = ran but failed,
    'unverified' = no readable check. Mirrors eval.artifacts.parse_synthesis_check
    without importing the eval layer into the server."""
    if ws is None:
        return "unverified"
    try:
        sc = json.loads(
            (_kicraft_dir(ws) / "synthesis_check.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return "unverified"
    if not isinstance(sc, dict) or sc.get("status") is None:
        return "unverified"
    failed = sc.get("failed_checks")
    if failed is None:
        failed = [c.get("name") for c in (sc.get("checks") or []) if c.get("ok") is False]
    return "fab_ready" if (sc.get("status") == "ok" and not failed) else "erc_errors"


def _board_thumb_url(dir_path, stem) -> str | None:
    """A small board-preview URL for a browse card: the routed front render of the
    project's first leaf (falling back to its placement render), served via a signed
    token. None when no render exists yet (the card shows a placeholder)."""
    gen = _persisted_generated_dir(dir_path, stem)
    if gen is None:
        return None
    sub = gen / ".experiments" / "subcircuits"
    if not sub.is_dir():
        return None
    best = None
    for leaf in sorted(sub.iterdir()):
        renders = leaf / "renders"
        if not renders.is_dir():
            continue
        routed = _latest_render(renders, "routed_front_all")
        if routed is not None:
            best = routed
            break
        if best is None:
            best = _latest_render(renders, "pre_route_front_all")
    if best is None:
        return None
    tok = _register_project_dir(gen)
    rel = best.relative_to(gen).as_posix()
    return f"/project/{tok}/render/{rel}?v={int(best.stat().st_mtime)}"


def _board_source(gen: Path, stem: str, token: str):
    """(url, filename) for the project's board PCB, or None. Prefers <stem>.kicad_pcb
    (the file KiCanvas + serve_project_file expect in the dir root)."""
    cand = gen / f"{stem}.kicad_pcb"
    if cand.is_file():
        return (f"/project/{token}/{cand.name}", cand.name)
    pcbs = sorted(gen.glob("*.kicad_pcb"))
    if pcbs:
        return (f"/project/{token}/{pcbs[0].name}", pcbs[0].name)
    return None


def _load_persisted_state(dir_path) -> dict | None:
    """Read a persisted project's state.json for the detail page's BOM. The
    metadata dir is always `.kicraft/` via _state_path (one layout, no fallback;
    pre-Phase-4a projects were purged). None if unreadable."""
    if not dir_path:
        return None
    p = _state_path(Path(dir_path))
    if p.is_file():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
    return None


def _public_project_or_none(project_id):
    """The project for a public, completed id, else None. This is the detail page's
    privacy gate: a file token is minted only for a non-None result, so a private,
    failed, or missing project's files are never served through the public page."""
    try:
        pid = int(project_id)
    except (TypeError, ValueError):
        return None
    p = _store().get_project(pid)
    if p is None or p.status != "ok" or not p.is_public:
        return None
    return p


def _clone_project(source, cloner, make_private: bool):
    """Copy a public project into `cloner`'s account as a new, re-runnable project.

    Returns (new_project_id, None) on success or (None, reason) on failure. Consumes
    a quota slot like a normal design (it is an owned, re-runnable copy). The copied
    tree keeps the `.kicraft/` + `generated/` layout so the clone opens + rebuilds in
    place. events.jsonl is NOT copied: the clone starts a fresh history."""
    store = _store()
    if not store.can_design(cloner):
        return None, "quota"
    make_private = bool(make_private and _can_make_private(cloner))
    src = Path(source.dir_path) if source.dir_path else None
    if src is None or not src.is_dir():
        return None, "missing"
    pid = store.create_project(cloner.id, source.brief or "", is_public=not make_private)
    dst = store.projects_dir / str(cloner.id) / str(pid)
    zip_path = None
    try:
        dst.mkdir(parents=True, exist_ok=True)
        if (src / "brief.txt").is_file():
            shutil.copy2(src / "brief.txt", dst / "brief.txt")
        for sub in (".kicraft", "generated"):
            if (src / sub).is_dir():
                shutil.copytree(src / sub, dst / sub, dirs_exist_ok=True)
        if (src / "kicraft_project.zip").is_file():
            zip_path = str(dst / "kicraft_project.zip")
            shutil.copy2(src / "kicraft_project.zip", zip_path)
    except Exception:
        shutil.rmtree(dst, ignore_errors=True)
        try:
            store.finish_project(pid, "failed")  # free the reserved quota slot
        except Exception:
            pass
        return None, "copy_error"
    store.finish_project(pid, "ok", stem=source.project_stem, cost_usd=None,
                         dir_path=str(dst), zip_path=zip_path)
    store.set_cloned_from(pid, source.id)
    if source.quality:
        store.set_quality(pid, source.quality)
    store.increment_clone_count(source.id)
    try:
        store.reindex_search(pid)  # make the clone searchable if it is public
    except Exception:
        pass
    return pid, None


def _project_card(r: dict) -> None:
    """One browse-grid card for a public project dict (from list_public_projects)."""
    stem = r.get("project_stem") or "Untitled board"
    thumb = _board_thumb_url(r.get("dir_path"), r.get("project_stem"))
    card = ui.card().classes("w-72 gap-1 cursor-pointer") \
        .style("background:var(--kc-surface);border:1px solid var(--kc-border)")
    with card:
        if thumb:
            ui.image(thumb).props("fit=contain") \
                .style("height:150px;background:#0a0f1e").classes("w-full rounded")
        else:
            with ui.element("div").classes("w-full rounded flex items-center justify-center") \
                    .style("height:150px;background:#0a0f1e"):
                ui.icon("developer_board").style("color:var(--kc-border-strong);font-size:46px")
        with ui.row().classes("w-full items-center justify-between gap-1"):
            ui.label(stem).classes("text-base font-semibold text-white")
            _quality_chip(r.get("quality"))
        with ui.row().classes("items-center gap-3"):
            _stat_icon("visibility", r.get("view_count"))
            _stat_icon("content_copy", r.get("clone_count"))
            _stat_icon("favorite", r.get("like_count"))
        brief = (r.get("brief") or "").strip()
        if brief:
            ui.label(brief).classes("text-xs").style(
                "color:#94a3b8;display:-webkit-box;-webkit-line-clamp:2;"
                "-webkit-box-orient:vertical;overflow:hidden")
    card.on("click", lambda rr=r: ui.navigate.to(f"/p/{rr['id']}"))


@ui.page("/browse")
def browse_page():
    """The community browser: every public, completed design, searchable by part or
    function and sortable by popularity / newest / most-cloned. Login + consent gated
    like the rest of the app; cloning and liking happen on a project's detail page."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    if user.accepted_terms_version != LEGAL_VERSION:
        return RedirectResponse("/consent")

    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    _mobile_head()

    with ui.header().classes("items-center justify-between") \
            .style("background:var(--kc-surface);border-bottom:1px solid var(--kc-border)"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("community browser").classes("text-sm kc-tagline").style("color:#94a3b8")
        ui.button("Back to workspace", icon="arrow_back",
                  on_click=lambda: ui.navigate.to("/")) \
            .props("flat dense no-caps color=white").classes("text-xs")

    PAGE = 24
    state = {"offset": 0, "deb": None}

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1200px"):
        ui.label("Community projects").classes("text-2xl font-bold text-white")
        ui.label("Browse boards the KiCraft community has built. Search by a part "
                 "(like esp32) or by what it does (like plant watering), then open "
                 "one to view it and clone it into your own workspace.") \
            .classes("text-sm").style("color:#94a3b8")

        with ui.row().classes("w-full items-center gap-3"):
            search = ui.input(
                placeholder="Search by part (esp32) or function (plant watering)…") \
                .props("dense outlined clearable dark").classes("flex-grow") \
                .style("min-width:240px")
            sort_toggle = ui.toggle(
                {"popularity": "Popular", "new": "New", "clones": "Most clones"},
                value="popularity").props("dense no-caps")
            badge_toggle = ui.toggle(
                {"all": "All", "fab_ready": "Fab-ready", "erc_errors": "Has ERC issues"},
                value="all").props("dense no-caps")
        count_label = ui.label().classes("text-xs").style("color:#64748b")
        grid = ui.row().classes("w-full flex-wrap gap-4")
        more_row = ui.row().classes("w-full justify-center")

        def _q():
            return (search.value or "").strip() or None

        def _badge():
            return None if badge_toggle.value == "all" else badge_toggle.value

        def _maybe_more(total):
            more_row.clear()
            if state["offset"] < total:
                with more_row:
                    ui.button(f"Load more ({total - state['offset']} more)",
                              on_click=load_more).props("flat no-caps color=primary")

        def add_cards(rows):
            with grid:
                for r in rows:
                    _project_card(r)

        def render():
            grid.clear()
            state["offset"] = 0
            q, badge = _q(), _badge()
            total = _store().count_public_projects(query=q, badge=badge)
            rows = _store().list_public_projects(
                sort=sort_toggle.value, query=q, badge=badge, limit=PAGE, offset=0)
            suffix = " found" if (q or badge) else ""
            count_label.text = f"{total} project{'' if total == 1 else 's'}{suffix}"
            add_cards(rows)
            state["offset"] = len(rows)
            _maybe_more(total)

        def load_more():
            q, badge = _q(), _badge()
            rows = _store().list_public_projects(
                sort=sort_toggle.value, query=q, badge=badge,
                limit=PAGE, offset=state["offset"])
            add_cards(rows)
            state["offset"] += len(rows)
            _maybe_more(_store().count_public_projects(query=q, badge=badge))

        def schedule_render():
            if state["deb"] is not None:
                state["deb"].cancel()
            state["deb"] = ui.timer(0.25, render, once=True)  # debounce typing

        search.on_value_change(lambda: schedule_render())
        sort_toggle.on_value_change(lambda: render())
        badge_toggle.on_value_change(lambda: render())
        render()


@ui.page("/p/{project_id}")
def public_project_page(project_id: str):
    """A public project's detail page: schematic + board (KiCanvas), BOM, community
    metrics, and the Like + Clone actions. Login + consent gated. A private, failed,
    or missing project renders a neutral 'not available' panel and mints no token."""
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    if user.accepted_terms_version != LEGAL_VERSION:
        return RedirectResponse("/consent")

    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    kicanvas_head()
    _mobile_head()

    with ui.header().classes("items-center justify-between") \
            .style("background:var(--kc-surface);border-bottom:1px solid var(--kc-border)"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("community project").classes("text-sm kc-tagline").style("color:#94a3b8")
        ui.button("Back to browse", icon="arrow_back",
                  on_click=lambda: ui.navigate.to("/browse")) \
            .props("flat dense no-caps color=white").classes("text-xs")

    p = _public_project_or_none(project_id)
    if p is None:
        with ui.column().classes("w-full mx-auto p-8 gap-2 items-center") \
                .style("max-width:760px"):
            ui.icon("lock").style("color:#64748b;font-size:40px")
            ui.label("This project isn't available.").classes("text-lg text-white")
            ui.label("It may be private, still building, or no longer exists.") \
                .classes("text-sm").style("color:#94a3b8")
            ui.button("Browse community projects", icon="travel_explore",
                      on_click=lambda: ui.navigate.to("/browse")).props("flat no-caps")
        return

    # One view per browser session per project, so a refresh doesn't inflate the count.
    viewed = app.storage.user.setdefault("viewed_projects", [])
    if p.id not in viewed:
        try:
            _store().record_view(p.id)
        except Exception:
            pass
        viewed.append(p.id)
        app.storage.user["viewed_projects"] = viewed

    gen = _persisted_generated_dir(p.dir_path, p.project_stem)
    token = _register_project_dir(gen) if gen else None

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1200px"):
        with ui.row().classes("w-full items-center justify-between gap-2"):
            ui.label(p.project_stem or "Untitled board") \
                .classes("text-2xl font-bold text-white")
            _quality_chip(p.quality)
        if (p.brief or "").strip():
            ui.label(p.brief).classes("text-sm").style("color:#94a3b8")

        with ui.row().classes("items-center gap-4 flex-wrap"):
            _stat_icon("visibility", p.view_count)
            _stat_icon("content_copy", p.clone_count)
            like_state = {"liked": _store().has_liked(user.id, p.id), "n": p.like_count}

            def _refresh_like():
                like_btn.props(
                    f"icon={'favorite' if like_state['liked'] else 'favorite_border'}")
                like_btn.set_text(str(like_state["n"]))

            def _on_like():
                like_state["liked"] = _store().toggle_like(user.id, p.id)
                fresh = _store().get_project(p.id)
                like_state["n"] = fresh.like_count if fresh else like_state["n"]
                _refresh_like()

            like_btn = ui.button(on_click=_on_like) \
                .props("flat dense no-caps color=white").classes("text-xs")
            _refresh_like()
            _clone_button(p, user)
            ui.label("Community project").classes("text-xs").style("color:#64748b")

        if gen and token:
            srcs = _schematic_sources(gen, p.project_stem or "", token)
            if srcs:
                with ui.card().classes("w-full") \
                        .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                    _render_synth_view(srcs, p.project_stem or "", gen)
            board = _board_source(gen, p.project_stem or "", token)
            if board:
                with ui.card().classes("w-full") \
                        .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                    ui.label("Board").classes("text-xs font-medium").style("color:#94a3b8")
                    KiCanvasView([KiCanvasSource(board[0], board[1])], height="h-[520px]")
        else:
            ui.label("This project's files aren't available to preview.") \
                .classes("text-sm").style("color:#64748b")

        _render_bom_table(_load_persisted_state(p.dir_path))


def _clone_button(source, user) -> None:
    """Render the Clone action: paid users get a 'make private' dialog (private by
    default), free users clone publicly in one click. The tier gate is re-checked in
    _clone_project, so the dialog is convenience, not the security boundary."""
    def do_clone(make_private):
        pid, err = _clone_project(source, user, make_private)
        if err == "quota":
            ui.notify("You've used your design quota for this period. "
                      "See /pricing to upgrade.",
                      color="warning")
            return
        if err is not None or pid is None:
            ui.notify("Couldn't clone this project. Please try again.", color="negative")
            return
        ui.notify("Cloned into your workspace.", color="positive")
        # Deep-link straight into the fresh copy: a plain "/" runs the default
        # pick, where an older parked run would outrank the new clone.
        ui.navigate.to(f"/?project={pid}")

    if _can_make_private(user):
        def open_dialog():
            with ui.dialog() as dlg, ui.card().classes("gap-2") \
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                ui.label("Clone this project").classes("text-base font-bold text-white")
                ui.label("A copy lands in your workspace; you can open and re-run it.") \
                    .classes("text-xs").style("color:#94a3b8")
                priv = ui.switch("Make my clone private", value=True)
                with ui.row().classes("w-full justify-end gap-2"):
                    ui.button("Cancel", on_click=dlg.close).props("flat no-caps")
                    ui.button("Clone",
                              on_click=lambda: (dlg.close(), do_clone(priv.value))) \
                        .props("color=primary unelevated no-caps")
            dlg.open()
        ui.button("Clone", icon="content_copy", on_click=open_dialog) \
            .props("color=primary unelevated no-caps")
    else:
        ui.button("Clone", icon="content_copy", on_click=lambda: do_clone(False)) \
            .props("color=primary unelevated no-caps")


def _render_bom_table(state) -> None:
    """A compact, read-only bill of materials for the detail page."""
    parts = (((state or {}).get("bom") or {}).get("parts")) or []
    with ui.card().classes("w-full gap-1") \
            .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
        ui.label(f"Bill of materials ({len(parts)} parts)") \
            .classes("text-xs font-medium").style("color:#94a3b8")
        if not parts:
            ui.label("No parts listed.").classes("text-xs").style("color:#64748b")
            return
        with ui.row().classes("w-full items-center gap-2 text-xs font-bold") \
                .style("color:#64748b"):
            ui.label("ref").style("width:64px")
            ui.label("value").style("width:170px")
            ui.label("mpn / sourcing").classes("flex-grow")
            ui.label("sheet").style("width:120px")
        for prt in parts:
            with ui.row().classes("w-full items-center gap-2 text-xs") \
                    .style("border-top:1px solid var(--kc-border);padding:3px 0"):
                ui.label(str(prt.get("ref") or "")).classes("font-mono") \
                    .style("width:64px;color:#e2e8f0")
                ui.label(str(prt.get("value") or "")).style("width:170px;color:#cbd5e1")
                ui.label(str(prt.get("mpn") or prt.get("sourcing_note") or "")) \
                    .classes("flex-grow font-mono").style("color:#94a3b8")
                ui.label(str(prt.get("sheet") or "")).style("width:120px;color:#64748b")


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
    ui.query("body").style("background:var(--kc-bg)")
    kicanvas_head()
    _mobile_head()
    if any(s.has_3d() for s in available_samples()):
        # <model-viewer> (Google @google/model-viewer 4.0.0, BSD-3) — self-hosted
        # like kicanvas.js. The explorer is where the interactive 3D board lives.
        ui.add_head_html(
            '<script type="module" src="/static/model-viewer.min.js"></script>')

    with ui.header().classes("items-center justify-between") \
            .style("background:var(--kc-surface);border-bottom:1px solid var(--kc-border)"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("example boards").classes("text-sm kc-tagline").style("color:#94a3b8")
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
                # Interactive 3D board first (the show-off view), then the real
                # schematic and routed board in KiCanvas.
                if s.has_3d():
                    with ui.card().classes("w-full") \
                            .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                        ui.label("3D model").classes("text-xs font-medium") \
                            .style("color:#94a3b8")
                        # sanitize=False: NiceGUI would strip the custom element.
                        ui.html(_sample_model_viewer(s), sanitize=False) \
                            .classes("w-full")
                # Schematic and board are both rendered visible (not in tabs/dialogs):
                # a KiCanvas WebGL canvas built inside a hidden container can size to
                # zero and never repaint, so keeping both on-screen avoids that.
                with ui.card().classes("w-full") \
                        .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                    _render_synth_view(s.schematic_sources(), s.stem, s.dir)
                with ui.card().classes("w-full") \
                        .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                    ui.label("Board").classes("text-xs font-medium").style("color:#94a3b8")
                    url, name = s.board_source()
                    KiCanvasView([KiCanvasSource(url, name)], height="h-[520px]")
            ui.run_javascript(
                "document.querySelector('.kc-viewer')?."
                "scrollIntoView({behavior:'smooth',block:'start'})")

        with grid:
            for s in samples:
                card = ui.card().classes("w-72 gap-1 cursor-pointer") \
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border)")
                with card:
                    ui.image(s.board_png_url).props("fit=contain") \
                        .style("height:150px;background:#0a0f1e").classes("w-full rounded")
                    ui.label(s.title).classes("text-base font-semibold text-white")
                    ui.label(f"{s.sheets} sheets / {s.parts} parts / routed") \
                        .classes("text-xs").style("color:#64748b")
                    ui.label(s.blurb).classes("text-xs").style("color:#94a3b8")
                card.on("click", lambda ss=s: open_sample(ss))


def _mobile_head() -> None:
    """Load the mobile/tablet-only stylesheet. Every rule in it sits under a
    max-width media query, so desktop (>=1024px) rendering is unaffected."""
    ui.add_head_html('<link rel="stylesheet" href="/static/kc_mobile.css">')


def _parts_header(subtitle_btn_label: str, subtitle_btn_target: str) -> None:
    """The shared dark header for the /parts pages: brand + a single back button."""
    _mobile_head()
    with ui.header().classes("items-center justify-between") \
            .style("background:var(--kc-surface);border-bottom:1px solid var(--kc-border)"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("part library").classes("text-sm kc-tagline").style("color:#94a3b8")
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
    ui.query("body").style("background:var(--kc-bg)")
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
                        "background:var(--kc-surface);border:1px solid var(--kc-border)")
                    with row:
                        with ui.column().classes("gap-0").style("min-width:170px"):
                            ui.label(m.mpn).classes("text-sm font-bold text-white")
                            ui.label(m.name).classes("text-xs").style("color:#64748b")
                        ui.label(m.description).classes("text-xs flex-grow").style(
                            "color:#94a3b8;display:-webkit-box;-webkit-line-clamp:2;"
                            "-webkit-box-orient:vertical;overflow:hidden")
                        _tier_badge(p.tier)
                        ui.badge(m.maturity, color="grey-7")
                        code = (m.sourcing or {}).get("lcsc", "").strip().upper()
                        if _LCSC_CODE_RE.match(code):
                            ui.label(code).classes("text-xs font-mono rounded") \
                                .style("background:var(--kc-border);color:#94a3b8;padding:2px 8px")
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
    ui.query("body").style("background:var(--kc-bg)")
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
                    "background:var(--kc-surface);border:1px solid var(--kc-border);min-width:300px"):
                ui.label("Symbol").classes("text-xs font-medium").style("color:#94a3b8")
                # ?v=<content-hash> makes the URL content-addressed so the
                # long-lived immutable cache header in serve_part_preview is
                # safe: an edited bundle changes the hash, hence the URL.
                ver = _content_hash_key(m)
                syms = symbol_svgs(part) if kicad_cli_available() else []
                if syms:
                    for i in range(len(syms)):
                        ui.image(f"/part-preview/{m.name}/symbol-{i + 1}.svg?v={ver}") \
                            .props("fit=contain").classes("w-full rounded").style(img_style)
                else:
                    ui.label("Preview unavailable").classes("text-xs") \
                        .style("color:#64748b;padding:16px")
            with ui.card().classes("flex-grow").style(
                    "background:var(--kc-surface);border:1px solid var(--kc-border);min-width:300px"):
                ui.label("Footprint").classes("text-xs font-medium") \
                    .style("color:#94a3b8")
                fp = footprint_svg(part) if kicad_cli_available() else None
                if fp:
                    ui.image(f"/part-preview/{m.name}/footprint.svg?v={ver}") \
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

        # Live LCSC pricing: the C-number plus a unit price from the easyeda.com
        # endpoint. The qty-break ladder source (JLCPCB) is WAF-blocked, so the
        # 10/100-pc columns show "n/a" rather than a guessed number.
        code = (m.sourcing or {}).get("lcsc", "").strip().upper()
        if _LCSC_CODE_RE.match(code):
            with ui.card().classes("w-full") \
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                with ui.row().classes("items-center gap-2"):
                    ui.label("LCSC pricing").classes("text-sm font-medium") \
                        .style("color:#94a3b8")
                    ui.label(code).classes("text-xs font-mono rounded") \
                        .style("background:var(--kc-border);color:#cbd5e1;padding:2px 8px")
                price_row = ui.row().classes("items-center gap-6")
                with price_row:
                    ui.label("Loading live price…").classes("text-sm") \
                        .style("color:#64748b")

                def _fill_price(row=price_row, cid=code) -> bool:
                    res = _price_for_lcsc(cid)
                    if res is None:
                        return False  # still fetching -> keep polling
                    row.clear()
                    with row:
                        if isinstance(res, dict):
                            for qty, val in (("1", res["unit_price"]),
                                             ("10", res.get("price_10")),
                                             ("100", res.get("price_100"))):
                                with ui.column().classes("gap-0 items-start"):
                                    ui.label(f"@{qty} pc").classes("text-xs") \
                                        .style("color:#64748b")
                                    ui.label(_fmt_price(val) if val is not None
                                             else "n/a") \
                                        .classes("text-sm font-mono text-white")
                            with ui.column().classes("gap-0 items-start"):
                                ui.label("JLC stock").classes("text-xs") \
                                    .style("color:#64748b")
                                ui.label(f"{res.get('stock') or 0:,}") \
                                    .classes("text-sm font-mono text-white")
                            with ui.column().classes("gap-0 items-start"):
                                ui.label("LCSC retail").classes("text-xs") \
                                    .style("color:#64748b")
                                retail = res.get("retail_stock")
                                if retail is None:
                                    ui.label("unverified").classes("text-xs") \
                                        .style("color:#64748b;padding-top:2px")
                                else:
                                    ui.label(f"{retail:,}") \
                                        .classes("text-sm font-mono") \
                                        .style("color:#f87171" if retail == 0
                                               else "color:#ffffff")
                        else:  # _UNAVAILABLE
                            ui.label("Live pricing unavailable (vendor API "
                                     "blocked).").classes("text-sm") \
                                .style("color:#f59e0b")
                    return True

                if not _fill_price():
                    timer = ui.timer(1.0,
                                     lambda: _fill_price() and timer.deactivate())
                ui.label("LCSC pricing; 10/100-pc breaks come from the offline "
                         "JLC catalog when it covers the part. JLC stock = "
                         "JLCPCB assembly inventory; LCSC retail = live "
                         "lcsc.com storefront.").classes("text-xs") \
                    .style("color:#64748b")

        with ui.card().classes("w-full") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
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


def _sample_media(s, *, hero: bool = False) -> str:
    """A static, high-quality isometric render of the board. The landing page is
    all stills now — the hero and every gallery card — so it paints instantly and
    shows the board the way the renderer drew it. The interactive 3D model lives
    on the logged-in explorer instead (see ``_sample_model_viewer``)."""
    alt = f"{s.title} board, designed by KiCraft"
    if hero:
        # The featured board's dedicated hero render (hero.png, else board.png).
        return (f'<img class="kc-board" src="{s.board_hero_url}" alt="{alt}" '
                f'fetchpriority="high">')
    return f'<img src="{s.board_png_url}" alt="{alt}" loading="lazy">'


def _sample_model_viewer(s) -> str:
    """Interactive 3D ``<model-viewer>`` of the board's GLB, for the explorer.
    Orbit/zoom enabled and gently auto-rotating, grounded with a soft shadow; the
    static render is the poster so it paints instantly before the GLB streams in.
    Starts at a flat, mostly-top-down angle to match the landing stills."""
    alt = f"{s.title} board, designed by KiCraft"
    return (
        f'<model-viewer src="{s.board_glb_url}" poster="{s.board_png_url}" '
        f'alt="{alt}" camera-controls auto-rotate auto-rotate-delay="1500" '
        f'rotation-per-second="14deg" camera-orbit="16deg 58deg 105%" '
        f'interaction-prompt="none" shadow-intensity="1" shadow-softness="0.85" '
        f'exposure="1.05" touch-action="pan-y" reveal="auto" loading="lazy" '
        f'style="width:100%;height:520px;background:transparent;'
        f'--poster-color:transparent;"></model-viewer>')


def _landing_sample_card(s) -> str:
    badge = '<span class="kc-badge">Featured</span>' if s.featured else ""
    href = f"/signup?prompt={quote(s.prompt)}"
    return (
        f'<a class="kc-sample kc-reveal" href="{href}">'
        f'<div class="kc-sample-art">{badge}{_sample_media(s)}</div>'
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
    ui.query("body").style("background:var(--kc-bg)")
    ui.add_head_html('<link rel="stylesheet" href="/static/kc_landing.css">')
    ui.add_head_html('<style>html{scroll-behavior:smooth}</style>')
    ui.add_head_html(
        f"<script>window.KICRAFT_PROMPTS={json.dumps(EXAMPLE_PROMPTS)};</script>")
    ui.add_head_html('<script src="/static/kc_landing.js" defer></script>')

    samples = available_samples()
    hero = featured_sample()
    # The landing page is all static renders now — no <model-viewer> and no
    # model-viewer bundle loaded here. The interactive 3D board lives on the
    # logged-in explorer (samples_page), which pulls the bundle in itself.

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
    hero_art = (f'<div class="kc-hero-art">{_sample_media(hero, hero=True)}</div>'
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
      <a class="kc-nav-signin" href="/pricing">Pricing</a>
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
      <a href="/pricing">Pricing</a>
      <a href="/terms" target="_blank" rel="noopener">Terms of Service</a>
      <a href="/privacy" target="_blank" rel="noopener">Privacy Policy</a>
      <a href="/login">Sign in</a>
    </div>
  </div></footer>
</div>
"""
    ui.html(html, sanitize=False)


@ui.page("/")
def index(prompt: str = "", project: str = ""):
    user = _current_user()
    if user is None:
        _render_landing()
        return
    if user.accepted_terms_version != LEGAL_VERSION:
        return RedirectResponse("/consent")
    q0 = _store().quota_status(user)

    kicanvas_head()
    _mobile_head()
    ui.add_head_html('<link rel="stylesheet" href="/static/kc_onboarding.css">')
    ui.add_head_html(
        f"<script>window.KICRAFT_PROMPTS={json.dumps(EXAMPLE_PROMPTS)};"
        f"window.KICRAFT_PLACEHOLDER_FALLBACK="
        f"{json.dumps('Describe your board, big or small. Be bold.')};</script>")
    ui.add_head_html('<script src="/static/kc_onboarding.js" defer></script>')
    ui.dark_mode().enable()
    ui.query("body").style("background:var(--kc-bg)")
    first_run = not _store().list_projects(user.id)
    welcome_card = None
    arrow_hint = None
    # The open design's run state. Shared with the worker thread while a run is
    # live (and with other pages attached to the same run via _LIVE_RUNS), so
    # opening a different design RE-BINDS this name to another dict (nonlocal in
    # open_project/start/start_fresh) -- it is never recycled in place.
    state: dict = _fresh_run_state()

    # Page-local render bookkeeping: this page's event cursor, widget handles
    # and mtime caches. Kept OUT of `state` so two pages can watch the same run
    # without fighting over the cursor or refreshing each other's widgets. The
    # dict object is stable (closures capture it); only its contents reset.
    view: dict = {}

    def _reset_view():
        live_sig = view.get("live_sig")  # page-level, survives project switches
        view.clear()
        view.update(rendered=0, build_lines=[], fab_done=False,
                    sch_view=None, pcb_view=None,
                    sch_revealed=False, pcb_revealed=False,
                    pcb_mtime=None, state_mtime=None, run_mtime=None,
                    leaf_progress_sig=None, questions_rendered=None,
                    prices_rev_seen=0, prices_loaded_ws=None,
                    account_refreshed=False, viewed_marked=False,
                    support_prompted=False, live_sig=live_sig,
                    # Manual layout editor: while True the place/route
                    # view slot belongs to the editor and the timer must
                    # not repaint the gallery/board over it.
                    layout_editor=False, rescue_offered=False)

    _reset_view()

    def logout():
        for k in ("user_id", "email"):
            app.storage.user.pop(k, None)
        ui.navigate.to("/login")

    with ui.header().classes("items-center justify-between") \
            .style("background:var(--kc-surface);border-bottom:1px solid var(--kc-border)"):
        with ui.row().classes("items-center gap-2"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("design a PCB from a sentence").classes("text-sm kc-tagline") \
                .style("color:#94a3b8")
        # Full nav row on desktop (>=1024px) ...
        with ui.row().classes("items-center gap-3 gt-sm"):
            # Start a fresh design. Lives here (not in the composer) because the
            # composer's prompt chrome collapses once a design is open. start_fresh
            # is defined further down; the lambda resolves it at click time.
            ui.button("New design", icon="add",
                      on_click=lambda: start_fresh()) \
                .props("flat dense no-caps color=primary").classes("text-xs") \
                .tooltip("Start a fresh design. The open one keeps running in the "
                         "background and stays under My projects.")
            ui.button("My projects", icon="folder",
                      on_click=lambda: ui.navigate.to("/projects")) \
                .props("flat dense no-caps color=white").classes("text-xs") \
                .tooltip("Your designs -- open them and publish to the community")
            ui.button("Examples", icon="dashboard",
                      on_click=lambda: ui.navigate.to("/samples")) \
                .props("flat dense no-caps color=white").classes("text-xs") \
                .tooltip("Explore boards KiCraft designed")
            ui.button("Part library", icon="memory",
                      on_click=lambda: ui.navigate.to("/parts")) \
                .props("flat dense no-caps color=white").classes("text-xs") \
                .tooltip("Browse the standard library and parts you've added")
            ui.button("Browse", icon="travel_explore",
                      on_click=lambda: ui.navigate.to("/browse")) \
                .props("flat dense no-caps color=white").classes("text-xs") \
                .tooltip("Browse and clone community projects")
            ui.button("Pricing", icon="workspace_premium",
                      on_click=lambda: ui.navigate.to("/pricing")) \
                .props("flat dense no-caps color=white").classes("text-xs") \
                .tooltip("Plans and upgrades")
            ui.button("Support", icon="support_agent",
                      on_click=lambda: open_support_dialog(auto=False)) \
                .props("flat dense no-caps color=white").classes("text-xs") \
                .tooltip("Report a problem (the open board's ID and error "
                         "details are attached automatically)")
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
        # ... collapsed to a badge + hamburger menu on phones/tablets (<1024px).
        with ui.row().classes("items-center gap-2 lt-md"):
            m_tier_badge = ui.badge(q0["label"], color="primary")
            with ui.button(icon="menu").props("flat dense color=white"):
                with ui.menu().props("auto-close") \
                        .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                    ui.menu_item("New design", lambda: start_fresh())
                    ui.menu_item("My projects", lambda: ui.navigate.to("/projects"))
                    ui.menu_item("Examples", lambda: ui.navigate.to("/samples"))
                    ui.menu_item("Part library", lambda: ui.navigate.to("/parts"))
                    ui.menu_item("Browse", lambda: ui.navigate.to("/browse"))
                    ui.menu_item("Pricing", lambda: ui.navigate.to("/pricing"))
                    ui.menu_item("Support", lambda: open_support_dialog(auto=False))
                    if is_admin(user):
                        ui.menu_item("Admin", lambda: ui.navigate.to("/admin"))
                    ui.separator()
                    ui.menu_item(user.email, lambda: ui.navigate.to("/profile"))
                    ui.menu_item("Log out", logout)

    with ui.column().classes("w-full mx-auto p-4 gap-3").style("max-width:1600px"):
        # Site-wide LLM budget is admin-only telemetry; users never see a cost figure.
        if is_admin(user):
            try:
                budget = SpendGuard(Settings.from_env()).status()
                ui.label(f"Daily budget remaining: ${budget['daily_remaining_usd']:.2f} "
                         f"of ${budget['daily_ceiling_usd']:.0f}").classes("text-xs").style("color:#64748b")
            except Exception:
                ui.label("").classes("hidden")

        with ui.row().classes("items-center gap-2"):
            quota_label = ui.label().classes("text-xs").style("color:#94a3b8")
            upgrade_link = ui.button("Upgrade", icon="workspace_premium",
                                     on_click=lambda: ui.navigate.to("/pricing")) \
                .props("flat dense no-caps color=primary").classes("text-xs")
            upgrade_link.set_visibility(False)
        # Unverified-email banner: shown when a non-staff user hasn't confirmed
        # their signup email. The Design button (managed in refresh_account_ui)
        # stays disabled until they verify; the Resend button mints a fresh link.
        with ui.row().classes("items-center gap-2 kc-unverified") as unverified_row:
            unverified_label = ui.label(
                "Verify your email to start designing. Check your inbox for the "
                "confirmation link.").classes("text-xs").style("color:#f59e0b")

            def _resend_verify():
                u = _current_user()
                if u is None:
                    return
                token = _store().create_verification_token(u.id)
                if token is None:
                    ui.notify("A verification link was sent recently — check your "
                              "inbox in a minute.", color="warning")
                    return
                try:
                    s = Settings.from_env()
                    send_verification_email(
                        s, u.email, f"{s.public_url}/verify?token={token}",
                        _VERIFY_TTL_HOURS)
                except Exception:
                    pass
                ui.notify("Verification link sent. Check your email.",
                          color="positive")

            ui.button("Resend verification link", on_click=_resend_verify) \
                .props("flat dense no-caps color=primary").classes("text-xs")
        unverified_row.set_visibility(False)

        if first_run:
            with ui.row().classes("w-full items-start justify-between kc-welcome") \
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border);"
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

        # Once a design is open (started, attached, or reopened) the compose chrome
        # below collapses and only this read-only prompt header stays at the top.
        # _enter_run_view / _enter_compose_view (defined below) toggle the two.
        prompt_display = ui.column().classes("w-full gap-1 kc-prompt-shown") \
            .style("background:var(--kc-surface);border:1px solid var(--kc-border);"
                   "border-radius:8px;padding:12px 14px")
        with prompt_display:
            ui.label("Your prompt").classes("text-xs uppercase tracking-wide") \
                .style("color:#64748b")
            prompt_label = ui.label("").classes("text-sm") \
                .style("color:#e2e8f0;white-space:pre-wrap")
        prompt_display.set_visibility(False)

        brief = ui.textarea(
            "Describe your board",
            placeholder="Describe your board, big or small. Be bold.") \
            .props("rows=4 stack-label").classes("w-full kc-brief")

        # One-click inspiration: "Surprise me" streams the vetted self-eval corpus
        # (the cycling placeholder in kc_onboarding.js supplies passive ideas).
        # Created here for position; its click handler is wired below once `start`
        # exists so it can both load the next brief AND launch the run.
        chips_row = ui.row().classes("items-center gap-2 kc-chips")

        with ui.row().classes("items-center gap-2"):
            design_btn = ui.button("Design").props("color=primary unelevated") \
                .classes("kc-design")
            continue_btn = ui.button("Continue design", icon="play_arrow") \
                .props("color=primary outline")
            continue_btn.set_visibility(False)
            # Escape hatch from the auto-opened design: detach to a blank
            # composer so a second design can run in parallel (the open one
            # keeps running in the background, reachable from My projects).
            new_btn = ui.button("New design", icon="add") \
                .props("outline color=white") \
                .tooltip("Start a fresh design. The open one keeps running in "
                         "the background and stays under My projects.")
            new_btn.set_visibility(False)
            if first_run:
                design_btn.classes(add="kc-pulse")
                with ui.row().classes("items-center kc-arrow") as arrow_hint:
                    ui.icon("arrow_back").classes("kc-arrow-icon")
                    ui.label("click to start")

        # Walk-away notifications: persisted per user; the run worker reads the
        # stored preference at send time, so flipping it mid-run takes effect.
        notify_chk = ui.checkbox("Email me when a run finishes or needs my input",
                    value=bool(user.notify_email),
                    on_change=lambda e: _store().set_notify_email(
                        user.id, bool(e.value))) \
            .props("dense size=xs").classes("text-xs").style("color:#94a3b8")

        def use_prompt(text: str):
            brief.value = text
            brief.run_method("focus")
            design_btn.classes(add="kc-pulse")  # draw the eye to the next click

        with chips_row:
            # Streams the vetted self-eval corpus in order and runs each one
            # (handler wired below, once `start` is defined).
            surprise_btn = ui.button("Surprise me", icon="casino") \
                .props("flat rounded dense no-caps").classes("kc-chip") \
                .tooltip("Run the next vetted self-eval brief — a known-good "
                         "design. Click again for the next one.")

        def _enter_run_view(prompt_text: str) -> None:
            """A design is open (started / attached / reopened): collapse the
            compose chrome to the read-only prompt header so only the user's
            prompt stays at the top. Starting a fresh one happens from the
            header's "New design" button now."""
            prompt_label.text = (prompt_text or "").strip()
            prompt_display.set_visibility(True)
            for el in (welcome_card, brief, chips_row, notify_chk,
                       design_btn, new_btn, arrow_hint):
                if el is not None:
                    el.set_visibility(False)

        def _enter_compose_view() -> None:
            """Back to a blank composer: restore the prompt box and its chrome."""
            prompt_display.set_visibility(False)
            prompt_label.text = ""
            for el in (brief, chips_row, notify_chk, design_btn):
                if el is not None:
                    el.set_visibility(True)

        # Prefill from a sample the visitor chose before signing up (carried via the
        # ?prompt= query or stashed across the signup hop). No run starts: the user
        # still clicks Design themselves, so no model is called without a signup.
        prefill = (prompt or app.storage.user.pop("pending_prompt", "") or "").strip()
        if prefill:
            use_prompt(prefill)

        with ui.row().classes("items-center gap-3"):
            status = ui.label("").classes("text-sm").style("color:#e2e8f0")
            # The open design's unique, human-quotable id. Always on screen while
            # a board is open so a user can quote it in any support report.
            board_label = ui.label("").classes("text-xs font-mono cursor-pointer") \
                .style("color:#94a3b8;border:1px solid var(--kc-border-strong);"
                       "border-radius:4px;padding:1px 8px") \
                .tooltip("This board's unique ID. Click to copy; quote it when "
                         "reporting an issue.")
            board_label.set_visibility(False)

            def _copy_board_code():
                code = state.get("board_code")
                if code:
                    ui.run_javascript(
                        f"navigator.clipboard.writeText({json.dumps(code)})")
                    ui.notify(f"Copied {code}.", color="positive")
            board_label.on("click", _copy_board_code)
        # Per-design LLM spend: admin-only. Users never see what a design costs;
        # the spend is still tracked server-side (state["spend"] -> ledger / admin).
        spend = ui.label("").classes("text-sm").style("color:#64748b")
        spend.set_visibility(is_admin(user))
        question_box = ui.column().classes("w-full")

        support_dialog = ui.dialog()

        def open_support_dialog(auto: bool = False):
            """(Re)build and open the support dialog over the open design.

            auto=True is the post-error flavor: the failure was ALREADY filed
            for automated review by the run worker (_file_failure_report), so
            submitting only attaches the user's optional feedback to that row.
            The manual flavor (the header's Support button) files a fresh
            report with the same diagnostics snapshot."""
            diag = _collect_support_diagnostics(state)
            code = state.get("board_code")
            support_dialog.clear()
            with support_dialog, ui.card().classes("w-[680px] max-w-[95vw] gap-2") \
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border-strong)"):
                ui.label("Something went wrong" if auto else "Contact support") \
                    .classes("text-lg font-bold text-white")
                if auto:
                    ui.label("This run failed. The technical details below were "
                             "logged for review. Anything you can add about what "
                             "you were trying to build helps us fix it faster.") \
                        .classes("text-sm").style("color:#94a3b8")
                else:
                    ui.label("Report a problem with the open design or the app. "
                             "The technical details below are attached "
                             "automatically.") \
                        .classes("text-sm").style("color:#94a3b8")
                with ui.row().classes("items-center gap-2"):
                    ui.label("Board ID").classes("text-xs").style("color:#64748b")
                    ui.label(code or "(no design open)") \
                        .classes("text-sm font-mono font-bold") \
                        .style("color:#e2e8f0" if code else "color:#64748b")
                with ui.expansion("Details that will be sent").classes("w-full") \
                        .style("background:var(--kc-bg);border:1px solid var(--kc-border)"):
                    ui.label(json.dumps(diag, indent=2, ensure_ascii=False)) \
                        .classes("text-xs font-mono whitespace-pre-wrap") \
                        .style("color:#94a3b8;max-height:240px;overflow:auto")
                feedback = ui.textarea(
                    "Anything you'd like to add? (optional)",
                    placeholder="What were you trying to build? What did you "
                                "expect to happen?") \
                    .props("rows=3").classes("w-full")

                def submit():
                    msg = (feedback.value or "").strip()
                    rid = state.get("support_report_id") if auto else None
                    try:
                        if rid is None:
                            rid = _store().create_support_report(
                                user_id=user.id,
                                project_id=state.get("project_id"),
                                board_code=code,
                                kind=("error_auto" if auto else "user"),
                                message=(msg or None), diagnostics=diag)
                        elif msg:
                            _store().set_support_report_message(rid, msg)
                    except Exception:
                        ui.notify("Could not record the report. Please try "
                                  "again.", color="negative")
                        return
                    # A user just engaged (manual report, or feedback on a
                    # failure) -- auto-investigate if the admin toggle is on. The
                    # bare per-failure auto-file (auto=True, no message) is left
                    # alone so we don't investigate every failed build.
                    if rid is not None and (not auto or msg):
                        _auto_investigate_if_enabled(rid)
                    support_dialog.close()
                    ref = code or f"report #{rid}"
                    ui.notify(f"Thanks, that's logged. Your reference is {ref}.",
                              color="positive")

                with ui.row().classes("justify-end gap-2 w-full"):
                    ui.button("Close", on_click=support_dialog.close) \
                        .props("flat color=white")
                    ui.button("Send report", icon="send", on_click=submit) \
                        .props("color=primary")
            support_dialog.open()

        # Per-stage tabs: each phase gets its own tab with a project-state inspector
        # (left) over the LLM thinking + activity/log windows (right). The native
        # KiCad schematic/board (KiCanvas) and the download land in the build tabs.
        tabs = StageTabs(show_cost=is_admin(user))

        # A KiCanvas view built while its tab is hidden sizes its WebGL canvas to zero
        # and never repaints; re-fit it the first time the user reveals that tab. The
        # flag is reset when each view is (re)created (see the render loop below).
        def _reveal_view(view_key: str, seen_flag: str) -> None:
            v = view.get(view_key)
            if v is not None and not view.get(seen_flag):
                view[seen_flag] = True
                v.refresh()
        tabs.on_show("synthesize", lambda: _reveal_view("sch_view", "sch_revealed"))
        tabs.on_show("place_route", lambda: _reveal_view("pcb_view", "pcb_revealed"))

        with ui.expansion("Edit a stage & re-run").classes("w-full mt-2") \
                .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
            edit_box = ui.column().classes("w-full gap-2 p-2")
        edit_ctx: dict = {"getter": None, "raw": None, "instr": None}

        def build_question_panel():
            """(Re)build the clarifying-question panel for a parked run as a
            conversational card: each question is an agent 'bubble', suggested
            options are tappable quick-pick chips that fill the reply box, and a
            clearly-outlined answer field carries a green focus ring. Freeform
            text is always accepted; chips are just shortcuts."""
            question_box.clear()
            view["questions_rendered"] = state.get("questions")
            qs = state.get("questions") or []
            if not (state.get("awaiting_input") and qs):
                return
            stage = qs[0].get("stage", "")
            with question_box:
                with ui.column().classes("w-full kc-qcard gap-3") \
                        .style("padding:16px 18px"):
                    with ui.row().classes("items-center gap-2 w-full"):
                        ui.icon("smart_toy").classes("text-lg") \
                            .style("color:var(--kc-brand)")
                        ui.label("KiCraft needs a detail") \
                            .classes("text-sm font-semibold") \
                            .style("color:var(--kc-text)")
                        ui.space()
                        if stage:
                            ui.label(stage).classes("text-xs kc-stage-pill")
                    widgets = []
                    for q in qs:
                        with ui.row().classes("w-full"):
                            with ui.element("div").classes("kc-bubble") \
                                    .style("padding:10px 14px;max-width:640px"):
                                ui.label(q.get("text", "")).classes("text-sm") \
                                    .style("color:var(--kc-text);white-space:pre-wrap")
                        ans = ui.input(placeholder="Type your answer…") \
                            .props("outlined dense") \
                            .classes("w-full").style("max-width:640px")
                        opts = q.get("options") or []
                        if opts:
                            with ui.row().classes("items-center gap-2 flex-wrap"):
                                ui.label("Quick pick").classes("text-xs") \
                                    .style("color:var(--kc-dim)")
                                for opt in opts:
                                    ui.button(
                                        opt,
                                        on_click=lambda o=opt, a=ans: (
                                            a.set_value(o), a.run_method("focus"))) \
                                        .props("flat dense no-caps") \
                                        .classes("kc-qchip text-xs")
                        widgets.append((q, ans))

                    def submit_answers():
                        answers = [{"text": q.get("text", ""),
                                    "answer": (a.value or "").strip()}
                                   for q, a in widgets]
                        if not any(x["answer"] for x in answers):
                            ui.notify("Type or pick at least one answer.",
                                      color="warning")
                            return
                        _answer_and_resume(stage, answers)

                    # Enter submits from any answer field (single-question is the
                    # common case; multi still validates every field).
                    for _q, _a in widgets:
                        _a.on("keydown.enter", lambda: submit_answers())

                    with ui.row().classes("w-full justify-end mt-1"):
                        ui.button("Submit & continue", icon="send",
                                  on_click=submit_answers) \
                            .props("unelevated no-caps color=primary")

        def _answer_and_resume(stage, answers):
            if _project_run_live(state):
                return
            _ensure_workspace(state)  # rehydrate the durable project before the write
            ws = state["ws"]
            if not ws:
                ui.notify("No open design.", color="warning")
                return
            record_answers(ws, stage, answers)
            runs = [stage] + downstream_stages(stage)
            if state["project_id"]:  # same project, no new quota slot
                _store().update_project_status(state["project_id"], "running")
            state.update(running=True, done=False, ok=None,
                         awaiting_input=False, questions=[])
            view.update(fab_done=False, account_refreshed=False,
                        viewed_marked=False, questions_rendered=None,
                        support_prompted=False)
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
                read_root = Path(state["ws"]) if state.get("ws") else None  # durable root in view mode (ws=None)
                if not read_root:
                    ui.label("Open or run a design first, then edit a stage here.") \
                        .classes("text-xs").style("color:#64748b")
                    return
                sj = read_state(read_root)
                editable = [s for s in ("intent", "functional_spec", "architecture", "bom")
                            if sj.get(s)]
                if not editable:
                    ui.label("No committed stages to edit yet.") \
                        .classes("text-xs").style("color:#64748b")
                    return
                ui.label("Editing a stage re-runs the stages after it.") \
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
                    .style("background:var(--kc-surface);border:1px solid var(--kc-border)"):
                ui.label("Re-run stages?").classes("text-lg font-bold text-white")
                ui.label(verb + tail + ".") \
                    .classes("text-sm").style("color:#94a3b8")
                with ui.row().classes("gap-2 justify-end w-full"):
                    ui.button("Cancel", on_click=dlg.close).props("flat color=white")
                    ui.button("Confirm & run", color="primary",
                              on_click=lambda: (dlg.close(),
                                                _do_rerun(stage, slot_dict, instruction, runs)))
            dlg.open()

        def _do_rerun(stage, slot_dict, instruction, runs):
            if _project_run_live(state):
                ui.notify("A run is already in progress.", color="warning")
                return
            _ensure_workspace(state)  # rehydrate the durable project before the edit/rerun
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
            # The re-run stages' panels are invalid now: back to placeholders,
            # and repaint every tab from the durable state (edited stage stays
            # green, cleared downstream stages drop to pending, build phases
            # drop to pending because the design is no longer complete).
            for s in runs:
                tabs.reset_stage(s)
            sj = read_state(ws)
            tabs.set_statuses(_derived_statuses(Path(ws), sj, None, False),
                              sj.get("stage_status"))
            if state["project_id"]:
                _store().update_project_status(state["project_id"], "running")
            state.update(running=True, done=False, ok=None, pcb_ready=False,
                         awaiting_input=False, questions=[])
            view.update(fab_done=False, account_refreshed=False, viewed_marked=False,
                        sch_view=None, pcb_view=None, pcb_mtime=None,
                        state_mtime=None, run_mtime=None, support_prompted=False)
            continue_btn.set_visibility(False)
            design_btn.disable()
            status.text = "Re-running: " + " -> ".join(runs) + " ..."
            threading.Thread(target=_run_design, args=(state, runs),
                             kwargs={"instruction": instruction}, daemon=True).start()

        def _continue():
            """Run the stages still missing from the current (reopened) design."""
            if _project_run_live(state):
                return
            _ensure_workspace(state)  # rehydrate the durable project before continuing
            sj = read_state(state["ws"]) if state["ws"] else {}
            rem = remaining_stages(sj)
            if not rem:
                ui.notify("Nothing left to run for this design.", color="info")
                return
            if state["project_id"]:  # same project, no new quota slot
                _store().update_project_status(state["project_id"], "running")
            state.update(running=True, done=False, ok=None)
            view.update(fab_done=False, account_refreshed=False, viewed_marked=False,
                        support_prompted=False)
            continue_btn.set_visibility(False)
            design_btn.disable()
            status.text = "Continuing: " + " -> ".join(rem) + " ..."
            threading.Thread(target=_run_design, args=(state, rem), daemon=True).start()

        continue_btn.on_click(_continue)

        def open_project(p, *, notify=True):
            """Open a project. If its run is live in this process, attach to the
            running state dict so progress streams into this page; otherwise
            rehydrate the saved workspace and render its committed slots. Either
            way the design this page was showing keeps any background worker it
            had (its dict stays in _LIVE_RUNS, reopenable from the list), and
            the same project_id is reused (no new quota slot)."""
            nonlocal state
            # A design is open: collapse the compose chrome to the prompt header.
            _enter_run_view(p.brief or "")
            live = _LIVE_RUNS.get(p.id)
            if live is not None:
                state = live
                _reset_view()
                tabs.reset()
                # Restore the durable stage statuses first: a live run resumed
                # after a reopen only streams events for the stages it re-runs,
                # so the replay below cannot repaint the earlier, already-
                # committed stages.
                live_ws = Path(state["ws"]) if state.get("ws") else None
                live_sj = read_state(state["ws"]) if state.get("ws") else {}
                tabs.set_statuses(
                    _derived_statuses(live_ws, live_sj, p.status,
                                      bool(state.get("zip"))),
                    live_sj.get("stage_status"))
                continue_btn.set_visibility(False)
                new_btn.set_visibility(False)  # "New design" lives in the header now
                if state.get("running"):
                    design_btn.disable()
                    what = p.project_stem or (p.brief or "design")[:60]
                    status.text = (f'Designing "{what}" -- live progress is in '
                                   "the tabs below.")
                elif state.get("awaiting_input"):
                    view["account_refreshed"] = True
                    status.text = "Reopened. This design is waiting for your answer below."
                # Notify BEFORE refresh_account_ui: rebuilding the projects list
                # deletes the clicked Open button's slot, and ui.notify resolves
                # the client through that slot -- notifying after the rebuild
                # raises "parent element ... has been deleted" (seen live).
                if notify:
                    ui.notify(f"Attached to {p.project_stem or 'your running design'}.",
                              color="positive")
                refresh_account_ui()
                return
            # Build-in-place: the durable project dir IS the workspace -- reads AND any
            # later writes (continue/edit/rebuild) happen here, no scratch, no copytree.
            # (A listed project always has dir_path.)
            read_root = Path(p.dir_path) if p.dir_path else None
            ws_str = str(p.dir_path) if p.dir_path else None
            project_dir = (_persisted_generated_dir(p.dir_path, p.project_stem)
                           if p.dir_path else None)
            sj = read_state(read_root) if read_root else {}
            zip_ok = bool(p.zip_path and Path(p.zip_path).is_file())
            completed = p.status == "ok"
            state = _fresh_run_state()
            state.update(done=completed, ok=(True if completed else None),
                         failed=(p.status == "failed"),
                         spend=p.cost_usd, zip=(p.zip_path if zip_ok else None),
                         ws=ws_str, stem=p.project_stem,
                         user_id=user.id, project_id=p.id, brief=p.brief or "",
                         board_code=p.board_code,
                         status=("awaiting_input" if p.status == "awaiting_input" else None),
                         awaiting_input=(p.status == "awaiting_input"),
                         questions=[q for q in (sj.get("open_questions") or [])
                                    if not q.get("answer")])
            # Reopen the build timeline + LLM reasoning: events.jsonl is persisted at
            # finalize but was never read back, so the timeline rendered blank. The
            # render loop replays these into the tabs (display-only: tabs.push paints).
            state["events"] = _load_events(p.dir_path)
            _reset_view()
            view["account_refreshed"] = True
            tabs.reset()
            new_btn.set_visibility(False)  # "New design" lives in the header now
            # project_dir resolved above (durable or rehydrated workspace); restored
            # artifacts -> schematic / PCB render, even if the run FAILED.
            if project_dir is not None:
                state["stem"] = project_dir.name
                state["project_dir"] = str(project_dir)
                state["token"] = _register_project_dir(project_dir)
                state["pcb_ready"] = (project_dir / f"{project_dir.name}.kicad_pcb").is_file()
            # Stage icons reflect the persisted progress, not just live events:
            # without this every reopened project showed all-pending tabs.
            tabs.set_statuses(_derived_statuses(read_root, sj, p.status, zip_ok),
                              sj.get("stage_status"))
            if state["failed"] and state["pcb_ready"]:
                _mark_fab_invalid()
            rem = remaining_stages(sj)
            continue_btn.set_visibility(bool(rem) and not state["awaiting_input"])
            if state["awaiting_input"]:
                status.text = "Reopened. This design is waiting for your answer below."
            elif rem:
                status.text = ("Reopened. Remaining: " + " -> ".join(rem)
                               + ". Click Continue design when ready.")
            elif state["failed"]:
                # All LLM stages done but the build (place/route) failed or was
                # killed -- there is no routed/fab board. Do NOT call this
                # "Design complete": that mislabels a timed-out build and offers
                # a download of an unrouted board (KC-NZXXEE).
                status.text = ("Reopened. The PCB place/route did not finish "
                               "(it failed or timed out), so there is no routed "
                               "board to download. The schematic stages are "
                               "complete -- edit a stage and rebuild to retry.")
            elif not zip_ok:
                status.text = ("Reopened. Routing finished but no fab package "
                               "was produced (not fab-ready). Edit a stage to "
                               "revise, or rebuild.")
            else:
                status.text = ("Reopened. Design complete: download below, "
                               "or edit a stage to revise.")
            if p.status in ("ok", "failed"):
                view["viewed_marked"] = True
                _store().mark_viewed(p.id)
            # Notify BEFORE refresh_account_ui (see the attach branch above).
            if notify:
                ui.notify(f"Opened {p.project_stem or 'project'}.", color="positive")
            refresh_account_ui()

        def refresh_account_ui():
            u = _current_user()
            if u is None:
                return
            q = _store().quota_status(u)
            period = "week" if q["window_days"] <= 7 else "month"
            # Unverified non-staff users get the verify banner and a disabled
            # Design button, shown instead of a misleading "0 of 1 left".
            needs_verify = (not is_admin(u)) and not u.email_verified
            unverified_row.set_visibility(needs_verify)
            if needs_verify:
                quota_label.text = "Email not verified — verify to start designing."
                quota_label.style("color:#f59e0b")
                design_btn.disable()
                # A free unverified user has no quota story to upgrade past yet.
                upgrade_link.set_visibility(False)
                build_edit_panel()
                return
            if q.get("unlimited"):
                quota_label.text = f"{q['label']} tier: unlimited designs (staff)."
            else:
                quota_label.text = (f"{q['label']} tier: {q['remaining']} of {q['limit']} "
                                    f"designs left this {period}.")
            tier_badge.text = q["label"]
            m_tier_badge.text = q["label"]
            # Quietly offer the paid tiers to free users; insist once the
            # quota is spent (the Design button below goes dark with it).
            upgrade_link.set_visibility(
                not q.get("unlimited")
                and (u.tier == "free" or q["remaining"] <= 0))
            if q["remaining"] <= 0:
                design_btn.disable()
                quota_label.style("color:#f59e0b")
            else:
                if not state["running"]:
                    design_btn.enable()
                quota_label.style("color:#94a3b8")
            build_edit_panel()

        def start_fresh():
            """Detach from the open design and reset to a blank composer. A
            still-running design keeps its worker and its _LIVE_RUNS entry, so
            this is how a second design gets started in parallel."""
            nonlocal state
            state = _fresh_run_state()
            _reset_view()
            tabs.reset()
            brief.value = ""
            status.text = ""
            spend.text = ""
            board_label.set_visibility(False)
            continue_btn.set_visibility(False)
            new_btn.set_visibility(False)
            # Restore the prompt box + chrome that collapsed when a design opened.
            _enter_compose_view()
            refresh_account_ui()  # re-enables Design when quota allows

        new_btn.on_click(start_fresh)

        def start():
            nonlocal state
            if state["running"]:
                return
            u = _current_user()
            if u is None:
                ui.navigate.to("/login")
                return
            if not (brief.value or "").strip():
                ui.notify("Enter a brief first.", color="warning")
                return
            if not is_admin(u) and not u.email_verified:
                ui.notify("Verify your email first — check your inbox for the "
                          "confirmation link (or click Resend above).",
                          color="warning")
                return
            q = _store().quota_status(u)
            if q["remaining"] <= 0:
                period = "week" if q["window_days"] <= 7 else "month"
                ui.notify(f"You've used your {q['limit']} design(s) this {period}. "
                          "See Pricing to upgrade.", color="warning")
                return
            pid = _store().create_project(u.id, brief.value)
            proj = _store().get_project(pid)
            state = _fresh_run_state()
            state.update(running=True, user_id=u.id, project_id=pid,
                         brief=brief.value,
                         board_code=(proj.board_code if proj else None))
            _reset_view()
            continue_btn.set_visibility(False)
            new_btn.set_visibility(False)  # "New design" lives in the header now
            tabs.reset()
            status.text = ("Designing... (intent -> functional_spec -> architecture -> bom -> "
                           "wiring -> synthesize -> place/route -> fab)")
            design_btn.disable()
            design_btn.classes(remove="kc-pulse")
            # Collapse the compose chrome -- only the user's prompt stays at the top.
            _enter_run_view(brief.value)
            threading.Thread(target=_design_worker, args=(brief.value, state), daemon=True).start()

        design_btn.on_click(start)

        def surprise():
            """Load the next vetted self-eval brief and launch it. The corpus
            (kicraft.tuning.benchmark) is walked in order via a persistent global
            counter, so repeated clicks stream the whole set one at a time — a
            continuous feed of known-good designs. `start` enforces the usual
            quota / verification gates."""
            if state["running"]:
                return
            briefs = _selfeval_briefs()
            if not briefs:
                ui.notify("No self-eval briefs are configured.", color="warning")
                return
            idx = _store().next_cycle_index("surprise_me_idx", len(briefs))
            brief.value = briefs[idx]
            start()

        surprise_btn.on_click(surprise)

        def _close_layout_editor():
            """Leave the manual layout editor; the render timer repaints
            the board view (pcb_view None) or gallery on its next tick.
            Deferred one tick: the Back button lives inside the slot
            being cleared, and deleting the clicked element during its
            own event dispatch is unsafe."""

            def _do_close():
                view["layout_editor"] = False
                view["layout_panel"] = None
                tabs.view_slot("place_route").clear()
                view["pcb_view"] = None
                view["pcb_mtime"] = None
                view["run_mtime"] = None
                view["leaf_progress_sig"] = None
                view["rescue_offered"] = False

            ui.timer(0.05, _do_close, once=True)

        def _open_layout_editor():
            if state["running"]:
                ui.notify("Wait for the current run to finish first.",
                          color="warning")
                return
            if not state["project_dir"] or not state["token"]:
                return
            u = _current_user()
            if not user_may_edit_layout(u):
                ui.notify("The layout editor needs a Pro or Max plan. "
                          "See Pricing.", color="warning")
                return
            # The editor's _on_save writes manual_layout.json/preview/stamp.log into
            # state["project_dir"]; in view mode that is the DURABLE tree, so
            # materialize a scratch workspace first and re-point project_dir at it.
            _ensure_workspace(state)
            view["layout_editor"] = True

            def _do_open():
                panel = LayoutEditorPanel(
                    project_dir=Path(state["project_dir"]),
                    stem=state["stem"],
                    token=state["token"],
                    user=u,
                    on_exit=_close_layout_editor,
                    is_run_active=lambda: bool(state["running"]),
                    on_route=_start_manual_route,
                )
                view["layout_panel"] = panel
                slot = tabs.view_slot("place_route")
                slot.clear()  # deletes the entry button (deferred, see above)
                with slot:
                    # render() builds its shell synchronously and fills it
                    # from a background task (leaf discovery + PNG rendering
                    # run in the executor, never on the UI event loop). It
                    # must stay synchronous HERE: this timer element lives in
                    # the slot cleared above, so the clear cancels this very
                    # task the moment it awaits.
                    panel.render()

            ui.timer(0.05, _do_open, once=True)

        def _start_manual_route():
            """Enqueue a manual_route job for this project's workspace and
            return the tab to the live build view; logs + queue position
            stream through the same plumbing as a normal build."""
            if _project_run_live(state):
                ui.notify("A run is already in progress.", color="warning")
                return
            u = _current_user()
            if not user_may_edit_layout(u):
                ui.notify("Routing a manual layout needs a Pro or Max plan.",
                          color="warning")
                return
            _ensure_workspace(state)  # defensive: editor-open should have made it
            if not state.get("ws") or not state.get("project_dir"):
                return
            ml = (Path(state["project_dir"]) / ".experiments" / "manual"
                  / "manual_layout.json")
            if not ml.is_file():
                ui.notify("Save the layout first.", color="warning")
                return
            state.update(running=True, done=False, ok=None, status=None)
            status.text = ("Routing your manual layout (FreeRouting, may take "
                           "minutes) -- live progress is in the Place/Route tab.")
            design_btn.disable()
            _close_layout_editor()
            threading.Thread(target=_rerun_build_worker,
                             args=(state, "manual_route"), daemon=True).start()

        def _open_rules_panel():
            if state["running"]:
                ui.notify("Wait for the current run to finish first.",
                          color="warning")
                return
            _ensure_workspace(state)  # rules edits (commit_slot) must land in scratch
            if not state["project_dir"] or not state.get("ws"):
                return
            u = _current_user()
            if not user_may_edit_layout(u):
                ui.notify("Placement rules need a Pro or Max plan. "
                          "See Pricing.", color="warning")
                return
            view["layout_editor"] = True  # the panel owns the view slot

            def _do_open():
                panel = PlacementRulesPanel(
                    ws=Path(state["ws"]),
                    project_dir=Path(state["project_dir"]),
                    stem=state["stem"],
                    user=u,
                    on_exit=_close_layout_editor,
                    on_rebuild=_start_replace_build,
                    is_run_active=lambda: bool(state["running"]),
                )
                view["layout_panel"] = panel
                slot = tabs.view_slot("place_route")
                slot.clear()
                with slot:
                    panel.render()

            ui.timer(0.05, _do_open, once=True)

        def _start_replace_build(
                message: str = ("Re-placing with your rules (place + route + "
                                "fab, may take minutes) -- live progress is "
                                "in the tabs.")):
            """LLM-free rebuild (synthesize -> place -> route -> fab): after a
            committed placement-rules edit, or directly from the Rebuild
            button (e.g. to retry a failed board on a newer pipeline)."""
            if _project_run_live(state):
                ui.notify("A run is already in progress.", color="warning")
                return
            _ensure_workspace(state)  # rehydrate the durable project before the build enqueue
            if not state.get("ws") or not state.get("project_dir"):
                return
            state.update(running=True, done=False, ok=None, status=None)
            status.text = message
            design_btn.disable()
            _close_layout_editor()
            threading.Thread(target=_rerun_build_worker,
                             args=(state, "build"), daemon=True).start()

        def _start_rebuild():
            _start_replace_build(
                "Rebuilding (synthesize + place + route + fab, may take "
                "minutes) -- live progress is in the tabs.")

        def _mark_fab_invalid():
            """Fail loudly in the FAB tab: red icon + banner. The board on
            display is a failed verify candidate (kept for inspection), so no
            valid fab package exists and any earlier download is stale."""
            if view.get("fab_invalid"):
                return
            view["fab_invalid"] = True
            tabs.set_statuses({"fab": "failed"})
            stale = bool(state.get("zip"))
            with tabs.view_slot("fab"):
                with ui.card().classes("w-full p-3").style(
                        "border:1px solid #b91c1c;background:#450a0a"):
                    ui.label("Fab package invalid -- this build FAILED "
                             "verification.") \
                        .classes("text-sm font-medium").style("color:#fecaca")
                    ui.label(
                        "The board in PLACE/ROUTE is the failed candidate, "
                        "kept so the problem can be inspected. Do not "
                        "fabricate it. "
                        + ("Any package downloaded earlier is from an older "
                           "successful build and does NOT match this board."
                           if stale else
                           "No fab package was exported for this build.")
                    ).classes("text-xs").style("color:#fca5a5")

        def _layout_editor_entry_row(label: str = "Edit layout") -> None:
            """Buttons into the manual layout editor + placement rules (both
            tier-gated visually; the open handlers and the panels' apply
            paths re-check server-side) + an ungated deterministic Rebuild
            (same machinery as the original build, no LLM spend)."""
            gated = not user_may_edit_layout(user)
            with ui.row().classes("items-center gap-2 mt-1"):
                ui.button("Rebuild board", icon="restart_alt",
                          on_click=_start_rebuild).props("dense outline") \
                    .tooltip("Re-run the deterministic build on the current "
                             "design: synthesize, place, route, verify, "
                             "export. No AI step. Picks up pipeline fixes "
                             "deployed since the last build.")
                btn = ui.button(label, icon="design_services",
                                on_click=_open_layout_editor).props("dense outline")
                rules_btn = ui.button("Placement rules", icon="rule",
                                      on_click=_open_rules_panel) \
                    .props("dense outline")
                if gated:
                    btn.disable()
                    btn.tooltip("Manual layout editing (drag blocks, board "
                                "size and shape, mounting holes) is a "
                                "Pro/Max feature.")
                    rules_btn.disable()
                    rules_btn.tooltip("Per-component placement rules are a "
                                      "Pro/Max feature.")
                    ui.link("Upgrade", "/pricing").classes("text-xs") \
                        .style("color:#38bdf8")

        def _live_sig():
            # Includes the running flag so a run parking on a question (it stays
            # registered) still refreshes the list's status label. list() first:
            # design/build threads insert and pop entries concurrently, and
            # iterating the live dict here raises 'dictionary changed size'.
            return tuple(sorted(
                (pid, bool(st.get("running")))
                for pid, st in list(_LIVE_RUNS.items())
                if st.get("user_id") == user.id))

        def render():
            # This timer only ticks while the page's websocket is connected, so it
            # is the "still watching" signal that suppresses walk-away emails.
            notify.mark_active(state.get("user_id"))
            evs = state["events"]
            changed = False
            while view["rendered"] < len(evs):
                e = evs[view["rendered"]]
                if e.get("kind") == "build_log":
                    view["build_lines"].append(e.get("text", ""))
                tabs.push(e)
                view["rendered"] += 1
                changed = True
            if changed:
                tabs.flush()
            if is_admin(user) and state["spend"] is not None:
                spend.text = f"Spent this design: ${state['spend']:.4f}"

            # Board ID chip: tracks the open design (state rebinding included).
            code = state.get("board_code")
            want = f"Board ID: {code}" if code else ""
            if board_label.text != want:
                board_label.text = want
                board_label.set_visibility(bool(want))

            # A background run starting or ending (this page's, another tab's,
            # or one this page detached from) changes what the project list
            # should offer (Open/Download, status), so rebuild it on roster
            # changes -- this is what used to need a manual page reload.
            sig = _live_sig()
            if sig != view.get("live_sig"):
                view["live_sig"] = sig
                refresh_account_ui()

            # Clarifying-question panel: (re)build only when the question set changes
            # (a worker parks the run from its thread; this picks it up next tick).
            if state.get("questions") != view.get("questions_rendered"):
                build_question_panel()

            # Design-stage inspectors: rebuild from state.json whenever it changes.
            # read_root is the scratch workspace, or (on a reopen) the durable
            # project root -- the readers resolve either via the storage accessors.
            read_root = Path(state["ws"]) if state.get("ws") else None
            if read_root:
                # Seed the price cache from this project's persisted prices once
                # (so a reopen shows costs immediately, before any new fetch).
                if view.get("prices_loaded_ws") != str(read_root):
                    view["prices_loaded_ws"] = str(read_root)
                    _load_price_cache(read_root)
                mt = _mtime(_state_path(read_root))
                if mt and mt != view["state_mtime"]:
                    view["state_mtime"] = mt
                    sj = _read_state_json(read_root)
                    if not sj:
                        # Caught the file mid-write: un-consume the mtime and
                        # retry next tick instead of wiping the inspectors.
                        view["state_mtime"] = None
                    else:
                        # Unconditional set: an empty spec CLEARS a stage whose
                        # slot was nulled by an edit (it used to stay stale).
                        # set_inspector keeps an in-progress live draft on screen.
                        for stg in ("intent", "functional_spec", "architecture",
                                    "bom", "wiring"):
                            tabs.set_inspector(stg, _inspector_spec(
                                stg, sj, {}, None, view["build_lines"]))
                        # Live-price any BOM parts in the background (fills in the
                        # cost column + total once the fetch lands; cached parts
                        # are instant).
                        bom_parts = (sj.get("bom") or {}).get("parts") or []
                        if bom_parts:
                            # In view mode str(read_root) is the durable root, so the
                            # background _save_price_cache writes through to durable
                            # kicraft/<price file> -- the intended cache, no workspace.
                            _ensure_bom_prices(bom_parts, str(read_root), state)

            # Prices arrive on a background thread; re-render the BOM when they do.
            if read_root and state.get("prices_rev") != view.get("prices_rev_seen"):
                view["prices_rev_seen"] = state.get("prices_rev")
                spec = _inspector_spec(
                    "bom", _read_state_json(read_root), {}, None,
                    view["build_lines"])
                if spec:
                    tabs.set_inspector("bom", spec)

            # Even when the build later FAILS, show the schematic as soon as synthesis
            # writes the sheets: discover the generated dir from the workspace so the
            # viewer never depends on a project_stem being recorded or on the pre-build
            # wiring having found one. Self-heals on both live and reopened runs.
            if state["project_dir"] is None and read_root:
                pd = _discover_generated_dir(read_root)
                if pd is not None:
                    state["stem"] = pd.name
                    state["project_dir"] = str(pd)
                    state["token"] = _register_project_dir(pd)

            project_dir = Path(state["project_dir"]) if state["project_dir"] else None

            # Schematic appears in the Synthesize tab once synth writes the sheets.
            if project_dir is not None and view["sch_view"] is None:
                srcs = _schematic_sources(project_dir, state["stem"], state["token"])
                if srcs:
                    sj = _read_state_json(read_root) if read_root else {}
                    tabs.set_inspector("synthesize", _inspector_spec(
                        "synthesize", sj, {}, project_dir, view["build_lines"]))
                    slot = tabs.view_slot("synthesize")
                    slot.clear()  # an edit-rerun renders again; never stack views
                    with slot:
                        view["sch_view"] = _render_synth_view(
                            srcs, state["stem"], project_dir)
                    # Painted already if synthesize is the visible tab now; otherwise
                    # mark it for a re-fit when the user first reveals it.
                    view["sch_revealed"] = tabs.active() == "synthesize"

            # Place/route: live progress bar (inspector) + KiCanvas board views
            # (view_slot) during the build, replaced by the final PCB when done.
            if project_dir is not None:
                rmt = _mtime(project_dir / ".experiments" / "run_status.json")
                if rmt and rmt != view["run_mtime"]:
                    view["run_mtime"] = rmt
                    rs = _read_run_status(project_dir)
                    if rs and state.get("token"):
                        # Inject live board URLs + leaf progress so the inspector
                        # and view_slot renderers can use them without re-reading disk.
                        urls = _live_board_urls(rs, project_dir, state["token"])
                        rs["_live_leaf_source"] = urls["leaf"]
                        rs["_live_parent_source"] = urls["parent"]
                        rs["_leaf_progress"] = _leaf_layout_progress(project_dir, state["token"])
                    tabs.set_inspector("place_route", _inspector_spec(
                        "place_route", {}, rs, project_dir, view["build_lines"]))
                    if (not state["pcb_ready"] and state["token"]
                            and not view.get("layout_editor")):
                        prog = rs.get("_leaf_progress") or []
                        # Rebuild whenever run_status changes (leaf completion,
                        # phase transitions, board URLs update).
                        slot = tabs.view_slot("place_route")
                        slot.clear()
                        with slot:
                            _render_leaf_gallery(prog, rs)
                if state["pcb_ready"] and not view.get("layout_editor"):
                    pcb_name = f"{state['stem']}.kicad_pcb"
                    pcb_path = project_dir / pcb_name
                    pcb_url = f"/project/{state['token']}/{pcb_name}"
                    if view["pcb_view"] is None:
                        view["pcb_mtime"] = _mtime(pcb_path)
                        slot = tabs.view_slot("place_route")
                        slot.clear()  # drop the progress gallery; show the final board
                        with slot:
                            ui.label("PCB").classes("text-xs font-medium").style("color:#94a3b8")
                            view["pcb_view"] = KiCanvasView(
                                [KiCanvasSource(pcb_url, pcb_name)],
                                height="", style=_BUILD_VIEW_STYLE)
                            view["pcb_revealed"] = tabs.active() == "place_route"
                            # Click during a still-running build is refused
                            # by the open handler with a notify.
                            _layout_editor_entry_row()
                    else:
                        mt = _mtime(pcb_path)
                        if mt != view["pcb_mtime"]:
                            view["pcb_mtime"] = mt
                            view["pcb_view"].refresh()

                # Rescue path: the parent place/route failed (live this
                # session, ok=False, or a reopened failed project,
                # state["failed"]) but the individual circuit blocks
                # routed; offer the manual layout editor so the user can
                # finish the board by hand instead of abandoning the
                # design.
                if (not view.get("rescue_offered")
                        and not view.get("layout_editor")
                        and not state["running"]
                        and (state.get("ok") is False or state.get("failed"))
                        and state["token"]
                        and leaf_artifacts_exist(project_dir)):
                    view["rescue_offered"] = True
                    with tabs.view_slot("place_route"):
                        with ui.card().classes("w-full p-3").style(
                                "border:1px solid #b45309;background:#451a03"):
                            ui.label(
                                "Automatic board layout failed, but the "
                                "circuit blocks themselves routed. You can "
                                "place them on the board yourself."
                            ).classes("text-sm").style("color:#fcd34d")
                            _layout_editor_entry_row(
                                "Rescue: lay out the board manually")

            if state["done"]:
                design_btn.enable()
                if not view["account_refreshed"]:
                    view["account_refreshed"] = True
                    refresh_account_ui()
                # The user is looking at the finished result, so it no longer
                # counts as "unseen" for the auto-open default. Parked runs are
                # done=True too but not terminal -- their result is still owed.
                if (state["project_id"] and not view["viewed_marked"]
                        and state.get("status") != "awaiting_input"):
                    view["viewed_marked"] = True
                    try:
                        _store().mark_viewed(state["project_id"])
                    except Exception:
                        pass
                if state["ok"]:
                    status.text = (
                        "Done (with a caution). Your KiCad project is ready."
                        if view.get("fab_caution")
                        else "Done. Your KiCad project is ready."
                    )
                    if not view["fab_done"]:
                        view["fab_done"] = True
                        sj = _read_state_json(read_root) if read_root else {}
                        rs = _read_run_status(project_dir) if project_dir else {}
                        # Non-blocking fab warnings (e.g. a minor courtyard clip):
                        # the project IS ready + 3D-rendered, but we flag the gap
                        # in yellow rather than a plain green "Done".
                        build_warnings = list(
                            (sj.get("artifacts") or {}).get("build_warnings") or []
                        )
                        view["fab_caution"] = bool(build_warnings)
                        if build_warnings:
                            status.text = "Done (with a caution). Your KiCad project is ready."
                        for stg in ("synthesize", "place_route", "electrical_review", "fab"):  # finalize build logs
                            tabs.set_inspector(stg, _inspector_spec(
                                stg, sj, rs, project_dir, view["build_lines"]))
                        if state["zip"]:
                            with tabs.view_slot("fab"):
                                if build_warnings:
                                    with ui.element("div").classes(
                                        "w-full max-w-3xl q-mb-sm q-pa-sm rounded-borders"
                                    ).style("background:#422006;border:1px solid #eab308"):
                                        ui.label("⚠ Fabricable, with a caution").classes(
                                            "text-sm text-weight-medium"
                                        ).style("color:#eab308")
                                        for w in build_warnings:
                                            ui.label(w).classes("text-xs").style(
                                                "color:#fde68a"
                                            )
                                # Assembled-board 3D render from the fab stage
                                # (best-effort artifact: absent when kicad-cli
                                # render failed; the package is still complete).
                                png = (project_dir / "fab" / "board_3d.png"
                                       if project_dir else None)
                                if png is not None and png.is_file() and state["token"]:
                                    ui.image(
                                        f"/project/{state['token']}/render/fab/"
                                        f"board_3d.png?v={int(png.stat().st_mtime)}"
                                    ).classes("w-full max-w-3xl rounded-borders q-mb-sm")
                                ui.button("Download KiCad project (.zip)", icon="download",
                                          on_click=lambda: ui.download(state["zip"])) \
                                    .props("color=positive")
                elif state["ok"] is False:
                    status.text = ("Build failed. The synthesized schematic is shown in "
                                   "the Synthesize tab (red) for review."
                                   + (f" Board ID: {code}." if code else ""))
                    if state.get("pcb_ready"):
                        _mark_fab_invalid()
                    # Surface the support dialog ONCE per failed run on this page
                    # (the failure itself is already auto-filed by the worker);
                    # closing it without sending stays closed.
                    if not view.get("support_prompted"):
                        view["support_prompted"] = True
                        open_support_dialog(auto=True)

        refresh_account_ui()
        view["live_sig"] = _live_sig()
        # Default view: an explicit /?project=<id> deep link (the clone flow and
        # the notification emails) wins; otherwise surface the design that needs
        # the user instead of a blank composer -- a run still going (or parked on
        # a question) first, else the newest finished design they haven't seen.
        # Arriving with a prefilled brief (?prompt= / a sample picked before
        # signup) means "start a new one", so that keeps the composer.
        requested = None
        if project.strip().isdigit():
            rp = _store().get_project(int(project.strip()))
            if rp is not None and rp.user_id == user.id:  # own projects only
                requested = rp
        if requested is not None:
            open_project(requested, notify=False)
        elif not prefill:
            _auto = _pick_default_project(user.id)
            if _auto is not None:
                open_project(_auto, notify=False)
        ui.timer(0.2, render)

    # Non-sticky: the branding sits at the end of the page content and scrolls
    # away with it (fixed=False) rather than pinning to the viewport bottom --
    # on mobile the pinned banner used to overlap the composer.
    with ui.footer(fixed=False).classes("justify-center py-1") \
            .style("background:var(--kc-bg);border-top:1px solid var(--kc-border)"):
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
            {"ref": "U1", "value": "TP4056", "symbol": "tp4056:TP4056_C725790",
             "footprint": "tp4056:ESOP-8", "sheet": "MAIN"},
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
    # total render with no network. TP4056/USB-C come from curated bundles, so
    # they resolve to their manifest LCSC id (id:C…); the LED is a generic passive.
    _DEMO_PRICES = {
        "id:C725790": {"unit_price": 0.18, "lcsc": "C725790", "stock": 9999},
        "id:C165948": {"unit_price": 0.0667, "lcsc": "C165948", "stock": 9999},
        "kw:white LED 0603": {"unit_price": 0.014, "lcsc": "C72043", "stock": 9999},
    }

    @ui.page("/demo")
    def demo_page():
        """Dev-only: replay a canned design through the per-stage tabs so the layout
        and styling can be previewed and screenshotted with no spend or network.
        Registered only when KICRAFT_WEB_DEMO is set (off in production)."""
        ui.dark_mode().enable()
        ui.query("body").style("background:var(--kc-bg)")
        _mobile_head()
        with ui.header().classes("items-center justify-between") \
                .style("background:var(--kc-surface);border-bottom:1px solid var(--kc-border)"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("design preview (demo)").classes("text-sm kc-tagline").style("color:#94a3b8")
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
                        for stg in ("synthesize", "place_route", "electrical_review", "fab"):
                            tabs.set_inspector(stg, _inspector_spec(
                                stg, _DEMO_STATE, rs, None, d["build_lines"]))
                    d["i"] += 1
                    pushed += 1
                tabs.flush()
                if d["i"] >= len(evs):
                    timer.cancel()  # replay finished; stop ticking

            timer = ui.timer(0.25, step)


# Admin dashboards live in their own module; importing it registers the
# /admin/* @ui.page routes (and pulls its shared helpers from this module).
# Reload-safety: the web test harnesses call importlib.reload(web) to reset
# module state between cases. That re-runs web's own @ui.page decorators but NOT
# an already-imported submodule's, and NiceGUI drops a page route whose module
# didn't (re-)register -- so on a *reload* (not the first import) we must re-run
# routes_admin too, or every /admin/* page 404s. (render_serving is immune: its
# @app.get routes live on the FastAPI app, which a reload doesn't clear.)
from . import routes_admin  # noqa: E402,F401  side-effecting: registers routes
if "_ADMIN_ROUTES_REGISTERED" in globals():  # set below => this is a reload
    importlib.reload(routes_admin)
_ADMIN_ROUTES_REGISTERED = True


def main() -> None:
    Settings.from_env()  # fail fast if OPENROUTER_API_KEY is missing; also loads .env
    store = _store()
    if not (_signup_code() or store.signup_open()
            or any(c["enabled"] for c in store.list_invite_codes())):
        print("WARNING: public signup is off and no invite code exists; no one can "
              "register. Mint a code at /admin/invites or set KICRAFT_SIGNUP_CODE.")
    _gc_workspaces()
    # Recover builds that outlived the previous web process (see _orphan_reaper).
    threading.Thread(target=_orphan_reaper, daemon=True).start()
    # Always-on host-resource sampler feeding the /admin usage charts
    # (drive / RAM / CPU over time). Idempotent + daemon; failures are
    # swallowed inside the sampler so they never affect serving.
    try:
        from .host_metrics import start_host_metrics_sampler
        start_host_metrics_sampler()
    except Exception:  # pragma: no cover - diagnostics only
        print("WARNING: host-metrics sampler failed to start; "
              "/admin usage charts will be empty until restart.")
    ui.run(
        host=os.environ.get("KICRAFT_WEB_HOST", "0.0.0.0"),
        port=int(os.environ.get("KICRAFT_WEB_PORT", "8080")),
        title="KiCraft",
        # Shared with the capability tokens; never falls open to a public
        # default (render_serving generates a per-process secret when unset).
        storage_secret=render_serving.storage_secret(),
        reload=False,
        show=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
