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

import hmac
import json
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

from nicegui import app, ui
from starlette.responses import FileResponse, PlainTextResponse, RedirectResponse

from .accounts import AccountStore
from .config import LEGAL_VERSION, Settings, default_legal_dir
from .kicanvas import KICANVAS_ASSET, KiCanvasSource, KiCanvasView, kicanvas_head
from .spend_guard import SpendGuard
from .stage_driver import DESIGN_STAGES, KICRAFT, drive_chain
from .stagetabs import StageTabs, demo_events

# Self-host the KiCanvas ES module bundle so the browser fetches it same-origin.
app.add_static_files("/static", str(KICANVAS_ASSET.parent))

# Raw-file serving: a capability token maps to a project dir. The token is minted
# only inside the authed page, so it gates access without depending on
# app.storage.user (whose getter can assert outside the page/connection flow).
_PROJECT_TOKENS: dict[str, Path] = {}
_ALLOWED_SUFFIXES = (".kicad_sch", ".kicad_pcb", ".kicad_pro")


_STORE: AccountStore | None = None


def _store() -> AccountStore:
    """The shared accounts store, built once per process from settings."""
    global _STORE
    if _STORE is None:
        s = Settings.from_env()
        _STORE = AccountStore(s.users_db_path, s.projects_dir)
    return _STORE


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
    return _store().get_user(int(uid))


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


def _register_project_dir(project_dir: Path) -> str:
    """Mint an unguessable token for a project dir so the browser can fetch its raw
    KiCad files. Cap the map so a long-lived server does not grow without bound."""
    token = secrets.token_urlsafe(16)
    _PROJECT_TOKENS[token] = project_dir
    if len(_PROJECT_TOKENS) > 256:
        for old in list(_PROJECT_TOKENS)[:-128]:
            _PROJECT_TOKENS.pop(old, None)
    return token


@app.get("/project/{token}/{filename}")
def serve_project_file(token: str, filename: str):
    """Serve one KiCad file from a tokened project dir to the browser (KiCanvas).

    Defends three ways against traversal: basename-only (any slash rejected), a
    suffix whitelist, and a check that the resolved target sits directly in the
    project dir. `no-store` so a rewritten board is always re-fetched.
    """
    base = _PROJECT_TOKENS.get(token)
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


def _zip_generated(ws: Path) -> str | None:
    gen = ws / "generated"
    if not gen.is_dir():
        return None
    base = str(ws / "kicraft_project")
    return shutil.make_archive(base, "zip", root_dir=str(gen))


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


def _inspector_spec(stage: str, sj: dict, run_status: dict, project_dir: Path | None,
                    build_lines: list[str]) -> list[dict]:
    """Build the structured project-state spec for a stage's inspector window.

    Pure-data stages read their committed slot from `sj` (state.json); the build
    stages read filesystem signals (sheets, run_status, build log). Returns the
    section list consumed by StagePanel.set_inspector; [] means "nothing yet".
    """
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
        secs = [{"type": "kv", "title": "Summary", "rows": [("parts", len(parts))]},
                {"type": "table", "title": "Parts",
                 "columns": ["ref", "value", "symbol", "footprint", "sheet"],
                 "rows": [[p.get("ref"), p.get("value"), p.get("symbol"),
                           p.get("footprint"), p.get("sheet")] for p in parts]}]
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
    status = "ok" if state.get("ok") else "failed"
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


def _design_worker(brief: str, state: dict) -> None:
    """Run the full pipeline in a background thread, streaming into `state`.

    The thread only mutates `state` (appends progress events, sets flags); every
    NiceGUI element update happens in the page render timer (elements must not be
    touched off the UI context).
    """
    ws = Path(tempfile.mkdtemp(prefix="kicraft_web_"))
    state["ws"] = str(ws)
    state["brief"] = brief

    def progress(ev):
        state["events"].append(ev)

    try:
        results, guard, _ = drive_chain(list(DESIGN_STAGES), brief, ws, progress=progress)
        state["spend"] = guard.get("spent_total_usd")
        if not all(r.get("commit_ok") for r in results):
            state["ok"] = False
            return

        # Hand off to the deterministic (zero-LLM) build: synthesize -> place ->
        # route -> verify -> fab. `build` re-runs synthesize as its first step, so
        # the schematic appears as soon as that step writes the sheets.
        stem = _read_project_stem(ws)
        if stem:
            project_dir = ws / "generated" / stem
            state["stem"] = stem
            state["project_dir"] = str(project_dir)
            state["token"] = _register_project_dir(project_dir)
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


def _legal_footer() -> None:
    """Public links to the Terms and Privacy Policy (shown on auth cards)."""
    with ui.row().classes("items-center gap-3 w-full justify-center"):
        ui.link("Terms of Service", "/terms", new_tab=True) \
            .classes("text-xs").style("color:#64748b")
        ui.link("Privacy Policy", "/privacy", new_tab=True) \
            .classes("text-xs").style("color:#64748b")


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
def login_page():
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
                ui.navigate.to("/")
            else:
                ui.notify("Wrong email or password.", color="negative")

        pw.on("keydown.enter", submit)
        ui.button("Sign in", on_click=submit).classes("w-full")
        ui.separator().style("background:#1e293b")
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("New to KiCraft?").classes("text-xs").style("color:#94a3b8")
            ui.button("Create an account", on_click=lambda: ui.navigate.to("/signup")) \
                .props("flat dense")
        _legal_footer()


@ui.page("/signup")
def signup_page():
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    with ui.card().classes("absolute-center w-96") \
            .style("background:#0f172a;border:1px solid #1e293b"):
        ui.label("Create your account").classes("text-2xl font-bold text-white")
        ui.label("Free tier: one design per week. No credit card.") \
            .classes("text-sm").style("color:#94a3b8")
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
        ui.label("Optional, and changeable later in Account & privacy.") \
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
            ui.navigate.to("/")

        pw.on("keydown.enter", submit)
        code.on("keydown.enter", submit)
        ui.button("Create account", on_click=submit).classes("w-full")
        with ui.row().classes("items-center justify-between w-full"):
            ui.label("Already registered?").classes("text-xs").style("color:#94a3b8")
            ui.button("Sign in", on_click=lambda: ui.navigate.to("/login")).props("flat dense")


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


@ui.page("/")
def index():
    user = _current_user()
    if user is None:
        return RedirectResponse("/login")
    if user.accepted_terms_version != LEGAL_VERSION:
        return RedirectResponse("/consent")
    q0 = _store().quota_status(user)

    kicanvas_head()
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    state: dict = {
        "events": [], "rendered": 0, "running": False, "done": False, "ok": None,
        "spend": None, "zip": None, "fab_done": False, "ws": None,
        "token": None, "project_dir": None, "stem": None, "pcb_ready": False,
        "sch_view": None, "pcb_view": None, "pcb_mtime": None,
        "state_mtime": None, "run_mtime": None, "build_lines": [],
        "user_id": None, "project_id": None, "brief": "", "account_refreshed": False,
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
            ui.label(user.email).classes("text-xs").style("color:#cbd5e1")
            tier_badge = ui.badge(q0["label"], color="primary")
            ui.button("Log out", on_click=logout).props("flat dense color=white").classes("text-xs")

    with ui.column().classes("w-full max-w-7xl mx-auto p-4 gap-3"):
        try:
            budget = SpendGuard(Settings.from_env()).status()
            ui.label(f"Daily budget remaining: ${budget['daily_remaining_usd']:.2f} "
                     f"of ${budget['daily_ceiling_usd']:.0f}").classes("text-xs").style("color:#64748b")
        except Exception:
            ui.label("").classes("hidden")

        quota_label = ui.label().classes("text-xs").style("color:#94a3b8")

        brief = ui.textarea(
            "Describe your board",
            placeholder="e.g. A USB-C powered LED night light with a slide switch and "
                        "three white LEDs, no microcontroller.").props("rows=4").classes("w-full")

        design_btn = ui.button("Design").props("color=primary unelevated")
        status = ui.label("").classes("text-sm").style("color:#e2e8f0")
        spend = ui.label("").classes("text-sm").style("color:#64748b")

        # Per-stage tabs: each phase gets its own tab with a project-state inspector
        # (left) over the LLM thinking + activity/log windows (right). The native
        # KiCad schematic/board (KiCanvas) and the download land in the build tabs.
        tabs = StageTabs()

        with ui.expansion("Your projects").classes("w-full mt-2") \
                .style("background:#0f172a;border:1px solid #1e293b"):
            proj_container = ui.column().classes("w-full gap-1 p-2")

        with ui.expansion("Account & privacy").classes("w-full") \
                .style("background:#0f172a;border:1px solid #1e293b"):
            with ui.column().classes("w-full gap-1 p-2"):
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
                        if p.zip_path and Path(p.zip_path).is_file():
                            ui.button("Download", icon="download",
                                      on_click=lambda zp=p.zip_path: ui.download(zp)) \
                                .props("flat dense")

        def refresh_account_ui():
            u = _current_user()
            if u is None:
                return
            q = _store().quota_status(u)
            period = "week" if q["window_days"] <= 7 else "month"
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
                         build_lines=[], account_refreshed=False)
            state["user_id"] = u.id
            state["project_id"] = pid
            state["brief"] = brief.value
            tabs.reset()
            status.text = ("Designing... (intent -> functional_spec -> architecture -> bom -> "
                           "wiring -> synthesize -> place/route -> fab)")
            design_btn.disable()
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
                if state["running"]:  # keep the live tab pinned to its newest output
                    tabs.scroll_active_to_bottom()
            if state["spend"] is not None:
                spend.text = f"Spent this design: ${state['spend']:.4f}"

            # Design-stage inspectors: rebuild from state.json whenever it changes.
            if state["ws"]:
                mt = _mtime(Path(state["ws"]) / ".kicraft" / "state.json")
                if mt and mt != state["state_mtime"]:
                    state["state_mtime"] = mt
                    sj = _read_state_json(Path(state["ws"]))
                    for stg in ("intent", "functional_spec", "architecture", "bom", "wiring"):
                        spec = _inspector_spec(stg, sj, {}, None, state["build_lines"])
                        if spec:
                            tabs.set_inspector(stg, spec)

            project_dir = Path(state["project_dir"]) if state["project_dir"] else None

            # Schematic appears in the Synthesize tab once synth writes the sheets.
            if project_dir is not None and state["sch_view"] is None:
                srcs = _schematic_sources(project_dir, state["stem"], state["token"])
                if srcs:
                    sj = _read_state_json(Path(state["ws"])) if state["ws"] else {}
                    tabs.set_inspector("synthesize", _inspector_spec(
                        "synthesize", sj, {}, project_dir, state["build_lines"]))
                    with tabs.view_slot("synthesize"):
                        ui.label("Schematic").classes("text-xs font-medium").style("color:#94a3b8")
                        state["sch_view"] = KiCanvasView(
                            [KiCanvasSource(u, f) for u, f in srcs], height="h-[360px]")

            # Place/route inspector (on run_status change) + the board in its tab.
            if project_dir is not None:
                rmt = _mtime(project_dir / ".experiments" / "run_status.json")
                if rmt and rmt != state["run_mtime"]:
                    state["run_mtime"] = rmt
                    tabs.set_inspector("place_route", _inspector_spec(
                        "place_route", {}, _read_run_status(project_dir), project_dir,
                        state["build_lines"]))
                if state["pcb_ready"]:
                    pcb_name = f"{state['stem']}.kicad_pcb"
                    pcb_path = project_dir / pcb_name
                    pcb_url = f"/project/{state['token']}/{pcb_name}"
                    if state["pcb_view"] is None:
                        state["pcb_mtime"] = _mtime(pcb_path)
                        with tabs.view_slot("place_route"):
                            ui.label("PCB").classes("text-xs font-medium").style("color:#94a3b8")
                            state["pcb_view"] = KiCanvasView(
                                [KiCanvasSource(pcb_url, pcb_name)], height="h-[420px]")
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
                    status.text = "Stopped. See the failing stage's tab."

        refresh_account_ui()
        ui.timer(0.2, render)


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
        with ui.column().classes("w-full max-w-7xl mx-auto p-4 gap-3"):
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
                        tabs.set_inspector(stg, _inspector_spec(stg, _DEMO_STATE, {}, None, []))
                    elif e.get("kind") == "build_done":
                        rs = {"phase": "done", "progress_percent": 100}
                        for stg in ("synthesize", "place_route", "fab"):
                            tabs.set_inspector(stg, _inspector_spec(
                                stg, _DEMO_STATE, rs, None, d["build_lines"]))
                    d["i"] += 1
                    pushed += 1
                tabs.flush()
                tabs.scroll_active_to_bottom()
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
