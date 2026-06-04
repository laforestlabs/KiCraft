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

from .config import Settings
from .feedview import FeedView, demo_events
from .kicanvas import KICANVAS_ASSET, KiCanvasSource, KiCanvasView, kicanvas_head
from .spend_guard import SpendGuard
from .stage_driver import DESIGN_STAGES, KICRAFT, drive_chain

# Self-host the KiCanvas ES module bundle so the browser fetches it same-origin.
app.add_static_files("/static", str(KICANVAS_ASSET.parent))

# All-phases pipeline: the design stages (DESIGN_STAGES) plus the build phases.
PHASES: list[tuple[str, str]] = [
    ("intent", "Intent"),
    ("functional_spec", "Functional"),
    ("architecture", "Architecture"),
    ("bom", "BOM"),
    ("wiring", "Wiring"),
    ("synthesize", "Synthesize"),
    ("place_route", "Place/Route"),
    ("fab", "Fab"),
]
_STEP_COLORS = {
    "pending": "bg-slate-700 text-slate-300",
    "active": "bg-amber-500 text-black",
    "done": "bg-green-600 text-white",
    "failed": "bg-red-600 text-white",
}
_CHIP_BASE = "px-2 py-1 rounded text-xs whitespace-nowrap"

# Raw-file serving: a capability token maps to a project dir. The token is minted
# only inside the authed page, so it gates access without depending on
# app.storage.user (whose getter can assert outside the page/connection flow).
_PROJECT_TOKENS: dict[str, Path] = {}
_ALLOWED_SUFFIXES = (".kicad_sch", ".kicad_pcb", ".kicad_pro")


def _access_password() -> str:
    """Read the gate password live. os.environ is populated from .env in main()
    (after this module is imported), so reading it at import time would capture
    an empty string. Read it per request instead."""
    return os.environ.get("KICRAFT_ACCESS_PASSWORD", "")


def _authed() -> bool:
    return bool(app.storage.user.get("authed"))


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


def _design_worker(brief: str, state: dict) -> None:
    """Run the full pipeline in a background thread, streaming into `state`.

    The thread only mutates `state` (appends progress events, sets flags); every
    NiceGUI element update happens in the page render timer (elements must not be
    touched off the UI context).
    """
    ws = Path(tempfile.mkdtemp(prefix="kicraft_web_"))
    state["ws"] = str(ws)

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
        state["done"] = True
        state["running"] = False


@ui.page("/login")
def login_page():
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    with ui.card().classes("absolute-center w-96") \
            .style("background:#0f172a;border:1px solid #1e293b"):
        ui.label("KiCraft").classes("text-2xl font-bold text-white")
        ui.label("Enter the access password to design a board.") \
            .classes("text-sm").style("color:#94a3b8")
        pw = ui.input("Access password", password=True, password_toggle_button=True).classes("w-full")

        def submit():
            access_pw = _access_password()
            if access_pw and hmac.compare_digest(pw.value or "", access_pw):
                app.storage.user["authed"] = True
                ui.navigate.to("/")
            elif not access_pw:
                ui.notify("Access is not configured (set KICRAFT_ACCESS_PASSWORD).",
                          color="negative")
            else:
                ui.notify("Wrong password.", color="negative")

        pw.on("keydown.enter", submit)
        ui.button("Enter", on_click=submit).classes("w-full")


@ui.page("/")
def index():
    if not _authed():
        return RedirectResponse("/login")

    kicanvas_head()
    ui.dark_mode().enable()
    ui.query("body").style("background:#0b1120")
    state: dict = {
        "events": [], "rendered": 0, "running": False, "done": False, "ok": None,
        "spend": None, "zip": None, "dl_added": False,
        "token": None, "project_dir": None, "stem": None, "pcb_ready": False,
        "sch_view": None, "pcb_view": None, "pcb_mtime": None, "synth_done": False,
    }

    with ui.header().classes("items-center justify-between") \
            .style("background:#0f172a;border-bottom:1px solid #1e293b"):
        ui.label("KiCraft").classes("text-xl font-bold text-white")
        ui.label("design a PCB from a sentence").classes("text-sm").style("color:#94a3b8")

    with ui.column().classes("w-full max-w-3xl mx-auto p-4 gap-3"):
        try:
            budget = SpendGuard(Settings.from_env()).status()
            ui.label(f"Daily budget remaining: ${budget['daily_remaining_usd']:.2f} "
                     f"of ${budget['daily_ceiling_usd']:.0f}").classes("text-xs").style("color:#64748b")
        except Exception:
            ui.label("").classes("hidden")

        brief = ui.textarea(
            "Describe your board",
            placeholder="e.g. A USB-C powered LED night light with a slide switch and "
                        "three white LEDs, no microcontroller.").props("rows=4").classes("w-full")

        design_btn = ui.button("Design").props("color=primary unelevated")
        status = ui.label("").classes("text-sm").style("color:#e2e8f0")
        spend = ui.label("").classes("text-sm").style("color:#64748b")

        # All-phases stepper: one chip per phase, recoloured as the run progresses.
        step_chips: dict[str, ui.label] = {}
        with ui.row().classes("w-full flex-wrap gap-1 items-center"):
            for key, label in PHASES:
                step_chips[key] = ui.label(label).classes(f"{_CHIP_BASE} {_STEP_COLORS['pending']}")

        # Live feed: a structured, themed render of each stage's reasoning, tool
        # calls, retries and the build log. FeedView appends only new elements per
        # tick (and grows one label per active block), so it stays cheap.
        with ui.scroll_area().classes("w-full h-96 rounded") \
                .style("background:#0f172a;border:1px solid #1e293b") as feed_scroll:
            feed_col = ui.column().classes("w-full p-3 gap-0")
        feed = FeedView(feed_col)

        # Native KiCad views (schematic after synth, board after route).
        sch_holder = ui.column().classes("w-full")
        pcb_holder = ui.column().classes("w-full")
        download_holder = ui.row()

        def set_step(key: str, status_name: str) -> None:
            chip = step_chips.get(key)
            if chip is not None:
                chip.classes(replace=f"{_CHIP_BASE} {_STEP_COLORS[status_name]}")

        def start():
            if state["running"]:
                return
            if not (brief.value or "").strip():
                ui.notify("Enter a brief first.", color="warning")
                return
            state.update(events=[], rendered=0, running=True, done=False, ok=None,
                         spend=None, zip=None, dl_added=False, token=None,
                         project_dir=None, stem=None, pcb_ready=False, sch_view=None,
                         pcb_view=None, pcb_mtime=None, synth_done=False)
            feed.clear()
            sch_holder.clear()
            pcb_holder.clear()
            download_holder.clear()
            for key, _ in PHASES:
                set_step(key, "pending")
            status.text = ("Designing... (intent -> functional_spec -> architecture -> bom -> "
                           "wiring -> synthesize -> place/route -> fab)")
            design_btn.disable()
            threading.Thread(target=_design_worker, args=(brief.value, state), daemon=True).start()

        design_btn.on_click(start)

        def _step_from_event(e) -> None:
            k = e.get("kind")
            if k == "stage_start":
                set_step(e.get("stage"), "active")
            elif k == "stage_done":
                set_step(e.get("stage"), "done" if e.get("ok") else "failed")
            elif k == "build_start":
                set_step("synthesize", "active")
            elif k == "build_done" and not e.get("ok"):
                set_step("place_route" if state["synth_done"] else "synthesize", "failed")

        def render():
            evs = state["events"]
            changed = False
            while state["rendered"] < len(evs):
                e = evs[state["rendered"]]
                feed.push(e)
                _step_from_event(e)
                state["rendered"] += 1
                changed = True
            if changed:
                feed.flush()
                if state["running"]:  # don't yank the view while reading folded stages
                    feed_scroll.scroll_to(percent=1.0)
            if state["spend"] is not None:
                spend.text = f"Spent this design: ${state['spend']:.4f}"

            # Schematic appears once the synthesize step has written the sheets.
            if state["project_dir"] and state["sch_view"] is None:
                project_dir = Path(state["project_dir"])
                srcs = _schematic_sources(project_dir, state["stem"], state["token"])
                if srcs:
                    set_step("synthesize", "done")
                    state["synth_done"] = True
                    with sch_holder:
                        ui.label("Schematic").classes("text-sm font-medium mt-2")
                        state["sch_view"] = KiCanvasView(
                            [KiCanvasSource(u, f) for u, f in srcs], height="h-[420px]")

            # Place/route progress + the board, while/after build runs.
            if state["project_dir"]:
                project_dir = Path(state["project_dir"])
                st = _read_run_status(project_dir)
                if st and state["synth_done"] and not state["pcb_ready"]:
                    set_step("place_route", "active")
                if state["pcb_ready"]:
                    set_step("place_route", "done")
                    pcb_name = f"{state['stem']}.kicad_pcb"
                    pcb_path = project_dir / pcb_name
                    pcb_url = f"/project/{state['token']}/{pcb_name}"
                    if state["pcb_view"] is None:
                        state["pcb_mtime"] = _mtime(pcb_path)
                        with pcb_holder:
                            ui.label("PCB").classes("text-sm font-medium mt-2")
                            state["pcb_view"] = KiCanvasView(
                                [KiCanvasSource(pcb_url, pcb_name)], height="h-[520px]")
                    else:
                        mt = _mtime(pcb_path)
                        if mt != state["pcb_mtime"]:
                            state["pcb_mtime"] = mt
                            state["pcb_view"].refresh()

            if state["done"]:
                design_btn.enable()
                if state["ok"]:
                    set_step("fab", "done")
                    status.text = "Done. Your KiCad project is ready."
                    if state["zip"] and not state["dl_added"]:
                        state["dl_added"] = True
                        with download_holder:
                            ui.button("Download KiCad project (.zip)", icon="download",
                                      on_click=lambda: ui.download(state["zip"])).props("color=positive")
                elif state["ok"] is False:
                    status.text = "Stopped. See the log above."

        ui.timer(0.2, render)


if os.environ.get("KICRAFT_WEB_DEMO"):

    @ui.page("/demo")
    def demo_page():
        """Dev-only: replay a canned design through FeedView so the live-log styling
        can be previewed and screenshotted with no spend or network. Registered only
        when KICRAFT_WEB_DEMO is set (off in production)."""
        ui.dark_mode().enable()
        ui.query("body").style("background:#0b1120")
        with ui.header().classes("items-center justify-between") \
                .style("background:#0f172a;border-bottom:1px solid #1e293b"):
            ui.label("KiCraft").classes("text-xl font-bold text-white")
            ui.label("design log preview (demo)").classes("text-sm").style("color:#94a3b8")
        with ui.column().classes("w-full max-w-3xl mx-auto p-4 gap-3"):
            ui.label("Replaying a canned design to preview the live-log styling.") \
                .classes("text-sm").style("color:#94a3b8")
            with ui.scroll_area().classes("w-full rounded") \
                    .style("height:36rem;background:#0f172a;border:1px solid #1e293b") as feed_scroll:
                feed_col = ui.column().classes("w-full p-3 gap-0")
            feed = FeedView(feed_col)
            d = {"events": demo_events(), "i": 0}

            def step():
                evs = d["events"]
                pushed = 0
                while d["i"] < len(evs) and pushed < 2:  # ~2 events/tick = streaming feel
                    feed.push(evs[d["i"]])
                    d["i"] += 1
                    pushed += 1
                feed.flush()
                feed_scroll.scroll_to(percent=1.0)

            ui.timer(0.25, step)


def main() -> None:
    Settings.from_env()  # fail fast if OPENROUTER_API_KEY is missing; also loads .env
    if not _access_password():
        print("WARNING: KICRAFT_ACCESS_PASSWORD is not set; the site will refuse logins. "
              "Set it before exposing kicraft.io.")
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
