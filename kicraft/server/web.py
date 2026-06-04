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
        "spend": None, "zip": None, "fab_done": False, "ws": None,
        "token": None, "project_dir": None, "stem": None, "pcb_ready": False,
        "sch_view": None, "pcb_view": None, "pcb_mtime": None,
        "state_mtime": None, "run_mtime": None, "build_lines": [],
    }

    with ui.header().classes("items-center justify-between") \
            .style("background:#0f172a;border-bottom:1px solid #1e293b"):
        ui.label("KiCraft").classes("text-xl font-bold text-white")
        ui.label("design a PCB from a sentence").classes("text-sm").style("color:#94a3b8")

    with ui.column().classes("w-full max-w-7xl mx-auto p-4 gap-3"):
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

        # Per-stage tabs: each phase gets its own tab with a project-state inspector
        # (left) over the LLM thinking + activity/log windows (right). The native
        # KiCad schematic/board (KiCanvas) and the download land in the build tabs.
        tabs = StageTabs()

        def start():
            if state["running"]:
                return
            if not (brief.value or "").strip():
                ui.notify("Enter a brief first.", color="warning")
                return
            state.update(events=[], rendered=0, running=True, done=False, ok=None,
                         spend=None, zip=None, fab_done=False, ws=None, token=None,
                         project_dir=None, stem=None, pcb_ready=False, sch_view=None,
                         pcb_view=None, pcb_mtime=None, state_mtime=None, run_mtime=None,
                         build_lines=[])
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
