"""kicraft.io web app: a gated, capped, live front end over the agent loop.

A single page: enter a brief, watch each design stage stream in as it commits,
see the running spend, and download the synthesized KiCad project. Every model
call still flows through the capped gateway (SpendGuard), so the global spend
ceilings and kill switch apply to the whole site. Access is gated by a shared
password (KICRAFT_ACCESS_PASSWORD) so only invited users spend the balance.

Run locally:   KICRAFT_ACCESS_PASSWORD=secret python -m kicraft.server.web
"""
from __future__ import annotations

import hmac
import os
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from nicegui import app, ui
from starlette.responses import RedirectResponse

from .config import Settings
from .spend_guard import SpendGuard
from .stage_driver import DESIGN_STAGES, KICRAFT, drive_chain

ACCESS_PASSWORD = os.environ.get("KICRAFT_ACCESS_PASSWORD", "")


def _authed() -> bool:
    return bool(app.storage.user.get("authed"))


def _zip_generated(ws: Path) -> str | None:
    gen = ws / "generated"
    if not gen.is_dir():
        return None
    base = str(ws / "kicraft_project")
    return shutil.make_archive(base, "zip", root_dir=str(gen))


def _design_worker(brief: str, state: dict) -> None:
    """Run the full pipeline in a background thread, streaming into `state`."""
    ws = Path(tempfile.mkdtemp(prefix="kicraft_web_"))
    state["ws"] = str(ws)
    try:
        results, guard, _ = drive_chain(
            list(DESIGN_STAGES), brief, ws,
            on_stage=lambda r: state["events"].append(r),
        )
        state["spend"] = guard.get("spent_total_usd")
        if not all(r.get("commit_ok") for r in results):
            state["ok"] = False
            return
        # Deterministic synthesize (zero LLM) -> KiCad files.
        syn = subprocess.run(
            KICRAFT + ["synthesize", ".kicraft/state.json", "generated", "--no-archive"],
            cwd=str(ws), capture_output=True, text=True, timeout=600)
        if syn.returncode != 0:
            state["events"].append({"stage": "synthesize", "commit_ok": False,
                                    "error": (syn.stdout or syn.stderr)[-400:]})
            state["ok"] = False
            return
        state["events"].append({"stage": "synthesize", "commit_ok": True})
        state["zip"] = _zip_generated(ws)
        state["ok"] = bool(state["zip"])
    except Exception as e:  # surface, never crash the UI thread
        state["events"].append({"stage": "error", "commit_ok": False, "error": str(e)})
        state["ok"] = False
    finally:
        state["done"] = True
        state["running"] = False


@ui.page("/login")
def login_page():
    ui.query("body").classes("bg-slate-50")
    with ui.card().classes("absolute-center w-96"):
        ui.label("KiCraft").classes("text-2xl font-bold")
        ui.label("Enter the access password to design a board.").classes("text-sm text-grey")
        pw = ui.input("Access password", password=True, password_toggle_button=True).classes("w-full")

        def submit():
            if ACCESS_PASSWORD and hmac.compare_digest(pw.value or "", ACCESS_PASSWORD):
                app.storage.user["authed"] = True
                ui.navigate.to("/")
            elif not ACCESS_PASSWORD:
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

    state: dict = {"events": [], "rendered": 0, "running": False, "done": False,
                   "ok": None, "spend": None, "zip": None, "dl_added": False}

    with ui.header().classes("items-center justify-between bg-slate-800"):
        ui.label("KiCraft").classes("text-xl font-bold text-white")
        ui.label("design a PCB from a sentence").classes("text-sm text-slate-300")

    with ui.column().classes("w-full max-w-3xl mx-auto p-4 gap-3"):
        try:
            budget = SpendGuard(Settings.from_env()).status()
            ui.label(f"Daily budget remaining: ${budget['daily_remaining_usd']:.2f} "
                     f"of ${budget['daily_ceiling_usd']:.0f}").classes("text-xs text-grey")
        except Exception:
            ui.label("").classes("hidden")

        brief = ui.textarea(
            "Describe your board",
            placeholder="e.g. A USB-C powered LED night light with a slide switch and "
                        "three white LEDs, no microcontroller.").props("rows=4").classes("w-full")

        design_btn = ui.button("Design")
        status = ui.label("").classes("text-sm")
        spend = ui.label("").classes("text-sm text-grey")
        log = ui.column().classes("w-full gap-1 font-mono text-sm")
        download_holder = ui.row()

        def start():
            if state["running"]:
                return
            if not (brief.value or "").strip():
                ui.notify("Enter a brief first.", color="warning")
                return
            state.update(events=[], rendered=0, running=True, done=False, ok=None,
                         spend=None, zip=None, dl_added=False)
            log.clear()
            download_holder.clear()
            status.text = "Designing... (intent -> functional_spec -> architecture -> bom -> "\
                          "wiring -> synthesize)"
            design_btn.disable()
            threading.Thread(target=_design_worker, args=(brief.value, state), daemon=True).start()

        design_btn.on_click(start)

        def render():
            evs = state["events"]
            while state["rendered"] < len(evs):
                e = evs[state["rendered"]]
                state["rendered"] += 1
                ok = e.get("commit_ok")
                mark, color = ("✓", "text-green-700") if ok else ("✗", "text-red-700")
                cost = e.get("cost_usd")
                cost_str = f"  (${cost:.4f})" if isinstance(cost, (int, float)) else ""
                with log:
                    ui.label(f"{mark} {e.get('stage')}{cost_str}").classes(color)
                    if not ok and (e.get("error") or e.get("commit")):
                        ui.label(str(e.get("error") or e.get("commit"))[:300]).classes(
                            "text-xs text-grey ml-4")
            if state["spend"] is not None:
                spend.text = f"Spent this design: ${state['spend']:.4f}"
            if state["done"]:
                design_btn.enable()
                if state["ok"]:
                    status.text = "Done. Your KiCad project is ready."
                    if state["zip"] and not state["dl_added"]:
                        state["dl_added"] = True
                        with download_holder:
                            ui.button("Download KiCad project (.zip)", icon="download",
                                      on_click=lambda: ui.download(state["zip"])).props("color=positive")
                else:
                    status.text = "Stopped. See the stage that failed above."

        ui.timer(0.6, render)


def main() -> None:
    Settings.from_env()  # fail fast if OPENROUTER_API_KEY is missing
    if not ACCESS_PASSWORD:
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
