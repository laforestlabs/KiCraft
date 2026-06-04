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
import json
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

def _access_password() -> str:
    """Read the gate password live. os.environ is populated from .env in main()
    (after this module is imported), so reading it at import time would capture
    an empty string. Read it per request instead."""
    return os.environ.get("KICRAFT_ACCESS_PASSWORD", "")


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

    def progress(ev):
        state["events"].append(ev)

    try:
        results, guard, _ = drive_chain(list(DESIGN_STAGES), brief, ws, progress=progress)
        state["spend"] = guard.get("spent_total_usd")
        if not all(r.get("commit_ok") for r in results):
            state["ok"] = False
            return
        # Deterministic synthesize (zero LLM) -> KiCad files.
        progress({"kind": "stage_start", "stage": "synthesize"})
        syn = subprocess.run(
            KICRAFT + ["synthesize", ".kicraft/state.json", "generated", "--no-archive"],
            cwd=str(ws), capture_output=True, text=True, timeout=600)
        if syn.returncode != 0:
            progress({"kind": "tool_result", "name": "synthesize",
                      "output": (syn.stdout or syn.stderr)[-500:]})
            progress({"kind": "stage_done", "stage": "synthesize", "ok": False})
            state["ok"] = False
            return
        progress({"kind": "stage_done", "stage": "synthesize", "ok": True})
        state["zip"] = _zip_generated(ws)
        state["ok"] = bool(state["zip"])
    except Exception as e:  # surface, never crash the UI thread
        progress({"kind": "tool_result", "name": "error", "output": str(e)})
        progress({"kind": "stage_done", "stage": "error", "ok": False})
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

    state: dict = {"events": [], "rendered": 0, "running": False, "done": False,
                   "ok": None, "spend": None, "zip": None, "dl_added": False, "transcript": ""}

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
        with ui.scroll_area().classes("w-full h-96 border rounded bg-slate-50") as scroll:
            transcript = ui.label("").classes("whitespace-pre-wrap font-mono text-xs")
        download_holder = ui.row()

        def start():
            if state["running"]:
                return
            if not (brief.value or "").strip():
                ui.notify("Enter a brief first.", color="warning")
                return
            state.update(events=[], rendered=0, running=True, done=False, ok=None,
                         spend=None, zip=None, dl_added=False, transcript="")
            transcript.set_text("")
            download_holder.clear()
            status.text = "Designing... (intent -> functional_spec -> architecture -> bom -> "\
                          "wiring -> synthesize)"
            design_btn.disable()
            threading.Thread(target=_design_worker, args=(brief.value, state), daemon=True).start()

        design_btn.on_click(start)

        def _frag(e):
            k = e.get("kind")
            if k == "reasoning_delta":
                return e.get("text", "")
            if k == "answer_delta":
                return ""  # the JSON draft is the result, not "thinking"; keep it out of the feed
            if k == "stage_start":
                return f"\n\n=== {e.get('stage')} ===\n"
            if k == "tool":
                return f"\n> {e.get('name')}({json.dumps(e.get('args', {}))[:140]})\n"
            if k == "tool_result":
                return f"  -> {str(e.get('output', ''))[:240]}\n"
            if k == "retry":
                return f"\n! retry {e.get('stage')}: {json.dumps(e.get('errors'))[:200]}\n"
            if k == "stage_done":
                c = e.get("cost")
                cs = f"  (${c:.4f})" if isinstance(c, (int, float)) else ""
                return f"\n[{'OK' if e.get('ok') else 'FAIL'} {e.get('stage')}]{cs}\n"
            return ""

        def render():
            evs = state["events"]
            changed = False
            while state["rendered"] < len(evs):
                state["transcript"] += _frag(evs[state["rendered"]])
                state["rendered"] += 1
                changed = True
            if changed:
                transcript.set_text(state["transcript"])
                scroll.scroll_to(percent=1.0)
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
                    status.text = "Stopped. See the log above."

        ui.timer(0.2, render)


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
