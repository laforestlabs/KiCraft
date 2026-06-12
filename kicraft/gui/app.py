"""NiceGUI application shell — main tabbed layout."""

from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

from nicegui import app, ui
from nicegui.elements.timer import Timer as _NgTimer


def _patch_timer_context() -> None:
    """Stop NiceGUI 3.x timers from spamming tracebacks on page reload.

    Upstream race: ``Timer._get_context`` is written as
    ``return self.parent_slot or nullcontext()``, but ``Element.parent_slot``
    *raises* RuntimeError when its weakref has died (instead of returning
    None). The ``or`` never fires, so an in-flight tick whose parent slot
    was GC'd between ``_should_stop()`` and ``_get_context()`` always
    raises before the next stop check. ``on_disconnect``-based
    ``timer.cancel()`` cannot close this race -- the cancelled timer's
    next tick still enters ``_get_context``.

    This shim catches that one RuntimeError and falls back to nullcontext,
    letting the loop iterate once more, observe ``_should_stop()``, and
    exit cleanly. Targeted to ``Timer._get_context`` so no other element's
    behaviour is touched.
    """
    original = _NgTimer._get_context

    def _safe_get_context(self):
        try:
            return original(self)
        except RuntimeError:
            return nullcontext()

    _NgTimer._get_context = _safe_get_context


_patch_timer_context()

from .components.pipeline_tracker import pipeline_tracker
from .pages.design import design_page
from .pages.leaf_library import leaf_library_page
from .pages.monitor import monitor_page
from .pages.setup import setup_page
from .state import get_state


def _mount_experiment_assets() -> None:
    """Expose .experiments/ as /experiments/ for render previews.

    The monitor's pipeline graph serves per-leaf/round PNGs through
    this static mount. Mounting at /experiments/ keeps the URL space
    namespaced so nothing else collides with KiCraft artifact paths.
    """
    state = get_state()
    experiments_dir = state.experiments_dir
    # Create the dir before mounting so the static route is ALWAYS
    # registered. add_static_files runs once at startup; if .experiments/
    # does not exist yet (fresh project, GUI launched before the first
    # run), a guarded mount is skipped and every render <img> 404s even
    # after a later run creates the dir. mkdir keeps the mount
    # unconditional; it is idempotent when the dir already exists.
    experiments_dir.mkdir(parents=True, exist_ok=True)
    app.add_static_files("/experiments", str(experiments_dir))


_mount_experiment_assets()


def _load_json(path: Path) -> dict | None:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _import_hierarchical_best_preset() -> None:
    """Import the best hierarchical run summary as a preset if available."""
    state = get_state()
    if not state.toggles.get("import_best_as_preset", True):
        return

    summary_path = state.experiments_dir / "best" / "best_hierarchical_round.json"
    summary = _load_json(summary_path)
    if not summary:
        return

    from . import presets as preset_store

    if any(
        p.name == "Best Hierarchical (imported)" for p in preset_store.list_presets()
    ):
        return

    preset_config = {
        "_strategy": {
            "rounds": state.strategy.get("rounds", 50),
            "workers": state.strategy.get("workers", 0),
            "seed": summary.get("seed", 0),
            "pcb_file": state.strategy["pcb_file"],
        },
        "_hierarchical_best": summary,
    }

    notes = (
        "Auto-imported from hierarchical best summary "
        f"(round={summary.get('round_num', '?')}, score={summary.get('score', '?')})"
    )
    preset_store.save_preset(
        "Best Hierarchical (imported)", preset_config, notes
    )


def _auto_import_on_startup() -> None:
    """Run startup hooks: import the best preset and restore session state."""
    state = get_state()
    _import_hierarchical_best_preset()
    state.restore_session_state()


_auto_import_on_startup()


@ui.page("/")
def index() -> None:
    """Main page with tabbed layout."""
    state = get_state()

    ui.dark_mode(True)
    ui.add_head_html(
        """
    <style>
        .nicegui-content { max-width: 1400px; margin: 0 auto; }
        .q-tab-panel { padding: 16px 0 !important; }
    </style>
    <script>
        /* Workaround for a NiceGUI Socket.IO reconnect bug.
         *
         * The page template embeds the outbox's next_message_id at
         * render time into options.query.next_message_id. nicegui.js
         * passes that object straight into io(url, {query: ...}), and
         * Socket.IO-client reuses the same query object on every
         * reconnect attempt. window.nextMessageId is correctly
         * incremented as messages arrive, but options.query.next_message_id
         * is never updated -- so on any brief WS reconnect (a
         * round-end message burst that stalls the asyncio loop past
         * Socket.IO keepalive is enough), the server's handshake
         * handler gets a stale next_message_id (usually 0), runs
         * outbox.try_rewind, can't find that id in the pruned message
         * history, and falls through to its
         * window.location.reload() recovery -- snapping the active
         * tab back to Setup mid-run.
         *
         * Patch: when Socket.IO is about to reconnect, sync the
         * query value to whatever the client actually has. Then the
         * server's try_rewind finds the id in history and replays
         * the missed messages instead of reloading.
         */
        (function() {
            function attach() {
                if (!window.socket || !window.socket.io) {
                    setTimeout(attach, 50);
                    return;
                }
                const sync = function() {
                    try {
                        window.socket.io.opts.query.next_message_id =
                            window.nextMessageId;
                    } catch (e) { /* opts may not be writable on some versions */ }
                };
                window.socket.io.on("reconnect_attempt", sync);
                // Belt-and-suspenders: also sync just before the
                // engine.io transport opens, which is when the query
                // is actually serialised onto the URL.
                window.socket.io.on("open", sync);
            }
            attach();
        })();
    </script>
    """
    )

    with ui.header().classes("items-center justify-between px-6"):
        ui.label(f"{state.project_name} Experiment Manager" if state.project_name != "project" else "KiCad Experiment Manager").classes("text-xl font-bold tracking-wide")
        with ui.row().classes("items-center gap-3"):
            ui.label(f"Project: {state.project_root.name}").classes(
                "text-sm text-gray-400"
            )
            ui.badge("Hierarchical Subcircuits", color="green").classes("text-xs")

    # Whole-pipeline tracker (design stages + build progress), observable from
    # the very start of a project, not just the place/route phase.
    with ui.row().classes("w-full px-6 pb-1 border-b border-gray-800"):
        pipeline_tracker(state.project_root)

    # The Manual Layout tab moved to the web app's place/route panel
    # (kicraft.server.layout_panel); both built on kicraft.layout_editor.
    with ui.tabs().classes("w-full") as tabs:
        design_tab = ui.tab("Design", icon="schema")
        leaf_library_tab = ui.tab("Leaf Library", icon="library_books")
        setup_tab = ui.tab("Setup", icon="tune")
        monitor_tab = ui.tab("Monitor", icon="monitor")

    with ui.tab_panels(tabs, value=setup_tab).classes("w-full px-4"):
        with ui.tab_panel(design_tab):
            design_page()
        with ui.tab_panel(leaf_library_tab):
            leaf_library_page()
        with ui.tab_panel(setup_tab):
            setup_page()
        with ui.tab_panel(monitor_tab):
            monitor_page()
