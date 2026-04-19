"""Setup page -- hierarchical experiment configuration UI."""

from __future__ import annotations

from nicegui import ui

from ..state import PLACEMENT_PARAMS, get_state


def setup_page():
    state = get_state()

    ui.label("Hierarchical Experiment Setup").classes("text-2xl font-bold mb-2")
    ui.label(
        "Configure the bottom-up subcircuit experiment flow: routed leaf solving, "
        "parent composition, and canonical parent routing. The defaults aim "
        "for a balanced baseline: fast enough for routine iteration, but still "
        "useful as a quality signal."
    ).classes("text-sm text-gray-400 mb-6")

    with ui.tabs().classes("w-full") as tabs:
        strategy_tab = ui.tab("Run Strategy", icon="play_circle")
        placement_tab = ui.tab("Placement & Routing", icon="tune")
        hierarchy_tab = ui.tab("Hierarchy Scope", icon="account_tree")
        presets_tab = ui.tab("Presets", icon="bookmark")

    with ui.tab_panels(tabs, value=strategy_tab).classes("w-full"):
        with ui.tab_panel(strategy_tab):
            _strategy_panel(state)

        with ui.tab_panel(placement_tab):
            _placement_routing_panel(state)

        with ui.tab_panel(hierarchy_tab):
            _hierarchy_panel(state)

        with ui.tab_panel(presets_tab):
            _presets_panel(state)


def _strategy_panel(state):
    ui.label("Run Strategy").classes("text-lg font-bold mb-2")
    ui.label(
        "These settings control the outer experiment loop. Each round runs the "
        "hierarchical pipeline from routed leaves upward. Start with the balanced "
        "defaults, then widen settings only when the results justify the extra time."
    ).classes("text-sm text-gray-400 mb-4")

    with ui.grid(columns=2).classes("w-full gap-4"):
        ui.number(
            "Experiment rounds",
            value=state.strategy.get("rounds", 10),
            min=1,
            max=1000,
            step=1,
            on_change=lambda e: state.strategy.update({"rounds": int(e.value)}),
        ).tooltip(
            "How many full hierarchical attempts to run. A balanced default is 10 "
            "rounds for routine comparison without turning every run into an "
            "overnight job."
        )

        ui.number(
            "Leaf solve rounds per experiment round",
            value=state.strategy.get("leaf_rounds", 2),
            min=1,
            max=6,
            step=1,
            on_change=lambda e: state.strategy.update({"leaf_rounds": int(e.value)}),
        ).tooltip(
            "How many local solve attempts each leaf subcircuit gets inside one "
            "experiment round. A balanced default is 2: usually enough to improve "
            "stability over 1, without the runtime jump of deeper search."
        )

        ui.number(
            "Leaf workers",
            value=state.strategy.get("workers", 0),
            min=0,
            max=64,
            step=1,
            on_change=lambda e: state.strategy.update({"workers": int(e.value)}),
        ).tooltip(
            "Parallel workers for leaf solving. 0 = auto-select based on CPU "
            "count and number of leaves. Use 1 for debugging or a fixed value "
            "on machines where you want explicit control."
        )

        ui.number(
            "Base seed",
            value=state.strategy.get("seed", 0),
            min=0,
            step=1,
            on_change=lambda e: state.strategy.update({"seed": int(e.value)}),
        ).tooltip(
            "Master seed for reproducible hierarchical runs. Keep this fixed when "
            "comparing nearby settings; change it when you want to test whether a "
            "result is robust instead of lucky."
        )

    ui.separator().classes("my-4")

    with ui.grid(columns=2).classes("w-full gap-4"):
        ui.input(
            "PCB file",
            value=state.strategy.get("pcb_file", ""),
            on_change=lambda e: state.strategy.update({"pcb_file": e.value.strip()}),
        ).classes("w-full").tooltip("Top-level PCB used as the project anchor.")

        ui.input(
            "Schematic file",
            value=state.strategy.get("schematic_file", ""),
            on_change=lambda e: state.strategy.update(
                {"schematic_file": e.value.strip()}
            ),
        ).classes("w-full").tooltip("Top-level schematic for hierarchy parsing.")

        ui.input(
            "Parent selector",
            value=state.strategy.get("parent", "/"),
            on_change=lambda e: state.strategy.update(
                {"parent": e.value.strip() or "/"}
            ),
        ).classes("w-full").tooltip(
            "Parent node to compose and visualize. Use '/' for the top-level parent."
        )

        ui.input(
            "Only selectors (comma-separated)",
            value=", ".join(state.strategy.get("only", [])),
            on_change=lambda e: state.strategy.update({"only": _split_csv(e.value)}),
        ).classes("w-full").tooltip(
            "Optional leaf filters. Leave empty to solve the full leaf set."
        )

    ui.separator().classes("my-4")

    with ui.row().classes("w-full items-start gap-8"):
        with ui.column().classes("gap-2"):
            ui.switch(
                "Render PNG previews",
                value=state.toggles.get("render_png", True),
                on_change=lambda e: state.toggles.update({"render_png": bool(e.value)}),
            ).tooltip(
                "Keep visual artifacts up to date so the monitor and analysis pages "
                "can show progression. This is usually worth leaving on unless you "
                "are doing quick throughput-focused checks."
            )

        with ui.column().classes("gap-2"):
            ui.switch(
                "Keep per-round detail artifacts",
                value=state.toggles.get("save_round_details", True),
                on_change=lambda e: state.toggles.update(
                    {"save_round_details": bool(e.value)}
                ),
            ).tooltip("Preserve round JSON and related metadata for later inspection.")


def _placement_routing_panel(state):
    ui.label("Placement & Routing Parameters").classes("text-lg font-bold mb-2")
    ui.label(
        "These parameters control the placement engine physics, board geometry, "
        "connector behavior, simulated annealing refinement, and routing settings. "
        "Only values you change from their defaults are written to the config overlay "
        "passed to the solver. Reset a value to its default to remove it from the overlay."
    ).classes("text-sm text-gray-400 mb-4")

    groups: dict[str, list[dict]] = {}
    for param in PLACEMENT_PARAMS:
        groups.setdefault(param["group"], []).append(param)

    group_icons = {
        "Placement Physics": "science",
        "Board Geometry": "straighten",
        "Edge & Connectors": "electrical_services",
        "Component Behavior": "memory",
        "SA Refinement": "local_fire_department",
        "Routing": "route",
        "Thermal": "thermostat",
    }

    for group_name, params in groups.items():
        icon = group_icons.get(group_name, "settings")
        with ui.expansion(group_name, icon=icon).classes("w-full"):
            with ui.grid(columns=2).classes("w-full gap-3 p-2"):
                for param in params:
                    _render_param_control(state, param)


def _render_param_control(state, param: dict):
    key = param["key"]
    is_bool = param.get("type") == "bool"
    current_value = state.placement_config.get(key, param["default"])

    if is_bool:
        ui.switch(
            param["label"],
            value=bool(current_value),
            on_change=lambda e, k=key, p=param: _on_param_change(state, k, p, e.value),
        ).tooltip(param["description"])
    else:
        ui.number(
            param["label"],
            value=float(current_value),
            min=param["min"],
            max=param["max"],
            step=param["step"],
            on_change=lambda e, k=key, p=param: _on_param_change(state, k, p, e.value),
        ).tooltip(param["description"])


def _on_param_change(state, key: str, param: dict, value):
    if value is None:
        return
    is_bool = param.get("type") == "bool"
    if is_bool:
        typed_value = bool(value)
    elif isinstance(param["default"], int):
        typed_value = int(value)
    else:
        typed_value = float(value)

    if typed_value == param["default"]:
        state.placement_config.pop(key, None)
    else:
        state.placement_config[key] = typed_value


def _hierarchy_panel(state):
    ui.label("Hierarchy Scope").classes("text-lg font-bold mb-2")
    ui.label(
        "Define what part of the hierarchy the experiment should focus on and how "
        "the GUI should present progression."
    ).classes("text-sm text-gray-400 mb-4")

    with ui.card().classes("w-full p-4"):
        ui.label("Current hierarchical target").classes("text-md font-bold mb-2")
        with ui.grid(columns=2).classes("w-full gap-4"):
            ui.input(
                "Top-level parent selector",
                value=state.strategy.get("parent", "/"),
                on_change=lambda e: state.strategy.update(
                    {"parent": e.value.strip() or "/"}
                ),
            ).tooltip("The parent node that composition and visible assembly target.")

            ui.input(
                "Leaf filter selectors",
                value=", ".join(state.strategy.get("only", [])),
                on_change=lambda e: state.strategy.update(
                    {"only": _split_csv(e.value)}
                ),
            ).tooltip(
                "Optional list of leaf names, files, or instance paths to restrict solving."
            )

    ui.separator().classes("my-4")

    with ui.card().classes("w-full p-4"):
        ui.label("Hierarchy behavior").classes("text-md font-bold mb-2")
        with ui.column().classes("gap-3"):
            ui.switch(
                "Prefer full top-level progression in monitor",
                value=state.toggles.get("show_top_level_progress", True),
                on_change=lambda e: state.toggles.update(
                    {"show_top_level_progress": bool(e.value)}
                ),
            ).tooltip(
                "Bias the monitor toward showing parent/top-level readiness alongside leaf progress."
            )

            ui.switch(
                "Show accepted leaf artifacts prominently",
                value=state.toggles.get("show_leaf_artifacts", True),
                on_change=lambda e: state.toggles.update(
                    {"show_leaf_artifacts": bool(e.value)}
                ),
            ).tooltip(
                "Keep accepted routed leaf artifacts front-and-center in the GUI."
            )

            ui.switch(
                "Track composition outputs",
                value=state.toggles.get("track_composition_outputs", True),
                on_change=lambda e: state.toggles.update(
                    {"track_composition_outputs": bool(e.value)}
                ),
            ).tooltip(
                "Expose parent composition JSON and visible output artifacts in the GUI."
            )

    ui.separator().classes("my-4")

    with ui.card().classes("w-full p-4"):
        ui.label("Summary").classes("text-md font-bold mb-2")
        ui.markdown(
            f"""
- **Schematic:** `{state.strategy.get("schematic_file", "")}`
- **PCB:** `{state.strategy.get("pcb_file", "")}`
- **Parent:** `{state.strategy.get("parent", "/")}`
- **Leaf filters:** `{", ".join(state.strategy.get("only", [])) or "all leaves"}`

"""
        )


def _presets_panel(state):
    ui.label("Presets").classes("text-lg font-bold mb-2")
    ui.label(
        "Save and restore focused hierarchical experiment configurations."
    ).classes("text-sm text-gray-400 mb-4")

    preset_name = ui.input("Preset name", value="").classes("w-64")
    preset_notes = ui.textarea("Notes", value="").classes("w-full").props("rows=2")

    with ui.row().classes("gap-2 mb-4"):

        async def _save():
            name = preset_name.value.strip()
            if not name:
                ui.notify("Enter a preset name", type="warning")
                return
            config = state.to_config_dict()
            state.db.save_preset(name, config, preset_notes.value)
            ui.notify(f"Saved preset '{name}'", type="positive")
            _refresh_presets()

        ui.button("Save Current Config", on_click=_save, icon="save")

    ui.separator()

    ui.label("Saved Presets").classes("text-lg font-bold mt-3")
    presets_container = ui.column().classes("w-full gap-2")

    def _refresh_presets():
        presets_container.clear()
        presets = state.db.get_presets()
        if not presets:
            with presets_container:
                ui.label("No presets saved yet").classes("text-gray-500 italic")
            return

        with presets_container:
            for preset in presets:
                if not state.gui_cleanup.get(
                    "show_legacy_presets", False
                ) and preset.name in {
                    "Best (imported)",
                    "Best Hierarchical (imported)",
                }:
                    continue
                with ui.card().classes("w-full p-3"):
                    with ui.row().classes("items-center gap-3"):
                        ui.label(preset.name).classes("font-bold")
                        ui.label(
                            preset.created_at.strftime("%Y-%m-%d %H:%M")
                            if preset.created_at
                            else ""
                        ).classes("text-xs text-gray-500")
                        ui.space()
                        ui.button(
                            "Load",
                            icon="download",
                            on_click=lambda _, pn=preset.name: _load(pn),
                        ).props("flat dense")
                        ui.button(
                            "Delete",
                            icon="delete",
                            on_click=lambda _, pn=preset.name: _delete(pn),
                            color="red",
                        ).props("flat dense")
                    if preset.notes:
                        ui.label(preset.notes).classes("text-xs text-gray-400 mt-1")

    def _load(name: str):
        config = state.db.load_preset(name)
        if config:
            state.load_from_config(config)
            ui.notify(f"Loaded preset '{name}'", type="positive")
            ui.navigate.reload()
        else:
            ui.notify(f"Preset '{name}' not found", type="warning")

    def _delete(name: str):
        state.db.delete_preset(name)
        ui.notify(f"Deleted preset '{name}'")
        _refresh_presets()

    _refresh_presets()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]
