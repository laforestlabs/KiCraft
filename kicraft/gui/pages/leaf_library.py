"""NiceGUI page: Leaf Library — promote and manage golden leaves.

Two sections:

(A) Promote wizard (top) — pick a source project + sheet + round, fill
    in metadata, write the leaf to ``$KICRAFT_LEAF_LIB``.
(B) Installed leaves (bottom) — card grid with thumbnail, description,
    provenance, and a Remove button.

The promote wizard requires the source project to have:
- ``.experiments/subcircuits/<leaf_key>/round_NNNN_*`` complete triads
- A ``<project_stem>_circuitchat_state.json`` so we can slice BOM rows
  and the autoplacer fragment for the chosen sheet. (Projects synthesized
  via CircuitChat write this; hand-rolled projects must save state
  alongside before promoting.)
"""

from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path
from typing import Any

from nicegui import ui

from kicraft.autoplacer.brain import pins as pins_module
from kicraft.leaf_library import (
    BrokenLeaf,
    LeafLibrary,
    LoadedLeaf,
    PromoteRequest,
    extract_leaf,
    resolve_library_dir,
)

logger = logging.getLogger(__name__)


def leaf_library_page() -> None:
    """Render the Leaf Library tab."""
    state: dict[str, Any] = {
        "lib": LeafLibrary(resolve_library_dir()),
        "source_project": "",
        "candidates_container": None,
        "installed_container": None,
    }

    with ui.column().classes("w-full gap-4"):
        ui.label("Leaf Library").classes("text-h6")
        ui.label(
            f"Library path: {state['lib'].base_dir}"
        ).classes("text-caption text-grey-7 font-mono")

        # =========================================================
        # (A) Promote wizard
        # =========================================================
        with ui.card().classes("w-full"):
            ui.label("Promote a leaf").classes("text-subtitle1")

            with ui.row().classes("w-full no-wrap items-center"):
                proj_input = ui.input(
                    placeholder="/path/to/source/project",
                    label="Source project directory",
                ).classes("col-grow")
                ui.button(
                    "Load",
                    icon="folder_open",
                    on_click=lambda: _reload_candidates(state, proj_input.value),
                )

            state["candidates_container"] = ui.column().classes("w-full gap-2")

        # =========================================================
        # (B) Installed leaves
        # =========================================================
        with ui.card().classes("w-full"):
            ui.label("Installed leaves").classes("text-subtitle1")
            state["installed_container"] = ui.column().classes("w-full gap-2")
            _render_installed(state)


def _reload_candidates(state: dict[str, Any], proj_path: str) -> None:
    container = state["candidates_container"]
    container.clear()
    proj_path = (proj_path or "").strip()
    if not proj_path:
        with container:
            ui.label("(enter a project directory)").classes("text-grey")
        return
    proj = Path(proj_path).expanduser()
    if not proj.is_dir():
        with container:
            ui.label(f"not a directory: {proj}").classes("text-negative")
        return
    state["source_project"] = str(proj)

    exp_dir = proj / ".experiments"
    sub_root = exp_dir / "subcircuits"
    if not sub_root.is_dir():
        with container:
            ui.label(f"no .experiments/subcircuits in {proj}").classes("text-warning")
        return

    candidates = _discover_candidates(proj)
    if not candidates:
        with container:
            ui.label(
                "no leaf-pinnable rounds found "
                "(run solve-subcircuits and pin a round first)"
            ).classes("text-warning")
        return

    state_path = _find_state_json(proj)
    state["state_data"] = _load_state_json(state_path) if state_path else None

    for cand in candidates:
        _render_candidate_form(state, cand)


def _discover_candidates(proj_dir: Path) -> list[dict[str, Any]]:
    """Return a list of {leaf_key, sheet_name, available_rounds, metadata}
    for every subcircuit dir under .experiments/subcircuits/ that has at
    least one complete round snapshot.
    """
    out: list[dict[str, Any]] = []
    sub_root = proj_dir / ".experiments" / "subcircuits"
    for leaf_dir in sorted(sub_root.iterdir()):
        if not leaf_dir.is_dir():
            continue
        if leaf_dir.name.startswith("subcircuit__"):
            continue
        rounds = pins_module.list_available_rounds(
            proj_dir / ".experiments", leaf_dir.name
        )
        if not rounds:
            continue
        # Read the latest metadata.json for the sheet name.
        meta_path = leaf_dir / "metadata.json"
        sheet_name = leaf_dir.name
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                sid = meta.get("subcircuit_id", {})
                sheet_name = sid.get("sheet_name", sheet_name)
            except Exception:
                pass
        out.append({
            "leaf_key": leaf_dir.name,
            "sheet_name": sheet_name,
            "available_rounds": rounds,
        })
    return out


def _find_state_json(proj_dir: Path) -> Path | None:
    """Locate a CircuitChat state.json next to the project."""
    for candidate in [
        proj_dir / f"{proj_dir.name}_circuitchat_state.json",
        proj_dir / "circuitchat_state.json",
        proj_dir / "state.json",
    ]:
        if candidate.exists():
            return candidate
    return None


def _load_state_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("could not load state json %s: %s", path, exc)
        return None


def _render_candidate_form(state: dict[str, Any], cand: dict[str, Any]) -> None:
    """Promote form for one candidate sheet."""
    container = state["candidates_container"]
    with container:
        with ui.card().classes("w-full"):
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(f"{cand['sheet_name']}").classes("text-subtitle2 font-mono")
                ui.label(
                    f"leaf_key={cand['leaf_key'][:18]}... "
                    f"rounds={cand['available_rounds']}"
                ).classes("text-caption font-mono text-grey-7")

            round_select = ui.select(
                cand["available_rounds"],
                label="Round",
                value=cand["available_rounds"][-1],
            ).classes("w-40")
            name_input = ui.input(
                label="Leaf name (kebab-case)",
                value=cand["sheet_name"].lower().replace("_", "-"),
            ).classes("w-full")
            version_input = ui.input(label="Version", value="0.1.0").classes("w-40")
            desc_input = ui.textarea(
                label="Description",
                placeholder="What it does, key parts, important parameters.",
            ).classes("w-full")
            tags_input = ui.input(
                label="Tags (comma-separated)", value=""
            ).classes("w-full")
            watch_input = ui.textarea(label="Watch out for (optional)").classes("w-full")

            confirm_btn = ui.button(
                "Promote",
                icon="upload",
                on_click=lambda: _do_promote(
                    state, cand, round_select.value, name_input.value,
                    version_input.value, desc_input.value, tags_input.value,
                    watch_input.value,
                ),
            )
            confirm_btn.props("color=primary")


def _do_promote(
    state: dict[str, Any],
    cand: dict[str, Any],
    round_val: int,
    name: str,
    version: str,
    description: str,
    tags_csv: str,
    watch_out_for: str,
) -> None:
    """Run the extractor with the form values."""
    if not name or not version or not description:
        ui.notify("name, version, description required", type="warning")
        return

    proj = Path(state["source_project"])
    state_data = state.get("state_data")
    bom_rows = _slice_bom_for_sheet(state_data, cand["sheet_name"])
    fragment = _slice_autoplacer_for_sheet(
        proj, cand["sheet_name"], [row.get("ref", "") for row in bom_rows]
    )

    tags = [t.strip() for t in tags_csv.split(",") if t.strip()]

    req = PromoteRequest(
        source_project_dir=proj,
        source_project_stem=proj.name,
        source_sheet_name=cand["sheet_name"],
        source_sheet_stem=cand["sheet_name"],
        source_leaf_key=cand["leaf_key"],
        source_round=int(round_val),
        name=name.strip(),
        version=version.strip(),
        description=description.strip(),
        tags=tags,
        watch_out_for=watch_out_for.strip() or None,
        bom_rows=bom_rows,
        autoplacer_fragment=fragment,
    )

    try:
        target = extract_leaf(req, state["lib"].base_dir, render=True)
    except FileExistsError as exc:
        ui.notify(str(exc), type="warning")
        return
    except Exception as exc:
        logger.exception("promotion failed")
        ui.notify(f"promotion failed: {exc}", type="negative")
        return

    ui.notify(f"promoted to {target}", type="positive")
    _render_installed(state)


def _slice_bom_for_sheet(state_data: dict | None, sheet_name: str) -> list[dict[str, str]]:
    if not state_data:
        return []
    bom = state_data.get("bom") or {}
    parts = bom.get("parts") or []
    return [p for p in parts if p.get("sheet") == sheet_name]


def _slice_autoplacer_for_sheet(
    proj_dir: Path, sheet_name: str, refs_in_sheet: list[str]
) -> dict[str, Any]:
    """Slice the project's autoplacer JSON to keys/values referring to
    refs in this sheet.

    Best effort: scans for the project's ``_autoplacer.json`` and copies
    in only the entries whose keys or members are in ``refs_in_sheet``.
    """
    candidates = list(proj_dir.glob("*_autoplacer.json")) + [proj_dir / "autoplacer.json"]
    payload: dict[str, Any] = {}
    for c in candidates:
        if c.exists():
            try:
                payload = json.loads(c.read_text(encoding="utf-8"))
            except Exception:
                continue
            break
    if not payload:
        return {}
    refs = set(refs_in_sheet)

    out: dict[str, Any] = {}
    ic_groups = payload.get("ic_groups", {})
    if isinstance(ic_groups, dict):
        out_ic = {
            k: list(v) for k, v in ic_groups.items()
            if k in refs and isinstance(v, list)
        }
        if out_ic:
            out["ic_groups"] = out_ic
    group_labels = payload.get("group_labels", {})
    if isinstance(group_labels, dict):
        out_gl = {k: v for k, v in group_labels.items() if k in refs}
        if out_gl:
            out["group_labels"] = out_gl
    for list_key in ("thermal_refs", "signal_flow_order"):
        vals = payload.get(list_key, [])
        if isinstance(vals, list):
            sliced = [v for v in vals if v in refs]
            if sliced:
                out[list_key] = sliced
    component_zones = payload.get("component_zones", {})
    if isinstance(component_zones, dict):
        sliced = {k: v for k, v in component_zones.items() if k in refs}
        if sliced:
            out["component_zones"] = sliced
    return out


def _render_installed(state: dict[str, Any]) -> None:
    container = state["installed_container"]
    container.clear()
    lib = state["lib"]
    loaded, broken = lib.load_all()
    if not loaded and not broken:
        with container:
            ui.label("(no installed leaves yet)").classes("text-grey")
        return

    with container:
        for leaf in loaded:
            _render_installed_card(state, leaf)
        for b in broken:
            _render_broken_card(b)


def _render_installed_card(state: dict[str, Any], leaf: LoadedLeaf) -> None:
    m = leaf.manifest
    thumb = leaf.dir / "renders" / "thumbnail.png"
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full no-wrap"):
            if thumb.exists():
                ui.image(str(thumb)).classes("w-32 h-32")
            else:
                ui.label("(no thumbnail)").classes("text-grey w-32 h-32")
            with ui.column().classes("col-grow gap-1"):
                ui.label(f"{m.name}@{m.version}").classes("text-subtitle2")
                ui.label(m.description).classes("text-caption")
                if m.tags:
                    ui.label("tags: " + ", ".join(m.tags)).classes("text-caption text-grey-7")
                ui.label(
                    f"from {m.provenance.source_project_stem} / "
                    f"{m.provenance.source_sheet_name}, "
                    f"round {m.provenance.source_experiment_round}, "
                    f"{m.provenance.promoted_at}"
                ).classes("text-caption text-grey-7")
            ui.button(
                "Remove",
                icon="delete",
                color="negative",
                on_click=lambda leaf=leaf: _confirm_remove(state, leaf),
            )


def _render_broken_card(broken: BrokenLeaf) -> None:
    with ui.card().classes("w-full"):
        with ui.row().classes("w-full items-center gap-2"):
            ui.icon("error", color="negative")
            ui.label(broken.dir.name).classes("text-subtitle2 font-mono")
            ui.label(broken.reason).classes("text-caption text-negative")


def _confirm_remove(state: dict[str, Any], leaf: LoadedLeaf) -> None:
    with ui.dialog() as dialog, ui.card():
        ui.label(f"Remove {leaf.slug}?").classes("text-subtitle1")
        ui.label(
            "This will not affect projects that have already imported this "
            "leaf. Existing projects keep their imported copy in their own files."
        ).classes("text-caption text-grey-7")
        with ui.row().classes("w-full justify-end gap-2"):
            ui.button("Cancel", on_click=dialog.close).props("flat")

            def _do_remove() -> None:
                shutil.rmtree(leaf.dir)
                ui.notify(f"removed {leaf.slug}", type="positive")
                dialog.close()
                _render_installed(state)

            ui.button("Remove", color="negative", on_click=_do_remove)
    dialog.open()


__all__ = ["leaf_library_page"]
