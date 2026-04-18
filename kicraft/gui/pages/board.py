"""Board viewer page — static PCB renders with layer controls."""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from nicegui import ui

from ..state import get_state


def board_page():
    state = get_state()

    ui.label("Board Viewer").classes("text-2xl font-bold mb-4")

    # Find available PCB files
    pcb_files = sorted(state.project_root.glob("*.kicad_pcb"))
    best_pcb = state.experiments_dir / "best" / "best.kicad_pcb"
    if best_pcb.exists():
        pcb_files.insert(0, best_pcb)

    if not pcb_files:
        ui.label("No PCB files found").classes("text-gray-500 italic")
        return

    options = {str(p): p.name for p in pcb_files}
    selected_pcb = {"path": str(pcb_files[0])}

    # ── Layer views ──
    LAYER_VIEWS = {
        "Front Copper (F.Cu)": ["F.Cu", "Edge.Cuts"],
        "Back Copper (B.Cu)": ["B.Cu", "Edge.Cuts"],
        "Front Silkscreen": ["F.SilkS", "F.Fab", "Edge.Cuts"],
        "Both Copper": ["F.Cu", "B.Cu", "Edge.Cuts"],
        "Courtyard": ["F.CrtYd", "B.CrtYd", "Edge.Cuts"],
        "Full Front": ["F.Cu", "F.SilkS", "F.Fab", "F.Mask", "Edge.Cuts"],
    }

    active_view = {"name": "Full Front"}

    image_container = ui.column().classes(
        "w-full items-center justify-center min-h-96")

    def _render_board():
        """Render selected PCB with selected layers via kicad-cli."""
        pcb_path = selected_pcb["path"]
        view_name = active_view["name"]
        layers = LAYER_VIEWS.get(view_name, ["F.Cu", "Edge.Cuts"])

        image_container.clear()
        with image_container:
            ui.label("Rendering...").classes("text-gray-400 italic")

        try:
            with tempfile.NamedTemporaryFile(suffix=".svg", delete=False) as tf:
                svg_path = tf.name

            cmd = [
                "kicad-cli", "pcb", "export", "svg",
                "-o", svg_path,
                "--layers", ",".join(layers),
                "--page-size-mode", "2",  # board area only
                pcb_path,
            ]
            result = subprocess.run(cmd, capture_output=True, text=True,
                                    timeout=30)

            if result.returncode != 0:
                image_container.clear()
                with image_container:
                    ui.label(f"Render failed: {result.stderr}").classes(
                        "text-red-400")
                return

            # Convert SVG to PNG via ImageMagick if available
            png_path = svg_path.replace(".svg", ".png")
            try:
                subprocess.run(
                    ["magick", svg_path, "-resize", "1200x900",
                     "-background", "white", "-flatten", png_path],
                    capture_output=True, timeout=15,
                )
                display_path = png_path
            except (FileNotFoundError, subprocess.TimeoutExpired):
                display_path = svg_path

            image_container.clear()
            with image_container:
                if display_path.endswith(".svg"):
                    with open(display_path) as f:
                        svg_content = f.read()
                    ui.html(svg_content).classes("max-w-4xl")
                else:
                    ui.image(display_path).classes("max-w-4xl")

        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            image_container.clear()
            with image_container:
                ui.label(f"Error: {e}").classes("text-red-400")
                ui.label("Make sure kicad-cli is installed and on PATH"
                         ).classes("text-gray-500 text-sm")

    # ── Controls ──
    with ui.row().classes("w-full items-center gap-4 mb-4"):
        ui.select(
            options=options,
            value=selected_pcb["path"],
            label="PCB File",
            on_change=lambda e: (
                selected_pcb.update({"path": e.value}),
                _render_board(),
            ),
        ).classes("w-96")

        ui.select(
            options=list(LAYER_VIEWS.keys()),
            value=active_view["name"],
            label="Layer View",
            on_change=lambda e: (
                active_view.update({"name": e.value}),
                _render_board(),
            ),
        ).classes("w-64")

        ui.button("Render", icon="refresh", on_click=_render_board)

    # ── Component list ──
    with ui.expansion("Component List", icon="list", value=False
                      ).classes("w-full mt-4"):
        _component_list(selected_pcb)

    # Initial render
    _render_board()


def _component_list(selected_pcb: dict):
    """Show a simple component list by parsing kicad_pcb file references."""
    try:
        pcb_path = selected_pcb["path"]
        # Quick grep for footprint references
        import re
        with open(pcb_path) as f:
            text = f.read()

        # Parse footprint blocks — extract ref, value, position, layer
        pattern = re.compile(
            r'\(footprint\s+"[^"]*".*?'
            r'\(layer\s+"([^"]+)"\).*?'
            r'\(at\s+([\d.\-]+)\s+([\d.\-]+).*?\).*?'
            r'\(property\s+"Reference"\s+"([^"]+)".*?\).*?'
            r'\(property\s+"Value"\s+"([^"]+)".*?\)',
            re.DOTALL,
        )

        rows = []
        for m in pattern.finditer(text):
            layer, x, y, ref, value = m.groups()
            rows.append({
                "ref": ref,
                "value": value,
                "x_mm": float(x),
                "y_mm": float(y),
                "layer": layer,
            })

        if rows:
            rows.sort(key=lambda r: r["ref"])
            ui.aggrid({
                "columnDefs": [
                    {"field": "ref", "headerName": "Ref", "width": 80,
                     "sortable": True},
                    {"field": "value", "headerName": "Value", "width": 150,
                     "sortable": True},
                    {"field": "x_mm", "headerName": "X (mm)", "width": 90,
                     "valueFormatter": "x.value?.toFixed(2)"},
                    {"field": "y_mm", "headerName": "Y (mm)", "width": 90,
                     "valueFormatter": "x.value?.toFixed(2)"},
                    {"field": "layer", "headerName": "Layer", "width": 100,
                     "sortable": True},
                ],
                "rowData": rows,
                "domLayout": "autoHeight",
            }).classes("w-full")
        else:
            ui.label("Could not parse component data").classes(
                "text-gray-500 italic")
    except Exception as e:
        ui.label(f"Error reading PCB: {e}").classes("text-red-400")
