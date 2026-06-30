"""Stamp a parent composition onto a pcbnew board (subprocess) + pre-route DRC.

Split out of ``compose_subcircuits.py`` (Lever 2.5); re-exported there.
"""
from __future__ import annotations

import shutil
import sys

from kicraft.cli._compose_validate import _fit_requested_shape
from kicraft.cli._compose_validate import _repair_parent_outline
from kicraft.cli._compose_validate import _validate_parent_geometry
from pathlib import Path


def _stamp_parent_board(
    state: ParentCompositionState,
    pcb_path: Path,
    project_dir: Path,
    cfg: dict[str, Any],
    output_pcb_path: Path | None = None,
) -> Path:
    """Stamp the parent composition onto a real .kicad_pcb file.

    Uses a subprocess to run pcbnew operations so the main process does not
    need pcbnew installed.  The subprocess:
    1. Loads the copied board
    2. Moves footprints to their composed positions
    3. Clears existing tracks/zones
    4. Recreates traces/vias from the merged child copper
    5. Rebuilds connectivity and saves

    If ``output_pcb_path`` is provided, the stamped board is written there
    instead of the canonical ``<artifact_dir>/parent_pre_freerouting.kicad_pcb``.
    The candidate-search loop uses this to stamp each trial to a distinct
    file under ``<artifact_dir>/_search/``.

    Returns the stamped board path.
    """
    import json as _json
    import os
    import tempfile

    from kicraft.autoplacer.brain.subcircuit_artifacts import slugify_subcircuit_id
    from kicraft.autoplacer.brain.types import Layer
    from kicraft.autoplacer.freerouting_runner import _run_pcbnew_script_file

    composition = state.composition
    if composition is None:
        raise RuntimeError("ParentCompositionState has no composition object")

    parent_id = composition.hierarchy_state.subcircuit.id
    slug = slugify_subcircuit_id(parent_id)
    artifact_dir = project_dir / ".experiments" / "subcircuits" / slug
    artifact_dir.mkdir(parents=True, exist_ok=True)

    if output_pcb_path is not None:
        output_pcb = Path(output_pcb_path)
        output_pcb.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_pcb = artifact_dir / "parent_pre_freerouting.kicad_pcb"
    shutil.copy2(str(pcb_path), str(output_pcb))

    # Serialize board state for the subprocess
    board_state = composition.board_state
    components_json = []
    for ref, comp in (board_state.components or {}).items():
        components_json.append(
            {
                "ref": ref,
                "x": comp.pos.x,
                "y": comp.pos.y,
                "rotation": comp.rotation,
                "layer": 0 if comp.layer == Layer.FRONT else 1,
            }
        )

    traces_json = []
    for trace in board_state.traces or []:
        traces_json.append(
            {
                "start_x": trace.start.x,
                "start_y": trace.start.y,
                "end_x": trace.end.x,
                "end_y": trace.end.y,
                "width": trace.width_mm,
                "layer": "F.Cu" if trace.layer == Layer.FRONT else "B.Cu",
                "net_name": trace.net or "",
            }
        )

    vias_json = []
    for via in board_state.vias or []:
        vias_json.append(
            {
                "x": via.pos.x,
                "y": via.pos.y,
                "size": via.size_mm,
                "drill": via.drill_mm,
                "net_name": via.net or "",
            }
        )

    silkscreen_json = []
    for elem in board_state.silkscreen or []:
        if elem.kind == "poly":
            silkscreen_json.append({
                "kind": "poly",
                "layer": elem.layer,
                "points": [{"x": p.x, "y": p.y} for p in elem.points],
                "stroke_width": elem.stroke_width,
            })
        elif elem.kind == "text":
            silkscreen_json.append({
                "kind": "text",
                "layer": elem.layer,
                "text": elem.text,
                "pos": {"x": elem.pos.x, "y": elem.pos.y},
                "font_height": elem.font_height,
                "font_width": elem.font_width,
                "font_thickness": elem.font_thickness,
            })

    # Grow the parent outline to enclose all placed geometry BEFORE deriving
    # Edge.Cuts and validating. The constraint-aware outline can snap smaller
    # than the placed-content bbox (edge-anchored sides snap to their anchor),
    # leaving a few footprints/pads/traces outside Edge.Cuts. That fails geometry
    # validation, which gates routing -- so every candidate goes un-routed and
    # the parent is rejected as illegal_routed_geometry. Repairing here (grow
    # only) makes geometry valid for every candidate, so the search judges
    # placements on quality (overlap/packing/net-distance) rather than on
    # overflowing a too-small outline, and FreeRouting gets a valid board.
    _repair_parent_outline(state)

    # Circumscribe a brief-requested non-rect shape around the grown AABB (auto
    # path; no-op for manual layouts and rectangular boards). Sets
    # state.manual_outline so the polyline stamp + shape-aware geometry
    # validation below consume the true shape.
    _fit_requested_shape(state)

    # Compute the board outline from the composition
    outline = board_state.board_outline
    outline_data = None
    if outline and len(outline) >= 2:
        outline_data = {
            "tl_x": outline[0].x,
            "tl_y": outline[0].y,
            "br_x": outline[1].x,
            "br_y": outline[1].y,
        }
        # Non-rect manual shapes stamp Edge.Cuts as the shape's closed
        # polyline instead of a 4-segment rectangle. Generated from the
        # board_state outline AABB (not the spec's own min/max) so the
        # stamped shape always brackets exactly the validated outline.
        if (
            state.manual_outline is not None
            and state.manual_outline.get("shape", "rect") != "rect"
        ):
            from kicraft.layout_editor.outline import OutlineSpec as _OutlineSpec

            _spec = _OutlineSpec.from_dict(
                {
                    **state.manual_outline,
                    "min": {"x": outline[0].x, "y": outline[0].y},
                    "max": {"x": outline[1].x, "y": outline[1].y},
                }
            )
            outline_data["polyline"] = [[p.x, p.y] for p in _spec.polyline()]

    geometry_validation = _validate_parent_geometry(state)
    if not geometry_validation.get("accepted", False):
        # Don't raise -- the caller wants to stamp + render even on
        # geometry rejection so the user has a diagnostic image showing
        # where components ended up. The outer code's geometry_accepted
        # flag still gates routing, so we won't attempt to route an
        # off-board layout. Just surface a warning so the failure is
        # visible in the log.
        print(
            "warning: parent composition geometry is invalid before "
            "stamping (continuing to stamp for diagnostic render): "
            f"outside_components={geometry_validation.get('outside_component_count', 0)} "
            f"outside_pads={geometry_validation.get('outside_pad_count', 0)} "
            f"outside_traces={geometry_validation.get('outside_trace_count', 0)} "
            f"outside_vias={geometry_validation.get('outside_via_count', 0)}",
            file=sys.stderr,
        )

    keepout_json = [
        {
            "tl_x": rect[0].x,
            "tl_y": rect[0].y,
            "br_x": rect[1].x,
            "br_y": rect[1].y,
        }
        for rect in (state.parent_local_keep_in_rects or [])
    ]

    synthesize_json = [dict(e) for e in (state.synthesized_footprints or [])]

    # Center the assembly on a standard A4 drawing sheet (297 x 210 mm)
    # so the PCB opens centered in the title block rather than crammed
    # against the top-left corner. The composer's native origin is the
    # parent search-space's top-left, which lands at (0, 0) on the
    # sheet and looks lopsided in the schematic / board editor.
    _PAGE_W_MM = float(cfg.get("parent_page_width_mm", 297.0))
    _PAGE_H_MM = float(cfg.get("parent_page_height_mm", 210.0))
    if outline_data:
        _board_w = outline_data["br_x"] - outline_data["tl_x"]
        _board_h = outline_data["br_y"] - outline_data["tl_y"]
        _dx = (_PAGE_W_MM - _board_w) / 2.0 - outline_data["tl_x"]
        _dy = (_PAGE_H_MM - _board_h) / 2.0 - outline_data["tl_y"]
        if abs(_dx) > 1e-6 or abs(_dy) > 1e-6:
            for _c in components_json:
                _c["x"] += _dx
                _c["y"] += _dy
            for _t in traces_json:
                _t["start_x"] += _dx
                _t["start_y"] += _dy
                _t["end_x"] += _dx
                _t["end_y"] += _dy
            for _v in vias_json:
                _v["x"] += _dx
                _v["y"] += _dy
            for _s in silkscreen_json:
                if _s.get("kind") == "poly":
                    for _pt in _s.get("points", []):
                        _pt["x"] += _dx
                        _pt["y"] += _dy
                elif _s.get("kind") == "text":
                    _s["pos"]["x"] += _dx
                    _s["pos"]["y"] += _dy
            for _k in keepout_json:
                _k["tl_x"] += _dx
                _k["tl_y"] += _dy
                _k["br_x"] += _dx
                _k["br_y"] += _dy
            for _sf in synthesize_json:
                _sf["x"] += _dx
                _sf["y"] += _dy
            outline_data["tl_x"] += _dx
            outline_data["tl_y"] += _dy
            outline_data["br_x"] += _dx
            outline_data["br_y"] += _dy
            for _pt in outline_data.get("polyline") or []:
                _pt[0] += _dx
                _pt[1] += _dy

    payload = {
        "pcb_path": str(output_pcb),
        "output_path": str(output_pcb),
        "components": components_json,
        "traces": traces_json,
        "vias": vias_json,
        "silkscreen": silkscreen_json,
        "outline": outline_data,
        "keepouts": keepout_json,
        "synthesize_footprints": synthesize_json,
    }

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json", prefix="stamp_parent_")
    try:
        with os.fdopen(tmp_fd, "w") as f:
            _json.dump(payload, f)

        _run_pcbnew_script_file(_PARENT_STAMP_SCRIPT_PATH, tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    # KiCad's board.Save() in the stamp subprocess emits a *default* sidecar
    # .kicad_pro (Default netclass 0.20 mm), dropping the project's real
    # netclasses (e.g. Power 0.30 mm). Overwrite it with the source project's
    # .kicad_pro so the stamped board carries the true netclass clearances and
    # patterns; otherwise FreeRouting routes power nets at the default clearance
    # and the promoted board fails DRC against the real Power rule
    # (illegal_routed_geometry). freerouting_runner._inject_netclass_clearances
    # then carries these into the DSN handed to FreeRouting.
    try:
        src_pro = Path(pcb_path).with_suffix(".kicad_pro")
        if not src_pro.is_file():
            src_pro = next(iter(sorted(project_dir.glob("*.kicad_pro"))), None)
        sibling_pro = output_pcb.with_suffix(".kicad_pro")
        if (
            src_pro
            and src_pro.is_file()
            and src_pro.resolve() != sibling_pro.resolve()
        ):
            shutil.copy2(str(src_pro), str(sibling_pro))
    except OSError:
        pass

    print(f"Parent board stamped to {output_pcb} (subprocess)")
    return output_pcb



# ---------------------------------------------------------------------------
# Self-contained pcbnew script executed in a subprocess by
# _stamp_parent_board(). Lifted to its own file (_parent_stamp_subprocess.py)
# so import-time errors fire when the file is parsed and so linters / IDEs
# see the pcbnew API calls.
# ---------------------------------------------------------------------------
_PARENT_STAMP_SCRIPT_PATH = str(
    Path(__file__).parent / "_parent_stamp_subprocess.py"
)
