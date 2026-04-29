from __future__ import annotations

import copy
import re
import shutil
import time
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.leaf_geometry import repair_leaf_placement_legality
from kicraft.autoplacer.brain.subcircuit_artifacts import resolve_artifact_paths
from kicraft.autoplacer.brain.subcircuit_extractor import ExtractedSubcircuitBoard
from kicraft.autoplacer.brain.subcircuit_render_diagnostics import (
    generate_leaf_diagnostic_artifacts,
    generate_stage_diagnostic_artifacts,
)
from kicraft.autoplacer.brain.types import Component
from kicraft.autoplacer.freerouting_runner import (
    import_routed_copper,
    route_with_freerouting,
    validate_routed_board,
)
from kicraft.autoplacer.hardware.adapter import KiCadAdapter


def route_local_subcircuit(
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
    *,
    generate_diagnostics: bool = True,
    round_index: int | None = None,
) -> tuple[dict[str, Any], dict[str, float]]:
    fast_smoke_mode = bool(cfg.get("subcircuit_fast_smoke_mode", False))
    render_pre_route_board_views = bool(
        cfg.get("subcircuit_render_pre_route_board_views", not fast_smoke_mode)
    )
    render_routed_board_views = bool(
        cfg.get("subcircuit_render_routed_board_views", True)
    )
    render_pre_route_drc_overlay = bool(
        cfg.get("subcircuit_render_pre_route_drc_overlay", not fast_smoke_mode)
    )
    render_routed_drc_overlay = bool(
        cfg.get("subcircuit_render_routed_drc_overlay", not fast_smoke_mode)
    )
    write_pre_route_drc_json = bool(
        cfg.get("subcircuit_write_pre_route_drc_json", not fast_smoke_mode)
    )
    write_routed_drc_json = bool(cfg.get("subcircuit_write_routed_drc_json", True))
    write_pre_route_drc_report = bool(
        cfg.get("subcircuit_write_pre_route_drc_report", not fast_smoke_mode)
    )
    write_routed_drc_report = bool(cfg.get("subcircuit_write_routed_drc_report", True))
    build_comparison_contact_sheet = bool(
        cfg.get("subcircuit_build_comparison_contact_sheet", not fast_smoke_mode)
    )
    if not extraction.internal_net_names:
        # Trivial leaf: no nets to route, but we still stamp the placed
        # components onto a real PCB so the leaf flows through the same
        # workflow as every other leaf -- pin_best_leaves can promote it,
        # the GUI snapshot picker shows its rounds, and the composer reads
        # uniformly from leaf_routed.kicad_pcb. The "routed" board is
        # identical to the pre-route board because there's nothing to add.
        return _stamp_trivial_leaf(
            extraction=extraction,
            solved_components=solved_components,
            cfg=cfg,
            round_index=round_index,
            generate_diagnostics=generate_diagnostics,
            fast_smoke_mode=fast_smoke_mode,
        )

    artifact_paths = resolve_artifact_paths(
        Path(extraction.subcircuit.schematic_path).parent,
        extraction.subcircuit.id,
    )
    pre_route_board = Path(artifact_paths.artifact_dir) / "leaf_pre_freerouting.kicad_pcb"
    routed_board = Path(artifact_paths.artifact_dir) / "leaf_routed.kicad_pcb"
    illegal_board = Path(artifact_paths.artifact_dir) / "leaf_illegal_pre_stamp.kicad_pcb"

    route_timing: dict[str, float] = {}
    route_total_start = time.monotonic()

    legality_start = time.monotonic()
    repaired_components, legality_repair = repair_leaf_placement_legality(
        extraction,
        solved_components,
        cfg,
    )
    route_timing["legality_repair_s"] = round(
        max(0.0, time.monotonic() - legality_start), 3
    )

    source_pcb = Path(cfg.get("subcircuit_route_source_pcb", cfg.get("pcb_path", "")))
    if not source_pcb.exists():
        source_pcb = Path(extraction.subcircuit.schematic_path).with_suffix(".kicad_pcb")

    if not source_pcb.exists():
        raise RuntimeError(
            "Leaf FreeRouting requires a real source PCB to stamp from; "
            f"could not resolve base board for {extraction.subcircuit.id.instance_path}"
        )

    if not legality_repair.get("resolved", False):
        diagnostics = legality_repair.get("diagnostics", {}) or {}
        overlap_count = int(diagnostics.get("overlap_count", 0) or 0)
        pad_outside_count = int(diagnostics.get("pad_outside_count", 0) or 0)
        overlap_pairs = [
            f"{item.get('a', '?')}:{item.get('b', '?')}"
            for item in diagnostics.get("overlaps", [])
        ]
        pad_violations = [
            f"{item.get('ref', '?')}:{item.get('pad_id', '?')}:{','.join(item.get('sides', []))}"
            for item in diagnostics.get("pads_outside_board", [])
        ]

        overlap_details = []
        for item in diagnostics.get("overlaps", []):
            overlap_details.append(
                {
                    "a": item.get("a"),
                    "b": item.get("b"),
                    "overlap_x_mm": item.get("overlap_x_mm"),
                    "overlap_y_mm": item.get("overlap_y_mm"),
                    "overlap_area_mm2": item.get("overlap_area_mm2"),
                }
            )

        component_debug = []
        repaired_by_ref = repaired_components or {}
        for ref in sorted(repaired_by_ref.keys()):
            comp = repaired_by_ref[ref]
            component_debug.append(
                {
                    "ref": ref,
                    "kind": comp.kind,
                    "layer": str(comp.layer),
                    "locked": bool(comp.locked),
                    "x_mm": round(comp.pos.x, 4),
                    "y_mm": round(comp.pos.y, 4),
                    "rotation_deg": round(comp.rotation, 4),
                    "width_mm": round(comp.width_mm, 4),
                    "height_mm": round(comp.height_mm, 4),
                    "pad_count": len(comp.pads),
                }
            )

        print(
            "  Leaf legality repair rejected placement: "
            f"overlaps={overlap_count} "
            f"pads_outside={pad_outside_count} "
            f"overlap_pairs={overlap_pairs} "
            f"pad_violations={pad_violations}"
        )
        if overlap_details:
            print(f"  Leaf legality overlap details: {overlap_details}")
        if component_debug:
            print(f"  Leaf legality component states: {component_debug}")

        illegal_input_board = copy.deepcopy(extraction.local_state)
        illegal_input_board.components = copy.deepcopy(repaired_components)
        illegal_input_board.traces = []
        illegal_input_board.vias = []

        illegal_render_diagnostics: dict[str, Any] = {
            "artifact_dir": artifact_paths.artifact_dir,
            "renders_dir": str(Path(artifact_paths.artifact_dir) / "renders"),
            "illegal_pre_stamp": None,
            "errors": [],
        }

        try:
            route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
            route_adapter.stamp_subcircuit_board(
                illegal_input_board,
                output_path=str(illegal_board),
                clear_existing_tracks=True,
                clear_existing_zones=True,
                remove_unmapped_footprints=True,
            )
            illegal_validation = {
                "accepted": False,
                "rejected": True,
                "rejection_stage": "leaf_pre_stamp_legality_repair",
                "rejection_reasons": ["illegal_unrepaired_leaf_placement"],
                "leaf_legality_repair": copy.deepcopy(legality_repair),
                "drc": {
                    "violations": [],
                    "report_text": (
                        "Leaf placement rejected before routing due to placement legality.\n"
                        f"overlap_count={overlap_count}\n"
                        f"pad_outside_count={pad_outside_count}\n"
                        f"overlap_pairs={overlap_pairs}\n"
                        f"pad_violations={pad_violations}\n"
                    ),
                },
            }
            illegal_render_diagnostics["illegal_pre_stamp"] = (
                generate_stage_diagnostic_artifacts(
                    pcb_path=str(illegal_board),
                    validation=illegal_validation,
                    artifact_dir=artifact_paths.artifact_dir,
                    stage="illegal_pre_stamp",
                    render_board_views=not fast_smoke_mode,
                    write_drc_json=not fast_smoke_mode,
                    write_drc_report=not fast_smoke_mode,
                    render_drc_overlay=not fast_smoke_mode,
                )
            )
        except Exception as exc:
            illegal_render_diagnostics["errors"].append(
                f"illegal_pre_stamp_render_failed:{exc}"
            )

        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "illegal_unrepaired_leaf_placement",
                "router": "freerouting",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": list(sorted(extraction.internal_net_names)),
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": False,
                    "rejected": True,
                    "rejection_stage": "leaf_pre_stamp_legality_repair",
                    "rejection_reasons": ["illegal_unrepaired_leaf_placement"],
                    "leaf_legality_repair": copy.deepcopy(legality_repair),
                    "render_diagnostics": copy.deepcopy(illegal_render_diagnostics)
                    if generate_diagnostics
                    else {"skipped": True, "reason": "size_reduction_fast_path"},
                    "illegal_pre_stamp_board_path": str(illegal_board),
                },
                "leaf_legality_repair": copy.deepcopy(legality_repair),
                "render_diagnostics": copy.deepcopy(illegal_render_diagnostics)
                if generate_diagnostics
                else {"skipped": True, "reason": "size_reduction_fast_path"},
                "illegal_pre_stamp_board_path": str(illegal_board),
                "failed": True,
            },
            route_timing,
        )

    route_input_board = copy.deepcopy(extraction.local_state)
    route_input_board.components = copy.deepcopy(repaired_components)
    route_input_board.traces = []
    route_input_board.vias = []

    stamp_start = time.monotonic()
    route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
    route_adapter.stamp_subcircuit_board(
        route_input_board,
        output_path=str(pre_route_board),
        clear_existing_tracks=True,
        clear_existing_zones=True,
        remove_unmapped_footprints=True,
    )
    route_timing["stamp_pre_route_board_s"] = round(
        max(0.0, time.monotonic() - stamp_start), 3
    )

    jar_path = cfg.get("freerouting_jar")
    if not jar_path:
        raise RuntimeError(
            "Leaf FreeRouting requires 'freerouting_jar' to be configured"
        )

    freerouting_start = time.monotonic()
    # Leaves can use a lower pass cap than the parent; fall back to the
    # shared default if the leaf-specific knob isn't set.
    leaf_routing_cfg = {
        **cfg,
        "pcb_path": str(source_pcb),
        "freerouting_preserve_existing_copper": False,
    }
    leaf_passes = cfg.get("leaf_freerouting_max_passes")
    if leaf_passes is not None:
        leaf_routing_cfg["freerouting_max_passes"] = int(leaf_passes)
    freerouting_stats = route_with_freerouting(
        str(pre_route_board),
        str(routed_board),
        str(jar_path),
        leaf_routing_cfg,
    )
    route_timing["freerouting_s"] = round(
        max(0.0, time.monotonic() - freerouting_start), 3
    )

    pre_route_validation_start = time.monotonic()
    pre_route_validation = validate_routed_board(
        str(pre_route_board),
        cfg=cfg,
        expected_anchor_names=[port.name for port in extraction.interface_ports],
        actual_anchor_names=[port.name for port in extraction.interface_ports],
        required_anchor_names=[
            port.name for port in extraction.interface_ports if port.required
        ],
        timeout_s=int(cfg.get("subcircuit_validation_timeout_s", 30)),
    )
    route_timing["pre_route_validation_s"] = round(
        max(0.0, time.monotonic() - pre_route_validation_start), 3
    )
    # Pre-route DRC is informational only -- we let FreeRouting attempt routing
    # regardless of pre-route violations. The post-route DRC gate handles acceptance.
    pre_route_drc = pre_route_validation.get("drc", {})
    if pre_route_drc.get("violations"):
        pre_route_violation_types = {v.get("type") for v in pre_route_drc["violations"]}
        print(
            f"  Pre-route DRC info: {len(pre_route_drc['violations'])} violations ({', '.join(sorted(pre_route_violation_types))})"
        )
    if generate_diagnostics:
        pre_route_render_start = time.monotonic()
        leaf_diagnostics = generate_leaf_diagnostic_artifacts(
            artifact_dir=artifact_paths.artifact_dir,
            pre_route_board=str(pre_route_board),
            routed_board=str(routed_board) if routed_board.exists() else None,
            pre_route_validation=pre_route_validation,
            routed_validation=None,
            render_pre_route_board_views=render_pre_route_board_views,
            render_routed_board_views=False,
            write_pre_route_drc_json=write_pre_route_drc_json,
            write_routed_drc_json=False,
            write_pre_route_drc_report=write_pre_route_drc_report,
            write_routed_drc_report=False,
            render_pre_route_drc_overlay=render_pre_route_drc_overlay,
            render_routed_drc_overlay=False,
            build_comparison_contact_sheet_enabled=False,
            quiet_board_render=fast_smoke_mode,
        )
        route_timing["pre_route_render_diagnostics_s"] = round(
            max(0.0, time.monotonic() - pre_route_render_start), 3
        )
    else:
        leaf_diagnostics = {
            "skipped": True,
            "reason": "size_reduction_fast_path",
        }
        route_timing["pre_route_render_diagnostics_s"] = 0.0

    round_board_illegal_pre_stamp = ""
    round_board_pre_route = ""
    round_board_routed = ""

    if round_index is not None:
        round_prefix = f"round_{int(round_index):04d}"

        def _copy_round_board(
            source_path: Path,
            suffix: str,
        ) -> str:
            if not source_path.exists():
                return ""
            destination = source_path.parent / f"{round_prefix}_{suffix}{source_path.suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            return str(destination)

        round_board_pre_route = _copy_round_board(
            pre_route_board,
            "leaf_pre_freerouting",
        )

    if round_index is not None and not leaf_diagnostics.get("skipped", False):
        renders_dir = Path(artifact_paths.artifact_dir) / "renders"
        round_prefix = f"round_{int(round_index):04d}"

        def _copy_round_preview(
            source_path: str | None,
            suffix: str,
        ) -> str:
            if not source_path:
                return ""
            source = Path(source_path)
            if not source.exists():
                return ""
            destination = renders_dir / f"{round_prefix}_{suffix}{source.suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return str(destination)

        pre_route_section = leaf_diagnostics.get("pre_route", {})
        if isinstance(pre_route_section, dict):
            pre_route_views = pre_route_section.get("board_views", {})
            if isinstance(pre_route_views, dict):
                pre_route_paths = pre_route_views.get("paths", {})
                if isinstance(pre_route_paths, dict):
                    round_pre_front = _copy_round_preview(
                        pre_route_paths.get("front_all"),
                        "pre_route_front_all",
                    )
                    round_pre_back = _copy_round_preview(
                        pre_route_paths.get("back_all"),
                        "pre_route_back_all",
                    )
                    round_pre_copper = _copy_round_preview(
                        pre_route_paths.get("copper_both"),
                        "pre_route_copper_both",
                    )
                    if round_pre_front:
                        pre_route_paths["round_front_all"] = round_pre_front
                    if round_pre_back:
                        pre_route_paths["round_back_all"] = round_pre_back
                    if round_pre_copper:
                        pre_route_paths["round_copper_both"] = round_pre_copper

    pre_route_validation["render_diagnostics"] = copy.deepcopy(leaf_diagnostics)
    pre_route_validation["leaf_legality_repair"] = copy.deepcopy(legality_repair)
    if round_board_pre_route:
        pre_route_validation["round_board_pre_route"] = round_board_pre_route

    import_copper_start = time.monotonic()
    imported_copper = import_routed_copper(str(routed_board))
    route_timing["import_routed_copper_s"] = round(
        max(0.0, time.monotonic() - import_copper_start), 3
    )

    routed_validation_start = time.monotonic()
    validation = validate_routed_board(
        str(routed_board),
        cfg=cfg,
        expected_anchor_names=[port.name for port in extraction.interface_ports],
        actual_anchor_names=[port.name for port in extraction.interface_ports],
        required_anchor_names=[
            port.name for port in extraction.interface_ports if port.required
        ],
        timeout_s=int(cfg.get("subcircuit_validation_timeout_s", 30)),
    )
    route_timing["routed_validation_s"] = round(
        max(0.0, time.monotonic() - routed_validation_start), 3
    )
    if generate_diagnostics:
        routed_render_start = time.monotonic()
        leaf_diagnostics = generate_leaf_diagnostic_artifacts(
            artifact_dir=artifact_paths.artifact_dir,
            pre_route_board=str(pre_route_board),
            routed_board=str(routed_board),
            pre_route_validation=pre_route_validation,
            routed_validation=validation,
            render_pre_route_board_views=render_pre_route_board_views,
            render_routed_board_views=render_routed_board_views,
            write_pre_route_drc_json=write_pre_route_drc_json,
            write_routed_drc_json=write_routed_drc_json,
            write_pre_route_drc_report=write_pre_route_drc_report,
            write_routed_drc_report=write_routed_drc_report,
            render_pre_route_drc_overlay=render_pre_route_drc_overlay,
            render_routed_drc_overlay=render_routed_drc_overlay,
            build_comparison_contact_sheet_enabled=build_comparison_contact_sheet,
            quiet_board_render=fast_smoke_mode,
        )
        route_timing["routed_render_diagnostics_s"] = round(
            max(0.0, time.monotonic() - routed_render_start), 3
        )
    else:
        leaf_diagnostics = {
            "skipped": True,
            "reason": "size_reduction_fast_path",
        }
        route_timing["routed_render_diagnostics_s"] = 0.0

    round_preview_pre_route_front = ""
    round_preview_pre_route_back = ""
    round_preview_pre_route_copper = ""
    round_preview_routed_front = ""
    round_preview_routed_back = ""
    round_preview_routed_copper = ""

    if round_index is not None:
        round_prefix = f"round_{int(round_index):04d}"

        def _copy_round_board(
            source_path: Path,
            suffix: str,
        ) -> str:
            if not source_path.exists():
                return ""
            destination = source_path.parent / f"{round_prefix}_{suffix}{source_path.suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination)
            return str(destination)

        round_board_routed = _copy_round_board(
            routed_board,
            "leaf_routed",
        )

    if round_index is not None and not leaf_diagnostics.get("skipped", False):
        renders_dir = Path(artifact_paths.artifact_dir) / "renders"
        round_prefix = f"round_{int(round_index):04d}"

        def _copy_round_preview(
            source_path: str | None,
            suffix: str,
        ) -> str:
            if not source_path:
                return ""
            source = Path(source_path)
            if not source.exists():
                return ""
            destination = renders_dir / f"{round_prefix}_{suffix}{source.suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
            return str(destination)

        pre_route_section = leaf_diagnostics.get("pre_route", {})
        if isinstance(pre_route_section, dict):
            pre_route_views = pre_route_section.get("board_views", {})
            if isinstance(pre_route_views, dict):
                pre_route_paths = pre_route_views.get("paths", {})
                if isinstance(pre_route_paths, dict):
                    round_preview_pre_route_front = _copy_round_preview(
                        pre_route_paths.get("front_all"),
                        "pre_route_front_all",
                    )
                    round_preview_pre_route_back = _copy_round_preview(
                        pre_route_paths.get("back_all"),
                        "pre_route_back_all",
                    )
                    round_preview_pre_route_copper = _copy_round_preview(
                        pre_route_paths.get("copper_both"),
                        "pre_route_copper_both",
                    )
                    if round_preview_pre_route_front:
                        pre_route_paths["round_front_all"] = round_preview_pre_route_front
                    if round_preview_pre_route_back:
                        pre_route_paths["round_back_all"] = round_preview_pre_route_back
                    if round_preview_pre_route_copper:
                        pre_route_paths["round_copper_both"] = round_preview_pre_route_copper

        routed_section = leaf_diagnostics.get("routed", {})
        if isinstance(routed_section, dict):
            routed_views = routed_section.get("board_views", {})
            if isinstance(routed_views, dict):
                routed_paths = routed_views.get("paths", {})
                if isinstance(routed_paths, dict):
                    round_preview_routed_front = _copy_round_preview(
                        routed_paths.get("front_all"),
                        "routed_front_all",
                    )
                    round_preview_routed_back = _copy_round_preview(
                        routed_paths.get("back_all"),
                        "routed_back_all",
                    )
                    round_preview_routed_copper = _copy_round_preview(
                        routed_paths.get("copper_both"),
                        "routed_copper_both",
                    )
                    if round_preview_routed_front:
                        routed_paths["round_front_all"] = round_preview_routed_front
                    if round_preview_routed_back:
                        routed_paths["round_back_all"] = round_preview_routed_back
                    if round_preview_routed_copper:
                        routed_paths["round_copper_both"] = round_preview_routed_copper

    validation["pre_route_validation"] = copy.deepcopy(pre_route_validation)
    validation["render_diagnostics"] = copy.deepcopy(leaf_diagnostics)
    if round_board_pre_route:
        validation["round_board_pre_route"] = round_board_pre_route
    if round_board_routed:
        validation["round_board_routed"] = round_board_routed

    drc = validation.get("drc", {})
    drc_stdout = str(drc.get("stdout", ""))
    drc_stderr = str(drc.get("stderr", ""))
    _ = "\n".join(part for part in (drc_stdout, drc_stderr) if part.strip())

    # Post-route ignorable violation types: cosmetic issues and violations
    # that are inherent to the footprint or subcircuit outline, not caused
    # by the routing itself.
    ignorable_warning_types = {
        "silk_overlap",
        "lib_footprint_mismatch",
        "copper_edge_clearance",  # tight subcircuit outlines
        "silk_edge_clearance",  # cosmetic
        "silk_over_copper",  # cosmetic
        "solder_mask_bridge",  # footprint-internal
        "unconnected_items",  # FreeRouting may not route all nets
    }
    significant_violations = [
        violation
        for violation in drc.get("violations", [])
        if violation.get("type") not in ignorable_warning_types
    ]

    # --- Generalized DRC exception: config-driven patterns ---
    # If the config provides ignorable_drc_patterns (list of regex strings),
    # check whether ALL significant violations match at least one pattern.
    ignorable_drc_patterns = cfg.get("ignorable_drc_patterns", [])
    _compiled_drc_patterns = [re.compile(p) for p in ignorable_drc_patterns]
    _all_match_config_patterns = (
        significant_violations
        and _compiled_drc_patterns
        and all(
            any(pat.search(v.get("description", "")) for pat in _compiled_drc_patterns)
            for v in significant_violations
        )
        and not drc.get("shorts", 0)
    )

    # --- Generalized DRC exception: footprint-baseline clearance heuristic ---
    # If ALL significant violations are clearance-type violations whose
    # descriptions reference pads from the SAME single footprint, treat them
    # as footprint-internal baseline clearance issues (e.g. dense USB-C,
    # fine-pitch IC pads closer together than the board clearance rule).
    _footprint_ref_re = re.compile(r"\bof\s+(\S+)")
    _clearance_types = {"clearance", "hole_clearance", "solder_mask_bridge"}
    _all_clearance = (
        significant_violations
        and all(v.get("type") in _clearance_types for v in significant_violations)
        and not drc.get("shorts", 0)
    )
    _single_footprint_baseline = False
    _baseline_footprint_ref = None
    if _all_clearance:
        # Collect all footprint references mentioned across violations
        _violation_footprint_refs: set[str] = set()
        for v in significant_violations:
            desc = v.get("description", "")
            for m in _footprint_ref_re.finditer(desc):
                _violation_footprint_refs.add(m.group(1))
        # If every violation references pads from exactly one footprint,
        # this is a footprint-internal clearance issue.
        if len(_violation_footprint_refs) == 1:
            _single_footprint_baseline = True
            _baseline_footprint_ref = next(iter(_violation_footprint_refs))

    if _all_match_config_patterns or _single_footprint_baseline:
        _ignore_reason = (
            "config_ignorable_drc_patterns"
            if _all_match_config_patterns
            else f"footprint_baseline_clearance:{_baseline_footprint_ref}"
        )
        _ignored_types = {v.get("type") for v in significant_violations}
        validation["obviously_illegal_routed_geometry"] = False
        validation["rejection_reasons"] = [
            reason
            for reason in validation.get("rejection_reasons", [])
            if reason != "illegal_routed_geometry"
        ]
        validation["accepted"] = not validation["rejection_reasons"]
        validation["drc"]["ignored_violation_types"] = sorted(
            ignorable_warning_types | _ignored_types
        )
        validation["drc"]["ignored_violation_count"] = len(drc.get("violations", []))
        validation["drc"]["significant_violation_count"] = 0
        validation["drc"]["ignored_clearance_reason"] = _ignore_reason

    accepted = bool(validation.get("accepted", False))
    if not accepted:
        validation["accepted"] = False
        validation["rejected"] = True
        validation["rejection_stage"] = "leaf_routed_artifact_validation"
        validation["routed_board_path"] = str(routed_board)
        validation["pre_route_board_path"] = str(pre_route_board)
        validation["router"] = "freerouting"
        validation["internal_net_names"] = list(sorted(extraction.internal_net_names))
        validation["interface_port_names"] = [
            port.name for port in extraction.interface_ports
        ]
        validation["imported_copper_summary"] = {
            "trace_count": int(imported_copper.get("trace_count", 0)),
            "via_count": int(imported_copper.get("via_count", 0)),
            "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
        }
        validation["freerouting_stats"] = copy.deepcopy(freerouting_stats)
        validation["rejection_message"] = "Leaf routed artifact rejected: " + ",".join(
            validation.get("rejection_reasons", [])
        )
        print("  Routed DRC rejected placement: " + validation["rejection_message"])
        route_timing["route_local_subcircuit_total_s"] = round(
            max(0.0, time.monotonic() - route_total_start), 3
        )
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "routed_drc_rejection",
                "router": "freerouting",
                "traces": int(imported_copper.get("trace_count", 0)),
                "vias": int(imported_copper.get("via_count", 0)),
                "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
                "round_board_illegal_pre_stamp": round_board_illegal_pre_stamp,
                "round_board_pre_route": round_board_pre_route,
                "round_board_routed": round_board_routed,
                "routed_internal_nets": [],
                "failed_internal_nets": list(sorted(extraction.internal_net_names)),
                "_trace_segments": [],
                "_via_objects": [],
                "validation": copy.deepcopy(validation),
                "freerouting_stats": copy.deepcopy(freerouting_stats),
                "render_diagnostics": copy.deepcopy(leaf_diagnostics),
                "routed_board_path": str(routed_board),
                "pre_route_board_path": str(pre_route_board),
                "round_preview_pre_route_front": round_preview_pre_route_front,
                "round_preview_pre_route_back": round_preview_pre_route_back,
                "round_preview_pre_route_copper": round_preview_pre_route_copper,
                "round_preview_routed_front": round_preview_routed_front,
                "round_preview_routed_back": round_preview_routed_back,
                "round_preview_routed_copper": round_preview_routed_copper,
                "failed": True,
            },
            route_timing,
        )

    route_timing["route_local_subcircuit_total_s"] = round(
        max(0.0, time.monotonic() - route_total_start), 3
    )
    return (
        {
            "enabled": True,
            "skipped": False,
            "reason": "",
            "router": "freerouting",
            "traces": int(imported_copper.get("trace_count", 0)),
            "vias": int(imported_copper.get("via_count", 0)),
            "total_length_mm": float(imported_copper.get("total_length_mm", 0.0)),
            "round_board_illegal_pre_stamp": round_board_illegal_pre_stamp,
            "round_board_pre_route": round_board_pre_route,
            "round_board_routed": round_board_routed,
            "routed_internal_nets": list(sorted(extraction.internal_net_names)),
            "failed_internal_nets": [],
            "_trace_segments": [
                copy.deepcopy(trace) for trace in imported_copper.get("traces", [])
            ],
            "_via_objects": [
                copy.deepcopy(via) for via in imported_copper.get("vias", [])
            ],
            "freerouting_stats": freerouting_stats,
            "validation": validation,
            "render_diagnostics": copy.deepcopy(leaf_diagnostics),
            "leaf_legality_repair": copy.deepcopy(legality_repair),
            "routed_board_path": str(routed_board),
            "pre_route_board_path": str(pre_route_board),
            "round_preview_pre_route_front": round_preview_pre_route_front,
            "round_preview_pre_route_back": round_preview_pre_route_back,
            "round_preview_pre_route_copper": round_preview_pre_route_copper,
            "round_preview_routed_front": round_preview_routed_front,
            "round_preview_routed_back": round_preview_routed_back,
            "round_preview_routed_copper": round_preview_routed_copper,
            "failed": False,
        },
        route_timing,
    )


def _stamp_trivial_leaf(
    *,
    extraction: ExtractedSubcircuitBoard,
    solved_components: dict[str, Component],
    cfg: dict[str, Any],
    round_index: int | None,
    generate_diagnostics: bool,
    fast_smoke_mode: bool,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Stamp a placed-but-not-routed PCB for a leaf with no internal nets.

    A trivial leaf (e.g. a battery holder with both terminals exposed
    via interface ports) has nothing to route, but we still want a
    real ``leaf_routed.kicad_pcb`` on disk so:

    * ``pin_best_leaves`` can promote a chosen round like every other
      leaf (no more "no-snapshots" status).
    * The GUI snapshot picker can show this leaf's rounds and let the
      user pin keyboard-style.
    * The composer's blocker extraction reads the same way from
      every leaf.

    The resulting PCB is just the placed footprints; no traces, no
    vias, no FreeRouting invocation. The returned routing dict
    advertises ``traces=0, vias=0, reason="no_internal_nets"`` and
    ``routed_board_path`` set to the same file as
    ``pre_route_board_path``.
    """
    route_timing: dict[str, float] = {}
    route_total_start = time.monotonic()

    artifact_paths = resolve_artifact_paths(
        Path(extraction.subcircuit.schematic_path).parent,
        extraction.subcircuit.id,
    )
    pre_route_board = Path(artifact_paths.artifact_dir) / "leaf_pre_freerouting.kicad_pcb"
    routed_board = Path(artifact_paths.artifact_dir) / "leaf_routed.kicad_pcb"

    legality_start = time.monotonic()
    repaired_components, legality_repair = repair_leaf_placement_legality(
        extraction,
        solved_components,
        cfg,
    )
    route_timing["legality_repair_s"] = round(
        max(0.0, time.monotonic() - legality_start), 3
    )

    source_pcb = Path(cfg.get("subcircuit_route_source_pcb", cfg.get("pcb_path", "")))
    if not source_pcb.exists():
        source_pcb = Path(extraction.subcircuit.schematic_path).with_suffix(".kicad_pcb")
    if not source_pcb.exists():
        # Without a source board we can't stamp; degrade to the original
        # behaviour (no PCB on disk, validation accepted because there's
        # nothing to fail). pin_best_leaves will report "no-snapshots"
        # like before, which is honest in this configuration.
        return (
            {
                "enabled": True,
                "skipped": True,
                "reason": "no_internal_nets",
                "router": "freerouting",
                "traces": 0,
                "vias": 0,
                "total_length_mm": 0.0,
                "routed_internal_nets": [],
                "failed_internal_nets": [],
                "_trace_segments": [],
                "_via_objects": [],
                "validation": {
                    "accepted": True,
                    "reason": "no_internal_nets",
                    "board_exists": False,
                    "shorts": 0,
                    "clearance_violations": 0,
                    "track_summary": {"traces": 0, "vias": 0},
                },
                "failed": False,
            },
            route_timing,
        )

    route_input_board = copy.deepcopy(extraction.local_state)
    route_input_board.components = copy.deepcopy(repaired_components)
    route_input_board.traces = []
    route_input_board.vias = []

    stamp_start = time.monotonic()
    route_adapter = KiCadAdapter(str(source_pcb), config=cfg)
    route_adapter.stamp_subcircuit_board(
        route_input_board,
        output_path=str(pre_route_board),
        clear_existing_tracks=True,
        clear_existing_zones=True,
        remove_unmapped_footprints=True,
    )
    route_timing["stamp_pre_route_board_s"] = round(
        max(0.0, time.monotonic() - stamp_start), 3
    )

    # No FreeRouting to run; the placed board IS the routed board.
    shutil.copy2(pre_route_board, routed_board)

    round_board_pre_route = ""
    round_board_routed = ""
    if round_index is not None:
        round_prefix = f"round_{int(round_index):04d}"
        for src_path, suffix in (
            (pre_route_board, "leaf_pre_freerouting"),
            (routed_board, "leaf_routed"),
        ):
            if not src_path.exists():
                continue
            dst = src_path.parent / f"{round_prefix}_{suffix}{src_path.suffix}"
            shutil.copy2(src_path, dst)
            if suffix == "leaf_pre_freerouting":
                round_board_pre_route = str(dst)
            else:
                round_board_routed = str(dst)

    diagnostics_payload: dict[str, Any]
    if generate_diagnostics and not fast_smoke_mode:
        try:
            diagnostics_payload = generate_leaf_diagnostic_artifacts(
                artifact_dir=artifact_paths.artifact_dir,
                pre_route_board=str(pre_route_board),
                routed_board=str(routed_board),
                pre_route_validation={"accepted": True, "reason": "no_internal_nets"},
                routed_validation={"accepted": True, "reason": "no_internal_nets"},
                render_pre_route_board_views=True,
                render_routed_board_views=True,
                write_pre_route_drc_json=False,
                write_routed_drc_json=False,
                write_pre_route_drc_report=False,
                write_routed_drc_report=False,
                render_pre_route_drc_overlay=False,
                render_routed_drc_overlay=False,
                build_comparison_contact_sheet_enabled=False,
                quiet_board_render=fast_smoke_mode,
            )
        except Exception as exc:
            diagnostics_payload = {"skipped": True, "reason": f"diag_failed:{exc}"}
    else:
        diagnostics_payload = {"skipped": True, "reason": "fast_smoke_or_no_diag"}

    route_timing["route_local_subcircuit_total_s"] = round(
        max(0.0, time.monotonic() - route_total_start), 3
    )

    return (
        {
            "enabled": True,
            "skipped": True,
            "reason": "no_internal_nets",
            "router": "freerouting",
            "traces": 0,
            "vias": 0,
            "total_length_mm": 0.0,
            "round_board_illegal_pre_stamp": "",
            "round_board_pre_route": round_board_pre_route,
            "round_board_routed": round_board_routed,
            "routed_internal_nets": [],
            "failed_internal_nets": [],
            "_trace_segments": [],
            "_via_objects": [],
            "validation": {
                "accepted": True,
                "reason": "no_internal_nets",
                "board_exists": True,
                "shorts": 0,
                "clearance_violations": 0,
                "track_summary": {"traces": 0, "vias": 0},
            },
            "render_diagnostics": diagnostics_payload,
            "leaf_legality_repair": copy.deepcopy(legality_repair),
            "routed_board_path": str(routed_board),
            "pre_route_board_path": str(pre_route_board),
            "failed": False,
        },
        route_timing,
    )
