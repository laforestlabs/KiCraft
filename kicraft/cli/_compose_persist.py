"""Persist a routed parent composition to the project's experiment tree.

Split out of ``compose_subcircuits.py`` (Lever 2.5); re-exported there.
"""
from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

from kicraft.autoplacer.brain.copper_accounting import verify_copper_preservation
from pathlib import Path

if TYPE_CHECKING:
    from kicraft.cli._compose_state import ParentCompositionState


def _persist_parent_artifact(
    state: ParentCompositionState,
    routing_result: dict[str, Any],
    project_dir: Path,
    cfg: dict[str, Any],
) -> str:
    """Persist a parent-level solved layout artifact.

    1. Build a SubCircuitLayout from the composition's board_state
       with routed copper from the routing result
    2. Build and save the solved layout artifact payload
    3. Save metadata and debug payloads
    4. Return the artifact directory path
    """
    from kicraft.autoplacer.brain.subcircuit_artifacts import (
        build_solved_layout_artifact,
        resolve_artifact_paths,
        save_solved_layout_artifact,
    )
    from kicraft.autoplacer.brain.types import SubCircuitLayout

    composition = state.composition
    if composition is None:
        raise RuntimeError("ParentCompositionState has no composition object")

    parent_id = composition.hierarchy_state.subcircuit.id
    parent_def = composition.hierarchy_state.subcircuit

    # Use routed copper (all traces: child + new parent interconnect)
    all_traces = routing_result.get("_trace_segments", [])
    all_vias = routing_result.get("_via_objects", [])

    # Fall back to composition board_state copper if routing returned nothing
    if not all_traces and not all_vias:
        all_traces = list(composition.board_state.traces)
        all_vias = list(composition.board_state.vias)

    # Compute bounding box from the composition's board outline
    tl, br = composition.board_state.board_outline
    width = max(0.0, br.x - tl.x)
    height = max(0.0, br.y - tl.y)

    layout = SubCircuitLayout(
        subcircuit_id=parent_id,
        components=dict(composition.board_state.components),
        traces=list(all_traces),
        vias=list(all_vias),
        bounding_box=(width, height),
        ports=list(parent_def.ports),
        interface_anchors=[],
        score=state.score_total,
        frozen=True,
    )

    # Build notes for the artifact
    # Use copper manifest for accurate child-only counts (state.trace_count
    # includes parent interconnect traces, which inflates the expectation).
    if state.copper_manifest is not None:
        expected_child_trace_count = state.copper_manifest.total_child_traces
        expected_child_via_count = state.copper_manifest.total_child_vias
    else:
        expected_child_trace_count = int(state.trace_count)
        expected_child_via_count = int(state.via_count)
    routed_total_trace_count = len(all_traces)
    routed_total_via_count = len(all_vias)

    # -- Real copper accounting via fingerprint matching --
    if state.copper_manifest is not None:
        copper_verification = verify_copper_preservation(
            manifest=state.copper_manifest,
            post_route_traces=all_traces,
            post_route_vias=all_vias,
        )
        preserved_child_trace_count = copper_verification["matched_child_traces"]
        preserved_child_via_count = copper_verification["matched_child_vias"]
        added_parent_trace_count = copper_verification["new_route_traces"]
        added_parent_via_count = copper_verification["new_route_vias"]
    else:
        # Fallback to count-based estimation when manifest unavailable
        copper_verification = None
        preserved_child_trace_count = min(
            expected_child_trace_count, routed_total_trace_count
        )
        preserved_child_via_count = min(
            expected_child_via_count, routed_total_via_count
        )
        added_parent_trace_count = max(
            0, routed_total_trace_count - preserved_child_trace_count
        )
        added_parent_via_count = max(
            0, routed_total_via_count - preserved_child_via_count
        )

    state.expected_preserved_child_trace_count = expected_child_trace_count
    state.expected_preserved_child_via_count = expected_child_via_count
    state.preserved_child_trace_count = preserved_child_trace_count
    state.preserved_child_via_count = preserved_child_via_count
    state.routed_total_trace_count = routed_total_trace_count
    state.routed_total_via_count = routed_total_via_count
    state.added_parent_trace_count = added_parent_trace_count
    state.added_parent_via_count = added_parent_via_count

    notes = [
        "parent_composition=true",
        f"child_count={len(state.entries)}",
        f"interconnect_nets={state.interconnect_net_count}",
        f"inferred_interconnects={state.inferred_interconnect_net_count}",
        f"expected_child_traces={expected_child_trace_count}",
        f"expected_child_vias={expected_child_via_count}",
        f"preserved_child_traces={preserved_child_trace_count}",
        f"preserved_child_vias={preserved_child_via_count}",
        f"routed_total_traces={routed_total_trace_count}",
        f"routed_total_vias={routed_total_via_count}",
        f"added_parent_traces={added_parent_trace_count}",
        f"added_parent_vias={added_parent_via_count}",
    ]
    if copper_verification:
        notes.append(f"copper_status={copper_verification['status']}")
        notes.append(
            f"copper_trace_preservation="
            f"{copper_verification['matched_child_traces']}/"
            f"{copper_verification['expected_child_traces']}"
        )
        notes.append(
            f"copper_via_preservation="
            f"{copper_verification['matched_child_vias']}/"
            f"{copper_verification['expected_child_vias']}"
        )
    validation = routing_result.get("validation", {})
    if validation:
        notes.append(f"validation_accepted={validation.get('accepted', False)}")

    payload = build_solved_layout_artifact(
        layout,
        project_dir=str(project_dir),
        solver_version="parent-compose-v1",
        notes=notes,
    )

    save_solved_layout_artifact(payload)

    # Save additional metadata alongside the solved layout
    artifact_paths = resolve_artifact_paths(str(project_dir), parent_id)
    artifact_dir = Path(artifact_paths.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    # Write a metadata.json for the parent artifact
    renders_dir = artifact_dir / "renders"
    parent_leaf_placed_board = artifact_dir / "parent_placed.kicad_pcb"
    parent_routed_board_path = artifact_dir / "parent_routed.kicad_pcb"
    parent_stamped_preview_path = renders_dir / "parent_stamped.png"
    parent_routed_preview_path = renders_dir / "parent_routed.png"

    # Provenance: which run produced this artifact (KICRAFT_RUN_ID is inherited
    # from the parent process; see kicraft/cli/artifact_paths.py). Lets the
    # promote freshness gate and `kicraft artifacts` answer "is this current?".
    from kicraft.cli.artifact_paths import current_run_id

    metadata_payload = {
        "schema_version": "parent-compose-v1",
        "run_id": current_run_id(),
        "generated_at": time.time(),
        "subcircuit_id": {
            "sheet_name": parent_id.sheet_name,
            "sheet_file": parent_id.sheet_file,
            "instance_path": parent_id.instance_path,
            "parent_instance_path": parent_id.parent_instance_path,
        },
        "parent_composition": True,
        "child_count": len(state.entries),
        "component_count": state.component_count,
        "trace_count": len(all_traces),
        "via_count": len(all_vias),
        "interconnect_net_count": state.interconnect_net_count,
        "inferred_interconnect_net_count": state.inferred_interconnect_net_count,
        "preserved_child_trace_count": state.preserved_child_trace_count,
        "preserved_child_via_count": state.preserved_child_via_count,
        "expected_preserved_child_trace_count": state.expected_preserved_child_trace_count,
        "expected_preserved_child_via_count": state.expected_preserved_child_via_count,
        "routed_total_trace_count": state.routed_total_trace_count,
        "routed_total_via_count": state.routed_total_via_count,
        "added_parent_trace_count": state.added_parent_trace_count,
        "added_parent_via_count": state.added_parent_via_count,
        "score_total": state.score_total,
        "copper_verification": copper_verification if copper_verification else {},
        "validation": validation,
        "artifact_paths": {
            "artifact_dir": str(artifact_dir),
            "renders_dir": str(renders_dir),
            "parent_placed_board": str(parent_leaf_placed_board)
            if parent_leaf_placed_board.exists()
            else "",
            "parent_routed_board": str(parent_routed_board_path)
            if parent_routed_board_path.exists()
            else "",
            "parent_stamped_preview": str(parent_stamped_preview_path)
            if parent_stamped_preview_path.exists()
            else "",
            "parent_routed_preview": str(parent_routed_preview_path)
            if parent_routed_preview_path.exists()
            else "",
        },
        "notes": notes,
    }
    metadata_path = artifact_dir / "metadata.json"
    # Atomic write: GUI and other downstream consumers may read this
    # while compose is still finishing other artifacts in the same dir.
    tmp_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    tmp_metadata.write_text(
        json.dumps(metadata_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    tmp_metadata.replace(metadata_path)

    # Write a debug.json with routing details
    debug_payload = {
        "schema_version": "parent-compose-v1",
        "parent_composition": True,
        "geometry_validation": dict(state.geometry_validation),
        "artifact_paths": {
            "artifact_dir": str(artifact_dir),
            "renders_dir": str(renders_dir),
            "parent_placed_board": str(parent_leaf_placed_board)
            if parent_leaf_placed_board.exists()
            else "",
            "parent_routed_board": str(parent_routed_board_path)
            if parent_routed_board_path.exists()
            else "",
            "parent_stamped_preview": str(parent_stamped_preview_path)
            if parent_stamped_preview_path.exists()
            else "",
            "parent_routed_preview": str(parent_routed_preview_path)
            if parent_routed_preview_path.exists()
            else "",
        },
        "routing_result": {
            "routed_board_path": routing_result.get("routed_board_path", ""),
            "trace_count": len(all_traces),
            "via_count": len(all_vias),
            "routing_stats": routing_result.get("routing_stats", {}),
            "preview_paths": {
                "parent_stamped_preview": str(parent_stamped_preview_path)
                if parent_stamped_preview_path.exists()
                else "",
                "parent_routed_preview": str(parent_routed_preview_path)
                if parent_routed_preview_path.exists()
                else "",
            },
            "board_paths": {
                "parent_placed_board": str(parent_leaf_placed_board)
                if parent_leaf_placed_board.exists()
                else "",
                "parent_routed_board": str(parent_routed_board_path)
                if parent_routed_board_path.exists()
                else "",
            },
            "copper_accounting": {
                "expected_preserved_child_trace_count": state.expected_preserved_child_trace_count,
                "expected_preserved_child_via_count": state.expected_preserved_child_via_count,
                "preserved_child_trace_count": state.preserved_child_trace_count,
                "preserved_child_via_count": state.preserved_child_via_count,
                "routed_total_trace_count": state.routed_total_trace_count,
                "routed_total_via_count": state.routed_total_via_count,
                "added_parent_trace_count": state.added_parent_trace_count,
                "added_parent_via_count": state.added_parent_via_count,
            },
            "copper_verification": copper_verification,
            "copper_manifest": state.copper_manifest.to_dict() if state.copper_manifest else None,
        },
        "validation": validation,
        "hierarchical_status": {
            "current_parent": state.parent_sheet_name,
            "current_node": state.parent_instance_path,
            "top_level_status": "accepted"
            if validation.get("accepted")
            else "rejected",
            "composition_status": "routed"
            if not routing_result.get("failed")
            else "failed",
            "preview_paths": {
                "parent_stamped_preview": str(parent_stamped_preview_path)
                if parent_stamped_preview_path.exists()
                else "",
                "parent_routed_preview": str(parent_routed_preview_path)
                if parent_routed_preview_path.exists()
                else "",
                "parent_placed_board": str(parent_leaf_placed_board)
                if parent_leaf_placed_board.exists()
                else "",
                "parent_routed_board": str(parent_routed_board_path)
                if parent_routed_board_path.exists()
                else "",
            },
            "leaf_workers": {
                "total": 0,
                "active": 0,
                "idle": 0,
                "queued": 0,
                "completed": len(state.entries),
            },
            "copper_accounting": {
                "expected_preserved_child_trace_count": state.expected_preserved_child_trace_count,
                "expected_preserved_child_via_count": state.expected_preserved_child_via_count,
                "preserved_child_trace_count": state.preserved_child_trace_count,
                "preserved_child_via_count": state.preserved_child_via_count,
                "routed_total_trace_count": state.routed_total_trace_count,
                "routed_total_via_count": state.routed_total_via_count,
                "added_parent_trace_count": state.added_parent_trace_count,
                "added_parent_via_count": state.added_parent_via_count,
            },
        },
        "composition_state": state.to_dict(),
    }
    debug_path = artifact_dir / "debug.json"
    # Atomic write: same rationale as metadata.json above.
    tmp_debug = debug_path.with_suffix(debug_path.suffix + ".tmp")
    tmp_debug.write_text(
        json.dumps(debug_payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    tmp_debug.replace(debug_path)

    return str(artifact_dir)
