#!/usr/bin/env python3
"""Compose solved subcircuits into a parent composition state.

This CLI is the first parent-composition entrypoint for the subcircuits
redesign. It loads solved leaf artifacts from `.experiments/subcircuits`,
instantiates them as rigid modules, applies translation/rotation transforms,
and emits a machine-readable composition snapshot.

Current scope:
- load canonical solved subcircuit artifacts
- instantiate rigid child modules
- apply translation + rotation transforms
- build a parent composition state summary
- emit JSON and optional saved composition snapshot
- support simple placement modes for initial composition experiments
- stamp composition onto a real .kicad_pcb file (--stamp)
- route parent interconnects via FreeRouting (--route)
- persist parent-level solved layout artifacts

This command does NOT yet:
- optimize parent placement
- recurse through non-leaf schematic hierarchy automatically

It is intended as a composition-side scaffold so later milestones can build:
- parent-level placement optimization
- interconnect routing
- recursive upward propagation
- final top-level board assembly
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from kicraft.autoplacer.brain.hierarchy_parser import parse_hierarchy


from kicraft.autoplacer.brain.copper_accounting import (
    CopperManifest,
    build_copper_manifest,
    verify_copper_preservation,
)
from kicraft.autoplacer.brain.parent_adapter import (
    artifact_to_component,
    attachment_constraints_to_zones,
    infer_interconnect_nets_pre_placement,
    placements_from_solved_state,
    synthetic_block_ref,
)
from kicraft.layout_editor.model import ManualLayout, load_manual_layout
from kicraft.autoplacer.brain.placement_solver import PlacementSolver
from kicraft.autoplacer.brain.subcircuit_composer import (
    AttachmentConstraint,
    ChildArtifactPlacement,
    DerivedAttachmentConstraints,
    ParentComposition,
    build_parent_composition,
    dominant_blocker_side,
    edge_anchor_target_coordinate,
    derive_attachment_constraints,
    child_layer_envelopes,
    can_overlap,
    can_overlap_sparse,
    constraint_aware_outline,
    extract_leaf_blocker_set,
)
from kicraft.autoplacer.brain import geometry
from kicraft.autoplacer.brain.types import BoardState
from kicraft.autoplacer.brain.subcircuit_extractor import extract_parent_local_components
from kicraft.autoplacer.brain.subcircuit_instances import (
    LoadedSubcircuitArtifact,
    artifact_debug_dict,
    artifact_summary,
    load_solved_artifacts,
    transform_loaded_artifact,
    transformed_debug_dict,
    transformed_summary,
)
from kicraft.autoplacer.brain.types import (
    Component,
    Point,
    SubCircuitDefinition,
    SubCircuitId,
    SubCircuitLayout,
    angles_close,
    edge_outward_angle,
    opening_board_angle,
)


# CompositionEntry / ParentCompositionState live in _compose_state (Lever 2.5
# split); re-exported so existing references + the external API keep resolving.
from kicraft.cli._compose_state import (  # noqa: E402
    CompositionEntry,
    ParentCompositionState,
)


def _emit_inspector_bundle(routed_pcb: Path) -> None:
    """Run the parent-PCB inspector and print bundle paths.

    Emits a structured JSON report, a markdown summary, and annotated
    PNGs that downstream callers (especially AI agents) can read to
    understand the layout. Failures are non-fatal -- the inspector is
    a diagnostic, not a gate.
    """
    if not routed_pcb.is_file():
        return
    try:
        from kicraft.cli.inspect_parent import collect, render_annotated_top, \
            render_stacking_heatmap, to_markdown
        out_dir = routed_pcb.parent / "inspect"
        out_dir.mkdir(parents=True, exist_ok=True)
        report = collect(routed_pcb)
        json_path = out_dir / "report.json"
        # Atomic writes: this bundle is auto-emitted after every parent
        # route, and downstream tools (GUI, agents) may read mid-write.
        tmp_json = json_path.with_suffix(json_path.suffix + ".tmp")
        tmp_json.write_text(
            json.dumps(report.to_dict(), indent=2), encoding="utf-8"
        )
        tmp_json.replace(json_path)
        pngs: dict[str, Path] = {}
        try:
            pngs["annotated_top"] = render_annotated_top(
                report, out_dir / "annotated_top.png"
            )
            pngs["stacking_heatmap"] = render_stacking_heatmap(
                report, out_dir / "stacking_heatmap.png"
            )
        except Exception as exc:
            print(f"inspect: render failed: {exc}", file=sys.stderr)
        md_path = out_dir / "summary.md"
        tmp_md = md_path.with_suffix(md_path.suffix + ".tmp")
        tmp_md.write_text(to_markdown(report, png_paths=pngs), encoding="utf-8")
        tmp_md.replace(md_path)
        print(f"inspect_summary    : {md_path}")
        print(f"inspect_json       : {json_path}")
        for label, p in pngs.items():
            print(f"inspect_{label:<10s}: {p}")
    except Exception as exc:
        print(f"inspect: failed: {exc}", file=sys.stderr)


def _discover_artifact_dirs(project_dir: Path) -> list[Path]:
    """Find solved subcircuit artifact directories under a project."""
    root = project_dir / ".experiments" / "subcircuits"
    if not root.exists():
        return []

    artifact_dirs: list[Path] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        metadata = child / "metadata.json"
        debug = child / "debug.json"
        solved_layout = child / "solved_layout.json"
        if metadata.exists() and debug.exists() and solved_layout.exists():
            artifact_dirs.append(child)
    return artifact_dirs


def _resolve_artifact_dirs(
    project_dir: Path | None,
    artifact_args: list[str],
) -> list[str | Path]:
    """Resolve artifact directories from CLI inputs."""
    resolved: list[str | Path] = []

    for artifact in artifact_args:
        path = Path(artifact).resolve()
        if path not in resolved:
            resolved.append(path)

    if project_dir is not None:
        for path in _discover_artifact_dirs(project_dir.resolve()):
            if path not in resolved:
                resolved.append(path)

    return resolved


def _filter_loaded_artifacts(loaded_artifacts, only: list[str]) -> list[Any]:
    """Filter loaded artifacts by sheet name, file name, or instance path."""
    if not only:
        return list(loaded_artifacts)

    only_set = {item.strip().lower() for item in only if item.strip()}
    filtered = []
    for artifact in loaded_artifacts:
        candidates = {
            artifact.layout.subcircuit_id.sheet_name.lower(),
            artifact.layout.subcircuit_id.sheet_file.lower(),
            artifact.layout.subcircuit_id.instance_path.lower(),
        }
        if candidates & only_set:
            filtered.append(artifact)
    return filtered


def _missing_child_artifacts(parent_definition, loaded_artifacts) -> list:
    """Expected child subcircuit ids that have no loaded solved artifact.

    A child whose solve FAILED produces no ``solved_layout.json`` artifact, so it
    is absent from ``loaded_artifacts``. Composing the parent without it would
    strand its components as loose parent-level parts (force/SA'd at the parent)
    -- a fallback that cannot place a failed leaf. The caller aborts loudly when
    this is non-empty. Empty ``parent_definition`` -> nothing to require."""
    if parent_definition is None:
        return []
    loaded_paths = {a.layout.subcircuit_id.instance_path for a in loaded_artifacts}
    return [
        cid for cid in parent_definition.child_ids
        if cid.instance_path not in loaded_paths
    ]


def _select_parent_definition(
    project_dir: Path | None,
    parent_selector: str | None,
) -> SubCircuitDefinition | None:
    """Resolve a real parent definition from schematic hierarchy."""
    if project_dir is None or not parent_selector:
        return None

    graph = parse_hierarchy(project_dir=project_dir.resolve())
    selector = parent_selector.strip().lower()
    if not selector:
        return None

    root_candidates = {
        graph.root.id.sheet_name.lower(),
        graph.root.id.sheet_file.lower(),
        graph.root.id.instance_path.lower(),
    }
    if selector in root_candidates:
        return graph.root.definition

    for node in graph.non_leaf_nodes():
        if node.id.instance_path == "/":
            continue
        candidates = {
            node.id.sheet_name.lower(),
            node.id.sheet_file.lower(),
            node.id.instance_path.lower(),
        }
        if selector in candidates:
            return node.definition

    raise ValueError(f"Unknown parent subcircuit: {parent_selector}")


def _filter_artifacts_for_parent(
    loaded_artifacts,
    parent_definition: SubCircuitDefinition | None,
) -> list[Any]:
    """Restrict artifacts to direct children of the selected parent."""
    if parent_definition is None:
        return list(loaded_artifacts)

    child_paths = {child_id.instance_path for child_id in parent_definition.child_ids}
    return [
        artifact
        for artifact in loaded_artifacts
        if artifact.layout.subcircuit_id.instance_path in child_paths
    ]


# Pure bbox/rect/envelope geometry helpers live in _compose_geometry (Lever 2.5
# split); imported here for internal use.
from kicraft.cli._compose_geometry import (  # noqa: E402
    _bbox_disjoint,
    _component_geometry_bbox,
    _rect_area,
    _rect_lists_disjoint,
    _shift_envelope,
    _shift_rect,
)


def _resolve_parent_local_allowlist(component_zones: dict[str, Any], loaded_artifacts) -> set[str]:
    child_refs = set()
    for artifact in loaded_artifacts:
        child_refs.update(artifact.layout.components.keys())
    return {
        ref
        for ref in component_zones.keys()
        if ref not in child_refs
    }


def _edge_bank_geometry(
    derived: DerivedAttachmentConstraints,
    widths: dict[int, float],
    heights: dict[int, float],
    spacing_mm: float,
) -> dict[str, Any]:
    """Per-edge bank extents for parent seed sizing.

    Children pinned to the LEFT/RIGHT edges stack VERTICALLY along that
    edge: the bank consumes *height* (Σ member heights + inter/end gaps) and
    its *depth* into the board is the widest single member.  TOP/BOTTOM banks
    are the transpose.  Opposing banks (left vs right, top vs bottom) sit on
    OPPOSITE sides with interior room between them, so their depths ADD across
    the board while their stack extents are independent (``max``, not ``sum``).

    This replaces the old floor that summed *every* edge child's width into a
    single horizontal row regardless of which edge pinned it -- the arithmetic
    that inflated a ~100 mm board's seed to 218 mm when 9 of 10 leaves were
    edge-pinned (KC-AXHQTP, RC-P1 in the compactness plan).

    ``widths``/``heights`` are per-child-index dims (a dict so the same helper
    serves the un-rotated seed estimate and the rotation-aware post-solve
    re-fit); indices absent from them are skipped.
    """
    sides: dict[str, list[int]] = {"left": [], "right": [], "top": [], "bottom": []}
    corners: list[int] = []
    for spec in derived.child_specs.values():
        if not spec.constraints:
            continue
        primary = next((c for c in spec.constraints if c.strict), spec.constraints[0])
        idx = spec.child_index
        if idx not in widths or idx not in heights:
            continue
        if primary.target == "edge" and primary.value in sides:
            sides[primary.value].append(idx)
        elif primary.target == "corner":
            corners.append(idx)

    def _stack(idxs: list[int], dims: dict[int, float]) -> float:
        # members laid end to end along the edge + inter/end gaps
        if not idxs:
            return 0.0
        return sum(dims[i] for i in idxs) + spacing_mm * (len(idxs) + 1)

    def _depth(idxs: list[int], dims: dict[int, float]) -> float:
        # how far the bank reaches into the board = widest single member
        return max((dims[i] for i in idxs), default=0.0)

    return {
        "lr_present": bool(sides["left"] or sides["right"]),
        "lr_stack_h": max(
            _stack(sides["left"], heights), _stack(sides["right"], heights)
        ),
        "lr_depth_w": _depth(sides["left"], widths) + _depth(sides["right"], widths),
        "tb_present": bool(sides["top"] or sides["bottom"]),
        "tb_stack_w": max(
            _stack(sides["top"], widths), _stack(sides["bottom"], widths)
        ),
        "tb_depth_h": _depth(sides["top"], heights) + _depth(sides["bottom"], heights),
        "corner_w": _depth(corners, widths),
        "corner_h": _depth(corners, heights),
    }


def _seed_outline_dimensions(
    loaded_artifacts,
    derived: DerivedAttachmentConstraints,
    spacing_mm: float,
    *,
    area_overhead: float = 2.5,
    aspect_target: float = 1.0,
    seed_cap: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Estimate (width, height) for the seed board outline the unified
    placer runs in.

    The legacy ``estimate_layer_aware_parent_board_size`` operates on
    layer envelopes (pad rects), which under-counts space for parent
    composition: synthetic blocks need room for their full bbox, not
    just their pad cluster. We bound by the children's whole-bbox area
    with slack, then floor by the per-axis constraint span when any
    axis-pinning constraint exists -- so an aggressive ``USB on left +
    power on right`` config cannot collapse the seed below the children's
    natural width.

    ``area_overhead`` (default 2.5) multiplies child total area to set
    the seed's nominal area: lower = tighter seed = forces compaction;
    higher = looser seed = lets the placer sprawl.

    ``aspect_target`` (default 1.0 = square) is the seed's width/height
    ratio.  Setting <1.0 produces a tall seed (favours vertical
    layouts), >1.0 a wide seed (favours horizontal-strip layouts).  The
    floors below (max single child + spacing, and the per-edge bank spans
    from :func:`_edge_bank_geometry`) still apply, so an aggressive aspect
    cannot collapse the seed below what the children actually need.

    ``seed_cap`` (from ``inscribed_rect_bound``) bounds the aspect-driven
    base to the largest content rect the brief-requested outline shape can
    contain at its requested ``size_mm`` — the placement-side half of the
    shape contract, so the solver packs INTO the shape instead of having
    the shape rejected around a sprawled rectangle at stamp time. Advisory:
    the solvability floors below still win, and a placement that genuinely
    cannot fit fails loudly at the stamp-time guard, never silently.
    """
    if not loaded_artifacts:
        return (max(20.0, spacing_mm * 4),) * 2

    # Use the content bbox of each artifact (not the leaf-PCB outline) so
    # the seed reflects the placer's actual space need.
    widths: list[float] = []
    heights: list[float] = []
    for art in loaded_artifacts:
        transformed = transform_loaded_artifact(art, origin=Point(0.0, 0.0), rotation=0.0)
        tl, br = transformed.bounding_box
        widths.append(max(0.0, br.x - tl.x))
        heights.append(max(0.0, br.y - tl.y))
    total_area = sum(w * h for w, h in zip(widths, heights))
    target_area = max(total_area, 1.0) * max(0.5, area_overhead)
    aspect = max(0.1, aspect_target)
    base_w = math.sqrt(target_area * aspect) + spacing_mm * 2.0
    base_h = math.sqrt(target_area / aspect) + spacing_mm * 2.0
    if seed_cap is not None:
        base_w = min(base_w, seed_cap[0])
        base_h = min(base_h, seed_cap[1])
    # Solvability floor: the single biggest child must fit with spacing. The
    # old ``sum*0.6`` fallback was DROPPED -- it summed every child's width
    # into one row on the same axis-blind arithmetic that RC-P1 fixes below.
    seed_w = max(base_w, max(widths) + spacing_mm * 4)
    seed_h = max(base_h, max(heights) + spacing_mm * 4)

    # Per-edge constraint floors (RC-P1): keep the seed wide/tall enough for
    # each BANK of edge-pinned children without the old single-row sum that
    # collapsed opposing banks into one horizontal strip. Left/right banks
    # define height (their members stack vertically) and add their depths to
    # the width; top/bottom banks are the transpose; corners contribute their
    # own extent to both axes.
    banks = _edge_bank_geometry(
        derived, dict(enumerate(widths)), dict(enumerate(heights)), spacing_mm
    )
    if banks["lr_present"]:
        seed_h = max(seed_h, banks["lr_stack_h"])
        seed_w = max(seed_w, banks["lr_depth_w"] + spacing_mm * 3)
    if banks["tb_present"]:
        seed_w = max(seed_w, banks["tb_stack_w"])
        seed_h = max(seed_h, banks["tb_depth_h"] + spacing_mm * 3)
    if banks["corner_w"] > 0.0 or banks["corner_h"] > 0.0:
        seed_w = max(seed_w, banks["corner_w"] + spacing_mm * 2)
        seed_h = max(seed_h, banks["corner_h"] + spacing_mm * 2)
    return seed_w, seed_h


# Absolute floor for copper-to-board-edge breathing room. KiCad's default
# board-setup edge clearance is 0.2 mm; anything the outline math emits below
# this margin ships a guaranteed copper_edge_clearance DRC error (self-eval
# 2026-07-17 batch: run_02/09/11 rejected at 0.0-0.18 mm actual). +0.1 mm
# guard over the constraint, same philosophy as the DSN 10 um guards.
_COPPER_EDGE_MARGIN_MM = 0.3


def _refit_seed_from_placement(
    placed_child_bboxes: dict[int, tuple[Point, Point]],
    derived: DerivedAttachmentConstraints,
    spacing_mm: float,
    seed_wh: tuple[float, float],
) -> tuple[float, float] | None:
    """Right-size the seed from a completed (pass-1) placement (Fix 2).

    The pass-1 seed is an area-basis *estimate* (``√(Σ child area · overhead)``)
    that deliberately over-provisions so the solver has room. On a board whose
    children are mostly edge-pinned leaves that leaves a big empty interior:
    the banks pin flush to the oversized seed edges and never re-fit (RC-P2).
    After the solve the true content need is measurable -- the interior
    (non-edge) blocks' union extent packed between the per-edge banks -- so we
    derive a tighter seed to re-solve on.

    Returns the tighter ``(w, h)``, or ``None`` when pass 1 was already tight
    (< 10 % slack on BOTH axes) so the caller skips the re-solve. A re-fit only
    ever tightens: the result is clamped to never exceed ``seed_wh``.
    """
    seed_w, seed_h = seed_wh
    if seed_w <= 0.0 or seed_h <= 0.0 or not placed_child_bboxes:
        return None

    # Placed (rotation-aware) child dims, keyed by child index.
    widths: dict[int, float] = {}
    heights: dict[int, float] = {}
    for idx, (tl, br) in placed_child_bboxes.items():
        widths[idx] = max(0.0, br.x - tl.x)
        heights[idx] = max(0.0, br.y - tl.y)
    if not widths:
        return None
    banks = _edge_bank_geometry(derived, widths, heights, spacing_mm)

    # Interior = child blocks NOT pinned to an edge/corner. They floated inside
    # the seed; their pass-1 union bbox is the room they need between opposing
    # banks. Keyed off the constraints (robust to the unlock_all_footprints
    # path where the edge blocks are not flagged ``locked``).
    edge_pinned = {
        spec.child_index
        for spec in derived.child_specs.values()
        if spec.constraints
        and any(c.target in ("edge", "corner") for c in spec.constraints)
    }
    interior_boxes = [
        placed_child_bboxes[i] for i in placed_child_bboxes if i not in edge_pinned
    ]
    if interior_boxes:
        int_w = max(b[1].x for b in interior_boxes) - min(b[0].x for b in interior_boxes)
        int_h = max(b[1].y for b in interior_boxes) - min(b[0].y for b in interior_boxes)
    else:
        int_w = int_h = 0.0

    # Per-axis padding floored at the copper-edge margin: at tiny configured
    # spacing the seed must still leave DRC-clearable room between child copper
    # (the bboxes include trace copper) and the board edge on both sides.
    pad2 = max(spacing_mm * 2, _COPPER_EDGE_MARGIN_MM * 2)
    pad3 = max(spacing_mm * 3, _COPPER_EDGE_MARGIN_MM * 3)
    pad4 = max(spacing_mm * 4, _COPPER_EDGE_MARGIN_MM * 4)
    floor_w = max(widths.values()) + pad4
    floor_h = max(heights.values()) + pad4
    need_w = max(
        banks["lr_depth_w"] + int_w + pad3,  # L-bank | interior | R-bank
        banks["tb_stack_w"],
        int_w + pad2,
        banks["corner_w"] + pad2,
        floor_w,
    )
    need_h = max(
        banks["lr_stack_h"],
        banks["tb_depth_h"] + int_h + pad3,  # T-bank / interior / B-bank
        int_h + pad2,
        banks["corner_h"] + pad2,
        floor_h,
    )
    # A re-fit only tightens; never grow past the pass-1 seed. When a floor
    # exceeds the pass-1 seed on an axis, that axis clamps back to the seed
    # (no tightening there -- never below the copper-margin floors) while the
    # OTHER axis may still legitimately shrink (single-axis sprawl is the
    # common KC-AXHQTP shape).
    need_w = min(need_w, seed_w)
    need_h = min(need_h, seed_h)
    # Worth a re-solve only when it removes meaningful slack (>10% on an axis).
    if need_w > seed_w * 0.9 and need_h > seed_h * 0.9:
        return None
    return (need_w, need_h)


def _post_solve_geometry(
    placements: dict[str, ChildArtifactPlacement],
    loaded_artifacts,
) -> tuple[
    dict[int, tuple[Point, Point]],
    list[dict[str, Any]],
    dict[str, Point],
    dict[int, Any],
]:
    """Build per-child bbox, envelopes, anchor positions, and transformed
    cache from the solver output. Used by the validation block and the
    final-outline computation downstream."""
    bboxes_by_index: dict[int, tuple[Point, Point]] = {}
    placed_envelopes: list[dict[str, Any]] = []
    anchor_positions: dict[str, Point] = {}
    transformed_by_index: dict[int, Any] = {}

    for child_index, artifact in enumerate(loaded_artifacts):
        placement = placements.get(artifact.instance_path)
        if placement is None:
            continue
        transformed = transform_loaded_artifact(
            artifact,
            origin=placement.origin,
            rotation=placement.rotation,
        )
        bbox = transformed.bounding_box
        bboxes_by_index[child_index] = bbox
        transformed_by_index[child_index] = transformed
        envelopes = child_layer_envelopes(transformed)
        blocker_set = extract_leaf_blocker_set(artifact)
        placed_envelopes.append(
            {
                "bbox": bbox,
                "envelopes": envelopes,
                "blocker_set": blocker_set,
                "origin": placement.origin,
                "rotation": placement.rotation,
                "label": artifact.sheet_name,
            }
        )
        # Anchor position for each constrained ref: the existing per-spec
        # constraint_entries already encode local_anchor_offset at the
        # CHOSEN rotation, so we walk transformed.transformed_components
        # for the constrained refs and recover the world-frame anchor
        # using the same local-anchor math the legacy compose used.
    return bboxes_by_index, placed_envelopes, anchor_positions, transformed_by_index


def _resolve_constraint_anchor_positions(
    derived: DerivedAttachmentConstraints,
    placements: dict[str, ChildArtifactPlacement],
    loaded_artifacts,
    transformed_by_index: dict[int, Any],
    parent_local: dict[str, Component],
) -> dict[str, Point]:
    """Compute the world-frame anchor position for every constrained ref.

    For child-artifact constraints, recompute local_anchor_offset at the
    chosen rotation (the rotation the solver picked, not the spec's first
    candidate) and add it to the placement origin. For parent-local
    constraints, use the component's pad centroid or body center.
    """
    anchors: dict[str, Point] = {}
    for spec in derived.child_specs.values():
        artifact = loaded_artifacts[spec.child_index]
        placement = placements.get(artifact.instance_path)
        if placement is None:
            continue
        transformed = transformed_by_index.get(spec.child_index)
        if transformed is None:
            continue
        blocker_set = extract_leaf_blocker_set(artifact)
        for constraint in spec.constraints:
            try:
                from kicraft.autoplacer.brain.subcircuit_composer import (
                    _compute_local_anchor_offset,
                )
                local_offset = _compute_local_anchor_offset(
                    transformed,
                    constraint,
                    spec.constraints,
                    blocker_set,
                    placement.rotation,
                )
            except Exception:
                continue
            anchors[constraint.ref] = Point(
                placement.origin.x + local_offset.x,
                placement.origin.y + local_offset.y,
            )

    from kicraft.autoplacer.brain.subcircuit_composer import pad_centroid_anchor

    for constraint in derived.parent_local_constraints:
        comp = parent_local.get(constraint.ref)
        if comp is None:
            continue
        # Same body-pinned anchor the leaf path uses for holes -- one formula.
        anchors[constraint.ref] = pad_centroid_anchor(comp)
    return anchors


def _block_artifact_origin(comp: Component) -> Point:
    """Inverse of synthetic-block-pos -> artifact-origin mapping. For a
    synthetic block, ``world_origin = pos - rotate_vector(body_center_offset,
    +rot)`` -- the true inverse of the KiCad-CW forward body-center transform;
    matches ``_recover_artifact_placements`` / ``_world_artifact_origin`` (all
    three share the SAME convention; see parent_adapter._rotated). For non-block
    components (parent-local mounting holes) the origin is simply ``comp.pos``.
    """
    if comp.kind != "subcircuit" or comp.block_artifact_origin_offset is None:
        return comp.pos
    rotated = geometry.rotate_vector(comp.block_artifact_origin_offset, comp.rotation)
    return Point(comp.pos.x - rotated.x, comp.pos.y - rotated.y)


def _apply_slide(comp: Component, free_axis_y: bool, delta: float) -> None:
    """Translate a Component (its ``pos``, ``body_center``, and pad
    positions) along one axis by ``delta``."""
    if free_axis_y:
        comp.pos = Point(comp.pos.x, comp.pos.y + delta)
        if comp.body_center is not None:
            comp.body_center = Point(comp.body_center.x, comp.body_center.y + delta)
        for pad in comp.pads:
            pad.pos = Point(pad.pos.x, pad.pos.y + delta)
    else:
        comp.pos = Point(comp.pos.x + delta, comp.pos.y)
        if comp.body_center is not None:
            comp.body_center = Point(comp.body_center.x + delta, comp.body_center.y)
        for pad in comp.pads:
            pad.pos = Point(pad.pos.x + delta, pad.pos.y)


def _slide_clearance_ok(solved: dict[str, Component], moving_ref: str) -> bool:
    """True iff the bbox of ``solved[moving_ref]`` does not introduce an
    incompatible overlap with any other placed component. Compatible
    block-on-block overlaps (per ``can_overlap_sparse``) are allowed --
    those represent the dual-layer stacking the parent solver already
    builds. Non-block components and same-side blocks are not allowed
    to overlap.
    """
    moving = solved[moving_ref]
    moving_bbox = moving.bbox()
    moving_blocker = moving.block_blocker_set
    moving_origin = _block_artifact_origin(moving)
    for other_ref, other in solved.items():
        if other_ref == moving_ref:
            continue
        if _bbox_disjoint(moving_bbox, other.bbox()):
            continue
        if moving_blocker is not None and other.block_blocker_set is not None:
            other_origin = _block_artifact_origin(other)
            if can_overlap_sparse(
                moving_blocker,
                moving_origin,
                moving.rotation,
                other.block_blocker_set,
                other_origin,
                other.rotation,
            ):
                continue
        return False
    return True


def _largest_safe_slide(
    solved: dict[str, Component],
    moving_ref: str,
    free_axis_y: bool,
    desired_delta: float,
) -> float:
    """Binary-search the largest |delta| in [0, |desired_delta|] (same sign
    as desired) that keeps ``_slide_clearance_ok`` true. Mutates and
    restores the moving component during probes."""
    if abs(desired_delta) < 1e-3:
        return 0.0

    moving = solved[moving_ref]
    saved_pos = moving.pos
    saved_body = moving.body_center
    saved_pad_positions = [pad.pos for pad in moving.pads]

    def restore() -> None:
        moving.pos = saved_pos
        moving.body_center = saved_body
        for pad, original in zip(moving.pads, saved_pad_positions):
            pad.pos = original

    def safe_at(d: float) -> bool:
        _apply_slide(moving, free_axis_y, d)
        try:
            return _slide_clearance_ok(solved, moving_ref)
        finally:
            restore()

    if safe_at(desired_delta):
        return desired_delta

    sign = 1.0 if desired_delta > 0 else -1.0
    lo, hi = 0.0, abs(desired_delta)
    for _ in range(8):
        mid = (lo + hi) / 2.0
        if safe_at(sign * mid):
            lo = mid
        else:
            hi = mid
    return sign * lo


def _collision_aware_corner_snap(
    solved: dict[str, Component], ref: str, dx: float, dy: float
) -> bool:
    """Slide ``ref`` toward a corner/edge target by ``(dx, dy)``, one axis at a
    time, stopping each axis at the largest offset that keeps clearance.

    A raw corner snap slides a parent-local mounting hole onto the exact cluster
    corner -- which is precisely where a corner leaf's header pads sit -- stamping
    the hole's PTH pad on top of leaf copper at 0.0 mm (the encoder-oled-panel
    ``candidate-search ... shorts=10..16`` abort). Reusing ``_largest_safe_slide``
    lands the hole as close to the corner as clearance allows and no closer (WS4).
    Returns True if any movement was applied.
    """
    moved = False
    if abs(dx) > 1e-3:
        safe_dx = _largest_safe_slide(solved, ref, False, dx)
        if abs(safe_dx) > 1e-3:
            _apply_slide(solved[ref], free_axis_y=False, delta=safe_dx)
            moved = True
    if abs(dy) > 1e-3:
        safe_dy = _largest_safe_slide(solved, ref, True, dy)
        if abs(safe_dy) > 1e-3:
            _apply_slide(solved[ref], free_axis_y=True, delta=safe_dy)
            moved = True
    return moved


def _cluster_bbox(
    solved: dict[str, Component], exclude_refs: set[str]
) -> tuple[Point, Point] | None:
    """Bbox union of every component in ``solved`` whose ref is not in
    ``exclude_refs``. Returns None when nothing is left."""
    lo_x = lo_y = math.inf
    hi_x = hi_y = -math.inf
    for ref, comp in solved.items():
        if ref in exclude_refs:
            continue
        b_min, b_max = comp.bbox()
        if b_min.x < lo_x:
            lo_x = b_min.x
        if b_min.y < lo_y:
            lo_y = b_min.y
        if b_max.x > hi_x:
            hi_x = b_max.x
        if b_max.y > hi_y:
            hi_y = b_max.y
    if lo_x == math.inf:
        return None
    return (Point(lo_x, lo_y), Point(hi_x, hi_y))


def _slide_constrained_to_cluster(
    solved: dict[str, Component],
    derived: DerivedAttachmentConstraints,
    synthetic_refs: dict[int, str],
) -> None:
    """In-place: pull each constrained component back to the cluster.

    The parent solver pins constrained components to the seed frame's
    edges or corners. The seed frame is oversized (2.5x area slack) to
    leave routing room, so a constrained component pinned to its corner
    can sit far outside the actual cluster's extent. ``_compute_final_outline``
    then snaps the corresponding board side to include the constrained
    component, inflating PCB area.

    Two operations:

      * **Edge-pinned components** keep their pinned axis fixed but slide
        along the free axis so their bbox falls inside the cluster's
        perpendicular span. Walked back via ``_largest_safe_slide`` if
        the slide would create an incompatible block overlap.
      * **Corner-pinned parent-local components** (mounting holes) jump
        to the corresponding corner of the cluster bbox. They are placed
        BY the constraint -- their absolute position is meant to track
        the board corner -- so dragging them off the seed-frame corner
        onto the cluster corner is the right operation.
    """
    edge_targets: list[tuple[str, str]] = []
    for child_index, spec in derived.child_specs.items():
        block_ref = synthetic_refs.get(child_index)
        if block_ref is None or block_ref not in solved:
            continue
        primary = next((c for c in spec.constraints if c.strict), None)
        if primary is None or primary.target != "edge":
            continue
        edge_targets.append((block_ref, primary.value))
    for c in derived.parent_local_constraints:
        if c.target != "edge" or c.ref not in solved:
            continue
        edge_targets.append((c.ref, c.value))

    corner_targets: list[tuple[str, str]] = [
        (c.ref, c.value)
        for c in derived.parent_local_constraints
        if c.target == "corner" and c.ref in solved
    ]
    if not edge_targets and not corner_targets:
        return

    # Corner-pinned refs sit at board corners by design and would stretch
    # the span to the full seed frame; exclude them from cluster math.
    corner_refs = {c.ref for c in derived.constraints if c.target == "corner"}

    slides_applied = 0
    snapped_corners = 0

    # --- 1. Edge-pinned slides on the free axis ---
    for ref, edge_value in edge_targets:
        comp = solved[ref]
        free_axis_y = edge_value in ("left", "right")
        # Cluster span excludes the moving ref and every corner-pinned ref.
        exclude = {ref} | corner_refs
        cluster = _cluster_bbox(solved, exclude)
        if cluster is None:
            continue
        cluster_lo = cluster[0].y if free_axis_y else cluster[0].x
        cluster_hi = cluster[1].y if free_axis_y else cluster[1].x

        b_min, b_max = comp.bbox()
        cur_lo, cur_hi = (b_min.y, b_max.y) if free_axis_y else (b_min.x, b_max.x)

        if cur_lo >= cluster_lo and cur_hi <= cluster_hi:
            continue
        if (cur_hi - cur_lo) > (cluster_hi - cluster_lo) + 1e-3:
            continue

        if cur_hi > cluster_hi:
            delta = cluster_hi - cur_hi
        elif cur_lo < cluster_lo:
            delta = cluster_lo - cur_lo
        else:
            continue

        safe = _largest_safe_slide(solved, ref, free_axis_y, delta)
        if abs(safe) < 1e-3:
            continue
        _apply_slide(comp, free_axis_y, safe)
        slides_applied += 1

    # --- 2. Corner-pinned parent-local snap to cluster corner ---
    # Edge-slid components are now in the cluster's perpendicular span,
    # so include them in the cluster bbox used for corner placement.
    cluster_for_corners = _cluster_bbox(solved, corner_refs)
    if cluster_for_corners is not None:
        c_min, c_max = cluster_for_corners
        for ref, corner_value in corner_targets:
            comp = solved[ref]
            cur_anchor_x = (
                comp.body_center.x if comp.body_center is not None else comp.pos.x
            )
            cur_anchor_y = (
                comp.body_center.y if comp.body_center is not None else comp.pos.y
            )
            target_x = c_min.x if "left" in corner_value else c_max.x
            target_y = c_min.y if "top" in corner_value else c_max.y
            dx = target_x - cur_anchor_x
            dy = target_y - cur_anchor_y
            if _collision_aware_corner_snap(solved, ref, dx, dy):
                snapped_corners += 1

    if slides_applied or snapped_corners:
        print(
            f"  Cluster-slide: aligned {slides_applied} edge-constrained, "
            f"{snapped_corners} corner-constrained component(s) to the cluster"
        )


def _compute_final_outline(
    placed_bboxes: list[tuple[Point, Point]],
    constraints: list[AttachmentConstraint],
    anchor_positions: dict[str, Point],
    spacing_mm: float,
    *,
    edge_constrained_refs: set[str] | None = None,
    edge_zoned_outline_sides: frozenset[str] | None = None,
    pad_edge_clearance_mm: float = 0.2,
) -> tuple[Point, Point]:
    """Final outline tracks ``constraint_aware_outline`` (which already
    applies ``margin_mm`` to unconstrained sides and snaps constrained
    sides to anchor targets), expanded if necessary so the outline still
    contains every placed bbox -- never shrinking past geometry."""
    if not placed_bboxes:
        return (Point(0.0, 0.0), Point(0.0, 0.0))

    # Margin floored at the copper-edge minimum: placed bboxes include trace
    # copper (see _compute_layout_bbox), so any outline side closer than
    # _COPPER_EDGE_MARGIN_MM to a bbox is a guaranteed copper_edge_clearance
    # DRC error at stamp time.
    margin_mm = max(spacing_mm, _COPPER_EDGE_MARGIN_MM)

    if not constraints:
        geom_min_x = min(b[0].x for b in placed_bboxes) - margin_mm
        geom_min_y = min(b[0].y for b in placed_bboxes) - margin_mm
        geom_max_x = max(b[1].x for b in placed_bboxes) + margin_mm
        geom_max_y = max(b[1].y for b in placed_bboxes) + margin_mm
        return (Point(geom_min_x, geom_min_y), Point(geom_max_x, geom_max_y))

    constraint_outline = constraint_aware_outline(
        placed_bboxes=placed_bboxes,
        attachment_constraints=constraints,
        constrained_ref_world_anchors=anchor_positions,
        margin_mm=margin_mm,
    )
    # On a side constrained by an EDGE constraint (e.g. J1 edge=left)
    # the outline must STOP at the constraint anchor target -- expanding
    # past it would push the connector inboard so its body no longer sits
    # flush with the PCB edge. CORNER-only sides (e.g. only H4 corner=
    # top-left constrains the top edge) are different: the corner mount
    # may have been escaped by _stack_compatible_blocks / corner-escape
    # to dodge an edge connector, and using the escaped mount's centroid
    # as the precise outline edge clips legitimately-placed leaves outside
    # the board. For corner-only sides, expand to fit geometry so escaped
    # mounts never shrink the outline below placed bboxes.
    edge_constrained_sides = {
        "left": False, "right": False, "top": False, "bottom": False,
    }
    corner_constrained_sides = {
        "left": False, "right": False, "top": False, "bottom": False,
    }
    # Sides pinned by a long-barrel connector's PAD face (right-angle BNC,
    # barrel jack): the board edge legitimately sits far inboard of the placed
    # geometry edge (the overhanging barrel tip), so the anchor-slack clamp
    # below must TRUST the anchor instead of rejecting it as a transform bug.
    barrel_sides = {
        "left": False, "right": False, "top": False, "bottom": False,
    }
    for c in constraints:
        if c.target == "edge":
            edge_constrained_sides[c.value] = True
            if getattr(c, "barrel_overhang", False) and c.value in barrel_sides:
                barrel_sides[c.value] = True
        elif c.target == "corner":
            for side in c.value.split("-"):
                if side in corner_constrained_sides:
                    corner_constrained_sides[side] = True

    geom_min_x = min(b[0].x for b in placed_bboxes)
    geom_min_y = min(b[0].y for b in placed_bboxes)
    geom_max_x = max(b[1].x for b in placed_bboxes)
    geom_max_y = max(b[1].y for b in placed_bboxes)

    # For corner-constrained sides, ``c_val`` is the corner anchor's
    # coordinate (no margin -- corner anchors are points, and
    # ``constraint_aware_outline`` populates ``{top,bottom,left,right}_edges``
    # from corner constraints without adding ``margin_mm``). When a leaf's
    # geometry extends past the corner anchor on that side, we must inflate
    # ``g_val`` by ``spacing_mm`` so pad copper keeps its full edge clearance
    # -- the unconstrained branch already gets margin via constraint_aware_outline.
    #
    # Edge-snap sanity clamp: a flush-mount anchor legitimately sits within
    # a couple mm of the placed geometry's edge on its side (pad inset /
    # housing overhang). An anchor further out is a frame/transform bug
    # upstream, and snapping the outline to it bakes a phantom bare-FR4 strip
    # into the board (the outline repair pass only ever grows). The old
    # ``spacing_mm + 10`` (~11mm) was loose enough to wave through an 8.8mm
    # stranded USB-C anchor; ``spacing_mm + 2`` keeps the real flush case
    # (delta ~= overhang, well under the clamp) while rejecting the
    # part-height-scale reflection the convention bug produced. The fallback
    # (geometry + spacing) is benign -- it just tracks geometry instead of a
    # bogus anchor. Fall back and say so.
    anchor_slack_mm = spacing_mm + 2.0

    def _resolve_min(side: str, c_val: float, g_val: float) -> float:
        if barrel_sides[side]:
            # Pad-anchored long-barrel connector: trust the anchor (the barrel
            # overhangs by design, so c_val is meant to sit inboard of g_val).
            return c_val
        if edge_constrained_sides[side]:
            if abs(c_val - g_val) > anchor_slack_mm:
                print(
                    f"[outline] {side} edge anchor {c_val:.2f}mm is "
                    f"{abs(c_val - g_val):.1f}mm from placed geometry edge "
                    f"{g_val:.2f}mm (> {anchor_slack_mm:.1f}mm slack); "
                    "ignoring anchor, using geometry + margin"
                )
                return g_val - margin_mm
            return c_val
        if corner_constrained_sides[side]:
            return min(c_val, g_val - margin_mm)
        return min(c_val, g_val)

    def _resolve_max(side: str, c_val: float, g_val: float) -> float:
        if barrel_sides[side]:
            # Pad-anchored long-barrel connector: trust the anchor (the barrel
            # overhangs by design, so the board edge c_val sits inboard of the
            # placed geometry edge g_val at the barrel tip).
            return c_val
        if edge_constrained_sides[side]:
            if abs(c_val - g_val) > anchor_slack_mm:
                print(
                    f"[outline] {side} edge anchor {c_val:.2f}mm is "
                    f"{abs(c_val - g_val):.1f}mm from placed geometry edge "
                    f"{g_val:.2f}mm (> {anchor_slack_mm:.1f}mm slack); "
                    "ignoring anchor, using geometry + margin"
                )
                return g_val + margin_mm
            return c_val
        if corner_constrained_sides[side]:
            return max(c_val, g_val + margin_mm)
        return max(c_val, g_val)

    out_min_x = _resolve_min("left", constraint_outline[0].x, geom_min_x)
    out_min_y = _resolve_min("top", constraint_outline[0].y, geom_min_y)
    out_max_x = _resolve_max("right", constraint_outline[1].x, geom_max_x)
    out_max_y = _resolve_max("bottom", constraint_outline[1].y, geom_max_y)

    # --- Containment invariant (Phase 3A) ---
    # The outline must enclose every placed block bbox with spacing_mm of
    # copper-to-edge breathing room on NON-connector sides, mirroring the
    # grow _repair_parent_outline applies downstream. Connector-defined sides
    # (edge_zoned_outline_sides) are anchor-authoritative: the barrel/edge
    # branches above already placed the edge at the mouth + overhang, and a
    # bbox-level floor there would wrongly enclose the overhanging barrel body
    # (the block bbox includes the barrel tip). Pad/trace/via containment on
    # those sides is finalized by the mutating _repair_parent_outline.
    # ``edge_constrained_refs`` is accepted for the future pad-level path but
    # unused at the bbox level (no pad data here).
    _ = edge_constrained_refs
    conn_sides = (
        edge_zoned_outline_sides
        if edge_zoned_outline_sides is not None
        else frozenset(
            side
            for c in constraints
            if c.target in ("edge", "corner")
            and not _is_mounting_hole_ref(c.ref)
            for side in (
                [c.value]
                if c.target == "edge"
                and c.value in ("left", "right", "top", "bottom")
                else [
                    s
                    for s in c.value.split("-")
                    if s in ("left", "right", "top", "bottom")
                ]
            )
        )
    )
    _ = pad_edge_clearance_mm  # reserved for the pad-level containment path
    if "left" not in conn_sides:
        out_min_x = min(out_min_x, geom_min_x - margin_mm)
    if "top" not in conn_sides:
        out_min_y = min(out_min_y, geom_min_y - margin_mm)
    if "right" not in conn_sides:
        out_max_x = max(out_max_x, geom_max_x + margin_mm)
    if "bottom" not in conn_sides:
        out_max_y = max(out_max_y, geom_max_y + margin_mm)
    return (Point(out_min_x, out_min_y), Point(out_max_x, out_max_y))


def _is_mounting_hole_ref(ref: str, comp: Component | None = None) -> bool:
    """Heuristic: refs like 'H1', 'H86' are mounting holes.

    KiCad convention is 'H' or 'MH' prefix. Components fed to manual
    mode's parent_local list are already filtered to constraint-pinned
    items, so a name match is sufficient here.
    """
    if not ref:
        return False
    upper = ref.upper()
    return upper.startswith("H") and (len(upper) == 1 or upper[1].isdigit() or upper[1] == "_")


def _warn_non_board_level_parent_local(parent_local: dict[str, Component]) -> list[str]:
    """Surface the parent-only-leaves invariant (Part 2, Lever 2.1) at its source.

    A top-level/parent sheet should carry only child leaves plus board-level
    structure (mounting holes / fiducials). Any OTHER loose component on the
    parent sheet is a "parent-local" ref that flows through compose's SECOND
    placement path (``_snap_parent_local`` + its connector branch) instead of
    the single leaf path -- the exact duplication this plan collapses. Warn
    (don't fail) so the violation is visible; the eventual collapse auto-wraps
    each offender as a single-component leaf so it flows through the one path.
    Returns the offending refs (board-level refs excluded)."""
    import logging

    offenders = [
        ref for ref, comp in parent_local.items()
        if not _is_mounting_hole_ref(ref, comp)
    ]
    if offenders:
        logging.getLogger(__name__).warning(
            "parent-only-leaves invariant: %d non-board-level parent-local "
            "component(s) take the parent-local placement path rather than a "
            "leaf: %s (Lever 2.1 will auto-wrap these as single-component leaves)",
            len(offenders), ", ".join(sorted(offenders)),
        )
    return offenders


def _wrap_loose_parent_components_as_leaves(
    parent_local: dict[str, Component],
    loaded_artifacts: list,
) -> tuple[list, dict[str, Component]]:
    """Lever 2.1: wrap each loose parent-level non-board component (an edge
    connector etc. that sits on the parent sheet, in no leaf) as a
    single-component leaf so it flows through the ONE leaf placement path --
    the solver edge-pins its synthetic block as the board extremity and the
    composer flush-aligns it, exactly like a leaf connector.

    This replaces the parent-local snap, which only pinned a connector to the
    pre-repair outline and stranded it inboard whenever a taller leaf defined
    the board extremity on that edge (the PARENT_LOCAL_CONN fixture). Wrapping
    routes it through the path that actually makes it the extremity.

    Board-level structure (mounting holes / fiducials) is NOT wrapped: it has
    no mating direction and wants the generic corner/zone parent-local snap.
    Returns the augmented artifact list and the trimmed parent_local dict."""
    offenders = _warn_non_board_level_parent_local(parent_local)
    if not offenders:
        return loaded_artifacts, parent_local

    import logging

    wrapped = list(loaded_artifacts)
    remaining = dict(parent_local)
    for ref in offenders:
        comp = copy.deepcopy(remaining.pop(ref))
        # Re-base the component into a (0,0)-anchored leaf-local frame sized to
        # its own extent. A real leaf's components sit INSIDE its board outline;
        # the parent-local component still carries its absolute seed-PCB
        # position, so without this the body-center origin recovery and the
        # anchor frame math don't cancel and the connector's edge anchor lands
        # ~(seed offset) mm out of frame (slack fallback -> stranded).
        half_w, half_h = comp.width_mm / 2.0, comp.height_mm / 2.0
        bc = comp.body_center if comp.body_center is not None else comp.pos
        min_x = min([bc.x - half_w] + [p.pos.x for p in comp.pads])
        min_y = min([bc.y - half_h] + [p.pos.y for p in comp.pads])
        max_x = max([bc.x + half_w] + [p.pos.x for p in comp.pads])
        max_y = max([bc.y + half_h] + [p.pos.y for p in comp.pads])
        dx, dy = -min_x, -min_y
        comp.pos = Point(comp.pos.x + dx, comp.pos.y + dy)
        comp.body_center = Point(bc.x + dx, bc.y + dy)
        for pad in comp.pads:
            pad.pos = Point(pad.pos.x + dx, pad.pos.y + dy)
        layout = SubCircuitLayout(
            subcircuit_id=SubCircuitId(
                sheet_name=f"__auto_{ref}",
                sheet_file=f"__auto_{ref}.kicad_sch",
                instance_path=f"/__auto_{ref}",
            ),
            components={ref: comp},
            bounding_box=(max(0.1, max_x - min_x), max(0.1, max_y - min_y)),
        )
        wrapped.append(
            LoadedSubcircuitArtifact(
                artifact_dir=f"<auto:{ref}>", metadata={}, debug={}, layout=layout
            )
        )
    logging.getLogger(__name__).info(
        "Lever 2.1: auto-wrapped %d loose parent-level component(s) as "
        "single-component leaves: %s",
        len(offenders), ", ".join(offenders),
    )
    return wrapped, remaining


def _move_component_to(comp: Component, new_pos: Point) -> None:
    """Translate a Component (and its pads / body_center) so its anchor
    lands at ``new_pos``. Preserves rotation; mirrors the same delta
    pattern used by _snap_parent_local.
    """
    dx = new_pos.x - comp.pos.x
    dy = new_pos.y - comp.pos.y
    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        return
    comp.pos = Point(comp.pos.x + dx, comp.pos.y + dy)
    if comp.body_center is not None:
        comp.body_center = Point(comp.body_center.x + dx, comp.body_center.y + dy)
    for pad in comp.pads:
        pad.pos = Point(pad.pos.x + dx, pad.pos.y + dy)


def _snap_parent_local(
    comps: dict[str, Component],
    constraints: list[AttachmentConstraint],
    outline: tuple[Point, Point],
) -> None:
    """Translate each parent-local component with an edge/corner/zone
    constraint so its anchor lands at the exact constraint coordinate
    via edge_anchor_target_coordinate. Restores sub-mm precision the
    geometry validator enforces (the solver's edge-pinning math may
    leave it within the connector_inset_mm jitter window).

    Lever 2.1: loose parent-level CONNECTORS no longer reach here -- compose
    auto-wraps them as single-component leaves (_wrap_loose_parent_components_as_leaves)
    so they take the SAME edge-pin/flush-align path as leaf connectors. What
    remains parent-local is board-level structure (mounting holes / fiducials),
    which wants this generic corner/zone snap (no mating direction)."""
    min_pt, max_pt = outline
    for c in constraints:
        comp = comps.get(c.ref)
        if comp is None:
            continue

        target_x = comp.pos.x
        target_y = comp.pos.y
        if c.target == "edge":
            if c.value in ("left", "right"):
                target_x = edge_anchor_target_coordinate(c.value, c, min_pt, max_pt)
            elif c.value in ("top", "bottom"):
                target_y = edge_anchor_target_coordinate(c.value, c, min_pt, max_pt)
        elif c.target == "corner":
            corner_sides = {
                "top-left": ("left", "top"),
                "top-right": ("right", "top"),
                "bottom-left": ("left", "bottom"),
                "bottom-right": ("right", "bottom"),
            }.get(c.value)
            if corner_sides is None:
                continue
            target_x = edge_anchor_target_coordinate(corner_sides[0], c, min_pt, max_pt)
            target_y = edge_anchor_target_coordinate(corner_sides[1], c, min_pt, max_pt)
        elif c.target == "zone" and c.value == "bottom":
            target_y = max_pt.y - c.inward_keep_in_mm
        else:
            continue

        # Compute centroid of pads (or body center fallback) and shift to
        # the target. Anchor is what the validator checks, so we move the
        # whole comp by the delta needed to bring the anchor on target.
        if comp.pads:
            anchor_x = sum(p.pos.x for p in comp.pads) / len(comp.pads)
            anchor_y = sum(p.pos.y for p in comp.pads) / len(comp.pads)
        else:
            ref = comp.body_center if comp.body_center is not None else comp.pos
            anchor_x, anchor_y = ref.x, ref.y

        dx = target_x - anchor_x
        dy = target_y - anchor_y
        if abs(dx) < 1e-6 and abs(dy) < 1e-6:
            continue
        # Collision-aware snap: what reaches _snap_parent_local is board-level
        # structure (mounting holes / fiducials -- connectors are wrapped as
        # leaves upstream), so snapping the anchor to the exact corner/edge must
        # not stamp the hole onto a leaf's pads. Slide as far to the target as
        # clearance allows (WS4); a fully-clear target still lands exactly on it.
        _collision_aware_corner_snap(comps, c.ref, dx, dy)


def _synthetic_parent_definition(loaded_artifacts) -> SubCircuitDefinition:
    return SubCircuitDefinition(
        id=SubCircuitId(
            sheet_name="COMPOSED_PARENT",
            sheet_file="COMPOSED_PARENT.kicad_sch",
            instance_path="/COMPOSED_PARENT",
            parent_instance_path=None,
        ),
        schematic_path="",
        component_refs=[],
        ports=[],
        child_ids=[artifact.layout.subcircuit_id for artifact in loaded_artifacts],
        parent_id=None,
        is_leaf=False,
        sheet_uuid="",
        notes=[
            "synthetic_parent=true",
            f"artifact_count={len(loaded_artifacts)}",
        ],
    )


def _ensure_edge_blocks_extremal(
    solved: dict[str, Component],
    block_zones: dict[str, Any],
    margin_mm: float = 0.1,
) -> list[str]:
    """Shift each edge-zoned block OUTBOARD so it is the extremity on its zoned
    side, so the connector it contains defines the board edge and stays flush.

    Without this, the parent solver may place another leaf's block a hair more
    outboard than the edge-zoned connector's block; the board edge (= bbox union
    of all blocks) is then drawn at that other block and the connector reads as
    stranded inboard -- the KC-S8PC37 J1 signature once R8 is no longer masking
    the edge. The shift is OUTBOARD-only (toward the empty board edge), so it
    never creates an overlap; it just grows the board by the small amount the
    other block had crowded past the connector. Returns the refs shifted.
    """
    shifted: list[str] = []
    for bref, zone in (block_zones or {}).items():
        side = (zone or {}).get("edge")
        if side not in ("left", "right", "top", "bottom") or bref not in solved:
            continue
        e_tl, e_br = solved[bref].bbox(0.0)
        others = [solved[r].bbox(0.0) for r in solved if r != bref]
        if not others:
            continue
        dx = dy = 0.0
        if side == "left":
            m = min(b[0].x for b in others)
            if m < e_tl.x:
                dx = m - e_tl.x - margin_mm
        elif side == "right":
            m = max(b[1].x for b in others)
            if m > e_br.x:
                dx = m - e_br.x + margin_mm
        elif side == "top":
            m = min(b[0].y for b in others)
            if m < e_tl.y:
                dy = m - e_tl.y - margin_mm
        else:  # bottom
            m = max(b[1].y for b in others)
            if m > e_br.y:
                dy = m - e_br.y + margin_mm
        if dx == 0.0 and dy == 0.0:
            continue
        c = solved[bref]
        c.pos = Point(c.pos.x + dx, c.pos.y + dy)
        if c.body_center is not None:
            c.body_center = Point(c.body_center.x + dx, c.body_center.y + dy)
        for pad in c.pads:
            pad.pos = Point(pad.pos.x + dx, pad.pos.y + dy)
        shifted.append(bref)
    return shifted


def _compose_artifacts(
    loaded_artifacts,
    *,
    spacing_mm: float,
    rotation_step_deg: float,
    parent_definition: SubCircuitDefinition | None = None,
    pcb_path: Path | None = None,
    cfg: dict[str, Any] | None = None,
    seed: int = 0,
    seed_area_overhead: float = 2.5,
    seed_aspect_target: float = 1.0,
    seed_size_override: tuple[float, float] | None = None,
    manual_layout: ManualLayout | None = None,
) -> tuple[ParentCompositionState, list[dict[str, Any]]]:
    """Compose loaded artifacts into a parent composition snapshot.

    Replaces the legacy 3-iteration constraint-placement loop with a
    single pass through the unified ``PlacementSolver``: each artifact
    becomes a synthetic block component carrying its sparse blocker
    set, child attachment constraints become block-level zone entries
    with anchor offsets, and the solver picks positions and rotations
    in one shot.
    """
    from kicraft.autoplacer.config import discover_project_config, load_project_config
    import logging

    logger = logging.getLogger(__name__)

    # Merge caller-supplied cfg over project config so explicit overrides
    # (e.g., opposite_side_attraction_k from the parent route command,
    # connector_edge_inset_mm tuning) survive. Without this merge,
    # load_project_config used to clobber whatever the caller passed in.
    user_cfg = dict(cfg or {})
    cfg = dict(user_cfg)
    component_zones: dict[str, Any] = cfg.get("component_zones", {})
    parent_local: dict[str, Component] = {}
    # Standard form-factor scaffold (replace & rewire, compose half). Resolved up
    # front so its connector refs are known before extraction: the standard
    # headers are placed parent-local at their fixed template positions (injected
    # locked, below), so they must NOT also arrive via a leaf or the loose-
    # connector wrap (either would duplicate the ref and collide at compose).
    # None -> dormant for every other board and while enforcement is off.
    from kicraft.form_factors.compose_scaffold import (
        resolve_scaffold as _resolve_ff_scaffold,
    )

    _ff_scaffold = _resolve_ff_scaffold(cfg) if manual_layout is None else None
    _ff_refs: set[str] = set(_ff_scaffold.components) if _ff_scaffold is not None else set()

    if pcb_path:
        # No try/except here: a project-config parse error or a
        # parent_local extraction failure leaves the solver without
        # mounting-hole keepouts, edge-connector pins, etc. -- which
        # silently produces broken placements (items_not_allowed
        # violations, leaves over mounting holes). Better to fail loudly
        # at the cfg loading boundary than to ship a bad layout.
        project_dir = Path(pcb_path).resolve().parent
        cfg_file = discover_project_config(project_dir)
        if cfg_file is not None:
            project_cfg = load_project_config(str(cfg_file))
            cfg = {**project_cfg, **user_cfg}
            component_zones = cfg.get("component_zones", {})
            # Re-resolve against the merged project cfg (the scaffold gate reads
            # form_factor_enforce / form_factor_standard, which live there).
            _ff_scaffold = _resolve_ff_scaffold(cfg) if manual_layout is None else None
            _ff_refs = set(_ff_scaffold.components) if _ff_scaffold is not None else set()
        # Drop a leaf made ENTIRELY of the standard headers -- the scaffold owns
        # their placement. (The reconcile consolidates the headers onto one
        # sheet, so this leaf is exactly them.)
        if _ff_refs:
            loaded_artifacts = [
                art
                for art in loaded_artifacts
                if not (
                    set(art.layout.components)
                    and set(art.layout.components) <= _ff_refs
                )
            ]
        parent_local = extract_parent_local_components(
            str(pcb_path),
            loaded_artifacts,
            allowlist=_resolve_parent_local_allowlist(
                component_zones, loaded_artifacts
            ),
        )
        # The real header footprints (loose on the seed PCB now their leaf is
        # gone) drop out of parent_local before the wrap; the scaffold re-adds a
        # single locked copy per ref below, so the stamp moves the seed footprint
        # to the fixed position with no duplicate.
        for _r in _ff_refs:
            parent_local.pop(_r, None)
        # Lever 2.1: loose parent-level connectors flow through the leaf path
        # (edge-pinned as the board extremity), not the parent-local snap.
        loaded_artifacts, parent_local = _wrap_loose_parent_components_as_leaves(
            parent_local, loaded_artifacts
        )

    derived = derive_attachment_constraints(
        loaded_artifacts,
        parent_local,
        component_zones,
        cfg,
        rotation_step_deg=rotation_step_deg,
    )
    all_constraints = derived.constraints
    logger.info("composition: %d attachment constraints derived", len(all_constraints))

    # --- Build synthetic block components for each child artifact ---
    synthetic_refs = {
        i: synthetic_block_ref(i, art.sheet_name)
        for i, art in enumerate(loaded_artifacts)
    }
    block_zones, allowed_rotations = attachment_constraints_to_zones(
        derived, synthetic_refs, list(loaded_artifacts)
    )
    # Project-level override for the layer-intent heuristic. Sheet
    # names listed here force can_overlap_sparse to treat the leaf as
    # having no front-side copper intent, so SMT-on-front candidates
    # may stack on it. Generalises the THT-back-anchor case (battery
    # holders, terminal blocks) without per-project hardcoding -- any
    # project lists its own offending sheet names. Default empty: the
    # heuristic alone handles the common shadow-PTH case.
    _back_through_hole_leaves = set(
        cfg.get("parent_placement", {}).get("backside_through_hole_leaves", []) or []
    )
    synthetic_comps: dict[str, Component] = {}
    for i, art in enumerate(loaded_artifacts):
        ref = synthetic_refs[i]
        rot = float(block_zones.get(ref, {}).get("rotation", 0.0))
        spec = derived.child_specs.get(i)
        rotation_models = spec.models if spec is not None else None
        comp = artifact_to_component(
            art, ref=ref, rotation=rot, rotation_models=rotation_models
        )
        if ref in allowed_rotations:
            comp.allowed_rotations = list(allowed_rotations[ref])
        if art.sheet_name in _back_through_hole_leaves:
            comp.block_force_back_only = True
        synthetic_comps[ref] = comp

    # Parent-local components (mounting holes etc.) join the same solver
    # state. They keep their loaded positions; _snap_parent_local applies
    # the exact constraint-target snap after solve.
    # Standard form-factor enforcement (replace & rewire, compose half): a
    # validated standard pins the parent to the standard's exact outline (below)
    # and locks its connectors at their fixed board positions, so the solver
    # auto-places every leaf/local around them. The scaffold was resolved up
    # front (its refs drove the leaf/parent-local exclusion above); here we add
    # the single locked copy per header ref. The stamp then moves the real seed
    # footprint (matched by ref) to this fixed pos + rotation.
    if _ff_scaffold is not None:
        for _ref, _comp in _ff_scaffold.components.items():
            parent_local[_ref] = _comp

    for ref, comp in parent_local.items():
        synthetic_comps[ref] = comp

    # Brief-requested outline shape (autoplacer.json ``board_outline``) with a
    # size target bounds the seed to its largest inscribable content rect, so
    # placement happens INSIDE the requested ⌀/size instead of the shape being
    # circumscribe-rejected around a sprawled rectangle at stamp time.
    _shape_seed_cap = None
    if manual_layout is None and _ff_scaffold is None:
        _board_outline_req = cfg.get("board_outline")
        if isinstance(_board_outline_req, dict):
            _shape_seed_cap = inscribed_rect_bound(
                _board_outline_req, max(0.1, seed_aspect_target)
            )
    seed_w, seed_h = _seed_outline_dimensions(
        loaded_artifacts,
        derived,
        spacing_mm,
        area_overhead=seed_area_overhead,
        aspect_target=seed_aspect_target,
        seed_cap=_shape_seed_cap,
    )
    # Fix 2 (parent-compose compactness): the candidate search's pass-2 re-fit
    # passes a right-sized seed measured from THIS placement's pass 1, replacing
    # the area-basis estimate above. Only ever supplied for auto rect/parametric
    # boards (a shaped/ff/manual seed is authoritative, never an estimate).
    if seed_size_override is not None:
        seed_w, seed_h = seed_size_override
    if _ff_scaffold is not None:
        # Place inside the standard frame, not a content-derived one.
        seed_w, seed_h = _ff_scaffold.width_mm, _ff_scaffold.height_mm

    parent_subcircuit = parent_definition or _synthetic_parent_definition(loaded_artifacts)
    interconnect_nets = infer_interconnect_nets_pre_placement(
        parent_subcircuit, loaded_artifacts, synthetic_refs
    )

    state_in = BoardState(
        components=synthetic_comps,
        nets=interconnect_nets,
        traces=[],
        vias=[],
        silkscreen=[],
        board_outline=(Point(0.0, 0.0), Point(seed_w, seed_h)),
    )
    # Forward project-level cfg so the solver sees connector_edge_inset_mm,
    # edge_margin_mm, force_attract_k, etc. parent_placement is an optional
    # override layer for parent-specific tuning. component_zones is forced
    # to block-level only so the leaf-pad warning doesn't fire and so the
    # solver doesn't try to pin J1/J2/J3 directly (those flow through
    # attachment_constraints_to_zones into the synthetic block zones).
    #
    # parent_keep_in_rects: hard-keepout zones derived from parent-local
    # attachment constraints (e.g. mounting-hole inward_keep_in_mm). The
    # solver's _resolve_keep_in_rects pass pushes any unlocked component
    # whose bbox enters one of these zones back toward the board center.
    # Without this, fast-cfg / SA-disabled runs were producing layouts
    # with leaves stamped on top of mounting-hole keep-ins -> stamped
    # DRC items_not_allowed violations (14 on a recent failed candidate).
    parent_keep_in_specs = [
        {"ref": c.ref, "margin_mm": float(c.inward_keep_in_mm)}
        for c in derived.parent_local_constraints
        if c.ref in parent_local
    ]
    # Stock-footprint load instructions for user mounting holes without
    # a backing H-ref; populated by the manual branch, executed by the
    # stamp subprocess. Empty in auto mode.
    synthesized_footprints: list[dict[str, Any]] = []
    # Fix 2: a right-sized seed measured from THIS placement, when pass 1
    # over-provisioned the interior. Set only on the auto (solver) path for
    # rect/parametric boards; stays None for manual/ff/shaped/override runs.
    _refit_seed: tuple[float, float] | None = None
    if manual_layout is not None:
        # Manual mode: user-supplied placements + outline. Skip the solver
        # and the auto outline-fit pass entirely. Validation, stamping and
        # routing run unchanged on these placements.
        manual_by_path = manual_layout.placement_by_path()
        missing = [
            art.instance_path
            for art in loaded_artifacts
            if art.instance_path not in manual_by_path
        ]
        if missing:
            raise ValueError(
                "manual layout missing placements for instance paths: "
                + ", ".join(missing)
            )
        placements_dict = {
            art.instance_path: ChildArtifactPlacement(
                artifact=art,
                origin=manual_by_path[art.instance_path].origin,
                rotation=manual_by_path[art.instance_path].rotation,
            )
            for art in loaded_artifacts
        }
        # Honour user-supplied parent-local positions when present; else
        # keep the extracted positions and let _snap_parent_local apply
        # constraint targets within the manual outline.
        parent_local_solved = {
            ref: copy.deepcopy(comp) for ref, comp in parent_local.items()
        }
        manual_pl_by_ref = manual_layout.parent_local_by_ref()
        for ref, mpl in manual_pl_by_ref.items():
            comp = parent_local_solved.get(ref)
            if comp is not None:
                comp.pos = mpl.pos

        # Editor mounting-hole panel: holes map onto the parent's
        # existing parent-local mounting-hole footprints in alphabetical
        # ref order (so H4 < H86 etc.), overriding each paired
        # component's position with the user's chosen corner+inset.
        # SURPLUS holes (the common case: schematics rarely carry H
        # refs) are synthesized: a parent-local Component here for
        # validation/keep-ins, plus a stock-footprint load instruction
        # the stamp subprocess executes (see _parent_stamp_subprocess
        # synthesize_footprints). All are marked user-positioned so the
        # constraint snap below leaves them alone.
        user_positioned_refs: set[str] = set()
        gui_holes = sorted(
            getattr(manual_layout, "mounting_holes", []) or [],
            key=lambda h: h.index,
        )
        if gui_holes:
            from kicraft.autoplacer.brain.types import Layer as _Layer
            from kicraft.layout_editor.holes import (
                plan_mounting_holes,
                require_stock_mounting_hole_lib,
                screw_spec,
            )

            mh_refs = sorted(
                ref for ref, comp in parent_local_solved.items()
                if _is_mounting_hole_ref(ref, comp)
            )
            taken_refs: set[str] = set(parent_local_solved)
            for art in loaded_artifacts:
                taken_refs.update(art.metadata.get("component_refs") or [])
            mapped_holes, synth_holes = plan_mounting_holes(
                gui_holes, mh_refs, taken_refs
            )
            for hole, ref in mapped_holes:
                comp = parent_local_solved.get(ref)
                if comp is None:
                    continue
                _move_component_to(comp, hole.pos)
                user_positioned_refs.add(ref)
            if synth_holes:
                synth_lib_dir = require_stock_mounting_hole_lib()
                for hole, ref in synth_holes:
                    spec = screw_spec(getattr(hole, "screw", None))
                    parent_local_solved[ref] = Component(
                        ref=ref,
                        value=spec.screw,
                        pos=Point(hole.pos.x, hole.pos.y),
                        rotation=0.0,
                        layer=_Layer.FRONT,
                        width_mm=spec.courtyard_mm,
                        height_mm=spec.courtyard_mm,
                        pads=[],
                        locked=True,
                        kind="mounting_hole",
                        body_center=Point(hole.pos.x, hole.pos.y),
                    )
                    user_positioned_refs.add(ref)
                    synthesized_footprints.append(
                        {
                            "ref": ref,
                            "x": hole.pos.x,
                            "y": hole.pos.y,
                            "lib_dir": str(synth_lib_dir),
                            "fp_name": spec.fp_name,
                            "screw": spec.screw,
                        }
                    )
                print(
                    f"[manual-layout] synthesized {len(synth_holes)} mounting "
                    f"hole footprint(s): "
                    + ", ".join(
                        f"{e['ref']}={e['fp_name']}" for e in synthesized_footprints
                    )
                )

        solver_phase_timings = {}

        placed_child_bboxes, placed_envelopes, _ignored_anchors, transformed_by_index = (
            _post_solve_geometry(placements_dict, loaded_artifacts)
        )
        child_anchor_positions = _resolve_constraint_anchor_positions(
            derived, placements_dict, loaded_artifacts, transformed_by_index, parent_local_solved
        )
        exact_outline = manual_layout.board_outline
        _snap_parent_local(
            parent_local_solved,
            [
                c for c in derived.parent_local_constraints
                if c.ref not in user_positioned_refs
            ],
            exact_outline,
        )
    else:
        solver_cfg = {
            **cfg,
            **cfg.get("parent_placement", {}),
            "component_zones": dict(block_zones),
            "placement_clearance_mm": spacing_mm,
            "clearance_mm": spacing_mm,
            "parent_keep_in_rects": parent_keep_in_specs,
        }
        solver = PlacementSolver(state_in, config=solver_cfg, seed=seed)
        solved = solver.solve()
        solver_phase_timings = dict(getattr(solver, "last_solve_phase_timings", {}))

        # 3D step-1: Instrument the three compose post-passes to count when
        # they change anything. The goal is to eventually move edge-extremity
        # into the parent solver itself so these post-passes become
        # verify-only. This instrumentation reports the counts so we know
        # which boards still need the compose mutations.
        _3d_slide_changes = 0
        _3d_extremal_changes = 0
        _3d_courtyard_resolved = 0

        # Slide any edge-constrained block whose free axis drifted outside the
        # rest of the cluster's perpendicular span. The solver pins X (or Y)
        # to the board edge but lets the free axis float; a leaf parked in a
        # corner inflates the final outline because the orthogonal sides snap
        # to include it. Bringing it back inside the cluster span lets
        # _compute_final_outline shrink the board.
        _pre_slide_positions = {r: (c.pos.x, c.pos.y) for r, c in solved.items()}
        _slide_constrained_to_cluster(solved, derived, synthetic_refs)
        _3d_slide_changes = sum(
            1 for r, c in solved.items()
            if r in _pre_slide_positions
            and (abs(c.pos.x - _pre_slide_positions[r][0]) > 1e-6
                 or abs(c.pos.y - _pre_slide_positions[r][1]) > 1e-6)
        )

        # Make each edge-zoned block the extremity on its side so its connector
        # defines the board edge and stays flush (KC-S8PC37 J1) instead of being
        # stranded inboard by another block edging past it.
        if cfg.get("connector_edge_block_extremity", True):
            _pre_extremal_positions = {r: (c.pos.x, c.pos.y) for r, c in solved.items()}
            _shifted = _ensure_edge_blocks_extremal(solved, block_zones)
            _3d_extremal_changes = sum(
                1 for r, c in solved.items()
                if r in _pre_extremal_positions
                and (abs(c.pos.x - _pre_extremal_positions[r][0]) > 1e-6
                     or abs(c.pos.y - _pre_extremal_positions[r][1]) > 1e-6)
            )
            if _shifted:
                logger.info("composition: shifted edge blocks to extremity: %s", _shifted)

        # Re-run the solver's courtyard-separation pass as the GENUINE last
        # geometry step. The solver runs it at the end of solve(), but the
        # cluster-slide and edge-extremity shifts above move blocks AFTER that
        # -- aligning two same-edge blocks to the same perpendicular extreme can
        # collapse the separation the solver had relied on, reintroducing a
        # same-side courtyards_overlap that nothing else re-resolves. Running it
        # here (only unlocked blocks move, along the smaller-overlap axis, so
        # edge flush + extremity are preserved) guarantees the stamped parent
        # has no same-side courtyard overlap.
        if cfg.get("resolve_courtyard_overlaps", True):
            _pre_courtyard_positions = {r: (c.pos.x, c.pos.y) for r, c in solved.items()}
            solver._resolve_courtyard_overlaps(solved)
            _3d_courtyard_resolved = sum(
                1 for r, c in solved.items()
                if r in _pre_courtyard_positions
                and (abs(c.pos.x - _pre_courtyard_positions[r][0]) > 1e-6
                     or abs(c.pos.y - _pre_courtyard_positions[r][1]) > 1e-6)
            )

        if _3d_slide_changes or _3d_extremal_changes or _3d_courtyard_resolved:
            print(
                f"[3d-instrument] post-pass changes: "
                f"slide={_3d_slide_changes} "
                f"extremal={_3d_extremal_changes} "
                f"courtyard={_3d_courtyard_resolved}",
                flush=True,
            )

        # --- Recover artifact placements from solver output ---
        placements_dict = placements_from_solved_state(solved, list(loaded_artifacts), synthetic_refs)
        parent_local_solved: dict[str, Component] = {
            ref: solved[ref] for ref in parent_local if ref in solved
        }

        # Build per-child geometry for outline + validation.
        placed_child_bboxes, placed_envelopes, _ignored_anchors, transformed_by_index = (
            _post_solve_geometry(placements_dict, loaded_artifacts)
        )
        child_anchor_positions = _resolve_constraint_anchor_positions(
            derived, placements_dict, loaded_artifacts, transformed_by_index, parent_local_solved
        )

        # Fix 2: measure a right-sized seed from this (pass-1) placement so the
        # candidate search can re-solve tighter. Only for the auto rect/parametric
        # path -- an override IS a re-fit already, and shaped/ff seeds are
        # authoritative (their own fit logic owns sizing).
        if (
            seed_size_override is None
            and _ff_scaffold is None
            and _shape_seed_cap is None
        ):
            _refit_seed = _refit_seed_from_placement(
                placed_child_bboxes, derived, spacing_mm, (seed_w, seed_h)
            )

        placed_bbox_list = [
            placed_child_bboxes[index] for index in sorted(placed_child_bboxes)
        ]
        if _ff_scaffold is not None:
            # Standard form factor: the outline IS the standard rect, not grown
            # from content. _repair_parent_outline early-returns on it
            # (state.outline_authoritative) and _validate_parent_geometry then
            # enforces exact containment fail-loud -- a design that doesn't fit
            # the standard is rejected, not silently up-sized.
            exact_outline = _ff_scaffold.outline
        else:
            exact_outline = _compute_final_outline(
                placed_bbox_list, all_constraints, child_anchor_positions, spacing_mm,
                edge_constrained_refs={
                    c.ref for c in all_constraints if c.target in ("edge", "corner")
                },
            edge_zoned_outline_sides=frozenset(
                side
                for c in all_constraints
                if c.target in ("edge", "corner")
                and not _is_mounting_hole_ref(c.ref)
                for side in (
                    [c.value]
                    if c.target == "edge"
                    and c.value in ("left", "right", "top", "bottom")
                    else [
                        s
                        for s in c.value.split("-")
                        if s in ("left", "right", "top", "bottom")
                    ]
                )
            ),
        )

        # Snap parent-local components to exact constraint coordinates.
        _snap_parent_local(
            parent_local_solved,
            derived.parent_local_constraints,
            exact_outline,
        )

    # --- Build CompositionEntry list + transformed_payloads in artifact order ---
    entries: list[CompositionEntry] = []
    transformed_payloads: list[dict[str, Any]] = []
    child_artifact_placements: list[ChildArtifactPlacement] = []
    final_transformed_by_index: dict[int, Any] = {}
    for child_index, artifact in enumerate(loaded_artifacts):
        placement = placements_dict.get(artifact.instance_path)
        if placement is None:
            raise ValueError(
                f"Solver did not produce a placement for {artifact.instance_path}"
            )
        transformed = transformed_by_index.get(child_index)
        if transformed is None:
            transformed = transform_loaded_artifact(
                artifact, origin=placement.origin, rotation=placement.rotation
            )
        final_transformed_by_index[child_index] = transformed
        entry = CompositionEntry(
            artifact_dir=artifact.artifact_dir,
            sheet_name=artifact.sheet_name,
            instance_path=artifact.instance_path,
            origin=placement.origin,
            rotation=placement.rotation,
            transformed_bbox=transformed.instance.transformed_bbox,
            component_count=len(transformed.transformed_components),
            trace_count=len(transformed.transformed_traces),
            via_count=len(transformed.transformed_vias),
            anchor_count=len(transformed.transformed_anchors),
        )
        entries.append(entry)
        child_artifact_placements.append(placement)
        transformed_payloads.append(
            {
                "artifact": artifact_debug_dict(artifact),
                "transformed": transformed_debug_dict(transformed),
                "summary": transformed_summary(transformed),
            }
        )

    composition = build_parent_composition(
        parent_subcircuit,
        child_artifact_placements=child_artifact_placements,
        board_outline=exact_outline,
        local_components=parent_local_solved,
    )

    copper_manifest = build_copper_manifest(
        composed_children=composition.composed_children,
        final_child_bboxes={
            loaded_artifacts[index].instance_path: (
                (bbox[0].x, bbox[0].y),
                (bbox[1].x, bbox[1].y),
            )
            for index, bbox in placed_child_bboxes.items()
        },
    )

    for ref, comp in parent_local_solved.items():
        if ref in composition.board_state.components:
            composition.board_state.components[ref] = copy.deepcopy(comp)
        if ref in composition.hierarchy_state.local_components:
            composition.hierarchy_state.local_components[ref] = copy.deepcopy(comp)

    # --- Validation block (preserved from legacy flow) ---
    import itertools

    edge_attachment_satisfied: dict[str, bool] = {}
    mounting_hole_keep_in_satisfied: dict[str, bool] = {}

    for c in all_constraints:
        expected_x = None
        expected_y = None
        actual_x = None
        actual_y = None

        min_pt, max_pt = exact_outline

        if c.source == "child_artifact":
            anchor = child_anchor_positions.get(c.ref)
            if anchor is None:
                continue
            actual_x = anchor.x
            actual_y = anchor.y
        else:
            comp = parent_local_solved.get(c.ref)
            if not comp:
                continue
            if comp.pads:
                actual_x = sum(pad.pos.x for pad in comp.pads) / len(comp.pads)
                actual_y = sum(pad.pos.y for pad in comp.pads) / len(comp.pads)
            else:
                anchor = comp.body_center if comp.body_center is not None else comp.pos
                actual_x = anchor.x
                actual_y = anchor.y

        if c.target == "edge":
            if c.value in ("left", "right"):
                expected_x = edge_anchor_target_coordinate(c.value, c, min_pt, max_pt)
            elif c.value in ("top", "bottom"):
                expected_y = edge_anchor_target_coordinate(c.value, c, min_pt, max_pt)
        elif c.target == "corner":
            corner_sides = {
                "top-left": ("left", "top"),
                "top-right": ("right", "top"),
                "bottom-left": ("left", "bottom"),
                "bottom-right": ("right", "bottom"),
            }.get(c.value)
            if corner_sides is not None:
                x_side, y_side = corner_sides
                expected_x = edge_anchor_target_coordinate(x_side, c, min_pt, max_pt)
                expected_y = edge_anchor_target_coordinate(y_side, c, min_pt, max_pt)
        elif c.target == "zone" and c.value == "bottom":
            expected_y = max_pt.y - c.inward_keep_in_mm

        ok_x = expected_x is None or abs(actual_x - expected_x) <= 1e-3
        ok_y = expected_y is None or abs(actual_y - expected_y) <= 1e-3
        ok = ok_x and ok_y

        edge_attachment_satisfied[c.ref] = ok

        is_hole = c.ref.startswith("H") or (
            c.inward_keep_in_mm > 0 and "hole" in c.ref.lower()
        )
        if is_hole:
            mounting_hole_keep_in_satisfied[c.ref] = ok

    # --- Edge connector orientation gate -------------------------------
    # Verify every edge-pinned connector with a detectable mouth ended up
    # facing OUTWARD on its assigned edge. _filter_rotations_for_connector_opening
    # should have guaranteed this, so a violation here means an unsatisfiable
    # multi-connector leaf or an undetected regression -- surface it loudly
    # rather than silently shipping a board whose USB port faces inward.
    misoriented_connectors: list[str] = []
    for c in all_constraints:
        if c.target != "edge" or c.source != "child_artifact" or c.child_index is None:
            continue
        transformed = final_transformed_by_index.get(int(c.child_index))
        if transformed is None:
            continue
        comp = transformed.transformed_components.get(c.ref)
        if comp is None or comp.opening_direction is None:
            continue
        board_opening = opening_board_angle(comp.opening_direction, comp.rotation)
        want = edge_outward_angle(comp.layer, c.value)
        if not angles_close(board_opening, want):
            misoriented_connectors.append(c.ref)
    if misoriented_connectors:
        logger.warning(
            "Edge connector(s) %s face INWARD after composition (mouth not at "
            "the board edge) -- the port may be unmateable. Check that the "
            "leaf rotation candidates were not over-constrained and that the "
            "connector's opening_direction was detected correctly.",
            ", ".join(sorted(misoriented_connectors)),
        )

    same_side_overlap_conflicts: list[tuple[str, str]] = []
    tht_keepout_violations: list[tuple[str, str]] = []

    ordered_entry_indices = [
        next(
            index
            for index, artifact in enumerate(loaded_artifacts)
            if artifact.instance_path == entry.instance_path
        )
        for entry in entries
    ]
    for i, j in itertools.combinations(range(len(placed_envelopes)), 2):
        item_a = placed_envelopes[i]
        item_b = placed_envelopes[j]
        env_a = item_a["envelopes"]
        env_b = item_b["envelopes"]
        art_a = loaded_artifacts[ordered_entry_indices[i]]
        art_b = loaded_artifacts[ordered_entry_indices[j]]
        rect_a = placed_child_bboxes[ordered_entry_indices[i]]
        rect_b = placed_child_bboxes[ordered_entry_indices[j]]

        if not _bbox_disjoint(rect_a, rect_b):
            blocker_a = item_a.get("blocker_set")
            blocker_b = item_b.get("blocker_set")
            if blocker_a is not None and blocker_b is not None:
                overlap_ok = can_overlap_sparse(
                    blocker_a,
                    item_a["origin"],
                    item_a["rotation"],
                    blocker_b,
                    item_b["origin"],
                    item_b["rotation"],
                )
            else:
                overlap_ok = can_overlap(env_a, env_b)
            if not overlap_ok:
                a_label = getattr(
                    art_a,
                    "label",
                    getattr(art_a, "slug", getattr(art_a, "sheet_name", f"child[{i}]")),
                )
                b_label = getattr(
                    art_b,
                    "label",
                    getattr(art_b, "slug", getattr(art_b, "sheet_name", f"child[{j}]")),
                )

                a_front, a_back, a_tht = env_a
                b_front, b_back, b_tht = env_b

                if (
                    not _rect_lists_disjoint(a_tht, b_front)
                    or not _rect_lists_disjoint(a_tht, b_back)
                    or not _rect_lists_disjoint(b_tht, a_front)
                    or not _rect_lists_disjoint(b_tht, a_back)
                    or not _rect_lists_disjoint(a_tht, b_tht)
                ):
                    tht_keepout_violations.append((a_label, b_label))
                elif (
                    not _rect_lists_disjoint(a_front, b_front)
                    or not _rect_lists_disjoint(a_back, b_back)
                    or (
                        blocker_a is not None
                        and blocker_b is not None
                        and dominant_blocker_side(blocker_a) in {"front", "back"}
                        and dominant_blocker_side(blocker_a)
                        == dominant_blocker_side(blocker_b)
                    )
                ):
                    same_side_overlap_conflicts.append((a_label, b_label))

    validation_data = {
        "edge_attachment_satisfied": edge_attachment_satisfied,
        "mounting_hole_keep_in_satisfied": mounting_hole_keep_in_satisfied,
        "same_side_overlap_conflicts": same_side_overlap_conflicts,
        "tht_keepout_violations": tht_keepout_violations,
        "constraint_count": len(all_constraints),
        "parent_local_count": len(parent_local),
    }
    unsatisfied_edges = sum(
        1 for ok in edge_attachment_satisfied.values() if not ok
    )
    logger.info(
        "composition: %d constraints, %d unsatisfied edges, %d overlap conflicts, "
        "%d THT violations",
        len(all_constraints),
        unsatisfied_edges,
        len(same_side_overlap_conflicts),
        len(tht_keepout_violations),
    )

    parent_local_keep_in_rects: list[tuple[Point, Point]] = []
    for constraint in derived.parent_local_constraints:
        comp = parent_local_solved.get(constraint.ref)
        if comp is None:
            continue
        bbox_min, bbox_max = _component_geometry_bbox(comp)
        keep_in = (
            Point(
                bbox_min.x - constraint.inward_keep_in_mm,
                bbox_min.y - constraint.inward_keep_in_mm,
            ),
            Point(
                bbox_max.x + constraint.inward_keep_in_mm,
                bbox_max.y + constraint.inward_keep_in_mm,
            ),
        )
        parent_local_keep_in_rects.append(keep_in)

    # Synthesized mounting holes have no attachment constraint, so they
    # get their keep-in rect here: a square reaching
    # mounting_holes.keepout.size_mm from the hole center (the config's
    # documented semantic), stamped as a rule-area so FreeRouting can't
    # route through the screw head.
    if synthesized_footprints:
        _mh_keepout_mm = float(
            ((cfg.get("mounting_holes") or {}).get("keepout") or {}).get(
                "size_mm", 4.0
            )
        )
        for _entry in synthesized_footprints:
            parent_local_keep_in_rects.append(
                (
                    Point(_entry["x"] - _mh_keepout_mm, _entry["y"] - _mh_keepout_mm),
                    Point(_entry["x"] + _mh_keepout_mm, _entry["y"] + _mh_keepout_mm),
                )
            )

    outline_w = exact_outline[1].x - exact_outline[0].x
    outline_h = exact_outline[1].y - exact_outline[0].y
    packing_metadata: dict[str, Any] = {
        "strategy": "unified_solver",
        "board_width_mm": round(outline_w, 2),
        "board_height_mm": round(outline_h, 2),
    }
    # Area-waste visibility (PCB area-compaction plan, Phase 0): parent-level
    # utilization/aspect metrics ride in packing_metadata -> parent_pipeline.json
    # (state.packing_metadata + composition.debug.packing_metadata), where the
    # autoexperiment status writer picks them up for run_status.json.
    from kicraft.autoplacer.brain.placement_utils import board_utilization_metrics

    packing_metadata["board_metrics"] = board_utilization_metrics(
        composition.board_state.components, outline_w, outline_h
    )

    project_dir = ""
    if loaded_artifacts:
        try:
            project_dir = str(
                Path(loaded_artifacts[0].artifact_dir).resolve().parents[2]
            )
        except IndexError:
            project_dir = ""

    state = ParentCompositionState(
        project_dir=project_dir,
        spacing_mm=spacing_mm,
        entries=entries,
        bounding_box=composition.board_state.board_outline,
        parent_sheet_name=composition.hierarchy_state.subcircuit.id.sheet_name,
        parent_instance_path=composition.hierarchy_state.subcircuit.id.instance_path,
        component_count=composition.component_count,
        trace_count=composition.trace_count,
        via_count=composition.via_count,
        interconnect_net_count=len(composition.hierarchy_state.interconnect_nets),
        inferred_interconnect_net_count=len(composition.inferred_interconnect_nets),
        preserved_child_trace_count=composition.trace_count,
        preserved_child_via_count=composition.via_count,
        expected_preserved_child_trace_count=composition.trace_count,
        expected_preserved_child_via_count=composition.via_count,
        routed_total_trace_count=composition.trace_count,
        routed_total_via_count=composition.via_count,
        added_parent_trace_count=0,
        added_parent_via_count=0,
        packing_metadata=packing_metadata,
        geometry_validation=validation_data,
        score_total=composition.score.total if composition.score else 0.0,
        score_breakdown=dict(composition.score.breakdown) if composition.score else {},
        score_notes=list(composition.score.notes) if composition.score else [],
        composition_notes=list(composition.notes),
        copper_manifest=copper_manifest,
        composition=composition,
        parent_local_keep_in_rects=list(parent_local_keep_in_rects),
        edge_constrained_refs=frozenset(
            c.ref for c in all_constraints if c.target in ("edge", "corner")
        ),
        edge_zoned_outline_sides=frozenset(
            side
            for c in all_constraints
            if c.target in ("edge", "corner") and not _is_mounting_hole_ref(c.ref)
            for side in (
                [c.value]
                if c.target == "edge" and c.value in ("left", "right", "top", "bottom")
                else [s for s in c.value.split("-") if s in ("left", "right", "top", "bottom")]
            )
        ),
        manual_outline=(
            manual_layout.outline.to_dict() if manual_layout is not None else None
        ),
        # A standard form-factor scaffold's rect IS the spec: never grow it
        # (a design that doesn't fit the standard is rejected, not up-sized).
        outline_authoritative=(_ff_scaffold is not None),
        # Brief-captured outline shape (autoplacer.json ``board_outline``), only
        # when no manual layout supplies an authoritative outline. Resolved to
        # Edge.Cuts geometry by the compose pipeline (Phase 3).
        requested_shape=(
            cfg.get("board_outline")
            if manual_layout is None and isinstance(cfg.get("board_outline"), dict)
            else None
        ),
        synthesized_footprints=list(synthesized_footprints),
        refit_seed=_refit_seed,
    )
    state.phase_timings.update(solver_phase_timings)
    return state, transformed_payloads


def _save_composition_snapshot(
    output_path: Path,
    state: ParentCompositionState,
    transformed_payloads: list[dict[str, Any]],
) -> str:
    """Write a composition snapshot JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    composition_payload = {
        "summary": (
            f"{state.parent_sheet_name} "
            f"[{state.parent_instance_path}] "
            f"children={len(state.entries)} "
            f"components={state.component_count} "
            f"traces={state.trace_count} "
            f"vias={state.via_count} "
            f"interconnects={state.interconnect_net_count} "
            f"score={state.score_total:.1f} "
            f"size={state.width_mm:.1f}x{state.height_mm:.1f}mm"
        ),
        "debug": {
            "parent": {
                "sheet_name": state.parent_sheet_name,
                "instance_path": state.parent_instance_path,
            },
            "child_count": len(state.entries),
            "component_count": state.component_count,
            "trace_count": state.trace_count,
            "via_count": state.via_count,
            "interconnect_net_count": state.interconnect_net_count,
            "inferred_interconnect_net_count": state.inferred_interconnect_net_count,
            "score": {
                "total": state.score_total,
                "breakdown": dict(state.score_breakdown),
                "notes": list(state.score_notes),
            },
            "notes": list(state.composition_notes),
            "board_outline": state.to_dict()["bounding_box"],
            "packing_metadata": dict(state.packing_metadata),
            "geometry_validation": dict(state.geometry_validation),
        },
    }
    payload = {
        "composition": composition_payload,
        "state": state.to_dict(),
        "artifacts": transformed_payloads,
    }
    # Atomic write: autoexperiment reads this output via
    # _read_composer_quality_score to extract the round score; a torn
    # mid-write read would crash the score-extraction path and reject
    # an otherwise-valid round.
    tmp_output = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp_output.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    tmp_output.replace(output_path)
    return str(output_path)


def _print_human_summary(
    loaded_artifacts,
    state: ParentCompositionState,
    transformed_payloads: list[dict[str, Any]],
    output_path: str | None,
) -> None:
    print("=== Subcircuit Composition ===")
    print(f"artifacts              : {len(loaded_artifacts)}")
    print(f"spacing_mm             : {state.spacing_mm:.2f}")
    print(f"parent                 : {state.parent_sheet_name}")
    print(f"parent_instance_path   : {state.parent_instance_path}")
    print(f"composition_mm         : {state.width_mm:.2f} x {state.height_mm:.2f}")
    print(f"components             : {state.component_count}")
    print(f"traces                 : {state.trace_count}")
    print(f"vias                   : {state.via_count}")
    print(f"interconnect_nets      : {state.interconnect_net_count}")
    print(f"inferred_interconnects : {state.inferred_interconnect_net_count}")
    print(f"score_total            : {state.score_total:.2f}")
    if output_path:
        print(f"output_json            : {output_path}")
    print()

    for artifact, transformed in zip(loaded_artifacts, transformed_payloads):
        print(f"- {artifact_summary(artifact)}")
        print(f"  artifact_dir : {artifact.artifact_dir}")
        print(f"  transformed  : {transformed['summary']}")
        print()

    if state.score_breakdown:
        print("score_breakdown:")
        for key, value in sorted(state.score_breakdown.items()):
            print(f"  - {key}: {value:.2f}")

    # Copper accounting summary
    if state.preserved_child_trace_count or state.added_parent_trace_count:
        print()
        print("copper_accounting:")
        print(
            f"  child_traces         : {state.preserved_child_trace_count}"
            f" / {state.expected_preserved_child_trace_count} preserved"
        )
        print(
            f"  child_vias           : {state.preserved_child_via_count}"
            f" / {state.expected_preserved_child_via_count} preserved"
        )
        print(f"  parent_traces        : +{state.added_parent_trace_count} new")
        print(f"  parent_vias          : +{state.added_parent_via_count} new")
        print(
            f"  total_routed         : {state.routed_total_trace_count} traces,"
            f" {state.routed_total_via_count} vias"
        )
        print()

    if state.score_notes:
        print("score_notes:")
        for note in state.score_notes:
            print(f"  - {note}")
        print()

    if state.composition_notes:
        print("composition_notes:")
        for note in state.composition_notes:
            print(f"  - {note}")
        print()


def _json_payload(
    loaded_artifacts,
    state: ParentCompositionState,
    transformed_payloads: list[dict[str, Any]],
    output_path: str | None,
) -> dict[str, Any]:
    composition_payload = {
        "summary": (
            f"{state.parent_sheet_name} "
            f"[{state.parent_instance_path}] "
            f"children={len(state.entries)} "
            f"components={state.component_count} "
            f"traces={state.trace_count} "
            f"vias={state.via_count} "
            f"interconnects={state.interconnect_net_count} "
            f"score={state.score_total:.1f} "
            f"size={state.width_mm:.1f}x{state.height_mm:.1f}mm"
        ),
        "debug": {
            "parent": {
                "sheet_name": state.parent_sheet_name,
                "instance_path": state.parent_instance_path,
            },
            "child_count": len(state.entries),
            "component_count": state.component_count,
            "trace_count": state.trace_count,
            "via_count": state.via_count,
            "interconnect_net_count": state.interconnect_net_count,
            "inferred_interconnect_net_count": state.inferred_interconnect_net_count,
            "score": {
                "total": state.score_total,
                "breakdown": dict(state.score_breakdown),
                "notes": list(state.score_notes),
            },
            "notes": list(state.composition_notes),
            "board_outline": state.to_dict()["bounding_box"],
        },
    }
    return {
        "artifact_count": len(loaded_artifacts),
        "composition": composition_payload,
        "state": state.to_dict(),
        "output_json": output_path,
        "artifacts": transformed_payloads,
    }


# ---------------------------------------------------------------------------
# Parent board stamping, routing, validation, rendering, and artifact persistence
# ---------------------------------------------------------------------------


def _compact_routed_validation(validation: dict[str, Any]) -> dict[str, Any]:
    """Strip the ~100 KB DRC report_text, keep the diagnostic signal.

    Preserved fields: accepted, rejection_reasons, obviously_illegal_routed_geometry,
    DRC violation counts, first N violation excerpts (~1 KB), track/anchor
    summaries. Callers that need the full report can still run
    `validate_routed_board` themselves.
    """
    if not isinstance(validation, dict):
        return {}
    drc = validation.get("drc", {}) or {}
    report_text = drc.get("report_text", "") or ""
    drc_slim: dict[str, Any] = {
        k: v for k, v in drc.items() if k != "report_text"
    }
    if report_text:
        # Retain only the first few violation blocks for post-hoc analysis.
        # A KiCad DRC report is line-oriented; "[category]" lines introduce
        # each violation. Take the first 800 chars, which is usually enough
        # to identify the dominant failure mode without bloating the JSON.
        drc_slim["report_excerpt"] = report_text[:800]
    out = {
        "accepted": bool(validation.get("accepted", False)),
        "rejection_reasons": list(validation.get("rejection_reasons", []) or []),
        "obviously_illegal_routed_geometry": bool(
            validation.get("obviously_illegal_routed_geometry", False)
        ),
        "malformed_board_geometry": bool(
            validation.get("malformed_board_geometry", False)
        ),
        "track_summary": dict(validation.get("track_summary", {}) or {}),
        "anchor_summary": {
            k: v
            for k, v in (validation.get("anchor_summary", {}) or {}).items()
            if k
            in {
                "expected_count",
                "actual_count",
                "required_count",
                "all_required_present",
            }
        },
        "drc": drc_slim,
    }
    for k in (
        "footprint_internal_clearance_count",
        "footprint_internal_copper_edge_count",
    ):
        if k in validation:
            out[k] = validation[k]
    return out


# Outline repair + geometry validation + view rendering live in
# _compose_validate (Lever 2.5 split); re-exported (external API).
from kicraft.cli._compose_validate import (  # noqa: E402
    _render_parent_board_views,
    _repair_parent_outline,
    _requested_size_pair,
    _validate_parent_geometry,
    inscribed_rect_bound,
)


# stamp lives in _compose_stamp (Lever 2.5 split); re-exported.
from kicraft.cli._compose_stamp import (  # noqa: E402
    _PARENT_STAMP_SCRIPT_PATH,
    _stamp_parent_board,
)


def _promotable_strand_only(
    reasons: list[str],
    connector_strand_reasons: list[str],
    drc: dict[str, Any] | None,
) -> bool:
    """Whether a *rejected* routed parent should still be promoted as a routed
    but NOT-fab-ready board.

    True only when the board's sole defect is connector stranding -- every
    rejection reason is a recorded ``connector_stranded:*`` finding -- AND the
    board is electrically complete (no shorts, no unconnected nets). Such a
    board is fully routed and usable for inspection; failing it outright as
    ``rc=6 "no routed parent"`` (with no artifact) misrepresents a placement
    defect as an infra failure. Boards with any *other* defect (illegal routed
    geometry, unconnected nets, shorts) stay hard-rejected.
    """
    if not connector_strand_reasons:
        return False
    if any(reason not in connector_strand_reasons for reason in reasons):
        return False
    drc = drc or {}
    return not drc.get("shorts") and not drc.get("unconnected")


@dataclass(slots=True)
class CandidateRecord:
    """One trial in the layout search loop."""

    seed: int
    shorts: int
    score: float
    place_solve_ms: float
    stamp_ms: float
    stamp_drc_ms: float
    accepted: bool
    pcb_path: str
    bbox_h_mm: float = 0.0
    bbox_w_mm: float = 0.0
    outline_h_mm: float = 0.0
    outline_w_mm: float = 0.0
    breakdown: dict[str, float] = field(default_factory=dict)
    geometry_accepted: bool = False
    outside_component_count: int = 0
    outside_pad_count: int = 0
    phase_timings: dict[str, float] = field(default_factory=dict)
    # True when no outline shape was requested OR the requested shape
    # committed at stamp time for this candidate (see state.shape_fit).
    shape_fitted: bool = True
    # Stamp-time copper-to-board-edge DRC error count. A candidate with a
    # nonzero count ships a guaranteed fab-gate failure, so winner selection
    # hard-prefers clean candidates (self-eval 2026-07-17 T2).
    stamp_edge_clearance: int = 0
    # True when this candidate came from the pass-2 re-fit ("r" pass) rather
    # than the pass-1 area-basis seed. Surfaced as candidate_search.winner_refit
    # so the round scheduler can back off the re-fit after a routed rejection.
    refit: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "shorts": self.shorts,
            "score": self.score,
            "place_solve_ms": self.place_solve_ms,
            "stamp_ms": self.stamp_ms,
            "stamp_drc_ms": self.stamp_drc_ms,
            "accepted": self.accepted,
            "pcb_path": self.pcb_path,
            "bbox_h_mm": self.bbox_h_mm,
            "bbox_w_mm": self.bbox_w_mm,
            "outline_h_mm": self.outline_h_mm,
            "outline_w_mm": self.outline_w_mm,
            "breakdown": dict(self.breakdown),
            "geometry_accepted": self.geometry_accepted,
            "outside_component_count": self.outside_component_count,
            "outside_pad_count": self.outside_pad_count,
            "phase_timings": dict(self.phase_timings),
            "shape_fitted": self.shape_fitted,
            "stamp_edge_clearance": self.stamp_edge_clearance,
            "refit": self.refit,
        }


@dataclass(slots=True)
class SearchResult:
    """Outcome of _search_best_layout: winner state + per-candidate records."""

    winner_idx: int
    winner_seed: int
    winner_state: ParentCompositionState
    winner_payloads: list[dict[str, Any]]
    winner_pcb_path: Path
    candidates: list[CandidateRecord]
    total_search_ms: float


def _winner_key(c: "CandidateRecord") -> tuple:
    """Lexicographic winner ordering, in one place so every selection site
    (initial pick, strand-screen re-picks) agrees: a shape-fitted candidate
    beats any rect fallback (the circle IS the deliverable), then a
    stamp-edge-clean candidate beats one whose stamped board already violates
    copper-to-edge clearance (a guaranteed fab-gate failure the packing score
    otherwise PREFERS -- self-eval 2026-07-17 runs 02/09/11), then score."""
    return (c.shape_fitted, c.stamp_edge_clearance == 0, c.score)


def _net_dist_score(ratsnest_mm: float, scale_mm: float = 1000.0) -> float:
    """Compactness score (0..100, higher = tighter) from total ratsnest length.

    Bounded inverse rather than the old linear ``100 - 0.1*ratsnest`` clamp,
    which floored to 0 for any ratsnest > 1000mm -- killing the signal on the
    large/sprawled boards that need it. Matches the old slope near 0 (scale
    1000) but never flatlines, so a 1500mm candidate still ranks below 1200mm.
    """
    return 100.0 * scale_mm / (scale_mm + max(0.0, ratsnest_mm))


def _sprawl_penalty(
    outline_area_mm2: float, summed_courtyard_area_mm2: float
) -> tuple[float, float]:
    """Return ``(sprawl_ratio, penalty)`` for an outline vs the copper it holds.

    ``sprawl = outline_area / summed_courtyard_area`` (i.e. ``1 / packing``).
    The denominator is the SUMMED component courtyard area, NOT the area their
    spread spans: tight leaf clusters flung apart span ~= the whole outline
    (ratio ~1) and would never trip a span-based gate -- the exact
    big-board-tiny-parts sprawl we must catch. No penalty above ~33% packing
    (sprawl 3), ramping linearly to a 25-pt cap by ~12% packing (sprawl 8).
    """
    summed = max(1.0, summed_courtyard_area_mm2)
    sprawl = outline_area_mm2 / summed
    penalty = min(25.0, 5.0 * (sprawl - 3.0)) if sprawl > 3.0 else 0.0
    return sprawl, penalty


def _edge_demotion_candidates(
    loaded_artifacts, cfg: dict[str, Any] | None
) -> list[str]:
    """Strict edge/corner-zoned refs whose OWNING leaf could instead NEST
    inside another leaf's enclosed interior hole (PR-N4 of
    docs/plans/shaped-compose-leaf-nesting.md).

    A ref qualifies when it carries an edge/corner ``component_zones`` entry
    and its owning leaf's occupied bbox (plus nest margins) fits inside some
    OTHER leaf's interior hole. Pure geometry over the loaded artifacts --
    no side effects; deterministic output order.
    """
    from kicraft.autoplacer.brain.subcircuit_composer import (
        _blocker_occupied_rects,
        extract_leaf_blocker_set,
    )

    zones = (cfg or {}).get("component_zones") or {}
    pinned = sorted(
        ref for ref, zone in zones.items()
        if isinstance(zone, dict) and ("edge" in zone or "corner" in zone)
    )
    if not pinned or len(loaded_artifacts) < 2:
        return []
    margin = float((cfg or {}).get("nest_margin_mm", 1.0))

    blockers = []
    for art in loaded_artifacts:
        try:
            blockers.append((art, extract_leaf_blocker_set(art, cfg=cfg)))
        except Exception:
            blockers.append((art, None))
    holes: list[tuple[Any, float, float]] = []
    for art, bs in blockers:
        for hole in (getattr(bs, "interior_free_rects", ()) or ()):
            holes.append((art, hole[1].x - hole[0].x, hole[1].y - hole[0].y))
    if not holes:
        return []

    out: list[str] = []
    near_misses: list[str] = []
    for ref in pinned:
        owner = next(
            ((art, bs) for art, bs in blockers
             if ref in getattr(art.layout, "components", {})),
            None,
        )
        if owner is None or owner[1] is None:
            continue
        art, bs = owner
        rects = _blocker_occupied_rects(bs) or [bs.leaf_outline]
        guest_w = max(r[1].x for r in rects) - min(r[0].x for r in rects)
        guest_h = max(r[1].y for r in rects) - min(r[0].y for r in rects)
        best: tuple[float, str] | None = None
        for hole_art, hole_w, hole_h in holes:
            if hole_art is art:
                continue
            dx = hole_w - (guest_w + 2.0 * margin)
            dy = hole_h - (guest_h + 2.0 * margin)
            deficit = -min(dx, dy)
            if best is None or deficit < best[0]:
                best = (deficit, (
                    f"{ref}: guest {guest_w:.1f}x{guest_h:.1f} + 2x{margin:.1f}"
                    f" margin vs hole {hole_w:.1f}x{hole_h:.1f}"
                    f" -> short {max(0.0, -dx):.2f}/{max(0.0, -dy):.2f} mm"
                ))
        if best is None:
            continue
        if best[0] <= 0.0:
            out.append(ref)
        else:
            near_misses.append(best[1])
    if near_misses:
        # No-silent-miss rule: a 0.5 mm shortfall must read as "missed by
        # 0.5 mm" in the compose log, not as an unexplained rect fallback.
        print(
            "[candidate-search] nest-demotion fit check failed for "
            "pinned leaf/leaves: " + "; ".join(near_misses)
        )
    return out


def _search_best_layout(
    loaded_artifacts,
    *,
    spacing_mm: float,
    rotation_step_deg: float,
    parent_definition,
    pcb_path: Path | None,
    project_dir: Path,
    cfg: dict[str, Any] | None,
    base_seed: int,
    k: int = 8,
    time_budget_s: float = 60.0,
    manual_layout: ManualLayout | None = None,
) -> SearchResult:
    """Generate K placement candidates, hard-prefer shorts==0, return the winner.

    Each iteration calls ``_compose_artifacts`` with the project cfg
    UNCHANGED -- one solver pipeline, full fidelity per candidate. K is a
    diversity knob across seeds, not a quality mode.

    Winner selection is lexicographic: shorts ascending first, composite
    score descending second. If no candidate has shorts==0, the function
    raises ``RuntimeError`` -- the round fails loudly. There is no
    "best of bad options" fallback: a stamped board with shorts is not
    a useful artifact, and silently emitting one masks solver bugs that
    deserve attention.
    """
    from kicraft.autoplacer.brain.graph import total_ratsnest_length
    from kicraft.autoplacer.brain.placement_scorer import PlacementScorer
    from kicraft.autoplacer.freerouting_runner import _run_kicad_cli_drc

    base_cfg = dict(cfg or {})
    base_parent_placement = dict(base_cfg.get("parent_placement", {}))
    _search_cfg = dict(base_parent_placement.get("candidate_search", {}))

    # Manual mode: collapse the candidate search to a single trial. The
    # user has chosen the layout, so seed-aspect sweeping is meaningless
    # and stamp_drc is informational rather than gating.
    if manual_layout is not None:
        k = 1

    # Resolve artifact_dir for candidate stamping. Mirrors the path
    # _stamp_parent_board() builds so downstream code finds the winner
    # under the canonical slug.
    from kicraft.autoplacer.brain.subcircuit_artifacts import slugify_subcircuit_id

    if parent_definition is not None and getattr(parent_definition, "id", None):
        slug = slugify_subcircuit_id(parent_definition.id)
    else:
        slug = "parent"
    artifact_dir = project_dir / ".experiments" / "subcircuits" / slug
    search_dir = artifact_dir / "_search"
    search_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[CandidateRecord] = []
    cand_states: list[ParentCompositionState] = []
    cand_payloads: list[list[dict[str, Any]]] = []
    cand_pcb_paths: list[Path] = []

    # Brief-requested outline shape: center the aspect sweep on the shape's
    # own aspect (a circle wants square-ish content; a 60x40 rounded_rect
    # wants ~1.5) instead of burning candidates on elongated packs whose
    # circumscribed shape can never pass the stamp-time size guard.
    _shape_target_aspect: float | None = None
    if manual_layout is None:
        _bo_req = base_cfg.get("board_outline")
        if (
            isinstance(_bo_req, dict)
            and str(_bo_req.get("shape", "rect")).strip().lower() not in ("", "rect")
        ):
            _size_pair = _requested_size_pair(_bo_req.get("size_mm"))
            _shape_target_aspect = (
                _size_pair[0] / _size_pair[1] if _size_pair else 1.0
            )

    t_search_start = time.perf_counter()
    # Candidate waves (PR-N4, docs/plans/shaped-compose-leaf-nesting.md):
    # wave 1 runs with the caller's cfg verbatim. When a shaped brief gets
    # ZERO shape-fitted candidates and a strict edge/corner-pinned leaf could
    # instead NEST inside another leaf's interior hole, ONE more wave runs
    # with those component_zones entries dropped (same seeds -> deterministic)
    # -- the brief-requested shape outranks a derived edge zone that
    # contradicts it. Kill switch: candidate_search.edge_demotion = false.
    pending_waves: list[tuple[dict[str, Any], str]] = [(base_cfg, "cand")]
    edge_pins_demoted: list[str] = []
    wave_no = 0
    wave1_count = 0
    while pending_waves:
        wave_cfg, _cand_prefix = pending_waves.pop(0)
        wave_no += 1
        t_wave_start = time.perf_counter()
        for i in range(k):
            elapsed_ms = (time.perf_counter() - t_wave_start) * 1000.0
            if elapsed_ms > time_budget_s * 1000.0:
                print(
                    f"[candidate-search] time budget {time_budget_s:.0f}s exhausted "
                    f"after {i} candidate(s)",
                    file=sys.stderr,
                )
                break
            seed_i = base_seed + i
            # Sweep the seed aspect linearly across the K candidates so each
            # parents-only invocation explores a spread of horizontal-strip
            # to vertical-stack shapes instead of K placements all starting
            # from the same square seed.  i=0 is the most horizontal
            # (aspect=0.6, wide+short), i=k-1 is the most vertical
            # (aspect=1.7, tall+narrow).  K=1 stays at aspect=1.0 (square)
            # for backward compatibility with single-candidate searches.
            # area_overhead stays fixed -- it is config-tunable per round
            # via parent_seed_area_overhead so users can dial overall seed
            # tightness without forking the per-candidate aspect sweep.
            if _shape_target_aspect is not None:
                # Shaped board: sweep +/-25% around the shape's target aspect.
                if k > 1:
                    seed_aspect_i = _shape_target_aspect * (0.75 + (i / (k - 1)) * 0.5)
                else:
                    seed_aspect_i = _shape_target_aspect
            elif k > 1:
                seed_aspect_i = 0.6 + (i / (k - 1)) * 1.1
            else:
                seed_aspect_i = 1.0
            seed_overhead_i = float(wave_cfg.get("parent_seed_area_overhead", 2.5))
            # Fix 2 (parent-compose compactness): pass 1 solves on the area-basis
            # seed; if the solved placement reveals it over-provisioned the
            # interior, a pass-2 re-solve on the measured right-sized seed enters
            # as an ADDITIONAL, tighter candidate. Both compete on stamp+DRC
            # score, so pass 1 stays in the pool as the route-congestion fallback.
            # The worklist grows in place: the ("r", refit_seed) entry is appended
            # once, after pass 1 measures it. Kill switch:
            # candidate_search.parent_refit = false.
            _passes: list[tuple[str, tuple[float, float] | None]] = [("", None)]
            _pass_idx = 0
            while _pass_idx < len(_passes):
                _pass_suffix, _seed_override = _passes[_pass_idx]
                _pass_idx += 1
                t_solve = time.perf_counter()
                state, payloads = _compose_artifacts(
                    loaded_artifacts,
                    spacing_mm=spacing_mm,
                    rotation_step_deg=rotation_step_deg,
                    parent_definition=parent_definition,
                    pcb_path=pcb_path,
                    cfg=wave_cfg,
                    seed=seed_i,
                    seed_area_overhead=seed_overhead_i,
                    seed_aspect_target=seed_aspect_i,
                    seed_size_override=_seed_override,
                    manual_layout=manual_layout,
                )
                place_solve_ms = (time.perf_counter() - t_solve) * 1000.0
                state.phase_timings["place_solve_ms"] = place_solve_ms

                if pcb_path is None:
                    # No source PCB → no stamping, no DRC. Search degrades to
                    # placement-quality ranking (composite score only); shorts
                    # is unknowable so all candidates are treated as
                    # accepted=True. This keeps a single search code path
                    # for the rare interactive `compose-subcircuits --output`
                    # case without --pcb.
                    stamp_ms = 0.0
                    stamp_drc_ms = 0.0
                    shorts = 0
                    stamp_edge_clr = 0
                    stamped = Path("")
                else:
                    cand_pcb = search_dir / f"{_cand_prefix}_{i:02d}{_pass_suffix}.kicad_pcb"
                    t_stamp = time.perf_counter()
                    stamped = _stamp_parent_board(
                        state, pcb_path, project_dir, wave_cfg,
                        output_pcb_path=cand_pcb,
                    )
                    stamp_ms = (time.perf_counter() - t_stamp) * 1000.0
                    state.phase_timings["stamp_ms"] = stamp_ms

                    t_drc = time.perf_counter()
                    drc = _run_kicad_cli_drc(str(stamped), timeout_s=30) or {}
                    stamp_drc_ms = (time.perf_counter() - t_drc) * 1000.0
                    state.phase_timings["stamp_drc_ms"] = stamp_drc_ms
                    state.stamp_drc = dict(drc)
                    shorts = int(drc.get("shorts", 0) or 0)
                    stamp_edge_clr = int(drc.get("copper_edge_clearance", 0) or 0)

                board_state = state.composition.board_state if state.composition else None
                if board_state is None:
                    raise RuntimeError(
                        f"candidate-search cand={i}{_pass_suffix} seed={seed_i}: "
                        "composition has no board_state; cannot compute composite score"
                    )
                scorer = PlacementScorer(board_state, base_parent_placement)
                opp_side = float(scorer._score_block_opposite_side())
                overlap = float(scorer._score_courtyard_overlap())
                ratsnest_mm = float(total_ratsnest_length(board_state))
                net_dist = _net_dist_score(ratsnest_mm)
                # Compactness penalty: candidates with sprawling placements grow
                # the parent outline (sometimes 200+ mm tall) and break geometry
                # validation. The previous composite saturated to 4.37 across
                # most candidates (opp_side=0, overlap=12.5, net_dist=0),
                # leaving K=4 selection essentially random. Add a direct bbox
                # term so smaller boards beat bigger ones when the rest is
                # tied. Uses _score_bbox_packing (0..100, higher = denser).
                bbox_packing = float(scorer._score_bbox_packing())
                composite = (
                    0.30 * opp_side
                    + 0.25 * overlap
                    + 0.15 * net_dist
                    + 0.30 * bbox_packing
                )

                # Diagnostic capture: unlocked-component cluster AABB (the
                # spread the solver actually controls) and the auto-grown
                # outline are NOT the same. Locked components (corner-pinned
                # mounting holes, edge-pinned connectors) sit at fixed
                # template positions and would constant-pad the cluster
                # measurement -- including them here would compare the
                # outline cap against a frame the solver can't shrink, so a
                # template board larger than the cap would always fail
                # regardless of how compact the unlocked placement was.
                # Mirrors _record_placed_extent so this measurement is
                # apples-to-apples with the per-phase solve_*_placed_*_mm
                # extents persisted alongside.
                comps = [
                    c for c in board_state.components.values() if not c.locked
                ]
                if comps:
                    phys = [c.physical_bbox() for c in comps]
                    placed_w_mm = max(b[1].x for b in phys) - min(b[0].x for b in phys)
                    placed_h_mm = max(b[1].y for b in phys) - min(b[0].y for b in phys)
                else:
                    placed_w_mm = 0.0
                    placed_h_mm = 0.0
                outline_tl, outline_br = board_state.board_outline
                outline_w_mm = max(0.0, outline_br.x - outline_tl.x)
                outline_h_mm = max(0.0, outline_br.y - outline_tl.y)
                # Outline-sprawl gate: an outline whose area dwarfs the copper it holds
                # means a phantom edge anchor or runaway auto-grow baked bare FR4 into
                # the board. The denominator is the SUMMED component courtyard area, NOT
                # the area their spread SPANS: when tight leaf clusters are flung apart
                # across the board the span ~= the whole outline (ratio ~1) and the
                # penalty never fires -- the exact 215x222mm-for-9%-packing sprawl we
                # want to catch (KC-8AG6FU). Summed area makes a board of small parts in
                # a big outline read its true low packing. Penalize so a compact
                # candidate always beats a sprawled one when the other terms tie.
                sprawl = 0.0
                all_phys = [c.physical_bbox() for c in board_state.components.values()]
                if all_phys and outline_w_mm > 0.0 and outline_h_mm > 0.0:
                    summed_courtyard_area = sum(
                        (b[1].x - b[0].x) * (b[1].y - b[0].y) for b in all_phys
                    )
                    sprawl, sprawl_penalty = _sprawl_penalty(
                        outline_w_mm * outline_h_mm, summed_courtyard_area
                    )
                    if sprawl_penalty > 0.0:
                        composite -= sprawl_penalty
                        print(
                            f"[candidate-search] cand={i}{_pass_suffix} outline "
                            f"{outline_w_mm:.1f}x{outline_h_mm:.1f}mm is {sprawl:.1f}x "
                            f"its summed courtyard area; score -{sprawl_penalty:.1f}"
                        )
                # Requested-shape fit for THIS candidate (set by _fit_requested_shape
                # inside _stamp_parent_board). A candidate whose placement the shape
                # could not wrap at the requested size loses to any that fit: the
                # penalty separates ties, and winner selection below hard-prefers
                # fitted candidates. No shape requested (or no stamping) => True.
                if state.requested_shape is None or pcb_path is None:
                    shape_fitted = True
                else:
                    shape_fitted = bool((state.shape_fit or {}).get("fitted"))
                if not shape_fitted:
                    composite -= 30.0
                    print(
                        f"[candidate-search] cand={i}{_pass_suffix} requested shape "
                        f"{str(state.requested_shape.get('shape'))!r} did not fit: "
                        f"{(state.shape_fit or {}).get('reason')}; score -30.0"
                    )
                # state.geometry_validation is populated inside _stamp_parent_board
                # via _validate_parent_geometry. When pcb_path is None the stamp
                # path is skipped, leaving geometry_validation = {} -- treat that
                # as accepted=True (no stamping happened, so nothing to reject).
                gv = state.geometry_validation or {}
                geometry_accepted = bool(gv.get("accepted", True))
                outside_component_count = int(gv.get("outside_component_count", 0) or 0)
                outside_pad_count = int(gv.get("outside_pad_count", 0) or 0)

                # Hard gate: shorts==0 only. Stamped electrical shorts are an
                # objective truth (DRC counts them), so a candidate with shorts
                # cannot win regardless of its routing prospects. Geometry
                # violations (components/pads outside the auto-grown outline)
                # are RECORDED on the CandidateRecord (geometry_accepted,
                # outside_component_count, outside_pad_count) but no longer
                # short-circuit the picker -- they're a guess at unfabricability
                # that prevents FreeRouting from running and starves the search
                # of real signal. Let routing run; a layout that violates
                # geometry will produce a routed PNG showing exactly where the
                # problem is, which is more actionable than "round aborted".
                #
                # Manual mode bypass: the user explicitly chose this placement.
                # stamp_drc is informational (surfaced in the GUI), not a gate;
                # auto-rejecting on shorts would force the user to redo the
                # whole drag-route cycle when often a single track tweak in the
                # routed result is enough.
                if manual_layout is not None:
                    accepted = True
                else:
                    accepted = shorts == 0

                rec = CandidateRecord(
                    seed=seed_i,
                    shorts=shorts,
                    score=composite,
                    place_solve_ms=place_solve_ms,
                    stamp_ms=stamp_ms,
                    stamp_drc_ms=stamp_drc_ms,
                    accepted=accepted,
                    pcb_path=str(stamped),
                    stamp_edge_clearance=stamp_edge_clr,
                    refit=_pass_suffix == "r",
                    bbox_h_mm=placed_h_mm,
                    bbox_w_mm=placed_w_mm,
                    outline_h_mm=outline_h_mm,
                    outline_w_mm=outline_w_mm,
                    breakdown={
                        "opp_side": opp_side,
                        "overlap": overlap,
                        "net_dist": net_dist,
                        "bbox_packing": bbox_packing,
                        "sprawl": sprawl,
                    },
                    geometry_accepted=geometry_accepted,
                    outside_component_count=outside_component_count,
                    outside_pad_count=outside_pad_count,
                    phase_timings=dict(state.phase_timings),
                    shape_fitted=shape_fitted,
                )
                candidates.append(rec)
                cand_states.append(state)
                cand_payloads.append(payloads)
                cand_pcb_paths.append(stamped)
                print(
                    f"[candidate-search] cand={i}{_pass_suffix} seed={seed_i} "
                    f"aspect={seed_aspect_i:.2f} "
                    f"shorts={shorts} score={composite:.1f} "
                    f"bh={placed_h_mm:.1f}mm bw={placed_w_mm:.1f}mm "
                    f"oh={outline_h_mm:.1f}mm geom_ok={geometry_accepted} "
                    f"place={place_solve_ms / 1000:.1f}s drc={stamp_drc_ms / 1000:.1f}s"
                )

                # Queue the pass-2 re-fit exactly once, when pass 1 (no override)
                # measured meaningful interior slack. state.refit_seed is None on
                # shaped/ff/manual/override runs, so those never re-fit.
                if (
                    _seed_override is None
                    and state.refit_seed is not None
                    and bool(_search_cfg.get("parent_refit", True))
                ):
                    _passes.append(("r", state.refit_seed))

        if wave_no == 1:
            wave1_count = len(candidates)
            if (
                manual_layout is None
                and pcb_path is not None
                and _shape_target_aspect is not None
                and candidates
                and not any(c.shape_fitted for c in candidates)
                and bool(_search_cfg.get("edge_demotion", True))
            ):
                edge_pins_demoted = _edge_demotion_candidates(
                    loaded_artifacts, base_cfg
                )
                if edge_pins_demoted:
                    print(
                        f"[candidate-search] no candidate fit the requested "
                        f"shape; re-running K={k} with edge pin(s) "
                        f"{', '.join(edge_pins_demoted)} demoted -- the "
                        f"brief-requested shape outranks the derived edge "
                        f"zone (recorded in candidate_search.edge_pins_demoted)"
                    )
                    demoted_cfg = dict(base_cfg)
                    demoted_cfg["component_zones"] = {
                        r: z
                        for r, z in (base_cfg.get("component_zones") or {}).items()
                        if r not in set(edge_pins_demoted)
                    }
                    pending_waves.append((demoted_cfg, "cand_d"))

    total_search_ms = (time.perf_counter() - t_search_start) * 1000.0

    if not candidates:
        # K iterations exited the budget loop with zero successful
        # appends. Either k <= 0 was passed or the time_budget_s was so
        # tight that the very first compose blew it. Either way it's a
        # configuration problem, not a placement bug -- surface it.
        raise RuntimeError(
            f"candidate-search ran zero candidates in K={k}, "
            f"time_budget_s={time_budget_s:.0f}; check candidate_search cfg"
        )

    accepted_recs = [c for c in candidates if c.accepted]
    if not accepted_recs:
        # All K candidates failed at least one hard gate (shorts > 0
        # or geometry_accepted False). Fail loudly -- per "no
        # fallbacks", a masked failure here would let real solver
        # bugs ship under a green checkmark.
        #
        # Before raising, persist per-candidate diagnostics. Each record
        # already carries the per-phase solve_*_placed_{w,h}_mm extents
        # captured by _record_placed_extent(); without writing them to
        # disk now, the round abort throws this data away and the next
        # operator has nothing to triage from. The success path scrubs
        # search_dir at line ~2703; the abort path leaves cand_*.kicad_pcb
        # AND now this JSON for inspection.
        rejected_payload = {
            "k": k,
            "tried": len(candidates),
            "total_search_ms": total_search_ms,
            "candidates": [c.to_dict() for c in candidates],
        }
        try:
            (search_dir / "_rejected_candidates.json").write_text(
                json.dumps(rejected_payload, indent=2, default=float)
            )
        except OSError as exc:
            print(
                f"[candidate-search] failed to write rejected-candidate "
                f"diagnostics to {search_dir}: {exc}",
                file=sys.stderr,
            )
        rejection_summary = ", ".join(
            (
                f"seed={c.seed}:"
                f"shorts={c.shorts},"
                f"geom_ok={c.geometry_accepted},"
                f"bh={c.bbox_h_mm:.1f}mm,"
                f"bw={c.bbox_w_mm:.1f}mm"
            )
            for c in candidates
        )
        raise RuntimeError(
            f"candidate-search produced no acceptable placement in K={len(candidates)} "
            f"(per-candidate: {rejection_summary}). "
            f"Round aborted; investigate solver before re-running. "
            f"Per-candidate phase timings written to "
            f"{search_dir / '_rejected_candidates.json'}."
        )
    edge_clean_count = sum(
        1 for c in accepted_recs if c.stamp_edge_clearance == 0
    )
    if not edge_clean_count:
        print(
            "[candidate-search] WARNING: every accepted candidate has stamp "
            "copper_edge_clearance violations -- the routed board will fail "
            "the fab gate; picking best-of-bad for inspection "
            "(see docs/plans/self-eval-2026-07-17-fix-plan.md T1/T2)"
        )
    winner_rec = max(accepted_recs, key=_winner_key)

    # Winner strand screen (self-eval 2026-07-17 T3): connector stranding is a
    # STAMP-time property, so a would-be winner whose edge connector sits
    # inboard is knowable before FreeRouting burns minutes routing a board the
    # validation must reject (the re-fit re-solve strands connectors this way;
    # runs 14/30 + the run_02 repro). Demote and re-pick, bounded by the pool;
    # if EVERY candidate strands, keep the original best (the round fails
    # downstream exactly as before -- never invent a new failure mode). Checks
    # run only on would-be winners: one pcbnew load each, not K.
    if (
        manual_layout is None
        and pcb_path is not None
        and not edge_pins_demoted  # demoted-wave zones are stale by design
        and bool(_search_cfg.get("winner_strand_screen", True))
        and base_cfg.get("enforce_connector_edge_gap", True)
    ):
        try:
            from kicraft.autoplacer.brain.connector_edge_gap import (
                connector_edge_gaps,
            )
            _zones = base_cfg.get("component_zones", {}) or {}
            _tol = float(base_cfg.get("connector_edge_inboard_tol_mm", 1.0))
            _original_winner = winner_rec
            _screen_pool = list(accepted_recs)
            while _screen_pool:
                _gaps = connector_edge_gaps(
                    str(winner_rec.pcb_path), _zones, inboard_tol_mm=_tol
                )
                _stranded = [g for g in _gaps if g.gap_mm < -_tol]
                if not _stranded:
                    break
                print(
                    f"[candidate-search] winner (seed={winner_rec.seed}"
                    f"{'r' if winner_rec.refit else ''}) strands "
                    + ", ".join(f"{g.ref}@{g.gap_mm:.2f}mm({g.edge})"
                                for g in _stranded)
                    + " at stamp time; demoting and re-picking"
                )
                _screen_pool.remove(winner_rec)
                if not _screen_pool:
                    print(
                        "[candidate-search] every candidate strands a "
                        "connector; keeping the original best (round will "
                        "fail the edge-gap gate downstream)"
                    )
                    winner_rec = _original_winner
                    break
                winner_rec = max(_screen_pool, key=_winner_key)
        except Exception as _screen_exc:  # noqa: BLE001 - screen, not a gate
            print(
                f"warning: winner strand screen skipped: {_screen_exc}",
                file=sys.stderr,
            )

    winner_idx = candidates.index(winner_rec)
    winner_state = cand_states[winner_idx]
    winner_payloads = cand_payloads[winner_idx]
    winner_pcb = cand_pcb_paths[winner_idx]

    winner_from_demoted_wave = bool(edge_pins_demoted) and winner_idx >= wave1_count
    winner_state.candidate_search = {
        "k": k,
        "tried": len(candidates),
        "accepted": len(accepted_recs),
        "rejected_drc": len(candidates) - len(accepted_recs),
        "shape_fitted": sum(1 for c in candidates if c.shape_fitted),
        "best_index": winner_idx,
        "best_seed": winner_rec.seed,
        "winner_refit": winner_rec.refit,
        "winner_stamp_edge_clearance": winner_rec.stamp_edge_clearance,
        "edge_clean_count": edge_clean_count,
        "total_search_ms": total_search_ms,
        "edge_pins_demoted": edge_pins_demoted,
        "winner_from_demoted_wave": winner_from_demoted_wave,
        "candidates": [c.to_dict() for c in candidates],
    }
    winner_state.phase_timings["candidate_search_ms"] = total_search_ms

    # Artifact-scoped record for the build-tail gates: when the SHIPPED board
    # came from the demoted wave, the connector-stranded gate must not flag
    # the demoted refs against the (now overridden) component_zones in
    # *_autoplacer.json. Stale files from earlier rounds are removed so the
    # record always describes the current winner.
    demoted_sidecar = artifact_dir / "edge_pins_demoted.json"
    if winner_from_demoted_wave:
        try:
            demoted_sidecar.write_text(json.dumps({
                "refs": edge_pins_demoted,
                "winner_seed": winner_rec.seed,
                "reason": "requested outline shape could not fit any "
                          "wave-1 candidate; leaf nested instead "
                          "(shaped-compose-leaf-nesting PR-N4)",
            }, indent=2))
        except OSError as exc:
            print(f"[candidate-search] failed to write {demoted_sidecar}: {exc}",
                  file=sys.stderr)
    else:
        demoted_sidecar.unlink(missing_ok=True)

    # Drop the per-trial stamped boards now that the winner is selected.
    # The search dir leaks ~16 MB/round of intermediates and the winner
    # is captured in winner_state + winner_pcb_path.
    try:
        shutil.rmtree(search_dir)
    except OSError:
        pass

    return SearchResult(
        winner_idx=winner_idx,
        winner_seed=winner_rec.seed,
        winner_state=winner_state,
        winner_payloads=winner_payloads,
        winner_pcb_path=winner_pcb,
        candidates=candidates,
        total_search_ms=total_search_ms,
    )


# route lives in _compose_route (Lever 2.5 split); re-exported.
from kicraft.cli._compose_route import (  # noqa: E402
    _route_parent_board,
)


# persist lives in _compose_persist (Lever 2.5 split); re-exported.
from kicraft.cli._compose_persist import (  # noqa: E402
    _persist_parent_artifact,
)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compose solved subcircuits into a parent composition state"
    )
    parser.add_argument(
        "--project",
        help="Project directory containing .experiments/subcircuits",
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="Specific solved artifact directory to include (repeatable)",
    )
    parser.add_argument(
        "--only",
        action="append",
        default=[],
        help="Restrict composition to sheet name, sheet file, or instance path",
    )
    parser.add_argument(
        "--parent",
        help="Compose a real parent by sheet name, sheet file, or instance path (including root)",
    )
    parser.add_argument(
        "--spacing-mm",
        type=float,
        default=2.0,
        help="Spacing between rigid child modules in mm (default: 2)",
    )
    parser.add_argument(
        "--rotation-step-deg",
        type=float,
        default=0.0,
        help="Per-artifact rotation increment in degrees (default: 0)",
    )
    parser.add_argument(
        "--output",
        help="Optional JSON file path to save the composition snapshot",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON instead of human-readable text",
    )
    # --- New flags for parent board stamping and routing ---
    parser.add_argument(
        "--pcb",
        help="Source .kicad_pcb file (template for stamping; needed for --stamp/--route)",
    )
    parser.add_argument(
        "--stamp",
        action="store_true",
        help="Stamp composition into a real .kicad_pcb file",
    )
    parser.add_argument(
        "--route",
        action="store_true",
        help="Route parent interconnects via FreeRouting (implies --stamp)",
    )
    parser.add_argument(
        "--jar",
        help="Path to FreeRouting JAR (overrides config)",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of placement rounds for parent composition (default: 1)",
    )
    parser.add_argument(
        "--config",
        help="Optional JSON config file to merge on top of default/project config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "RNG seed forwarded to the parent PlacementSolver. Different seeds "
            "produce different parent placements at fixed config; required for "
            "random-search to actually explore parent layouts (default: 0)."
        ),
    )
    parser.add_argument(
        "--manual-layout",
        help=(
            "Path to a manual_layout.v1 JSON file. When set, the candidate "
            "search is skipped: leaf placements and the board outline are "
            "taken verbatim from the file, and the rest of the pipeline "
            "(stamp, stamp_drc, route) runs unchanged."
        ),
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])

    project_dir = Path(args.project).resolve() if args.project else None
    artifact_dirs = _resolve_artifact_dirs(project_dir, args.artifact)

    if not artifact_dirs:
        print(
            "error: no solved subcircuit artifacts found; provide --artifact or --project",
            file=sys.stderr,
        )
        return 2

    # Apply any active leaf pins so the canonical artifact files reflect
    # the pinned round, then load. ensure_applied is idempotent and a no-op
    # when no pins.json exists, so this is safe on every compose run. If
    # ensure_applied fails (corrupt pins.json, missing pinned round file),
    # we let it raise -- silently continuing with stale leaf state would
    # produce wrong placements without surfacing why.
    if project_dir is not None:
        from kicraft.autoplacer.brain import pins
        pin_status = pins.ensure_applied(project_dir / ".experiments")
        for leaf_key, status in pin_status.items():
            print(f"[pins] {leaf_key}: {status}")

    try:
        loaded_artifacts = load_solved_artifacts(list(artifact_dirs))
        loaded_artifacts = _filter_loaded_artifacts(loaded_artifacts, args.only)
        parent_definition = _select_parent_definition(project_dir, args.parent)
        loaded_artifacts = _filter_artifacts_for_parent(
            loaded_artifacts,
            parent_definition,
        )

        # NO FALLBACKS: every expected child subcircuit must have produced a
        # solved artifact. A child whose solve FAILED yields none, and absorbing
        # its components as loose parent-level parts (extract_parent_local +
        # _wrap_loose_parent_components_as_leaves -> force/SA at the parent) has
        # no reason to place them -- it only masks the failure and burns CPU.
        # Abort loudly so the failing leaf is fixed, instead of degrading to that
        # fallback. (--only is an explicit partial-compose debug path: skip it.)
        if parent_definition is not None and not args.only:
            missing = _missing_child_artifacts(parent_definition, loaded_artifacts)
            if missing:
                names = ", ".join(
                    f"{getattr(c, 'sheet_name', '') or c.instance_path}"
                    for c in missing
                )
                print(
                    "error: compose aborted -- child subcircuit(s) produced no "
                    f"solved artifact (their solve failed): {names}. Refusing to "
                    "strand their components as loose parent-level parts and "
                    "force/SA them at the parent -- that fallback cannot place a "
                    "failed leaf and only hides the failure. Fix the failing "
                    "leaf(s) and rebuild.",
                    file=sys.stderr,
                )
                return 1

        if not loaded_artifacts:
            if args.parent:
                print(
                    "error: no solved child artifacts found for selected parent",
                    file=sys.stderr,
                )
            else:
                print(
                    "error: no matching solved artifacts after filtering",
                    file=sys.stderr,
                )
            return 1

        # Resolve project clearance early so the composer's pad-margin can adapt
        # to the project's design rules even on plain compose runs without
        # --stamp/--route.
        from kicraft.autoplacer.config import (
            DEFAULT_CONFIG,
            discover_project_config,
            load_project_config,
        )
        # Seed from DEFAULT_CONFIG so defaults (clearances, margins, jar path)
        # are present without requiring --config, matching
        # solve_subcircuits._load_config.
        compose_cfg: dict[str, Any] = {**DEFAULT_CONFIG}
        if project_dir:
            proj_cfg_path = discover_project_config(str(project_dir))
            if proj_cfg_path:
                compose_cfg.update(load_project_config(str(proj_cfg_path)))
        if args.config:
            compose_cfg.update(load_project_config(args.config))

        # Candidate-search loop: K fast-cfg solves + DRC, pick shorts==0
        # winner (lex order: shorts asc, composite score desc). K=1 collapses
        # to a single fast solve; same code path. K and time_budget come
        # from cfg["parent_placement"]["candidate_search"], with project
        # config overrides applied via load_project_config above.
        search_cfg_raw = (
            compose_cfg.get("parent_placement", {}).get("candidate_search", {})
        )
        try:
            k = max(1, int(search_cfg_raw.get("k", 4)))
        except (TypeError, ValueError):
            k = 4
        try:
            time_budget_s = max(1.0, float(search_cfg_raw.get("time_budget_s", 240.0)))
        except (TypeError, ValueError):
            time_budget_s = 240.0

        manual_layout: ManualLayout | None = None
        if args.manual_layout:
            manual_layout = load_manual_layout(args.manual_layout)
            print(
                f"[manual-layout] loaded {len(manual_layout.placements)} "
                f"leaf placements + {len(manual_layout.parent_local)} parent-local "
                f"overrides; outline={manual_layout.board_outline}"
            )

        search_result = _search_best_layout(
            loaded_artifacts,
            spacing_mm=max(0.0, args.spacing_mm),
            rotation_step_deg=args.rotation_step_deg,
            parent_definition=parent_definition,
            pcb_path=Path(args.pcb) if args.pcb else None,
            project_dir=project_dir if project_dir else Path("."),
            cfg=compose_cfg,
            base_seed=int(args.seed),
            k=k,
            time_budget_s=time_budget_s,
            manual_layout=manual_layout,
        )
        state = search_result.winner_state
        transformed_payloads = search_result.winner_payloads

        output_path = None
        if args.output:
            output_path = _save_composition_snapshot(
                Path(args.output).resolve(),
                state,
                transformed_payloads,
            )

    except Exception as exc:
        print(f"error: failed to compose subcircuits: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                _json_payload(
                    loaded_artifacts, state, transformed_payloads, output_path
                ),
                indent=2,
            )
        )
        return 0

    _print_human_summary(loaded_artifacts, state, transformed_payloads, output_path)

    # --- Parent board stamping and routing ---
    if args.route or args.stamp:
        if not args.pcb:
            print(
                "error: --pcb is required for --stamp and --route",
                file=sys.stderr,
            )
            return 2

        pcb_path = Path(args.pcb)
        if not pcb_path.exists():
            print(f"error: PCB file not found: {pcb_path}", file=sys.stderr)
            return 2

        if project_dir is None:
            print(
                "error: --project is required for --stamp and --route",
                file=sys.stderr,
            )
            return 2

        # Build config: DEFAULT_CONFIG base -> project config -> --config
        # overlay -> --jar override. Seeding DEFAULT_CONFIG ensures
        # freerouting_jar (and other defaults) are present so --route works
        # without --jar, matching solve_subcircuits._load_config.
        from kicraft.autoplacer.config import (
            DEFAULT_CONFIG,
            discover_project_config,
            load_project_config,
        )

        cfg: dict[str, Any] = {**DEFAULT_CONFIG, "pcb_path": str(pcb_path)}
        proj_cfg_path = discover_project_config(str(project_dir))
        if proj_cfg_path:
            cfg.update(load_project_config(str(proj_cfg_path)))
        if args.config:
            cfg.update(load_project_config(args.config))
        if args.jar:
            cfg["freerouting_jar"] = args.jar

        try:
            # Grow the parent outline to enclose all placed geometry before
            # validating/stamping. The constraint-aware outline can snap
            # smaller than the placed-content bbox (edge-anchored sides),
            # which leaves copper outside Edge.Cuts and makes FreeRouting
            # return no SES (rc=-1). Repairing here keeps the in-memory state
            # and the stamped Edge.Cuts in sync.
            outline_repair = _repair_parent_outline(
                state,
                pad_edge_clearance_mm=float(
                    cfg.get("connector_edge_pad_clearance_mm", 0.2)
                ),
            )
            if outline_repair.get("repaired"):
                print(
                    "parent_outline_repaired: "
                    f"{outline_repair['old_size_mm']} -> {outline_repair['new_size_mm']} mm "
                    "(grown to enclose placed geometry)"
                )
            geometry_validation = _validate_parent_geometry(state)
            geometry_accepted = bool(geometry_validation.get("accepted", False))

            # Always stamp + render, even when geometry validation fails.
            # The PNG is the user's diagnostic for why composition was
            # rejected, and the Monitor tab falls back to it when
            # parent_routed.png is absent (failed routing).
            try:
                _t_stamp = time.perf_counter()
                stamped_pcb = _stamp_parent_board(state, pcb_path, project_dir, cfg)
                state.phase_timings["stamp_ms"] = (
                    time.perf_counter() - _t_stamp
                ) * 1000.0
                print(f"parent_stamped_pcb : {stamped_pcb}")

                stamped_render_dir = stamped_pcb.parent / "renders"
                stamped_renders = _render_parent_board_views(
                    stamped_pcb,
                    stamped_render_dir / ".tmp_parent_stamped_views",
                )
                if stamped_renders.get("front_all"):
                    shutil.copy2(
                        stamped_renders["front_all"],
                        stamped_render_dir / "parent_stamped.png",
                    )
                    print(
                        f"parent_stamped_png : {stamped_render_dir / 'parent_stamped.png'}"
                    )
            except Exception as stamp_exc:
                if geometry_accepted:
                    raise
                # If geometry was already going to be rejected, a stamping
                # failure here is secondary -- keep the original rejection
                # as the reason but note the stamp problem.
                print(
                    f"warning: stamp/render failed after geometry rejection: {stamp_exc}",
                    file=sys.stderr,
                )

            if not geometry_accepted:
                # Diagnostic dump but DON'T exit. The previous behaviour
                # was to return 1 here, which is the THIRD pre-route
                # rejection gate that starved the search of signal:
                # the stamped board exists, the user wants to see what
                # came out, and FreeRouting can still produce a useful
                # routed render even on a layout whose components extend
                # past the auto-grown outline. Surface as a warning and
                # let routing run.
                print(
                    json.dumps(
                        {
                            "parent_geometry_validation_failed": True,
                            "geometry_validation": geometry_validation,
                        },
                        indent=2,
                    ),
                    file=sys.stderr,
                )
                print(
                    "warning: parent composition geometry validation failed; "
                    "continuing to route so the routed render is available "
                    "as a diagnostic",
                    file=sys.stderr,
                )

            # Stamp-time DRC guard: kicad-cli DRC on the pre-route board so
            # composer-introduced shorts (two leaves' locked tracks stamped
            # on top of each other) are caught and labeled as such, instead
            # of being misattributed to FreeRouting later. When shorts are
            # detected, routing is skipped: FreeRouting cannot fix
            # overlapping copper, and a 200 s+ routing pass on a known-bad
            # layout is wasted CPU. The composer-vs-router attribution is
            # recorded in state.stamp_drc and surfaces in the round JSON.
            stamp_shorts = 0
            stamp_clearance = 0
            try:
                from kicraft.autoplacer.freerouting_runner import _run_kicad_cli_drc
                _t_stamp_drc = time.perf_counter()
                _stamp_drc = _run_kicad_cli_drc(str(stamped_pcb), timeout_s=30)
                state.phase_timings["stamp_drc_ms"] = (
                    time.perf_counter() - _t_stamp_drc
                ) * 1000.0
                stamp_shorts = int(_stamp_drc.get("shorts", 0))
                stamp_clearance = int(_stamp_drc.get("clearance", 0))
                state.stamp_drc = {
                    "ran": bool(_stamp_drc.get("ran", False)),
                    "shorts": stamp_shorts,
                    "clearance": stamp_clearance,
                    "copper_edge_clearance": int(_stamp_drc.get("copper_edge_clearance", 0)),
                    "courtyard": int(_stamp_drc.get("courtyard", 0)),
                    "report_excerpt": (str(_stamp_drc.get("report_text", ""))[:2000]),
                }
                if stamp_shorts > 0:
                    print(
                        f"warning: stamp-time DRC found {stamp_shorts} shorts on "
                        f"parent_pre_freerouting -- composer stamped overlapping "
                        f"leaf tracks; skipping FreeRouting",
                        file=sys.stderr,
                    )
            except Exception as drc_exc:
                # kicad-cli failure or subprocess crash. Without a stamp
                # DRC result we can't verify the placement is routable;
                # proceeding to FreeRouting on an unverified board would
                # let composer-introduced shorts ship as routing failures
                # without attribution. Persist what we know and re-raise
                # so the round fails loudly.
                state.stamp_drc = {"ran": False, "error": str(drc_exc)}
                if args.output:
                    _save_composition_snapshot(
                        Path(args.output).resolve(),
                        state,
                        transformed_payloads,
                    )
                raise

            # Re-save the snapshot with stamp + stamp_drc timings populated
            # so --stamp-only runs (and the route-skipped path below) carry
            # the full phase_timings breakdown into the output JSON.
            if args.output:
                _save_composition_snapshot(
                    Path(args.output).resolve(),
                    state,
                    transformed_payloads,
                )

            # Early bail: stamp DRC shows the placement is unroutable as
            # stamped. Surface as a routing rejection so callers (the
            # autoexperiment harness, etc.) treat the round as failed.
            if stamp_shorts > 0 and args.route:
                state.routed_validation = {
                    "accepted": False,
                    "rejection_reasons": [f"stamp_shorts={stamp_shorts}"],
                    "drc": {
                        "shorts": stamp_shorts,
                        "clearance": stamp_clearance,
                        "skipped_routing": True,
                    },
                }
                if args.output:
                    _save_composition_snapshot(
                        Path(args.output).resolve(),
                        state,
                        transformed_payloads,
                    )
                print(
                    f"parent_status      : rejected (stamp_shorts={stamp_shorts})"
                )
                print(
                    f"error: parent stamped with {stamp_shorts} shorts; "
                    f"skipped routing",
                    file=sys.stderr,
                )
                return 1

            # Connector edge-mount gate. An edge-zoned connector (USB-C, screw
            # terminal, etc.) whose mouth is pulled inboard of the board edge it
            # is zoned to is an UNMATEABLE port -- a real defect DRC cannot see
            # (the board is electrically fine, so it would otherwise ship
            # "fab-ready"). Stranding is a PLACEMENT property fixed by stamp time,
            # so measure it on the stamped board here. Rather than skip routing
            # and fail the whole build (which surfaces as a misleading "no routed
            # parent / route-infra failure" with NO board), we route the board,
            # then persist + promote it flagged NOT fab-ready below: a routed
            # board the user can inspect, honestly graded, beats nothing. The
            # fab-readiness verify gate (_verify_routed_board) independently
            # re-checks stranding so a stranded board can never reach "fab-ready".
            # Config-gated (default on); a generous inboard tolerance so only
            # genuine stranding -- not pad-inset / rounding noise -- is flagged.
            connector_strand_reasons: list[str] = []
            connector_edge_gaps_payload: list[dict[str, Any]] = []
            if args.route and cfg.get("enforce_connector_edge_gap", True):
                component_zones = cfg.get("component_zones", {}) or {}
                try:
                    from kicraft.autoplacer.brain.connector_edge_gap import (
                        connector_edge_gaps,
                    )

                    inboard_tol = float(
                        cfg.get("connector_edge_inboard_tol_mm", 1.0)
                    )
                    edge_gaps = connector_edge_gaps(
                        str(stamped_pcb),
                        component_zones,
                        inboard_tol_mm=inboard_tol,
                    )
                    # Flag only genuine INBOARD stranding -- not the rarer
                    # excessive-overhang failure, which is far less clearly a
                    # defect.
                    strand = [g for g in edge_gaps if g.gap_mm < -inboard_tol]
                except Exception as gate_exc:  # noqa: BLE001
                    # The gate must never invent a new failure mode: a pcbnew
                    # load hiccup here should not fail an otherwise-good round.
                    print(
                        f"warning: connector edge-gap gate skipped: {gate_exc}",
                        file=sys.stderr,
                    )
                    strand = []
                    edge_gaps = []
                if strand:
                    connector_strand_reasons = [
                        f"connector_stranded:{g.ref}@{g.gap_mm:.2f}mm({g.edge})"
                        for g in strand
                    ]
                    connector_edge_gaps_payload = [
                        {"ref": g.ref, "edge": g.edge, "gap_mm": g.gap_mm, "ok": g.ok}
                        for g in edge_gaps
                    ]
                    print(
                        "warning: connector(s) stranded inboard of their zoned "
                        f"board edge: {', '.join(connector_strand_reasons)}; "
                        "routing anyway and flagging the board NOT fab-ready",
                        file=sys.stderr,
                    )

            if args.route:
                _t_route = time.perf_counter()
                routing_result = _route_parent_board(
                    stamped_pcb, state, project_dir, cfg
                )
                state.phase_timings["freerouting_ms"] = (
                    time.perf_counter() - _t_route
                ) * 1000.0
                if not routing_result.get("failed"):
                    routed_board_path = Path(routing_result["routed_board_path"])
                    routed_renders = _render_parent_board_views(
                        routed_board_path,
                        stamped_render_dir / ".tmp_parent_routed_views",
                    )
                    if routed_renders.get("front_all"):
                        shutil.copy2(
                            routed_renders["front_all"],
                            stamped_render_dir / "parent_routed.png",
                        )
                        print(
                            f"parent_routed_png  : {stamped_render_dir / 'parent_routed.png'}"
                        )

                    validation = routing_result.get("validation", {})
                    if connector_strand_reasons:
                        # A stranded edge connector makes the board not fab-ready
                        # even when its copper is electrically clean. Fold the
                        # stamp-time finding into the routed validation so the
                        # board is promoted (below) but never accepted.
                        validation = dict(validation)
                        validation["accepted"] = False
                        merged_reasons = list(validation.get("rejection_reasons", []))
                        for _reason in connector_strand_reasons:
                            if _reason not in merged_reasons:
                                merged_reasons.append(_reason)
                        validation["rejection_reasons"] = merged_reasons
                        validation["connector_edge_gaps"] = connector_edge_gaps_payload
                    state.routed_validation = _compact_routed_validation(validation)
                    if args.output:
                        # Re-save the snapshot so the --output file reflects
                        # the post-route DRC summary (rejection reasons, DRC
                        # category counts). Necessary because the first save
                        # ran before routing; otherwise the data is lost when
                        # the run exits non-zero on rejection.
                        _save_composition_snapshot(
                            Path(args.output).resolve(),
                            state,
                            transformed_payloads,
                        )

                    # Apply post-routing DRC penalty to parent score.
                    # Shorts tank the score to near-zero; clearance
                    # violations apply proportional reduction.
                    drc = validation.get("drc", {})
                    shorts = drc.get("shorts", 0)
                    clearance = drc.get("clearance", 0)
                    if shorts > 0:
                        state.score_total *= 0.01
                        state.score_breakdown["drc_penalty"] = 0.01
                        state.score_notes.append(
                            f"DRC penalty: {shorts} short(s) -- score *= 0.01"
                        )
                    elif clearance > 0:
                        penalty = min(0.9, clearance * 0.1)
                        state.score_total *= (1.0 - penalty)
                        state.score_breakdown["drc_penalty"] = 1.0 - penalty
                        state.score_notes.append(
                            f"DRC penalty: {clearance} clearance violation(s) -- score *= {1.0 - penalty:.3f}"
                        )

                    if validation.get("accepted"):
                        artifact_dir = _persist_parent_artifact(
                            state, routing_result, project_dir, cfg
                        )
                        print(f"parent_artifact    : {artifact_dir}")
                        print("parent_status      : accepted")
                        _emit_inspector_bundle(
                            Path(artifact_dir) / "parent_routed.kicad_pcb"
                        )
                    else:
                        reasons = validation.get("rejection_reasons", [])
                        reason_str = ', '.join(reasons) if reasons else 'unknown'
                        if _promotable_strand_only(
                            reasons,
                            connector_strand_reasons,
                            validation.get("drc", {}),
                        ):
                            # The board is fully routed and electrically complete;
                            # its ONLY defect is an inboard-stranded connector (a
                            # placement-quality issue, not a route/infra failure).
                            # Persist + promote it so the build yields a routed
                            # board honestly marked NOT fab-ready (rc=7) rather
                            # than discarding it as "no routed parent" (rc=6) with
                            # nothing to inspect. The verify gate re-checks the
                            # stranding and keeps it out of "fab-ready".
                            artifact_dir = _persist_parent_artifact(
                                state, routing_result, project_dir, cfg
                            )
                            print(f"parent_artifact    : {artifact_dir}")
                            print(
                                "parent_status      : routed, NOT fab-ready "
                                f"({reason_str})"
                            )
                            _emit_inspector_bundle(
                                Path(artifact_dir) / "parent_routed.kicad_pcb"
                            )
                        else:
                            print(
                                f"parent_status      : rejected ({reason_str})"
                            )
                            # Still run the inspector on the rejected board so
                            # an AI agent can see exactly what failed: which
                            # DRC violations triggered the rejection, where
                            # the marker is vs. the board edge, etc.
                            rejected_pcb = (
                                Path(routing_result.get("routed_pcb", ""))
                                if routing_result.get("routed_pcb")
                                else None
                            )
                            if rejected_pcb and rejected_pcb.is_file():
                                _emit_inspector_bundle(rejected_pcb)
                            print(
                                f"error: parent board rejected by acceptance gate: {reason_str}",
                                file=sys.stderr,
                            )
                            return 1
                else:
                    error_msg = routing_result.get("error", "unknown error")
                    print(
                        f"error: parent routing failed: {error_msg}",
                        file=sys.stderr,
                    )
                    return 1
        except Exception as exc:
            print(f"error: parent stamping/routing failed: {exc}", file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
